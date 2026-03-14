import asyncio
import json
import re
import os
import logging
from datetime import datetime
import threading

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import networkx as nx

# ─────────────────────────────────────────────
# Heavy models — încărcate în background thread
# ─────────────────────────────────────────────
similarity_model  = None
cosine_similarity = None
ner_pipeline      = None
sentiment_pipeline = None
nlp_ready  = False
nlp_status = "Așteptare..."

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# ─────────────────────────────────────────────
# State global
# ─────────────────────────────────────────────
channels_set: set = set()
running    = False
paused     = False
target_channel  = None
keywords_list: list = []
similarity_mode = "similar"   # "repost" | "similar" | "stylography"

nodes_data: dict = {}   # {ch: {id, label, subscribers}}
edges_data: dict = {}   # {(c1,c2): float}  — forta relatie

posts_history: list = []

# Cache mesaje: {ch: [str, ...]}  (ultimele 50, deduplicate)
ch_msgs_cache: dict = {}

# Cache embedding incremental:
# ch_embs_cache[ch] = {"texts": [str,...], "matrix": np.ndarray shape(N,dim)}
# Mesajele noi sunt encodate INCREMENTAL — cele vechi nu se re-encodează.
ch_embs_cache: dict = {}

ch_style_cache: dict = {}
ch_lang_cache:  dict = {}

# Set paralel pentru lookup O(1) la deduplicarea mesajelor.
# ch_msgs_set[ch] oglindeste ch_msgs_cache[ch] dar ca set pentru O(1).
ch_msgs_set: dict = {}   # {ch: set(str)}

# Cache NLP: {hash(text): result_dict}
nlp_msg_cache: dict = {}

# Entitati globale agregate
global_entities: dict = {"PER": {}, "ORG": {}}

# Dirty tracking: canale care au primit mesaje noi de la ultimul ciclu de analiza.
# Analizatorul re-evalueaza DOAR perechile (c1,c2) unde cel putin unul e dirty.
dirty_channels: set = set()

background_tasks: list = []

THRESHOLD = {
    "repost":      0.88,
    "similar":     0.72,
    "stylography": 0.72,
}
# Prag minim pentru scorul de aliniere NLP (bonus, nu conditie blocanta)
NLP_ALIGNMENT_MIN = 0.3

# Decay de baza per ciclu; relatiile puternice decad mai lent (logaritmic).
DECAY_BASE = 0.88

# ─────────────────────────────────────────────
# Incarcare modele NLP
# ─────────────────────────────────────────────
def _load_nlp_models(loop):
    global ner_pipeline, sentiment_pipeline, similarity_model, cosine_similarity
    global nlp_ready, nlp_status
    try:
        logger.info("[NLP] Importuri grele...")
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cos
        from transformers import pipeline as hf_pipeline
        cosine_similarity = sk_cos
        logger.info("[NLP] Librarii importate.")

        nlp_status = "Descarc Similarity Model..."
        _notify_state(loop)
        similarity_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        logger.info("[NLP] Similarity model ready.")

        nlp_status = "Incarcare NER..."
        _notify_state(loop)
        ner_pipeline = hf_pipeline(
            "ner",
            model="Babelscape/wikineural-multilingual-ner",
            aggregation_strategy="simple",
            device=-1,
        )
        logger.info("[NLP] NER ready.")

        nlp_status = "Incarcare Sentiment..."
        _notify_state(loop)
        sentiment_pipeline = hf_pipeline(
            "text-classification",
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            device=-1,
        )
        logger.info("[NLP] Sentiment ready.")

        nlp_ready  = True
        nlp_status = "Sistem NLP Pregatit."
        logger.info("[NLP] TOATE MODELELE ACTIVE.")
        _notify_state(loop)

    except Exception as e:
        logger.error(f"[NLP] EROARE INCARCARE: {e}", exc_info=True)


def _notify_state(loop):
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(manager.send_state(), loop)


def start_nlp_loading(loop=None):
    # Bug #4 fix: primim loop-ul din startup_event(), nu il obtinem cu
    # get_event_loop() care e deprecat in Python 3.10+ si RuntimeError in 3.12+.
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Dacă nu există loop running, creăm unul nou
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    threading.Thread(target=_load_nlp_models, args=(loop,), daemon=True).start()

# ─────────────────────────────────────────────
# Utilitare
# ─────────────────────────────────────────────
def detect_language(text: str) -> str:
    if not text:
        return "other"
    cyr = len(re.findall(r'[\u0400-\u04FF]', text))
    lat = len(re.findall(r'[a-zA-Z]', text))
    if cyr > lat: return "ru"
    if lat > 5:   return "ro"
    return "other"


def clean_text(text: str) -> str:
    t = re.sub(r'http\S+', '', text)
    t = re.sub(r'[^\w\s]', ' ', t)
    return t.strip().lower()


def text_matches_keywords(text: str, keywords: list) -> bool:
    if not keywords:
        return True
    lo = text.lower()
    return any(k.lower() in lo for k in keywords)


# Greutăți per dimensiune stilometrică — ticurile de scriere primesc greutate mai mare.
# Suma = 1.0 după normalizare (se face la prima utilizare).
_STYLE_DIM_WEIGHTS_RAW = np.array([
    0.5,   # 0  char_len      — util, dar nu decisiv
    1.5,   # 1  punc          — semn de stil clar
    1.5,   # 2  special/emoji — semn de stil clar
    1.0,   # 3  word_len
    2.0,   # 4  ellipsis      — tic de scriere
    2.0,   # 5  exclaim       — tic de scriere
    1.5,   # 6  uppercase     — stil agresiv vs. normal
    1.0,   # 7  ttr
    1.0,   # 8  hapax
    1.0,   # 9  sent_len
    1.5,   # 10 stopword      — densitate funcțională
    1.0,   # 11 digit
    1.0,   # 12 vocab_richness
    1.5,   # 13 short_sent
    1.5,   # 14 long_sent
    2.0,   # 15 questions     — tic de scriere
    1.0,   # 16 para_var
    1.5,   # 17 has_link
], dtype=float)
_STYLE_DIM_WEIGHTS = _STYLE_DIM_WEIGHTS_RAW / _STYLE_DIM_WEIGHTS_RAW.sum()

_STYLE_DIM_NAMES = [
    "char_len", "punc", "special", "word_len", "ellipsis", "exclaim",
    "uppercase", "ttr", "hapax", "sent_len", "stopword", "digit",
    "vocab_richness", "short_sent", "long_sent", "questions", "para_var", "has_link",
]

# Stopwords pentru română și rusă (cuvinte funcționale, nu de conținut)
_STOPWORDS_STYLE = {
    "si", "in", "la", "de", "cu", "ca", "este", "sunt", "nu", "se",
    "pe", "din", "o", "un", "sau", "care", "pentru", "dar", "mai",
    "ne", "a", "ai", "ale", "cel", "cea", "cei", "cele", "lui", "lor",
    "i", "v", "na", "s", "ne", "eto", "kak", "no",
}


def get_stylometric_fingerprint(texts: list) -> np.ndarray:
    """
    Vector stilografic cu 18 dimensiuni grupate în trei categorii:

    Suprafață textuală (6):
        0  avg_char_len       — lungimea medie a mesajului în caractere
        1  punc_density       — densitate punctuație (.,!?;:) per char
        2  special_density    — densitate caractere speciale/emoji per char
        3  avg_word_len       — lungimea medie a cuvântului în caractere
        4  ellipsis_rate      — frecvența „..." per char
        5  exclaim_rate       — frecvența „!!" per char
        6  uppercase_ratio    — proporția literelor mari din total litere

    Lexic & vocabular (6):
        7  ttr               — type-token ratio (vocabular unic / total cuvinte)
        8  hapax_ratio       — cuvinte care apar o singură dată / total unice
        9  avg_sent_len      — cuvinte per propoziție (split pe [.!?])
        10 stopword_ratio    — proporția stopwords din total cuvinte
        11 digit_ratio       — proporția cifrelor din total caractere
        12 vocab_richness    — log(nr. cuvinte unice + 1) / log(nr. total + 1)

    Ritm & structură (6):
        13 short_sent_ratio  — proporția propozițiilor cu < 5 cuvinte
        14 long_sent_ratio   — proporția propozițiilor cu > 20 cuvinte
        15 question_rate     — frecvența „?" per mesaj
        16 paragraph_var     — deviația standard a lungimii mesajelor (normalizată)
        17 link_density      — proporția mesajelor care conțin URL-uri

    Compararea între canale se face per-mesaj (distribuție), nu doar ca medii —
    folosim mediană + IQR pentru robustețe la outlieri.
    """
    if not texts or len(texts) < 2:
        return np.zeros(18)

    def _msg_vec(t: str) -> np.ndarray:
        """Extrage vectorul de 18 trăsături pentru un mesaj, toate pre-normalizate în [0,1]."""
        from collections import Counter
        n_char   = len(t) + 1
        words_lo = [w.lower().strip(".,!?;:\"'()[]") for w in t.split() if w.strip()]
        sents    = [s.strip() for s in re.split(r'[.!?]+', t) if s.strip()]
        n_sents  = len(sents) + 1
        sent_lens = [len(s.split()) for s in sents] if sents else [0]
        # Bug #8 fix: Counter O(n) in loc de words_lo.count(w) care era O(n²)
        freq     = Counter(words_lo)
        n_unique = len(freq) + 1
        n_total  = len(words_lo) + 1
        letters  = re.findall(r'[a-zA-Z\u00C0-\u024F\u0400-\u04FF]', t)
        upper    = re.findall(r'[A-Z\u00C0-\u00DE\u0400-\u042F]', t)
        return np.array([
            # 0  char_len normalizat la 500 chars max
            min(len(t) / 500.0, 1.0),
            # 1  densitate punctuatie
            len(re.findall(r'[.,!?;:]', t)) / n_char,
            # 2  densitate caractere speciale/emoji
            len(re.findall(r'[^\w\s]', t)) / n_char,
            # 3  lungime medie cuvant normalizata la 15
            min(sum(len(w) for w in words_lo) / ((len(words_lo) + 1) * 15.0), 1.0),
            # 4  rata ellipsis (... per char, x100 pentru vizibilitate)
            min(len(re.findall(r'\.\.\.', t)) / n_char * 100, 1.0),
            # 5  rata exclamare dubla (!! per char, x100)
            min(len(re.findall(r'!!', t)) / n_char * 100, 1.0),
            # 6  proportia literelor mari
            len(upper) / (len(letters) + 1),
            # 7  type-token ratio
            n_unique / n_total,
            # 8  hapax legomena ratio — Bug #8 fix: Counter O(n) nu O(n²)
            sum(1 for w, c in freq.items() if c == 1) / n_unique,
            # 9  lungime medie propozitie normalizata la 30 cuvinte
            min(float(np.mean(sent_lens)) / 30.0, 1.0),
            # 10 rata stopwords
            sum(1 for w in words_lo if w in _STOPWORDS_STYLE) / n_total,
            # 11 rata cifre
            len(re.findall(r'\d', t)) / n_char,
            # 12 bogatia vocabularului (log ratio)
            float(np.log(n_unique) / (np.log(n_total) + 1e-9)),
            # 13 proportia propozitiilor scurte (< 5 cuvinte)
            sum(1 for l in sent_lens if l < 5) / n_sents,
            # 14 proportia propozitiilor lungi (> 20 cuvinte)
            sum(1 for l in sent_lens if l > 20) / n_sents,
            # 15 rata intrebari normalizata la 3 per mesaj
            min(t.count("?") / 3.0, 1.0),
            # 16 placeholder para_var — se calculeaza la nivel corpus
            float(len(t)),
            # 17 prezenta URL
            1.0 if re.search(r'https?://', t) else 0.0,
        ], dtype=float)

    # Calculam vectorul per mesaj si luam MEDIANA (robusta la outlieri)
    all_vecs = [_msg_vec(t) for t in texts]
    medians  = np.median(np.stack(all_vecs), axis=0)

    # Dim 16: inlocuim placeholder cu variatia lungimii mesajelor (normalizata)
    char_lens = np.array([len(t) for t in texts], dtype=float)
    medians[16] = min(np.std(char_lens) / (np.mean(char_lens) + 1.0), 1.0)

    return medians


def analyse_text(text: str) -> dict:
    if not nlp_ready or not text or not ner_pipeline or not sentiment_pipeline:
        return {}
    h = hash(text)
    if h in nlp_msg_cache:
        return nlp_msg_cache[h]
    try:
        s_res = sentiment_pipeline(text[:512])[0]
        n_res = ner_pipeline(text[:512])
        seen, entities = set(), []
        for ent in n_res:
            name  = ent['word'].replace('##', '')
            etype = ent['entity_group']
            if etype in ("PER", "ORG") and len(name) > 2 and name not in seen:
                entities.append({"name": name, "type": etype})
                seen.add(name)
        res = {"sentiment": s_res, "entities": entities}
        nlp_msg_cache[h] = res
        return res
    except Exception:
        return {}


def update_entity_stats(nlp_data: dict):
    if not nlp_data or "sentiment" not in nlp_data:
        return
    label = nlp_data["sentiment"]["label"].lower()
    val   = 1.0 if "pos" in label else (-1.0 if "neg" in label else 0.0)
    for ent in nlp_data.get("entities", []):
        etype, ename = ent["type"], ent["name"]
        if etype in global_entities:
            if ename not in global_entities[etype]:
                global_entities[etype][ename] = {"count": 0, "sum": 0.0}
            global_entities[etype][ename]["count"] += 1
            global_entities[etype][ename]["sum"]   += val


def parse_subscribers(text: str) -> int:
    if not text: return 0
    t = text.lower()
    for w in ["subscribers", "subscriber", "abonati", "abonat", "podpischikov", "podpischika"]:
        t = t.replace(w, "")
    t = t.strip().replace(" ", "").replace(",", "")
    if t.endswith("k"):
        try: return int(float(t[:-1]) * 1_000)
        except: pass
    if t.endswith("m"):
        try: return int(float(t[:-1]) * 1_000_000)
        except: pass
    try: return int(t)
    except: return 0


def scrape_channel(username: str) -> dict:
    u = username.lstrip("@")
    try:
        r1    = requests.get(f"https://t.me/{u}", timeout=8)
        s1    = BeautifulSoup(r1.text, "html.parser")
        t_el  = s1.find("div", class_="tgme_page_title")
        title = t_el.text.strip() if t_el else u
        e_el  = s1.find("div", class_="tgme_page_extra")
        subs  = parse_subscribers(e_el.text if e_el else "0")
        r2    = requests.get(f"https://t.me/s/{u}", timeout=8)
        s2    = BeautifulSoup(r2.text, "html.parser")
        msgs  = []
        for w in s2.find_all("div", class_="tgme_widget_message_text"):
            txt = w.get_text(separator=" ", strip=True)
            if len(txt) > 30 and len(txt.split()) >= 5:
                msgs.append(txt)
        return {"username": u, "title": title, "subscribers": subs, "messages": msgs[-15:]}
    except Exception as e:
        logger.error(f"Scrape error {username}: {e}")
        return {"username": u, "title": u, "subscribers": 0, "messages": []}

# ─────────────────────────────────────────────
# Embedding incremental
# ─────────────────────────────────────────────
def update_embeddings_incremental(ch: str, new_texts: list):
    """
    Encodează DOAR mesajele noi și le concatenează la matricea existentă.

    IMPORTANT — aliniament index:
    ch_embs_cache[ch]["orig_texts"] păstrează textele ORIGINALE (nemodificate)
    în aceeași ordine cu rândurile din matrice. Astfel sims[i,j] corespunde
    întotdeauna corect cu orig_texts[i] și orig_texts[j].

    Structura ch_embs_cache[ch]:
        "orig_texts": [str, ...]   — texte originale, paralele cu rândurile matricei
        "matrix":     np.ndarray   — shape (N, embedding_dim)
    """
    if not new_texts or similarity_model is None:
        return

    # Curățăm pentru encoding, dar păstrăm și originalele
    clean_new = [clean_text(t) for t in new_texts]
    valid_pairs = [(orig, cl) for orig, cl in zip(new_texts, clean_new) if len(cl) > 10]
    if not valid_pairs:
        return

    orig_valid  = [p[0] for p in valid_pairs]
    clean_valid = [p[1] for p in valid_pairs]

    new_embs = similarity_model.encode(clean_valid, show_progress_bar=False)

    if ch not in ch_embs_cache or ch_embs_cache[ch] is None:
        ch_embs_cache[ch] = {"orig_texts": orig_valid, "matrix": new_embs}
    else:
        existing = ch_embs_cache[ch]
        # Bug #5 fix: Verificăm dacă existing are cheile corecte
        if "orig_texts" not in existing or "matrix" not in existing:
            ch_embs_cache[ch] = {"orig_texts": orig_valid, "matrix": new_embs}
        else:
            combined_orig = existing["orig_texts"] + orig_valid
            combined_matrix = np.vstack([existing["matrix"], new_embs])
            if len(combined_orig) > 50:
                combined_orig = combined_orig[-50:]
                combined_matrix = combined_matrix[-50:]
            ch_embs_cache[ch] = {"orig_texts": combined_orig, "matrix": combined_matrix}


def get_embedding_matrix(ch: str):
    entry = ch_embs_cache.get(ch)
    if entry is None:
        return None, None
    return entry["matrix"], entry["orig_texts"]

# ─────────────────────────────────────────────
# Decay logaritmic
# ─────────────────────────────────────────────
def _decay_edge(strength: float) -> float:
    """
    Relatiile puternice (forta ~ 10) decad mai lent decat cele slabe (forta ~ 1).
    Factor efectiv: intre DECAY_BASE si DECAY_BASE + 0.1
    """
    log_factor = 0.1 * (np.log10(strength + 1) / np.log10(11))
    return strength * (DECAY_BASE + log_factor)

# ─────────────────────────────────────────────
# Analiza globala a unei perechi de canale
# ─────────────────────────────────────────────
def _analyse_pair_global(c1: str, c2: str, mode: str, threshold: float):
    """
    Evalueaza similaritatea GLOBALA intre doua canale folosind INTREAGA
    matrice de embeddings disponibila — nu doar un singur best-match.

    Returneaza (edge_score: float, best_match: dict | None).
    """
    # Bug #7 fix: Adăugat MAX_HITS la începutul funcției
    MAX_HITS = 10
    score = 0.0
    match = None

    # ── Stilografie (detectare autor similar) ────────────────────────
    if mode == "stylography":
        lang1 = ch_lang_cache.get(c1, "other")
        lang2 = ch_lang_cache.get(c2, "other")
        if lang1 != lang2 or lang1 == "other":
            return 0.0, None

        v1 = ch_style_cache.get(c1)
        v2 = ch_style_cache.get(c2)
        if v1 is None or v2 is None:
            return 0.0, None

        # Mascăm dimensiunile inactive (ambele zero) — nu discriminează nimic
        active = (np.abs(v1) + np.abs(v2)) > 1e-5
        if active.sum() < 4:
            return 0.0, None

        w  = _STYLE_DIM_WEIGHTS * active.astype(float)
        w /= w.sum() + 1e-9   # renormalizăm pe dimensiunile active

        # ── Metrica 1: Bray-Curtis similaritate ponderată (cea mai discriminativă) ──
        # BC dissimilarity = sum(w * |a-b|) / sum(w * (|a|+|b|))
        # Diferit de cosinus — măsoară proporțional diferența per dimensiune
        bc_diss = np.sum(w * np.abs(v1 - v2)) / (np.sum(w * (np.abs(v1) + np.abs(v2))) + 1e-9)
        bc_sim  = 1.0 - bc_diss

        # ── Metrica 2: Convergență ponderată ──────────────────────────────────────
        # Câte dimensiuni au diferență relativă < 15%, ponderat după importanță
        conv_w = 0.0
        active_w_sum = float(_STYLE_DIM_WEIGHTS[active].sum()) + 1e-9
        for i, (a, b) in enumerate(zip(v1, v2)):
            if not active[i]:
                continue
            denom = max(abs(a), abs(b), 1e-5)
            if abs(a - b) / denom < 0.15:
                conv_w += _STYLE_DIM_WEIGHTS[i]
        conv_w /= active_w_sum

        # ── Metrica 3: Cosinus pe vectorii ponderați (sqrt pentru a evita saturarea) ──
        v1w = np.sqrt(np.abs(v1) * _STYLE_DIM_WEIGHTS)
        v2w = np.sqrt(np.abs(v2) * _STYLE_DIM_WEIGHTS)
        cos = float(cosine_similarity(v1w.reshape(1, -1), v2w.reshape(1, -1))[0, 0])

        # ── Scor combinat ─────────────────────────────────────────────────────────
        # Bray-Curtis (45%) + Convergență (35%) + Cosinus (20%)
        combined = 0.45 * bc_sim + 0.35 * conv_w + 0.20 * cos

        if combined < threshold:
            return 0.0, None

        # ── Detaliu per dimensiune pentru raportul în UI ──────────────────────────
        n_agree  = sum(1 for i, (a, b) in enumerate(zip(v1, v2))
                       if active[i] and abs(a-b) / max(abs(a), abs(b), 1e-5) < 0.15)
        n_active = int(active.sum())
        dim_details = {
            name: {
                "c1": round(float(v1[i]), 4),
                "c2": round(float(v2[i]), 4),
                "rel_diff": round(abs(float(v1[i]) - float(v2[i])) / max(abs(float(v1[i])), abs(float(v2[i])), 1e-5), 3),
                "agrees": abs(float(v1[i]) - float(v2[i])) / max(abs(float(v1[i])), abs(float(v2[i])), 1e-5) < 0.15,
                "weight": round(float(_STYLE_DIM_WEIGHTS[i]), 4),
            }
            for i, name in enumerate(_STYLE_DIM_NAMES)
            if active[i]
        }

        # ── Mesajele reprezentative: cel mai aproape de profilul median al canalului ──
        def pick_representative(ch: str, vec: np.ndarray) -> str:
            msgs = ch_msgs_cache.get(ch, [])
            if not msgs:
                return ""
            best_msg, best_dist = msgs[0], float("inf")
            for m in msgs:
                if not m.strip():
                    continue
                # Proxy rapid pe primele 4 dimensiuni (char_len, punc, special, word_len)
                words = m.split()
                n_char = len(m) + 1
                proxy = np.array([
                    min(len(m) / 500.0, 1.0),
                    len(re.findall(r'[.,!?;:]', m)) / n_char,
                    len(re.findall(r'[^\w\s]',  m)) / n_char,
                    min(sum(len(w) for w in words) / ((len(words)+1) * 15.0), 1.0),
                ])
                dist = float(np.linalg.norm(proxy - vec[:4]))
                if dist < best_dist:
                    best_dist, best_msg = dist, m
            return best_msg

        ex1 = pick_representative(c1, v1)
        ex2 = pick_representative(c2, v2)

        edge_score = combined * 3.0   # amplificare la scala [0, 10]
        match = {
            "type":                "stylography",
            "ch1":                 c1,
            "ch2":                 c2,
            "score":               round(combined, 4),
            "bc_sim":              round(bc_sim,   4),
            "conv":                round(conv_w,   4),
            "cosine":              round(cos,       4),
            "active_dimensions":   n_active,
            "agreeing_dimensions": n_agree,
            "dim_details":         dim_details,
            "ch1_msg":             ex1,
            "ch2_msg":             ex2,
        }
        return edge_score, match

    # ── Matrice de embeddings ─────────────────────────────────────────
    m1, orig1 = get_embedding_matrix(c1)
    m2, orig2 = get_embedding_matrix(c2)
    if m1 is None or m2 is None or m1.shape[0] == 0 or m2.shape[0] == 0:
        return 0.0, None

    # sims[i,j] = similaritate cosinus intre mesajul i din c1 si mesajul j din c2
    sims = cosine_similarity(m1, m2)   # shape (N1, N2)

    # ── Repostari ─────────────────────────────────────────────────────
    if mode == "repost":
        hits   = np.where(sims > threshold)
        n_hits = len(hits[0])
        if n_hits == 0:
            return 0.0, None

        score     = n_hits / 5.0
        best_flat = int(np.argmax(sims))
        bi, bj    = divmod(best_flat, sims.shape[1])
        match = {
            "type": "repost", "ch1": c1, "ch2": c2,
            "score": float(sims[bi, bj]),
            # Bug #5 fix: folosim orig_texts aliniate cu matricea, nu msgs_cache
            "ch1_msg": orig1[bi] if bi < len(orig1) else "",
            "ch2_msg": orig2[bj] if bj < len(orig2) else "",
            "n_hits": n_hits,
        }
        return score, match

    # ── Aliniere ideologica profunda (similar) ────────────────────────
    above = sims > threshold
    if not above.any():
        return 0.0, None

    # Bug #7 fix: limiteaza perechile procesate la primele MAX_HITS sortate
    # dupa scor descrescator. zip(*np.where(above)) poate produce sute de
    # perechi daca threshold e mic — fiecare apeleaza analyse_text() x2.
    hit_rows, hit_cols = np.where(above)
    hit_scores = sims[hit_rows, hit_cols]
    # Sortam descrescator si luam primele MAX_HITS
    top_idx  = np.argsort(hit_scores)[::-1][:MAX_HITS]
    hit_pairs = list(zip(hit_rows[top_idx], hit_cols[top_idx]))

    total_strength = 0.0
    best_match     = None
    best_val       = 0.0

    for (i, j) in hit_pairs:
        # Bug #5 fix: indexam direct orig_texts, nu msgs_cache (care poate
        # avea lungime diferita fata de matrice dupa fereastra glisanta)
        if i >= len(orig1) or j >= len(orig2):
            continue

        m1_text = orig1[i]
        m2_text = orig2[j]
        sem_sim = float(sims[i, j])

        # Filtru heuristic: suprapunere mare de cuvinte scurte = zgomot lexical
        w1 = set(m1_text.lower().split())
        w2 = set(m2_text.lower().split())
        common = w1 & w2
        if len(common) > 1 and len(m1_text) < 50:
            if len(common) / max(len(w1), len(w2), 1) > 0.4:
                continue

        # Bug #1 fix: sem_sim > threshold este SUFICIENT pentru detectie.
        # NLP-ul adauga un bonus de aliniere, dar NU mai este o conditie
        # blocanta. Daca NLP nu e gata sau nu gaseste entitati, scorul
        # semantic singur trece perechea drept relevanta.
        base_strength = sem_sim  # detectie garantata peste prag

        nlp1 = analyse_text(m1_text)
        nlp2 = analyse_text(m2_text)

        nlp_bonus = 0.0
        if nlp1 and nlp2:
            if nlp1.get("sentiment", {}).get("label") == nlp2.get("sentiment", {}).get("label"):
                nlp_bonus += 0.3
            e1 = {e['name'] for e in nlp1.get("entities", []) if len(e['name']) > 3}
            e2 = {e['name'] for e in nlp2.get("entities", []) if len(e['name']) > 3}
            if e1 & e2:
                nlp_bonus += 0.2
            update_entity_stats(nlp1)
            update_entity_stats(nlp2)

        # Scorul final al perechii: semantic + bonus NLP (max 1.5×)
        pair_strength = base_strength * (1.0 + nlp_bonus)

        total_strength += pair_strength
        if pair_strength > best_val:
            best_val   = pair_strength
            best_match = {
                "type":     "similar",
                "ch1":      c1,
                "ch2":      c2,
                "score":    round(sem_sim, 4),
                "nlp_bonus": round(nlp_bonus, 3),
                "ch1_msg":  m1_text,
                "ch2_msg":  m2_text,
            }

    if total_strength == 0.0:
        return 0.0, None

    edge_score = min(total_strength * 3.0, 10.0)
    return edge_score, best_match

# ─────────────────────────────────────────────
# Connection Manager
# ─────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
        await self.send_state()

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, msg: str):
        for c in list(self.active_connections):
            try:
                await c.send_text(msg)
            except Exception:
                pass

    async def send_state(self):
        state = {
            "type": "state_update",
            "channels": list(channels_set),
            "running": running,
            "paused": paused,
            "target_channel": target_channel,
            "keywords": keywords_list,
            "nlp_ready": nlp_ready,
            "nlp_status": nlp_status,
            "similarity_mode": similarity_mode,
            "global_entities": global_entities,
        }
        await self.broadcast(json.dumps(state))


manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    logger.info("Pornire aplicatie — initiere incarcare modele NLP.")
    loop = asyncio.get_running_loop()   # loop-ul real al uvicorn
    start_nlp_loading(loop)

# ─────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global running, paused, channels_set, nodes_data, edges_data, posts_history
    global target_channel, keywords_list, similarity_mode, global_entities
    global ch_msgs_cache, ch_embs_cache, ch_style_cache, background_tasks, dirty_channels

    await manager.connect(websocket)
    try:
        while True:
            data   = await websocket.receive_text()
            cmd    = json.loads(data)
            action = cmd.get("action")

            if action == "add_channel":
                ch = cmd.get("channel", "").strip()
                if ch:
                    ch = "@" + ch.lstrip("@")
                    channels_set.add(ch)
                    if ch not in nodes_data:
                        nodes_data[ch] = {"id": ch, "label": ch, "subscribers": 0}
                    await manager.send_state()

            elif action == "remove_channel":
                ch = cmd.get("channel", "").strip()
                channels_set.discard(ch)
                dirty_channels.discard(ch)
                if ch == target_channel:
                    target_channel = None
                nodes_data.pop(ch, None)
                ch_msgs_cache.pop(ch, None)
                ch_msgs_set.pop(ch, None)
                ch_embs_cache.pop(ch, None)
                ch_style_cache.pop(ch, None)
                ch_lang_cache.pop(ch, None)
                for k in [k for k in edges_data if ch in k]:
                    edges_data.pop(k, None)
                await manager.send_state()

            elif action == "reset":
                channels_set.clear()
                nodes_data.clear()
                edges_data.clear()
                posts_history.clear()
                ch_msgs_cache.clear()
                ch_msgs_set.clear()
                ch_embs_cache.clear()
                ch_style_cache.clear()
                ch_lang_cache.clear()
                nlp_msg_cache.clear()
                dirty_channels.clear()
                global_entities["PER"].clear()
                global_entities["ORG"].clear()
                target_channel = None
                running = False
                paused  = False
                for t in background_tasks:
                    if not t.done():
                        t.cancel()
                background_tasks.clear()
                await manager.broadcast(json.dumps({"type": "clear_graph"}))
                await manager.send_state()

            elif action == "set_target":
                ch = cmd.get("channel", "").strip()
                target_channel = ch if (ch in channels_set and target_channel != ch) else None
                await manager.send_state()

            elif action == "set_mode":
                similarity_mode = cmd.get("mode", "similar")
                # Schimbarea modului invalideaza toate edges — fortam re-analiza completa
                edges_data.clear()
                dirty_channels.update(channels_set)
                await manager.send_state()

            elif action == "start":
                if not running:
                    running = True
                    paused  = False
                    t1 = asyncio.create_task(background_scraper())
                    t2 = asyncio.create_task(background_analyzer())
                    background_tasks.extend([t1, t2])
                await manager.send_state()

            elif action == "stop":
                running = False
                await manager.send_state()

            elif action == "pause":
                paused = not paused
                await manager.send_state()

            elif action == "save_request":  # Bug #10 fix: Schimbat din "save_request" în "save_state" pentru consistență
                payload = {
                    "channels": list(channels_set),
                    "edges": [
                        {"from": k[0], "to": k[1], "strength": v}
                        for k, v in edges_data.items()
                    ],
                    "entities": global_entities,
                    "posts_history": posts_history[-50:],  # Limităm la ultimele 50
                }
                await manager.broadcast(json.dumps({"type": "save_file", "data": payload}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ─────────────────────────────────────────────
# Background scraper
# ─────────────────────────────────────────────
async def background_scraper():
    """
    Scraper paralel cu actualizare progresiva a UI.
    Marcheaza canalele modificate ca 'dirty' → analizatorul stie
    exact ce perechi sa re-evalueze in ciclul urmator.
    """
    global running, paused, channels_set, nodes_data
    global ch_msgs_cache, ch_embs_cache, ch_style_cache, dirty_channels

    # Cu 204 canale si timeout 8s/canal, semaphore(5) = ~5 canale simultan.
    # Anterior semaphore(10) nu ajuta: tot 20 batch-uri × 8s = 160s.
    semaphore = asyncio.Semaphore(5)
    # Bug #2 fix: Adăugat BATCH_SIZE
    BATCH_SIZE = 20

    async def process_single_channel(ch: str):
        async with semaphore:
            if not running or paused:
                return
            try:
                await manager.broadcast(json.dumps({"type": "status", "msg": f"Scanez {ch}..."}))
                # Timeout explicit per canal: nu blocam tot ciclul daca un canal e lent
                data = await asyncio.wait_for(
                    asyncio.to_thread(scrape_channel, ch),
                    timeout=12.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"[Scraper] Timeout pentru {ch}, sarit.")
                return
            except Exception as e:
                logger.error(f"[Scraper] Eroare {ch}: {e}")
                return

            nodes_data[ch] = {
                "id": ch,
                "label": data["title"],
                "subscribers": data["subscribers"],
            }

            msgs = data.get("messages", [])
            if not msgs:
                await manager.broadcast(json.dumps({"type": "node_update", "node": nodes_data[ch]}))
                return

            # Detectare limba si amprenta stilometrica pe setul complet
            ch_lang_cache[ch]  = detect_language(" ".join(msgs[:10]))
            ch_style_cache[ch] = await asyncio.to_thread(get_stylometric_fingerprint, msgs)

            # Bug #6 fix: Verificăm dacă ch_msgs_set[ch] există
            existing_set = ch_msgs_set.get(ch)
            if existing_set is None:
                existing_set = set()
                ch_msgs_set[ch] = existing_set

            filtered     = (
                [m for m in msgs if text_matches_keywords(m, keywords_list)]
                if keywords_list else msgs
            )
            new_msgs = [m for m in filtered if m not in existing_set]

            if new_msgs:
                # Actualizam cache-ul de text (fereastra glisanta de 50)
                combined = (ch_msgs_cache.get(ch, []) + new_msgs)[-50:]
                ch_msgs_cache[ch] = combined
                # Mentinem setul paralel in sync
                ch_msgs_set[ch] = set(combined)

                # Encoding INCREMENTAL: doar mesajele noi sunt encodate
                # Mesajele vechi din ch_embs_cache[ch]["matrix"] sunt refolosite ca atare
                await asyncio.to_thread(update_embeddings_incremental, ch, new_msgs)

                # Pre-warm NLP cache pentru mesajele noi (primele 5)
                for m in new_msgs[:5]:
                    await asyncio.to_thread(analyse_text, m)

                # Marcam canalul dirty: analizatorul va re-evalua toate perechile cu acest canal
                dirty_channels.add(ch)
                logger.info(f"[Scraper] {ch}: {len(new_msgs)} mesaje noi → dirty")

            # Trimitem nodul actualizat in UI imediat dupa scrape
            await manager.broadcast(json.dumps({"type": "node_update", "node": nodes_data[ch]}))

    # Bug #2 fix: nu lansam 204 taskuri simultan in gather.
    # Impartim in batch-uri de BATCH_SIZE si procesam secvential.

    while running:
        if paused or not channels_set or not nlp_ready:
            await asyncio.sleep(2)
            continue

        ch_list = list(channels_set)
        for start in range(0, len(ch_list), BATCH_SIZE):
            if not running or paused:
                break
            batch = ch_list[start:start + BATCH_SIZE]
            await asyncio.gather(*[process_single_channel(ch) for ch in batch])

        await asyncio.sleep(10)

# ─────────────────────────────────────────────
# Background analyzer
# ─────────────────────────────────────────────
async def background_analyzer():
    """
    Analizator global de similaritate cu evaluare selectiva bazata pe dirty tracking.

    Ciclul de analiza:
    1. Decay logaritmic pe TOATE edges existente (relatiile slabe dispar natural).
    2. Re-evaluare SELECTIVA: recalculeaza doar perechile (c1, c2) unde cel putin
       unul e in dirty_channels SAU perechea e noua (nu are inca un edge).
       → Celelalte perechi supravietuiesc prin decay fara recalcul inutil.
    3. Similaritatea e calculata GLOBAL pe intreaga matrice de embeddings —
       fiecare mesaj din c1 e comparat cu fiecare mesaj din c2 (nu doar argmax).
    4. Graful NetworkX e reconstruit si trimis complet in UI dupa fiecare ciclu.
    """
    global running, paused, edges_data, posts_history, dirty_channels
    global channels_set, nodes_data, target_channel, similarity_mode
    global global_entities, nlp_ready, similarity_model, cosine_similarity

    # Bug #3 fix: Adăugat MAX_PAIRS_PER_CYCLE
    MAX_PAIRS_PER_CYCLE = 300

    while running:
        if paused or not channels_set or not nlp_ready or not similarity_model or not cosine_similarity:
            await asyncio.sleep(5)
            continue

        try:
            n_ch    = len(channels_set)
            n_dirty = len(dirty_channels)
            await manager.broadcast(json.dumps({
                "type": "status",
                "msg":  f"Analiza globala: {n_ch} canale, {n_dirty} cu continut nou",
            }))
            logger.info(f"[Analyzer] mod={similarity_mode} canale={n_ch} dirty={n_dirty}")

            # ── 1. Decay logaritmic pe edges existente ────────────────────
            for pair in list(edges_data.keys()):
                edges_data[pair] = _decay_edge(edges_data[pair])

            # ── 2. Re-evaluare selectiva ──────────────────────────────────
            # Bug #9 fix: Filtrăm cheile valide din ch_embs_cache
            all_keys = [k for k in ch_embs_cache.keys() 
                        if ch_embs_cache[k] is not None 
                        and "matrix" in ch_embs_cache[k] 
                        and ch_embs_cache[k]["matrix"].shape[0] > 0]
            threshold     = THRESHOLD.get(similarity_mode, 0.6)
            current_dirty = set(dirty_channels)

            # Bug #1 fix: force_new_pairs logic corectata.
            # Dupa primul ciclu complet, toate perechile au edge si dirty=0 intre
            # cicluri → analizatorul nu mai rula. Acum re-analizam si perechile
            # cu edge slab (< 1.0) care ar putea fi consolidate sau eliminate.
            force_recheck = len(current_dirty) == 0 and len(edges_data) > 0

            # Bug #3 fix: C(204,2) = 20706 perechi = prea mult pentru un singur ciclu.
            # Limitam la MAX_PAIRS_PER_CYCLE perechi per ciclu, prioritizand:
            # 1. Perechile dirty (cel mai important)
            # 2. Perechile noi (fara edge)
            # 3. Perechile cu edge slab (< 1.0) — candidati la eliminare/consolidare

            if len(all_keys) >= 2:
                # Construim lista de perechi prioritizata
                pairs_to_check = []
                for i, c1 in enumerate(all_keys):
                    for c2 in all_keys[i + 1:]:
                        pair        = tuple(sorted([c1, c2]))
                        is_new_pair = pair not in edges_data
                        is_affected = bool(current_dirty & {c1, c2})
                        is_weak     = edges_data.get(pair, 0.0) < 1.0

                        if is_affected:
                            priority = 0   # cel mai urgent
                        elif is_new_pair:
                            priority = 1
                        elif force_recheck and is_weak:
                            priority = 2   # re-verifica relatii slabe
                        else:
                            continue       # sarim perechile stabile cu edge puternic

                        pairs_to_check.append((priority, pair, c1, c2))

                # Sortam dupa prioritate si limitam numarul
                pairs_to_check.sort(key=lambda x: x[0])
                pairs_to_check = pairs_to_check[:MAX_PAIRS_PER_CYCLE]

                for _, pair, c1, c2 in pairs_to_check:
                    if not running:
                        break

                    edge_score, match = await asyncio.to_thread(
                        _analyse_pair_global, c1, c2, similarity_mode, threshold
                    )

                    if edge_score > 0:
                        edges_data[pair] = min(
                            edges_data.get(pair, 0.0) + edge_score, 10.0
                        )
                        if match:
                            match["time"] = datetime.now().strftime("%H:%M:%S")
                            posts_history.append(match)
                            if len(posts_history) > 50:
                                posts_history.pop(0)
                            await manager.broadcast(
                                json.dumps({"type": "new_post_match", "data": match})
                            )

            # ── 3. Eliminam edges sub pragul minim ───────────────────────
            dead = [p for p, s in edges_data.items() if s < 0.15]
            for p in dead:
                edges_data.pop(p, None)

            # ── 4. Consumam dirty_channels dupa procesare ─────────────────
            dirty_channels -= current_dirty

            # ── 5. Construim graful NetworkX ──────────────────────────────
            G = nx.Graph()
            for (u, v), w in edges_data.items():
                G.add_edge(u, v, weight=w)
            for c in list(channels_set):
                if c not in G:
                    G.add_node(c)

            comms = {}
            try:
                from networkx.algorithms.community import louvain_communities
                if G.size() > 0:
                    for idx, s in enumerate(louvain_communities(G, weight="weight")):
                        for n in s:
                            comms[n] = idx
            except Exception as e:
                logger.debug(f"Community error: {e}")

            deg, btw, cls, pgr, eig = {}, {}, {}, {}, {}
            if G.size() > 0:
                try: deg = nx.degree_centrality(G)
                except Exception: pass
                try: btw = nx.betweenness_centrality(G, weight="weight")
                except Exception: pass
                try: cls = nx.closeness_centrality(G)
                except Exception: pass
                try: pgr = nx.pagerank(G, weight="weight")
                except Exception: pass
                try: eig = nx.eigenvector_centrality_numpy(G, weight="weight")
                except Exception: pass

            displayed = set(G.nodes())
            if target_channel and target_channel in G and G.degree(target_channel) > 0:
                try:
                    displayed = nx.node_connected_component(G, target_channel)
                except Exception:
                    pass

            f_nodes = []
            for n in displayed:
                if n not in nodes_data:
                    continue
                nd = nodes_data[n].copy()
                nd.update({
                    "community": comms.get(n, 0),
                    "metrics": {
                        "degree":      deg.get(n, 0),
                        "betweenness": btw.get(n, 0),
                        "closeness":   cls.get(n, 0),
                        "eigenvector": eig.get(n, 0),
                        "diffusion":   pgr.get(n, 0),
                    },
                })
                f_nodes.append(nd)

            f_edges = [
                {"from": k[0], "to": k[1], "value": round(v, 3), "title": f"Forta: {v:.2f}"}
                for k, v in edges_data.items()
                if k[0] in displayed and k[1] in displayed
            ]

            await manager.broadcast(json.dumps({
                "type": "graph_update",
                "nodes": f_nodes,
                "edges": f_edges,
            }))

        except Exception as e:
            logger.error(f"[Analyzer] Eroare: {e}", exc_info=True)

        await asyncio.sleep(10)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
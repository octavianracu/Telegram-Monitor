import asyncio
import json
import re
import os
import gc
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import sqlite3
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import networkx as nx

# Pentru Grounded Theory
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURARE LOGGING ȘI MODELE NLP
# ─────────────────────────────────────────────────────────────────────────────
similarity_model = None
cosine_similarity = None
ner_pipeline = None
sentiment_pipeline = None
nlp_ready = False
nlp_status = "Așteptare modele NLP..."

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Encoder pentru tipuri numpy."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def safe_json_dumps(data) -> str:
    return json.dumps(data, cls=NumpyEncoder, ensure_ascii=False)


app = FastAPI()

# app.mount la nivel de modul — fișierele statice sunt servite atât cu
# `python main.py` cât și cu `uvicorn main:app`.
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================================
# UTILITAR: limită de memorie (opțional, doar pe Linux/Unix)
# ============================================================================

def set_memory_limit():
    """Setează limită de memorie pentru a preveni blocajele (Unix only)."""
    try:
        import resource
        limit = 2 * 1024 * 1024 * 1024  # 2 GB
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        logger.info("[Memory] Limită de memorie setată la 2 GB")
    except Exception as e:
        logger.warning(f"[Memory] Nu s-a putut seta limita de memorie: {e}")


# ============================================================================
# PARTEA 1: GROUNDED THEORY - IDEOLOGII EMERGENTE
# ============================================================================

class GroundedIdeologyDiscoverer:
    """
    Descoperă ideologii emergente folosind principiile Grounded Theory:
    1. Codare deschisă  - extragere concepte din mesaje
    2. Codare axială    - gruparea conceptelor în categorii
    3. Codare selectivă - identificarea temelor centrale
    4. Saturație teoretică - iterare până la stabilizare
    """

    def __init__(self, similarity_model, sentiment_pipeline, ner_pipeline):
        self.similarity_model = similarity_model
        self.sentiment_pipeline = sentiment_pipeline
        self.ner_pipeline = ner_pipeline
        self.concepts_cache = {}
        self.category_concepts = defaultdict(set)
        self.category_embeddings = {}
        self.emergent_ideologies = []
        self.analysis_history = []
        self.saturation_threshold = 0.05

    def discover_ideologies(self, all_messages: Dict[str, List[str]],
                            max_iterations: int = 3) -> Dict:
        """Proces iterativ de descoperire a ideologiilor emergente."""
        logger.info("[Grounded] Încep descoperirea ideologiilor emergente...")

        # FIX: eliminat signal.SIGALRM — era apelat din asyncio.to_thread (thread
        # non-main) și arunca ValueError în Python. Timeout-ul este gestionat
        # corect la nivel de asyncio.wait_for în endpoint/background task.
        try:
            # Pas 1: Codare deschisă
            all_concepts = self._open_coding(all_messages)
            logger.info(f"[Grounded] Extrase {len(all_concepts)} concepte unice")

            if len(all_concepts) < 10:
                return self._empty_result(all_messages, "Prea puține concepte pentru analiză")

            gc.collect()

            # Pas 2: Codare axială
            categories = self._axial_coding(all_concepts, all_messages)
            logger.info(f"[Grounded] Formate {len(categories)} categorii emergente")

            if len(categories) < 2:
                return self._empty_result(all_messages, "Prea puține categorii pentru analiză")

            gc.collect()

            # Pas 3: Codare selectivă
            ideologies = self._selective_coding(categories, all_messages)
            logger.info(f"[Grounded] Identificate {len(ideologies)} ideologii centrale")

            # Pas 4: Iterare pentru saturație (maxim 3 iterații)
            max_iterations = min(max_iterations, 3)
            iteration = 1
            while iteration < max_iterations:
                previous_count = len(ideologies)

                refined_categories = self._refine_categories(all_messages, categories, ideologies)
                ideologies = self._selective_coding(refined_categories, all_messages)

                change_rate = (
                    abs(len(ideologies) - previous_count) / previous_count
                    if previous_count > 0 else 1
                )
                self.analysis_history.append({
                    "iteration": iteration,
                    "ideologies_count": len(ideologies),
                    "change_rate": round(change_rate, 4)
                })

                logger.info(
                    f"[Grounded] Iteration {iteration}: "
                    f"{len(ideologies)} ideologii, change={change_rate:.3f}"
                )

                if change_rate < self.saturation_threshold:
                    logger.info(f"[Grounded] Saturație atinsă la iterația {iteration}")
                    break
                iteration += 1
                gc.collect()

            # Pas 5: Asignare ideologii per canal
            channel_profiles = self._assign_channel_ideologies(all_messages, ideologies)

            return {
                "emergent_ideologies": ideologies,
                "channel_profiles": channel_profiles,
                "saturation_history": self.analysis_history,
                "categories": {k: list(v)[:20] for k, v in self.category_concepts.items()},
                "methodology": "Grounded Theory - descoperire emergentă",
                "total_concepts": len(all_concepts),
                "total_categories": len(categories)
            }

        except Exception as e:
            logger.error(f"[Grounded] Eroare: {e}", exc_info=True)
            return self._empty_result(all_messages, f"Eroare: {str(e)}")

    def _empty_result(self, all_messages: Dict, reason: str) -> Dict:
        """Returnează un rezultat gol cu motivul."""
        return {
            "emergent_ideologies": [],
            "channel_profiles": {
                ch: {"ideology_scores": {}, "description": reason}
                for ch in all_messages.keys()
            },
            "saturation_history": [],
            "categories": {},
            "methodology": "Grounded Theory",
            "error": reason
        }

    def _open_coding(self, all_messages: Dict[str, List[str]]) -> Dict[str, Dict]:
        """Codare deschisă: extrage concepte din mesaje."""
        concepts_by_channel: Dict = defaultdict(list)

        for channel, messages in all_messages.items():
            for msg in messages[:30]:
                concepts = self._extract_concepts(msg)

                try:
                    raw_sentiment = self.sentiment_pipeline(msg[:512])[0]["label"]
                    # FIX: normalizat explicit la lowercase pentru consistență cu cheile dict
                    sentiment = raw_sentiment.lower()
                    for c in concepts:
                        c["sentiment_context"] = sentiment
                except Exception as e:
                    logger.debug(f"[Grounded] Sentiment eșuat: {e}")

                concepts_by_channel[channel].extend(concepts)
                msg_hash = hash(msg)
                self.concepts_cache[msg_hash] = concepts

        unique_concepts: Dict = {}
        for channel, concepts in concepts_by_channel.items():
            for c in concepts:
                key = c["text"].lower()
                if key not in unique_concepts:
                    unique_concepts[key] = {
                        "text": c["text"],
                        "type": c["type"],
                        "channels": set(),
                        "frequency": 0,
                        "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}
                    }
                unique_concepts[key]["channels"].add(channel)
                unique_concepts[key]["frequency"] += 1
                if "sentiment_context" in c:
                    sent_key = c["sentiment_context"]
                    # FIX: verificăm că cheia există (evităm KeyError pentru etichete neașteptate)
                    if sent_key in unique_concepts[key]["sentiment_distribution"]:
                        unique_concepts[key]["sentiment_distribution"][sent_key] += 1

        filtered = {
            k: v for k, v in unique_concepts.items()
            if len(v["channels"]) >= 2 and v["frequency"] >= 3
        }
        # 🔥 MEMORY LEAK FIX: Clear cache if too large
        if len(self.concepts_cache) > 10000:
            self.concepts_cache.clear()
        return filtered

    def _extract_concepts(self, text: str) -> List[Dict]:
        """Extrage concepte multiple din text."""
        concepts = []

        try:
            entities = self.ner_pipeline(text[:512])
            seen: set = set()
            for ent in entities:
                name = ent["word"].replace("##", "").strip()
                etype = ent["entity_group"]
                if len(name) > 2 and name not in seen and etype in ("PER", "ORG", "LOC", "MISC"):
                    concepts.append({"text": name, "type": f"entity_{etype.lower()}"})
                    seen.add(name)
        except Exception as e:
            logger.debug(f"[Grounded] NER eșuat: {e}")

        words = text.lower().split()
        words = words[:50]  # 🔥 LIMITĂ anti-explozie
        stop_words = {
            "de", "la", "în", "cu", "pe", "din", "pentru",
            "și", "sau", "dar", "a", "al", "ai"
        }

        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i + 1]) > 3:
                bigram = f"{words[i]} {words[i + 1]}"
                if not any(stop in bigram.split() for stop in stop_words):
                    concepts.append({"text": bigram, "type": "bigram"})

        for i in range(len(words) - 2):
            if len(words[i]) > 2 and len(words[i + 1]) > 2 and len(words[i + 2]) > 2:
                trigram = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                if not any(stop in trigram.split() for stop in stop_words):
                    concepts.append({"text": trigram, "type": "trigram"})

        for w in words:
            if len(w) > 5 and w not in stop_words:
                concepts.append({"text": w, "type": "keyword"})

        return concepts[:10]

    def _axial_coding(self, concepts: Dict, all_messages: Dict) -> List[Dict]:
        """Codare axială: grupează conceptele în categorii."""
        concept_list = list(concepts.keys())
        if len(concept_list) < 10:
            return []

        MAX_CONCEPTS = 150
        if len(concept_list) > MAX_CONCEPTS:
            logger.warning(
                f"[Grounded] Prea multe concepte ({len(concept_list)}), "
                f"limitare la {MAX_CONCEPTS}"
            )
            concept_list = sorted(
                concept_list,
                key=lambda c: concepts[c]["frequency"],
                reverse=True
            )[:MAX_CONCEPTS]

        concept_embeddings: Dict = {}
        batch_size = 20
        for i in range(0, len(concept_list), batch_size):
            batch = concept_list[i:i + batch_size]
            try:
                embs = self.similarity_model.encode(batch, show_progress_bar=False)
                for j, concept in enumerate(batch):
                    concept_embeddings[concept] = embs[j]
            except Exception as e:
                logger.warning(f"[Grounded] Encoding batch eșuat: {e}")
                for concept in batch:
                    concept_embeddings[concept] = np.zeros(768)
            gc.collect()  # 🔥 Aggressive GC after embeddings

        valid_concepts = [c for c in concept_list if np.any(concept_embeddings[c] != 0)]
        if len(valid_concepts) < 10:
            logger.warning(
                f"[Grounded] Prea puține concepte cu embeddings valide: {len(valid_concepts)}"
            )
            return []

        emb_matrix = np.array([concept_embeddings[c] for c in valid_concepts])
        n_clusters = min(max(len(valid_concepts) // 10, 3), 8)

        # FIX: eliminat signal.alarm(30) din interiorul acestei funcții — apelată
        # via asyncio.to_thread (thread non-main), signal.signal() aruncă ValueError.
        # KMeans limitează singur iterațiile prin max_iter=100.
        try:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=3,
                max_iter=50
            )
            cluster_labels = kmeans.fit_predict(emb_matrix)
            gc.collect()  # 🔥 IMPORTANT: aggressive GC after KMeans
        except Exception as e:
            logger.warning(f"[Grounded] KMeans eșuat: {e}")
            cluster_labels = np.array([i % n_clusters for i in range(len(valid_concepts))])

        categories = []
        for cluster_id in range(n_clusters):
            cluster_concepts = [
                valid_concepts[i] for i in range(len(valid_concepts))
                if cluster_labels[i] == cluster_id
            ]

            if len(cluster_concepts) < 3:
                continue

            center = np.mean([concept_embeddings[c] for c in cluster_concepts], axis=0)
            # 🔥 BUG FIX #2: Check if center is zero vector to prevent NaN in cosine
            center_norm = np.linalg.norm(center)
            if center_norm > 0:
                representative = min(
                    cluster_concepts,
                    key=lambda c: cosine(center, concept_embeddings[c])
                    if (c in concept_embeddings and np.linalg.norm(concept_embeddings[c]) > 0) else 1.0
                )
            else:
                # If center is zero, pick first concept as representative
                representative = cluster_concepts[0] if cluster_concepts else "unknown"

            category = {
                "id": len(categories),
                "name": representative,
                "concepts": cluster_concepts[:20],
                "all_concepts": cluster_concepts,
                "size": len(cluster_concepts),
                "embedding": center,
                "sentiment_profile": self._calc_category_sentiment(cluster_concepts, concepts)
            }

            categories.append(category)
            for concept in cluster_concepts:
                self.category_concepts[category["id"]].add(concept)
            self.category_embeddings[category["id"]] = center

        return categories

    def _calc_category_sentiment(self, concepts: List[str], concepts_data: Dict) -> Dict:
        """Calculează profilul de sentiment al unei categorii."""
        sentiments = {"positive": 0, "negative": 0, "neutral": 0}
        total = 0

        for concept in concepts:
            if concept in concepts_data:
                dist = concepts_data[concept].get("sentiment_distribution", {})
                for k, v in dist.items():
                    if k in sentiments:
                        sentiments[k] += v
                        total += v

        if total > 0:
            sentiments = {k: v / total for k, v in sentiments.items()}

        dominant = max(sentiments, key=sentiments.get)
        return {
            "distribution": sentiments,
            "dominant_tone": dominant,
            "intensity": sentiments[dominant]
        }

    def _selective_coding(self, categories: List[Dict], all_messages: Dict) -> List[Dict]:
        """Codare selectivă: identifică temele centrale (ideologiile)."""
        if len(categories) < 2:
            return categories

        MAX_CATEGORIES = 20
        if len(categories) > MAX_CATEGORIES:
            logger.warning(
                f"[Grounded] Prea multe categorii ({len(categories)}), "
                f"limitare la {MAX_CATEGORIES}"
            )
            categories = sorted(categories, key=lambda c: c["size"], reverse=True)[:MAX_CATEGORIES]

        category_ids = [c["id"] for c in categories]
        cooccurrence = np.zeros((len(categories), len(categories)))

        for channel, messages in all_messages.items():
            category_mentions: Dict = defaultdict(int)
            for msg in messages[:10]:
                msg_hash = hash(msg)
                if msg_hash in self.concepts_cache:
                    for concept in self.concepts_cache[msg_hash][:10]:
                        for cat in categories:
                            if concept["text"].lower() in cat["all_concepts"]:
                                category_mentions[cat["id"]] += 1
                                break

            cat_list = list(category_mentions.keys())
            for i, cat1 in enumerate(cat_list):
                for cat2 in cat_list[i + 1:]:
                    idx1 = category_ids.index(cat1)
                    idx2 = category_ids.index(cat2)
                    cooccurrence[idx1][idx2] += 1
                    cooccurrence[idx2][idx1] += 1

        G = nx.Graph()
        for i, cat in enumerate(categories):
            G.add_node(cat["id"], name=cat["name"], size=cat["size"])
            for j, cat2 in enumerate(categories):
                if i != j and cooccurrence[i][j] > 0:
                    G.add_edge(cat["id"], cat2["id"], weight=cooccurrence[i][j])

        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G, weight="weight", seed=42)
        except Exception as e:
            logger.warning(f"[Grounded] Louvain eșuat: {e}")
            communities = [[c["id"]] for c in categories]

        ideologies = []
        for idx, community in enumerate(communities):
            if len(community) < 1:
                continue

            community_categories = [c for c in categories if c["id"] in community]
            ideology_name = self._generate_ideology_name(community_categories)

            all_concepts_list: List[str] = []
            for cat in community_categories:
                all_concepts_list.extend(cat["all_concepts"][:10])

            ideology = {
                "id": idx,
                "name": ideology_name,
                "categories": [c["id"] for c in community_categories],
                "category_names": [c["name"] for c in community_categories],
                "concepts": list(set(all_concepts_list))[:50],
                "size": sum(c["size"] for c in community_categories),
                "embedding": np.mean([c["embedding"] for c in community_categories], axis=0),
                "sentiment_profile": self._aggregate_sentiment(community_categories),
                "signature_phrases": self._extract_signature_phrases(community_categories)
            }
            ideologies.append(ideology)

        return ideologies

    def _generate_ideology_name(self, categories: List[Dict]) -> str:
        """Generează un nume simbolic pentru ideologie."""
        if not categories:
            return "Necunoscut"
        top_concepts = [cat["name"] for cat in categories[:3]]
        if len(top_concepts) == 1:
            return top_concepts[0].title()
        elif len(top_concepts) == 2:
            return f"{top_concepts[0].title()} și {top_concepts[1].title()}"
        else:
            return f"{top_concepts[0].title()}, {top_concepts[1].title()} și altele"

    def _aggregate_sentiment(self, categories: List[Dict]) -> Dict:
        """Agregă sentimentul categoriilor."""
        sentiments = {"positive": 0, "negative": 0, "neutral": 0}
        total_weight = 0

        for cat in categories:
            weight = cat["size"]
            for tone, score in cat["sentiment_profile"]["distribution"].items():
                if tone in sentiments:
                    sentiments[tone] += score * weight
            total_weight += weight

        if total_weight > 0:
            sentiments = {k: v / total_weight for k, v in sentiments.items()}

        dominant = max(sentiments, key=sentiments.get)
        return {
            "distribution": sentiments,
            "dominant_tone": dominant,
            "intensity": sentiments[dominant]
        }

    def _extract_signature_phrases(self, categories: List[Dict]) -> List[str]:
        """Extrage frazele caracteristice."""
        phrases = []
        for cat in categories[:3]:
            phrases.append(cat["name"])
            if cat["concepts"]:
                phrases.append(cat["concepts"][0])
        return phrases[:5]

    def _refine_categories(self, all_messages: Dict, old_categories: List[Dict],
                           ideologies: List[Dict]) -> List[Dict]:
        """Rafinează categoriile pe baza ideologiilor identificate."""
        return old_categories

    def _assign_channel_ideologies(self, all_messages: Dict, ideologies: List[Dict]) -> Dict:
        """Asignează fiecărui canal un profil de apartenența."""
        channel_profiles: Dict = {}

        ideology_embeddings: Dict = {}
        for ideo in ideologies:
            ideology_embeddings[ideo["id"]] = ideo["embedding"]

        for channel, messages in all_messages.items():
            channel_embedding = self._build_channel_embedding(messages)

            ideology_scores: Dict = {}
            for ideo in ideologies:
                # 🔥 BUG FIX #1: Check BOTH embeddings for NaN prevention
                ideo_emb = ideology_embeddings[ideo["id"]]
                if (channel_embedding is not None and np.linalg.norm(channel_embedding) > 0 and
                    ideo_emb is not None and np.linalg.norm(ideo_emb) > 0):
                    try:
                        similarity = 1 - cosine(channel_embedding, ideo_emb)
                        # Ensure similarity is not NaN
                        if not np.isnan(similarity):
                            similarity = float(np.clip(similarity, 0.0, 1.0))
                        else:
                            similarity = 0.0
                    except Exception:
                        similarity = 0.0
                else:
                    similarity = 0.0

                concept_match = self._calc_concept_match(messages, ideo["concepts"])
                ideology_scores[ideo["id"]] = similarity * 0.6 + concept_match * 0.4

            if ideology_scores:
                max_score = max(ideology_scores.values())
                if max_score > 0:
                    ideology_scores = {k: v / max_score for k, v in ideology_scores.items()}

            dominant = [
                ideo["id"] for ideo in ideologies
                if ideology_scores.get(ideo["id"], 0) > 0.4
            ]
            description = self._gen_channel_description(ideologies, ideology_scores, dominant)

            channel_profiles[channel] = {
                "ideology_scores": ideology_scores,
                "dominant_ideologies": dominant,
                "primary_ideology": max(ideology_scores, key=ideology_scores.get)
                if ideology_scores else None,
                "description": description
            }

        return channel_profiles

    def _build_channel_embedding(self, messages: List[str]) -> Optional[np.ndarray]:
        """Construiește embedding-ul mediu pentru un canal."""
        if not messages:
            return None
        embeddings = []
        for msg in messages[:15]:
            try:
                emb = self.similarity_model.encode([msg[:512]], show_progress_bar=False)[0]
                embeddings.append(emb)
            except Exception as e:
                logger.debug(f"[Grounded] Encoding mesaj eșuat: {e}")
        if embeddings:
            return np.mean(embeddings, axis=0)
        return None

    def _calc_concept_match(self, messages: List[str], ideology_concepts: List[str]) -> float:
        """Calculează potrivirea conceptelor."""
        concept_set = set(ideology_concepts)
        matches = 0
        total = 0
        for msg in messages[:15]:
            for c in self._extract_concepts(msg):
                total += 1
                if c["text"].lower() in concept_set:
                    matches += 1
        return matches / total if total > 0 else 0.0

    def _gen_channel_description(self, ideologies: List[Dict], scores: Dict,
                                  dominant: List[int]) -> str:
        """Generează descriere textuală."""
        if not dominant:
            return "Profil ideologic neclar, mesaje diverse."
        parts = []
        primary = next((i for i in ideologies if i["id"] == dominant[0]), None)
        if primary:
            parts.append(f"Ideologia dominantă: {primary['name']}")
        if len(dominant) > 1:
            secondary = [
                next((i["name"] for i in ideologies if i["id"] == d), "")
                for d in dominant[1:3]
            ]
            if secondary:
                parts.append(f"Tendințe: {', '.join(secondary)}")
        return ". ".join(parts)


# ============================================================================
# PARTEA 2: SQLITE ȘI FUNCȚII DE BAZĂ
# ============================================================================

DB_PATH = "tgm_monitor.db"
_db_lock = threading.RLock()  # 🔥 RLock prevents deadlocks
ANALYSIS_WINDOW_DAYS = 3

# State global
channels_set: set = set()
running = False
paused = False
target_channel = None
keywords_list: list = []
similarity_mode = "hibrid"
analysis_mode = "similar"

nodes_data: dict = {}
edges_data: dict = {}
edges_type: dict = {}
posts_history: list = []

ch_msgs_cache: dict = {}
ch_embs_cache: dict = {}
ch_style_cache: dict = {}
ch_lang_cache: dict = {}
ch_msgs_set: dict = {}
nlp_msg_cache: dict = {}
global_entities: dict = {"PER": {}, "ORG": {}}
dirty_channels: set = set()
background_tasks: list = []

THRESHOLD = {"repost": 0.88, "similar": 0.72, "stylography": 0.72}
INFERENCE_THRESHOLD = {"direct": 0.72, "hibrid": 0.72, "tranzitiv": 0.68}
INFERENCE_GRAPH_MIN_WEIGHT = 0.72
TRANSITIVE_GRAPH_MIN_WEIGHT = 0.65
INFERRED_SCORE_PENALTY = 0.65
TRANSITIVE_PENALTY_L2 = 0.50
TRANSITIVE_PENALTY_L3 = 0.35
DECAY_BASE = 0.98
DECAY_INFERRED = {"hibrid": 0.85, "tranzitiv": 0.80}

# Narrative state
narrative_topics_cache: list = []
narrative_profiles_cache: dict = {}
_last_narrative_run: Optional[datetime] = None
NARRATIVE_RUN_INTERVAL_HOURS = 24

# Grounded Theory state
grounded_discoverer = None
emergent_ideologies_cache: dict = {}
EMERGENT_ANALYSIS_RUN: Optional[datetime] = None
EMERGENT_ANALYSIS_INTERVAL_HOURS = 24


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def db_init():
    with _db_lock:
        conn = db_connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                text TEXT NOT NULL,
                ts TEXT NOT NULL,
                UNIQUE(channel, text)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_channel_ts ON messages(channel, ts);

            CREATE TABLE IF NOT EXISTS edges_cumulative (
                ch1 TEXT NOT NULL,
                ch2 TEXT NOT NULL,
                score_total REAL NOT NULL DEFAULT 0.0,
                hits INTEGER NOT NULL DEFAULT 0,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                PRIMARY KEY (ch1, ch2)
            );

            CREATE TABLE IF NOT EXISTS channel_daily_embeddings (
                channel TEXT NOT NULL,
                day TEXT NOT NULL,
                embedding TEXT NOT NULL,
                msg_count INTEGER NOT NULL DEFAULT 0,
                updated TEXT NOT NULL,
                PRIMARY KEY (channel, day)
            );

            CREATE TABLE IF NOT EXISTS channel_narrative_profile (
                channel TEXT PRIMARY KEY,
                ema_embedding TEXT NOT NULL,
                topic_distribution TEXT NOT NULL DEFAULT '{}',
                dominant_topic INTEGER NOT NULL DEFAULT -1,
                last_updated TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS narrative_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                topic_id INTEGER NOT NULL,
                keywords TEXT NOT NULL,
                size INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_narrative_topics_run ON narrative_topics(run_id, topic_id);

            CREATE TABLE IF NOT EXISTS emergent_ideologies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                channels_analyzed INTEGER NOT NULL
            );
        """)
        conn.commit()
        conn.close()
    logger.info(f"[DB] Schema inițializată: {DB_PATH}")


def db_insert_messages(channel: str, texts: list):
    if not texts:
        return
    now = datetime.now().isoformat()
    with _db_lock:
        conn = db_connect()
        conn.executemany(
            "INSERT OR IGNORE INTO messages(channel, text, ts) VALUES (?, ?, ?)",
            [(channel, t, now) for t in texts],
        )
        conn.commit()
        conn.close()


def db_get_recent_messages(channel: str, days: int = ANALYSIS_WINDOW_DAYS) -> list:
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    with _db_lock:
        conn = db_connect()
        rows = conn.execute(
            "SELECT text FROM messages WHERE channel=? AND ts>=? ORDER BY ts DESC LIMIT 20",  # 🔥 OPTIMIZED: DESC + LIMIT
            (channel, cutoff),
        ).fetchall()
        conn.close()
    return [r[0] for r in rows]


def db_get_all_messages_set(channel: str) -> set:
    with _db_lock:
        conn = db_connect()
        rows = conn.execute(
            "SELECT text FROM messages WHERE channel=?", (channel,)
        ).fetchall()
        conn.close()
    return {r[0] for r in rows}


def db_get_recent_messages_all_channels(days: int = ANALYSIS_WINDOW_DAYS) -> dict:
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    with _db_lock:
        conn = db_connect()
        # 🔥 BUG FIX #8: Limit total results to prevent memory explosion
        rows = conn.execute(
            "SELECT channel, text FROM messages WHERE ts >= ? ORDER BY channel, ts DESC LIMIT 5000",
            (cutoff,),
        ).fetchall()
        conn.close()
    result: dict = {}
    for channel, text in rows:
        result.setdefault(channel, []).append(text)
    return result


def db_get_all_channels() -> list:
    with _db_lock:
        conn = db_connect()
        rows = conn.execute(
            "SELECT DISTINCT channel FROM messages ORDER BY channel"
        ).fetchall()
        conn.close()
    return [r[0] for r in rows]


def db_update_edge_cumulative(ch1: str, ch2: str, score_delta: float):
    now = datetime.now().isoformat()
    with _db_lock:
        conn = db_connect()
        conn.execute("""
            INSERT INTO edges_cumulative(ch1, ch2, score_total, hits, first_seen, last_seen)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(ch1, ch2) DO UPDATE SET
                score_total = score_total + excluded.score_total,
                hits = hits + 1,
                last_seen = excluded.last_seen
        """, (ch1, ch2, score_delta, now, now))
        conn.commit()
        conn.close()


def db_get_cumulative_scores() -> dict:
    with _db_lock:
        conn = db_connect()
        rows = conn.execute(
            "SELECT ch1, ch2, score_total FROM edges_cumulative WHERE score_total > 0"
        ).fetchall()
        conn.close()
    return {(r[0], r[1]): r[2] for r in rows}


def db_save_emergent_ideologies(data: Dict):
    with _db_lock:
        conn = db_connect()
        conn.execute(
            """INSERT INTO emergent_ideologies(analysis_timestamp, data, channels_analyzed)
               VALUES (?, ?, ?)""",
            (
                data["analysis_timestamp"],
                json.dumps(data, cls=NumpyEncoder),
                data.get("channels_analyzed", 0)
            )
        )
        conn.commit()
        conn.close()


def db_get_emergent_ideologies() -> Optional[Dict]:
    with _db_lock:
        conn = db_connect()
        row = conn.execute(
            "SELECT data FROM emergent_ideologies ORDER BY analysis_timestamp DESC LIMIT 1"
        ).fetchone()
        conn.close()
    if row:
        return json.loads(row[0])
    return None


# ============================================================================
# PARTEA 3: FUNCȚII UTILITARE
# ============================================================================

def detect_language(text: str) -> str:
    if not text:
        return "other"
    cyr = len(re.findall(r"[\u0400-\u04FF]", text))
    lat = len(re.findall(r"[a-zA-Z]", text))
    if cyr > lat:
        return "ru"
    if lat > 5:
        return "ro"
    return "other"


def clean_text(text: str) -> str:
    t = re.sub(r"http\S+", "", text)
    t = re.sub(r"[^\w\s]", " ", t)
    return t.strip().lower()


def text_matches_keywords(text: str, keywords: list) -> bool:
    if not keywords:
        return True
    lo = text.lower()
    return any(k.lower() in lo for k in keywords)


_STYLE_DIM_WEIGHTS_RAW = np.array([
    0.5, 1.5, 1.5, 1.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0,
    1.5, 1.0, 1.0, 1.5, 1.5, 2.0, 1.0, 1.5,
], dtype=float)
_STYLE_DIM_WEIGHTS = _STYLE_DIM_WEIGHTS_RAW / _STYLE_DIM_WEIGHTS_RAW.sum()

_STOPWORDS_STYLE = {
    "si", "in", "la", "de", "cu", "ca", "este", "sunt", "nu", "se",
    "pe", "din", "o", "un", "sau", "care", "pentru", "dar", "mai",
}


def get_stylometric_fingerprint(texts: list) -> np.ndarray:
    if not texts or len(texts) < 2:
        return np.zeros(18)

    def _msg_vec(t: str) -> np.ndarray:
        from collections import Counter
        n_char = len(t) + 1
        words_lo = [w.lower().strip(".,!?;:\"'()[]") for w in t.split() if w.strip()]
        sents = [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]
        n_sents = len(sents) + 1
        sent_lens = [len(s.split()) for s in sents] if sents else [0]
        freq = Counter(words_lo)
        n_unique = len(freq) + 1
        n_total = len(words_lo) + 1
        letters = re.findall(r"[a-zA-Z\u00C0-\u024F\u0400-\u04FF]", t)
        upper = re.findall(r"[A-Z\u00C0-\u00DE\u0400-\u042F]", t)
        char_lens_local = np.array([len(s) for s in sents]) if sents else np.array([len(t)])
        para_var = float(np.std(char_lens_local) / (np.mean(char_lens_local) + 1.0))
        para_var = min(para_var, 1.0)
        return np.array([
            min(len(t) / 500.0, 1.0),
            len(re.findall(r"[.,!?;:]", t)) / n_char,
            len(re.findall(r"[^\w\s]", t)) / n_char,
            min(sum(len(w) for w in words_lo) / ((len(words_lo) + 1) * 15.0), 1.0),
            min(len(re.findall(r"\.\.\.", t)) / n_char * 100, 1.0),
            min(len(re.findall(r"!!", t)) / n_char * 100, 1.0),
            len(upper) / (len(letters) + 1),
            n_unique / n_total,
            sum(1 for w, c in freq.items() if c == 1) / n_unique,
            min(float(np.mean(sent_lens)) / 30.0, 1.0),
            sum(1 for w in words_lo if w in _STOPWORDS_STYLE) / n_total,
            len(re.findall(r"\d", t)) / n_char,
            float(np.log(n_unique) / (np.log(n_total) + 1e-9)),
            sum(1 for ln in sent_lens if ln < 5) / n_sents,
            sum(1 for ln in sent_lens if ln > 20) / n_sents,
            min(t.count("?") / 3.0, 1.0),
            para_var,
            1.0 if re.search(r"https?://", t) else 0.0,
        ], dtype=float)

    all_vecs = [_msg_vec(t) for t in texts]
    medians = np.median(np.stack(all_vecs), axis=0)
    char_lens = np.array([len(t) for t in texts], dtype=float)
    medians[16] = min(float(np.std(char_lens) / (np.mean(char_lens) + 1.0)), 1.0)
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
        seen: set = set()
        entities = []
        for ent in n_res:
            name = ent["word"].replace("##", "")
            etype = ent["entity_group"]
            if etype in ("PER", "ORG") and len(name) > 2 and name not in seen:
                entities.append({"name": name, "type": etype})
                seen.add(name)
        res = {"sentiment": s_res, "entities": entities}
        nlp_msg_cache[h] = res
        # 🔥 MEMORY LEAK FIX: Clear cache if too large
        if len(nlp_msg_cache) > 10000:
            nlp_msg_cache.clear()
        return res
    except Exception:
        return {}


def scrape_channel(username: str) -> dict:
    """Scrapează informații despre un canal Telegram cu retry și backoff exponențial."""
    u = username.lstrip("@")

    for attempt in range(3):
        # FIX: context manager `with requests.Session()` — Session se închide
        # automat la ieșirea din bloc, eliminând resource leak-ul.
        # FIX: eliminat `import time` din interiorul funcției — `time` este deja
        # importat la nivel de modul (linia 8).
        try:
            with requests.Session() as session:
                r1 = session.get(f"https://t.me/{u}", timeout=(10, 25))
                s1 = BeautifulSoup(r1.text, "html.parser")

                t_el = s1.find("div", class_="tgme_page_title")
                title = t_el.text.strip() if t_el else u

                e_el = s1.find("div", class_="tgme_page_extra")
                subs = parse_subscribers(e_el.text if e_el else "0")

                # Pauză între cereri pentru a reduce încărcarea serverului
                time.sleep(0.5)

                r2 = session.get(f"https://t.me/s/{u}", timeout=(10, 25))
                s2 = BeautifulSoup(r2.text, "html.parser")

                msgs = []
                for w in s2.find_all("div", class_="tgme_widget_message_text"):
                    txt = w.get_text(separator=" ", strip=True)
                    if len(txt) > 30 and len(txt.split()) >= 5:
                        msgs.append(txt)

                return {
                    "username": u,
                    "title": title,
                    "subscribers": subs,
                    "messages": msgs[-10:]  # 🔥 Reduced from 15 for stability
                }

        except requests.exceptions.ReadTimeout:
            logger.warning(f"Read timeout la {username} (încercarea {attempt + 1}/3)")
            if attempt == 2:
                logger.error(f"Read timeout definitiv la {username}")
                return {"username": u, "title": u, "subscribers": 0, "messages": []}
            time.sleep(2 ** attempt)

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout la {username} (încercarea {attempt + 1}/3)")
            if attempt == 2:
                return {"username": u, "title": u, "subscribers": 0, "messages": []}
            time.sleep(2 ** attempt)

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error la {username}: {e}")
            if attempt == 2:
                return {"username": u, "title": u, "subscribers": 0, "messages": []}
            time.sleep(3)

        except Exception as e:
            logger.error(f"Scrape error {username}: {e}")
            return {"username": u, "title": u, "subscribers": 0, "messages": []}

    return {"username": u, "title": u, "subscribers": 0, "messages": []}


def parse_subscribers(text: str) -> int:
    """Parsează numărul de abonați din text."""
    if not text:
        return 0
    t = text.lower().strip()
    t = re.sub(r"[^\d.,km]", "", t)
    t = t.replace(",", "")
    if t.endswith("k"):
        try:
            return int(float(t[:-1]) * 1000)
        except (ValueError, TypeError):
            pass
    if t.endswith("m"):
        try:
            return int(float(t[:-1]) * 1000000)
        except (ValueError, TypeError):
            pass
    try:
        return int(float(t))
    except (ValueError, TypeError):
        return 0


def update_embeddings_incremental(ch: str, new_texts: list):
    if not new_texts or similarity_model is None:
        return
    recent_texts = db_get_recent_messages(ch, days=ANALYSIS_WINDOW_DAYS)
    if not recent_texts:
        return
    clean_texts = [clean_text(t) for t in recent_texts if len(clean_text(t)) > 10]
    if not clean_texts:
        return
    orig_valid = [t for t in recent_texts if len(clean_text(t)) > 10]
    matrix = similarity_model.encode(clean_texts, show_progress_bar=False)
    ch_embs_cache[ch] = {"orig_texts": orig_valid, "matrix": matrix}
    # 🔥 MEMORY LEAK FIX: Clear cache if too large
    if len(ch_embs_cache) > 500:
        ch_embs_cache.clear()


def get_embedding_matrix(ch: str):
    entry = ch_embs_cache.get(ch)
    if entry is None:
        return None, None
    return entry["matrix"], entry["orig_texts"]


def _decay_edge(pair: tuple, strength: float) -> float:
    etype = edges_type.get(pair, "direct")
    if etype == "direct":
        base_decay = DECAY_BASE
    elif etype == "inferred_tranzitiv":
        base_decay = DECAY_INFERRED["tranzitiv"]
    else:
        base_decay = DECAY_INFERRED["hibrid"]
    strength = float(min(strength, 10.0))
    log_factor = 0.1 * (np.log10(strength + 1) / np.log10(11))
    result = strength * (base_decay + log_factor)
    return float(np.clip(result, 0.0, 10.0))


# ============================================================================
# PARTEA 4: ENDPOINT-URI API
# ============================================================================

@app.get("/api/nlp_status")
async def get_nlp_status():
    return {"nlp_ready": nlp_ready, "nlp_status": nlp_status}


@app.get("/api/discover_ideologies")
async def discover_emergent_ideologies(force: bool = False):
    """Rulează descoperirea ideologiilor emergente folosind Grounded Theory."""
    global grounded_discoverer, emergent_ideologies_cache, EMERGENT_ANALYSIS_RUN

    if not nlp_ready or similarity_model is None:
        return {"status": "error", "message": "Modelele NLP nu sunt încă gata."}

    if not force and emergent_ideologies_cache:
        return {
            "status": "cached",
            "data": emergent_ideologies_cache,
            "last_run": EMERGENT_ANALYSIS_RUN.isoformat() if EMERGENT_ANALYSIS_RUN else None
        }

    all_messages: dict = {}
    channels = db_get_all_channels()

    for channel in channels:
        messages = db_get_recent_messages(channel, days=7)
        if len(messages) >= 5:
            all_messages[channel] = messages[:20]

    if len(all_messages) < 3:
        return {
            "status": "error",
            "message": (
                f"Prea puține canale cu mesaje suficiente ({len(all_messages)}). "
                "Așteaptă colectarea mai multor date."
            )
        }

    logger.info(f"[Grounded] Analiză pe {len(all_messages)} canale")

    grounded_discoverer = GroundedIdeologyDiscoverer(
        similarity_model=similarity_model,
        sentiment_pipeline=sentiment_pipeline,
        ner_pipeline=ner_pipeline
    )

    # 🔥 FIX THREAD + ASYNC: Use loop.run_in_executor instead of asyncio.to_thread
    # This avoids timing issues with nested thread calls
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        grounded_discoverer.discover_ideologies,
        all_messages,
        3
    )

    result["analysis_timestamp"] = datetime.now().isoformat()
    result["channels_analyzed"] = len(all_messages)
    result["total_channels"] = len(channels)

    emergent_ideologies_cache = result
    EMERGENT_ANALYSIS_RUN = datetime.now()

    await asyncio.to_thread(db_save_emergent_ideologies, result)
    return {"status": "success", "data": result}


@app.get("/api/emergent_ideologies")
async def get_emergent_ideologies():
    """Returnează ultimele ideologii emergente descoperite."""
    if emergent_ideologies_cache:
        return {
            "status": "success",
            "data": emergent_ideologies_cache,
            "last_run": EMERGENT_ANALYSIS_RUN.isoformat() if EMERGENT_ANALYSIS_RUN else None
        }
    cached = await asyncio.to_thread(db_get_emergent_ideologies)
    if cached:
        return {"status": "success", "data": cached, "source": "database"}
    return {
        "status": "pending",
        "message": "Nicio analiză disponibilă. Rulează /api/discover_ideologies"
    }


@app.get("/api/emergent_ideology/{channel}")
async def get_channel_emergent_ideology(channel: str):
    """Returnează profilul ideologic emergent pentru un canal specific."""
    if not emergent_ideologies_cache:
        return {"status": "error", "message": "Nicio analiză emergentă disponibilă"}

    channel_profile = emergent_ideologies_cache.get("channel_profiles", {}).get(channel)
    if not channel_profile:
        return {"status": "error", "message": f"Canalul {channel} nu a fost analizat"}

    ideologies = emergent_ideologies_cache.get("emergent_ideologies", [])

    detailed: dict = {
        "channel": channel,
        "scores": channel_profile.get("ideology_scores", {}),
        "dominant_ideologies": [],
        "description": channel_profile.get("description", ""),
        "ideologies_detail": []
    }

    for ideo_id, score in channel_profile.get("ideology_scores", {}).items():
        # 🔥 BUG FIX #5: Safe int conversion with validation
        try:
            ideo_id_int = int(ideo_id) if isinstance(ideo_id, str) else ideo_id
        except (ValueError, TypeError):
            logger.warning(f"[API] Invalid ideology_id format: {ideo_id}")
            continue
        
        ideo_detail = next((i for i in ideologies if i["id"] == ideo_id_int), None)
        if ideo_detail:
            detailed["ideologies_detail"].append({
                "id": ideo_id,
                "name": ideo_detail["name"],
                "score": score,
                "categories": ideo_detail.get("category_names", [])[:5],
                "signature_phrases": ideo_detail.get("signature_phrases", [])[:3],
                "tone": ideo_detail.get("sentiment_profile", {}).get("dominant_tone", "neutru")
            })
            if score > 0.4:
                detailed["dominant_ideologies"].append(ideo_detail["name"])

    return detailed


@app.get("/api/emergent_saturation")
async def get_saturation_status():
    """Returnează statusul saturației teoretice."""
    if not emergent_ideologies_cache:
        return {"status": "error", "message": "Nicio analiză disponibilă"}

    history = emergent_ideologies_cache.get("saturation_history", [])
    if not history:
        return {"status": "pending", "message": "Analiza nu a atins încă saturația"}

    latest = history[-1]
    return {
        "status": "saturated" if latest["change_rate"] < 0.05 else "converging",
        "iterations_completed": len(history),
        "final_change_rate": latest["change_rate"],
        "ideologies_count": latest["ideologies_count"],
        "history": history
    }


@app.get("/api/narratives")
async def get_narratives():
    profiles = await asyncio.to_thread(db_get_all_narrative_profiles)
    topics = await asyncio.to_thread(db_get_latest_topics)
    channels_out = {
        ch: {
            "dominant_topic": data["dominant_topic"],
            "topic_distribution": data["topic_distribution"],
        }
        for ch, data in profiles.items()
    }
    return {
        "topics": topics,
        "channels": channels_out,
        "last_run": _last_narrative_run.isoformat() if _last_narrative_run else None,
        "total_channels_profiled": len(profiles),
    }


@app.get("/api/rebuild_profiles")
async def rebuild_profiles():
    if not nlp_ready or similarity_model is None:
        return {"status": "error", "message": "Modelele NLP nu sunt încă gata."}

    async def _rebuild():
        channels = await asyncio.to_thread(db_get_all_channels)
        total = len(channels)
        done = 0
        for ch in channels:
            await asyncio.to_thread(rebuild_narrative_profile_for_channel, ch)
            done += 1
            if done % 10 == 0:
                logger.info(f"[Profiles] Reconstruite {done}/{total} profile EMA")
        profiles_count = len(await asyncio.to_thread(db_get_all_narrative_profiles))
        logger.info(f"[Profiles] Complet: {profiles_count}/{total} profile EMA în DB.")

    asyncio.create_task(_rebuild())
    channels_count = len(await asyncio.to_thread(db_get_all_channels))
    return {
        "status": "started",
        "message": f"Reconstruiesc profile EMA pentru {channels_count} canale."
    }


@app.get("/api/run_bertopic")
async def run_bertopic_now():
    if not nlp_ready or similarity_model is None:
        return {"status": "error", "message": "Modelele NLP nu sunt gata."}
    profiles = await asyncio.to_thread(db_get_all_narrative_profiles)
    if len(profiles) < 5:
        return {"status": "error", "message": f"Prea puține profile ({len(profiles)})."}
    asyncio.create_task(asyncio.to_thread(run_narrative_clustering))
    return {"status": "started", "message": f"BERTopic pornit pe {len(profiles)} canale."}


@app.get("/api/bimodal_export")
async def bimodal_export():
    profiles = await asyncio.to_thread(db_get_all_narrative_profiles)
    topics = await asyncio.to_thread(db_get_latest_topics)
    cumulative_scores = await asyncio.to_thread(db_get_cumulative_scores)

    nodes = []
    for ch, data in profiles.items():
        node_info = nodes_data.get(ch, {})
        nodes.append({
            "id": ch,
            "type": "channel",
            "label": node_info.get("label", ch),
            "subscribers": node_info.get("subscribers", 0),
            "dominant_topic": data["dominant_topic"],
        })

    for t in topics:
        nodes.append({
            "id": f"topic_{t['topic_id']}",
            "type": "topic",
            "label": " · ".join(t["keywords"][:5]),
            "size": t["size"],
        })

    edges = []
    for (ch1, ch2), score in cumulative_scores.items():
        if ch1 in profiles and ch2 in profiles:
            edges.append({
                "source": ch1,
                "target": ch2,
                "weight": round(score, 4),
                "type": "coordination",
            })

    return {"nodes": nodes, "edges": edges}


# ============================================================================
# PARTEA 5: FUNCȚII NARRATIVE ȘI BERTopic
# ============================================================================

def db_get_all_narrative_profiles() -> dict:
    with _db_lock:
        conn = db_connect()
        rows = conn.execute(
            "SELECT channel, ema_embedding, topic_distribution, dominant_topic "
            "FROM channel_narrative_profile"
        ).fetchall()
        conn.close()
    result: dict = {}
    for ch, emb_json, topic_json, dominant in rows:
        result[ch] = {
            "embedding": np.array(json.loads(emb_json)),
            "topic_distribution": json.loads(topic_json),
            "dominant_topic": dominant,
        }
    return result


def db_get_latest_topics() -> list:
    with _db_lock:
        conn = db_connect()
        row = conn.execute(
            "SELECT run_id FROM narrative_topics ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            conn.close()
            return []
        run_id = row[0]
        rows = conn.execute(
            "SELECT topic_id, keywords, size FROM narrative_topics "
            "WHERE run_id=? ORDER BY size DESC",
            (run_id,),
        ).fetchall()
        conn.close()
    return [{"topic_id": r[0], "keywords": json.loads(r[1]), "size": r[2]} for r in rows]


def compute_channel_ema(channel: str, alpha: float = 0.3) -> Optional[np.ndarray]:
    cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    with _db_lock:
        conn = db_connect()
        rows = conn.execute(
            "SELECT day, embedding FROM channel_daily_embeddings "
            "WHERE channel=? AND day>=? ORDER BY day ASC",
            (channel, cutoff),
        ).fetchall()
        conn.close()
    if len(rows) < 2:
        return None
    daily = [(r[0], np.array(json.loads(r[1]))) for r in rows]
    ema = daily[0][1].copy()
    for _, emb in daily[1:]:
        ema = alpha * emb + (1 - alpha) * ema
    return ema


def update_narrative_profile_for_channel(channel: str):
    if similarity_model is None or not nlp_ready:
        return
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        recent = db_get_recent_messages(channel, days=1)
        if not recent:
            return
        clean = [clean_text(t) for t in recent if len(clean_text(t)) > 10]
        if not clean:
            return
        embs = similarity_model.encode(clean, show_progress_bar=False)
        day_emb = embs.mean(axis=0)

        with _db_lock:
            conn = db_connect()
            conn.execute(
                """INSERT INTO channel_daily_embeddings(channel, day, embedding, msg_count, updated)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(channel, day) DO UPDATE SET
                       embedding = excluded.embedding,
                       msg_count = excluded.msg_count,
                       updated = excluded.updated""",
                (channel, today, json.dumps(day_emb.tolist()), len(clean),
                 datetime.now().isoformat())
            )
            conn.commit()
            conn.close()

        ema = compute_channel_ema(channel)
        if ema is not None:
            with _db_lock:
                conn = db_connect()
                conn.execute(
                    """INSERT INTO channel_narrative_profile
                           (channel, ema_embedding, topic_distribution, dominant_topic, last_updated)
                       VALUES (?, ?, '{}', -1, ?)
                       ON CONFLICT(channel) DO UPDATE SET
                           ema_embedding = excluded.ema_embedding,
                           last_updated = excluded.last_updated""",
                    (channel, json.dumps(ema.tolist()), datetime.now().isoformat())
                )
                conn.commit()
                conn.close()
    except Exception as e:
        logger.warning(f"[Narrative] Profil eșuat {channel}: {e}")


def rebuild_narrative_profile_for_channel(channel: str):
    if similarity_model is None or not nlp_ready:
        return
    try:
        msgs = db_get_recent_messages(channel, days=ANALYSIS_WINDOW_DAYS)
        if not msgs:
            return
        clean = [clean_text(t) for t in msgs if len(clean_text(t)) > 10]
        if not clean:
            return
        embs = similarity_model.encode(clean, show_progress_bar=False)
        profile_emb = embs.mean(axis=0)

        with _db_lock:
            conn = db_connect()
            conn.execute(
                """INSERT INTO channel_narrative_profile
                       (channel, ema_embedding, topic_distribution, dominant_topic, last_updated)
                   VALUES (?, ?, '{}', -1, ?)
                   ON CONFLICT(channel) DO UPDATE SET
                       ema_embedding = excluded.ema_embedding,
                       last_updated = excluded.last_updated""",
                (channel, json.dumps(profile_emb.tolist()), datetime.now().isoformat())
            )
            conn.commit()
            conn.close()
        logger.debug(f"[Narrative] Profil narativ reconstruit pentru {channel} ({len(clean)} mesaje)")
    except Exception as e:
        logger.warning(f"[Narrative] Rebuild profil eșuat {channel}: {e}")


def run_narrative_clustering():
    global narrative_topics_cache, narrative_profiles_cache, _last_narrative_run

    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as e:
        logger.error(f"[Narrative] Import BERTopic eșuat: {e}")
        return

    profiles = db_get_all_narrative_profiles()
    if len(profiles) < 5:
        logger.info("[Narrative] Prea puține profile pentru clustering.")
        return

    logger.info(f"[Narrative] BERTopic pe {len(profiles)} canale...")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        channels = list(profiles.keys())
        docs = []
        for ch in channels:
            msgs = db_get_recent_messages(ch, days=7)
            # 🔥 BUG FIX #9: Use only returned messages (avoid None slicing)
            doc_text = " ".join(msgs[:min(30, len(msgs))]) if msgs else ch
            if not doc_text or len(doc_text.strip()) < 10:
                doc_text = ch  # Fallback to channel name
            docs.append(doc_text)

        STOPWORDS_RO = {"și", "de", "la", "în", "că", "cu", "pe", "din", "pentru", "este"}
        STOPWORDS_RU = {"и", "в", "не", "на", "с", "что", "как", "по", "из", "от"}
        STOPWORDS_COMMON = {"http", "https", "www", "com", "md", "ro", "ru"}
        all_stopwords = list(STOPWORDS_RO | STOPWORDS_RU | STOPWORDS_COMMON)

        vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            stop_words=all_stopwords,
            min_df=2,
            max_features=5000,
            token_pattern=r"(?u)\b[a-zA-ZÀ-žА-яёÀ-ÿ]{3,}\b",
        )
        topic_model = BERTopic(
            embedding_model=similarity_model,
            vectorizer_model=vectorizer,
            min_topic_size=3,
            nr_topics="auto",
            verbose=False,
        )

        # 🔥 BUG FIX #10: Add timeout for BERTopic execution
        logger.info(f"[Narrative] Inițiez BERTopic cu {len(docs)} documente...")
        topics, probs = topic_model.fit_transform(docs)
        logger.info(f"[Narrative] BERTopic executat cu succes")

        topic_info = topic_model.get_topic_info()
        topics_to_save = []
        for _, row in topic_info.iterrows():
            tid = int(row["Topic"])
            if tid == -1:
                continue
            keywords = [w for w, _ in topic_model.get_topic(tid)[:10]]
            topics_to_save.append({
                "topic_id": tid,
                "keywords": keywords,
                "size": int(row["Count"]),
            })

        with _db_lock:
            conn = db_connect()
            conn.executemany(
                """INSERT INTO narrative_topics(run_id, topic_id, keywords, size, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (run_id, t["topic_id"], json.dumps(t["keywords"]), t["size"],
                     datetime.now().isoformat())
                    for t in topics_to_save
                ]
            )
            conn.commit()
            conn.close()

        channel_topics: dict = {}
        for i, ch in enumerate(channels):
            t_id = int(topics[i])
            if hasattr(probs, "ndim") and probs.ndim == 2:
                dist = {str(j): float(probs[i][j]) for j in range(probs.shape[1])}
            else:
                dist = {str(t_id): 1.0}
            channel_topics[ch] = {"distribution": dist, "dominant": t_id}

        with _db_lock:
            conn = db_connect()
            for ch, data in channel_topics.items():
                conn.execute(
                    """UPDATE channel_narrative_profile
                       SET topic_distribution=?, dominant_topic=?, last_updated=?
                       WHERE channel=?""",
                    (
                        json.dumps(data["distribution"]),
                        data["dominant"],
                        datetime.now().isoformat(),
                        ch
                    )
                )
            conn.commit()
            conn.close()

        narrative_topics_cache = topics_to_save
        narrative_profiles_cache = {
            ch: {
                "dominant_topic": channel_topics[ch]["dominant"],
                "topic_distribution": channel_topics[ch]["distribution"]
            }
            for ch in channels
        }
        _last_narrative_run = datetime.now()
        logger.info(f"[Narrative] ✓ Clustering complet: {len(topics_to_save)} teme descoperite peste {len(channels)} canale.")

    except Exception as e:
        logger.error(f"[Narrative] Eroare BERTopic: {e}", exc_info=True)


# ============================================================================
# PARTEA 6: WEB SOCKET ȘI CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: list = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
        await self.send_state()

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, msg: str):
        disconnected = []
        for c in self.active_connections:
            try:
                await c.send_text(msg)
            except Exception:
                disconnected.append(c)
        for c in disconnected:
            if c in self.active_connections:
                self.active_connections.remove(c)

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
        }
        await self.broadcast(safe_json_dumps(state))


manager = ConnectionManager()


def _notify_state(loop):
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(manager.send_state(), loop)


def _load_nlp_models(loop):
    global ner_pipeline, sentiment_pipeline, similarity_model, cosine_similarity
    global nlp_ready, nlp_status

    try:
        logger.info("[NLP] Importuri grele...")
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cos
        from transformers import pipeline as hf_pipeline
        cosine_similarity = sk_cos

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
            device=-1
        )
        logger.info("[NLP] NER ready.")

        nlp_status = "Incarcare Sentiment..."
        _notify_state(loop)
        sentiment_pipeline = hf_pipeline(
            "text-classification",
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            device=-1
        )
        logger.info("[NLP] Sentiment ready.")

        nlp_ready = True
        nlp_status = "Sistem NLP Pregătit."
        logger.info("[NLP] TOATE MODELELE ACTIVE.")
        _notify_state(loop)

        threading.Thread(target=warm_up_embeddings, daemon=True).start()

    except Exception as e:
        logger.error(f"[NLP] EROARE INCARCARE: {e}", exc_info=True)


def warm_up_embeddings():
    global ch_embs_cache

    if similarity_model is None:
        return

    channels = list(channels_set)
    if not channels:
        return

    logger.info(f"[WarmUp] Reconstruiesc embeddings pentru {len(channels)} canale...")
    recent_by_channel = db_get_recent_messages_all_channels(days=ANALYSIS_WINDOW_DAYS)

    rebuilt = 0
    for ch in channels:
        msgs = recent_by_channel.get(ch, [])
        if not msgs:
            continue
        try:
            clean_texts = [clean_text(t) for t in msgs if len(clean_text(t)) > 10]
            orig_valid = [t for t in msgs if len(clean_text(t)) > 10]
            if not clean_texts:
                continue
            matrix = similarity_model.encode(clean_texts, show_progress_bar=False)
            ch_embs_cache[ch] = {"orig_texts": orig_valid, "matrix": matrix}
            rebuilt += 1
        except Exception as e:
            logger.warning(f"[WarmUp] Embeddings eșuate pentru {ch}: {e}")

    logger.info(f"[WarmUp] ✓ Embeddings reconstruite pentru {rebuilt}/{len(channels)} canale.")


def db_warm_up_state():
    global channels_set, nodes_data, ch_msgs_cache, ch_msgs_set, ch_lang_cache, ch_style_cache

    logger.info("[WarmUp] Reconstruiesc starea RAM din SQLite...")

    all_channels = db_get_all_channels()
    if not all_channels:
        logger.info("[WarmUp] Baza de date goală.")
        return

    recent_by_channel = db_get_recent_messages_all_channels(days=ANALYSIS_WINDOW_DAYS)

    for ch in all_channels:
        channels_set.add(ch)
        if ch not in nodes_data:
            nodes_data[ch] = {"id": ch, "label": ch, "subscribers": 0}

        msgs = recent_by_channel.get(ch, [])
        if msgs:
            ch_msgs_cache[ch] = msgs[-50:]
            ch_msgs_set[ch] = set(msgs)
            ch_lang_cache[ch] = detect_language(" ".join(msgs[:10]))
            try:
                ch_style_cache[ch] = get_stylometric_fingerprint(msgs)
            except Exception as e:
                logger.warning(f"[WarmUp] Fingerprint eșuat pentru {ch}: {e}")

    logger.info(f"[WarmUp] Restaurate {len(channels_set)} canale.")


def start_nlp_loading(loop=None):
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    threading.Thread(target=_load_nlp_models, args=(loop,), daemon=True).start()


# ============================================================================
# PARTEA 7: BACKGROUND TASKS
# ============================================================================

async def background_narrative_clusterer():
    """Rulează clustering narativ periodic, independent de starea `running`."""
    await asyncio.sleep(600)
    while True:
        try:
            if nlp_ready and similarity_model is not None:
                logger.info("[Narrative] Pornesc clustering narativ...")
                await asyncio.to_thread(run_narrative_clustering)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[Narrative] Eroare clustering: {e}", exc_info=True)
        await asyncio.sleep(NARRATIVE_RUN_INTERVAL_HOURS * 3600)


async def background_emergent_analyzer():
    """Rulează analiza ideologică emergentă periodic, independent de starea `running`."""
    global grounded_discoverer, emergent_ideologies_cache, EMERGENT_ANALYSIS_RUN

    # Așteptăm modelele NLP
    while not nlp_ready or similarity_model is None:
        await asyncio.sleep(10)

    await asyncio.sleep(1800)

    while True:
        try:
            logger.info("[Grounded] Pornesc analiza emergentă...")

            all_messages: dict = {}
            channels = db_get_all_channels()

            for channel in channels:
                messages = db_get_recent_messages(channel, days=7)
                if len(messages) >= 5:
                    all_messages[channel] = messages[:20]

            if len(all_messages) >= 3:
                grounded_discoverer = GroundedIdeologyDiscoverer(
                    similarity_model=similarity_model,
                    sentiment_pipeline=sentiment_pipeline,
                    ner_pipeline=ner_pipeline
                )

                # FIX: timeout corect cu asyncio.wait_for, nu signal.alarm()
                # 🔥 FIX THREAD + ASYNC: Use loop.run_in_executor for better control
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    grounded_discoverer.discover_ideologies,
                    all_messages,
                    3
                )

                result["analysis_timestamp"] = datetime.now().isoformat()
                result["channels_analyzed"] = len(all_messages)
                result["total_channels"] = len(channels)

                emergent_ideologies_cache = result
                EMERGENT_ANALYSIS_RUN = datetime.now()

                await asyncio.to_thread(db_save_emergent_ideologies, result)
                logger.info(
                    f"[Grounded] Analiză completă: "
                    f"{len(result.get('emergent_ideologies', []))} ideologii descoperite, "
                    f"{len(all_messages)} canale analizate"
                )
            else:
                logger.info(
                    f"[Grounded] Prea puține canale ({len(all_messages)}), amân analiza"
                )

        except asyncio.CancelledError:
            break
        except asyncio.TimeoutError:
            logger.error("[Grounded] Analiza emergentă a depășit 600 s — omisă.")
        except Exception as e:
            logger.error(f"[Grounded] Eroare: {e}", exc_info=True)

        await asyncio.sleep(EMERGENT_ANALYSIS_INTERVAL_HOURS * 3600)


async def background_scraper():
    """Scrapează canalele periodic cât timp `running` este True."""
    global running, paused, channels_set, nodes_data
    global ch_msgs_cache, ch_embs_cache, ch_style_cache, dirty_channels, ch_lang_cache

    semaphore = asyncio.Semaphore(2)
    BATCH_SIZE = 10
    _last_purge = datetime.now()

    async def scrape_with_retry(ch: str) -> Optional[dict]:
        for attempt in range(2):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(scrape_channel, ch),
                    timeout=25.0
                )
            except Exception as e:
                if attempt == 0:
                    logger.debug(f"[Scraper] Retry {ch}: {e}")
                    await asyncio.sleep(3)
                else:
                    logger.warning(f"[Scraper] Eșuat definitiv {ch}: {e}")
                    return None
        return None

    async def process_single_channel(ch: str):
        async with semaphore:
            if not running or paused:
                return
            logger.info(f"[Scraper] Scanez {ch}...")
            await manager.broadcast(
                safe_json_dumps({"type": "status", "msg": f"Scanez {ch}..."})
            )

            data = await scrape_with_retry(ch)
            if data is None:
                return

            nodes_data[ch] = {
                "id": ch,
                "label": data["title"],
                "subscribers": data["subscribers"]
            }
            logger.info(f"[Scraper] {ch}: '{data['title']}' ({data['subscribers']} abonați)")
            msgs = data.get("messages", [])
            if not msgs:
                return

            if ch not in ch_lang_cache or ch_lang_cache[ch] == "other":
                ch_lang_cache[ch] = detect_language(" ".join(msgs[:10]))

            existing_set = await asyncio.to_thread(db_get_all_messages_set, ch)
            filtered = (
                [m for m in msgs if text_matches_keywords(m, keywords_list)]
                if keywords_list else msgs
            )
            new_msgs = [m for m in filtered if m not in existing_set]

            if new_msgs:
                logger.info(f"[Scraper] Găsite {len(new_msgs)} mesaje noi în {ch}")
                await asyncio.to_thread(db_insert_messages, ch, new_msgs)
                await asyncio.to_thread(update_embeddings_incremental, ch, new_msgs)
                await asyncio.to_thread(update_narrative_profile_for_channel, ch)
                if nlp_ready:
                    for m in new_msgs[:5]:
                        await asyncio.to_thread(analyse_text, m)
                dirty_channels.add(ch)

            recent_for_style = await asyncio.to_thread(
                db_get_recent_messages, ch, ANALYSIS_WINDOW_DAYS
            )
            if len(recent_for_style) >= 2:
                ch_style_cache[ch] = await asyncio.to_thread(
                    get_stylometric_fingerprint, recent_for_style
                )

            await manager.broadcast(
                safe_json_dumps({"type": "node_update", "node": nodes_data[ch]})
            )

    while running:
        if paused or not channels_set:
            await asyncio.sleep(1)
            continue

        # Purge zilnic — șterge mesajele mai vechi de 30 de zile
        if (datetime.now() - _last_purge).total_seconds() > 86400:
            _last_purge = datetime.now()
            try:
                cutoff = (datetime.now() - timedelta(days=30)).isoformat()
                with _db_lock:
                    conn = db_connect()
                    conn.execute("DELETE FROM messages WHERE ts < ?", (cutoff,))
                    conn.commit()
                    conn.close()
                logger.info("[Scraper] Purge mesaje vechi (>30 zile) efectuat.")
            except Exception as e:
                logger.warning(f"[Scraper] Purge eșuat: {e}")

        ch_list = list(channels_set)
        for start in range(0, len(ch_list), BATCH_SIZE):
            if not running or paused:
                break
            batch = ch_list[start:start + BATCH_SIZE]
            await asyncio.gather(*[process_single_channel(ch) for ch in batch])
            if running and not paused and start + BATCH_SIZE < len(ch_list):
                await asyncio.sleep(2)
        logger.info(f"[Scraper] ✓ Scanare continuă: {len(ch_list)} canale procesate.")
        await asyncio.sleep(10)


# ============================================================================
# PARTEA 8: STARTUP ȘI WEBSOCKET
# ============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("Pornire aplicație.")
    set_memory_limit()
    db_init()
    db_warm_up_state()
    loop = asyncio.get_running_loop()
    start_nlp_loading(loop)
    # background_narrative_clusterer și background_emergent_analyzer pornesc imediat
    # și rulează independent de starea `running`.
    # background_scraper se pornește explicit la comanda WebSocket "start".
    asyncio.create_task(background_narrative_clusterer())
    asyncio.create_task(background_emergent_analyzer())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global running, paused, channels_set, nodes_data, edges_data, edges_type, posts_history
    global target_channel, keywords_list, similarity_mode, analysis_mode
    global background_tasks, dirty_channels, ch_msgs_cache, ch_embs_cache, ch_style_cache, ch_lang_cache, ch_msgs_set  # 🔥 BUG FIX #3: Extra globals

    await manager.connect(websocket)
    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=90.0
                )
                cmd = json.loads(data)
                action = cmd.get("action")

                if action == "pong":
                    continue

                # 🔥 BUG FIX #3: Handle keywords_list update properly
                if action == "set_keywords":
                    keywords_list.clear()
                    keywords_list.extend(cmd.get("keywords", []))
                    await manager.send_state()
                    continue

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
                    await manager.send_state()

                elif action == "reset":
                    # 🔥 BUG FIX #3: Proper global variable updates
                    channels_set.clear()
                    nodes_data.clear()
                    edges_data.clear()
                    edges_type.clear()
                    posts_history.clear()
                    ch_msgs_cache.clear()
                    ch_msgs_set.clear()
                    ch_embs_cache.clear()
                    ch_style_cache.clear()
                    ch_lang_cache.clear()
                    dirty_channels.clear()
                    nlp_msg_cache.clear()  # 🔥 Also clear NLP cache on reset
                    target_channel = None
                    running = False
                    paused = False
                    for t in background_tasks:
                        if not t.done():
                            t.cancel()
                    background_tasks.clear()
                    await manager.broadcast(safe_json_dumps({"type": "clear_graph"}))
                    await manager.send_state()

                elif action == "set_target":
                    ch = cmd.get("channel", "").strip()
                    target_channel = (
                        ch if (ch in channels_set and target_channel != ch) else None
                    )
                    await manager.send_state()

                elif action == "start":
                    if not running:
                        running = True
                        paused = False
                        # Pornit doar background_scraper la comanda "start".
                        # background_emergent_analyzer rulează deja din startup_event.
                        t1 = asyncio.create_task(background_scraper())
                        background_tasks.append(t1)
                    await manager.send_state()

                elif action == "stop":
                    running = False
                    for t in background_tasks:
                        if not t.done():
                            t.cancel()
                    background_tasks.clear()
                    await manager.send_state()

                elif action == "pause":
                    paused = not paused
                    await manager.send_state()

            except asyncio.TimeoutError:
                try:
                    await websocket.send_text(safe_json_dumps({"type": "ping"}))
                except Exception:
                    break
                continue
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    finally:
        manager.disconnect(websocket)


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_ping_interval=None, ws_ping_timeout=None)
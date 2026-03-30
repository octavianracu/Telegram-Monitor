"""
classify_channel.py — Clasificator zero-shot de hub narativ
============================================================
Citește profilurile EMA și temele BERTopic din tgm_monitor.db,
calculează centroizii fiecărui hub și clasifică un canal sau text
nou prin similaritate cosinus.

Nu depinde de serverul FastAPI. Rulează independent.

Utilizare:
  python classify_channel.py --channel @exemplu
  python classify_channel.py --text "Guvernul a anunțat noi măsuri economice"
  python classify_channel.py --all
  python classify_channel.py --export hub_map.csv
  python classify_channel.py --eval
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

DB_PATH    = "tgm_monitor.db"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


# ─────────────────────────────────────────────
# Utilitar DB
# ─────────────────────────────────────────────

def db_connect(path: str = DB_PATH) -> sqlite3.Connection:
    if not Path(path).exists():
        print(f"[EROARE] Baza de date '{path}' nu există.")
        print("  Asigură-te că rulezi scriptul din același director cu main.py.")
        sys.exit(1)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def load_profiles(conn) -> dict:
    """
    Returnează {channel: {"embedding": np.array, "dominant_topic": int,
                           "topic_distribution": dict}}
    """
    rows = conn.execute(
        "SELECT channel, ema_embedding, topic_distribution, dominant_topic "
        "FROM channel_narrative_profile"
    ).fetchall()
    if not rows:
        print("[AVERTISMENT] Nu există profile narative în DB.")
        print("  Rulează serverul cel puțin 10 minute cu date pentru a genera profile EMA.")
        sys.exit(1)
    profiles = {}
    for r in rows:
        profiles[r["channel"]] = {
            "embedding":          np.array(json.loads(r["ema_embedding"])),
            "dominant_topic":     r["dominant_topic"],
            "topic_distribution": json.loads(r["topic_distribution"]),
        }
    return profiles


def load_topics(conn) -> dict:
    """
    Returnează {topic_id: {"keywords": [...], "size": int}} din ultima rulare BERTopic.
    """
    row = conn.execute(
        "SELECT run_id FROM narrative_topics ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return {}
    run_id = row["run_id"]
    rows = conn.execute(
        "SELECT topic_id, keywords, size FROM narrative_topics WHERE run_id=?",
        (run_id,)
    ).fetchall()
    return {
        r["topic_id"]: {
            "keywords": json.loads(r["keywords"]),
            "size":     r["size"],
        }
        for r in rows
    }


def load_recent_messages(conn, channel: str, days: int = 7) -> list:
    """Returnează mesajele unui canal din ultimele `days` zile."""
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT text FROM messages WHERE channel=? AND ts>=? ORDER BY ts ASC",
        (channel, cutoff)
    ).fetchall()
    return [r["text"] for r in rows]


# ─────────────────────────────────────────────
# Centroizi hub-uri
# ─────────────────────────────────────────────

def compute_hub_centroids(profiles: dict) -> dict:
    """
    Grupează canalele după dominant_topic și calculează centroidul
    (media aritmetică a vectorilor EMA) pentru fiecare hub.

    Returnează {topic_id: centroid_np_array}
    """
    groups: dict = {}
    for ch, data in profiles.items():
        tid = data["dominant_topic"]
        if tid == -1:
            continue  # outlieri BERTopic — nu formează hub
        groups.setdefault(tid, []).append(data["embedding"])

    return {
        tid: np.mean(np.stack(vecs), axis=0)
        for tid, vecs in groups.items()
    }


# ─────────────────────────────────────────────
# Similaritate cosinus
# ─────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def classify_vector(
    vec: np.ndarray,
    centroids: dict,
    topics: dict,
    top_k: int = 3,
) -> list:
    """
    Clasifică un vector față de centroizii tuturor hub-urilor.

    Returnează lista primelor top_k hub-uri ordonate descrescător după scor:
    [{"topic_id": int, "score": float, "keywords": [...], "size": int}, ...]
    """
    scores = []
    for tid, centroid in centroids.items():
        sim = cosine_similarity(vec, centroid)
        info = topics.get(tid, {})
        scores.append({
            "topic_id": tid,
            "score":    round(sim, 4),
            "keywords": info.get("keywords", []),
            "size":     info.get("size", 0),
        })
    scores.sort(key=lambda x: -x["score"])
    return scores[:top_k]


# ─────────────────────────────────────────────
# Encoding text nou cu modelul
# ─────────────────────────────────────────────

def load_model():
    """Încarcă sentence-transformers (același model ca în main.py)."""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[Model] Se încarcă {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        print("[Model] Gata.")
        return model
    except ImportError:
        print("[EROARE] sentence-transformers nu e instalat.")
        print("  Rulează: pip install sentence-transformers")
        sys.exit(1)


def encode_texts(model, texts: list) -> np.ndarray:
    """Encodează o listă de texte și returnează media vectorilor."""
    import re
    clean = []
    for t in texts:
        t = re.sub(r"http\S+", "", t)
        t = re.sub(r"[^\w\s]", " ", t)
        t = t.strip().lower()
        if len(t) > 10:
            clean.append(t)
    if not clean:
        return None
    embs = model.encode(clean, show_progress_bar=False)
    return embs.mean(axis=0)


# ─────────────────────────────────────────────
# Formatare output
# ─────────────────────────────────────────────

def format_result(channel_or_text: str, results: list, topics: dict) -> str:
    lines = [f"\n{'─'*60}", f"  {channel_or_text}", f"{'─'*60}"]
    for i, r in enumerate(results):
        kw    = " · ".join(r["keywords"][:5]) if r["keywords"] else "—"
        bar   = "█" * int(r["score"] * 20)
        label = " ◄ HUB DOMINANT" if i == 0 else ""
        lines.append(
            f"  #{i+1}  Hub {r['topic_id']:>3}  [{bar:<20}]  {r['score']:.3f}"
            f"  ({r['size']} canale){label}"
        )
        lines.append(f"         {kw}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Comenzi principale
# ─────────────────────────────────────────────

def cmd_classify_channel(channel: str, conn, profiles, centroids, topics, model=None):
    """Clasifică un canal existent în DB sau unul nou prin scraping mesaje."""
    channel = "@" + channel.lstrip("@")

    # Canal existent cu profil EMA
    if channel in profiles:
        vec     = profiles[channel]["embedding"]
        results = classify_vector(vec, centroids, topics)
        print(format_result(channel, results, topics))
        current = profiles[channel]["dominant_topic"]
        print(f"\n  Etichetă curentă în DB: Hub {current}")
        if results and results[0]["topic_id"] == current:
            print("  ✓ Clasificarea zero-shot confirmă eticheta existentă.")
        else:
            print(f"  ⚠ Clasificarea zero-shot sugerează Hub {results[0]['topic_id']} "
                  f"(scor {results[0]['score']:.3f}) față de eticheta curentă Hub {current}.")
        return

    # Canal nou — necesită mesaje din DB sau model
    msgs = load_recent_messages(conn, channel, days=7)
    if not msgs:
        print(f"[AVERTISMENT] Canalul {channel} nu are mesaje în DB.")
        print("  Adaugă-l în monitor și lasă-l să se scrapeze înainte de clasificare.")
        return

    if model is None:
        model = load_model()
    vec = encode_texts(model, msgs)
    if vec is None:
        print(f"[EROARE] Mesajele canalului {channel} nu pot fi encodate.")
        return
    results = classify_vector(vec, centroids, topics)
    print(format_result(f"{channel} (canal nou)", results, topics))


def cmd_classify_text(text: str, centroids, topics, model=None):
    """Clasifică un text arbitrar."""
    if model is None:
        model = load_model()
    vec = encode_texts(model, [text])
    if vec is None:
        print("[EROARE] Textul nu poate fi encodat.")
        return
    results = classify_vector(vec, centroids, topics)
    preview = text[:60] + "..." if len(text) > 60 else text
    print(format_result(f'"{preview}"', results, topics))


def cmd_classify_all(profiles, centroids, topics):
    """Clasifică toate canalele din DB și arată discrepanțele față de eticheta BERTopic."""
    print(f"\nClasificare zero-shot pentru {len(profiles)} canale...\n")
    matches    = 0
    mismatches = []

    for ch, data in sorted(profiles.items()):
        vec     = data["embedding"]
        results = classify_vector(vec, centroids, topics, top_k=1)
        if not results:
            continue
        predicted = results[0]["topic_id"]
        current   = data["dominant_topic"]
        if predicted == current:
            matches += 1
        else:
            mismatches.append({
                "channel":   ch,
                "current":   current,
                "predicted": predicted,
                "score":     results[0]["score"],
                "keywords":  results[0]["keywords"][:3],
            })

    total = len(profiles)
    print(f"  Acord zero-shot / BERTopic: {matches}/{total} ({100*matches//total}%)")
    if mismatches:
        print(f"\n  Discrepanțe ({len(mismatches)} canale):")
        for m in mismatches:
            kw = " · ".join(m["keywords"])
            print(f"    {m['channel']:<35}  curent=Hub {m['current']}  "
                  f"→  zero-shot=Hub {m['predicted']} ({m['score']:.3f})  [{kw}]")
    print()


def cmd_export_csv(output_path: str, profiles, centroids, topics):
    """Exportă clasificarea tuturor canalelor ca CSV pentru R/Excel/Gephi."""
    import csv
    rows = []
    for ch, data in profiles.items():
        vec     = data["embedding"]
        results = classify_vector(vec, centroids, topics, top_k=3)
        row = {
            "channel":          ch,
            "hub_dominant":     data["dominant_topic"],
            "hub_zeroshot":     results[0]["topic_id"] if results else -1,
            "score_1":          results[0]["score"]    if len(results) > 0 else 0,
            "hub_2":            results[1]["topic_id"] if len(results) > 1 else -1,
            "score_2":          results[1]["score"]    if len(results) > 1 else 0,
            "hub_3":            results[2]["topic_id"] if len(results) > 2 else -1,
            "score_3":          results[2]["score"]    if len(results) > 2 else 0,
            "keywords_hub1":    " | ".join(results[0]["keywords"][:5]) if results else "",
        }
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Export] {len(rows)} canale salvate în '{output_path}'")
    print(f"  Importă în R: df <- read.csv('{output_path}')")


def cmd_show_hubs(profiles, centroids, topics):
    """Afișează un rezumat al hub-urilor existente."""
    print(f"\n{'═'*60}")
    print(f"  Hub-uri narative detectate ({len(centroids)} hub-uri, "
          f"{len(profiles)} canale)\n")
    for tid in sorted(centroids.keys()):
        info     = topics.get(tid, {})
        kw       = " · ".join(info.get("keywords", [])[:6])
        channels = [ch for ch, d in profiles.items() if d["dominant_topic"] == tid]
        print(f"  Hub {tid:>3}  ({len(channels):>3} canale)  {kw}")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────
# Evaluare cross-validare simplă
# ─────────────────────────────────────────────

def cmd_eval(profiles, topics):
    """
    Leave-one-out cross-validare:
    pentru fiecare canal, recalculează centroizii fără el și verifică
    dacă zero-shot îl clasifică corect.
    Oferă o estimare realistă a acurateței modelului.
    """
    channels  = [(ch, d) for ch, d in profiles.items() if d["dominant_topic"] != -1]
    correct   = 0
    total     = len(channels)

    print(f"\nEvaluare leave-one-out pe {total} canale...\n")

    for i, (ch, data) in enumerate(channels):
        # Centroizi fără canalul curent
        temp_profiles = {c: d for c, d in profiles.items() if c != ch}
        centroids_loo = compute_hub_centroids(temp_profiles)

        vec     = data["embedding"]
        results = classify_vector(vec, centroids_loo, topics, top_k=1)
        if results and results[0]["topic_id"] == data["dominant_topic"]:
            correct += 1

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{total}...", end="\r")

    acc = 100 * correct / total if total else 0
    print(f"\n  Acuratețe leave-one-out: {correct}/{total} = {acc:.1f}%")
    print()
    if acc >= 80:
        print("  ✓ Model robust — zero-shot funcționează bine pe aceste date.")
    elif acc >= 60:
        print("  △ Acuratețe moderată — hub-urile se suprapun parțial.")
        print("    Consideră un clasificator antrenat (SVM/MLP) pentru performanță mai bună.")
    else:
        print("  ✗ Acuratețe scăzută — hub-urile nu sunt bine separate semantic.")
        print("    Poate fi nevoie de mai multe date sau de re-rularea BERTopic.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clasificator zero-shot de hub narativ pentru TGM Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemple:
  python classify_channel.py --hubs
  python classify_channel.py --channel @tv8md
  python classify_channel.py --text "Guvernul anunță măsuri economice"
  python classify_channel.py --all
  python classify_channel.py --export rezultate.csv
  python classify_channel.py --eval
  python classify_channel.py --db /cale/catre/alt/tgm_monitor.db --channel @canal
        """
    )
    parser.add_argument("--db",      default=DB_PATH, help="Calea către tgm_monitor.db")
    parser.add_argument("--channel", help="Clasifică un canal (@username)")
    parser.add_argument("--text",    help="Clasifică un text arbitrar")
    parser.add_argument("--all",     action="store_true", help="Clasifică toate canalele")
    parser.add_argument("--export",  metavar="FILE.csv",  help="Exportă clasificarea ca CSV")
    parser.add_argument("--eval",    action="store_true", help="Evaluare leave-one-out")
    parser.add_argument("--hubs",    action="store_true", help="Afișează hub-urile existente")
    parser.add_argument("--top",     type=int, default=3, help="Numărul de hub-uri returnate")
    args = parser.parse_args()

    if not any([args.channel, args.text, args.all, args.export, args.eval, args.hubs]):
        parser.print_help()
        sys.exit(0)

    conn     = db_connect(args.db)
    profiles = load_profiles(conn)
    topics   = load_topics(conn)
    centroids = compute_hub_centroids(profiles)

    if not centroids:
        print("[EROARE] Nu există centroizi — profilurile nu au etichete BERTopic.")
        print("  Rulează serverul și asteaptă prima rulare BERTopic (10 min după pornire).")
        sys.exit(1)

    if args.hubs:
        cmd_show_hubs(profiles, centroids, topics)

    if args.channel:
        cmd_classify_channel(args.channel, conn, profiles, centroids, topics)

    if args.text:
        cmd_classify_text(args.text, centroids, topics)

    if args.all:
        cmd_classify_all(profiles, centroids, topics)

    if args.export:
        cmd_export_csv(args.export, profiles, centroids, topics)

    if args.eval:
        cmd_eval(profiles, topics)

    conn.close()


if __name__ == "__main__":
    main()

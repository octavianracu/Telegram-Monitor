import asyncio
import json
import re
import os
import logging
from datetime import datetime, timedelta
import threading
import pickle
import hashlib
import glob
import sqlite3

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import networkx as nx

# ─────────────────────────────────────────────
# Heavy models — încărcate în background thread
# ─────────────────────────────────────────────
similarity_model   = None
cosine_similarity  = None
ner_pipeline       = None
sentiment_pipeline = None
nlp_ready   = False
nlp_status  = "Așteptare..."

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """
    Encoder care converteste tipurile numpy in tipuri Python native.
    cosine_similarity (sklearn) returneaza np.float32, np.where returneaza
    np.int64 — ambele fac json.dumps sa arunce TypeError, prins silentios
    de except Exception, care golea fisierele .tgm salvate (doar channels).
    """
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
    """json.dumps cu NumpyEncoder — folosit peste tot in aplicatie."""
    return json.dumps(data, cls=NumpyEncoder, ensure_ascii=False)


app = FastAPI()

# ─────────────────────────────────────────────
# Backup endpoint
# ─────────────────────────────────────────────
@app.get("/api/backup_now")
async def backup_now():
    try:
        os.makedirs("backups", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"backups/backup_{timestamp}.json"
        backup_data = {
            "timestamp":      timestamp,
            "channels":       list(channels_set),
            "target":         target_channel,
            "keywords":       keywords_list,
            "mode":           similarity_mode,
            "nodes":          nodes_data,
            "edges":          {f"{k[0]}|{k[1]}": v for k, v in edges_data.items()},
            "posts_history":  posts_history[-100:],
            "global_entities":global_entities,
            "stats": {
                "total_channels": len(channels_set),
                "total_edges":    len(edges_data),
            },
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Backup creat: {filename}")
        return {"success": True, "message": f"Backup salvat: {filename}",
                "filename": filename, "stats": backup_data["stats"]}
    except Exception as e:
        logger.error(f"❌ Eroare backup: {e}")
        return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────
# Proiecte (.tgm)
# ─────────────────────────────────────────────
PROJECTS_DIR = "projects"
os.makedirs(PROJECTS_DIR, exist_ok=True)

@app.get("/api/project/list")
async def list_projects():
    try:
        projects = []
        for filepath in glob.glob(f"{PROJECTS_DIR}/*.tgm"):
            filename = os.path.basename(filepath)
            stat     = os.stat(filepath)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                project_name = data.get("name", filename)
            except (json.JSONDecodeError, OSError):
                project_name = filename
            projects.append({
                "filename": filename,
                "name":     project_name,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size":     stat.st_size,
            })
        projects.sort(key=lambda x: x["modified"], reverse=True)
        return {"status": "success", "projects": projects}
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/project/save")
async def save_project(request: Request):
    try:
        data         = await request.json()
        project_name = data.get("name", "unnamed")
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name    = "".join(c for c in project_name if c.isalnum() or c in (" ", "-", "_")).strip()
        safe_name    = safe_name.replace(" ", "_")
        filename     = f"{PROJECTS_DIR}/{safe_name}_{timestamp}.tgm"

        project_data = {
            "version":         "2.0",
            "name":            project_name,
            "saved_at":        datetime.now().isoformat(),
            "channels":        list(channels_set),
            "target_channel":  target_channel,
            "keywords":        keywords_list,
            "similarity_mode": similarity_mode,
            "analysis_mode":   analysis_mode,
            "running":         running,
            "paused":          paused,
            "nodes":           nodes_data,
            "edges":           {f"{k[0]}|{k[1]}": v for k, v in edges_data.items()},
            # FIX: salvăm și tipul fiecărei muchii (directă vs inferată)
            "edges_type":      {f"{k[0]}|{k[1]}": v for k, v in edges_type.items()},
            "posts_history":   posts_history,
            "ch_msgs_cache":   ch_msgs_cache,
            "ch_msgs_set":     {k: list(v) for k, v in ch_msgs_set.items()},
            "ch_style_cache":  {},
            "ch_lang_cache":   ch_lang_cache,
            "ch_embs_cache":   {},
            "nlp_msg_cache":   {str(k): v for k, v in nlp_msg_cache.items()},
            "global_entities": global_entities,
            "dirty_channels":  list(dirty_channels),
            "stats": {
                "total_channels": len(channels_set),
                "total_edges":    len(edges_data),
                "total_messages": sum(len(v) for v in ch_msgs_cache.values()),
            },
        }
        for k, v in ch_style_cache.items():
            project_data["ch_style_cache"][k] = v.tolist() if isinstance(v, np.ndarray) else v
        for ch, emb in ch_embs_cache.items():
            if emb and "matrix" in emb and emb["matrix"] is not None:
                project_data["ch_embs_cache"][ch] = {
                    "orig_texts": emb["orig_texts"],
                    "matrix":     emb["matrix"].tolist(),
                }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(project_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Proiect .tgm salvat: {filename}")
        return {"status": "success", "filename": filename}
    except Exception as e:
        logger.error(f"Error saving project: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/project/load")
async def load_project(request: Request):
    try:
        data     = await request.json()
        filename = data.get("filename")
        if not filename:
            return {"status": "error", "message": "Filename required"}
        filepath = os.path.join(PROJECTS_DIR, filename)
        if not os.path.exists(filepath):
            return {"status": "error", "message": "File not found"}
        with open(filepath, "r", encoding="utf-8") as f:
            project_data = json.load(f)
        return {"status": "success", "data": project_data}
    except Exception as e:
        logger.error(f"Error loading project: {e}")
        return {"status": "error", "message": str(e)}


@app.delete("/api/project/delete")
async def delete_project(request: Request):
    try:
        data     = await request.json()
        filename = data.get("filename")
        if not filename:
            return {"status": "error", "message": "Filename required"}
        filepath = os.path.join(PROJECTS_DIR, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"✅ Proiect șters: {filename}")
            return {"status": "success"}
        return {"status": "error", "message": "File not found"}
    except Exception as e:
        logger.error(f"Error deleting project: {e}")
        return {"status": "error", "message": str(e)}


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")


# ─────────────────────────────────────────────
# State global
# ─────────────────────────────────────────────
channels_set: set  = set()
running     = False
paused      = False
target_channel     = None
keywords_list: list = []
similarity_mode    = "hibrid"   # "direct" | "hibrid" | "tranzitiv"
analysis_mode      = "similar"  # "repost" | "similar" | "stylography"

nodes_data: dict = {}
edges_data: dict = {}   # {(c1,c2): float}

# FIX: dicționar separat pentru tipul fiecărei muchii
# "direct" = relație confirmată semantic; "inferred" = inferată tranzitiv
edges_type: dict = {}   # {(c1,c2): "direct" | "inferred"}

posts_history: list = []

ch_msgs_cache: dict = {}
ch_embs_cache: dict = {}
ch_style_cache: dict = {}
ch_lang_cache:  dict = {}
ch_msgs_set:    dict = {}
nlp_msg_cache:  dict = {}
global_entities: dict = {"PER": {}, "ORG": {}}
dirty_channels:  set  = set()
background_tasks: list = []

THRESHOLD = {
    "repost":      0.88,
    "similar":     0.72,
    "stylography": 0.72,
}

# FIX: pragurile de inferență sunt EGALE cu pragul direct — nu mai mici.
# Inferențele nu trebuie să fie mai permisive decât analiza directă.
INFERENCE_THRESHOLD = {
    "direct":    0.72,
    "hibrid":    0.72,   # FIX: era 0.65 — prea permisiv, genera hub-uri false
    "tranzitiv": 0.68,   # FIX: era 0.60 — prea permisiv
}

# FIX: pragul minim pentru ca o muchie să intre în graful de inferențe
# Trebuie să fie ≥ pragul direct, nu 0.5 cum era înainte.
INFERENCE_GRAPH_MIN_WEIGHT = 0.72   # FIX: era 0.5 — principala cauză a hub-urilor false

# FIX: penalizare aplicată scorului inferit față de cel direct
# Relațiile inférate sunt mai puțin sigure și decad mai rapid.
INFERRED_SCORE_PENALTY = 0.65   # FIX: era 1.3 (amplificare!) — acum penalizăm

NLP_ALIGNMENT_MIN = 0.3
# FIX: 0.88 → 0.95 pentru muchii directe — la 0.88 o muchie confirmată murea în ~13 cicluri
# (~2 min), mai repede decât round-robin-ul (68 cicluri, ~11 min) o putea reconfirma.
# La 0.95 durata de viață crește la ~45 cicluri (~7.5 min), permițând grafului să acumuleze
# muchii stabile și comunităților Louvain să convergă în loc să oscileze.
DECAY_BASE        = 0.95
# Decay agresiv pentru muchii inférate — rămâne 0.75, inferențele greșite
# dispar rapid dacă nu sunt confirmate semantic direct.
DECAY_INFERRED    = 0.75

# ─────────────────────────────────────────────
# SQLite — persistență mesaje + scoruri cumulative
# ─────────────────────────────────────────────
DB_PATH             = "tgm_monitor.db"
# Fereastra de analiză: doar mesajele din ultimele 3 zile intră în embeddings.
# Captează reacțiile la același val de știri fără a acumula conținut expirat.
ANALYSIS_WINDOW_DAYS = 3

_db_lock = threading.Lock()


def db_connect() -> sqlite3.Connection:
    """Returnează o conexiune SQLite cu WAL mode pentru scrieri concurente sigure."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def db_init():
    """
    Creează schema SQLite la primul pornire.
    Tabelul messages stochează fiecare mesaj cu timestamp ISO pentru filtrare
    pe fereastra glisantă de 3 zile.
    Tabelul edges_cumulative stochează scorul total pe toată durata monitorizării —
    nu scade niciodată, nu e afectat de decay-ul din RAM.
    """
    with _db_lock:
        conn = db_connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                channel   TEXT    NOT NULL,
                text      TEXT    NOT NULL,
                ts        TEXT    NOT NULL,
                UNIQUE(channel, text)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_channel_ts
                ON messages(channel, ts);

            CREATE TABLE IF NOT EXISTS edges_cumulative (
                ch1         TEXT NOT NULL,
                ch2         TEXT NOT NULL,
                score_total REAL NOT NULL DEFAULT 0.0,
                hits        INTEGER NOT NULL DEFAULT 0,
                first_seen  TEXT NOT NULL,
                last_seen   TEXT NOT NULL,
                PRIMARY KEY (ch1, ch2)
            );
        """)
        conn.commit()
        conn.close()
    logger.info(f"[DB] Schema inițializată: {DB_PATH}")


def db_insert_messages(channel: str, texts: list):
    """
    Inserează mesajele noi cu timestamp curent.
    IGNORE pe conflictul UNIQUE evită duplicatele fără excepții.
    """
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
    """
    Returnează mesajele canalului din ultimele `days` zile.
    Aceasta este fereastra glisantă: la 50+ msg/zi, un canal poate avea
    150–500 de mesaje în fereastră — mult mai relevant decât ultimele 50 indiferent de dată.
    """
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    with _db_lock:
        conn = db_connect()
        rows = conn.execute(
            "SELECT text FROM messages WHERE channel=? AND ts>=? ORDER BY ts ASC",
            (channel, cutoff),
        ).fetchall()
        conn.close()
    return [r[0] for r in rows]


def db_get_all_messages_set(channel: str) -> set:
    """Set complet de texte pentru deduplicare rapidă (indiferent de dată)."""
    with _db_lock:
        conn = db_connect()
        rows = conn.execute(
            "SELECT text FROM messages WHERE channel=?", (channel,)
        ).fetchall()
        conn.close()
    return {r[0] for r in rows}


def db_update_edge_cumulative(ch1: str, ch2: str, score_delta: float):
    """
    Adaugă score_delta la scorul cumulativ al perechii (ch1, ch2).
    Scorul cumulativ nu scade niciodată — reflectă întreaga istorie a
    coordonării detectate între cele două canale pe durata monitorizării.
    ch1 < ch2 garantat de tuple(sorted(...)) înainte de apel.
    """
    now = datetime.now().isoformat()
    with _db_lock:
        conn = db_connect()
        conn.execute("""
            INSERT INTO edges_cumulative(ch1, ch2, score_total, hits, first_seen, last_seen)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(ch1, ch2) DO UPDATE SET
                score_total = score_total + excluded.score_total,
                hits        = hits + 1,
                last_seen   = excluded.last_seen
        """, (ch1, ch2, score_delta, now, now))
        conn.commit()
        conn.close()


def db_get_cumulative_scores() -> dict:
    """
    Returnează dicționarul {(ch1, ch2): score_total} pentru toate perechile cunoscute.
    Folosit la construirea grafului Louvain — graful cumulativ e stabil și convergent.
    """
    with _db_lock:
        conn = db_connect()
        rows = conn.execute(
            "SELECT ch1, ch2, score_total FROM edges_cumulative WHERE score_total > 0"
        ).fetchall()
        conn.close()
    return {(r[0], r[1]): r[2] for r in rows}


def db_purge_old_messages(days: int = 7):
    """
    Curăță mesajele mai vechi de `days` zile pentru a limita creșterea DB.
    Apelat automat la fiecare 24h din scraper.
    Fereastra de analiză e 3 zile, dar păstrăm 7 pentru auditul manual.
    """
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    with _db_lock:
        conn = db_connect()
        deleted = conn.execute(
            "DELETE FROM messages WHERE ts < ?", (cutoff,)
        ).rowcount
        conn.commit()
        conn.close()
    if deleted:
        logger.info(f"[DB] Curățat {deleted} mesaje mai vechi de {days} zile")


# ─────────────────────────────────────────────
# Backup complet
# ─────────────────────────────────────────────
def full_backup():
    os.makedirs("backups", exist_ok=True)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename     = f"backups/full_backup_{timestamp}.pkl"
    backup = {
        "timestamp":       timestamp,
        "version":         "2.1",
        "channels":        list(channels_set),
        "target":          target_channel,
        "keywords":        keywords_list,
        "mode":            similarity_mode,
        "analysis_mode":   analysis_mode,
        "running":         running,
        "paused":          paused,
        "nodes":           nodes_data,
        "edges":           {f"{k[0]}|{k[1]}": v for k, v in edges_data.items()},
        "edges_type":      {f"{k[0]}|{k[1]}": v for k, v in edges_type.items()},
        "ch_msgs_cache":   ch_msgs_cache,
        "ch_msgs_set":     {k: list(v) for k, v in ch_msgs_set.items()},
        "ch_embs_cache":   {},
        "ch_style_cache":  {},
        "ch_lang_cache":   ch_lang_cache,
        "global_entities": global_entities,
        "posts_history":   posts_history[-500:],
        "nlp_msg_cache":   {str(k): v for k, v in nlp_msg_cache.items()},
        "dirty_channels":  list(dirty_channels),
        "stats": {
            "total_channels":   len(channels_set),
            "total_edges":      len(edges_data),
            "total_messages":   sum(len(v) for v in ch_msgs_cache.values()),
            "total_embeddings": sum(1 for v in ch_embs_cache.values() if v),
        },
    }
    for ch, data in ch_embs_cache.items():
        if data and "matrix" in data and data["matrix"] is not None:
            backup["ch_embs_cache"][ch] = {
                "orig_texts": data["orig_texts"],
                "matrix":     data["matrix"].tolist(),
            }
    for k, v in ch_style_cache.items():
        backup["ch_style_cache"][k] = v.tolist() if isinstance(v, np.ndarray) else v

    with open(filename, "wb") as f:
        pickle.dump(backup, f)

    json_filename = f"backups/backup_{timestamp}.json"
    json_backup   = {k: v for k, v in backup.items() if k != "ch_embs_cache"}
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(json_backup, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Backup complet: {filename}")
    return filename


def restore_full_backup(filename):
    global channels_set, nodes_data, edges_data, edges_type, posts_history
    global target_channel, keywords_list, similarity_mode, analysis_mode
    global ch_msgs_cache, ch_msgs_set, ch_embs_cache, ch_style_cache
    global ch_lang_cache, dirty_channels, running, paused, nlp_msg_cache
    # BUG 5 FIX: global_entities lipsea din lista global — atribuirea crea
    # o variabilă locală și nu modifica starea globală a aplicației.
    global global_entities
    try:
        with open(filename, "rb") as f:
            backup = pickle.load(f)
        logger.info(f"Restaurare backup v{backup.get('version','1.0')} din {backup.get('timestamp')}")

        channels_set    = set(backup["channels"])
        target_channel  = backup.get("target")
        keywords_list   = backup.get("keywords", [])
        similarity_mode = backup.get("mode", "hibrid")
        analysis_mode   = backup.get("analysis_mode", "similar")
        running         = backup.get("running", False)
        paused          = backup.get("paused", False)
        nodes_data      = backup.get("nodes", {})

        edges_data.clear()
        edges_type.clear()
        for k_str, v in backup.get("edges", {}).items():
            c1, c2 = k_str.split("|")
            edges_data[(c1, c2)] = v
        for k_str, v in backup.get("edges_type", {}).items():
            c1, c2 = k_str.split("|")
            edges_type[(c1, c2)] = v

        ch_msgs_cache = backup.get("ch_msgs_cache", {})
        ch_msgs_set.clear()
        for k, v in backup.get("ch_msgs_set", {}).items():
            ch_msgs_set[k] = set(v)

        ch_embs_cache.clear()
        for ch, data in backup.get("ch_embs_cache", {}).items():
            if data and "matrix" in data and data["matrix"]:
                ch_embs_cache[ch] = {
                    "orig_texts": data["orig_texts"],
                    "matrix":     np.array(data["matrix"]),
                }

        ch_style_cache.clear()
        for k, v in backup.get("ch_style_cache", {}).items():
            ch_style_cache[k] = np.array(v) if isinstance(v, list) else v

        ch_lang_cache   = backup.get("ch_lang_cache", {})
        global_entities = backup.get("global_entities", {"PER": {}, "ORG": {}})
        posts_history   = backup.get("posts_history", [])

        nlp_msg_cache.clear()
        for k_str, v in backup.get("nlp_msg_cache", {}).items():
            try:
                nlp_msg_cache[int(k_str)] = v
            except (ValueError, TypeError):
                pass

        dirty_channels = set(backup.get("dirty_channels", []))
        logger.info(f"✅ Backup restaurat: {len(channels_set)} canale, {len(edges_data)} muchii")
        return True
    except Exception as e:
        logger.error(f"❌ Eroare restaurare: {e}", exc_info=True)
        return False


# ─────────────────────────────────────────────
# Salvare / încărcare stare proiect
# ─────────────────────────────────────────────
def save_project_state():
    project_data = {
        "version":         "2.1",
        "timestamp":       datetime.now().isoformat(),
        "channels":        list(channels_set),
        "target_channel":  target_channel,
        "keywords":        keywords_list,
        "similarity_mode": similarity_mode,
        "analysis_mode":   analysis_mode,
        "running":         running,
        "paused":          paused,
        "nodes":           nodes_data,
        "edges":           {f"{k[0]}|{k[1]}": v for k, v in edges_data.items()},
        "edges_type":      {f"{k[0]}|{k[1]}": v for k, v in edges_type.items()},
        "ch_msgs_cache":   ch_msgs_cache,
        "ch_msgs_set":     {k: list(v) for k, v in ch_msgs_set.items()},
        "ch_style_cache":  {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in ch_style_cache.items()
        },
        "ch_lang_cache":   ch_lang_cache,
        "global_entities": global_entities,
        "posts_history":   posts_history[-100:],
        "nlp_msg_cache":   {str(k): v for k, v in nlp_msg_cache.items()},
        "dirty_channels":  list(dirty_channels),
        "last_save":       datetime.now().timestamp(),
        "ch_embs_cache":   {},
    }
    for ch, emb_data in ch_embs_cache.items():
        if emb_data and "matrix" in emb_data and emb_data["matrix"] is not None:
            project_data["ch_embs_cache"][ch] = {
                "orig_texts": emb_data["orig_texts"],
                "matrix":     emb_data["matrix"].tolist(),
            }
    return project_data


def load_project_state(project_data):
    global channels_set, nodes_data, edges_data, edges_type, posts_history
    global target_channel, keywords_list, similarity_mode, analysis_mode
    global ch_msgs_cache, ch_msgs_set, ch_embs_cache, ch_style_cache, ch_lang_cache
    global dirty_channels, global_entities, running, paused, nlp_msg_cache
    try:
        channels_set    = set(project_data.get("channels", []))
        target_channel  = project_data.get("target_channel")
        keywords_list   = project_data.get("keywords", [])
        similarity_mode = project_data.get("similarity_mode", "hibrid")
        analysis_mode   = project_data.get("analysis_mode", "similar")
        running         = project_data.get("running", False)
        paused          = project_data.get("paused", False)
        nodes_data      = project_data.get("nodes", {})

        edges_data.clear()
        edges_type.clear()
        for k_str, v in project_data.get("edges", {}).items():
            c1, c2 = k_str.split("|")
            edges_data[(c1, c2)] = v
        for k_str, v in project_data.get("edges_type", {}).items():
            c1, c2 = k_str.split("|")
            edges_type[(c1, c2)] = v

        ch_msgs_cache = project_data.get("ch_msgs_cache", {})
        ch_msgs_set.clear()
        for k, v in project_data.get("ch_msgs_set", {}).items():
            ch_msgs_set[k] = set(v)

        ch_style_cache.clear()
        for k, v in project_data.get("ch_style_cache", {}).items():
            ch_style_cache[k] = np.array(v) if isinstance(v, list) else v

        ch_lang_cache   = project_data.get("ch_lang_cache", {})
        global_entities = project_data.get("global_entities", {"PER": {}, "ORG": {}})
        posts_history   = project_data.get("posts_history", [])

        ch_embs_cache.clear()
        for ch, emb_data in project_data.get("ch_embs_cache", {}).items():
            if emb_data and "matrix" in emb_data and emb_data["matrix"]:
                ch_embs_cache[ch] = {
                    "orig_texts": emb_data["orig_texts"],
                    "matrix":     np.array(emb_data["matrix"]),
                }

        nlp_msg_cache.clear()
        for k_str, v in project_data.get("nlp_msg_cache", {}).items():
            try:
                nlp_msg_cache[int(k_str)] = v
            except (ValueError, TypeError):
                pass

        dirty_channels = set(channels_set)
        logger.info(f"Proiect încărcat: {len(channels_set)} canale, {len(edges_data)} conexiuni")
        return True
    except Exception as e:
        logger.error(f"Eroare încărcare proiect: {e}")
        return False


def hash_project_state(state):
    state_str = json.dumps(state, sort_keys=True, default=str, cls=NumpyEncoder)
    return hashlib.md5(state_str.encode()).hexdigest()[:8]


def build_graph():
    G = nx.Graph()
    for (u, v), w in edges_data.items():
        G.add_edge(u, v, weight=w)
    for c in channels_set:
        if c not in G:
            G.add_node(c)
    return G


async def send_graph_update(websocket, G):
    comms = {}
    try:
        from networkx.algorithms.community import louvain_communities
        if G.size() > 0:
            # FIX: seed=42 elimină nedeterminismul Louvain — numărul de comunități
            # nu mai oscilează între cicluri când graful este identic sau aproape identic.
            for idx, s in enumerate(louvain_communities(G, weight="weight", seed=42)):
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

    f_edges = []
    for k, v in edges_data.items():
        if k[0] in displayed and k[1] in displayed:
            etype = edges_type.get(k, "direct")
            f_edges.append({
                "from":  k[0],
                "to":    k[1],
                "value": round(v, 3),
                "title": f"Forta: {v:.2f} ({'inferată' if etype == 'inferred' else 'directă'})",
                "type":  etype,  # FIX: trimitem tipul muchiei către UI
            })

    return {"nodes": f_nodes, "edges": f_edges}


# ─────────────────────────────────────────────
# Încărcare modele NLP
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
        logger.info("[NLP] Librării importate.")

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
        nlp_status = "Sistem NLP Pregătit."
        logger.info("[NLP] TOATE MODELELE ACTIVE.")
        _notify_state(loop)
    except Exception as e:
        logger.error(f"[NLP] EROARE INCARCARE: {e}", exc_info=True)


def _notify_state(loop):
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(manager.send_state(), loop)


def start_nlp_loading(loop=None):
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    threading.Thread(target=_load_nlp_models, args=(loop,), daemon=True).start()


# ─────────────────────────────────────────────
# Utilitare
# ─────────────────────────────────────────────
def detect_language(text: str) -> str:
    if not text:
        return "other"
    cyr = len(re.findall(r"[\u0400-\u04FF]", text))
    lat = len(re.findall(r"[a-zA-Z]", text))
    if cyr > lat: return "ru"
    if lat > 5:   return "ro"
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

_STYLE_DIM_NAMES = [
    "char_len", "punc", "special", "word_len", "ellipsis", "exclaim",
    "uppercase", "ttr", "hapax", "sent_len", "stopword", "digit",
    "vocab_richness", "short_sent", "long_sent", "questions", "para_var", "has_link",
]

_STOPWORDS_STYLE = {
    "si", "in", "la", "de", "cu", "ca", "este", "sunt", "nu", "se",
    "pe", "din", "o", "un", "sau", "care", "pentru", "dar", "mai",
    "ne", "a", "ai", "ale", "cel", "cea", "cei", "cele", "lui", "lor",
    "i", "v", "na", "s", "ne", "eto", "kak", "no",
}


def get_stylometric_fingerprint(texts: list) -> np.ndarray:
    if not texts or len(texts) < 2:
        return np.zeros(18)

    def _msg_vec(t: str) -> np.ndarray:
        from collections import Counter
        n_char    = len(t) + 1
        words_lo  = [w.lower().strip(".,!?;:\"'()[]") for w in t.split() if w.strip()]
        sents     = [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]
        n_sents   = len(sents) + 1
        sent_lens = [len(s.split()) for s in sents] if sents else [0]
        freq      = Counter(words_lo)
        n_unique  = len(freq) + 1
        n_total   = len(words_lo) + 1
        letters   = re.findall(r"[a-zA-Z\u00C0-\u024F\u0400-\u04FF]", t)
        upper     = re.findall(r"[A-Z\u00C0-\u00DE\u0400-\u042F]", t)
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
            sum(1 for l in sent_lens if l < 5) / n_sents,
            sum(1 for l in sent_lens if l > 20) / n_sents,
            min(t.count("?") / 3.0, 1.0),
            float(len(t)),
            1.0 if re.search(r"https?://", t) else 0.0,
        ], dtype=float)

    all_vecs  = [_msg_vec(t) for t in texts]
    medians   = np.median(np.stack(all_vecs), axis=0)
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
            name  = ent["word"].replace("##", "")
            etype = ent["entity_group"]
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
    if not text:
        return 0
    t = text.lower().strip()
    t = re.sub(r"[^\d.,km]", "", t)
    t = t.replace(",", "")
    if t.endswith("k"):
        try: return int(float(t[:-1]) * 1_000)
        except (ValueError, TypeError): pass
    if t.endswith("m"):
        try: return int(float(t[:-1]) * 1_000_000)
        except (ValueError, TypeError): pass
    try: return int(float(t))
    except (ValueError, TypeError): return 0


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
# Embeddings pe fereastră glisantă de 3 zile
# ─────────────────────────────────────────────
def update_embeddings_incremental(ch: str, new_texts: list):
    """
    Recalculează embeddings-urile canalului pe fereastra glisantă de ANALYSIS_WINDOW_DAYS zile.
    La 50+ msg/zi, fereastra conține 150–500 de mesaje — mult mai reprezentativ
    decât ultimele 50 indiferent de dată.
    Embeddings-urile sunt păstrate în RAM (ch_embs_cache) pentru viteză,
    dar sursa de adevăr este SQLite.
    """
    if not new_texts or similarity_model is None:
        return
    recent_texts = db_get_recent_messages(ch, days=ANALYSIS_WINDOW_DAYS)
    if not recent_texts:
        return
    clean_texts = [clean_text(t) for t in recent_texts if len(clean_text(t)) > 10]
    if not clean_texts:
        return
    orig_valid = [t for t in recent_texts if len(clean_text(t)) > 10]
    matrix     = similarity_model.encode(clean_texts, show_progress_bar=False)
    ch_embs_cache[ch] = {"orig_texts": orig_valid, "matrix": matrix}


def get_embedding_matrix(ch: str):
    entry = ch_embs_cache.get(ch)
    if entry is None:
        return None, None
    return entry["matrix"], entry["orig_texts"]


# ─────────────────────────────────────────────
# Decay
# ─────────────────────────────────────────────
def _decay_edge(pair: tuple, strength: float) -> float:
    """
    FIX: decay diferențiat în funcție de tipul muchiei.
    Muchiile inférate decad mai rapid (DECAY_INFERRED = 0.75) decât
    cele directe (DECAY_BASE = 0.95). Astfel, o inferență greșită
    dispare din graf în câteva cicluri dacă nu este confirmată semantic.
    """
    etype      = edges_type.get(pair, "direct")
    base_decay = DECAY_INFERRED if etype == "inferred" else DECAY_BASE
    log_factor = 0.1 * (np.log10(strength + 1) / np.log10(11))
    return strength * (base_decay + log_factor)


# ─────────────────────────────────────────────
# FIX MAJOR: Inferență hibridă corectată
# ─────────────────────────────────────────────
async def perform_hybrid_inference(max_inferences: int):
    """
    Mod hibrid: completează cu inferențe NUMAI pe baza muchiilor directe puternice.

    MODIFICĂRI față de versiunea anterioară:
    1. Pragul de intrare în graful de inferențe: INFERENCE_GRAPH_MIN_WEIGHT (0.72)
       în loc de 0.5. Elimină principala sursă de hub-uri false.
    2. Formula: media geometrică FĂRĂ multiplicator de amplificare.
       inferred = (w_ab * w_bc) ** 0.5  — fără × 1.3
    3. Pragul de acceptare al inferenței: INFERENCE_THRESHOLD["hibrid"] (0.72)
       în loc de 0.65. Inferența nu e mai permisivă decât analiza directă.
    4. Scorul adăugat în edges_data: inferred * INFERRED_SCORE_PENALTY (0.65)
       Relația inferată pornește cu scor mai mic decât una directă.
    5. Iterăm doar pe noduri cu degree ≤ mediana grafului, nu pe primele 50.
       Evităm ca hub-urile mari să genereze O(k²) inferențe.
    6. Verificare: perechea (a, c) nu trebuie să existe deja ca inferată
       cu scor mare — evităm re-adăugarea în fiecare ciclu.
    """
    global edges_data, edges_type, posts_history

    if len(edges_data) < 20:
        return

    try:
        # FIX 1: prag de filtrare egal cu pragul direct
        G = nx.Graph()
        for (u, v), w in edges_data.items():
            if w > INFERENCE_GRAPH_MIN_WEIGHT and edges_type.get((u, v), "direct") == "direct":
                G.add_edge(u, v, weight=w)

        if G.number_of_edges() < 10:
            return

        # FIX 5: iterăm doar pe noduri cu degree mic-mediu
        degrees    = dict(G.degree())
        if not degrees:
            return
        median_deg = sorted(degrees.values())[len(degrees) // 2]
        # Noduri cu degree ≤ median + 2 pentru a evita hub-urile care explodează combinatorial
        candidate_nodes = [n for n, d in degrees.items() if d <= median_deg + 2]

        inferences = []
        for b in candidate_nodes[:40]:  # FIX: limitat la 40 noduri intermediare
            neighbors = list(G.neighbors(b))
            for i, a in enumerate(neighbors):
                for c in neighbors[i + 1:]:
                    if G.has_edge(a, c):
                        continue
                    pair = tuple(sorted([a, c]))
                    # FIX 6: nu re-adăugăm dacă există deja o inferență cu scor bun
                    existing = edges_data.get(pair, 0.0)
                    if existing > INFERENCE_GRAPH_MIN_WEIGHT * INFERRED_SCORE_PENALTY:
                        continue
                    w_ab = G[a][b]["weight"]
                    w_bc = G[b][c]["weight"]
                    # FIX 2: media geometrică simplă, FĂRĂ amplificare
                    inferred = (w_ab * w_bc) ** 0.5
                    # FIX 3: prag de acceptare egal cu pragul direct
                    if inferred > INFERENCE_THRESHOLD["hibrid"]:
                        inferences.append((a, c, inferred, b))

        inferences.sort(key=lambda x: -x[2])
        added = 0
        for a, c, w, via in inferences[:max_inferences]:
            pair = tuple(sorted([a, c]))
            if pair not in edges_data:
                # FIX 4: scor penalizat față de relații directe
                edges_data[pair] = min(w * INFERRED_SCORE_PENALTY, 10.0)
                edges_type[pair] = "inferred"
                posts_history.append({
                    "type":       "hibrid",
                    "ch1":        a,
                    "ch2":        c,
                    "via":        via,
                    "confidence": round(w, 3),
                    "time":       datetime.now().strftime("%H:%M:%S"),
                })
                if len(posts_history) > 500:
                    posts_history[:] = posts_history[-500:]
                added += 1
                logger.info(f"[Hibrid] Inferență: {a} ~ {c} (via {via}, {w:.2f})")

        if added > 0:
            logger.info(f"[Hibrid] Adăugate {added} muchii noi")

    except Exception as e:
        logger.error(f"[Hibrid] Eroare: {e}")


# ─────────────────────────────────────────────
# FIX MAJOR: Inferență tranzitivă corectată
# ─────────────────────────────────────────────
async def perform_transitive_inference(max_inferences: int):
    """
    Mod tranzitiv: inferențe prin lanțuri, cu aceleași corecții ca hibrid.

    MODIFICĂRI față de versiunea anterioară:
    1. Același prag de intrare ridicat: INFERENCE_GRAPH_MIN_WEIGHT (0.72).
    2. Lanțuri de lungime 2: (w_ab * w_bc)^0.5 fără bonus × 1.2.
    3. Lanțuri de lungime 3: (w_ab * w_bc * w_cd)^(1/3) fără bonus × 1.5.
    4. Pragul de acceptare: INFERENCE_THRESHOLD["tranzitiv"] (0.68).
    5. Scor penalizat mai agresiv pentru lanțuri lungi.
    6. Iterăm pe noduri puține, limitate strict.
    """
    global edges_data, edges_type, posts_history

    if len(edges_data) < 10:
        return

    try:
        # FIX 1: același prag ridicat, numai muchii directe
        G = nx.Graph()
        for (u, v), w in edges_data.items():
            if w > INFERENCE_GRAPH_MIN_WEIGHT and edges_type.get((u, v), "direct") == "direct":
                G.add_edge(u, v, weight=w)

        if G.number_of_edges() < 5:
            return

        degrees    = dict(G.degree())
        median_deg = sorted(degrees.values())[len(degrees) // 2] if degrees else 1
        candidate_nodes = [n for n, d in degrees.items() if d <= median_deg + 2]

        inferences = []

        # Lanțuri de lungime 2 (A–B–C)
        for b in candidate_nodes[:25]:  # FIX: limitat strict
            neighbors = list(G.neighbors(b))
            for i, a in enumerate(neighbors):
                for c in neighbors[i + 1:]:
                    if G.has_edge(a, c):
                        continue
                    pair     = tuple(sorted([a, c]))
                    existing = edges_data.get(pair, 0.0)
                    if existing > INFERENCE_GRAPH_MIN_WEIGHT * INFERRED_SCORE_PENALTY:
                        continue
                    w_ab     = G[a][b]["weight"]
                    w_bc     = G[b][c]["weight"]
                    # FIX 2: fără bonus × 1.2
                    inferred = (w_ab * w_bc) ** 0.5
                    if inferred > INFERENCE_THRESHOLD["tranzitiv"]:
                        inferences.append((a, c, inferred, str(b), 2))

        # Lanțuri de lungime 3 (A–B–C–D) — numai dacă avem puține inferențe de lungime 2
        if len(inferences) < max_inferences * 0.5:
            nodes_list = candidate_nodes[:10]  # FIX: foarte limitat
            for a in nodes_list:
                for b in G.neighbors(a):
                    w_ab = G[a][b]["weight"]
                    for c in G.neighbors(b):
                        if c == a:
                            continue
                        w_bc = G[b][c]["weight"]
                        for d in G.neighbors(c):
                            if d in (b, a) or G.has_edge(a, d):
                                continue
                            pair     = tuple(sorted([a, d]))
                            existing = edges_data.get(pair, 0.0)
                            if existing > INFERENCE_GRAPH_MIN_WEIGHT * INFERRED_SCORE_PENALTY * 0.8:
                                continue
                            w_cd     = G[c][d]["weight"]
                            # FIX 3: fără bonus × 1.5
                            inferred = (w_ab * w_bc * w_cd) ** (1 / 3)
                            if inferred > INFERENCE_THRESHOLD["tranzitiv"]:
                                inferences.append((a, d, inferred, f"{b}→{c}", 3))

        inferences.sort(key=lambda x: -x[2])
        added = 0
        for a, c, w, via, length in inferences[:max_inferences]:
            pair = tuple(sorted([a, c]))
            if pair not in edges_data:
                # FIX: penalizare mai mare pentru lanțuri lungi
                penalty = INFERRED_SCORE_PENALTY if length == 2 else INFERRED_SCORE_PENALTY * 0.7
                edges_data[pair] = min(w * penalty, 10.0)
                edges_type[pair] = "inferred"
                posts_history.append({
                    "type":       "tranzitiv",
                    "ch1":        a,
                    "ch2":        c,
                    "via":        via,
                    "length":     length,
                    "confidence": round(w, 3),
                    "time":       datetime.now().strftime("%H:%M:%S"),
                })
                if len(posts_history) > 500:
                    posts_history[:] = posts_history[-500:]
                added += 1
                logger.info(f"[Tranzitiv-{length}] {a} ~ {c} (via {via}, {w:.2f})")

        if added > 0:
            logger.info(f"[Tranzitiv] Adăugate {added} muchii noi")

    except Exception as e:
        logger.error(f"[Tranzitiv] Eroare: {e}")


# ─────────────────────────────────────────────
# Analiza unei perechi de canale
# ─────────────────────────────────────────────
def _analyse_pair_global(c1: str, c2: str, mode: str, threshold: float):
    # BUG 2 FIX: cosine_similarity e None până când modelele NLP termină de
    # încărcat. Fără acest guard, apelul de la linia 1134 (cosine_similarity(m1,m2))
    # genera TypeError: 'NoneType' object is not callable și îneca analyzer-ul.
    if cosine_similarity is None:
        return 0.0, None

    MAX_HITS = 10
    score    = 0.0
    match    = None

    if mode == "stylography":
        lang1 = ch_lang_cache.get(c1, "other")
        lang2 = ch_lang_cache.get(c2, "other")
        if lang1 != lang2 or lang1 == "other":
            return 0.0, None
        v1 = ch_style_cache.get(c1)
        v2 = ch_style_cache.get(c2)
        if v1 is None or v2 is None:
            return 0.0, None
        active = (np.abs(v1) + np.abs(v2)) > 1e-5
        if active.sum() < 4:
            return 0.0, None
        w  = _STYLE_DIM_WEIGHTS * active.astype(float)
        w /= w.sum() + 1e-9
        bc_diss = np.sum(w * np.abs(v1 - v2)) / (np.sum(w * (np.abs(v1) + np.abs(v2))) + 1e-9)
        bc_sim  = 1.0 - bc_diss
        conv_w  = 0.0
        active_w_sum = float(_STYLE_DIM_WEIGHTS[active].sum()) + 1e-9
        for i, (a, b) in enumerate(zip(v1, v2)):
            if not active[i]:
                continue
            denom = max(abs(a), abs(b), 1e-5)
            if abs(a - b) / denom < 0.15:
                conv_w += _STYLE_DIM_WEIGHTS[i]
        conv_w /= active_w_sum
        v1w = np.sqrt(np.abs(v1) * _STYLE_DIM_WEIGHTS)
        v2w = np.sqrt(np.abs(v2) * _STYLE_DIM_WEIGHTS)
        cos = float(cosine_similarity(v1w.reshape(1, -1), v2w.reshape(1, -1))[0, 0])
        combined = 0.45 * bc_sim + 0.35 * conv_w + 0.20 * cos
        if combined < threshold:
            return 0.0, None
        n_agree  = sum(1 for i, (a, b) in enumerate(zip(v1, v2))
                       if active[i] and abs(a - b) / max(abs(a), abs(b), 1e-5) < 0.15)
        n_active = int(active.sum())
        dim_details = {
            name: {
                "c1":       round(float(v1[i]), 4),
                "c2":       round(float(v2[i]), 4),
                "rel_diff": round(abs(float(v1[i]) - float(v2[i])) / max(abs(float(v1[i])), abs(float(v2[i])), 1e-5), 3),
                "agrees":   abs(float(v1[i]) - float(v2[i])) / max(abs(float(v1[i])), abs(float(v2[i])), 1e-5) < 0.15,
                "weight":   round(float(_STYLE_DIM_WEIGHTS[i]), 4),
            }
            for i, name in enumerate(_STYLE_DIM_NAMES) if active[i]
        }

        def pick_representative(ch: str, vec: np.ndarray) -> str:
            msgs = ch_msgs_cache.get(ch, [])
            if not msgs:
                return ""
            best_msg, best_dist = msgs[0], float("inf")
            for m in msgs:
                if not m.strip():
                    continue
                words = m.split()
                n_char = len(m) + 1
                proxy = np.array([
                    min(len(m) / 500.0, 1.0),
                    len(re.findall(r"[.,!?;:]", m)) / n_char,
                    len(re.findall(r"[^\w\s]", m)) / n_char,
                    min(sum(len(w) for w in words) / ((len(words) + 1) * 15.0), 1.0),
                ])
                dist = float(np.linalg.norm(proxy - vec[:4]))
                if dist < best_dist:
                    best_dist, best_msg = dist, m
            return best_msg

        ex1        = pick_representative(c1, v1)
        ex2        = pick_representative(c2, v2)
        edge_score = combined * 3.0
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

    m1, orig1 = get_embedding_matrix(c1)
    m2, orig2 = get_embedding_matrix(c2)
    if m1 is None or m2 is None or m1.shape[0] == 0 or m2.shape[0] == 0:
        return 0.0, None

    sims = cosine_similarity(m1, m2)

    if mode == "repost":
        hits   = np.where(sims > threshold)
        n_hits = len(hits[0])
        if n_hits == 0:
            return 0.0, None
        score     = n_hits / 5.0
        best_flat = int(np.argmax(sims))
        bi, bj    = divmod(best_flat, sims.shape[1])
        match = {
            "type":    "repost",
            "ch1":     c1,
            "ch2":     c2,
            "score":   float(sims[bi, bj]),
            "ch1_msg": orig1[bi] if bi < len(orig1) else "",
            "ch2_msg": orig2[bj] if bj < len(orig2) else "",
            "n_hits":  n_hits,
        }
        return score, match

    above = sims > threshold
    if not above.any():
        return 0.0, None

    hit_rows, hit_cols = np.where(above)
    hit_scores = sims[hit_rows, hit_cols]
    top_idx    = np.argsort(hit_scores)[::-1][:MAX_HITS]
    hit_pairs  = list(zip(hit_rows[top_idx], hit_cols[top_idx]))

    total_strength = 0.0
    best_match     = None
    best_val       = 0.0

    for (i, j) in hit_pairs:
        if i >= len(orig1) or j >= len(orig2):
            continue
        m1_text = orig1[i]
        m2_text = orig2[j]
        sem_sim = float(sims[i, j])
        w1 = set(m1_text.lower().split())
        w2 = set(m2_text.lower().split())
        common = w1 & w2
        if len(common) > 1 and len(m1_text) < 50:
            if len(common) / max(len(w1), len(w2), 1) > 0.4:
                continue
        base_strength = sem_sim
        nlp1 = analyse_text(m1_text)
        nlp2 = analyse_text(m2_text)
        nlp_bonus = 0.0
        if nlp1 and nlp2:
            if nlp1.get("sentiment", {}).get("label") == nlp2.get("sentiment", {}).get("label"):
                nlp_bonus += 0.3
            e1 = {e["name"] for e in nlp1.get("entities", []) if len(e["name"]) > 3}
            e2 = {e["name"] for e in nlp2.get("entities", []) if len(e["name"]) > 3}
            if e1 & e2:
                nlp_bonus += 0.2
            update_entity_stats(nlp1)
            update_entity_stats(nlp2)
        pair_strength   = base_strength * (1.0 + nlp_bonus)
        total_strength += pair_strength
        if pair_strength > best_val:
            best_val   = pair_strength
            best_match = {
                "type":      "similar",
                "ch1":       c1,
                "ch2":       c2,
                "score":     round(sem_sim, 4),
                "nlp_bonus": round(nlp_bonus, 3),
                "ch1_msg":   m1_text,
                "ch2_msg":   m2_text,
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
        # keep_alive ELIMINAT: uvicorn are propriul ping WS. Două surse de
        # ping pe aceeași conexiune → "keepalive ping timeout" cod 1011.

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
            "type":             "state_update",
            "channels":         list(channels_set),
            "running":          running,
            "paused":           paused,
            "target_channel":   target_channel,
            "keywords":         keywords_list,
            "nlp_ready":        nlp_ready,
            "nlp_status":       nlp_status,
            "similarity_mode":  similarity_mode,
            "global_entities":  global_entities,
        }
        await self.broadcast(safe_json_dumps(state))


manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    logger.info("Pornire aplicație — inițiere încărcare modele NLP.")
    db_init()
    loop = asyncio.get_running_loop()
    start_nlp_loading(loop)


# ─────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global running, paused, channels_set, nodes_data, edges_data, edges_type, posts_history
    global target_channel, keywords_list, similarity_mode, analysis_mode
    global global_entities, ch_msgs_cache, ch_embs_cache, ch_style_cache
    global background_tasks, dirty_channels

    await manager.connect(websocket)
    try:
        while True:
            try:
                # Timeout mărit la 60s — cu 203 canale, broadcast-urile din
                # analyzer vin la fiecare 10s, deci 60s e suficient de relaxat
                # fără a lăsa conexiunea moartă prea mult timp.
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                cmd  = json.loads(data)
                action = cmd.get("action")

                # Ignorăm pong-urile venite de la client (răspuns la ping-ul uvicorn)
                if action == "pong" or cmd.get("type") == "pong":
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
                    for k in [k for k in edges_data if ch in k]:
                        edges_data.pop(k, None)
                        edges_type.pop(k, None)
                    await manager.send_state()

                elif action == "reset":
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
                    await manager.broadcast(safe_json_dumps({"type": "clear_graph"}))
                    await manager.send_state()

                elif action == "set_target":
                    ch = cmd.get("channel", "").strip()
                    target_channel = ch if (ch in channels_set and target_channel != ch) else None
                    await manager.send_state()

                elif action == "set_mode":
                    new_mode = cmd.get("mode", "hibrid")
                    if new_mode in ("direct", "hibrid", "tranzitiv"):
                        similarity_mode = new_mode
                        # FIX: la schimbarea modului, ștergem NUMAI muchiile inférate,
                        # păstrând relațiile directe confirmate semantic.
                        inferred_pairs = [p for p, t in edges_type.items() if t == "inferred"]
                        for p in inferred_pairs:
                            edges_data.pop(p, None)
                            edges_type.pop(p, None)
                        dirty_channels.update(channels_set)
                        logger.info(f"✅ Mod similaritate: {new_mode} — {len(inferred_pairs)} muchii inférate șterse")
                        await manager.send_state()
                    elif new_mode in ("repost", "similar", "stylography"):
                        analysis_mode = new_mode
                        dirty_channels.update(channels_set)
                        logger.info(f"✅ Mod analiză: {new_mode}")
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
                    # BUG 4 FIX: oprirea nu anula task-urile asyncio — scraper-ul
                    # și analyzer-ul continuau să ruleze în background după "stop".
                    # Acum le anulăm explicit și curățăm lista.
                    for t in background_tasks:
                        if not t.done():
                            t.cancel()
                    background_tasks.clear()
                    await manager.send_state()

                elif action == "pause":
                    paused = not paused
                    await manager.send_state()

                elif action == "save_request":
                    payload = {
                        "channels":      list(channels_set),
                        "edges":         [
                            {"from": k[0], "to": k[1], "strength": v, "type": edges_type.get(k, "direct")}
                            for k, v in edges_data.items()
                        ],
                        "entities":      global_entities,
                        "posts_history": posts_history[-50:],
                    }
                    await manager.broadcast(safe_json_dumps({"type": "save_file", "data": payload}))

                elif action == "full_backup":
                    try:
                        filename = full_backup()
                        await websocket.send_text(safe_json_dumps({
                            "type":     "backup_complete",
                            "success":  True,
                            "filename": filename,
                            "message":  f"Backup complet salvat: {filename}",
                        }))
                    except Exception as e:
                        await websocket.send_text(safe_json_dumps({
                            "type":    "backup_complete",
                            "success": False,
                            "message": f"Eroare la backup: {str(e)}",
                        }))

                elif action == "restore_backup":
                    filename = cmd.get("filename")
                    if filename and os.path.exists(filename):
                        success = restore_full_backup(filename)
                        if success:
                            await manager.send_state()
                            G          = build_graph()
                            graph_data = await send_graph_update(websocket, G)
                            await websocket.send_text(safe_json_dumps({
                                "type":  "graph_update",
                                "nodes": graph_data["nodes"],
                                "edges": graph_data["edges"],
                            }))
                            await websocket.send_text(safe_json_dumps({
                                "type":    "restore_complete",
                                "success": True,
                                "message": f"Backup restaurat: {filename}",
                            }))
                        else:
                            await websocket.send_text(safe_json_dumps({
                                "type":    "restore_complete",
                                "success": False,
                                "message": "Eroare la restaurarea backup-ului",
                            }))
                    else:
                        await websocket.send_text(safe_json_dumps({
                            "type":    "restore_complete",
                            "success": False,
                            "message": "Fișierul nu există",
                        }))

                elif action == "list_backups":
                    backups = []
                    if os.path.exists("backups"):
                        for f in os.listdir("backups"):
                            if f.endswith(".pkl") or f.endswith(".json"):
                                filepath = os.path.join("backups", f)
                                stat     = os.stat(filepath)
                                backups.append({
                                    "filename": f,
                                    "size":     stat.st_size,
                                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                })
                    await websocket.send_text(safe_json_dumps({
                        "type":    "backups_list",
                        "backups": sorted(backups, key=lambda x: x["modified"], reverse=True),
                    }))

                elif action == "save_project":
                    project_state = save_project_state()
                    project_hash  = hash_project_state(project_state)
                    await websocket.send_text(safe_json_dumps({
                        "type":     "project_saved",
                        "data":     project_state,
                        "hash":     project_hash,
                        "filename": f"telegram_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tgm",
                    }))

                elif action == "load_project":
                    project_data = cmd.get("data")
                    if project_data and load_project_state(project_data):
                        await manager.send_state()
                        G          = build_graph()
                        graph_data = await send_graph_update(websocket, G)
                        await websocket.send_text(safe_json_dumps({
                            "type":  "graph_update",
                            "nodes": graph_data["nodes"],
                            "edges": graph_data["edges"],
                        }))
                        await websocket.send_text(safe_json_dumps({
                            "type":    "project_loaded",
                            "success": True,
                            "message": f"Proiect încărcat cu {len(channels_set)} canale",
                        }))
                    else:
                        await websocket.send_text(safe_json_dumps({
                            "type":    "project_loaded",
                            "success": False,
                            "message": "Eroare la încărcarea proiectului",
                        }))

                elif action == "export_project":
                    project_state = save_project_state()
                    await websocket.send_text(safe_json_dumps({
                        "type":     "project_export",
                        "data":     project_state,
                        "filename": f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tgm",
                    }))

                elif action == "import_project":
                    project_data = cmd.get("data")
                    if project_data and load_project_state(project_data):
                        await manager.send_state()
                        G          = build_graph()
                        graph_data = await send_graph_update(websocket, G)
                        await websocket.send_text(safe_json_dumps({
                            "type":  "graph_update",
                            "nodes": graph_data["nodes"],
                            "edges": graph_data["edges"],
                        }))
                        await websocket.send_text(safe_json_dumps({
                            "type":    "project_imported",
                            "success": True,
                        }))
                    else:
                        await websocket.send_text(safe_json_dumps({
                            "type":    "project_imported",
                            "success": False,
                        }))

            except asyncio.TimeoutError:
                # La timeout simplu continuăm — nu trimitem ping manual,
                # uvicorn se ocupă de keepalive la nivel de transport.
                continue
            except Exception as e:
                err_str = str(e)
                # Deconectare normală — nu logăm ca eroare
                if "1000" in err_str or "1001" in err_str or "ConnectionClosed" in type(e).__name__:
                    logger.info(f"WebSocket închis normal: {e}")
                else:
                    logger.error(f"WebSocket error: {e}")
                break

    except WebSocketDisconnect:
        logger.info("WebSocket deconectat normal")
    except Exception as e:
        logger.error(f"WebSocket eroare neașteptată: {e}")
    finally:
        manager.disconnect(websocket)
        logger.info("WebSocket conexiune închisă")


# ─────────────────────────────────────────────
# Background scraper
# ─────────────────────────────────────────────
async def background_scraper():
    global running, paused, channels_set, nodes_data
    global ch_msgs_cache, ch_embs_cache, ch_style_cache, dirty_channels

    semaphore  = asyncio.Semaphore(5)
    BATCH_SIZE = 20

    async def process_single_channel(ch: str):
        async with semaphore:
            if not running or paused:
                return
            try:
                await manager.broadcast(safe_json_dumps({"type": "status", "msg": f"Scanez {ch}..."}))
                data = await asyncio.wait_for(
                    asyncio.to_thread(scrape_channel, ch),
                    timeout=20.0,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[Scraper] Timeout {ch}, sărit.")
                return
            except Exception as e:
                logger.error(f"[Scraper] Eroare {ch}: {e}")
                return

            nodes_data[ch] = {
                "id":          ch,
                "label":       data["title"],
                "subscribers": data["subscribers"],
            }
            msgs = data.get("messages", [])
            if not msgs:
                await manager.broadcast(safe_json_dumps({"type": "node_update", "node": nodes_data[ch]}))
                return

            ch_lang_cache[ch]  = detect_language(" ".join(msgs[:10]))
            ch_style_cache[ch] = await asyncio.to_thread(get_stylometric_fingerprint, msgs)

            # Deduplicare față de tot istoricul din SQLite (nu doar ultimele 50 din RAM)
            existing_set = await asyncio.to_thread(db_get_all_messages_set, ch)

            filtered = (
                [m for m in msgs if text_matches_keywords(m, keywords_list)]
                if keywords_list else msgs
            )
            new_msgs = [m for m in filtered if m not in existing_set]

            if new_msgs:
                # Persistăm în SQLite cu timestamp — sursa de adevăr pentru fereastra de 3 zile
                await asyncio.to_thread(db_insert_messages, ch, new_msgs)
                # Rebuilding embeddings din fereastra glisantă (3 zile din SQLite)
                await asyncio.to_thread(update_embeddings_incremental, ch, new_msgs)
                # BUG 3 FIX: analyse_text folosește ner_pipeline și sentiment_pipeline
                # care sunt None până când nlp_ready devine True. Fără această
                # verificare, apelul returnează {} silențios dar consumă timp inutil.
                if nlp_ready:
                    for m in new_msgs[:5]:
                        await asyncio.to_thread(analyse_text, m)
                dirty_channels.add(ch)
                logger.info(f"[Scraper] {ch}: {len(new_msgs)} mesaje noi → dirty")

            await manager.broadcast(safe_json_dumps({"type": "node_update", "node": nodes_data[ch]}))

    _last_purge = datetime.now()

    while running:
        # BUG 1 FIX: scraper-ul NU are nevoie de nlp_ready — modelele NLP
        # sunt folosite doar de analyzer. Condiția anterioară bloca complet
        # scraping-ul până când modelele (câteva minute) terminau de încărcat.
        if paused or not channels_set:
            await asyncio.sleep(2)
            continue

        # Curățăm mesajele vechi din SQLite o dată la 24h
        if (datetime.now() - _last_purge).total_seconds() > 86400:
            await asyncio.to_thread(db_purge_old_messages, 7)
            _last_purge = datetime.now()

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
    global running, paused, edges_data, edges_type, posts_history, dirty_channels
    global channels_set, nodes_data, target_channel, similarity_mode, analysis_mode
    global global_entities, nlp_ready, similarity_model, cosine_similarity

    MAX_PAIRS_PER_CYCLE = 300

    # ── Stare round-robin ────────────────────────────────────────────────────
    # Indexul din lista plată de perechi de unde continuăm în ciclul următor.
    # Garantăm că TOATE cele 20.503 perechi sunt analizate periodic (~68 cicluri),
    # nu doar primele 300 (primele ~25 canale din ordinea inserției cache-ului).
    rr_offset: int = 0

    while running:
        if paused or not channels_set or not nlp_ready or not similarity_model or not cosine_similarity:
            await asyncio.sleep(5)
            continue

        try:
            n_ch    = len(channels_set)
            n_dirty = len(dirty_channels)

            all_keys = [
                k for k in ch_embs_cache.keys()
                if ch_embs_cache[k] is not None
                and "matrix" in ch_embs_cache[k]
                and ch_embs_cache[k]["matrix"].shape[0] > 0
            ]
            total_pairs = len(all_keys) * (len(all_keys) - 1) // 2 if len(all_keys) >= 2 else 0

            await manager.broadcast(safe_json_dumps({
                "type": "status",
                "msg":  (f"Analiză: {n_ch} canale, {n_dirty} dirty, "
                         f"{total_pairs} perechi, offset={rr_offset}, mod={similarity_mode}"),
            }))
            logger.info(f"[Analyzer] canale={n_ch} dirty={n_dirty} perechi={total_pairs} "
                        f"offset={rr_offset} mod={similarity_mode}/{analysis_mode}")

            # Decay diferențiat pe tip de muchie
            for pair in list(edges_data.keys()):
                edges_data[pair] = _decay_edge(pair, edges_data[pair])

            threshold     = THRESHOLD.get(analysis_mode, 0.6)
            current_dirty = set(dirty_channels)

            # ── Construim listele de perechi ──────────────────────────────────
            # P0/1/2: dirty, noi, inférate slabe — procesate întotdeauna primele
            # all_rr_pairs: perechi existente confirmate — procesate în rotație
            priority_0_1_2 = []
            all_rr_pairs   = []

            if len(all_keys) >= 2:
                for i, c1 in enumerate(all_keys):
                    for c2 in all_keys[i + 1:]:
                        pair             = tuple(sorted([c1, c2]))
                        is_affected      = bool(current_dirty & {c1, c2})
                        is_new_pair      = pair not in edges_data
                        is_inferred_weak = (
                            edges_type.get(pair) == "inferred"
                            and edges_data.get(pair, 0.0) < 1.0
                        )
                        if is_affected:
                            priority_0_1_2.append((0, pair, c1, c2))
                        elif is_new_pair:
                            priority_0_1_2.append((1, pair, c1, c2))
                        elif is_inferred_weak:
                            priority_0_1_2.append((2, pair, c1, c2))
                        else:
                            all_rr_pairs.append((pair, c1, c2))

            priority_0_1_2.sort(key=lambda x: x[0])

            # Buget pentru round-robin = MAX - perechi prioritare
            if similarity_mode == "direct":
                rr_budget = max(0, MAX_PAIRS_PER_CYCLE - len(priority_0_1_2))
            elif similarity_mode == "hibrid":
                rr_budget = max(0, int(MAX_PAIRS_PER_CYCLE * 0.7) - len(priority_0_1_2))
            else:
                rr_budget = max(0, int(MAX_PAIRS_PER_CYCLE * 0.5) - len(priority_0_1_2))

            # Felie round-robin din offset curent
            n_rr = len(all_rr_pairs)
            if n_rr > 0 and rr_budget > 0:
                if rr_offset >= n_rr:
                    rr_offset = 0
                end_idx   = min(rr_offset + rr_budget, n_rr)
                rr_slice  = all_rr_pairs[rr_offset:end_idx]
                rr_offset = end_idx % n_rr
            else:
                rr_slice  = []
                rr_offset = 0

            # Lista finală de analizat
            priority_slice = priority_0_1_2[:MAX_PAIRS_PER_CYCLE - len(rr_slice)]
            pairs_to_check = [(p, pair, c1, c2) for p, pair, c1, c2 in priority_slice]
            pairs_to_check += [(3, pair, c1, c2) for pair, c1, c2 in rr_slice]

            logger.info(f"[Analyzer] P0/1/2={len(priority_slice)} rr={len(rr_slice)} "
                        f"total={len(pairs_to_check)} rr_offset->{rr_offset}/{n_rr}")

            # ── Analizăm perechile selectate ──────────────────────────────────
            for _, pair, c1, c2 in pairs_to_check:
                if not running:
                    break
                edge_score, match = await asyncio.to_thread(
                    _analyse_pair_global, c1, c2, analysis_mode, threshold
                )
                if edge_score > 0:
                    existing = edges_data.get(pair, 0.0)
                    edges_data[pair] = min(existing + edge_score, 10.0)
                    edges_type[pair] = "direct"
                    # Persistăm scorul în SQLite — scorul cumulativ nu scade niciodată,
                    # reflectă întreaga istorie a coordonării pe durata monitorizării.
                    ch1, ch2 = pair
                    await asyncio.to_thread(db_update_edge_cumulative, ch1, ch2, edge_score)
                    if match:
                        match["time"] = datetime.now().strftime("%H:%M:%S")
                        posts_history.append(match)
                        if len(posts_history) > 500:
                            posts_history[:] = posts_history[-500:]
                        await manager.broadcast(
                            safe_json_dumps({"type": "new_post_match", "data": match})
                        )

            analyzed_pairs = len(pairs_to_check)

            # Inferențe (hibrid/tranzitiv) cu buget rămas
            if similarity_mode == "hibrid":
                remaining = MAX_PAIRS_PER_CYCLE - analyzed_pairs
                if remaining > 10:
                    await perform_hybrid_inference(min(remaining // 3, 30))
            elif similarity_mode == "tranzitiv":
                remaining = MAX_PAIRS_PER_CYCLE - analyzed_pairs
                if remaining > 5:
                    await perform_transitive_inference(min(remaining // 3, 50))

            # Eliminăm muchiile moarte
            dead = [p for p, s in edges_data.items() if s < 0.15]
            for p in dead:
                edges_data.pop(p, None)
                edges_type.pop(p, None)

            dirty_channels -= current_dirty

            # ── Construim graful pe scoruri cumulative din SQLite ──────────────
            # Graful pentru Louvain folosește scorurile cumulative (nu decay-ul din RAM)
            # pentru că scopul este detectarea coordonării pe termen lung (zile).
            # edges_data (RAM, cu decay) rămâne pentru logica de prioritizare a perechilor.
            cumulative_scores = await asyncio.to_thread(db_get_cumulative_scores)
            G = nx.Graph()
            for (u, v), w in cumulative_scores.items():
                if u in channels_set and v in channels_set:
                    G.add_edge(u, v, weight=w)
            for c in list(channels_set):
                if c not in G:
                    G.add_node(c)

            comms = {}
            try:
                from networkx.algorithms.community import louvain_communities
                if G.size() > 0:
                    # FIX: seed=42 elimină nedeterminismul Louvain — numărul de comunități
                    # nu mai oscilează între cicluri când graful este identic sau aproape identic.
                    for idx, s in enumerate(louvain_communities(G, weight="weight", seed=42)):
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

            f_edges = []
            for k, v in edges_data.items():
                if k[0] in displayed and k[1] in displayed:
                    etype    = edges_type.get(k, "direct")
                    cum_score = cumulative_scores.get(k, cumulative_scores.get((k[1], k[0]), 0.0))
                    f_edges.append({
                        "from":  k[0],
                        "to":    k[1],
                        "value": round(cum_score if cum_score > 0 else v, 3),
                        "title": (f"Sesiune: {v:.2f} | Cumulativ: {cum_score:.2f} "
                                  f"({'inferată' if etype == 'inferred' else 'directă'})"),
                        "type":  etype,
                    })

            await manager.broadcast(safe_json_dumps({
                "type":  "graph_update",
                "nodes": f_nodes,
                "edges": f_edges,
            }))

        except Exception as e:
            logger.error(f"[Analyzer] Eroare: {e}", exc_info=True)

        await asyncio.sleep(10)


if __name__ == "__main__":
    import uvicorn
    os.makedirs("backups", exist_ok=True)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # Ping WebSocket la fiecare 30s, timeout răspuns 20s.
        # Valorile implicite uvicorn (20s/20s) sunt prea agresive când
        # event loop-ul e ocupat cu scraping/encoding NLP pe 200+ canale.
        ws_ping_interval=30,
        ws_ping_timeout=20,
    )
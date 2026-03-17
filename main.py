import asyncio
import json
import re
import os
import logging
from datetime import datetime
import threading
import pickle
import base64
import hashlib
import glob
from pathlib import Path

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
similarity_model  = None
cosine_similarity = None
ner_pipeline      = None
sentiment_pipeline = None
nlp_ready  = False
nlp_status = "Așteptare..."

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ─────────────────────────────────────────────
# Backup endpoint
# ─────────────────────────────────────────────
@app.get("/api/backup_now")
async def backup_now():
    """Salvează datele curente în folderul backups"""
    try:
        # Creează folderul dacă nu există
        os.makedirs("backups", exist_ok=True)
        
        # Nume fișier cu timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backups/backup_{timestamp}.json"
        
        # Colectează datele
        backup_data = {
            "timestamp": timestamp,
            "channels": list(channels_set),
            "target": target_channel,
            "keywords": keywords_list,
            "mode": similarity_mode,
            "nodes": nodes_data,
            "edges": {f"{k[0]}|{k[1]}": v for k, v in edges_data.items()},
            "posts_history": posts_history[-100:],
            "global_entities": global_entities,
            "stats": {
                "total_channels": len(channels_set),
                "total_edges": len(edges_data)
            }
        }
        
        # Salvează în fișier
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2)
        
        # Log în consolă
        print(f"✅ Backup creat: {filename}")
        
        return {
            "success": True, 
            "message": f"Backup salvat: {filename}",
            "filename": filename,
            "stats": backup_data["stats"]
        }
        
    except Exception as e:
        print(f"❌ Eroare backup: {e}")
        return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────
# Endpointuri pentru gestionarea proiectelor (.tgm)
# ─────────────────────────────────────────────
PROJECTS_DIR = "projects"
os.makedirs(PROJECTS_DIR, exist_ok=True)

@app.get("/api/project/list")
async def list_projects():
    """Listează toate proiectele salvate (.tgm)"""
    try:
        projects = []
        for filepath in glob.glob(f"{PROJECTS_DIR}/*.tgm"):
            filename = os.path.basename(filepath)
            stat = os.stat(filepath)
            
            # Încearcă să citești numele din fișier
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    project_name = data.get('name', filename)
            except (json.JSONDecodeError, OSError):
                project_name = filename
            
            projects.append({
                "filename": filename,
                "name": project_name,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size": stat.st_size
            })
        
        # Sortează după data modificării (cele mai noi primele)
        projects.sort(key=lambda x: x["modified"], reverse=True)
        
        return {"status": "success", "projects": projects}
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/project/save")
async def save_project(request: Request):
    """Salvează un proiect COMPLET în format .tgm (JSON)"""
    try:
        data = await request.json()
        project_name = data.get('name', 'unnamed')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Elimină caracterele speciale din nume
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        filename = f"{PROJECTS_DIR}/{safe_name}_{timestamp}.tgm"
        
        # Pregătește backup COMPLET (absolut tot)
        project_data = {
            "version": "2.0",
            "name": project_name,
            "saved_at": datetime.now().isoformat(),
            
            # Configurații
            "channels": list(channels_set),
            "target_channel": target_channel,
            "keywords": keywords_list,
            "similarity_mode": similarity_mode,
            "analysis_mode": analysis_mode,
            "running": running,
            "paused": paused,
            
            # Date rețea
            "nodes": nodes_data,
            "edges": {f"{k[0]}|{k[1]}": v for k, v in edges_data.items()},
            "posts_history": posts_history,
            
            # Cache-uri text
            "ch_msgs_cache": ch_msgs_cache,
            "ch_msgs_set": {k: list(v) for k, v in ch_msgs_set.items()},
            
            # Cache-uri analiză
            "ch_style_cache": {},
            "ch_lang_cache": ch_lang_cache,
            
            # FIX #6: ch_embs_cache inițializat mereu ca {}, nu condiționat
            "ch_embs_cache": {},
            
            # Cache NLP
            "nlp_msg_cache": {str(k): v for k, v in nlp_msg_cache.items()},
            
            # Entități
            "global_entities": global_entities,
            
            # Stare sistem
            "dirty_channels": list(dirty_channels),
            
            # Statistici
            "stats": {
                "total_channels": len(channels_set),
                "total_edges": len(edges_data),
                "total_messages": sum(len(v) for v in ch_msgs_cache.values()),
            }
        }
        
        # Procesează stilometria (numpy arrays -> liste)
        for k, v in ch_style_cache.items():
            if isinstance(v, np.ndarray):
                project_data["ch_style_cache"][k] = v.tolist()
            else:
                project_data["ch_style_cache"][k] = v
        
        # Procesează embedding-urile (numpy arrays -> liste)
        for ch, emb_data in ch_embs_cache.items():
            if emb_data and "matrix" in emb_data and emb_data["matrix"] is not None:
                project_data["ch_embs_cache"][ch] = {
                    "orig_texts": emb_data["orig_texts"],
                    "matrix": emb_data["matrix"].tolist()
                }
        
        # Salvează în format JSON (cu extensia .tgm)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Proiect .tgm salvat: {filename}")
        return {"status": "success", "filename": filename}
    except Exception as e:
        logger.error(f"Error saving project: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/project/load")
async def load_project(request: Request):
    """Încarcă un proiect din format .tgm"""
    try:
        data = await request.json()
        filename = data.get('filename')
        
        if not filename:
            return {"status": "error", "message": "Filename required"}
        
        filepath = os.path.join(PROJECTS_DIR, filename)
        if not os.path.exists(filepath):
            return {"status": "error", "message": "File not found"}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            project_data = json.load(f)
        
        return {"status": "success", "data": project_data}
    except Exception as e:
        logger.error(f"Error loading project: {e}")
        return {"status": "error", "message": str(e)}


@app.delete("/api/project/delete")
async def delete_project(request: Request):
    """Șterge un proiect"""
    try:
        data = await request.json()
        filename = data.get('filename')
        
        if not filename:
            return {"status": "error", "message": "Filename required"}
        
        filepath = os.path.join(PROJECTS_DIR, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"✅ Proiect șters: {filename}")
            return {"status": "success"}
        else:
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
channels_set: set = set()
running    = False
paused     = False
target_channel  = None
keywords_list: list = []
# MODIFICARE: Am adăugat noile moduri de similaritate
similarity_mode = "hibrid"      # "direct" | "hibrid" | "tranzitiv"
analysis_mode = "similar"       # păstrăm modul intern "repost" | "similar" | "stylography"

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

# NOU: Praguri pentru modurile de inferență
INFERENCE_THRESHOLD = {
    "direct":   0.72,  # pragul actual pentru comparații directe
    "hibrid":   0.65,  # mai permisiv pentru hibrid
    "tranzitiv": 0.60,  # și mai permisiv pentru tranzitiv
}

# Prag minim pentru scorul de aliniere NLP (bonus, nu conditie blocanta)
NLP_ALIGNMENT_MIN = 0.3

# Decay de baza per ciclu; relatiile puternice decad mai lent (logaritmic).
DECAY_BASE = 0.88

# ─────────────────────────────────────────────
# Funcția pentru backup complet (METODA 4)
# ─────────────────────────────────────────────
def full_backup():
    """
    Backup complet - salvează ABSOLUT TOATE datele în format pickle și JSON
    Include toate cache-urile, embedding-urile, și starea completă a sistemului.
    """
    # Asigură-te că directorul de backup există
    os.makedirs("backups", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backups/full_backup_{timestamp}.pkl"  # folosește pickle pentru obiecte complexe
    
    # Pregătește datele pentru backup
    backup = {
        "timestamp": timestamp,
        "version": "2.0",
        
        # Date de bază
        "channels": list(channels_set),
        "target": target_channel,
        "keywords": keywords_list,
        "mode": similarity_mode,
        "analysis_mode": analysis_mode,  # BUG A FIX: lipsea — restore_full_backup îl citea dar nu era salvat
        "running": running,
        "paused": paused,
        
        # Graf
        "nodes": nodes_data,
        "edges": {f"{k[0]}|{k[1]}": v for k, v in edges_data.items()},
        
        # Cache-uri text
        "ch_msgs_cache": ch_msgs_cache,
        "ch_msgs_set": {k: list(v) for k, v in ch_msgs_set.items()},
        
        # Embedding-uri (partea critică!)
        "ch_embs_cache": {},
        # Cache-uri analiză
        "ch_style_cache": {},
        "ch_lang_cache": ch_lang_cache,
        
        # Entități și istoric
        "global_entities": global_entities,
        "posts_history": posts_history[-500:],
        "nlp_msg_cache": {str(k): v for k, v in nlp_msg_cache.items()},  # hash e int
        
        # Stare sistem
        "dirty_channels": list(dirty_channels),
        "stats": {
            "total_channels": len(channels_set),
            "total_edges": len(edges_data),
            "total_messages": sum(len(v) for v in ch_msgs_cache.values()),
            "total_embeddings": sum(1 for v in ch_embs_cache.values() if v)
        }
    }
    
    # Procesează embedding-urile (converteste numpy array în listă)
    for ch, data in ch_embs_cache.items():
        if data and "matrix" in data and data["matrix"] is not None:
            backup["ch_embs_cache"][ch] = {
                "orig_texts": data["orig_texts"],
                "matrix": data["matrix"].tolist()
            }
    
    # Procesează stilometria (converteste numpy array în listă)
    for k, v in ch_style_cache.items():
        if isinstance(v, np.ndarray):
            backup["ch_style_cache"][k] = v.tolist()
        else:
            backup["ch_style_cache"][k] = v
    
    # Salvează în format pickle (mai eficient pentru obiecte complexe)
    with open(filename, 'wb') as f:
        pickle.dump(backup, f)
    
    # Și o copie JSON pentru citire ușoară (fără embeddings)
    json_filename = f"backups/backup_{timestamp}.json"
    json_backup = {k: v for k, v in backup.items() if k != "ch_embs_cache"}
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_backup, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Backup complet salvat: {filename}")
    logger.info(f"✅ Backup JSON salvat: {json_filename}")
    logger.info(f"📊 Statistici: {backup['stats']['total_channels']} canale, {backup['stats']['total_edges']} muchii, {backup['stats']['total_messages']} mesaje")
    
    return filename


def restore_full_backup(filename):
    """
    Restaurează un backup complet creat cu full_backup()
    """
    global channels_set, nodes_data, edges_data, posts_history
    global target_channel, keywords_list, similarity_mode, analysis_mode  # FIX #2: + analysis_mode
    global ch_msgs_cache, ch_msgs_set, ch_embs_cache, ch_style_cache
    global ch_lang_cache, dirty_channels, running, paused
    global nlp_msg_cache  # FIX #1: era prezent dar nlp_msg_cache nu era restaurat corect
    
    try:
        with open(filename, 'rb') as f:
            backup = pickle.load(f)
        
        version = backup.get("version", "1.0")
        logger.info(f"Restaurare backup versiunea {version} din {backup.get('timestamp')}")
        
        channels_set    = set(backup["channels"])
        target_channel  = backup.get("target")
        keywords_list   = backup.get("keywords", [])
        similarity_mode = backup.get("mode", "hibrid")
        # FIX #2: analysis_mode nu era restaurat deloc
        analysis_mode   = backup.get("analysis_mode", "similar")
        running         = backup.get("running", False)
        paused          = backup.get("paused", False)
        
        nodes_data = backup.get("nodes", {})
        
        edges_data.clear()
        for k_str, v in backup.get("edges", {}).items():
            c1, c2 = k_str.split("|")
            edges_data[(c1, c2)] = v
        
        ch_msgs_cache = backup.get("ch_msgs_cache", {})
        
        ch_msgs_set.clear()
        for k, v in backup.get("ch_msgs_set", {}).items():
            ch_msgs_set[k] = set(v)
        
        ch_embs_cache.clear()
        for ch, data in backup.get("ch_embs_cache", {}).items():
            if data and "matrix" in data and data["matrix"]:
                ch_embs_cache[ch] = {
                    "orig_texts": data["orig_texts"],
                    "matrix": np.array(data["matrix"])
                }
        
        ch_style_cache.clear()
        for k, v in backup.get("ch_style_cache", {}).items():
            if isinstance(v, list):
                ch_style_cache[k] = np.array(v)
            else:
                ch_style_cache[k] = v
        
        ch_lang_cache   = backup.get("ch_lang_cache", {})
        global_entities = backup.get("global_entities", {"PER": {}, "ORG": {}})
        posts_history   = backup.get("posts_history", [])
        
        # FIX #1: nlp_msg_cache.clear() + restaurare corectă cu except specific
        nlp_msg_cache.clear()
        for k_str, v in backup.get("nlp_msg_cache", {}).items():
            try:
                nlp_msg_cache[int(k_str)] = v
            except (ValueError, TypeError):
                pass
        
        dirty_channels = set(backup.get("dirty_channels", []))
        
        logger.info(f"✅ Backup restaurat cu succes: {len(channels_set)} canale, {len(edges_data)} muchii")
        return True
        
    except Exception as e:
        logger.error(f"❌ Eroare la restaurarea backup-ului: {e}", exc_info=True)
        return False


# ─────────────────────────────────────────────
# Funcții pentru salvarea/încărcarea proiectului
# ─────────────────────────────────────────────
def save_project_state():
    """
    Salvează starea completă a proiectului pentru monitorizare continuă.
    FIX #5: adăugate running, paused, analysis_mode, nlp_msg_cache, ch_embs_cache mereu inițializat.
    """
    project_data = {
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "channels": list(channels_set),
        "target_channel": target_channel,
        "keywords": keywords_list,
        "similarity_mode": similarity_mode,
        "analysis_mode": analysis_mode,      # FIX #5: lipsea
        "running": running,                  # FIX #5: lipsea
        "paused": paused,                    # FIX #5: lipsea
        "nodes": nodes_data,
        "edges": {f"{k[0]}|{k[1]}": v for k, v in edges_data.items()},
        "ch_msgs_cache": ch_msgs_cache,
        "ch_msgs_set": {k: list(v) for k, v in ch_msgs_set.items()},
        "ch_style_cache": {k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in ch_style_cache.items()},
        "ch_lang_cache": ch_lang_cache,
        "global_entities": global_entities,
        "posts_history": posts_history[-100:],
        "nlp_msg_cache": {str(k): v for k, v in nlp_msg_cache.items()},  # FIX #5: lipsea
        "dirty_channels": list(dirty_channels),
        "last_save": datetime.now().timestamp(),
        "ch_embs_cache": {},  # FIX #6: inițializat mereu ca {}, nu condiționat cu if
    }
    
    for ch, emb_data in ch_embs_cache.items():
        if emb_data and "matrix" in emb_data and emb_data["matrix"] is not None:
            project_data["ch_embs_cache"][ch] = {
                "orig_texts": emb_data["orig_texts"],
                "matrix": emb_data["matrix"].tolist()
            }
    
    return project_data


def load_project_state(project_data):
    """
    Încarcă o stare salvată anterior a proiectului.
    FIX #3: adăugate running, paused, analysis_mode, nlp_msg_cache — toate lipseau.
    """
    global channels_set, nodes_data, edges_data, posts_history
    global target_channel, keywords_list, similarity_mode, analysis_mode  # FIX #3: + analysis_mode
    global ch_msgs_cache, ch_msgs_set, ch_embs_cache, ch_style_cache, ch_lang_cache
    global dirty_channels, global_entities
    global running, paused, nlp_msg_cache  # FIX #3: toate lipseau
    
    try:
        channels_set    = set(project_data.get("channels", []))
        target_channel  = project_data.get("target_channel")
        keywords_list   = project_data.get("keywords", [])
        similarity_mode = project_data.get("similarity_mode", "hibrid")
        # FIX #3: analysis_mode nu era restaurat
        analysis_mode   = project_data.get("analysis_mode", "similar")
        # FIX #3: running și paused nu erau restaurate
        running         = project_data.get("running", False)
        paused          = project_data.get("paused", False)
        
        nodes_data = project_data.get("nodes", {})
        
        edges_data.clear()
        for k_str, v in project_data.get("edges", {}).items():
            c1, c2 = k_str.split("|")
            edges_data[(c1, c2)] = v
        
        ch_msgs_cache = project_data.get("ch_msgs_cache", {})
        
        ch_msgs_set.clear()
        for k, v in project_data.get("ch_msgs_set", {}).items():
            ch_msgs_set[k] = set(v)
        
        ch_style_cache.clear()
        for k, v in project_data.get("ch_style_cache", {}).items():
            if isinstance(v, list):
                ch_style_cache[k] = np.array(v)
            else:
                ch_style_cache[k] = v
        
        ch_lang_cache   = project_data.get("ch_lang_cache", {})
        global_entities = project_data.get("global_entities", {"PER": {}, "ORG": {}})
        posts_history   = project_data.get("posts_history", [])
        
        ch_embs_cache.clear()
        for ch, emb_data in project_data.get("ch_embs_cache", {}).items():
            if emb_data and "matrix" in emb_data and emb_data["matrix"]:
                ch_embs_cache[ch] = {
                    "orig_texts": emb_data["orig_texts"],
                    "matrix": np.array(emb_data["matrix"])
                }
        
        # FIX #3: nlp_msg_cache nu era restaurat deloc
        nlp_msg_cache.clear()
        for k_str, v in project_data.get("nlp_msg_cache", {}).items():
            try:
                nlp_msg_cache[int(k_str)] = v
            except (ValueError, TypeError):
                pass
        
        dirty_channels = set(channels_set)
        
        logger.info(f"Proiect încărcat cu succes: {len(channels_set)} canale, {len(edges_data)} conexiuni")
        return True
        
    except Exception as e:
        logger.error(f"Eroare la încărcarea proiectului: {e}")
        return False


def hash_project_state(state):
    """Generează un hash unic pentru starea proiectului"""
    state_str = json.dumps(state, sort_keys=True, default=str)
    return hashlib.md5(state_str.encode()).hexdigest()[:8]


def build_graph():
    """Construiește graful NetworkX pentru trimitere în UI"""
    G = nx.Graph()
    for (u, v), w in edges_data.items():
        G.add_edge(u, v, weight=w)
    for c in channels_set:
        if c not in G:
            G.add_node(c)
    return G


async def send_graph_update(websocket, G):
    """Trimite actualizarea grafului către client"""
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
    
    return {"nodes": f_nodes, "edges": f_edges}

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
        n_char   = len(t) + 1
        words_lo = [w.lower().strip(".,!?;:\"'()[]") for w in t.split() if w.strip()]
        sents    = [s.strip() for s in re.split(r'[.!?]+', t) if s.strip()]
        n_sents  = len(sents) + 1
        sent_lens = [len(s.split()) for s in sents] if sents else [0]
        freq     = Counter(words_lo)
        n_unique = len(freq) + 1
        n_total  = len(words_lo) + 1
        letters  = re.findall(r'[a-zA-Z\u00C0-\u024F\u0400-\u04FF]', t)
        upper    = re.findall(r'[A-Z\u00C0-\u00DE\u0400-\u042F]', t)
        return np.array([
            min(len(t) / 500.0, 1.0),
            len(re.findall(r'[.,!?;:]', t)) / n_char,
            len(re.findall(r'[^\w\s]', t)) / n_char,
            min(sum(len(w) for w in words_lo) / ((len(words_lo) + 1) * 15.0), 1.0),
            min(len(re.findall(r'\.\.\.', t)) / n_char * 100, 1.0),
            min(len(re.findall(r'!!', t)) / n_char * 100, 1.0),
            len(upper) / (len(letters) + 1),
            n_unique / n_total,
            sum(1 for w, c in freq.items() if c == 1) / n_unique,
            min(float(np.mean(sent_lens)) / 30.0, 1.0),
            sum(1 for w in words_lo if w in _STOPWORDS_STYLE) / n_total,
            len(re.findall(r'\d', t)) / n_char,
            float(np.log(n_unique) / (np.log(n_total) + 1e-9)),
            sum(1 for l in sent_lens if l < 5) / n_sents,
            sum(1 for l in sent_lens if l > 20) / n_sents,
            min(t.count("?") / 3.0, 1.0),
            float(len(t)),
            1.0 if re.search(r'https?://', t) else 0.0,
        ], dtype=float)

    all_vecs = [_msg_vec(t) for t in texts]
    medians  = np.median(np.stack(all_vecs), axis=0)
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
    # FIX #4: înlocuirile manuale cu string erau fragile și dependente de ordine
    # (ex: "podpischikov" conține "podpischika" — ordinea conta).
    # Acum folosim regex pentru a păstra doar cifre, punct/virgulă și sufixele k/m.
    if not text:
        return 0
    t = text.lower().strip()
    t = re.sub(r'[^\d.,km]', '', t)
    t = t.replace(',', '')
    if t.endswith("k"):
        try:
            return int(float(t[:-1]) * 1_000)
        except (ValueError, TypeError):
            pass
    if t.endswith("m"):
        try:
            return int(float(t[:-1]) * 1_000_000)
        except (ValueError, TypeError):
            pass
    try:
        return int(float(t))
    except (ValueError, TypeError):
        return 0


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
    if not new_texts or similarity_model is None:
        return

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
    log_factor = 0.1 * (np.log10(strength + 1) / np.log10(11))
    return strength * (DECAY_BASE + log_factor)

# ─────────────────────────────────────────────
# NOI: Funcții pentru modurile de 
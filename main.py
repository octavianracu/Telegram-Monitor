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
            except:
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
            
            # Embedding-uri
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
similarity_mode = "direct"      # "direct" | "hibrid" | "tranzitiv"
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
                "matrix": data["matrix"].tolist()  # numpy -> list
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
    global target_channel, keywords_list, similarity_mode, global_entities
    global ch_msgs_cache, ch_msgs_set, ch_embs_cache, ch_style_cache
    global ch_lang_cache, dirty_channels, running, paused
    global nlp_msg_cache
    
    try:
        with open(filename, 'rb') as f:
            backup = pickle.load(f)
        
        # Verifică versiunea
        version = backup.get("version", "1.0")
        logger.info(f"Restaurare backup versiunea {version} din {backup.get('timestamp')}")
        
        # Restaurează datele de bază
        channels_set = set(backup["channels"])
        target_channel = backup.get("target")
        keywords_list = backup.get("keywords", [])
        similarity_mode = backup.get("mode", "direct")
        running = backup.get("running", False)
        paused = backup.get("paused", False)
        
        # Restaurează date rețea
        nodes_data = backup.get("nodes", {})
        
        # Restaurează edges (convertește cheile înapoi în tuple)
        edges_data.clear()
        for k_str, v in backup.get("edges", {}).items():
            c1, c2 = k_str.split("|")
            edges_data[(c1, c2)] = v
        
        # Restaurează cache-uri text
        ch_msgs_cache = backup.get("ch_msgs_cache", {})
        
        # Restaurează set-uri (convertește listă în set)
        ch_msgs_set.clear()
        for k, v in backup.get("ch_msgs_set", {}).items():
            ch_msgs_set[k] = set(v)
        
        # Restaurează embedding-uri (reconstruiește numpy arrays)
        ch_embs_cache.clear()
        for ch, data in backup.get("ch_embs_cache", {}).items():
            if data and "matrix" in data and data["matrix"]:
                ch_embs_cache[ch] = {
                    "orig_texts": data["orig_texts"],
                    "matrix": np.array(data["matrix"])
                }
        
        # Restaurează stilometrie (convertește listă înapoi în numpy array)
        ch_style_cache.clear()
        for k, v in backup.get("ch_style_cache", {}).items():
            if isinstance(v, list):
                ch_style_cache[k] = np.array(v)
            else:
                ch_style_cache[k] = v
        
        ch_lang_cache = backup.get("ch_lang_cache", {})
        
        # Restaurează entități
        global_entities = backup.get("global_entities", {"PER": {}, "ORG": {}})
        
        # Restaurează istoric
        posts_history = backup.get("posts_history", [])
        
        # Restaurează cache NLP (convertește cheile înapoi în int)
        nlp_msg_cache.clear()
        for k_str, v in backup.get("nlp_msg_cache", {}).items():
            try:
                nlp_msg_cache[int(k_str)] = v
            except:
                pass
        
        # Restaurează dirty channels
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
    Include toate cache-urile, datele și configurațiile.
    """
    project_data = {
        # Configurații de bază
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        
        # Canale și setări
        "channels": list(channels_set),
        "target_channel": target_channel,
        "keywords": keywords_list,
        "similarity_mode": similarity_mode,
        
        # Date rețea
        "nodes": nodes_data,
        "edges": {f"{k[0]}|{k[1]}": v for k, v in edges_data.items()},
        
        # Cache-uri text
        "ch_msgs_cache": ch_msgs_cache,
        "ch_msgs_set": {k: list(v) for k, v in ch_msgs_set.items()},
        
        # Cache-uri analiză
        "ch_style_cache": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in ch_style_cache.items()},
        "ch_lang_cache": ch_lang_cache,
        
        # Entități și istoric
        "global_entities": global_entities,
        "posts_history": posts_history[-100:],  # Ultimele 100 de potriviri
        
        # Timestamps pentru decay
        "last_save": datetime.now().timestamp()
    }
    
    # Adaugă embedding-urile (compresie optională)
    if ch_embs_cache:
        project_data["ch_embs_cache"] = {}
        for ch, emb_data in ch_embs_cache.items():
            if emb_data and "matrix" in emb_data and emb_data["matrix"] is not None:
                # Convertim numpy array în listă pentru serializare
                project_data["ch_embs_cache"][ch] = {
                    "orig_texts": emb_data["orig_texts"],
                    "matrix": emb_data["matrix"].tolist() if emb_data["matrix"] is not None else None
                }
    
    return project_data


def load_project_state(project_data):
    """
    Încarcă o stare salvată anterior a proiectului.
    """
    global channels_set, nodes_data, edges_data, posts_history
    global target_channel, keywords_list, similarity_mode, global_entities
    global ch_msgs_cache, ch_msgs_set, ch_embs_cache, ch_style_cache, ch_lang_cache
    global dirty_channels
    
    try:
        # Verifică versiunea
        version = project_data.get("version", "1.0.0")
        
        # Restaurează configurații de bază
        channels_set = set(project_data.get("channels", []))
        target_channel = project_data.get("target_channel")
        keywords_list = project_data.get("keywords", [])
        similarity_mode = project_data.get("similarity_mode", "direct")
        
        # Restaurează date rețea
        nodes_data = project_data.get("nodes", {})
        
        # Restaurează edges (convertește cheile înapoi în tuple)
        edges_data.clear()
        for k_str, v in project_data.get("edges", {}).items():
            c1, c2 = k_str.split("|")
            edges_data[(c1, c2)] = v
        
        # Restaurează cache-uri text
        ch_msgs_cache = project_data.get("ch_msgs_cache", {})
        
        # Restaurează set-uri (convertește listă în set)
        ch_msgs_set.clear()
        for k, v in project_data.get("ch_msgs_set", {}).items():
            ch_msgs_set[k] = set(v)
        
        # Restaurează stilometrie (convertește listă înapoi în numpy array)
        ch_style_cache.clear()
        for k, v in project_data.get("ch_style_cache", {}).items():
            if isinstance(v, list):
                ch_style_cache[k] = np.array(v)
            else:
                ch_style_cache[k] = v
        
        ch_lang_cache = project_data.get("ch_lang_cache", {})
        
        # Restaurează entități
        global_entities = project_data.get("global_entities", {"PER": {}, "ORG": {}})
        
        # Restaurează istoric
        posts_history = project_data.get("posts_history", [])
        
        # Restaurează embedding-uri
        ch_embs_cache.clear()
        for ch, emb_data in project_data.get("ch_embs_cache", {}).items():
            if emb_data and "matrix" in emb_data and emb_data["matrix"]:
                ch_embs_cache[ch] = {
                    "orig_texts": emb_data["orig_texts"],
                    "matrix": np.array(emb_data["matrix"])
                }
        
        # Marchează toate canalele ca dirty pentru re-analiză
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
# NOI: Funcții pentru modurile de inferență
# ─────────────────────────────────────────────
async def perform_hybrid_inference(max_inferences: int):
    """Mod hibrid: completează cu inferențe după analiza directă"""
    global edges_data, posts_history
    
    if len(edges_data) < 20:
        return
        
    try:
        G = nx.Graph()
        for (u, v), w in edges_data.items():
            if w > 0.5:  # doar relații semnificative
                G.add_edge(u, v, weight=w)
        
        if G.number_of_edges() < 10:
            return
            
        inferences = []
        for b in list(G.nodes())[:50]:  # limităm pentru performanță
            neighbors = list(G.neighbors(b))
            for i, a in enumerate(neighbors):
                for c in neighbors[i+1:]:
                    if not G.has_edge(a, c):
                        w_ab = G[a][b]['weight']
                        w_bc = G[b][c]['weight']
                        inferred = (w_ab * w_bc) ** 0.5
                        if inferred > INFERENCE_THRESHOLD["hibrid"]:
                            inferences.append((a, c, inferred, b))
        
        # Adaugă cele mai bune inferențe
        inferences.sort(key=lambda x: -x[2])
        added = 0
        for a, c, w, via in inferences[:max_inferences]:
            pair = tuple(sorted([a, c]))
            if pair not in edges_data:
                edges_data[pair] = min(w * 1.3, 10.0)
                posts_history.append({
                    "type": "hibrid",
                    "ch1": a,
                    "ch2": c,
                    "via": via,
                    "confidence": round(w, 3),
                    "time": datetime.now().strftime("%H:%M:%S")
                })
                added += 1
                logger.info(f"[Hibrid] Inferență: {a} ~ {c} (via {via}, {w:.2f})")
        
        if added > 0:
            logger.info(f"[Hibrid] Adăugate {added} muchii noi")
            
    except Exception as e:
        logger.error(f"[Hibrid] Eroare: {e}")


async def perform_transitive_inference(max_inferences: int):
    """Mod tranzitiv: prioritizăm descoperirea de conexiuni prin lanțuri"""
    global edges_data, posts_history
    
    if len(edges_data) < 10:
        return
        
    try:
        G = nx.Graph()
        for (u, v), w in edges_data.items():
            if w > 0.4:  # prag mai mic în modul tranzitiv
                G.add_edge(u, v, weight=w)
        
        if G.number_of_edges() < 5:
            return
            
        inferences = []
        
        # Lanțuri de lungime 2 (A-B-C)
        for b in list(G.nodes())[:30]:
            neighbors = list(G.neighbors(b))
            for i, a in enumerate(neighbors):
                for c in neighbors[i+1:]:
                    if not G.has_edge(a, c):
                        w_ab = G[a][b]['weight']
                        w_bc = G[b][c]['weight']
                        inferred = (w_ab * w_bc) ** 0.5 * 1.2  # bonus în modul tranzitiv
                        if inferred > INFERENCE_THRESHOLD["tranzitiv"]:
                            inferences.append((a, c, inferred, f"{b}", 2))
        
        # Lanțuri de lungime 3 (A-B-C-D) - doar dacă avem nevoie de mai multe
        if len(inferences) < max_inferences * 0.7:
            nodes_list = list(G.nodes())[:15]  # limitat pentru performanță
            for a in nodes_list:
                for b in G.neighbors(a):
                    for c in G.neighbors(b):
                        if c != a and not G.has_edge(a, c):
                            for d in G.neighbors(c):
                                if d != b and d != a and not G.has_edge(a, d):
                                    w_ab = G[a][b]['weight']
                                    w_bc = G[b][c]['weight']
                                    w_cd = G[c][d]['weight']
                                    inferred = (w_ab * w_bc * w_cd) ** (1/3) * 1.5
                                    if inferred > INFERENCE_THRESHOLD["tranzitiv"] * 0.9:
                                        inferences.append((a, d, inferred, f"{b}→{c}", 3))
        
        # Adaugă inferențele
        inferences.sort(key=lambda x: -x[2])
        added = 0
        for a, c, w, via, length in inferences[:max_inferences]:
            pair = tuple(sorted([a, c]))
            if pair not in edges_data:
                edges_data[pair] = min(w * 1.5, 10.0)
                posts_history.append({
                    "type": "tranzitiv",
                    "ch1": a,
                    "ch2": c,
                    "via": via,
                    "length": length,
                    "confidence": round(w, 3),
                    "time": datetime.now().strftime("%H:%M:%S")
                })
                added += 1
                logger.info(f"[Tranzitiv-{length}] {a} ~ {c} (via {via}, {w:.2f})")
        
        if added > 0:
            logger.info(f"[Tranzitiv] Adăugate {added} muchii noi")
            
    except Exception as e:
        logger.error(f"[Tranzitiv] Eroare: {e}")

# ─────────────────────────────────────────────
# Analiza globala a unei perechi de canale
# ─────────────────────────────────────────────
def _analyse_pair_global(c1: str, c2: str, mode: str, threshold: float):
    MAX_HITS = 10
    score = 0.0
    match = None

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

        conv_w = 0.0
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
            "type": "repost", "ch1": c1, "ch2": c2,
            "score": float(sims[bi, bj]),
            "ch1_msg": orig1[bi] if bi < len(orig1) else "",
            "ch2_msg": orig2[bj] if bj < len(orig2) else "",
            "n_hits": n_hits,
        }
        return score, match

    above = sims > threshold
    if not above.any():
        return 0.0, None

    hit_rows, hit_cols = np.where(above)
    hit_scores = sims[hit_rows, hit_cols]
    top_idx  = np.argsort(hit_scores)[::-1][:MAX_HITS]
    hit_pairs = list(zip(hit_rows[top_idx], hit_cols[top_idx]))

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
            e1 = {e['name'] for e in nlp1.get("entities", []) if len(e['name']) > 3}
            e2 = {e['name'] for e in nlp2.get("entities", []) if len(e['name']) > 3}
            if e1 & e2:
                nlp_bonus += 0.2
            update_entity_stats(nlp1)
            update_entity_stats(nlp2)

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
# Connection Manager - VERSIUNE ÎMBUNĂTĂȚITĂ
# ─────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active_connections = []
        self.ping_interval = 20  # secunde

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
        # Pornește task-ul de ping
        asyncio.create_task(self.keep_alive(ws))
        await self.send_state()

    async def keep_alive(self, ws: WebSocket):
        """Trimite ping periodic pentru a menține conexiunea"""
        try:
            while ws in self.active_connections:
                await asyncio.sleep(self.ping_interval)
                try:
                    await ws.send_text(json.dumps({"type": "ping"}))
                except:
                    break
        except:
            pass

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
        
        # Elimină conexiunile moarte
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
            "similarity_mode": similarity_mode,  # "direct", "hibrid", "tranzitiv"
            "global_entities": global_entities,
        }
        await self.broadcast(json.dumps(state))


manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    logger.info("Pornire aplicatie — initiere incarcare modele NLP.")
    loop = asyncio.get_running_loop()
    start_nlp_loading(loop)

# ─────────────────────────────────────────────
# WebSocket endpoint - VERSIUNE ÎMBUNĂTĂȚITĂ
# ─────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global running, paused, channels_set, nodes_data, edges_data, posts_history
    global target_channel, keywords_list, similarity_mode, global_entities
    global ch_msgs_cache, ch_embs_cache, ch_style_cache, background_tasks, dirty_channels

    await manager.connect(websocket)
    try:
        # Setează timeout mai mare pentru ping
        await websocket.receive_text()
        
        while True:
            try:
                # Setează timeout pentru fiecare mesaj
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0  # 30 secunde timeout
                )
                
                cmd = json.loads(data)
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

                # MODIFICARE: Acțiune pentru setarea modului de similaritate
                elif action == "set_mode":
                    new_mode = cmd.get("mode", "direct")
                    if new_mode in ["direct", "hibrid", "tranzitiv"]:
                        similarity_mode = new_mode
                        edges_data.clear()  # resetăm muchiile la schimbarea modului
                        dirty_channels.update(channels_set)
                        logger.info(f"✅ Mod similaritate schimbat: {new_mode}")
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

                elif action == "save_request":
                    payload = {
                        "channels": list(channels_set),
                        "edges": [
                            {"from": k[0], "to": k[1], "strength": v}
                            for k, v in edges_data.items()
                        ],
                        "entities": global_entities,
                        "posts_history": posts_history[-50:],
                    }
                    await manager.broadcast(json.dumps({"type": "save_file", "data": payload}))

                # ── Acțiuni pentru backup complet (METODA 4) ────────────
                elif action == "full_backup":
                    try:
                        filename = full_backup()
                        await websocket.send_text(json.dumps({
                            "type": "backup_complete",
                            "success": True,
                            "filename": filename,
                            "message": f"Backup complet salvat: {filename}"
                        }))
                    except Exception as e:
                        await websocket.send_text(json.dumps({
                            "type": "backup_complete",
                            "success": False,
                            "message": f"Eroare la backup: {str(e)}"
                        }))

                elif action == "restore_backup":
                    filename = cmd.get("filename")
                    if filename and os.path.exists(filename):
                        success = restore_full_backup(filename)
                        if success:
                            await manager.send_state()
                            G = build_graph()
                            graph_data = await send_graph_update(websocket, G)
                            await websocket.send_text(json.dumps({
                                "type": "graph_update",
                                "nodes": graph_data["nodes"],
                                "edges": graph_data["edges"]
                            }))
                            await websocket.send_text(json.dumps({
                                "type": "restore_complete",
                                "success": True,
                                "message": f"Backup restaurat: {filename}"
                            }))
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "restore_complete",
                                "success": False,
                                "message": "Eroare la restaurarea backup-ului"
                            }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "restore_complete",
                            "success": False,
                            "message": "Fișierul nu există"
                        }))

                elif action == "list_backups":
                    backups = []
                    if os.path.exists("backups"):
                        for f in os.listdir("backups"):
                            if f.endswith('.pkl') or f.endswith('.json'):
                                filepath = os.path.join("backups", f)
                                stat = os.stat(filepath)
                                backups.append({
                                    "filename": f,
                                    "size": stat.st_size,
                                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                                })
                    await websocket.send_text(json.dumps({
                        "type": "backups_list",
                        "backups": sorted(backups, key=lambda x: x["modified"], reverse=True)
                    }))

                # ── Acțiuni existente pentru salvare/încărcare proiect ────────────
                elif action == "save_project":
                    project_state = save_project_state()
                    project_hash = hash_project_state(project_state)
                    response = {
                        "type": "project_saved",
                        "data": project_state,
                        "hash": project_hash,
                        "filename": f"telegram_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tgm"
                    }
                    await websocket.send_text(json.dumps(response))

                elif action == "load_project":
                    project_data = cmd.get("data")
                    if project_data and load_project_state(project_data):
                        await manager.send_state()
                        G = build_graph()
                        graph_data = await send_graph_update(websocket, G)
                        await websocket.send_text(json.dumps({
                            "type": "graph_update",
                            "nodes": graph_data["nodes"],
                            "edges": graph_data["edges"]
                        }))
                        await websocket.send_text(json.dumps({
                            "type": "project_loaded",
                            "success": True,
                            "message": f"Proiect încărcat cu {len(channels_set)} canale"
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "project_loaded",
                            "success": False,
                            "message": "Eroare la încărcarea proiectului"
                        }))

                elif action == "export_project":
                    project_state = save_project_state()
                    await websocket.send_text(json.dumps({
                        "type": "project_export",
                        "data": project_state,
                        "filename": f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tgm"
                    }))

                elif action == "import_project":
                    project_data = cmd.get("data")
                    if project_data and load_project_state(project_data):
                        await manager.send_state()
                        G = build_graph()
                        graph_data = await send_graph_update(websocket, G)
                        await websocket.send_text(json.dumps({
                            "type": "graph_update",
                            "nodes": graph_data["nodes"],
                            "edges": graph_data["edges"]
                        }))
                        await websocket.send_text(json.dumps({
                            "type": "project_imported",
                            "success": True
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "project_imported",
                            "success": False
                        }))

            except asyncio.TimeoutError:
                # Trimite ping pentru a menține conexiunea vie
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                    continue
                except:
                    logger.warning("WebSocket ping failed, closing connection")
                    break
                    
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket unexpected error: {e}")
    finally:
        manager.disconnect(websocket)
        logger.info("WebSocket connection closed")

# ─────────────────────────────────────────────
# Background scraper - CU TIMEOUT MĂRIT
# ─────────────────────────────────────────────
async def background_scraper():
    global running, paused, channels_set, nodes_data
    global ch_msgs_cache, ch_embs_cache, ch_style_cache, dirty_channels

    semaphore = asyncio.Semaphore(5)
    BATCH_SIZE = 20

    async def process_single_channel(ch: str):
        async with semaphore:
            if not running or paused:
                return
            try:
                await manager.broadcast(json.dumps({"type": "status", "msg": f"Scanez {ch}..."}))
                data = await asyncio.wait_for(
                    asyncio.to_thread(scrape_channel, ch),
                    timeout=20.0  # mărit de la 12 la 20 secunde
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

            ch_lang_cache[ch]  = detect_language(" ".join(msgs[:10]))
            ch_style_cache[ch] = await asyncio.to_thread(get_stylometric_fingerprint, msgs)

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
                combined = (ch_msgs_cache.get(ch, []) + new_msgs)[-50:]
                ch_msgs_cache[ch] = combined
                ch_msgs_set[ch] = set(combined)

                await asyncio.to_thread(update_embeddings_incremental, ch, new_msgs)

                for m in new_msgs[:5]:
                    await asyncio.to_thread(analyse_text, m)

                dirty_channels.add(ch)
                logger.info(f"[Scraper] {ch}: {len(new_msgs)} mesaje noi → dirty")

            await manager.broadcast(json.dumps({"type": "node_update", "node": nodes_data[ch]}))

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
# Background analyzer - MODIFICAT pentru noile moduri
# ─────────────────────────────────────────────
async def background_analyzer():
    global running, paused, edges_data, posts_history, dirty_channels
    global channels_set, nodes_data, target_channel, similarity_mode
    global global_entities, nlp_ready, similarity_model, cosine_similarity

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
                "msg":  f"Analiza globala: {n_ch} canale, {n_dirty} cu continut nou, mod: {similarity_mode}",
            }))
            logger.info(f"[Analyzer] mod_similaritate={similarity_mode} canale={n_ch} dirty={n_dirty}")

            # Decay pentru toate muchiile existente
            for pair in list(edges_data.keys()):
                edges_data[pair] = _decay_edge(edges_data[pair])

            all_keys = [k for k in ch_embs_cache.keys() 
                        if ch_embs_cache[k] is not None 
                        and "matrix" in ch_embs_cache[k] 
                        and ch_embs_cache[k]["matrix"].shape[0] > 0]
            threshold = THRESHOLD.get(analysis_mode, 0.6)  # folosim analysis_mode pentru prag
            current_dirty = set(dirty_channels)

            force_recheck = len(current_dirty) == 0 and len(edges_data) > 0

            # Selectarea perechilor pentru analiză directă
            pairs_to_check = []
            if len(all_keys) >= 2:
                for i, c1 in enumerate(all_keys):
                    for c2 in all_keys[i + 1:]:
                        pair        = tuple(sorted([c1, c2]))
                        is_new_pair = pair not in edges_data
                        is_affected = bool(current_dirty & {c1, c2})
                        is_weak     = edges_data.get(pair, 0.0) < 1.0

                        if is_affected:
                            priority = 0
                        elif is_new_pair:
                            priority = 1
                        elif force_recheck and is_weak:
                            priority = 2
                        else:
                            continue

                        pairs_to_check.append((priority, pair, c1, c2))

                pairs_to_check.sort(key=lambda x: x[0])
                
                # Limităm numărul de perechi în funcție de mod
                if similarity_mode == "direct":
                    pairs_to_check = pairs_to_check[:MAX_PAIRS_PER_CYCLE]
                elif similarity_mode == "hibrid":
                    # În modul hibrid, folosim 70% pentru analiză directă
                    direct_limit = int(MAX_PAIRS_PER_CYCLE * 0.7)
                    pairs_to_check = pairs_to_check[:direct_limit]
                elif similarity_mode == "tranzitiv":
                    # În modul tranzitiv, folosim doar 50% pentru analiză directă
                    direct_limit = int(MAX_PAIRS_PER_CYCLE * 0.5)
                    pairs_to_check = pairs_to_check[:direct_limit]

                # Analizăm perechile selectate
                for _, pair, c1, c2 in pairs_to_check:
                    if not running:
                        break

                    edge_score, match = await asyncio.to_thread(
                        _analyse_pair_global, c1, c2, analysis_mode, threshold
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

            # Aplicăm logica specifică modului ales
            analyzed_pairs = len(pairs_to_check)
            
            if similarity_mode == "hibrid" and analyzed_pairs < MAX_PAIRS_PER_CYCLE * 0.7:
                # În modul hibrid, completăm cu inferențe
                remaining = MAX_PAIRS_PER_CYCLE - analyzed_pairs
                if remaining > 10:
                    await perform_hybrid_inference(min(remaining, 50))
                    
            elif similarity_mode == "tranzitiv":
                # În modul tranzitiv, facem inferențe mai agresive
                remaining = MAX_PAIRS_PER_CYCLE - analyzed_pairs
                if remaining > 5:
                    await perform_transitive_inference(min(remaining, 100))

            # Eliminăm muchiile slabe
            dead = [p for p, s in edges_data.items() if s < 0.15]
            for p in dead:
                edges_data.pop(p, None)

            dirty_channels -= current_dirty

            # Construim și trimitem graful actualizat
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
    os.makedirs("backups", exist_ok=True)
    
    # Opțional: Rulează un backup automat la pornire
    # full_backup()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
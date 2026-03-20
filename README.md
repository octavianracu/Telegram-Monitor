# TGM Monitor

Instrument de monitorizare a coordonării narative între canale Telegram. Detectează canale care reacționează similar la aceleași evenimente, amplifică același conținut sau fac parte din rețele de distribuție indirectă — pe perioade de zile, nu de minute.

---

## Funcționalitate principală

- **Scraping automat** al mesajelor din canale publice Telegram prin `t.me/s/{username}`
- **Analiză semantică** cu embeddings multilinguale (`paraphrase-multilingual-mpnet-base-v2`) și similaritate cosinus
- **NER + Sentiment** cu modele HuggingFace pentru entități (persoane, organizații) și tonalitate
- **Trei moduri de similaritate:** Direct, Hibrid, Tranzitiv — fiecare cu logică de inferență diferită
- **Persistență SQLite** — mesajele și scorurile cumulative supraviețuiesc repornirii serverului
- **Graf interactiv** vizualizat în timp real prin WebSocket, cu detecție de comunități Louvain
- **Fereastră glisantă de 3 zile** — analiza folosește numai mesajele recente, relevante pentru valul curent de știri

---

## Arhitectură

```
tgm_monitor.db (SQLite)
├── messages          — toate mesajele scrapate, cu timestamp
└── edges_cumulative  — scoruri de coordonare acumulate pe toată durata monitorizării

RAM (volatil, se reface la repornire)
├── edges_data        — scoruri de sesiune cu decay
├── edges_type        — tipul fiecărei muchii (direct / inferred_hibrid / inferred_tranzitiv)
├── ch_embs_cache     — embeddings reconstruite din fereastra de 3 zile
└── ch_style_cache    — amprente stilometrice per canal
```

Graful Louvain pentru detecția comunităților se construiește **din `edges_cumulative`** (SQLite), nu din starea de sesiune. Comunitatea unui canal reflectă astfel istoricul complet al coordonării, nu doar ultimele 10 minute.

---

## Moduri de similaritate

| Mod | Ce detectează | Inferențe | Buget perechi/ciclu |
|-----|--------------|-----------|---------------------|
| **Direct** | Același subiect în fereastra de 3 zile, confirmat semantic | Niciuna | 300 (100%) |
| **Hibrid** | Coordonare directă + relații inférate printr-un hub comun | ≤ 30 hibrid/ciclu | 210 (70%) |
| **Tranzitiv** | Rețele de amplificare indirectă A→B→C, chiar dacă A și C nu se aseamănă direct | ≤ 50 tranzitiv/ciclu | 150 (50%) |

### Moduri de analiză

- **similar** — similaritate semantică generală (prag 0.72)
- **repost** — conținut aproape identic / repost (prag 0.88)
- **stylography** — amprentă stilometrică pe 18 dimensiuni (prag 0.72)

---

## Instalare

### Cerințe

- Python 3.10+
- ~4 GB RAM pentru modelele NLP la încărcare completă
- Conexiune internet pentru scraping Telegram

### Dependențe

```bash
pip install fastapi uvicorn[standard] requests beautifulsoup4 \
            numpy networkx sentence-transformers transformers \
            scikit-learn torch
```

### Pornire

```bash
python main.py
```

Serverul pornește pe `http://0.0.0.0:8000`. La primul pornire se creează automat `tgm_monitor.db` în directorul curent.

---

## Utilizare

### Interfața web

Accesează `http://localhost:8000` după pornire. Fișierele statice (HTML/JS/CSS) trebuie plasate în directorul `static/`.

### API REST

| Endpoint | Metodă | Descriere |
|----------|--------|-----------|
| `/api/backup_now` | GET | Salvează backup JSON în `backups/` |
| `/api/project/list` | GET | Listează proiectele `.tgm` salvate |
| `/api/project/save` | POST | Salvează starea curentă ca proiect `.tgm` |
| `/api/project/load` | POST | Încarcă un proiect `.tgm` existent |
| `/api/project/delete` | DELETE | Șterge un proiect `.tgm` |

### WebSocket

Conexiunea principală la `ws://localhost:8000/ws` acceptă comenzi JSON:

```json
{ "action": "add_channel",    "channel": "@exemplu" }
{ "action": "remove_channel", "channel": "@exemplu" }
{ "action": "set_target",     "channel": "@exemplu" }
{ "action": "set_mode",       "mode": "hibrid" }
{ "action": "set_mode",       "mode": "similar" }
{ "action": "start" }
{ "action": "stop" }
{ "action": "pause" }
{ "action": "reset" }
{ "action": "full_backup" }
{ "action": "save_request" }
```

Serverul emite events:

```json
{ "type": "state_update",    ... }
{ "type": "graph_update",    "nodes": [...], "edges": [...] }
{ "type": "new_post_match",  "data": { "ch1": "...", "ch2": "...", "score": 0.84 } }
{ "type": "node_update",     "node": { "id": "...", "subscribers": 12000 } }
{ "type": "status",          "msg": "Analiză: 203 canale, 42 dirty..." }
```

---

## Parametri cheie

```python
ANALYSIS_WINDOW_DAYS     = 3      # fereastra de mesaje analizate
MAX_PAIRS_PER_CYCLE      = 300    # perechi analizate per ciclu de 10s

DECAY_BASE               = 0.95   # decay muchii directe (~45 cicluri durată de viață)
DECAY_INFERRED           = {
    "hibrid":    0.75,             # ~8 cicluri fără confirmare
    "tranzitiv": 0.70,             # ~6 cicluri fără confirmare
}

INFERENCE_GRAPH_MIN_WEIGHT    = 0.72   # prag intrare graf inferențe hibrid
TRANSITIVE_GRAPH_MIN_WEIGHT   = 0.65   # prag intrare graf inferențe tranzitive
INFERRED_SCORE_PENALTY        = 0.65   # penalizare scor inferit hibrid
TRANSITIVE_PENALTY_L2         = 0.50   # penalizare lanț de lungime 2
TRANSITIVE_PENALTY_L3         = 0.35   # penalizare lanț de lungime 3
```

---

## Persistență și backup

### SQLite (`tgm_monitor.db`)

Baza de date se creează automat la primul pornire și persistă între reporniri.

- `messages` — mesajele scrapate cu timestamp; curățate automat la 7 zile
- `edges_cumulative` — scorurile totale de coordonare; **nu scad niciodată**

### Backup manual

```bash
curl http://localhost:8000/api/backup_now
```

Sau prin WebSocket: `{ "action": "full_backup" }` — salvează atât `.pkl` cât și `.json` în `backups/`.

### Proiecte `.tgm`

Fișiere JSON care salvează starea completă a unei sesiuni: lista de canale, configurație, noduri, muchii, istoricul de potriviri. Portabile între mașini.

---

## Scalabilitate

| Canale | Perechi totale | Cicluri pentru acoperire completă | Timp per rotație |
|--------|---------------|----------------------------------|-----------------|
| 100 | 4.950 | ~17 | ~2.8 min |
| 203 | 20.503 | ~68 | ~11.3 min |
| 300 | 44.850 | ~150 | ~25 min |
| 500 | 124.750 | ~416 | ~69 min |

La 500+ canale, construirea listei O(N²) la fiecare ciclu și `betweenness_centrality` devin vizibile în latență. Se recomandă dezactivarea `betweenness_centrality` sau reducerea frecvenței la cicluri alternative.

---

## Modele NLP utilizate

| Model | Sarcină | Dimensiune |
|-------|---------|-----------|
| `paraphrase-multilingual-mpnet-base-v2` | Embeddings semantice | ~420 MB |
| `Babelscape/wikineural-multilingual-ner` | Recunoaștere entități (PER, ORG) | ~480 MB |
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | Clasificare sentiment | ~260 MB |

Modelele sunt descărcate automat la primul pornire din HuggingFace Hub. Încărcarea durează 2–5 minute; scraping-ul pornește imediat, analiza NLP începe după ce modelele sunt gata.

---

## Structura proiectului

```
.
├── main.py              — server principal (FastAPI + WebSocket + NLP + SQLite)
├── tgm_monitor.db       — baza de date SQLite (creată automat)
├── static/              — fișiere frontend (index.html, JS, CSS)
├── projects/            — proiecte .tgm salvate
└── backups/             — backup-uri automate și manuale
```

---

## Licență

Proiect privat. Utilizare exclusivă în scopuri de cercetare și monitorizare media.

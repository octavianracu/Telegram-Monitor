Telegram Channel Network Analyzer
https://img.shields.io/badge/python-3.8+-blue.svg
https://img.shields.io/badge/FastAPI-0.68+-green.svg
https://img.shields.io/badge/license-MIT-blue.svg
https://img.shields.io/badge/PRs-welcome-brightgreen.svg

Telegram Channel Network Analyzer este un instrument avansat de analiză și vizualizare a rețelelor de canale Telegram, care detectează automat relații de similaritate, repostări și pattern-uri stilometrice între canale. Sistemul rulează în timp real și oferă o interfață web interactivă pentru explorarea rețelei sociale descoperite.

https://via.placeholder.com/800x400/0d0d0f/00f2ff?text=Telegram+Channel+Network+Analyzer

✨ Caracteristici Principale
🎯 Analiză Multi-dimensională
Repostări - Detectează canale care repostează același conținut (prag: 0.88)

Similaritate semantică - Identifică canale cu conținut ideologic similar folosind embeddings multilingve (prag: 0.72)

Stilometrie avansată - Recunoaște același autor după 18 dimensiuni stilistice de scriere (prag: 0.72)

🧠 Procesare NLP de Ultimă Generație
Modele încărcate asincron în background thread

Cache inteligent pentru embeddings și analize NLP

Procesare incrementală - doar mesajele noi sunt encodate

Suport multilingv complet (română, rusă, engleză)

NER (Named Entity Recognition) pentru persoane și organizații

Analiză de sentiment per mesaj

📊 Vizualizare Interactivă în Timp Real
Grafice de rețea dinamice cu librăria vis.js

Comunități detectate automat (algoritm Louvain)

6 metrici de centralitate:

Grad (degree)

Betweenness (punte)

Closeness (proximitate)

Eigenvector (influență)

PageRank (difuzie)

Sistem de culori pentru comunități

Tooltip-uri cu informații detaliate

Panou de informații la click pe nod

⚡ Performanță Optimizată pentru Scenarii Reale
Scraping paralel cu semafor (max 5 canale simultan)

Procesare în batch-uri de 20 canale pentru evitarea supraîncărcării

Dirty tracking - reanalizează doar canalele modificate

Decay logaritmic al relațiilor în timp (relațiile puternice persistă)

Limitare la 300 perechi analizate per ciclu

Timeout de 12 secunde per canal pentru scraping

🎨 Interfață Modernă
Design Cyberpunk cu temă întunecată

Gradient și efecte de blur

Animații fluide pentru notificări

Butoane de navigare în graf:

Zoom In/Out

Fit to Screen

Toggle Physics (oprește/pornește animația)

Export Graph în JSON

Reset View

Filtrare canale în timp real

Upload fișiere .txt cu liste de canale

🏗️ Arhitectură
text
telegram-channel-analyzer/
├── main.py                 # Aplicația principală FastAPI
├── static/
│   └── index.html         # Interfață web (SPA)
├── requirements.txt       # Dependințe Python
├── .gitignore             # Fișiere ignorate în Git
└── README.md              # Documentație
Componente Tehnice
Componentă	Tehnologie	Rol
Backend	FastAPI + WebSocket	Server și comunicare real-time
Frontend	vis.js + vanilla JS	Vizualizare graf interactivă
Embeddings	sentence-transformers	Similaritate semantică
NLP	Hugging Face Transformers	NER și sentiment
Analiză rețea	NetworkX	Algoritmi de comunități și centralitate
Scraping	requests + BeautifulSoup4	Colectare date Telegram
Matematică	NumPy + scikit-learn	Calcul similarități
📦 Instalare
Cerințe sistem
Python 3.8 sau mai nou

4GB RAM minim (recomandat 8GB)

Conexiune internet pentru descărcarea modelelor

Pași instalare
bash
# Clonează repository-ul
git clone https://github.com/username/telegram-channel-analyzer.git
cd telegram-channel-analyzer

# Creează și activează virtual environment
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate

# Instalează dependințele
pip install -r requirements.txt

# Pornește serverul
python main.py
Serverul va rula la http://localhost:8000. Modelele NLP se vor descărca automat la prima pornire (1-2GB, poate dura câteva minute).

🎮 Utilizare
Interfață Web
Adaugă canale - Introdu nume de canale Telegram (cu sau fără @)

Selectează modul de analiză:

🔄 Repostări - pentru conținut identic

🎯 Tematici & Ideologie - pentru similaritate semantică

✍️ Stilografie - pentru același autor

Pornește analiza - click START

Explorează graful:

Click pe nod pentru detalii

Dublu-click pentru focus

Butoanele din dreapta sus pentru navigare

Selectează metrica din dropdown

Comenzi disponibile
Buton	Acțiune
START	Pornește analiza în timp real
PAUZĂ	Suspendă analiza temporar
STOP	Oprește analiza complet
RESETARE	Șterge toate datele
JSON	Exportă rețeaua în format JSON
+ ADAUGĂ	Adaugă canal manual
📁 .TXT	Încarcă listă de canale din fișier
Navigare Graf
Buton	Funcție
➕	Apropie zoom
➖	Depărtează zoom
⬛	Potrivește în ecran
⚙	Pornește/oprește fizica
⬇	Exportă graful
⟲	Resetează vederea
📊 Dimensiuni Stilometrice
Sistemul analizează 18 dimensiuni pentru detectarea autorului:

Suprafață textuală (pondere 0.5-1.5)
Lungime medie mesaj

Densitate punctuație

Densitate caractere speciale/emoji

Lungime medie cuvânt

Ticuri de scriere (pondere 2.0)
Rată ellipsis (...)

Rată exclamații (!!)

Rată întrebări (?)

Lexic și vocabular (pondere 1.0)
Type-token ratio (TTR)

Rată hapax legomena

Bogăție vocabular

Rată stopwords

Structură (pondere 1.0-1.5)
Lungime medie propoziție

Proporție propoziții scurte/lungi

Variație lungime mesaje

Densitate link-uri

⚙️ Configurare Avansată
Parametrii pot fi ajustați în main.py:

python
# Praguri similaritate
THRESHOLD = {
    "repost":      0.88,  # Prag pentru repostări
    "similar":     0.72,  # Prag pentru similaritate semantică
    "stylography": 0.72,  # Prag pentru stilometrie
}

# Performanță
DECAY_BASE = 0.88        # Factor decay pentru relații
BATCH_SIZE = 20          # Canale per batch la scraping
MAX_PAIRS_PER_CYCLE = 300  # Perechi analizate per ciclu
TIMEOUT = 12             # Timeout scraping per canal (secunde)
🧪 Exemple de Utilizare
Adăugare manuală canale
bash
# În interfață, introduceți:
@stiri
@news_ro
@actualitate
Încărcare fișier .txt
txt
stiri
news_ro
actualitate
politic
economie
Export date
json
{
  "channels": ["@stiri", "@news_ro"],
  "edges": [
    {"from": "@stiri", "to": "@news_ro", "strength": 0.92}
  ],
  "entities": {
    "PER": {"Ion Popescu": {"count": 5, "sum": 3.2}},
    "ORG": {"Guvern": {"count": 3, "sum": -1.5}}
  }
}
🤝 Contribuții
Contribuțiile sunt binevenite! Te rugăm să:

Fork the repository

Creează branch pentru feature (git checkout -b feature/amazing)

Commit changes (git commit -m 'Add amazing feature')

Push to branch (git push origin feature/amazing)

Deschide Pull Request

Reguli de contribuție
Păstrează stilul de cod existent

Adaugă comentarii pentru funcții noi

Actualizează documentația

Testează înainte de PR

🐛 Bug Reporting
Pentru bug-uri, te rugăm să deschizi un issue cu:

Descrierea problemei

Pași de reproducere

Comportament așteptat vs. actual

Log-uri de eroare (dacă există)

Versiuni software (Python, pachete)

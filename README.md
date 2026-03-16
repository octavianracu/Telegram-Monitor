Telegram Monitor - Analiză Rețele de Canale Telegram
https://img.shields.io/badge/version-2.0-blue
https://img.shields.io/badge/python-3.8%252B-green
https://img.shields.io/badge/FastAPI-0.68%252B-teal
https://img.shields.io/badge/license-MIT-orange

📋 Descriere
Telegram Monitor este o aplicație web pentru monitorizarea și analiza rețelelor de canale Telegram. Sistemul scanează canale specificate, extrage postări și construiește un graf al relațiilor dintre canale bazat pe similaritatea conținutului.

✨ Caracteristici principale
🎯 Analiză în trei moduri
Mod	Descriere	Utilizare
🔍 Direct	Compară direct doar perechile cu conținut nou	Acuratețe maximă
🔄 Hibrid	70% comparații directe + 30% inferențe din graf	Balanță viteză/acuratețe
🌐 Tranzitiv	Prioritizează descoperirea de conexiuni prin lanțuri	Descoperire rapidă de comunități
📊 Analiză avansată
Similaritate semantică - folosește modele transformer (paraphrase-multilingual-mpnet-base-v2)

Stilometrie - analiză pe 18 dimensiuni stilistice

Detecție repostări - identifică conținut duplicat

Analiză sentiment - clasificare sentiment pe mesaje

NER - extragere entități (persoane, organizații)

🕸️ Vizualizare rețea
Graf interactiv cu canale și conexiuni

Comunități detectate automat (algoritm Louvain)

Metrici de rețea (grad, betweenness, PageRank)

Filtrare după canal țintă

💾 Persistență date
Backup automat la fiecare oră

Backup manual prin API

Salvare/încărcare proiecte

Export în format JSON

🏗️ Arhitectură
text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Backend       │────▶│   Baze de date  │
│   (HTML/JS)     │◀────│   (FastAPI)     │◀────│   (Memorie)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                        │
        ▼                       ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Vizualizare   │     │   WebSocket     │     │   Embedding     │
│   vis.js        │     │   în timp real  │     │   Cache         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
🚀 Instalare
Cerințe sistem
Python 3.8 sau mai nou

8GB RAM recomandat (pentru modele NLP)

Conexiune internet pentru descărcarea modelelor

Pași instalare
bash
# 1. Clonează repository
git clone https://github.com/utilizator/telegram-monitor.git
cd telegram-monitor

# 2. Creează și activează virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instalează dependențe
pip install -r requirements.txt

# 4. Pornește aplicația
python main.py
📦 Dependențe principale
txt
fastapi==0.68.0
uvicorn==0.15.0
websockets==10.0
requests==2.26.0
beautifulsoup4==4.10.0
numpy==1.21.0
networkx==2.6.3
sentence-transformers==2.1.0
scikit-learn==0.24.2
transformers==4.11.3
torch==1.9.0
🎮 Utilizare
1. Adăugare canale
text
@canal1
@canal2
@canal3
2. Selectare mod analiză
Alege între:

DIRECT - pentru acuratețe maximă

HIBRID - pentru balanță optimă

TRANZITIV - pentru descoperire rapidă

3. Monitorizare
Sistemul scanează automat canalele la fiecare 10 secunde

Conexiunile descoperite apar în graf și în fluxul de postări

Poți urmări metrici de rețea în timp real

4. Salvare proiecte
javascript
// Backup manual
// Din interfață, click pe "PROIECTE" → "Salvează"
// Fișierele sunt salvate în directorul `projects/`
📁 Structură proiect
text
telegram-monitor/
├── main.py                 # Aplicația principală
├── static/
│   └── index.html          # Interfață web
├── projects/               # Proiecte salvate
├── backups/                # Backup-uri automate
├── requirements.txt        # Dependențe
└── README.md               # Documentație
🔧 Configurare avansată
Praguri similaritate
În main.py poți ajusta pragurile:

python
THRESHOLD = {
    "repost":      0.88,    # Prag pentru identificare repostări
    "similar":     0.72,    # Prag pentru similaritate semantică
    "stylography": 0.72,    # Prag pentru similaritate stilometrică
}

INFERENCE_THRESHOLD = {
    "direct":   0.72,        # Prag mod direct
    "hibrid":   0.65,        # Prag mod hibrid
    "tranzitiv": 0.60,       # Prag mod tranzitiv
}
Endpointuri API
Metodă	Endpoint	Descriere
GET	/api/backup_now	Backup manual
GET	/api/project/list	Listare proiecte
POST	/api/project/save	Salvare proiect
POST	/api/project/load	Încărcare proiect
DELETE	/api/project/delete	Ștergere proiect
WebSocket	/ws	Conexiune în timp real
📊 Algoritmi de similaritate
1. Similaritate semantică
Embedding-uri generate cu SentenceTransformer

Similaritate cosinus între vectori

Bonus NLP pentru sentiment și entități comune

2. Stilometrie
18 dimensiuni analizate:

Lungime caracter/mesaj

Frecvență punctuație

Tip-token ratio (TTR)

Proporție cuvinte unice (hapax)

Și altele...

3. Inferență tranzitivă
Lanțuri de lungime 2 (A-B-C)

Lanțuri de lungime 3 (A-B-C-D)

Propagare cu decay exponențial

🤝 Contribuții
Contribuțiile sunt binevenite! Te rugăm să:

Fork repository

Creează branch nou (git checkout -b feature/amazing-feature)

Commit modificări (git commit -m 'Add amazing feature')

Push branch (git push origin feature/amazing-feature)

Deschide Pull Request

📝 To-Do
Autentificare utilizatori

Salvare în baze de date (PostgreSQL/MongoDB)

Export în formate multiple (CSV, GEXF)

Analiză timeline (evoluție în timp)

Notificări în timp real

Dockerizare aplicație

📄 Licență
Acest proiect este licențiat sub MIT License.

✉️ Contact
Autor: Octavian Racu

Telegram: @racumd

Canal monitorizare: @socialcomputing

⚠️ Notă: Asigură-te că respecți termenii de utilizare Telegram când folosești acest instrument.

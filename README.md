**TGM Monitor** este un instrument de cercetare pentru analiza coordonării narative în rețelele de canale Telegram publice. Sistemul combină scraping continuu, modele NLP multilingve și clustering nesupervizat pentru a detecta automat grupuri de canale care promovează narrative similare — pe perioade de zile sau săptămâni, nu de minute.

### Ce face

- Scrapează automat mesajele din canale publice Telegram și le stochează persistent în SQLite
- Calculează similaritatea semantică între canale folosind embeddings multilingve (`paraphrase-multilingual-mpnet-base-v2`) și similaritate cosinus
- Detectează trei tipuri de relații: **directe** (același subiect în 3 zile), **hibride** (coordonare prin hub comun) și **tranzitive** (rețele de amplificare indirectă A→B→C)
- Construiește profiluri narative per canal prin media mobilă exponențială (EMA) pe 7 zile de embeddings zilnice
- Rulează **BERTopic** automat o dată pe zi pentru a descoperi temele narative dominante fără supraveghere umană
- Vizualizează graful de coordonare în timp real prin WebSocket cu detecție de comunități Louvain (seed determinist)
- Expune API REST pentru interogarea temelor narative, statusului NLP și backup-urilor

### Stack tehnic

| Componentă | Tehnologie |
|---|---|
| Backend | Python 3.10+, FastAPI, uvicorn |
| Persistență | SQLite cu WAL mode |
| NLP | sentence-transformers, HuggingFace transformers |
| Clustering | BERTopic, scikit-learn |
| Graf | NetworkX, algoritm Louvain |
| Frontend | HTML/JS single-page, WebSocket |

### Cazuri de utilizare

Monitorizarea coordonării narative în campanii electorale, detectarea rețelelor de dezinformare, analiza ecosistemelor media regionale, cercetare academică în comunicare politică.

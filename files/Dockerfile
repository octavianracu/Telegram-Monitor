FROM python:3.11-slim

WORKDIR /app

# Dependențe sistem necesare pentru compilarea unor pachete Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiem doar requirements primul — layer cacheuit separat.
# La rebuild după modificarea main.py, pip install nu rulează din nou.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiem codul sursă și fișierele statice
COPY main.py .
COPY static/ ./static/

# Directoarele de date — create în imagine, populate de volume la runtime
RUN mkdir -p projects backups

# Portul pe care ascultă uvicorn
EXPOSE 8000

# HuggingFace descarcă modelele în /root/.cache/huggingface.
# Acest director este montat ca volum în docker-compose.yml
# ca modelele să nu fie descărcate la fiecare recreare a containerului.
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

CMD ["python", "main.py"]

# backup_manual.py
import urllib.request
import json
import os
from datetime import datetime

# Crează folder backups
os.makedirs("backups", exist_ok=True)

# Încearcă să faci backup prin API
try:
    response = urllib.request.urlopen("http://localhost:8000/api/backup_now")
    data = json.loads(response.read())
    print("Backup reușit:", data)
except:
    # Dacă API-ul nu există, fă backup manual
    print("API-ul nu există. Backup manual:")
    
    # Aici nu putem accesa datele serverului din exterior
    print("Pentru backup manual, trebuie să rulezi codul ÎN CONSOLA SERVERULUI")
    print("Sau să adaugi endpointul în cod și să testezi din nou în browser")
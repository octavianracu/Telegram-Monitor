#!/usr/bin/env python3
"""
Script pentru verificarea stării serverului și analizei emergente.
"""

import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def check_server():
    """Verifică dacă serverul răspunde."""
    try:
        resp = requests.get(f"{BASE_URL}/api/nlp_status", timeout=5)
        print(f"✅ Serverul rulează - NLP ready: {resp.json().get('nlp_ready')}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Serverul NU rulează!")
        return False
    except requests.exceptions.Timeout:
        print("⚠️ Serverul rulează dar nu răspunde (timeout)")
        return True
    except Exception as e:
        print(f"❌ Eroare: {e}")
        return False

def check_emergent_status():
    """Verifică starea analizei emergente."""
    try:
        resp = requests.get(f"{BASE_URL}/api/emergent_ideologies", timeout=10)
        data = resp.json()
        
        if data.get('status') == 'success':
            result = data.get('data', {})
            ideologies = result.get('emergent_ideologies', [])
            print(f"✅ Analiză completă: {len(ideologies)} ideologii descoperite")
            
            for ideo in ideologies[:5]:
                print(f"   - {ideo.get('name')}")
            return True
            
        elif data.get('status') == 'cached':
            print(f"✅ Date din cache disponibile")
            return True
            
        elif data.get('status') == 'pending':
            print(f"⏳ Analiză în curs: {data.get('message', '')}")
            return False
            
        else:
            print(f"⚠️ Status necunoscut: {data}")
            return False
            
    except requests.exceptions.Timeout:
        print("⚠️ Timeout la verificare - analiza poate fi în curs")
        return False
    except Exception as e:
        print(f"❌ Eroare la verificare: {e}")
        return False

def check_narratives():
    """Verifică starea temelor narative."""
    try:
        resp = requests.get(f"{BASE_URL}/api/narratives", timeout=10)
        data = resp.json()
        topics = data.get('topics', [])
        channels = data.get('total_channels_profiled', 0)
        print(f"📊 Teme narative: {len(topics)} teme, {channels} canale profilate")
        return True
    except Exception as e:
        print(f"❌ Eroare: {e}")
        return False

def main():
    print("=" * 60)
    print("DIAGNOSTIC SERVER - ANALIZĂ EMERGENTĂ")
    print("=" * 60)
    
    if not check_server():
        print("\n🔥 Serverul nu rulează! Pornește-l cu: python main.py")
        sys.exit(1)
    
    print("\n--- Stare analiză emergentă ---")
    completed = check_emergent_status()
    
    print("\n--- Stare teme narative ---")
    check_narratives()
    
    if not completed:
        print("\n" + "=" * 60)
        print("⚠️ ANALIZA ÎNCĂ RULEAZĂ")
        print("=" * 60)
        print("\nDacă analiza durează mai mult de 10 minute, probabil s-a blocat.")
        print("\nSoluții:")
        print("  1. Așteaptă încă 5 minute")
        print("  2. Oprește serverul (Ctrl+C) și pornește din nou")
        print("  3. Rulează analiza cu parametri reduși:")
        print("     curl http://localhost:8000/api/discover_ideologies?force=true")
        print("\nPentru a verifica din nou, rulează din nou acest script.")

if __name__ == "__main__":
    main()
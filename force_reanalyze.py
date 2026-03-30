#!/usr/bin/env python3
"""
Script pentru forțarea reevaluării complete a datelor colectate.
Rulează: python force_reanalyze.py
"""

import requests
import time
import sys
import sqlite3
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"
DB_PATH = "tgm_monitor.db"

# Timeout-uri mai mari
LONG_TIMEOUT = 300  # 5 minute pentru operații lungi
SHORT_TIMEOUT = 30  # 30 secunde pentru verificări

def check_nlp_status():
    """Verifică dacă modelele NLP sunt încărcate."""
    try:
        resp = requests.get(f"{BASE_URL}/api/nlp_status", timeout=SHORT_TIMEOUT)
        data = resp.json()
        if data.get("nlp_ready"):
            print(f"✅ NLP ready: {data.get('nlp_status')}")
            return True
        else:
            print(f"⏳ NLP încărcare: {data.get('nlp_status')}")
            return False
    except Exception as e:
        print(f"❌ Eroare conectare: {e}")
        return False

def wait_for_nlp():
    """Așteaptă până când NLP-ul este gata."""
    print("Aștept încărcarea modelelor NLP...")
    attempts = 0
    while attempts < 60:
        if check_nlp_status():
            return True
        attempts += 1
        time.sleep(5)
    print("❌ Timeout: NLP nu s-a încărcat în 5 minute")
    return False

def reset_via_sqlite():
    """Resetare manuală prin SQLite."""
    print("\n=== Resetare manuală via SQLite ===\n")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Verificăm ce avem înainte
        cursor.execute("SELECT COUNT(*) FROM narrative_topics")
        old_topics = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM channel_narrative_profile")
        old_channel_profiles = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM messages")
        old_messages = cursor.fetchone()[0]
        
        print(f"Date existente:")
        print(f"  - narrative_topics: {old_topics}")
        print(f"  - channel_narrative_profile: {old_channel_profiles}")
        print(f"  - messages: {old_messages}")
        
        # Ștergem datele narative
        cursor.execute("DELETE FROM narrative_topics")
        cursor.execute("DELETE FROM channel_daily_embeddings")
        
        # Resetăm profilele canalelor
        cursor.execute("""
            UPDATE channel_narrative_profile 
            SET topic_distribution = '{}', dominant_topic = -1
        """)
        
        # Ștergem analizele emergente
        cursor.execute("DELETE FROM emergent_ideologies")
        
        conn.commit()
        
        # Verificăm după reset
        cursor.execute("SELECT COUNT(*) FROM narrative_topics")
        new_topics = cursor.fetchone()[0]
        
        print(f"\nDupă reset:")
        print(f"  - narrative_topics: {new_topics} (șterse {old_topics - new_topics})")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Eroare SQLite: {e}")
        return False

def rebuild_profiles_via_api():
    """Reconstruiește profilele prin API (operație lungă)."""
    print("\n=== Reconstruire profile ===\n")
    
    try:
        # Folosim timeout mai mare
        resp = requests.get(f"{BASE_URL}/api/rebuild_profiles", timeout=LONG_TIMEOUT)
        data = resp.json()
        print(f"   → {data.get('status')}: {data.get('message', '')}")
        return data.get('status') == 'started'
    except requests.exceptions.Timeout:
        print("   ⚠️ Timeout la rebuild_profiles - operația continuă în background")
        return True  # Considerăm că a pornit
    except Exception as e:
        print(f"   ❌ Eroare: {e}")
        return False

def run_bertopic_via_api():
    """Rulează BERTopic prin API (operație lungă)."""
    print("\n=== Pornire BERTopic ===\n")
    
    try:
        resp = requests.get(f"{BASE_URL}/api/run_bertopic", timeout=LONG_TIMEOUT)
        data = resp.json()
        print(f"   → {data.get('status')}: {data.get('message', '')}")
        return data.get('status') == 'started'
    except requests.exceptions.Timeout:
        print("   ⚠️ Timeout la run_bertopic - operația continuă în background")
        return True
    except Exception as e:
        print(f"   ❌ Eroare: {e}")
        return False

def discover_emergent_ideologies_async():
    """Pornește analiza emergentă și verifică periodic statusul."""
    print("\n=== Pornire analiză emergentă (background) ===\n")
    
    try:
        # Pornim analiza
        resp = requests.get(f"{BASE_URL}/api/discover_ideologies?force=true", timeout=SHORT_TIMEOUT)
        data = resp.json()
        
        if data.get('status') == 'success':
            print(f"   ✅ Analiză pornită cu succes")
            return True
        elif data.get('status') == 'started':
            print(f"   ⏳ Analiză pornită în background")
            return True
        else:
            print(f"   → {data.get('status')}: {data.get('message', '')}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ⚠️ Timeout la pornire - analiza rulează în background")
        return True
    except Exception as e:
        print(f"   ❌ Eroare: {e}")
        return False

def wait_for_emergent_analysis(max_wait_minutes=10):
    """Așteaptă finalizarea analizei emergente."""
    print(f"\n⏳ Aștept finalizarea analizei emergente (max {max_wait_minutes} minute)...")
    
    start_time = time.time()
    last_status = ""
    
    while time.time() - start_time < max_wait_minutes * 60:
        try:
            resp = requests.get(f"{BASE_URL}/api/emergent_ideologies", timeout=SHORT_TIMEOUT)
            data = resp.json()
            
            if data.get('status') == 'success':
                result_data = data.get('data', {})
                ideologies = result_data.get('emergent_ideologies', [])
                
                if ideologies:
                    print(f"\n   ✅ Analiză completă! {len(ideologies)} ideologii descoperite")
                    return True
                else:
                    status_msg = f"   ⏳ Procesare... {len(ideologies)} ideologii până acum"
                    if status_msg != last_status:
                        print(status_msg)
                        last_status = status_msg
            elif data.get('status') == 'cached':
                print(f"   ✅ Date din cache disponibile")
                return True
            else:
                if last_status != data.get('status'):
                    print(f"   ⏳ Status: {data.get('status')} - {data.get('message', '')}")
                    last_status = data.get('status')
                    
        except Exception as e:
            if last_status != f"Eroare: {e}":
                print(f"   ⏳ Verificare... ({e})")
                last_status = f"Eroare: {e}"
        
        # Așteaptă 10 secunde între verificări
        for i in range(10):
            time.sleep(1)
            # Arată un indicator de progres
            if i == 9:
                print(".", end="", flush=True)
    
    print("\n   ⚠️ Timeout: analiza durează mai mult decât estimat")
    print("   Verifică manual mai târziu cu /api/emergent_ideologies")
    return False

def check_narratives():
    """Verifică starea narativă."""
    print("\n=== Verificare teme narative ===\n")
    
    try:
        resp = requests.get(f"{BASE_URL}/api/narratives", timeout=SHORT_TIMEOUT)
        data = resp.json()
        topics = data.get('topics', [])
        channels = data.get('total_channels_profiled', 0)
        
        print(f"   → Teme narative: {len(topics)}")
        print(f"   → Canale profilate: {channels}")
        
        for t in topics[:5]:
            keywords = t.get('keywords', [])[:3]
            print(f"      - Topic {t.get('topic_id')}: {' · '.join(keywords)}")
        
        return len(topics) > 0
        
    except Exception as e:
        print(f"   ❌ Eroare: {e}")
        return False

def check_emergent_ideologies():
    """Verifică starea ideologiilor emergente."""
    print("\n=== Verificare ideologii emergente ===\n")
    
    try:
        resp = requests.get(f"{BASE_URL}/api/emergent_ideologies", timeout=SHORT_TIMEOUT)
        data = resp.json()
        
        if data.get('status') == 'success':
            result_data = data.get('data', {})
            ideologies = result_data.get('emergent_ideologies', [])
            channels = result_data.get('channels_analyzed', 0)
            
            print(f"   → Ideologii emergente: {len(ideologies)}")
            print(f"   → Canale analizate: {channels}")
            
            for ideo in ideologies[:5]:
                print(f"\n      --- {ideo.get('name')} ---")
                print(f"          Categorii: {', '.join(ideo.get('category_names', [])[:3])}")
                phrases = ideo.get('signature_phrases', [])[:2]
                if phrases:
                    print(f"          Fraze: {', '.join(phrases)}")
                tone = ideo.get('sentiment_profile', {}).get('dominant_tone', 'neutru')
                print(f"          Ton: {tone}")
            
            return True
        elif data.get('status') == 'pending':
            print(f"   ⏳ {data.get('message', 'Analiză în curs...')}")
            return False
        else:
            print(f"   → {data.get('status')}: {data.get('message', '')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Eroare: {e}")
        return False

def get_channel_ideology(channel: str):
    """Obține profilul ideologic al unui canal."""
    print(f"\n=== Profil ideologic pentru {channel} ===\n")
    
    try:
        resp = requests.get(f"{BASE_URL}/api/emergent_ideology/{channel}", timeout=SHORT_TIMEOUT)
        data = resp.json()
        
        if 'error' not in data:
            print(f"   📝 {data.get('description', 'N/A')}")
            print(f"\n   🏷️  Ideologii dominante: {', '.join(data.get('dominant_ideologies', [])) or 'Niciuna'}")
            
            for ideo in data.get('ideologies_detail', [])[:3]:
                print(f"\n   📌 {ideo.get('name')} (scor: {ideo.get('score', 0):.2f})")
                print(f"      Categorii: {', '.join(ideo.get('categories', [])[:4])}")
                print(f"      Ton: {ideo.get('tone', 'neutru')}")
                if ideo.get('signature_phrases'):
                    print(f"      Fraze cheie: {', '.join(ideo.get('signature_phrases', [])[:3])}")
            
            return True
        else:
            print(f"   → {data.get('error', 'Eroare necunoscută')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Eroare: {e}")
        return False

def get_saturation_status():
    """Verifică statusul saturației."""
    print("\n=== Status saturație teoretică ===\n")
    
    try:
        resp = requests.get(f"{BASE_URL}/api/emergent_saturation", timeout=SHORT_TIMEOUT)
        data = resp.json()
        
        if 'error' not in data:
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Iterații: {data.get('iterations_completed', 0)}")
            print(f"   Rată schimbare finală: {data.get('final_change_rate', 0):.4f}")
            print(f"   Număr ideologii: {data.get('ideologies_count', 0)}")
            
            history = data.get('history', [])
            if history:
                print("\n   Istoric iterații:")
                for h in history[-3:]:
                    print(f"      - Iter {h['iteration']}: {h['ideologies_count']} ideologii, change={h['change_rate']:.4f}")
            
            return True
        else:
            print(f"   → {data.get('error', 'N/A')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Eroare: {e}")
        return False

def main():
    """Funcția principală."""
    print("=" * 70)
    print("FORȚAREA REEVALUĂRII CU GROUNDED THEORY")
    print("=" * 70)
    
    # Verificăm serverul
    try:
        resp = requests.get(f"{BASE_URL}/api/nlp_status", timeout=5)
        print(f"✅ Serverul rulează la {BASE_URL}")
    except Exception as e:
        print(f"❌ Serverul nu răspunde la {BASE_URL}")
        print(f"   Asigură-te că aplicația rulează: python main.py")
        sys.exit(1)
    
    # Așteptăm NLP
    if not wait_for_nlp():
        print("Nu pot continua fără modele NLP încărcate")
        sys.exit(1)
    
    # Întrebăm utilizatorul
    print("\n" + "=" * 70)
    print("Alege metoda de reevaluare:")
    print("  1. Resetare completă + rebuild + BERTopic + emergent (recomandat)")
    print("  2. Doar rebuild profile și BERTopic")
    print("  3. Doar analiză emergentă (pe datele existente)")
    print("  4. Verifică starea curentă")
    print("=" * 70)
    
    choice = input("\nAlegere (1/2/3/4): ").strip()
    
    if choice == "1":
        # Resetare completă
        if reset_via_sqlite():
            print("\n✅ Resetare SQLite completă")
        else:
            print("\n❌ Resetare eșuată")
            sys.exit(1)
        
        # Așteptăm puțin
        time.sleep(2)
        
        # Rebuild profile
        if rebuild_profiles_via_api():
            print("\n⏳ Aștept reconstruirea profilelor (2-5 minute)...")
            time.sleep(30)
            
            # Rulează BERTopic
            if run_bertopic_via_api():
                print("\n⏳ Aștept clustering BERTopic (1-3 minute)...")
                time.sleep(60)
        
        # Verifică temele narative
        check_narratives()
        
        # Pornește analiza emergentă
        if discover_emergent_ideologies_async():
            # Așteaptă finalizarea
            wait_for_emergent_analysis(max_wait_minutes=10)
            
            # Verifică rezultatele
            check_emergent_ideologies()
            get_saturation_status()
        
    elif choice == "2":
        # Doar rebuild și BERTopic
        if rebuild_profiles_via_api():
            print("\n⏳ Aștept reconstruirea profilelor (2-5 minute)...")
            time.sleep(30)
            
            if run_bertopic_via_api():
                print("\n⏳ Aștept clustering BERTopic (1-3 minute)...")
                time.sleep(60)
        
        check_narratives()
        
        # Opțional: analiză emergentă
        run_emergent = input("\nRulează și analiza emergentă? (y/n): ").strip().lower()
        if run_emergent == 'y':
            if discover_emergent_ideologies_async():
                wait_for_emergent_analysis(max_wait_minutes=8)
                check_emergent_ideologies()
        
    elif choice == "3":
        # Doar analiză emergentă
        if discover_emergent_ideologies_async():
            wait_for_emergent_analysis(max_wait_minutes=10)
            check_emergent_ideologies()
            get_saturation_status()
        
    elif choice == "4":
        # Verificare stare
        check_narratives()
        check_emergent_ideologies()
        get_saturation_status()
        
        # Dacă există canale, arată un exemplu
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT channel FROM messages LIMIT 5")
        channels = [r[0] for r in cursor.fetchall()]
        conn.close()
        
        if channels:
            print("\n--- Exemplu profil pentru primul canal ---")
            get_channel_ideology(channels[0])
    
    else:
        print("Alegere invalidă")
        sys.exit(1)
    
    # Rezumat final
    print("\n" + "=" * 70)
    print("REZUMAT FINAL")
    print("=" * 70)
    
    check_narratives()
    check_emergent_ideologies()
    get_saturation_status()
    
    print("\n" + "=" * 70)
    print("✅ REEVALUARE COMPLETĂ")
    print("=" * 70)
    print("\n📌 Pentru a vizualiza rezultatele:")
    print("   - Interfață web: http://localhost:8000/static/index.html")
    print("   - Teme narative: http://localhost:8000/api/narratives")
    print("   - Ideologii emergente: http://localhost:8000/api/emergent_ideologies")
    print("   - Status saturație: http://localhost:8000/api/emergent_saturation")
    print("\n📌 Pentru un canal specific:")
    print("   - http://localhost:8000/api/emergent_ideology/@nume_canal")

if __name__ == "__main__":
    main()
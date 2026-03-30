#!/usr/bin/env python3
"""
Script pentru analiză rapidă cu parametri reduși.
Rulează separat de server.
"""

import asyncio
import sys
import os
import sqlite3
import json
from datetime import datetime
from collections import defaultdict

# Adaugă calea pentru importuri
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    db_connect, db_get_recent_messages, db_get_all_channels,
    similarity_model, sentiment_pipeline, ner_pipeline, nlp_ready,
    GroundedIdeologyDiscoverer, _load_nlp_models
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def quick_analysis():
    """Analiză rapidă cu parametri reduși."""
    
    print("=" * 60)
    print("ANALIZĂ RAPIDĂ - GROUNDED THEORY")
    print("=" * 60)
    
    # Asigură-te că modelele sunt încărcate
    if not nlp_ready or similarity_model is None:
        print("Încărcare modele NLP...")
        loop = asyncio.get_event_loop()
        await asyncio.to_thread(_load_nlp_models, loop)
        
        # Așteaptă puțin
        for i in range(30):
            if nlp_ready and similarity_model is not None:
                break
            await asyncio.sleep(2)
            print(".", end="", flush=True)
        print()
    
    if not nlp_ready:
        print("❌ Modelele NLP nu s-au încărcat")
        return
    
    print("✅ Modele NLP gata")
    
    # Colectare mesaje - parametri reduși
    all_messages = {}
    channels = db_get_all_channels()
    
    print(f"Colectare mesaje din {len(channels)} canale...")
    
    for i, channel in enumerate(channels):
        # Doar ultimele 3 zile, nu 14
        messages = db_get_recent_messages(channel, days=3)
        
        # Filtrăm mesaje scurte
        messages = [m for m in messages if len(m) > 50]
        
        if len(messages) >= 3:  # Prag mai mic
            # Limităm la 20 mesaje per canal
            all_messages[channel] = messages[:20]
            
        if (i + 1) % 50 == 0:
            print(f"   Procesate {i+1}/{len(channels)} canale, {len(all_messages)} cu mesaje")
    
    print(f"Analiză pe {len(all_messages)} canale cu mesaje suficiente")
    
    if len(all_messages) < 3:
        print("❌ Prea puține canale cu mesaje")
        return
    
    # Analiză
    discoverer = GroundedIdeologyDiscoverer(
        similarity_model=similarity_model,
        sentiment_pipeline=sentiment_pipeline,
        ner_pipeline=ner_pipeline
    )
    
    print("Pornesc analiza Grounded Theory...")
    print("(acest pas poate dura 2-5 minute)")
    
    try:
        result = await asyncio.to_thread(
            discoverer.discover_ideologies,
            all_messages,
            max_iterations=3  # Doar 3 iterații
        )
        
        ideologies = result.get("emergent_ideologies", [])
        
        print("\n" + "=" * 60)
        print(f"✅ ANALIZĂ COMPLETĂ: {len(ideologies)} ideologii descoperite")
        print("=" * 60)
        
        for i, ideo in enumerate(ideologies[:10]):
            print(f"\n{i+1}. {ideo['name']}")
            print(f"   Categorii: {', '.join(ideo.get('category_names', [])[:5])}")
            print(f"   Ton: {ideo.get('sentiment_profile', {}).get('dominant_tone', 'neutru')}")
            if ideo.get('signature_phrases'):
                print(f"   Fraze cheie: {', '.join(ideo['signature_phrases'][:3])}")
        
        # Salvează rezultatul
        result["analysis_timestamp"] = datetime.now().isoformat()
        result["channels_analyzed"] = len(all_messages)
        
        # Salvează în fișier JSON
        with open("emergent_ideologies.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Rezultat salvat în emergent_ideologies.json")
        
    except Exception as e:
        print(f"❌ Eroare: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(quick_analysis())
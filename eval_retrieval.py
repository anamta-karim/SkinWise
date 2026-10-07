"""Measure how reliably SkinWise can find a product's ingredient list from the live web.

Run from the project folder (venv active):   python eval_retrieval.py
Uses SerpApi credits for products that are not cached yet (about 1-2 each; repeats are free).
Writes retrieval_results.json and prints a table you can paste into the README.
"""
import json
import time
import statistics
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()

import app as skinwise            # loads the models once (takes ~20 s)
import serpapi_utils as su

PRODUCTS = [
    "CeraVe Moisturising Cream",
    "Cetaphil Gentle Skin Cleanser",
    "Neutrogena Hydro Boost Water Gel",
    "La Roche-Posay Effaclar Duo+M",
    "The Ordinary Niacinamide 10% + Zinc 1%",
    "Minimalist 10% Niacinamide Serum",
    "The Derma Co 2% Salicylic Acid Serum",
    "Dot & Key Watermelon Sunscreen SPF 50",
    "Foxtale Vitamin C Serum",
    "Plum 10% Niacinamide Serum",
    "Pyunkang Yul Essence Toner",
    "COSRX Advanced Snail 96 Mucin Power Essence",
]

rows = []
for name in PRODUCTS:
    before = su.credits_used_this_session
    start = time.time()
    try:
        ings, sources = skinwise.get_product_ingredients(name)
        err = ""
    except Exception as e:                      # a timeout etc. counts as a miss
        ings, sources, err = [], [], str(e)[:60]
    host = urlparse(sources[0]["link"]).netloc.replace("www.", "") if (sources and ings) else "-"
    rows.append({
        "product": name,
        "found": len(ings) >= 8,          # "usable" = 8+ ingredients; shorter lists are low-confidence
        "n_ingredients": len(ings),
        "match": ("closest variant" if sources and sources[0].get("note") else "exact") if ings else "-",
        "confidence": skinwise.source_confidence(len(ings), sources) if ings else "-",
        "source": host,
        "credits": su.credits_used_this_session - before,
        "seconds": round(time.time() - start, 1),
        "error": err,
    })
    r = rows[-1]
    print(f"{'OK ' if r['found'] else 'MISS'} {name:45} {r['n_ingredients']:>3} ingredients  {r['match']:15} {r['source']}")

found = [r for r in rows if r["found"]]
low = [r for r in rows if not r["found"] and r["n_ingredients"] > 0]
missed = [r for r in rows if r["n_ingredients"] == 0]
good = [r for r in found if r["confidence"] == "good"]
exact = [r for r in found if r["match"] == "exact"]
print("\n| Product | Found | Ingredients | Match | Confidence | Source |")
print("|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['product']} | {'yes' if r['found'] else 'no'} | {r['n_ingredients'] or '-'} | {r['match']} | {r['confidence']} | {r['source']} |")

print(f"\nUsable ingredient list (8+ ingredients) for {len(found)}/{len(rows)} products; "
      f"{len(low)} low-confidence (<8 ingredients); {len(missed)} not found. "
      f"({len(exact)} exact-product pages, {len(found) - len(exact)} closest-variant, {len(good)} with 15+ ingredients). "
      f"Median list length: {statistics.median([r['n_ingredients'] for r in found]) if found else 0}. "
      f"Credits used this run: {su.credits_used_this_session}.")
print("Spot-check 3-4 rows against the real pack: the source column shows which site the list came from.")

with open("retrieval_results.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=1, ensure_ascii=False)
print("Saved retrieval_results.json")
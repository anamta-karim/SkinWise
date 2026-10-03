"""SerpApi helpers for SkinWise (hackathon build).

- Every response is cached on disk (data/serpapi_cache.json), so repeated
  searches cost 0 credits. The free plan only gives 250 searches/month.
- Uses plain `requests` against the SerpApi REST endpoint (no extra SDK needed).
"""
import os
import re
import json
import math
import html as htmllib
import threading
import requests
from rapidfuzz import fuzz

SERPAPI_URL = "https://serpapi.com/search.json"
CACHE_FILE = os.path.join("data", "serpapi_cache.json")

_lock = threading.Lock()
_cache = {}
credits_used_this_session = 0

if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            _cache = json.load(f)
    except Exception:
        _cache = {}


def _save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False, indent=1)


def serpapi_search(engine, **params):
    """Run one SerpApi search. Returns (json, from_cache). Raises on API errors."""
    global credits_used_this_session
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is not set in .env")

    cache_key = engine + "|" + json.dumps(params, sort_keys=True)
    with _lock:
        if cache_key in _cache:
            return _cache[cache_key], True

    r = requests.get(SERPAPI_URL,
                     params={"engine": engine, "api_key": api_key, **params},
                     timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(data["error"])

    with _lock:
        _cache[cache_key] = data
        credits_used_this_session += 1
        _save_cache()
    return data, False


def shopping_search(query):
    """Google Shopping (India). Returns a list of normalised offers."""
    data, cached = serpapi_search("google_shopping", q=query, gl="in", hl="en")
    offers = []
    for item in data.get("shopping_results", []):
        offers.append({
            "title": item.get("title", ""),
            "merchant": item.get("source", ""),
            "price": item.get("price", ""),
            "price_value": item.get("extracted_price"),
            "rating": item.get("rating"),
            "reviews": item.get("reviews"),
            "link": item.get("link") or item.get("product_link", ""),
            "thumbnail": item.get("thumbnail", ""),
        })
    return offers, cached


def price_compare(brand, name, min_match=55, max_offers=6):
    """Live price comparison for one product across merchants."""
    query = f"{brand} {name}".strip()
    try:
        offers, cached = shopping_search(query)
    except Exception as e:
        return {"query": query, "offers": [], "error": str(e)}

    # Drop unrelated results: title must fuzzy-match "brand name"
    good = [o for o in offers
            if fuzz.token_set_ratio(query.lower(), o["title"].lower()) >= min_match
            and o["price_value"] is not None]
    good.sort(key=lambda o: o["price_value"])
    good = good[:max_offers]

    return {
        "query": query,
        "offers": good,
        "cheapest": good[0] if good else None,
        "from_cache": cached,
    }


# ---------------------------------------------------------------------------
# Live product discovery (Google Shopping) - powers the recommender
# ---------------------------------------------------------------------------
CATEGORY_QUERY = {
    "face_wash": "face wash",
    "moisturizer": "face moisturizer",
    "serum_essence": "face serum",
    "sunscreen": "sunscreen for face",
    "toner_mist": "face toner",
    "scrub_exfoliator": "face exfoliator",
    "mask_peel": "face mask",
}


def discover_products(category, skin_type, concerns, limit=5):
    """Find real, currently-listed products for a skin profile via Google Shopping."""
    label = CATEGORY_QUERY.get(category, category.replace("_", " "))
    query = f"{label} for {skin_type} skin {' '.join(concerns[:2])}".strip()
    offers, cached = shopping_search(query)

    seen, products = set(), []
    for o in offers:
        key = o["title"].lower()[:40]
        if not o["title"] or o["price_value"] is None or key in seen:
            continue
        seen.add(key)
        rating = o["rating"] or 0
        reviews = o["reviews"] or 0
        # trust ratings backed by many reviews more than a lone 5.0
        o["score"] = round((rating or 3.5) * math.log10(2 + reviews), 3)
        o["category"] = category
        products.append(o)

    products.sort(key=lambda p: p["score"], reverse=True)
    return products[:limit]


# ---------------------------------------------------------------------------
# Google Search snippets - used for ingredient retrieval and brand evidence
# ---------------------------------------------------------------------------
def search_snippets(query, num=6):
    """Google Search (India). Returns [{title, link, snippet}], from_cache."""
    data, cached = serpapi_search("google", q=query, gl="in", hl="en", num=num)
    out = []
    for r in data.get("organic_results", [])[:num]:
        out.append({
            "title": r.get("title", ""),
            "link": r.get("link", ""),
            "snippet": r.get("snippet", ""),
        })
    return out, cached


# ---------------------------------------------------------------------------
# Read a result page for ingredient lists (plain HTTP fetch, 0 SerpApi credits)
# ---------------------------------------------------------------------------
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

_ITEM = r"[A-Za-z0-9][A-Za-z0-9 ()/\-\.%'+&]{1,45}"
_RUN = re.compile(r"(?:%s,\s*){5,}%s" % (_ITEM, _ITEM))


def find_comma_lists(text, max_lists=8):
    """Find runs of 6+ short comma-separated items (what an INCI list looks like)."""
    lists = []
    for m in _RUN.finditer(text[:400_000]):
        items = [re.sub(r"^.*?ingredients?\s*:?\s*", "", i.strip(), flags=re.I).strip(" .;").lower()
                 for i in m.group(0).split(",")]
        items = [i for i in items if i]
        if len(items) >= 6:
            lists.append(items)
        if len(lists) >= max_lists:
            break
    return lists


def fetch_ingredient_context(url, window=900, max_hits=3):
    """Download a page. Returns {'lists': [candidate ingredient lists], 'windows': text near 'ingredients'}."""
    key = "page2|" + url
    with _lock:
        if key in _cache:
            return _cache[key]
    empty = {"lists": [], "windows": ""}
    try:
        r = requests.get(url, headers=_UA, timeout=8)
        r.raise_for_status()
        raw = r.text[:1_500_000]
    except Exception:
        return empty

    raw = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw)
    text = htmllib.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))
    text = re.sub(r"\s+", " ", text)

    chunks, last_end = [], -1
    for m in re.finditer(r"ingredients?", text, re.I):
        if m.start() < last_end:
            continue
        chunk = text[m.start(): m.start() + window]
        if chunk.count(",") >= 4:
            chunks.append(chunk)
            last_end = m.start() + window
        if len(chunks) >= max_hits:
            break

    out = {"lists": find_comma_lists(text), "windows": " ... ".join(chunks)}
    with _lock:
        _cache[key] = out
        _save_cache()
    return out
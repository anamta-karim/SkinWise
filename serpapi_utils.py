"""SerpApi helpers for SkinWise (hackathon build).

- Every response is cached on disk (data/serpapi_cache.json), so repeated
searches cost 0 credits. The free plan only gives 250 searches/month.
- Uses plain `requests` against the SerpApi REST endpoint (no extra SDK needed).
"""
import os
import re
import time
import json
import math
import html as htmllib
import threading
import requests
from urllib.parse import quote_plus
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

    for attempt in range(2):                     
        try:
            r = requests.get(SERPAPI_URL,
                             params={"engine": engine, "api_key": api_key, **params},
                            timeout=60)
            break
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            if attempt == 1:
                raise
            time.sleep(2)
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
        title = item.get("title", "")
        offers.append({
            "title": item.get("title", ""),
            "merchant": item.get("source", ""),
            "price": item.get("price", ""),
            "price_value": item.get("extracted_price"),
            "rating": item.get("rating"),
            "reviews": item.get("reviews"),
            "link": (item.get("link") or item.get("product_link")
                    or "https://www.google.com/search?tbm=shop&q=" + quote_plus(title)),
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
        if "₹" not in (o["price"] or ""):          
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


# ---------------------------------------------------------------------------
# Amazon Product API - structured ingredient list for an ASIN (1 credit)
# ---------------------------------------------------------------------------
def amazon_item_ingredients(asin, domain="amazon.in"):
    """Returns a cleaned list of ingredients from the Amazon listing's Item Ingredients section."""
    data, _ = serpapi_search("amazon_product", asin=asin, amazon_domain=domain)
    raw = ", ".join(str(i) for i in (data.get("item_ingredients") or []))
    raw = re.sub(r"(?i)^\s*ingredients?\s*:?", "", raw)
    items = [re.sub(r"\s+", " ", x).strip(" .;").lower() for x in raw.split(",")]
    return [i for i in items if i]


def _all_strings(obj):
    """Yield every text value inside a nested JSON response."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _all_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_strings(v)


def amazon_ingredient_candidates(asin, domain="amazon.in"):
    """1 credit (cached). Returns (explicit_item_ingredients, candidate_lists_found_anywhere_in_the_listing)."""
    data, _ = serpapi_search("amazon_product", asin=asin, amazon_domain=domain)
    raw = ", ".join(str(i) for i in (data.get("item_ingredients") or []))
    raw = re.sub(r"(?i)^\s*ingredients?\s*:?", "", raw)
    explicit = [re.sub(r"\s+", " ", x).strip(" .;").lower() for x in raw.split(",")]
    explicit = [i for i in explicit if i]

    text = " | ".join(_all_strings(data))          
    candidates = find_comma_lists(text)
    if not explicit and not candidates:
        print("   amazon response keys:", list(data.keys())[:14])
    return explicit, candidates


# ---------------------------------------------------------------------------
# EthiScan evidence: Google Search (certifications) + Google News (recent coverage)
# ---------------------------------------------------------------------------
TRUSTED_SOURCES = {
    "peta.org": "PETA",
    "leapingbunny.org": "Leaping Bunny",
    "crueltyfreekitty.com": "Cruelty Free Kitty",
    "vegansociety.com": "Vegan Society",
}


def brand_evidence(brand):
    """2 credits (cached): web results about certifications + recent news about the brand."""
    key = brand.lower().split()[0]

    web, _ = search_snippets(f"{brand} cruelty free vegan certified PETA Leaping Bunny", num=8)
    web = [r for r in web
        if key in f"{r['title']} {r['link']}".lower() or r["snippet"].lower().count(key) >= 2]
    for r in web:
        r["trusted"] = next((label for dom, label in TRUSTED_SOURCES.items() if dom in r["link"]), None)
    web.sort(key=lambda r: r["trusted"] is None)          

    news = []
    try:
        data, _ = serpapi_search("google_news", q=f"{brand} skincare animal testing OR cruelty-free", gl="in", hl="en")
        for item in data.get("news_results", []):
            nested = [item.get("highlight") or {}] + (item.get("stories") or [])
            for it in [item] + nested:
                title, link = it.get("title"), it.get("link")
                if not title or not link or key not in title.lower():
                    continue
                news.append({"title": title, "link": link,
                            "source": (it.get("source") or {}).get("name", ""),
                            "date": (it.get("iso_date") or "")[:10]})
    except Exception as e:
        print(f"   ⚠️ news lookup failed: {e}")

    seen, uniq = set(), []
    for n in news:
        if n["link"] not in seen:
            seen.add(n["link"])
            uniq.append(n)
    return {"web": web[:5], "news": uniq[:4]}


# ---------------------------------------------------------------------------
# Ingredient Pulse: Google Trends (India) + related queries + Google Shopping
# ---------------------------------------------------------------------------
def _trends(q, data_type):
    data, _ = serpapi_search("google_trends", q=q, geo="IN", date="today 12-m",
                            data_type=data_type, hl="en", tz="-330")
    return data


def trend_products(ingredient, limit=3):
    """Top-rated live products for an ingredient (INR listings only)."""
    offers, _ = shopping_search(f"{ingredient} serum")
    good = [o for o in offers if o["price_value"] is not None and "\u20b9" in (o["price"] or "")
            and o["price_value"] <= 2500]
    good.sort(key=lambda o: (o["rating"] or 0) * ((o["reviews"] or 0) ** 0.5), reverse=True)
    return good[:limit]


def ingredient_pulse(ingredients):
    """3 credits (cached): interest over time for up to 5 ingredients in India, momentum,
    rising searches and live products for the fastest-rising one."""
    n_ing = len(ingredients)
    data = _trends(",".join(ingredients), "TIMESERIES")
    timeline = (data.get("interest_over_time") or {}).get("timeline_data", [])

    series = []
    for pt in timeline:
        row = [0] * n_ing
        for k, v in enumerate(pt.get("values", [])):
            idx = v.get("query_index", k)
            if 0 <= idx < n_ing:
                row[idx] = v.get("extracted_value", 0) or 0
        series.append({"date": pt.get("date", ""), "values": row})
    if not series:
        raise RuntimeError("Google Trends returned no data for these ingredients")

    stats = []
    k = max(1, len(series) // 6)                      
    for idx, name in enumerate(ingredients):
        col = [p["values"][idx] for p in series]
        first, last = sum(col[:k]) / k, sum(col[-k:]) / k
        stats.append({"name": name,
                    "avg": round(sum(col) / len(col)),
                    "momentum": round((last - first) / first * 100) if first > 0 else None})

    movers = [s for s in stats if s["momentum"] is not None]
    focus = max(movers, key=lambda s: s["momentum"])["name"] if movers else max(stats, key=lambda s: s["avg"])["name"]

    rising, top, products = [], [], []
    try:
        rq = (_trends(focus, "RELATED_QUERIES").get("related_queries") or {})
        rising = [{"query": r.get("query", ""), "value": r.get("value", "")} for r in rq.get("rising", [])[:6]]
        top = [{"query": r.get("query", ""), "value": r.get("value", "")} for r in rq.get("top", [])[:6]]
    except Exception as e:
        print(f"   \u26a0\ufe0f related queries failed: {e}")
    try:
        products = trend_products(focus)
    except Exception as e:
        print(f"   \u26a0\ufe0f trend products failed: {e}")

    return {"ingredients": ingredients, "series": series, "stats": stats,
            "focus": focus, "rising": rising, "top": top, "products": products}

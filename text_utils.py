"""Pure text helpers used by the ingredient-retrieval pipeline.

Kept free of heavy imports (no models, no Flask) so they can be unit tested quickly.
"""
import re

# Words that describe a product type or filler, not its identity
GENERIC_WORDS = {"serum", "toner", "cream", "face", "gel", "lotion", "moisturizer", "moisturiser",
                 "moisturising", "moisturizing", "hydrating", "wash", "cleanser", "sunscreen", "mask",
                 "essence", "skin", "with", "for", "the", "and", "spf", "acid", "free", "pack"}

# Words that signal a *different product* in the same line (a psoriasis cream is not the plain cream)
VARIANT_WORDS = {"psoriasis", "eczema", "baume", "intensive", "renewing", "kids", "baby",
                 "travel", "mini", "sample", "refill", "kit", "combo", "pm", "lotion", "gel"}


def clean_title(title):
    """Shorten a marketplace title ('X Serum for glowing skin, 30 ml | free shipping') to brand + product."""
    t = title.split('|')[0]
    t = re.sub(r'\(.*?\)', ' ', t)
    t = t.split(',')[0]
    t = re.sub(r'(?i)\b(with|for)\b.*$', ' ', t)
    t = re.sub(r'(?i)\b(pack of \d+|\d+\s?(ml|g|gm|gms|oz))\b', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return ' '.join(t.split()[:8])


def title_tokens(title):
    """Distinctive words of a product title (brand + key active)."""
    words = re.findall(r"[a-z0-9]+", clean_title(title).lower())
    return [w for w in words if len(w) >= 4 and w not in GENERIC_WORDS and not w.isdigit()]


def brand_of(title):
    """First meaningful word of the title, which is almost always the brand."""
    ws = [w for w in re.findall(r"[a-z0-9]+", clean_title(title).lower()) if w != 'the']
    return ws[0] if ws else ''


def is_relevant(result, tokens, brand=''):
    """Whole-word match (so 'derma' doesn't match 'dermaquest'); a 4+ letter brand must appear."""
    blob = f"{result['title']} {result['link']} {result['snippet']}".lower()
    words = set(re.findall(r"[a-z0-9]+", blob))
    if len(brand) >= 4 and brand not in words:
        return False
    need = 1 if len(tokens) <= 1 else 2
    return sum(1 for t in tokens if t in words) >= need


def percentages(text):
    """Concentrations mentioned in a text, e.g. 'Niacinamide 10% Serum' -> {'10'}."""
    return set(re.findall(r"(\d+(?:\.\d+)?)\s*%", text))


def variant_conflict(result, query_title):
    """True if the page is about a different variant than the one asked for:
    a variant word the user didn't type (e.g. 'psoriasis'), or a different concentration (5% vs 10%)."""
    q = set(re.findall(r"[a-z]+", query_title.lower()))
    blob = set(re.findall(r"[a-z]+", f"{result['title']} {result['link']}".lower()))
    if (blob & VARIANT_WORDS) - q:
        return True
    wanted, found = percentages(query_title), percentages(result['title'])
    return bool(wanted and found and not (wanted & found))


def list_confidence(n, from_web):
    """How much to trust a score. Labels/pasted lists are complete; web-found lists may be partial."""
    if not from_web:
        return 'complete'
    if n >= 15:
        return 'good'
    if n >= 8:
        return 'partial'
    return 'low'


def source_confidence(n, sources):
    c = list_confidence(n, bool(sources))
    if sources and sources[0].get('note') and c == 'good':
        return 'partial'          # a closest-match variant page: treat the score as a best case
    return c


def norm_ingredient(i):
    """lowercase, drop (parentheses) and percentages like 10%"""
    return re.sub(r'\s+', ' ', re.sub(r'\([^)]*\)|\d+(\.\d+)?\s*%', ' ', i.lower())).strip()
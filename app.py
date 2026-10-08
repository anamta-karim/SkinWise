from flask import Flask, render_template, request, jsonify
import os
import json
import time
from ocr_utils import extract_ingredients
from serpapi_utils import price_compare, discover_products, search_snippets, fetch_ingredient_context, find_comma_lists, shopping_search, CATEGORY_QUERY, amazon_ingredient_candidates, brand_evidence, ingredient_pulse
from analysis_utils import terms_only
from analysis_utils import analyze_ingredients, calculate_safety_score
from rapidfuzz import process, fuzz
from groq import Groq
import torch
from analysis_utils import embedding_model
from sentence_transformers import util
from text_utils import (GENERIC_WORDS, VARIANT_WORDS, clean_title, title_tokens, brand_of, is_relevant,
                        variant_conflict, list_confidence, source_confidence, norm_ingredient as _norm)

from dotenv import load_dotenv
import os
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
print("Groq model:", GROQ_MODEL)
_orig_create = groq_client.chat.completions.create
os.makedirs('static/uploads', exist_ok=True)

def _create(*args, **kwargs):
    kwargs["model"] = GROQ_MODEL
    kwargs["max_tokens"] = kwargs.get("max_tokens", 300) + 700
    kwargs.setdefault("extra_body", {"reasoning_effort": "low"})
    r = _orig_create(*args, **kwargs)
    try:
        for ch in r.choices:
            if ch.message and ch.message.content:
                ch.message.content = ch.message.content.replace("**", "").replace("__", "")
    except Exception:
        pass
    return r

groq_client.chat.completions.create = _create

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

with open('data/products.json', 'r', encoding='utf-8') as f:
    PRODUCTS = json.load(f)

with open('data/brands.json', 'r', encoding='utf-8') as f:
    BRANDS = json.load(f)['brands']

print("✅ All data loaded successfully!")
print(f"   • Products: {sum(len(cat) for cat in PRODUCTS.values())} items")
print(f"   • Brands: {len(BRANDS)} brands")

print("⏳ Creating embeddings for all brands...")

brand_texts = []
brand_metadata = []

for brand in BRANDS:
    search_text = brand['name']
    if brand.get('aliases'):
        search_text += " " + " ".join(brand['aliases'])
    
    brand_texts.append(search_text)
    brand_metadata.append(brand)

brand_embeddings = embedding_model.encode(brand_texts, convert_to_tensor=True)

print(f"✅ Brand embeddings ready for {len(BRANDS)} brands")

def clean_ocr_with_groq(raw_ocr_text):
    """Use Groq to extract clean ingredient list from messy OCR text"""
    try:
        prompt = f"""You are an expert skincare ingredient extractor.
Extract ONLY the ingredient names as a clean Python list from this messy OCR text.
Return ONLY the list, nothing else. Example: ["water", "glycerin", "niacinamide"]

Text: {raw_ocr_text}"""

        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500
        )
        
        response_text = completion.choices[0].message.content.strip()

        import ast
        ingredients = ast.literal_eval(response_text)
        if isinstance(ingredients, list):
            return [str(ing).strip().lower() for ing in ingredients if ing]
        return []
        
    except Exception as e:
        print(f"⚠️ Groq cleaning failed: {e} → falling back to old method")
        return None
    
def generate_ai_explanation(ingredient_name, concern_level, skin_type=None):
    """Use Groq to generate natural, personalized explanation"""
    try:
        prompt = f"""You are a friendly skincare expert.
Explain in 1-2 short, simple sentences why '{ingredient_name}' is marked as '{concern_level}'.
Make it easy to understand. Mention skin type if given.

Skin type: {skin_type or 'general'}
Ingredient: {ingredient_name}
Concern level: {concern_level}

Reply in natural, helpful tone. Do not use technical jargon unless necessary."""

        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        explanation = completion.choices[0].message.content.strip()
        return explanation
        
    except Exception as e:
        print(f"⚠️ Groq explanation failed: {e}")
        return None

def generate_product_explanation(product, skin_type, concerns):
    """Generate short, concise personalized explanation"""
    try:
        concerns_str = ", ".join(concerns) if concerns else "general skincare"
        
        prompt = f"""You are a friendly skincare expert.
Write **only 1-2 short sentences** (maximum 2) explaining why "{product['name']}" by {product['brand']} 
is a good recommendation for {skin_type} skin concerned with {concerns_str}.

Keep it concise, natural and helpful. Do not write long paragraphs."""

        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"⚠️ Groq explanation failed: {e}")
        return f"Good match for your {skin_type} skin."

def generate_daily_routine(skin_type, concerns, user_products=None):
    """Generate a personalized morning + night routine"""
    try:
        concerns_str = ", ".join(concerns) if concerns else "general skincare"
        
        prompt = f"""You are an expert skincare routine planner.
Create a simple, realistic **Morning** and **Night** routine for someone with **{skin_type}** skin who has these concerns: **{concerns_str}**.

Important rules:
- Do NOT use any markdown formatting like ** or *.
- Do NOT use bold or italic.
- Suggest product types along with key helpful ingredients.
- Example: "Gentle niacinamide cleanser", "Salicylic acid serum", "Lightweight hyaluronic acid moisturizer"
- Keep routines to 3–5 steps each.
- Use logical order (cleanse → treat → moisturize → protect).
- Be friendly, practical and encouraging.

Return in this exact plain text format (no markdown):

Morning:
1. Product type with key ingredient - short reason
2. ...

Night:
1. Product type with key ingredient - short reason
2. ..."""

        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=450
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"⚠️ Groq routine generation failed: {e}")
        return "Unable to generate routine at the moment. Please try again."
    
def check_compatibility(products_or_ingredients):
    """Check for ingredient conflicts using Groq"""
    try:
        items_str = "\n".join([f"- {item}" for item in products_or_ingredients])
        
        prompt = f"""You are a skincare compatibility expert.
The user wants to use these products/ingredients together:

{items_str}

Analyze for dangerous combinations (e.g. Retinol + Vitamin C, AHAs/BHAs + Retinol, etc.).
Return in this exact format:

Compatibility: [Safe / Caution / High Risk]

Warnings:
- Warning 1: short explanation
- Warning 2: ...

Recommended Order:
1. Step...
2. Step...

Be honest, helpful, and concise."""

        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=400
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"⚠️ Compatibility check failed: {e}")
        return "Unable to check compatibility at the moment."
    
def generate_brand_analysis(brand_data):
    """Use Groq to generate a natural, friendly explanation about the brand"""
    try:
        name = brand_data.get('name', '')
        cruelty_free = brand_data.get('cruelty_free')
        vegan = brand_data.get('vegan')
        note = brand_data.get('note', '')
        
        status = []
        if cruelty_free is True:
            status.append("cruelty-free")
        elif cruelty_free is False:
            status.append("not cruelty-free")
        
        if vegan is True:
            status.append("fully vegan")
        elif vegan is False:
            status.append("not fully vegan")
        
        status_str = " and ".join(status) if status else "has mixed ethics status"
        
        prompt = f"""You are a friendly skincare ethics expert.
Write a **short and natural** 2-sentence explanation about "{name}".
Mention if it is cruelty-free and/or vegan, and add one useful insight from this note: {note}

Keep it warm, concise, and helpful. Do not repeat basic facts."""

        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=220
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"⚠️ Groq brand analysis failed: {e}")
        return None
    
def get_product_embedding(product):
    """Create a rich description and get its embedding"""
    description = f"{product['name']} by {product['brand']}. " \
                f"For {', '.join(product.get('skin_types', []))}. " \
                f"Helps with {', '.join(product.get('concerns', []))}. " \
                f"Key ingredients: {', '.join(product.get('key_ingredients', []))}"
    
    return embedding_model.encode(description, convert_to_tensor=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyzer')
def analyzer():
    return render_template('analyzer.html')

@app.route('/recommender')
def recommender():
    return render_template('recommender.html')

@app.route('/ethiscan')
def ethiscan():
    return render_template('ethiscan.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    skin_type = request.form.getlist('skin_type')
    image = request.files.get('image')
    manual_text = request.form.get('manual_text', '')
    product_name = request.form.get('product_name', '').strip()
    sources = []

    if image and image.filename:
        start_ocr = time.time()
        image_bytes = image.read()
        
        raw_ingredients = extract_ingredients(image_bytes)
        
        cleaned_ingredients = clean_ocr_with_groq(' '.join(raw_ingredients))
        
        if cleaned_ingredients is not None and len(cleaned_ingredients) > 0:
            ingredients = cleaned_ingredients
            print(f"✅ Groq AI cleaned {len(ingredients)} ingredients")
        else:
            ingredients = raw_ingredients
            print("⚠️ Groq failed → using old cleaning method")
        
        ocr_time = time.time() - start_ocr
        print(f"⏱️ OCR took {ocr_time:.2f} seconds")
    
    elif manual_text.strip():
        ingredients = [i.strip() for i in manual_text.split(',') if i.strip()]

    elif product_name:
        try:
            ingredients, sources = get_product_ingredients(product_name)
        except Exception as e:
            return jsonify({'error': f'Live search failed: {e}'}), 502
        if len(ingredients) < 3:
            return jsonify({'error': "Couldn't find a reliable ingredient list for that product online. Try the full product name, or paste the ingredients."}), 404

    else:
        ingredients = []

    try:
        start_analysis = time.time()
        skin_type_str = skin_type[0] if skin_type else None
        results = analyze_ingredients(ingredients, skin_type=skin_type_str)
        score = calculate_safety_score(results)

        for item in results.get('matched', []):
            if item.get('concern_level') in ['Caution', 'Avoid']:
                ai_exp = generate_ai_explanation(
                    item.get('ingredient_name', ''),
                    item.get('concern_level', ''),
                    skin_type_str
                )
                if ai_exp:
                    item['ai_explanation'] = ai_exp
                else:
                    item['ai_explanation'] = item.get('explanation', '')

        analysis_time = time.time() - start_analysis
        print(f"⏱️ Analysis + AI explanations took {analysis_time:.2f} seconds")

        safe_results = []
        for r in results.get('matched', []):
            safe_results.append({
                'ingredient': r.get('ingredient_name', ''),
                'matched_name': r.get('ingredient_name', ''),
                'concern_level': r.get('concern_level', ''),
                'category': r.get('category', ''),
                'explanation': r.get('ai_explanation') or r.get('explanation', ''),
                'benefit': r.get('benefit', ''),
                'flagged': r.get('concern_level') in ['Caution', 'Avoid']
            })

        safe_flagged = []
        for r in results.get('flagged', []):
            safe_flagged.append({
                'ingredient': r.get('ingredient_name', ''),
                'matched_name': r.get('ingredient_name', ''),
                'concern_level': r.get('concern_level', ''),
                'explanation': r.get('ai_explanation') or r.get('explanation', ''),
                'skin_types_to_avoid': r.get('skin_types_to_avoid', '') or ''
            })

        return jsonify({
            'ingredients': ingredients,
            'results': safe_results,
            'score': score,
            'flagged': safe_flagged,
            'safe_count': results.get('safe_count', 0),
            'caution_count': results.get('caution_count', 0),
            'avoid_count': results.get('avoid_count', 0),
            'sources': sources,
            'confidence': source_confidence(len(ingredients), sources)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    skin_type = data.get('skin_type', '').lower()
    concerns = [c.lower() for c in data.get('concerns', [])]

    user_profile = f"{skin_type} skin"
    if concerns:
        user_profile += f" concerned with {', '.join(concerns)}"

    user_embedding = embedding_model.encode(user_profile, convert_to_tensor=True)

    recommendations = {}

    for category, products in PRODUCTS.items():
        scored_products = []
        
        for p in products:
            skin_match = skin_type in [s.lower() for s in p.get('skin_types', [])]
            concern_match = any(c in [x.lower() for x in p.get('concerns', [])] for c in concerns)
            
            if skin_match or (not concerns and skin_match):
                prod_embedding = get_product_embedding(p)
                similarity = util.cos_sim(user_embedding, prod_embedding)[0][0].item()
                
                final_score = (p.get('rating', 0) * 0.4) + (similarity * 0.6)
                
                p_copy = p.copy()
                p_copy['semantic_score'] = round(similarity, 3)
                p_copy['final_score'] = round(final_score, 3)
                scored_products.append(p_copy)

        scored_products.sort(key=lambda x: x['final_score'], reverse=True)
        
        top_products = scored_products[:5]
        
        for product in top_products:
            explanation = generate_product_explanation(product, skin_type, concerns)
            if explanation:
                product['ai_explanation'] = explanation

        recommendations[category] = top_products

    return jsonify(recommendations)

def check_brand_base():
    data = request.get_json()
    brand_input = data.get('brand_name', '').strip().lower()

    if not brand_input:
        return jsonify({'error': 'Please enter a brand name'}), 400

    for brand in BRANDS:
        name_lower = brand['name'].lower()
        aliases = [a.lower() for a in brand.get('aliases', [])]
        if brand_input == name_lower or brand_input in aliases:
            brand_data = brand.copy()
            ai_analysis = generate_brand_analysis(brand_data)
            if ai_analysis:
                brand_data['ai_explanation'] = ai_analysis
            return jsonify(brand_data)

    candidates = {}
    for b in BRANDS:
        candidates[b['name'].lower()] = b
        for a in b.get('aliases', []):
            candidates[a.lower()] = b
    match = process.extractOne(brand_input, list(candidates.keys()), scorer=fuzz.ratio)
    if match and match[1] >= 90:
        brand_data = candidates[match[0]].copy()
        ai_analysis = generate_brand_analysis(brand_data)
        if ai_analysis:
            brand_data['ai_explanation'] = ai_analysis
        return jsonify(brand_data)

    return jsonify({
        'name': brand_input.title(),
        'cruelty_free': None,
        'vegan': None,
        'note': "Not in our curated database. The live Google check below is based on search results only.",
        'source': "Live search only"
    })

@app.route('/api/routine', methods=['POST'])
def generate_routine():
    data = request.get_json()
    skin_type = data.get('skin_type', '').lower()
    concerns = [c.lower() for c in data.get('concerns', [])]

    if not skin_type:
        return jsonify({'error': 'Please select your skin type'}), 400

    routine = generate_daily_routine(skin_type, concerns)

    return jsonify({
        'skin_type': skin_type,
        'concerns': concerns,
        'routine': routine
    })

@app.route('/api/compatibility', methods=['POST'])
def check_compatibility_route():
    data = request.get_json()
    items = data.get('items', [])

    if not items or len(items) < 2:
        return jsonify({'error': 'Please provide at least 2 products or ingredients'}), 400

    result = check_compatibility(items)

    return jsonify({
        'items': items,
        'compatibility_result': result
    })

@app.route('/api/prices', methods=['POST'])
def api_prices():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'name required'}), 400
    return jsonify(price_compare(data.get('brand', ''), name))

import ast, re

def extract_ingredients_from_snippets(title, snippets):
    text = "\n".join(f"- {s['title']}: {s['snippet']}" for s in snippets if s.get('snippet'))
    if not text:
        return []
    prompt = f"""Below is text from web pages about the skincare product "{title}".
Extract the product's ingredient list (INCI names) ONLY if it is explicitly written in the text.
Do not guess or add ingredients from your own knowledge. If no ingredient list appears, return [].
Return ONLY a Python list of lowercase strings.

Text:
{text}"""
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=500)
        raw = completion.choices[0].message.content.strip()
        m = re.search(r"\[.*\]", raw, re.S)
        items = ast.literal_eval(m.group(0)) if m else []
        items = [str(i).strip().lower() for i in items if str(i).strip()]
        text_l = text.lower()
        kept = [i for i in items if i in text_l]
        print(f"   model returned {len(items)} ingredients, {len(kept)} found in the source text")
        return kept
    except Exception as e:
        print(f"⚠️ Ingredient extraction failed: {e}")
        return []

PREFERRED_SITES = ("incidecoder.com", "skincarisma.com", "minimalist.co", "nykaa.com", "purplle.com")
SKIP_SITES = ("amazon.", "flipkart.", "youtube.", "instagram.", "facebook.", "reddit.", "pinterest.", "taobao.", "inkeedecoder.", "joom.")
USE_AMAZON = False

TYPE_WORDS = {
    "face_wash": ("wash", "cleanser"),
    "moisturizer": ("moistur", "cream", "gel"),
    "serum_essence": ("serum", "essence"),
    "sunscreen": ("sunscreen", "spf"),
    "toner_mist": ("toner", "mist"),
    "scrub_exfoliator": ("scrub", "exfoliat", "peel"),
    "mask_peel": ("mask", "peel"),
}

def looks_like_inci(items):
    """A candidate list only counts if it mostly matches our ingredient database."""
    if len(items) < 6:
        return False
    hits = sum(1 for it in items
            if process.extractOne(_norm(it), terms_only, scorer=fuzz.ratio, score_cutoff=85))
    return hits / len(items) >= 0.35

def get_product_ingredients(title, retry=True):
    """Strict pass (exact-product pages), then a lenient pass over the same cached results that
    accepts other variants of the product, flagged as a closest match. Returns (ingredients, sources)."""
    short = clean_title(title)
    tokens = title_tokens(title)
    brand = brand_of(title)
    queries = (f"{short} skincare ingredients", f"{short} incidecoder")[: 2 if retry else 1]

    def scan(snippets, allow_variants):
        relevant = [s for s in snippets if is_relevant(s, tokens, brand)
                    and (allow_variants or not variant_conflict(s, short))]
        note = 'closest match, may be a different variant' if allow_variants else ''

        # 1) a valid-looking list inside a relevant snippet
        for s in relevant:
            for l in find_comma_lists(s['snippet']):
                if looks_like_inci(l):
                    return l, [{'title': s['title'], 'link': s['link'], 'note': note}]

        # 2) read the relevant pages (plain fetch, 0 credits, cached)
        def rank(s):
            return 0 if any(d in s['link'] for d in PREFERRED_SITES) else 1
        candidates = [s for s in sorted(relevant, key=rank)
                    if s['link'] and not any(d in s['link'] for d in SKIP_SITES)]
        for s in candidates[:3]:
            page = fetch_ingredient_context(s['link'])
            valid = [l for l in page['lists'] if looks_like_inci(l)]
            print(f"   page {s['link'][:50]} -> {len(page['lists'])} candidate lists, {len(valid)} valid")
            if valid:
                return max(valid, key=len), [{'title': s['title'], 'link': s['link'], 'note': note}]
            if page['windows']:
                found = extract_ingredients_from_snippets(short, [{'title': s['title'], 'snippet': page['windows']}])
                if len(found) >= 3:
                    return found, [{'title': s['title'], 'link': s['link'], 'note': note}]
        return None

    fetched = []
    for query in queries:
        snippets, _ = search_snippets(query)
        fetched.append(snippets)
        print(f"🔎 '{query}' -> {len(snippets)} results")
        hit = scan(snippets, False)
        if hit:
            return hit

    for snippets in fetched:
        hit = scan(snippets, True)
        if hit:
            print("   ↪ no exact-product page had a list; using a closest-match variant page")
            return hit
    return [], []

@app.route('/api/recommend_live', methods=['POST'])
def recommend_live():
    d = request.get_json() or {}
    skin_type = (d.get('skin_type') or 'sensitive').lower()
    concerns = [c.lower() for c in d.get('concerns', [])]
    categories = d.get('categories') or ['serum_essence', 'moisturizer', 'sunscreen']
    out = {}
    for cat in categories:
        try:
            out[cat] = discover_products(cat, skin_type, concerns)
        except Exception as e:
            print(f"⚠️ SerpApi discovery failed for {cat}: {e}")
            out[cat] = []
    return jsonify(out)

@app.route('/api/live_analyze', methods=['POST'])
def live_analyze():
    d = request.get_json() or {}
    title = d.get('title', '').strip()
    skin_type = (d.get('skin_type') or '').lower() or None
    if not title:
        return jsonify({'error': 'title required'}), 400
    try:
        ingredients, sources = get_product_ingredients(title)
        if len(ingredients) < 3:
            return jsonify({'found': False, 'sources': sources})
        results = analyze_ingredients(ingredients, skin_type=skin_type)
        return jsonify({
            'found': True,
            'ingredients': ingredients,
            'score': calculate_safety_score(results),
            'confidence': source_confidence(len(ingredients), sources),
            'flagged': [{'ingredient': r.get('ingredient_name', ''),
                        'concern_level': r.get('concern_level', ''),
                        'explanation': r.get('explanation', '')} for r in results['flagged']],
            'sources': sources
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'found': False, 'error': str(e)}), 500

ACTIVE_CATEGORIES = {'beneficial', 'antioxidant', 'aha', 'bha', 'pha', 'retinoid', 'peptide', 'uv filter'}
FILLERS = {'water', 'aqua', 'eau', 'glycerin', 'butylene glycol', 'propylene glycol', 'propanediol',
        'pentylene glycol', 'caprylyl glycol', 'ethylhexylglycerin'}
CORE_STOP = GENERIC_WORDS | {'extract', 'root', 'leaf', 'water', 'oil', 'flower', 'juice', 'seed', 'fruit'}

def active_list(ings):
    """Ordered, de-duplicated key actives: beneficial-type ingredients from our database, minus fillers."""
    out = []
    for r in analyze_ingredients(ings)['matched']:
        name = str(r['ingredient_name']).lower().split('(')[0].strip()
        if (str(r.get('category', '')).strip().lower() in ACTIVE_CATEGORIES
                and name not in FILLERS and name not in out):
            out.append(name)
    return out

def active_overlap(orig_ings, other_ings):
    a, b = active_list(orig_ings), set(active_list(other_ings))
    if not a:
        return 0, []
    shared = [x for x in a if x in b]
    return round(len(shared) / len(a) * 100), shared

def generic_query(title, product_type='', actives=()):
    """Brand-free shopping phrase grounded in the product's real actives (no guessing)."""
    short = clean_title(title)
    type_hint = f' It is a {product_type}.' if product_type else ''
    act_hint = (f' Its key active ingredients are: {", ".join(actives[:3])}. Use ONLY these actives, do not add others.'
                if actives else ' Do not guess any active ingredients; use only the product type.')
    try:
        c = groq_client.chat.completions.create(
            model=GROQ_MODEL, temperature=0.0, max_tokens=60,
            messages=[{"role": "user", "content":
                f'Rewrite "{short}" as a generic shopping search phrase with NO brand name, max 6 words, '
                f'keeping the product type and the concentration if the title states it.{type_hint}{act_hint} '
                f'Return only the phrase.'}])
        phrase = c.choices[0].message.content.strip().strip('"')
        if phrase:
            return phrase
    except Exception as e:
        print(f"⚠️ generic_query failed: {e}")
    return ' '.join(short.split()[1:]) or short

@app.route('/api/dupes', methods=['POST'])
def api_dupes():
    d = request.get_json() or {}
    title = d.get('title', '').strip()
    price = d.get('price_value')
    category = d.get('category', '')
    skin_type = (d.get('skin_type') or '').lower() or None
    if not title or not price:
        return jsonify({'error': 'title and price required'}), 400
    try:
        try:
            orig_ings, _ = get_product_ingredients(title)
        except Exception as e:
            print(f"⚠️ original lookup failed: {e}")
            return jsonify({'error': 'The search service timed out. Please try again in a moment.'}), 503
        if len(orig_ings) < 6:
            return jsonify({'error': "Couldn't read enough of this product's ingredients to compare dupes."}), 404
        actives = active_list(orig_ings)
        if not actives:
            return jsonify({'error': "This product's key actives aren't in our ingredient database, so I can't judge dupes reliably."}), 404

        phrase = generic_query(title, CATEGORY_QUERY.get(category, ''), actives)
        offers, _ = shopping_search(phrase)
        print(f"🧬 actives {actives[:3]} -> search '{phrase}' -> {len(offers)} offers")

        brand_words = [w for w in clean_title(title).lower().split() if w != 'the']
        brand = brand_words[0] if brand_words else ''
        core = [w for a in actives[:2] for w in a.split() if len(w) > 3 and w not in CORE_STOP][:3]
        type_words = TYPE_WORDS.get(category, ())
        BAD = ("hair", "shampoo", "conditioner", "body", "lip ", "foot", "hand ")

        cands, seen = [], set()
        for o in offers:
            t = o['title'].lower()
            if o['price_value'] is None or o['price_value'] >= float(price) * 0.9:
                continue
            if '₹' not in (o['price'] or ''):
                continue
            if (brand and brand in t) or t[:40] in seen:
                continue
            if core and not any(w in t for w in core):
                continue
            if type_words and not any(w in t for w in type_words):
                continue
            if any(b in t for b in BAD):
                continue
            seen.add(t[:40])
            cands.append(o)
        cands.sort(key=lambda o: (o['rating'] or 0) * ((o['reviews'] or 0) ** 0.5), reverse=True)

        verified, similar = [], []
        for i, o in enumerate(cands[:5]):
            base = {'title': o['title'], 'merchant': o['merchant'], 'price': o['price'],
                    'link': o['link'], 'thumbnail': o['thumbnail'],
                    'saving': int(float(price) - o['price_value'])}
            ings = []
            if i < 2:
                try:
                    ings, _ = get_product_ingredients(o['title'], retry=False)
                except Exception as e:
                    print(f"   ⚠️ couldn't read '{o['title'][:40]}': {e}")
            if len(ings) >= 6:
                pct, shared = active_overlap(orig_ings, ings)
                headline = any(w in ing for ing in ings for w in core)
                print(f"   '{o['title'][:40]}': {pct}% active overlap {shared} (headline active: {headline})")
                if pct >= 30 or (headline and pct >= 15):
                    res = analyze_ingredients(ings, skin_type=skin_type)
                    verified.append({**base, 'overlap': pct, 'shared': shared[:4],
                                    'safety': calculate_safety_score(res), 'n_ingredients': len(ings)})
                continue
            similar.append(base)
        verified.sort(key=lambda x: (x['overlap'], x['safety']), reverse=True)
        return jsonify({'phrase': phrase, 'dupes': verified, 'similar': similar[:3]})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def summarize_evidence(name, web, db_payload):
    lines = "\n".join(f"- [{r['trusted'] or 'other site'}] {r['title']}: {r['snippet']}" for r in web)
    db_line = ""
    if db_payload.get('cruelty_free') is not None:
        db_line = f"Our database says: cruelty-free={db_payload['cruelty_free']}, vegan={db_payload.get('vegan')}. "
    prompt = (f'Brand: {name}. {db_line}Using ONLY the search results below, write at most 2 short sentences: '
            f'does an independent source (PETA, Leaping Bunny, Cruelty Free Kitty, Vegan Society) list this brand as '
            f'cruelty-free, and does any result raise doubts (e.g. animal testing, sales in mainland China)? '
            f'If the results do not clearly say, say that plainly. Do not invent facts.\n\nResults:\n{lines}')
    try:
        c = groq_client.chat.completions.create(model=GROQ_MODEL, temperature=0.0, max_tokens=200,
                                                messages=[{"role": "user", "content": prompt}])
        return c.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ evidence summary failed: {e}")
        return ''

@app.route('/api/check_brand', methods=['POST'])
def check_brand():
    result = check_brand_base()
    if isinstance(result, tuple):
        return result
    payload = result.get_json()
    try:
        ev = brand_evidence(payload.get('name', ''))
        payload['evidence'] = ev
        if ev['web']:
            payload['evidence_summary'] = summarize_evidence(payload['name'], ev['web'], payload)
    except Exception as e:
        print(f"⚠️ evidence lookup failed: {e}")
    return jsonify(payload)

@app.route('/trends')
def trends_page():
    return render_template('trends.html')

@app.route('/api/trends', methods=['POST'])
def api_trends():
    d = request.get_json() or {}
    ings = [str(i).strip().lower() for i in d.get('ingredients', []) if str(i).strip()][:5]
    if len(ings) < 2:
        return jsonify({'error': 'Pick at least 2 ingredients to compare.'}), 400
    try:
        return jsonify(ingredient_pulse(ings))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Couldn't load Google Trends right now: {e}"}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
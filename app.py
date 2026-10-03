from flask import Flask, render_template, request, jsonify
import os
import json
import time
from ocr_utils import extract_ingredients
from serpapi_utils import price_compare, discover_products, search_snippets, fetch_ingredient_context, find_comma_lists
from analysis_utils import terms_only
from analysis_utils import analyze_ingredients, calculate_safety_score
from rapidfuzz import process, fuzz
from groq import Groq
import torch
from analysis_utils import embedding_model
from sentence_transformers import util

from dotenv import load_dotenv
import os
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
print("Groq model:", GROQ_MODEL)
_orig_create = groq_client.chat.completions.create
def _create(*args, **kwargs):
    kwargs["model"] = GROQ_MODEL
    kwargs["max_tokens"] = kwargs.get("max_tokens", 300) + 700
    kwargs.setdefault("extra_body", {"reasoning_effort": "low"})
    return _orig_create(*args, **kwargs)
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
            'sources': sources
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

@app.route('/api/check_brand', methods=['POST'])
def check_brand():
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

    query_embedding = embedding_model.encode(brand_input, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, brand_embeddings)[0]
    
    best_score, best_idx = torch.max(similarities, dim=0)
    best_score = float(best_score)

    if best_score > 0.55:
        matched_brand = brand_metadata[best_idx.item()].copy()
        ai_analysis = generate_brand_analysis(matched_brand)
        if ai_analysis:
            matched_brand['ai_explanation'] = ai_analysis
        return jsonify(matched_brand)

    brand_names = [b['name'].lower() for b in BRANDS]
    match = process.extractOne(brand_input, brand_names, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= 80:
        matched_brand = next((b for b in BRANDS if b['name'].lower() == match[0]), None)
        if matched_brand:
            brand_data = matched_brand.copy()
            ai_analysis = generate_brand_analysis(brand_data)
            if ai_analysis:
                brand_data['ai_explanation'] = ai_analysis
            return jsonify(brand_data)

    return jsonify({
        'name': brand_input.title(),
        'cruelty_free': None,
        'vegan': None,
        'note': "Brand not found in our database. Try searching on Cruelty-Free Kitty or PETA directly!",
        'source': "Not found"
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
        kept = [i for i in items if i in text_l]   # drop anything not literally in the source
        print(f"   model returned {len(items)} ingredients, {len(kept)} found in the source text")
        return kept
    except Exception as e:
        print(f"⚠️ Ingredient extraction failed: {e}")
        return []

PREFERRED_SITES = ("incidecoder.com", "skincarisma.com", "minimalist.co", "nykaa.com", "purplle.com")
SKIP_SITES = ("amazon.", "flipkart.", "youtube.", "instagram.", "facebook.", "reddit.", "pinterest.", "taobao.")

def clean_title(title):
    t = title.split('|')[0]
    t = re.sub(r'\(.*?\)', ' ', t)
    t = t.split(',')[0]
    t = re.sub(r'(?i)\b(with|for)\b.*$', ' ', t)
    t = re.sub(r'(?i)\b(pack of \d+|\d+\s?(ml|g|gm|gms|oz))\b', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return ' '.join(t.split()[:8])

def _norm(i):
    """lowercase, drop (parentheses) and percentages like 10%"""
    return re.sub(r'\s+', ' ', re.sub(r'\([^)]*\)|\d+(\.\d+)?\s*%', ' ', i.lower())).strip()

def looks_like_inci(items):
    """A candidate list only counts if it mostly matches our ingredient database."""
    if len(items) < 6:
        return False
    hits = sum(1 for it in items
            if process.extractOne(_norm(it), terms_only, scorer=fuzz.ratio, score_cutoff=85))
    return hits / len(items) >= 0.35

def get_product_ingredients(title):
    """Free list-scan on snippets, then whole pages, then AI extraction as last resort."""
    short = clean_title(title)
    sources = []
    for query in (f"{short} ingredients", f"{short} incidecoder ingredients"):
        snippets, _ = search_snippets(query)
        sources = [{'title': s['title'], 'link': s['link']} for s in snippets[:3]]
        print(f"🔎 '{query}' -> {[s['link'][:50] for s in snippets]}")

        # 1) does any snippet already contain a valid-looking list?
        for l in find_comma_lists(" | ".join(s['snippet'] for s in snippets)):
            if looks_like_inci(l):
                print("   ✅ valid list found in snippets")
                return l, sources

        # 2) read the pages themselves
        def rank(s):
            return 0 if any(d in s['link'] for d in PREFERRED_SITES) else 1
        candidates = [s for s in sorted(snippets, key=rank)
                    if s['link'] and not any(d in s['link'] for d in SKIP_SITES)]
        for s in candidates[:4]:
            page = fetch_ingredient_context(s['link'])
            valid = [l for l in page['lists'] if looks_like_inci(l)]
            print(f"   page {s['link'][:50]} -> {len(page['lists'])} candidate lists, {len(valid)} valid")
            if valid:
                return max(valid, key=len), [{'title': s['title'], 'link': s['link']}]
            if page['windows']:
                found = extract_ingredients_from_snippets(short, [{'title': s['title'], 'snippet': page['windows']}])
                if len(found) >= 3:
                    return found, [{'title': s['title'], 'link': s['link']}]
    return [], sources

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
            'flagged': [{'ingredient': r.get('ingredient_name', ''),
                        'concern_level': r.get('concern_level', ''),
                        'explanation': r.get('explanation', '')} for r in results['flagged']],
            'sources': sources
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'found': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
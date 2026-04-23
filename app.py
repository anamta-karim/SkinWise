from flask import Flask, render_template, request, jsonify
import os
import json
from ocr_utils import extract_ingredients
from analysis_utils import analyze_ingredients, calculate_safety_score
from rapidfuzz import process, fuzz

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ====================== LOAD ALL DATA ======================
# Products for Recommender
with open('data/products.json', 'r', encoding='utf-8') as f:
    PRODUCTS = json.load(f)

# Brands for EthiScan
with open('data/brands.json', 'r', encoding='utf-8') as f:
    BRANDS = json.load(f)['brands']

print("✅ All data loaded successfully!")
print(f"   • Products: {sum(len(cat) for cat in PRODUCTS.values())} items")
print(f"   • Brands: {len(BRANDS)} brands")

# ====================== ROUTES ======================

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

# ====================== API ENDPOINTS ======================

@app.route('/api/analyze', methods=['POST'])
def analyze():
    skin_type = request.form.getlist('skin_type')
    image = request.files.get('image')
    manual_text = request.form.get('manual_text', '')

    if image and image.filename:
        path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(path)
        ingredients = extract_ingredients(path)
    else:
        ingredients = [i.strip() for i in manual_text.split(',') if i.strip()]

    # Run analysis
    results = analyze_ingredients(ingredients, skin_type=skin_type)
    score = calculate_safety_score(results)

    # Convert pandas dicts to plain dicts for JSON
    safe_results = []
    for r in results.get('matched', []):
        safe_results.append({
            'ingredient': r.get('ingredient_name', ''),
            'matched_name': r.get('ingredient_name', ''),
            'concern_level': r.get('concern_level', ''),
            'category': r.get('category', ''),
            'explanation': r.get('explanation', ''),
            'benefit': r.get('benefit', ''),
            'flagged': r.get('concern_level') in ['Caution', 'Avoid']
        })

    return jsonify({
        'ingredients': ingredients,
        'results': safe_results,
        'score': score,
        'flagged': results.get('flagged', []),
        'safe_count': results.get('safe_count', 0),
        'caution_count': results.get('caution_count', 0),
        'avoid_count': results.get('avoid_count', 0)
    })

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    skin_type = data.get('skin_type', '').lower()
    concerns = [c.lower() for c in data.get('concerns', [])]

    recommendations = {}

    for category, products in PRODUCTS.items():
        matched = []
        for p in products:
            skin_match = skin_type in [s.lower() for s in p.get('skin_types', [])]
            concern_match = any(c in [x.lower() for x in p.get('concerns', [])] for c in concerns)
            if skin_match and (not concerns or concern_match):
                matched.append(p)
        matched.sort(key=lambda x: x.get('rating', 0), reverse=True)
        recommendations[category] = matched[:5]

    return jsonify(recommendations)

@app.route('/api/check_brand', methods=['POST'])
def check_brand():
    data = request.get_json()
    brand_input = data.get('brand_name', '').strip().lower()

    if not brand_input:
        return jsonify({'error': 'Please enter a brand name'}), 400

    # 1. Exact match (including aliases)
    for brand in BRANDS:
        name_lower = brand['name'].lower()
        aliases = [a.lower() for a in brand.get('aliases', [])]
        if brand_input == name_lower or brand_input in aliases:
            return jsonify(brand)

    # 2. Fuzzy match
    brand_names = [b['name'].lower() for b in BRANDS]
    match = process.extractOne(brand_input, brand_names, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= 80:
        matched_brand = next((b for b in BRANDS if b['name'].lower() == match[0]), None)
        if matched_brand:
            return jsonify(matched_brand)

    # 3. Not found
    return jsonify({
        'name': brand_input.title(),
        'cruelty_free': None,
        'vegan': None,
        'note': "Brand not found in our database. Try searching on Cruelty-Free Kitty or PETA directly!",
        'source': "Not found"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
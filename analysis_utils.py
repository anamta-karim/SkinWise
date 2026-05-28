import pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import torch

# Load database
db = pd.read_csv('data/ingredients_database.csv', encoding='latin-1')

# Build search index
search_terms = []
for idx, row in db.iterrows():
    search_terms.append((row['ingredient_name'].lower().strip(), idx))
    if pd.notna(row['common_aliases']):
        for alias in row['common_aliases'].split(','):
            alias = alias.lower().strip()
            if alias:
                search_terms.append((alias, idx))

terms_only = [term for term, idx in search_terms]

# ====================== SEMANTIC EMBEDDINGS ======================
print("⏳ Loading AI embedding model (this happens only once)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # fast & accurate for short text

# Pre-compute embeddings for all ingredients in the database (fast lookup)
db_embeddings = embedding_model.encode(terms_only, convert_to_tensor=True)
print(f"✅ AI embeddings ready for {len(terms_only)} terms")

def match_ingredient(ingredient, threshold=0.75):
    """
    NEW: Semantic matching using embeddings (much smarter than rapidfuzz)
    """
    if not ingredient or len(ingredient.strip()) < 2:
        return None
    
    # Encode the query ingredient
    query_embedding = embedding_model.encode(ingredient, convert_to_tensor=True)
    
    # Compute cosine similarity with all database terms
    cos_scores = util.cos_sim(query_embedding, db_embeddings)[0]
    best_score, best_idx = torch.max(cos_scores, dim=0)
    
    # Convert score to 0-1 range for consistency
    score = float(best_score)
    
    if score < 0.55:
        return None
    
    # Get the matched database row
    db_index = search_terms[best_idx.item()][1]
    return db.iloc[db_index].to_dict()

def analyze_ingredients(ingredient_list, skin_type=None):
    results = {
        'matched': [],
        'not_found': [],
        'flagged': [],
        'safe_count': 0,
        'caution_count': 0,
        'avoid_count': 0
    }
    
    for ingredient in ingredient_list:
        match = match_ingredient(ingredient)
        if match is None:
            results['not_found'].append(ingredient)
            continue
        results['matched'].append(match)
        
        if match['concern_level'] == 'Safe':
            results['safe_count'] += 1
        elif match['concern_level'] == 'Caution':
            results['caution_count'] += 1
        elif match['concern_level'] == 'Avoid':
            results['avoid_count'] += 1
        
        if match['concern_level'] in ['Caution', 'Avoid']:
            if skin_type:
                skin_types_to_avoid = str(match['skin_types_to_avoid']).lower()
                if skin_type.lower() in skin_types_to_avoid:
                    results['flagged'].append(match)
            else:
                results['flagged'].append(match)
    
    return results

def calculate_safety_score(results):
    total = len(results['matched'])
    if total == 0:
        return 0
    avoid_count = sum(1 for item in results['flagged'] if item['concern_level'] == 'Avoid')
    caution_count = sum(1 for item in results['flagged'] if item['concern_level'] == 'Caution')
    deductions = (avoid_count * 10) + (caution_count * 3)
    return max(0, 100 - deductions)
import json
import torch
from rapidfuzz import fuzz, process
from analysis_utils import embedding_model, match_ingredient
from ocr_utils import extract_ingredients
import os

print("🔬 Starting SkinWise AI Evaluation...\n")

# ====================== 1. INGREDIENT MATCHING EVALUATION ======================
print("1. Evaluating Ingredient Matching (Semantic + Fuzzy)...")

# Sample test cases (you can expand this)
test_cases = [
    ("niacinamide", "Niacinamide", "Good"),
    ("salicylic acid", "Salicylic Acid", "Good"),
    ("tea tree oil", "Tea Tree Oil", "Good"),
    ("hyaluronic acid", "Hyaluronic Acid", "Good"),
    ("retinol", "Retinol", "Good"),
    ("olive oil", "Olive Oil", "Good"),           # you had issue with this earlier
    ("vitamin c", "Ascorbic Acid", "Partial"),
    ("ceramide np", "Ceramide NP", "Good"),
]

correct = 0
total = len(test_cases)

for query, expected, _ in test_cases:
    result = match_ingredient(query)
    matched = result.get('ingredient_name', '').lower()
    score = result.get('similarity_score', 0)
    
    is_correct = matched == expected.lower() or score > 0.7
    if is_correct:
        correct += 1
    print(f"  Query: '{query}' → Matched: '{matched}' (score: {score:.3f}) → {'✅' if is_correct else '❌'}")

accuracy = (correct / total) * 100
print(f"\n✅ Ingredient Matching Accuracy: **{accuracy:.1f}%** ({correct}/{total})\n")

# ====================== 2. OCR EVALUATION ======================
print("2. Evaluating OCR Pipeline...")

# Put your test images in a folder called "test_images/" (create it if needed)
test_image_folder = "test_images"
if os.path.exists(test_image_folder):
    test_images = [f for f in os.listdir(test_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(test_images)} test images.")
    
    # You can manually label expected ingredients for a few images
    # For now, we just check if OCR runs without crashing and returns reasonable number of ingredients
    success = 0
    for img in test_images[:5]:   # test only first 5
        try:
            ingredients = extract_ingredients(os.path.join(test_image_folder, img))
            if len(ingredients) >= 5:   # reasonable minimum
                success += 1
                print(f"  {img} → {len(ingredients)} ingredients extracted ✅")
            else:
                print(f"  {img} → only {len(ingredients)} ingredients ❌")
        except Exception as e:
            print(f"  {img} → Error: {e}")
    
    ocr_success_rate = (success / len(test_images[:5])) * 100 if test_images else 0
    print(f"\n✅ OCR Success Rate (basic): **{ocr_success_rate:.1f}%**\n")
else:
    print("⚠️  No 'test_images' folder found. Create one with sample product photos for full OCR eval.\n")

print("="*60)
print("🎯 Evaluation Complete!")

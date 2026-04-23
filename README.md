# SkinWise 🧴

**AI-Powered Skincare Ingredient Analyzer & Recommender**

A full-stack web application that helps users analyze skincare product ingredients, get safety scores, and receive personalized product recommendations based on their skin type and concerns.

---

## ✨ Features

### 📸 Ingredient Analyzer
- Upload product label image or paste ingredients manually
- AI-powered OCR extraction (EasyOCR)
- Safety scoring (0-100)
- Benefit aggregation (shows only benefits supported by 2+ ingredients)
- Flagged ingredients with explanations
- Auto-generated personalized recommendations

### ✨ Product Recommender
- Select skin type(s) and concerns
- Get top 5 personalized product recommendations per category
- 105+ curated products across 7 categories (face wash, moisturizer, serum, sunscreen, etc.)

### 🐰 EthiScan
- Check if any brand is cruelty-free and vegan
- Exact + fuzzy search on 100+ brands database
- Shows certifications, notes, and sources

---

## 🛠️ Tech Stack

- **Backend:** Flask, Python 3.10
- **Frontend:** HTML5, CSS3, JavaScript (Tailwind-inspired lavender theme)
- **OCR:** EasyOCR + custom post-processing
- **Matching:** rapidfuzz (fuzzy string matching)
- **Data:** 385-ingredient database, 105 curated products, 100+ brand ethics database

---

## 🚀 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/anamta-karim/SkinWise.git
cd SkinWise

# 2. Activate virtual environment
venv310\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py

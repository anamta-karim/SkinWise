---
title: SkinWise
emoji: 🌿
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

# SkinWise 🧴

**AI-Powered Skincare Intelligence Platform**

🚀 **Live Demo**: [huggingface.co/spaces/anamta-karim/SkinWise](https://huggingface.co/spaces/anamta-karim/SkinWise)

A full-stack web application that uses **OCR, semantic embeddings, and LLM reasoning** to help users analyze skincare ingredients, receive personalized product recommendations with complete daily routines, and verify brand ethics.

---

## ✨ Key Features

### 📸 **Ingredient Analyzer**
- Upload product label image or paste ingredients manually
- **EasyOCR** with custom post-processing + **Groq AI** for intelligent ingredient cleaning
- Semantic matching using **sentence-transformers** embeddings
- Personalized safety scoring (0-100) based on skin type
- AI-generated natural explanations for flagged ingredients (Caution / Avoid)

### ✨ **Personal Regimen**
- Multi-select skin type(s) and concerns
- Get top 5 personalized product recommendations per category
- Hybrid recommendation engine (rule-based + semantic ranking)
- **Groq-powered Daily & Night Routine Builder**
- 105+ curated products across 7 categories

### 🐰 **EthiScan**
- Check if any brand is cruelty-free and vegan
- Semantic brand search using embeddings
- **Groq AI** generates natural, detailed ethics explanations
- Shows certifications, notes, and sources

---

## 🧠 AI/ML Components

- **OCR Pipeline**: EasyOCR + Groq LLM for post-correction
- **Semantic Search**: `all-MiniLM-L6-v2` embeddings + cosine similarity
- **LLM Reasoning**: Groq (llama-3.1-8b-instant) for explanations, routine generation, and brand analysis
- **Hybrid Recommendation**: Rule-based filtering + semantic ranking

---

## 📊 Evaluation Results

- **Ingredient Matching Accuracy**: 100% on 8-query test set including synonym resolution (e.g. "vitamin c" → Ascorbic Acid)
- **Brand Semantic Search**: Handles partial names, typos, and Indian brand variations effectively
- **OCR Pipeline**: Successfully extracts 5+ ingredients from real product label images

Full evaluation script available in `evaluation.py`.

---

## 🛠️ Tech Stack

- **Backend**: Flask (Python 3.10)
- **AI/ML**: Groq LLM, sentence-transformers, EasyOCR, rapidfuzz
- **Frontend**: HTML, CSS, Vanilla JS (lavender UI)
- **Data**: JSON + CSV + in-memory embeddings

---

## 🚀 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/anamta-karim/SkinWise.git
cd SkinWise

# 2. Activate virtual environment
venv310\Scripts\activate

# 3. Create a .env file and add your Groq API key
echo GROQ_API_KEY=ygsk_AX7BmtWQeTUAXIPhwKbgWGdyb3FYEPO0C4PhxqMD9VO6SyU > .env

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
python app.py
```

---

## 📁 Project Structure
SkinWise/
├── app.py                        # Main Flask application + all routes
├── ocr_utils.py                  # OCR extraction + cleaning logic
├── analysis_utils.py             # Semantic matching, safety scoring, embeddings
├── evaluation.py                 # AI component evaluation
├── requirements.txt
├── Dockerfile
├── data/
│   ├── products.json             # 105 curated products
│   ├── brands.json               # 100+ brand ethics database
│   └── ingredients_database.csv  # 385+ ingredients with safety profiles
├── templates/
│   ├── index.html                # Landing page
│   ├── analyzer.html             # Ingredient Analyzer
│   ├── recommender.html          # Personal Regimen
│   └── ethiscan.html             # EthiScan
└── static/

---

## ⚠️ Limitations & Future Work

**Current Limitations**
- Product database is relatively small (105 items)
- Groq API calls are not cached, can feel slow on repeated requests
- No user accounts or persistent history
- Static JSON and CSV files instead of a proper database

**Planned Improvements**
- Migrate to a proper database for products, brands, and user history
- Add user authentication and saved analysis history
- Add Product Compatibility Checker for dangerous ingredient combinations
- Expand product and brand databases significantly
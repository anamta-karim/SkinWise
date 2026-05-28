import easyocr
import re
from PIL import Image
import io

reader = easyocr.Reader(['en'])

def extract_ingredients(input_data, confidence_threshold=0.3):
    if isinstance(input_data, (bytes, bytearray)):
        img_bytes = input_data
    else:
        # fallback for old path usage
        img = Image.open(input_data)
        if img.size[0] > 1200:
            img = img.resize((1200, int(1200 * img.size[1] / img.size[0])), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_bytes = buf.getvalue()

    # Fast OCR on bytes
    result = reader.readtext(img_bytes, detail=0, low_text=0.3, text_threshold=0.5,
                             width_ths=0.7, height_ths=0.7, paragraph=False)

    lines = result
    raw_text = ' '.join(lines)
    
    ocr_corrections = {
        '1oo': '100',
        '10o': '100',
        '0o': '00',
        'o0': '00',
        '2O': '20',
        '- ': '-',
        '_': '',
    }
    
    for wrong, correct in ocr_corrections.items():
        raw_text = raw_text.replace(wrong, correct)
    
    raw_text = raw_text.replace(';', ',')
    raw_text = raw_text.strip()
    
    if 'ingredient' in raw_text.lower():
        start = raw_text.lower().find('ingredient')
        start = raw_text.find(',', start)
        raw_text = raw_text[start+1:]
    
    ingredients = raw_text.split(',')
    
    cleaned = []
    for ingredient in ingredients:
        ingredient = ingredient.strip()
        ingredient = ingredient.strip('.')
        ingredient = re.sub(r'^\d+\s*', '', ingredient)
        ingredient = re.sub(r'\s+', ' ', ingredient).strip()
        ingredient = ingredient.lower()
        if len(ingredient) > 2:
            cleaned.append(ingredient)
    
    further_cleaned = []
    for ingredient in cleaned:
        if '.' in ingredient:
            parts = ingredient.split('.')
            for part in parts:
                part = part.strip()
                if len(part) > 2:
                    further_cleaned.append(part)
        else:
            further_cleaned.append(ingredient)
    
    final_ingredients = []
    for ingredient in further_cleaned:
        if 'ceramide' in ingredient and ingredient.count('ceramide') > 1:
            parts = re.split(r'(?<!^)(?=ceramide)', ingredient)
            for part in parts:
                part = part.strip()
                if len(part) > 2:
                    final_ingredients.append(part)
        else:
            final_ingredients.append(ingredient)
    
    return final_ingredients
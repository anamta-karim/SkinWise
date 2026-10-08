from text_utils import (brand_of, clean_title, is_relevant, list_confidence, norm_ingredient,
                        percentages, source_confidence, title_tokens, variant_conflict)


def result(title, link, snippet=""):
    return {"title": title, "link": link, "snippet": snippet}


# ---- clean_title / tokens / brand -------------------------------------------------------------
def test_clean_title_strips_marketing_text():
    messy = "Dot & Key Sunscreen SPF 50+, PA+++, UV Filters, Fragrance Free, 30 gm | chemical sunscreen"
    assert clean_title(messy) == "Dot & Key Sunscreen SPF 50+"


def test_clean_title_removes_pack_and_size_noise():
    assert clean_title("Minimalist Vitamin C 10% Face Serum for Glowing Skin 10 ml X 2 (Pack of 2)") \
        == "Minimalist Vitamin C 10% Face Serum"


def test_title_tokens_keep_brand_and_active_only():
    assert title_tokens("The Derma Co 2% Salicylic Acid Serum") == ["derma", "salicylic"]


def test_brand_skips_leading_the():
    assert brand_of("The Ordinary Glycolic Acid 7%") == "ordinary"
    assert brand_of("CeraVe Moisturising Cream") == "cerave"


# ---- relevance gate ---------------------------------------------------------------------------
def test_derma_does_not_match_dermaquest():
    title = "The Derma Co 2% Salicylic Acid Serum"
    wrong = result("DermaQuest SkinBrite Salicylic", "https://x.com/products/dermaquest-skinbrite", "salicylic acid serum")
    assert not is_relevant(wrong, title_tokens(title), brand_of(title))


def test_real_product_page_is_relevant():
    title = "The Derma Co 2% Salicylic Acid Serum"
    right = result("The Derma Co 2% Salicylic Acid Serum", "https://thedermaco.com/product/2-salicylic-acid-serum")
    assert is_relevant(right, title_tokens(title), brand_of(title))


def test_unrelated_word_overlap_is_rejected():
    title = "Plum 10% Niacinamide Serum"
    page = result("Plumber guide", "https://x.com/plumber", "niacinamide")
    assert not is_relevant(page, title_tokens(title), brand_of(title))


# ---- variant guard ----------------------------------------------------------------------------
def test_psoriasis_variant_is_flagged():
    page = result("CeraVe Psoriasis Moisturizing Cream", "https://skinsort.com/products/cerave/psoriasis-moisturizing-cream")
    assert variant_conflict(page, "CeraVe Moisturising Cream")


def test_plain_product_page_is_not_flagged():
    page = result("CeraVe Moisturizing Cream", "https://example.com/cerave-moisturizing-cream")
    assert not variant_conflict(page, "CeraVe Moisturising Cream")


def test_variant_word_in_query_is_allowed():
    page = result("Effaclar Duo+M", "https://example.com/effaclar-duo-m")
    assert not variant_conflict(page, "La Roche-Posay Effaclar Duo+M gel")


# ---- confidence -------------------------------------------------------------------------------
def test_confidence_levels():
    assert list_confidence(30, False) == "complete"       # pasted / photographed label
    assert list_confidence(20, True) == "good"
    assert list_confidence(10, True) == "partial"
    assert list_confidence(4, True) == "low"


def test_closest_variant_never_counts_as_good():
    variant = [{"note": "closest match, may be a different variant"}]
    assert source_confidence(30, variant) == "partial"
    assert source_confidence(30, [{"note": ""}]) == "good"


def test_norm_ingredient_drops_percentages_and_brackets():
    assert norm_ingredient("Niacinamide 10%") == "niacinamide"
    assert norm_ingredient("Zinc PCA (1%)") == "zinc pca"


# ---- concentration guard ----------------------------------------------------------------------
def test_percentages_are_extracted():
    assert percentages("Minimalist 10% Niacinamide Serum") == {"10"}
    assert percentages("Glycolic Acid 7% + Salicylic 1%") == {"7", "1"}
    assert percentages("CeraVe Moisturising Cream") == set()


def test_different_concentration_is_a_variant():
    page = result("Minimalist Niacinamide 5% Face Serum", "https://example.com/minimalist-niacinamide-serum")
    assert variant_conflict(page, "Minimalist 10% Niacinamide Serum")


def test_same_concentration_is_not_a_variant():
    page = result("Minimalist Niacinamide 10% Face Serum 30ml", "https://example.com/minimalist-niacinamide")
    assert not variant_conflict(page, "Minimalist 10% Niacinamide Serum")


def test_page_without_a_percentage_is_not_penalised():
    page = result("Minimalist Niacinamide Serum", "https://example.com/minimalist-niacinamide")
    assert not variant_conflict(page, "Minimalist 10% Niacinamide Serum")
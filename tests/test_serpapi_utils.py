import pytest
import requests

import serpapi_utils as su


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Every test gets an empty cache file and a fake API key; no real network calls."""
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr(su, "CACHE_FILE", str(tmp_path / "cache.json"))
    monkeypatch.setattr(su, "_cache", {})
    monkeypatch.setattr(su, "credits_used_this_session", 0)
    monkeypatch.setattr(su.time, "sleep", lambda s: None)


def serve(monkeypatch, handler):
    calls = []

    def fake_get(url, params=None, **kwargs):
        calls.append(params)
        return FakeResponse(handler(params))

    monkeypatch.setattr(su.requests, "get", fake_get)
    return calls


# ---- caching, retries, errors -----------------------------------------------------------------
def test_second_identical_search_is_served_from_cache(monkeypatch):
    calls = serve(monkeypatch, lambda p: {"shopping_results": []})
    su.serpapi_search("google_shopping", q="serum")
    _, from_cache = su.serpapi_search("google_shopping", q="serum")
    assert from_cache is True
    assert len(calls) == 1 and su.credits_used_this_session == 1


def test_timeout_is_retried_once(monkeypatch):
    attempts = {"n": 0}

    def flaky(url, params=None, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise requests.exceptions.ReadTimeout("slow")
        return FakeResponse({"ok": True})

    monkeypatch.setattr(su.requests, "get", flaky)
    data, _ = su.serpapi_search("google", q="x")
    assert data == {"ok": True} and attempts["n"] == 2


def test_api_error_message_is_raised(monkeypatch):
    serve(monkeypatch, lambda p: {"error": "Invalid API key"})
    with pytest.raises(RuntimeError, match="Invalid API key"):
        su.serpapi_search("google", q="x")


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY")
    with pytest.raises(RuntimeError, match="SERPAPI_API_KEY"):
        su.serpapi_search("google", q="x")


# ---- ingredient-list detection ----------------------------------------------------------------
def test_find_comma_lists_finds_inci_and_ignores_prose():
    text = ("Free shipping on all orders. Ingredients: Aqua, Niacinamide, Zinc PCA, Glycerin, "
            "Sodium Hyaluronate, Panthenol, Phenoxyethanol. Buy now.")
    lists = su.find_comma_lists(text)
    assert lists and lists[0][:3] == ["aqua", "niacinamide", "zinc pca"]
    assert su.find_comma_lists("Great serum, loved it, will buy again") == []


# ---- shopping ---------------------------------------------------------------------------------
SHOPPING = {"shopping_results": [
    {"title": "Minimalist 10% Niacinamide Serum", "source": "Nykaa", "price": "₹599", "extracted_price": 599,
     "rating": 4.5, "reviews": 1200, "link": "https://nykaa.com/p"},
    {"title": "Foreign Serum", "source": "Shop", "price": "$12", "extracted_price": 12, "rating": 5, "reviews": 10},
    {"title": "Lone Five Star Serum", "source": "Tiny", "price": "₹300", "extracted_price": 300, "rating": 5, "reviews": 2},
]}


def test_missing_store_link_falls_back_to_google_shopping_search(monkeypatch):
    serve(monkeypatch, lambda p: SHOPPING)
    offers, _ = su.shopping_search("niacinamide serum")
    foreign = next(o for o in offers if o["title"] == "Foreign Serum")
    assert foreign["link"].startswith("https://www.google.com/search?tbm=shop&q=")


def test_discover_products_drops_foreign_currency_and_ranks_by_trust(monkeypatch):
    serve(monkeypatch, lambda p: SHOPPING)
    picks = su.discover_products("serum_essence", "oily", ["acne"])
    titles = [p["title"] for p in picks]
    assert "Foreign Serum" not in titles
    assert titles[0] == "Minimalist 10% Niacinamide Serum"      # 4.5 stars x 1200 reviews beats 5 stars x 2


def test_price_compare_sorts_cheapest_first_and_drops_unrelated(monkeypatch):
    data = {"shopping_results": [
        {"title": "Cetaphil Gentle Skin Cleanser 250ml", "source": "A", "price": "₹459", "extracted_price": 459},
        {"title": "Cetaphil Gentle Skin Cleanser 125ml", "source": "B", "price": "₹330", "extracted_price": 330},
        {"title": "Random Sofa", "source": "C", "price": "₹9999", "extracted_price": 9999},
    ]}
    serve(monkeypatch, lambda p: data)
    result = su.price_compare("Cetaphil", "Gentle Skin Cleanser")
    assert [o["merchant"] for o in result["offers"]] == ["B", "A"]
    assert result["cheapest"]["merchant"] == "B"


# ---- brand evidence ---------------------------------------------------------------------------
def test_brand_evidence_puts_trusted_sources_first_and_filters_news(monkeypatch):
    def handler(p):
        if p["engine"] == "google":
            return {"organic_results": [
                {"title": "Plum Goodness review", "link": "https://blog.com/plum", "snippet": "plum skincare"},
                {"title": "Plum - cruelty free", "link": "https://crueltyfree.peta.org/plum/", "snippet": "plum is cruelty free"},
            ]}
        return {"news_results": [
            {"title": "Plum raises funds", "link": "https://n1", "source": {"name": "ET"}, "iso_date": "2026-09-01T10:00:00Z"},
            {"title": "Unrelated story", "link": "https://n2", "source": {"name": "X"}},
        ]}
    serve(monkeypatch, handler)
    ev = su.brand_evidence("Plum")
    assert ev["web"][0]["trusted"] == "PETA"
    assert [n["title"] for n in ev["news"]] == ["Plum raises funds"]


# ---- trends -----------------------------------------------------------------------------------
def test_ingredient_pulse_picks_the_fastest_riser(monkeypatch):
    def handler(p):
        if p["engine"] == "google_trends" and p.get("data_type") == "TIMESERIES":
            points = []
            for i in range(48):
                points.append({"date": f"Week {i}", "values": [
                    {"query": "niacinamide", "query_index": 0, "extracted_value": 40 + i},      # rising
                    {"query": "retinol", "query_index": 1, "extracted_value": 80 - i}]})       # falling
            return {"interest_over_time": {"timeline_data": points}}
        if p["engine"] == "google_trends":
            return {"related_queries": {"rising": [{"query": "niacinamide for pigmentation", "value": "+300%"}], "top": []}}
        return {"shopping_results": [{"title": "Niacinamide Serum", "source": "N", "price": "₹499", "extracted_price": 499}]}
    serve(monkeypatch, handler)
    pulse = su.ingredient_pulse(["niacinamide", "retinol"])
    assert pulse["focus"] == "niacinamide"
    momentum = {s["name"]: s["momentum"] for s in pulse["stats"]}
    assert momentum["niacinamide"] > 0 > momentum["retinol"]
    assert pulse["rising"][0]["query"] == "niacinamide for pigmentation"
    assert pulse["products"][0]["title"] == "Niacinamide Serum"
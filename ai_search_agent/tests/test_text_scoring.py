from search_agent.utils.text_scoring import build_query_terms, rank_products, tokenize


def test_tokenize_lowercases_and_keeps_hyphenated_terms():
    assert tokenize("Water-Resistant Jacket!") == ["water-resistant", "jacket"]


def test_build_query_terms_pulls_structured_fields_and_keywords():
    intent = {
        "category": "Outerwear",
        "occasion": "Wedding",
        "weather_attribute": "waterproof",
        "keywords": ["stylish", "coastal"],
    }
    terms = build_query_terms(intent)
    for expected in ["outerwear", "wedding", "waterproof", "stylish", "coastal"]:
        assert expected in terms


def test_rank_products_prefers_stronger_matches():
    products = [
        {
            "product_id": "PRD-1",
            "name": "Summit Parka",
            "description": "A waterproof, insulated parka for coastal weather.",
            "category": "Outerwear",
            "material": "Gore-Tex",
            "occasion": "Outdoor",
        },
        {
            "product_id": "PRD-2",
            "name": "Classic Tee",
            "description": "A soft cotton t-shirt for everyday casual wear.",
            "category": "Tops",
            "material": "Cotton",
            "occasion": "Casual",
        },
    ]
    intent = {"category": "Outerwear", "weather_attribute": "waterproof", "keywords": ["coastal"]}

    # min_score=-1 so the zero-scoring product (no matching terms) isn't
    # filtered out — we want to see both scores side by side here.
    results = rank_products(products, intent, top_k=2, min_score=-1)

    assert results[0]["product_id"] == "PRD-1"
    assert results[0]["_relevance_score"] > results[1]["_relevance_score"]


def test_rank_products_respects_top_k_and_min_score():
    products = [
        {"product_id": f"PRD-{i}", "name": "Item", "description": "generic item", "category": "Tops",
         "material": "Cotton", "occasion": "Casual"}
        for i in range(5)
    ]
    intent = {"category": "Tops"}

    results = rank_products(products, intent, top_k=2)
    assert len(results) <= 2

    # A query with no usable terms scores everything at 0, so min_score=0
    # (strictly greater-than) should filter every result out.
    empty_intent_results = rank_products(products, {}, top_k=5, min_score=0.0)
    assert empty_intent_results == []

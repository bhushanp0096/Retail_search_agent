import search_agent.nodes.stage2a_candidate_retrieval as stage2a


FAKE_PRODUCTS = [
    {
        "product_id": "PRD-1",
        "name": "Summit Parka",
        "description": "A waterproof, insulated parka built for coastal weather.",
        "category": "Outerwear",
        "material": "Gore-Tex",
        "occasion": "Wedding",
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


def test_fetch_candidates_node_returns_ranked_candidates(monkeypatch):
    monkeypatch.setattr(stage2a, "load_products", lambda: FAKE_PRODUCTS)

    state = {
        "raw_query": "waterproof jacket for a wedding",
        "intent": {"category": "Outerwear", "weather_attribute": "waterproof", "occasion": "Wedding"},
        "errors": [],
    }
    update = stage2a.fetch_candidates_node(state)
    print(f'test_fetch_candidates_node_returns_ranked_candidates update : {update}')
    assert "candidate_products" in update
    assert update["candidate_products"][0]["product_id"] == "PRD-1"
    assert "errors" not in update or update["errors"] == []


def test_fetch_candidates_node_handles_missing_intent_gracefully(monkeypatch):
    monkeypatch.setattr(stage2a, "load_products", lambda: FAKE_PRODUCTS)

    state = {"raw_query": "anything", "errors": []}  # no "intent" key at all
    update = stage2a.fetch_candidates_node(state)

    print(f'test_fetch_candidates_node_handles_missing_intent_gracefully update : {update}')
    # No usable query terms -> nothing scores above min_score -> empty list, not a crash.
    assert update["candidate_products"] == []


def test_fetch_candidates_node_reports_error_on_load_failure(monkeypatch):
    def _raise():
        raise FileNotFoundError("products.csv missing")

    monkeypatch.setattr(stage2a, "load_products", _raise)

    state = {"raw_query": "anything", "intent": {"category": "Tops"}, "errors": []}
    update = stage2a.fetch_candidates_node(state)

    print(f'test_fetch_candidates_node_reports_error_on_load_failure update : {update}')
    
    assert update["candidate_products"] == []
    assert len(update["errors"]) == 1
    assert "stage2a_candidate_retrieval" in update["errors"][0]

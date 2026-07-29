import search_agent.nodes.stage2b_profile_fetch as stage2b


FAKE_CUSTOMERS = {
    "CUST-001": {
        "customer_id": "CUST-001",
        "name": "Jamie Rivera",
        "size_profile": {"tops": "M", "bottoms": "32", "shoes": "10"},
        "style_preferences": ["minimalist", "waterproof"],
        "purchase_history": ["PRD-1001"],
    }
}


def test_fetch_customer_profile_node_returns_none_for_guest_session():
    state = {"raw_query": "anything", "customer_id": None, "errors": []}
    update = stage2b.fetch_customer_profile_node(state)
    assert update == {"customer_profile": None}


def test_fetch_customer_profile_node_returns_none_when_customer_id_absent():
    state = {"raw_query": "anything", "errors": []}  # no "customer_id" key at all
    update = stage2b.fetch_customer_profile_node(state)
    assert update == {"customer_profile": None}


def test_fetch_customer_profile_node_returns_profile_for_known_customer(monkeypatch):
    monkeypatch.setattr(stage2b, "load_customers", lambda: FAKE_CUSTOMERS)

    state = {"raw_query": "anything", "customer_id": "CUST-001", "errors": []}
    update = stage2b.fetch_customer_profile_node(state)

    print(f'test_fetch_customer_profile_node_returns_profile_for_known_customer update : {update}')
    
    assert update["customer_profile"]["name"] == "Jamie Rivera"


def test_fetch_customer_profile_node_returns_none_for_unknown_customer(monkeypatch):
    monkeypatch.setattr(stage2b, "load_customers", lambda: FAKE_CUSTOMERS)

    state = {"raw_query": "anything", "customer_id": "CUST-999", "errors": []}
    update = stage2b.fetch_customer_profile_node(state)

    print(f'test_fetch_customer_profile_node_returns_none_for_unknown_customer update : {update}')
    assert update == {"customer_profile": None}


def test_fetch_customer_profile_node_reports_error_on_load_failure(monkeypatch):
    def _raise():
        raise FileNotFoundError("customers.json missing")

    monkeypatch.setattr(stage2b, "load_customers", _raise)

    state = {"raw_query": "anything", "customer_id": "CUST-001", "errors": []}
    update = stage2b.fetch_customer_profile_node(state)

    print(f'test_fetch_customer_profile_node_reports_error_on_load_failure update : {update}')
    assert update["customer_profile"] is None
    assert len(update["errors"]) == 1
    assert "stage2b_profile_fetch" in update["errors"][0]

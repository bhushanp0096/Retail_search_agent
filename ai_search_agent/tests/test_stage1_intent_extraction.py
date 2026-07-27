"""
Unit tests for Stage 1 (intent extraction).

Uses a fake Anthropic client (no network calls, no API key required) so
these tests run in CI / offline. The fake client mimics just enough of the
`messages.create(...)` response shape (a list of content blocks with
`.type`, `.name`, `.input`) for `AnthropicStructuredClient` to operate on.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from search_agent.nodes.stage1_intent_extraction import extract_intent_node
from search_agent.schemas.intent import QueryIntent
from search_agent.utils.llm_client import AnthropicStructuredClient, StructuredExtractionError


def _tool_use_response(tool_name: str, tool_input: Dict[str, Any]) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", name=tool_name, input=tool_input)
    return SimpleNamespace(content=[block])


class FakeMessages:
    """Stand-in for `anthropic.Anthropic().messages`, returning a queued
    sequence of canned responses (one per `.create()` call), so a test can
    simulate "fails validation once, then succeeds" retry behavior."""

    def __init__(self, responses: List[SimpleNamespace]):
        self._responses = list(responses)
        self.call_count = 0

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.call_count += 1
        if not self._responses:
            raise AssertionError("FakeMessages.create called more times than responses were queued")
        return self._responses.pop(0)


class FakeAnthropicClient:
    def __init__(self, responses: List[SimpleNamespace]):
        self.messages = FakeMessages(responses)


@pytest.fixture
def make_client():
    def _make(responses: List[SimpleNamespace]) -> AnthropicStructuredClient:
        return AnthropicStructuredClient(client=FakeAnthropicClient(responses))

    return _make


def test_extract_intent_node_success(make_client):
    tool_input = {
        "raw_query": "waterproof jacket for an October coastal wedding",
        "category": "Outerwear",
        "occasion": "Wedding",
        "weather_attribute": "waterproof",
        "material": None,
        "size": None,
        "color": None,
        "max_price": None,
        "keywords": ["October", "coastal"],
    }
    client = make_client([_tool_use_response("extract_query_intent", tool_input)])

    state = {"raw_query": tool_input["raw_query"], "errors": []}
    update = extract_intent_node(state, client=client)

    assert update["errors"] == []
    assert update["intent"]["category"] == "Outerwear"
    assert update["intent"]["occasion"] == "Wedding"
    assert update["intent"]["weather_attribute"] == "waterproof"
    assert "coastal" in update["intent"]["keywords"]


def test_extract_intent_node_retries_on_invalid_enum_then_succeeds(make_client):
    # First response uses an invalid category (not in the allowed Literal set);
    # second (retry) response is valid. Confirms the self-correction loop works.
    bad_input = {
        "raw_query": "stylish blazer for a business dinner",
        "category": "Suits",  # not a valid Category value -> ValidationError
        "occasion": "Business",
    }
    good_input = {
        "raw_query": "stylish blazer for a business dinner",
        "category": "Tops",
        "occasion": "Business",
        "keywords": ["stylish", "blazer"],
    }
    client = make_client(
        [
            _tool_use_response("extract_query_intent", bad_input),
            _tool_use_response("extract_query_intent", good_input),
        ]
    )

    state = {"raw_query": bad_input["raw_query"], "errors": []}
    update = extract_intent_node(state, client=client)

    assert client._client.messages.call_count == 2
    assert update["errors"] == []
    assert update["intent"]["category"] == "Tops"


def test_extract_intent_node_falls_back_gracefully_after_exhausting_retries(make_client):
    bad_input = {"raw_query": "x", "category": "NotARealCategory"}
    # settings.max_retries defaults to 2 => 3 total attempts; queue 3 bad responses.
    client = make_client([_tool_use_response("extract_query_intent", bad_input) for _ in range(3)])

    state = {"raw_query": "some unparseable query", "errors": []}
    update = extract_intent_node(state, client=client)

    assert update["intent"]["raw_query"] == "some unparseable query"
    assert update["intent"]["category"] is None
    assert len(update["errors"]) == 1
    assert "stage1_intent_extraction" in update["errors"][0]


def test_query_intent_rejects_unknown_fields():
    with pytest.raises(Exception):
        QueryIntent(raw_query="test", not_a_real_field="oops")

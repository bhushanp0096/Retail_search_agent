"""
Stage 1 - State Schema & Intent Extraction.

Single node: `extract_intent_node`. Parses `state["raw_query"]` into a
structured `QueryIntent` via forced tool-use, and returns a *partial* state
update (per LangGraph convention — nodes return dict fragments, not the
full state) containing `intent` and, on failure, an `errors` entry.

This node never raises: extraction failures degrade gracefully to an
"empty" intent (just `raw_query` populated, everything else None/[]) plus a
logged error, so a single bad LLM call can't take down the whole graph.
Stage 2 can decide how to handle a low-signal intent (e.g. fall back to
pure keyword search).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from search_agent.prompts.intent_extraction_prompts import (
    INTENT_EXTRACTION_SYSTEM_PROMPT,
    build_intent_extraction_user_prompt,
)
from search_agent.schemas.intent import QueryIntent
from search_agent.schemas.state import GraphState
from search_agent.utils.common import log_node_execution
from search_agent.utils.llm_client import GroqStructuredClient, StructuredExtractionError

logger = logging.getLogger(__name__)

NODE_NAME = "stage1_intent_extraction"


def _empty_intent(raw_query: str) -> Dict[str, Any]:
    """Fallback intent when extraction fails: preserves the raw query so
    downstream keyword-based retrieval (Stage 2) still has something to work
    with, everything else left unset.
    """
    return QueryIntent(raw_query=raw_query).model_dump()


@log_node_execution(NODE_NAME)
def extract_intent_node(
    state: GraphState,
    *,
    client: Optional[GroqStructuredClient] = None,
) -> Dict[str, Any]:
    """LangGraph node: extracts structured intent from `state["raw_query"]`.

    Args:
        state: Current graph state. Requires `raw_query`.
        client: Optional injected `GroqStructuredClient` (used by tests
            / callers who want a custom model config). Defaults to a
            fresh client built from `search_agent.config.settings.llm`.

    Returns:
        Partial state update: `{"intent": {...}}` on success, plus
        `{"errors": [...]}` appended on failure (empty list otherwise,
        which is a no-op under the `operator.add` reducer).
    """
    raw_query = state["raw_query"]
    llm_client = client or GroqStructuredClient()

    try:
        completion = llm_client.extract_structured(
            system_prompt=INTENT_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=build_intent_extraction_user_prompt(raw_query),
            schema_model=QueryIntent,
            tool_name="extract_query_intent",
            tool_description="Return the structured search intent extracted from the customer's query.",
        )
        intent: QueryIntent = completion.parsed  # type: ignore[assignment]
        logger.info(
            "[%s] extracted intent for query=%r -> category=%s occasion=%s weather=%s",
            NODE_NAME, raw_query, intent.category, intent.occasion, intent.weather_attribute,
        )
        return {"intent": intent.model_dump(), "errors": []}

    except StructuredExtractionError as exc:
        logger.error("[%s] extraction failed for query=%r: %s", NODE_NAME, raw_query, exc)
        return {
            "intent": _empty_intent(raw_query),
            "errors": [f"{NODE_NAME}: {exc}"],
        }

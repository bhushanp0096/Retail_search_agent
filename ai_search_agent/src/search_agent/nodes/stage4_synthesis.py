"""
Stage 4 - Synthesis.

Two nodes:
- `synthesis_node` — the "has_results" branch. Calls the Groq-hosted model
  with the prompt built in `prompts/synthesis_prompts.py`, using
  `filtered_skus` (Stage 3) and `customer_profile` (Stage 2B) as grounding
  context.
- `no_results_node` — the "no_results" branch (see `graph/routing.py`).
  A static, honest response — no LLM call needed for "we don't have
  anything," and one less thing that can go wrong on an already-negative
  outcome for the customer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from search_agent.prompts.synthesis_prompts import (
    NO_RESULTS_RESPONSE,
    SYNTHESIS_SYSTEM_PROMPT,
    build_synthesis_user_prompt,
)
from search_agent.schemas.state import GraphState
from search_agent.utils.common import log_node_execution
from search_agent.utils.llm_client import GroqTextClient

logger = logging.getLogger(__name__)

SYNTHESIS_NODE_NAME = "stage4_synthesis"
NO_RESULTS_NODE_NAME = "stage4_no_results"


@log_node_execution(SYNTHESIS_NODE_NAME)
def synthesis_node(
    state: GraphState,
    *,
    client: Optional[GroqTextClient] = None,
) -> Dict[str, Any]:
    """LangGraph node: generates the final customer-facing response from
    `state["filtered_skus"]`, `state["customer_profile"]`, and
    `state["raw_query"]`.

    Only reached via the `has_results` branch of `route_after_scoring`, so
    `filtered_skus` is expected to be non-empty here — but this still
    degrades gracefully (a short generic message) rather than raising if
    the LLM call itself fails, matching Stage 1's "never crash the graph"
    pattern for LLM-backed nodes.
    """
    raw_query = state["raw_query"]
    filtered_skus = state.get("filtered_skus") or []
    customer_profile = state.get("customer_profile")
    text_client = client or GroqTextClient()

    user_prompt = build_synthesis_user_prompt(raw_query, filtered_skus, customer_profile)

    try:
        response_text = text_client.generate_text(
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        logger.info("[%s] generated response for query=%r", SYNTHESIS_NODE_NAME, raw_query)
        return {"final_response": response_text}

    except Exception as exc:  # noqa: BLE001 - node boundary: never let this crash the graph
        logger.error("[%s] failed: %s", SYNTHESIS_NODE_NAME, exc)
        fallback = f"Here are {len(filtered_skus)} result(s) matching your search."
        return {"final_response": fallback, "errors": [f"{SYNTHESIS_NODE_NAME}: {exc}"]}


@log_node_execution(NO_RESULTS_NODE_NAME)
def no_results_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node: the `no_results` branch. Static response, no LLM
    call — see module docstring for why.
    """
    return {"final_response": NO_RESULTS_RESPONSE}

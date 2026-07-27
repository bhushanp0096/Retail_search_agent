"""
Stage 2A - Candidate Retrieval.

Single node: `fetch_candidates_node`. Scores the full product catalog
against `state["intent"]` (produced by Stage 1) using the custom TF-IDF
scorer in `utils/text_scoring.py`, and returns the top-k candidates.

Designed to run in parallel with Stage 2B (`fetch_customer_profile_node`)
once wired into the graph in Stage 4 — this node only reads `state["intent"]`
and only writes `state["candidate_products"]`, so it never conflicts with 2B.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from search_agent.config import settings
from search_agent.schemas.state import GraphState
from search_agent.utils.common import log_node_execution
from search_agent.utils.data_loaders import load_products
from search_agent.utils.text_scoring import rank_products

logger = logging.getLogger(__name__)

NODE_NAME = "stage2a_candidate_retrieval"


@log_node_execution(NODE_NAME)
def fetch_candidates_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node: ranks the catalog against `state["intent"]`.

    Args:
        state: Current graph state. Requires `intent` (Stage 1's output).

    Returns:
        Partial state update: `{"candidate_products": [...]}` on success,
        or `{"candidate_products": [], "errors": [...]}` on failure. Kept
        deliberately simple — no retry loop — since this is deterministic
        code (CSV read + arithmetic), not a nondeterministic LLM call.
    """
    intent = state.get("intent") or {}

    try:
        products = load_products()
        candidates = rank_products(
            products,
            intent,
            top_k=settings.retrieval.top_k_candidates,
            min_score=settings.retrieval.min_relevance_score,
        )
        logger.info(
            "[%s] ranked %d candidates from %d products for intent=%r",
            NODE_NAME, len(candidates), len(products), intent,
        )
        return {"candidate_products": candidates}

    except Exception as exc:  # noqa: BLE001 - node boundary: never let this crash the graph
        logger.error("[%s] failed: %s", NODE_NAME, exc)
        return {"candidate_products": [], "errors": [f"{NODE_NAME}: {exc}"]}

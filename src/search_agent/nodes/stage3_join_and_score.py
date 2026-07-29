"""
Stage 3 - Relational Join & Personalization Engine.

Single node: `join_and_score_node`. Joins Stage 2A's `candidate_products`
with `inventory_pricing.csv`, applies in-stock + size filters, and ranks
the survivors by a composite score blending text relevance, category/
occasion match, personalization, and discount.

See `utils/join_and_score.py` for the actual logic (pulled from
`stage3_exploration.ipynb` once validated against real data) and
`config.ScoringSettings` for the tunable weights.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from search_agent.schemas.state import GraphState
from search_agent.utils.common import log_node_execution
from search_agent.utils.data_loaders import load_inventory
from search_agent.utils.join_and_score import rank_skus

logger = logging.getLogger(__name__)

NODE_NAME = "stage3_join_and_score"


@log_node_execution(NODE_NAME)
def join_and_score_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node: joins/filters/scores `state["candidate_products"]`
    against inventory + `state["customer_profile"]`.

    Args:
        state: Current graph state. Requires `candidate_products` (Stage 2A)
            and `intent` (Stage 1). `customer_profile` (Stage 2B) may be
            `None` (guest session) — that's handled, not an error.

    Returns:
        Partial state update: `{"filtered_skus": [...]}` — an empty list is
        a valid, expected result (e.g. nothing in stock in the customer's
        size), not a failure. Kept deliberately simple — no retry loop —
        since this is deterministic code (join + arithmetic), not a
        nondeterministic LLM call.
    """
    candidates = state.get("candidate_products") or []
    intent = state.get("intent") or {}
    customer_profile = state.get("customer_profile")

    try:
        inventory_df = load_inventory()
        filtered_skus = rank_skus(
            candidates=candidates,
            intent=intent,
            customer_profile=customer_profile,
            inventory_df=inventory_df,
        )
        logger.info(
            "[%s] ranked %d SKUs from %d candidates (customer=%s)",
            NODE_NAME, len(filtered_skus), len(candidates),
            (customer_profile or {}).get("customer_id"),
        )
        return {"filtered_skus": filtered_skus}

    except Exception as exc:  # noqa: BLE001 - node boundary: never let this crash the graph
        logger.error("[%s] failed: %s", NODE_NAME, exc)
        return {"filtered_skus": [], "errors": [f"{NODE_NAME}: {exc}"]}

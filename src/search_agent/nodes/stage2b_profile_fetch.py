"""
Stage 2B - Profile Fetching.

Single node: `fetch_customer_profile_node`. Looks up `state["customer_id"]`
in the customers CRM data. A missing/None `customer_id` is a normal guest
session, not an error — only unexpected I/O failures are treated as errors.

Designed to run in parallel with Stage 2A once wired into the graph in
Stage 4 — this node only reads `state["customer_id"]` and only writes
`state["customer_profile"]`, so it never conflicts with 2A.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from search_agent.schemas.state import GraphState
from search_agent.utils.common import log_node_execution
from search_agent.utils.data_loaders import load_customers

logger = logging.getLogger(__name__)

NODE_NAME = "stage2b_profile_fetch"


@log_node_execution(NODE_NAME)
def fetch_customer_profile_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node: looks up the customer profile for `state["customer_id"]`.

    Args:
        state: Current graph state. `customer_id` may be None (guest session).

    Returns:
        Partial state update: `{"customer_profile": {...} | None}`. A
        missing customer_id or an unknown id both resolve to `None` (not an
        error). Only an unexpected exception (e.g. malformed customers.json)
        is reported via `state["errors"]`.
    """
    customer_id = state.get("customer_id")

    if not customer_id:
        logger.info("[%s] no customer_id on state (guest session) — skipping lookup", NODE_NAME)
        return {"customer_profile": None}

    try:
        customers_by_id = load_customers()
    except Exception as exc:  # noqa: BLE001 - node boundary: never let this crash the graph
        logger.error("[%s] failed to load customers: %s", NODE_NAME, exc)
        return {"customer_profile": None, "errors": [f"{NODE_NAME}: {exc}"]}

    profile = customers_by_id.get(customer_id)
    if profile is None:
        logger.warning("[%s] customer_id=%r not found in CRM data", NODE_NAME, customer_id)
    else:
        logger.info("[%s] found profile for customer_id=%r", NODE_NAME, customer_id)

    return {"customer_profile": profile}

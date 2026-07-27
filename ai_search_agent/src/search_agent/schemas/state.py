"""
Global LangGraph state schema.

This is intentionally the *full* state shape for all four stages, even
though only Stage 1 is implemented right now. Defining it up front means:
  - Stage 2/3/4 nodes can be added later without changing the graph's
    state contract (no breaking changes to already-wired nodes).
  - Every node's signature is `def node(state: GraphState) -> dict`, and the
    dict it returns is a partial update merged into this TypedDict by
    LangGraph — so nodes stay decoupled from each other.

Fields are grouped by which stage first populates them. `total=False` is
avoided in favor of explicit `NotRequired[...]` per field so it's obvious
at a glance which keys are optional vs guaranteed present at graph entry.
"""

from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional

from typing_extensions import Annotated, NotRequired, TypedDict


class GraphState(TypedDict):
    # ---- Graph input (required at invocation) ----------------------------
    raw_query: str
    customer_id: NotRequired[Optional[str]]  # None => anonymous / guest session

    # ---- Stage 1: Intent Extraction --------------------------------------
    # Serialized `QueryIntent` (via `.model_dump()`), kept as a plain dict in
    # state so the TypedDict has no hard dependency on the pydantic model
    # import (keeps schemas/state.py lightweight and framework-agnostic).
    intent: NotRequired[Optional[Dict[str, Any]]]

    # ---- Stage 2: Candidate Retrieval & Profile Fetching (parallel) ------
    candidate_products: NotRequired[List[Dict[str, Any]]]      # Node 2A output
    customer_profile: NotRequired[Optional[Dict[str, Any]]]    # Node 2B output

    # ---- Stage 3: Relational Join & Personalization ----------------------
    filtered_skus: NotRequired[List[Dict[str, Any]]]

    # ---- Stage 4: Synthesis ------------------------------------------------
    final_response: NotRequired[Optional[str]]

    # ---- Cross-cutting ------------------------------------------------------
    # `operator.add` reducer: safe to be written by multiple (parallel) nodes
    # in the same super-step without one overwriting the other's errors.
    errors: Annotated[List[str], operator.add]

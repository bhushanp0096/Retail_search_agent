"""
Conditional routing for the compiled graph.

Kept as its own small module (rather than inline in `builder.py`) since
routing logic and edge-wiring are different concerns: this file answers
"which node comes next," `builder.py` answers "how is the graph shaped."
"""

from __future__ import annotations

from search_agent.schemas.state import GraphState

# Branch name constants — used both as the router's return values and as
# keys in `add_conditional_edges`'s destination mapping in builder.py, so
# there's exactly one place each string is spelled out.
HAS_RESULTS = "has_results"
NO_RESULTS = "no_results"


def route_after_scoring(state: GraphState) -> str:
    """Router run after Stage 3 (`join_and_score_node`): sends the graph to
    the real synthesis node if there's at least one ranked SKU, or to the
    no-results node otherwise.

    An empty `filtered_skus` is a real, expected outcome (e.g. nothing in
    stock in the customer's exact size — see stage3/stage4 exploration
    notebooks) and gets its own honest response, not a hallucinated one.
    """
    if state.get("filtered_skus"):
        return HAS_RESULTS
    return NO_RESULTS

"""
Stage 4 - LangGraph Compilation.

`build_graph()` wires every stage's node into one `StateGraph`:

    START -> extract_intent -> [fetch_candidates, fetch_profile]  (fan-out)
          -> join_and_score  (fan-in)
          -> route_after_scoring -> synthesis | no_results  (conditional)
          -> END

Compiled with a `MemorySaver` checkpointer — an easy, no-cost-elsewhere
add-on (see `stage4_exploration.ipynb`, section 6 discussion): it needed no
changes to any node or to `GraphState`, just a checkpointer instance passed
to `.compile()` and a `thread_id` passed at invoke time. This is in-memory
only (lost on process restart) and gives per-conversation state history for
free; swapping in a persistent checkpointer later (e.g. Postgres) is a
one-line change here, not a redesign.

`invoke_search_graph(...)` is the single entrypoint both `main.py` and the
FastAPI app call — neither needs to know about `StateGraph` internals.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from search_agent.graph.routing import HAS_RESULTS, NO_RESULTS, route_after_scoring
from search_agent.nodes.stage1_intent_extraction import extract_intent_node
from search_agent.nodes.stage2a_candidate_retrieval import fetch_candidates_node
from search_agent.nodes.stage2b_profile_fetch import fetch_customer_profile_node
from search_agent.nodes.stage3_join_and_score import join_and_score_node
from search_agent.nodes.stage4_synthesis import no_results_node, synthesis_node
from search_agent.schemas.state import GraphState

logger = logging.getLogger(__name__)

_compiled_graph: Optional[CompiledStateGraph] = None


def build_graph() -> CompiledStateGraph:
    """Constructs and compiles the full 4-stage search agent graph.

    Cached at module level (see `get_compiled_graph`) since compilation is
    pure setup work — no reason to rebuild it per request.
    """
    graph = StateGraph(GraphState)

    graph.add_node("extract_intent", extract_intent_node)
    graph.add_node("fetch_candidates", fetch_candidates_node)
    graph.add_node("fetch_profile", fetch_customer_profile_node)
    graph.add_node("join_and_score", join_and_score_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("no_results", no_results_node)

    graph.add_edge(START, "extract_intent")
    graph.add_edge("extract_intent", "fetch_candidates")  # fan-out
    graph.add_edge("extract_intent", "fetch_profile")     # fan-out
    graph.add_edge("fetch_candidates", "join_and_score")  # fan-in
    graph.add_edge("fetch_profile", "join_and_score")     # fan-in
    graph.add_conditional_edges(
        "join_and_score",
        route_after_scoring,
        {HAS_RESULTS: "synthesis", NO_RESULTS: "no_results"},
    )
    graph.add_edge("synthesis", END)
    graph.add_edge("no_results", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def get_compiled_graph() -> CompiledStateGraph:
    """Returns the module-level compiled graph, building it on first call."""
    global _compiled_graph
    if _compiled_graph is None:
        logger.info("Compiling search agent graph")
        _compiled_graph = build_graph()
    return _compiled_graph


def invoke_search_graph(
    raw_query: str,
    *,
    customer_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs one query through the full graph.

    Args:
        raw_query: The customer's free-text search query.
        customer_id: Optional CRM id; omit/None for a guest session.
        thread_id: Optional conversation id for the checkpointer. If not
            given, a new one is generated — meaning by default every call
            is its own fresh conversation. Pass the same `thread_id` across
            calls to accumulate state in one LangGraph-tracked thread.

    Returns:
        The final `GraphState` dict — includes `final_response`,
        `filtered_skus`, `intent`, `customer_profile`, and `errors`.
    """
    resolved_thread_id = thread_id or str(uuid.uuid4())
    compiled_graph = get_compiled_graph()

    initial_state: GraphState = {
        "raw_query": raw_query,
        "customer_id": customer_id,
        "errors": [],
    }
    config = {"configurable": {"thread_id": resolved_thread_id}}

    result = compiled_graph.invoke(initial_state, config=config)
    return {**result, "thread_id": resolved_thread_id}  # thread_id isn't part of GraphState; convenience for callers

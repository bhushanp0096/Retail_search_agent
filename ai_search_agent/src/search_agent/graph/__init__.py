"""
Stage 4 - LangGraph Compilation & Synthesis.

Public API: `build_graph()` / `get_compiled_graph()` / `invoke_search_graph()`
in `builder.py`, and the routing constants/function in `routing.py`.
"""

from search_agent.graph.builder import build_graph, get_compiled_graph, invoke_search_graph
from search_agent.graph.routing import HAS_RESULTS, NO_RESULTS, route_after_scoring

__all__ = [
    "build_graph",
    "get_compiled_graph",
    "invoke_search_graph",
    "route_after_scoring",
    "HAS_RESULTS",
    "NO_RESULTS",
]

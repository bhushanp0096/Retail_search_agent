"""
search_agent
============
Modular LangGraph-based AI search agent MVP for a retail/e-commerce catalog.

Package layout
--------------
schemas/    Pydantic + TypedDict data contracts (LLM I/O + LangGraph state).
prompts/    All LLM prompt text, kept out of node logic so prompts can be
            iterated on / versioned independently of code.
nodes/      One module per LangGraph node (one node per pipeline stage).
utils/      Small, reusable, side-effect-light helper functions (dataclasses,
            LLM client wrapper, data loaders, logging).
graph/      StateGraph construction + compilation (wired up in Stage 4).

Execution plan
--------------
Stage 1 - State Schema & Intent Extraction        (this package, implemented)
Stage 2 - Candidate Retrieval & Profile Fetching  (parallel nodes, TODO)
Stage 3 - Relational Join & Personalization       (deterministic scoring, TODO)
Stage 4 - LangGraph Compilation & Synthesis       (final wiring, TODO)
"""

__version__ = "0.1.0"

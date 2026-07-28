"""
FastAPI wrapper around the compiled search agent graph.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    uvicorn app:app --reload

Then:
    curl -X POST http://127.0.0.1:8000/search \\
        -H "Content-Type: application/json" \\
        -d '{"query": "waterproof jacket for an October coastal wedding", "customer_id": "CUST-001"}'

The graph is compiled once at process startup (`lifespan`), not per request
— compilation is pure setup work, invoking is the per-request cost.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from search_agent.graph import get_compiled_graph, invoke_search_graph
from search_agent.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Compiling search agent graph at startup")
    get_compiled_graph()  # build once, cached for the process lifetime
    yield


app = FastAPI(
    title="AI Search Agent",
    description="LangGraph-based conversational search agent over a retail catalog.",
    version="0.1.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The customer's free-text search query.")
    customer_id: Optional[str] = Field(default=None, description="Optional CRM customer id (e.g. CUST-001).")
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional conversation/session id. Omit to start a fresh conversation each call.",
    )


class SearchResponse(BaseModel):
    final_response: str
    thread_id: str
    filtered_skus: List[Dict[str, Any]] = Field(default_factory=list)
    intent: Optional[Dict[str, Any]] = None
    customer_profile: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)


@app.get("/health")
def health() -> Dict[str, str]:
    """Basic liveness check — doesn't touch the graph or the LLM."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Runs one query through the full 4-stage graph and returns the result.

    Node-level failures (a bad LLM call, a missing file, etc.) don't raise
    here — every node degrades gracefully and reports into `errors` instead
    (see each stage's node module). This endpoint only raises a 500 for a
    genuinely unexpected failure in the graph invocation itself.
    """
    try:
        result = invoke_search_graph(
            request.query,
            customer_id=request.customer_id,
            thread_id=request.thread_id,
        )
    except Exception as exc:  # noqa: BLE001 - last-resort boundary; nodes already handle their own errors
        logger.error("Unexpected failure invoking search graph: %s", exc)
        raise HTTPException(status_code=500, detail="Internal error while processing the search request.")

    return SearchResponse(
        final_response=result["final_response"],
        thread_id=result["thread_id"],
        filtered_skus=result.get("filtered_skus") or [],
        intent=result.get("intent"),
        customer_profile=result.get("customer_profile"),
        errors=result.get("errors") or [],
    )

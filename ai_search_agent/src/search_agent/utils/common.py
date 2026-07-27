"""
Small, generic, reusable helpers shared by every stage's node(s).

Kept deliberately minimal for now (Stage 1 only needs the timing decorator +
NodeTiming dataclass for consistent observability). Stages 2-4 will likely
add: a `load_products()` / `load_inventory()` / `load_customers()` trio of
cached data-loaders, a text-normalization helper for the TF-IDF/string
matching in Stage 2, etc.
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class NodeTiming:
    """Standardized timing/observability record for a single LangGraph node
    invocation. Every node decorated with `@log_node_execution` logs one of
    these, giving consistent per-node latency visibility once the graph is
    compiled in Stage 4 (and making it trivial to plug into real tracing
    later without touching node logic itself).
    """

    node_name: str
    duration_ms: float
    success: bool
    error: str | None = None


def log_node_execution(node_name: str) -> Callable[[F], F]:
    """Decorator for LangGraph node functions: logs entry/exit, duration, and
    surfaces exceptions as structured log records rather than letting them
    propagate silently.

    Usage:
        @log_node_execution("stage1_intent_extraction")
        def extract_intent_node(state: GraphState) -> dict:
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            logger.info("[%s] starting", node_name)
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                timing = NodeTiming(node_name=node_name, duration_ms=duration_ms, success=True)
                logger.info("[%s] completed in %.1fms", timing.node_name, timing.duration_ms)
                return result
            except Exception as exc:  # noqa: BLE001 - intentionally broad; re-raised below
                duration_ms = (time.perf_counter() - start) * 1000
                timing = NodeTiming(
                    node_name=node_name, duration_ms=duration_ms, success=False, error=str(exc)
                )
                logger.error(
                    "[%s] failed after %.1fms: %s", timing.node_name, timing.duration_ms, timing.error
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator

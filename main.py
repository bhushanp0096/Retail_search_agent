"""
Main CLI entrypoint — runs a single query through the full 4-stage graph.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py "waterproof jacket for an October coastal wedding"
    python main.py "waterproof jacket for an October coastal wedding" --customer-id CUST-001
    python main.py "something minimalist for the office" --customer-id CUST-001 --thread-id demo-1

Without an API key, this still runs end to end — Stage 1 degrades to an
empty intent (logged as an error, not a crash) and Stage 4 will typically
land on the no-results branch. See README for details.
"""

from __future__ import annotations

import argparse
import json
import sys

from search_agent.graph import invoke_search_graph
from search_agent.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a query through the AI search agent.")
    parser.add_argument("query", help="The customer's free-text search query.")
    parser.add_argument("--customer-id", default=None, help="Optional CRM customer id (e.g. CUST-001).")
    parser.add_argument("--thread-id", default=None, help="Optional conversation/session id for checkpointing.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ...).")
    parser.add_argument(
        "--show-skus", action="store_true", help="Also print the ranked SKU list, not just the final response."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    result = invoke_search_graph(
        args.query,
        customer_id=args.customer_id,
        thread_id=args.thread_id,
    )

    print("\n" + "=" * 80)
    print(f"QUERY: {args.query}")
    print("=" * 80)
    print(result["final_response"])

    if args.show_skus:
        print("\n--- Ranked SKUs ---")
        print(json.dumps(result.get("filtered_skus") or [], indent=2))

    if result.get("errors"):
        print(f"\n(non-fatal issues encountered: {result['errors']})", file=sys.stderr)


if __name__ == "__main__":
    main()

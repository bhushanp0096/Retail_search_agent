"""
Stage 1 demo entrypoint.

Runs `extract_intent_node` against a handful of sample queries and prints
the resulting structured intent, so Stage 1 can be validated end-to-end
before Stage 2 (which will consume `state["intent"]`) is built.

Requires a real ANTHROPIC_API_KEY in the environment (or a `.env` file at
the project root — see `.env.example`).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_stage1_demo.py
"""

from __future__ import annotations

import json
import sys

from search_agent.logging_config import setup_logging
from search_agent.nodes.stage1_intent_extraction import extract_intent_node

SAMPLE_QUERIES = [
    "waterproof jacket for an October coastal wedding",
    "something stylish and breathable for the office, under $120",
    "durable size 10 hiking boots for the trail",
    "a minimalist black leather belt as a gift for my dad",
]


def main() -> None:
    setup_logging(level="INFO")

    for query in SAMPLE_QUERIES:
        state = {"raw_query": query, "errors": []}
        update = extract_intent_node(state)

        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)
        print(json.dumps(update["intent"], indent=2))
        if update.get("errors"):
            print(f"!! errors: {update['errors']}", file=sys.stderr)


if __name__ == "__main__":
    main()

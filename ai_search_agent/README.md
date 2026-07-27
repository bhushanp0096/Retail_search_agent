# AI Search Agent MVP

A LangGraph-based conversational search agent over a synthetic retail/e-commerce
catalog (`products.csv`, `inventory_pricing.csv`, `customers.json`).

## Architecture

```
ai_search_agent/
├── src/search_agent/
│   ├── config.py                 # env-driven settings (paths, model, retries) — single source of truth
│   ├── logging_config.py         # one setup_logging() call, consistent format everywhere
│   ├── schemas/
│   │   ├── state.py              # GraphState TypedDict — the full LangGraph state contract (all 4 stages)
│   │   └── intent.py             # QueryIntent Pydantic model — Stage 1's structured-output contract
│   ├── prompts/
│   │   └── intent_extraction_prompts.py   # Stage 1 system/user prompt text, isolated from node logic
│   ├── nodes/
│   │   └── stage1_intent_extraction.py    # extract_intent_node — the actual Stage 1 LangGraph node
│   ├── utils/
│   │   ├── llm_client.py         # AnthropicStructuredClient — tool-use structured output + retry/validation
│   │   └── common.py             # log_node_execution decorator, NodeTiming dataclass (used by every stage)
│   └── graph/
│       └── __init__.py           # placeholder — StateGraph wiring lands in Stage 4
├── tests/
│   └── test_stage1_intent_extraction.py   # fully offline (fake Anthropic client, no API key needed)
├── data/                          # products.csv / inventory_pricing.csv / customers.json
├── run_stage1_demo.py             # standalone script to exercise Stage 1 against sample queries
├── requirements.txt
├── pyproject.toml                 # `pip install -e .` for the search_agent package
├── pytest.ini                     # adds src/ to pythonpath so tests import search_agent cleanly
└── .env.example
```

**Design principles this follows:**
- **State schema defined once, up front.** `GraphState` already has placeholder keys for
  Stages 2-4 (`candidate_products`, `customer_profile`, `filtered_skus`, `final_response`),
  so later stages are additive — no breaking changes to Stage 1's node signature.
- **Prompts are data, not code.** Every prompt lives in `prompts/`, never inline in a node,
  so prompt tuning doesn't require touching pipeline logic or re-reviewing business logic.
- **Structured output via tool-use, not "please return JSON."** `AnthropicStructuredClient`
  forces a tool call whose schema is generated directly from the Pydantic model
  (`schema_model.model_json_schema()`), so there's one definition of the schema, not two
  (a prompt description *and* a parser) that can drift apart.
- **Nodes never raise.** `extract_intent_node` catches extraction failures and degrades to
  an empty-but-valid intent (keeping `raw_query` for keyword fallback) plus a logged error
  in `state["errors"]` — a single bad LLM call can't crash the graph.
- **Every node is `@log_node_execution`-wrapped** for consistent latency/success logging,
  which becomes free observability once Stage 4 compiles the full graph.

## Execution Plan

| Stage | Focus | Status |
|---|---|---|
| **1** | State schema (`GraphState`) + intent extraction node (Pydantic + tool-use) | ✅ Implemented |
| **2** | Parallel nodes: 2A TF-IDF/string-match candidate retrieval over `products.csv`; 2B CRM lookup against `customers.json` | 🔜 Next |
| **3** | Join candidates with `inventory_pricing.csv`, apply in-stock + size filters, compute composite relevance score | 🔜 |
| **4** | Wire `StateGraph`, conditional routing + parallel branches, compile, final LLM synthesis node | 🔜 |

## Stage 1 — What it does

`extract_intent_node(state) -> dict` takes `state["raw_query"]` (e.g.
*"waterproof jacket for an October coastal wedding"*) and returns a partial
state update:

```json
{
  "intent": {
    "raw_query": "waterproof jacket for an October coastal wedding",
    "category": "Outerwear",
    "occasion": "Wedding",
    "weather_attribute": "waterproof",
    "material": null,
    "size": null,
    "color": null,
    "max_price": null,
    "keywords": ["October", "coastal"]
  },
  "errors": []
}
```

`category`, `occasion`, and `weather_attribute` are constrained `Literal` types
matching the catalog's actual vocabulary (see `schemas/intent.py`), so Stage 3's
relational join can filter on them directly — no fuzzy string matching needed
for the structured fields. Anything that doesn't fit a structured field (style
adjectives, recipient, season, location, etc.) falls into `keywords`, which
Stage 2's text-matching node will use as additional signal.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs `search_agent` from src/ in editable mode
cp .env.example .env      # then fill in ANTHROPIC_API_KEY
```

## Running Stage 1

```bash
python run_stage1_demo.py
```

## Running tests

Tests use a fake Anthropic client (see `tests/test_stage1_intent_extraction.py`),
so no API key or network access is required:

```bash
pytest -v
```

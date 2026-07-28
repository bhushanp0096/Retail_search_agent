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
│   │   ├── stage1_intent_extraction.py    # extract_intent_node — the actual Stage 1 LangGraph node
│   │   ├── stage2a_candidate_retrieval.py # fetch_candidates_node — TF-IDF ranks products.csv
│   │   ├── stage2b_profile_fetch.py       # fetch_customer_profile_node — CRM lookup in customers.json
│   │   ├── stage3_join_and_score.py       # join_and_score_node — join + filter + composite rank
│   │   └── stage4_synthesis.py            # synthesis_node / no_results_node — final response generation
│   ├── prompts/
│   │   ├── intent_extraction_prompts.py   # Stage 1 system/user prompt text, isolated from node logic
│   │   └── synthesis_prompts.py           # Stage 4 system/user prompt text + NO_RESULTS_RESPONSE
│   ├── utils/
│   │   ├── llm_client.py         # AnthropicStructuredClient (Stage 1) + AnthropicTextClient (Stage 4)
│   │   ├── common.py             # log_node_execution decorator, NodeTiming dataclass (used by every stage)
│   │   ├── data_loaders.py       # load_products() / load_customers() / load_inventory() — lru_cache'd readers
│   │   ├── text_scoring.py       # custom stdlib TF-IDF scorer (tokenize, IDF table, rank_products)
│   │   └── join_and_score.py     # pandas join/filter/composite-score logic (Stage 3), ported from the notebook
│   └── graph/
│       ├── builder.py            # build_graph() / get_compiled_graph() / invoke_search_graph()
│       └── routing.py            # route_after_scoring() + HAS_RESULTS/NO_RESULTS constants
├── tests/
│   └── test_stage1_intent_extraction.py   # fully offline (fake Anthropic client, no API key needed)
├── data/                          # products.csv / inventory_pricing.csv / customers.json
├── run_stage1_demo.py             # standalone script to exercise Stage 1 against sample queries
├── main.py                        # CLI entrypoint — runs one query through the full graph
├── app.py                         # FastAPI wrapper around the compiled graph
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
| **2** | Parallel nodes: 2A TF-IDF/string-match candidate retrieval over `products.csv`; 2B CRM lookup against `customers.json` | ✅ Implemented |
| **3** | Join candidates with `inventory_pricing.csv`, apply in-stock + size filters, compute composite relevance score | ✅ Implemented |
| **4** | Wire `StateGraph`, conditional routing + parallel branches, compile, final LLM synthesis node | ✅ Implemented |

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

## Stage 2 — What it does

Two independent nodes, both reading from state produced earlier in the graph,
designed to run in parallel once Stage 4 wires the graph together:

- **`fetch_candidates_node`** (2A) reads `state["intent"]`, scores all 500
  products via a pure-stdlib TF-IDF (`utils/text_scoring.py`: tokenize ->
  build an IDF table over the catalog -> score each product by summed
  `tf * idf` over the query's terms), and returns the top `top_k_candidates`
  (default 20, configurable via `SEARCH_AGENT_TOP_K`) as `candidate_products`,
  each annotated with a `_relevance_score`.
- **`fetch_customer_profile_node`** (2B) reads `state["customer_id"]` and
  looks it up in `customers.json` (loaded once, cached, keyed by id for O(1)
  lookup). A missing/`None` `customer_id`, or an id not found in the CRM
  data, both resolve to `customer_profile: None` — that's a normal guest
  session, not an error.

Both nodes follow a **simple** error-handling pattern (deliberately no
retry loop, unlike Stage 1's LLM calls): wrap the risky I/O in `try/except`,
log it, return an empty/`None` result plus one entry in `state["errors"]`.
Deterministic code doesn't need self-correction — just a clean failure mode.

**Known limitation, fixed in Stage 3:** the TF-IDF scorer alone weighs all
fields equally, so a strong keyword match (e.g. a product literally named
"Coastal Shorts") could outrank a correct-category item with weaker keyword
overlap. Stage 3's composite scoring adds a category/occasion match bonus
on top of this raw text score to correct for it — see below.

## Stage 3 — What it does

One node: `join_and_score_node`, wrapping pandas-based logic in
`utils/join_and_score.py` that was built and validated in
`stage3_exploration.ipynb` before being ported into the module.

1. **Join** — `state["candidate_products"]` (Stage 2A's ~20 products) is
   merged with `inventory_pricing.csv` on `product_id`, fanning out to
   SKU-level rows (one per size), since in-stock/size are SKU-level facts.
2. **Filter: in-stock** — drops any SKU with `stock_count == 0`.
3. **Filter: size match** — if `state["customer_profile"]` has a
   `size_profile`, keeps only SKUs matching that customer's size for the
   relevant category (`Tops`/`Outerwear` -> `size_profile["tops"]`,
   `Bottoms` -> `size_profile["bottoms"]`, `Footwear` -> `size_profile["shoes"]`,
   `Accessories` -> never filtered). A guest session (`customer_profile is
   None`) skips this filter entirely.
4. **Composite score** — `_relevance_score` (Stage 2A) + category/occasion
   match bonus + personalization bonus (customer's `style_preferences`
   overlap) + discount bonus. Weights live in `config.ScoringSettings`
   (`SEARCH_AGENT_CATEGORY_BONUS`, `SEARCH_AGENT_OCCASION_BONUS`,
   `SEARCH_AGENT_PERSONALIZATION_WEIGHT`, `SEARCH_AGENT_DISCOUNT_WEIGHT`,
   `SEARCH_AGENT_FINAL_TOP_K`), not hardcoded in the scoring functions.
5. Returns the top `final_top_k` (default 10) SKUs, sorted descending, as
   `state["filtered_skus"]`.

**A real, expected edge case surfaced during exploration:** for the query
*"waterproof jacket for an October coastal wedding"*, the single best
category/occasion match ("Alpine Parka") only exists in size L — a customer
who wears S never sees it, and a correctly-scored-but-different item ranks
#1 instead. This isn't a scoring bug; it's the filter doing its job. Stage
4's synthesis node should account for this (e.g. "we don't have the Alpine
Parka in your size, but here's...") rather than silently presenting the
next-best match as if it were the top pick.

Error handling here is intentionally simple — no retry loop, unlike Stage
1's LLM calls: wrap the join/load in `try/except`, log it, return
`{"filtered_skus": [], "errors": [...]}`. An **empty result is not an
error** on its own (it can legitimately mean "nothing in stock in your
size") — only unexpected exceptions go into `errors`.

## Stage 4 — What it does

`graph/builder.py` wires every stage into one `StateGraph`:

```
START -> extract_intent -> [fetch_candidates, fetch_profile]   (fan-out, run concurrently)
       -> join_and_score                                       (fan-in)
       -> route_after_scoring -> synthesis | no_results         (conditional)
       -> END
```

- **`route_after_scoring`** (`graph/routing.py`) checks `state["filtered_skus"]`
  after Stage 3: non-empty goes to `synthesis`, empty goes to `no_results`.
  An empty result is a real, expected outcome (see the Stage 3/4 exploration
  notebooks) — it gets its own honest response, not a hallucinated one.
- **`synthesis_node`** calls Claude (via `AnthropicTextClient`, plain text
  generation — no forced tool-use needed here) with a prompt built from
  `filtered_skus`, `customer_profile`, and the original query. Like Stage 1,
  it never raises: an LLM failure falls back to a generic "Here are N
  result(s)" message plus a logged error, rather than crashing the request.
- **`no_results_node`** is a static response — no LLM call needed for
  "we don't have anything," which is one less thing that can fail on an
  already-negative outcome.
- **Checkpointing**: the compiled graph uses LangGraph's `MemorySaver`,
  keyed by `thread_id`. This was genuinely a free add-on — no changes to
  any node or to `GraphState` — so it's included now rather than deferred.
  It's in-memory only (state is lost on process restart); swapping in a
  persistent checkpointer later is a one-line change in `build_graph()`.
- **`invoke_search_graph(raw_query, customer_id=None, thread_id=None)`** is
  the one function both `main.py` and `app.py` call — neither needs to
  touch `StateGraph` directly.

**A fix made along the way:** while wiring this up, `extract_intent_node`
turned out to only catch `StructuredExtractionError` — a raw API/auth/
network error from the Anthropic SDK would have propagated uncaught and
crashed the graph, contradicting its own "never raise" design goal from
Stage 1. Broadened its except clause to cover any exception, not just
validation-retry exhaustion.

## Running the CLI

```bash
python main.py "waterproof jacket for an October coastal wedding" --customer-id CUST-001 --show-skus
```

Works even without `ANTHROPIC_API_KEY` set — Stage 1 degrades to an empty
intent (logged as a non-fatal error) and Stage 4 typically lands on the
no-results branch, so you can exercise the full graph's wiring without a key.

## Running the API

```bash
uvicorn app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/search \\
    -H "Content-Type: application/json" \\
    -d '{"query": "waterproof jacket for an October coastal wedding", "customer_id": "CUST-001"}'
```

Interactive docs at `http://127.0.0.1:8000/docs`. `GET /health` is a plain
liveness check that doesn't touch the graph or the LLM.

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

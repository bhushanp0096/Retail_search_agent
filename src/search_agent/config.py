"""
Central configuration for the search_agent package.

Everything environment- or deployment-specific lives here (or is overridable
via environment variables) so nodes/utils never hardcode paths or model
names inline. Import this module rather than reading os.environ elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load a local .env file if present (no-op if it doesn't exist).
load_dotenv()

# --------------------------------------------------------------------------
# Filesystem paths
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = Path(os.getenv("SEARCH_AGENT_DATA_DIR", PROJECT_ROOT / "data"))
LOG_DIR: Path = Path(os.getenv("SEARCH_AGENT_LOG_DIR", PROJECT_ROOT / "logs"))

PRODUCTS_CSV: Path = DATA_DIR / "products.csv"
INVENTORY_CSV: Path = DATA_DIR / "inventory_pricing.csv"
CUSTOMERS_JSON: Path = DATA_DIR / "customers.json"


@dataclass(frozen=True)
class LLMSettings:
    """Settings for the Grow client used for structured extraction / synthesis."""

    # Current Claude model lineup (see Grow docs for the full/latest list).
    # Overridable per-environment without touching code.
    model: str = field(default_factory=lambda: os.getenv("SEARCH_AGENT_LLM_MODEL", "claude-sonnet-5"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("SEARCH_AGENT_MAX_TOKENS", "1024")))
    temperature: float = field(default_factory=lambda: float(os.getenv("SEARCH_AGENT_TEMPERATURE", "0.0")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("SEARCH_AGENT_LLM_MAX_RETRIES", "2")))
    api_key_env_var: str = "GROQ_API_KEY"


@dataclass(frozen=True)
class RetrievalSettings:
    """Tunables for Stage 2 candidate retrieval (kept here now so Stage 1's
    config surface doesn't need to change shape when Stage 2 lands)."""

    top_k_candidates: int = field(default_factory=lambda: int(os.getenv("SEARCH_AGENT_TOP_K", "20")))
    # Products scoring at/below this are dropped before the top-k cut, so a
    # near-empty/irrelevant query doesn't return 20 essentially-random products.
    min_relevance_score: float = field(
        default_factory=lambda: float(os.getenv("SEARCH_AGENT_MIN_RELEVANCE_SCORE", "0.0"))
    )


@dataclass(frozen=True)
class ScoringSettings:
    """Composite-score weights for Stage 3's join/filter/rank pipeline.

    Values here are the ones validated in `stage3_exploration.ipynb` — not
    guessed in the abstract. See that notebook (section 4) for the reasoning
    behind category_match_bonus/occasion_match_bonus in particular: without
    them, raw text relevance alone lets an incidentally-keyword-matching
    product outrank a correctly-categorized one.
    """

    category_match_bonus: float = field(
        default_factory=lambda: float(os.getenv("SEARCH_AGENT_CATEGORY_BONUS", "5.0"))
    )
    occasion_match_bonus: float = field(
        default_factory=lambda: float(os.getenv("SEARCH_AGENT_OCCASION_BONUS", "3.0"))
    )
    personalization_weight: float = field(
        default_factory=lambda: float(os.getenv("SEARCH_AGENT_PERSONALIZATION_WEIGHT", "1.5"))
    )
    discount_weight: float = field(
        default_factory=lambda: float(os.getenv("SEARCH_AGENT_DISCOUNT_WEIGHT", "2.0"))
    )
    # Final number of ranked SKUs returned to the customer — distinct from
    # RetrievalSettings.top_k_candidates (Stage 2A's broader candidate pool).
    final_top_k: int = field(default_factory=lambda: int(os.getenv("SEARCH_AGENT_FINAL_TOP_K", "10")))


@dataclass(frozen=True)
class AppSettings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    scoring: ScoringSettings = field(default_factory=ScoringSettings)
    log_level: str = field(default_factory=lambda: os.getenv("SEARCH_AGENT_LOG_LEVEL", "INFO"))
    log_to_file: bool = field(default_factory=lambda: os.getenv("SEARCH_AGENT_LOG_TO_FILE", "false").lower() == "true")


settings = AppSettings()

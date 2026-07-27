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
load_dotenv(override=True)

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
    """Settings for the ChatGroq client used for structured extraction / synthesis.

    All fields are overridable via environment variables so the codebase never
    hardcodes deployment-specific values.
    """

    # Groq model to use.  Switch via SEARCH_AGENT_LLM_MODEL env var without
    # touching code (e.g. "llama-3.1-8b-instant" for a faster/cheaper variant).
    model: str = field(default_factory=lambda: os.getenv("SEARCH_AGENT_LLM_MODEL", "llama-3.3-70b-versatile"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("SEARCH_AGENT_MAX_TOKENS", "1024")))
    temperature: float = field(default_factory=lambda: float(os.getenv("SEARCH_AGENT_TEMPERATURE", "0.7")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("SEARCH_AGENT_LLM_MAX_RETRIES", "2")))
    # Name of the environment variable that holds the Groq API key.
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
class AppSettings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    log_level: str = field(default_factory=lambda: os.getenv("SEARCH_AGENT_LOG_LEVEL", "INFO"))
    log_to_file: bool = field(default_factory=lambda: os.getenv("SEARCH_AGENT_LOG_TO_FILE", "false").lower() == "true")


settings = AppSettings()

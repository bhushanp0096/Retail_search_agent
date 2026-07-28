"""
Data loaders for the static catalog/CRM/inventory files.

`load_products` / `load_customers` return plain dicts (Stage 1/2's nodes
are pure-stdlib, no pandas dependency). `load_inventory` returns a pandas
DataFrame since Stage 3's join/filter/score logic is pandas-based — see
`utils/join_and_score.py`.

All three are `lru_cache`d: the underlying files don't change during a
process's lifetime in this MVP, so every query re-reading/re-parsing them
from disk would be pure waste. If you need to pick up edited data without
restarting the process, call `.cache_clear()` on the relevant function.
"""

from __future__ import annotations

import csv
import json
import logging
from functools import lru_cache
from typing import Any, Dict, List

import pandas as pd

from search_agent.config import CUSTOMERS_JSON, INVENTORY_CSV, PRODUCTS_CSV

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_products() -> List[Dict[str, Any]]:
    """Loads products.csv into a list of dicts (one per product row).

    Returns:
        List of product dicts with keys: product_id, name, description,
        category, material, occasion.
    """
    if not PRODUCTS_CSV.exists():
        raise FileNotFoundError(f"products.csv not found at {PRODUCTS_CSV}")

    with open(PRODUCTS_CSV, newline="", encoding="utf-8") as f:
        products = list(csv.DictReader(f))

    logger.info("Loaded %d products from %s", len(products), PRODUCTS_CSV)
    return products


@lru_cache(maxsize=1)
def load_customers() -> Dict[str, Dict[str, Any]]:
    """Loads customers.json into a dict keyed by customer_id for O(1) lookup.

    Returns:
        Dict mapping customer_id -> customer record (name, size_profile,
        style_preferences, purchase_history).
    """
    if not CUSTOMERS_JSON.exists():
        raise FileNotFoundError(f"customers.json not found at {CUSTOMERS_JSON}")

    with open(CUSTOMERS_JSON, encoding="utf-8") as f:
        customers = json.load(f)

    by_id = {c["customer_id"]: c for c in customers}
    logger.info("Loaded %d customers from %s", len(by_id), CUSTOMERS_JSON)
    return by_id


@lru_cache(maxsize=1)
def load_inventory() -> pd.DataFrame:
    """Loads inventory_pricing.csv as a pandas DataFrame, used by Stage 3's
    join against candidate products.

    Returns:
        DataFrame with columns: sku_id, product_id, size, stock_count,
        base_price, discount_percentage.
    """
    if not INVENTORY_CSV.exists():
        raise FileNotFoundError(f"inventory_pricing.csv not found at {INVENTORY_CSV}")

    inventory_df = pd.read_csv(INVENTORY_CSV)
    logger.info("Loaded %d inventory rows from %s", len(inventory_df), INVENTORY_CSV)
    return inventory_df

"""
Stage 3 join, filter, and composite-scoring logic — pandas-based.

Ported directly from `stage3_exploration.ipynb` once the logic was
validated against the real dataset (see that notebook for the reasoning,
including the "Coastal Shorts" / "Alpine Parka" case that motivated the
category/occasion bonus, and the size-exclusion edge case it also
surfaced). Kept as pure, side-effect-free functions operating on
DataFrames/dicts — no GraphState dependency here, so this stays reusable
and testable independent of the graph.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from search_agent.config import ScoringSettings
from search_agent.config import settings as default_settings

# category -> which key in a customer's size_profile it should be checked
# against. Accessories intentionally maps to None: never filtered on size.
CATEGORY_TO_SIZE_PROFILE_KEY: Dict[str, Optional[str]] = {
    "Tops": "tops",
    "Outerwear": "tops",
    "Bottoms": "bottoms",
    "Footwear": "shoes",
    "Accessories": None,
}


def join_candidates_with_inventory(
    candidates: List[Dict[str, Any]], inventory_df: pd.DataFrame
) -> pd.DataFrame:
    """Fans candidate products out to SKU-level rows via an inner join on
    product_id. A product with no matching inventory rows is simply
    dropped (shouldn't happen given the synthetic data generator, but
    handled defensively rather than assumed away).
    """
    candidates_df = pd.DataFrame(candidates)
    if candidates_df.empty:
        return candidates_df
    return candidates_df.merge(inventory_df, on="product_id", how="inner")


def filter_in_stock(skus_df: pd.DataFrame) -> pd.DataFrame:
    """Drops any SKU with stock_count <= 0."""
    if skus_df.empty:
        return skus_df
    return skus_df[skus_df["stock_count"] > 0].copy()


def matches_customer_size(row: pd.Series, size_profile: Optional[Dict[str, str]]) -> bool:
    """True if this SKU's size is wearable for this customer, or if sizing
    doesn't apply / isn't known (accessories, guest session, or the
    customer has no recorded preference for this category).
    """
    if size_profile is None:
        return True  # guest session: don't filter by size at all
    profile_key = CATEGORY_TO_SIZE_PROFILE_KEY.get(row["category"])
    if profile_key is None:
        return True  # accessories: one-size-fits-all, nothing to check
    preferred_size = size_profile.get(profile_key)
    if preferred_size is None:
        return True  # customer has no preference recorded for this category
    return str(row["size"]) == str(preferred_size)


def filter_by_size(skus_df: pd.DataFrame, size_profile: Optional[Dict[str, str]]) -> pd.DataFrame:
    """Applies `matches_customer_size` across all rows."""
    if skus_df.empty:
        return skus_df
    mask = skus_df.apply(lambda row: matches_customer_size(row, size_profile), axis=1)
    return skus_df[mask].copy()


def category_occasion_bonus(
    row: pd.Series, intent: Dict[str, Any], category_bonus: float, occasion_bonus: float
) -> float:
    """Rewards an exact category and/or occasion match — the fix for raw
    text relevance letting an incidentally-keyword-matching product (e.g.
    "Coastal Shorts") outrank a correctly-categorized one (e.g. "Alpine
    Parka") for a query like "waterproof jacket for a coastal wedding".
    """
    bonus = 0.0
    if intent.get("category") and row["category"] == intent["category"]:
        bonus += category_bonus
    if intent.get("occasion") and row["occasion"] == intent["occasion"]:
        bonus += occasion_bonus
    return bonus


def personalization_bonus(
    row: pd.Series, style_preferences: Optional[List[str]], weight: float
) -> float:
    """Counts how many of the customer's style tags appear in this
    product's searchable text, weighted. Contributes 0 for guest sessions
    or customers with no recorded style_preferences.
    """
    if not style_preferences:
        return 0.0
    haystack = " ".join(
        str(row.get(field, "")) for field in ("name", "description", "category", "material", "occasion")
    ).lower()
    matches = sum(1 for tag in style_preferences if tag.lower() in haystack)
    return matches * weight


def compute_composite_score(
    skus_df: pd.DataFrame,
    intent: Dict[str, Any],
    customer_profile: Optional[Dict[str, Any]],
    scoring_settings: Optional[ScoringSettings] = None,
) -> pd.DataFrame:
    """Adds `_composite_score` (plus its component columns, kept for
    debugging/inspection) to `skus_df`.

    composite_score = text_relevance (from Stage 2A's `_relevance_score`)
                       + category/occasion bonus
                       + personalization bonus
                       + discount bonus
    """
    if skus_df.empty:
        return skus_df

    s = scoring_settings or default_settings.scoring
    style_preferences = (customer_profile or {}).get("style_preferences")

    skus_df = skus_df.copy()
    skus_df["_category_bonus"] = skus_df.apply(
        lambda row: category_occasion_bonus(row, intent, s.category_match_bonus, s.occasion_match_bonus),
        axis=1,
    )
    skus_df["_personalization_bonus"] = skus_df.apply(
        lambda row: personalization_bonus(row, style_preferences, s.personalization_weight),
        axis=1,
    )
    skus_df["_discount_bonus"] = skus_df["discount_percentage"] * s.discount_weight

    skus_df["_composite_score"] = (
        skus_df["_relevance_score"]
        + skus_df["_category_bonus"]
        + skus_df["_personalization_bonus"]
        + skus_df["_discount_bonus"]
    )
    return skus_df


def rank_skus(
    candidates: List[Dict[str, Any]],
    intent: Dict[str, Any],
    customer_profile: Optional[Dict[str, Any]],
    inventory_df: pd.DataFrame,
    top_k: Optional[int] = None,
    scoring_settings: Optional[ScoringSettings] = None,
) -> List[Dict[str, Any]]:
    """Orchestrates the full Stage 3 pipeline: join -> in-stock filter ->
    size filter -> composite score -> sort -> top-k.

    Returns a list of plain dicts (JSON-serializable, matching GraphState's
    `filtered_skus` field) — empty list if nothing survives the filters,
    which is a real, expected outcome (e.g. no stock in the customer's
    exact size), not an error condition.
    """
    s = scoring_settings or default_settings.scoring
    resolved_top_k = top_k if top_k is not None else s.final_top_k
    size_profile = (customer_profile or {}).get("size_profile")

    skus_df = join_candidates_with_inventory(candidates, inventory_df)
    skus_df = filter_in_stock(skus_df)
    skus_df = filter_by_size(skus_df, size_profile)
    skus_df = compute_composite_score(skus_df, intent, customer_profile, s)

    if skus_df.empty:
        return []

    ranked = skus_df.sort_values("_composite_score", ascending=False).head(resolved_top_k)
    return ranked.to_dict(orient="records")

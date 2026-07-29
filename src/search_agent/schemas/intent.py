"""
Structured-output contract for Stage 1 (Intent Extraction).

`QueryIntent` is the shape we force the LLM to respond in (via Anthropic
tool-use / forced function-calling — see utils/llm_client.py). Keeping the
enums aligned with the actual vocab in products.csv (category, occasion,
material) means Stage 3's relational join can filter directly on these
fields with no fuzzy-matching required.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# NOTE: kept in sync with the vocab banks used by the synthetic data generator
# (CATEGORIES / OCCASIONS / MATERIALS_BY_CATEGORY). If the catalog's vocab
# changes, update these Literals too — that's what keeps structured output
# reliable instead of "reliable most of the time".
Category = Literal["Outerwear", "Footwear", "Tops", "Bottoms", "Accessories","Both top and Bottom", "Complete Outfit"]

Occasion = Literal[
    "Casual", "Formal", "Wedding", "Outdoor", "Activewear", "Business", "Travel"
]

WeatherAttribute = Literal[
    "waterproof", "water-resistant", "weatherproof", "windproof", "insulated", "quick-drying"
]


class QueryIntent(BaseModel):
    """Parsed representation of a free-text customer search query.

    All fields except `raw_query` and `keywords` are optional: a query like
    "something for a wedding" should populate `occasion` and leave everything
    else `None` rather than force the model to hallucinate a category.
    """

    raw_query: str = Field(..., description="The original, unmodified user query.")

    category: Optional[Category] = Field(
        default=None,
        description="Product category implied by the query, if any (must match the catalog's category vocabulary).",
    )
    occasion: Optional[Occasion] = Field(
        default=None,
        description="Occasion/use-case implied by the query, if any.",
    )
    weather_attribute: Optional[WeatherAttribute] = Field(
        default=None,
        description="Weather/performance attribute implied by the query (e.g. 'waterproof'), if any.",
    )
    material: Optional[str] = Field(
        default=None,
        description="Specific material mentioned or strongly implied (e.g. 'Gore-Tex', 'leather'), if any.",
    )
    size: Optional[str] = Field(
        default=None,
        description="Size mentioned by the customer (letter size, waist size, or shoe size), if any.",
    )
    color: Optional[str] = Field(
        default=None,
        description="Color mentioned or strongly implied, if any.",
    )
    max_price: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum price the customer is willing to pay, if mentioned (e.g. 'under $150' -> 150.0).",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description=(
            "Any other descriptive, subjective, or contextual terms from the query that don't fit "
            "the structured fields above (e.g. 'stylish', 'for my mom', 'coastal'). Used as a fallback "
            "text-search signal in Stage 2."
        ),
    )

    model_config = ConfigDict(extra="forbid")

"""
Prompt bank for Stage 4 — Synthesis.

Wording matches what was proven out in `stage4_exploration.ipynb` (section
5): explicitly constrained to not invent products/prices, and explicitly
told what to say when the results list is empty rather than left to guess.
"""

import json
from typing import Any, Dict, List, Optional

SYNTHESIS_SYSTEM_PROMPT = """\
You are a friendly, knowledgeable retail shopping assistant. You'll be given
a customer's original search query, a short ranked list of matching products
(with size, price, and discount info), and optionally the customer's name and
style preferences.

Write a brief, warm, natural-language response (2-4 sentences) that:
- Directly addresses what they searched for
- Highlights the top 1-3 results by name, mentioning price and any discount
- If the provided list is empty, say so honestly and suggest trying a
  different size or checking back later — don't pretend there are results
- Do NOT invent products, prices, or details not present in the provided list
"""


def build_synthesis_user_prompt(
    raw_query: str,
    filtered_skus: List[Dict[str, Any]],
    customer_profile: Optional[Dict[str, Any]],
) -> str:
    """Builds the user-turn prompt: original query + top few SKUs (as JSON,
    trimmed to the fields the model actually needs) + customer context.
    """
    customer_line = (
        f"Customer: {customer_profile['name']}, style preferences: {customer_profile['style_preferences']}"
        if customer_profile
        else "Customer: guest (no profile)"
    )
    skus_summary = [
        {
            "name": s["name"],
            "size": s["size"],
            "price": s["base_price"],
            "discount_percentage": s["discount_percentage"],
        }
        for s in filtered_skus[:5]
    ]
    return (
        f'Original query: "{raw_query}"\n'
        f"{customer_line}\n"
        f"Top matching results (JSON): {json.dumps(skus_summary)}\n\n"
        "Write the customer-facing response."
    )


NO_RESULTS_RESPONSE = (
    "We couldn't find anything matching in your size right now — "
    "try a different size or check back soon."
)

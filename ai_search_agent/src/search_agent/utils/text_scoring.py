"""
Custom TF-IDF-style relevance scoring, pure standard library (no sklearn),
per the original spec for this MVP.

Pipeline: tokenize each product's searchable text -> build an IDF table over
the whole catalog -> for a given query's terms, score each product by
summing (term frequency in that product) * (IDF of that term) across the
query's terms -> rank descending.

This intentionally lives outside `nodes/` — it's pure, side-effect-free
scoring logic with no dependency on GraphState, so it can be unit-tested
and reused (e.g. Stage 3 may want to re-score a smaller candidate set)
without importing anything graph-related.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")  # keeps hyphenated terms like "water-resistant" whole


def tokenize(text: str) -> List[str]:
    """Lowercases and splits text into word tokens, keeping hyphenated
    compounds (e.g. "water-resistant", "quick-drying") as single tokens
    since that's how they appear in both the catalog vocab and QueryIntent's
    weather_attribute enum.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def build_document_text(product: Dict[str, Any]) -> str:
    """Concatenates a product's searchable fields into one scoring string."""
    fields = (
        product.get("name", ""),
        product.get("description", ""),
        product.get("category", ""),
        product.get("material", ""),
        product.get("occasion", ""),
    )
    return " ".join(f for f in fields if f)


def build_query_terms(intent: Dict[str, Any]) -> List[str]:
    """Flattens a QueryIntent dict into a list of search tokens.

    Structured fields (category/material/occasion/weather_attribute/color)
    and free-text `keywords` are all treated as equally-weighted query terms
    here — the IDF weighting in `score_product` is what naturally makes rare,
    specific terms (e.g. "waterproof") count for more than common ones
    (e.g. "casual").
    """
    terms: List[str] = []
    for field_name in ("category", "material", "occasion", "weather_attribute", "color"):
        value = intent.get(field_name)
        if value:
            terms.extend(tokenize(value))

    for keyword in intent.get("keywords") or []:
        terms.extend(tokenize(keyword))

    return terms


def compute_idf(document_tokens: List[List[str]]) -> Dict[str, float]:
    """Standard smoothed IDF over the corpus: idf(t) = ln(N / (1 + df(t))) + 1.

    The +1 smoothing avoids divide-by-zero and keeps every term's IDF
    positive, so a term appearing in every document still contributes a
    small positive score rather than zeroing everything out.
    """
    n_docs = len(document_tokens)
    doc_freq: Counter = Counter()
    for tokens in document_tokens:
        doc_freq.update(set(tokens))


    return {
        term: math.log(n_docs / (1 + df)) + 1
        for term, df in doc_freq.items()
    }


def score_product(query_terms: List[str], doc_tokens: List[str], idf: Dict[str, float]) -> float:
    """TF-IDF relevance score: sum over query terms of (term frequency in
    this document) * (that term's corpus-wide IDF). Terms with zero IDF
    weight (unseen in the corpus) contribute nothing, which is the desired
    behavior rather than an error.
    """
    if not query_terms or not doc_tokens:
        return 0.0

    term_frequency = Counter(doc_tokens)
    return sum(term_frequency[term] * idf.get(term, 0.0) for term in query_terms)


def rank_products(
    products: List[Dict[str, Any]],
    intent: Dict[str, Any],
    top_k: int = 20,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """Scores every product against the query intent and returns the top-k.

    Each returned product dict is a shallow copy of the original with an
    added `_relevance_score` field (useful for debugging/tests and for
    Stage 3, which may want to factor this score into its composite ranking).
    """
    query_terms = build_query_terms(intent)
    doc_tokens_by_product = [tokenize(build_document_text(p)) for p in products]
    idf = compute_idf(doc_tokens_by_product)

    scored = []
    for product, doc_tokens in zip(products, doc_tokens_by_product):
        score = score_product(query_terms, doc_tokens, idf)
        if score > min_score:
            scored.append({**product, "_relevance_score": score})

    scored.sort(key=lambda p: p["_relevance_score"], reverse=True)
    return scored[:top_k]

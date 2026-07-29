"""
Streamlit frontend for the AI search agent — talks to the FastAPI backend
(`app.py`) over HTTP, no direct import of `search_agent` needed here. Keeps
the frontend fully decoupled from the graph/LLM code, so it can be deployed
as its own container hitting the API over the network (see docker-compose.yml).

Run standalone (backend already running separately):
    export BACKEND_URL=http://localhost:8000
    streamlit run streamlit_app.py

Run via Docker Compose: `docker compose up` — BACKEND_URL is set there to
the API service's container name.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 60

st.set_page_config(page_title="AI Search Agent", page_icon="🔎", layout="wide")


def check_backend_health() -> bool:
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def call_search_api(query: str, customer_id: Optional[str], thread_id: Optional[str]) -> Dict[str, Any]:
    payload = {"query": query, "customer_id": customer_id or None, "thread_id": thread_id or None}
    response = requests.post(f"{BACKEND_URL}/search", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def render_skus_table(filtered_skus: List[Dict[str, Any]]) -> None:
    if not filtered_skus:
        st.info("No ranked SKUs to show.")
        return

    rows = [
        {
            "SKU": s.get("sku_id"),
            "Product": s.get("name"),
            "Category": s.get("category"),
            "Size": s.get("size"),
            "Price": s.get("base_price"),
            "Discount": s.get("discount_percentage"),
            "Stock": s.get("stock_count"),
            "Score": round(s.get("_composite_score", 0), 2) if s.get("_composite_score") is not None else None,
        }
        for s in filtered_skus
    ]
    st.dataframe(rows, width='stretch', hide_index=True)


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("Session")

    backend_ok = check_backend_health()
    if backend_ok:
        st.success(f"Backend reachable at {BACKEND_URL}")
    else:
        st.error(f"Backend not reachable at {BACKEND_URL}")

    customer_id = st.text_input(
        "Customer ID (optional)",
        placeholder="e.g. CUST-001",
        help="Leave blank for a guest session — results won't be filtered by size or personalized.",
    )

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None

    persist_thread = st.checkbox(
        "Persist conversation (checkpointing)",
        value=False,
        help="If checked, reuses the same thread_id across searches in this session, "
        "so the graph's LangGraph checkpointer accumulates state for this conversation.",
    )
    if not persist_thread:
        st.session_state.thread_id = None

    if st.session_state.thread_id:
        st.caption(f"thread_id: `{st.session_state.thread_id}`")

    if st.button("Reset session"):
        st.session_state.thread_id = None
        st.session_state.pop("last_result", None)
        st.rerun()

# --- Main ----------------------------------------------------------------------
st.title("🔎 AI Search Agent")
st.caption("Ask for a product the way you would describe it to a person.")

query = st.text_input(
    "What are you looking for?",
    placeholder="e.g. waterproof jacket for an October coastal wedding",
)
submitted = st.button("Search", type="primary", disabled=not backend_ok)

if submitted and query.strip():
    with st.spinner("Searching..."):
        try:
            result = call_search_api(query.strip(), customer_id.strip() or None, st.session_state.thread_id)
            st.session_state.last_result = result
            if persist_thread:
                st.session_state.thread_id = result.get("thread_id")
        except requests.RequestException as exc:
            st.error(f"Request to backend failed: {exc}")
            st.session_state.pop("last_result", None)

result = st.session_state.get("last_result")
if result:
    st.subheader("Response")
    st.write(result["final_response"])

    if result.get("errors"):
        with st.expander("⚠️ Non-fatal issues encountered", expanded=False):
            for err in result["errors"]:
                st.warning(err)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ranked results")
        render_skus_table(result.get("filtered_skus") or [])

    with col2:
        st.subheader("Extracted intent")
        st.json(result.get("intent") or {})

        st.subheader("Customer profile")
        st.json(result.get("customer_profile") or {"note": "guest session"})
elif submitted and not query.strip():
    st.warning("Enter a search query first.")

import os
import requests
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ---------- Load environment ----------
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-minilm")

TOP_K = int(os.getenv("TOP_K", "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.1"))

# ---------- Helpers ----------
def get_ollama_embedding(text: str) -> np.ndarray:
    """Call Ollama embeddings API."""
    url = f"{OLLAMA_URL}/api/embeddings"
    try:
        r = requests.post(url, json={"model": EMBEDDING_MODEL, "prompt": text}, timeout=30)
        r.raise_for_status()
        data = r.json()
        emb = np.array(data["embedding"], dtype=np.float32)
        # normalize for cosine (Qdrant uses raw vectors; normalized input makes cosine stable)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    except Exception as e:
        raise RuntimeError(f"Ollama embedding error: {e}")

@st.cache_resource(show_spinner=False)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def qdrant_search(query: str, filters: Dict[str, Any] | None = None, limit: int = TOP_K):
    """Search Qdrant using an Ollama-derived query embedding."""
    emb = get_ollama_embedding(query)
    client = get_qdrant_client()

    q_filter = None
    if filters:
        conds = []
        for k, v in filters.items():
            conds.append(FieldCondition(key=k, match=MatchValue(value=v)))
        q_filter = Filter(must=conds)

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=emb.tolist(),
        limit=limit,
        score_threshold=SCORE_THRESHOLD,
        query_filter=q_filter,
    )
    out = []
    for r in results:
        payload = r.payload or {}
        out.append({
            "text": payload.get("text", ""),
            "source": payload.get("source", ""),
            "chunk_index": payload.get("chunk_index", 0),
            "score": r.score,
        })
    return out

def build_answer_from_hits(hits: List[Dict[str, Any]]) -> str:
    """Very simple answer synthesis: return the best snippet + cite others."""
    if not hits:
        return "No relevant chunks found."

    best = hits[0]
    answer = best["text"].strip()
    # Append tiny citations
    cites = []
    for h in hits[:3]:
        cites.append(f"{h['source']}#chunk{h['chunk_index']} (score {h['score']:.3f})")
    citation_line = "Sources: " + " | ".join(cites)
    return f"{answer}\n\n_{citation_line}_"

# ---------- UI ----------
st.set_page_config(page_title="Qdrant RAG Chat", page_icon="🔎", layout="wide")
st.title("🔎 Qdrant RAG Chat (Ollama + Qdrant)")

with st.sidebar:
    st.subheader("Settings")
    st.caption("These are read from .env; you can override here at runtime.")
    col_name = st.text_input("Collection", value=COLLECTION_NAME, help="Qdrant collection to query")
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=TOP_K, step=1)
    score_thr = st.number_input("Score threshold", min_value=0.0, max_value=1.0, value=SCORE_THRESHOLD, step=0.01)
    model_name = st.text_input("Ollama Model", value=EMBEDDING_MODEL)
    st.caption("Optional payload filters (exact match):")
    filter_key = st.text_input("Filter key (payload key)", value="")
    filter_val = st.text_input("Filter value", value="")

# allow sidebar overrides (in-memory only)
COLLECTION_NAME = col_name
TOP_K = int(top_k)
SCORE_THRESHOLD = float(score_thr)
EMBEDDING_MODEL = model_name

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_query = st.chat_input("Ask anything about your documents…")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Run search
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                filters = {filter_key: filter_val} if filter_key and filter_val else None
                hits = qdrant_search(user_query, filters=filters, limit=TOP_K)
                answer = build_answer_from_hits(hits)
            except Exception as e:
                answer = f"Error: {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Footer health check
with st.expander("Health"):
    st.write("**Qdrant URL:**", QDRANT_URL)
    st.write("**Ollama URL:**", OLLAMA_URL)
    st.write("**Collection:**", COLLECTION_NAME)

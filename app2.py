import os
import json
import time
import requests
import numpy as np
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ========= Load env =========
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-minilm")

TOP_K = int(os.getenv("TOP_K", "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.1"))

# Groq (optional)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
USE_GROQ_DEFAULT = os.getenv("USE_GROQ", "true").lower() in ("1", "true", "yes")

# ========= Helpers =========
def get_ollama_embedding(text: str) -> np.ndarray:
    """Call Ollama embeddings API and return a normalized vector."""
    url = f"{OLLAMA_URL}/api/embeddings"
    r = requests.post(url, json={"model": EMBEDDING_MODEL, "prompt": text}, timeout=60)
    r.raise_for_status()
    data = r.json()
    emb = np.array(data["embedding"], dtype=np.float32)
    # Normalize for cosine stability
    n = np.linalg.norm(emb)
    if n > 0:
        emb = emb / n
    return emb

@st.cache_resource(show_spinner=False)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def make_filter(k: Optional[str], v: Optional[str]) -> Optional[Filter]:
    if not k or not v:
        return None
    return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v))])

def qdrant_search(query: str, limit: int, score_threshold: float,
                  filter_k: Optional[str] = None, filter_v: Optional[str] = None) -> List[Dict[str, Any]]:
    emb = get_ollama_embedding(query)
    client = get_qdrant_client()
    q_filter = make_filter(filter_k, filter_v)

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=emb.tolist(),
        limit=limit,
        score_threshold=score_threshold,
        query_filter=q_filter
    )
    hits = []
    for r in results:
        payload = r.payload or {}
        hits.append({
            "text": payload.get("text", ""),
            "source": payload.get("source", ""),
            "chunk_index": payload.get("chunk_index", 0),
            "score": float(r.score),
        })
    return hits

def format_context(hits: List[Dict[str, Any]], max_chars: int = 900) -> str:
    """Compact, numbered context with source and chunk, truncated for token safety."""
    lines = []
    for i, h in enumerate(hits, 1):
        txt = (h.get("text") or "").replace("\n", " ").strip()
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "..."
        src = h.get("source", "unknown")
        idx = h.get("chunk_index", 0)
        lines.append(f"[{i}] ({src}#chunk{idx}) {txt}")
    return "\n".join(lines)

def groq_generate_answer(query: str, hits: List[Dict[str, Any]]) -> str:
    if not GROQ_API_KEY:
        return "Groq not configured: set GROQ_API_KEY in .env."

    context = format_context(hits)
    system_prompt = (
        "You are a concise assistant. Answer strictly using ONLY the provided context. "
        "If the answer is not in the context, say you don't know. "
        "Cite snippets with bracketed indices like [1], [2]."
    )
    user_prompt = f"Question: {query}\n\nContext (numbered snippets):\n{context}\n\nAnswer:"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    if resp.status_code >= 400:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        return f"Groq error ({resp.status_code}): {err}"
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

def fallback_answer(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "No relevant chunks found."
    best = hits[0]
    cites = []
    for i, h in enumerate(hits[:3], 1):
        cites.append(f"[{i}] {h['source']}#chunk{h['chunk_index']} (score {h['score']:.3f})")
    return f"{best['text'].strip()}\n\n_Sources: {' | '.join(cites)}_"

# ========= UI =========
st.set_page_config(page_title="Qdrant RAG Chat", page_icon="🔎", layout="wide")
st.title("🔎 Qdrant RAG Chat (Ollama embeddings, optional Groq generation)")

with st.sidebar:
    st.subheader("Settings")
    COLLECTION_NAME = st.text_input("Collection name", value=COLLECTION_NAME)
    EMBEDDING_MODEL = st.text_input("Ollama embedding model", value=EMBEDDING_MODEL, help="e.g., all-minilm")
    TOP_K = st.number_input("Top K", 1, 20, value=TOP_K, step=1)
    SCORE_THRESHOLD = st.number_input("Score threshold", 0.0, 1.0, value=SCORE_THRESHOLD, step=0.01)

    st.markdown("---")
    st.caption("Optional exact-match payload filter (e.g., limit to one PDF)")
    filter_k = st.text_input("Filter key", value="")
    filter_v = st.text_input("Filter value", value="")

    st.markdown("---")
    use_groq = st.checkbox("Use Groq for answer generation", value=USE_GROQ_DEFAULT)
    if use_groq:
        GROQ_MODEL = st.text_input("Groq model", value=GROQ_MODEL)
        MAX_TOKENS = st.number_input("Max tokens", 128, 4096, value=MAX_TOKENS, step=64)
        if not GROQ_API_KEY:
            st.warning("GROQ_API_KEY missing in .env; generation will fail until set.")

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
user_query = st.chat_input("Ask about your documents…")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve + generate
    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            try:
                hits = qdrant_search(
                    user_query,
                    limit=int(TOP_K),
                    score_threshold=float(SCORE_THRESHOLD),
                    filter_k=filter_k or None,
                    filter_v=filter_v or None
                )
            except Exception as e:
                st.error(f"Search error: {e}")
                hits = []

        if hits:
            if use_groq:
                with st.spinner("Synthesizing answer…"):
                    answer = groq_generate_answer(user_query, hits)
            else:
                answer = fallback_answer(hits)
        else:
            answer = "No relevant chunks found."

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Health panel
with st.expander("Health / Debug"):
    st.write("**Qdrant URL**:", QDRANT_URL)
    st.write("**Collection**:", COLLECTION_NAME)
    st.write("**Ollama URL**:", OLLAMA_URL)
    st.write("**Embedding model**:", EMBEDDING_MODEL)
    st.write("**Use Groq**:", use_groq)

"""
config.py — Shared configuration and client initialisation.

Memory architecture (Letta OS):
  Core Memory     → Letta blocks    (always in context: persona + human state)
  Archival Memory → Letta passages  (replaces Mem0 + Qdrant mem0 collection + Neo4j)
  Recall Memory   → JSON file store (memory/sessions_data.json — no external service)

Qdrant is still used for the RAG knowledge_base (uploaded documents).
"""

import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# ── Xazna LLM (OpenAI-compatible) ────────────────────────────────────────────
LLM_BASE_URL = "https://ai.xazna.uz/llm/v1"
LLM_API_KEY  = "sk-raximberdi-cmF4aW1iZXJkaQ"

llm_client = AsyncOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

# Model ID resolved at startup via initialize() in agent.py
MODEL_ID: str = ""

# ── Letta OS configuration ────────────────────────────────────────────────────
# The Letta server runs in Docker (port 8283) and provides Core + Archival memory.
# Recall Memory is handled locally (memory/session.py → sessions_data.json).
LETTA_BASE_URL    = os.getenv("LETTA_BASE_URL",    "http://localhost:8283")
LETTA_SERVER_PASS = os.getenv("LETTA_SERVER_PASS", "")

# ── Qdrant (RAG knowledge_base only) ─────────────────────────────────────────
# Used for document uploads (PDF/TXT/MD → Qdrant knowledge_base collection).
# The mem0 collection has been removed — LTM now lives in Letta Archival Memory.
QDRANT_HOST    = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT    = int(os.getenv("QDRANT_PORT", "6334"))
RAG_COLLECTION = "knowledge_base"
EMBED_MODEL    = "/models/embedding"

# Ensure the RAG collection exists at the correct 2048-dim configuration.
try:
    _qc      = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    _existing = {c.name: c for c in _qc.get_collections().collections}
    if RAG_COLLECTION not in _existing:
        _qc.create_collection(
            RAG_COLLECTION,
            vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
        )
    elif _qc.get_collection(RAG_COLLECTION).config.params.vectors.size != 2048:
        _qc.delete_collection(RAG_COLLECTION)
        _qc.create_collection(
            RAG_COLLECTION,
            vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
        )
    _qc.close()
    del _qc, _existing
except Exception as _e:
    print(
        f"[config] WARNING: Qdrant unavailable ({_e})"
        " — RAG document search will fail until Qdrant is running."
    )

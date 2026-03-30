"""
RAG (Retrieval-Augmented Generation) engine
Uses Gemini v1 REST API directly - bypasses SDK version issues
"""
import json
import os
import numpy as np
import requests
from pathlib import Path
from typing import Optional

KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "data" / "knowledge_base.json"
TOP_K = 8
BASE = "https://generativelanguage.googleapis.com"
EMBED_CANDIDATES = ["gemini-embedding-001", "gemini-embedding-2-preview"]
CHAT_URL = f"{BASE}/v1beta/models/gemini-2.0-flash:generateContent"

_knowledge_base: list[dict] = []
_embeddings_matrix: Optional[np.ndarray] = None
_api_key: str = ""
_embed_url: str = ""


def _detect_embed_url() -> str:
    for version in ("v1", "v1beta"):
        for model in EMBED_CANDIDATES:
            url = f"{BASE}/{version}/models/{model}:embedContent"
            try:
                resp = requests.post(
                    url, params={"key": _api_key},
                    json={"model": f"models/{model}", "content": {"parts": [{"text": "test"}]}},
                    timeout=10
                )
                if resp.status_code == 200:
                    print(f"[RAG] Embedding: {model} ({version})")
                    return url
            except Exception:
                pass
    raise RuntimeError("No working Gemini embedding model found. Check your API key.")


def init():
    global _api_key, _embed_url
    _api_key = os.environ.get("GEMINI_API_KEY", "")
    if not _api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    _embed_url = _detect_embed_url()
    _load_knowledge_base()
    print(f"[RAG] Loaded {len(_knowledge_base)} chunks from knowledge base")


def _load_knowledge_base():
    global _knowledge_base, _embeddings_matrix
    if not KNOWLEDGE_BASE_PATH.exists():
        print("[RAG] Warning: knowledge_base.json not found")
        _knowledge_base = []
        _embeddings_matrix = None
        return
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    _knowledge_base = data.get("chunks", [])
    if _knowledge_base:
        _embeddings_matrix = np.array([c["embedding"] for c in _knowledge_base], dtype=np.float32)
        norms = np.linalg.norm(_embeddings_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        _embeddings_matrix = _embeddings_matrix / norms


def _embed_query(text: str) -> np.ndarray:
    model_name = _embed_url.split("/models/")[1].split(":")[0]
    payload = {
        "model": f"models/{model_name}",
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_QUERY"
    }
    resp = requests.post(_embed_url, params={"key": _api_key}, json=payload, timeout=30)
    resp.raise_for_status()
    vec = np.array(resp.json()["embedding"]["values"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _retrieve(query_vec: np.ndarray, top_k: int = TOP_K) -> list[dict]:
    if _embeddings_matrix is None or not _knowledge_base:
        return []
    scores = _embeddings_matrix @ query_vec
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        chunk = _knowledge_base[idx].copy()
        chunk["score"] = float(scores[idx])
        chunk.pop("embedding", None)
        results.append(chunk)
    return results


SYSTEM_PROMPT = """אתה עוזר מקצועי לעובדי רשות המיסים בישראל, המתמחים בטיפול בהחזרי מס.
תפקידך לענות על שאלות מקצועיות בנושאי מס הכנסה בעברית בלבד.

כללים:
1. ענה אך ורק על בסיס המידע שסופק לך בהקשר.
2. אם המידע אינו מספיק לתשובה מלאה, ציין זאת בבירור.
3. ציין את מקור המידע (שם המסמך, חוזר, סעיף חוק) בסוף התשובה.
4. השתמש בשפה מקצועית וברורה.
5. אם שאלה נוגעת לפקודת מס הכנסה, ציין את הסעיף הרלוונטי.
6. אל תמציא מידע שאינו מופיע בהקשר."""


def answer_question(question: str) -> dict:
    if not _knowledge_base:
        return {
            "answer": "מאגר הידע טרם נטען. אנא פנה למנהל המערכת.",
            "sources": [],
            "has_knowledge": False
        }

    query_vec = _embed_query(question)
    chunks = _retrieve(query_vec)
    relevant = [c for c in chunks if c["score"] > 0.3] or chunks[:3]

    context_parts = []
    sources = []
    seen = set()
    for chunk in relevant:
        context_parts.append(f"--- {chunk.get('source', 'מקור לא ידוע')} ---\n{chunk['text']}")
        src = chunk.get("source", "")
        if src and src not in seen:
            seen.add(src)
            sources.append({"name": src, "url": chunk.get("url", "")})

    prompt = f"""הקשר (מידע רלוונטי ממאגר הידע):
{chr(10).join(context_parts)}

שאלת עובד: {question}

ענה על השאלה בעברית על בסיס ההקשר לעיל:"""

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt}]}]
    }
    resp = requests.post(CHAT_URL, params={"key": _api_key}, json=payload, timeout=60)
    resp.raise_for_status()
    answer = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    return {"answer": answer, "sources": sources, "has_knowledge": True}

"""
RAG (Retrieval-Augmented Generation) engine
Uses Gemini v1 REST API directly - bypasses SDK version issues
"""
import base64
import json
import os
import time
import numpy as np
import requests
from pathlib import Path
from typing import Optional

KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "data" / "knowledge_base.json"
TOP_K = 8
BASE = "https://generativelanguage.googleapis.com"

EMBED_CANDIDATES = ["gemini-embedding-001", "gemini-embedding-2-preview", "embedding-001"]
CHAT_CANDIDATES = [
    "gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-001",
    "gemini-1.5-pro", "gemini-pro",
]

_knowledge_base: list[dict] = []
_embeddings_matrix: Optional[np.ndarray] = None
_api_key: str = ""
_embed_url: str = ""
_chat_url: str = ""
_available_models: list[str] = []
_model_methods: dict = {}


def _list_models() -> tuple[list[str], dict]:
    """Fetch available models. Returns (name_list, methods_dict)."""
    for version in ("v1beta", "v1"):
        try:
            resp = requests.get(
                f"{BASE}/{version}/models",
                params={"key": _api_key},
                timeout=15
            )
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                names = [m["name"].split("/")[-1] for m in models]
                methods = {
                    m["name"].split("/")[-1]: m.get("supportedGenerationMethods", [])
                    for m in models
                }
                print(f"[RAG] Found {len(names)} models via {version}")
                return names, methods
        except Exception as e:
            print(f"[RAG] ListModels failed ({version}): {e}")
    return [], {}


def _detect_embed_url(available: list[str], methods: dict) -> str:
    for model in EMBED_CANDIDATES:
        if model in available:
            if "embedContent" in methods.get(model, []) or not methods:
                url = f"{BASE}/v1beta/models/{model}:embedContent"
                print(f"[RAG] Embedding model: {model}")
                return url
    # Fallback: test each candidate
    for model in EMBED_CANDIDATES:
        for version in ("v1", "v1beta"):
            url = f"{BASE}/{version}/models/{model}:embedContent"
            try:
                resp = requests.post(
                    url, params={"key": _api_key},
                    json={"model": f"models/{model}", "content": {"parts": [{"text": "test"}]}},
                    timeout=15
                )
                if resp.status_code == 200:
                    print(f"[RAG] Embedding model (tested): {model} ({version})")
                    return url
            except Exception:
                pass
    raise RuntimeError("No working Gemini embedding model found.")


def _detect_chat_url(available: list[str], methods: dict) -> str:
    # Allow manual override via env var
    override = os.environ.get("GEMINI_CHAT_MODEL", "")
    if override:
        url = f"{BASE}/v1beta/models/{override}:generateContent"
        print(f"[RAG] Chat model (env override): {override}")
        return url

    # Use supportedGenerationMethods from ListModels - no test calls needed
    for model in CHAT_CANDIDATES:
        if model in available:
            model_methods = methods.get(model, [])
            if "generateContent" in model_methods or not model_methods:
                url = f"{BASE}/v1beta/models/{model}:generateContent"
                print(f"[RAG] Chat model: {model}")
                return url

    # Last resort: pick first available model that has generateContent
    for name, model_methods in methods.items():
        if "generateContent" in model_methods and "embedding" not in name.lower():
            url = f"{BASE}/v1beta/models/{name}:generateContent"
            print(f"[RAG] Chat model (fallback): {name}")
            return url

    raise RuntimeError("No working Gemini chat model found.")


def init():
    global _api_key, _embed_url, _chat_url, _available_models
    _api_key = os.environ.get("GEMINI_API_KEY", "")
    if not _api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    # Load knowledge base first (no API needed)
    _load_knowledge_base()
    print(f"[RAG] Loaded {len(_knowledge_base)} chunks from knowledge base")

    # Get available models list (1 API call, no quota used)
    _available_models, _model_methods = _list_models()

    # Detect working models using supportedGenerationMethods (no test calls)
    try:
        _embed_url = _detect_embed_url(_available_models, _model_methods)
    except RuntimeError as e:
        print(f"[RAG] WARNING: {e}")

    try:
        _chat_url = _detect_chat_url(_available_models, _model_methods)
    except RuntimeError as e:
        print(f"[RAG] WARNING: {e}")

    print(f"[RAG] embed_url={_embed_url}")
    print(f"[RAG] chat_url={_chat_url}")


def get_debug_info() -> dict:
    return {
        "chunks_loaded": len(_knowledge_base),
        "embed_url": _embed_url,
        "chat_url": _chat_url,
        "available_models": _available_models,
        "embed_ready": bool(_embed_url),
        "chat_ready": bool(_chat_url),
    }


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
        _rebuild_matrix()


def _rebuild_matrix():
    global _embeddings_matrix
    if not _knowledge_base:
        _embeddings_matrix = None
        return

    def _decode(e):
        if isinstance(e, str):
            return np.frombuffer(base64.b64decode(e), dtype=np.float16).astype(np.float32)
        return np.array(e, dtype=np.float32)

    mat = np.array([_decode(c["embedding"]) for c in _knowledge_base], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    _embeddings_matrix = mat / norms


def add_preembedded_chunks(chunks: list[dict]):
    if not chunks:
        return
    for chunk in chunks:
        _knowledge_base.append(chunk)
    _rebuild_matrix()
    print(f"[RAG] Added {len(chunks)} uploaded chunks from DB")


def embed_and_add_chunks(chunks: list[dict]) -> list[dict]:
    embedded = []
    for i, chunk in enumerate(chunks):
        vec_raw = _embed_doc(chunk["text"])
        arr = np.array(vec_raw, dtype=np.float16)
        c = dict(chunk)
        c["embedding"] = base64.b64encode(arr.tobytes()).decode("ascii")
        embedded.append(c)
        if i > 0 and i % 10 == 0:
            time.sleep(0.5)
    for chunk in embedded:
        _knowledge_base.append(chunk)
    _rebuild_matrix()
    return embedded


def _embed_doc(text: str) -> list[float]:
    model_name = _embed_url.split("/models/")[1].split(":")[0]
    payload = {
        "model": f"models/{model_name}",
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_DOCUMENT"
    }
    resp = requests.post(_embed_url, params={"key": _api_key}, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


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
        return {"answer": "מאגר הידע טרם נטען. אנא פנה למנהל המערכת.", "sources": [], "has_knowledge": False}

    if not _embed_url:
        return {"answer": "שגיאת הגדרה: מודל ה-embedding לא זוהה. בדוק את הלוגים.", "sources": [], "has_knowledge": False}

    if not _chat_url:
        return {"answer": "שגיאת הגדרה: מודל השיחה לא זוהה. בדוק את הלוגים.", "sources": [], "has_knowledge": False}

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
    resp = requests.post(_chat_url, params={"key": _api_key}, json=payload, timeout=60)
    resp.raise_for_status()
    answer = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    return {"answer": answer, "sources": sources, "has_knowledge": True}

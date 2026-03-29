"""
RAG (Retrieval-Augmented Generation) engine
Loads knowledge base from JSON, finds relevant chunks, generates answers via Gemini
"""
import json
import os
import numpy as np
from pathlib import Path
from typing import Optional
import google.generativeai as genai

KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "data" / "knowledge_base.json"
TOP_K = 8  # number of relevant chunks to retrieve

_knowledge_base: list[dict] = []
_embeddings_matrix: Optional[np.ndarray] = None
_model = None
_embed_model = "models/text-embedding-004"
_chat_model = "gemini-1.5-flash"


def init():
    """Initialize Gemini and load knowledge base."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    genai.configure(api_key=api_key)
    _load_knowledge_base()
    print(f"[RAG] Loaded {len(_knowledge_base)} chunks from knowledge base")


def _load_knowledge_base():
    global _knowledge_base, _embeddings_matrix
    if not KNOWLEDGE_BASE_PATH.exists():
        print(f"[RAG] Warning: knowledge_base.json not found at {KNOWLEDGE_BASE_PATH}")
        _knowledge_base = []
        _embeddings_matrix = None
        return
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    _knowledge_base = data.get("chunks", [])
    if _knowledge_base:
        _embeddings_matrix = np.array([c["embedding"] for c in _knowledge_base], dtype=np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(_embeddings_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        _embeddings_matrix = _embeddings_matrix / norms


def _embed_query(text: str) -> np.ndarray:
    result = genai.embed_content(
        model=_embed_model,
        content=text,
        task_type="retrieval_query"
    )
    vec = np.array(result["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _retrieve(query_vec: np.ndarray, top_k: int = TOP_K) -> list[dict]:
    if _embeddings_matrix is None or len(_knowledge_base) == 0:
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
6. אל תמציא מידע שאינו מופיע בהקשר.
"""


def answer_question(question: str) -> dict:
    """
    Main entry point: given a question, retrieve context and generate answer.
    Returns dict with 'answer', 'sources', 'has_knowledge'.
    """
    if not _knowledge_base:
        return {
            "answer": "מאגר הידע טרם נטען. אנא פנה למנהל המערכת.",
            "sources": [],
            "has_knowledge": False
        }

    query_vec = _embed_query(question)
    chunks = _retrieve(query_vec)

    # Filter low-relevance chunks
    relevant = [c for c in chunks if c["score"] > 0.3]
    if not relevant:
        relevant = chunks[:3]  # fallback: take top 3 anyway

    # Build context
    context_parts = []
    sources = []
    seen_sources = set()
    for chunk in relevant:
        context_parts.append(f"--- {chunk.get('source', 'מקור לא ידוע')} ---\n{chunk['text']}")
        src = chunk.get("source", "")
        src_url = chunk.get("url", "")
        if src and src not in seen_sources:
            seen_sources.add(src)
            sources.append({"name": src, "url": src_url})

    context = "\n\n".join(context_parts)

    prompt = f"""הקשר (מידע רלוונטי ממאגר הידע):
{context}

שאלת עובד: {question}

ענה על השאלה בעברית על בסיס ההקשר לעיל:"""

    model = genai.GenerativeModel(
        model_name=_chat_model,
        system_instruction=SYSTEM_PROMPT
    )
    response = model.generate_content(prompt)
    answer = response.text.strip()

    return {
        "answer": answer,
        "sources": sources,
        "has_knowledge": True
    }

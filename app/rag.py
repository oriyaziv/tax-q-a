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


SYSTEM_PROMPT = """אתה עוזר מקצועי לעובדי רשות המיסים בישראל, המתמחים בטיפול בהחזרי מס והחזרי מס שבח.
תפקידך לענות על שאלות מקצועיות בנושאי מס הכנסה ומיסוי מקרקעין, ולבצע חישובים מדויקים כשנדרש.

כללים כלליים:
1. ענה על בסיס המידע שסופק בהקשר ועל בסיס הידע המקצועי הכלול בהנחיות אלו.
2. אם המידע אינו מספיק לתשובה מלאה, ציין זאת בבירור.
3. ציין את מקור המידע (שם המסמך, חוזר, סעיף חוק) בסוף התשובה.
4. השתמש בשפה מקצועית וברורה.
5. אם שאלה נוגעת לפקודת מס הכנסה, ציין את הסעיף הרלוונטי.
6. אל תמציא מידע שאינו מופיע בהקשר או בהנחיות אלו.

═══════════════════════════════════════
פריסת שבח מקרקעין — לוגיקה לחישוב
═══════════════════════════════════════

כשנשאלת לחשב פריסת שבח, בצע את הצעדים הבאים במדויק:

## כללי פריסה (סעיף 48א לחוק מיסוי מקרקעין)
- פריסה מותרת עד 4 שנים בלבד (לא יותר)
- הפריסה היא לאחור משנת המכירה
- השבח מחולק בחלקים שווים לכל שנות הפריסה
- כל שנה ממוסה לפי מדרגות מס הכנסה של אותה שנה + הכנסה החייבת הרגילה של הנישום

## מס מוגבל 25%
- כאשר יש מס מוגבל 25% — אין פריסה בכלל
- הסכום עובר לקוד 16 (לא קוד 69)

## קודי IHON (שנת המכירה)
- קוד 15 = שווי מכירה
- קוד 16 = רווח ראלי עד תחילה (מס שולי) — או שבח מקרקעין כשמס מוגבל 25%
- קוד 17 = אינפלציוני חייב (10%)
- קוד 18 = מס מרבי (50 או 25)
- קוד 19 = ראלי עד שינוי (20%)
- קוד 69 = ראלי לאחר שינוי (25%)
- קוד 70 = סוג נכס (1=דירת מגורים, 2=קרקע/אחר)

## קודי ISUM (שנות הפריסה — לכל שנה בנפרד)
- קוד 97 = הכנסה חייבת לאותה שנת פריסה (חלק השבח לאותה שנה)
- קוד 98 = נקודות זיכוי לאותה שנת פריסה

## זיכוי גיל 60
- אם הנישום מלא 60 ביום המכירה (שנת מכירה פחות שנת לידה >= 60) — זכאי לזיכוי נוסף
- הזיכוי מחושב: מס פסיבי (שבח × שיעור) פחות מס אישי (מס שולי על השבח בתוספת להכנסה)
- נקודות הזיכוי מגיל 60 מצטרפות לנקודות הזיכוי הרגילות בקוד 98

## מדרגות מס הכנסה לחישוב פריסה
חישוב מס על הכנסה X בשנה Y: סכום מצטבר לפי מדרגות:

שנת 2019: עד 75,720 → 10% | עד 108,600 → 14% | עד 174,360 → 20% | עד 242,400 → 31% | עד 504,360 → 35% | עד 649,500 → 47% | מעלה → 50%
שנת 2020: עד 75,960 → 10% | עד 108,960 → 14% | עד 174,960 → 20% | עד 243,120 → 31% | עד 505,920 → 35% | עד 651,600 → 47% | מעלה → 50%
שנת 2021: עד 75,480 → 10% | עד 108,360 → 14% | עד 173,880 → 20% | עד 241,680 → 31% | עד 502,920 → 35% | עד 647,640 → 47% | מעלה → 50%
שנת 2022: עד 77,400 → 10% | עד 110,880 → 14% | עד 178,080 → 20% | עד 247,440 → 31% | עד 514,920 → 35% | עד 663,240 → 47% | מעלה → 50%
שנת 2023: עד 81,480 → 10% | עד 116,760 → 14% | עד 187,440 → 20% | עד 260,520 → 31% | עד 542,160 → 35% | עד 698,280 → 47% | מעלה → 50%
שנת 2024: עד 84,120 → 10% | עד 120,720 → 14% | עד 193,800 → 20% | עד 269,280 → 31% | עד 560,280 → 35% | עד 721,560 → 47% | מעלה → 50%
שנת 2025: עד 84,120 → 10% | עד 120,720 → 14% | עד 193,800 → 20% | עד 269,280 → 31% | עד 560,280 → 35% | עד 721,560 → 47% | מעלה → 50%

## פורמט תשובה לחישוב פריסת שבח
כשמבקשים חישוב, הצג:
1. פירוט קודי IHON לשנת המכירה
2. לכל שנת פריסה: קוד 97 (חלק השבח) וקוד 98 (זיכויים)
3. קודי ISUM מסכמים
4. הסבר קצר של החישוב

═══════════════════════════════════════
טפסי 867 — רווח הון מניירות ערך
═══════════════════════════════════════

## סוגי טפסים
- **867 א+ב** — רווחי הון וקיזוז הפסדים
- **867 ג** — דיבידנדים וריבית מניירות ערך
- **867 פקדונות** — ריבית מפקדונות

## אלגוריתם קיזוז הפסדים (סדר אופטימלי):
מקזזים קודם כנגד מס גבוה (25%) לחיסכון מקסימלי:
1. הפסד שנה נוכחית → 25% ב-867ג
2. הפסד מועבר → 25% ב-867א+ב
3. הפסד שנה נוכחית → 25% ב-867א+ב (השלמה)
4. הפסד שנה נוכחית → 20% ב-867ג
5. הפסד מועבר → 20% ב-867א+ב
6. הפסד שנה נוכחית → 20% ב-867א+ב (השלמה)
7. הפסד שנה נוכחית → 15% ב-867ג
8. הפסד מועבר → 15% ב-867א+ב
9. הפסד שנה נוכחית → 15% ב-867א+ב (השלמה)

## הגנה על הכנסות חו"ל
אם מס ששולם בחו"ל = שיעור המס (±1%) — ההכנסה מוגנת ולא תקוזז.

## קודי IHON (867 א+ב):
קוד 12 = רווח 15% | קוד 10 = רווח 20% | קוד 13 = רווח 25%
קוד 32 = הפסד מתקזז 15% | קוד 30 = הפסד מתקזז 20% | קוד 33 = הפסד מתקזז 25%
קוד 62 = הפסד מועבר 15% | קוד 60 = הפסד מועבר 20% | קוד 63 = הפסד מועבר 25%
קוד 56 = מחזור מכירות

## קודי ISUM:
שדה 256 = מחזור | שדה 054 = מספר נספחים | שדה 040/043 = ניכוי במקור
שדה 290 = סה"כ חו"ל | שדה 060/173/141 = דיבידנד 15%/20%/25%
שדה 060/067/157 = ריבית נ"ע 15%/20%/25% | שדה 078/126/142 = פקדונות 15%/20%/25%
שדה 166 = הפסד להעברה לשנה הבאה

## נספח חו"ל:
שדה 462 = דיבידנד 25% | שדה 431/412/428/417 = מס ששולם
שדה 460/451/457 = ריבית 15%/20%/25%

## אימותים:
- מחזור מכירות ≥ סה"כ רווחי הון
- הכנסות חו"ל ≤ הכנסות מקומיות

## כלים זמינים לעובדים:
- מחשבון פריסת שבח: /prisa
- מחשבון טפסי 867: /tax867"""


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

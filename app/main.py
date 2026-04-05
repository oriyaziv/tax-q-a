"""
Tax Q&A - FastAPI application
Knowledge portal for Israeli Tax Authority employees
"""
import io
import os
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

import base64
import requests

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import PyPDF2
from docx import Document

import app.rag as rag

DB_PATH = Path(__file__).parent.parent / "data" / "chat_history.db"
STATIC_PATH = Path(__file__).parent.parent / "static"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MAX_CHUNKS_PER_UPLOAD = 120


# ─── DATABASE ───

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            source TEXT,
            url TEXT,
            embedding TEXT NOT NULL,
            uploaded_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_to_history(question: str, answer: str, sources: list):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO chat_history (question, answer, sources, created_at) VALUES (?, ?, ?, ?)",
            (question, answer, json.dumps(sources, ensure_ascii=False), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Error saving history: {e}")


def get_history(limit: int = 50) -> list:
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT id, question, answer, sources, created_at FROM chat_history ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "question": r[1],
                "answer": r[2],
                "sources": json.loads(r[3]) if r[3] else [],
                "created_at": r[4]
            }
            for r in rows
        ]
    except Exception:
        return []


def save_uploaded_chunks(chunks: list[dict]):
    try:
        conn = sqlite3.connect(DB_PATH)
        now = datetime.now().isoformat()
        conn.executemany(
            "INSERT INTO uploaded_chunks (text, source, url, embedding, uploaded_at) VALUES (?, ?, ?, ?, ?)",
            [(c["text"], c.get("source", ""), c.get("url", ""), c["embedding"], now) for c in chunks]
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Error saving uploaded chunks: {e}")


def load_uploaded_chunks_from_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT text, source, url, embedding FROM uploaded_chunks"
        ).fetchall()
        conn.close()
        if rows:
            chunks = [{"text": r[0], "source": r[1] or "", "url": r[2] or "", "embedding": r[3]} for r in rows]
            rag.add_preembedded_chunks(chunks)
    except Exception as e:
        print(f"[DB] Error loading uploaded chunks: {e}")


def get_uploaded_sources() -> list[dict]:
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT source, COUNT(*) as cnt, MAX(uploaded_at) FROM uploaded_chunks GROUP BY source ORDER BY MAX(uploaded_at) DESC"
        ).fetchall()
        conn.close()
        return [{"source": r[0], "chunks": r[1], "uploaded_at": r[2]} for r in rows]
    except Exception:
        return []


# ─── TEXT EXTRACTION ───

def extract_text_from_pdf(content: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(content))
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def extract_text_from_docx(content: bytes) -> str:
    doc = Document(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def chunk_text(text: str, source: str) -> list[dict]:
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start:start + CHUNK_SIZE]
        if piece.strip():
            chunks.append({"text": piece.strip(), "source": source, "url": ""})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ─── STARTUP ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        rag.init()
        load_uploaded_chunks_from_db()
    except Exception as e:
        print(f"[RAG] Init error: {e}")
    yield


app = FastAPI(title="מס הכנסה - מערכת שאלות ותשובות", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")


# ─── MODELS ───

class QuestionRequest(BaseModel):
    question: str


# ─── FORM 106 ANALYSIS ───

FORM106_PROMPT = """אתה מומחה מיסוי ישראלי. לפניך תמונות של טפסי 106 (אישורי מעסיק שנתיים).
עליך לחלץ את הנתונים ולמפות אותם לקודי השאילתא במס הכנסה (הדומה לטופס 135).

## מיפוי שדות טופס 106 → קודי שאילתה:

### הכנסות מיגיעה אישית:
- משכורת / שכר עבודה (שדה 158) → קוד 158
- מענק פרישה חייב (שדה 243) → קוד 243
- קצבה חייבת (שדה 242) → קוד 242
- שווי שימוש ברכב (שדה 237) → קוד 237
- הכנסות אחרות מעבודה (שדה 236) → קוד 236
- שווי ארוחות / תנאים סוציאליים (שדה 245) → קוד 245
- מענק פטור (שדה 9020) → לא מופיע בשאילתא (פטור)

### ניכויים שנוכו במקור:
- מס הכנסה שנוכה (שדה 042) → קוד 042
- דמי ביטוח לאומי (שדה 045) → קוד 045
- דמי ביטוח בריאות (שדה 047) → קוד 047
- תשלומי עובד לפנסיה/ביטוח מנהלים (שדה 048) → קוד 048
- תשלומי עובד לקרן השתלמות (שדה 049) → קוד 049
- תשלומי עובד לקופת גמל (שדה 050) → קוד 050

### זיכויים ממס:
- זיכוי ממס בגין פנסיה/ביטוח מנהלים (שדה 064) → קוד 064
- זיכוי ממס בגין קופת גמל (שדה 065) → קוד 065

### נקודות זיכוי:
- נקודות זיכוי שניתנו (שדה 047א) → קוד 047א / 2.25 כברירת מחדל

## הוראות חילוץ:
1. חלץ את שם המעסיק ומספר העוסק/ח.פ. מכל טופס
2. אם יש מספר טפסים — **צבור** את הסכומים לפי קוד
3. כלול רק קודים שהסכום שלהם גדול מ-0
4. עגל לשקלים שלמים
5. ציין את שנת המס (מופיעה בטופס)

## פורמט תשובה (JSON בלבד, ללא טקסט נוסף):
{
  "tax_year": "2024",
  "employers": [
    {"name": "שם מעסיק", "employer_id": "מספר עוסק", "period": "01/24-12/24"}
  ],
  "codes": [
    {"code": "158", "name": "הכנסת עבודה", "amount": 120000},
    {"code": "042", "name": "מס הכנסה שנוכה", "amount": 18000}
  ],
  "notes": "הערות על החילוץ (אם יש אי-ודאות)"
}"""


# ─── ROUTES ───

@app.get("/", response_class=HTMLResponse)
async def home():
    index_path = STATIC_PATH / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.get("/prisa", response_class=HTMLResponse)
async def prisa_calculator():
    return (STATIC_PATH / "prisa.html").read_text(encoding="utf-8")


@app.get("/tax867", response_class=HTMLResponse)
async def tax867_calculator():
    return (STATIC_PATH / "tax867.html").read_text(encoding="utf-8")


@app.get("/form106", response_class=HTMLResponse)
async def form106_page():
    return (STATIC_PATH / "form106.html").read_text(encoding="utf-8")


@app.post("/api/ask")
async def ask_question(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="שאלה ריקה")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="השאלה ארוכה מדי (מקסימום 1000 תווים)")
    try:
        result = rag.answer_question(question)
        save_to_history(question, result["answer"], result.get("sources", []))
        return result
    except Exception as e:
        print(f"[API] Error answering question: {e}")
        raise HTTPException(status_code=500, detail="שגיאה בעיבוד השאלה. אנא נסה שוב.")


@app.post("/api/analyze-106")
async def analyze_form_106(
    files: list[UploadFile] = File(default=[]),
    notes: str = Form(default="")
):
    if not files or all(f.size == 0 for f in files if hasattr(f, "size")):
        raise HTTPException(status_code=400, detail="יש לצרף לפחות תמונה אחת של טופס 106.")

    if not rag._chat_url or not rag._api_key:
        raise HTTPException(status_code=500, detail="מודל הבינה המלאכותית אינו זמין.")

    parts = []
    for file in files:
        content = await file.read()
        if not content:
            continue
        suffix = Path(file.filename or "").suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        elif suffix == ".png":
            mime = "image/png"
        elif suffix == ".webp":
            mime = "image/webp"
        elif suffix == ".heic":
            mime = "image/heic"
        else:
            mime = "image/jpeg"
        parts.append({
            "inlineData": {
                "mimeType": mime,
                "data": base64.b64encode(content).decode("ascii")
            }
        })

    if not parts:
        raise HTTPException(status_code=400, detail="לא נמצאו תמונות תקינות.")

    extra = f"\n\nנתונים נוספים שהמשתמש ציין: {notes}" if notes.strip() else ""
    parts.append({"text": FORM106_PROMPT + extra})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0}
    }

    try:
        resp = requests.post(
            rag._chat_url,
            params={"key": rag._api_key},
            json=payload,
            timeout=90
        )
        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="שגיאה בפענוח תשובת המודל. נסה שוב.")
    except Exception as e:
        print(f"[Form106] Error: {e}")
        raise HTTPException(status_code=500, detail="שגיאה בניתוח הטופס. אנא נסה שוב.")

    return result


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), password: str = ""):
    upload_password = os.environ.get("UPLOAD_PASSWORD", "")
    if upload_password and password != upload_password:
        raise HTTPException(status_code=401, detail="סיסמה שגויה.")

    filename = file.filename or "מסמך"
    suffix = Path(filename).suffix.lower()

    if suffix not in (".pdf", ".docx", ".doc", ".txt"):
        raise HTTPException(status_code=400, detail="סוג קובץ לא נתמך. השתמש ב-PDF, Word או TXT.")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=400, detail="הקובץ גדול מדי (מקסימום 20MB).")

    try:
        if suffix == ".pdf":
            text = extract_text_from_pdf(content)
        elif suffix in (".docx", ".doc"):
            text = extract_text_from_docx(content)
        else:
            text = content.decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"שגיאה בקריאת הקובץ: {e}")

    if not text or len(text.strip()) < 50:
        raise HTTPException(status_code=400, detail="הקובץ ריק או לא ניתן לחילוץ טקסט.")

    chunks = chunk_text(text, source=filename)
    if len(chunks) > MAX_CHUNKS_PER_UPLOAD:
        chunks = chunks[:MAX_CHUNKS_PER_UPLOAD]

    try:
        embedded = rag.embed_and_add_chunks(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"שגיאה ביצירת embeddings: {e}")

    save_uploaded_chunks(embedded)

    return {
        "success": True,
        "filename": filename,
        "chunks_added": len(embedded),
        "message": f"המסמך '{filename}' נוסף בהצלחה ({len(embedded)} קטעים)."
    }


@app.get("/api/uploaded-sources")
async def uploaded_sources():
    return {"sources": get_uploaded_sources()}


@app.get("/api/debug")
async def debug():
    return rag.get_debug_info()


@app.get("/api/history")
async def get_chat_history():
    return {"history": get_history(50)}


@app.get("/api/status")
async def status():
    kb_path = Path(__file__).parent.parent / "data" / "knowledge_base.json"
    base_chunks = 0
    if kb_path.exists():
        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        base_chunks = len(data.get("chunks", []))

    uploaded = get_uploaded_sources()
    uploaded_chunks = sum(s["chunks"] for s in uploaded)

    return {
        "status": "ok",
        "chunks_loaded": base_chunks + uploaded_chunks,
        "base_chunks": base_chunks,
        "uploaded_chunks": uploaded_chunks,
        "rag_ready": (base_chunks + uploaded_chunks) > 0
    }

"""
Tax Q&A - FastAPI application
Knowledge portal for Israeli Tax Authority employees
"""
import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app.rag as rag

DB_PATH = Path(__file__).parent.parent / "data" / "chat_history.db"
STATIC_PATH = Path(__file__).parent.parent / "static"


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        rag.init()
    except Exception as e:
        print(f"[RAG] Init error: {e}")
    yield


app = FastAPI(title="מס הכנסה - מערכת שאלות ותשובות", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")


class QuestionRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def home():
    index_path = STATIC_PATH / "index.html"
    return index_path.read_text(encoding="utf-8")


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


@app.get("/api/history")
async def get_chat_history():
    return {"history": get_history(50)}


@app.get("/api/status")
async def status():
    kb_path = Path(__file__).parent.parent / "data" / "knowledge_base.json"
    chunk_count = 0
    if kb_path.exists():
        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunk_count = len(data.get("chunks", []))
    return {
        "status": "ok",
        "chunks_loaded": chunk_count,
        "rag_ready": chunk_count > 0
    }

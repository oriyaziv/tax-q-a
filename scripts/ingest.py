"""
Ingestion script - run ONCE locally to build the knowledge base.
Processes:
  1. Local PDF/Word documents
  2. kolzchut.org.il (income-tax related pages)
  3. Israeli Tax Authority circulars (ניתוב שלב א')

Usage:
    pip install -r requirements.txt
    set GEMINI_API_KEY=your_key_here
    python scripts/ingest.py --docs "D:\אוריה\יצירת אפליקציות קלוד\קלוד קוד מס הכנסה"
"""
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "knowledge_base.json"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RATE_LIMIT_DELAY = 0.6
BASE = "https://generativelanguage.googleapis.com"

_api_key: str = ""
_embed_model: str = ""   # determined at runtime

# Candidate embedding models in preference order (tried against both v1 and v1beta)
EMBED_CANDIDATES = [
    "gemini-embedding-001",
    "gemini-embedding-2-preview",
]


def detect_embed_model() -> str:
    """Try each candidate model on v1 then v1beta and return the first that works."""
    for version in ("v1", "v1beta"):
        for model in EMBED_CANDIDATES:
            url = f"{BASE}/{version}/models/{model}:embedContent"
            try:
                resp = requests.post(
                    url,
                    params={"key": _api_key},
                    json={"model": f"models/{model}",
                          "content": {"parts": [{"text": "test"}]}},
                    timeout=15
                )
                if resp.status_code == 200:
                    print(f"[INFO] Using embedding model: {model} (API {version})")
                    # store version+model as embed URL
                    return f"{BASE}/{version}/models/{model}:embedContent"
            except Exception:
                pass
    raise RuntimeError(
        "No working embedding model found. "
        "Please verify your GEMINI_API_KEY is valid and the Generative Language API is enabled."
    )


# ─────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────

def extract_text_from_pdf(path: Path) -> str:
    text_parts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def extract_text_from_docx(path: Path) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text_from_file(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return extract_text_from_pdf(path)
        elif suffix in (".docx", ".doc"):
            return extract_text_from_docx(path)
        elif suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")
        else:
            print(f"  [skip] Unsupported format: {path.name}")
            return None
    except Exception as e:
        print(f"  [error] Failed to read {path.name}: {e}")
        return None


# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────

def chunk_text(text: str, source: str, url: str = "") -> list[dict]:
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({"text": chunk.strip(), "source": source, "url": url})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ─────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """Call Gemini REST API directly for a single embedding."""
    model_name = _embed_model.split("/models/")[1].split(":")[0]
    payload = {
        "model": f"models/{model_name}",
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_DOCUMENT"
    }
    resp = requests.post(
        _embed_model,
        params={"key": _api_key},
        json=payload,
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add embeddings to all chunks using Gemini v1 REST API directly."""
    embedded = []
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        try:
            chunk["embedding"] = embed_text(chunk["text"])
            embedded.append(chunk)
            if (i + 1) % 20 == 0:
                print(f"  Embedded {i + 1}/{total} chunks...")
            time.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            print(f"  [error] Embedding failed for chunk {i}: {e}")
            time.sleep(3)
    return embedded


# ─────────────────────────────────────────────
# LOCAL DOCUMENTS
# ─────────────────────────────────────────────

def ingest_local_documents(folder: Path) -> list[dict]:
    print(f"\n[1/3] Processing local documents from: {folder}")
    all_chunks = []
    supported = [".pdf", ".docx", ".doc", ".txt"]
    files = [f for f in folder.iterdir() if f.suffix.lower() in supported]
    print(f"  Found {len(files)} files")

    for file_path in sorted(files):
        print(f"  Reading: {file_path.name}")
        text = extract_text_from_file(file_path)
        if text and len(text.strip()) > 50:
            chunks = chunk_text(text, source=file_path.name)
            all_chunks.extend(chunks)
            print(f"    → {len(chunks)} chunks")
        else:
            print(f"    → Empty or too short, skipped")

    print(f"  Total local chunks: {len(all_chunks)}")
    return all_chunks


# ─────────────────────────────────────────────
# KOLZCHUT.ORG.IL SCRAPER
# ─────────────────────────────────────────────

ZCHUT_BASE = "https://www.kolzchut.org.il"
ZCHUT_TAX_KEYWORDS = [
    "מס הכנסה", "החזר מס", "פקודת מס הכנסה", "ניכוי מס",
    "זיכוי מס", "נקודות זיכוי", "הכנסה חייבת", 'דו"ח שנתי'
]


def is_tax_relevant(text: str) -> bool:
    return any(kw in text for kw in ZCHUT_TAX_KEYWORDS)


def scrape_zchut_page(url: str) -> Optional[dict]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; TaxBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")

        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        content_div = (
            soup.find("div", class_="entry-content") or
            soup.find("div", id="main-content") or
            soup.find("article") or
            soup.find("main") or
            soup.find("div", class_="content")
        )
        if not content_div:
            return None

        for tag in content_div.find_all(["script", "style", "nav"]):
            tag.decompose()
        text = content_div.get_text(separator="\n", strip=True)

        if not is_tax_relevant(text):
            return None

        return {"title": title, "text": text, "url": url}
    except Exception as e:
        print(f"  [error] scrape_zchut_page({url}): {e}")
        return None


def get_zchut_tax_urls() -> list[str]:
    urls = set()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TaxBot/1.0)"}

    known_pages = [
        "מדרגות_מס_הכנסה", "תיאום_מס_הכנסה", "החזר_מס_הכנסה",
        "נקודות_זיכוי_ממס_הכנסה", "ניכויים_ממס_הכנסה", "זיכויים_ממס_הכנסה",
        "הגשת_דוח_שנתי_למס_הכנסה", "מס_הכנסה_לשכירים", "פטור_ממס_הכנסה",
        "ניכוי_הוצאות_ממס_הכנסה", "נקודות_זיכוי_לעולים_חדשים",
        "נקודות_זיכוי_לבן_זוג_שאינו_עובד", "נקודות_זיכוי_עבור_ילדים",
        "פטור_ממס_הכנסה_לנכים", 'מס_הכנסה_על_הכנסות_מחו"ל',
        "מקדמות_מס_הכנסה", "שומת_מס_הכנסה", "ערעור_על_שומת_מס_הכנסה",
        "מס_הכנסה_לפנסיונרים", "זיכוי_ממס_עבור_תרומות",
    ]
    for page in known_pages:
        urls.add(f"{ZCHUT_BASE}/he/{page}")

    seed_urls = [
        f"{ZCHUT_BASE}/he/קטגוריה:מס_הכנסה",
        f"{ZCHUT_BASE}/he/קטגוריה:החזרי_מס",
        f"{ZCHUT_BASE}/he/קטגוריה:זכויות_עובדים_ומעסיקים",
    ]
    for seed in seed_urls:
        try:
            resp = requests.get(seed, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/he/") and "קטגוריה" not in href and "מיוחד" not in href:
                    urls.add(ZCHUT_BASE + href)
            time.sleep(0.3)
        except Exception:
            pass

    return list(urls)


def ingest_zchut(max_pages: int = 200) -> list[dict]:
    print(f"\n[2/3] Scraping kolzchut.org.il (tax-related pages)...")
    urls = get_zchut_tax_urls()
    print(f"  Found {len(urls)} candidate URLs")

    all_chunks = []
    processed = 0
    for url in urls[:max_pages]:
        page = scrape_zchut_page(url)
        if page:
            chunks = chunk_text(page["text"], source=f"זכותי - {page['title']}", url=url)
            all_chunks.extend(chunks)
            processed += 1
            if processed % 20 == 0:
                print(f"  Processed {processed} pages, {len(all_chunks)} chunks so far...")
        time.sleep(0.3)

    print(f"  Total zchut chunks: {len(all_chunks)} from {processed} pages")
    return all_chunks


# ─────────────────────────────────────────────
# TAX AUTHORITY CIRCULARS - ניתוב שלב א'
# ─────────────────────────────────────────────

NITUB_KNOWN_PDFS = [
    {"title": "ניתוב שלב א' 2025 - הוראת ביצוע 07/2025", "url": "https://www.gov.il/BlobFolder/policy/inst-07-2025/he/IncomeTax_inst-07-2025.pdf"},
    {"title": "ניתוב שלב א' 2024 - הוראת ביצוע 05/2024", "url": "https://www.gov.il/BlobFolder/policy/inst-05-2024/he/IncomeTax_inst-05-2024.pdf"},
    {"title": "ניתוב שלב א' 2023 - הוראת ביצוע 03/2023", "url": "https://www.gov.il/BlobFolder/policy/inst-03-2023/he/IncomeTax_inst-03-2023.pdf"},
    {"title": "ניתוב שלב א' 2022 - הוראת ביצוע 04/2022", "url": "https://www.gov.il/BlobFolder/policy/inst-04-2022/he/IncomeTax_inst-04-2022.pdf"},
    {"title": "ניתוב שלב א' 2021 - הוראת ביצוע 05/2021", "url": "https://www.gov.il/BlobFolder/policy/inst-05-2021/he/IncomeTax_inst-05-2021.pdf"},
    {"title": "ניתוב שלב א' 2020 - הוראת ביצוע 04/2020", "url": "https://www.gov.il/BlobFolder/policy/inst-04-2020/he/IncomeTax_inst-04-2020.pdf"},
    {"title": "ניתוב שלב א' 2019 - הוראת ביצוע 05/2019", "url": "https://www.gov.il/BlobFolder/policy/inst-05-2019/he/IncomeTax_inst-05-2019.pdf"},
    {"title": "ניתוב שלב א' 2018 - הוראת ביצוע 06/2018", "url": "https://www.gov.il/BlobFolder/policy/inst-06-2018/he/IncomeTax_inst-06-2018.pdf"},
    {"title": "ניתוב שלב א' - ארכיון הוראות ביצוע", "url": "https://claltax.com/הוראות-ביצוע-מס-הכנסה/"},
]


def fetch_circular_text(url: str) -> Optional[str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; TaxBot/1.0)"}
        if url.lower().endswith(".pdf"):
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                import io
                reader = PyPDF2.PdfReader(io.BytesIO(resp.content))
                return "\n".join(p.extract_text() or "" for p in reader.pages)
        else:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                return None
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup.find_all(["script", "style", "nav", "footer"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.find("body")
            return main.get_text(separator="\n", strip=True) if main else None
    except Exception as e:
        print(f"  [error] fetch_circular_text({url}): {e}")
        return None


def ingest_tax_circulars() -> list[dict]:
    print(f"\n[3/3] Fetching Tax Authority circulars (ניתוב שלב א')...")
    all_chunks = []
    for circ in NITUB_KNOWN_PDFS:
        print(f"  Fetching: {circ['title']}")
        text = fetch_circular_text(circ["url"])
        if text and len(text.strip()) > 100:
            chunks = chunk_text(text, source=circ["title"], url=circ["url"])
            all_chunks.extend(chunks)
            print(f"    → {len(chunks)} chunks")
        else:
            print(f"    → לא נמצא תוכן")
        time.sleep(0.5)
    print(f"  Total circular chunks: {len(all_chunks)}")
    return all_chunks


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    global _api_key, _embed_model

    parser = argparse.ArgumentParser(description="Build knowledge base for Tax Q&A")
    parser.add_argument("--docs", type=str, required=True, help="Path to local documents folder")
    parser.add_argument("--skip-web", action="store_true", help="Skip all web scraping")
    parser.add_argument("--skip-zchut", action="store_true", help="Skip kolzchut.org.il")
    parser.add_argument("--skip-circulars", action="store_true", help="Skip tax circulars")
    args = parser.parse_args()

    _api_key = os.environ.get("GEMINI_API_KEY", "")
    if not _api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set!")
        print("Get a free key at: https://aistudio.google.com/app/apikey")
        exit(1)

    print("[INFO] Detecting available embedding model...")
    _embed_model = detect_embed_model()

    docs_folder = Path(args.docs)
    if not docs_folder.exists():
        print(f"ERROR: Documents folder not found: {docs_folder}")
        exit(1)

    all_chunks = []
    all_chunks.extend(ingest_local_documents(docs_folder))

    if not args.skip_web and not args.skip_zchut:
        all_chunks.extend(ingest_zchut())
    else:
        print("\n[2/3] Skipping kolzchut.org.il")

    if not args.skip_web and not args.skip_circulars:
        all_chunks.extend(ingest_tax_circulars())
    else:
        print("\n[3/3] Skipping tax circulars")

    print(f"\n[EMBED] Total chunks to embed: {len(all_chunks)}")
    print("[EMBED] Starting embedding (this may take several minutes)...")

    embedded_chunks = embed_chunks(all_chunks)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"chunks": embedded_chunks}, f, ensure_ascii=False)

    print(f"\n[DONE] Knowledge base saved: {OUTPUT_PATH}")
    print(f"       {len(embedded_chunks)} chunks embedded successfully")
    print(f"\nNext steps:")
    print(f"  git add data/knowledge_base.json")
    print(f"  git commit -m 'Add knowledge base'")
    print(f"  git push origin claude/tax-knowledge-portal-fcaXi")


if __name__ == "__main__":
    main()

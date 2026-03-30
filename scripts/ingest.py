"""
Ingestion script - run ONCE locally to build the knowledge base.
Processes:
  1. Local PDF/Word documents
  2. Zchut.org.il (income-tax related pages)
  3. Israeli Tax Authority circulars (ניתוב שלב א')

Usage:
    pip install -r requirements.txt
    export GEMINI_API_KEY=your_key_here
    python scripts/ingest.py --docs /path/to/your/documents
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
import google.generativeai as genai

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "knowledge_base.json"
EMBED_MODEL = "models/embedding-001"
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap between chunks
RATE_LIMIT_DELAY = 0.5  # seconds between API calls

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
    """Split text into overlapping chunks."""
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({
                "text": chunk.strip(),
                "source": source,
                "url": url
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ─────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add embeddings to chunks using Gemini."""
    embedded = []
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        try:
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=chunk["text"],
                task_type="retrieval_document"
            )
            chunk["embedding"] = result["embedding"]
            embedded.append(chunk)
            if (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{total} chunks...")
            time.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            print(f"  [error] Embedding failed for chunk {i}: {e}")
            time.sleep(2)
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
# ZCHUT.ORG.IL SCRAPER
# ─────────────────────────────────────────────

ZCHUT_BASE = "https://www.kolzchut.org.il"
ZCHUT_TAX_KEYWORDS = [
    "מס הכנסה", "החזר מס", "פקודת מס הכנסה", "ניכוי מס",
    "זיכוי מס", "נקודות זיכוי", "הכנסה חייבת", "דו\"ח שנתי"
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

        # Get page title
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        # Get main content
        content_div = (
            soup.find("div", class_="entry-content") or
            soup.find("div", id="main-content") or
            soup.find("article") or
            soup.find("main") or
            soup.find("div", class_="content")
        )
        if not content_div:
            return None

        # Extract text
        for tag in content_div.find_all(["script", "style", "nav"]):
            tag.decompose()
        text = content_div.get_text(separator="\n", strip=True)

        if not is_tax_relevant(text):
            return None

        # Check for relevant laws mentioned
        has_income_tax_law = "פקודת מס הכנסה" in text

        return {
            "title": title,
            "text": text,
            "url": url,
            "has_income_tax_law": has_income_tax_law
        }
    except Exception as e:
        print(f"  [error] scrape_zchut_page({url}): {e}")
        return None


def get_zchut_tax_urls() -> list[str]:
    """Discover tax-related URLs on kolzchut.org.il"""
    urls = set()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TaxBot/1.0)"}

    # Known income-tax pages on kolzchut - Hebrew wiki-style URLs
    known_pages = [
        "מדרגות_מס_הכנסה",
        "תיאום_מס_הכנסה",
        "החזר_מס_הכנסה",
        "נקודות_זיכוי_ממס_הכנסה",
        "ניכויים_ממס_הכנסה",
        "זיכויים_ממס_הכנסה",
        "הגשת_דוח_שנתי_למס_הכנסה",
        "מס_הכנסה_לשכירים",
        "פטור_ממס_הכנסה",
        "ניכוי_הוצאות_ממס_הכנסה",
        "נקודות_זיכוי_לעולים_חדשים",
        "נקודות_זיכוי_לבן_זוג_שאינו_עובד",
        "נקודות_זיכוי_עבור_ילדים",
        "פטור_ממס_הכנסה_לנכים",
        "מס_הכנסה_על_הכנסות_מחו\"ל",
        "מקדמות_מס_הכנסה",
        "שומת_מס_הכנסה",
        "ערעור_על_שומת_מס_הכנסה",
        "מס_הכנסה_לפנסיונרים",
        "זיכוי_ממס_עבור_תרומות",
    ]
    for page in known_pages:
        urls.add(f"{ZCHUT_BASE}/he/{page}")

    # Crawl category pages for more links
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
                    full_url = ZCHUT_BASE + href
                    urls.add(full_url)
            time.sleep(0.3)
        except Exception:
            pass

    return list(urls)


def ingest_zchut(max_pages: int = 200) -> list[dict]:
    print(f"\n[2/3] Scraping zchut.org.il (tax-related pages)...")
    urls = get_zchut_tax_urls()
    print(f"  Found {len(urls)} candidate URLs")

    all_chunks = []
    processed = 0
    for url in urls[:max_pages]:
        page = scrape_zchut_page(url)
        if page:
            source_name = f"זכותי - {page['title']}"
            chunks = chunk_text(page["text"], source=source_name, url=url)
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

GOV_IL_BASE = "https://www.gov.il"

# הוראות ביצוע - ניתוב שלב א' published on gov.il
# URL pattern: https://www.gov.il/BlobFolder/policy/inst-XX-YYYY/he/IncomeTax_inst-XX-YYYY.pdf
# Known instruction numbers by year (approximate)
NITUB_KNOWN_PDFS = [
    {"title": "ניתוב שלב א' 2025 - הוראת ביצוע 07/2025", "url": "https://www.gov.il/BlobFolder/policy/inst-07-2025/he/IncomeTax_inst-07-2025.pdf"},
    {"title": "ניתוב שלב א' 2024 - הוראת ביצוע 05/2024", "url": "https://www.gov.il/BlobFolder/policy/inst-05-2024/he/IncomeTax_inst-05-2024.pdf"},
    {"title": "ניתוב שלב א' 2023 - הוראת ביצוע 03/2023", "url": "https://www.gov.il/BlobFolder/policy/inst-03-2023/he/IncomeTax_inst-03-2023.pdf"},
    {"title": "ניתוב שלב א' 2022 - הוראת ביצוע 04/2022", "url": "https://www.gov.il/BlobFolder/policy/inst-04-2022/he/IncomeTax_inst-04-2022.pdf"},
    {"title": "ניתוב שלב א' 2021 - הוראת ביצוע 05/2021", "url": "https://www.gov.il/BlobFolder/policy/inst-05-2021/he/IncomeTax_inst-05-2021.pdf"},
    {"title": "ניתוב שלב א' 2020 - הוראת ביצוע 04/2020", "url": "https://www.gov.il/BlobFolder/policy/inst-04-2020/he/IncomeTax_inst-04-2020.pdf"},
    {"title": "ניתוב שלב א' 2019 - הוראת ביצוע 05/2019", "url": "https://www.gov.il/BlobFolder/policy/inst-05-2019/he/IncomeTax_inst-05-2019.pdf"},
    {"title": "ניתוב שלב א' 2018 - הוראת ביצוע 06/2018", "url": "https://www.gov.il/BlobFolder/policy/inst-06-2018/he/IncomeTax_inst-06-2018.pdf"},
    # Older ones on claltax archive
    {"title": "ניתוב שלב א' - ארכיון הוראות ביצוע", "url": "https://claltax.com/הוראות-ביצוע-מס-הכנסה/"},
]


def fetch_tax_circular_text(url: str) -> Optional[str]:
    """Fetch text from a tax authority page or PDF."""
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
            # For claltax - find all PDF links for circulars
            if "claltax.com" in url:
                links = []
                for a in soup.find_all("a", href=True):
                    if ".pdf" in a["href"].lower() and ("inst" in a["href"].lower() or "ניתוב" in a.get_text()):
                        links.append(a["href"])
                # Return list of PDF URLs as text for further processing
                return "\n".join(links) if links else None
            main = soup.find("main") or soup.find("article") or soup.find("body")
            return main.get_text(separator="\n", strip=True) if main else None
    except Exception as e:
        print(f"  [error] fetch_tax_circular_text({url}): {e}")
        return None


def search_nitub_circulars() -> list[dict]:
    """Return known ניתוב שלב א' circular URLs."""
    return NITUB_KNOWN_PDFS


def ingest_tax_circulars() -> list[dict]:
    print(f"\n[3/3] Fetching Tax Authority circulars (ניתוב שלב א')...")
    circulars = search_nitub_circulars()
    print(f"  Found {len(circulars)} candidate circular pages")

    all_chunks = []
    for circ in circulars:
        print(f"  Fetching: {circ['title']}")
        text = fetch_tax_circular_text(circ["url"])
        if text and len(text.strip()) > 100:
            chunks = chunk_text(text, source=circ["title"], url=circ["url"])
            all_chunks.extend(chunks)
            print(f"    → {len(chunks)} chunks")
        else:
            print(f"    → לא נמצא תוכן (ייתכן שה-PDF אינו זמין)")
        time.sleep(0.5)

    print(f"  Total circular chunks: {len(all_chunks)}")
    return all_chunks


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build knowledge base for Tax Q&A")
    parser.add_argument("--docs", type=str, required=True, help="Path to local documents folder")
    parser.add_argument("--skip-web", action="store_true", help="Skip web scraping (only process local docs)")
    parser.add_argument("--skip-zchut", action="store_true", help="Skip zchut.org.il scraping")
    parser.add_argument("--skip-circulars", action="store_true", help="Skip tax circulars scraping")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set!")
        print("Get a free key at: https://aistudio.google.com/app/apikey")
        exit(1)

    genai.configure(api_key=api_key)
    print(f"[INFO] Using Gemini API for embeddings")

    docs_folder = Path(args.docs)
    if not docs_folder.exists():
        print(f"ERROR: Documents folder not found: {docs_folder}")
        exit(1)

    all_chunks = []

    # 1. Local documents
    local_chunks = ingest_local_documents(docs_folder)
    all_chunks.extend(local_chunks)

    # 2. Zchut.org.il
    if not args.skip_web and not args.skip_zchut:
        zchut_chunks = ingest_zchut()
        all_chunks.extend(zchut_chunks)
    else:
        print("\n[2/3] Skipping zchut.org.il")

    # 3. Tax circulars
    if not args.skip_web and not args.skip_circulars:
        circular_chunks = ingest_tax_circulars()
        all_chunks.extend(circular_chunks)
    else:
        print("\n[3/3] Skipping tax circulars")

    print(f"\n[EMBED] Total chunks to embed: {len(all_chunks)}")
    print("[EMBED] This may take a few minutes (rate-limited API calls)...")

    embedded_chunks = embed_chunks(all_chunks)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"chunks": embedded_chunks}, f, ensure_ascii=False)

    print(f"\n[DONE] Knowledge base saved to: {OUTPUT_PATH}")
    print(f"       {len(embedded_chunks)} chunks ready")
    print(f"\nNext steps:")
    print(f"  1. git add data/knowledge_base.json")
    print(f"  2. git commit -m 'Update knowledge base'")
    print(f"  3. git push origin claude/tax-knowledge-portal-fcaXi")


if __name__ == "__main__":
    main()

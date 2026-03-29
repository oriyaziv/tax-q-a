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
EMBED_MODEL = "models/text-embedding-004"
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

ZCHUT_BASE = "https://www.zchut.org.il"
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
    """Discover tax-related URLs on zchut.org.il"""
    urls = set()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TaxBot/1.0)"}

    # Known tax-related sections
    seed_urls = [
        f"{ZCHUT_BASE}/topics/taxes",
        f"{ZCHUT_BASE}/topics/income-tax",
        f"{ZCHUT_BASE}/topics/%D7%9E%D7%A1-%D7%94%D7%9B%D7%A0%D7%A1%D7%94",
        f"{ZCHUT_BASE}/topics/%D7%94%D7%97%D7%96%D7%A8-%D7%9E%D7%A1",
        f"{ZCHUT_BASE}/topics/%D7%A0%D7%A7%D7%95%D7%93%D7%95%D7%AA-%D7%96%D7%99%D7%9B%D7%95%D7%99",
    ]

    # Also try the sitemap
    try:
        sitemap_resp = requests.get(f"{ZCHUT_BASE}/sitemap.xml", headers=headers, timeout=15)
        if sitemap_resp.status_code == 200:
            soup = BeautifulSoup(sitemap_resp.text, "xml")
            for loc in soup.find_all("loc"):
                u = loc.get_text(strip=True)
                if ZCHUT_BASE in u:
                    urls.add(u)
    except Exception:
        pass

    # Crawl seed URLs for links
    for seed in seed_urls:
        try:
            resp = requests.get(seed, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/"):
                    href = ZCHUT_BASE + href
                if ZCHUT_BASE in href and href not in urls:
                    urls.add(href)
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

TAX_GOV_BASE = "https://www.misim.gov.il"
TAX_ALT_BASE = "https://taxes.gov.il"

NITUB_SEARCH_URLS = [
    "https://www.misim.gov.il/mfnhbhvhm/clsMain.aspx",
    "https://taxes.gov.il/Pages/ListAgafimAndMasovim.aspx",
    "https://www.misim.gov.il/mfnhbhvhm/",
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
            main = soup.find("main") or soup.find("article") or soup.find("body")
            return main.get_text(separator="\n", strip=True) if main else None
    except Exception as e:
        print(f"  [error] fetch_tax_circular_text({url}): {e}")
        return None


def search_nitub_circulars() -> list[dict]:
    """Search for ניתוב שלב א' circulars on tax authority websites."""
    circulars = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TaxBot/1.0)"}

    search_queries = [
        "https://www.misim.gov.il/mfnhbhvhm/clsMain.aspx?nType=2&nYear=2024",
        "https://taxes.gov.il/incomeTax/Pages/MaasHavara.aspx",
        "https://www.gov.il/he/departments/publications/reports/nitub_shlavA",
        "https://www.gov.il/he/search?q=%D7%A0%D7%99%D7%AA%D7%95%D7%91+%D7%A9%D7%9C%D7%91+%D7%90&skip=0&limit=20&OfficeId=0130&topics=income_tax",
    ]

    for url in search_queries:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")

            # Look for links containing ניתוב
            for a in soup.find_all("a", href=True):
                link_text = a.get_text(strip=True)
                href = a["href"]
                if "ניתוב" in link_text or "nitub" in href.lower():
                    full_url = href if href.startswith("http") else "https://www.gov.il" + href
                    circulars.append({"title": link_text, "url": full_url})
            time.sleep(0.5)
        except Exception:
            pass

    # Direct known URLs for recent years
    known_circulars = [
        {"title": f"ניתוב שלב א' {year}", "url": f"https://www.misim.gov.il/mfnhbhvhm/clsMain.aspx?nType=2&nYear={year}"}
        for year in range(2005, 2025)
    ]
    circulars.extend(known_circulars)

    return circulars


def ingest_tax_circulars() -> list[dict]:
    print(f"\n[3/3] Fetching Tax Authority circulars (ניתוב שלב א')...")
    circulars = search_nitub_circulars()
    print(f"  Found {len(circulars)} candidate circular pages")

    all_chunks = []
    for circ in circulars:
        print(f"  Fetching: {circ['title']}")
        text = fetch_tax_circular_text(circ["url"])
        if text and len(text.strip()) > 100 and "ניתוב" in text:
            chunks = chunk_text(text, source=circ["title"], url=circ["url"])
            all_chunks.extend(chunks)
            print(f"    → {len(chunks)} chunks")
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

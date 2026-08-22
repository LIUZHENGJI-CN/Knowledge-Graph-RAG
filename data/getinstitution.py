import json
import os
import random
import re
import threading
import time
from io import BytesIO
from typing import List, Optional, Tuple

import pdfplumber
import requests
from openai import OpenAI
from py2neo import Graph
from concurrent.futures import ThreadPoolExecutor, as_completed


# =====================================================
# 1. Config
# =====================================================
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "your_neo4j_password_here")
INPUT_JSON = r"C:\Users\lzj\Desktop\申请博士\复现知识图谱\data\full_paper_knowledge_base.json"

# OpenAI API config
# You can paste your API key directly here for quick local use.
# Environment variable OPENAI_API_KEY still takes precedence if it is set.
LOCAL_OPENAI_API_KEY = "your_openai_api_key_here"
API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or LOCAL_OPENAI_API_KEY.strip()
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()

# Only process papers still missing institutions
FORCE_REFRESH = False
LIMIT = None
MAX_WORKERS = 3
SAVE_EVERY = 10
MIN_CONFIDENCE = 0.75

REQUEST_TIMEOUT = 35
PDF_TEXT_CHAR_LIMIT = 4500

graph = Graph(NEO4J_URI, auth=NEO4J_AUTH)
client = OpenAI(api_key=API_KEY) if API_KEY else None
session = requests.Session()
session.headers.update(
    {
        "User-Agent": "kg-institution-enrichment/1.0",
        "Accept": "application/pdf,application/octet-stream,*/*",
    }
)
data_lock = threading.Lock()


# =====================================================
# 2. Helpers
# =====================================================
def save_json(data: List[dict]) -> None:
    with open(INPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def should_process(metadata: dict) -> bool:
    if FORCE_REFRESH:
        return True

    institutions = metadata.get("institutions") or []
    if institutions:
        return False

    institution = (metadata.get("institution") or "").strip()
    if institution:
        return False

    return True


def build_pdf_url(metadata: dict) -> Optional[str]:
    pdf_url = (metadata.get("pdf_url") or "").strip()
    if pdf_url:
        return pdf_url

    entry_id = (metadata.get("entry_id") or "").strip()
    if not entry_id:
        return None

    pdf_url = entry_id.replace("/abs/", "/pdf/")
    if not pdf_url.endswith(".pdf"):
        pdf_url += ".pdf"
    return pdf_url


def clean_institution_name(name: str) -> Optional[str]:
    if not name:
        return None

    value = re.sub(r"\s+", " ", str(name)).strip(" \t\r\n,;")
    if len(value) < 4:
        return None

    banned_fragments = [
        "national science foundation",
        "grant",
        "funding",
        "project",
        "supplementary",
        "correspondence",
        "author contributions",
    ]
    lowered = value.lower()
    if any(fragment in lowered for fragment in banned_fragments):
        return None

    return value


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        cleaned = clean_institution_name(item)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def extract_pdf_text(pdf_url: str) -> str:
    # Small jitter to reduce pressure on arXiv
    time.sleep(random.uniform(0.8, 1.6))

    response = session.get(pdf_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    parts: List[str] = []
    with pdfplumber.open(BytesIO(response.content)) as pdf:
        for page in pdf.pages[:2]:
            text = page.extract_text() or ""
            if text:
                parts.append(text)

    merged = "\n".join(parts).strip()
    return merged[:PDF_TEXT_CHAR_LIMIT]


def extract_institutions_via_llm(title: str, authors: List[str], pdf_text: str) -> Tuple[List[str], float]:
    if not client:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")

    authors_text = ", ".join(authors[:12]) if authors else "Unknown"
    prompt = f"""
You are extracting author affiliations from a research paper.

Paper title: {title}
Authors: {authors_text}
First-page text:
{pdf_text}
"""

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Extract only the authors' real affiliations from the paper text. "
                            "Prefer official full institution names. Exclude funding agencies, "
                            "grants, publishers, and repositories such as arXiv. "
                            "If evidence is weak, return an empty institution list and low confidence."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "institution_extraction",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "institutions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["institutions", "confidence"],
                },
            }
        },
    )

    parsed = json.loads(response.output_text)
    institutions = dedupe_preserve_order(parsed.get("institutions") or [])

    confidence = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(confidence, 1.0))
    return institutions, confidence


def sync_institutions_to_neo4j(entry_id: str, institutions: List[str], best_institution: Optional[str], confidence: float) -> None:
    graph.run(
        """
        MATCH (p:Paper {entry_id: $entry_id})
        SET p.institutions = $institutions,
            p.institution = $institution,
            p.institution_source = 'llm_pdf',
            p.institution_confidence = $confidence
        """,
        entry_id=entry_id,
        institutions=institutions,
        institution=best_institution,
        confidence=confidence,
    )

    graph.run(
        """
        MATCH (p:Paper {entry_id: $entry_id})-[r:AFFILIATED_WITH]->()
        DELETE r
        """,
        entry_id=entry_id,
    )

    for institution in institutions:
        graph.run(
            """
            MATCH (p:Paper {entry_id: $entry_id})
            MERGE (i:Institution {name: $institution})
            MERGE (p)-[:AFFILIATED_WITH]->(i)
            """,
            entry_id=entry_id,
            institution=institution,
        )


# =====================================================
# 3. Worker
# =====================================================
def process_institution_task(paper_item: dict) -> str:
    metadata = paper_item.get("metadata", {})
    title = metadata.get("title", "Unknown Title")
    entry_id = metadata.get("entry_id")
    authors = metadata.get("authors") or []

    if not entry_id:
        return f"SKIP: {title[:30]} -> missing entry_id"

    pdf_url = build_pdf_url(metadata)
    if not pdf_url:
        return f"SKIP: {title[:30]} -> missing pdf_url"

    try:
        pdf_text = extract_pdf_text(pdf_url)
    except Exception as exc:
        return f"PDF_FAIL: {title[:30]} -> {type(exc).__name__}"

    if not pdf_text.strip():
        return f"NO_TEXT: {title[:30]}"

    try:
        institutions, confidence = extract_institutions_via_llm(title, authors, pdf_text)
    except Exception as exc:
        return f"LLM_FAIL: {title[:30]} -> {type(exc).__name__}"

    institutions = dedupe_preserve_order(institutions)
    if confidence < MIN_CONFIDENCE:
        institutions = []

    with data_lock:
        metadata["institutions"] = institutions
        metadata["institution"] = institutions[0] if institutions else metadata.get("institution")
        metadata["institution_source"] = "llm_pdf" if institutions else metadata.get("institution_source")
        metadata["institution_confidence"] = confidence

    try:
        sync_institutions_to_neo4j(
            entry_id=entry_id,
            institutions=institutions,
            best_institution=institutions[0] if institutions else None,
            confidence=confidence,
        )
    except Exception as exc:
        return f"DB_FAIL: {title[:30]} -> {type(exc).__name__}"

    if institutions:
        return f"SUCCESS: {title[:30]} -> {institutions[0]} ({confidence:.2f})"
    return f"NO_INST: {title[:30]} ({confidence:.2f})"


# =====================================================
# 4. Main
# =====================================================
def main() -> None:
    if not os.path.exists(INPUT_JSON):
        print(f"Input file not found: {INPUT_JSON}")
        return

    if not API_KEY:
        print("Missing OPENAI_API_KEY. Set it before running.")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        papers_data = json.load(f)

    to_process = [paper for paper in papers_data if should_process(paper.get("metadata", {}))]
    if LIMIT is not None:
        to_process = to_process[:LIMIT]

    total = len(to_process)
    print(f"Need to enrich institutions for {total} papers with {MAX_WORKERS} workers")

    if total == 0:
        print("Nothing to do")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_institution_task, paper): paper for paper in to_process}

        for idx, future in enumerate(as_completed(futures), 1):
            print(f"[{idx}/{total}] {future.result()}")

            if idx % SAVE_EVERY == 0:
                with data_lock:
                    save_json(papers_data)
                print(f">>> Progress saved ({idx}/{total})")

    save_json(papers_data)
    print("Institution enrichment completed")


if __name__ == "__main__":
    main()


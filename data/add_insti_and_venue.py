import json
import os
import random
import re
import time
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import fitz  # PyMuPDF
import requests
from py2neo import Graph


# =====================================================
# 1. Config
# =====================================================
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "your_neo4j_password_here")

INPUT_JSON = r"C:\Users\lzj\Desktop\申请博士\复现知识图谱\data\full_paper_knowledge_base.json"
INSTITUTION_MAP_FILE = r"C:\Users\lzj\Desktop\申请博士\复现知识图谱\data\institution_map.json"

# Save progress every N papers
SAVE_EVERY = 20

# Set to None for all papers
LIMIT = None

# If True, papers that already have institution/venue will also be refreshed
FORCE_REFRESH = False

# OpenAlex is currently rate-limiting heavily; disable it to keep the pipeline moving.
OPENALEX_ENABLED = False

# Crossref recommends using a mailto for better service
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "").strip()


graph = Graph(NEO4J_URI, auth=NEO4J_AUTH)
session = requests.Session()
session.headers.update(
    {
        "User-Agent": "kg-metadata-enrichment/1.0 (contact: local-script)",
        "Accept": "application/json",
    }
)

with open(INSTITUTION_MAP_FILE, "r", encoding="utf-8") as f:
    INSTITUTION_MAP = json.load(f)


# =====================================================
# 2. Helpers
# =====================================================
def save_json(data: List[dict]) -> None:
    with open(INPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def normalize_doi(doi: str) -> Optional[str]:
    if not doi:
        return None

    doi = doi.strip()
    if not doi or doi.upper() == "N/A":
        return None

    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("https://dx.doi.org/", "").replace("http://dx.doi.org/", "")
    return doi.strip()


def classify_work_type(raw_type: Optional[str]) -> str:
    raw = (raw_type or "").lower()
    if raw in {"journal", "journal-article", "article"}:
        return "Journal"
    if raw in {"conference", "proceedings", "proceedings-article", "book-series"}:
        return "Conference"
    if raw in {"repository", "posted-content", "preprint"}:
        return "Preprint"
    return "Venue"


def build_query_params(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    params: Dict[str, str] = {}
    if CROSSREF_MAILTO:
        params["mailto"] = CROSSREF_MAILTO
    if extra:
        params.update(extra)
    return params


def request_json(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, str]] = None,
    json_body: Optional[dict] = None,
    timeout: int = 30,
    max_retries: int = 5,
) -> Optional[dict]:
    for attempt in range(1, max_retries + 1):
        try:
            response = session.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                timeout=timeout,
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 404:
                return None

            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = max(float(retry_after), 1.0)
                else:
                    wait = min(60.0, (2 ** attempt) + random.uniform(0.5, 2.0))
                print(f"HTTP {response.status_code} from {url}, retrying in {wait:.1f}s")
                time.sleep(wait)
                continue

            print(f"Skip {url}: HTTP {response.status_code}")
            return None
        except requests.RequestException as exc:
            if attempt == max_retries:
                print(f"Request failed for {url}: {exc}")
                return None
            wait = min(30.0, (2 ** attempt) + random.uniform(0.5, 1.5))
            print(f"Request error for {url}: {exc}; retrying in {wait:.1f}s")
            time.sleep(wait)

    return None


# =====================================================
# 3. OpenAlex
# =====================================================
def parse_openalex_work(work: dict) -> Dict[str, object]:
    source = ((work.get("primary_location") or {}).get("source") or {})
    venue_name = source.get("display_name")
    venue_type = classify_work_type(source.get("type"))

    institutions: List[str] = []
    for authorship in work.get("authorships") or []:
        for inst in authorship.get("institutions") or []:
            display_name = inst.get("display_name")
            if display_name and display_name not in institutions:
                institutions.append(display_name)

    return {
        "venue": venue_name,
        "publication_type": venue_type,
        "institutions": institutions,
        "doi": normalize_doi(work.get("doi") or ((work.get("ids") or {}).get("doi"))),
        "source": "openalex",
    }


def fetch_openalex_by_doi(doi: str) -> Optional[Dict[str, object]]:
    if not OPENALEX_ENABLED:
        return None

    doi = normalize_doi(doi)
    if not doi:
        return None

    encoded_filter = f"doi:https://doi.org/{doi}"
    params = {
        "filter": encoded_filter,
        "per-page": "1",
        "select": "doi,display_name,primary_location,authorships",
    }
    data = request_json("GET", "https://api.openalex.org/works", params=params)
    if not data:
        return None

    results = data.get("results") or []
    if not results:
        return None

    return parse_openalex_work(results[0])


def fetch_openalex_by_title(title: str) -> Optional[Dict[str, object]]:
    if not OPENALEX_ENABLED:
        return None

    params = {
        "filter": f"title.search:{title}",
        "per-page": "5",
        "select": "doi,display_name,primary_location,authorships",
    }
    data = request_json("GET", "https://api.openalex.org/works", params=params)
    if not data:
        return None

    best_match: Optional[dict] = None
    best_score = 0.0
    for item in data.get("results") or []:
        candidate_title = item.get("display_name") or ""
        score = title_similarity(title, candidate_title)
        if score > best_score:
            best_score = score
            best_match = item

    if best_match and best_score >= 0.88:
        return parse_openalex_work(best_match)
    return None


# =====================================================
# 4. Crossref
# =====================================================
def parse_crossref_work(message: dict) -> Dict[str, object]:
    container_titles = message.get("container-title") or []
    short_titles = message.get("short-container-title") or []
    venue_name = None
    if container_titles:
        venue_name = container_titles[0]
    elif short_titles:
        venue_name = short_titles[0]

    return {
        "venue": venue_name,
        "publication_type": classify_work_type(message.get("type")),
        "doi": normalize_doi(message.get("DOI")),
        "source": "crossref",
    }


def fetch_crossref_by_doi(doi: str) -> Optional[Dict[str, object]]:
    doi = normalize_doi(doi)
    if not doi:
        return None

    encoded = quote(doi, safe="")
    url = f"https://api.crossref.org/works/{encoded}"
    data = request_json("GET", url, params=build_query_params())
    if not data or "message" not in data:
        return None
    return parse_crossref_work(data["message"])


def fetch_crossref_by_title(title: str) -> Optional[Dict[str, object]]:
    params = build_query_params(
        {
            "query.title": title,
            "rows": "5",
            "select": "DOI,title,container-title,short-container-title,type",
        }
    )
    data = request_json("GET", "https://api.crossref.org/works", params=params)
    if not data or "message" not in data:
        return None

    items = (data["message"].get("items") or [])
    best_match: Optional[dict] = None
    best_score = 0.0
    for item in items:
        titles = item.get("title") or []
        candidate_title = titles[0] if titles else ""
        score = title_similarity(title, candidate_title)
        if score > best_score:
            best_score = score
            best_match = item

    if best_match and best_score >= 0.88:
        return parse_crossref_work(best_match)
    return None


# =====================================================
# 5. PDF fallback for institution
# =====================================================
EMAIL_DOMAIN_PATTERN = re.compile(r"@([\w\.-]+\.\w+)")


def fetch_institutions_from_pdf(pdf_url: str) -> List[str]:
    try:
        response = session.get(pdf_url, timeout=30)
        if response.status_code != 200:
            return []

        document = fitz.open(stream=response.content, filetype="pdf")
        if document.page_count == 0:
            return []

        first_page_text = document[0].get_text("text")
        matches = EMAIL_DOMAIN_PATTERN.findall(first_page_text)

        institutions: List[str] = []
        for domain in matches:
            domain = domain.lower().strip(".")
            for key, value in INSTITUTION_MAP.items():
                if domain == key or domain.endswith("." + key):
                    if value not in institutions:
                        institutions.append(value)
                    break
        return institutions
    except Exception as exc:
        print(f"PDF parse failed: {exc}")
        return []


# =====================================================
# 6. Enrichment logic
# =====================================================
def enrich_metadata(metadata: dict) -> Tuple[dict, List[str]]:
    title = metadata.get("title", "")
    doi = normalize_doi(metadata.get("doi"))
    pdf_url = metadata.get("pdf_url") or metadata.get("entry_id", "").replace("/abs/", "/pdf/") + ".pdf"

    result = {
        "venue": metadata.get("publication_venue"),
        "publication_type": metadata.get("publication_type"),
        "doi": doi,
    }

    need_venue = FORCE_REFRESH or not result["venue"] or result["venue"] == "N/A"
    need_inst = FORCE_REFRESH or not (
        metadata.get("institutions") or metadata.get("institution")
    )

    institutions: List[str] = []

    if doi and need_venue:
        openalex = fetch_openalex_by_doi(doi)
        if openalex:
            if openalex.get("venue"):
                result["venue"] = openalex["venue"]
                result["publication_type"] = openalex["publication_type"]
            if openalex.get("doi"):
                result["doi"] = openalex["doi"]

    if doi and need_venue and (not result["venue"] or result["venue"] == "N/A"):
        crossref = fetch_crossref_by_doi(doi)
        if crossref and crossref.get("venue"):
            result["venue"] = crossref["venue"]
            result["publication_type"] = crossref["publication_type"]
            if crossref.get("doi"):
                result["doi"] = crossref["doi"]

    if need_venue:
        openalex_title = fetch_openalex_by_title(title)
        if openalex_title:
            if (not result["venue"] or result["venue"] == "N/A") and openalex_title.get("venue"):
                result["venue"] = openalex_title["venue"]
                result["publication_type"] = openalex_title["publication_type"]
            if openalex_title.get("doi") and not result["doi"]:
                result["doi"] = openalex_title["doi"]

    if need_venue and (not result["venue"] or result["venue"] == "N/A"):
        crossref_title = fetch_crossref_by_title(title)
        if crossref_title and crossref_title.get("venue"):
            result["venue"] = crossref_title["venue"]
            result["publication_type"] = crossref_title["publication_type"]
            if crossref_title.get("doi") and not result["doi"]:
                result["doi"] = crossref_title["doi"]

    # 保留 Institution 节点/关系逻辑，但不再写 institution 相关 property key
    if doi and need_inst:
        openalex = fetch_openalex_by_doi(doi)
        if openalex and openalex.get("institutions"):
            institutions = openalex["institutions"]
            if openalex.get("doi"):
                result["doi"] = openalex["doi"]

    if need_inst and not institutions:
        openalex_title = fetch_openalex_by_title(title)
        if openalex_title and openalex_title.get("institutions"):
            institutions = openalex_title["institutions"]
            if openalex_title.get("doi") and not result["doi"]:
                result["doi"] = openalex_title["doi"]

    if need_inst and not institutions and pdf_url:
        pdf_institutions = fetch_institutions_from_pdf(pdf_url)
        if pdf_institutions:
            institutions = pdf_institutions

    if not result["venue"]:
        result["venue"] = "N/A"
    if not result["publication_type"]:
        result["publication_type"] = "Venue"

    return result, institutions


# =====================================================
# 7. Neo4j sync
# =====================================================
def ensure_constraints() -> None:
    graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Institution) REQUIRE i.name IS UNIQUE")


def sync_to_neo4j(entry_id: str, enriched: dict, institutions: List[str]) -> None:
    graph.run(
        """
        MATCH (p:Paper {entry_id: $entry_id})
        SET p.doi = coalesce($doi, p.doi)
        """,
        entry_id=entry_id,
        doi=enriched.get("doi"),
    )

    graph.run(
        """
        MATCH (p:Paper {entry_id: $entry_id})-[r:PUBLISHED_IN]->()
        DELETE r
        """,
        entry_id=entry_id,
    )
    graph.run(
        """
        MATCH (p:Paper {entry_id: $entry_id})-[r:AFFILIATED_WITH]->()
        DELETE r
        """,
        entry_id=entry_id,
    )

    venue = enriched.get("venue")
    if venue and venue != "N/A":
        venue_label = classify_work_type(enriched.get("publication_type"))
        if venue_label == "Journal":
            label = "Journal"
        elif venue_label == "Conference":
            label = "Conference"
        else:
            label = None

        if label is not None:
            query = f"""
            MATCH (p:Paper {{entry_id: $entry_id}})
            MERGE (v:{label} {{name: $venue}})
            MERGE (p)-[:PUBLISHED_IN]->(v)
            """
            graph.run(query, entry_id=entry_id, venue=venue)

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
# 8. Main
# =====================================================
def should_process(metadata: dict) -> bool:
    if FORCE_REFRESH:
        return True
    has_inst = bool(metadata.get("institutions")) or bool(metadata.get("institution"))
    has_venue = bool(metadata.get("publication_venue")) and metadata.get("publication_venue") != "N/A"
    return not (has_inst and has_venue)


def main() -> None:
    ensure_constraints()

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        papers_data = json.load(f)

    to_process = [paper for paper in papers_data if should_process(paper.get("metadata", {}))]
    if LIMIT is not None:
        to_process = to_process[:LIMIT]

    total = len(to_process)
    print(f"Need to enrich {total} papers")

    if total == 0:
        print("Nothing to do")
        return

    for idx, paper in enumerate(to_process, start=1):
        metadata = paper["metadata"]
        entry_id = metadata["entry_id"]

        try:
            enriched, institutions = enrich_metadata(metadata)

            if enriched.get("doi"):
                metadata["doi"] = enriched["doi"]

            sync_to_neo4j(entry_id, enriched, institutions)
            print(
                f"[{idx}/{total}] SUCCESS: "
                f"{metadata.get('title', '')[:40]} -> "
                f"{enriched.get('venue', 'N/A')} | "
                f"{institutions[0] if institutions else 'No Inst'}"
            )
        except KeyboardInterrupt:
            print("Interrupted, saving progress...")
            save_json(papers_data)
            raise
        except Exception as exc:
            print(f"[{idx}/{total}] ERROR: {metadata.get('title', '')[:40]} -> {exc}")

        if idx % SAVE_EVERY == 0:
            save_json(papers_data)
            time.sleep(random.uniform(0.5, 1.0))

    save_json(papers_data)
    print("Metadata enrichment completed")


if __name__ == "__main__":
    main()

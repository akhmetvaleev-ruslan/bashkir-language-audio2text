"""
Search books from books_metadata.csv on kitaptar.org (kitaptar.bashkort.org) -
the Bashkort open digital library - to see which ones already exist there
as text.

The site's search is an Algolia/instantsearch.js widget: typing into the
search box fires a live query and results are rendered client-side into
#hits, so this uses Playwright (real browser) rather than plain HTTP requests.

Caveats about the input data:
  The "title" column in books_metadata.csv was auto-derived from Telegram
  audio filenames like "Author_BookTitle_Reader_уҡый.mp3", so it's a mix of
  author + title + reader/recording-note, not a clean book title. This script
  applies a best-effort heuristic (clean_query) to strip reader/recording
  annotations and part/episode numbers before searching, then groups rows
  that collapse to the same cleaned query (e.g. multi-part recordings of the
  same book) to avoid redundant searches. Always sanity-check the results -
  the heuristic will not be perfect on every row.

Requirements:
  pip install playwright tqdm
  playwright install chromium

Usage:
  python search_kitaptar.py
  python search_kitaptar.py --input books_metadata.csv --output kitaptar_matches.csv
  python search_kitaptar.py --limit 20 --no-headless   # quick visual test run
"""

import argparse
import csv
import difflib
import re
import sys
from pathlib import Path

SITE_URL = "https://kitaptar.bashkort.org/"
SEARCH_INPUT_SELECTOR = "#search-box input.ais-search-box--input"

# Markers that introduce reader / recording annotations rather than the book
# title itself (e.g. "... Наилә Ғәләүетдинова уҡый", "... 2024 йылғы яҙма").
_ANNOTATION_MARKERS = re.compile(r"(уҡый|уҡыу|яҙма)", re.IGNORECASE)
_LEADING_INDEX = re.compile(r"^\d+[\s_.\-]+")
_TRAILING_PART = re.compile(
    r"[\s,]*\d+[\s_-]*(се|сы|со|сө|ce)?\s*(бүлек|өлөш|китап)?\s*$", re.IGNORECASE
)
_MULTI_SPACE = re.compile(r"\s+")


def clean_query(title: str, author: str = "") -> str:
    text = (title or "").replace("_", " ").replace(",,", " ")
    text = _LEADING_INDEX.sub("", text).strip()

    m = _ANNOTATION_MARKERS.search(text)
    if m:
        text = text[: m.start()]

    text = _TRAILING_PART.sub("", text)
    text = re.sub(r"[.,;:\-\s]+$", "", text).strip()
    text = _MULTI_SPACE.sub(" ", text).strip()

    if not text:
        text = _MULTI_SPACE.sub(" ", (title or "").replace("_", " ")).strip()

    return text


def load_rows(input_csv: str) -> list[dict]:
    with open(input_csv, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_queries(rows: list[dict]) -> dict[str, list[str]]:
    """Group message_ids by cleaned search query."""
    groups: dict[str, list[str]] = {}
    for row in rows:
        title = (row.get("title") or "").strip()
        if not title:
            continue
        query = clean_query(title, row.get("author", ""))
        if not query:
            continue
        groups.setdefault(query, []).append(row.get("message_id", ""))
    return groups


def search_one(page, query: str, timeout_ms: int = 8000) -> tuple[str, list[dict]]:
    """Type a query into the search box and scrape the resulting hits."""
    try:
        with page.expect_response(
            lambda r: "algolia.net" in r.url, timeout=timeout_ms
        ):
            page.fill(SEARCH_INPUT_SELECTOR, query)
    except Exception:
        page.wait_for_timeout(1000)

    page.wait_for_timeout(150)  # let the DOM render after the response

    stats = ""
    stats_el = page.query_selector("#stats")
    if stats_el:
        stats = stats_el.inner_text().strip()
    if not stats and page.query_selector("#no-results-message"):
        stats = "0 results found"

    hits = []
    for hit in page.query_selector_all("#hits .hit"):
        link = hit.query_selector("a")
        title_el = hit.query_selector("h2")
        author_el = hit.query_selector("h4")
        href = link.get_attribute("href") if link else ""
        hits.append(
            {
                "title": title_el.inner_text().strip() if title_el else "",
                "author": author_el.inner_text().strip() if author_el else "",
                "url": (SITE_URL.rstrip("/") + href) if href else "",
            }
        )
    return stats, hits


def best_match(query: str, hits: list[dict]) -> tuple[str, float]:
    best_title, best_score = "", 0.0
    for hit in hits:
        score = difflib.SequenceMatcher(None, query.lower(), hit["title"].lower()).ratio()
        if score > best_score:
            best_title, best_score = hit["title"], score
    return best_title, round(best_score, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="books_metadata.csv")
    parser.add_argument("--output", default="kitaptar_search_results.csv")
    parser.add_argument("--limit", type=int, default=None, help="only search the first N unique queries (for testing)")
    parser.add_argument("--no-headless", action="store_true", help="show the browser window")
    parser.add_argument("--delay-ms", type=int, default=250, help="pause between searches, be polite to the server")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    try:
        from playwright.sync_api import sync_playwright
        from tqdm import tqdm
    except ImportError as e:
        raise SystemExit(f"ImportError: {e}\nInstall dependencies first:\n  pip install playwright tqdm\n  playwright install chromium")

    rows = load_rows(args.input)
    groups = build_queries(rows)
    queries = list(groups.items())
    if args.limit:
        queries = queries[: args.limit]

    print(f"{len(rows)} rows -> {len(queries)} unique search queries")

    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.no_headless)
        page = browser.new_page()
        page.goto(SITE_URL, wait_until="networkidle")

        for query, message_ids in tqdm(queries, desc="Searching"):
            stats, hits = search_one(page, query)
            match_title, match_score = best_match(query, hits)
            results.append(
                {
                    "query": query,
                    "message_ids": ";".join(message_ids),
                    "num_hits": len(hits),
                    "stats": stats,
                    "best_match_title": match_title,
                    "best_match_score": match_score,
                    "hits": " | ".join(
                        f"{h['title']} — {h['author']} ({h['url']})" for h in hits
                    ),
                }
            )
            page.wait_for_timeout(args.delay_ms)

        browser.close()

    fieldnames = ["query", "message_ids", "num_hits", "stats", "best_match_title", "best_match_score", "hits"]
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    found = sum(1 for r in results if r["num_hits"] > 0)
    print(f"Done. {found}/{len(results)} queries returned at least one hit.")
    print(f"Results written to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

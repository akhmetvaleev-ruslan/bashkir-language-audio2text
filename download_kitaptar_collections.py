"""
Search kitaptar.bashkort.org (the Bashkort open digital library) for collected-works
volumes ("Сочинения" / "Әҫәрҙәр" / "Һайланма әҫәрҙәр") across all authors, and download
the PDF for every hit found.

Uses the same Playwright-driven live-search approach as search_kitaptar.py (the site's
search is a client-side Algolia/instantsearch.js widget), then fetches each book's detail
page with a plain HTTP request to extract its storage/files/*.pdf link (book detail pages
are server-rendered, no browser needed for that part).

Usage:
  python download_kitaptar_collections.py
  python download_kitaptar_collections.py --out kitaptar_downloads --no-headless
"""

import argparse
import csv
import re
import sys
import time
import urllib.request
from pathlib import Path

SITE_URL = "https://kitaptar.bashkort.org/"
SEARCH_INPUT_SELECTOR = "#search-box input.ais-search-box--input"
QUERIES = ["Сочинения", "Әҫәрҙәр", "Һайланма"]


def search_one(page, query: str, timeout_ms: int = 8000) -> list[dict]:
    try:
        with page.expect_response(
            lambda r: "algolia.net" in r.url, timeout=timeout_ms
        ):
            page.fill(SEARCH_INPUT_SELECTOR, query)
    except Exception:
        page.wait_for_timeout(1000)
    page.wait_for_timeout(300)

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
    return hits


def get_pdf_url(book_url: str) -> str:
    req = urllib.request.Request(book_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    m = re.search(r'storage/files/[^"\'<> ]+\.pdf', html)
    if not m:
        return ""
    return SITE_URL.rstrip("/") + "/" + m.group(0)


def download(url: str, dest: Path) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    dest.write_bytes(data)
    return len(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="kitaptar_collections")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--delay-ms", type=int, default=300)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from playwright.sync_api import sync_playwright

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    all_hits = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.no_headless)
        page = browser.new_page()
        page.goto(SITE_URL, wait_until="networkidle")

        for query in QUERIES:
            print(f"Searching: {query}")
            hits = search_one(page, query)
            print(f"  {len(hits)} hits")
            for h in hits:
                if h["url"]:
                    all_hits[h["url"]] = h
            page.wait_for_timeout(args.delay_ms)

        browser.close()

    print(f"\n{len(all_hits)} unique book pages found across {len(QUERIES)} queries")

    results = []
    for i, (url, h) in enumerate(all_hits.items(), 1):
        print(f"[{i}/{len(all_hits)}] {h['title']} — {h['author']}")
        row = {**h, "pdf_url": "", "local_file": "", "bytes": 0, "status": ""}
        try:
            pdf_url = get_pdf_url(url)
            if not pdf_url:
                row["status"] = "no_pdf_link_found"
                print("    no PDF link found on page")
            else:
                row["pdf_url"] = pdf_url
                fname = pdf_url.rsplit("/", 1)[-1]
                # avoid collisions
                dest = out_dir / fname
                n = 1
                while dest.exists():
                    dest = out_dir / f"{dest.stem}_{n}{dest.suffix}"
                    n += 1
                size = download(pdf_url, dest)
                row["local_file"] = str(dest)
                row["bytes"] = size
                row["status"] = "ok"
                print(f"    downloaded {size/1024/1024:.1f} MB -> {dest}")
        except Exception as e:
            row["status"] = f"error: {e}"
            print(f"    ERROR: {e}")
        results.append(row)
        time.sleep(0.3)

    fieldnames = ["title", "author", "url", "pdf_url", "local_file", "bytes", "status"]
    report_path = out_dir / "download_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone. {ok}/{len(results)} downloaded successfully.")
    print(f"Report: {report_path.resolve()}")


if __name__ == "__main__":
    main()

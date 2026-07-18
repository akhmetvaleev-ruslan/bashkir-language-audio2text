"""
Скачивает PDF-файлы книг, перечисленных в author_works.csv, с сайта
kitaptar.bashkort.org, и сохраняет их в папку pdf/. Другие форматы
(fb2, docx, epub) не скачиваются.

Ссылка на сам файл не хранится в author_works.csv (там только ссылка на
страницу книги), поэтому для каждой книги дополнительно запрашивается
её объект в поисковом индексе Algolia (тот же API, что и find_author_works.py) —
там есть список доступных форматов и путь к файлу.

Запуск:
    python download_pdfs.py
"""
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

ALGOLIA_APP_ID = "D5PMFTLE8T"
ALGOLIA_API_KEY = "e58b23282bf0ab9d4a7c961f267be78b"
ALGOLIA_INDEX = "books"
OBJECT_URL_TEMPLATE = f"https://{ALGOLIA_APP_ID}-dsn.algolia.net/1/indexes/{ALGOLIA_INDEX}/{{object_id}}"
STORAGE_BASE = "https://kitaptar.bashkort.org/storage/"

WORKS_PATH = "author_works.csv"
OUTPUT_DIR = "pdf"
REQUEST_DELAY_SEC = 0.3
USER_AGENT = "Mozilla/5.0 (compatible; kitaptar-pdf-downloader/1.0)"


def object_id_from_url(book_url: str) -> str:
    return book_url.rstrip("/").rsplit("/", 1)[-1]


def get_book_object(object_id: str) -> dict:
    url = OBJECT_URL_TEMPLATE.format(object_id=object_id)
    req = urllib.request.Request(url, headers={
        "X-Algolia-API-Key": ALGOLIA_API_KEY,
        "X-Algolia-Application-Id": ALGOLIA_APP_ID,
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp)


def sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "untitled"


def download_file(url: str, dest_path: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with open(dest_path, "wb") as f:
        f.write(data)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(WORKS_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    downloaded = 0
    skipped_exists = 0
    skipped_no_pdf = 0
    failed = 0

    for i, row in enumerate(rows, 1):
        object_id = object_id_from_url(row["book_url"])
        title = row["title"] or f"book_{object_id}"
        dest_name = f"{sanitize_filename(title)} ({object_id}).pdf"
        dest_path = os.path.join(OUTPUT_DIR, dest_name)

        if os.path.exists(dest_path):
            skipped_exists += 1
            print(f"[{i}/{len(rows)}] {title}: уже скачано, пропуск")
            continue

        try:
            obj = get_book_object(object_id)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            failed += 1
            print(f"[{i}/{len(rows)}] {title}: ошибка запроса метаданных ({exc})")
            continue

        pdf_file = next((f for f in obj.get("files", []) if f.get("ext") == "pdf"), None)
        if not pdf_file:
            skipped_no_pdf += 1
            print(f"[{i}/{len(rows)}] {title}: PDF не найден на сайте, пропуск")
            continue

        file_url = STORAGE_BASE + urllib.parse.quote(pdf_file["href"], safe="/")

        try:
            download_file(file_url, dest_path)
            downloaded += 1
            print(f"[{i}/{len(rows)}] {title}: скачано")
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            failed += 1
            print(f"[{i}/{len(rows)}] {title}: ошибка скачивания ({exc})")

        time.sleep(REQUEST_DELAY_SEC)

    print()
    print(f"Скачано: {downloaded}")
    print(f"Уже было (пропущено): {skipped_exists}")
    print(f"Нет PDF на сайте: {skipped_no_pdf}")
    print(f"Ошибок: {failed}")


if __name__ == "__main__":
    main()

"""
Ищет произведения авторов, перечисленных в authors.csv, в электронной
библиотеке kitaptar.bashkort.org, и сохраняет найденные ссылки в author_works.csv.

Библиотека использует поиск Algolia (индекс "books"), тот же, что работает
в поисковой строке на https://kitaptar.bashkort.org. Публичный search-only
ключ виден в открытом виде в js/search.js сайта и не даёт прав на запись.

Запуск:
    python find_author_works.py
"""
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# На Windows консоль по умолчанию использует cp1251, который не умеет
# кодировать часть башкирских букв (ә, ғ, ҡ, ң, ҙ, ҫ, һ, ө, ү) — без этой
# строки print() с именами авторов падает с UnicodeEncodeError.
sys.stdout.reconfigure(encoding="utf-8")

ALGOLIA_APP_ID = "D5PMFTLE8T"
ALGOLIA_API_KEY = "e58b23282bf0ab9d4a7c961f267be78b"
ALGOLIA_INDEX = "books"
SEARCH_URL = f"https://{ALGOLIA_APP_ID}-dsn.algolia.net/1/indexes/{ALGOLIA_INDEX}/query"
BOOK_URL_TEMPLATE = "https://kitaptar.bashkort.org/book/{id}"

AUTHORS_PATH = "authors.csv"
OUTPUT_PATH = "author_works.csv"
REQUEST_DELAY_SEC = 0.2  # пауза между запросами, чтобы не долбить API

# Башкирские буквы, которых нет в обычной кириллице, приводим к ближайшим
# аналогам, чтобы сравнивать имя автора из authors.csv с полем authors_names
# на сайте даже при небольших расхождениях в написании.
_NORMALIZE_MAP = str.maketrans({
    "ә": "а", "ғ": "г", "ҡ": "к", "ң": "н",
    "ҙ": "д", "ҫ": "с", "һ": "х", "ө": "о", "ү": "у",
})


def normalize(text: str) -> str:
    text = text.lower().translate(_NORMALIZE_MAP)
    text = re.sub(r"[^a-zа-яё0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def surnames(author: str) -> list[str]:
    """Последнее слово каждого имени в записи (на случай нескольких соавторов через запятую)."""
    parts = [p.strip() for p in author.split(",") if p.strip()]
    result = []
    for part in parts:
        words = normalize(part).split()
        if words:
            result.append(words[-1])
    return result


def load_authors(path: str) -> list[str]:
    with open(path, encoding="utf-8-sig") as f:
        return [row["author"] for row in csv.DictReader(f)]


def search_books(query: str, hits_per_page: int = 50) -> dict:
    params = urllib.parse.urlencode({
        "query": query,
        "hitsPerPage": hits_per_page,
        "filters": "active = 1",
    })
    body = json.dumps({"params": params}).encode("utf-8")
    req = urllib.request.Request(
        SEARCH_URL,
        data=body,
        headers={
            "X-Algolia-API-Key": ALGOLIA_API_KEY,
            "X-Algolia-Application-Id": ALGOLIA_APP_ID,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp)


def main() -> None:
    authors = load_authors(AUTHORS_PATH)
    rows = []

    for i, author in enumerate(authors, 1):
        wanted_surnames = [s for s in surnames(author) if len(s) > 2]
        try:
            data = search_books(author)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"[{i}/{len(authors)}] {author}: ошибка запроса ({exc})")
            continue

        for hit in data.get("hits", []):
            hit_authors_norm = normalize(hit.get("authors_names", ""))
            if not any(surname in hit_authors_norm for surname in wanted_surnames):
                continue
            rows.append({
                "author": author,
                "title": hit.get("name", "").strip(),
                "book_url": BOOK_URL_TEMPLATE.format(id=hit.get("objectID")),
                "authors_names_on_site": hit.get("authors_names", ""),
                "language": hit.get("language", ""),
            })

        print(f"[{i}/{len(authors)}] {author}: найдено совпадений — "
              f"{sum(1 for r in rows if r['author'] == author)}")
        time.sleep(REQUEST_DELAY_SEC)

    rows.sort(key=lambda r: (r["author"].lower(), r["title"].lower()))

    fieldnames = ["author", "title", "book_url", "authors_names_on_site", "language"]
    with open(OUTPUT_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nВсего найдено произведений: {len(rows)}")
    print(f"Результат сохранён в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

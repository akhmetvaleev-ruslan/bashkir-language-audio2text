"""
Ищет на Wikisource ссылки на страницы, соответствующие произведениям из
bashkir_poems_bashkir.csv, сверяя их с полным списком заголовков категории
"Шиғыр" (башкирская поэзия).

Подход:
  1. Один раз выгружает список ВСЕХ заголовков категории (это только
     названия страниц, не текст — как оглавление).
  2. Локально (без новых запросов к API) ищет среди них точные/близкие
     совпадения по названию произведения и фамилии автора.
  3. Сохраняет только найденные СОВПАДЕНИЯ (заголовок + URL) — текст
     страниц НЕ скачивается и не сохраняется.

ВАЖНО: даже найденный URL нужно проверять вручную — само наличие
страницы на Wikisource не гарантирует, что произведение в общественном
достоянии (категория содержит и современных авторов).

Запуск:
  python find_wikisource_urls.py
"""

import csv
import json
import re
import time
import urllib.parse
import urllib.request

CSV_PATH = "bashkir_poems_bashkir.csv"
CATEGORY_CACHE = "wikisource_category_titles.json"
OUT_PATH = "wikisource_urls.csv"
API = "https://wikisource.org/w/api.php"
PAGE_BASE = "https://wikisource.org/wiki/"
CATEGORY = "Category:Шиғыр"
REQUEST_PAUSE = 2.0  # секунды между запросами к API


def api_get(params, retries=5):
    params = dict(params, format="json")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "bashkir-corpus-url-finder/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = REQUEST_PAUSE * (attempt + 2)
                print(f"  429, жду {wait:.0f} сек...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("превышено число повторов после 429")


def fetch_category_titles():
    titles = []
    cmcontinue = None
    page = 0
    while True:
        page += 1
        print(f"Категория: страница {page} (собрано {len(titles)})")
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": CATEGORY,
            "cmlimit": "500",
            "cmnamespace": "0",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(params)
        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            titles.append({"pageid": m["pageid"], "title": m["title"]})
        cont = data.get("continue", {}).get("cmcontinue")
        time.sleep(REQUEST_PAUSE)
        if not cont:
            break
        cmcontinue = cont
    return titles


def page_url(title):
    return PAGE_BASE + urllib.parse.quote(title.replace(" ", "_"))


def normalize(s):
    s = s.lower()
    s = re.sub(r"[«»\"'.,!?…()\-–—:;]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def author_surname(author):
    # берём последнее слово как фамилию (для "Р.Сафин" — "Сафин", для "Мостай Кәрим" — "Кәрим")
    parts = re.split(r"[.\s]+", author.strip())
    parts = [p for p in parts if p]
    return parts[-1].lower() if parts else ""


def find_matches(author, our_title, catalog):
    norm_title = normalize(our_title)
    surname = author_surname(author)
    matches = []
    for entry in catalog:
        norm_page = normalize(entry["title"])
        title_hit = norm_title and norm_title in norm_page
        author_hit = surname and surname in norm_page
        if title_hit and author_hit:
            matches.append((entry["title"], "точное совпадение (название+автор)"))
        elif title_hit and len(norm_title) > 8:
            matches.append((entry["title"], "совпадение только по названию"))
    return matches


def main():
    try:
        with open(CATEGORY_CACHE, encoding="utf-8") as f:
            catalog = json.load(f)
        print(f"Используется кэш категории: {len(catalog)} заголовков")
    except FileNotFoundError:
        catalog = fetch_category_titles()
        with open(CATEGORY_CACHE, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=0)
        print(f"Сохранено {len(catalog)} заголовков категории в {CATEGORY_CACHE}")

    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    results = []
    for i, row in enumerate(rows, 1):
        author = row["Автор"]
        title = row["Название стихотворения"]
        print(f"[{i}/{len(rows)}] сверка...")
        matches = find_matches(author, title, catalog)
        if not matches:
            results.append([author, title, "", "", "не найдено в категории"])
        else:
            for found_title, quality in matches:
                results.append([author, title, found_title, page_url(found_title), quality])

    with open(OUT_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Автор", "Название (наше)", "Заголовок на Wikisource", "URL", "Качество совпадения"])
        w.writerows(results)

    print(f"\nГотово. {OUT_PATH} — только ссылки, без текста.")
    print("Проверяйте вручную соответствие и права автора перед использованием.")


if __name__ == "__main__":
    main()

"""
Генерирует статистику по каждой папке с аудиофайлами проекта и сохраняет её
в файл stats.csv прямо внутри этой папки: авторы (или дикторы/локации,
смотря по структуре папки), названия произведений (или сессий записи) и
количество аудиозаписей.

Структура папок в проекте неоднородна, поэтому для каждой применяется свой
способ сбора статистики:

  pairs/, pairs_poems/
      Дерево <Автор>/<Произведение>/audio/*.mp3 — считаем mp3 в каждой
      папке произведения.

  books/
      Плоская папка с mp3, обогащается метаданными из books_metadata.csv
      (автор/название, распознанные scrape_tg_books.py из подписи поста,
      имени файла и ID3-тегов). Для файлов без метаданных автор помечается
      как "Билгеһеҙ автор", а название берётся из имени файла.

  common_voice_pairs/, common_voice_pairs_poems/
      Корпуса, сгенерированные pairs_to_common_voice.py: строки train.tsv
      группируются по столбцам author/work, "записи" — это количество
      речевых клипов (после разбиения на предложения), а не число
      исходных mp3.

  common_voice_elang/
      Корпус, сгенерированный eaf_to_common_voice.py: колонок author/work
      нет (это диалектологические полевые записи, а не литературные
      произведения), поэтому группируем по speaker/source_file — строка
      "автор" здесь означает диктора, а "книга" — исходную запись.

  elang_data/
      Исходные записи ELAN (.eaf+.wav) под <локация>/<запись N>/. Имени
      автора тут тоже нет — вместо этого из разметки .eaf извлекаются
      теги диктора вида "[Имя Фамилия]" (та же логика, что и в
      eaf_to_common_voice.py); строка "книга" — это "<локация>/<запись N>".

Запуск:
  python generate_folder_stats.py
"""

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
SPEAKER_TAG_RE = re.compile(r"^\[([^\]]*)\]\s*")

STATS_FIELDNAMES = ["author", "book", "records"]


def write_stats(rows: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {dest}: {len(rows)} строк")


def stats_author_work_tree(root: Path) -> list[dict]:
    """pairs/, pairs_poems/: <author>/<work>/audio/*.mp3"""
    rows = []
    for author_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for work_dir in sorted(p for p in author_dir.iterdir() if p.is_dir()):
            audio_dir = work_dir / "audio" if (work_dir / "audio").is_dir() else work_dir
            count = len(list(audio_dir.glob("*.mp3")))
            if count:
                rows.append({"author": author_dir.name, "book": work_dir.name, "records": count})
    return rows


def stats_books(root: Path, metadata_csv: Path) -> list[dict]:
    """books/: плоская папка + books_metadata.csv (по имени файла)"""
    meta = {}
    if metadata_csv.exists():
        with open(metadata_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                meta[row["filename"]] = row

    counts: dict[tuple[str, str], int] = {}
    for mp3 in sorted(root.glob("*.mp3")):
        row = meta.get(mp3.name, {})
        author = (row.get("author") or "").strip() or "Билгеһеҙ автор"
        title = (row.get("title") or "").strip() or mp3.stem
        key = (author, title)
        counts[key] = counts.get(key, 0) + 1

    return [
        {"author": a, "book": b, "records": c}
        for (a, b), c in sorted(counts.items(), key=lambda kv: (kv[0][0].lower(), kv[0][1].lower()))
    ]


def stats_common_voice_tsv(tsv_path: Path, group_cols: tuple[str, str]) -> list[dict]:
    """common_voice_*/train.tsv, сгруппировано по (author, work) либо (speaker, source_file)"""
    df = pd.read_csv(tsv_path, sep="\t", usecols=list(group_cols), dtype=str, keep_default_na=False)
    grouped = df.groupby(list(group_cols), dropna=False).size().reset_index(name="records")
    grouped = grouped.sort_values(list(group_cols), key=lambda s: s.str.lower())
    return [
        {"author": row[group_cols[0]] or "(не указано)", "book": row[group_cols[1]] or "(не указано)", "records": row["records"]}
        for _, row in grouped.iterrows()
    ]


def extract_speakers(eaf_path: Path) -> set[str]:
    root = ET.parse(eaf_path).getroot()
    speakers = set()
    for tier in root.findall("TIER"):
        for ann in tier.findall("ANNOTATION/ALIGNABLE_ANNOTATION"):
            value_el = ann.find("ANNOTATION_VALUE")
            text = (value_el.text or "").strip() if value_el is not None else ""
            m = SPEAKER_TAG_RE.match(text)
            if m and m.group(1).strip():
                speakers.add(m.group(1).strip())
    return speakers


def stats_elang_data(root: Path) -> list[dict]:
    """elang_data/: <локация>/<запись N>/*.eaf — дикторы разбираются из тегов [Имя] в разметке"""
    rows = []
    for location_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for session_dir in sorted(p for p in location_dir.iterdir() if p.is_dir()):
            eaf_files = list(session_dir.glob("*.eaf"))
            wav_count = len(list(session_dir.glob("*.wav")))
            if not eaf_files or wav_count == 0:
                continue
            speakers = extract_speakers(eaf_files[0])
            book = f"{location_dir.name}/{session_dir.name}"
            for speaker in sorted(speakers) or ["unknown"]:
                rows.append({"author": speaker, "book": book, "records": wav_count})
    return rows


def main() -> None:
    print("pairs/")
    write_stats(stats_author_work_tree(ROOT / "pairs"), ROOT / "pairs" / "stats.csv")

    print("pairs_poems/")
    write_stats(stats_author_work_tree(ROOT / "pairs_poems"), ROOT / "pairs_poems" / "stats.csv")

    print("books/")
    write_stats(stats_books(ROOT / "books", ROOT / "books_metadata.csv"), ROOT / "books" / "stats.csv")

    print("common_voice_pairs/")
    write_stats(
        stats_common_voice_tsv(ROOT / "common_voice_pairs" / "train.tsv", ("author", "work")),
        ROOT / "common_voice_pairs" / "stats.csv",
    )

    print("common_voice_pairs_poems/")
    write_stats(
        stats_common_voice_tsv(ROOT / "common_voice_pairs_poems" / "train.tsv", ("author", "work")),
        ROOT / "common_voice_pairs_poems" / "stats.csv",
    )

    print("common_voice_elang/")
    write_stats(
        stats_common_voice_tsv(ROOT / "common_voice_elang" / "train.tsv", ("speaker", "source_file")),
        ROOT / "common_voice_elang" / "stats.csv",
    )

    print("elang_data/")
    write_stats(stats_elang_data(ROOT / "elang_data"), ROOT / "elang_data" / "stats.csv")


if __name__ == "__main__":
    main()

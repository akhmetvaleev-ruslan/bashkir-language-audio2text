"""
Скачивает аудио (mp3) для всех записей из bashkir_poems_bashkir.csv через yt-dlp
и раскладывает их по структуре pairs_poems/<Автор>/<Название>/audio/, как в остальном проекте.

Перед запуском:
  1. Установите yt-dlp:  pip install -U yt-dlp
  2. Установите ffmpeg (нужен для конвертации в mp3). Скрипт сначала пытается
     найти ffmpeg в PATH, а если не находит — использует портативный бинарник
     из пакета imageio-ffmpeg (pip install --user imageio-ffmpeg).
  3. Убедитесь, что у вас есть право использовать эти записи для ваших целей
     (личный/исследовательский корпус, разрешение правообладателя и т.п.) —
     скрипт не проверяет это за вас.

Запуск:
  python download_bashkir_audio.py

Скрипт можно безопасно прерывать (Ctrl+C) и перезапускать — уже скачанные
ролики отмечаются в pairs_poems/.downloaded_bashkir_poems.txt и повторно не скачиваются.
"""

import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

CSV_PATH = Path(__file__).parent / "bashkir_poems_bashkir.csv"
PAIRS_DIR = Path(__file__).parent / "pairs_poems"
ARCHIVE_FILE = PAIRS_DIR / ".downloaded_bashkir_poems.txt"
SLEEP_SECONDS = 2  # пауза между запросами, чтобы не спамить YouTube


def find_ffmpeg() -> str | None:
    """Ищет ffmpeg в PATH, иначе пробует портативный бинарник из imageio-ffmpeg."""
    in_path = shutil.which("ffmpeg")
    if in_path:
        return in_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def sanitize(name: str) -> str:
    for ch in '\\/:*?"<>|':
        name = name.replace(ch, "")
    return name.strip().rstrip(".")


def main() -> int:
    if not CSV_PATH.exists():
        print(f"Не найден файл {CSV_PATH}")
        return 1

    PAIRS_DIR.mkdir(exist_ok=True)

    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        print(
            "ffmpeg не найден ни в PATH, ни через imageio-ffmpeg.\n"
            "Установите: pip install --user imageio-ffmpeg"
        )
        return 1
    print(f"Используется ffmpeg: {ffmpeg_path}")

    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Найдено записей: {len(rows)}")
    print(f"Файлы будут разложены в: {PAIRS_DIR}\\<Автор>\\<Название>\\audio\\\n")

    failures = []
    for i, row in enumerate(rows, 1):
        num = row.get("№", str(i)).strip()
        performer = sanitize(row.get("Исполнитель", ""))
        author = sanitize(row.get("Автор", "")) or "Неизвестный автор"
        title = sanitize(row.get("Название стихотворения", "")) or "Без названия"
        url = row.get("URL", "").strip()

        if not url:
            continue

        try:
            num_fmt = f"{int(num):03d}"
        except ValueError:
            num_fmt = num

        audio_dir = PAIRS_DIR / author / title / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{num_fmt}_{performer}"[:150]
        out_template = str(audio_dir / f"{filename}.%(ext)s")

        print(f"[{i}/{len(rows)}] {author}/{title}/audio/{filename}")

        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-x", "--audio-format", "mp3",
            "--audio-quality", "0",
            "--ffmpeg-location", ffmpeg_path,
            "--download-archive", str(ARCHIVE_FILE),
            "-o", out_template,
            url,
        ]

        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print("Модуль yt_dlp не найден. Установите его: pip install -U yt-dlp")
            return 1
        except subprocess.CalledProcessError as e:
            print(f"  Ошибка скачивания ({url}): {e}")
            failures.append((num_fmt, url))

        time.sleep(SLEEP_SECONDS)

    print("\nГотово.")
    if failures:
        print(f"Не удалось скачать {len(failures)} записей:")
        for num_fmt, url in failures:
            print(f"  {num_fmt}: {url}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

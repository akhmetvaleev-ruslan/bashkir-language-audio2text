"""
Scrape book metadata (and optionally download audio) from Telegram channel.
Metadata is collected from three sources:
  1. Post caption/text (regex-based parsing)
  2. Filename patterns like "Author - Title.mp3"
  3. Audio file tags (ID3/MP4 via mutagen)

Requirements:
  pip install telethon mutagen tqdm

Usage:
  # Metadata only (fast)
  python scrape_tg_books.py --api-id YOUR_ID --api-hash YOUR_HASH

  # Download all audio files to books/
  python scrape_tg_books.py --api-id YOUR_ID --api-hash YOUR_HASH --download --download-dir books

  # Resume interrupted download (already existing files are skipped)
  python scrape_tg_books.py --api-id YOUR_ID --api-hash YOUR_HASH --download --download-dir books

Get api_id and api_hash at https://my.telegram.org -> App configuration.
"""

import asyncio
import csv
import io
import json
import re
import argparse
import os
from pathlib import Path
from datetime import timezone


CHANNEL = "bashaudiokitap"
SESSION_FILE = "tg_session"
OUTPUT_FILE = "books_metadata.json"

_FIELD_PATTERNS = {
    "title":    re.compile(r"(?:Китап|Книга|Ат|Название)[:\s]*(.+)", re.IGNORECASE),
    "author":   re.compile(r"(?:Автор|Яҙыусы|Авт\.?)[:\s]*(.+)", re.IGNORECASE),
    "reader":   re.compile(r"(?:Уҡыусы|Читает|Исполнитель)[:\s]*(.+)", re.IGNORECASE),
    "genre":    re.compile(r"(?:Жанр|Тема)[:\s]*(.+)", re.IGNORECASE),
    "year":     re.compile(r"(?:Йыл|Год)[:\s]*(\d{4})", re.IGNORECASE),
    "duration": re.compile(r"(?:Ваҡыт|Длительность|Продолжительность)[:\s]*(.+)", re.IGNORECASE),
    "language": re.compile(r"(?:Тел|Язык)[:\s]*(.+)", re.IGNORECASE),
}


def parse_filename_metadata(filename: str) -> dict:
    stem = Path(filename).stem.strip()
    stem = re.sub(r"^\d+[\.\-\s]+", "", stem).strip()
    parts = [p.strip() for p in re.split(r"\s[-–—]\s", stem)]
    if len(parts) == 1:
        return {"title": parts[0]}
    if len(parts) == 2:
        return {"author": parts[0], "title": parts[1]}
    return {"author": parts[0], "title": parts[1], "reader": parts[2]}


def parse_text_metadata(text: str) -> dict:
    if not text:
        return {}
    result = {}
    for field, pattern in _FIELD_PATTERNS.items():
        m = pattern.search(text)
        if m:
            result[field] = m.group(1).strip()
    return result


def parse_audio_tags(source: "str | bytes", filename: str = "") -> dict:
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        return {}
    try:
        if isinstance(source, (str, Path)):
            audio = MutagenFile(str(source), easy=True)
        else:
            buf = io.BytesIO(source)
            buf.name = filename
            audio = MutagenFile(buf, easy=True)
        if audio is None:
            return {}

        tags = {}
        mapping = {
            "title":   ["title", "TIT2"],
            "author":  ["artist", "TPE1", "author", "composer"],
            "album":   ["album", "TALB"],
            "year":    ["date", "TDRC", "year"],
            "genre":   ["genre", "TCON"],
            "comment": ["comment", "COMM"],
        }
        for field, keys in mapping.items():
            for key in keys:
                val = audio.get(key)
                if val:
                    tags[field] = str(val[0]) if isinstance(val, list) else str(val)
                    break

        if hasattr(audio, "info") and audio.info:
            info = audio.info
            if hasattr(info, "length"):
                tags["duration_sec"] = round(info.length)
            if hasattr(info, "sample_rate"):
                tags["sample_rate"] = info.sample_rate
            if hasattr(info, "channels"):
                tags["channels"] = info.channels
        return tags
    except Exception:
        return {}


def _get_doc_filename(doc) -> str:
    for attr in getattr(doc, "attributes", []):
        fn = getattr(attr, "file_name", None)
        if fn:
            return fn
    return "audio.mp3"


async def fetch_channel(
    api_id: int,
    api_hash: str,
    channel: str = CHANNEL,
    limit: int | None = None,
    output: str = OUTPUT_FILE,
    download_dir: str | None = None,
) -> None:
    try:
        from telethon import TelegramClient
        from telethon.tl.types import MessageMediaDocument
        from tqdm import tqdm
    except ImportError as e:
        raise SystemExit(
            f"ImportError: {e}\n"
            "Install dependencies first:\n  pip install telethon mutagen tqdm"
        )

    audio_dir = Path(download_dir) if download_dir else None
    if audio_dir:
        audio_dir.mkdir(parents=True, exist_ok=True)

    client = TelegramClient(SESSION_FILE, api_id, api_hash)

    async with client:
        print(f"Connected. Fetching posts from @{channel} ...")
        entity = await client.get_entity(channel)
        posts = await client.get_messages(entity, limit=limit)
        results = []

        for msg in tqdm(posts, desc="Processing"):
            if msg is None or not hasattr(msg, "id"):
                continue

            entry: dict = {
                "message_id": msg.id,
                "date": msg.date.astimezone(timezone.utc).isoformat() if msg.date else None,
                "url": f"https://t.me/{channel}/{msg.id}",
                "raw_text": msg.text or "",
                "source": [],
            }

            text_meta = parse_text_metadata(msg.text or "")
            if text_meta:
                entry.update(text_meta)
                entry["source"].append("text")

            media = getattr(msg, "media", None)
            is_audio = isinstance(media, MessageMediaDocument)

            if is_audio:
                doc = getattr(media, "document", None)
                if doc:
                    entry["file_size"] = getattr(doc, "size", None)
                    entry["mime_type"] = getattr(doc, "mime_type", None)

                    filename = _get_doc_filename(doc)
                    entry["filename"] = filename

                    for k, v in parse_filename_metadata(filename).items():
                        entry.setdefault(k, v)
                    if filename != "audio.mp3":
                        entry["source"].append("filename")

                    if audio_dir:
                        dest = audio_dir / filename
                        if dest.exists():
                            tqdm.write(f"  skip (exists): {filename}")
                        else:
                            tqdm.write(f"  downloading: {filename}")
                            await client.download_media(msg, file=str(dest))

                        entry["local_path"] = str(dest)
                        audio_tags = parse_audio_tags(dest, filename)
                        if audio_tags:
                            for k, v in audio_tags.items():
                                entry.setdefault(k, v)
                            entry["source"].append("audio_tags")
                    else:
                        # metadata-only: read just first 128 KB for tags
                        try:
                            chunk = await client.download_media(msg, file=bytes, part_size_kb=128)
                            if chunk:
                                audio_tags = parse_audio_tags(chunk[:131072], filename)
                                if audio_tags:
                                    for k, v in audio_tags.items():
                                        entry.setdefault(k, v)
                                    entry["source"].append("audio_tags")
                        except Exception as exc:
                            entry["audio_tag_error"] = str(exc)

            entry["source"] = list(set(entry["source"])) or ["none"]
            results.append(entry)

        output_path = Path(output)
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved {len(results)} entries → {output_path.resolve()}")

        csv_path = output_path.with_suffix(".csv")
        csv_fields = [
            "message_id", "date", "url", "local_path", "filename",
            "title", "author", "reader", "genre", "year", "duration",
            "language", "file_size", "mime_type", "source",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for entry in results:
                row = {**entry, "source": ", ".join(entry.get("source", []))}
                writer.writerow(row)
        print(f"Saved CSV → {csv_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape (and optionally download) audio books from a Telegram channel"
    )
    parser.add_argument("--api-id", type=int,
                        default=int(os.environ.get("TG_API_ID", 0)) or None,
                        help="Telegram API id (or set TG_API_ID env var)")
    parser.add_argument("--api-hash", default=os.environ.get("TG_API_HASH"),
                        help="Telegram API hash (or set TG_API_HASH env var)")
    parser.add_argument("--channel", default=CHANNEL,
                        help=f"Channel username without @ (default: {CHANNEL})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of posts to fetch (default: all)")
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help=f"Output JSON file (default: {OUTPUT_FILE})")
    parser.add_argument("--download", action="store_true",
                        help="Download full audio files (not just metadata)")
    parser.add_argument("--download-dir", default="books",
                        help="Directory for downloaded audio files (default: books)")
    args = parser.parse_args()

    api_id = args.api_id
    api_hash = args.api_hash

    if not api_id:
        print("Get api_id and api_hash at https://my.telegram.org -> App configuration")
        try:
            api_id = int(input("api_id: ").strip())
        except ValueError:
            raise SystemExit("api_id must be an integer")
    if not api_hash:
        api_hash = input("api_hash: ").strip()
    if not api_hash:
        raise SystemExit("api_hash is required")

    asyncio.run(
        fetch_channel(
            api_id=api_id,
            api_hash=api_hash,
            channel=args.channel,
            limit=args.limit,
            output=args.output,
            download_dir=args.download_dir if args.download else None,
        )
    )


if __name__ == "__main__":
    main()

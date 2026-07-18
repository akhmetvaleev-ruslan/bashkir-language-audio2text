"""
Convert the ELAN-annotated recordings in elang_data/ into a Common Voice-style
corpus: a clips/ folder of per-utterance audio plus a train.tsv using the same
columns as Common Voice's own split files (see train.tsv / whisper.py).

Each directory containing a .eaf (ELAN) file is expected to hold exactly one,
paired with a .wav of the same basename; such directories are found by
recursively scanning --input-dir, so recordings may be nested arbitrarily
deep (e.g. elang_data/<speaker>/<session>/*.eaf). Only tiers built from
ALIGNABLE_ANNOTATION (the time-aligned transcript, tier "default" or "ved")
are used; ELAN's dependent
REF_ANNOTATION tiers ("*_poln"/"vyst*") are near-empty comment tracks and are
skipped automatically.

Annotation text may open with "[Speaker Name]" to mark a speaker change; that
speaker carries forward to subsequent untagged segments until the next tag,
and is hashed into a Common-Voice-style client_id so utterances from the same
speaker share an id across clips (and across files, since the hash is stable).

Requires only the standard library (wave + xml.etree) - no ffmpeg needed since
clips are cut straight from the source PCM WAV.

Usage:
  python eaf_to_common_voice.py --input-dir elang_data --output-dir common_voice_elang
"""

import argparse
import csv
import hashlib
import re
import wave
import xml.etree.ElementTree as ET
from pathlib import Path

LOCALE = "ba"
SPEAKER_TAG_RE = re.compile(r"^\[([^\]]*)\]\s*")

TSV_COLUMNS = [
    "client_id", "path", "sentence_id", "sentence", "sentence_domain",
    "up_votes", "down_votes", "age", "gender", "accents", "variant", "locale", "segment",
    "speaker", "source_file", "start_ms", "end_ms", "duration_ms",
]


def clean_sentence(text: str, slash_mode: str) -> str:
    text = text.strip()
    if slash_mode == "comma":
        text = re.sub(r"\s*/\s*", ", ", text)
    elif slash_mode == "remove":
        text = re.sub(r"\s*/\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_eaf(eaf_path: Path) -> list[dict]:
    root = ET.parse(eaf_path).getroot()

    time_slots = {}
    for ts in root.find("TIME_ORDER"):
        value = ts.get("TIME_VALUE")
        if value is not None:
            time_slots[ts.get("TIME_SLOT_ID")] = int(value)

    segments = []
    for tier in root.findall("TIER"):
        aligned = tier.findall("ANNOTATION/ALIGNABLE_ANNOTATION")
        if not aligned:
            continue  # dependent/translation tier, no time-aligned speech
        speaker = None
        for ann in aligned:
            value_el = ann.find("ANNOTATION_VALUE")
            text = (value_el.text or "").strip() if value_el is not None else ""
            start = time_slots.get(ann.get("TIME_SLOT_REF1"))
            end = time_slots.get(ann.get("TIME_SLOT_REF2"))
            if not text or start is None or end is None or end <= start:
                continue

            m = SPEAKER_TAG_RE.match(text)
            if m:
                speaker = m.group(1).strip() or speaker
                text = text[m.end():].strip()
            if not text:
                continue

            segments.append({
                "tier": tier.get("TIER_ID"),
                "speaker": speaker or "unknown",
                "start_ms": start,
                "end_ms": end,
                "text": text,
            })
    return segments


def slice_wav(src: wave.Wave_read, start_ms: int, end_ms: int, dst_path: Path) -> None:
    framerate = src.getframerate()
    n_frames_total = src.getnframes()

    start_frame = min(int(start_ms / 1000 * framerate), n_frames_total)
    end_frame = min(int(end_ms / 1000 * framerate), n_frames_total)

    src.setpos(start_frame)
    frames = src.readframes(max(0, end_frame - start_frame))

    with wave.open(str(dst_path), "wb") as out:
        out.setnchannels(src.getnchannels())
        out.setsampwidth(src.getsampwidth())
        out.setframerate(framerate)
        out.writeframes(frames)


def process_folder(
    folder: Path, clips_dir: Path, slash_mode: str, min_dur_ms: int, max_dur_ms: int
) -> list[dict]:
    eaf_files = list(folder.glob("*.eaf"))
    if not eaf_files:
        return []
    eaf_path = eaf_files[0]
    wav_path = eaf_path.with_suffix(".wav")
    if not wav_path.exists():
        print(f"  skip {folder.name}: no matching wav for {eaf_path.name}")
        return []

    segments = parse_eaf(eaf_path)
    rows = []
    with wave.open(str(wav_path), "rb") as src:
        for i, seg in enumerate(segments, start=1):
            duration = seg["end_ms"] - seg["start_ms"]
            if duration < min_dur_ms or duration > max_dur_ms:
                continue
            sentence = clean_sentence(seg["text"], slash_mode)
            if not sentence:
                continue

            clip_name = f"{eaf_path.stem}_{seg['tier']}_{i:04d}.wav"
            slice_wav(src, seg["start_ms"], seg["end_ms"], clips_dir / clip_name)

            client_id = hashlib.sha256(seg["speaker"].encode("utf-8")).hexdigest()
            sentence_id = hashlib.sha256(sentence.encode("utf-8")).hexdigest()

            rows.append({
                "client_id": client_id,
                "path": clip_name,
                "sentence_id": sentence_id,
                "sentence": sentence,
                "sentence_domain": "",
                "up_votes": 0,
                "down_votes": 0,
                "age": "",
                "gender": "",
                "accents": "",
                "variant": "",
                "locale": LOCALE,
                "segment": "",
                "speaker": seg["speaker"],
                "source_file": eaf_path.stem,
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "duration_ms": duration,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Convert elang_data ELAN annotations into a Common Voice-style corpus"
    )
    parser.add_argument("--input-dir", default="elang_data")
    parser.add_argument("--output-dir", default="common_voice_elang")
    parser.add_argument("--tsv-name", default="train.tsv")
    parser.add_argument(
        "--slash-mode", choices=["comma", "remove", "keep"], default="comma",
        help="How to handle '/' pause markers in transcripts (default: turn into ', ')",
    )
    parser.add_argument("--min-duration-ms", type=int, default=300)
    parser.add_argument("--max-duration-ms", type=int, default=20000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    folders = sorted({p.parent for p in input_dir.rglob("*.eaf")}, key=str)
    for folder in folders:
        print(f"Processing {folder.relative_to(input_dir)} ...")
        rows = process_folder(
            folder, clips_dir, args.slash_mode, args.min_duration_ms, args.max_duration_ms
        )
        print(f"  {len(rows)} clips")
        all_rows.extend(rows)

    tsv_path = output_dir / args.tsv_name
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} clips written to {clips_dir}")
    print(f"Metadata written to {tsv_path}")


if __name__ == "__main__":
    main()

"""
Convert forced-alignment word-level JSON (ctc-forced-aligner output) plus their
source MP3 recordings, under pairs/ or pairs_poems/, into a Common Voice-style
corpus: a clips/ folder of per-sentence WAV clips and a train.tsv, following
the same column convention as eaf_to_common_voice.py's output.

Expected layout: <input-dir>/<author>/<work>/audio/ containing one or more
*.alignment.json files (word-level {"text", "segments": [{start,end,text,score}]})
each paired with one *.mp3 in the same directory. Filenames don't always share a
stem (e.g. "Мостай Кәрим Ауыл адвокаттары 1 часть.alignment.json" vs
"Ауыл_адвокаттары_Мостай_Кәрим_1_часть_...mp3"), so json/mp3 are paired
positionally after sorting both lists by their first embedded number.

Word segments are grouped into sentences by splitting after tokens ending in
sentence-final punctuation (. ! ? …), then each sentence's audio span is cut
straight from the source MP3 via soundfile (seek + read), avoiding ffmpeg.

Usage:
  python pairs_to_common_voice.py --input-dir pairs --output-dir common_voice_pairs
  python pairs_to_common_voice.py --input-dir pairs_poems --output-dir common_voice_pairs_poems --speaker-mode filename
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import soundfile as sf

LOCALE = "ba"
SENTENCE_END_RE = re.compile(r'[.!?…]+["»)]*$')
READER_NAME_RE = re.compile(r"^\d{2,4}_(.+)$")

TSV_COLUMNS = [
    "client_id", "path", "sentence_id", "sentence", "sentence_domain",
    "up_votes", "down_votes", "age", "gender", "accents", "variant", "locale", "segment",
    "speaker", "source_file", "author", "work", "start_ms", "end_ms", "duration_ms",
]


def group_sentences(segments: list[dict]) -> list[list[dict]]:
    groups = []
    current = []
    for seg in segments:
        if not seg.get("text"):
            continue
        current.append(seg)
        if SENTENCE_END_RE.search(seg["text"]):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def sort_key(path: Path):
    nums = [int(n) for n in re.findall(r"\d+", path.stem)]
    return (nums, path.stem)


def process_pair(
    json_path: Path, mp3_path: Path, clips_dir: Path, author: str, work: str,
    speaker_mode: str, min_dur_ms: int, max_dur_ms: int,
) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    groups = group_sentences(data.get("segments", []))

    if speaker_mode == "filename":
        m = READER_NAME_RE.match(mp3_path.stem)
        speaker = m.group(1) if m else mp3_path.stem
    else:
        speaker = author
    client_id = hashlib.sha256(speaker.encode("utf-8")).hexdigest()

    rows = []
    with sf.SoundFile(str(mp3_path)) as src:
        sr = src.samplerate
        total_frames = src.frames
        for i, group in enumerate(groups, start=1):
            start = group[0]["start"]
            end = group[-1]["end"]
            duration_ms = round((end - start) * 1000)
            if duration_ms < min_dur_ms or duration_ms > max_dur_ms:
                continue
            sentence = re.sub(r"\s+", " ", " ".join(w["text"] for w in group)).strip()
            if not sentence:
                continue

            start_frame = min(int(start * sr), total_frames)
            end_frame = min(int(end * sr), total_frames)
            n_frames = max(0, end_frame - start_frame)
            if n_frames == 0:
                continue

            src.seek(start_frame)
            frames = src.read(n_frames, dtype="int16")

            clip_name = f"{json_path.stem}_{i:04d}.wav"
            sf.write(str(clips_dir / clip_name), frames, sr, subtype="PCM_16")

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
                "speaker": speaker,
                "source_file": mp3_path.stem,
                "author": author,
                "work": work,
                "start_ms": round(start * 1000),
                "end_ms": round(end * 1000),
                "duration_ms": duration_ms,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Convert pairs/pairs_poems forced-alignment data into a Common Voice-style corpus"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tsv-name", default="train.tsv")
    parser.add_argument(
        "--speaker-mode", choices=["author", "filename"], default="author",
        help="'author': speaker = author folder name. "
             "'filename': speaker = name parsed from 'NNN_Name.mp3'-style filenames "
             "(use for pairs_poems, where filenames are reciter entries).",
    )
    parser.add_argument("--min-duration-ms", type=int, default=300)
    parser.add_argument("--max-duration-ms", type=int, default=25000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    audio_dirs = sorted({p.parent for p in input_dir.rglob("*.alignment.json")}, key=str)
    for audio_dir in audio_dirs:
        work_dir = audio_dir.parent
        author_dir = work_dir.parent
        author = author_dir.name
        work = work_dir.name

        jsons = sorted(audio_dir.glob("*.alignment.json"), key=sort_key)
        mp3s = sorted(audio_dir.glob("*.mp3"), key=sort_key)
        if len(jsons) != len(mp3s):
            print(f"  skip {audio_dir}: {len(jsons)} json vs {len(mp3s)} mp3 (count mismatch)")
            continue

        print(f"Processing {author} / {work} ({len(jsons)} file(s)) ...")
        for json_path, mp3_path in zip(jsons, mp3s):
            try:
                rows = process_pair(
                    json_path, mp3_path, clips_dir, author, work, args.speaker_mode,
                    args.min_duration_ms, args.max_duration_ms,
                )
            except Exception as e:
                print(f"  ERROR {json_path.name} / {mp3_path.name}: {e}")
                continue
            print(f"  {json_path.name}: {len(rows)} clips")
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

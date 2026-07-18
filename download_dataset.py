"""
Download Mozilla Common Voice dataset for Bashkir (ba) language.
Source: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
Requires: pip install datasets huggingface_hub soundfile
"""

import csv
import os
import argparse
from pathlib import Path


def save_audio(sample: dict, audio_path: Path) -> None:
    import soundfile as sf
    import numpy as np
    audio = sample["audio"]
    sf.write(str(audio_path), np.array(audio["array"]), audio["sampling_rate"])


def download_common_voice(
    output_dir: str = "data",
    splits: list[str] | None = None,
    version: str = "17_0",
    token: str | None = None,
):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "Install dependencies first:\n  pip install datasets huggingface_hub soundfile"
        )

    if splits is None:
        splits = ["train", "validation", "test"]

    dataset_name = f"mozilla-foundation/common_voice_{version}"
    output_path = Path(output_dir)
    audio_path = output_path / "audio"
    audio_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_name} | language: ba | splits: {splits}")
    print(f"Audio  → {audio_path.resolve()}")
    print(f"CSV    → {output_path.resolve()}/<split>.csv\n")

    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit(
            "Common Voice is a gated dataset. Provide your Hugging Face token:\n"
            "  Option 1: set environment variable HF_TOKEN=hf_...\n"
            "  Option 2: pass --token hf_...\n"
            "  Get token at: https://huggingface.co/settings/tokens\n"
            "  Then accept the dataset license at:\n"
            f"  https://huggingface.co/datasets/{dataset_name}"
        )

    for split in splits:
        print(f"[{split}] Loading...")
        ds = load_dataset(
            dataset_name,
            "ba",
            split=split,
            token=hf_token,
            trust_remote_code=True,
        )

        split_audio_dir = audio_path / split
        split_audio_dir.mkdir(exist_ok=True)

        csv_path = output_path / f"{split}.csv"
        print(f"[{split}] Saving {len(ds)} samples...")

        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["audio_path", "sentence"])
            writer.writeheader()

            for i, sample in enumerate(ds):
                filename = f"{i:06d}.wav"
                rel_path = f"audio/{split}/{filename}"
                save_audio(sample, split_audio_dir / filename)
                writer.writerow({"audio_path": rel_path, "sentence": sample["sentence"]})

                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(ds)}", end="\r")

        print(f"[{split}] Done → {csv_path}\n")

    print("All splits downloaded successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Common Voice Bashkir dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Root directory: audio saved to <dir>/audio/, CSV to <dir>/<split>.csv (default: data)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test", "other", "invalidated"],
        help="Dataset splits to download",
    )
    parser.add_argument(
        "--version",
        default="17_0",
        help="Common Voice version, e.g. 17_0, 16_1, 11_0 (default: 17_0)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    download_common_voice(
        output_dir=args.output_dir,
        splits=args.splits,
        version=args.version,
        token=args.token,
    )

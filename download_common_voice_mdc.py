"""
Download Common Voice Scripted Speech 26.0 (Bashkir) from Mozilla Data Collective.
Source: https://mozilladatacollective.com/datasets/cmqinmgw400w4nr07r24gw0pm

Requires: pip install datacollective
Auth: set MDC_API_KEY env var (get one from your MDC profile settings)
      accept the dataset's terms on the site before downloading, or the
      download will be rejected by the API.
"""

import argparse
import os
import tarfile
from pathlib import Path

DATASET_ID = "cmqinmgw400w4nr07r24gw0pm"


def download_and_extract(
    output_dir: str = "data", clean_archive: bool = False, api_key: str | None = None
) -> Path:
    try:
        from datacollective import download_dataset
    except ImportError:
        raise SystemExit("Install dependencies first:\n  pip install datacollective")

    if api_key:
        os.environ["MDC_API_KEY"] = api_key

    if not os.environ.get("MDC_API_KEY"):
        raise SystemExit(
            "Mozilla Data Collective requires an API key.\n"
            "  1. Sign in at https://mozilladatacollective.com and accept the dataset's terms\n"
            "  2. Grab an API key from your profile settings\n"
            "  3. export MDC_API_KEY=your-api-key-here   (or pass --api-key your-api-key-here)"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset {DATASET_ID} ...")
    archive_path = Path(download_dataset(DATASET_ID))
    print(f"Downloaded: {archive_path}")

    extract_dir = output_path / "common_voice_ba"
    extract_dir.mkdir(exist_ok=True)

    print(f"Extracting to {extract_dir.resolve()} ...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(extract_dir)

    if clean_archive:
        archive_path.unlink()
        print(f"Removed archive {archive_path}")

    print("Done. Contents:")
    for entry in sorted(extract_dir.rglob("*"))[:20]:
        print(f"  {entry.relative_to(extract_dir)}")

    return extract_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Common Voice Bashkir (26.0) from Mozilla Data Collective"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to download the archive into and extract it under <output-dir>/common_voice_ba (default: data)",
    )
    parser.add_argument(
        "--clean-archive",
        action="store_true",
        help="Delete the downloaded .tar.gz after extraction (default: keep it, location is wherever the SDK saved it)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Mozilla Data Collective API key (or set MDC_API_KEY env var)",
    )
    args = parser.parse_args()

    download_and_extract(
        output_dir=args.output_dir, clean_archive=args.clean_archive, api_key=args.api_key
    )

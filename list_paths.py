"""
Walk a directory tree and write every file/folder path to a text file.

Usage:
  python list_paths.py [root] [-o output.txt]

Examples:
  python list_paths.py                       # walk cwd, write paths.txt
  python list_paths.py data                  # walk ./data, write paths.txt
  python list_paths.py D: -o d_drive.txt      # walk whole D: drive
"""

import argparse
import os
from pathlib import Path


def collect_paths(root: Path) -> list[str]:
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        for name in dirnames + filenames:
            paths.append(str(Path(dirpath) / name))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description="List all file/folder paths under a root directory")
    parser.add_argument("root", nargs="?", default=".", help="Directory to walk (default: current directory)")
    parser.add_argument("-o", "--output", default="paths.txt", help="Output text file (default: paths.txt)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    paths = collect_paths(root)
    Path(args.output).write_text("\n".join(paths), encoding="utf-8")
    print(f"Wrote {len(paths)} paths to {args.output}")


if __name__ == "__main__":
    main()

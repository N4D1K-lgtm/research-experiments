#!/usr/bin/env python3
"""Download IPA dictionary TSV files from open-dict-data/ipa-dict."""

import argparse
import os
from pathlib import Path

import requests

LANGUAGES = ["en_US", "fr_FR", "es_ES", "de", "nl"]
BASE_URL = "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data"


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for lang in LANGUAGES:
        filename = f"{lang}.txt"
        dest = data_dir / filename
        if dest.exists():
            print(f"  {filename} already exists, skipping")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"  Downloading {url} ...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_text(resp.text, encoding="utf-8")
        print(f"  -> {dest} ({len(resp.text)} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download IPA dict data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Directory to store downloaded TSV files",
    )
    args = parser.parse_args()
    print("Downloading IPA dictionary files...")
    download(args.data_dir)
    print("Done.")


if __name__ == "__main__":
    main()

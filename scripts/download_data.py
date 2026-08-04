"""Download the OurAirports reference data this project routes over.

    python scripts/download_data.py

Files land in ``data/`` and are gitignored; re-run after cloning. Pass
--force to re-download files that already exist.
"""

import argparse
import sys
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BASE_URL = "https://davidmegginson.github.io/ourairports-data"

FILES = ("airports.csv", "navaids.csv", "runways.csv")


def download(filename: str, force: bool = False) -> None:
    destination = DATA_DIR / filename
    if destination.exists() and not force:
        print(f"  {filename}: already present, skipping (--force to refresh)")
        return

    url = f"{BASE_URL}/{filename}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # Write via a temp file so an interrupted download can't leave a
    # truncated CSV behind that later parses as valid-but-incomplete.
    temp = destination.with_suffix(destination.suffix + ".part")
    temp.write_bytes(response.content)
    temp.replace(destination)

    kib = len(response.content) / 1024
    print(f"  {filename}: {kib:,.0f} KiB")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Re-download files that already exist"
    )
    args = parser.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading OurAirports data into {DATA_DIR}")

    for filename in FILES:
        try:
            download(filename, force=args.force)
        except requests.RequestException as exc:
            print(f"  {filename}: FAILED ({exc})", file=sys.stderr)
            return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

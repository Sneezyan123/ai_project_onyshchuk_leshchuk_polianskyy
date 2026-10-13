"""Завантажує сирі дані з figshare у data/raw/. Вже скачані файли пропускає."""
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

FILES = {
    "bawa/data_schema.json": 53632100,
    "bawa/pro_kremlin_channels_list.csv": 53632103,
    "bawa/anti_kremlin_channel_list.csv": 53632106,
    "bawa/anti_kremlin_dataset.zip": 53632109,
    "bawa/pro_kremlin_dataset.zip": 53632112,
    "belarus/dat_mapping.csv": 66644348,
    "belarus/readme.md": 66644351,
    "belarus/themes_automated_analysis.csv": 66644363,
}


def main():
    for rel, file_id in FILES.items():
        dst = RAW / rel
        if dst.exists() and dst.stat().st_size > 0:
            print("є", rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print("качаю", rel, file=sys.stderr)
        urllib.request.urlretrieve(f"https://ndownloader.figshare.com/files/{file_id}", dst)


if __name__ == "__main__":
    main()

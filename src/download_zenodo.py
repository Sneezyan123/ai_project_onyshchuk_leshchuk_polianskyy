"""Завантажує датасет Kyrychenko (Zenodo 16949193) у data/raw/zenodo/: список каналів, кластери,
граф форвардів і 8 файлів з текстами постів (~48 ГБ). Великі файли качаємо шматками в 24 потоки,
бо Zenodo віддає ~1.7 МБ/с на одне з'єднання. Уже скачані файли пропускаємо."""
import json
import sys
import urllib.request
from pathlib import Path

from pdownload import main as pdownload

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "zenodo"
RECORD = "https://zenodo.org/api/records/16949193"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RECORD) as r:
        files = json.load(r)["files"]
    # спершу маленькі файли, потім тексти
    files.sort(key=lambda f: f["size"])
    for f in files:
        dst = RAW / f["key"]
        if dst.exists() and dst.stat().st_size == f["size"]:
            print("є", f["key"], flush=True)
            continue
        print("качаю", f["key"], round(f["size"] / 1e9, 2), "ГБ", flush=True)
        if f["size"] < 50_000_000:
            urllib.request.urlretrieve(f["links"]["self"], dst)
        else:
            pdownload(f["links"]["self"], f["size"], dst, workers=24)


if __name__ == "__main__":
    sys.exit(main())

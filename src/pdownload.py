"""Паралельне завантаження великого файлу з figshare шматками (Range).
figshare/S3 ріже швидкість на одне зʼєднання, тому качаємо в N потоків."""
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 32 * 1024 * 1024


def fetch(url, start, end, part):
    if part.exists() and part.stat().st_size == end - start + 1:
        return
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
            if part.stat().st_size == end - start + 1:
                return
        except Exception as e:
            print("retry", part.name, e, file=sys.stderr)
    raise RuntimeError(f"не вдалося {part}")


def main(file_id, size, dst, workers=16):
    # file_id - або id файлу на figshare, або повний URL (Zenodo)
    url = file_id if str(file_id).startswith("http") else f"https://ndownloader.figshare.com/files/{file_id}"
    dst = Path(dst)
    parts_dir = dst.with_suffix(".parts")
    parts_dir.mkdir(exist_ok=True)
    ranges = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    parts = [parts_dir / f"{i:05d}" for i in range(len(ranges))]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda a: fetch(url, *a[0], a[1]), zip(ranges, parts)))
    with open(dst, "wb") as out:
        for p in parts:
            out.write(p.read_bytes())
    for p in parts:
        p.unlink()
    parts_dir.rmdir()
    print("готово", dst, dst.stat().st_size)


if __name__ == "__main__":
    fid = sys.argv[1] if sys.argv[1].startswith("http") else int(sys.argv[1])
    main(fid, int(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 16)

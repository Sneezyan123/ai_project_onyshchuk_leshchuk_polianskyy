"""Потокова вибірка з двох датасетів: Bawa & Kursuncu (figshare) і Kyrychenko (Zenodo).

Обидва читаємо чанками, нічого не тримаючи в пам'яті цілком. Для кожної комірки
(група, канал, місяць) тримаємо до CAP_READ постів із найменшим випадковим ключем: це той самий
reservoir sampling, але детермінований (ключ = хеш від seed, каналу й id поста), тому повторний
запуск дає ту саму вибірку. Наприкінці для кожної групи окремо підбираємо спільну межу на
комірку, щоб група не перевищила свій TARGET.

Що викидаємо одразу: форварди, пости коротші за MIN_CHARS після очистки, точні дублікати
(за нормалізованим текстом, лишається перша поява), пости поза вікном 24.02.2021-24.02.2023.
Групи каналів задає groups.py.
"""
import hashlib
import heapq
import re
import sys
import zipfile
from collections import Counter, defaultdict

import pandas as pd

from common import INTERIM, ROOT, TAB
from groups import DROP

RAW = ROOT / "data" / "raw"
SEED = 42
MIN_CHARS = 80
START, END = pd.Timestamp("2021-02-24", tz="UTC"), pd.Timestamp("2023-02-24", tz="UTC")

# скільки постів хочемо на групу в підсумку і скільки тримаємо на комірку під час читання.
# Українських каналів 750, тому на комірку їм потрібно менше, інакше не влізе в пам'ять.
TARGET = {"kremlin": 200_000, "opposition": 80_000, "ukraine": 120_000, "belarus": 30_000}
CAP_READ = {"kremlin": 40, "opposition": 45, "ukraine": 15, "belarus": 45}

URL = re.compile(r"https?://\S+|t\.me/\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
SPACES = re.compile(r"\s+")


def clean(text):
    text = URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def key(*parts):
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


def bawa_chunks(camp):
    with zipfile.ZipFile(RAW / "bawa" / f"{camp}_kremlin_dataset.zip") as z:
        names = sorted(n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
        for name in names:
            print("читаю", name, file=sys.stderr, flush=True)
            with z.open(name) as f:
                for chunk in pd.read_csv(
                        f, chunksize=50_000,
                        usecols=["post_id", "post_datetime", "message", "post_views", "fwd_from", "channel"],
                        dtype={"message": "string", "channel": "string"}):
                    chunk = chunk[chunk["fwd_from"].fillna("[False, -1]").str.startswith("[False")]
                    yield chunk.rename(columns={"post_datetime": "date", "post_views": "views"}), camp


def zenodo_chunks(peer_to_user, forwards):
    for path in sorted((RAW / "zenodo").glob("post_texts_part_*.csv")):
        print("читаю", path.name, file=sys.stderr, flush=True)
        for chunk in pd.read_csv(path, chunksize=200_000, usecols=["id", "peer_id", "date", "message", "views"],
                                 dtype={"message": "string"}, on_bad_lines="skip"):
            chunk = chunk[chunk["peer_id"].isin(peer_to_user)]
            if chunk.empty:
                continue
            fwd = pd.MultiIndex.from_arrays([chunk["peer_id"], chunk["id"]]).isin(forwards)
            chunk = chunk[~fwd]
            chunk = chunk.assign(channel=chunk["peer_id"].map(peer_to_user)).rename(columns={"id": "post_id"})
            yield chunk, "zenodo"


def main():
    if (INTERIM / "sample.parquet").exists():
        print("вибірка вже є, пропускаю")
        return
    groups = pd.read_csv(INTERIM / "groups.csv")
    group_of = dict(zip(groups["username"].str.lower(), groups["group"]))
    z = groups[groups["source"] == "zenodo"]
    peer_to_user = dict(zip(z["peer_id"].astype("int64"), z["username"]))

    # форварди в Zenodo лежать окремим графом: (канал, пост) -> звідки переслано
    fwd = pd.read_csv(RAW / "zenodo" / "post_fwd.csv", usecols=["id", "peer_id"])
    fwd = fwd[fwd["peer_id"].isin(peer_to_user)]
    forwards = pd.MultiIndex.from_arrays([fwd["peer_id"], fwd["id"]])
    print("форвардів у вибраних каналах Zenodo:", len(forwards), file=sys.stderr)

    cells = defaultdict(list)
    seen = set()
    stats = Counter()
    sources = [bawa_chunks("anti"), bawa_chunks("pro"), zenodo_chunks(peer_to_user, forwards)]
    for src in sources:
        for chunk, origin in src:
            chunk = chunk[chunk["message"].notna()]
            dt = pd.to_datetime(chunk["date"], utc=True, errors="coerce")
            inside = (dt >= START) & (dt < END)
            chunk, dt = chunk[inside], dt[inside]
            for row, ts in zip(chunk.itertuples(index=False), dt):
                ch = str(row.channel)
                lu = ch.lower()
                if lu in DROP:
                    continue
                # канали Bawa, яких немає у списках, лишаються в таборі свого архіву
                group = group_of.get(lu, {"pro": "kremlin", "anti": "opposition"}.get(origin))
                if group is None:
                    continue
                text = clean(row.message)
                if len(text) < MIN_CHARS:
                    stats[f"{group}_short"] += 1
                    continue
                h = key(text.lower()[:400])
                if h in seen:
                    stats[f"{group}_dupes"] += 1
                    continue
                seen.add(h)
                stats[f"{group}_kept"] += 1

                cell = (group, ch, ts.strftime("%Y-%m"))
                k = key(SEED, ch, row.post_id)
                views = int(row.views) if pd.notna(row.views) else -1
                rec = (group, "zenodo" if origin == "zenodo" else "bawa", ch, int(row.post_id), ts, text, views)
                heap = cells[cell]
                if len(heap) < CAP_READ[group]:
                    heapq.heappush(heap, (-k, k, rec))
                elif k < heap[0][1]:
                    heapq.heapreplace(heap, (-k, k, rec))

    rows, caps = [], {}
    for group, target in TARGET.items():
        sizes = [len(h) for c, h in cells.items() if c[0] == group]
        caps[group] = max(c for c in range(1, CAP_READ[group] + 1) if sum(min(s, c) for s in sizes) <= target)
        for c, heap in cells.items():
            if c[0] == group:
                rows += [rec for _, _, rec in sorted(heap, key=lambda x: x[1])[:caps[group]]]
    df = pd.DataFrame(rows, columns=["group", "source", "channel", "post_id", "date", "text", "views"])
    df["period"] = (df["date"] >= pd.Timestamp("2022-02-24", tz="UTC")).map({True: "after", False: "before"})
    df = df.sort_values(["date", "channel", "post_id"]).reset_index(drop=True)
    df.to_parquet(INTERIM / "sample.parquet", index=False)

    for g, c in caps.items():
        stats[f"{g}_cap_per_cell"] = c
        stats[f"{g}_sample"] = int((df["group"] == g).sum())
        stats[f"{g}_channels"] = int(df.loc[df["group"] == g, "channel"].nunique())
    stats["sample"] = len(df)
    pd.Series(stats).sort_index().to_csv(TAB / "sampling_stats.csv", header=["n"])
    print(pd.Series(stats).sort_index())


if __name__ == "__main__":
    main()

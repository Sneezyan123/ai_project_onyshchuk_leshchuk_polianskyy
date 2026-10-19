"""Ембединги вибірки multilingual-e5-base. E5 очікує префікс "query: " для симетричних задач
(кластеризація, схожість), без нього якість помітно падає."""
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from common import INTERIM, MODELS

MODEL = "intfloat/multilingual-e5-base"


_cache = {}


def embed(texts, model_name=MODEL, prefix="query: ", batch=64, max_len=256):
    if model_name in _cache:
        return _encode(_cache[model_name], texts, prefix, batch)
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = max_len
    if torch.cuda.is_available():
        model.half()
    _cache[model_name] = model
    return _encode(model, texts, prefix, batch)


def _encode(model, texts, prefix, batch):
    # сортуємо за довжиною, щоб батчі були рівніші і менше паддингу
    order = np.argsort([len(t) for t in texts])
    emb = model.encode([prefix + texts[i] for i in order], batch_size=batch, show_progress_bar=False,
                       normalize_embeddings=True, convert_to_numpy=True)
    out = np.empty_like(emb)
    out[order] = emb
    return out.astype(np.float16)


SHARD = 20_000


def main():
    """Рахуємо шардами по 20k і зберігаємо кожен, щоб обрив не губив години роботи."""
    texts = pd.read_parquet(INTERIM / "sample.parquet", columns=["text"])["text"].tolist()
    shards = MODELS / "emb_shards"
    shards.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, s in enumerate(range(0, len(texts), SHARD)):
        path = shards / f"{i:03d}.npy"
        if not path.exists():
            np.save(path, embed(texts[s:s + SHARD]))
            print("shard", i, file=sys.stderr, flush=True)
        parts.append(np.load(path))
    emb = np.concatenate(parts)
    np.save(MODELS / "emb_me5_base.npy", emb)
    print(emb.shape, file=sys.stderr)


if __name__ == "__main__":
    main()

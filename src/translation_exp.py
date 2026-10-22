"""Чи варто перекладати пости англійською перед кластеризацією?

Беремо 12 тис. російськомовних постів (порівну з кожної пари «група × період» для Кремля й опозиції), перекладаємо
opus-mt-ru-en і порівнюємо три конфігурації з однаковими UMAP/HDBSCAN:
  A  multilingual-e5-base на оригіналі
  B  англійська all-mpnet-base-v2 на перекладі
  C  multilingual-e5-base на перекладі (щоб відділити ефект перекладу від ефекту моделі)
Ключові слова тем у всіх трьох рахуємо по російських токенах, інакше когерентність не порівняти.
Перекладаємо тільки перші ~200 токенів поста: далі ембединг-моделі все одно обрізають.
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score
from transformers import MarianMTModel, MarianTokenizer

from common import INTERIM, MODELS, SEED, TAB, coherence, reference_corpus, topic_diversity
from embed import embed
from lemmatize import lemmatize
from topics_bertopic import cluster, keywords, reduce, topic_model

N_PER_CELL = 3000
MCS = 30
# з min_samples=10, як на повному корпусі, HDBSCAN на 12k постів злипається в 2 кластери
# (один на 99% даних), тому тут 5
MIN_SAMPLES = 5


def translate(texts, batch=16, ckpt=None):
    """Перекладаємо батчами і кожні 50 батчів скидаємо прогрес на диск."""
    name = "Helsinki-NLP/opus-mt-ru-en"
    done = pd.read_pickle(ckpt) if ckpt is not None and ckpt.exists() else {}
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name).cuda().eval()
    order = np.argsort([len(t) for t in texts])
    t0 = time.time()
    for s in range(0, len(texts), batch):
        idx = [i for i in order[s:s + batch] if i not in done]
        if not idx:
            continue
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=200).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=256, num_beams=1)
        for i, g in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            done[int(i)] = g
        if s % (batch * 50) == 0:
            print(f"{s}/{len(texts)} {time.time() - t0:.0f} c", file=sys.stderr, flush=True)
            if ckpt is not None:
                pd.to_pickle(done, ckpt)
    return [done[i] for i in range(len(texts))]


def main():
    """Переклади кешуються в data/interim/translations.parquet разом із каналом і id поста,
    тому експеримент не залежить від того, як пронумерована поточна вибірка."""
    cache = INTERIM / "translations.parquet"
    if cache.exists():
        sub = pd.read_parquet(cache)
    else:
        df = pd.read_parquet(INTERIM / "sample.parquet", columns=["group", "channel", "post_id", "period", "lang", "text"])
        ru = df[(df["lang"] == "ru") & df["group"].isin(["kremlin", "opposition"])]
        parts = [g.sample(min(N_PER_CELL, len(g)), random_state=SEED) for _, g in ru.groupby(["group", "period"])]
        sub = pd.concat(parts).reset_index(drop=True)
        sub["en"] = translate(sub["text"].tolist(), ckpt=INTERIM / "translation_ckpt.pkl")
        sub.to_parquet(cache, index=False)

    # ключові слова тем рахуємо по російських леммах оригіналу, однаково для всіх конфігурацій
    docs = [" ".join(lemmatize(t, "ru")) for t in sub["text"]]
    ref_texts, ref_dict = reference_corpus(pd.read_pickle(INTERIM / "tokens.pkl").tolist())

    configs = {
        "A: me5 / оригінал": lambda: embed(sub["text"].tolist()),
        "B: mpnet-en / переклад": lambda: embed(sub["en"].tolist(), "sentence-transformers/all-mpnet-base-v2", prefix=""),
        "C: me5 / переклад": lambda: embed(sub["en"].tolist()),
    }
    rows, labels_by = [], {}
    for name, fn in configs.items():
        emb = fn()
        runs = []
        for seed in [SEED, 7, 2024]:
            red = reduce(emb, seed=seed)
            runs.append(cluster(red, MCS, MIN_SAMPLES))
        labels = runs[0]
        labels_by[name] = labels
        # на 12k постів кластерів мало, тому min_df=2, інакше він більший за max_df
        kw = keywords(topic_model(docs, labels, min_df=2))
        topics = list(kw.values())
        rows.append({
            "config": name, "k": len(kw),
            "c_v": coherence(topics, ref_texts, ref_dict, "c_v")[0],
            "npmi": coherence(topics, ref_texts, ref_dict, "c_npmi")[0],
            "diversity": topic_diversity(topics),
            "outliers": float((labels == -1).mean()),
            "ari_seeds": float(np.mean([adjusted_rand_score(runs[i], runs[j]) for i in range(3) for j in range(i + 1, 3)])),
        })
        print(rows[-1], file=sys.stderr, flush=True)

    res = pd.DataFrame(rows)
    names = list(labels_by)
    agree = pd.DataFrame([[adjusted_rand_score(labels_by[a], labels_by[b]) for b in names] for a in names],
                         index=names, columns=names)
    res.to_csv(TAB / "translation_experiment.csv", index=False)
    agree.round(3).to_csv(TAB / "translation_agreement_ari.csv")
    print(res)
    print(agree.round(3))
    # кілька прикладів перекладу для звіту / перевірки очима
    ex = sub.sample(20, random_state=SEED)[["text", "en"]].copy()
    ex[["text", "en"]] = ex[["text", "en"]].apply(lambda s: s.str[:200])  # лише короткі цитати, див. data/README.md
    ex.to_csv(TAB / "translation_examples.csv", index=False)


if __name__ == "__main__":
    main()

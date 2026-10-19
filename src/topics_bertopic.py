"""BERTopic на всій вибірці.

UMAP рахуємо один раз і кешуємо (на 300k постів це найдовший крок), а далі перебираємо
min_cluster_size у HDBSCAN: це і є експеримент із гранулярністю. Ключові слова тем рахує
c-TF-IDF по тих самих токенах, що й у LDA, тому когерентність двох моделей порівнювана.
"""
import json
import sys
import time

import hdbscan
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.cluster import BaseCluster
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from common import INTERIM, MODELS, SEED, TAB, coherence, reference_corpus, topic_diversity

MIN_CLUSTER_SIZES = [50, 100, 200, 400]
CHOSEN = 200  # перезаписується, якщо передати аргумент


def reduce(emb, seed=SEED, path=None):
    if path is not None and path.exists():
        return np.load(path)
    t = time.time()
    red = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
               random_state=seed, low_memory=True, verbose=True).fit_transform(emb.astype(np.float32))
    print(f"UMAP {time.time() - t:.0f} c", file=sys.stderr, flush=True)
    if path is not None:
        np.save(path, red)
    return red


def cluster(red, mcs, min_samples=10):
    return hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=min_samples, metric="euclidean",
                           cluster_selection_method="eom", core_dist_n_jobs=6).fit_predict(red)


def topic_model(docs, labels, min_df=10):
    """BERTopic поверх уже готових кластерів: потрібен лише для c-TF-IDF і зручних методів.
    Ембединги передаємо заглушкою: без них BERTopic сам почне переембедити всі документи."""
    model = BERTopic(
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=BaseCluster(),
        vectorizer_model=CountVectorizer(min_df=min_df, max_df=0.3, token_pattern=r"\S+"),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        language="multilingual",  # з "english" BERTopic вирізає всю кирилицю перед c-TF-IDF
        calculate_probabilities=False,
        verbose=False,
    )
    model.fit(docs, embeddings=np.zeros((len(docs), 2), dtype=np.float32), y=labels)
    return model


def keywords(model, topn=10):
    return {t: [w for w, _ in model.get_topic(t)[:topn]] for t in model.get_topics() if t != -1}


def main(chosen=CHOSEN):
    df = pd.read_parquet(INTERIM / "sample.parquet", columns=["group", "period", "lang"])
    tokens = pd.read_pickle(INTERIM / "tokens.pkl").tolist()
    docs = [" ".join(t) for t in tokens]
    emb = np.load(MODELS / "emb_me5_base.npy")
    red = reduce(emb, path=MODELS / "umap5_seed42.npy")
    ref_texts, ref_dict = reference_corpus(tokens)

    rows = []
    for mcs in MIN_CLUSTER_SIZES:
        t = time.time()
        model = topic_model(docs, cluster(red, mcs))
        # BERTopic перенумеровує теми за розміром (0 = найбільша), тому далі беремо його мітки,
        # а не сирі з HDBSCAN, інакше ключові слова приклеються до чужих тем
        labels = np.array(model.topics_)
        kw = keywords(model)
        topics = list(kw.values())
        cv, _ = coherence(topics, ref_texts, ref_dict, "c_v")
        npmi, _ = coherence(topics, ref_texts, ref_dict, "c_npmi")
        rows.append({"model": "BERTopic", "min_cluster_size": mcs, "k": len(kw), "c_v": cv, "npmi": npmi,
                     "diversity": topic_diversity(topics), "outliers": float((labels == -1).mean()),
                     "seconds": round(time.time() - t)})
        print(rows[-1], file=sys.stderr, flush=True)
        np.save(MODELS / f"labels_mcs{mcs}.npy", labels)
        with open(TAB / f"bertopic_mcs{mcs}_topics.json", "w", encoding="utf-8") as f:
            json.dump({int(k): v for k, v in kw.items()}, f, ensure_ascii=False, indent=1)

    res = pd.DataFrame(rows)
    res.to_csv(TAB / "bertopic_granularity.csv", index=False)
    print(res)

    # детальна таблиця тем для обраної гранулярності
    labels = np.load(MODELS / f"labels_mcs{chosen}.npy")
    model = topic_model(docs, labels)
    assert (np.array(model.topics_) == labels).all()
    info = model.get_topic_info()
    info["keywords"] = info["Topic"].map(lambda t: ", ".join(w for w, _ in model.get_topic(t)[:10]) if t != -1 else "")
    info = info.drop(columns=[c for c in ["Representation", "Representative_Docs", "Name"] if c in info])
    shares = pd.crosstab(labels, df["group"])
    info = info.merge(shares, left_on="Topic", right_index=True, how="left")
    info.to_csv(TAB / f"bertopic_mcs{chosen}_info.csv", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else CHOSEN)

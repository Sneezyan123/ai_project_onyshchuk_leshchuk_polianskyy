"""LDA baseline на gensim. Перебираємо кількість тем, рахуємо c_v, NPMI, diversity,
для найкращого K ще два запуски з іншими seed і ARI між ними."""
import json
import sys

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from sklearn.metrics import adjusted_rand_score

from common import INTERIM, MODELS, SEED, TAB, coherence, reference_corpus, topic_diversity

KS = [20, 40, 60, 80]


def fit(corpus, dictionary, k, seed):
    return LdaMulticore(corpus, id2word=dictionary, num_topics=k, passes=3, chunksize=4000,
                        random_state=seed, workers=5, alpha="symmetric", eta="auto")


def dominant(lda, corpus):
    return np.array([max(lda.get_document_topics(b, minimum_probability=0), key=lambda x: x[1])[0]
                     for b in corpus])


def main():
    tokens = pd.read_pickle(INTERIM / "tokens.pkl").tolist()  # лематизація з lemmatize.py

    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=20, no_above=0.3, keep_n=50_000)
    corpus = [dictionary.doc2bow(t) for t in tokens]
    ref_texts, ref_dict = reference_corpus(tokens)

    rows = []
    for k in KS:
        print("K =", k, file=sys.stderr, flush=True)
        path = MODELS / "lda" / f"lda_k{k}"
        lda = LdaMulticore.load(str(path)) if path.exists() else fit(corpus, dictionary, k, SEED)
        topics = [[w for w, _ in lda.show_topic(i, topn=10)] for i in range(k)]
        cv, _ = coherence(topics, ref_texts, ref_dict, "c_v")
        npmi, _ = coherence(topics, ref_texts, ref_dict, "c_npmi")
        rows.append({"model": "LDA", "k": k, "c_v": cv, "npmi": npmi,
                     "diversity": topic_diversity(topics)})
        print(rows[-1], file=sys.stderr, flush=True)
        (MODELS / "lda").mkdir(parents=True, exist_ok=True)
        lda.save(str(MODELS / "lda" / f"lda_k{k}"))
        with open(TAB / f"lda_k{k}_topics.json", "w", encoding="utf-8") as f:
            json.dump(topics, f, ensure_ascii=False, indent=1)

    res = pd.DataFrame(rows)
    best_k = int(res.sort_values("npmi", ascending=False).iloc[0]["k"])

    # стабільність: ARI домінантних тем між запусками з різними seed (на 30k документів)
    sub = np.random.default_rng(SEED).choice(len(corpus), 30_000, replace=False)
    sub_corpus = [corpus[i] for i in sub]
    labels = []
    for s in [SEED, 7, 2024]:
        lda = fit(corpus, dictionary, best_k, s) if s != SEED else LdaMulticore.load(str(MODELS / "lda" / f"lda_k{best_k}"))
        labels.append(dominant(lda, sub_corpus))
    aris = [adjusted_rand_score(labels[i], labels[j]) for i in range(3) for j in range(i + 1, 3)]
    res["ari_mean"] = np.nan
    res.loc[res["k"] == best_k, "ari_mean"] = float(np.mean(aris))
    res.to_csv(TAB / "lda_metrics.csv", index=False)
    print(res)
    print("best K", best_k, "ARI", aris)


if __name__ == "__main__":
    main()

"""Чому в BERTopic NPMI нижчий, ніж у LDA?

Гіпотеза: c-TF-IDF обирає рідкісні специфічні слова (прізвища, топоніми), які у 30-тисячному
референсному корпусі майже не зустрічаються разом, і кожна така пара дає NPMI = -1. Перевіряємо:
1) середня документна частота ключових слів;
2) частка пар ключових слів, які не зустрічаються в одному документі жодного разу;
3) NPMI на рівні документа, порахований на всіх ~300k постах, а не на 30k.
"""
import json
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from common import INTERIM, TAB, reference_corpus


def doc_index(token_lists):
    index = defaultdict(set)
    for i, toks in enumerate(token_lists):
        for t in set(toks):
            index[t].add(i)
    return index


def stats(topics, index, n_docs):
    dfs, zero, npmis = [], 0, []
    pairs = 0
    for t in topics:
        words = [w for w in t[:10] if w in index]
        dfs += [len(index[w]) for w in words]
        for a, b in combinations(words, 2):
            pairs += 1
            co = len(index[a] & index[b])
            if co == 0:
                zero += 1
                npmis.append(-1.0)
                continue
            p_ab = co / n_docs
            p_a, p_b = len(index[a]) / n_docs, len(index[b]) / n_docs
            npmis.append(np.log(p_ab / (p_a * p_b)) / -np.log(p_ab))
    return {"median_df": float(np.median(dfs)), "zero_pairs": zero / pairs, "npmi_doc": float(np.mean(npmis))}


def main():
    tokens = pd.read_pickle(INTERIM / "tokens.pkl").tolist()
    ref_texts, _ = reference_corpus(tokens)
    full, small = doc_index(tokens), doc_index(ref_texts)

    best_k = int(pd.read_csv(TAB / "lda_metrics.csv").sort_values("npmi", ascending=False).iloc[0]["k"])
    with open(TAB / f"lda_k{best_k}_topics.json", encoding="utf-8") as f:
        lda = json.load(f)
    with open(TAB / "bertopic_mcs200_topics.json", encoding="utf-8") as f:
        bt = list(json.load(f).values())

    rows = []
    for name, topics in [(f"LDA K={best_k}", lda), ("BERTopic mcs=200", bt)]:
        for ref, idx, n in [("30k", small, len(ref_texts)), ("увесь корпус", full, len(tokens))]:
            rows.append({"model": name, "reference": ref, **stats(topics, idx, n)})
    res = pd.DataFrame(rows).round(3)
    res.to_csv(TAB / "coherence_check.csv", index=False)
    print(res.to_string())


if __name__ == "__main__":
    main()

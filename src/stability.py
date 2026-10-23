"""Стабільність BERTopic: ще два запуски UMAP з іншими seed на тих самих ембедингах,
HDBSCAN з mcs=200, ARI між трьома розбиттями. Рахуємо на тих самих 30k документів, що й
для LDA, двома способами: з шумом як окремою міткою і лише на документах, які в обох
запусках потрапили в якусь тему."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from common import MODELS, SEED, TAB
from topics_bertopic import cluster, reduce

MCS = 200


def main():
    emb = np.load(MODELS / "emb_me5_base.npy")
    runs = [np.load(MODELS / f"labels_mcs{MCS}.npy")]
    for seed in [7, 2024]:
        red = reduce(emb, seed=seed, path=MODELS / f"umap5_seed{seed}.npy")
        runs.append(cluster(red, MCS))
    sub = np.random.default_rng(SEED).choice(len(runs[0]), 30_000, replace=False)
    rows = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = runs[i][sub], runs[j][sub]
            both = (a >= 0) & (b >= 0)
            rows.append({"pair": f"{i}-{j}", "ari_with_noise": adjusted_rand_score(a, b),
                         "ari_clustered_only": adjusted_rand_score(a[both], b[both]),
                         "share_clustered_in_both": both.mean(),
                         "k_i": len(set(runs[i])) - 1, "k_j": len(set(runs[j])) - 1})
    res = pd.DataFrame(rows).round(3)
    res.to_csv(TAB / "bertopic_stability.csv", index=False)
    print(res.to_string())
    print(res.mean(numeric_only=True).round(3))


if __name__ == "__main__":
    main()

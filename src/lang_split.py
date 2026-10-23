"""Чи ділить модель пости за мовою, а не за змістом? Дивимось, куди потрапляють українськомовні
пости: скільки з них опиняються в темах, які майже повністю україномовні (>50%)."""
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore

from common import INTERIM, MODELS, TAB


def concentration(topic, lang):
    d = pd.DataFrame({"topic": topic, "lang": lang})
    d = d[d.topic >= 0]
    uk_share = d.groupby("topic")["lang"].apply(lambda s: (s == "uk").mean())
    uk_topics = uk_share[uk_share > 0.5].index
    uk = d[d.lang == "uk"]
    return {"uk_posts": len(uk), "uk_topics": len(uk_topics),
            "uk_posts_in_uk_topics_%": round(uk.topic.isin(uk_topics).mean() * 100, 1)}


def main():
    df = pd.read_parquet(INTERIM / "sample.parquet", columns=["lang"])
    tokens = pd.read_pickle(INTERIM / "tokens.pkl")
    best_k = int(pd.read_csv(TAB / "lda_metrics.csv").sort_values("npmi", ascending=False).iloc[0]["k"])
    lda = LdaMulticore.load(str(MODELS / "lda" / f"lda_k{best_k}"))
    dictionary = lda.id2word
    lda_topic = np.array([max(lda.get_document_topics(dictionary.doc2bow(t), minimum_probability=0),
                              key=lambda x: x[1])[0] for t in tokens])
    np.save(MODELS / f"lda_k{best_k}_dominant.npy", lda_topic)
    bt_topic = np.load(MODELS / "labels_mcs200.npy")
    res = pd.DataFrame([{"model": f"LDA K={best_k}", **concentration(lda_topic, df.lang)},
                        {"model": "BERTopic mcs=200 (без шуму)", **concentration(bt_topic, df.lang)}])
    res.to_csv(TAB / "lang_split.csv", index=False)
    print(res.to_string())


if __name__ == "__main__":
    main()

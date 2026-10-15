"""Language ID + описова статистика вибірки. Пише results/tables/eda_*.csv і results/figures/eda_*.png,
а у вибірку додає колонку lang."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lingua import Language, LanguageDetectorBuilder

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "interim" / "sample.parquet"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

LANGS = {Language.RUSSIAN: "ru", Language.UKRAINIAN: "uk", Language.BELARUSIAN: "be", Language.ENGLISH: "en"}


def detect(texts):
    det = LanguageDetectorBuilder.from_languages(*LANGS).with_minimum_relative_distance(0.1).build()
    out = det.detect_languages_in_parallel_of([t[:500] for t in texts])
    return [LANGS.get(l, "other") if l is not None else "other" for l in out]


def main():
    df = pd.read_parquet(SAMPLE)
    df["views"] = df["views"].where(df["views"] >= 0)
    if "lang" not in df:
        df["lang"] = detect(df["text"].tolist())
        df.to_parquet(SAMPLE, index=False)

    df["month"] = df["date"].dt.strftime("%Y-%m")
    df["len_words"] = df["text"].str.split().str.len()

    overview = df.groupby("group").agg(
        posts=("text", "size"), channels=("channel", "nunique"),
        median_words=("len_words", "median"), median_views=("views", "median"),
    )
    overview.to_csv(TAB / "eda_overview.csv")
    print(overview)

    pd.crosstab(df["group"], df["source"]).to_csv(TAB / "eda_group_source.csv")
    by_period = pd.crosstab(df["group"], df["period"])
    by_period.to_csv(TAB / "eda_group_period.csv")
    print(by_period)

    lang = pd.crosstab(df["group"], df["lang"], normalize="index").round(4)
    lang.to_csv(TAB / "eda_lang_share.csv")
    print(lang)

    ch_lang = (df.groupby(["group", "channel"])["lang"]
                 .agg(lambda s: s.value_counts(normalize=True).idxmax()).rename("main_lang"))
    ch_uk = (df.assign(uk=df["lang"].eq("uk")).groupby(["group", "channel"])["uk"].mean()
               .rename("uk_share").sort_values(ascending=False))
    ch = pd.concat([ch_lang, ch_uk, df.groupby(["group", "channel"]).size().rename("posts")], axis=1)
    ch.sort_values("uk_share", ascending=False).to_csv(TAB / "eda_channels.csv")
    print(ch[ch["main_lang"] != "ru"])

    # графіки
    fig, ax = plt.subplots(figsize=(10, 4))
    pd.crosstab(df["month"], df["group"]).plot(ax=ax, marker="o", ms=3)
    ax.axvline(list(sorted(df["month"].unique())).index("2022-02"), color="grey", ls="--", lw=1)
    ax.set_title("Постів у вибірці по місяцях")
    ax.set_xlabel("")
    ax.set_ylabel("постів")
    plt.xticks(rotation=60, fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG / "eda_posts_per_month.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    for group, g in df.groupby("group"):
        ax.hist(g["len_words"].clip(upper=600), bins=60, alpha=0.5, density=True, label=group)
    ax.set_title("Довжина поста, слів (обрізано на 600)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG / "eda_length.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    per_ch = df.groupby(["group", "channel"])["views"].median().reset_index()
    for group, g in per_ch.groupby("group"):
        ax.hist(g["views"].clip(lower=1).apply(lambda v: __import__("math").log10(v)), bins=30, alpha=0.5, label=group)
    ax.set_title("Медіана переглядів на канал, log10")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG / "eda_views.png", dpi=150)

    top = df.groupby(["group", "channel"]).size().rename("posts").reset_index()
    top.groupby("group")["posts"].describe().to_csv(TAB / "eda_posts_per_channel.csv")


if __name__ == "__main__":
    main()

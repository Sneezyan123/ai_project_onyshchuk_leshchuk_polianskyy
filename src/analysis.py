"""Відповіді на питання 1-3 на чотирьох групах каналів: Кремль, опозиція, Україна, Білорусь.

Розподіл тем рахуємо без шуму HDBSCAN (-1). Відстань між розподілами - дивергенція
Дженсена-Шеннона з основою 2 (0 = однакові, 1 = не перетинаються). Значущість перевіряємо
перестановочним тестом по каналах: перемішуємо, який канал у якій групі, зберігаючи розміри
груп, і дивимось, як часто випадкове розбиття дає таку саму або більшу JSD. Середнє цього
нуль-розподілу заодно показує, скільки JSD дає сама лише різнорідність каналів.
"""
import json
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from common import FIG, INTERIM, MODELS, SEED, TAB

MCS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
GROUPS = ["kremlin", "opposition", "ukraine", "belarus"]
NAMES = {"kremlin": "Кремль", "opposition": "опозиція", "ukraine": "Україна", "belarus": "Білорусь"}
COLORS = {"kremlin": "#d8433b", "opposition": "#e0a030", "ukraine": "#3b7dd8", "belarus": "#2e9e6a"}
N_PERM = 1000
rng = np.random.default_rng(SEED)


def jsd(a, b):
    a, b = a / a.sum(), b / b.sum()
    return float(jensenshannon(a, b, base=2) ** 2)


def channel_matrix(d, k):
    """Канал x тема: скільки постів каналу в кожній темі."""
    m = pd.crosstab(d["channel"], d["topic"]).reindex(columns=range(k), fill_value=0)
    return m


def perm_test(ma, mb, n=N_PERM):
    """JSD між групами каналів a і b та перестановочний тест по каналах."""
    obs = jsd(ma.sum(0), mb.sum(0))
    pool = np.vstack([ma, mb])
    na = len(ma)
    null = np.empty(n)
    for i in range(n):
        idx = rng.permutation(len(pool))
        null[i] = jsd(pool[idx[:na]].sum(0), pool[idx[na:]].sum(0))
    return obs, float(null.mean()), float((np.sum(null >= obs) + 1) / (n + 1))


def split_half(m, n=100):
    """JSD між двома випадковими половинами каналів однієї групи."""
    out = []
    for _ in range(n):
        idx = rng.permutation(len(m))
        h = len(m) // 2
        out.append(jsd(m[idx[:h]].sum(0), m[idx[h:]].sum(0)))
    return float(np.mean(out))


def main():
    df = pd.read_parquet(INTERIM / "sample.parquet", columns=["group", "source", "channel", "date", "period", "lang", "views"])
    df["topic"] = np.load(MODELS / f"labels_mcs{MCS}.npy")
    with open(TAB / f"bertopic_mcs{MCS}_topics.json", encoding="utf-8") as f:
        kw = {int(k): ", ".join(v[:8]) for k, v in json.load(f).items()}

    noise = df.groupby("group")["topic"].apply(lambda s: round((s == -1).mean(), 3))
    print("частка шуму:", noise.to_dict())
    d = df[df["topic"] >= 0].copy()
    k = d["topic"].max() + 1
    mats = {g: channel_matrix(d[d.group == g], k).to_numpy() for g in GROUPS}

    # ---------- матриця відстаней між групами, весь період і окремо до/після ----------
    rows = []
    for period in ["усе", "before", "after"]:
        dd = d if period == "усе" else d[d.period == period]
        pm = {g: channel_matrix(dd[dd.group == g], k).to_numpy() for g in GROUPS}
        for a, b in combinations(GROUPS, 2):
            obs, null_mean, p = perm_test(pm[a], pm[b], n=N_PERM if period == "усе" else 200)
            rows.append({"period": period, "a": a, "b": b, "jsd": round(obs, 4),
                         "null_mean": round(null_mean, 4), "p": round(p, 4)})
        for g in GROUPS:
            rows.append({"period": period, "a": g, "b": g, "jsd": round(split_half(pm[g]), 4),
                         "null_mean": np.nan, "p": np.nan})
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "jsd_groups.csv", index=False)
    print(res.to_string())

    # ---------- перевірки, чи не міряємо ми мову або джерело замість змісту ----------
    checks = []

    def add_check(name, a, b):
        ma, mb = channel_matrix(a, k).to_numpy(), channel_matrix(b, k).to_numpy()
        obs, null_mean, p = perm_test(ma, mb, n=200)
        checks.append({"check": name, "jsd": round(obs, 4), "null_mean": round(null_mean, 4), "p": round(p, 4),
                       "channels_a": len(ma), "channels_b": len(mb)})

    opp = d[d.group == "opposition"]
    add_check("опозиція: канали з Bawa vs канали з Zenodo", opp[opp.source == "bawa"], opp[opp.source == "zenodo"])
    ua = d[d.group == "ukraine"]
    ua_main = ua.groupby("channel")["lang"].agg(lambda s: s.value_counts().idxmax())
    add_check("Україна: канали, що пишуть переважно українською vs російською",
              ua[ua.channel.isin(ua_main[ua_main == "uk"].index)], ua[ua.channel.isin(ua_main[ua_main == "ru"].index)])
    add_check("Україна: канали з Bawa vs канали з Zenodo", ua[ua.source == "bawa"], ua[ua.source == "zenodo"])
    kr = d[d.group == "kremlin"]
    add_check("Кремль vs Україна, лише російськомовні пости", kr[kr.lang == "ru"], ua[ua.lang == "ru"])
    checks = pd.DataFrame(checks)
    checks.to_csv(TAB / "jsd_checks.csv", index=False)
    print(checks.to_string())

    # ---------- профіль тем по групах ----------
    cnt = pd.crosstab(d["topic"], d["group"]).reindex(columns=GROUPS, fill_value=0)
    share = (cnt + 1) / (cnt + 1).sum()
    topics = pd.DataFrame({f"{g}_%": (cnt[g] / cnt[g].sum() * 100).round(3) for g in GROUPS})
    for g in GROUPS:
        others = share.drop(columns=g).mean(axis=1)
        topics[f"log2_{g}_vs_rest"] = np.log2(share[g] / others).round(2)
    topics["log2_kremlin_vs_opposition"] = np.log2(share["kremlin"] / share["opposition"]).round(2)
    topics["log2_belarus_vs_kremlin"] = np.log2(share["belarus"] / share["kremlin"]).round(2)
    topics["n_posts"] = cnt.sum(axis=1)
    topics["n_channels"] = d.groupby("topic")["channel"].nunique()
    topics["top_channel_share"] = d.groupby("topic")["channel"].agg(lambda s: s.value_counts(normalize=True).iloc[0]).round(2)
    topics["uk_share"] = d.groupby("topic")["lang"].agg(lambda s: (s == "uk").mean()).round(2)
    topics["keywords"] = topics.index.map(kw)
    topics.to_csv(TAB / f"topics_by_group_mcs{MCS}.csv")
    # теми, що тримаються на одному каналі, для висновків про групу не беремо
    solid = topics[(topics.n_channels >= 5) & (topics.top_channel_share < 0.5) & (topics.n_posts >= 300)]
    print("тем:", len(topics), " не одноканальних:", len(solid),
          " одноканальні займають % постів:", round(topics.loc[~topics.index.isin(solid.index), "n_posts"].sum() / topics.n_posts.sum() * 100, 1))
    cols = [f"{g}_%" for g in GROUPS] + ["keywords"]
    for g in GROUPS:
        print(f"\n== найхарактерніші для {g} ==\n", solid.sort_values(f"log2_{g}_vs_rest", ascending=False)[cols].head(12).to_string())
    # Білорусь vs Кремль (питання 3): спільні великі теми і характерні для кожного
    both = solid[(solid["belarus_%"] >= 0.3) & (solid["kremlin_%"] >= 0.3) & (solid.log2_belarus_vs_kremlin.abs() < 1)]
    print("\n== спільні для Білорусі й Кремля ==\n", both.sort_values("belarus_%", ascending=False)[cols].head(12).to_string())
    print("\n== Білорусь, але не Кремль ==\n", solid.sort_values("log2_belarus_vs_kremlin", ascending=False)[cols].head(12).to_string())
    print("\n== Кремль, але не Білорусь ==\n", solid.sort_values("log2_belarus_vs_kremlin")[cols].head(12).to_string())

    # ---------- питання 2: до / після 24.02, лише канали, активні в обох періодах ----------
    act = d.groupby(["channel", "period"]).size().unstack(fill_value=0)
    stable = act[(act.get("before", 0) >= 20) & (act.get("after", 0) >= 20)].index
    shift, drift = [], []
    mid = pd.Timestamp("2021-08-24", tz="UTC")
    for g in GROUPS:
        gs = d[(d.group == g) & d.channel.isin(stable)]
        if gs.empty:
            continue
        mb = channel_matrix(gs[gs.period == "before"], k).to_numpy()
        ma = channel_matrix(gs[gs.period == "after"], k).to_numpy()
        b = gs[gs.period == "before"]
        drift.append({"group": g, "channels": gs.channel.nunique(),
                      "jsd_before_after": round(jsd(mb.sum(0), ma.sum(0)), 4),
                      "jsd_halfyears_before": round(jsd(channel_matrix(b[b.date < mid], k).to_numpy().sum(0),
                                                        channel_matrix(b[b.date >= mid], k).to_numpy().sum(0)), 4)})
        c = pd.crosstab(gs["topic"], gs["period"]).reindex(range(k), fill_value=0)
        s = c / c.sum() * 100
        t = pd.DataFrame({"group": g, "topic": range(k), "before_%": s["before"].round(3), "after_%": s["after"].round(3),
                          "delta_pp": (s["after"] - s["before"]).round(3)})
        t["status"] = np.select([(t["before_%"] < 0.1 * t["after_%"]) & (t["after_%"] >= 0.2),
                                 (t["after_%"] < 0.1 * t["before_%"]) & (t["before_%"] >= 0.2)], ["нова", "зникла"], "")
        shift.append(t)
    drift = pd.DataFrame(drift)
    drift.to_csv(TAB / "shift_drift.csv", index=False)
    print("\n", drift.to_string())
    shift = pd.concat(shift)
    shift["keywords"] = shift["topic"].map(kw)
    shift.to_csv(TAB / f"shift_mcs{MCS}.csv", index=False)
    for g in GROUPS:
        s = shift[shift.group == g]
        print(f"\n== {g}: нових {(s.status == 'нова').sum()}, зниклих {(s.status == 'зникла').sum()} ==")
        print(" зросли:\n", s.sort_values("delta_pp", ascending=False).head(8)[["topic", "before_%", "after_%", "status", "keywords"]].to_string())
        print(" впали:\n", s.sort_values("delta_pp").head(8)[["topic", "before_%", "after_%", "status", "keywords"]].to_string())

    # ---------- графіки ----------
    full = res[res.period == "усе"]
    mat = pd.DataFrame(np.nan, index=GROUPS, columns=GROUPS)
    for _, r in full.iterrows():
        mat.loc[r.a, r.b] = mat.loc[r.b, r.a] = r.jsd
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat.values, cmap="Reds", vmin=0)
    ax.set_xticks(range(4), [NAMES[g] for g in GROUPS])
    ax.set_yticks(range(4), [NAMES[g] for g in GROUPS])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title("JSD між групами (на діагоналі - дві половини\nканалів однієї групи)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(FIG / "jsd_heatmap.png", dpi=150)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, g in zip(axes.flat, GROUPS):
        top = solid.sort_values(f"log2_{g}_vs_rest", ascending=False).head(10)[::-1]
        ax.barh([f"{t}: {kw[t][:42]}" for t in top.index], top[f"log2_{g}_vs_rest"], color=COLORS[g])
        ax.set_title(f"{NAMES[g]}: найхарактерніші теми")
        ax.set_xlabel("log2(частка в групі / середня частка в інших)")
        ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    fig.savefig(FIG / "distinctive_topics.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    sx, sy = solid["kremlin_%"].clip(lower=0.005), solid["belarus_%"].clip(lower=0.005)
    ax.scatter(sx, sy, s=12, alpha=0.6, color="#555")
    lim = [0.005, max(sx.max(), sy.max()) * 1.3]
    ax.plot(lim, lim, color="grey", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for t in list(solid.sort_values("log2_belarus_vs_kremlin").head(4).index) + \
            list(solid.sort_values("log2_belarus_vs_kremlin").tail(5).index) + \
            list(both.sort_values("belarus_%").tail(3).index):
        ax.annotate(kw[t].split(",")[0], (sx[t], sy[t]), fontsize=7)
    ax.set_xlabel("частка теми в Кремлі, %")
    ax.set_ylabel("частка теми в Білорусі, %")
    ax.set_title("Білорусь vs Кремль: кожна точка - тема")
    plt.tight_layout()
    fig.savefig(FIG / "belarus_vs_kremlin.png", dpi=150)

    d["week"] = d["date"].dt.tz_localize(None).dt.to_period("W").dt.start_time
    for g in GROUPS:
        s = shift[shift.group == g]
        chosen = list(s.sort_values("delta_pp", ascending=False).head(4)["topic"]) + list(s.sort_values("delta_pp").head(2)["topic"])
        dd = d[(d.group == g) & d.channel.isin(stable)]
        w = pd.crosstab(dd["week"], dd["topic"], normalize="index").reindex(columns=chosen, fill_value=0) * 100
        fig, ax = plt.subplots(figsize=(10, 4.5))
        w.rolling(2, min_periods=1).mean().plot(ax=ax, lw=1.3)
        ax.axvline(pd.Timestamp("2022-02-24"), color="black", ls="--", lw=1)
        ax.legend([f"{t}: {kw[t][:40]}" for t in chosen], fontsize=7, loc="upper left")
        ax.set_ylabel("% постів тижня")
        ax.set_xlabel("")
        ax.set_title(f"{NAMES[g]}: теми з найбільшою зміною частки")
        plt.tight_layout()
        fig.savefig(FIG / f"weekly_shift_{g}.png", dpi=150)


if __name__ == "__main__":
    main()

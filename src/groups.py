"""Розкладає канали двох датасетів на чотири групи і пише data/interim/groups.csv.

kremlin      прокремлівські канали з Bawa & Kursuncu
opposition   російська і білоруська опозиція: антикремлівські канали Bawa без українських
             + кластер Zenodo "Russian/Belarusian Liberal Opposition" зі стороною pro-Ukrainian
ukraine      українські канали з антикремлівського списку Bawa
             + кластер Zenodo "Pro-Ukrainian Channels" зі стороною pro-Ukrainian
belarus      білоруські провладні канали з кластера Zenodo "Pro-Russian Belarusian Channels"

Якщо канал є в обох датасетах, беремо його пости з Bawa, а з Zenodo не дублюємо.
Ручні правки міток описані нижче, кожна з причиною.
"""
import re

import pandas as pd

from common import INTERIM, ROOT

RAW = ROOT / "data" / "raw"

# українські канали в антикремлівському списку Bawa (перевіряли за назвою і описом каналу)
UA_IN_BAWA = {
    "kyivoperativ", "tch_channel", "tipichnoe_xtz", "uniannet", "insiderukr", "ukraine_365news",
    "berezoview", "oleksiihoncharenko", "gordonuacom", "riamelitopol", "novosti_donbass24", "sbukr",
    "vklochok", "o_arestovich_official", "kievreal1", "forbes_ukraine_official", "spravdi",
    "shrikenews", "gruntmedia", "mov_tribunal",
}
# у списку Bawa як прокремлівські, але це російська опозиція (Zenodo теж відносить їх до опозиції)
KREMLIN_TO_OPPOSITION = {"tvrain", "roman_popkov"}
# американський ультраправий блогер про політику США, до нашої теми стосунку не має
DROP = {"joeldavisx"}

# ознаки білоруського каналу в назві, описі чи username
BY = re.compile(r"беларус|белорус|білорус|минск|мінск|belarus|minsk|gov\.by|belta|\bонт\b|ontnews|atn_|mirby|"
                r"батьк|лукашенк|гомел|гродн|брест|витеб|могил|belvestnik|🇧🇾", re.I)
# білоруські, але фільтр їх не ловить: пул пресслужби Лукашенка, пропагандист СТВ Азарьонок,
# "Шкварка" (є і в датасеті Rudnik & Dukach), "ZмагарОК" (змагар - зневажлива назва опозиції)
BY_MANUAL = {"pul_1", "belshkvarka", "zmagarok"}


def main():
    pro = pd.read_csv(RAW / "bawa" / "pro_kremlin_channels_list.csv")
    anti = pd.read_csv(RAW / "bawa" / "anti_kremlin_channel_list.csv")
    zen = pd.read_csv(RAW / "zenodo" / "leiden_clusters.csv", low_memory=False)
    rud = set(pd.read_csv(RAW / "belarus" / "dat_mapping.csv", usecols=["source"])["source"].str.lower())

    rows = []
    for u in pro["username"].str.strip():
        lu = u.lower()
        if lu in DROP:
            continue
        rows.append((u, None, "opposition" if lu in KREMLIN_TO_OPPOSITION else "kremlin", "bawa"))
    for u in anti["username"].str.strip():
        lu = u.lower()
        if lu in DROP:
            continue
        rows.append((u, None, "ukraine" if lu in UA_IN_BAWA else "opposition", "bawa"))
    bawa = {r[0].lower() for r in rows}

    zen["u"] = zen["username"].fillna("").str.lower()
    text = zen["title"].fillna("") + " " + zen["about"].fillna("") + " " + zen["u"]
    is_by = (zen["cluster"] == "Pro-Russian Belarusian Channels") & (
        text.str.contains(BY) | zen["u"].isin(rud) | zen["u"].isin(BY_MANUAL) | zen["title"].str.contains("Азарёнок", na=False))
    pick = {
        "ukraine": (zen["cluster"] == "Pro-Ukrainian Channels") & (zen["side"] == "pro-Ukrainian"),
        "opposition": (zen["cluster"] == "Russian/Belarusian Liberal Opposition") & (zen["side"] == "pro-Ukrainian"),
        "belarus": is_by,
    }
    for group, mask in pick.items():
        for _, r in zen[mask & ~zen["u"].isin(bawa)].iterrows():
            rows.append((r["username"], int(r["peer_id"]), group, "zenodo"))

    g = pd.DataFrame(rows, columns=["username", "peer_id", "group", "source"]).drop_duplicates("username")
    g.to_csv(INTERIM / "groups.csv", index=False)
    print(pd.crosstab(g["group"], g["source"], margins=True))


if __name__ == "__main__":
    main()

"""Власна лематизація для LDA і c-TF-IDF: pymorphy3 зі словниками російської та української.

Раніше ми брали готове поле processed_message з датасету Bawa, але в Zenodo такого поля немає,
а лематизатор Bawa до того ж калічив українські слова ("даними" -> "дание"). Тепер усі пости
обробляє один і той самий код: мову поста беремо з language ID (eda.py), українські слова
розбирає український словник, решту російський. Англійські слова лишаємо як є, у нижньому
регістрі. Розбір кешуємо по слову, бо унікальних слів у кілька десятків разів менше, ніж токенів.
"""
import sys
from functools import lru_cache

import pandas as pd
import pymorphy3
import stopwordsiso

from common import EXTRA_STOP, INTERIM, TOKEN

MORPH = {"ru": pymorphy3.MorphAnalyzer(lang="ru"), "uk": pymorphy3.MorphAnalyzer(lang="uk")}

# Російський стоп-лист stopwordsiso викидає й змістовні слова: "россия", "война", "москва",
# "народ". Для нашої задачі це неприйнятно, тому з нього лишаємо тільки службові частини мови.
FUNCTION_POS = {"PREP", "CONJ", "PRCL", "NPRO", "INTJ", "PRED", "ADVB", "NUMR", "COMP"}
STOP_RU = {w for w in stopwordsiso.stopwords("ru")
           if MORPH["ru"].parse(w)[0].tag.POS in FUNCTION_POS or "Apro" in str(MORPH["ru"].parse(w)[0].tag)}
STOP = STOP_RU | set(stopwordsiso.stopwords(["uk", "en"])) | EXTRA_STOP | {"быть", "мочь", "стать", "являться"}


@lru_cache(maxsize=2_000_000)
def lemma(word, lang):
    if word.isascii():
        return word
    return MORPH[lang].parse(word)[0].normal_form


def lemmatize(text, lang="ru"):
    lang = "uk" if lang == "uk" else "ru"
    out = []
    for w in TOKEN.findall(text.lower()):
        if w in STOP:
            continue
        l = lemma(w, lang)
        if len(l) > 2 and l not in STOP:
            out.append(l)
    return out


def main():
    df = pd.read_parquet(INTERIM / "sample.parquet", columns=["text", "lang"])
    tokens = []
    for i, (t, l) in enumerate(zip(df["text"], df["lang"])):
        tokens.append(lemmatize(t, l))
        if i % 50_000 == 0:
            print(i, lemma.cache_info().currsize, file=sys.stderr, flush=True)
    pd.Series(tokens).to_pickle(INTERIM / "tokens.pkl")


if __name__ == "__main__":
    main()

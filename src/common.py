"""Спільні речі: шляхи, токени, стоп-слова і метрики якості тем."""
import re
from pathlib import Path

import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
MODELS = ROOT / "models"
SEED = 42

TOKEN = re.compile(r"[а-яёіїєґ][а-яёіїєґ'-]{2,}|[a-z]{3,}")

# Стандартні стоп-листи (stopwordsiso) не знають телеграмних хвостів («подписаться», «канал»)
# і частини українських службових слів, тому доповнюємо вручну.
EXTRA_STOP = set("""
це цей ця ці той та які який яка яке також тому щоб коли вже лише якщо через після може буде були було
його її їх нас вас вони він вона ми ви ще навіть дуже більше саме можна треба тут там зараз
подписаться подписывайтесь подпишись канал канала наш наша ссылка источник читать смотреть видео фото
telegram это весь свой который мочь год также ещё еще підписатися підписуйтесь каналу джерело
стати бути мати сказати говорити заявить заявити сообщать повідомляти
""".split())

def topic_diversity(topics, topn=10):
    words = [w for t in topics for w in t[:topn]]
    return len(set(words)) / max(len(words), 1)


def coherence(topics, texts, dictionary, measure, topn=10):
    """c_v або c_npmi на референсному корпусі. Теми, де менше 2 слів зі словника, пропускаємо."""
    topics = [[w for w in t[:topn] if w in dictionary.token2id] for t in topics]
    topics = [t for t in topics if len(t) >= 2]
    cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary,
                        coherence=measure, topn=topn, processes=1)
    per_topic = cm.get_coherence_per_topic()
    return float(np.nanmean(per_topic)), per_topic


def reference_corpus(token_lists, n=30_000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(token_lists), size=min(n, len(token_lists)), replace=False)
    texts = [token_lists[i] for i in idx]
    return texts, Dictionary(texts)

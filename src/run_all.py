"""Проганяє весь пайплайн мідтерму по черзі. Завантаження, вибірка,
ембединги, UMAP і переклад кешуються: якщо результат уже лежить на диску, крок його не перераховує."""
import subprocess
import sys
from pathlib import Path

STEPS = [
    "download.py", "download_zenodo.py", "groups.py", "sample.py", "eda.py", "lemmatize.py",
    "lda.py", "embed.py", "topics_bertopic.py",
    "analysis.py", "translation_exp.py", "coherence_check.py", "lang_split.py", "stability.py",
]

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for step in STEPS:
        print(f"=== {step}", flush=True)
        subprocess.run([sys.executable, step], cwd=here, check=True)

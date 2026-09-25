"""
embeddings.py — a single embedding function used for BOTH documents and
queries (the task explicitly requires this), so results are comparable.

Primary model: sentence-transformers/all-MiniLM-L6-v2
  - small (~80MB), fast on CPU, strong quality/cost trade-off for a
    knowledge base of this size (dozens of short career-advice PDFs) —
    no need for a heavier model like bge-m3 here.

Fallback: if the sentence-transformers weights can't be downloaded
(e.g. no internet, as in this sandbox), we fall back to a deterministic
TF-IDF vectorizer (scikit-learn) so the pipeline still runs end-to-end.
The interface (`.encode(list[str]) -> np.ndarray`) is identical either
way, so nothing downstream needs to know which backend is active.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.backend = None
        self.model_name = model_name
        self._model = None
        self._tfidf = None
        self._load()

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.backend = "sentence-transformers"
        except Exception as e:  # offline sandbox, first run w/o internet, etc.
            print(
                f"[embeddings] Could not load '{self.model_name}' "
                f"({type(e).__name__}: {e}). Falling back to a local "
                f"TF-IDF embedder so the pipeline can still run."
            )
            self.backend = "tfidf"

    def fit_corpus(self, texts: list[str]) -> None:
        """Only needed for the TF-IDF fallback (it must fit its vocabulary
        on the corpus once, before encoding). No-op for sentence-transformers."""
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
            self._tfidf.fit(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "sentence-transformers":
            return np.asarray(
                self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            )
        elif self.backend == "tfidf":
            if self._tfidf is None:
                raise RuntimeError("Call fit_corpus() before encoding with the TF-IDF fallback.")
            mat = self._tfidf.transform(texts).toarray().astype("float32")
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return mat / norms
        raise RuntimeError("Embedder not loaded.")

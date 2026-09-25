"""
reranker.py — cross-encoder reranking stage.

Why a reranker at all: bi-encoder similarity (used for the initial Top-K
retrieval) embeds the query and each chunk independently, which is fast
but misses fine-grained query-chunk interaction. A cross-encoder scores
the (query, chunk) pair jointly and is much more accurate at judging
"is this chunk actually relevant" — worth the extra latency for a small
Top-K re-score.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 — a standard, small, fast
reranker well-suited to short passages like ours.

Fallback: if the model can't be downloaded, we fall back to a lexical
overlap score (shared-token ratio between query and chunk) so reranking
still does *something* meaningful rather than silently no-op'ing.
"""

from __future__ import annotations

import re
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "of", "in", "on", "at", "to", "for", "with", "about", "as", "by",
    "and", "or", "but", "if", "so", "than", "then", "do", "does", "did",
    "i", "you", "your", "my", "it", "its", "me", "we", "they", "he", "she",
    "how", "when", "where", "why", "can", "could", "should", "would",
    "also", "such", "not", "no", "yes",
}


def _tokenize(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS}


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.backend = None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
            self.backend = "cross-encoder"
        except Exception as e:
            print(
                f"[reranker] Could not load '{model_name}' "
                f"({type(e).__name__}: {e}). Falling back to a lexical "
                f"overlap reranker."
            )
            self.backend = "lexical"

    # Below this lexical-overlap score a chunk is treated as "not actually
    # relevant" rather than just "ranked last" — only meaningful for the
    # offline fallback, since with a real cross-encoder + an LLM in the
    # loop, the LLM itself is instructed to say when context is missing.
    LEXICAL_MIN_SCORE = 0.12

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_n: int = 4
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        if self.backend == "cross-encoder":
            pairs = [(query, c["text"]) for c in candidates]
            scores = self._model.predict(pairs)
        else:
            q_tokens = _tokenize(query)
            scores = []
            for c in candidates:
                c_tokens = _tokenize(c["text"])
                if not q_tokens or not c_tokens:
                    scores.append(0.0)
                    continue
                overlap = len(q_tokens & c_tokens) / len(q_tokens)
                scores.append(overlap)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

        if self.backend == "lexical":
            ranked = [c for c in ranked if c["rerank_score"] >= self.LEXICAL_MIN_SCORE]

        return _dedupe(ranked)[:top_n]


def _dedupe(chunks: list[dict[str, Any]], sim_threshold: float = 0.9) -> list[dict[str, Any]]:
    """Drop near-duplicate chunks (same source+page prose repeated via
    chunk overlap) while never dropping table chunks, which carry unique
    row data even if their surrounding text looks similar."""
    kept: list[dict[str, Any]] = []
    for c in chunks:
        if c["metadata"]["content_type"] == "table":
            kept.append(c)
            continue
        is_dup = False
        for k in kept:
            if k["metadata"]["content_type"] == "table":
                continue
            if k["metadata"]["source"] == c["metadata"]["source"] and k["metadata"]["page"] == c["metadata"]["page"]:
                a, b = _tokenize(k["text"]), _tokenize(c["text"])
                if a and b and len(a & b) / len(a | b) > sim_threshold:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(c)
    return kept

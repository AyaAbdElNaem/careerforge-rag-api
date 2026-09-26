"""
query.py — the ONLINE half of the pipeline.

Question -> query embedding -> Top-K retrieval -> rerank -> context ->
LLM -> {"answer", "sources"}

Models (embedder, reranker) and the vector store connection are created
once per process and reused across calls — this is the "avoid reloading
models unnecessarily" requirement.
"""

from __future__ import annotations

from typing import Any

from .embeddings import Embedder
from .generation import generate_answer
from .reranker import Reranker
from .vectorstore import VectorStore

TOP_K_RETRIEVE = 10
TOP_N_RERANK = 4
MIN_SIMILARITY = 0.05  # filters out clearly-irrelevant retrieval hits


class RagPipeline:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.store = VectorStore(persist_dir=persist_dir)
        self.embedder = Embedder()
        if self.embedder.backend == "tfidf":
            # the fallback embedder needs to be fit on the same corpus it
            # was built on; refit it from what's already indexed in Chroma.
            existing = self.store.collection.get(include=["documents"])
            self.embedder.fit_corpus(existing["documents"])
        self.reranker = Reranker()

    def query(self, question: str) -> dict[str, Any]:
        question = question.strip()
        normalized = question.lower().rstrip("!.؟?، ")

        # trivial greeting — no retrieval needed
        GREETINGS = {"hi", "hello", "hey", "yo", "مرحبا", "أهلا", "اهلا", "السلام عليكم"}
        if normalized in GREETINGS:
            result = {
                "answer": (
                    "Hello! Ask me anything about resumes, job descriptions, "
                    "interviews, salary negotiation, career growth, or data "
                    "career roadmaps — I'll answer from the CareerForge "
                    "knowledge base."
                ),
                "sources": [],
            }
            return result, [], []

        # trivial thanks / gratitude — no retrieval needed, no sources to show
        THANKS = {
            "thanks", "thank you", "thanks!", "thx", "ty",
            "شكرا", "شكراً", "شكرا لك", "شكرًا", "تسلم", "تسلمي",
            "يعطيك العافية", "الله يسلمك", "متشكر", "متشكرة",
        }
        if normalized in THANKS:
            result = {
                "answer": (
                    "العفو! 🙏 تحت أمرك لو عندك أي سؤال تاني عن السيرة الذاتية، "
                    "مقابلات الشغل، التفاوض على الراتب، أو التطور المهني."
                ),
                "sources": [],
            }
            return result, [], []

        q_vec = self.embedder.encode([question])[0]
        retrieved = self.store.query(q_vec, top_k=TOP_K_RETRIEVE)
        retrieved = [r for r in retrieved if r["similarity"] >= MIN_SIMILARITY]

        reranked = self.reranker.rerank(question, retrieved, top_n=TOP_N_RERANK)

        return generate_answer(question, reranked), retrieved, reranked


def query_rag(question: str, pipeline: "RagPipeline | None" = None, persist_dir: str = "./chroma_db") -> dict[str, Any]:
    """Convenience wrapper matching the deployment guide's expected
    signature: query_rag(question: str) -> {"answer": ..., "sources": ...}
    """
    pipe = pipeline or RagPipeline(persist_dir=persist_dir)
    result, _retrieved, _reranked = pipe.query(question)
    return result

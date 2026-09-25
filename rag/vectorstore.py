"""
vectorstore.py — thin wrapper around Chroma (persistent, local, no server
to run — a practical choice for a project this size; FAISS would work
too but lacks Chroma's built-in metadata filtering/storage, which we need
to return source/page/table info with every result).
"""

from __future__ import annotations

import json
from typing import Any

import chromadb


class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "career_kb"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def add(self, chunks: list[dict[str, Any]], embeddings) -> None:
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": c["source"],
                "page": c["page"],
                "pages": json.dumps(c["pages"]),
                "section": c.get("section") or "",
                "content_type": c["content_type"],
                "table_id": c["table_id"] if c["table_id"] is not None else -1,
            }
            for c in chunks
        ]
        self.collection.add(
            ids=ids, embeddings=embeddings.tolist(), documents=documents, metadatas=metadatas
        )

    def query(self, query_embedding, top_k: int = 10) -> list[dict[str, Any]]:
        res = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            meta = dict(meta)
            meta["pages"] = json.loads(meta["pages"])
            out.append({"text": doc, "metadata": meta, "similarity": 1 - dist})
        return out

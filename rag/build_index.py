"""
build_index.py — the OFFLINE half of the pipeline.

PDF -> Extraction -> Multi-page table merge -> Cleaning -> Chunking ->
Embeddings -> Vector DB (persisted to disk)

Run once (or whenever the source PDFs change) — `query_rag()` never
re-extracts or re-embeds anything; it just opens the persisted Chroma
collection created here.
"""

from __future__ import annotations

import argparse
import json
import time

from .chunking import chunk_pages
from .embeddings import Embedder
from .extraction import clean_pages, extract_pdfs, load_pdfs_from_dir
from .vectorstore import VectorStore


def build(pdf_dir: str, persist_dir: str = "./chroma_db", force: bool = False) -> VectorStore:
    store = VectorStore(persist_dir=persist_dir)

    if not force and not store.is_empty():
        print(f"[build_index] Vector store already populated "
              f"({store.collection.count()} chunks) — skipping re-embedding. "
              f"Pass --force to rebuild.")
        return store

    t0 = time.time()
    raw = load_pdfs_from_dir(pdf_dir)
    pages = extract_pdfs(raw)
    pages = clean_pages(pages)
    chunks = chunk_pages(pages)
    print(f"[build_index] {len(pages)} pages -> {len(chunks)} chunks "
          f"({time.time() - t0:.1f}s)")

    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embedder.fit_corpus(texts)  # no-op unless using the TF-IDF fallback

    t1 = time.time()
    vectors = embedder.encode(texts)
    print(f"[build_index] embedded {len(texts)} chunks with "
          f"backend={embedder.backend} ({time.time() - t1:.1f}s)")

    if not store.is_empty() and force:
        # Chroma has no "clear collection" one-liner; recreate it.
        store.client.delete_collection(store.collection.name)
        store = VectorStore(persist_dir=persist_dir)

    store.add(chunks, vectors)
    print(f"[build_index] persisted {store.collection.count()} chunks to "
          f"'{persist_dir}' ({time.time() - t0:.1f}s total)")

    with open(f"{persist_dir}_meta.json", "w") as f:
        json.dump({"embedder_backend": embedder.backend, "n_chunks": len(chunks)}, f)

    return store


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="./kb")
    ap.add_argument("--persist-dir", default="./chroma_db")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    build(args.pdf_dir, args.persist_dir, args.force)

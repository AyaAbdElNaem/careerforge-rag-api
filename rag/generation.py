"""
generation.py — builds the grounded context + calls the LLM.

API key handling: never hard-coded. Reads ANTHROPIC_API_KEY (default) or
OPENAI_API_KEY from the environment (see .env.example). If neither is
set, falls back to a plain extractive answer built directly from the
top reranked chunks, so `query_rag()` still returns a well-formed
{"answer", "sources"} response for local testing without a key.
"""

from __future__ import annotations

import os
from typing import Any

SYSTEM_PROMPT = (
    "You are a career-advice assistant. Answer ONLY using the CONTEXT "
    "provided below, which comes from the CareerForge knowledge base. "
    "Rules:\n"
    "1. Do not use outside knowledge and do not guess.\n"
    "2. If the context does not contain the answer, say clearly that the "
    "knowledge base does not cover it — do not fabricate one.\n"
    "3. Preserve exact numbers, percentages, and table values from the "
    "context rather than paraphrasing them loosely.\n"
    "4. Keep the answer concise and directly responsive to the question."
)


def build_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        meta = c["metadata"]
        pages = meta["pages"]
        page_str = f"p.{pages[0]}" if len(pages) == 1 else f"pp.{pages[0]}-{pages[-1]}"
        tag = f"[{i}] {meta['source']} ({page_str}"
        if meta.get("section"):
            tag += f", section: {meta['section']}"
        if meta["content_type"] == "table":
            tag += f", table {meta['table_id']}"
        tag += ")"
        parts.append(f"{tag}\n{c['text']}")
    return "\n\n".join(parts)


def _sources_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for c in chunks:
        meta = c["metadata"]
        sources.append(
            {
                "source": meta["source"],
                "pages": meta["pages"],
                "section": meta.get("section") or None,
                "content_type": meta["content_type"],
                "table_id": meta["table_id"] if meta["table_id"] != -1 else None,
            }
        )
    return sources


def _call_anthropic(question: str, context: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
            }
        ],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _call_openai(question: str, context: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
        ],
    )
    return resp.choices[0].message.content.strip()


def _call_cohere(question: str, context: str) -> str:
    import cohere

    co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    resp = co.chat(
        model="command-a-plus-05-2026",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
            },
        ],
    )
    # command-a-plus-05-2026 can return a leading "thinking" content block
    # before the actual text block, so filter by type instead of assuming
    # content[0] is the answer (same pattern as _call_anthropic above).
    text_blocks = [b.text for b in resp.message.content if b.type == "text"]
    return "".join(text_blocks).strip()


def _extractive_fallback(question: str, chunks: list[dict[str, Any]]) -> str:
    """No API key configured: return the most relevant retrieved chunks
    verbatim instead of a fabricated 'answer', so the pipeline is honest
    about not having an LLM in the loop."""
    if not chunks:
        return (
            "I don't have an LLM API key configured, and no relevant "
            "context was retrieved for this question either way."
        )
    lines = [
        "[No LLM API key configured — showing the most relevant "
        "retrieved context instead of a generated answer.]\n"
    ]
    for i, c in enumerate(chunks[:2], start=1):
        meta = c["metadata"]
        lines.append(f"({i}) From {meta['source']} p.{meta['page']}:\n{c['text'][:500]}")
    return "\n\n".join(lines)


def generate_answer(question: str, reranked_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    context = build_context(reranked_chunks)
    sources = _sources_from_chunks(reranked_chunks)

    if not reranked_chunks:
        return {
            "answer": (
                "The knowledge base doesn't contain information to answer "
                "that question."
            ),
            "sources": [],
        }

    if os.environ.get("ANTHROPIC_API_KEY"):
        answer = _call_anthropic(question, context)
    elif os.environ.get("OPENAI_API_KEY"):
        answer = _call_openai(question, context)
    elif os.environ.get("COHERE_API_KEY"):
        answer = _call_cohere(question, context)
    else:
        answer = _extractive_fallback(question, reranked_chunks)

    return {"answer": answer, "sources": sources}

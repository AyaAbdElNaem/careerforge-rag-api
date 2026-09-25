"""
chunking.py — context-aware chunking with overlap, tables kept intact.

Fixes applied vs. the original rag_epsilon.py:
  * The original used `llama_index`'s SemanticSplitterNodeParser, which
    (a) forces an extra embedding pass at chunk time on top of the later
    embedding pass for the vector DB (wasted compute / "unnecessary
    repeated embedding" the task explicitly warns against), and
    (b) has no concept of tables, so a table's rows could be split across
    two semantic chunks.
    Replaced with a lightweight recursive character splitter that respects
    paragraph/sentence boundaries and never splits a table mid-row.
  * Tables are chunked as their own atomic unit (one table = one chunk,
    unless it's larger than the max chunk size, in which case it's split
    row-wise so header + table_id + source travel with every piece).
  * Every chunk keeps full metadata: source, page(s), section, content_type,
    table_id (when applicable).
  * True overlap between adjacent text chunks (the original had none).
"""

from __future__ import annotations

import re
from typing import Any

CHUNK_SIZE = 800          # target characters per text chunk
CHUNK_OVERLAP = 120       # characters of overlap between adjacent chunks
MAX_TABLE_ROWS_PER_CHUNK = 12  # split large tables at this row count


def _split_text_with_overlap(text: str, size: int, overlap: int) -> list[str]:
    """Recursive-ish splitter: prefer paragraph breaks, then sentence
    breaks, then hard cut — always leaves `overlap` chars of context at
    the start of the next chunk."""
    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            # prefer the last paragraph/sentence boundary in the window
            for pat in ("\n\n", ". ", "\n", " "):
                idx = window.rfind(pat)
                if idx > size * 0.4:  # don't cut too early
                    end = start + idx + len(pat)
                    break

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks


def _table_chunks(page: dict[str, Any], table: dict[str, Any]) -> list[dict[str, Any]]:
    """One table -> one or more chunks, split by rows only if it's large,
    never mid-row."""
    from .extraction import _table_to_text  # local import, avoids cycle at module load

    rows = table["rows"]
    row_groups = [
        rows[i : i + MAX_TABLE_ROWS_PER_CHUNK]
        for i in range(0, len(rows), MAX_TABLE_ROWS_PER_CHUNK)
    ] or [[]]

    out = []
    for part_idx, group in enumerate(row_groups, start=1):
        sub_table = {"columns": table["columns"], "rows": group}
        table_text = _table_to_text(sub_table)
        suffix = f" (part {part_idx}/{len(row_groups)})" if len(row_groups) > 1 else ""
        out.append(
            {
                "text": (
                    f"[Table {table['table_id']} from {page['source']}"
                    f"{suffix}, columns: {', '.join(table['columns'])}]\n"
                    f"{table_text}"
                ),
                "source": page["source"],
                "page": page["page"],
                "pages": table.get("continued_on_pages", [page["page"]]),
                "section": page.get("section"),
                "content_type": "table",
                "table_id": table["table_id"],
            }
        )
    return out


def chunk_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turns cleaned pages into retrieval-ready chunks with full metadata."""
    chunks: list[dict[str, Any]] = []

    for page in pages:
        # 1) tables -> atomic chunk(s), never mixed into prose text
        for table in page["tables"]:
            chunks.extend(_table_chunks(page, table))

        # 2) prose text -> section-aware, overlapping text chunks.
        #    (Table markdown already lives in page["content"]; use
        #    page["text"] here so prose chunks don't duplicate table rows.)
        prose = page["text"]
        if not prose:
            continue

        heading_prefix = f"{page['section']}\n" if page.get("section") else ""
        for piece in _split_text_with_overlap(prose, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(
                {
                    "text": (heading_prefix + piece) if heading_prefix not in piece else piece,
                    "source": page["source"],
                    "page": page["page"],
                    "pages": [page["page"]],
                    "section": page.get("section"),
                    "content_type": "text",
                    "table_id": None,
                }
            )

    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{c['source']}::p{c['pages'][0]}-{c['pages'][-1]}::{c['content_type']}::{i}"

    return chunks

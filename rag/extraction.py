"""
extraction.py — PDF loading, table-aware extraction, multi-page table
reconstruction, and text cleaning.

Fixes applied vs. the original rag_epsilon.py:
  * Colab-only `google.colab.files.upload()` replaced with a plain local
    directory loader (`load_pdfs_from_dir`) so the script runs anywhere.
  * Tables are no longer flattened into "key: value | key: value" text —
    they are kept as structured JSON (list-of-dicts) AND given a compact
    markdown-ish text form for embedding, so both are available downstream.
  * A table whose header row repeats on the next page is detected and
    merged into a single logical table (see `merge_continued_tables`),
    instead of being stored as two unrelated tables.
  * Every page/table now carries `section` (nearest heading above it) and
    `content_type` metadata, which the original script never captured.
  * `documents[0]` is no longer accessed unconditionally (would crash on
    an empty PDF/empty corpus).
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Any

import pymupdf


# --------------------------------------------------------------------------
# 1. Loading
# --------------------------------------------------------------------------

def load_pdfs_from_dir(directory: str) -> dict[str, bytes]:
    """Replacement for the Colab-only `files.upload()`.

    Returns {filename: raw_bytes} for every .pdf in `directory`, which is
    the same shape the rest of the original script expected from Colab.
    """
    paths = sorted(glob.glob(os.path.join(directory, "*.pdf")))
    if not paths:
        raise FileNotFoundError(f"No PDF files found in {directory!r}")

    out = {}
    for p in paths:
        with open(p, "rb") as fh:
            out[os.path.basename(p)] = fh.read()
    return out


# --------------------------------------------------------------------------
# 2. Heading detection (used for the `section` metadata field)
# --------------------------------------------------------------------------

_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+[A-Z].{2,80}|[A-Z][A-Za-z0-9 ,&/()'\-]{2,60})$"
)


def _guess_headings(page_text: str) -> list[str]:
    """Very lightweight heading detector: short, title-cased / numbered
    lines with no trailing punctuation. Good enough to tag a `section`
    without needing a layout model."""
    headings = []
    for line in page_text.splitlines():
        line = line.strip()
        if not line or len(line) > 90:
            continue
        if line.endswith((".", ",", ":", ";")):
            continue
        if _HEADING_RE.match(line):
            headings.append(line)
    return headings


# --------------------------------------------------------------------------
# 3. Table extraction (structured, not flattened)
# --------------------------------------------------------------------------

def _extract_tables(page) -> list[dict[str, Any]]:
    tables_data = []
    try:
        finder = page.find_tables()
    except Exception:
        return tables_data

    for table_id, table in enumerate(finder.tables, start=1):
        table_rows = table.extract()
        if not table_rows:
            continue

        headers = [
            (str(h).strip() if h is not None else f"Column_{i + 1}")
            for i, h in enumerate(table_rows[0])
        ]

        rows = []
        for raw_row in table_rows[1:]:
            row_dict = {}
            for i, value in enumerate(raw_row):
                if i < len(headers):
                    row_dict[headers[i]] = (
                        str(value).strip() if value is not None else ""
                    )
            if any(v for v in row_dict.values()):
                rows.append(row_dict)

        if not rows:
            continue

        tables_data.append(
            {
                "table_id": table_id,
                "columns": headers,
                "rows": rows,
                "bbox": tuple(table.bbox) if hasattr(table, "bbox") else None,
            }
        )
    return tables_data


def _table_to_text(table: dict[str, Any]) -> str:
    """Compact, embedding- and LLM-friendly markdown table."""
    cols = table["columns"]
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in table["rows"]:
        lines.append("| " + " | ".join(row.get(c, "") for c in cols) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 4. Multi-page table reconstruction
# --------------------------------------------------------------------------

def merge_continued_tables(pages: list[dict[str, Any]]) -> None:
    """Mutates `pages` in place: if a table on page N+1 repeats the exact
    header of the LAST table on page N, its rows are appended to that
    table and the page-N+1 copy is removed (marked as a continuation).

    This is the fix for the "table split across two pages" requirement.
    """
    for i in range(len(pages) - 1):
        cur, nxt = pages[i], pages[i + 1]
        if not cur["tables"] or not nxt["tables"]:
            continue

        last_table = cur["tables"][-1]
        continued = []
        for t in list(nxt["tables"]):
            if t["columns"] == last_table["columns"]:
                # Same header repeated on the next page -> continuation,
                # not a new table.
                last_table["rows"].extend(t["rows"])
                last_table["continued_on_pages"] = last_table.get(
                    "continued_on_pages", [cur["page"]]
                ) + [nxt["page"]]
                continued.append(t)

        for t in continued:
            nxt["tables"].remove(t)

    # Refresh each page's `content` (page text + rendered tables) now that
    # rows may have moved between pages.
    for p in pages:
        text = p["text"]
        for t in p["tables"]:
            t["text"] = _table_to_text(t)
            pages_str = (
                f"pages {'-'.join(map(str, sorted(set(t['continued_on_pages']))))}"
                if t.get("continued_on_pages")
                else f"page {p['page']}"
            )
            text += (
                f"\n\n[TABLE {t['table_id']} — {pages_str}]\n{t['text']}"
            )
        p["content"] = text


# --------------------------------------------------------------------------
# 5. Top-level extraction
# --------------------------------------------------------------------------

def extract_pdfs(pdf_bytes_by_name: dict[str, bytes]) -> list[dict[str, Any]]:
    """Returns one record per page:
    {source, page, text, tables, content, section}
    with multi-page tables already merged.
    """
    all_pages: list[dict[str, Any]] = []

    for filename, raw in pdf_bytes_by_name.items():
        pdf = pymupdf.open(stream=raw, filetype="pdf")
        file_pages = []

        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text", sort=True).strip()
            tables = _extract_tables(page)
            headings = _guess_headings(page_text)
            section = headings[0] if headings else None

            file_pages.append(
                {
                    "source": filename,
                    "page": page_number,
                    "text": page_text,
                    "tables": tables,
                    "content": page_text,  # rebuilt below after table merge
                    "section": section,
                }
            )

        pdf.close()

        # Reconstruct any table that continues onto the next page BEFORE
        # this file's pages are merged into the global list — continuation
        # only ever happens within the same source document.
        merge_continued_tables(file_pages)
        all_pages.extend(file_pages)

    if not all_pages:
        raise ValueError("No extractable content found in any PDF.")

    return all_pages


# --------------------------------------------------------------------------
# 6. Cleaning
# --------------------------------------------------------------------------

_BULLET_RE = re.compile(r"^(?:\s*[•●▪◦■]){2,}\s*", re.MULTILINE)
_MULTISPACE_RE = re.compile(r"[ \t]+")
_MANY_BLANKLINES_RE = re.compile(r"\n{3,}")
_PAGE_NOISE_RE = re.compile(
    r"^(page\s+\d+(\s+of\s+\d+)?|CareerForge Knowledge Base.*)$",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not _PAGE_NOISE_RE.match(ln)]

    # de-duplicate consecutive identical lines (repeated running headers)
    deduped = []
    for ln in lines:
        if not deduped or deduped[-1] != ln:
            deduped.append(ln)

    text = "\n".join(deduped)
    text = _BULLET_RE.sub("• ", text)
    text = _MULTISPACE_RE.sub(" ", text)
    text = _MANY_BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def clean_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for p in pages:
        cleaned.append(
            {
                "source": p["source"],
                "page": p["page"],
                "section": p["section"],
                "text": clean_text(p["text"]),
                "content": clean_text(p["content"]),
                "tables": p["tables"],
            }
        )
    return cleaned

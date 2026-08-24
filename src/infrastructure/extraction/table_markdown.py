"""
src/infrastructure/extraction/table_markdown.py

Shared Markdown conversion for extracted table rows, used identically by
both PdfplumberTableExtractionStrategy and PyMuPdfTableExtractionStrategy
so the output format stays consistent regardless of which one wins.
"""

from __future__ import annotations


def table_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned_rows = [
        [
            str(cell).strip().replace("|", "\\|").replace("\n", " ") if cell else ""
            for cell in row
        ]
        for row in rows
    ]
    cleaned_rows = [row for row in cleaned_rows if any(cell for cell in row)]
    if not cleaned_rows:
        return ""

    lines: list[str] = []
    for i, row in enumerate(cleaned_rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * len(row)) + " |")
    return "\n".join(lines)
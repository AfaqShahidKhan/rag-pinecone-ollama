"""
src/infrastructure/loaders/html_loader.py

IDocumentLoader adapter for HTML files backed by BeautifulSoup4.

Phase 3 update: <table> elements are converted to Markdown tables BEFORE
general text extraction. This preserves tabular structure so:
  - MetadataEnricher can detect has_tables=True
  - The LLM receives structured table data instead of garbled concatenated cells
  - TableTransformer pre-processor (future) can further process Markdown tables

Strategy:
  - Convert all <table> → Markdown in-place on the BeautifulSoup tree
  - Remove non-content tags: <script>, <style>, <nav>, <footer>, <head>
  - Split into sections at <h1>/<h2> boundaries (one Document per section)
  - Fallback to single Document from <body> text if no headings exist

Install:  pip install beautifulsoup4
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from src.domain.entities import Document
from src.domain.interfaces import IDocumentLoader, ILogger

SUPPORTED_EXTENSIONS = {".html", ".htm"}

_NOISE_TAGS = ["script", "style", "nav", "footer", "head", "noscript", "iframe"]
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


class HtmlLoader(IDocumentLoader):
    def __init__(self, logger: ILogger) -> None:
        self._logger = logger

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_EXTENSIONS

    def load(self, path: Path) -> list[Document]:
        raw_html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw_html, "html.parser")

        # 1. Remove noise tags
        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        # 2. Convert <table> elements to Markdown in-place (Phase 3)
        for table in soup.find_all("table"):
            md_text = self._table_to_markdown(table)
            table.replace_with(NavigableString(f"\n\n{md_text}\n\n"))

        # 3. Split into sections
        sections = self._split_into_sections(soup, path)

        if not sections:
            self._logger.warning(f"'{path.name}' yielded no text after parsing — skipping.")
            return []

        self._logger.info(f"Loaded '{path.name}': {len(sections)} section(s).")
        return sections

    def _split_into_sections(self, soup: BeautifulSoup, path: Path) -> list[Document]:
        body = soup.find("body") or soup
        sections: list[Document] = []
        current_heading = path.stem
        current_lines: list[str] = []
        section_index = 0
        headings = soup.find_all(["h1", "h2"])
        total_sections = max(len(headings), 1)

        for element in body.children:
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    current_lines.append(text)
                continue

            if not isinstance(element, Tag):
                continue

            if element.name in ("h1", "h2"):
                doc = self._make_document(
                    current_lines, current_heading, section_index, total_sections, path
                )
                if doc:
                    sections.append(doc)
                    section_index += 1
                current_heading = element.get_text(separator=" ", strip=True)
                current_lines = []
            else:
                text = element.get_text(separator="\n", strip=True)
                if text:
                    current_lines.append(text)

        # Flush last section
        doc = self._make_document(
            current_lines, current_heading, section_index, total_sections, path
        )
        if doc:
            sections.append(doc)

        return sections

    @staticmethod
    def _table_to_markdown(table: Tag) -> str:
        """Convert a BeautifulSoup <table> Tag to a Markdown table string."""
        rows: list[str] = []
        for i, tr in enumerate(table.find_all("tr")):
            cells = [
                cell.get_text(separator=" ", strip=True).replace("|", "\\|")
                for cell in tr.find_all(["th", "td"])
            ]
            if not cells:
                continue
            rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")

        return "\n".join(rows)

    def _make_document(
        self,
        lines: list[str],
        heading: str,
        section_index: int,
        total_sections: int,
        path: Path,
    ) -> Document | None:
        landing_zone = "\n".join(lines)
        text = _MULTI_BLANK_RE.sub("\n\n", _MULTI_SPACE_RE.sub(" ", landing_zone)).strip()
        if not text:
            return None
        return Document(
            page_content=text,
            metadata={
                "source":        path.name,
                "source_path":   str(path),
                "page":          section_index + 1,
                "total_pages":   total_sections,
                "section_title": heading,
                "file_type":     "html",
            },
        )
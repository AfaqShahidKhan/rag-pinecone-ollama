"""
src/infrastructure/loaders/html_loader.py

IDocumentLoader adapter for HTML files backed by BeautifulSoup4.

Strategy:
  - Removes non-content tags: <script>, <style>, <nav>, <footer>, <head>
  - Splits the document into sections at every <h1> or <h2> boundary,
    producing one Document per section (mirrors the page-per-Document
    approach used by PdfDocumentLoader).
  - Falls back to a single Document from <body> text if no headings exist.

Install:  pip install beautifulsoup4
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from src.domain.entities import Document
from src.domain.interfaces import IDocumentLoader, ILogger

SUPPORTED_EXTENSIONS = {".html", ".htm"}

# Tags whose content adds no semantic value
_NOISE_TAGS = ["script", "style", "nav", "footer", "head", "noscript", "iframe"]

# Collapse runs of whitespace / blank lines produced by tag removal
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

        # Remove noise tags in-place
        for tag in soup(SUPPORTED_EXTENSIONS | set(_NOISE_TAGS)):
            tag.decompose()
        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        sections = self._split_into_sections(soup, path)

        if not sections:
            self._logger.warning(f"'{path.name}' yielded no text after parsing — skipping.")
            return []

        self._logger.info(f"Loaded '{path.name}': {len(sections)} section(s).")
        return sections

    def _split_into_sections(self, soup: BeautifulSoup, path: Path) -> list[Document]:
        """Split at h1/h2 boundaries → one Document per section."""
        body = soup.find("body") or soup

        sections: list[Document] = []
        current_heading = path.stem          # use filename as heading for pre-heading content
        current_lines: list[str] = []
        section_index = 0
        total_headings = len(soup.find_all(["h1", "h2"]))
        total_sections = max(total_headings, 1)

        for element in body.children:
            if not isinstance(element, Tag):
                continue

            if element.name in ("h1", "h2"):
                # Flush the current section
                doc = self._make_document(
                    current_lines, current_heading, section_index,
                    total_sections, path
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

        # Flush the last section
        doc = self._make_document(
            current_lines, current_heading, section_index, total_sections, path
        )
        if doc:
            sections.append(doc)

        return sections

    def _make_document(
        self,
        lines: list[str],
        heading: str,
        section_index: int,
        total_sections: int,
        path: Path,
    ) -> Document | None:
        raw = "\n".join(lines)
        text = _MULTI_BLANK_RE.sub("\n\n", _MULTI_SPACE_RE.sub(" ", raw)).strip()
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
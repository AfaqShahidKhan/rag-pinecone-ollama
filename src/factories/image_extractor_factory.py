"""
src/factories/image_extractor_factory.py

Abstract factory: holds the set of registered IImageExtractor adapters and
resolves the correct one per file — mirrors DocumentLoaderFactory. Returns
None for file types with no registered extractor (html/json/ocr sources),
so callers skip cleanly instead of erroring.
"""

from __future__ import annotations

from pathlib import Path

from src.domain.interfaces import IImageExtractor, IImageExtractorResolver, ILogger


class ImageExtractorFactory(IImageExtractorResolver):
    def __init__(self, extractors: list[IImageExtractor], logger: ILogger) -> None:
        self._extractors = extractors
        self._logger = logger

    def resolve_for_file(self, path: Path) -> IImageExtractor | None:
        for extractor in self._extractors:
            if extractor.supports(path):
                return extractor
        return None
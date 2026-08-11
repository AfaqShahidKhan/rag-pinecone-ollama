"""
src/infrastructure/validation/unprocessed_file_mover.py

Moves a rejected file into data/unprocessed/<reason>/, preserving it for
manual inspection instead of silently dropping it or leaving it stuck in
landing_zone. Filename collisions are resolved with a numeric suffix
rather than overwriting.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.domain.interfaces import ILogger, IUnprocessedFileMover


class UnprocessedFileMover(IUnprocessedFileMover):
    def __init__(self, logger: ILogger, unprocessed_dir: str) -> None:
        self._logger = logger
        self._base_dir = Path(unprocessed_dir)

    def move(self, path: Path, reason: str) -> Path:
        dest_dir = self._base_dir / reason
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest = dest_dir / path.name
        counter = 1
        while dest.exists():
            dest = dest_dir / f"{path.stem}_{counter}{path.suffix}"
            counter += 1

        shutil.move(str(path), str(dest))
        self._logger.info(f"Moved '{path.name}' -> '{dest}'")
        return dest
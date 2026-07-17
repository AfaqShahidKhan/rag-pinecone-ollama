"""
src/infrastructure/landing_zone/file_system_watcher.py

ILandingZoneWatcher adapter backed by the `watchdog` library.

Behaviour:
  - Watches a directory (non-recursively by default) for file creation events.
  - Skips temporary files (dot-files, .tmp, .part) that signal an in-progress write.
  - Applies a short stabilisation delay before notifying the adapter, so the
    file is fully written before ingestion begins (prevents partial reads on
    slow copies).
  - Runs in a background daemon thread — calling start() returns immediately.
  - stop() blocks until the observer thread exits cleanly.

Install:  pip install watchdog
"""

from __future__ import annotations

import time
from pathlib import Path

from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.domain.interfaces import IIngestionAdapter, ILandingZoneWatcher, ILogger

# Extensions that indicate an in-progress write — skip these
_SKIP_SUFFIXES = {".tmp", ".part", ".crdownload", ".download"}

# Seconds to wait after a creation event before notifying the adapter
# (allows OS to finish flushing large files to disk)
_STABILISE_SECONDS = 2.0


class _FileEventHandler(FileSystemEventHandler):
    """Internal watchdog handler — bridges OS events to IIngestionAdapter."""

    def __init__(self, adapter: IIngestionAdapter, logger: ILogger) -> None:
        super().__init__()
        self._adapter = adapter
        self._logger = logger

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return

        path = Path(event.src_path)

        # Skip hidden and temporary files
        if path.name.startswith(".") or path.suffix.lower() in _SKIP_SUFFIXES:
            self._logger.debug(f"FileSystemWatcher: skipping temp file '{path.name}'")
            return

        self._logger.info(f"FileSystemWatcher: detected new file '{path.name}'")

        # Brief stabilisation pause
        time.sleep(_STABILISE_SECONDS)

        # Verify file still exists (may have been moved/deleted)
        if not path.exists():
            self._logger.warning(
                f"FileSystemWatcher: '{path.name}' disappeared before ingestion."
            )
            return

        self._adapter.on_file_arrived(path)


class FileSystemWatcher(ILandingZoneWatcher):
    def __init__(
        self,
        adapter: IIngestionAdapter,
        logger: ILogger,
        recursive: bool = False,
    ) -> None:
        """
        Args:
            adapter:   IIngestionAdapter that handles each detected file.
            logger:    Injected logger.
            recursive: If True, watch subdirectories as well.
        """
        self._adapter = adapter
        self._logger = logger
        self._recursive = recursive
        self._observer: Observer | None = None

    def start(self, path: Path) -> None:
        if self._observer and self._observer.is_alive():
            self._logger.warning("FileSystemWatcher: already running — ignoring start().")
            return

        handler = _FileEventHandler(
            adapter=self._adapter,
            logger=self._logger,
        )
        self._observer = Observer()
        self._observer.schedule(handler, str(path), recursive=self._recursive)
        self._observer.daemon = True
        self._observer.start()

        self._logger.info(
            f"FileSystemWatcher: watching '{path}' "
            f"(recursive={self._recursive}). Press Ctrl+C to stop."
        )

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._logger.info("FileSystemWatcher: stopped.")
            self._observer = None

    def is_running(self) -> bool:
        return self._observer is not None and self._observer.is_alive()
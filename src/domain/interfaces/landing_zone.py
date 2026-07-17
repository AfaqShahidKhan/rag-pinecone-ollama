"""
src/domain/interfaces/landing_zone.py

Two ports for the landing zone layer:

  ILandingZoneWatcher  — monitors a directory for new/modified files and
                         notifies a registered IIngestionAdapter.

  IIngestionAdapter    — receives a file path from the watcher and converts
                         it into an internal ingestion call (Interface
                         Segregation: the adapter only knows about file paths,
                         never about OS-level events or watchdog internals).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class IIngestionAdapter(ABC):
    """Converts a raw file-system event into an internal ingestion call."""

    @abstractmethod
    def on_file_arrived(self, path: Path) -> None:
        """
        Called when a new or fully-written file is detected.
        Implementations should be non-blocking where possible.
        """
        ...


class ILandingZoneWatcher(ABC):
    """Monitors a directory and notifies an IIngestionAdapter on new files."""

    @abstractmethod
    def start(self, path: Path) -> None:
        """Begin watching the given directory. Non-blocking — runs in background."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop watching and release all resources."""
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """Return True if the watcher background thread is active."""
        ...
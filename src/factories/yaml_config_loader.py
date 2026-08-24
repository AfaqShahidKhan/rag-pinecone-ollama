"""
src/factories/yaml_config_loader.py

Loads and deep-merges YAML configuration files. Used by SettingsFactory to
build the layered config: config/default.yml as the base, optionally
overridden by a per-user file (e.g. config/user_afaq.yml).

This is the only module allowed to call yaml.safe_load() — nothing else in
the codebase should read a YAML config file directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class YamlConfigLoader:
    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root

    def load(self, config_file: str | Path | None = None) -> dict[str, Any]:
        """
        Load config/default.yml (if present) and deep-merge an optional
        user config file on top of it. Returns {} if neither file exists —
        callers then fall back to env vars / dataclass defaults entirely.
        """
        merged = self._merge({}, self._read(self._project_root / "config" / "default.yml"))

        if config_file:
            path = Path(config_file)
            if not path.is_absolute():
                path = self._project_root / path
            merged = self._merge(merged, self._read(path))

        return merged

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}

    @classmethod
    def _merge(cls, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge override into base, returning a new dict."""
        result = dict(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = cls._merge(result[key], value)
            else:
                result[key] = value
        return result
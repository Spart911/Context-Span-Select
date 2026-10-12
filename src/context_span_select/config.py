from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from context_span_select.paths import CONFIGS, ROOT


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config; support one-level `extends: relative/or/absolute`."""
    path = Path(path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "extends" in raw:
        parent = Path(raw["extends"])
        if not parent.is_absolute():
            parent = (ROOT / parent).resolve()
        base = load_config(parent)
        return _deep_merge(base, raw)
    return raw


def default_config() -> dict[str, Any]:
    return load_config(CONFIGS / "default.yaml")

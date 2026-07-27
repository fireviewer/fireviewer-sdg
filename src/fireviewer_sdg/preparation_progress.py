"""Persistent, truthful progress for input and asset preparation."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROGRESS_NAME = "input-preparation-progress.json"
PROGRESS_SCHEMA_VERSION = 1


def progress_path(volume_root: Path) -> Path:
    return volume_root.resolve() / "input" / PROGRESS_NAME


def write_progress(
    volume_root: Path,
    *,
    phase: str,
    state: str = "running",
    message: str,
    **facts: Any,
) -> dict[str, Any]:
    if state not in {"pending", "running", "completed", "blocked"}:
        raise ValueError(f"unsupported preparation progress state: {state}")
    payload = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "state": state,
        "phase": phase,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **facts,
    }
    path = progress_path(volume_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
    return payload


def load_progress(volume_root: Path) -> dict[str, Any] | None:
    path = progress_path(volume_root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PROGRESS_SCHEMA_VERSION
    ):
        return None
    return payload


__all__ = [
    "PROGRESS_NAME",
    "load_progress",
    "progress_path",
    "write_progress",
]

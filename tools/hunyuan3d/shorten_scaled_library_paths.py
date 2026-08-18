#!/usr/bin/env python3
"""Migrate a scaled library to short, Windows-safe per-index directories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def rewrite_paths(value: Any, old_root: Path, new_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: rewrite_paths(item, old_root, new_root) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_paths(item, old_root, new_root) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        try:
            relative = candidate.resolve().relative_to(old_root.resolve())
        except (OSError, ValueError):
            return value
        return str((new_root / relative).resolve())
    return value


def run(args: argparse.Namespace) -> int:
    manifest_path = args.active_manifest.resolve()
    library_root = manifest_path.parent
    assets_root = library_root / "assets"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = list(manifest.get("assets", []))
    if len(assets) != int(manifest.get("asset_count", -1)):
        raise RuntimeError("active manifest asset count mismatch")

    moved = 0
    already_short = 0
    for asset in assets:
        index = int(asset["index"])
        asset_id = str(asset["asset_id"])
        old_root = assets_root / f"{index:03d}_{asset_id}"
        new_root = assets_root / f"{index:03d}"
        if not inside(old_root, assets_root) or not inside(new_root, assets_root):
            raise RuntimeError(f"unsafe asset directory for index {index}")
        if old_root.is_dir() and not new_root.exists():
            os.replace(old_root, new_root)
            moved += 1
        elif new_root.is_dir() and not old_root.exists():
            already_short += 1
        else:
            raise RuntimeError(
                f"ambiguous migration state for {index}: old={old_root.exists()}, new={new_root.exists()}"
            )

        report_path = new_root / "asset-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report = rewrite_paths(report, old_root, new_root)
        for key in ("glb", "usd", "texture"):
            path = Path(report[key])
            if not inside(path, new_root) or not path.is_file():
                raise RuntimeError(f"rewritten {key} is invalid for index {index}: {path}")
        atomic_json(report_path, report)
        asset.clear()
        asset.update(report)

    directories = [path for path in assets_root.iterdir() if path.is_dir()]
    directory_indices = {int(path.name) for path in directories if path.name.isdigit()}
    expected_indices = {int(asset["index"]) for asset in assets}
    if len(directories) != len(assets) or directory_indices != expected_indices:
        raise RuntimeError("short directory inventory mismatch after migration")
    manifest["path_policy"] = "windows_safe_index_directories"
    manifest["assets"] = assets
    atomic_json(manifest_path, manifest)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "asset_count": len(assets),
        "moved_count": moved,
        "already_short_count": already_short,
        "path_policy": "assets/<three-digit-index>/<full-asset-filename>",
        "passed": True,
    }
    atomic_json(library_root / "path-migration.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-manifest", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

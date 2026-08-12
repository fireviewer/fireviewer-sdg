#!/usr/bin/env python3
"""Inventory Asset4Sim references and route 3D, terrain, and documentation inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return result[:72] or "asset"


def route(relative: Path) -> str:
    top = relative.parts[0].lower()
    if top.startswith("pack_dalles_terrain_2d"):
        return "terrain_2d"
    if top == "validation_documentation":
        return "documentation"
    return "hunyuan3d"


def build(reference_root: Path) -> dict[str, object]:
    assets = []
    for path in sorted(reference_root.rglob("*.png")):
        relative = path.relative_to(reference_root)
        relative_posix = relative.as_posix()
        stable_id = hashlib.sha256(relative_posix.lower().encode("utf-8")).hexdigest()[:12]
        asset_id = f"{stable_id}_{slug(relative.stem)}"
        assets.append(
            {
                "asset_id": asset_id,
                "route": route(relative),
                "source": str(path.resolve()),
                "source_relative": relative_posix,
                "input_name": f"asset4sim/reference/{relative_posix}",
                "source_bytes": path.stat().st_size,
                "source_sha256": file_sha256(path),
                "seed": int(stable_id[:8], 16),
            }
        )

    counts: dict[str, int] = {}
    for asset in assets:
        counts[asset["route"]] = counts.get(asset["route"], 0) + 1
    return {
        "schema_version": 1,
        "reference_root": str(reference_root.resolve()),
        "asset_count": len(assets),
        "route_counts": counts,
        "assets": assets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = build(args.reference_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["route_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

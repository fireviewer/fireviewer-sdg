#!/usr/bin/env python3
"""Validate geometry, UVs and embedded base-color textures in delivered GLBs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def texture_size(material: Any) -> list[int] | None:
    for name in ("baseColorTexture", "image"):
        image = getattr(material, name, None)
        if image is not None and getattr(image, "size", None):
            return [int(image.size[0]), int(image.size[1])]
    return None


def inspect_glb(path: Path) -> dict[str, Any]:
    scene = trimesh.load(path, force="scene", process=False)
    meshes = [mesh for mesh in scene.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("no triangle mesh")
    faces = sum(len(mesh.faces) for mesh in meshes)
    vertices = sum(len(mesh.vertices) for mesh in meshes)
    uv_meshes = 0
    textured_meshes = 0
    texture_sizes: list[list[int]] = []
    for mesh in meshes:
        uv = getattr(mesh.visual, "uv", None)
        if uv is not None and len(uv) == len(mesh.vertices) and np.all(np.isfinite(uv)):
            uv_meshes += 1
        material = getattr(mesh.visual, "material", None)
        size = texture_size(material) if material is not None else None
        if size:
            textured_meshes += 1
            texture_sizes.append(size)
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    bounds_valid = bool(bounds.shape == (2, 3) and np.all(np.isfinite(bounds)) and np.all(bounds[1] > bounds[0]))
    passed = bool(faces > 0 and vertices > 0 and bounds_valid and uv_meshes == len(meshes) and textured_meshes == len(meshes))
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "meshes": len(meshes),
        "vertices": vertices,
        "faces": faces,
        "uv_meshes": uv_meshes,
        "textured_meshes": textured_meshes,
        "texture_sizes": texture_sizes,
        "bounds": bounds.tolist(),
        "bounds_diagonal": float(math.dist(bounds[0], bounds[1])) if bounds_valid else None,
        "passed": passed,
    }


def run(args: argparse.Namespace) -> int:
    files = sorted(args.input_dir.rglob("*.glb"))
    assets: list[dict[str, Any]] = []
    for path in files:
        try:
            assets.append({"asset_id": path.parent.name, **inspect_glb(path)})
        except Exception as exc:
            assets.append({"asset_id": path.parent.name, "path": str(path.resolve()), "passed": False, "error": str(exc)})
    passed_count = sum(bool(item.get("passed")) for item in assets)
    report = {
        "schema_version": 1,
        "input_dir": str(args.input_dir.resolve()),
        "expected_count": args.expected_count,
        "asset_count": len(assets),
        "passed_count": passed_count,
        "failed_count": len(assets) - passed_count,
        "passed": bool(len(assets) == args.expected_count and passed_count == len(assets)),
        "assets": assets,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("asset_count", "passed_count", "failed_count", "passed")}))
    return 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

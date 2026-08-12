#!/usr/bin/env python3
"""Prepare the calibrated, source-backed library for the compact Z16 scene.

The recovered assets have deliberately been kept as authored by their original
publishers.  Several of those source layers use centimetre-scale coordinates,
which is valid data but would make a fire appliance, a tree or a building
hundreds of metres wide if it were placed directly in the stage.  This command
creates a small *composition manifest*: it never edits an asset and it only
uses the real HERO/MID/FAR files proved by the native LOD receipt.

The manifest has two purposes:

* apply the fixed, explicit real-world scale at the prototype boundary;
* prevent a scene builder from using a low-detail forest pack as a near-camera
  tree or from turning a scanned building into a repeated generic house.

It is intentionally source data under ``workspace/`` rather than repository
content.  The scene builder consumes this manifest after it has validated the
compact raster and vector inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


MATERIALIZATION_RECEIPT = "materialization-receipt.json"
LOD_RECEIPT = "lod-receipt.json"
OUTPUT_DIRECTORY = "scene-library"
OUTPUT_MANIFEST = "z16-scene-library.json"
STATE = "Z16_SCENE_LIBRARY_READY"
SCHEMA_VERSION = 1


class SceneLibraryError(RuntimeError):
    """Raised when a Z16 source-backed scene library cannot be proved."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneLibraryError(f"invalid JSON document: {path}") from exc
    if not isinstance(value, dict):
        raise SceneLibraryError(f"JSON document must contain an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _safe_relative(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise SceneLibraryError(f"{label} must be a non-empty relative path")
    normalized = PurePosixPath(raw.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise SceneLibraryError(f"{label} is an unsafe relative path")
    return Path(*normalized.parts)


def _asset_id(raw: object) -> str:
    if not isinstance(raw, str) or len(raw) != 32:
        raise SceneLibraryError(f"invalid asset identifier: {raw!r}")
    value = raw.lower()
    if any(character not in "0123456789abcdef" for character in value):
        raise SceneLibraryError(f"invalid asset identifier: {raw!r}")
    return value


def _world_dimensions(raw: object, *, asset_id: str) -> list[float]:
    if not isinstance(raw, dict):
        raise SceneLibraryError(f"{asset_id} has no native world bounds")
    dimensions = raw.get("dimensions")
    if not isinstance(dimensions, list) or len(dimensions) != 3:
        raise SceneLibraryError(f"{asset_id} has invalid native dimensions")
    values = [float(value) for value in dimensions]
    if any(value <= 0.0 for value in values):
        raise SceneLibraryError(f"{asset_id} has non-positive native dimensions")
    return values


def _world_minimum(raw: object, *, asset_id: str) -> list[float]:
    if not isinstance(raw, dict):
        raise SceneLibraryError(f"{asset_id} has no native world bounds")
    minimum = raw.get("minimum")
    if not isinstance(minimum, list) or len(minimum) != 3:
        raise SceneLibraryError(f"{asset_id} has invalid native world minimum")
    return [float(value) for value in minimum]


def _normalised_dimensions(dimensions: list[float], scale: float) -> list[float]:
    return [round(value * scale, 6) for value in dimensions]


# Every decision here is deliberately explicit.  A generic "scale down every
# large asset" heuristic caused the former pile-up / invisible-scene failures.
# The source filenames and measured native bounds identify the five metric
# assets and the four centimetre-authored ones.  Composition may not add an
# asset outside this table without a new library revision.
_PROFILES: dict[str, dict[str, Any]] = {
    "94ef5c37c3c543fd9efbaa571a7a7590": {
        "family": "ground_response_vehicle",
        "placement_class": "actor_ground",
        "uniform_scale": 0.01,
        "max_instances_per_scene": 3,
        "minimum_uniform_scale": 0.85,
        "maximum_uniform_scale": 1.15,
        "near_camera_allowed": True,
        "dense_forest_fill_allowed": False,
    },
    "8f62ab4eacbc430186d85a7029d7d156": {
        "family": "fixed_wing_response_aircraft",
        "placement_class": "actor_air",
        "uniform_scale": 0.01,
        "max_instances_per_scene": 2,
        "minimum_uniform_scale": 0.9,
        "maximum_uniform_scale": 1.1,
        "near_camera_allowed": True,
        "dense_forest_fill_allowed": False,
    },
    "6246617aeb874e4793b21d5861eea8c9": {
        "family": "heavy_lift_helicopter",
        "placement_class": "actor_air",
        "uniform_scale": 1.0,
        "max_instances_per_scene": 2,
        "minimum_uniform_scale": 0.9,
        "maximum_uniform_scale": 1.1,
        "near_camera_allowed": True,
        "dense_forest_fill_allowed": False,
    },
    "fc2b5eb692ca40c2b44357b62eb149df": {
        "family": "response_truck",
        "placement_class": "actor_ground",
        "uniform_scale": 0.01,
        "max_instances_per_scene": 4,
        "minimum_uniform_scale": 0.8,
        "maximum_uniform_scale": 1.2,
        "near_camera_allowed": True,
        "dense_forest_fill_allowed": False,
    },
    "c573303be1f04e0c94cfa245c2f2ddcf": {
        "family": "heavy_ground_vehicle",
        "placement_class": "actor_ground",
        "uniform_scale": 0.01,
        "max_instances_per_scene": 3,
        "minimum_uniform_scale": 0.85,
        "maximum_uniform_scale": 1.15,
        "near_camera_allowed": True,
        "dense_forest_fill_allowed": False,
    },
    "cf138b8eb2d340cda643ed59f824989c": {
        "family": "hero_tree",
        "placement_class": "vegetation",
        "uniform_scale": 0.01,
        "max_instances_per_scene": None,
        "minimum_uniform_scale": 0.55,
        "maximum_uniform_scale": 2.25,
        "near_camera_allowed": True,
        "dense_forest_fill_allowed": True,
    },
    "9153c2b370934758bf14c395abe36b27": {
        "family": "forest_canopy_cluster",
        "placement_class": "vegetation",
        "uniform_scale": 0.01,
        "max_instances_per_scene": None,
        "minimum_uniform_scale": 0.8,
        "maximum_uniform_scale": 1.25,
        "near_camera_allowed": False,
        "dense_forest_fill_allowed": True,
    },
    "b75d4fbbee614c4898ee5214b9fd04aa": {
        "family": "rural_building",
        "placement_class": "building",
        "uniform_scale": 0.01,
        "max_instances_per_scene": 18,
        "minimum_uniform_scale": 0.7,
        "maximum_uniform_scale": 1.35,
        "near_camera_allowed": True,
        "dense_forest_fill_allowed": False,
    },
    "17861ac10c5e480d84339ed6d6cf8073": {
        "family": "village_anchor",
        "placement_class": "building",
        "uniform_scale": 0.01,
        "max_instances_per_scene": 3,
        "minimum_uniform_scale": 0.8,
        "maximum_uniform_scale": 1.2,
        "near_camera_allowed": True,
        "dense_forest_fill_allowed": False,
    },
}


def _validate_receipts(site_root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    materialization_path = site_root / "assets" / "materialized" / MATERIALIZATION_RECEIPT
    lod_root = site_root / "assets" / "lod"
    lod_path = lod_root / LOD_RECEIPT
    materialization = _read_json(materialization_path)
    lod = _read_json(lod_path)
    if materialization.get("state") != "Z16_RECOVERED_ASSETS_MATERIALIZED":
        raise SceneLibraryError("materialized source receipt is not ready")
    if lod.get("state") != "Z16_RECOVERED_ASSET_LODS_READY":
        raise SceneLibraryError("native LOD receipt is not ready")
    assets = lod.get("assets")
    if not isinstance(assets, list) or len(assets) != len(_PROFILES):
        raise SceneLibraryError("native LOD receipt must contain the complete asset set")
    if set(_PROFILES) != {_asset_id(item.get("asset_id")) for item in assets if isinstance(item, dict)}:
        raise SceneLibraryError("native LOD receipt does not match the fixed Z16 asset set")
    return materialization, lod, lod_root


def prepare_scene_library(*, site_root: Path) -> dict[str, Any]:
    root = site_root.resolve()
    materialization, lod, lod_root = _validate_receipts(root)
    lod_assets = lod["assets"]
    assert isinstance(lod_assets, list)
    library_assets: list[dict[str, Any]] = []
    for item in sorted(lod_assets, key=lambda value: _asset_id(value.get("asset_id"))):
        if not isinstance(item, dict):
            raise SceneLibraryError("native LOD receipt contains a non-object asset")
        asset_id = _asset_id(item.get("asset_id"))
        role = item.get("role")
        profile = _PROFILES[asset_id]
        if role not in {"actor", "vegetation", "building"}:
            raise SceneLibraryError(f"{asset_id} has invalid role: {role!r}")
        if profile["placement_class"] == "vegetation" and role != "vegetation":
            raise SceneLibraryError(f"{asset_id} has inconsistent vegetation role")
        if profile["placement_class"] == "building" and role != "building":
            raise SceneLibraryError(f"{asset_id} has inconsistent building role")
        if profile["placement_class"].startswith("actor_") and role != "actor":
            raise SceneLibraryError(f"{asset_id} has inconsistent actor role")
        raw_lods = item.get("lod_paths")
        if not isinstance(raw_lods, dict) or set(raw_lods) != {"HERO", "MID", "FAR"}:
            raise SceneLibraryError(f"{asset_id} lacks a complete real LOD chain")
        lod_paths: dict[str, dict[str, str]] = {}
        for level in ("HERO", "MID", "FAR"):
            record = raw_lods[level]
            if not isinstance(record, dict):
                raise SceneLibraryError(f"{asset_id}.{level} record is invalid")
            relative = _safe_relative(record.get("path"), label=f"{asset_id}.{level}")
            path = (lod_root / relative).resolve()
            if not _inside(lod_root, path) or not path.is_file() or path.is_symlink():
                raise SceneLibraryError(f"{asset_id}.{level} LOD is missing or unsafe")
            checksum = record.get("sha256")
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise SceneLibraryError(f"{asset_id}.{level} LOD checksum is invalid")
            if _sha256(path) != checksum:
                raise SceneLibraryError(f"{asset_id}.{level} LOD checksum drifted")
            lod_paths[level] = {
                "path": path.relative_to(root).as_posix(),
                "sha256": checksum,
            }
        metrics = item.get("native_metrics")
        if not isinstance(metrics, dict) or not isinstance(metrics.get("HERO"), dict):
            raise SceneLibraryError(f"{asset_id} has no native HERO metric")
        hero = metrics["HERO"]
        raw_bounds = hero.get("world_bounds")
        dimensions = _world_dimensions(raw_bounds, asset_id=asset_id)
        minimum = _world_minimum(raw_bounds, asset_id=asset_id)
        scale = float(profile["uniform_scale"])
        library_assets.append(
            {
                "asset_id": asset_id,
                "role": role,
                "family": profile["family"],
                "placement_class": profile["placement_class"],
                "lod_paths": lod_paths,
                "uniform_scale": scale,
                "native_dimensions_as_authored": dimensions,
                "dimensions_metres_after_scale": _normalised_dimensions(dimensions, scale),
                "native_dimensions_m": {
                    axis: value
                    for axis, value in zip(
                        ("x", "y", "z"),
                        _normalised_dimensions(dimensions, scale),
                        strict=True,
                    )
                },
                "ground_anchor_m": _normalised_dimensions(minimum, scale),
                "minimum_uniform_scale": profile["minimum_uniform_scale"],
                "maximum_uniform_scale": profile["maximum_uniform_scale"],
                "max_instances_per_scene": profile["max_instances_per_scene"],
                "near_camera_allowed": profile["near_camera_allowed"],
                "dense_forest_fill_allowed": profile["dense_forest_fill_allowed"],
                "source_is_real_asset": True,
                "primitive_substitution": "forbidden",
            }
        )
    output = root / "assets" / OUTPUT_DIRECTORY / OUTPUT_MANIFEST
    payload = {
        "schema_version": SCHEMA_VERSION,
        "state": STATE,
        "site_id": "Z16-base-01",
        "source_materialization_receipt": {
            "path": (root / "assets" / "materialized" / MATERIALIZATION_RECEIPT)
            .relative_to(root)
            .as_posix(),
            "sha256": _sha256(root / "assets" / "materialized" / MATERIALIZATION_RECEIPT),
        },
        "source_lod_receipt": {
            "path": (root / "assets" / "lod" / LOD_RECEIPT).relative_to(root).as_posix(),
            "sha256": _sha256(root / "assets" / "lod" / LOD_RECEIPT),
        },
        "asset_count": len(library_assets),
        "assets": library_assets,
        "composition_rules": {
            "terrain": "one real MNT-derived mesh payload per one-kilometre source tile",
            "imagery": "each terrain payload uses its matching BD ORTHO texture; no stretched zone mosaic",
            "vegetation": "MNH-guided placement; hero_tree near cameras and canopy_cluster only as dense-forest fill",
            "buildings": "real vector footprint context plus bounded real-asset placement; no primitive buildings",
            "roads_and_water": "source vector and orthophoto context; no synthetic replacement curves",
            "lod": "only real native HERO/MID/FAR asset chains",
        },
        "editor_opening_contract": {
            "must_open_before_campaign": True,
            "must_verify": [
                "asset metre scale and ground anchoring",
                "tree axis and density across all sixteen payloads",
                "building placement without overlaps or repeated scan tiling",
                "orthophoto continuity and absence of terrain-grid seams",
            ],
        },
        "next_required_step": "build_compact_sim_01_with_raster_tile_payloads",
    }
    _write_json(output, payload)
    return {
        "state": STATE,
        "manifest": output.relative_to(root).as_posix(),
        "asset_count": len(library_assets),
        "manifest_sha256": _sha256(output),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = prepare_scene_library(site_root=args.site_root)
    except SceneLibraryError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

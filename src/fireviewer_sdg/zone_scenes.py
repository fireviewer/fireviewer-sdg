"""Fail-closed, sequential production for the external 20-zone USD catalog.

The catalog itself is deliberately external to Git.  This module records its
fingerprints in the production workspace and never writes into it.  It is also
deliberately independent from :mod:`fireviewer_sdg.event_catalog`: the latter
describes synthetic fire events, while this module describes geographic scene
sources and their resulting OpenUSD packages.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from threading import Event, Lock
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from fireviewer_sdg.artifacts import sha256, write_json


PIPELINE_ID = "fireviewer-zone-scenes-v1"
SCHEMA_VERSION = 1
WFS_BASE = "https://data.geopf.fr/wfs/ows"
WFS_DATASETS = {
    "lidar": (
        "IGNF_NUAGES-DE-POINTS-LIDAR-HD:dalle",
        "IGNF_NUAGES-DE-POINTS-LIDAR-HD:dalles",
        "IGNF_LIDAR-HD_TA:nuage-dalle",
        "IGNF_LIDAR-HD_TA:nuage-dalles",
    ),
    "mnt": (
        "IGNF_MNT-LIDAR-HD:dalle",
        "IGNF_LIDAR-HD_TA:mnt-dalle",
        "IGNF_LIDAR-HD_TA:mnt-dalles",
    ),
    "mns": (
        "IGNF_MNS-LIDAR-HD:dalle",
        "IGNF_LIDAR-HD_TA:mns-dalle",
        "IGNF_LIDAR-HD_TA:mns-dalles",
    ),
    "mnh": (
        "IGNF_MNH-LIDAR-HD:dalle",
        "IGNF_LIDAR-HD_TA:mnh-dalle",
        "IGNF_LIDAR-HD_TA:mnh-dalles",
    ),
}
# These are the published BDTOPO V3 feature types verified from the live IGN
# WFS capabilities.  They are deliberately kept as distinct scene concerns:
# production must not silently substitute an untraceable generic asset for a
# missing building, road, hydrography or vegetation source.
VECTOR_SOURCE_LAYERS = {
    "buildings": ("BDTOPO_V3:batiment",),
    "roads": ("BDTOPO_V3:troncon_de_route",),
    "hydrology": (
        "BDTOPO_V3:surface_hydrographique",
        "BDTOPO_V3:cours_d_eau",
    ),
    "vegetation": ("BDTOPO_V3:zone_de_vegetation",),
}
VECTOR_PAGE_SIZE = 1000
ALL_DATASETS = ("lidar", "mnt", "mns", "mnh", "ortho20", "ortho50")
BASELINE_DATASETS = ("mnt", "mns", "mnh", "ortho50")
LOD0_DATASETS = ("lidar", "ortho20")
DIRECT_ELEVATION_DATASETS = frozenset(("lidar", "mnt", "mns", "mnh"))
SOURCE_PROFILES = ("full", "light")
LIGHT_PROFILE = "light"
LIGHT_PROFILE_VERSION = 3
LIGHT_TERRAIN_RESOLUTION_METRES = 8
LIGHT_TERRAIN_WIDTH = 2500
LIGHT_ORTHOPHOTO_RESOLUTION_METRES = 2
LIGHT_ORTHOPHOTO_WIDTH = 5000
LIGHT_ORTHOPHOTO_LAYER = "ORTHOIMAGERY.ORTHOPHOTOS.BDORTHO"
# A 2 by 2 kilometre review square is small enough to stay local, while four
# 20 cm source tiles give a real ground-level surface around the hero cameras.
LIGHT_HERO_TILE_COUNT = 4
MAX_RASTER_DOWNLOAD_WORKERS = 128
DEFAULT_RASTER_DOWNLOAD_WORKERS = 128
MAX_DIRECT_DOWNLOAD_WORKERS = 128
DEFAULT_DIRECT_DOWNLOAD_WORKERS = 128
# The IGN Géoplateforme contract permits 10 requests/s/IP. Keep a safety
# margin below that ceiling while overlapping independent tile transfers.
DIRECT_REQUEST_START_INTERVAL_SECONDS = 0.11
TAIL_SEGMENTATION_POOL_DIVISOR = 4
TAIL_SEGMENT_MIN_BYTES = 8 * 1024 * 1024
LIGHT_EXTERNAL_ASSETS = (
    {
        "id": "polyhaven-pine-sapling-small-gltf",
        "asset_id": "pine_sapling_small",
        "relative_path": "vegetation/pine_sapling_small.gltf",
        "url": "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/pine_sapling_small/pine_sapling_small_1k.gltf",
        "version": "1k glTF",
    },
    {
        "id": "polyhaven-pine-sapling-small-bin",
        "asset_id": "pine_sapling_small",
        "relative_path": "vegetation/pine_sapling_small.bin",
        "url": "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/pine_sapling_small/pine_sapling_small.bin",
        "version": "1k glTF",
    },
    {
        "id": "polyhaven-fir-sapling-medium-gltf",
        "asset_id": "fir_sapling_medium",
        "relative_path": "vegetation/fir_sapling_medium.gltf",
        "url": "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/fir_sapling_medium/fir_sapling_medium_1k.gltf",
        "version": "1k glTF",
    },
    {
        "id": "polyhaven-fir-sapling-medium-bin",
        "asset_id": "fir_sapling_medium",
        "relative_path": "vegetation/fir_sapling_medium.bin",
        "url": "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/fir_sapling_medium/fir_sapling_medium.bin",
        "version": "1k glTF",
    },
    {
        "id": "polyhaven-red-slate-roof-diffuse",
        "asset_id": "red_slate_roof_tiles_01",
        "relative_path": "materials/red_slate_roof_tiles_01_diff_1k.jpg",
        "url": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/red_slate_roof_tiles_01/red_slate_roof_tiles_01_diff_1k.jpg",
        "version": "1k JPG",
    },
    {
        "id": "polyhaven-red-slate-roof-roughness",
        "asset_id": "red_slate_roof_tiles_01",
        "relative_path": "materials/red_slate_roof_tiles_01_rough_1k.jpg",
        "url": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/red_slate_roof_tiles_01/red_slate_roof_tiles_01_rough_1k.jpg",
        "version": "1k JPG",
    },
)
REVIEW_CAMERA_TARGET_OFFSETS_METRES = (
    (-7000, -7000),
    (-3000, -7000),
    (1000, -7000),
    (5000, -7000),
    (-7000, -2000),
    (-3000, -2000),
    (1000, -2000),
    (5000, -2000),
    (-7000, 3000),
    (-3000, 3000),
    (1000, 3000),
    (5000, 3000),
)
PHASES = (
    "preflight",
    "resolve",
    "acquire",
    "build",
    "review",
    "render",
    "qa",
    "archive",
    "cleanup",
)
ZONE_ORDER = (
    "Z16",
    "Z10",
    "Z08",
    "Z19",
    "Z17",
    "Z18",
    "Z01",
    "Z02",
    "Z03",
    "Z04",
    "Z05",
    "Z06",
    "Z07",
    "Z09",
    "Z11",
    "Z12",
    "Z13",
    "Z14",
    "Z15",
    "Z20",
)
TILE_RE = re.compile(r"(?<!\d)(\d{3,4})[_-](\d{4})(?!\d)")
SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
DEFAULT_WORKSPACE_ROOT = Path("D:/FVS/workspace/fireviewer-sdg")
CATALOG_DIRNAME = "zone-scenes"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _is_below(root: Path, candidate: Path) -> bool:
    root = root.resolve()
    candidate = candidate.resolve()
    return candidate == root or root in candidate.parents


def _require_absolute_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is absent or is not a directory: {resolved}")
    return resolved


def _catalog_root(value: str | Path | None) -> Path:
    raw = str(value or os.getenv("FW_SDG_ZONE_CATALOG_ROOT", "")).strip()
    if not raw:
        raise ValueError("--catalog-root or FW_SDG_ZONE_CATALOG_ROOT is required")
    return _require_absolute_directory(Path(raw), label="catalog root")


def _workspace_root(value: str | Path | None) -> Path:
    raw = str(value or os.getenv("FW_SDG_ZONE_WORKSPACE_ROOT", "")).strip()
    root = Path(raw) if raw else DEFAULT_WORKSPACE_ROOT
    if not root.is_absolute():
        raise ValueError("workspace root must be an absolute path")
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _validate_roots(catalog_root: Path, workspace_root: Path) -> None:
    if _is_below(catalog_root, workspace_root) or _is_below(workspace_root, catalog_root):
        raise ValueError("catalog root and workspace root must not contain one another")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is absent: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _open_csv(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def _load_csv(path: Path, *, label: str) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise ValueError(f"{label} is absent: {path}")
    with _open_csv(path) as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"{label} has no header: {path}")
        return list(reader), list(reader.fieldnames)


def _manifest_paths(catalog_root: Path) -> list[Path]:
    paths = sorted((catalog_root / "manifests").glob("Z*.json"))
    if len(paths) != 20:
        raise ValueError(f"catalog must contain exactly 20 zone manifests, got {len(paths)}")
    return paths


def _parse_checksums(catalog_root: Path) -> dict[Path, str]:
    path = catalog_root / "SHA256SUMS.txt"
    if not path.is_file():
        raise ValueError("catalog SHA256SUMS.txt is absent")
    result: dict[Path, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value, separator, relative = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"invalid SHA256SUMS entry: {line!r}")
        candidate = (catalog_root / relative).resolve()
        if not _is_below(catalog_root, candidate):
            raise ValueError("catalog checksum path escapes the catalog")
        result[candidate] = value
    return result


def _verify_catalog_checksums(catalog_root: Path) -> list[dict[str, Any]]:
    checksums = _parse_checksums(catalog_root)
    verified: list[dict[str, Any]] = []
    for path, expected in sorted(checksums.items(), key=lambda item: item[0].as_posix()):
        if not path.is_file():
            raise ValueError(f"catalog file listed in SHA256SUMS is absent: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"catalog checksum mismatch: {path.relative_to(catalog_root)}")
        verified.append(
            {
                "relpath": path.relative_to(catalog_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    return verified


def _zone_manifest(catalog_root: Path, zone_id: str) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in _manifest_paths(catalog_root):
        payload = _read_json(path, label="zone manifest")
        if str(payload.get("zone", {}).get("id", "")) == zone_id:
            matches.append((path, payload))
    if len(matches) != 1:
        raise ValueError(f"zone manifest is not unique for {zone_id}")
    return matches[0]


def validate_catalog(catalog_root: Path) -> dict[str, Any]:
    """Validate the external delivery without mutating it.

    The uncompressed inventory is deliberately fingerprinted separately because
    this delivery's SHA256SUMS file only lists its compressed variant.
    """

    package = _read_json(catalog_root / "package_manifest.json", label="package manifest")
    if package.get("crs") != "EPSG:2154":
        raise ValueError("catalog CRS must be EPSG:2154")
    if int(package.get("zone_count", 0)) != 20 or int(package.get("tile_count", 0)) != 8000:
        raise ValueError("catalog must declare 20 zones and 8000 tiles")

    zones, _ = _load_csv(catalog_root / "zones_summary.csv", label="zone summary")
    zone_ids = [str(row.get("id", "")).strip() for row in zones]
    if len(zone_ids) != 20 or len(set(zone_ids)) != 20 or set(zone_ids) != set(ZONE_ORDER):
        raise ValueError("zone summary must contain exactly the expected 20 zone ids")

    inventory, fields = _load_csv(catalog_root / "tiles_1km_inventory.csv", label="tile inventory")
    required_fields = {
        "zone_id", "tile_ref", "xmin", "ymin", "xmax", "ymax",
        "lidar_expected", "mnt_expected", "mns_expected", "mnh_expected",
        "ortho_ref", "ortho_wms_20cm", "ortho_wms_50cm",
    }
    if not required_fields.issubset(fields):
        raise ValueError("tile inventory is missing required production columns")
    if len(inventory) != 8000:
        raise ValueError(f"tile inventory must contain 8000 rows, got {len(inventory)}")
    rows_by_zone: dict[str, list[dict[str, str]]] = {zone_id: [] for zone_id in zone_ids}
    refs: set[str] = set()
    for row in inventory:
        zone_id = str(row.get("zone_id", "")).strip()
        tile_ref = str(row.get("tile_ref", "")).strip()
        if zone_id not in rows_by_zone or not tile_ref:
            raise ValueError("tile inventory contains an unknown zone or blank tile_ref")
        if tile_ref in refs:
            raise ValueError(f"tile inventory duplicates tile_ref: {tile_ref}")
        refs.add(tile_ref)
        try:
            xmin, ymin, xmax, ymax = (int(row[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"tile {tile_ref} has invalid Lambert-93 bounds") from exc
        if xmax - xmin != 1000 or ymax - ymin != 1000:
            raise ValueError(f"tile {tile_ref} is not a one-kilometre cell")
        rows_by_zone[zone_id].append(row)
    if any(len(items) != 400 for items in rows_by_zone.values()):
        raise ValueError("every zone must contain exactly 400 one-kilometre tiles")

    manifests: list[dict[str, Any]] = []
    for zone_id in zone_ids:
        path, payload = _zone_manifest(catalog_root, zone_id)
        tiles = payload.get("tiles")
        if not isinstance(tiles, list) or len(tiles) != 400:
            raise ValueError(f"manifest {zone_id} must contain exactly 400 tiles")
        manifest_refs = {str(item.get("tile_ref", "")) for item in tiles if isinstance(item, dict)}
        inventory_refs = {str(item["tile_ref"]) for item in rows_by_zone[zone_id]}
        if manifest_refs != inventory_refs:
            raise ValueError(f"manifest and CSV tiles disagree for {zone_id}")
        manifests.append(
            {
                "zone_id": zone_id,
                "relpath": path.relative_to(catalog_root).as_posix(),
                "sha256": sha256(path),
            }
        )

    inventory_path = catalog_root / "tiles_1km_inventory.csv"
    return {
        "pipeline": PIPELINE_ID,
        "validated_at": _utc_now(),
        "catalog_root": str(catalog_root),
        "package_manifest_sha256": sha256(catalog_root / "package_manifest.json"),
        "inventory": {
            "relpath": "tiles_1km_inventory.csv",
            "rows": len(inventory),
            "sha256": sha256(inventory_path),
            "bytes": inventory_path.stat().st_size,
        },
        "zones": {zone_id: len(rows_by_zone[zone_id]) for zone_id in zone_ids},
        "manifests": manifests,
        "verified_catalog_files": _verify_catalog_checksums(catalog_root),
    }


def _state_path(workspace_root: Path) -> Path:
    return workspace_root / CATALOG_DIRNAME / "production-state.json"


def _load_state(workspace_root: Path) -> dict[str, Any]:
    path = _state_path(workspace_root)
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "pipeline": PIPELINE_ID,
            "zone_order": list(ZONE_ORDER),
            "zones": {},
        }
    state = _read_json(path, label="zone production state")
    if state.get("schema_version") != SCHEMA_VERSION or state.get("pipeline") != PIPELINE_ID:
        raise ValueError("unsupported zone production state")
    if state.get("zone_order") != list(ZONE_ORDER) or not isinstance(state.get("zones"), dict):
        raise ValueError("zone production state is malformed")
    return state


def _write_state(workspace_root: Path, state: dict[str, Any]) -> Path:
    state["updated_at"] = _utc_now()
    path = _state_path(workspace_root)
    write_json(path, state)
    return path


def _zone_root(workspace_root: Path, zone_id: str) -> Path:
    root = workspace_root / CATALOG_DIRNAME / zone_id
    if not _is_below(workspace_root / CATALOG_DIRNAME, root):
        raise ValueError("zone workspace path escapes the zone-scenes root")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _zone_state(state: dict[str, Any], zone_id: str) -> dict[str, Any]:
    zones = state["zones"]
    assert isinstance(zones, dict)
    item = zones.setdefault(zone_id, {"phase": "not_started", "history": []})
    if not isinstance(item, dict) or not isinstance(item.get("history"), list):
        raise ValueError(f"zone state is malformed for {zone_id}")
    return item


def _variant_base_zones() -> tuple[str, ...] | None:
    """Return the explicit independent base portfolio, when configured."""

    raw = os.getenv("FW_SDG_VARIANT_BASE_ZONES", "").strip()
    if not raw:
        return None
    values = tuple(value.strip() for value in raw.split(","))
    if (
        len(values) != 4
        or len(set(values)) != 4
        or any(value not in ZONE_ORDER for value in values)
    ):
        raise ValueError(
            "FW_SDG_VARIANT_BASE_ZONES must contain exactly four distinct "
            "catalog zone identifiers"
        )
    return values


def _assert_turn(state: dict[str, Any], zone_id: str) -> None:
    variant_bases = _variant_base_zones()
    if variant_bases is not None:
        if zone_id not in variant_bases:
            raise RuntimeError(
                f"{zone_id} is outside the configured four-scene variant portfolio"
            )
        # The four source scenes must remain concurrently available because
        # their terrain, water and object identities feed five variants each.
        # The legacy sequential archive/cleanup rule would delete a base before
        # the portfolio authoring pass can bind it.
        return
    index = ZONE_ORDER.index(zone_id)
    if index == 0:
        return
    predecessor = _zone_state(state, ZONE_ORDER[index - 1])
    if predecessor.get("phase") != "cleanup_complete":
        raise RuntimeError(
            f"{zone_id} is blocked until {ZONE_ORDER[index - 1]} is archived and cleaned"
        )


def _record_phase(
    state: dict[str, Any], zone_id: str, phase: str, *, details: dict[str, Any]
) -> None:
    zone = _zone_state(state, zone_id)
    zone["phase"] = phase
    zone["updated_at"] = _utc_now()
    # Consumers such as review, render and QA use the current zone state to
    # locate the immutable artifacts from the preceding gate.  Retaining the
    # details only in history makes a valid build impossible to reopen even
    # though its receipt has just been verified.
    zone.update(details)
    history = zone["history"]
    assert isinstance(history, list)
    history.append({"phase": phase, "at": zone["updated_at"], **details})


def _zone_rows(catalog_root: Path, zone_id: str) -> list[dict[str, str]]:
    rows, _ = _load_csv(catalog_root / "tiles_1km_inventory.csv", label="tile inventory")
    selected = [row for row in rows if str(row.get("zone_id", "")) == zone_id]
    if len(selected) != 400:
        raise ValueError(f"zone {zone_id} must resolve to exactly 400 inventory rows")
    return selected


def _recursive_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _recursive_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_strings(item)


def _geometry_bounds(geometry: object) -> tuple[float, float, float, float] | None:
    if not isinstance(geometry, dict):
        return None
    values: list[tuple[float, float]] = []

    def walk(value: object) -> None:
        if isinstance(value, (list, tuple)) and len(value) >= 2 and isinstance(value[0], (float, int)) and isinstance(value[1], (float, int)):
            values.append((float(value[0]), float(value[1])))
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(geometry.get("coordinates"))
    if not values:
        return None
    return min(x for x, _ in values), min(y for _, y in values), max(x for x, _ in values), max(y for _, y in values)


def _tile_ref_from_feature(feature: object, expected: set[str]) -> str | None:
    if not isinstance(feature, dict):
        return None
    for text in _recursive_strings(feature.get("properties", {})):
        for match in TILE_RE.finditer(text):
            reference = f"L93_{int(match.group(1)):04d}_{int(match.group(2)):04d}"
            if reference in expected:
                return reference
    bounds = _geometry_bounds(feature.get("geometry"))
    if bounds:
        xmin, _ymin, _xmax, ymax = bounds
        reference = f"L93_{int((xmin + 1e-6) // 1000):04d}_{int(-(- (ymax - 1e-6) // 1000)):04d}"
        if reference in expected:
            return reference
    return None


def _pick_url(properties: object) -> str:
    if not isinstance(properties, dict):
        return ""
    preferred = ("url", "href", "download", "download_url", "lien", "uri", "location", "resource", "fichier", "file")
    lowered = {str(key).lower(): value for key, value in properties.items()}
    for key in preferred:
        value = lowered.get(key)
        if isinstance(value, str) and value.startswith(("https://", "http://")):
            return value
    return next((item for item in _recursive_strings(properties) if item.startswith(("https://", "http://"))), "")


def _pick_name(properties: object) -> str:
    if not isinstance(properties, dict):
        return ""
    for key in ("name", "nom", "filename", "file_name", "fichier", "title", "titre"):
        value = properties.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _request_collection(layer: str, bbox: str, *, timeout: float, retries: int) -> dict[str, Any]:
    query = urlencode(
        {
            "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
            "TYPENAMES": layer, "OUTPUTFORMAT": "application/json", "SRSNAME": "EPSG:2154",
            "BBOX": f"{bbox},EPSG:2154", "COUNT": "1000",
        }
    )
    request = Request(f"{WFS_BASE}?{query}", headers={"User-Agent": "FireViewer-Zone-Scenes/1.0"})
    failure: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
                raise RuntimeError("WFS did not return a GeoJSON FeatureCollection")
            return payload
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            failure = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"WFS resolution failed for {layer}: {failure}")


def _resolve_rows(rows: list[dict[str, str]], *, timeout: float, retries: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    expected = {row["tile_ref"] for row in rows}
    xmin = min(int(row["xmin"]) for row in rows)
    ymin = min(int(row["ymin"]) for row in rows)
    xmax = max(int(row["xmax"]) for row in rows)
    ymax = max(int(row["ymax"]) for row in rows)
    report: dict[str, Any] = {"bbox_epsg2154": [xmin, ymin, xmax, ymax], "datasets": {}}
    for dataset, candidates in WFS_DATASETS.items():
        resolved: dict[str, dict[str, str]] = {}
        errors: list[str] = []
        selected_layer = ""
        for layer in candidates:
            try:
                collection = _request_collection(layer, f"{xmin},{ymin},{xmax},{ymax}", timeout=timeout, retries=retries)
            except RuntimeError as exc:
                errors.append(str(exc))
                continue
            for feature in collection.get("features", []):
                tile_ref = _tile_ref_from_feature(feature, expected)
                if not tile_ref or not isinstance(feature, dict):
                    continue
                properties = feature.get("properties", {})
                resolved[tile_ref] = {"url": _pick_url(properties), "name": _pick_name(properties)}
            selected_layer = layer
            break
        for row in rows:
            item = resolved.get(row["tile_ref"], {})
            row[f"{dataset}_resolved_url"] = item.get("url", "")
            row[f"{dataset}_resolved_name"] = item.get("name", "")
            row[f"{dataset}_wfs_layer"] = selected_layer
            row[f"{dataset}_status"] = "available" if item.get("url") else "unresolved"
        report["datasets"][dataset] = {
            "layer": selected_layer,
            "resolved_tiles": sum(1 for item in resolved.values() if item.get("url")),
            "expected_tiles": len(expected),
            "errors": errors,
        }
    return rows, report


def _source_lock_from_rows(
    *, catalog_receipt: dict[str, Any], zone_id: str, rows: list[dict[str, str]], resolution_report: dict[str, Any]
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        for dataset in ALL_DATASETS:
            if dataset in WFS_DATASETS:
                url = row.get(f"{dataset}_resolved_url", "")
                name = row.get(f"{dataset}_resolved_name") or row.get(f"{dataset}_expected", "")
                status = row.get(f"{dataset}_status", "unresolved")
                layer = row.get(f"{dataset}_wfs_layer", "")
            else:
                suffix = "20cm" if dataset == "ortho20" else "50cm"
                url = row.get(f"ortho_wms_{suffix}", "")
                name = f"{row.get('ortho_ref', row['tile_ref'])}_{suffix}"
                status = "declared_wms_reference" if url else "unresolved"
                layer = "ORTHOIMAGERY.ORTHOPHOTOS.BDORTHO"
            entries.append(
                {
                    "id": f"{row['tile_ref']}:{dataset}",
                    "tile_ref": row["tile_ref"],
                    "dataset": dataset,
                    "expected_name": name,
                    "url": url,
                    "wfs_layer": layer,
                    "resolution_status": status,
                    "license": "Licence Ouverte / Etalab 2.0" if dataset != "ortho20" and dataset != "ortho50" else "IGN BD ORTHO stated resource terms",
                    "content_length_bytes": None,
                    "download": {"state": "pending"},
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE_ID,
        "zone_id": zone_id,
        "created_at": _utc_now(),
        "catalog_inventory_sha256": catalog_receipt["inventory"]["sha256"],
        "resolution_report": resolution_report,
        "entries": entries,
    }


def _source_lock_path(zone_root: Path) -> Path:
    return zone_root / "source-lock.json"


def _load_source_lock(zone_root: Path) -> dict[str, Any]:
    return _read_json(_source_lock_path(zone_root), label="zone source lock")


def _light_terrain_url(*, template_url: str, bbox: tuple[int, int, int, int], zone_id: str) -> str:
    """Build one locked 8 m MNT request for the complete 20 km zone."""

    parsed = urlparse(template_url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    if query.get("SERVICE", [""])[-1].upper() != "WMS":
        raise ValueError("the light terrain source requires an IGN WMS MNT template")
    xmin, ymin, xmax, ymax = bbox
    query.update(
        {
            "BBOX": [f"{xmin},{ymin},{xmax},{ymax}"],
            "WIDTH": [str(LIGHT_TERRAIN_WIDTH)],
            "HEIGHT": [str(LIGHT_TERRAIN_WIDTH)],
            "FILENAME": [f"{zone_id}_terrain_lod3_{LIGHT_TERRAIN_RESOLUTION_METRES}m.tif"],
        }
    )
    return parsed._replace(query=urlencode(query, doseq=True)).geturl()


def _light_orthophoto_entries(
    *, zone_id: str, bbox: tuple[int, int, int, int]
) -> list[dict[str, Any]]:
    """Lock four 10 km BD ORTHO images as the full-zone visual context.

    Two-metre imagery keeps the full 20 km landscape readable from review
    cameras.  It is deliberately paired with the camera-local 20 cm source
    below: neither is allowed to masquerade as the other.
    """

    xmin, ymin, xmax, ymax = bbox
    xmid = xmin + (xmax - xmin) // 2
    ymid = ymin + (ymax - ymin) // 2
    entries: list[dict[str, Any]] = []
    for column, left, right in ((0, xmin, xmid), (1, xmid, xmax)):
        for row, bottom, top in ((0, ymin, ymid), (1, ymid, ymax)):
            query = urlencode(
                {
                    "SERVICE": "WMS",
                    "VERSION": "1.3.0",
                    "REQUEST": "GetMap",
                    "LAYERS": LIGHT_ORTHOPHOTO_LAYER,
                    "FORMAT": "image/jpeg",
                    "STYLES": "",
                    "CRS": "EPSG:2154",
                    "BBOX": f"{left},{bottom},{right},{top}",
                    "WIDTH": str(LIGHT_ORTHOPHOTO_WIDTH),
                    "HEIGHT": str(LIGHT_ORTHOPHOTO_WIDTH),
                }
            )
            entries.append(
                {
                    "id": f"{zone_id}:ortho_lod2_{column}_{row}",
                    "tile_ref": "__zone__",
                    "dataset": "ortho_lod2",
                    "expected_name": f"{zone_id}_ortho_lod2_{column}_{row}.jpg",
                    "url": f"https://data.geopf.fr/wms-r?{query}",
                    "wfs_layer": LIGHT_ORTHOPHOTO_LAYER,
                    "bbox_epsg2154": [left, bottom, right, top],
                    "resolution_status": "derived_wms_context_2m",
                    "license": "Licence Ouverte / Etalab 2.0",
                    "attribution": "IGN orthophotos via Géoplateforme WMS",
                    "content_length_bytes": None,
                    "download": {"state": "pending"},
                }
            )
    return entries


def _light_hero_orthophoto_entries(
    *, zone_id: str, rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Lock four real 20 cm BD ORTHO tiles around the zone centre.

    The catalogue already gives final WMS requests for each kilometre tile.
    Selecting by centre distance makes this deterministic for every zone and
    avoids inventing a loose camera bounding box outside the delivered grid.
    """

    if len(rows) < LIGHT_HERO_TILE_COUNT:
        raise ValueError("a light scene needs at least four catalogue tiles")
    bbox = _zone_bbox(rows)
    centre_x = (bbox[0] + bbox[2]) * 0.5
    centre_y = (bbox[1] + bbox[3]) * 0.5
    ranked = sorted(
        rows,
        key=lambda row: (
            (float(row["xmin"]) + float(row["xmax"])) * 0.5 - centre_x
        ) ** 2
        + (
            (float(row["ymin"]) + float(row["ymax"])) * 0.5 - centre_y
        ) ** 2,
    )
    entries: list[dict[str, Any]] = []
    for row in ranked[:LIGHT_HERO_TILE_COUNT]:
        url = str(row.get("ortho_wms_20cm", "")).strip()
        if not url.startswith(("https://", "http://")):
            raise RuntimeError(
                f"catalogue has no traceable 20 cm orthophoto URL for {row['tile_ref']}"
            )
        entries.append(
            {
                "id": f"{row['tile_ref']}:ortho_lod0",
                "tile_ref": row["tile_ref"],
                "dataset": "ortho_lod0",
                "expected_name": f"{row['tile_ref']}_ortho_lod0_20cm.jpg",
                "url": url,
                "wfs_layer": "ORTHOIMAGERY.ORTHOPHOTOS.BDORTHO",
                "bbox_epsg2154": [
                    int(row["xmin"]),
                    int(row["ymin"]),
                    int(row["xmax"]),
                    int(row["ymax"]),
                ],
                "resolution_status": "catalogue_wms_lod0_20cm",
                "license": "IGN BD ORTHO stated resource terms",
                "attribution": "IGN BD ORTHO via Géoplateforme WMS",
                "download": {"state": "pending"},
            }
        )
    return entries


def _light_external_asset_entries() -> list[dict[str, Any]]:
    """Return the complete, traceable CC0 asset bundle used by the scene.

    The glTFs are not enough on their own: each sidecar is individually locked
    so the native converter cannot silently resolve a remote or untracked
    material at build time.
    """

    entries = [
        {
            **asset,
            "dataset": "assets",
            "expected_name": Path(str(asset["relative_path"])).name,
            "license": "CC0 1.0",
            "attribution": "Poly Haven",
            "source": "https://polyhaven.com/license",
            "download": {"state": "pending"},
        }
        for asset in LIGHT_EXTERNAL_ASSETS
    ]
    trees = {
        "pine_sapling_small": (
            "pine_sapling_small_bark_nor_gl_1k.jpg",
            "pine_sapling_small_bark_diff_1k.jpg",
            "pine_sapling_small_bark_arm_1k.jpg",
            "pine_sapling_small_twig_nor_gl_1k.jpg",
            "pine_sapling_small_twig_diff_1k.jpg",
            "pine_sapling_small_twig_arm_1k.jpg",
        ),
        "fir_sapling_medium": (
            "fir_sapling_medium_branches_nor_gl_1k.jpg",
            "fir_sapling_medium_branches_diff_1k.jpg",
            "fir_sapling_medium_branches_arm_1k.jpg",
            "fir_sapling_medium_twigs_nor_gl_1k.jpg",
            "fir_sapling_medium_twigs_diff_1k.jpg",
            "fir_sapling_medium_twigs_arm_1k.jpg",
        ),
    }
    for asset_id, texture_names in trees.items():
        for name in texture_names:
            entries.append(
                {
                    "id": f"polyhaven-{asset_id}-{Path(name).stem}",
                    "asset_id": asset_id,
                    "dataset": "assets",
                    "relative_path": f"vegetation/textures/{name}",
                    "expected_name": name,
                    "url": f"https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/{asset_id}/{name}",
                    "version": "1k glTF sidecar",
                    "license": "CC0 1.0",
                    "attribution": "Poly Haven",
                    "source": "https://polyhaven.com/license",
                    "download": {"state": "pending"},
                }
            )
    return entries


def _light_source_lock_from_full(
    *, full_lock: dict[str, Any], zone_id: str, rows: list[dict[str, str]]
) -> dict[str, Any]:
    """Reduce a resolved full tile catalogue to one traceable distant-terrain source.

    The output still has the source catalogue hash and the same EPSG:2154
    coverage, but it deliberately excludes high-resolution clouds, surface
    products and per-tile orthophotos.  The native USD builder splits this
    compact MNT back into its 400 payload boundaries.
    """

    entries = full_lock.get("entries")
    if not isinstance(entries, list):
        raise ValueError("full source lock entries are malformed")
    template = next(
        (
            item
            for item in entries
            if isinstance(item, dict)
            and item.get("dataset") == "mnt"
            and str(item.get("url", "")).startswith(("https://", "http://"))
        ),
        None,
    )
    if not isinstance(template, dict):
        raise RuntimeError("the resolved catalogue has no usable MNT WMS template")
    bbox = _zone_bbox(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE_ID,
        "zone_id": zone_id,
        "created_at": _utc_now(),
        "source_profile": LIGHT_PROFILE,
        "source_profile_version": LIGHT_PROFILE_VERSION,
        "catalog_inventory_sha256": full_lock.get("catalog_inventory_sha256"),
        "resolution_report": full_lock.get("resolution_report", {}),
        "light_profile": {
            "coverage_epsg2154": list(bbox),
            "terrain_resolution_metres": LIGHT_TERRAIN_RESOLUTION_METRES,
            "terrain_grid": [LIGHT_TERRAIN_WIDTH, LIGHT_TERRAIN_WIDTH],
            "excluded_products": ["lidar", "mns", "mnh", "ortho20", "ortho50"],
            "imagery": {
                "source": "IGN Géoplateforme WMS",
                "layer": LIGHT_ORTHOPHOTO_LAYER,
                "resolution_metres": LIGHT_ORTHOPHOTO_RESOLUTION_METRES,
                "grid": [2, 2],
                "role": "full-zone visual context",
            },
            "hero_imagery": {
                "source": "IGN BD ORTHO WMS",
                "resolution_metres": 0.2,
                "tile_count": LIGHT_HERO_TILE_COUNT,
                "role": "camera-local LOD0 terrain texture",
            },
            "external_assets": {
                "source": "Poly Haven",
                "license": "CC0 1.0",
                "role": "native vegetation prototypes and roof PBR",
            },
        },
        "entries": [
            {
                "id": f"{zone_id}:terrain_lod3_{LIGHT_TERRAIN_RESOLUTION_METRES}m",
                "tile_ref": "__zone__",
                "dataset": "terrain_lod3",
                "expected_name": f"{zone_id}_terrain_lod3_{LIGHT_TERRAIN_RESOLUTION_METRES}m.tif",
                "url": _light_terrain_url(
                    template_url=str(template["url"]), bbox=bbox, zone_id=zone_id
                ),
                "wfs_layer": str(template.get("wfs_layer", "")),
                "resolution_status": "derived_wms_lod3_8m",
                "license": "Licence Ouverte / Etalab 2.0",
                "content_length_bytes": None,
                "download": {"state": "pending"},
            },
            *_light_orthophoto_entries(zone_id=zone_id, bbox=bbox),
            *_light_hero_orthophoto_entries(zone_id=zone_id, rows=rows),
            *_light_external_asset_entries(),
        ],
    }


def _activate_light_source_lock(
    *, zone_root: Path, zone_id: str, rows: list[dict[str, str]]
) -> dict[str, Any]:
    """Preserve a full lock before replacing the active source plan with light."""

    current = _load_source_lock(zone_root)
    if (
        current.get("source_profile") == LIGHT_PROFILE
        and current.get("source_profile_version") == LIGHT_PROFILE_VERSION
    ):
        return current
    backup = zone_root / "source-lock-full-v1.json"
    if not backup.exists():
        write_json(backup, current)
    source_catalog = current if current.get("source_profile") != LIGHT_PROFILE else _read_json(
        backup, label="preserved full source lock"
    )
    light_lock = _light_source_lock_from_full(
        full_lock=source_catalog, zone_id=zone_id, rows=rows
    )
    # Preserve verified downloads while the source plan is enriched.  Every
    # retained entry still passes checksum verification in _download.
    previous = {
        str(item.get("id")): item
        for item in current.get("entries", [])
        if isinstance(item, dict)
    }
    for entry in light_lock["entries"]:
        prior = previous.get(str(entry["id"]))
        if isinstance(prior, dict) and prior.get("url") == entry.get("url"):
            download = prior.get("download")
            if isinstance(download, dict):
                entry["download"] = dict(download)
    if isinstance(current.get("vector_sources"), dict):
        light_lock["vector_sources"] = current["vector_sources"]
    write_json(_source_lock_path(zone_root), light_lock)
    return light_lock


def _zone_bbox(rows: Iterable[dict[str, str]]) -> tuple[int, int, int, int]:
    values = list(rows)
    if not values:
        raise ValueError("zone has no tile rows")
    return (
        min(int(row["xmin"]) for row in values),
        min(int(row["ymin"]) for row in values),
        max(int(row["xmax"]) for row in values),
        max(int(row["ymax"]) for row in values),
    )


def _vector_request_url(
    *, layer: str, bbox: tuple[int, int, int, int], start_index: int
) -> str:
    if start_index < 0:
        raise ValueError("WFS start index must not be negative")
    query = urlencode(
        {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "TYPENAMES": layer,
            "OUTPUTFORMAT": "application/json",
            "SRSNAME": "EPSG:2154",
            "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:2154",
            "COUNT": str(VECTOR_PAGE_SIZE),
            "STARTINDEX": str(start_index),
        }
    )
    return f"{WFS_BASE}?{query}"


def _read_vector_page(path: Path) -> dict[str, Any]:
    payload = _read_json(path, label="WFS vector page")
    if payload.get("type") != "FeatureCollection" or not isinstance(
        payload.get("features"), list
    ):
        raise ValueError(f"WFS vector page is malformed: {path}")
    return payload


def _download_vector_page(*, url: str, destination: Path, timeout: float) -> dict[str, Any]:
    """Get one bounded WFS page with an atomic, resumable raw artifact."""

    if destination.is_file():
        return _read_vector_page(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = Request(url, headers={"User-Agent": "FireViewer-Zone-Scenes/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
            raise RuntimeError("WFS vector page did not return a GeoJSON FeatureCollection")
        partial.write_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        partial.replace(destination)
        return payload
    except Exception:
        # Keep a partial response only when one was fully written as JSON.  It
        # remains inside raw/vectors and is never cleaned by this code path.
        if partial.is_file() and partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)
        raise


def _number_matched(payload: dict[str, Any]) -> int | None:
    """Return the WFS total when it is actually declared.

    GeoPlatform may omit ``numberMatched`` (or return the WFS sentinel
    ``"unknown"``).  Treating the current feature count as the total stops a
    full first page after 1,000 objects and silently truncates dense urban
    layers.
    """

    value = payload.get("numberMatched")
    if value is None or value == "unknown":
        return None
    if isinstance(value, bool):
        raise ValueError("WFS numberMatched must be a non-negative integer")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ValueError("WFS numberMatched must be a non-negative integer or unknown")


def _vector_page_fingerprint(features: list[object]) -> str:
    """Fingerprint a decoded page so an ignored STARTINDEX cannot loop forever."""

    canonical = json.dumps(
        features,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _vector_feature_identity(feature: object) -> str:
    if isinstance(feature, dict):
        feature_id = feature.get("id")
        if feature_id is not None and str(feature_id).strip():
            return f"id:{feature_id}"
    canonical = json.dumps(
        feature,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"content:{hashlib.sha256(canonical).hexdigest()}"


def _download_vector_layer(
    *,
    raw_root: Path,
    name: str,
    layer: str,
    bbox: tuple[int, int, int, int],
    timeout: float,
) -> dict[str, Any]:
    """Materialise all pages of one BDTOPO layer and lock its request URLs."""

    vector_root = raw_root / "vectors"
    if not _is_below(raw_root, vector_root):
        raise RuntimeError("vector source root escapes raw source root")
    pages_root = vector_root / f"{_safe_name(name, fallback='vector')}_{_safe_name(layer, fallback='layer')}.pages"
    pages_root.mkdir(parents=True, exist_ok=True)
    features: list[object] = []
    page_urls: list[str] = []
    start_index = 0
    number_matched: int | None = None
    seen_page_fingerprints: set[str] = set()
    seen_feature_identities: set[str] = set()
    while True:
        url = _vector_request_url(layer=layer, bbox=bbox, start_index=start_index)
        page_path = pages_root / f"{start_index:08d}.geojson"
        payload = _download_vector_page(url=url, destination=page_path, timeout=timeout)
        page_features = payload.get("features")
        if not isinstance(page_features, list):
            raise RuntimeError(f"WFS vector page has no features list: {url}")
        if not page_features:
            page_urls.append(url)
            break

        page_fingerprint = _vector_page_fingerprint(page_features)
        if page_fingerprint in seen_page_fingerprints:
            raise RuntimeError(
                f"WFS vector pagination made no progress; duplicate page for {layer}"
            )
        seen_page_fingerprints.add(page_fingerprint)

        page_feature_identities: set[str] = set()
        for feature in page_features:
            identity = _vector_feature_identity(feature)
            if (
                identity in page_feature_identities
                or identity in seen_feature_identities
            ):
                raise RuntimeError(
                    f"WFS vector pagination returned duplicate feature "
                    f"{identity!r} for {layer}"
                )
            page_feature_identities.add(identity)
        seen_feature_identities.update(page_feature_identities)

        features.extend(page_features)
        page_urls.append(url)
        reported_total = _number_matched(payload)
        if reported_total is not None:
            if number_matched is not None and reported_total != number_matched:
                raise RuntimeError(
                    f"WFS vector layer changed numberMatched during pagination: {layer}"
                )
            number_matched = reported_total
            if len(features) > number_matched:
                raise RuntimeError(
                    f"WFS vector layer returned more features than numberMatched: {layer}"
                )
        if number_matched is not None and len(features) >= number_matched:
            break
        if len(page_features) < VECTOR_PAGE_SIZE:
            break
        start_index += len(page_features)
        if start_index > 1_000_000:
            raise RuntimeError(f"WFS vector layer is unreasonably large: {layer}")
    final_path = vector_root / f"{_safe_name(name, fallback='vector')}_{_safe_name(layer, fallback='layer')}.geojson"
    final_payload = {
        "type": "FeatureCollection",
        "name": layer,
        "crs": {"type": "name", "properties": {"name": "EPSG:2154"}},
        "features": features,
    }
    temporary = final_path.with_suffix(final_path.suffix + ".partial")
    temporary.write_text(
        json.dumps(final_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(final_path)
    return {
        "name": name,
        "layer": layer,
        "license": "Licence Ouverte / Etalab 2.0",
        "attribution": "IGN BD TOPO V3 via Géoplateforme WFS",
        "queried_at": _utc_now(),
        "bbox_epsg2154": list(bbox),
        "urls": page_urls,
        "feature_count": len(features),
        "download": {
            "state": "downloaded",
            "relpath": final_path.relative_to(raw_root).as_posix(),
            "bytes": final_path.stat().st_size,
            "sha256": sha256(final_path),
        },
    }


def _acquire_vector_sources(
    *, raw_root: Path, rows: list[dict[str, str]], timeout: float
) -> dict[str, list[dict[str, Any]]]:
    """Acquire all material scene-vector sources in a traceable source lock."""

    bbox = _zone_bbox(rows)
    result: dict[str, list[dict[str, Any]]] = {}
    for name, layers in VECTOR_SOURCE_LAYERS.items():
        result[name] = [
            _download_vector_layer(
                raw_root=raw_root,
                name=name,
                layer=layer,
                bbox=bbox,
                timeout=timeout,
            )
            for layer in layers
        ]
    return result


def _locked_vector_sources_valid(*, raw_root: Path, lock: dict[str, Any]) -> bool:
    """Accept a vector cache only when its receipt still matches every file."""

    sources = lock.get("vector_sources")
    if not isinstance(sources, dict):
        return False
    for name, layers in VECTOR_SOURCE_LAYERS.items():
        records = sources.get(name)
        if not isinstance(records, list) or len(records) != len(layers):
            return False
        for record in records:
            if not isinstance(record, dict):
                return False
            download = record.get("download")
            if not isinstance(download, dict) or download.get("state") != "downloaded":
                return False
            path = (raw_root / str(download.get("relpath", ""))).resolve()
            if not _is_below(raw_root, path) or not path.is_file():
                return False
            if sha256(path) != str(download.get("sha256", "")):
                return False
    return True


def _safe_name(value: str, *, fallback: str) -> str:
    raw = SAFE_FILENAME.sub("_", value.strip())
    return raw.strip("._") or fallback


def _destination(raw_root: Path, entry: dict[str, Any]) -> Path:
    expected = _safe_name(str(entry.get("expected_name", "")), fallback=str(entry["id"]).replace(":", "_"))
    suffix = Path(urlparse(str(entry.get("url", ""))).path).suffix
    if not suffix:
        suffix = {
            "lidar": ".laz",
            "mnt": ".tif",
            "mns": ".tif",
            "mnh": ".tif",
            "ortho20": ".jpg",
            "ortho50": ".jpg",
        }.get(str(entry.get("dataset", "")), "")
    relative = str(entry.get("relative_path", expected)).replace("\\", "/")
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError("download relative path is unsafe")
    if suffix and not candidate.suffix:
        candidate = candidate.with_suffix(suffix)
    if suffix and not Path(expected).suffix:
        expected += suffix
    destination = raw_root / str(entry["dataset"]) / Path(*candidate.parts)
    if not _is_below(raw_root, destination):
        raise ValueError("download destination escapes raw source root")
    return destination


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        # Windows returns ERROR_INVALID_PARAMETER (WinError 87) for some
        # syntactically valid but impossible PIDs. Treat that stale receipt
        # marker exactly like a missing process.
        return False
    return True


def _review_process_matches(pid: int, editor: Path) -> bool:
    """Refuse PID reuse while allowing the real Kit launcher or child binary."""

    if not _process_is_running(pid):
        return False
    if sys.platform.startswith("linux"):
        command_line = Path(f"/proc/{pid}/cmdline")
        try:
            values = [
                item.decode("utf-8", errors="replace")
                for item in command_line.read_bytes().split(b"\0")
                if item
            ]
        except OSError:
            return False
        normalized = " ".join(values).casefold()
        return (
            str(editor.resolve()).casefold() in normalized
            or editor.name.casefold() in normalized
            or "fireviewer_usd_composer" in normalized
        )
    # Windows launches the Hub batch through cmd.exe, whose child command line
    # is not safely introspectable without another dependency.  A live PID
    # already bound to the exact root receipt is still safer than spawning a
    # second GPU-heavy Editor.
    return True


@contextmanager
def _acquisition_lock(zone_root: Path) -> Iterable[None]:
    """Serialise resumable raw downloads for one zone workspace."""

    lock_path = zone_root / ".acquisition.lock"
    payload = {"pid": os.getpid(), "created_at": _utc_now()}
    try:
        with lock_path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
    except FileExistsError:
        try:
            existing = _read_json(lock_path, label="zone acquisition lock")
        except (OSError, ValueError):
            existing = {}
        owner_pid = int(existing.get("pid", 0)) if isinstance(existing, dict) else 0
        if _process_is_running(owner_pid):
            raise RuntimeError(
                f"zone acquisition is already active (pid={owner_pid}); refusing concurrent resume"
            )
        # A dead process may leave a malformed marker. Only this marker is
        # removed; downloaded data and .partial files remain resumable.
        lock_path.unlink(missing_ok=True)
        with lock_path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
    try:
        yield
    finally:
        try:
            current = _read_json(lock_path, label="zone acquisition lock")
        except (OSError, ValueError):
            current = {}
        if isinstance(current, dict) and int(current.get("pid", 0)) == os.getpid():
            lock_path.unlink(missing_ok=True)


def _remote_version_validator(
    entry: dict[str, Any],
) -> dict[str, str] | None:
    value = entry.get("remote_version_validator")
    if not isinstance(value, dict):
        return None
    kind = str(value.get("kind", ""))
    validator = str(value.get("value", "")).strip()
    if kind == "strong_etag" and validator and not validator.startswith("W/"):
        return {"kind": kind, "value": validator}
    if kind == "last_modified" and validator:
        return {"kind": kind, "value": validator}
    return None


def _capture_remote_version_validator(
    entry: dict[str, Any], headers: Any
) -> dict[str, str] | None:
    etag = str(headers.get("ETag", "")).strip()
    last_modified = str(headers.get("Last-Modified", "")).strip()
    validator: dict[str, str] | None = None
    if etag and not etag.startswith("W/"):
        validator = {"kind": "strong_etag", "value": etag}
    elif last_modified:
        validator = {"kind": "last_modified", "value": last_modified}
    if validator is not None:
        entry["remote_version_validator"] = validator
    return validator


def _measurement_is_reusable(
    entry: dict[str, Any], *, require_remote_validator: bool = False
) -> bool:
    return (
        isinstance(entry.get("content_length_bytes"), int)
        and int(entry["content_length_bytes"]) > 0
        and entry.get("size_measurement")
        in {"head_content_length", "range_content_range_0_0"}
        and not entry.get("probe_error")
        and (
            not require_remote_validator
            or _remote_version_validator(entry) is not None
        )
    )


def _measure(
    entry: dict[str, Any],
    *,
    timeout: float,
    range_first: bool = False,
    before_request: Callable[[], None] | None = None,
    require_remote_validator: bool = False,
) -> None:
    """Measure one source without downloading its full payload.

    The IGN LiDAR endpoint answers HEAD without a size but exposes the total
    byte count through Content-Range for a one-byte request.  That is the
    authoritative capacity signal for a resumable direct download; treating it
    as unknown incorrectly blocks every LOD0 LiDAR acquisition.
    """

    if _measurement_is_reusable(
        entry, require_remote_validator=require_remote_validator
    ):
        return
    url = str(entry.get("url", ""))

    def start_request() -> None:
        if before_request is not None:
            before_request()

    def probe_head() -> bool:
        request = Request(
            url,
            method="HEAD",
            headers={"User-Agent": "FireViewer-Zone-Scenes/1.0"},
        )
        try:
            start_request()
            with urlopen(request, timeout=timeout) as response:
                length = response.headers.get("Content-Length")
                entry["content_type"] = response.headers.get("Content-Type", "")
                if length and length.isdigit():
                    validator = _capture_remote_version_validator(
                        entry, response.headers
                    )
                    if require_remote_validator and validator is None:
                        raise ValueError(
                            "HEAD response has no strong ETag or Last-Modified"
                        )
                    entry["content_length_bytes"] = int(length)
                    entry["size_measurement"] = "head_content_length"
                    entry.pop("probe_error", None)
                    return True
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            prior = str(entry.get("probe_error", "")).strip()
            detail = f"HEAD {type(exc).__name__}: {exc}"
            entry["probe_error"] = f"{prior}; {detail}" if prior else detail
        return False

    def probe_range() -> bool:
        range_request = Request(
            url,
            headers={
                "User-Agent": "FireViewer-Zone-Scenes/1.0",
                "Range": "bytes=0-0",
            },
        )
        try:
            start_request()
            with urlopen(range_request, timeout=timeout) as response:
                # Consume one byte so the range request is genuinely exercised;
                # Content-Length would describe that byte, not the source.
                response.read(1)
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(
                    r"bytes\s+0-0/(\d+)", content_range.strip()
                )
                if not match:
                    raise ValueError(
                        "range response did not provide a total Content-Range"
                    )
                validator = _capture_remote_version_validator(
                    entry, response.headers
                )
                if require_remote_validator and validator is None:
                    raise ValueError(
                        "range response has no strong ETag or Last-Modified"
                    )
                entry["content_length_bytes"] = int(match.group(1))
                entry["content_type"] = response.headers.get(
                    "Content-Type", entry.get("content_type", "")
                )
                entry["size_measurement"] = "range_content_range_0_0"
                entry.pop("probe_error", None)
                return True
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            prior = str(entry.get("probe_error", "")).strip()
            detail = f"range {type(exc).__name__}: {exc}"
            entry["probe_error"] = f"{prior}; {detail}" if prior else detail
        return False

    probes = (probe_range, probe_head) if range_first else (probe_head, probe_range)
    for probe in probes:
        if probe():
            return


def _measurement_samples(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Probe WMS by product and each selected direct-download source.

    WMS requests carry their decoded dimensions, so one product sample is
    enough.  Direct LiDAR assets have independent byte totals and must each be
    bounded before we reserve space for the small camera LOD0 set.
    """

    samples: dict[str, dict[str, Any]] = {}
    direct_sources: list[dict[str, Any]] = []
    for entry in entries:
        url = str(entry.get("url", ""))
        if url.startswith(("https://", "http://")) and _wms_uncompressed_bytes(entry) is None:
            direct_sources.append(entry)
        else:
            samples.setdefault(str(entry.get("dataset", "")), entry)
    return [*direct_sources, *[samples[key] for key in sorted(samples)]]


def _wms_uncompressed_bytes(entry: dict[str, Any]) -> int | None:
    """Conservative decoded size when an IGN WMS omits Content-Length.

    The estimate derives from the actual WIDTH/HEIGHT encoded in the locked
    request rather than an arbitrary per-zone quota.  Four bytes per pixel is
    safe for one Float32 elevation band and for an RGBA orthophoto response.
    """

    url = str(entry.get("url", ""))
    parsed = urlparse(url)
    query = {
        str(key).upper(): values[-1]
        for key, values in parse_qs(parsed.query).items()
        if values
    }
    if query.get("SERVICE", "").upper() != "WMS":
        return None
    try:
        width = int(query["WIDTH"])
        height = int(query["HEIGHT"])
    except (KeyError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width * height * 4


class _RequestStartPacer:
    """Serialize request starts while allowing transfers to overlap."""

    def __init__(self, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            raise ValueError("request start interval must be positive")
        self._interval_seconds = interval_seconds
        self._next_start = 0.0
        self._lock = Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next_start - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_start = max(self._next_start, now) + self._interval_seconds


def _measure_direct_entries(
    entries: list[dict[str, Any]],
    *,
    timeout: float,
    max_workers: int,
    request_pacer: _RequestStartPacer | None = None,
    require_remote_validator: bool = False,
) -> None:
    if not entries:
        return
    shared_pacer = request_pacer or _RequestStartPacer(
        DIRECT_REQUEST_START_INTERVAL_SECONDS
    )

    def measure_direct(entry: dict[str, Any]) -> None:
        options: dict[str, Any] = {
            "timeout": timeout,
            "range_first": True,
            "before_request": shared_pacer.wait,
        }
        if require_remote_validator:
            options["require_remote_validator"] = True
        _measure(entry, **options)

    with ThreadPoolExecutor(
        max_workers=min(max_workers, len(entries)),
        thread_name_prefix="fireviewer-zone-lidar-measure",
    ) as executor:
        futures = [executor.submit(measure_direct, entry) for entry in entries]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def _measure_direct_entries_with_retries(
    entries: list[dict[str, Any]],
    *,
    timeout: float,
    max_workers: int,
    attempts: int,
    request_pacer: _RequestStartPacer,
    checkpoint: Callable[[], None],
) -> list[dict[str, Any]]:
    """Persist successful size probes and retry only unresolved direct sources."""

    if attempts < 1:
        raise ValueError("direct source measurement attempts must be positive")
    unresolved = [
        entry
        for entry in entries
        if not (
            isinstance(entry.get("content_length_bytes"), int)
            and int(entry["content_length_bytes"]) > 0
        )
    ]
    for _attempt in range(attempts):
        if not unresolved:
            break
        _measure_direct_entries(
            unresolved,
            timeout=timeout,
            max_workers=max_workers,
            request_pacer=request_pacer,
        )
        checkpoint()
        unresolved = [
            entry
            for entry in unresolved
            if not (
                isinstance(entry.get("content_length_bytes"), int)
                and int(entry["content_length_bytes"]) > 0
            )
        ]
    return unresolved


def _expected_wms_format(entry: dict[str, Any]) -> str | None:
    parsed = urlparse(str(entry.get("url", "")))
    query = {
        str(key).upper(): values[-1].lower()
        for key, values in parse_qs(parsed.query).items()
        if values
    }
    if query.get("SERVICE", "").upper() != "WMS":
        return None
    requested = query.get("FORMAT", "")
    if "tif" in requested:
        return "tiff"
    if "jpeg" in requested or "jpg" in requested:
        return "jpeg"
    if "png" in requested:
        return "png"
    dataset = str(entry.get("dataset", ""))
    if dataset in {"mnt", "mns", "mnh", "terrain_lod3"}:
        return "tiff"
    if dataset in {"ortho20", "ortho50", "ortho_lod0", "ortho_lod2"}:
        return "jpeg"
    raise RuntimeError("WMS source has no verifiable output format")


def _wms_request_grid(
    entry: dict[str, Any],
) -> tuple[int, int, tuple[float, float, float, float]]:
    parsed = urlparse(str(entry.get("url", "")))
    query = {
        str(key).upper(): values[-1]
        for key, values in parse_qs(parsed.query).items()
        if values
    }
    try:
        width = int(query["WIDTH"])
        height = int(query["HEIGHT"])
        bbox_values = tuple(
            float(value) for value in query["BBOX"].split(",")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "WMS request lacks locked WIDTH/HEIGHT/BBOX"
        ) from exc
    if (
        width <= 0
        or height <= 0
        or len(bbox_values) != 4
        or bbox_values[2] <= bbox_values[0]
        or bbox_values[3] <= bbox_values[1]
    ):
        raise RuntimeError("WMS request grid is invalid")
    return width, height, bbox_values


def _close_geospatial_value(observed: float, expected: float) -> bool:
    return abs(observed - expected) <= max(1e-6, abs(expected) * 1e-9)


def _validate_wms_format(
    path: Path, *, entry: dict[str, Any], expected_format: str
) -> None:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for native WMS validation"
        ) from exc
    width, height, bbox = _wms_request_grid(entry)
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != expected_format.upper():
                raise RuntimeError(
                    "WMS response does not match requested "
                    f"{expected_format} format"
                )
            if image.size != (width, height):
                raise RuntimeError(
                    "WMS response dimensions differ from WIDTH/HEIGHT"
                )
            if expected_format != "tiff":
                return
            if image.mode != "F":
                raise RuntimeError(
                    "terrain WMS TIFF is not a Float32 elevation raster"
                )
            tags = image.tag_v2
            try:
                pixel_scale = tuple(float(value) for value in tags[33550])
                tiepoint = tuple(float(value) for value in tags[33922])
                geo_ascii = str(tags[34737])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "terrain WMS TIFF lacks required GeoTIFF tags"
                ) from exc
            if len(pixel_scale) < 2 or len(tiepoint) < 6:
                raise RuntimeError(
                    "terrain WMS TIFF has malformed GeoTIFF tags"
                )
            xmin, ymin, xmax, ymax = bbox
            pixel_x = (xmax - xmin) / width
            pixel_y = (ymax - ymin) / height
            # A WMS BBOX describes the raster area.  IGN GeoTIFF responses use
            # RasterPixelIsArea, so ModelTiepoint is the upper-left area corner
            # and must match the BBOX directly.  The half-pixel expansion is
            # already present in the locked LiDAR HD request when required.
            if not (
                _close_geospatial_value(pixel_scale[0], pixel_x)
                and _close_geospatial_value(pixel_scale[1], pixel_y)
                and _close_geospatial_value(tiepoint[0], 0.0)
                and _close_geospatial_value(tiepoint[1], 0.0)
                and _close_geospatial_value(tiepoint[3], xmin)
                and _close_geospatial_value(tiepoint[4], ymax)
            ):
                raise RuntimeError(
                    "terrain WMS TIFF georeferencing disagrees with BBOX"
                )
            if "EPSG:2154" not in geo_ascii:
                raise RuntimeError(
                    "terrain WMS TIFF is not tagged as EPSG:2154"
                )
    except UnidentifiedImageError as exc:
        raise RuntimeError(
            f"WMS response is not a native {expected_format} image"
        ) from exc


def _source_measurement_fingerprint(entry: dict[str, Any]) -> str:
    payload = {
        "url": str(entry.get("url", "")),
        "content_length_bytes": (
            int(entry["content_length_bytes"])
            if isinstance(entry.get("content_length_bytes"), int)
            else None
        ),
        "size_measurement": str(entry.get("size_measurement", "")),
        "remote_version_validator": _remote_version_validator(entry),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def _stable_file_sha256(path: Path) -> tuple[str, dict[str, int]]:
    if path.is_symlink():
        raise RuntimeError("download destination may not be a symlink")
    before = _file_identity(path)
    digest = sha256(path)
    after = _file_identity(path)
    if before != after:
        raise RuntimeError("download destination changed while hashing")
    return digest, after


def _final_download_receipt(
    entry: dict[str, Any],
    destination: Path,
    *,
    state: str,
    digest: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = _file_identity(destination)
    return {
        "state": state,
        "relpath": str(
            entry.get("relative_path", destination.name)
        ).replace("\\", "/"),
        "bytes": identity["bytes"],
        "sha256": digest,
        "file_identity": identity,
        "source_fingerprint_sha256": _source_measurement_fingerprint(entry),
        **(extra or {}),
    }


def _fast_download_receipt_matches(
    entry: dict[str, Any], destination: Path
) -> bool:
    download = entry.get("download")
    final_states = {
        "downloaded",
        "downloaded_segmented",
        "verified_existing",
        "recovered_complete_partial",
    }
    if (
        not isinstance(download, dict)
        or download.get("state") not in final_states
        or not destination.is_file()
        or destination.is_symlink()
        or not re.fullmatch(r"[0-9a-f]{64}", str(download.get("sha256", "")))
        or download.get("source_fingerprint_sha256")
        != _source_measurement_fingerprint(entry)
    ):
        return False
    identity = download.get("file_identity")
    if not isinstance(identity, dict):
        return False
    try:
        recorded_identity = {
            key: int(identity[key])
            for key in ("bytes", "mtime_ns", "device", "inode")
        }
    except (KeyError, TypeError, ValueError):
        return False
    expected_bytes = entry.get("content_length_bytes")
    if (
        isinstance(expected_bytes, int)
        and recorded_identity["bytes"] != expected_bytes
    ):
        return False
    return recorded_identity == _file_identity(destination)


def _download(
    entry: dict[str, Any],
    destination: Path,
    *,
    timeout: float,
    before_request: Callable[[], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.is_file() and destination.stat().st_size > 0:
        actual_bytes = destination.stat().st_size
        expected_bytes = entry.get("content_length_bytes")
        if isinstance(expected_bytes, int) and actual_bytes != expected_bytes:
            raise RuntimeError(
                f"existing source size differs from its locked measurement: {destination}"
            )
        prior_download = entry.get("download")
        if (
            isinstance(prior_download, dict)
            and prior_download.get("source_fingerprint_sha256")
            and prior_download.get("source_fingerprint_sha256")
            != _source_measurement_fingerprint(entry)
        ):
            raise RuntimeError(
                "existing source receipt belongs to another URL/measurement"
            )
        if _fast_download_receipt_matches(entry, destination):
            return
        actual_sha256, _identity = _stable_file_sha256(destination)
        if isinstance(prior_download, dict):
            expected_sha256 = str(prior_download.get("sha256", "")).strip().lower()
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"existing source SHA-256 differs from its prior receipt: {destination}"
                )
        entry["download"] = _final_download_receipt(
            entry,
            destination,
            state="verified_existing",
            digest=actual_sha256,
        )
        return
    start = partial.stat().st_size if partial.is_file() else 0
    expected_bytes = entry.get("content_length_bytes")
    prior_download = entry.get("download")
    expected_sha256 = ""
    if isinstance(prior_download, dict):
        expected_sha256 = str(
            prior_download.get("expected_sha256")
            or prior_download.get("sha256")
            or ""
        ).strip().lower()
    if start and isinstance(expected_bytes, int) and start > expected_bytes:
        partial.unlink()
        start = 0
    if start and isinstance(expected_bytes, int) and start == expected_bytes:
        actual_sha256 = sha256(partial)
        if not expected_sha256 or actual_sha256 == expected_sha256:
            partial.replace(destination)
            entry["download"] = _final_download_receipt(
                entry,
                destination,
                state="recovered_complete_partial",
                digest=actual_sha256,
            )
            return
        # A byte-complete partial is not complete when it disagrees with the
        # previously locked digest. Re-fetch it from byte zero rather than
        # silently promoting corrupt content.
        start = 0
    headers = {"User-Agent": "FireViewer-Zone-Scenes/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = Request(str(entry["url"]), headers=headers)
    expected_wms_format = _expected_wms_format(entry)
    restart_partial = destination.with_suffix(
        destination.suffix + ".restart.partial"
    )
    try:
        if should_cancel is not None and should_cancel():
            raise RuntimeError("direct source download cancelled after peer failure")
        if before_request is not None:
            before_request()
        if should_cancel is not None and should_cancel():
            raise RuntimeError("direct source download cancelled after peer failure")
        with urlopen(request, timeout=timeout) as response:
            final_url = str(getattr(response, "geturl", lambda: entry["url"])())
            if not final_url.startswith("https://"):
                raise RuntimeError("source download redirected outside HTTPS")
            status = int(getattr(response, "status", 200))
            declared_length = str(
                response.headers.get("Content-Length", "")
            ).strip()
            if (
                expected_wms_format is not None
                and declared_length
                and not declared_length.isdigit()
            ):
                raise RuntimeError(
                    "WMS response has no verifiable Content-Length"
                )
            declared_response_bytes = (
                int(declared_length) if declared_length.isdigit() else None
            )
            entry["content_type"] = response.headers.get(
                "Content-Type", entry.get("content_type", "")
            )
            append = start > 0 and status == 206
            target = partial
            mode = "wb"
            if append:
                if not isinstance(expected_bytes, int) or expected_bytes <= 0:
                    raise RuntimeError(
                        "resumed source has no locked byte count"
                    )
                content_range = str(
                    response.headers.get("Content-Range", "")
                ).strip()
                match = re.fullmatch(
                    r"bytes\s+(\d+)-(\d+)/(\d+)", content_range
                )
                if (
                    not match
                    or int(match.group(1)) != start
                    or int(match.group(2)) != expected_bytes - 1
                    or int(match.group(3)) != expected_bytes
                ):
                    raise RuntimeError(
                        "resumed source returned an invalid Content-Range: "
                        f"{content_range or '<missing>'}"
                    )
                mode = "ab"
            elif start:
                if status != 200:
                    raise RuntimeError(
                        f"resumed source returned unexpected HTTP status {status}"
                    )
                # Some providers ignore Range. Keep the known-good partial
                # untouched until the replacement full response is complete
                # and verified.
                target = restart_partial
            elif status not in {200, 206}:
                raise RuntimeError(
                    f"source returned unexpected HTTP status {status}"
                )
            response_written = 0
            with target.open(mode) as output:
                while chunk := response.read(1024 * 1024):
                    if should_cancel is not None and should_cancel():
                        raise RuntimeError(
                            "direct source download cancelled after peer failure"
                        )
                    response_written += len(chunk)
                    if (
                        declared_response_bytes is not None
                        and response_written > declared_response_bytes
                    ):
                        raise RuntimeError(
                            "source response exceeds its Content-Length"
                        )
                    output.write(chunk)
            if (
                declared_response_bytes is not None
                and response_written != declared_response_bytes
            ):
                raise RuntimeError(
                    "source response differs from its Content-Length"
                )
    except Exception:
        entry["download"] = {
            "state": "partial",
            "relpath": str(
                entry.get("relative_path", destination.name)
            ).replace("\\", "/"),
            **(
                {"expected_sha256": expected_sha256}
                if expected_sha256
                else {}
            ),
        }
        raise
    completed_partial = restart_partial if start and status == 200 else partial
    if (
        not isinstance(expected_bytes, int)
        and status == 200
        and declared_response_bytes is not None
    ):
        expected_bytes = declared_response_bytes
        entry["content_length_bytes"] = declared_response_bytes
        entry["size_measurement"] = "download_content_length"
    if (
        isinstance(expected_bytes, int)
        and completed_partial.stat().st_size != expected_bytes
    ):
        entry["download"] = {
            "state": "partial_size_mismatch",
            "relpath": str(entry.get("relative_path", destination.name)).replace("\\", "/"),
            "bytes": completed_partial.stat().st_size,
            "expected_bytes": expected_bytes,
            **(
                {"expected_sha256": expected_sha256}
                if expected_sha256
                else {}
            ),
        }
        raise RuntimeError(
            f"downloaded source size differs from its locked measurement: {destination}"
        )
    actual_sha256 = sha256(completed_partial)
    if expected_sha256 and actual_sha256 != expected_sha256:
        entry["download"] = {
            "state": "partial_sha256_mismatch",
            "relpath": str(
                entry.get("relative_path", destination.name)
            ).replace("\\", "/"),
            "bytes": completed_partial.stat().st_size,
            "sha256": actual_sha256,
            "expected_sha256": expected_sha256,
        }
        raise RuntimeError(
            f"downloaded source SHA-256 differs from its locked receipt: {destination}"
        )
    if expected_wms_format is not None:
        _validate_wms_format(
            completed_partial,
            entry=entry,
            expected_format=expected_wms_format,
        )
        if not isinstance(entry.get("content_length_bytes"), int):
            entry["content_length_bytes"] = completed_partial.stat().st_size
            entry["size_measurement"] = "download_native_validated_size"
    completed_partial.replace(destination)
    if completed_partial == restart_partial:
        # The old resumable prefix is retained throughout the ignored-Range
        # replacement and becomes redundant only after the full replacement
        # has passed size/hash validation and reached its final destination.
        partial.unlink(missing_ok=True)
    entry["download"] = _final_download_receipt(
        entry,
        destination,
        state="downloaded",
        digest=actual_sha256,
    )


def _download_with_retries(
    entry: dict[str, Any],
    destination: Path,
    *,
    timeout: float,
    retries: int,
    before_request: Callable[[], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    for attempt in range(1, retries + 1):
        if should_cancel is not None and should_cancel():
            raise RuntimeError("direct source download cancelled after peer failure")
        try:
            download_options: dict[str, Any] = {
                "timeout": timeout,
                "before_request": before_request,
            }
            if should_cancel is not None:
                download_options["should_cancel"] = should_cancel
            _download(entry, destination, **download_options)
            entry["download_attempts"] = attempt
            return
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            RuntimeError,
        ):
            if should_cancel is not None and should_cancel():
                raise
            if attempt >= retries:
                raise
            time.sleep(min(2 ** (attempt - 1), 5))


def _partition_byte_ranges(
    total_bytes: int, segment_count: int
) -> list[dict[str, int]]:
    if total_bytes <= 0:
        raise ValueError("segmented source byte count must be positive")
    if not 1 <= segment_count <= total_bytes:
        raise ValueError("segmented source range count is invalid")
    base, remainder = divmod(total_bytes, segment_count)
    ranges: list[dict[str, int]] = []
    start = 0
    for index in range(segment_count):
        length = base + (1 if index < remainder else 0)
        end = start + length - 1
        ranges.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "bytes": length,
                "total_bytes": total_bytes,
            }
        )
        start = end + 1
    _assert_exact_range_coverage(ranges, total_bytes=total_bytes)
    return ranges


def _assert_exact_range_coverage(
    ranges: Iterable[dict[str, int]], *, total_bytes: int
) -> None:
    ordered = sorted(ranges, key=lambda item: int(item["start"]))
    expected_start = 0
    for item in ordered:
        start = int(item["start"])
        end = int(item["end"])
        length = int(item["bytes"])
        if (
            start != expected_start
            or end < start
            or length != end - start + 1
            or int(item["total_bytes"]) != total_bytes
        ):
            raise RuntimeError(
                "segmented source ranges do not provide exact, disjoint coverage"
            )
        expected_start = end + 1
    if expected_start != total_bytes:
        raise RuntimeError(
            "segmented source ranges do not cover the complete source"
        )


def _segment_stage_root(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.segments"


def _segment_paths(
    destination: Path, byte_range: dict[str, int]
) -> tuple[Path, Path, Path]:
    stem = (
        f"{int(byte_range['index']):04d}-"
        f"{int(byte_range['start'])}-{int(byte_range['end'])}"
    )
    root = _segment_stage_root(destination)
    segment = root / f"{stem}.bin"
    return segment, root / f"{stem}.json", root / f"{stem}.partial"


def _range_receipt_is_valid(
    *,
    entry: dict[str, Any],
    destination: Path,
    byte_range: dict[str, int],
) -> dict[str, Any] | None:
    segment, receipt_path, _temporary = _segment_paths(
        destination, byte_range
    )
    if not segment.is_file() or not receipt_path.is_file():
        return None
    try:
        receipt = _read_json(receipt_path, label="LiDAR range receipt")
    except (OSError, ValueError):
        return None
    validator = _remote_version_validator(entry)
    try:
        valid = (
            isinstance(receipt, dict)
            and validator is not None
            and receipt.get("source_url") == str(entry.get("url", ""))
            and receipt.get("remote_version_validator") == validator
            and int(receipt.get("start", -1)) == int(byte_range["start"])
            and int(receipt.get("end", -1)) == int(byte_range["end"])
            and int(receipt.get("total_bytes", -1))
            == int(byte_range["total_bytes"])
            and int(receipt.get("bytes", -1)) == int(byte_range["bytes"])
            and segment.stat().st_size == int(byte_range["bytes"])
            and str(receipt.get("sha256", "")) == sha256(segment)
        )
    except (OSError, TypeError, ValueError):
        valid = False
    if not valid:
        return None
    return receipt


def _response_matches_remote_validator(
    headers: Any, validator: dict[str, str]
) -> bool:
    if validator["kind"] == "strong_etag":
        observed = str(headers.get("ETag", "")).strip()
        return observed == validator["value"] and not observed.startswith("W/")
    if validator["kind"] == "last_modified":
        return (
            str(headers.get("Last-Modified", "")).strip()
            == validator["value"]
        )
    return False


def _download_range_segment(
    *,
    entry: dict[str, Any],
    destination: Path,
    byte_range: dict[str, int],
    timeout: float,
    before_request: Callable[[], None],
    should_cancel: Callable[[], bool],
) -> dict[str, Any]:
    existing = _range_receipt_is_valid(
        entry=entry, destination=destination, byte_range=byte_range
    )
    if existing is not None:
        return existing
    validator = _remote_version_validator(entry)
    if validator is None:
        # Content-Range plus a final SHA detects truncation/corruption but
        # cannot prove concurrently fetched ranges came from one remote
        # version. Without a strong ETag or Last-Modified usable by If-Range,
        # segmented transfer is therefore deliberately unavailable.
        raise RuntimeError(
            "segmented source has no remote version validator"
        )
    segment, receipt_path, temporary = _segment_paths(
        destination, byte_range
    )
    root = segment.parent
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise RuntimeError("segmented source staging root may not be a symlink")
    start = int(byte_range["start"])
    end = int(byte_range["end"])
    total = int(byte_range["total_bytes"])
    request = Request(
        str(entry["url"]),
        headers={
            "User-Agent": "FireViewer-Zone-Scenes/1.0",
            "Range": f"bytes={start}-{end}",
            "If-Range": validator["value"],
        },
    )
    if should_cancel():
        raise RuntimeError("segmented source cancelled after peer failure")
    before_request()
    if should_cancel():
        raise RuntimeError("segmented source cancelled after peer failure")
    with urlopen(request, timeout=timeout) as response:
        final_url = str(getattr(response, "geturl", lambda: entry["url"])())
        if not final_url.startswith("https://"):
            raise RuntimeError("segmented source redirected outside HTTPS")
        if int(getattr(response, "status", 200)) != 206:
            raise RuntimeError(
                "segmented source did not honor the conditional byte range"
            )
        content_range = str(
            response.headers.get("Content-Range", "")
        ).strip()
        if content_range != f"bytes {start}-{end}/{total}":
            raise RuntimeError(
                "segmented source returned an invalid Content-Range: "
                f"{content_range or '<missing>'}"
            )
        if not _response_matches_remote_validator(
            response.headers, validator
        ):
            raise RuntimeError(
                "segmented source remote version validator changed or is absent"
            )
        written = 0
        with temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                if should_cancel():
                    raise RuntimeError(
                        "segmented source cancelled after peer failure"
                    )
                written += len(chunk)
                if written > int(byte_range["bytes"]):
                    raise RuntimeError(
                        "segmented source response exceeds its locked range"
                    )
                output.write(chunk)
    if written != int(byte_range["bytes"]):
        raise RuntimeError(
            "segmented source response is shorter than its locked range"
        )
    segment_sha256 = sha256(temporary)
    temporary.replace(segment)
    receipt = {
        "schema_version": 1,
        "source_url": str(entry["url"]),
        "remote_version_validator": validator,
        "index": int(byte_range["index"]),
        "start": start,
        "end": end,
        "total_bytes": total,
        "bytes": written,
        "sha256": segment_sha256,
    }
    write_json(receipt_path, receipt)
    return receipt


def _download_range_segment_with_retries(
    *,
    entry: dict[str, Any],
    destination: Path,
    byte_range: dict[str, int],
    timeout: float,
    retries: int,
    before_request: Callable[[], None],
    should_cancel: Callable[[], bool],
) -> dict[str, Any]:
    for attempt in range(1, retries + 1):
        try:
            return _download_range_segment(
                entry=entry,
                destination=destination,
                byte_range=byte_range,
                timeout=timeout,
                before_request=before_request,
                should_cancel=should_cancel,
            )
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError):
            if should_cancel() or attempt >= retries:
                raise
            time.sleep(min(2 ** (attempt - 1), 5))
    raise AssertionError("segmented source retry loop did not terminate")


def _destination_is_complete(
    entry: dict[str, Any], destination: Path
) -> bool:
    expected_bytes = entry.get("content_length_bytes")
    if (
        not destination.is_file()
        or not isinstance(expected_bytes, int)
        or destination.stat().st_size != expected_bytes
    ):
        return False
    return _fast_download_receipt_matches(entry, destination)


def _tail_segmentation_plan(
    entries: list[dict[str, Any]],
    *,
    raw_root: Path,
    max_workers: int,
) -> list[dict[str, Any]] | None:
    effective_workers = min(max_workers, MAX_DIRECT_DOWNLOAD_WORKERS)
    pending = [
        entry
        for entry in entries
        if not _destination_is_complete(
            entry, _destination(raw_root, entry)
        )
    ]
    if (
        effective_workers < 2
        or not pending
        or len(pending)
        > max(1, effective_workers // TAIL_SEGMENTATION_POOL_DIVISOR)
    ):
        return None
    per_file_budget = effective_workers // len(pending)
    plan: list[dict[str, Any]] = []
    for entry in pending:
        expected_bytes = entry.get("content_length_bytes")
        if (
            not isinstance(expected_bytes, int)
            or expected_bytes <= 0
            or _remote_version_validator(entry) is None
        ):
            return None
        size_limited_count = max(
            1, expected_bytes // TAIL_SEGMENT_MIN_BYTES
        )
        segment_count = min(per_file_budget, size_limited_count)
        if segment_count < 2:
            return None
        destination = _destination(raw_root, entry)
        ranges = _partition_byte_ranges(expected_bytes, segment_count)
        plan.append(
            {
                "entry": entry,
                "destination": destination,
                "ranges": ranges,
            }
        )
    if sum(len(item["ranges"]) for item in plan) > effective_workers:
        raise RuntimeError(
            "segmented source plan exceeds the global connection ceiling"
        )
    return plan


def _assemble_segmented_source(
    *,
    entry: dict[str, Any],
    destination: Path,
    ranges: list[dict[str, int]],
    receipts: list[dict[str, Any]],
) -> None:
    expected_bytes = int(entry["content_length_bytes"])
    _assert_exact_range_coverage(ranges, total_bytes=expected_bytes)
    receipts_by_index = {
        int(receipt["index"]): receipt for receipt in receipts
    }
    if set(receipts_by_index) != {
        int(item["index"]) for item in ranges
    }:
        raise RuntimeError("segmented source range receipts are incomplete")
    assembled = destination.with_suffix(
        destination.suffix + ".assembled.partial"
    )
    digest = hashlib.sha256()
    written = 0
    with assembled.open("wb") as output:
        for byte_range in sorted(ranges, key=lambda item: item["start"]):
            receipt = _range_receipt_is_valid(
                entry=entry,
                destination=destination,
                byte_range=byte_range,
            )
            if receipt is None:
                raise RuntimeError(
                    "segmented source range receipt failed revalidation"
                )
            segment, _receipt_path, _temporary = _segment_paths(
                destination, byte_range
            )
            with segment.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
    if written != expected_bytes:
        raise RuntimeError(
            "segmented source assembly has an invalid final byte count"
        )
    final_sha256 = digest.hexdigest()
    expected_sha256 = _locked_download_sha256(entry)
    if expected_sha256 and final_sha256 != expected_sha256:
        raise RuntimeError(
            "segmented source assembly differs from its locked SHA-256"
        )
    range_receipts = [
        receipts_by_index[int(item["index"])]
        for item in sorted(ranges, key=lambda item: item["start"])
    ]
    range_manifest = destination.with_suffix(
        destination.suffix + ".ranges.json"
    )
    write_json(
        range_manifest,
        {
            "schema_version": 1,
            "source_url": str(entry["url"]),
            "remote_version_validator": _remote_version_validator(entry),
            "bytes": written,
            "sha256": final_sha256,
            "ranges": range_receipts,
        },
    )
    assembled.replace(destination)
    destination.with_suffix(destination.suffix + ".partial").unlink(
        missing_ok=True
    )
    destination.with_suffix(
        destination.suffix + ".restart.partial"
    ).unlink(missing_ok=True)
    stage_root = _segment_stage_root(destination)
    resolved_stage = stage_root.resolve()
    if (
        stage_root.is_symlink()
        or resolved_stage.parent != destination.parent.resolve()
        or resolved_stage.name != f".{destination.name}.segments"
    ):
        raise RuntimeError("refusing unsafe segmented source cleanup")
    shutil.rmtree(stage_root)
    entry["download"] = _final_download_receipt(
        entry,
        destination,
        state="downloaded_segmented",
        digest=final_sha256,
        extra={
            "remote_version_validator": _remote_version_validator(entry),
            "range_manifest": range_manifest.name,
            "range_receipts": range_receipts,
        },
    )


def _download_tail_segmented_entries(
    plan: list[dict[str, Any]],
    *,
    timeout: float,
    max_workers: int,
    retries: int,
    checkpoint: Callable[[], None],
    request_pacer: _RequestStartPacer,
) -> None:
    stop_event = Event()
    work_items: list[dict[str, Any]] = []
    for item in plan:
        work_item = {
            **item,
            "work_entry": copy.deepcopy(item["entry"]),
            "receipts": {},
        }
        work_items.append(work_item)
    total_ranges = sum(len(item["ranges"]) for item in work_items)
    with ThreadPoolExecutor(
        max_workers=min(
            max_workers, MAX_DIRECT_DOWNLOAD_WORKERS, total_ranges
        ),
        thread_name_prefix="fireviewer-zone-lidar-range",
    ) as executor:
        futures = {}
        for item_index, item in enumerate(work_items):
            for byte_range in item["ranges"]:
                future = executor.submit(
                    _download_range_segment_with_retries,
                    entry=item["work_entry"],
                    destination=item["destination"],
                    byte_range=byte_range,
                    timeout=timeout,
                    retries=retries,
                    before_request=request_pacer.wait,
                    should_cancel=stop_event.is_set,
                )
                futures[future] = (item_index, byte_range)
        try:
            for future in as_completed(futures):
                item_index, byte_range = futures[future]
                item = work_items[item_index]
                item["receipts"][int(byte_range["index"])] = future.result()
                if len(item["receipts"]) == len(item["ranges"]):
                    _assemble_segmented_source(
                        entry=item["work_entry"],
                        destination=item["destination"],
                        ranges=item["ranges"],
                        receipts=list(item["receipts"].values()),
                    )
                    original = item["entry"]
                    original.clear()
                    original.update(item["work_entry"])
                    checkpoint()
        except BaseException:
            stop_event.set()
            for future in futures:
                future.cancel()
            raise


def _download_direct_entries(
    entries: list[dict[str, Any]],
    *,
    raw_root: Path,
    timeout: float,
    max_workers: int,
    retries: int,
    checkpoint: Callable[[], None],
    request_pacer: _RequestStartPacer | None = None,
) -> None:
    if not entries:
        return
    _assert_unique_destinations(entries, raw_root=raw_root)
    if max_workers < 1:
        raise ValueError("direct download workers must be positive")
    shared_pacer = request_pacer or _RequestStartPacer(
        DIRECT_REQUEST_START_INTERVAL_SECONDS
    )
    # A crash may occur after atomic file promotion but before the next batched
    # source-lock checkpoint. Reconcile only those complete files whose final
    # receipt is absent; this performs no network request and prevents a tail
    # run from omitting their full SHA evidence.
    for original in entries:
        destination = _destination(raw_root, original)
        expected_bytes = original.get("content_length_bytes")
        if (
            destination.is_file()
            and not destination.is_symlink()
            and isinstance(expected_bytes, int)
            and destination.stat().st_size == expected_bytes
            and not _fast_download_receipt_matches(original, destination)
        ):
            updated = copy.deepcopy(original)
            _download(updated, destination, timeout=timeout)
            original.clear()
            original.update(updated)
            checkpoint()
    remaining_entries = [
        entry
        for entry in entries
        if not _destination_is_complete(
            entry, _destination(raw_root, entry)
        )
    ]
    if not remaining_entries:
        return
    segmented_plan = _tail_segmentation_plan(
        remaining_entries, raw_root=raw_root, max_workers=max_workers
    )
    if segmented_plan is not None:
        _download_tail_segmented_entries(
            segmented_plan,
            timeout=timeout,
            max_workers=max_workers,
            retries=retries,
            checkpoint=checkpoint,
            request_pacer=shared_pacer,
        )
        return
    stop_event = Event()

    def download_direct(entry: dict[str, Any]) -> dict[str, Any]:
        _download_with_retries(
            entry,
            _destination(raw_root, entry),
            timeout=timeout,
            retries=retries,
            before_request=shared_pacer.wait,
            should_cancel=stop_event.is_set,
        )
        return entry

    with ThreadPoolExecutor(
        max_workers=min(
            max_workers, MAX_DIRECT_DOWNLOAD_WORKERS, len(remaining_entries)
        ),
        thread_name_prefix="fireviewer-zone-lidar",
    ) as executor:
        futures = {
            executor.submit(download_direct, copy.deepcopy(entry)): entry
            for entry in remaining_entries
        }
        try:
            for future in as_completed(futures):
                updated = future.result()
                original = futures[future]
                original.clear()
                original.update(updated)
                checkpoint()
        except BaseException:
            stop_event.set()
            for future in futures:
                future.cancel()
            raise


def _download_raster_entries(
    entries: list[dict[str, Any]],
    *,
    raw_root: Path,
    timeout: float,
    max_workers: int,
    retries: int,
    checkpoint: Callable[[], None],
    request_pacer: _RequestStartPacer,
) -> None:
    if not entries:
        return
    if not 1 <= max_workers <= MAX_RASTER_DOWNLOAD_WORKERS:
        raise ValueError(
            "raster download workers must be between "
            f"1 and {MAX_RASTER_DOWNLOAD_WORKERS}"
        )
    _assert_unique_destinations(entries, raw_root=raw_root)
    stop_event = Event()

    def download_raster(entry: dict[str, Any]) -> dict[str, Any]:
        _download_with_retries(
            entry,
            _destination(raw_root, entry),
            timeout=timeout,
            retries=retries,
            before_request=request_pacer.wait,
            should_cancel=stop_event.is_set,
        )
        return entry

    with ThreadPoolExecutor(
        max_workers=min(max_workers, len(entries)),
        thread_name_prefix="fireviewer-zone-raster",
    ) as executor:
        futures = {
            executor.submit(download_raster, copy.deepcopy(entry)): entry
            for entry in entries
        }
        try:
            for future in as_completed(futures):
                updated = future.result()
                original = futures[future]
                original.clear()
                original.update(updated)
                checkpoint()
        except BaseException:
            stop_event.set()
            for future in futures:
                future.cancel()
            raise


def _lock_unpublished_direct_coverage(lock: dict[str, Any]) -> set[str]:
    """Lock coherent tiles for which IGN publishes no direct elevation quartet.

    A coastal tile may legitimately have no LiDAR, MNT, MNS or MNH product.
    Such a tile is excluded only when the four direct datasets are all absent
    together and both orthophoto references remain available.  Any partial
    quartet, insecure URL or missing imagery still fails closed.
    """

    entries = lock.get("entries")
    if not isinstance(entries, list):
        raise ValueError("source lock entries are malformed")
    by_tile: dict[str, set[str]] = {}
    entry_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("source lock entry is malformed")
        dataset = str(item.get("dataset", ""))
        tile_ref = str(item.get("tile_ref", ""))
        entry_by_key[(tile_ref, dataset)] = item
        url = str(item.get("url", ""))
        if url.startswith("https://"):
            continue
        if url:
            raise RuntimeError(
                f"source is unresolved and cannot be acquired: {item.get('id')}"
            )
        if dataset not in DIRECT_ELEVATION_DATASETS:
            raise RuntimeError(
                f"source is unresolved and cannot be acquired: {item.get('id')}"
            )
        if str(item.get("resolution_status", "")) != "unresolved":
            raise RuntimeError(
                f"source is unresolved and cannot be acquired: {item.get('id')}"
            )
        by_tile.setdefault(tile_ref, set()).add(dataset)

    for tile_ref, datasets in by_tile.items():
        if datasets != DIRECT_ELEVATION_DATASETS:
            raise RuntimeError(
                "source is unresolved and cannot be acquired: "
                f"{tile_ref} has an incomplete direct elevation quartet"
            )
        for dataset in ("ortho20", "ortho50"):
            imagery = entry_by_key.get((tile_ref, dataset))
            if not isinstance(imagery, dict) or not str(
                imagery.get("url", "")
            ).startswith("https://"):
                raise RuntimeError(
                    "source is unresolved and cannot be acquired: "
                    f"{tile_ref} has no locked {dataset} context"
                )

    tile_refs = sorted(by_tile)
    if tile_refs:
        lock["unpublished_direct_coverage"] = {
            "schema_version": 1,
            "state": "UNPUBLISHED_DIRECT_ELEVATION_QUARTETS_LOCKED",
            "datasets": sorted(DIRECT_ELEVATION_DATASETS),
            "tile_refs": tile_refs,
            "tile_count": len(tile_refs),
            "representation": "excluded_no_synthetic_surface",
        }
    else:
        lock.pop("unpublished_direct_coverage", None)
    return set(tile_refs)


def _selected_entries(lock: dict[str, Any], *, lod0_tiles: set[str]) -> list[dict[str, Any]]:
    entries = lock.get("entries")
    if not isinstance(entries, list):
        raise ValueError("source lock entries are malformed")
    selected: list[dict[str, Any]] = []
    light_profile = lock.get("source_profile") == LIGHT_PROFILE
    unpublished_direct_tiles = (
        set() if light_profile else _lock_unpublished_direct_coverage(lock)
    )
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("source lock entry is malformed")
        dataset = str(item.get("dataset", ""))
        if light_profile:
            included = dataset in {"terrain_lod3", "ortho_lod2", "ortho_lod0", "assets"}
        else:
            included = dataset in BASELINE_DATASETS or (
                dataset in LOD0_DATASETS
                and str(item.get("tile_ref", "")) in lod0_tiles
            )
        if included:
            if (
                str(item.get("tile_ref", "")) in unpublished_direct_tiles
                and dataset in DIRECT_ELEVATION_DATASETS
            ):
                continue
            if not str(item.get("url", "")).startswith("https://"):
                raise RuntimeError(f"source is unresolved and cannot be acquired: {item.get('id')}")
            selected.append(item)
    if not selected:
        raise RuntimeError("acquisition selected no sources")
    return selected


def _review_camera_lod0_tiles(rows: Iterable[dict[str, str]]) -> set[str]:
    """Return the one-kilometre tiles that contain the prescribed review cameras.

    The supplied production contract reserves LiDAR and 20 cm imagery for the
    immediate camera neighbourhood.  The review-camera grid is deterministic,
    so an omitted CLI override must not silently produce a scene without LOD0.
    """

    materialised = list(rows)
    if not materialised:
        raise ValueError("cannot derive LOD0 tiles from an empty zone")
    zone_xmin, zone_ymin, zone_xmax, zone_ymax = _zone_bbox(materialised)
    origin_x = (zone_xmin + zone_xmax) // 2
    origin_y = (zone_ymin + zone_ymax) // 2
    selected: set[str] = set()
    for offset_x, offset_y in REVIEW_CAMERA_TARGET_OFFSETS_METRES:
        target_x = origin_x + offset_x
        target_y = origin_y + offset_y
        for row in materialised:
            if (
                int(row["xmin"]) <= target_x < int(row["xmax"])
                and int(row["ymin"]) <= target_y < int(row["ymax"])
            ):
                selected.add(row["tile_ref"])
                break
        else:
            raise RuntimeError(
                f"review-camera target outside the zone tile grid: {target_x},{target_y}"
            )
    return selected


def _minimum_free_bytes(value_gib: float) -> int:
    if value_gib < 0:
        raise ValueError("minimum free GiB must not be negative")
    return int(value_gib * 1024**3)


def _assert_unique_destinations(
    entries: Iterable[dict[str, Any]], *, raw_root: Path
) -> None:
    destinations: dict[str, str] = {}
    for entry in entries:
        destination = _destination(raw_root, entry)
        key = os.path.normcase(str(destination.resolve()))
        entry_id = str(entry.get("id", "unknown"))
        if key in destinations:
            raise RuntimeError(
                "sources resolve to the same destination: "
                f"{destinations[key]}, {entry_id}"
            )
        destinations[key] = entry_id


def _locked_download_sha256(entry: dict[str, Any]) -> str:
    download = entry.get("download")
    if not isinstance(download, dict):
        return ""
    return str(
        download.get("expected_sha256") or download.get("sha256") or ""
    ).strip().lower()


def _remaining_entry_bytes(
    entry: dict[str, Any], *, raw_root: Path | None, bounded_bytes: int
) -> int:
    if raw_root is None:
        return bounded_bytes
    destination = _destination(raw_root, entry)
    if destination.is_file() and destination.stat().st_size == bounded_bytes:
        return (
            0
            if _fast_download_receipt_matches(entry, destination)
            else bounded_bytes
        )
    partial = destination.with_suffix(destination.suffix + ".partial")
    if not partial.is_file():
        return bounded_bytes
    partial_bytes = partial.stat().st_size
    if partial_bytes <= 0 or partial_bytes > bounded_bytes:
        return bounded_bytes
    if partial_bytes == bounded_bytes:
        # A complete partial is hashed exactly once by _download before atomic
        # promotion. Capacity checks never consume the payload.
        return bounded_bytes
    return bounded_bytes - partial_bytes


def _assert_capacity(
    workspace_root: Path,
    entries: list[dict[str, Any]],
    *,
    minimum_free_gib: float,
    raw_root: Path | None = None,
    segmented_staging_entries: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    usage = shutil.disk_usage(workspace_root)
    segmented_ids = {id(item) for item in segmented_staging_entries}
    segmented_staging_bytes = 0
    announced = 0
    announced_remaining = 0
    estimates = 0
    estimates_remaining = 0
    unknown: list[str] = []
    for item in entries:
        content_length = item.get("content_length_bytes")
        if isinstance(content_length, int) and content_length > 0:
            announced += content_length
            if id(item) in segmented_ids:
                # Ranges and the assembled temporary coexist until the full
                # SHA is verified and atomically promoted. Existing sequential
                # partials are already reflected in current free space.
                required = content_length * 2
                announced_remaining += required
                segmented_staging_bytes += required
            else:
                announced_remaining += _remaining_entry_bytes(
                    item, raw_root=raw_root, bounded_bytes=content_length
                )
            continue
        estimated = _wms_uncompressed_bytes(item)
        if estimated is None:
            unknown.append(str(item.get("id", "unknown")))
        else:
            estimates += estimated
            estimates_remaining += _remaining_entry_bytes(
                item, raw_root=raw_root, bounded_bytes=estimated
            )
    if unknown:
        preview = ", ".join(unknown[:5])
        suffix = "…" if len(unknown) > 5 else ""
        raise RuntimeError(
            "source sizes cannot be bounded before acquisition: "
            f"{preview}{suffix}"
        )
    expected = announced_remaining + estimates_remaining
    reserve = _minimum_free_bytes(minimum_free_gib)
    if usage.free < expected + reserve:
        raise RuntimeError(
            "zone acquisition exceeds bounded remaining storage: "
            f"free_bytes={usage.free} expected_download_bytes={expected} reserve_bytes={reserve}"
        )
    return {
        "checked_at": _utc_now(),
        "free_bytes_before": usage.free,
        "announced_download_bytes": announced,
        "announced_remaining_download_bytes": announced_remaining,
        "segmented_staging_bytes": segmented_staging_bytes,
        "wms_uncompressed_bound_bytes": estimates,
        "wms_remaining_bound_bytes": estimates_remaining,
        "expected_download_bytes": expected,
        "unknown_size_entries": 0,
        "reserve_bytes": reserve,
    }


def _artifact_from_receipt(
    value: object, *, zone_root: Path, label: str, suffixes: tuple[str, ...] = ()
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    raw = str(value.get("path", "")).strip()
    if not raw:
        raise ValueError(f"{label}.path is required")
    path = (zone_root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if not _is_below(zone_root, path) or not path.is_file():
        raise ValueError(f"{label} must be an existing file inside the zone workspace")
    if suffixes and path.suffix.lower() not in suffixes:
        raise ValueError(f"{label} has an unsupported file type")
    expected = str(value.get("sha256", ""))
    actual = sha256(path)
    if expected != actual:
        raise ValueError(f"{label} checksum does not match")
    return {
        "path": path.relative_to(zone_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": actual,
    }


_USDA_PRIM_SPEC = re.compile(
    rb'(?m)^[ \t]*(?:def|over|class)[ \t]+'
    rb'(?:[A-Za-z_][A-Za-z0-9_]*[ \t]+)?["\']'
)


def _usd_artifact_from_receipt(
    value: object, *, zone_root: Path, label: str
) -> dict[str, Any]:
    """Validate a USD artifact and reject header-only ASCII stages.

    Binary Crate contents are inspected by the subsequent Kit/OpenUSD scene
    gate.  Here we still require their real Crate magic instead of accepting
    any file merely carrying a ``.usdc`` extension.
    """

    artifact = _artifact_from_receipt(
        value,
        zone_root=zone_root,
        label=label,
        suffixes=(".usd", ".usda", ".usdc"),
    )
    path = (zone_root / artifact["path"]).resolve()
    with path.open("rb") as stream:
        prefix = stream.read(4 * 1024 * 1024)
    is_ascii = path.suffix.lower() == ".usda" or prefix.startswith(b"#usda")
    if is_ascii:
        if not prefix.startswith(b"#usda"):
            raise ValueError(f"{label} is not a valid USDA layer")
        if _USDA_PRIM_SPEC.search(prefix) is None:
            raise ValueError(f"{label} contains no material USD prim")
    elif not prefix.startswith(b"PXR-USDC"):
        raise ValueError(f"{label} is not a valid USD Crate layer")
    return artifact


def _validate_build_receipt(path: Path, *, zone_root: Path, zone_id: str) -> dict[str, Any]:
    receipt = _read_json(path, label="scene build receipt")
    if (
        receipt.get("schema_version") != 2
        or receipt.get("zone_id") != zone_id
        or receipt.get("coordinate_convention")
        != "usd_z_up_meters_lambert93"
    ):
        raise ValueError("scene build receipt has an unsupported zone or coordinate convention")
    root_usd = _usd_artifact_from_receipt(
        receipt.get("root_usd"), zone_root=zone_root, label="root_usd"
    )
    payloads = receipt.get("payloads")
    if not isinstance(payloads, list) or len(payloads) != 400:
        raise ValueError("scene build receipt must contain exactly 400 one-kilometre payloads")
    payload_artifacts = [
        _usd_artifact_from_receipt(item, zone_root=zone_root, label="payload")
        for item in payloads
    ]
    if len({item["path"] for item in payload_artifacts}) != 400:
        raise ValueError("scene build receipt payload paths must be unique")
    source_profile = str(receipt.get("source_profile", ""))
    detail_artifacts_by_level: dict[str, list[dict[str, Any]]] = {
        "HERO": [],
        "MID": [],
        "FAR": [],
    }
    if source_profile == "full":
        for level, key in (
            ("HERO", "detail_payloads"),
            ("MID", "detail_mid_payloads"),
            ("FAR", "detail_far_payloads"),
        ):
            details = receipt.get(key)
            if not isinstance(details, list) or len(details) != 400:
                raise ValueError(
                    "full scene build receipt must contain exactly 400 "
                    f"{level} detail payloads"
                )
            artifacts = [
                _usd_artifact_from_receipt(
                    item,
                    zone_root=zone_root,
                    label=f"{level} detail payload",
                )
                for item in details
            ]
            if len({item["path"] for item in artifacts}) != 400:
                raise ValueError(
                    f"scene {level} detail payload paths must be unique"
                )
            detail_artifacts_by_level[level] = artifacts
        all_detail_paths = [
            item["path"]
            for artifacts in detail_artifacts_by_level.values()
            for item in artifacts
        ]
        if len(set(all_detail_paths)) != 1200:
            raise ValueError("HERO/MID/FAR detail artifacts must be distinct")
        lidar_quality = _artifact_from_receipt(
            receipt.get("lidar_quality"),
            zone_root=zone_root,
            label="lidar_quality",
            suffixes=(".json",),
        )
        if int(receipt.get("lidar_quality", {}).get("source_count", 0)) != 400:
            raise ValueError("full scene build must prove all 400 LiDAR sources")
    elif source_profile != "light":
        raise ValueError("scene build receipt has an unsupported source profile")
    else:
        lidar_quality = None

    coverage = receipt.get("tile_coverage")
    if not isinstance(coverage, list) or len(coverage) != 400:
        raise ValueError("scene build receipt must contain 400 tile coverage records")
    tile_refs: set[str] = set()
    payload_paths = {item["path"] for item in payload_artifacts}
    detail_paths = {
        level: {item["path"] for item in artifacts}
        for level, artifacts in detail_artifacts_by_level.items()
    }
    instance_namespaces: set[int] = set()
    for item in coverage:
        if not isinstance(item, dict):
            raise ValueError("scene tile coverage record is malformed")
        tile_ref = str(item.get("tile_ref", ""))
        if not tile_ref or tile_ref in tile_refs:
            raise ValueError("scene tile coverage references must be unique")
        tile_refs.add(tile_ref)
        if str(item.get("terrain_payload", "")) not in payload_paths:
            raise ValueError("scene tile coverage does not bind its terrain payload")
        lods = item.get("terrain_lods")
        if not isinstance(lods, list) or not {"LOD1", "LOD2", "LOD3"}.issubset(lods):
            raise ValueError("every tile must expose the complete non-empty terrain LOD chain")
        if item.get("collision_lods") != ["NEAR", "FAR"]:
            raise ValueError("every tile must expose NEAR/FAR collision LODs")
        if source_profile == "full":
            detail_lods = item.get("detail_lods")
            if not isinstance(detail_lods, dict) or set(detail_lods) != {
                "HERO",
                "MID",
                "FAR",
            }:
                raise ValueError(
                    "scene tile coverage must bind HERO/MID/FAR details"
                )
            for level in ("HERO", "MID", "FAR"):
                if str(detail_lods.get(level, "")) not in detail_paths[level]:
                    raise ValueError(
                        f"scene tile coverage does not bind its {level} detail"
                    )
            if str(item.get("detail_payload", "")) != str(
                detail_lods["HERO"]
            ):
                raise ValueError("legacy detail payload must equal HERO")
            counts = item.get("detail_counts")
            if not isinstance(counts, dict) or set(counts) != {
                "buildings",
                "roads",
                "hydrology",
                "vegetation",
            }:
                raise ValueError("scene tile detail counts are incomplete")
            lod_counts = item.get("detail_lod_counts")
            if not isinstance(lod_counts, dict) or set(lod_counts) != {
                "HERO",
                "MID",
                "FAR",
            }:
                raise ValueError("scene tile detail LOD counts are incomplete")
            for level, level_counts in lod_counts.items():
                if (
                    not isinstance(level_counts, dict)
                    or set(level_counts)
                    != {"buildings", "roads", "hydrology", "vegetation"}
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for value in level_counts.values()
                    )
                ):
                    raise ValueError(
                        f"scene tile {level} detail counts are invalid"
                    )
            namespace = item.get("instance_namespace")
            if (
                isinstance(namespace, bool)
                or not isinstance(namespace, int)
                or namespace <= 0
                or namespace in instance_namespaces
            ):
                raise ValueError(
                    "scene tile instance namespaces must be positive and unique"
                )
            instance_namespaces.add(namespace)
    layers = receipt.get("layers")
    required_layers = {
        "terrain",
        "imagery",
        "hydrology",
        "roads",
        "buildings",
        "vegetation",
        "collisions",
        "semantics",
        "detail_streaming",
    }
    if not isinstance(layers, dict) or not required_layers.issubset(layers):
        raise ValueError("scene build receipt is missing required scene layers")
    for layer in required_layers:
        if not isinstance(layers[layer], dict) or int(layers[layer].get("prim_count", 0)) <= 0:
            raise ValueError(f"scene layer is not materially authored: {layer}")
    streaming = layers["detail_streaming"]
    if streaming.get("levels") != ["HERO", "MID", "FAR"]:
        raise ValueError("detail streaming must declare HERO/MID/FAR levels")
    collision = layers["collisions"]
    if (
        collision.get("levels") != ["NEAR", "FAR"]
        or float(collision.get("near_spacing_m", 0.0)) > 4.0
        or float(collision.get("far_spacing_m", 0.0)) > 32.0
    ):
        raise ValueError("collision streaming must declare 4 m NEAR and 32 m FAR")
    return {
        "receipt": _artifact_from_receipt(
            {"path": str(path), "sha256": sha256(path)},
            zone_root=zone_root,
            label="build receipt",
            suffixes=(".json",),
        ),
        "root_usd": root_usd,
        "payloads": payload_artifacts,
        "detail_payloads": detail_artifacts_by_level["HERO"],
        "detail_mid_payloads": detail_artifacts_by_level["MID"],
        "detail_far_payloads": detail_artifacts_by_level["FAR"],
        "lidar_quality": lidar_quality,
    }


def _validate_render_receipt(path: Path, *, zone_root: Path, zone_id: str, root_sha256: str) -> dict[str, Any]:
    receipt = _read_json(path, label="scene render receipt")
    if receipt.get("zone_id") != zone_id or receipt.get("root_usd_sha256") != root_sha256:
        raise ValueError("render receipt is not bound to the validated root USD")
    runtime = _artifact_from_receipt(receipt.get("runtime_720p"), zone_root=zone_root, label="runtime_720p")
    if receipt.get("runtime_720p", {}).get("width") != 1280 or receipt.get("runtime_720p", {}).get("height") != 720:
        raise ValueError("runtime receipt must prove exactly 1280x720")
    review = receipt.get("review_renders")
    if not isinstance(review, list) or len(review) < 12:
        raise ValueError("render receipt must contain at least twelve review renders")
    review_artifacts = []
    for item in review:
        if not isinstance(item, dict) or int(item.get("width", 0)) < 3840 or int(item.get("height", 0)) < 2160:
            raise ValueError("every review render must be at least 3840x2160")
        review_artifacts.append(_artifact_from_receipt(item, zone_root=zone_root, label="review render"))
    return {"receipt": _artifact_from_receipt({"path": str(path), "sha256": sha256(path)}, zone_root=zone_root, label="render receipt", suffixes=(".json",)), "runtime_720p": runtime, "review_renders": review_artifacts}


class ZoneSceneProduction:
    """Stateful command surface for one zone at a time."""

    def __init__(self, *, catalog_root: Path, workspace_root: Path, zone_id: str) -> None:
        self.catalog_root = _catalog_root(catalog_root)
        self.workspace_root = _workspace_root(workspace_root)
        _validate_roots(self.catalog_root, self.workspace_root)
        if zone_id not in ZONE_ORDER:
            raise ValueError(f"unsupported zone id: {zone_id}")
        self.zone_id = zone_id
        self.zone_root = _zone_root(self.workspace_root, zone_id)

    def preflight(self) -> dict[str, Any]:
        receipt = validate_catalog(self.catalog_root)
        state = _load_state(self.workspace_root)
        catalog_receipt_path = self.zone_root / "catalog-receipt.json"
        write_json(catalog_receipt_path, receipt)
        zone = _zone_state(state, self.zone_id)
        details = {
            "catalog_receipt": str(
                catalog_receipt_path.relative_to(self.workspace_root)
            )
        }
        # Every production invocation performs the native runtime preflight
        # before reaching this command.  Rechecking the immutable catalog must
        # never roll an interrupted zone back before its acquired sources or
        # registered scene evidence; that would make resumable production
        # impossible after a safe restart.
        if zone.get("phase") == "not_started":
            _record_phase(
                state,
                self.zone_id,
                "catalog_validated",
                details=details,
            )
        else:
            verified_at = _utc_now()
            zone["updated_at"] = verified_at
            history = zone["history"]
            assert isinstance(history, list)
            history.append(
                {
                    "phase": "catalog_revalidated",
                    "at": verified_at,
                    "preserved_phase": zone.get("phase"),
                    **details,
                }
            )
        _write_state(self.workspace_root, state)
        return receipt

    def resolve(self, *, timeout: float = 60.0, retries: int = 3) -> dict[str, Any]:
        if timeout <= 0 or retries < 1:
            raise ValueError("resolution timeout must be positive and retries must be at least one")
        state = _load_state(self.workspace_root)
        _assert_turn(state, self.zone_id)
        catalog_receipt = _read_json(self.zone_root / "catalog-receipt.json", label="catalog receipt")
        rows, report = _resolve_rows(_zone_rows(self.catalog_root, self.zone_id), timeout=timeout, retries=retries)
        fieldnames = list(rows[0])
        resolved_path = self.zone_root / "source-resolution.csv"
        with resolved_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        report.update({"zone_id": self.zone_id, "resolved_at": _utc_now(), "resolved_inventory_sha256": sha256(resolved_path)})
        report_path = self.zone_root / "coverage-report.json"
        write_json(report_path, report)
        source_lock = _source_lock_from_rows(catalog_receipt=catalog_receipt, zone_id=self.zone_id, rows=rows, resolution_report=report)
        write_json(_source_lock_path(self.zone_root), source_lock)
        _record_phase(state, self.zone_id, "sources_resolved", details={"source_lock": str(_source_lock_path(self.zone_root).relative_to(self.workspace_root)), "coverage_report": str(report_path.relative_to(self.workspace_root))})
        _write_state(self.workspace_root, state)
        return source_lock

    def acquire(
        self,
        *,
        lod0_tiles: Iterable[str],
        minimum_free_gib: float = 20.0,
        timeout: float = 180.0,
        download_workers: int = DEFAULT_RASTER_DOWNLOAD_WORKERS,
        direct_download_workers: int = DEFAULT_DIRECT_DOWNLOAD_WORKERS,
        retries: int = 3,
        source_profile: str = "full",
    ) -> dict[str, Any]:
        if timeout <= 0:
            raise ValueError("download timeout must be positive")
        if not 1 <= download_workers <= MAX_RASTER_DOWNLOAD_WORKERS:
            raise ValueError(
                "raster download workers must be between "
                f"1 and {MAX_RASTER_DOWNLOAD_WORKERS}"
            )
        if not 1 <= direct_download_workers <= MAX_DIRECT_DOWNLOAD_WORKERS:
            raise ValueError(
                "direct download workers must be between "
                f"1 and {MAX_DIRECT_DOWNLOAD_WORKERS}"
            )
        if not 1 <= retries <= 10:
            raise ValueError("download retries must be between 1 and 10")
        if source_profile not in SOURCE_PROFILES:
            raise ValueError(f"unknown source profile: {source_profile}")
        state = _load_state(self.workspace_root)
        _assert_turn(state, self.zone_id)
        prior_phase = str(_zone_state(state, self.zone_id).get("phase", ""))
        refreshable_light_phase = prior_phase in {
            "scene_built",
            "review_launch_requested",
            "renders_registered",
        }
        if prior_phase not in {"sources_resolved", "sources_acquired"} and not (
            source_profile == LIGHT_PROFILE and refreshable_light_phase
        ):
            raise RuntimeError("sources must be resolved before acquisition")
        zone_rows = _zone_rows(self.catalog_root, self.zone_id)
        if source_profile == LIGHT_PROFILE:
            lock = _activate_light_source_lock(
                zone_root=self.zone_root, zone_id=self.zone_id, rows=zone_rows
            )
            requested_lod0: set[str] = set()
            lod0_selection = "excluded_by_light_profile"
        else:
            lock = _load_source_lock(self.zone_root)
            if lock.get("source_profile") == LIGHT_PROFILE:
                raise RuntimeError(
                    "the active source lock is light; run resolve before requesting the full profile"
                )
            requested_lod0 = {value.strip() for value in lod0_tiles if value.strip()}
            lod0_selection = "explicit"
            if not requested_lod0:
                requested_lod0 = _review_camera_lod0_tiles(zone_rows)
                lod0_selection = "derived_from_review_cameras"
        valid_tiles = {row["tile_ref"] for row in zone_rows}
        unknown = requested_lod0 - valid_tiles
        if unknown:
            raise ValueError(f"LOD0 tiles are outside {self.zone_id}: {sorted(unknown)}")
        selected = _selected_entries(lock, lod0_tiles=requested_lod0)
        with _acquisition_lock(self.zone_root):
            raw_root = self.zone_root / "raw"
            raw_root.mkdir(parents=True, exist_ok=True)
            if raw_root.is_symlink():
                raise RuntimeError("raw source root may not be a symlink")
            _assert_unique_destinations(selected, raw_root=raw_root)
            measurement_entries = _measurement_samples(selected)
            direct_measurement_entries = [
                entry
                for entry in measurement_entries
                if _wms_uncompressed_bytes(entry) is None
            ]
            raster_measurement_entries = [
                entry
                for entry in measurement_entries
                if _wms_uncompressed_bytes(entry) is not None
            ]
            direct_entries = [
                entry
                for entry in selected
                if _wms_uncompressed_bytes(entry) is None
            ]
            request_pacer = _RequestStartPacer(
                DIRECT_REQUEST_START_INTERVAL_SECONDS
            )
            _measure_direct_entries_with_retries(
                direct_measurement_entries,
                timeout=min(timeout, 15.0),
                max_workers=direct_download_workers,
                attempts=retries,
                request_pacer=request_pacer,
                checkpoint=lambda: write_json(
                    _source_lock_path(self.zone_root),
                    copy.deepcopy(lock),
                ),
            )
            pending_direct_entries = [
                entry
                for entry in direct_entries
                if not _destination_is_complete(
                    entry, _destination(raw_root, entry)
                )
            ]
            if (
                pending_direct_entries
                and len(pending_direct_entries)
                <= max(
                    1,
                    min(
                        direct_download_workers,
                        MAX_DIRECT_DOWNLOAD_WORKERS,
                    )
                    // TAIL_SEGMENTATION_POOL_DIVISOR,
                )
            ):
                missing_validators = [
                    entry
                    for entry in pending_direct_entries
                    if _remote_version_validator(entry) is None
                ]
                _measure_direct_entries(
                    missing_validators,
                    timeout=min(timeout, 15.0),
                    max_workers=direct_download_workers,
                    request_pacer=request_pacer,
                    require_remote_validator=True,
                )
            for entry in raster_measurement_entries:
                _measure(
                    entry,
                    timeout=min(timeout, 15.0),
                    before_request=request_pacer.wait,
                )
            tail_segmentation_plan = _tail_segmentation_plan(
                direct_entries,
                raw_root=raw_root,
                max_workers=direct_download_workers,
            )
            capacity = _assert_capacity(
                self.workspace_root,
                selected,
                minimum_free_gib=minimum_free_gib,
                raw_root=raw_root,
                segmented_staging_entries=(
                    [
                        item["entry"]
                        for item in tail_segmentation_plan
                    ]
                    if tail_segmentation_plan is not None
                    else ()
                ),
            )
            completed_downloads = 0

            def checkpoint_download() -> None:
                nonlocal completed_downloads
                completed_downloads += 1
                # The lock is large (2,400 entries). Checkpoint batches without
                # rewriting it once per file, while preserving resumability after a
                # bounded amount of work.
                if completed_downloads % 25 == 0:
                    write_json(
                        _source_lock_path(self.zone_root),
                        copy.deepcopy(lock),
                    )

            raster_entries = [
                entry for entry in selected if _wms_uncompressed_bytes(entry) is not None
            ]
            # Stay below the provider's advertised 10-request/s/IP ceiling
            # while overlapping long-lived, independent transfers. This
            # changes transport utilization only: every locked source remains
            # independently downloaded and verified.
            _download_direct_entries(
                direct_entries,
                raw_root=raw_root,
                timeout=timeout,
                max_workers=direct_download_workers,
                retries=retries,
                checkpoint=checkpoint_download,
                request_pacer=request_pacer,
            )
            _download_raster_entries(
                raster_entries,
                raw_root=raw_root,
                timeout=timeout,
                max_workers=download_workers,
                retries=retries,
                checkpoint=checkpoint_download,
                request_pacer=request_pacer,
            )
            # BDTOPO is a material input to the requested scene layers, not
            # optional decoration.  The WFS page URLs and content hashes are
            # persisted alongside the GeoTIFF receipts before a build may run.
            if not _locked_vector_sources_valid(raw_root=raw_root, lock=lock):
                lock["vector_sources"] = _acquire_vector_sources(
                    raw_root=raw_root,
                    rows=_zone_rows(self.catalog_root, self.zone_id),
                    timeout=timeout,
                )
            lock["acquisition"] = {
                "completed_at": _utc_now(),
                "lod0_tiles": sorted(requested_lod0),
                "lod0_selection": lod0_selection,
                "source_profile": source_profile,
                "raster_download_workers": download_workers,
                "direct_download_workers": direct_download_workers,
                "direct_request_start_interval_seconds": (
                    DIRECT_REQUEST_START_INTERVAL_SECONDS
                ),
                "tail_segmentation": {
                    "activation_max_pending_pool_divisor": (
                        TAIL_SEGMENTATION_POOL_DIVISOR
                    ),
                    "minimum_range_bytes": TAIL_SEGMENT_MIN_BYTES,
                    "requires_remote_version_validator": True,
                    "maximum_connections": MAX_DIRECT_DOWNLOAD_WORKERS,
                },
                "download_retries": retries,
                "capacity": capacity,
            }
            write_json(_source_lock_path(self.zone_root), lock)
        if refreshable_light_phase:
            # A visual-source revision invalidates only the prior generated
            # scene evidence.  Preserve the old files and history, but prevent
            # a stale root receipt from being reused before the rebuild.
            zone_state = _zone_state(state, self.zone_id)
            for key in ("build_receipt", "root_usd", "root_usd_sha256", "review_launch"):
                zone_state.pop(key, None)
        _record_phase(state, self.zone_id, "sources_acquired", details={"source_lock": str(_source_lock_path(self.zone_root).relative_to(self.workspace_root)), "raw_root": str(raw_root.relative_to(self.workspace_root)), "selected_sources": len(selected), "source_profile": source_profile})
        _write_state(self.workspace_root, state)
        return lock

    def register_build(self, receipt_path: Path) -> dict[str, Any]:
        state = _load_state(self.workspace_root)
        _assert_turn(state, self.zone_id)
        phase = str(_zone_state(state, self.zone_id).get("phase", ""))
        if phase not in {"sources_acquired", "scene_built", "review_launch_requested"}:
            raise RuntimeError("sources must be acquired before registering a scene build")
        receipt = _validate_build_receipt(receipt_path.resolve(), zone_root=self.zone_root, zone_id=self.zone_id)
        _record_phase(state, self.zone_id, "scene_built", details={"build_receipt": receipt["receipt"]["path"], "root_usd": receipt["root_usd"]["path"], "root_usd_sha256": receipt["root_usd"]["sha256"]})
        _write_state(self.workspace_root, state)
        return receipt

    def build(self, *, timeout: float = 7200.0) -> dict[str, Any]:
        """Run the actual native Isaac/OpenUSD builder, then validate its receipt."""

        if timeout <= 0:
            raise ValueError("native scene build timeout must be positive")
        state = _load_state(self.workspace_root)
        _assert_turn(state, self.zone_id)
        phase = str(_zone_state(state, self.zone_id).get("phase", ""))
        if phase not in {"sources_acquired", "scene_built", "review_launch_requested"}:
            raise RuntimeError("sources must be acquired before building a scene")
        command = [
            sys.executable,
            "-m",
            "fireviewer_sdg.native_zone_scene",
            "--catalog-root",
            str(self.catalog_root),
            "--workspace-root",
            str(self.workspace_root),
            "--zone",
            self.zone_id,
        ]
        environment = os.environ.copy()
        environment["FW_SDG_ZONE_NATIVE_BUILD"] = "1"
        subprocess.run(
            command,
            check=True,
            timeout=timeout,
            env=environment,
        )
        receipt_path = self.zone_root / "build" / "build-receipt.json"
        receipt = self.register_build(receipt_path)
        # Opening Composer is an explicit, separately resumable review phase.
        # Coupling it to every rebuild spawned an editor instance on each
        # visual iteration, exhausting the local GPU and leaving black
        # viewports.  A caller that wants human review must request `review`
        # after this validated build returns.
        return receipt

    def open_review(self) -> dict[str, Any]:
        """Open the validated root USD in the available native Omniverse editor."""

        state = _load_state(self.workspace_root)
        zone = _zone_state(state, self.zone_id)
        if zone.get("phase") not in {"scene_built", "review_launch_requested"}:
            raise RuntimeError("a validated scene build is required before opening manual review")
        # Builds recorded by the first production revision retained artifact
        # coordinates in history only.  Recover that already-validated data
        # once, rather than rebuilding or accepting an unverified path.
        required = ("root_usd", "root_usd_sha256")
        if any(not str(zone.get(key, "")).strip() for key in required):
            history = zone.get("history")
            recovered: dict[str, Any] | None = None
            if isinstance(history, list):
                for item in reversed(history):
                    if not isinstance(item, dict) or item.get("phase") != "scene_built":
                        continue
                    if all(str(item.get(key, "")).strip() for key in required):
                        recovered = {key: item[key] for key in required}
                        break
            if recovered is None:
                raise RuntimeError("validated root USD is absent from the scene build state")
            zone.update(recovered)
            _write_state(self.workspace_root, state)
        relative_root = str(zone.get("root_usd", ""))
        root_usd = (self.zone_root / relative_root).resolve()
        if not _is_below(self.zone_root, root_usd) or not root_usd.is_file():
            raise RuntimeError("validated root USD is absent for manual review")
        if sha256(root_usd) != str(zone.get("root_usd_sha256", "")):
            raise RuntimeError("validated root USD changed before manual review")
        configured = os.getenv("FW_SDG_REVIEW_EDITOR", "").strip()
        if configured:
            editor = Path(configured).expanduser()
        elif sys.platform == "win32":
            editor = Path(
                "D:/Programs/NVIDIA omni/kit-app-template/_build/windows-x86_64/release/fireviewer_usd_composer.kit.bat"
            )
        else:
            editor = Path(
                "/workspace/fireviewer-omniverse/runtime/kit-app-template/"
                "_build/linux-x86_64/release/fireviewer_usd_composer.kit.sh"
            )
        if not editor.is_file():
            raise RuntimeError(
                "FireViewer USD Composer is unavailable for review; set FW_SDG_REVIEW_EDITOR "
                "to its native Kit launcher"
            )
        review_script = Path(__file__).resolve().parents[2] / "tools" / "open-zone-scene-in-composer.py"
        if not review_script.is_file():
            raise RuntimeError("native review opener script is absent")
        opened_receipt = self.zone_root / "review-opened.json"
        launch_receipt = self.zone_root / "review-launch.json"
        if launch_receipt.is_file():
            prior_launch = _read_json(
                launch_receipt, label="existing review launch receipt"
            )
            prior_pid = int(prior_launch.get("launcher_pid", 0))
            if _process_is_running(prior_pid):
                if (
                    prior_launch.get("root_usd_sha256")
                    != str(zone.get("root_usd_sha256", ""))
                    or not _review_process_matches(prior_pid, editor)
                ):
                    raise RuntimeError(
                        "a live review PID does not match the current Editor/root; "
                        "refusing to spawn another process"
                    )
                return {
                    **prior_launch,
                    "reused_existing_process": True,
                    "human_review": "pending; launch is not QA acceptance",
                }
        environment = os.environ.copy()
        environment.update(
            {
                "FW_SDG_REVIEW_USD": str(root_usd),
                "FW_SDG_REVIEW_OPENED_RECEIPT": str(opened_receipt),
                "FW_SDG_REVIEW_ZONE": self.zone_id,
            }
        )
        pending_receipt = self.zone_root / "editor-review-pending.json"
        if pending_receipt.is_file():
            environment["FW_SDG_REVIEW_PENDING_RECEIPT"] = str(pending_receipt)
        arguments = [str(editor), "--no-ros-env", "--exec", str(review_script)]
        if sys.platform == "win32" and editor.suffix.lower() in {".bat", ".cmd"}:
            # Hub exposes the Windows app through a batch launcher.  `call` is
            # required for cmd.exe; the Linux pod uses an argv list and never
            # invokes a shell.
            composer_command: str | list[str] = "call " + subprocess.list2cmdline(
                arguments
            )
            process = subprocess.Popen(
                composer_command,
                cwd=str(editor.parent),
                env=environment,
                shell=True,
            )
            editor_kind = "fireviewer_usd_composer_via_omniverse_hub"
        else:
            process = subprocess.Popen(
                arguments,
                cwd=str(editor.parent),
                env=environment,
                shell=False,
                start_new_session=True,
            )
            editor_kind = "fireviewer_usd_composer_linux_x11"
        result = {
            "zone_id": self.zone_id,
            "launched_at": _utc_now(),
            "editor": str(editor.resolve()),
            "editor_kind": editor_kind,
            "root_usd": str(root_usd.relative_to(self.zone_root)).replace("\\", "/"),
            "root_usd_sha256": sha256(root_usd),
            "launcher_pid": process.pid,
            "opened_receipt": opened_receipt.name,
            "human_review": "pending; launch is not QA acceptance",
        }
        write_json(launch_receipt, result)
        _record_phase(
            state,
            self.zone_id,
            "review_launch_requested",
            details={"review_launch": launch_receipt.relative_to(self.workspace_root).as_posix()},
        )
        _write_state(self.workspace_root, state)
        return result

    def register_render(self, receipt_path: Path) -> dict[str, Any]:
        state = _load_state(self.workspace_root)
        _assert_turn(state, self.zone_id)
        zone = _zone_state(state, self.zone_id)
        if zone.get("phase") not in {"scene_built", "review_launch_requested"}:
            raise RuntimeError("scene build must be registered before render evidence")
        receipt = _validate_render_receipt(receipt_path.resolve(), zone_root=self.zone_root, zone_id=self.zone_id, root_sha256=str(zone.get("root_usd_sha256", "")))
        _record_phase(state, self.zone_id, "renders_registered", details={"render_receipt": receipt["receipt"]["path"], "review_render_count": len(receipt["review_renders"])})
        _write_state(self.workspace_root, state)
        return receipt

    def render(self, *, timeout: float = 7200.0) -> dict[str, Any]:
        """Run the native RTX/Flow renderer and validate its immutable receipt."""

        if timeout <= 0:
            raise ValueError("native scene render timeout must be positive")
        state = _load_state(self.workspace_root)
        _assert_turn(state, self.zone_id)
        if _zone_state(state, self.zone_id).get("phase") not in {
            "scene_built",
            "review_launch_requested",
        }:
            raise RuntimeError("a validated scene build is required before rendering")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "fireviewer_sdg.native_zone_render",
                "--workspace-root",
                str(self.workspace_root),
                "--zone",
                self.zone_id,
            ],
            check=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
        return self.register_render(self.zone_root / "renders" / "render-receipt.json")

    def accept_qa(self, receipt_path: Path, *, human_reviewed: bool) -> dict[str, Any]:
        if not human_reviewed:
            raise RuntimeError("QA acceptance requires --human-reviewed")
        state = _load_state(self.workspace_root)
        zone = _zone_state(state, self.zone_id)
        if zone.get("phase") != "renders_registered":
            raise RuntimeError("render evidence is required before QA acceptance")
        receipt = _read_json(receipt_path.resolve(), label="human QA receipt")
        if receipt.get("zone_id") != self.zone_id or receipt.get("decision") != "accepted":
            raise ValueError("human QA receipt must accept this zone")
        if not str(receipt.get("reviewer", "")).strip() or not str(receipt.get("reviewed_at", "")).strip():
            raise ValueError("human QA receipt requires reviewer and reviewed_at")
        if receipt.get("root_usd_sha256") != zone.get("root_usd_sha256"):
            raise ValueError("human QA receipt is not bound to the built root USD")
        artifact = _artifact_from_receipt({"path": str(receipt_path.resolve()), "sha256": sha256(receipt_path.resolve())}, zone_root=self.zone_root, label="human QA receipt", suffixes=(".json",))
        _record_phase(state, self.zone_id, "qa_accepted", details={"qa_receipt": artifact["path"], "reviewer": str(receipt["reviewer"]).strip()})
        _write_state(self.workspace_root, state)
        return artifact

    def archive(self) -> dict[str, Any]:
        state = _load_state(self.workspace_root)
        zone = _zone_state(state, self.zone_id)
        if zone.get("phase") != "qa_accepted":
            raise RuntimeError("human QA acceptance is required before archive")
        archive_root = self.zone_root / "archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        package = archive_root / f"{self.zone_id}_scene_package.zip"
        temporary = package.with_suffix(".zip.partial")
        include_roots = ("build", "renders", "catalog-receipt.json", "coverage-report.json", "source-lock.json", "source-lock-full-v1.json", "review-launch.json", "review-opened.json")
        files: list[Path] = []
        for relative in include_roots:
            candidate = self.zone_root / relative
            if candidate.is_file():
                files.append(candidate)
            elif candidate.is_dir():
                files.extend(path for path in candidate.rglob("*") if path.is_file())
        for receipt_name in ("build_receipt", "render_receipt", "qa_receipt"):
            relative = str(zone.get(receipt_name, ""))
            if relative:
                candidate = self.zone_root / relative
                if candidate.is_file() and candidate not in files:
                    files.append(candidate)
        if not files:
            raise RuntimeError("archive has no verified scene artifacts")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(files):
                archive.write(path, path.relative_to(self.zone_root).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            corrupt = archive.testzip()
            if corrupt:
                raise RuntimeError(f"archive verification failed for {corrupt}")
        temporary.replace(package)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "pipeline": PIPELINE_ID,
            "zone_id": self.zone_id,
            "created_at": _utc_now(),
            "archive": {"path": package.name, "bytes": package.stat().st_size, "sha256": sha256(package)},
            "files": [{"path": path.relative_to(self.zone_root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(files)],
        }
        manifest_path = archive_root / "archive-manifest.json"
        write_json(manifest_path, manifest)
        _record_phase(state, self.zone_id, "archived", details={"archive": str(package.relative_to(self.workspace_root)), "archive_sha256": manifest["archive"]["sha256"], "archive_manifest": str(manifest_path.relative_to(self.workspace_root))})
        _write_state(self.workspace_root, state)
        return manifest

    def cleanup(self, *, confirmation: str) -> dict[str, Any]:
        if confirmation != self.zone_id:
            raise RuntimeError(f"cleanup confirmation must exactly equal {self.zone_id}")
        state = _load_state(self.workspace_root)
        zone = _zone_state(state, self.zone_id)
        if zone.get("phase") != "archived":
            raise RuntimeError("archive verification is required before raw-source cleanup")
        archive = self.workspace_root / str(zone.get("archive", ""))
        if not archive.is_file() or sha256(archive) != zone.get("archive_sha256"):
            raise RuntimeError("verified archive is absent or corrupted; refusing raw-source cleanup")
        raw_root = (self.zone_root / "raw").resolve()
        if not raw_root.exists():
            raise RuntimeError("raw source root is absent; refusing to claim cleanup")
        if raw_root.is_symlink() or not _is_below(self.zone_root, raw_root) or raw_root == self.zone_root:
            raise RuntimeError("raw source cleanup target is unsafe")
        raw_files = sum(1 for path in raw_root.rglob("*") if path.is_file())
        raw_bytes = sum(path.stat().st_size for path in raw_root.rglob("*") if path.is_file())
        shutil.rmtree(raw_root)
        _record_phase(state, self.zone_id, "cleanup_complete", details={"deleted_raw_files": raw_files, "deleted_raw_bytes": raw_bytes})
        _write_state(self.workspace_root, state)
        return {"zone_id": self.zone_id, "deleted_raw_files": raw_files, "deleted_raw_bytes": raw_bytes}


def _path_argument(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FireViewer sequential Omniverse zone-scene production")
    parser.add_argument("--catalog-root", default=os.getenv("FW_SDG_ZONE_CATALOG_ROOT", ""))
    parser.add_argument("--workspace-root", default=os.getenv("FW_SDG_ZONE_WORKSPACE_ROOT", str(DEFAULT_WORKSPACE_ROOT)))
    parser.add_argument("--zone", default=os.getenv("FW_SDG_ZONE_ID", "Z16"), choices=ZONE_ORDER)
    parser.add_argument("--phase", default=os.getenv("FW_SDG_ZONE_PHASE", "preflight"), choices=PHASES)
    parser.add_argument(
        "--lod0-tile",
        action="append",
        default=[item for item in os.getenv("FW_SDG_ZONE_LOD0_TILES", "").split(",") if item],
    )
    parser.add_argument("--minimum-free-gib", type=float, default=float(os.getenv("FW_SDG_ZONE_MINIMUM_FREE_GIB", "20")))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--download-workers",
        type=int,
        default=int(
            os.getenv(
                "FW_SDG_ZONE_DOWNLOAD_WORKERS",
                str(DEFAULT_RASTER_DOWNLOAD_WORKERS),
            )
        ),
    )
    parser.add_argument(
        "--direct-download-workers",
        type=int,
        default=int(
            os.getenv(
                "FW_SDG_ZONE_DIRECT_DOWNLOAD_WORKERS",
                str(DEFAULT_DIRECT_DOWNLOAD_WORKERS),
            )
        ),
    )
    parser.add_argument(
        "--source-profile",
        choices=SOURCE_PROFILES,
        default=os.getenv("FW_SDG_ZONE_SOURCE_PROFILE", "full"),
    )
    parser.add_argument(
        "--build-timeout",
        type=float,
        default=float(os.getenv("FW_SDG_ZONE_BUILD_TIMEOUT", "7200")),
    )
    parser.add_argument(
        "--render-timeout",
        type=float,
        default=float(os.getenv("FW_SDG_ZONE_RENDER_TIMEOUT", "7200")),
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--receipt", type=_path_argument, default=_path_argument(os.getenv("FW_SDG_ZONE_RECEIPT")) if os.getenv("FW_SDG_ZONE_RECEIPT") else None)
    parser.add_argument("--human-reviewed", action="store_true", default=os.getenv("FW_SDG_ZONE_HUMAN_REVIEWED", "").strip().lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--confirm-cleanup", default=os.getenv("FW_SDG_ZONE_CONFIRM_CLEANUP", ""))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    production = ZoneSceneProduction(catalog_root=Path(args.catalog_root), workspace_root=Path(args.workspace_root), zone_id=args.zone)
    if args.phase == "preflight":
        result = production.preflight()
    elif args.phase == "resolve":
        result = production.resolve(timeout=min(args.timeout, 60.0), retries=args.retries)
    elif args.phase == "acquire":
        result = production.acquire(
            lod0_tiles=args.lod0_tile,
            minimum_free_gib=args.minimum_free_gib,
            timeout=args.timeout,
            download_workers=args.download_workers,
            direct_download_workers=args.direct_download_workers,
            retries=args.retries,
            source_profile=args.source_profile,
        )
    elif args.phase == "build":
        result = production.build(timeout=args.build_timeout)
    elif args.phase == "review":
        result = production.open_review()
    elif args.phase == "render":
        result = production.render(timeout=args.render_timeout)
    elif args.phase == "qa":
        if args.receipt is None:
            raise SystemExit("--receipt is required for qa")
        result = production.accept_qa(args.receipt, human_reviewed=args.human_reviewed)
    elif args.phase == "archive":
        result = production.archive()
    else:
        result = production.cleanup(confirmation=args.confirm_cleanup)
    summary: dict[str, Any]
    if args.phase == "resolve":
        report = result.get("resolution_report", {})
        summary = {
            "phase": "sources_resolved",
            "zone_id": args.zone,
            "source_entries": len(result.get("entries", [])),
            "coverage": report.get("datasets", {}),
        }
    elif args.phase == "acquire":
        acquisition = result.get("acquisition", {})
        summary = {
            "phase": "sources_acquired",
            "zone_id": args.zone,
            "lod0_tiles": acquisition.get("lod0_tiles", []),
            "capacity": acquisition.get("capacity", {}),
        }
    else:
        summary = result
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

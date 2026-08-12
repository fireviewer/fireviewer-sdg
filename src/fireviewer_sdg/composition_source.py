"""Atomic, import-safe export of one native scene composition source.

The native USD build is the only authority for object identity and placement.
This module therefore does not discover, derive or repair scene content.  Its
single public operation accepts the already selected native build records,
validates their complete interchange contract, and publishes:

* ``composition-source.json``;
* one bounded portable ``height-field.json``;
* streaming ``trees.jsonl`` and ``buildings.jsonl`` inventories.

All referenced artifacts stay below a caller-supplied persistent-volume root
and are bound by a freshly computed SHA-256.  Publication is an atomic
directory rename.  Invalid or incomplete input leaves no published directory.

The resulting manifest is schema-compatible with
``native_variant_campaign.load_native_base_layout``.  No ``pxr`` or Omniverse
module is imported, so the exporter can also be unit-tested on ordinary CI.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import uuid
from array import array
from collections import OrderedDict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from fireviewer_sdg.scene_variants import (
    SceneRoute,
    VariantConstraints,
    Vec3,
    route_topology,
)


SCHEMA_VERSION = 1
LOD_LEVELS = ("HERO", "MID", "FAR")
OBJECT_CATEGORIES = ("trees", "buildings")
MAX_HEIGHT_FIELD_SIDE = 1025
MAX_HEIGHT_FIELD_SAMPLES = MAX_HEIGHT_FIELD_SIDE * MAX_HEIGHT_FIELD_SIDE
MAX_SIGNED_INT64 = 2**63 - 1
MAX_COMPOSITION_CONTRACT_BYTES = 256 * 1024 * 1024
MAX_HEIGHT_FIELD_SOURCE_BYTES = 64 * 1024 * 1024
MAX_DETAIL_PAYLOAD_BYTES = 512 * 1024 * 1024
MAX_INSTANCES_PER_DETAIL_PAYLOAD = 100_000
MAX_PREPARED_ROUTE_COUNT = 1_000_000
MAX_PREPARED_WATER_COUNT = 1_000_000
MAX_ADJACENT_EDGE_HEIGHT_DELTA_M = 0.05
ROOT_LOCAL_COORDINATE_CONTRACT = (
    "root_local_xy_metres_from_epsg2154_origin__z_ign69_metres"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_USD_SUFFIXES = {".usd", ".usda", ".usdc"}
_FORBIDDEN_ASSET = re.compile(
    r"(?:^|[/_.-])(cube|cone|cylinder|sphere|capsule|primitive|placeholder)"
    r"(?:$|[/_.-])",
    re.IGNORECASE,
)


class CompositionSourceError(RuntimeError):
    """Raised when a native build cannot be exported without fabrication."""


@dataclass(frozen=True, slots=True)
class ArtifactSource:
    """One existing file on the persistent volume.

    ``expected_sha256`` is optional because the exporter always computes the
    current hash.  Supplying it turns the source into an additional immutable
    lock and is recommended for upstream receipts.
    """

    path: str | Path
    prim_path: str = ""
    expected_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class TerrainPayloadSource:
    """One native terrain tile and its immutable streaming identity."""

    artifact: ArtifactSource
    tile_ref: str
    local_bounds: Mapping[str, float]
    epsg2154_bounds: Mapping[str, float]
    instance_namespace: int
    terrain_lods: Sequence[str]
    collision_lods: Sequence[str]


@dataclass(frozen=True, slots=True)
class NativeArtifactsSource:
    """Hash-bound artifacts emitted or accepted by the native zone build."""

    native_build_receipt: ArtifactSource
    scene_auto_validation: ArtifactSource
    root_usd: ArtifactSource
    terrain_payloads: Sequence[TerrainPayloadSource]
    water_payloads: Sequence[ArtifactSource]
    water_validation_state: str
    water_validation_evidence: ArtifactSource


@dataclass(frozen=True, slots=True)
class HeightFieldSource:
    """Already bounded regular terrain samples; the exporter never resamples."""

    origin_x: float
    origin_y: float
    spacing_m: float
    samples: Iterable[Sequence[float]]


@dataclass(frozen=True, slots=True)
class GroundMaterialPayloadSource:
    """One terrain-tile PBR payload from the shared ground material library."""

    artifact: ArtifactSource
    tile_id: str
    tile_ref: str
    tile_bounds_m: Mapping[str, float] | Sequence[float]


@dataclass(frozen=True, slots=True)
class GroundSurfaceSource:
    """Validated object-free tiled PBR ground used after rearrangement."""

    # ``material`` is the lightweight payload index, not a scene-wide shader.
    material: ArtifactSource
    tile_material_payloads: Sequence[GroundMaterialPayloadSource]
    validation_evidence: ArtifactSource
    validation_state: str
    kind: str = "object_free_pbr"
    removed_object_classes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlacementHeightTileSource:
    """One high-resolution, lazy root-local terrain height tile."""

    artifact: ArtifactSource
    tile_ref: str
    local_bounds: Mapping[str, float]
    width: int
    height: int
    x_coordinates: Sequence[float]
    y_coordinates: Sequence[float]
    format: str = "float32-le-row-major-south-to-north"


@dataclass(frozen=True, slots=True)
class AssetSource:
    """One real tree/building asset with a common native LOD lineage."""

    key: str
    category: str
    family: str
    lods: Mapping[str, ArtifactSource]
    lod_lineage: str
    grounding_offsets_m: Mapping[str, float]
    simready_validation_state: str
    simready_validation_evidence: ArtifactSource


@dataclass(frozen=True, slots=True)
class WaterMaterialSource:
    """Visible real-water PBR material at all streamed LODs."""

    lods: Mapping[str, ArtifactSource]
    pbr_validation_state: str
    pbr_validation_evidence: ArtifactSource


@dataclass(frozen=True, slots=True)
class DetailPayloadExtractionSource:
    """Explicit coordinate/primvar contract for lazy native USD extraction.

    The extractor opens one HERO detail payload at a time and never loads the
    referenced photoreal assets.  Current native detail stages author X/Y either
    directly in root-local metres or in absolute EPSG:2154; callers must name
    which representation was written.  Z remains the IGN69 terrain elevation
    used by the portable height field.

    Per-instance stable IDs, footprint radii and group IDs must already exist as
    PointInstancer primvars.  Their absence is a build-contract failure, not an
    invitation for the exporter to synthesize metadata.
    """

    coordinate_space: str
    root_origin_epsg2154: Sequence[float]
    stable_id_primvar: str = "fireviewer_stable_id"
    footprint_radius_primvar: str = "fireviewer_footprint_radius_m"
    group_id_primvar: str = "fireviewer_group_id"
    tile_bounds_tolerance_m: float = 20.0


@dataclass(frozen=True, slots=True)
class NativePreparationSource:
    """Existing, validated inputs for autonomous pod-side preparation.

    Every path is an already produced artifact.  Geometry is read from the
    accepted native terrain/HERO payloads and the locked BDTOPO vectors; this
    source deliberately contains no operator-authored routes, water polygons,
    height samples or suitability geometry.
    """

    zone_root: str | Path
    scene_auto_validation: str | Path
    asset_manifest: str | Path
    asset_lod_validation: str | Path
    asset_pbr_validation: str | Path
    ground_artifact_root: str | Path
    ground_authoring_receipt: str | Path


@dataclass(frozen=True, slots=True)
class _ResolvedArtifact:
    physical_path: Path
    portable_path: str
    sha256: str
    prim_path: str


@dataclass(frozen=True, slots=True)
class _ResolvedTerrain:
    artifact: _ResolvedArtifact
    tile_ref: str
    local_bounds: Mapping[str, float]
    epsg2154_bounds: Mapping[str, float]
    instance_namespace: int
    terrain_lods: tuple[str, ...]
    collision_lods: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PreparedTerrainGrid:
    artifact: ArtifactSource
    tile_ref: str
    instance_namespace: int
    terrain_lods: tuple[str, ...]
    collision_lods: tuple[str, ...]
    epsg2154_bounds: tuple[float, float, float, float]
    local_bounds: dict[str, float]
    x_coordinates: tuple[float, ...]
    y_coordinates: tuple[float, ...]
    elevations: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class _PreparedPlacementHeight:
    tile_ref: str
    local_bounds: dict[str, float]
    x_coordinates: tuple[float, ...]
    y_coordinates: tuple[float, ...]
    physical_path: Path
    final_path: Path
    sha256: str
    south_edge: tuple[float, ...]
    north_edge: tuple[float, ...]
    west_edge: tuple[float, ...]
    east_edge: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _NativeSuitabilityObservations:
    tree_families_by_tile_ref: Mapping[str, tuple[str, ...]]
    building_group_bounds: Mapping[str, tuple[float, float, float, float, int]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_below(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _trimmed(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise CompositionSourceError(f"{label} must be a non-empty trimmed string")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompositionSourceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CompositionSourceError(f"{label} must be finite")
    return result


def _positive(value: object, *, label: str) -> float:
    result = _finite(value, label=label)
    if result <= 0.0:
        raise CompositionSourceError(f"{label} must be positive")
    return result


def _positive_id(value: object, *, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > MAX_SIGNED_INT64
    ):
        raise CompositionSourceError(
            f"{label} must be a positive signed int64 integer"
        )
    return value


def _portable_input_path(value: str | Path, *, label: str) -> Path:
    if isinstance(value, Path):
        raw = str(value)
    elif isinstance(value, str):
        raw = value
    else:
        raise CompositionSourceError(f"{label}.path must be a path")
    if not raw.strip() or raw != raw.strip():
        raise CompositionSourceError(f"{label}.path must be non-empty and trimmed")
    normalized = raw.replace("\\", "/")
    path = Path(normalized)
    if not path.is_absolute() and ".." in path.parts:
        raise CompositionSourceError(
            f"{label}.path must not escape the persistent volume"
        )
    return path


def _resolve_artifact(
    source: ArtifactSource,
    *,
    volume_root: Path,
    label: str,
    require_usd: bool = False,
    require_prim: bool = False,
) -> _ResolvedArtifact:
    if not isinstance(source, ArtifactSource):
        raise CompositionSourceError(f"{label} must be an ArtifactSource")
    raw_path = _portable_input_path(source.path, label=label)
    candidate = (
        raw_path.resolve()
        if raw_path.is_absolute()
        else (volume_root / raw_path).resolve()
    )
    if not _is_below(volume_root, candidate) or not candidate.is_file():
        raise CompositionSourceError(
            f"{label} must be an existing file below the persistent volume"
        )
    portable = candidate.relative_to(volume_root).as_posix()
    if require_usd and candidate.suffix.casefold() not in _USD_SUFFIXES:
        raise CompositionSourceError(f"{label} must reference a USD layer")
    if require_usd and _FORBIDDEN_ASSET.search(portable):
        raise CompositionSourceError(
            f"{label} references a forbidden primitive or placeholder"
        )
    prim_path = source.prim_path
    if not isinstance(prim_path, str):
        raise CompositionSourceError(f"{label}.prim_path must be a string")
    if require_prim and (not prim_path.startswith("/") or prim_path == "/"):
        raise CompositionSourceError(
            f"{label}.prim_path must identify a concrete absolute USD prim"
        )
    actual = _sha256(candidate)
    expected = source.expected_sha256
    if expected is not None:
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            raise CompositionSourceError(
                f"{label}.expected_sha256 must be lowercase SHA-256"
            )
        if expected != actual:
            raise CompositionSourceError(
                f"{label} hash mismatch: expected {expected}, found {actual}"
            )
    return _ResolvedArtifact(candidate, portable, actual, prim_path)


def _artifact_record(
    value: _ResolvedArtifact,
    *,
    isolated_content_roles: Sequence[str] = (),
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": value.portable_path,
        "sha256": value.sha256,
    }
    if value.prim_path:
        record["prim_path"] = value.prim_path
    if isolated_content_roles:
        record["isolated_content_roles"] = sorted(
            set(isolated_content_roles)
        )
    return record


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompositionSourceError(f"{label} is not readable JSON") from error
    if not isinstance(value, dict):
        raise CompositionSourceError(f"{label} must be a JSON object")
    return value


def _portable_receipt_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise CompositionSourceError(
            f"{label} must be a persistent-volume-relative path"
        )
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise CompositionSourceError(
            f"{label} must be a persistent-volume-relative path"
        )
    return normalized


def _receipt_artifact_tuple(value: object, *, label: str) -> tuple[str, str]:
    if not isinstance(value, Mapping):
        raise CompositionSourceError(f"{label} must be an artifact object")
    path = _portable_receipt_path(value.get("path"), label=f"{label}.path")
    sha = value.get("sha256")
    if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
        raise CompositionSourceError(f"{label}.sha256 must be lowercase SHA-256")
    return path, sha


def _validate_native_receipts(
    *,
    base_scene_id: str,
    native: NativeArtifactsSource,
    build_ref: _ResolvedArtifact,
    auto_ref: _ResolvedArtifact,
    root_ref: _ResolvedArtifact,
    terrain: Sequence[_ResolvedTerrain],
    volume_root: Path,
) -> dict[str, int]:
    build = _read_json(build_ref.physical_path, label="native build receipt")
    if (
        build.get("schema_version") != 2
        or build.get("zone_id") != base_scene_id
        or build.get("source_profile") != "full"
        or build.get("fire_simulation_status")
        != "blocked_pending_editor_review"
    ):
        raise CompositionSourceError(
            "native build receipt must be the full blocked schema-2 build "
            f"for {base_scene_id}"
        )
    if root_ref.prim_path != "/World":
        raise CompositionSourceError("root_usd.prim_path must be /World")
    if build_ref.physical_path.parent.name != "build":
        raise CompositionSourceError(
            "native build receipt must live in the native zone build directory"
        )
    zone_root = build_ref.physical_path.parent.parent

    def normalized_build_artifact(
        record: object,
        *,
        label: str,
    ) -> tuple[str, str]:
        raw_path, sha = _receipt_artifact_tuple(record, label=label)
        resolved = (zone_root / raw_path).resolve()
        if not _is_below(volume_root, resolved):
            raise CompositionSourceError(
                f"{label} escapes the persistent volume"
            )
        return resolved.relative_to(volume_root).as_posix(), sha

    root_record = normalized_build_artifact(
        build.get("root_usd"), label="native build receipt root_usd"
    )
    if root_record != (root_ref.portable_path, root_ref.sha256):
        raise CompositionSourceError(
            "native build receipt root_usd must resolve to the exact root path "
            "and hash"
        )
    raw_payloads = build.get("payloads")
    if not isinstance(raw_payloads, list) or len(raw_payloads) != 400:
        raise CompositionSourceError(
            "native build receipt must contain exactly 400 terrain payloads"
        )
    receipt_payloads = {
        normalized_build_artifact(
            record, label=f"native build receipt payloads[{index}]"
        )
        for index, record in enumerate(raw_payloads)
    }
    expected_payloads = {
        (tile.artifact.portable_path, tile.artifact.sha256)
        for tile in terrain
    }
    if len(receipt_payloads) != 400 or receipt_payloads != expected_payloads:
        raise CompositionSourceError(
            "terrain sources must be the exact 400 hash-bound payloads from "
            "the native build receipt"
        )
    for field in ("detail_payloads", "detail_mid_payloads", "detail_far_payloads"):
        records = build.get(field)
        if not isinstance(records, list) or len(records) != 400:
            raise CompositionSourceError(
                f"native build receipt must contain exactly 400 {field}"
            )
        for index, record in enumerate(records):
            _receipt_artifact_tuple(
                record, label=f"native build receipt {field}[{index}]"
            )
    coverage = build.get("tile_coverage")
    if not isinstance(coverage, list) or len(coverage) != 400:
        raise CompositionSourceError(
            "native build receipt must contain exact 400-tile coverage metadata"
        )
    expected_coverage = {
        (
            tile.tile_ref,
            tile.artifact.portable_path,
            tile.instance_namespace,
            tile.terrain_lods,
            tile.collision_lods,
        )
        for tile in terrain
    }
    receipt_by_raw_path = {
        str(record.get("path")): str(record.get("sha256"))
        for record in raw_payloads
        if isinstance(record, Mapping)
    }
    actual_coverage: set[
        tuple[str, str, int, tuple[str, ...], tuple[str, ...]]
    ] = set()
    for index, record in enumerate(coverage):
        if not isinstance(record, Mapping):
            raise CompositionSourceError(
                f"native build receipt tile_coverage[{index}] is malformed"
            )
        raw_path = record.get("terrain_payload")
        namespace = record.get("instance_namespace")
        terrain_lods = record.get("terrain_lods")
        collision_lods = record.get("collision_lods")
        if (
            not isinstance(raw_path, str)
            or isinstance(namespace, bool)
            or not isinstance(namespace, int)
            or not isinstance(terrain_lods, list)
            or any(not isinstance(value, str) for value in terrain_lods)
            or not isinstance(collision_lods, list)
            or any(not isinstance(value, str) for value in collision_lods)
        ):
            raise CompositionSourceError(
                f"native build receipt tile_coverage[{index}] is incomplete"
            )
        normalized_path, _ = normalized_build_artifact(
            {
                "path": raw_path,
                "sha256": receipt_by_raw_path.get(raw_path, ""),
            },
            label=f"native build receipt tile_coverage[{index}]",
        )
        actual_coverage.add(
            (
                str(record.get("tile_ref", "")),
                normalized_path,
                namespace,
                tuple(terrain_lods),
                tuple(collision_lods),
            )
        )
    if actual_coverage != expected_coverage:
        raise CompositionSourceError(
            "terrain tile refs, LODs, namespaces or paths differ from native "
            "coverage"
        )
    layers = build.get("layers")
    if not isinstance(layers, Mapping):
        raise CompositionSourceError("native build receipt has no layer inventory")
    counts: dict[str, int] = {}
    for layer in ("vegetation", "buildings", "hydrology"):
        record = layers.get(layer)
        count = record.get("prim_count") if isinstance(record, Mapping) else None
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise CompositionSourceError(
                f"native build layer {layer} must be non-empty"
            )
        counts[layer] = count
    roads = layers.get("roads")
    rendered_road_count = (
        roads.get("prim_count") if isinstance(roads, Mapping) else None
    )
    if (
        isinstance(rendered_road_count, bool)
        or not isinstance(rendered_road_count, int)
        or rendered_road_count < 0
    ):
        raise CompositionSourceError(
            "native build road layer has an invalid rendered-geometry count"
        )
    source_route_count = (
        roads.get("source_feature_count", rendered_road_count)
        if isinstance(roads, Mapping)
        else None
    )
    if (
        isinstance(source_route_count, bool)
        or not isinstance(source_route_count, int)
        or source_route_count < 1
    ):
        raise CompositionSourceError(
            "native build road layer has no source-backed route topology"
        )
    if rendered_road_count == 0 and (
        not isinstance(roads, Mapping)
        or roads.get("visible_representation")
        != "orthophoto_derived_terrain_material"
        or roads.get("geometry_authoring") != "disabled"
        or roads.get("asset_dependencies") != []
    ):
        raise CompositionSourceError(
            "native build road layer omits the orthophoto-only visual contract"
        )
    # Road topology is an input to placement/constraints, not a count of USD
    # road meshes.  The latter is deliberately zero for orthophoto rendering.
    counts["roads"] = source_route_count
    auto = _read_json(auto_ref.physical_path, label="scene auto validation")
    if (
        auto.get("state") != "AUTO_VALIDATED"
        or auto.get("fire_simulation_status")
        != "blocked_pending_editor_review"
        or auto.get("build_receipt_sha256") != build_ref.sha256
        or auto.get("root_usd_sha256") != root_ref.sha256
    ):
        raise CompositionSourceError(
            "scene auto validation is missing, stale or bound to other artifacts"
        )
    return counts


def _validate_water_evidence(
    *,
    native: NativeArtifactsSource,
    volume_root: Path,
    water_payloads: Sequence[_ResolvedArtifact],
) -> _ResolvedArtifact:
    if native.water_validation_state != "ISOLATED_WATER_VALIDATED":
        raise CompositionSourceError(
            "water payloads lack ISOLATED_WATER_VALIDATED evidence"
        )
    evidence = _resolve_artifact(
        native.water_validation_evidence,
        volume_root=volume_root,
        label="isolated water validation evidence",
    )
    payload = _read_json(evidence.physical_path, label="isolated water validation")
    if (
        payload.get("state") != "ISOLATED_WATER_VALIDATED"
        or payload.get("visible") is not True
        or payload.get("content_roles") != ["water"]
    ):
        raise CompositionSourceError(
            "isolated water validation must prove visible water-only content"
        )
    records = payload.get("payloads")
    if not isinstance(records, list):
        raise CompositionSourceError(
            "isolated water validation has no payload inventory"
        )
    actual = {
        _receipt_artifact_tuple(
            record, label=f"isolated water validation payloads[{index}]"
        )
        for index, record in enumerate(records)
    }
    expected = {
        (artifact.portable_path, artifact.sha256)
        for artifact in water_payloads
    }
    if len(actual) != len(water_payloads) or actual != expected:
        raise CompositionSourceError(
            "isolated water validation is stale or bound to other payloads"
        )
    return evidence


def _normalized_bounds(bounds: Mapping[str, object]) -> dict[str, float]:
    if not isinstance(bounds, Mapping):
        raise CompositionSourceError("bounds must be an object")
    result = {
        key: _finite(bounds.get(key), label=f"bounds.{key}")
        for key in ("min_x", "min_y", "max_x", "max_y")
    }
    if (
        result["max_x"] <= result["min_x"]
        or result["max_y"] <= result["min_y"]
    ):
        raise CompositionSourceError("bounds must have positive area")
    return result


def _normalized_bounds_record(
    value: Mapping[str, object] | Sequence[object],
    *,
    label: str,
) -> dict[str, float]:
    if isinstance(value, Mapping):
        return _normalized_bounds(value)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes)
    ) and len(value) == 4:
        result = {
            key: _finite(value[index], label=f"{label}[{index}]")
            for index, key in enumerate(
                ("min_x", "min_y", "max_x", "max_y")
            )
        }
        if (
            result["max_x"] <= result["min_x"]
            or result["max_y"] <= result["min_y"]
        ):
            raise CompositionSourceError(f"{label} must have positive area")
        return result
    raise CompositionSourceError(
        f"{label} must be a bounds object or four-value array"
    )


def _resolve_terrain_payloads(
    sources: Sequence[TerrainPayloadSource],
    *,
    volume_root: Path,
    scene_bounds: Mapping[str, float],
    epsg2154_origin: Sequence[float],
) -> tuple[_ResolvedTerrain, ...]:
    origin = _vector(
        epsg2154_origin, size=3, label="epsg2154_origin"
    )
    result: list[_ResolvedTerrain] = []
    tile_refs: set[str] = set()
    namespaces: set[int] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, TerrainPayloadSource):
            raise CompositionSourceError(
                f"terrain_payloads[{index}] must be a TerrainPayloadSource"
            )
        tile_ref = _trimmed(
            source.tile_ref, label=f"terrain_payloads[{index}].tile_ref"
        )
        if tile_ref in tile_refs:
            raise CompositionSourceError(f"terrain tile_ref repeats: {tile_ref}")
        namespace = source.instance_namespace
        if (
            isinstance(namespace, bool)
            or not isinstance(namespace, int)
            or namespace < 1
            or namespace >= 1 << 20
            or namespace in namespaces
        ):
            raise CompositionSourceError(
                f"terrain_payloads[{index}].instance_namespace is invalid/repeated"
            )
        terrain_lods = tuple(source.terrain_lods)
        collision_lods = tuple(source.collision_lods)
        if terrain_lods not in {
            ("LOD1", "LOD2", "LOD3"),
            ("LOD0", "LOD1", "LOD2", "LOD3"),
        }:
            raise CompositionSourceError(
                f"terrain_payloads[{index}].terrain_lods is not an accepted "
                "native LOD set"
            )
        if collision_lods != ("NEAR", "FAR"):
            raise CompositionSourceError(
                f"terrain_payloads[{index}].collision_lods must be NEAR/FAR"
            )
        local_bounds = _normalized_bounds(source.local_bounds)
        epsg_bounds = _normalized_bounds(source.epsg2154_bounds)
        if (
            local_bounds["min_x"] < scene_bounds["min_x"] - 1.0e-6
            or local_bounds["min_y"] < scene_bounds["min_y"] - 1.0e-6
            or local_bounds["max_x"] > scene_bounds["max_x"] + 1.0e-6
            or local_bounds["max_y"] > scene_bounds["max_y"] + 1.0e-6
        ):
            raise CompositionSourceError(
                f"terrain tile {tile_ref} leaves root-local scene bounds"
            )
        if not all(
            math.isclose(actual, expected, abs_tol=0.01, rel_tol=0.0)
            for actual, expected in (
                (
                    epsg_bounds["min_x"],
                    origin[0] + local_bounds["min_x"],
                ),
                (
                    epsg_bounds["min_y"],
                    origin[1] + local_bounds["min_y"],
                ),
                (
                    epsg_bounds["max_x"],
                    origin[0] + local_bounds["max_x"],
                ),
                (
                    epsg_bounds["max_y"],
                    origin[1] + local_bounds["max_y"],
                ),
            )
        ):
            raise CompositionSourceError(
                f"terrain tile {tile_ref} local/EPSG:2154 bounds diverge"
            )
        artifact = _resolve_artifact(
            source.artifact,
            volume_root=volume_root,
            label=f"terrain_payloads[{index}]",
            require_usd=True,
            require_prim=True,
        )
        result.append(
            _ResolvedTerrain(
                artifact=artifact,
                tile_ref=tile_ref,
                local_bounds=local_bounds,
                epsg2154_bounds=epsg_bounds,
                instance_namespace=namespace,
                terrain_lods=terrain_lods,
                collision_lods=collision_lods,
            )
        )
        tile_refs.add(tile_ref)
        namespaces.add(namespace)
    if len({tile.artifact.portable_path for tile in result}) != 400:
        raise CompositionSourceError("terrain payload paths must be unique")
    total_area = 0.0
    for index, tile in enumerate(result):
        bounds = tile.local_bounds
        total_area += (bounds["max_x"] - bounds["min_x"]) * (
            bounds["max_y"] - bounds["min_y"]
        )
        for other in result[index + 1 :]:
            overlap_x = min(bounds["max_x"], other.local_bounds["max_x"]) - max(
                bounds["min_x"], other.local_bounds["min_x"]
            )
            overlap_y = min(bounds["max_y"], other.local_bounds["max_y"]) - max(
                bounds["min_y"], other.local_bounds["min_y"]
            )
            if overlap_x > 1.0e-6 and overlap_y > 1.0e-6:
                raise CompositionSourceError(
                    f"terrain tiles {tile.tile_ref} and {other.tile_ref} overlap"
                )
    expected_area = (
        scene_bounds["max_x"] - scene_bounds["min_x"]
    ) * (scene_bounds["max_y"] - scene_bounds["min_y"])
    if not math.isclose(total_area, expected_area, abs_tol=1.0e-3, rel_tol=0.0):
        raise CompositionSourceError(
            "terrain tiles do not partition the complete root-local scene bounds"
        )
    return tuple(result)


def _resolve_ground_surface(
    *,
    source: GroundSurfaceSource,
    terrain: Sequence[_ResolvedTerrain],
    volume_root: Path,
) -> tuple[_ResolvedArtifact, list[dict[str, object]], _ResolvedArtifact]:
    if not isinstance(source, GroundSurfaceSource):
        raise CompositionSourceError(
            "ground_surface must be a GroundSurfaceSource"
        )
    if (
        source.kind != "object_free_pbr"
        or source.removed_object_classes
        or source.validation_state != "OBJECT_FREE_PBR_VALIDATED"
    ):
        raise CompositionSourceError(
            "ground surface must be explicitly OBJECT_FREE_PBR_VALIDATED; "
            "raw or object-bearing orthophotos are forbidden"
        )
    index = _resolve_artifact(
        source.material,
        volume_root=volume_root,
        label="ground_material payload index",
        require_usd=True,
        require_prim=True,
    )
    evidence = _resolve_artifact(
        source.validation_evidence,
        volume_root=volume_root,
        label="ground material object-free validation evidence",
    )
    payloads = tuple(source.tile_material_payloads)
    if len(payloads) != 400:
        raise CompositionSourceError(
            "ground surface requires exactly 400 tile material payloads"
        )
    terrain_by_ref = {tile.tile_ref: tile for tile in terrain}
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = {index.portable_path}
    for item_index, payload in enumerate(payloads):
        if not isinstance(payload, GroundMaterialPayloadSource):
            raise CompositionSourceError(
                f"ground tile payload {item_index} has the wrong type"
            )
        tile_id = _trimmed(
            payload.tile_id,
            label=f"ground tile payload {item_index}.tile_id",
        )
        tile_ref = _trimmed(
            payload.tile_ref,
            label=f"ground tile payload {item_index}.tile_ref",
        )
        if (
            tile_id != tile_ref
            or tile_id in seen_ids
            or tile_id not in terrain_by_ref
        ):
            raise CompositionSourceError(
                f"ground tile payload {item_index} has divergent, duplicate "
                "or unknown tile identity"
            )
        bounds = _normalized_bounds_record(
            payload.tile_bounds_m,
            label=f"ground tile payload {tile_id}.tile_bounds_m",
        )
        terrain_bounds = terrain_by_ref[tile_id].local_bounds
        if not all(
            math.isclose(
                bounds[key],
                terrain_bounds[key],
                abs_tol=0.01,
                rel_tol=0.0,
            )
            for key in ("min_x", "min_y", "max_x", "max_y")
        ):
            raise CompositionSourceError(
                f"ground tile payload {tile_id} bounds differ from terrain"
            )
        artifact = _resolve_artifact(
            payload.artifact,
            volume_root=volume_root,
            label=f"ground tile payload {tile_id}",
            require_usd=True,
            require_prim=True,
        )
        if artifact.prim_path != "/Ground":
            raise CompositionSourceError(
                f"ground tile payload {tile_id}.prim_path must be /Ground"
            )
        if artifact.portable_path in seen_paths:
            raise CompositionSourceError(
                f"ground tile payload path repeats: {artifact.portable_path}"
            )
        seen_paths.add(artifact.portable_path)
        seen_ids.add(tile_id)
        records.append(
            {
                "tile_id": tile_id,
                "tile_ref": tile_ref,
                "tile_bounds_m": [
                    bounds["min_x"],
                    bounds["min_y"],
                    bounds["max_x"],
                    bounds["max_y"],
                ],
                **_artifact_record(artifact),
            }
        )
    if seen_ids != set(terrain_by_ref):
        raise CompositionSourceError(
            "ground material/terrain tile identities differ"
        )
    records.sort(key=lambda item: str(item["tile_id"]))
    return index, records, evidence


def _resolve_placement_height_tiles(
    *,
    sources: Sequence[PlacementHeightTileSource],
    terrain: Sequence[_ResolvedTerrain],
    volume_root: Path,
) -> tuple[list[dict[str, object]], str]:
    if len(sources) != 400:
        raise CompositionSourceError(
            "placement height contract requires exactly 400 tiles"
        )
    terrain_by_ref = {tile.tile_ref: tile for tile in terrain}
    records: list[dict[str, object]] = []
    seen_refs: set[str] = set()
    seen_paths: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, PlacementHeightTileSource):
            raise CompositionSourceError(
                f"placement_height_tiles[{index}] has the wrong type"
            )
        tile_ref = _trimmed(
            source.tile_ref,
            label=f"placement_height_tiles[{index}].tile_ref",
        )
        if tile_ref in seen_refs or tile_ref not in terrain_by_ref:
            raise CompositionSourceError(
                f"placement height tile {tile_ref} is duplicated or unknown"
            )
        if source.format != "float32-le-row-major-south-to-north":
            raise CompositionSourceError(
                f"placement height tile {tile_ref} has unsupported format"
            )
        if (
            isinstance(source.width, bool)
            or not isinstance(source.width, int)
            or not 2 <= source.width <= 513
            or isinstance(source.height, bool)
            or not isinstance(source.height, int)
            or not 2 <= source.height <= 513
        ):
            raise CompositionSourceError(
                f"placement height tile {tile_ref} has invalid dimensions"
            )
        bounds = _normalized_bounds(source.local_bounds)
        terrain_bounds = terrain_by_ref[tile_ref].local_bounds
        if not all(
            math.isclose(
                bounds[key],
                terrain_bounds[key],
                abs_tol=0.01,
                rel_tol=0.0,
            )
            for key in ("min_x", "min_y", "max_x", "max_y")
        ):
            raise CompositionSourceError(
                f"placement height tile {tile_ref} bounds differ from terrain"
            )
        x_coordinates = tuple(
            _finite(
                value,
                label=f"placement height tile {tile_ref}.x_coordinates",
            )
            for value in source.x_coordinates
        )
        y_coordinates = tuple(
            _finite(
                value,
                label=f"placement height tile {tile_ref}.y_coordinates",
            )
            for value in source.y_coordinates
        )
        if (
            len(x_coordinates) != source.width
            or len(y_coordinates) != source.height
            or any(
                second <= first
                for first, second in zip(
                    x_coordinates, x_coordinates[1:]
                )
            )
            or any(
                second <= first
                for first, second in zip(
                    y_coordinates, y_coordinates[1:]
                )
            )
            or not all(
                math.isclose(actual, expected, abs_tol=0.01, rel_tol=0.0)
                for actual, expected in (
                    (x_coordinates[0], bounds["min_x"]),
                    (x_coordinates[-1], bounds["max_x"]),
                    (y_coordinates[0], bounds["min_y"]),
                    (y_coordinates[-1], bounds["max_y"]),
                )
            )
        ):
            raise CompositionSourceError(
                f"placement height tile {tile_ref} axes are invalid"
            )
        artifact = _resolve_artifact(
            source.artifact,
            volume_root=volume_root,
            label=f"placement height tile {tile_ref}",
        )
        if (
            artifact.physical_path.suffix.lower() != ".f32"
            or artifact.physical_path.stat().st_size
            != source.width * source.height * 4
            or artifact.portable_path in seen_paths
        ):
            raise CompositionSourceError(
                f"placement height tile {tile_ref} payload shape/path is invalid"
            )
        seen_refs.add(tile_ref)
        seen_paths.add(artifact.portable_path)
        records.append(
            {
                "tile_ref": tile_ref,
                "local_bounds": bounds,
                "path": artifact.portable_path,
                "sha256": artifact.sha256,
                "format": source.format,
                "width": source.width,
                "height": source.height,
                "x_coordinates": list(x_coordinates),
                "y_coordinates": list(y_coordinates),
            }
        )
    if seen_refs != set(terrain_by_ref):
        raise CompositionSourceError(
            "placement height/terrain tile identities differ"
        )
    records.sort(key=lambda item: str(item["tile_ref"]))
    fingerprint = hashlib.sha256(
        json.dumps(
            records,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return records, fingerprint


def _write_height_field(
    *,
    physical_path: Path,
    final_path: Path,
    volume_root: Path,
    source: HeightFieldSource,
    bounds: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, float | int]]:
    origin_x = _finite(source.origin_x, label="height_field.origin_x")
    origin_y = _finite(source.origin_y, label="height_field.origin_y")
    spacing = _positive(source.spacing_m, label="height_field.spacing_m")
    physical_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    width: int | None = None
    minimum_elevation = math.inf
    maximum_elevation = -math.inf
    with physical_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            '{"origin_x":'
            + json.dumps(origin_x, allow_nan=False)
            + ',"origin_y":'
            + json.dumps(origin_y, allow_nan=False)
            + ',"samples":['
        )
        for row_index, raw_row in enumerate(source.samples):
            if row_index >= MAX_HEIGHT_FIELD_SIDE:
                raise CompositionSourceError(
                    "height field exceeds the 1025-row portable bound"
                )
            if isinstance(raw_row, (str, bytes)) or not isinstance(
                raw_row, Sequence
            ):
                raise CompositionSourceError(
                    f"height_field.samples[{row_index}] must be a numeric row"
                )
            if width is None:
                width = len(raw_row)
                if width < 2:
                    raise CompositionSourceError(
                        "height field needs at least two columns"
                    )
                if width > MAX_HEIGHT_FIELD_SIDE:
                    raise CompositionSourceError(
                        "height field exceeds the 1025-column portable bound"
                    )
            elif len(raw_row) != width:
                raise CompositionSourceError(
                    "height field rows must have a constant width"
                )
            normalized = [
                _finite(
                    value,
                    label=f"height_field.samples[{row_index}][{column_index}]",
                )
                for column_index, value in enumerate(raw_row)
            ]
            minimum_elevation = min(minimum_elevation, min(normalized))
            maximum_elevation = max(maximum_elevation, max(normalized))
            if row_count:
                stream.write(",")
            stream.write(
                json.dumps(
                    normalized,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
            row_count += 1
        if row_count < 2 or width is None:
            raise CompositionSourceError("height field needs at least two rows")
        if row_count * width > MAX_HEIGHT_FIELD_SAMPLES:
            raise CompositionSourceError(
                "height field exceeds the portable sample-count bound"
            )
        stream.write(
            '],"spacing_m":'
            + json.dumps(spacing, allow_nan=False)
            + "}\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    max_x = origin_x + (width - 1) * spacing
    max_y = origin_y + (row_count - 1) * spacing
    epsilon = max(1.0e-6, spacing * 1.0e-9)
    if (
        origin_x > bounds["min_x"] + epsilon
        or origin_y > bounds["min_y"] + epsilon
        or max_x < bounds["max_x"] - epsilon
        or max_y < bounds["max_y"] - epsilon
    ):
        raise CompositionSourceError(
            "bounded height field must cover the complete scene bounds"
        )
    portable = final_path.relative_to(volume_root).as_posix()
    artifact = {
        "path": portable,
        "sha256": _sha256(physical_path),
    }
    metrics: dict[str, float | int] = {
        "width": width,
        "height": row_count,
        "sample_count": row_count * width,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "spacing_m": spacing,
        "max_x": max_x,
        "max_y": max_y,
        "minimum_elevation_m": minimum_elevation,
        "maximum_elevation_m": maximum_elevation,
    }
    return artifact, metrics


def _normalize_asset_library(
    *,
    sources: Iterable[AssetSource],
    volume_root: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    library: dict[str, Any] = {}
    category_by_key: dict[str, str] = {}
    evidence: dict[str, Any] = {}
    categories: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, AssetSource):
            raise CompositionSourceError(
                f"asset_library[{index}] must be an AssetSource"
            )
        key = _trimmed(source.key, label=f"asset_library[{index}].key")
        if key in library:
            raise CompositionSourceError(f"asset key repeats: {key}")
        category = source.category
        if category not in OBJECT_CATEGORIES:
            raise CompositionSourceError(
                f"asset {key} category must be trees or buildings"
            )
        family = _trimmed(source.family, label=f"asset {key}.family")
        lineage = _trimmed(
            source.lod_lineage, label=f"asset {key}.lod_lineage"
        )
        if source.simready_validation_state != "SIMREADY_VALIDATED":
            raise CompositionSourceError(
                f"asset {key} lacks SIMREADY_VALIDATED evidence"
            )
        if set(source.lods) != set(LOD_LEVELS):
            raise CompositionSourceError(
                f"asset {key} requires HERO, MID and FAR LODs"
            )
        if set(source.grounding_offsets_m) != set(LOD_LEVELS):
            raise CompositionSourceError(
                f"asset {key} requires HERO, MID and FAR grounding offsets"
            )
        resolved_lods = {
            level: _resolve_artifact(
                source.lods[level],
                volume_root=volume_root,
                label=f"asset {key}.{level}",
                require_usd=True,
                require_prim=True,
            )
            for level in LOD_LEVELS
        }
        if len({item.portable_path for item in resolved_lods.values()}) != 3:
            raise CompositionSourceError(
                f"asset {key} HERO, MID and FAR must be distinct files"
            )
        validation = _resolve_artifact(
            source.simready_validation_evidence,
            volume_root=volume_root,
            label=f"asset {key} SimReady validation evidence",
        )
        offsets = {
            level: _finite(
                source.grounding_offsets_m[level],
                label=f"asset {key}.grounding_offsets_m.{level}",
            )
            for level in LOD_LEVELS
        }
        library[key] = {
            "category": category,
            "family": family,
            "lods": {
                level: _artifact_record(resolved_lods[level])
                for level in LOD_LEVELS
            },
            "simready_validation": {
                "state": "SIMREADY_VALIDATED",
                "lod_lineage": lineage,
                "grounding_offsets_m": offsets,
                "evidence": _artifact_record(validation),
            },
        }
        evidence[key] = _artifact_record(validation)
        category_by_key[key] = category
        categories.add(category)
    if categories != set(OBJECT_CATEGORIES):
        raise CompositionSourceError(
            "asset library must contain real tree and building assets"
        )
    return library, category_by_key, evidence


def _normalize_water_materials(
    *,
    source: WaterMaterialSource,
    volume_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(source, WaterMaterialSource):
        raise CompositionSourceError(
            "water_material_lods must be a WaterMaterialSource"
        )
    if source.pbr_validation_state != "PBR_VALIDATED":
        raise CompositionSourceError(
            "water material lacks PBR_VALIDATED evidence"
        )
    if set(source.lods) != set(LOD_LEVELS):
        raise CompositionSourceError(
            "water material requires HERO, MID and FAR"
        )
    validation = _resolve_artifact(
        source.pbr_validation_evidence,
        volume_root=volume_root,
        label="water material PBR validation evidence",
    )
    materials = {
        level: _artifact_record(
            _resolve_artifact(
                source.lods[level],
                volume_root=volume_root,
                label=f"water material {level}",
                require_usd=True,
                require_prim=True,
            )
        )
        for level in LOD_LEVELS
    }
    return materials, _artifact_record(validation)


def _reserve_identity(
    database: sqlite3.Connection,
    *,
    stable_id: str,
    numeric_id: int | None,
    category: str,
) -> None:
    stable_owner = database.execute(
        "SELECT category FROM identities WHERE stable_id = ?", (stable_id,)
    ).fetchone()
    if stable_owner is not None:
        raise CompositionSourceError(
            f"stable ID {stable_id} repeats across "
            f"{stable_owner[0]} and {category}"
        )
    if numeric_id is not None:
        numeric_owner = database.execute(
            "SELECT stable_id, category FROM identities WHERE numeric_id = ?",
            (numeric_id,),
        ).fetchone()
        if numeric_owner is not None:
            raise CompositionSourceError(
                f"numeric ID {numeric_id} repeats for {numeric_owner[0]} "
                f"and {stable_id}"
            )
    database.execute(
        "INSERT INTO identities(stable_id, numeric_id, category) VALUES (?, ?, ?)",
        (stable_id, numeric_id, category),
    )


def _vector(
    value: object,
    *,
    size: int,
    label: str,
) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise CompositionSourceError(f"{label} must contain {size} coordinates")
    try:
        actual_size = len(value)  # type: ignore[arg-type]
        components = [value[index] for index in range(actual_size)]  # type: ignore[index]
    except (TypeError, KeyError, IndexError) as error:
        raise CompositionSourceError(
            f"{label} must contain {size} coordinates"
        ) from error
    if actual_size != size:
        raise CompositionSourceError(f"{label} must contain {size} coordinates")
    return [
        _finite(component, label=f"{label}[{index}]")
        for index, component in enumerate(components)
    ]


def _normalize_object(
    raw: Mapping[str, object],
    *,
    category: str,
    line_number: int,
    category_by_asset: Mapping[str, str],
    database: sqlite3.Connection,
    bounds: Mapping[str, float],
) -> dict[str, Any]:
    label = f"{category}[{line_number}]"
    if not isinstance(raw, Mapping):
        raise CompositionSourceError(f"{label} must be an object")
    stable_id = _trimmed(raw.get("stable_id"), label=f"{label}.stable_id")
    numeric_id = _positive_id(raw.get("numeric_id"), label=f"{label}.numeric_id")
    asset_key = _trimmed(raw.get("asset_key"), label=f"{label}.asset_key")
    if category_by_asset.get(asset_key) != category:
        raise CompositionSourceError(
            f"{label} references missing or wrong-category asset {asset_key}"
        )
    position = _vector(raw.get("position"), size=3, label=f"{label}.position")
    if not (
        bounds["min_x"] <= position[0] <= bounds["max_x"]
        and bounds["min_y"] <= position[1] <= bounds["max_y"]
    ):
        raise CompositionSourceError(
            f"{label}.position is outside root-local scene bounds"
        )
    heading = _finite(
        raw.get("heading_degrees"), label=f"{label}.heading_degrees"
    )
    scale = _positive(raw.get("uniform_scale"), label=f"{label}.uniform_scale")
    radius = _positive(
        raw.get("footprint_radius_m"),
        label=f"{label}.footprint_radius_m",
    )
    group_id = raw.get("group_id", "")
    if not isinstance(group_id, str) or group_id != group_id.strip():
        raise CompositionSourceError(f"{label}.group_id must be a trimmed string")
    _reserve_identity(
        database,
        stable_id=stable_id,
        numeric_id=numeric_id,
        category=category,
    )
    return {
        "stable_id": stable_id,
        "numeric_id": numeric_id,
        "asset_key": asset_key,
        "position": position,
        "heading_degrees": heading,
        "uniform_scale": scale,
        "footprint_radius_m": radius,
        "group_id": group_id,
    }


def _write_object_stream(
    *,
    physical_path: Path,
    final_path: Path,
    volume_root: Path,
    category: str,
    records: Iterable[Mapping[str, object]],
    category_by_asset: Mapping[str, str],
    database: sqlite3.Connection,
    bounds: Mapping[str, float],
) -> dict[str, Any]:
    physical_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with physical_path.open("w", encoding="utf-8", newline="\n") as stream:
        for line_number, raw in enumerate(records, start=1):
            normalized = _normalize_object(
                raw,
                category=category,
                line_number=line_number,
                category_by_asset=category_by_asset,
                database=database,
                bounds=bounds,
            )
            stream.write(
                json.dumps(
                    normalized,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
        if count < 1:
            raise CompositionSourceError(f"{category} stream must not be empty")
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "path": final_path.relative_to(volume_root).as_posix(),
        "sha256": _sha256(physical_path),
        "count": count,
        "format": "jsonl",
    }


def _reference_items(reference_list: object) -> list[object]:
    """Return direct Sdf.Reference items across supported OpenUSD bindings."""

    result: list[object] = []
    for attribute in (
        "explicitItems",
        "prependedItems",
        "appendedItems",
        "addedItems",
    ):
        values = getattr(reference_list, attribute, ())
        if values:
            result.extend(values)
    if not result:
        method = getattr(reference_list, "GetAddedOrExplicitItems", None)
        if callable(method):
            result.extend(method())
    return result


def _prototype_asset_path(
    prototype: object,
    *,
    volume_root: Path,
    label: str,
) -> Path:
    """Resolve the one direct external reference authored on a prototype."""

    candidates: set[Path] = set()
    for prim_spec in prototype.GetPrimStack():
        reference_list = getattr(prim_spec, "referenceList", None)
        if reference_list is None:
            continue
        layer = getattr(prim_spec, "layer", None)
        layer_path_raw = (
            getattr(layer, "realPath", "")
            or getattr(layer, "identifier", "")
        )
        if not layer_path_raw:
            continue
        layer_path = Path(str(layer_path_raw)).resolve()
        for reference in _reference_items(reference_list):
            asset_path = str(getattr(reference, "assetPath", ""))
            if not asset_path:
                continue
            raw = Path(asset_path.replace("\\", "/"))
            resolved = (
                raw.resolve()
                if raw.is_absolute()
                else (layer_path.parent / raw).resolve()
            )
            if not _is_below(volume_root, resolved) or not resolved.is_file():
                raise CompositionSourceError(
                    f"{label} prototype reference escapes/is absent from the "
                    "persistent volume"
                )
            candidates.add(resolved)
    if len(candidates) != 1:
        raise CompositionSourceError(
            f"{label} prototype must expose exactly one direct external USD asset"
        )
    return next(iter(candidates))


def _primvar_values(
    prim: object,
    *,
    name: str,
    count: int,
    label: str,
) -> list[object]:
    attribute = prim.GetAttribute(f"primvars:{name}")
    if not attribute or not attribute.IsValid() or not attribute.HasAuthoredValue():
        raise CompositionSourceError(
            f"{label} lacks required native primvar primvars:{name}"
        )
    values = list(attribute.Get() or [])
    if len(values) != count:
        raise CompositionSourceError(
            f"{label} primvars:{name} count does not match native instance count"
        )
    return values


def _heading_from_quaternion(value: object, *, label: str) -> float:
    try:
        real = float(value.GetReal())
        imaginary = value.GetImaginary()
        x = float(imaginary[0])
        y = float(imaginary[1])
        z = float(imaginary[2])
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise CompositionSourceError(f"{label} is not a native quaternion") from error
    components = (real, x, y, z)
    if any(not math.isfinite(component) for component in components):
        raise CompositionSourceError(f"{label} contains non-finite values")
    norm = math.sqrt(sum(component * component for component in components))
    if norm <= 1.0e-12:
        raise CompositionSourceError(f"{label} has zero quaternion norm")
    real, x, y, z = (component / norm for component in components)
    # Z-up yaw.  Native placement is uniform-scale plus heading only; pitch or
    # roll would be information the composition schema cannot preserve.
    pitch_roll_energy = abs(2.0 * (real * x + y * z)) + abs(
        2.0 * (real * y - z * x)
    )
    if pitch_roll_energy > 1.0e-4:
        raise CompositionSourceError(
            f"{label} contains pitch/roll unsupported by the native composition"
        )
    yaw = math.atan2(
        2.0 * (real * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return math.degrees(yaw)


def _detail_payload_artifacts(
    *,
    build_ref: _ResolvedArtifact,
    volume_root: Path,
) -> tuple[_ResolvedArtifact, ...]:
    build = _read_json(build_ref.physical_path, label="native build receipt")
    records = build.get("detail_payloads")
    if not isinstance(records, list) or len(records) != 400:
        raise CompositionSourceError(
            "lazy extraction requires exactly 400 HERO detail payloads"
        )
    result: list[_ResolvedArtifact] = []
    if build_ref.physical_path.parent.name != "build":
        raise CompositionSourceError(
            "native build receipt must live in the native zone build directory"
        )
    zone_root = build_ref.physical_path.parent.parent
    for index, record in enumerate(records):
        path, sha = _receipt_artifact_tuple(
            record, label=f"native build receipt detail_payloads[{index}]"
        )
        resolved_source = (zone_root / path).resolve()
        if not _is_below(volume_root, resolved_source):
            raise CompositionSourceError(
                f"HERO detail payload {index} escapes the persistent volume"
            )
        result.append(
            _resolve_artifact(
                ArtifactSource(path=resolved_source, expected_sha256=sha),
                volume_root=volume_root,
                label=f"HERO detail payload {index}",
                require_usd=True,
            )
        )
        if result[-1].physical_path.stat().st_size > MAX_DETAIL_PAYLOAD_BYTES:
            raise CompositionSourceError(
                f"HERO detail payload {index} exceeds the 512 MiB "
                "tile-streaming bound"
            )
    if len({item.portable_path for item in result}) != 400:
        raise CompositionSourceError("HERO detail payload paths must be unique")
    return tuple(result)


def _normalize_extraction_contract(
    source: DetailPayloadExtractionSource,
) -> tuple[str, float, float, float]:
    if not isinstance(source, DetailPayloadExtractionSource):
        raise CompositionSourceError(
            "native_detail_extraction must be a DetailPayloadExtractionSource"
        )
    if source.coordinate_space not in {
        "root_local_xy_ign69_z",
        "epsg2154_xy_ign69_z",
    }:
        raise CompositionSourceError(
            "detail extraction coordinate_space must explicitly be "
            "root_local_xy_ign69_z or epsg2154_xy_ign69_z"
        )
    origin = _vector(
        source.root_origin_epsg2154,
        size=2,
        label="native_detail_extraction.root_origin_epsg2154",
    )
    for label, name in (
        ("stable ID", source.stable_id_primvar),
        ("footprint radius", source.footprint_radius_primvar),
        ("group ID", source.group_id_primvar),
    ):
        _trimmed(name, label=f"native_detail_extraction {label} primvar")
    tolerance = _finite(
        source.tile_bounds_tolerance_m,
        label="native_detail_extraction.tile_bounds_tolerance_m",
    )
    if tolerance < 0.0 or tolerance > 100.0:
        raise CompositionSourceError(
            "detail extraction tile-bounds tolerance must be in [0,100] metres"
        )
    return source.coordinate_space, origin[0], origin[1], tolerance


def _write_extracted_object_streams(
    *,
    staging: Path,
    output: Path,
    volume_root: Path,
    build_ref: _ResolvedArtifact,
    source: DetailPayloadExtractionSource,
    assets: Mapping[str, Mapping[str, Any]],
    category_by_asset: Mapping[str, str],
    database: sqlite3.Connection,
    bounds: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stream two object inventories from the 400 accepted HERO USD layers."""

    try:
        from pxr import Usd, UsdGeom  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError) as error:
        raise CompositionSourceError(
            "lazy HERO detail extraction requires native Kit/Isaac pxr"
        ) from error
    coordinate_space, origin_x, origin_y, tolerance = (
        _normalize_extraction_contract(source)
    )
    asset_by_path = {
        (volume_root / record["lods"]["HERO"]["path"]).resolve(): key
        for key, record in assets.items()
    }
    payloads = _detail_payload_artifacts(
        build_ref=build_ref,
        volume_root=volume_root,
    )
    physical_paths = {
        category: staging / f"{category}.jsonl"
        for category in OBJECT_CATEGORIES
    }
    streams = {
        category: physical_paths[category].open(
            "w", encoding="utf-8", newline="\n"
        )
        for category in OBJECT_CATEGORIES
    }
    counts = {category: 0 for category in OBJECT_CATEGORIES}
    try:
        for payload_index, payload in enumerate(payloads):
            stage = Usd.Stage.Open(str(payload.physical_path), load=Usd.Stage.LoadNone)
            if stage is None:
                raise CompositionSourceError(
                    f"HERO detail payload cannot be opened: {payload.portable_path}"
                )
            detail = stage.GetPrimAtPath("/Detail")
            if not detail or not detail.IsValid():
                raise CompositionSourceError(
                    f"HERO detail payload has no /Detail root: {payload.portable_path}"
                )
            if detail.GetCustomDataByKey("fireviewer:detail_level") != "HERO":
                raise CompositionSourceError(
                    f"detail payload is not HERO: {payload.portable_path}"
                )
            raw_bounds = detail.GetCustomDataByKey("fireviewer:epsg2154_bounds")
            if not isinstance(raw_bounds, str):
                raise CompositionSourceError(
                    f"detail payload has no EPSG:2154 bounds: {payload.portable_path}"
                )
            try:
                xmin, ymin, xmax, ymax = (
                    float(value) for value in raw_bounds.split(",")
                )
            except (TypeError, ValueError) as error:
                raise CompositionSourceError(
                    f"detail payload has malformed EPSG:2154 bounds: "
                    f"{payload.portable_path}"
                ) from error
            if xmax <= xmin or ymax <= ymin:
                raise CompositionSourceError(
                    f"detail payload has invalid EPSG:2154 bounds: "
                    f"{payload.portable_path}"
                )
            instancers = sorted(
                (
                    UsdGeom.PointInstancer(prim)
                    for prim in stage.TraverseAll()
                    if prim.IsA(UsdGeom.PointInstancer)
                    and (
                        str(prim.GetPath()).startswith("/Detail/Vegetation/")
                        or str(prim.GetPath()).startswith("/Detail/Buildings/")
                    )
                ),
                key=lambda value: str(value.GetPath()),
            )
            payload_instance_count = 0
            for instancer in instancers:
                prim = instancer.GetPrim()
                prim_path = str(prim.GetPath())
                category = (
                    "trees"
                    if prim_path.startswith("/Detail/Vegetation/")
                    else "buildings"
                )
                positions = list(instancer.GetPositionsAttr().Get() or [])
                proto_indices = list(instancer.GetProtoIndicesAttr().Get() or [])
                orientations = list(instancer.GetOrientationsAttr().Get() or [])
                scales = list(instancer.GetScalesAttr().Get() or [])
                numeric_ids = list(instancer.GetIdsAttr().Get() or [])
                count = len(positions)
                if count < 1:
                    continue
                payload_instance_count += count
                if payload_instance_count > MAX_INSTANCES_PER_DETAIL_PAYLOAD:
                    raise CompositionSourceError(
                        f"{payload.portable_path} exceeds the "
                        "100000-instance per-tile streaming bound"
                    )
                if any(
                    len(values) != count
                    for values in (
                        proto_indices,
                        orientations,
                        scales,
                        numeric_ids,
                    )
                ):
                    raise CompositionSourceError(
                        f"{payload.portable_path}:{prim_path} has incomplete "
                        "native instance arrays"
                    )
                stable_ids = _primvar_values(
                    prim,
                    name=source.stable_id_primvar,
                    count=count,
                    label=f"{payload.portable_path}:{prim_path}",
                )
                radii = _primvar_values(
                    prim,
                    name=source.footprint_radius_primvar,
                    count=count,
                    label=f"{payload.portable_path}:{prim_path}",
                )
                group_ids = _primvar_values(
                    prim,
                    name=source.group_id_primvar,
                    count=count,
                    label=f"{payload.portable_path}:{prim_path}",
                )
                prototype_targets = list(
                    instancer.GetPrototypesRel().GetTargets() or []
                )
                if not prototype_targets:
                    raise CompositionSourceError(
                        f"{payload.portable_path}:{prim_path} has no prototypes"
                    )
                prototype_assets: list[str] = []
                for prototype_index, target in enumerate(prototype_targets):
                    prototype = stage.GetPrimAtPath(target)
                    if not prototype or not prototype.IsValid():
                        raise CompositionSourceError(
                            f"{payload.portable_path}:{prim_path} prototype "
                            f"{prototype_index} is absent"
                        )
                    asset_path = _prototype_asset_path(
                        prototype,
                        volume_root=volume_root,
                        label=(
                            f"{payload.portable_path}:{prim_path} prototype "
                            f"{prototype_index}"
                        ),
                    )
                    asset_key = asset_by_path.get(asset_path)
                    if asset_key is None or category_by_asset.get(asset_key) != category:
                        raise CompositionSourceError(
                            f"{payload.portable_path}:{prim_path} prototype "
                            f"{prototype_index} is not in the accepted HERO "
                            f"{category} asset library"
                        )
                    prototype_assets.append(asset_key)
                for instance_index in range(count):
                    prototype_index = int(proto_indices[instance_index])
                    if not 0 <= prototype_index < len(prototype_assets):
                        raise CompositionSourceError(
                            f"{payload.portable_path}:{prim_path} instance "
                            f"{instance_index} has an invalid prototype index"
                        )
                    point = _vector(
                        positions[instance_index],
                        size=3,
                        label=(
                            f"{payload.portable_path}:{prim_path} instance "
                            f"{instance_index} position"
                        ),
                    )
                    if coordinate_space == "epsg2154_xy_ign69_z":
                        point[0] -= origin_x
                        point[1] -= origin_y
                    local_bounds = (
                        xmin - origin_x,
                        ymin - origin_y,
                        xmax - origin_x,
                        ymax - origin_y,
                    )
                    if not (
                        local_bounds[0] - tolerance
                        <= point[0]
                        <= local_bounds[2] + tolerance
                        and local_bounds[1] - tolerance
                        <= point[1]
                        <= local_bounds[3] + tolerance
                    ):
                        raise CompositionSourceError(
                            f"{payload.portable_path}:{prim_path} instance "
                            f"{instance_index} is outside its explicit root-local "
                            "tile bounds"
                        )
                    scale_value = _vector(
                        scales[instance_index],
                        size=3,
                        label=(
                            f"{payload.portable_path}:{prim_path} instance "
                            f"{instance_index} scale"
                        ),
                    )
                    if (
                        max(scale_value) - min(scale_value)
                        > max(1.0e-6, max(scale_value) * 1.0e-6)
                    ):
                        raise CompositionSourceError(
                            f"{payload.portable_path}:{prim_path} instance "
                            f"{instance_index} is not uniformly scaled"
                        )
                    raw = {
                        "stable_id": str(stable_ids[instance_index]),
                        "numeric_id": int(numeric_ids[instance_index]),
                        "asset_key": prototype_assets[prototype_index],
                        "position": point,
                        "heading_degrees": _heading_from_quaternion(
                            orientations[instance_index],
                            label=(
                                f"{payload.portable_path}:{prim_path} instance "
                                f"{instance_index} orientation"
                            ),
                        ),
                        "uniform_scale": scale_value[0],
                        "footprint_radius_m": float(radii[instance_index]),
                        "group_id": str(group_ids[instance_index]),
                    }
                    normalized = _normalize_object(
                        raw,
                        category=category,
                        line_number=counts[category] + 1,
                        category_by_asset=category_by_asset,
                        database=database,
                        bounds=bounds,
                    )
                    streams[category].write(
                        json.dumps(
                            normalized,
                            ensure_ascii=False,
                            allow_nan=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    counts[category] += 1
            # Drop all stage references before opening the next kilometre.
            stage = None
    finally:
        for stream in streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
    for category in OBJECT_CATEGORIES:
        if counts[category] < 1:
            raise CompositionSourceError(
                f"lazy HERO extraction found no {category} instances"
            )
    return tuple(
        {
            "path": (output / f"{category}.jsonl")
            .relative_to(volume_root)
            .as_posix(),
            "sha256": _sha256(physical_paths[category]),
            "count": counts[category],
            "format": "jsonl",
            "source": "native_hero_detail_payloads",
        }
        for category in OBJECT_CATEGORIES
    )  # type: ignore[return-value]


def _polygon(
    value: object,
    *,
    label: str,
) -> list[list[float]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CompositionSourceError(f"{label} must be a polygon coordinate array")
    points = [
        _vector(point, size=2, label=f"{label}[{index}]")
        for index, point in enumerate(value)
    ]
    if len(points) < 3:
        raise CompositionSourceError(f"{label} must have at least three vertices")
    area = sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(points, points[1:] + points[:1])
    )
    if abs(area) <= 1.0e-9:
        raise CompositionSourceError(f"{label} must have non-zero area")
    return points


def _normalize_bridge(
    raw: object,
    *,
    label: str,
    database: sqlite3.Connection,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise CompositionSourceError(f"{label} must be an object")
    stable_id = _trimmed(raw.get("stable_id"), label=f"{label}.stable_id")
    values = {
        key: _finite(raw.get(key), label=f"{label}.{key}")
        for key in (
            "start_fraction",
            "water_start_fraction",
            "water_end_fraction",
            "end_fraction",
            "minimum_deck_clearance_m",
        )
    }
    if not (
        0.0
        <= values["start_fraction"]
        < values["water_start_fraction"]
        <= values["water_end_fraction"]
        < values["end_fraction"]
        <= 1.0
    ):
        raise CompositionSourceError(f"{label} fractions are not ordered in [0,1]")
    if values["minimum_deck_clearance_m"] <= 0.0:
        raise CompositionSourceError(f"{label} deck clearance must be positive")
    _reserve_identity(
        database,
        stable_id=stable_id,
        numeric_id=None,
        category="bridge",
    )
    return {"stable_id": stable_id, **values}


def _normalize_routes(
    records: Iterable[Mapping[str, object]],
    *,
    database: sqlite3.Connection,
    bounds: Mapping[str, float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        label = f"routes[{index}]"
        if not isinstance(raw, Mapping):
            raise CompositionSourceError(f"{label} must be an object")
        stable_id = _trimmed(raw.get("stable_id"), label=f"{label}.stable_id")
        numeric_id = _positive_id(
            raw.get("numeric_id"), label=f"{label}.numeric_id"
        )
        family = _trimmed(raw.get("family"), label=f"{label}.family")
        surface_class = _trimmed(
            raw.get("surface_class", raw.get("material_key")),
            label=f"{label}.surface_class",
        )
        raw_points = raw.get("points")
        if isinstance(raw_points, (str, bytes)) or not isinstance(
            raw_points, Sequence
        ):
            raise CompositionSourceError(f"{label}.points must be an array")
        points = [
            _vector(point, size=3, label=f"{label}.points[{point_index}]")
            for point_index, point in enumerate(raw_points)
        ]
        if len(points) < 2 or all(
            math.hypot(
                second[0] - first[0],
                second[1] - first[1],
            )
            <= 1.0e-9
            for first, second in zip(points, points[1:])
        ):
            raise CompositionSourceError(
                f"{label} must contain a non-zero-length polyline"
            )
        if any(
            not (
                bounds["min_x"] <= point[0] <= bounds["max_x"]
                and bounds["min_y"] <= point[1] <= bounds["max_y"]
            )
            for point in points
        ):
            raise CompositionSourceError(
                f"{label}.points are not in root-local scene bounds"
            )
        raw_bridges = raw.get("bridge_spans", [])
        if isinstance(raw_bridges, (str, bytes)) or not isinstance(
            raw_bridges, Sequence
        ):
            raise CompositionSourceError(f"{label}.bridge_spans must be an array")
        bridges = [
            _normalize_bridge(
                bridge,
                label=f"{label}.bridge_spans[{bridge_index}]",
                database=database,
            )
            for bridge_index, bridge in enumerate(raw_bridges)
        ]
        ordered = sorted(bridges, key=lambda value: value["start_fraction"])
        if any(
            first["end_fraction"] > second["start_fraction"]
            for first, second in zip(ordered, ordered[1:])
        ):
            raise CompositionSourceError(f"{label} bridge spans overlap")
        _reserve_identity(
            database,
            stable_id=stable_id,
            numeric_id=numeric_id,
            category="route",
        )
        result.append(
            {
                "stable_id": stable_id,
                "numeric_id": numeric_id,
                "family": family,
                "surface_class": surface_class,
                "width_m": _positive(
                    raw.get("width_m"), label=f"{label}.width_m"
                ),
                "points": points,
                "bridge_spans": bridges,
            }
        )
    if not result:
        raise CompositionSourceError("routes must not be empty")
    return result


def _normalize_waters(
    records: Iterable[Mapping[str, object]],
    *,
    database: sqlite3.Connection,
    bounds: Mapping[str, float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        label = f"waters[{index}]"
        if not isinstance(raw, Mapping):
            raise CompositionSourceError(f"{label} must be an object")
        stable_id = _trimmed(raw.get("stable_id"), label=f"{label}.stable_id")
        family = _trimmed(raw.get("family"), label=f"{label}.family")
        kind = raw.get("kind")
        if kind not in {"standing", "watercourse"}:
            raise CompositionSourceError(
                f"{label}.kind must be standing or watercourse"
            )
        outline = _polygon(raw.get("outline"), label=f"{label}.outline")
        if any(
            not (
                bounds["min_x"] <= point[0] <= bounds["max_x"]
                and bounds["min_y"] <= point[1] <= bounds["max_y"]
            )
            for point in outline
        ):
            raise CompositionSourceError(
                f"{label}.outline is not in root-local scene bounds"
            )
        raw_centreline = raw.get("centreline", [])
        raw_profile = raw.get("surface_profile_m")
        if (
            isinstance(raw_centreline, (str, bytes))
            or not isinstance(raw_centreline, Sequence)
            or isinstance(raw_profile, (str, bytes))
            or not isinstance(raw_profile, Sequence)
        ):
            raise CompositionSourceError(
                f"{label} centreline/profile must be arrays"
            )
        centreline = [
            _vector(
                point,
                size=2,
                label=f"{label}.centreline[{point_index}]",
            )
            for point_index, point in enumerate(raw_centreline)
        ]
        if any(
            not (
                bounds["min_x"] <= point[0] <= bounds["max_x"]
                and bounds["min_y"] <= point[1] <= bounds["max_y"]
            )
            for point in centreline
        ):
            raise CompositionSourceError(
                f"{label}.centreline is not in root-local scene bounds"
            )
        profile = [
            _finite(value, label=f"{label}.surface_profile_m[{profile_index}]")
            for profile_index, value in enumerate(raw_profile)
        ]
        expected_profile = len(centreline) if kind == "watercourse" else 1
        if (
            (kind == "watercourse" and len(centreline) < 2)
            or (kind == "standing" and centreline)
            or len(profile) != expected_profile
        ):
            raise CompositionSourceError(
                f"{label} geometry/profile is inconsistent with {kind}"
            )
        _reserve_identity(
            database,
            stable_id=stable_id,
            numeric_id=None,
            category="water",
        )
        result.append(
            {
                "stable_id": stable_id,
                "family": family,
                "outline": outline,
                "kind": kind,
                "centreline": centreline,
                "surface_profile_m": profile,
            }
        )
    if not result:
        raise CompositionSourceError("waters must not be empty")
    return result


def _normalize_suitability(
    records: Iterable[Mapping[str, object]],
    *,
    tree_families: set[str],
    database: sqlite3.Connection,
    bounds: Mapping[str, float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    covered_families: set[str] = set()
    for index, raw in enumerate(records):
        label = f"suitability_zones[{index}]"
        if not isinstance(raw, Mapping):
            raise CompositionSourceError(f"{label} must be an object")
        stable_id = _trimmed(raw.get("stable_id"), label=f"{label}.stable_id")
        families = raw.get("tree_families")
        if isinstance(families, (str, bytes)) or not isinstance(
            families, Sequence
        ):
            raise CompositionSourceError(f"{label}.tree_families must be an array")
        normalized_families = sorted(
            {
                _trimmed(
                    family,
                    label=f"{label}.tree_families[{family_index}]",
                )
                for family_index, family in enumerate(families)
            }
        )
        unknown = set(normalized_families) - tree_families
        if unknown:
            raise CompositionSourceError(
                f"{label} references unknown tree families: "
                + ", ".join(sorted(unknown))
            )
        buildable = raw.get("buildable")
        if not isinstance(buildable, bool):
            raise CompositionSourceError(f"{label}.buildable must be boolean")
        _reserve_identity(
            database,
            stable_id=stable_id,
            numeric_id=None,
            category="suitability",
        )
        outline = _polygon(raw.get("outline"), label=f"{label}.outline")
        if any(
            not (
                bounds["min_x"] <= point[0] <= bounds["max_x"]
                and bounds["min_y"] <= point[1] <= bounds["max_y"]
            )
            for point in outline
        ):
            raise CompositionSourceError(
                f"{label}.outline is not in root-local scene bounds"
            )
        result.append(
            {
                "stable_id": stable_id,
                "outline": outline,
                "biome": _trimmed(raw.get("biome"), label=f"{label}.biome"),
                "soil": _trimmed(raw.get("soil"), label=f"{label}.soil"),
                "tree_families": normalized_families,
                "buildable": buildable,
            }
        )
        covered_families.update(normalized_families)
    if not result:
        raise CompositionSourceError("suitability_zones must not be empty")
    missing = tree_families - covered_families
    if missing:
        raise CompositionSourceError(
            "suitability zones do not support all native tree families: "
            + ", ".join(sorted(missing))
        )
    if not any(zone["buildable"] for zone in result):
        raise CompositionSourceError(
            "suitability zones contain no native buildable area"
        )
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _fixed_variant_constraints() -> dict[str, int | float]:
    """Mirror the native campaign's reviewed numeric placement contract.

    The consumer deliberately rejects an omitted or partial constraint set.
    Reading the defaults from the shared dataclass keeps the interchange
    contract synchronized when a reviewed constraint is added upstream,
    without copying a second set of magic numbers into this exporter.
    ``tree_suitability`` is derived by the consumer from the source-backed
    suitability zones and therefore is not serialized here.
    """

    defaults = VariantConstraints()
    return {
        field.name: getattr(defaults, field.name)
        for field in fields(defaults)
        if field.name != "tree_suitability"
    }


def _normalize_variant_constraints(
    raw: Mapping[str, object] | None,
) -> dict[str, int | float]:
    """Validate the complete numeric campaign contract before publication."""

    defaults = _fixed_variant_constraints()
    if raw is None:
        return defaults
    if not isinstance(raw, Mapping) or set(raw) != set(defaults):
        missing = sorted(set(defaults) - set(raw if isinstance(raw, Mapping) else ()))
        extra = sorted(set(raw) - set(defaults)) if isinstance(raw, Mapping) else []
        raise CompositionSourceError(
            "variant_constraints must define the exact fixed numeric contract "
            f"(missing={missing}, extra={extra})"
        )
    normalized: dict[str, int | float] = {}
    for key, default in defaults.items():
        value = raw[key]
        if isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise CompositionSourceError(
                    f"variant_constraints.{key} must be an integer"
                )
            normalized[key] = value
        else:
            normalized[key] = _finite(
                value, label=f"variant_constraints.{key}"
            )
    try:
        VariantConstraints(**normalized)
    except (TypeError, ValueError) as error:
        raise CompositionSourceError(
            f"variant_constraints are invalid: {error}"
        ) from error
    return normalized


def _route_topology_contract(
    routes: Sequence[Mapping[str, object]],
    *,
    tolerance_m: float,
) -> dict[str, object]:
    try:
        scene_routes = tuple(
            SceneRoute(
                stable_id=str(route["stable_id"]),
                family=str(route["family"]),
                points=tuple(
                    Vec3(
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                    )
                    for point in route["points"]  # type: ignore[union-attr]
                ),
                width_m=float(route["width_m"]),
            )
            for route in routes
        )
        component_count, membership_sha256 = route_topology(
            scene_routes, tolerance_m
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CompositionSourceError(
            f"normalized route topology is invalid: {error}"
        ) from error
    if component_count < 1:
        raise CompositionSourceError(
            "normalized route network has no connected component"
        )
    return {
        "algorithm": "segment-connectivity-components-v1",
        "tolerance_m": tolerance_m,
        "source_component_count": component_count,
        "source_membership_sha256": membership_sha256,
    }


def _validated_route_source_evidence(
    raw: Mapping[str, object],
    *,
    volume_root: Path,
    prepared_route_count: int,
    native_fragment_count: int,
) -> dict[str, object]:
    if (
        raw.get("route_geometry_authority")
        != "locked_continuous_bdtopo_lines"
        or raw.get("prepared_route_count") != prepared_route_count
        or raw.get("native_hero_fragment_proof_count")
        != native_fragment_count
        or raw.get("native_hero_receipt_fragment_count")
        != native_fragment_count
    ):
        raise CompositionSourceError(
            "continuous route source evidence has divergent route/fragment "
            "counts or authority"
        )
    artifacts = raw.get("locked_source_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CompositionSourceError(
            "continuous route source evidence has no locked BDTOPO artifacts"
        )
    normalized_artifacts: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for index, record in enumerate(artifacts):
        path, expected = _receipt_artifact_tuple(
            record, label=f"route_source.locked_source_artifacts[{index}]"
        )
        physical = (volume_root / path).resolve()
        if (
            not _is_below(volume_root, physical)
            or not physical.is_file()
            or physical.is_symlink()
            or path in seen_paths
            or _sha256(physical) != expected
        ):
            raise CompositionSourceError(
                f"route source artifact {path} is absent, repeated or stale"
            )
        seen_paths.add(path)
        normalized_artifacts.append({"path": path, "sha256": expected})
    overlap = raw.get("overlap_validation")
    maximum_overlap = (
        overlap.get("maximum_interroute_collinear_overlap_m")
        if isinstance(overlap, Mapping)
        else None
    )
    if (
        not isinstance(overlap, Mapping)
        or overlap.get("algorithm")
        != "spatial-collinear-interior-overlap-gate-v1"
        or overlap.get("maximum_allowed_interroute_collinear_overlap_m")
        != 0.01
        or isinstance(maximum_overlap, bool)
        or not isinstance(maximum_overlap, (int, float))
        or not math.isfinite(float(maximum_overlap))
        or not 0.0 <= float(maximum_overlap) <= 0.01
    ):
        raise CompositionSourceError(
            "continuous route source evidence does not prove zero interior "
            "overlap"
        )
    for field in (
        "source_feature_count",
        "source_line_count",
        "placement_height_peak_cached_tiles",
    ):
        value = raw.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise CompositionSourceError(
                f"route_source.{field} must be a non-negative integer"
            )
    if (
        int(raw["source_feature_count"]) < 1
        or int(raw["source_line_count"]) < 1
        or int(raw["placement_height_peak_cached_tiles"]) > 2
    ):
        raise CompositionSourceError(
            "continuous route evidence has empty sources or exceeded its "
            "two-tile height cache"
        )
    if raw.get("placement_height_cache_tile_limit") != 2:
        raise CompositionSourceError(
            "continuous route source must retain the two-tile height cache"
        )
    return {
        "route_geometry_authority": "locked_continuous_bdtopo_lines",
        "locked_source_artifacts": normalized_artifacts,
        "source_feature_count": int(raw["source_feature_count"]),
        "source_line_count": int(raw["source_line_count"]),
        "prepared_route_count": prepared_route_count,
        "placement_height_cache_tile_limit": 2,
        "placement_height_peak_cached_tiles": int(
            raw["placement_height_peak_cached_tiles"]
        ),
        "overlap_validation": dict(overlap),
        "native_hero_fragment_proof_count": native_fragment_count,
        "native_hero_receipt_fragment_count": native_fragment_count,
    }


def export_composition_source(
    *,
    volume_root: Path,
    output_root: Path,
    base_scene_id: str,
    coordinate_contract: str,
    epsg2154_origin: Sequence[float],
    native_artifacts: NativeArtifactsSource,
    bounds: Mapping[str, object],
    height_field: HeightFieldSource,
    placement_height_tiles: Sequence[PlacementHeightTileSource],
    ground_surface: GroundSurfaceSource,
    asset_library: Iterable[AssetSource],
    water_material_lods: WaterMaterialSource,
    trees: Iterable[Mapping[str, object]] | None,
    buildings: Iterable[Mapping[str, object]] | None,
    routes: Iterable[Mapping[str, object]],
    waters: Iterable[Mapping[str, object]],
    suitability_zones: Iterable[Mapping[str, object]],
    variant_constraints: Mapping[str, object] | None = None,
    native_detail_extraction: DetailPayloadExtractionSource | None = None,
    expected_placement_height_fingerprint: str | None = None,
    expected_route_topology: Mapping[str, object] | None = None,
    route_source_evidence: Mapping[str, object] | None = None,
    source_contract_artifact: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Publish one complete native composition interchange atomically.

    Object iterables are consumed exactly once.  The caller must pass the
    stable IDs, numeric IDs, selected asset keys and transforms recorded during
    native authoring; the exporter intentionally has no fallback generation.
    ``output_root`` is a new directory and is never overwritten.
    """

    volume = volume_root.expanduser().resolve()
    if not volume.is_dir():
        raise CompositionSourceError(
            f"persistent volume root is absent: {volume}"
        )
    output = output_root.expanduser().resolve()
    if not _is_below(volume, output) or output == volume:
        raise CompositionSourceError(
            "composition output must be a dedicated directory below the "
            "persistent volume"
        )
    if output.exists():
        raise CompositionSourceError(
            f"refusing to overwrite composition output: {output}"
        )
    scene_id = _trimmed(base_scene_id, label="base_scene_id")
    if coordinate_contract != ROOT_LOCAL_COORDINATE_CONTRACT:
        raise CompositionSourceError(
            "coordinate_contract must explicitly use the root-local X/Y and "
            "IGN69 Z convention"
        )
    normalized_bounds = _normalized_bounds(bounds)
    normalized_origin = _vector(
        epsg2154_origin, size=3, label="epsg2154_origin"
    )
    normalized_source_contract: dict[str, str] | None = None
    if source_contract_artifact is not None:
        contract_path, contract_sha = _receipt_artifact_tuple(
            source_contract_artifact,
            label="source_contract_artifact",
        )
        physical_contract = (volume / contract_path).resolve()
        if (
            not _is_below(volume, physical_contract)
            or not physical_contract.is_file()
            or physical_contract.is_symlink()
            or physical_contract.stat().st_size
            > MAX_COMPOSITION_CONTRACT_BYTES
            or _sha256(physical_contract) != contract_sha
        ):
            raise CompositionSourceError(
                "source composition contract is absent, unsafe or stale"
            )
        normalized_source_contract = {
            "path": contract_path,
            "sha256": contract_sha,
        }
    if not isinstance(native_artifacts, NativeArtifactsSource):
        raise CompositionSourceError(
            "native_artifacts must be a NativeArtifactsSource"
        )
    terrain_sources = tuple(native_artifacts.terrain_payloads)
    water_sources = tuple(native_artifacts.water_payloads)
    if len(terrain_sources) != 400:
        raise CompositionSourceError(
            "native composition requires exactly 400 terrain payloads"
        )
    if not water_sources:
        raise CompositionSourceError(
            "native composition requires at least one isolated water payload"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    if staging.exists():
        raise CompositionSourceError(f"staging collision: {staging}")
    staging.mkdir()
    database: sqlite3.Connection | None = None
    try:
        build_ref = _resolve_artifact(
            native_artifacts.native_build_receipt,
            volume_root=volume,
            label="native_build_receipt",
        )
        auto_ref = _resolve_artifact(
            native_artifacts.scene_auto_validation,
            volume_root=volume,
            label="scene_auto_validation",
        )
        root_ref = _resolve_artifact(
            native_artifacts.root_usd,
            volume_root=volume,
            label="root_usd",
            require_usd=True,
            require_prim=True,
        )
        terrain = _resolve_terrain_payloads(
            terrain_sources,
            volume_root=volume,
            scene_bounds=normalized_bounds,
            epsg2154_origin=normalized_origin,
        )
        water_payloads = tuple(
            _resolve_artifact(
                source,
                volume_root=volume,
                label=f"water_payloads[{index}]",
                require_usd=True,
                require_prim=True,
            )
            for index, source in enumerate(water_sources)
        )
        if len({item.portable_path for item in water_payloads}) != len(
            water_payloads
        ):
            raise CompositionSourceError("water payload paths must be unique")
        water_evidence = _validate_water_evidence(
            native=native_artifacts,
            volume_root=volume,
            water_payloads=water_payloads,
        )
        layer_counts = _validate_native_receipts(
            base_scene_id=scene_id,
            native=native_artifacts,
            build_ref=build_ref,
            auto_ref=auto_ref,
            root_ref=root_ref,
            terrain=terrain,
            volume_root=volume,
        )
        ground_ref, ground_tiles, ground_evidence = _resolve_ground_surface(
            source=ground_surface,
            terrain=terrain,
            volume_root=volume,
        )
        placement_records, placement_fingerprint = (
            _resolve_placement_height_tiles(
                sources=placement_height_tiles,
                terrain=terrain,
                volume_root=volume,
            )
        )
        if expected_placement_height_fingerprint is not None and (
            not _SHA256.fullmatch(expected_placement_height_fingerprint)
            or expected_placement_height_fingerprint
            != placement_fingerprint
        ):
            raise CompositionSourceError(
                "placement height fingerprint is absent or stale"
            )
        assets, category_by_asset, asset_evidence = _normalize_asset_library(
            sources=asset_library,
            volume_root=volume,
        )
        water_materials, water_material_evidence = _normalize_water_materials(
            source=water_material_lods,
            volume_root=volume,
        )
        height_final = output / "height-field.json"
        height_record, height_metrics = _write_height_field(
            physical_path=staging / "height-field.json",
            final_path=height_final,
            volume_root=volume,
            source=height_field,
            bounds=normalized_bounds,
        )
        database_path = staging / ".identity-index.sqlite3"
        database = sqlite3.connect(database_path)
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute(
            "CREATE TABLE identities("
            "stable_id TEXT PRIMARY KEY, "
            "numeric_id INTEGER UNIQUE, "
            "category TEXT NOT NULL)"
        )
        if native_detail_extraction is None:
            if trees is None or buildings is None:
                raise CompositionSourceError(
                    "explicit tree/building streams are required unless lazy "
                    "native HERO detail extraction is configured"
                )
            trees_record = _write_object_stream(
                physical_path=staging / "trees.jsonl",
                final_path=output / "trees.jsonl",
                volume_root=volume,
                category="trees",
                records=trees,
                category_by_asset=category_by_asset,
                database=database,
                bounds=normalized_bounds,
            )
            buildings_record = _write_object_stream(
                physical_path=staging / "buildings.jsonl",
                final_path=output / "buildings.jsonl",
                volume_root=volume,
                category="buildings",
                records=buildings,
                category_by_asset=category_by_asset,
                database=database,
                bounds=normalized_bounds,
            )
            object_source = "native_build_in_memory_records"
        else:
            if trees is not None or buildings is not None:
                raise CompositionSourceError(
                    "choose either explicit native object streams or lazy HERO "
                    "detail extraction, never both"
                )
            trees_record, buildings_record = _write_extracted_object_streams(
                staging=staging,
                output=output,
                volume_root=volume,
                build_ref=build_ref,
                source=native_detail_extraction,
                assets=assets,
                category_by_asset=category_by_asset,
                database=database,
                bounds=normalized_bounds,
            )
            object_source = "native_hero_detail_payloads"
        normalized_routes = _normalize_routes(
            routes,
            database=database,
            bounds=normalized_bounds,
        )
        normalized_waters = _normalize_waters(
            waters,
            database=database,
            bounds=normalized_bounds,
        )
        tree_families = {
            record["family"]
            for record in assets.values()
            if record["category"] == "trees"
        }
        normalized_zones = _normalize_suitability(
            suitability_zones,
            tree_families=tree_families,
            database=database,
            bounds=normalized_bounds,
        )
        normalized_constraints = _normalize_variant_constraints(
            variant_constraints
        )
        topology = _route_topology_contract(
            normalized_routes,
            tolerance_m=float(
                normalized_constraints["road_connectivity_tolerance_m"]
            ),
        )
        if expected_route_topology is not None and dict(
            expected_route_topology
        ) != topology:
            raise CompositionSourceError(
                "prepared route topology fingerprint is stale"
            )
        source_component_count = int(
            topology["source_component_count"]
        )
        if variant_constraints is None:
            normalized_constraints["maximum_road_components"] = (
                source_component_count
            )
        elif (
            normalized_constraints["maximum_road_components"]
            != source_component_count
        ):
            raise CompositionSourceError(
                "variant_constraints.maximum_road_components must equal the "
                "exact source route component count"
            )
        if trees_record["count"] != layer_counts["vegetation"]:
            raise CompositionSourceError(
                "tree stream count does not equal the native HERO vegetation count"
            )
        if buildings_record["count"] != layer_counts["buildings"]:
            raise CompositionSourceError(
                "building stream count does not equal the native HERO building count"
            )
        normalized_route_source: dict[str, object] | None = None
        if route_source_evidence is None:
            if len(normalized_routes) != layer_counts["roads"]:
                raise CompositionSourceError(
                    "route count does not equal the native HERO road count"
                )
        else:
            normalized_route_source = _validated_route_source_evidence(
                route_source_evidence,
                volume_root=volume,
                prepared_route_count=len(normalized_routes),
                native_fragment_count=layer_counts["roads"],
            )
        if len(normalized_waters) != layer_counts["hydrology"]:
            raise CompositionSourceError(
                "water feature count does not equal the native HERO "
                "hydrology count"
            )
        database.commit()
        database.close()
        database = None
        database_path.unlink()
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "state": "COMPOSITION_SOURCE_READY",
            "base_scene_id": scene_id,
            "source_contract": normalized_source_contract,
            "coordinate_contract": ROOT_LOCAL_COORDINATE_CONTRACT,
            "epsg2154_origin": normalized_origin,
            "native_build_receipt": _artifact_record(build_ref),
            "scene_auto_validation": _artifact_record(auto_ref),
            "root_usd": _artifact_record(root_ref),
            "terrain_payloads": [
                {
                    **_artifact_record(
                        tile.artifact,
                        isolated_content_roles=("terrain",),
                    ),
                    "tile_ref": tile.tile_ref,
                    "local_bounds": dict(tile.local_bounds),
                    "epsg2154_bounds": dict(tile.epsg2154_bounds),
                    "instance_namespace": tile.instance_namespace,
                    "terrain_lods": list(tile.terrain_lods),
                    "collision_lods": list(tile.collision_lods),
                }
                for tile in terrain
            ],
            "water_payloads": [
                _artifact_record(item, isolated_content_roles=("water",))
                for item in water_payloads
            ],
            "ground_material": {
                **_artifact_record(
                    ground_ref,
                    isolated_content_roles=("object_free_pbr_ground",),
                ),
                "topology": (
                    "payload_tiled_materials_shared_pbr_library"
                ),
                "tile_material_payloads": ground_tiles,
            },
            "ground_surface": {
                "kind": "object_free_pbr",
                "content_fingerprint": ground_ref.sha256,
                "removed_object_classes": [],
            },
            "height_field": height_record,
            "height_field_contract": {
                **height_metrics,
                "role": "global_preview_and_water_validation_only",
                "maximum_side": MAX_HEIGHT_FIELD_SIDE,
                "maximum_samples": MAX_HEIGHT_FIELD_SAMPLES,
                "resampled_by_exporter": False,
            },
            "placement_height_tiles": placement_records,
            "placement_height_fingerprint": placement_fingerprint,
            "bounds": normalized_bounds,
            "asset_library": assets,
            "road_visual_contract": {
                "visible_representation": (
                    "orthophoto_derived_terrain_material"
                ),
                "geometry_authoring": "disabled",
                "asset_dependencies": [],
                "route_vectors_retained_for": [
                    "topology",
                    "actor_placement",
                    "annotations",
                    "composition_constraints",
                ],
            },
            "water_material_lods": water_materials,
            "trees": trees_record,
            "buildings": buildings_record,
            "routes": normalized_routes,
            "route_topology": topology,
            "route_source": normalized_route_source,
            "waters": normalized_waters,
            "suitability_zones": normalized_zones,
            "variant_constraints": normalized_constraints,
            "validation_evidence": {
                "ground_surface": _artifact_record(ground_evidence),
                "water": _artifact_record(water_evidence),
                "assets": asset_evidence,
                "water_materials": water_material_evidence,
            },
            "identity_contract": {
                "authority": "native_zone_build",
                "object_source": object_source,
                "stable_ids_preserved": True,
                "numeric_ids_preserved": True,
                "asset_keys_preserved": True,
                "transforms_preserved": True,
                "exporter_generated_geometry": False,
            },
            "streaming_memory_contract": {
                "height_field_maximum_samples": MAX_HEIGHT_FIELD_SAMPLES,
                "placement_height_tile_count": len(placement_records),
                "placement_height_cache_tile_limit": 2,
                "full_placement_height_surface_retained_in_ram": False,
                "hero_detail_payloads_open_concurrently": (
                    1 if native_detail_extraction is not None else 0
                ),
                "maximum_instances_per_hero_detail_payload": (
                    MAX_INSTANCES_PER_DETAIL_PAYLOAD
                    if native_detail_extraction is not None
                    else None
                ),
                "identity_index": "sqlite_disk_backed",
                "full_object_inventory_retained_in_ram": False,
            },
            "fire_simulation_status": "blocked_pending_editor_review",
        }
        _write_json(staging / "composition-source.json", manifest)
        os.replace(staging, output)
        return json.loads(
            (output / "composition-source.json").read_text(encoding="utf-8")
        )
    except BaseException:
        if database is not None:
            database.close()
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _required_file_below(
    value: str | Path,
    *,
    volume_root: Path,
    label: str,
) -> Path:
    path = Path(value).expanduser().resolve()
    if (
        not _is_below(volume_root, path)
        or not path.is_file()
        or path.is_symlink()
    ):
        raise CompositionSourceError(
            f"{label} must be a regular file below the persistent volume"
        )
    return path


def _required_directory_below(
    value: str | Path,
    *,
    volume_root: Path,
    label: str,
) -> Path:
    path = Path(value).expanduser().resolve()
    if (
        not _is_below(volume_root, path)
        or not path.is_dir()
        or path.is_symlink()
    ):
        raise CompositionSourceError(
            f"{label} must be a directory below the persistent volume"
        )
    return path


def _contract_artifact(
    path: Path,
    *,
    volume_root: Path,
    prim_path: str = "",
) -> dict[str, str]:
    record = {
        "path": path.relative_to(volume_root).as_posix(),
        "sha256": _sha256(path),
    }
    if prim_path:
        record["prim_path"] = prim_path
    return record


def _zone_receipt_artifact(
    raw: object,
    *,
    zone_root: Path,
    volume_root: Path,
    label: str,
    prim_path: str = "",
) -> tuple[Path, dict[str, str]]:
    relative, expected_sha = _receipt_artifact_tuple(raw, label=label)
    path = (zone_root / Path(relative)).resolve()
    if (
        not _is_below(zone_root, path)
        or not _is_below(volume_root, path)
        or not path.is_file()
        or path.is_symlink()
    ):
        raise CompositionSourceError(f"{label} is absent or escapes the zone")
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise CompositionSourceError(f"{label} SHA-256 changed after native build")
    return path, _contract_artifact(
        path,
        volume_root=volume_root,
        prim_path=prim_path,
    )


def _validated_asset_contract(
    *,
    volume_root: Path,
    manifest_path: Path,
    lod_validation_path: Path,
    pbr_validation_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    float,
]:
    """Convert the installed SimReady bundle into the interchange contract."""

    manifest = _read_json(manifest_path, label="SimReady asset manifest")
    manifest_sha = _sha256(manifest_path)
    lod_validation = _read_json(
        lod_validation_path, label="native asset LOD validation"
    )
    pbr_validation = _read_json(
        pbr_validation_path, label="native asset PBR validation"
    )
    for payload, expected_state, label in (
        (
            lod_validation,
            "NATIVE_ASSET_LODS_VALIDATED",
            "native asset LOD validation",
        ),
        (
            pbr_validation,
            "NATIVE_PBR_MATERIALS_VALIDATED",
            "native asset PBR validation",
        ),
    ):
        if (
            payload.get("state") != expected_state
            or payload.get("manifest_sha256") != manifest_sha
        ):
            raise CompositionSourceError(
                f"{label} is stale or bound to another manifest"
            )
    environment = manifest.get("environment")
    if not isinstance(environment, Mapping):
        raise CompositionSourceError(
            "SimReady manifest has no environment asset library"
        )
    lod_evidence = _contract_artifact(
        lod_validation_path, volume_root=volume_root
    )
    asset_library: dict[str, Any] = {}
    for source_kind, category in (
        ("vegetation", "trees"),
        ("buildings", "buildings"),
    ):
        families = environment.get(source_kind)
        if not isinstance(families, Mapping) or not families:
            raise CompositionSourceError(
                f"SimReady manifest has no {source_kind} families"
            )
        for family, entries in families.items():
            family_name = _trimmed(
                family, label=f"environment.{source_kind} family"
            )
            if not isinstance(entries, list) or not entries:
                raise CompositionSourceError(
                    f"environment.{source_kind}.{family_name} is empty"
                )
            for index, entry in enumerate(entries):
                label = f"environment.{source_kind}.{family_name}[{index}]"
                if not isinstance(entry, Mapping):
                    raise CompositionSourceError(f"{label} is malformed")
                key = _trimmed(entry.get("asset_id"), label=f"{label}.asset_id")
                if key in asset_library:
                    raise CompositionSourceError(
                        f"SimReady asset ID repeats: {key}"
                    )
                lineage = _trimmed(
                    entry.get("lod_lineage_id"),
                    label=f"{label}.lod_lineage_id",
                )
                raw_lods = entry.get("lod_paths")
                if not isinstance(raw_lods, Mapping) or set(raw_lods) != set(
                    LOD_LEVELS
                ):
                    raise CompositionSourceError(
                        f"{label} requires exact HERO/MID/FAR LOD records"
                    )
                lods: dict[str, dict[str, str]] = {}
                for level in LOD_LEVELS:
                    raw_lod = raw_lods[level]
                    if not isinstance(raw_lod, Mapping):
                        raise CompositionSourceError(
                            f"{label}.lod_paths.{level} is malformed"
                        )
                    relative = _portable_receipt_path(
                        raw_lod.get("path"),
                        label=f"{label}.lod_paths.{level}.path",
                    )
                    path = (manifest_path.parent / relative).resolve()
                    if (
                        not _is_below(volume_root, path)
                        or not path.is_file()
                        or path.is_symlink()
                        or path.suffix.lower() not in _USD_SUFFIXES
                    ):
                        raise CompositionSourceError(
                            f"{label}.lod_paths.{level} is absent or unsafe"
                        )
                    expected = raw_lod.get("sha256")
                    if (
                        not isinstance(expected, str)
                        or not _SHA256.fullmatch(expected)
                        or _sha256(path) != expected
                    ):
                        raise CompositionSourceError(
                            f"{label}.lod_paths.{level} SHA-256 changed"
                        )
                    if raw_lod.get("lineage_id", lineage) != lineage:
                        raise CompositionSourceError(
                            f"{label}.lod_paths.{level} breaks LOD lineage"
                        )
                    lods[level] = _contract_artifact(
                        path,
                        volume_root=volume_root,
                        prim_path="/Asset",
                    )
                if len({record["path"] for record in lods.values()}) != 3:
                    raise CompositionSourceError(
                        f"{label} LOD wrappers must be distinct"
                    )
                anchor = entry.get("ground_anchor_m")
                if (
                    not isinstance(anchor, list)
                    or len(anchor) != 3
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for value in anchor
                    )
                ):
                    raise CompositionSourceError(
                        f"{label}.ground_anchor_m is missing or invalid"
                    )
                asset_library[key] = {
                    "category": category,
                    "family": family_name,
                    "lods": lods,
                    "simready_validation": {
                        "state": "SIMREADY_VALIDATED",
                        "lod_lineage": lineage,
                        "grounding_offsets_m": {
                            level: float(anchor[2]) for level in LOD_LEVELS
                        },
                        "evidence": lod_evidence,
                    },
                }
    if {
        record["category"] for record in asset_library.values()
    } != set(OBJECT_CATEGORIES):
        raise CompositionSourceError(
            "SimReady manifest must contain tree and building assets"
        )

    materials = manifest.get("pbr_materials")
    if not isinstance(materials, Mapping):
        raise CompositionSourceError(
            "SimReady manifest has no PBR material library"
        )
    pbr_evidence = _contract_artifact(
        pbr_validation_path, volume_root=volume_root
    )

    def material_artifact(role: str) -> dict[str, str]:
        raw = materials.get(role)
        if not isinstance(raw, Mapping):
            raise CompositionSourceError(
                f"SimReady manifest lacks required PBR material {role}"
            )
        material_file = raw.get("material_file")
        if not isinstance(material_file, Mapping):
            raise CompositionSourceError(
                f"pbr_materials.{role}.material_file is malformed"
            )
        relative = _portable_receipt_path(
            material_file.get("path"),
            label=f"pbr_materials.{role}.material_file.path",
        )
        path = (manifest_path.parent / relative).resolve()
        expected = material_file.get("sha256")
        prim_path = _trimmed(
            raw.get("material_prim_path"),
            label=f"pbr_materials.{role}.material_prim_path",
        )
        if (
            not prim_path.startswith("/")
            or not _is_below(volume_root, path)
            or not path.is_file()
            or path.is_symlink()
            or path.suffix.lower() not in _USD_SUFFIXES
            or not isinstance(expected, str)
            or not _SHA256.fullmatch(expected)
            or _sha256(path) != expected
        ):
            raise CompositionSourceError(
                f"pbr_materials.{role} artifact is absent or stale"
            )
        return _contract_artifact(
            path, volume_root=volume_root, prim_path=prim_path
        )

    water_artifact = material_artifact("water")
    raw_water = materials.get("water")
    water_uv_metres = _positive(
        (
            raw_water.get("metres_per_uv_tile")
            if isinstance(raw_water, Mapping)
            else None
        ),
        label="pbr_materials.water.metres_per_uv_tile",
    )
    water_material_source = {
        "lods": {
            level: water_artifact for level in LOD_LEVELS
        },
        "pbr_validation_state": "PBR_VALIDATED",
        "pbr_validation_evidence": pbr_evidence,
    }
    return (
        asset_library,
        water_material_source,
        water_artifact,
        water_uv_metres,
    )


def _validated_ground_contract(
    *,
    volume_root: Path,
    artifact_root: Path,
    authoring_receipt_path: Path,
) -> dict[str, Any]:
    """Validate the native payload index and all 400 tiled PBR materials."""

    receipt = _read_json(
        authoring_receipt_path,
        label="composite ground authoring receipt",
    )
    if receipt.get("state") != "COMPOSITE_GROUND_MATERIAL_NATIVE_VALIDATED":
        raise CompositionSourceError(
            "ground receipt is not COMPOSITE_GROUND_MATERIAL_NATIVE_VALIDATED"
        )
    material = receipt.get("ground_material")
    native = receipt.get("native_validation")
    if not isinstance(material, Mapping) or not isinstance(native, Mapping):
        raise CompositionSourceError(
            "ground receipt lacks material/native validation records"
        )
    relative = _portable_receipt_path(
        material.get("path"), label="ground receipt material.path"
    )
    path = (artifact_root / relative).resolve()
    expected = material.get("sha256")
    prim_path = _trimmed(
        material.get("prim_path"), label="ground receipt material.prim_path"
    )
    if (
        not prim_path.startswith("/")
        or not _is_below(artifact_root, path)
        or not _is_below(volume_root, path)
        or not path.is_file()
        or path.is_symlink()
        or path.suffix.lower() not in _USD_SUFFIXES
        or not isinstance(expected, str)
        or not _SHA256.fullmatch(expected)
        or _sha256(path) != expected
    ):
        raise CompositionSourceError(
            "ground material is absent, unsafe or stale"
        )
    exact_native_checks = (
        native.get("surface_output_connected") is True,
        native.get("all_required_branches_surface_reachable") is True,
        native.get("uniform_fallback_present") is False,
        native.get("single_graph_for_all_tiles_present") is False,
        native.get("monolithic_generated_mask_atlas_present") is False,
        native.get("source_colour_feeds_base_color") is False,
        native.get("source_geometry_creates_rendered_objects") is False,
        native.get("native_stage_reopen_succeeded") is True,
    )
    if not all(exact_native_checks):
        raise CompositionSourceError(
            "ground receipt does not prove an object-free connected PBR graph"
        )
    raw_payloads = receipt.get("tile_material_payloads")
    if not isinstance(raw_payloads, list) or len(raw_payloads) != 400:
        raise CompositionSourceError(
            "ground receipt requires exactly 400 tile material payloads"
        )
    payloads: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = {path.relative_to(volume_root).as_posix()}
    for index, raw in enumerate(raw_payloads):
        if not isinstance(raw, Mapping):
            raise CompositionSourceError(
                f"ground tile material payload {index} is malformed"
            )
        tile_id = _trimmed(
            raw.get("tile_id"),
            label=f"ground tile material payload {index}.tile_id",
        )
        if (
            raw.get("tile_ref") != tile_id
            or tile_id in seen_ids
            or raw.get("prim_path") != "/Ground"
        ):
            raise CompositionSourceError(
                f"ground tile material payload {index} identity/prim drifted"
            )
        tile_bounds = _normalized_bounds_record(
            raw.get("tile_bounds_m", []),  # type: ignore[arg-type]
            label=f"ground tile material payload {tile_id}.tile_bounds_m",
        )
        relative_payload = _portable_receipt_path(
            raw.get("path"),
            label=f"ground tile material payload {tile_id}.path",
        )
        payload_path = (artifact_root / relative_payload).resolve()
        payload_sha = raw.get("sha256")
        payload_size = raw.get("size_bytes")
        portable_payload = (
            payload_path.relative_to(volume_root).as_posix()
            if _is_below(volume_root, payload_path)
            else ""
        )
        if (
            not _is_below(artifact_root, payload_path)
            or not _is_below(volume_root, payload_path)
            or not payload_path.is_file()
            or payload_path.is_symlink()
            or payload_path.suffix.lower() not in _USD_SUFFIXES
            or not isinstance(payload_sha, str)
            or not _SHA256.fullmatch(payload_sha)
            or _sha256(payload_path) != payload_sha
            or isinstance(payload_size, bool)
            or not isinstance(payload_size, int)
            or payload_size != payload_path.stat().st_size
            or portable_payload in seen_paths
        ):
            raise CompositionSourceError(
                f"ground tile material payload {tile_id} is absent, unsafe "
                "or stale"
            )
        seen_ids.add(tile_id)
        seen_paths.add(portable_payload)
        payloads.append(
            {
                "tile_id": tile_id,
                "tile_ref": tile_id,
                "tile_bounds_m": [
                    tile_bounds["min_x"],
                    tile_bounds["min_y"],
                    tile_bounds["max_x"],
                    tile_bounds["max_y"],
                ],
                "path": portable_payload,
                "sha256": payload_sha,
                "prim_path": "/Ground",
            }
        )
    payloads.sort(key=lambda item: str(item["tile_id"]))
    canonical_payloads_sha = hashlib.sha256(
        json.dumps(
            payloads,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    receipt_payloads_sha = receipt.get("tile_material_payloads_sha256")
    # Terrain PBR's canonical lock includes size_bytes. Recompute that exact
    # source representation separately from the portable layout records.
    source_payloads = sorted(
        (dict(raw) for raw in raw_payloads if isinstance(raw, Mapping)),
        key=lambda item: str(item.get("tile_id", "")),
    )
    source_payloads_sha = hashlib.sha256(
        json.dumps(
            source_payloads,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        receipt_payloads_sha != source_payloads_sha
        or native.get("tile_payload_layers_sha256")
        != source_payloads_sha
    ):
        raise CompositionSourceError(
            "ground tile material payload inventory is stale"
        )
    return {
        "material": _contract_artifact(
            path,
            volume_root=volume_root,
            prim_path=prim_path,
        ),
        "validation_evidence": _contract_artifact(
            authoring_receipt_path,
            volume_root=volume_root,
        ),
        "topology": "payload_tiled_materials_shared_pbr_library",
        "tile_material_payloads": payloads,
        "portable_tile_material_payloads_sha256": canonical_payloads_sha,
        "validation_state": "OBJECT_FREE_PBR_VALIDATED",
        "kind": "object_free_pbr",
        "removed_object_classes": [],
    }


def _geojson_polygons(geometry: object) -> Iterator[list[list[float]]]:
    if not isinstance(geometry, Mapping):
        return
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    candidates: object
    if geometry_type == "Polygon":
        candidates = [coordinates]
    elif geometry_type == "MultiPolygon":
        candidates = coordinates
    else:
        return
    if not isinstance(candidates, list):
        return
    for polygon in candidates:
        if not isinstance(polygon, list) or not polygon:
            continue
        outer = polygon[0]
        if not isinstance(outer, list):
            continue
        points: list[list[float]] = []
        for point in outer:
            if (
                not isinstance(point, list)
                or len(point) < 2
                or isinstance(point[0], bool)
                or isinstance(point[1], bool)
                or not isinstance(point[0], (int, float))
                or not isinstance(point[1], (int, float))
            ):
                points = []
                break
            x = float(point[0])
            y = float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                points = []
                break
            if not points or points[-1] != [x, y]:
                points.append([x, y])
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        if len(points) >= 3:
            yield points


def _geojson_lines(geometry: object) -> Iterator[list[list[float]]]:
    if not isinstance(geometry, Mapping):
        return
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    candidates: object
    if geometry_type == "LineString":
        candidates = [coordinates]
    elif geometry_type == "MultiLineString":
        candidates = coordinates
    else:
        return
    if not isinstance(candidates, list):
        return
    for line in candidates:
        if not isinstance(line, list):
            continue
        points: list[list[float]] = []
        for point in line:
            if (
                not isinstance(point, list)
                or len(point) < 2
                or isinstance(point[0], bool)
                or isinstance(point[1], bool)
                or not isinstance(point[0], (int, float))
                or not isinstance(point[1], (int, float))
            ):
                points = []
                break
            x = float(point[0])
            y = float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                points = []
                break
            if not points or math.hypot(
                x - points[-1][0], y - points[-1][1]
            ) > 1.0e-7:
                points.append([x, y])
        if len(points) >= 2:
            yield points


def _clip_segment_axis_aligned(
    first: Sequence[float],
    second: Sequence[float],
    *,
    bounds: tuple[float, float, float, float],
) -> tuple[list[float], list[float]] | None:
    """Liang-Barsky clipping without changing the source segment path."""

    xmin, ymin, xmax, ymax = bounds
    dx = float(second[0]) - float(first[0])
    dy = float(second[1]) - float(first[1])
    entering = 0.0
    leaving = 1.0
    for p, q in (
        (-dx, float(first[0]) - xmin),
        (dx, xmax - float(first[0])),
        (-dy, float(first[1]) - ymin),
        (dy, ymax - float(first[1])),
    ):
        if abs(p) <= 1.0e-12:
            if q < 0.0:
                return None
            continue
        ratio = q / p
        if p < 0.0:
            entering = max(entering, ratio)
        else:
            leaving = min(leaving, ratio)
        if entering > leaving:
            return None
    return (
        [
            float(first[0]) + entering * dx,
            float(first[1]) + entering * dy,
        ],
        [
            float(first[0]) + leaving * dx,
            float(first[1]) + leaving * dy,
        ],
    )


def _clip_line_axis_aligned(
    line: Sequence[Sequence[float]],
    *,
    bounds: tuple[float, float, float, float],
) -> list[list[list[float]]]:
    """Return continuous source-line fragments inside the scene rectangle."""

    fragments: list[list[list[float]]] = []
    current: list[list[float]] = []
    for first, second in zip(line, line[1:]):
        clipped = _clip_segment_axis_aligned(
            first, second, bounds=bounds
        )
        if clipped is None:
            if len(current) >= 2:
                fragments.append(current)
            current = []
            continue
        clipped_start, clipped_end = clipped
        if current and math.hypot(
            current[-1][0] - clipped_start[0],
            current[-1][1] - clipped_start[1],
        ) <= 1.0e-7:
            if math.hypot(
                current[-1][0] - clipped_end[0],
                current[-1][1] - clipped_end[1],
            ) > 1.0e-7:
                current.append(clipped_end)
        else:
            if len(current) >= 2:
                fragments.append(current)
            current = [clipped_start, clipped_end]
    if len(current) >= 2:
        fragments.append(current)
    return fragments


def _clip_polygon_axis_aligned(
    polygon: Sequence[Sequence[float]],
    *,
    bounds: tuple[float, float, float, float],
) -> list[list[float]]:
    """Sutherland-Hodgman clip against the accepted EPSG:2154 scene box."""

    result = [[float(point[0]), float(point[1])] for point in polygon]
    xmin, ymin, xmax, ymax = bounds
    edges = (
        ("left", xmin),
        ("right", xmax),
        ("bottom", ymin),
        ("top", ymax),
    )

    def inside(point: Sequence[float], edge: str, value: float) -> bool:
        if edge == "left":
            return point[0] >= value
        if edge == "right":
            return point[0] <= value
        if edge == "bottom":
            return point[1] >= value
        return point[1] <= value

    def intersection(
        first: Sequence[float],
        second: Sequence[float],
        edge: str,
        value: float,
    ) -> list[float]:
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        if edge in {"left", "right"}:
            if abs(dx) <= 1.0e-12:
                return [value, float(first[1])]
            ratio = (value - first[0]) / dx
            return [value, float(first[1] + ratio * dy)]
        if abs(dy) <= 1.0e-12:
            return [float(first[0]), value]
        ratio = (value - first[1]) / dy
        return [float(first[0] + ratio * dx), value]

    for edge, value in edges:
        if not result:
            break
        clipped: list[list[float]] = []
        previous = result[-1]
        previous_inside = inside(previous, edge, value)
        for current in result:
            current_inside = inside(current, edge, value)
            if current_inside:
                if not previous_inside:
                    clipped.append(
                        intersection(previous, current, edge, value)
                    )
                clipped.append(current)
            elif previous_inside:
                clipped.append(intersection(previous, current, edge, value))
            previous = current
            previous_inside = current_inside
        result = []
        for point in clipped:
            if not result or math.hypot(
                point[0] - result[-1][0],
                point[1] - result[-1][1],
            ) > 1.0e-7:
                result.append(point)
        if len(result) > 1 and math.hypot(
            result[0][0] - result[-1][0],
            result[0][1] - result[-1][1],
        ) <= 1.0e-7:
            result.pop()
    if len(result) < 3:
        return []
    twice_area = sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(result, result[1:] + result[:1])
    )
    return result if abs(twice_area) > 1.0e-6 else []


def _locked_vector_paths(
    *,
    zone_root: Path,
    source_lock: Mapping[str, object],
    category: str,
) -> tuple[Path, ...]:
    vector_sources = source_lock.get("vector_sources")
    records = (
        vector_sources.get(category)
        if isinstance(vector_sources, Mapping)
        else None
    )
    if not isinstance(records, list) or not records:
        raise CompositionSourceError(
            f"source-lock has no acquired BDTOPO {category} vectors"
        )
    result: list[Path] = []
    seen_paths: set[Path] = set()
    for index, record in enumerate(records):
        label = f"source-lock.vector_sources.{category}[{index}]"
        if (
            not isinstance(record, Mapping)
            or record.get("license") != "Licence Ouverte / Etalab 2.0"
        ):
            raise CompositionSourceError(f"{label} provenance is incomplete")
        download = record.get("download")
        if (
            not isinstance(download, Mapping)
            or download.get("state") != "downloaded"
        ):
            raise CompositionSourceError(f"{label} is not downloaded")
        relative = _portable_receipt_path(
            download.get("relpath"), label=f"{label}.download.relpath"
        )
        path = (zone_root / "raw" / relative).resolve()
        expected = download.get("sha256")
        if (
            not _is_below(zone_root / "raw", path)
            or not path.is_file()
            or path.is_symlink()
            or not isinstance(expected, str)
            or not _SHA256.fullmatch(expected)
            or _sha256(path) != expected
        ):
            raise CompositionSourceError(
                f"{label} file is absent, unsafe or stale"
            )
        if path in seen_paths:
            raise CompositionSourceError(
                f"{label} repeats an already locked BDTOPO artifact"
            )
        seen_paths.add(path)
        result.append(path)
    return tuple(result)


def _source_backed_suitability(
    *,
    zone_root: Path,
    source_lock: Mapping[str, object],
    epsg2154_bounds: tuple[float, float, float, float],
    root_origin: tuple[float, float],
    terrain: Sequence[_PreparedTerrainGrid],
    tree_families: Sequence[str],
    observations: _NativeSuitabilityObservations,
) -> list[dict[str, object]]:
    """Bind suitability to the families and settlements observed in native USD.

    The BDTOPO product does not provide a soil class.  The explicit
    ``source_not_provided`` label records that absence instead of inventing a
    loam/sand classification.  Vegetation polygons inherit only tree families
    observed in the exact native terrain tiles they intersect.  Buildability is
    limited to accepted native settlement-group extents plus the reviewed
    spacing margin; the former all-terrain buildable shortcut is forbidden.
    """

    accepted_families = set(tree_families)
    if not accepted_families:
        raise CompositionSourceError(
            "native suitability requires accepted SimReady tree families"
        )
    terrain_by_ref = {tile.tile_ref: tile for tile in terrain}
    if (
        len(terrain_by_ref) != len(terrain)
        or set(observations.tree_families_by_tile_ref) != set(terrain_by_ref)
    ):
        raise CompositionSourceError(
            "native tree-family observations must cover the exact terrain "
            "tile inventory"
        )
    observed_families = {
        family
        for families in observations.tree_families_by_tile_ref.values()
        for family in families
    }
    if observed_families != accepted_families:
        raise CompositionSourceError(
            "native suitability observations differ from the accepted "
            "SimReady tree library"
        )

    def polygon_intersects_tile(
        polygon: Sequence[Sequence[float]],
        tile: _PreparedTerrainGrid,
    ) -> bool:
        bounds = tile.local_bounds
        rectangle = [
            [bounds["min_x"], bounds["min_y"]],
            [bounds["max_x"], bounds["min_y"]],
            [bounds["max_x"], bounds["max_y"]],
            [bounds["min_x"], bounds["max_y"]],
        ]
        if any(
            bounds["min_x"] - 1.0e-9 <= float(point[0])
            <= bounds["max_x"] + 1.0e-9
            and bounds["min_y"] - 1.0e-9 <= float(point[1])
            <= bounds["max_y"] + 1.0e-9
            for point in polygon
        ):
            return True
        if any(_point_in_polygon_raw(point, polygon) for point in rectangle):
            return True
        return any(
            _segments_intersect_raw(
                polygon_start,
                polygon_end,
                rectangle_start,
                rectangle_end,
            )
            for polygon_start, polygon_end in zip(
                polygon, polygon[1:] + polygon[:1]  # type: ignore[operator]
            )
            for rectangle_start, rectangle_end in zip(
                rectangle, rectangle[1:] + rectangle[:1]
            )
        )

    vegetation_paths = _locked_vector_paths(
        zone_root=zone_root,
        source_lock=source_lock,
        category="vegetation",
    )
    result: list[dict[str, object]] = []
    covered_families: set[str] = set()
    stable_index = 0
    for path in vegetation_paths:
        collection = _read_json(path, label="locked BDTOPO vegetation GeoJSON")
        features = collection.get("features")
        if not isinstance(features, list):
            raise CompositionSourceError(
                f"locked vegetation GeoJSON has no features: {path}"
            )
        for feature_index, feature in enumerate(features):
            if not isinstance(feature, Mapping):
                continue
            properties = feature.get("properties")
            biome = "bdtopo_zone_de_vegetation"
            if isinstance(properties, Mapping):
                for key in ("nature", "type_de_vegetation", "usage"):
                    raw = properties.get(key)
                    if isinstance(raw, str) and raw.strip():
                        biome = f"bdtopo:{raw.strip()}"
                        break
            for polygon_index, polygon in enumerate(
                _geojson_polygons(feature.get("geometry"))
            ):
                clipped = _clip_polygon_axis_aligned(
                    polygon, bounds=epsg2154_bounds
                )
                if not clipped:
                    continue
                local = [
                    [
                        point[0] - root_origin[0],
                        point[1] - root_origin[1],
                    ]
                    for point in clipped
                ]
                local_families = sorted(
                    {
                        family
                        for tile in terrain
                        if polygon_intersects_tile(local, tile)
                        for family in observations.tree_families_by_tile_ref[
                            tile.tile_ref
                        ]
                    }
                )
                if not local_families:
                    continue
                result.append(
                    {
                        "stable_id": (
                            f"bdtopo-vegetation-{stable_index:08d}-"
                            f"{feature_index}-{polygon_index}"
                        ),
                        "outline": local,
                        "biome": biome,
                        "soil": "source_not_provided",
                        "tree_families": local_families,
                        "buildable": False,
                    }
                )
                covered_families.update(local_families)
                stable_index += 1
    if not result:
        raise CompositionSourceError(
            "locked BDTOPO vegetation has no polygon backed by observed "
            "native tree instances"
        )
    missing_families = accepted_families - covered_families
    if missing_families:
        raise CompositionSourceError(
            "locked vegetation does not spatially support observed native "
            "tree families: " + ", ".join(sorted(missing_families))
        )

    scene_min_x = min(tile.local_bounds["min_x"] for tile in terrain)
    scene_min_y = min(tile.local_bounds["min_y"] for tile in terrain)
    scene_max_x = max(tile.local_bounds["max_x"] for tile in terrain)
    scene_max_y = max(tile.local_bounds["max_y"] for tile in terrain)
    settlement_margin = float(
        _fixed_variant_constraints()["building_group_spacing_m"]
    )
    if not observations.building_group_bounds:
        raise CompositionSourceError(
            "native suitability requires accepted building settlement groups"
        )
    for group_id, raw_bounds in sorted(
        observations.building_group_bounds.items()
    ):
        min_x, min_y, max_x, max_y, instance_count = raw_bounds
        if (
            instance_count <= 0
            or not all(
                math.isfinite(value)
                for value in (min_x, min_y, max_x, max_y)
            )
            or max_x <= min_x
            or max_y <= min_y
        ):
            raise CompositionSourceError(
                f"native building group {group_id} has invalid bounds"
            )
        min_x = max(scene_min_x, min_x - settlement_margin)
        min_y = max(scene_min_y, min_y - settlement_margin)
        max_x = min(scene_max_x, max_x + settlement_margin)
        max_y = min(scene_max_y, max_y + settlement_margin)
        if max_x <= min_x or max_y <= min_y:
            raise CompositionSourceError(
                f"native building group {group_id} lies outside terrain bounds"
            )
        stable_group = hashlib.sha256(
            group_id.encode("utf-8")
        ).hexdigest()[:16]
        result.append(
            {
                "stable_id": f"native-settlement-{stable_group}",
                "outline": [
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y],
                ],
                "biome": "accepted_native_settlement_group",
                "soil": "source_not_provided",
                "tree_families": [],
                "buildable": True,
                "source_group_id": group_id,
                "source_instance_count": instance_count,
            }
        )
    return result


def _parse_epsg2154_bounds(
    value: object,
    *,
    label: str,
) -> tuple[float, float, float, float]:
    if not isinstance(value, str):
        raise CompositionSourceError(f"{label} is missing EPSG:2154 bounds")
    try:
        bounds = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise CompositionSourceError(
            f"{label} has malformed EPSG:2154 bounds"
        ) from error
    if (
        len(bounds) != 4
        or any(not math.isfinite(item) for item in bounds)
        or bounds[2] <= bounds[0]
        or bounds[3] <= bounds[1]
    ):
        raise CompositionSourceError(
            f"{label} has invalid EPSG:2154 bounds"
        )
    return bounds  # type: ignore[return-value]


def _prepared_terrain_grids(
    *,
    volume_root: Path,
    zone_root: Path,
    build: Mapping[str, object],
    root_origin: tuple[float, float],
    placement_physical_root: Path,
    placement_final_root: Path,
) -> tuple[
    tuple[_PreparedTerrainGrid, ...],
    tuple[_PreparedPlacementHeight, ...],
]:
    """Open one terrain payload at a time and retain only its 32² FAR grid."""

    try:
        from pxr import Usd, UsdGeom  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError) as error:
        raise CompositionSourceError(
            "native preparation requires the pinned Kit/Isaac pxr runtime"
        ) from error
    payloads = build.get("payloads")
    coverage = build.get("tile_coverage")
    if (
        not isinstance(payloads, list)
        or len(payloads) != 400
        or not isinstance(coverage, list)
        or len(coverage) != 400
    ):
        raise CompositionSourceError(
            "native preparation requires exact 400 terrain payloads/coverage"
        )
    coverage_by_path: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(coverage):
        if not isinstance(raw, Mapping):
            raise CompositionSourceError(
                f"tile_coverage[{index}] is malformed"
            )
        path = _portable_receipt_path(
            raw.get("terrain_payload"),
            label=f"tile_coverage[{index}].terrain_payload",
        )
        if path in coverage_by_path:
            raise CompositionSourceError(
                f"terrain coverage path repeats: {path}"
            )
        coverage_by_path[path] = raw
    result: list[_PreparedTerrainGrid] = []
    placements: list[_PreparedPlacementHeight] = []
    seen_refs: set[str] = set()
    seen_namespaces: set[int] = set()
    for index, raw in enumerate(payloads):
        relative, expected_sha = _receipt_artifact_tuple(
            raw, label=f"native build payloads[{index}]"
        )
        coverage_record = coverage_by_path.get(relative)
        if coverage_record is None:
            raise CompositionSourceError(
                f"terrain payload {relative} has no exact coverage record"
            )
        tile_ref = _trimmed(
            coverage_record.get("tile_ref"),
            label=f"tile_coverage[{index}].tile_ref",
        )
        namespace = _positive_id(
            coverage_record.get("instance_namespace"),
            label=f"tile_coverage[{index}].instance_namespace",
        )
        raw_terrain_lods = coverage_record.get("terrain_lods")
        raw_collision_lods = coverage_record.get("collision_lods")
        if (
            not isinstance(raw_terrain_lods, list)
            or any(not isinstance(value, str) for value in raw_terrain_lods)
            or tuple(raw_terrain_lods)
            not in {
                ("LOD1", "LOD2", "LOD3"),
                ("LOD0", "LOD1", "LOD2", "LOD3"),
            }
            or raw_collision_lods != ["NEAR", "FAR"]
        ):
            raise CompositionSourceError(
                f"tile_coverage[{index}] lacks exact native terrain/collision LODs"
            )
        terrain_lods = tuple(raw_terrain_lods)
        collision_lods = tuple(raw_collision_lods)
        if tile_ref in seen_refs or namespace in seen_namespaces:
            raise CompositionSourceError(
                "terrain tile refs and instance namespaces must be unique"
            )
        seen_refs.add(tile_ref)
        seen_namespaces.add(namespace)
        path = (zone_root / relative).resolve()
        if (
            not _is_below(zone_root, path)
            or not _is_below(volume_root, path)
            or not path.is_file()
            or path.is_symlink()
            or path.suffix.lower() not in _USD_SUFFIXES
            or _sha256(path) != expected_sha
        ):
            raise CompositionSourceError(
                f"terrain payload is absent or stale: {relative}"
            )
        stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
        if stage is None:
            raise CompositionSourceError(
                f"terrain payload cannot be opened: {relative}"
            )
        tile = stage.GetPrimAtPath("/Tile")
        if not tile or not tile.IsValid():
            raise CompositionSourceError(
                f"terrain payload has no /Tile prim: {relative}"
            )
        if tile.GetCustomDataByKey("fireviewer:tile_ref") != tile_ref:
            raise CompositionSourceError(
                f"terrain payload tile_ref differs from coverage: {relative}"
            )
        epsg_bounds = _parse_epsg2154_bounds(
            tile.GetCustomDataByKey("fireviewer:epsg2154_bounds"),
            label=relative,
        )
        variants = tile.GetVariantSets()
        terrain_lod = variants.GetVariantSet("terrainLOD")
        collision_lod = variants.GetVariantSet("collisionLOD")
        if (
            not terrain_lod.IsValid()
            or set(terrain_lod.GetVariantNames()) != set(terrain_lods)
            or not collision_lod.IsValid()
            or set(collision_lod.GetVariantNames()) != set(collision_lods)
        ):
            raise CompositionSourceError(
                f"terrain payload variants differ from coverage: {relative}"
            )
        collision_lod.SetVariantSelection("FAR")
        mesh_prim = stage.GetPrimAtPath("/Tile/Collision")
        if not mesh_prim or not mesh_prim.IsValid():
            raise CompositionSourceError(
                f"terrain payload has no FAR collision mesh: {relative}"
            )
        points = list(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get() or [])
        side = int(round(math.sqrt(len(points))))
        if (
            side < 2
            or side > MAX_HEIGHT_FIELD_SIDE
            or side * side != len(points)
        ):
            raise CompositionSourceError(
                f"terrain FAR collision grid is not square/bounded: {relative}"
            )
        rows: list[list[tuple[float, float, float]]] = []
        for row_index in range(side):
            row: list[tuple[float, float, float]] = []
            for column_index in range(side):
                point = points[row_index * side + column_index]
                try:
                    xyz = (
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                    )
                except (IndexError, TypeError, ValueError) as error:
                    raise CompositionSourceError(
                        f"terrain FAR collision point is malformed: {relative}"
                    ) from error
                if any(not math.isfinite(value) for value in xyz):
                    raise CompositionSourceError(
                        f"terrain FAR collision point is non-finite: {relative}"
                    )
                row.append(xyz)
            rows.append(row)
        x_coordinates = [point[0] for point in rows[0]]
        y_coordinates = [row[0][1] for row in rows]
        if x_coordinates[0] > x_coordinates[-1]:
            x_coordinates.reverse()
            rows = [list(reversed(row)) for row in rows]
        if y_coordinates[0] > y_coordinates[-1]:
            y_coordinates.reverse()
            rows.reverse()
        if any(
            second <= first
            for first, second in zip(x_coordinates, x_coordinates[1:])
        ) or any(
            second <= first
            for first, second in zip(y_coordinates, y_coordinates[1:])
        ):
            raise CompositionSourceError(
                f"terrain FAR collision axes are not strictly regular: {relative}"
            )
        tolerance = 1.0e-3
        if any(
            abs(row[column_index][0] - x_coordinates[column_index])
            > tolerance
            for row in rows
            for column_index in range(side)
        ) or any(
            abs(point[1] - y_coordinates[row_index]) > tolerance
            for row_index, row in enumerate(rows)
            for point in row
        ):
            raise CompositionSourceError(
                f"terrain FAR collision grid axes are inconsistent: {relative}"
            )
        local_bounds = {
            "min_x": epsg_bounds[0] - root_origin[0],
            "min_y": epsg_bounds[1] - root_origin[1],
            "max_x": epsg_bounds[2] - root_origin[0],
            "max_y": epsg_bounds[3] - root_origin[1],
        }
        if (
            abs(x_coordinates[0] - local_bounds["min_x"]) > 1.0
            or abs(x_coordinates[-1] - local_bounds["max_x"]) > 1.0
            or abs(y_coordinates[0] - local_bounds["min_y"]) > 1.0
            or abs(y_coordinates[-1] - local_bounds["max_y"]) > 1.0
        ):
            raise CompositionSourceError(
                f"terrain FAR grid does not cover its declared tile: {relative}"
            )
        result.append(
            _PreparedTerrainGrid(
                artifact=ArtifactSource(
                    path=path,
                    prim_path="/Tile",
                    expected_sha256=expected_sha,
                ),
                tile_ref=tile_ref,
                instance_namespace=namespace,
                terrain_lods=terrain_lods,
                collision_lods=collision_lods,
                epsg2154_bounds=epsg_bounds,
                local_bounds=local_bounds,
                x_coordinates=tuple(x_coordinates),
                y_coordinates=tuple(y_coordinates),
                elevations=tuple(
                    tuple(point[2] for point in row) for row in rows
                ),
            )
        )
        collision_lod.SetVariantSelection("NEAR")
        near_prim = stage.GetPrimAtPath("/Tile/Collision")
        if not near_prim or not near_prim.IsValid():
            raise CompositionSourceError(
                f"terrain payload has no NEAR collision mesh: {relative}"
            )
        near_points = list(
            UsdGeom.Mesh(near_prim).GetPointsAttr().Get() or []
        )
        near_side = int(round(math.sqrt(len(near_points))))
        if (
            near_side < 2
            or near_side > 513
            or near_side * near_side != len(near_points)
        ):
            raise CompositionSourceError(
                f"terrain NEAR collision grid is not square/bounded: {relative}"
            )
        near_rows: list[list[tuple[float, float, float]]] = []
        for near_row_index in range(near_side):
            near_row: list[tuple[float, float, float]] = []
            for near_column_index in range(near_side):
                point = near_points[
                    near_row_index * near_side + near_column_index
                ]
                try:
                    xyz = (
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                    )
                except (IndexError, TypeError, ValueError) as error:
                    raise CompositionSourceError(
                        f"terrain NEAR collision point is malformed: {relative}"
                    ) from error
                if any(not math.isfinite(value) for value in xyz):
                    raise CompositionSourceError(
                        f"terrain NEAR collision point is non-finite: {relative}"
                    )
                near_row.append(xyz)
            near_rows.append(near_row)
        near_x = [point[0] for point in near_rows[0]]
        near_y = [row[0][1] for row in near_rows]
        if near_x[0] > near_x[-1]:
            near_x.reverse()
            near_rows = [list(reversed(row)) for row in near_rows]
        if near_y[0] > near_y[-1]:
            near_y.reverse()
            near_rows.reverse()
        if any(
            second <= first
            for first, second in zip(near_x, near_x[1:])
        ) or any(
            second <= first
            for first, second in zip(near_y, near_y[1:])
        ):
            raise CompositionSourceError(
                f"terrain NEAR collision axes are not increasing: {relative}"
            )
        if (
            abs(near_x[0] - local_bounds["min_x"]) > 1.0
            or abs(near_x[-1] - local_bounds["max_x"]) > 1.0
            or abs(near_y[0] - local_bounds["min_y"]) > 1.0
            or abs(near_y[-1] - local_bounds["max_y"]) > 1.0
        ):
            raise CompositionSourceError(
                f"terrain NEAR grid does not cover its tile: {relative}"
            )
        if any(
            abs(row[column_index][0] - near_x[column_index]) > 1.0e-3
            for row in near_rows
            for column_index in range(near_side)
        ) or any(
            abs(point[1] - near_y[row_index]) > 1.0e-3
            for row_index, row in enumerate(near_rows)
            for point in row
        ):
            raise CompositionSourceError(
                f"terrain NEAR grid axes are inconsistent: {relative}"
            )
        safe_ref = re.sub(r"[^A-Za-z0-9_.-]", "_", tile_ref)
        physical_height = (
            placement_physical_root
            / f"{index:04d}-{safe_ref}.f32"
        )
        final_height = (
            placement_final_root / f"{index:04d}-{safe_ref}.f32"
        )
        physical_height.parent.mkdir(parents=True, exist_ok=True)
        values = array(
            "f",
            (
                point[2]
                for row in near_rows
                for point in row
            ),
        )
        south_edge = tuple(values[0:near_side])
        north_edge = tuple(
            values[(near_side - 1) * near_side : near_side * near_side]
        )
        west_edge = tuple(
            values[row_index * near_side]
            for row_index in range(near_side)
        )
        east_edge = tuple(
            values[(row_index + 1) * near_side - 1]
            for row_index in range(near_side)
        )
        if sys.byteorder != "little":
            values.byteswap()
        with physical_height.open("wb") as stream:
            values.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        expected_bytes = near_side * near_side * 4
        if physical_height.stat().st_size != expected_bytes:
            raise CompositionSourceError(
                f"terrain NEAR binary size is inconsistent: {relative}"
            )
        placements.append(
            _PreparedPlacementHeight(
                tile_ref=tile_ref,
                local_bounds=local_bounds,
                x_coordinates=tuple(near_x),
                y_coordinates=tuple(near_y),
                physical_path=physical_height,
                final_path=final_height,
                sha256=_sha256(physical_height),
                south_edge=south_edge,
                north_edge=north_edge,
                west_edge=west_edge,
                east_edge=east_edge,
            )
        )
        values = array("f")
        near_points = []
        near_rows = []
        stage = None
    if len(result) != 400:
        raise CompositionSourceError(
            "native terrain preparation did not inspect exactly 400 tiles"
        )
    if len(placements) != 400:
        raise CompositionSourceError(
            "native terrain preparation did not author 400 placement tiles"
        )
    return tuple(result), tuple(placements)


def _placement_height_records(
    *,
    placements: Sequence[_PreparedPlacementHeight],
    volume_root: Path,
) -> tuple[list[dict[str, object]], str, dict[str, object]]:
    """Validate all NEAR tile seams and bind their lazy float32 payloads.

    Only the four edge vectors per tile are retained after USD extraction.
    This proves a continuous placement surface without ever retaining the 400
    full-resolution grids in memory together.
    """

    if len(placements) != 400:
        raise CompositionSourceError(
            "placement height contract requires exactly 400 NEAR tiles"
        )
    x_starts = sorted(
        {tile.local_bounds["min_x"] for tile in placements}
    )
    y_starts = sorted(
        {tile.local_bounds["min_y"] for tile in placements}
    )
    by_origin = {
        (
            tile.local_bounds["min_x"],
            tile.local_bounds["min_y"],
        ): tile
        for tile in placements
    }
    if (
        len(by_origin) != len(placements)
        or len(x_starts) * len(y_starts) != len(placements)
    ):
        raise CompositionSourceError(
            "placement height tiles do not form one complete lattice"
        )
    maximum_delta = 0.0
    adjacent_pairs = 0

    def compare_axes_and_edges(
        *,
        first_axis: Sequence[float],
        second_axis: Sequence[float],
        first_edge: Sequence[float],
        second_edge: Sequence[float],
        label: str,
    ) -> None:
        nonlocal maximum_delta, adjacent_pairs
        if (
            len(first_axis) != len(second_axis)
            or len(first_edge) != len(second_edge)
            or len(first_axis) != len(first_edge)
            or any(
                not math.isclose(
                    first_value,
                    second_value,
                    abs_tol=0.01,
                    rel_tol=0.0,
                )
                for first_value, second_value in zip(
                    first_axis, second_axis
                )
            )
        ):
            raise CompositionSourceError(
                f"adjacent placement height axes diverge: {label}"
            )
        edge_delta = max(
            (
                abs(first_value - second_value)
                for first_value, second_value in zip(
                    first_edge, second_edge
                )
            ),
            default=0.0,
        )
        maximum_delta = max(maximum_delta, edge_delta)
        adjacent_pairs += 1
        if edge_delta > MAX_ADJACENT_EDGE_HEIGHT_DELTA_M:
            raise CompositionSourceError(
                f"adjacent placement height seam {label} differs by "
                f"{edge_delta:.6f} m (limit "
                f"{MAX_ADJACENT_EDGE_HEIGHT_DELTA_M:.6f} m)"
            )

    for x_index, x_start in enumerate(x_starts):
        for y_index, y_start in enumerate(y_starts):
            tile = by_origin.get((x_start, y_start))
            if tile is None:
                raise CompositionSourceError(
                    "placement height lattice contains a gap"
                )
            expected_max_x = (
                x_starts[x_index + 1]
                if x_index + 1 < len(x_starts)
                else max(
                    candidate.local_bounds["max_x"]
                    for candidate in placements
                )
            )
            expected_max_y = (
                y_starts[y_index + 1]
                if y_index + 1 < len(y_starts)
                else max(
                    candidate.local_bounds["max_y"]
                    for candidate in placements
                )
            )
            if not (
                math.isclose(
                    tile.local_bounds["max_x"],
                    expected_max_x,
                    abs_tol=0.01,
                    rel_tol=0.0,
                )
                and math.isclose(
                    tile.local_bounds["max_y"],
                    expected_max_y,
                    abs_tol=0.01,
                    rel_tol=0.0,
                )
            ):
                raise CompositionSourceError(
                    f"placement height tile {tile.tile_ref} overlaps or gaps"
                )
            if x_index + 1 < len(x_starts):
                east = by_origin.get(
                    (x_starts[x_index + 1], y_start)
                )
                if east is None:
                    raise CompositionSourceError(
                        "placement height lattice has no eastern neighbour"
                    )
                compare_axes_and_edges(
                    first_axis=tile.y_coordinates,
                    second_axis=east.y_coordinates,
                    first_edge=tile.east_edge,
                    second_edge=east.west_edge,
                    label=f"{tile.tile_ref}/{east.tile_ref}:east-west",
                )
            if y_index + 1 < len(y_starts):
                north = by_origin.get(
                    (x_start, y_starts[y_index + 1])
                )
                if north is None:
                    raise CompositionSourceError(
                        "placement height lattice has no northern neighbour"
                    )
                compare_axes_and_edges(
                    first_axis=tile.x_coordinates,
                    second_axis=north.x_coordinates,
                    first_edge=tile.north_edge,
                    second_edge=north.south_edge,
                    label=f"{tile.tile_ref}/{north.tile_ref}:north-south",
                )

    records: list[dict[str, object]] = []
    for tile in placements:
        if (
            not tile.physical_path.is_file()
            or tile.physical_path.is_symlink()
            or _sha256(tile.physical_path) != tile.sha256
        ):
            raise CompositionSourceError(
                f"placement height payload is absent or stale: {tile.tile_ref}"
            )
        records.append(
            {
                "tile_ref": tile.tile_ref,
                "local_bounds": dict(tile.local_bounds),
                "path": tile.final_path.relative_to(volume_root).as_posix(),
                "sha256": tile.sha256,
                "format": "float32-le-row-major-south-to-north",
                "width": len(tile.x_coordinates),
                "height": len(tile.y_coordinates),
                "x_coordinates": list(tile.x_coordinates),
                "y_coordinates": list(tile.y_coordinates),
            }
        )
    records.sort(key=lambda item: str(item["tile_ref"]))
    fingerprint = hashlib.sha256(
        json.dumps(
            records,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return (
        records,
        fingerprint,
        {
            "algorithm": "shared-edge-float32-near-grid-v1",
            "adjacent_pair_count": adjacent_pairs,
            "maximum_adjacent_edge_height_delta_m": maximum_delta,
            "maximum_allowed_adjacent_edge_height_delta_m": (
                MAX_ADJACENT_EDGE_HEIGHT_DELTA_M
            ),
        },
    )


class _PlacementHeightSampler:
    """Lazy bilinear access to the accepted NEAR grids with bounded cache."""

    def __init__(
        self,
        placements: Sequence[_PreparedPlacementHeight],
        *,
        cache_limit: int = 2,
    ) -> None:
        if not placements or not 1 <= cache_limit <= 8:
            raise CompositionSourceError(
                "placement sampler requires tiles and a 1..8 tile cache"
            )
        self._x_starts = sorted(
            {tile.local_bounds["min_x"] for tile in placements}
        )
        self._y_starts = sorted(
            {tile.local_bounds["min_y"] for tile in placements}
        )
        self._by_origin = {
            (
                tile.local_bounds["min_x"],
                tile.local_bounds["min_y"],
            ): tile
            for tile in placements
        }
        if (
            len(self._by_origin) != len(placements)
            or len(self._x_starts) * len(self._y_starts)
            != len(placements)
        ):
            raise CompositionSourceError(
                "placement sampler requires a complete tile lattice"
            )
        self._cache_limit = cache_limit
        self._cache: OrderedDict[str, array[float]] = OrderedDict()
        self.peak_cached_tiles = 0

    def _tile(self, x: float, y: float) -> _PreparedPlacementHeight:
        x_cell = min(
            max(bisect.bisect_right(self._x_starts, x) - 1, 0),
            len(self._x_starts) - 1,
        )
        y_cell = min(
            max(bisect.bisect_right(self._y_starts, y) - 1, 0),
            len(self._y_starts) - 1,
        )
        tile = self._by_origin.get(
            (self._x_starts[x_cell], self._y_starts[y_cell])
        )
        if (
            tile is None
            or x < tile.local_bounds["min_x"] - 1.0e-6
            or x > tile.local_bounds["max_x"] + 1.0e-6
            or y < tile.local_bounds["min_y"] - 1.0e-6
            or y > tile.local_bounds["max_y"] + 1.0e-6
        ):
            raise CompositionSourceError(
                f"root-local point ({x},{y}) has no accepted terrain tile"
            )
        return tile

    def _values(
        self, tile: _PreparedPlacementHeight
    ) -> array[float]:
        values = self._cache.get(tile.tile_ref)
        if values is not None:
            self._cache.move_to_end(tile.tile_ref)
            return values
        values = array("f")
        try:
            with tile.physical_path.open("rb") as stream:
                values.fromfile(
                    stream,
                    len(tile.x_coordinates)
                    * len(tile.y_coordinates),
                )
                if stream.read(1):
                    raise CompositionSourceError(
                        f"placement height tile {tile.tile_ref} has trailing bytes"
                    )
        except (EOFError, OSError) as error:
            raise CompositionSourceError(
                f"placement height tile {tile.tile_ref} cannot be sampled"
            ) from error
        if sys.byteorder != "little":
            values.byteswap()
        if len(values) != (
            len(tile.x_coordinates) * len(tile.y_coordinates)
        ):
            raise CompositionSourceError(
                f"placement height tile {tile.tile_ref} changed shape"
            )
        self._cache[tile.tile_ref] = values
        self._cache.move_to_end(tile.tile_ref)
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        self.peak_cached_tiles = max(
            self.peak_cached_tiles, len(self._cache)
        )
        return values

    def elevation(self, x: float, y: float) -> float:
        tile = self._tile(x, y)
        values = self._values(tile)
        xs = tile.x_coordinates
        ys = tile.y_coordinates
        column = min(
            max(bisect.bisect_right(xs, x) - 1, 0), len(xs) - 2
        )
        row = min(
            max(bisect.bisect_right(ys, y) - 1, 0), len(ys) - 2
        )
        x0, x1 = xs[column], xs[column + 1]
        y0, y1 = ys[row], ys[row + 1]
        fx = (min(max(x, x0), x1) - x0) / (x1 - x0)
        fy = (min(max(y, y0), y1) - y0) / (y1 - y0)
        width = len(xs)
        z00 = values[row * width + column]
        z10 = values[row * width + column + 1]
        z01 = values[(row + 1) * width + column]
        z11 = values[(row + 1) * width + column + 1]
        return (
            z00 * (1.0 - fx) * (1.0 - fy)
            + z10 * fx * (1.0 - fy)
            + z01 * (1.0 - fx) * fy
            + z11 * fx * fy
        )


def _terrain_scene_bounds(
    terrain: Sequence[_PreparedTerrainGrid],
) -> dict[str, float]:
    bounds = {
        "min_x": min(tile.local_bounds["min_x"] for tile in terrain),
        "min_y": min(tile.local_bounds["min_y"] for tile in terrain),
        "max_x": max(tile.local_bounds["max_x"] for tile in terrain),
        "max_y": max(tile.local_bounds["max_y"] for tile in terrain),
    }
    expected_area = (
        bounds["max_x"] - bounds["min_x"]
    ) * (bounds["max_y"] - bounds["min_y"])
    area = sum(
        (tile.local_bounds["max_x"] - tile.local_bounds["min_x"])
        * (tile.local_bounds["max_y"] - tile.local_bounds["min_y"])
        for tile in terrain
    )
    if not math.isclose(area, expected_area, rel_tol=0.0, abs_tol=1.0e-3):
        raise CompositionSourceError(
            "native terrain tiles do not partition one complete scene rectangle"
        )
    for index, tile in enumerate(terrain):
        for other in terrain[index + 1 :]:
            overlap_x = min(
                tile.local_bounds["max_x"], other.local_bounds["max_x"]
            ) - max(tile.local_bounds["min_x"], other.local_bounds["min_x"])
            overlap_y = min(
                tile.local_bounds["max_y"], other.local_bounds["max_y"]
            ) - max(tile.local_bounds["min_y"], other.local_bounds["min_y"])
            if overlap_x > 1.0e-6 and overlap_y > 1.0e-6:
                raise CompositionSourceError(
                    f"native terrain tiles {tile.tile_ref} and "
                    f"{other.tile_ref} overlap"
                )
    return bounds


def _grid_elevation(
    tile: _PreparedTerrainGrid,
    x: float,
    y: float,
) -> float:
    xs = tile.x_coordinates
    ys = tile.y_coordinates
    clamped_x = min(max(x, xs[0]), xs[-1])
    clamped_y = min(max(y, ys[0]), ys[-1])
    column = min(max(bisect.bisect_right(xs, clamped_x) - 1, 0), len(xs) - 2)
    row = min(max(bisect.bisect_right(ys, clamped_y) - 1, 0), len(ys) - 2)
    x0 = xs[column]
    x1 = xs[column + 1]
    y0 = ys[row]
    y1 = ys[row + 1]
    fx = (clamped_x - x0) / (x1 - x0)
    fy = (clamped_y - y0) / (y1 - y0)
    z00 = tile.elevations[row][column]
    z10 = tile.elevations[row][column + 1]
    z01 = tile.elevations[row + 1][column]
    z11 = tile.elevations[row + 1][column + 1]
    return (
        z00 * (1.0 - fx) * (1.0 - fy)
        + z10 * fx * (1.0 - fy)
        + z01 * (1.0 - fx) * fy
        + z11 * fx * fy
    )


def _height_rows(
    *,
    terrain: Sequence[_PreparedTerrainGrid],
    bounds: Mapping[str, float],
    spacing_m: float,
    width: int,
    height: int,
) -> Iterator[tuple[float, ...]]:
    x_starts = sorted({tile.local_bounds["min_x"] for tile in terrain})
    y_starts = sorted({tile.local_bounds["min_y"] for tile in terrain})
    by_origin = {
        (tile.local_bounds["min_x"], tile.local_bounds["min_y"]): tile
        for tile in terrain
    }
    if len(x_starts) * len(y_starts) != len(terrain):
        raise CompositionSourceError(
            "native terrain tiles do not form a complete regular tile lattice"
        )
    for y_index in range(height):
        y = min(
            bounds["min_y"] + y_index * spacing_m,
            bounds["max_y"],
        )
        row_values: list[float] = []
        y_cell = min(
            max(bisect.bisect_right(y_starts, y) - 1, 0),
            len(y_starts) - 1,
        )
        for x_index in range(width):
            x = min(
                bounds["min_x"] + x_index * spacing_m,
                bounds["max_x"],
            )
            x_cell = min(
                max(bisect.bisect_right(x_starts, x) - 1, 0),
                len(x_starts) - 1,
            )
            tile = by_origin.get((x_starts[x_cell], y_starts[y_cell]))
            if tile is None:
                raise CompositionSourceError(
                    "native terrain lattice has an uncovered height sample"
                )
            row_values.append(_grid_elevation(tile, x, y))
        yield tuple(row_values)


def _write_prepared_height_field(
    *,
    physical_path: Path,
    final_path: Path,
    volume_root: Path,
    terrain: Sequence[_PreparedTerrainGrid],
    bounds: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, float | int]]:
    span_x = bounds["max_x"] - bounds["min_x"]
    span_y = bounds["max_y"] - bounds["min_y"]
    maximum_span = max(span_x, span_y)
    spacing = maximum_span / float(MAX_HEIGHT_FIELD_SIDE - 1)
    width = int(math.ceil(span_x / spacing - 1.0e-12)) + 1
    height = int(math.ceil(span_y / spacing - 1.0e-12)) + 1
    if (
        width > MAX_HEIGHT_FIELD_SIDE
        or height > MAX_HEIGHT_FIELD_SIDE
        or width * height > MAX_HEIGHT_FIELD_SAMPLES
    ):
        raise CompositionSourceError(
            "native terrain cannot fit the bounded portable height field"
        )
    return _write_height_field(
        physical_path=physical_path,
        final_path=final_path,
        volume_root=volume_root,
        source=HeightFieldSource(
            origin_x=bounds["min_x"],
            origin_y=bounds["min_y"],
            spacing_m=spacing,
            samples=_height_rows(
                terrain=terrain,
                bounds=bounds,
                spacing_m=spacing,
                width=width,
                height=height,
            ),
        ),
        bounds=bounds,
    )


def _deduplicate_polyline(
    points: Iterable[Sequence[float]],
) -> list[list[float]]:
    result: list[list[float]] = []
    for point in points:
        current = [float(point[0]), float(point[1]), float(point[2])]
        if not result or math.dist(current, result[-1]) > 1.0e-7:
            result.append(current)
    return result


def _corridor_sides(
    points: Sequence[Sequence[float]],
    widths: Sequence[float],
) -> tuple[list[list[float]], list[list[float]]]:
    if len(points) < 2 or len(widths) != len(points):
        raise CompositionSourceError(
            "watercourse corridor needs one width per centreline point"
        )
    left: list[list[float]] = []
    right: list[list[float]] = []
    for index, point in enumerate(points):
        before = points[max(0, index - 1)]
        after = points[min(len(points) - 1, index + 1)]
        dx = float(after[0]) - float(before[0])
        dy = float(after[1]) - float(before[1])
        length = math.hypot(dx, dy)
        if length <= 1.0e-9:
            raise CompositionSourceError(
                "watercourse contains an unresolved zero-length tangent"
            )
        half_width = _positive(
            widths[index], label=f"watercourse widths[{index}]"
        ) * 0.5
        normal_x = -dy / length * half_width
        normal_y = dx / length * half_width
        left.append(
            [
                float(point[0]) + normal_x,
                float(point[1]) + normal_y,
                float(point[2]),
            ]
        )
        right.append(
            [
                float(point[0]) - normal_x,
                float(point[1]) - normal_y,
                float(point[2]),
            ]
        )
    return left, right


def _corridor_outline(
    points: Sequence[Sequence[float]],
    widths: Sequence[float],
) -> list[list[float]]:
    left, right = _corridor_sides(points, widths)
    outline = left + list(reversed(right))
    area = sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(outline, outline[1:] + outline[:1])
    )
    if abs(area) <= 1.0e-6:
        raise CompositionSourceError(
            "watercourse source widths produce a degenerate corridor"
        )
    return [[point[0], point[1]] for point in outline]


def _mesh_vertex_normals(
    points: Sequence[Sequence[float]],
    counts: Sequence[int],
    indices: Sequence[int],
) -> list[tuple[float, float, float]]:
    """Compute finite upward-facing vertex normals for polygonal water."""

    accumulators = [[0.0, 0.0, 0.0] for _ in points]
    cursor = 0
    for face_count in counts:
        face = list(indices[cursor : cursor + face_count])
        cursor += face_count
        if face_count < 3:
            raise CompositionSourceError(
                "water mesh contains a face with fewer than three vertices"
            )
        anchor = points[face[0]]
        for offset in range(1, face_count - 1):
            second = points[face[offset]]
            third = points[face[offset + 1]]
            first_vector = (
                float(second[0]) - float(anchor[0]),
                float(second[1]) - float(anchor[1]),
                float(second[2]) - float(anchor[2]),
            )
            second_vector = (
                float(third[0]) - float(anchor[0]),
                float(third[1]) - float(anchor[1]),
                float(third[2]) - float(anchor[2]),
            )
            normal = [
                first_vector[1] * second_vector[2]
                - first_vector[2] * second_vector[1],
                first_vector[2] * second_vector[0]
                - first_vector[0] * second_vector[2],
                first_vector[0] * second_vector[1]
                - first_vector[1] * second_vector[0],
            ]
            if normal[2] < 0.0:
                normal = [-value for value in normal]
            if math.sqrt(sum(value * value for value in normal)) <= 1.0e-12:
                continue
            for vertex_index in (face[0], face[offset], face[offset + 1]):
                for axis in range(3):
                    accumulators[vertex_index][axis] += normal[axis]
    result: list[tuple[float, float, float]] = []
    for accumulator in accumulators:
        length = math.sqrt(sum(value * value for value in accumulator))
        if length <= 1.0e-12:
            result.append((0.0, 0.0, 1.0))
        else:
            result.append(
                tuple(value / length for value in accumulator)  # type: ignore[arg-type]
            )
    return result


def _point_segment_distance_raw(
    point: Sequence[float],
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1.0e-12:
        return math.hypot(point[0] - first[0], point[1] - first[1])
    ratio = (
        (point[0] - first[0]) * dx + (point[1] - first[1]) * dy
    ) / length_squared
    ratio = min(max(ratio, 0.0), 1.0)
    return math.hypot(
        point[0] - (first[0] + ratio * dx),
        point[1] - (first[1] + ratio * dy),
    )


def _orientation_raw(
    first: Sequence[float],
    second: Sequence[float],
    third: Sequence[float],
) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _segments_intersect_raw(
    first_a: Sequence[float],
    first_b: Sequence[float],
    second_a: Sequence[float],
    second_b: Sequence[float],
) -> bool:
    values = (
        _orientation_raw(first_a, first_b, second_a),
        _orientation_raw(first_a, first_b, second_b),
        _orientation_raw(second_a, second_b, first_a),
        _orientation_raw(second_a, second_b, first_b),
    )
    if (
        (values[0] > 1.0e-9 and values[1] < -1.0e-9)
        or (values[0] < -1.0e-9 and values[1] > 1.0e-9)
    ) and (
        (values[2] > 1.0e-9 and values[3] < -1.0e-9)
        or (values[2] < -1.0e-9 and values[3] > 1.0e-9)
    ):
        return True
    return min(
        _point_segment_distance_raw(second_a, first_a, first_b),
        _point_segment_distance_raw(second_b, first_a, first_b),
        _point_segment_distance_raw(first_a, second_a, second_b),
        _point_segment_distance_raw(first_b, second_a, second_b),
    ) <= 1.0e-9


def _segment_distance_raw(
    first_a: Sequence[float],
    first_b: Sequence[float],
    second_a: Sequence[float],
    second_b: Sequence[float],
) -> float:
    if _segments_intersect_raw(first_a, first_b, second_a, second_b):
        return 0.0
    return min(
        _point_segment_distance_raw(first_a, second_a, second_b),
        _point_segment_distance_raw(first_b, second_a, second_b),
        _point_segment_distance_raw(second_a, first_a, first_b),
        _point_segment_distance_raw(second_b, first_a, first_b),
    )


def _point_in_polygon_raw(
    point: Sequence[float],
    polygon: Sequence[Sequence[float]],
) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (
            (current[1] > point[1]) != (previous[1] > point[1])
            and point[0]
            < (
                (previous[0] - current[0])
                * (point[1] - current[1])
                / (previous[1] - current[1])
                + current[0]
            )
        ):
            inside = not inside
        previous = current
    return inside


def _segment_polygon_distance_raw(
    first: Sequence[float],
    second: Sequence[float],
    polygon: Sequence[Sequence[float]],
) -> float:
    if (
        _point_in_polygon_raw(first, polygon)
        or _point_in_polygon_raw(second, polygon)
    ):
        return 0.0
    return min(
        _segment_distance_raw(
            first,
            second,
            edge_start,
            edge_end,
        )
        for edge_start, edge_end in zip(
            polygon, polygon[1:] + polygon[:1]  # type: ignore[operator]
        )
    )


def _segment_polygon_boundary_parameters(
    first: Sequence[float],
    second: Sequence[float],
    polygon: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Return exact normalized intersections with a polygon boundary."""

    dx = float(second[0]) - float(first[0])
    dy = float(second[1]) - float(first[1])
    length_squared = dx * dx + dy * dy
    if length_squared <= 1.0e-12:
        return ()
    values: list[float] = []
    for edge_start, edge_end in zip(
        polygon, polygon[1:] + polygon[:1]  # type: ignore[operator]
    ):
        sx = float(edge_end[0]) - float(edge_start[0])
        sy = float(edge_end[1]) - float(edge_start[1])
        offset_x = float(edge_start[0]) - float(first[0])
        offset_y = float(edge_start[1]) - float(first[1])
        denominator = dx * sy - dy * sx
        if abs(denominator) > 1.0e-12:
            route_fraction = (
                offset_x * sy - offset_y * sx
            ) / denominator
            edge_fraction = (
                offset_x * dy - offset_y * dx
            ) / denominator
            if (
                -1.0e-9 <= route_fraction <= 1.0 + 1.0e-9
                and -1.0e-9 <= edge_fraction <= 1.0 + 1.0e-9
            ):
                values.append(min(max(route_fraction, 0.0), 1.0))
            continue
        if abs(offset_x * dy - offset_y * dx) > 1.0e-9:
            continue
        for point in (edge_start, edge_end):
            values.append(
                (
                    (float(point[0]) - float(first[0])) * dx
                    + (float(point[1]) - float(first[1])) * dy
                )
                / length_squared
            )
    result: list[float] = []
    for value in sorted(values):
        if not -1.0e-9 <= value <= 1.0 + 1.0e-9:
            continue
        normalized = min(max(value, 0.0), 1.0)
        if not result or abs(normalized - result[-1]) > 1.0e-9:
            result.append(normalized)
    return tuple(result)


def _point_polygon_distance_raw(
    point: Sequence[float],
    polygon: Sequence[Sequence[float]],
) -> float:
    if _point_in_polygon_raw(point, polygon):
        return 0.0
    return min(
        _point_segment_distance_raw(point, first, second)
        for first, second in zip(
            polygon, polygon[1:] + polygon[:1]  # type: ignore[operator]
        )
    )


def _segment_crosses_polygon_interior_raw(
    first: Sequence[float],
    second: Sequence[float],
    polygon: Sequence[Sequence[float]],
) -> bool:
    """Distinguish a real water traversal from tangent/boundary contact."""

    parameters = (
        0.0,
        *_segment_polygon_boundary_parameters(first, second, polygon),
        1.0,
    )
    ordered = sorted(
        {
            min(max(float(value), 0.0), 1.0)
            for value in parameters
        }
    )
    for start, end in zip(ordered, ordered[1:]):
        if end - start <= 1.0e-10:
            continue
        midpoint = _interpolate_route_point(
            (float(first[0]), float(first[1]), 0.0),
            (float(second[0]), float(second[1]), 0.0),
            (start + end) * 0.5,
        )
        if _point_in_polygon_raw(midpoint, polygon):
            return True
    return False


def _interpolate_route_point(
    first: Sequence[float],
    second: Sequence[float],
    fraction: float,
) -> list[float]:
    return [
        float(first[index])
        + (float(second[index]) - float(first[index])) * fraction
        for index in range(3)
    ]


def _ensure_native_bridge_approaches(
    *,
    routes: list[dict[str, object]],
    waters: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    """Split continuous source segments so every crossing has two approaches.

    Routes reaching this function come from the continuous locked BDTOPO
    lines, not from per-tile HERO ribbons.  Only exact interpolation on the
    route's own source segment is permitted.  A line clipped by the *scene*
    boundary while already inside water fails closed because a complete bridge
    cannot be authored without geometry outside the accepted scene.
    """

    if not routes:
        return {"source_interpolated_approaches": 0}

    interpolated = 0
    for route_index, route in enumerate(routes):
        raw_points = route.get("points")
        if not isinstance(raw_points, (list, tuple)) or len(raw_points) < 2:
            raise CompositionSourceError(
                f"prepared route {route_index} has no bridge-ready points"
            )
        points = [
            [float(point[0]), float(point[1]), float(point[2])]
            for point in raw_points
        ]
        collision_clearance = (
            float(_fixed_variant_constraints()["road_water_buffer_m"])
            + _positive(
                route.get("width_m"),
                label=f"prepared route {route_index}.width_m",
            )
            * 0.5
        )
        for at_start in (True, False):
            first, second = (
                (points[0], points[1])
                if at_start
                else (points[-2], points[-1])
            )
            crossing_polygons = [
                outline
                for water in waters
                if isinstance((outline := water.get("outline")), list)
                and len(outline) >= 3
                and _segment_crosses_polygon_interior_raw(
                    first, second, outline
                )
            ]
            if not crossing_polygons:
                continue
            endpoint = first if at_start else second
            if any(
                _point_polygon_distance_raw(endpoint, polygon)
                <= collision_clearance + 1.0e-9
                for polygon in crossing_polygons
            ):
                raise CompositionSourceError(
                    f"continuous route {route.get('stable_id')} reaches the "
                    "scene boundary inside the reviewed water-clearance zone; "
                    "a complete source-backed bridge cannot be authored"
                )
            clearance_parameters: list[float] = []
            for polygon in crossing_polygons:
                boundaries = [
                    value
                    for value in _segment_polygon_boundary_parameters(
                        first, second, polygon
                    )
                    if 1.0e-9 < value < 1.0 - 1.0e-9
                ]
                if not boundaries:
                    raise CompositionSourceError(
                        f"route {route.get('stable_id')} intersects water "
                        "without a resolvable source boundary"
                    )
                if at_start:
                    outside = 0.0
                    inside = min(boundaries)
                    for _ in range(64):
                        midpoint = (outside + inside) * 0.5
                        point = _interpolate_route_point(
                            first, second, midpoint
                        )
                        if (
                            _point_polygon_distance_raw(point, polygon)
                            <= collision_clearance
                        ):
                            inside = midpoint
                        else:
                            outside = midpoint
                    clearance_parameters.append(inside)
                else:
                    inside = max(boundaries)
                    outside = 1.0
                    for _ in range(64):
                        midpoint = (inside + outside) * 0.5
                        point = _interpolate_route_point(
                            first, second, midpoint
                        )
                        if (
                            _point_polygon_distance_raw(point, polygon)
                            <= collision_clearance
                        ):
                            inside = midpoint
                        else:
                            outside = midpoint
                    clearance_parameters.append(outside)
            fraction = (
                min(clearance_parameters)
                if at_start
                else max(clearance_parameters)
            )
            if not 1.0e-9 < fraction < 1.0 - 1.0e-9:
                raise CompositionSourceError(
                    f"route {route.get('stable_id')} lacks a finite bridge "
                    "approach outside the reviewed water clearance"
                )
            point = _interpolate_route_point(first, second, fraction)
            if at_start:
                points.insert(1, point)
            else:
                points.insert(len(points) - 1, point)
            interpolated += 1
        route["points"] = points
    return {"source_interpolated_approaches": interpolated}


def _derive_native_bridge_spans(
    *,
    routes: list[dict[str, object]],
    waters: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Bind native route/water intersections to reviewed bridge geometry."""

    span_count = 0
    reviewed_clearance = float(
        _fixed_variant_constraints()["minimum_bridge_deck_clearance_m"]
    )
    approach_metrics = _ensure_native_bridge_approaches(
        routes=routes,
        waters=waters,
    )
    for route_index, route in enumerate(routes):
        raw_points = route.get("points")
        if not isinstance(raw_points, list) or len(raw_points) < 2:
            raise CompositionSourceError(
                f"prepared route {route_index} has no bridge-ready points"
            )
        _positive(
            route.get("width_m"),
            label=f"prepared route {route_index}.width_m",
        )
        lengths = [
            math.hypot(
                float(second[0]) - float(first[0]),
                float(second[1]) - float(first[1]),
            )
            for first, second in zip(raw_points, raw_points[1:])
        ]
        if any(length <= 1.0e-9 for length in lengths):
            raise CompositionSourceError(
                f"prepared route {route_index} has a zero-length segment"
            )
        cumulative = [0.0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        total = cumulative[-1]
        crossing_waters: list[list[Mapping[str, object]]] = []
        for first, second in zip(raw_points, raw_points[1:]):
            matches: list[Mapping[str, object]] = []
            for water in waters:
                outline = water.get("outline")
                if not isinstance(outline, list) or len(outline) < 3:
                    raise CompositionSourceError(
                        "prepared water outline is not bridge-ready"
                    )
                # A bridge is source-backed only where the route centreline
                # actually intersects the water footprint. Proximity to a bank
                # is a clearance check, never evidence of a bridge.
                if _segment_crosses_polygon_interior_raw(
                    first, second, outline
                ):
                    matches.append(water)
            crossing_waters.append(matches)
        spans: list[dict[str, object]] = []
        segment_index = 0
        while segment_index < len(crossing_waters):
            if not crossing_waters[segment_index]:
                segment_index += 1
                continue
            first_crossing = segment_index
            while (
                segment_index < len(crossing_waters)
                and crossing_waters[segment_index]
            ):
                segment_index += 1
            last_crossing = segment_index - 1
            if first_crossing == 0 or last_crossing == len(lengths) - 1:
                raise CompositionSourceError(
                    f"route {route.get('stable_id')} crosses water at a clipped "
                    "endpoint and has no source approach for a valid bridge"
                )
            water_start = cumulative[first_crossing] / total
            water_end = cumulative[last_crossing + 1] / total
            start = (
                cumulative[first_crossing - 1] / total + water_start
            ) * 0.5
            end = (
                water_end + cumulative[last_crossing + 2] / total
            ) * 0.5
            span_number = len(spans) + 1
            spans.append(
                {
                    "stable_id": (
                        f"{route.get('stable_id')}:bridge:{span_number:03d}"
                    ),
                    "start_fraction": start,
                    "water_start_fraction": water_start,
                    "water_end_fraction": water_end,
                    "end_fraction": end,
                    "minimum_deck_clearance_m": reviewed_clearance,
                }
            )
            span_count += 1
        route["bridge_spans"] = spans
    return {
        "algorithm": "native-route-water-segment-intersection-v1",
        "bridge_span_count": span_count,
        "reviewed_minimum_deck_clearance_m": reviewed_clearance,
        "clearance_authoring": (
            "variant_draping_then_final_geometry_revalidation"
        ),
        **approach_metrics,
    }


def _native_point_coordinates(
    prim: object,
    raw_points: Sequence[object],
    *,
    xform_cache: object,
    gf: object,
    label: str,
) -> list[list[float]]:
    matrix = xform_cache.GetLocalToWorldTransform(prim)
    result: list[list[float]] = []
    for index, point in enumerate(raw_points):
        try:
            transformed = matrix.Transform(
                gf.Vec3d(
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                )
            )
            coordinates = [
                float(transformed[0]),
                float(transformed[1]),
                float(transformed[2]),
            ]
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise CompositionSourceError(
                f"{label} point {index} is malformed"
            ) from error
        if any(not math.isfinite(value) for value in coordinates):
            raise CompositionSourceError(
                f"{label} point {index} is non-finite"
            )
        result.append(coordinates)
    return result


def _author_metric_uv(
    geometry: object,
    *,
    points: Sequence[Sequence[float]],
    metres_per_uv_tile: float,
    usd_geom: object,
    sdf: object,
    gf: object,
) -> None:
    primvar = usd_geom.PrimvarsAPI(geometry).CreatePrimvar(
        "st",
        sdf.ValueTypeNames.TexCoord2fArray,
        usd_geom.Tokens.vertex,
    )
    primvar.Set(
        [
            gf.Vec2f(
                float(point[0]) / metres_per_uv_tile,
                float(point[1]) / metres_per_uv_tile,
            )
            for point in points
        ]
    )


def _collinear_segment_overlap_length(
    first_a: Sequence[float],
    first_b: Sequence[float],
    second_a: Sequence[float],
    second_b: Sequence[float],
) -> float:
    first_dx = float(first_b[0]) - float(first_a[0])
    first_dy = float(first_b[1]) - float(first_a[1])
    second_dx = float(second_b[0]) - float(second_a[0])
    second_dy = float(second_b[1]) - float(second_a[1])
    first_length = math.hypot(first_dx, first_dy)
    second_length = math.hypot(second_dx, second_dy)
    if first_length <= 1.0e-9 or second_length <= 1.0e-9:
        return 0.0
    direction_cross = abs(
        first_dx * second_dy - first_dy * second_dx
    ) / (first_length * second_length)
    if direction_cross > 1.0e-8:
        return 0.0
    line_distance = abs(
        (float(second_a[0]) - float(first_a[0])) * first_dy
        - (float(second_a[1]) - float(first_a[1])) * first_dx
    ) / first_length
    if line_distance > 0.01:
        return 0.0
    unit_x = first_dx / first_length
    unit_y = first_dy / first_length

    def projection(point: Sequence[float]) -> float:
        return (
            (float(point[0]) - float(first_a[0])) * unit_x
            + (float(point[1]) - float(first_a[1])) * unit_y
        )

    second_interval = sorted(
        (projection(second_a), projection(second_b))
    )
    return max(
        0.0,
        min(first_length, second_interval[1])
        - max(0.0, second_interval[0]),
    )


def _validate_no_route_segment_overlap(
    routes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Reject duplicated collinear interiors while allowing point junctions.

    The check intentionally includes non-identical segments belonging to the
    same source route. A self-retracing centreline would otherwise author the
    same road surface twice and recreate the z-fighting this gate prevents.
    """

    cell_size = 250.0
    cells: dict[
        tuple[int, int],
        list[
            tuple[
                int,
                int,
                Sequence[float],
                Sequence[float],
            ]
        ],
    ] = {}
    segment_id = 0
    comparisons = 0
    maximum_overlap = 0.0
    for route_index, route in enumerate(routes):
        points = route.get("points")
        if not isinstance(points, list):
            raise CompositionSourceError(
                f"source route {route_index} has no points"
            )
        for first, second in zip(points, points[1:]):
            min_x = math.floor(
                min(float(first[0]), float(second[0])) / cell_size
            )
            max_x = math.floor(
                max(float(first[0]), float(second[0])) / cell_size
            )
            min_y = math.floor(
                min(float(first[1]), float(second[1])) / cell_size
            )
            max_y = math.floor(
                max(float(first[1]), float(second[1])) / cell_size
            )
            candidate_segments: dict[
                int,
                tuple[
                    int,
                    int,
                    Sequence[float],
                    Sequence[float],
                ],
            ] = {}
            for cell_x in range(min_x, max_x + 1):
                for cell_y in range(min_y, max_y + 1):
                    for candidate in cells.get((cell_x, cell_y), ()):
                        candidate_segments[candidate[0]] = candidate
            for (
                _candidate_id,
                candidate_route,
                candidate_first,
                candidate_second,
            ) in candidate_segments.values():
                comparisons += 1
                overlap = _collinear_segment_overlap_length(
                    first,
                    second,
                    candidate_first,
                    candidate_second,
                )
                maximum_overlap = max(maximum_overlap, overlap)
                if overlap > 0.01:
                    raise CompositionSourceError(
                        "locked BDTOPO routes contain duplicated collinear "
                        f"interiors ({overlap:.6f} m) between route "
                        f"{candidate_route} and {route_index}"
                    )
            record = (segment_id, route_index, first, second)
            for cell_x in range(min_x, max_x + 1):
                for cell_y in range(min_y, max_y + 1):
                    cells.setdefault((cell_x, cell_y), []).append(record)
            segment_id += 1
    return {
        "algorithm": "spatial-collinear-interior-overlap-gate-v1",
        "segment_count": segment_id,
        "candidate_comparison_count": comparisons,
        "maximum_interroute_collinear_overlap_m": maximum_overlap,
        "maximum_allowed_interroute_collinear_overlap_m": 0.01,
    }


def _source_backed_routes(
    *,
    volume_root: Path,
    zone_root: Path,
    source_lock: Mapping[str, object],
    epsg2154_bounds: tuple[float, float, float, float],
    root_origin: tuple[float, float],
    placement_tiles: Sequence[_PreparedPlacementHeight],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build continuous route identities from the exact locked BDTOPO lines."""

    road_paths = _locked_vector_paths(
        zone_root=zone_root,
        source_lock=source_lock,
        category="roads",
    )
    sampler = _PlacementHeightSampler(placement_tiles, cache_limit=2)
    pending: list[tuple[str, dict[str, object]]] = []
    source_features = 0
    source_lines = 0
    for source_index, path in enumerate(road_paths):
        collection = _read_json(path, label="locked BDTOPO road GeoJSON")
        features = collection.get("features")
        if not isinstance(features, list):
            raise CompositionSourceError(
                f"locked road GeoJSON has no features: {path}"
            )
        source_sha = _sha256(path)
        for feature_index, feature in enumerate(features):
            if not isinstance(feature, Mapping):
                continue
            geometry = feature.get("geometry")
            lines = list(_geojson_lines(geometry))
            if not lines:
                continue
            source_features += 1
            properties = feature.get("properties")
            raw_width = (
                properties.get("largeur_de_chaussee")
                if isinstance(properties, Mapping)
                else None
            )
            try:
                width = float(raw_width)
            except (TypeError, ValueError):
                width = 4.5
            if not math.isfinite(width):
                width = 4.5
            width = max(2.0, min(16.0, width))
            raw_feature_id = feature.get("id")
            if isinstance(raw_feature_id, str) and raw_feature_id.strip():
                feature_id = raw_feature_id.strip()
            else:
                feature_id = "geometry-" + hashlib.sha256(
                    json.dumps(
                        geometry,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
            for line_index, line in enumerate(lines):
                source_lines += 1
                fragments = _clip_line_axis_aligned(
                    line, bounds=epsg2154_bounds
                )
                for fragment_index, fragment in enumerate(fragments):
                    local_points = [
                        [
                            float(point[0]) - root_origin[0],
                            float(point[1]) - root_origin[1],
                        ]
                        for point in fragment
                    ]
                    points = [
                        [
                            point[0],
                            point[1],
                            sampler.elevation(point[0], point[1]),
                        ]
                        for point in local_points
                    ]
                    if len(points) < 2:
                        continue
                    identity = (
                        f"{source_sha}:{source_index}:{feature_id}:"
                        f"{feature_index}:{line_index}:{fragment_index}"
                    )
                    stable_hash = hashlib.sha256(
                        identity.encode("utf-8")
                    ).hexdigest()
                    stable_id = f"bdtopo-route-{stable_hash[:24]}"
                    pending.append(
                        (
                            stable_id,
                            {
                                "stable_id": stable_id,
                                "numeric_id": 0,
                                "family": "bdtopo_road",
                                "surface_class": "paved",
                                "width_m": width,
                                "points": points,
                                "bridge_spans": [],
                            },
                        )
                    )
                    if len(pending) > MAX_PREPARED_ROUTE_COUNT:
                        raise CompositionSourceError(
                            "locked BDTOPO route count exceeds the bounded "
                            "preparation limit"
                        )
    pending.sort(key=lambda item: item[0])
    if not pending or len({stable_id for stable_id, _ in pending}) != len(
        pending
    ):
        raise CompositionSourceError(
            "locked BDTOPO roads are empty or have unstable repeated identities"
        )
    routes: list[dict[str, object]] = []
    for index, (_stable_id, route) in enumerate(pending, start=1):
        numeric_id = (7 << 56) | index
        if numeric_id > MAX_SIGNED_INT64:
            raise CompositionSourceError(
                "locked BDTOPO route numeric identity exceeds signed int64"
            )
        route["numeric_id"] = numeric_id
        routes.append(route)
    overlap_metrics = _validate_no_route_segment_overlap(routes)
    return (
        routes,
        {
            "route_geometry_authority": "locked_continuous_bdtopo_lines",
            "locked_source_artifacts": [
                {
                    "path": path.relative_to(volume_root).as_posix(),
                    "sha256": _sha256(path),
                }
                for path in road_paths
            ],
            "source_feature_count": source_features,
            "source_line_count": source_lines,
            "prepared_route_count": len(routes),
            "placement_height_cache_tile_limit": 2,
            "placement_height_peak_cached_tiles": sampler.peak_cached_tiles,
            "overlap_validation": overlap_metrics,
        },
    )


def _extract_native_networks_and_water_payload(
    *,
    volume_root: Path,
    zone_root: Path,
    build: Mapping[str, object],
    physical_water_path: Path,
    final_water_path: Path,
    water_material: Mapping[str, str],
    water_uv_metres: float,
    accepted_tree_families: set[str],
    source_lock: Mapping[str, object],
    epsg2154_bounds: tuple[float, float, float, float],
    root_origin: tuple[float, float],
    placement_tiles: Sequence[_PreparedPlacementHeight],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    _NativeSuitabilityObservations,
]:
    """Verify HERO layers, derive continuous roads and author visible water."""

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError) as error:
        raise CompositionSourceError(
            "native route/water preparation requires pinned Kit/Isaac pxr"
        ) from error
    details = build.get("detail_payloads")
    coverage = build.get("tile_coverage")
    if (
        not isinstance(details, list)
        or len(details) != 400
        or not isinstance(coverage, list)
        or len(coverage) != 400
    ):
        raise CompositionSourceError(
            "network preparation requires exact 400 HERO detail payloads"
        )
    coverage_by_detail: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(coverage):
        if not isinstance(raw, Mapping):
            raise CompositionSourceError(
                f"tile_coverage[{index}] is malformed"
            )
        detail_path = _portable_receipt_path(
            raw.get("detail_payload"),
            label=f"tile_coverage[{index}].detail_payload",
        )
        if detail_path in coverage_by_detail:
            raise CompositionSourceError(
                f"HERO detail coverage repeats: {detail_path}"
            )
        coverage_by_detail[detail_path] = raw

    physical_water_path.parent.mkdir(parents=True, exist_ok=True)
    water_stage = Usd.Stage.CreateNew(str(physical_water_path))
    if water_stage is None:
        raise CompositionSourceError("cannot create isolated water USD")
    UsdGeom.SetStageMetersPerUnit(water_stage, 1.0)
    UsdGeom.SetStageUpAxis(water_stage, UsdGeom.Tokens.z)
    water_root = UsdGeom.Xform.Define(water_stage, "/Water")
    water_stage.SetDefaultPrim(water_root.GetPrim())
    water_root.GetPrim().SetCustomDataByKey(
        "fireviewer:content_role", "isolated_visible_water"
    )
    looks = UsdGeom.Scope.Define(water_stage, "/Water/Looks")
    looks.GetPrim().SetCustomDataByKey("fireviewer:layer", "water_material")
    material_proxy = UsdShade.Material.Define(
        water_stage, "/Water/Looks/Water"
    )
    material_path = (
        volume_root / str(water_material.get("path", ""))
    ).resolve()
    material_prim = _trimmed(
        water_material.get("prim_path"),
        label="water material prim_path",
    )
    if (
        not _is_below(volume_root, material_path)
        or not material_path.is_file()
        or _sha256(material_path) != water_material.get("sha256")
    ):
        raise CompositionSourceError(
            "isolated water material artifact is absent or stale"
        )
    relative_material = os.path.relpath(
        material_path, final_water_path.parent
    ).replace("\\", "/")
    material_proxy.GetPrim().GetReferences().AddReference(
        relative_material, material_prim
    )

    routes: list[dict[str, object]] = []
    waters: list[dict[str, object]] = []
    native_road_fragment_count = 0
    tree_families_by_tile: dict[str, tuple[str, ...]] = {}
    building_group_bounds: dict[
        str, tuple[float, float, float, float, int]
    ] = {}
    for payload_index, raw in enumerate(details):
        relative, expected_sha = _receipt_artifact_tuple(
            raw, label=f"native build detail_payloads[{payload_index}]"
        )
        coverage_record = coverage_by_detail.get(relative)
        if coverage_record is None:
            raise CompositionSourceError(
                f"HERO detail {relative} has no exact coverage record"
            )
        tile_ref = _trimmed(
            coverage_record.get("tile_ref"),
            label=f"detail coverage {relative}.tile_ref",
        )
        namespace = _positive_id(
            coverage_record.get("instance_namespace"),
            label=f"detail coverage {relative}.instance_namespace",
        )
        path = (zone_root / relative).resolve()
        if (
            not _is_below(zone_root, path)
            or not _is_below(volume_root, path)
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size > MAX_DETAIL_PAYLOAD_BYTES
            or _sha256(path) != expected_sha
        ):
            raise CompositionSourceError(
                f"HERO detail payload is absent, oversized or stale: {relative}"
            )
        stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
        if stage is None:
            raise CompositionSourceError(
                f"HERO detail payload cannot be opened: {relative}"
            )
        detail = stage.GetPrimAtPath("/Detail")
        if (
            not detail
            or not detail.IsValid()
            or detail.GetCustomDataByKey("fireviewer:detail_level") != "HERO"
            or detail.GetCustomDataByKey("fireviewer:tile_ref") != tile_ref
            or int(
                detail.GetCustomDataByKey("fireviewer:instance_namespace") or 0
            )
            != namespace
        ):
            raise CompositionSourceError(
                f"HERO detail identity differs from coverage: {relative}"
            )
        tile_scope = UsdGeom.Xform.Define(
            water_stage, f"/Water/Tile_{namespace:07d}"
        )
        tile_scope.GetPrim().SetCustomDataByKey(
            "fireviewer:tile_ref", tile_ref
        )
        cache = UsdGeom.XformCache()
        observed_tree_families: set[str] = set()
        instance_prims = sorted(
            (
                prim
                for prim in stage.TraverseAll()
                if prim.IsA(UsdGeom.PointInstancer)
                and (
                    str(prim.GetPath()).startswith(
                        "/Detail/Vegetation/"
                    )
                    or str(prim.GetPath()).startswith(
                        "/Detail/Buildings/"
                    )
                )
            ),
            key=lambda prim: str(prim.GetPath()),
        )
        for prim in instance_prims:
            instancer = UsdGeom.PointInstancer(prim)
            positions = list(instancer.GetPositionsAttr().Get() or [])
            if not positions:
                continue
            prim_path = str(prim.GetPath())
            if prim_path.startswith("/Detail/Vegetation/"):
                proto_indices = [
                    int(value)
                    for value in list(
                        instancer.GetProtoIndicesAttr().Get() or []
                    )
                ]
                targets = list(
                    instancer.GetPrototypesRel().GetTargets() or []
                )
                if (
                    len(proto_indices) != len(positions)
                    or not targets
                    or any(
                        not 0 <= value < len(targets)
                        for value in proto_indices
                    )
                ):
                    raise CompositionSourceError(
                        f"{relative}:{prim_path} has incomplete tree prototypes"
                    )
                for prototype_index in sorted(set(proto_indices)):
                    prototype = stage.GetPrimAtPath(
                        targets[prototype_index]
                    )
                    raw_family = (
                        prototype.GetCustomDataByKey(
                            "fireviewer:asset_family"
                        )
                        if prototype and prototype.IsValid()
                        else None
                    )
                    if not isinstance(raw_family, str):
                        raise CompositionSourceError(
                            f"{relative}:{prim_path} prototype lacks asset family"
                        )
                    family = (
                        raw_family.removeprefix("vegetation.")
                    )
                    if family not in accepted_tree_families:
                        raise CompositionSourceError(
                            f"{relative}:{prim_path} uses unaccepted tree "
                            f"family {family}"
                        )
                    observed_tree_families.add(family)
                continue
            group_ids = _primvar_values(
                prim,
                name="fireviewer_group_id",
                count=len(positions),
                label=f"{relative}:{prim_path}",
            )
            radii = _primvar_values(
                prim,
                name="fireviewer_footprint_radius_m",
                count=len(positions),
                label=f"{relative}:{prim_path}",
            )
            world_positions = _native_point_coordinates(
                prim,
                positions,
                xform_cache=cache,
                gf=Gf,
                label=f"{relative}:{prim_path}",
            )
            for instance_index, (position, raw_group, raw_radius) in enumerate(
                zip(world_positions, group_ids, radii)
            ):
                group_id = _trimmed(
                    raw_group,
                    label=(
                        f"{relative}:{prim_path} instance "
                        f"{instance_index}.group_id"
                    ),
                )
                radius = _positive(
                    raw_radius,
                    label=(
                        f"{relative}:{prim_path} instance "
                        f"{instance_index}.footprint_radius_m"
                    ),
                )
                candidate = (
                    position[0] - radius,
                    position[1] - radius,
                    position[0] + radius,
                    position[1] + radius,
                    1,
                )
                current = building_group_bounds.get(group_id)
                building_group_bounds[group_id] = (
                    candidate
                    if current is None
                    else (
                        min(current[0], candidate[0]),
                        min(current[1], candidate[1]),
                        max(current[2], candidate[2]),
                        max(current[3], candidate[3]),
                        current[4] + 1,
                    )
                )
        tree_families_by_tile[tile_ref] = tuple(
            sorted(observed_tree_families)
        )
        hydro_prims = sorted(
            (
                prim
                for prim in stage.TraverseAll()
                if str(prim.GetPath()).startswith("/Detail/Hydrology/")
                and (
                    prim.IsA(UsdGeom.Mesh)
                    or prim.IsA(UsdGeom.BasisCurves)
                )
            ),
            key=lambda prim: str(prim.GetPath()),
        )
        for local_index, prim in enumerate(hydro_prims):
            stable_id = (
                f"tile-{namespace}:waters:"
                f"{local_index + 1}:{prim.GetName()}"
            )
            output_path = (
                f"/Water/Tile_{namespace:07d}/"
                f"Feature_{local_index:06d}"
            )
            if prim.IsA(UsdGeom.Mesh):
                source = UsdGeom.Mesh(prim)
                points = _native_point_coordinates(
                    prim,
                    list(source.GetPointsAttr().Get() or []),
                    xform_cache=cache,
                    gf=Gf,
                    label=f"{relative}:{prim.GetPath()}",
                )
                counts = [
                    int(value)
                    for value in list(
                        source.GetFaceVertexCountsAttr().Get() or []
                    )
                ]
                indices = [
                    int(value)
                    for value in list(
                        source.GetFaceVertexIndicesAttr().Get() or []
                    )
                ]
                if (
                    len(counts) != 1
                    or counts[0] < 3
                    or len(indices) != counts[0]
                    or any(not 0 <= value < len(points) for value in indices)
                ):
                    raise CompositionSourceError(
                        f"native water surface topology is unsupported: "
                        f"{relative}:{prim.GetPath()}"
                    )
                outline = [[points[index][0], points[index][1]] for index in indices]
                output = UsdGeom.Mesh.Define(water_stage, output_path)
                output.CreatePointsAttr(
                    [Gf.Vec3f(*point) for point in points]
                )
                output.CreateFaceVertexCountsAttr(counts)
                output.CreateFaceVertexIndicesAttr(indices)
                output.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                output.CreateNormalsAttr(
                    [
                        Gf.Vec3f(*normal)
                        for normal in _mesh_vertex_normals(
                            points, counts, indices
                        )
                    ]
                )
                output.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
                _author_metric_uv(
                    output,
                    points=points,
                    metres_per_uv_tile=water_uv_metres,
                    usd_geom=UsdGeom,
                    sdf=Sdf,
                    gf=Gf,
                )
                profile = [sum(point[2] for point in points) / len(points)]
                water_record: dict[str, object] = {
                    "stable_id": stable_id,
                    "family": "bdtopo_water_surface",
                    "outline": outline,
                    "kind": "standing",
                    "centreline": [],
                    "surface_profile_m": profile,
                }
            else:
                source = UsdGeom.BasisCurves(prim)
                points = _native_point_coordinates(
                    prim,
                    list(source.GetPointsAttr().Get() or []),
                    xform_cache=cache,
                    gf=Gf,
                    label=f"{relative}:{prim.GetPath()}",
                )
                counts = [
                    int(value)
                    for value in list(
                        source.GetCurveVertexCountsAttr().Get() or []
                    )
                ]
                widths = [
                    float(value)
                    for value in list(source.GetWidthsAttr().Get() or [])
                ]
                if (
                    counts != [len(points)]
                    or len(points) < 2
                    or len(widths) not in {1, len(points)}
                    or any(
                        not math.isfinite(value) or value <= 0.0
                        for value in widths
                    )
                ):
                    raise CompositionSourceError(
                        f"native watercourse topology/width is unsupported: "
                        f"{relative}:{prim.GetPath()}"
                    )
                if len(widths) == 1:
                    widths *= len(points)
                centreline = _deduplicate_polyline(points)
                if len(centreline) != len(points):
                    raise CompositionSourceError(
                        f"native watercourse contains duplicate points: "
                        f"{relative}:{prim.GetPath()}"
                    )
                outline = _corridor_outline(centreline, widths)
                left, right = _corridor_sides(centreline, widths)
                corridor_points = [
                    point
                    for pair in zip(left, right)
                    for point in pair
                ]
                corridor_counts = [4] * (len(centreline) - 1)
                corridor_indices: list[int] = []
                for segment_index in range(len(centreline) - 1):
                    first = segment_index * 2
                    corridor_indices.extend(
                        (first, first + 1, first + 3, first + 2)
                    )
                output = UsdGeom.Mesh.Define(water_stage, output_path)
                output.CreatePointsAttr(
                    [Gf.Vec3f(*point) for point in corridor_points]
                )
                output.CreateFaceVertexCountsAttr(corridor_counts)
                output.CreateFaceVertexIndicesAttr(corridor_indices)
                output.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                output.CreateNormalsAttr(
                    [
                        Gf.Vec3f(*normal)
                        for normal in _mesh_vertex_normals(
                            corridor_points,
                            corridor_counts,
                            corridor_indices,
                        )
                    ]
                )
                output.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
                _author_metric_uv(
                    output,
                    points=corridor_points,
                    metres_per_uv_tile=water_uv_metres,
                    usd_geom=UsdGeom,
                    sdf=Sdf,
                    gf=Gf,
                )
                water_record = {
                    "stable_id": stable_id,
                    "family": "bdtopo_watercourse",
                    "outline": outline,
                    "kind": "watercourse",
                    "centreline": centreline,
                    "surface_profile_m": [
                        point[2] for point in centreline
                    ],
                }
            imageable = UsdGeom.Imageable(output.GetPrim())
            imageable.CreatePurposeAttr(UsdGeom.Tokens.render)
            imageable.CreateVisibilityAttr(UsdGeom.Tokens.inherited)
            output.GetPrim().SetCustomDataByKey(
                "fireviewer:semantic_class", "water"
            )
            UsdShade.MaterialBindingAPI.Apply(output.GetPrim()).Bind(
                material_proxy
            )
            waters.append(water_record)
            if len(waters) > MAX_PREPARED_WATER_COUNT:
                raise CompositionSourceError(
                    "native water count exceeds the bounded preparation limit"
                )
        stage = None
    routes, source_route_metrics = _source_backed_routes(
        volume_root=volume_root,
        zone_root=zone_root,
        source_lock=source_lock,
        epsg2154_bounds=epsg2154_bounds,
        root_origin=root_origin,
        placement_tiles=placement_tiles,
    )
    # Route vectors are the complete topology authority.  Native detail
    # payloads intentionally contain no road ribbons or road assets because
    # the orthophoto already renders the roads on the terrain surface.
    native_road_fragment_count = int(source_route_metrics["source_line_count"])
    if not routes or not waters:
        raise CompositionSourceError(
            "locked BDTOPO routes and native HERO hydrology must be non-empty"
        )
    layers = build.get("layers")
    if not isinstance(layers, Mapping):
        raise CompositionSourceError(
            "native build receipt has no layer inventory"
        )
    roads_layer = layers.get("roads")
    road_count = (
        roads_layer.get("source_line_count", roads_layer.get("source_feature_count"))
        if isinstance(roads_layer, Mapping)
        else None
    )
    hydro_count = (
        layers.get("hydrology", {}).get("prim_count")
        if isinstance(layers.get("hydrology"), Mapping)
        else None
    )
    if (
        road_count != native_road_fragment_count
        or hydro_count != len(waters)
    ):
        raise CompositionSourceError(
            "source-backed route-line/water counts differ from the native "
            "receipt"
        )
    bridge_metrics = _derive_native_bridge_spans(
        routes=routes,
        waters=waters,
    )
    bridge_metrics["route_source"] = {
        **source_route_metrics,
        "native_hero_fragment_proof_count": native_road_fragment_count,
        "native_hero_receipt_fragment_count": road_count,
    }
    water_root.GetPrim().SetCustomDataByKey(
        "fireviewer:feature_count", len(waters)
    )
    water_root.GetPrim().SetCustomDataByKey(
        "fireviewer:source_build_receipt_sha256",
        _sha256(zone_root / "build" / "build-receipt.json"),
    )
    water_stage.GetRootLayer().Save()
    water_stage = None

    reopened = Usd.Stage.Open(str(physical_water_path))
    if reopened is None:
        raise CompositionSourceError(
            "isolated water USD cannot be reopened after authoring"
        )
    geometry = [
        prim
        for prim in reopened.TraverseAll()
        if prim.IsA(UsdGeom.Mesh)
    ]
    if (
        len(geometry) != len(waters)
        or any(
            prim.IsA(UsdGeom.BasisCurves)
            for prim in reopened.TraverseAll()
            if str(prim.GetPath()).startswith("/Water/")
        )
        or any(
            not str(prim.GetPath()).startswith("/Water/")
            or UsdGeom.Imageable(prim).ComputeVisibility()
            != UsdGeom.Tokens.inherited
            or UsdGeom.Imageable(prim).ComputePurpose()
            != UsdGeom.Tokens.render
            or not UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            for prim in geometry
        )
    ):
        raise CompositionSourceError(
            "isolated water USD is incomplete, hidden or unmaterialized"
        )
    reopened = None
    topology = _route_topology_contract(
        routes,
        tolerance_m=float(
            _fixed_variant_constraints()["road_connectivity_tolerance_m"]
        ),
    )
    observed_families = {
        family
        for families in tree_families_by_tile.values()
        for family in families
    }
    if observed_families != accepted_tree_families:
        raise CompositionSourceError(
            "native tree family observations differ from the SimReady library "
            f"(observed={sorted(observed_families)}, "
            f"accepted={sorted(accepted_tree_families)})"
        )
    if not building_group_bounds:
        raise CompositionSourceError(
            "native building instances expose no accepted settlement groups"
        )
    return (
        routes,
        waters,
        topology,
        bridge_metrics,
        _NativeSuitabilityObservations(
            tree_families_by_tile_ref=tree_families_by_tile,
            building_group_bounds=building_group_bounds,
        ),
    )


def _orient_waters_and_required_tolerance(
    *,
    waters: list[dict[str, object]],
    placement_tiles: Sequence[_PreparedPlacementHeight],
) -> dict[str, object]:
    """Orient native watercourses and enforce the fixed reviewed tolerance."""

    allowed = float(
        _fixed_variant_constraints()["water_uphill_tolerance_m"]
    )
    sampler = _PlacementHeightSampler(placement_tiles, cache_limit=2)

    maximum_below_terrain = 0.0
    maximum_uphill = 0.0
    reversed_watercourses = 0
    for index, water in enumerate(waters):
        kind = water.get("kind")
        outline = water.get("outline")
        profile = water.get("surface_profile_m")
        centreline = water.get("centreline")
        if (
            not isinstance(outline, list)
            or not isinstance(profile, list)
            or not isinstance(centreline, list)
        ):
            raise CompositionSourceError(
                f"prepared water {index} is malformed"
            )
        if kind == "standing":
            surface = float(profile[0])
            maximum_below_terrain = max(
                maximum_below_terrain,
                0.0,
                max(
                    sampler.elevation(float(point[0]), float(point[1]))
                    - surface
                    for point in outline
                ),
            )
            continue
        if kind != "watercourse" or len(profile) != len(centreline):
            raise CompositionSourceError(
                f"prepared water {index} has inconsistent watercourse data"
            )

        def uphill(values: Sequence[object]) -> float:
            return max(
                (
                    float(second) - float(first)
                    for first, second in zip(values, values[1:])
                ),
                default=0.0,
            )

        forward_uphill = uphill(profile)
        reverse_uphill = uphill(list(reversed(profile)))
        if reverse_uphill < forward_uphill:
            centreline.reverse()
            profile.reverse()
            reversed_watercourses += 1
        maximum_uphill = max(maximum_uphill, uphill(profile))
        for point, surface in zip(centreline, profile):
            maximum_below_terrain = max(
                maximum_below_terrain,
                0.0,
                sampler.elevation(float(point[0]), float(point[1]))
                - float(surface),
            )
    if (
        maximum_uphill > allowed + 1.0e-9
        or maximum_below_terrain > allowed + 1.0e-9
    ):
        raise CompositionSourceError(
            "native water violates the fixed reviewed "
            f"{allowed:.6g} m downhill/terrain "
            f"tolerance (uphill={maximum_uphill:.6f}, "
            f"below_terrain={maximum_below_terrain:.6f})"
        )
    return {
        "algorithm": "near-grid-downhill-orientation-v1",
        "fixed_tolerance_m": allowed,
        "maximum_uphill_step_m": maximum_uphill,
        "maximum_below_terrain_m": maximum_below_terrain,
        "reversed_watercourse_count": reversed_watercourses,
        "placement_height_cache_tile_limit": 2,
        "placement_height_peak_cached_tiles": sampler.peak_cached_tiles,
    }


def _validate_build_asset_binding(
    *,
    build: Mapping[str, object],
    manifest_path: Path,
    volume_root: Path,
) -> None:
    asset_lock = build.get("asset_lock")
    records = (
        asset_lock.get("assets")
        if isinstance(asset_lock, Mapping)
        else None
    )
    if not isinstance(records, list):
        raise CompositionSourceError(
            "native build receipt has no shared asset lock"
        )
    manifest_sha = _sha256(manifest_path)
    matches = [
        record
        for record in records
        if isinstance(record, Mapping)
        and record.get("manifest_sha256") == manifest_sha
    ]
    if len(matches) != 1:
        raise CompositionSourceError(
            "native build is not bound to the supplied SimReady manifest"
        )
    raw_path = matches[0].get("manifest")
    if isinstance(raw_path, str) and raw_path.strip():
        candidate = (volume_root / raw_path).resolve()
        if candidate != manifest_path:
            raise CompositionSourceError(
                "native build asset lock points to another SimReady manifest"
            )


def _prepared_contract_artifact(
    *,
    final_path: Path,
    physical_path: Path,
    volume_root: Path,
    prim_path: str = "",
) -> dict[str, str]:
    record = {
        "path": final_path.relative_to(volume_root).as_posix(),
        "sha256": _sha256(physical_path),
    }
    if prim_path:
        record["prim_path"] = prim_path
    return record


def prepare_native_contract(
    *,
    volume_root: Path,
    output_root: Path,
    source: NativePreparationSource,
) -> dict[str, Any]:
    """Atomically derive a complete export contract from native artifacts.

    No route, water, suitability or height geometry is accepted from the
    operator.  The function reads the accepted 400 terrain/HERO payloads, the
    locked BDTOPO vegetation, the exact SimReady/PBR receipts and the native
    ground receipt.  USD stages are opened one at a time.
    """

    volume = volume_root.expanduser().resolve()
    if not volume.is_dir() or volume.is_symlink():
        raise CompositionSourceError(
            "persistent volume root must be a real directory"
        )
    if not isinstance(source, NativePreparationSource):
        raise CompositionSourceError(
            "source must be a NativePreparationSource"
        )
    output = output_root.expanduser().resolve()
    if (
        not _is_below(volume, output)
        or output == volume
        or output.exists()
    ):
        raise CompositionSourceError(
            "prepared contract output must be a new directory below the "
            "persistent volume"
        )
    zone_root = _required_directory_below(
        source.zone_root,
        volume_root=volume,
        label="native zone root",
    )
    build_path = zone_root / "build" / "build-receipt.json"
    build_path = _required_file_below(
        build_path,
        volume_root=volume,
        label="native build receipt",
    )
    build = _read_json(build_path, label="native build receipt")
    base_scene_id = _trimmed(
        build.get("zone_id"), label="native build receipt zone_id"
    )
    if (
        build.get("schema_version") != 2
        or build.get("source_profile") != "full"
        or build.get("fire_simulation_status")
        != "blocked_pending_editor_review"
    ):
        raise CompositionSourceError(
            "autonomous preparation requires a full blocked schema-2 native build"
        )
    scene_auto_validation = _required_file_below(
        source.scene_auto_validation,
        volume_root=volume,
        label="scene auto-validation",
    )
    auto = _read_json(
        scene_auto_validation, label="scene auto-validation"
    )
    root_path, root_record = _zone_receipt_artifact(
        build.get("root_usd"),
        zone_root=zone_root,
        volume_root=volume,
        label="native build receipt root_usd",
        prim_path="/World",
    )
    if (
        auto.get("state") != "AUTO_VALIDATED"
        or auto.get("fire_simulation_status")
        != "blocked_pending_editor_review"
        or auto.get("build_receipt_sha256") != _sha256(build_path)
        or auto.get("root_usd_sha256") != _sha256(root_path)
    ):
        raise CompositionSourceError(
            "scene auto-validation is stale or bound to another native build"
        )
    georeference_path = _required_file_below(
        zone_root / "build" / "metadata" / "georeference.json",
        volume_root=volume,
        label="native georeference",
    )
    georeference = _read_json(
        georeference_path, label="native georeference"
    )
    raw_origin = georeference.get("local_origin_epsg2154")
    if (
        georeference.get("zone_id") != base_scene_id
        or georeference.get("crs") != "EPSG:2154"
        or georeference.get("vertical_datum") != "IGN69"
        or not isinstance(raw_origin, list)
        or len(raw_origin) < 2
    ):
        raise CompositionSourceError(
            "native georeference is incomplete or bound to another zone"
        )
    root_origin = (
        _finite(raw_origin[0], label="georeference origin X"),
        _finite(raw_origin[1], label="georeference origin Y"),
    )
    source_lock_path, _source_lock_record = _zone_receipt_artifact(
        build.get("source_lock"),
        zone_root=zone_root,
        volume_root=volume,
        label="native build source_lock",
    )
    source_lock = _read_json(source_lock_path, label="native source-lock")
    if source_lock.get("zone_id") != base_scene_id:
        raise CompositionSourceError(
            "native source-lock is bound to another zone"
        )

    asset_manifest = _required_file_below(
        source.asset_manifest,
        volume_root=volume,
        label="SimReady asset manifest",
    )
    asset_lod_validation = _required_file_below(
        source.asset_lod_validation,
        volume_root=volume,
        label="native asset LOD validation",
    )
    asset_pbr_validation = _required_file_below(
        source.asset_pbr_validation,
        volume_root=volume,
        label="native asset PBR validation",
    )
    _validate_build_asset_binding(
        build=build,
        manifest_path=asset_manifest,
        volume_root=volume,
    )
    (
        asset_library,
        water_material_source,
        water_material,
        water_uv_metres,
    ) = _validated_asset_contract(
        volume_root=volume,
        manifest_path=asset_manifest,
        lod_validation_path=asset_lod_validation,
        pbr_validation_path=asset_pbr_validation,
    )
    ground_artifact_root = _required_directory_below(
        source.ground_artifact_root,
        volume_root=volume,
        label="ground artifact root",
    )
    ground_authoring_receipt = _required_file_below(
        source.ground_authoring_receipt,
        volume_root=volume,
        label="ground authoring receipt",
    )
    if not _is_below(ground_artifact_root, ground_authoring_receipt):
        raise CompositionSourceError(
            "ground authoring receipt must stay below its artifact root"
        )
    ground_surface = _validated_ground_contract(
        volume_root=volume,
        artifact_root=ground_artifact_root,
        authoring_receipt_path=ground_authoring_receipt,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(
        f".{output.name}.{uuid.uuid4().hex}.staging"
    )
    staging.mkdir()
    try:
        terrain, placement_tiles = _prepared_terrain_grids(
            volume_root=volume,
            zone_root=zone_root,
            build=build,
            root_origin=root_origin,
            placement_physical_root=staging / "placement-height",
            placement_final_root=output / "placement-height",
        )
        (
            placement_height_records,
            placement_height_fingerprint,
            placement_seam_metrics,
        ) = _placement_height_records(
            placements=placement_tiles,
            volume_root=volume,
        )
        bounds = _terrain_scene_bounds(terrain)
        epsg2154_bounds = (
            min(tile.epsg2154_bounds[0] for tile in terrain),
            min(tile.epsg2154_bounds[1] for tile in terrain),
            max(tile.epsg2154_bounds[2] for tile in terrain),
            max(tile.epsg2154_bounds[3] for tile in terrain),
        )
        final_height = output / "height-field.json"
        height_record, height_metrics = _write_prepared_height_field(
            physical_path=staging / "height-field.json",
            final_path=final_height,
            volume_root=volume,
            terrain=terrain,
            bounds=bounds,
        )
        tree_families = sorted(
            {
                str(record["family"])
                for record in asset_library.values()
                if record["category"] == "trees"
            }
        )
        final_water = output / "isolated-water.usdc"
        (
            routes,
            waters,
            route_topology_record,
            bridge_metrics,
            suitability_observations,
        ) = (
            _extract_native_networks_and_water_payload(
                volume_root=volume,
                zone_root=zone_root,
                build=build,
                physical_water_path=staging / "isolated-water.usdc",
                final_water_path=final_water,
                water_material=water_material,
                water_uv_metres=water_uv_metres,
                accepted_tree_families=set(tree_families),
                source_lock=source_lock,
                epsg2154_bounds=epsg2154_bounds,
                root_origin=root_origin,
                placement_tiles=placement_tiles,
            )
        )
        route_components = int(
            route_topology_record["source_component_count"]
        )
        water_validation_metrics = _orient_waters_and_required_tolerance(
            waters=waters,
            placement_tiles=placement_tiles,
        )
        suitability = _source_backed_suitability(
            zone_root=zone_root,
            source_lock=source_lock,
            epsg2154_bounds=epsg2154_bounds,
            root_origin=root_origin,
            terrain=terrain,
            tree_families=tree_families,
            observations=suitability_observations,
        )
        water_payload_record = _prepared_contract_artifact(
            final_path=final_water,
            physical_path=staging / "isolated-water.usdc",
            volume_root=volume,
            prim_path="/Water",
        )
        final_water_evidence = output / "isolated-water-validation.json"
        water_evidence_payload = {
            "schema_version": 1,
            "state": "ISOLATED_WATER_VALIDATED",
            "visible": True,
            "content_roles": ["water"],
            "feature_count": len(waters),
            "geometry_type": "UsdGeom.Mesh",
            "pbr_material": water_material,
            "physical_validation": water_validation_metrics,
            "payloads": [
                {
                    "path": water_payload_record["path"],
                    "sha256": water_payload_record["sha256"],
                }
            ],
        }
        _write_json(
            staging / "isolated-water-validation.json",
            water_evidence_payload,
        )
        water_evidence_record = _prepared_contract_artifact(
            final_path=final_water_evidence,
            physical_path=staging / "isolated-water-validation.json",
            volume_root=volume,
        )
        constraints = _fixed_variant_constraints()
        constraints["maximum_road_components"] = route_components
        terrain_records: list[dict[str, Any]] = []
        for tile in terrain:
            record = _contract_artifact(
                Path(tile.artifact.path),
                volume_root=volume,
                prim_path="/Tile",
            )
            terrain_records.append(
                {
                    **record,
                    "tile_ref": tile.tile_ref,
                    "local_bounds": tile.local_bounds,
                    "epsg2154_bounds": {
                        "min_x": tile.epsg2154_bounds[0],
                        "min_y": tile.epsg2154_bounds[1],
                        "max_x": tile.epsg2154_bounds[2],
                        "max_y": tile.epsg2154_bounds[3],
                    },
                    "instance_namespace": tile.instance_namespace,
                    "terrain_lods": list(tile.terrain_lods),
                    "collision_lods": list(tile.collision_lods),
                }
            )
        contract: dict[str, Any] = {
            "schema_version": 1,
            "state": "NATIVE_COMPOSITION_EXPORT_INPUT_READY",
            "base_scene_id": base_scene_id,
            "coordinate_contract": ROOT_LOCAL_COORDINATE_CONTRACT,
            "epsg2154_origin": [
                root_origin[0],
                root_origin[1],
                0.0,
            ],
            "object_source": "native_hero_detail_payloads",
            "native_artifacts": {
                "native_build_receipt": _contract_artifact(
                    build_path, volume_root=volume
                ),
                "scene_auto_validation": _contract_artifact(
                    scene_auto_validation, volume_root=volume
                ),
                "root_usd": root_record,
                "terrain_payloads": terrain_records,
                "water_payloads": [water_payload_record],
                "water_validation": {
                    "state": "ISOLATED_WATER_VALIDATED",
                    "evidence": water_evidence_record,
                },
            },
            "bounds": bounds,
            "height_field_source": height_record,
            "placement_height_tiles": placement_height_records,
            "placement_height_fingerprint": (
                placement_height_fingerprint
            ),
            "ground_surface": ground_surface,
            "asset_library": asset_library,
            "road_visual_contract": {
                "visible_representation": (
                    "orthophoto_derived_terrain_material"
                ),
                "geometry_authoring": "disabled",
                "asset_dependencies": [],
                "route_vectors_retained_for": [
                    "topology",
                    "actor_placement",
                    "annotations",
                    "composition_constraints",
                ],
            },
            "water_material_source": water_material_source,
            "routes": routes,
            "route_topology": route_topology_record,
            "route_source": bridge_metrics["route_source"],
            "waters": waters,
            "suitability_zones": suitability,
            "variant_constraints": constraints,
            "native_detail_extraction": {
                "coordinate_space": "root_local_xy_ign69_z",
                "root_origin_epsg2154": [
                    root_origin[0],
                    root_origin[1],
                ],
                "stable_id_primvar": "fireviewer_stable_id",
                "footprint_radius_primvar": (
                    "fireviewer_footprint_radius_m"
                ),
                "group_id_primvar": "fireviewer_group_id",
                "tile_bounds_tolerance_m": 20.0,
            },
            "preparation_evidence": {
                "source_lock": _contract_artifact(
                    source_lock_path, volume_root=volume
                ),
                "georeference": _contract_artifact(
                    georeference_path, volume_root=volume
                ),
                "asset_manifest": _contract_artifact(
                    asset_manifest, volume_root=volume
                ),
                "asset_lod_validation": _contract_artifact(
                    asset_lod_validation, volume_root=volume
                ),
                "asset_pbr_validation": _contract_artifact(
                    asset_pbr_validation, volume_root=volume
                ),
                "ground_authoring_receipt": _contract_artifact(
                    ground_authoring_receipt, volume_root=volume
                ),
            },
            "preparation_metrics": {
                "terrain_payload_count": len(terrain),
                "height_field": height_metrics,
                "placement_height_seams": placement_seam_metrics,
                "route_count": len(routes),
                "route_component_count": route_components,
                "route_topology": route_topology_record,
                "bridges": bridge_metrics,
                "water_feature_count": len(waters),
                "water_validation": water_validation_metrics,
                "suitability_zone_count": len(suitability),
            },
            "streaming_memory_contract": {
                "terrain_or_detail_usd_stages_open_concurrently": 1,
                "terrain_far_samples_retained": sum(
                    len(tile.x_coordinates) * len(tile.y_coordinates)
                    for tile in terrain
                ),
                "height_field_maximum_samples": MAX_HEIGHT_FIELD_SAMPLES,
                "placement_height_tile_count": len(placement_tiles),
                "placement_height_cache_tile_limit": 2,
                "full_placement_height_surface_retained_in_ram": False,
                "maximum_detail_payload_bytes": MAX_DETAIL_PAYLOAD_BYTES,
                "full_tree_or_building_inventory_retained_in_ram": False,
            },
            "fire_simulation_status": "blocked_pending_editor_review",
        }
        _write_json(
            staging / "composition-export-input.json",
            contract,
        )
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    contract_path = output / "composition-export-input.json"
    return {
        "schema_version": 1,
        "state": "NATIVE_COMPOSITION_EXPORT_CONTRACT_PREPARED",
        "base_scene_id": base_scene_id,
        "contract": _contract_artifact(
            contract_path, volume_root=volume
        ),
        "height_field": _contract_artifact(
            output / "height-field.json", volume_root=volume
        ),
        "placement_height_tiles": len(placement_tiles),
        "placement_height_fingerprint": placement_height_fingerprint,
        "placement_height_seams": placement_seam_metrics,
        "isolated_water": _contract_artifact(
            output / "isolated-water.usdc",
            volume_root=volume,
            prim_path="/Water",
        ),
        "routes": len(routes),
        "waters": len(waters),
        "suitability_zones": len(suitability),
        "memory_contract": contract["streaming_memory_contract"],
    }


def _artifact_source_from_record(
    raw: object,
    *,
    label: str,
) -> ArtifactSource:
    if not isinstance(raw, Mapping):
        raise CompositionSourceError(f"{label} must be an artifact object")
    path = raw.get("path")
    if not isinstance(path, (str, Path)):
        raise CompositionSourceError(f"{label}.path is required")
    prim_path = raw.get("prim_path", "")
    if not isinstance(prim_path, str):
        raise CompositionSourceError(f"{label}.prim_path must be a string")
    expected = raw.get("sha256")
    if expected is not None and (
        not isinstance(expected, str) or not _SHA256.fullmatch(expected)
    ):
        raise CompositionSourceError(f"{label}.sha256 must be lowercase SHA-256")
    return ArtifactSource(
        path=path,
        prim_path=prim_path,
        expected_sha256=expected,
    )


def _asset_sources_from_contract(raw: object) -> tuple[AssetSource, ...]:
    if not isinstance(raw, Mapping) or not raw:
        raise CompositionSourceError(
            "export contract asset_library must be a non-empty object"
        )
    result: list[AssetSource] = []
    for key, record in raw.items():
        label = f"asset_library.{key}"
        if not isinstance(key, str) or not isinstance(record, Mapping):
            raise CompositionSourceError(f"{label} is malformed")
        lods = record.get("lods")
        validation = record.get("simready_validation")
        if not isinstance(lods, Mapping) or not isinstance(validation, Mapping):
            raise CompositionSourceError(f"{label} lacks LOD/validation records")
        result.append(
            AssetSource(
                key=key,
                category=str(record.get("category", "")),
                family=str(record.get("family", "")),
                lods={
                    level: _artifact_source_from_record(
                        lods.get(level), label=f"{label}.lods.{level}"
                    )
                    for level in LOD_LEVELS
                },
                lod_lineage=str(validation.get("lod_lineage", "")),
                grounding_offsets_m=validation.get("grounding_offsets_m", {}),
                simready_validation_state=str(validation.get("state", "")),
                simready_validation_evidence=_artifact_source_from_record(
                    validation.get("evidence"),
                    label=f"{label}.simready_validation.evidence",
                ),
            )
        )
    return tuple(result)


def _water_material_source_from_contract(
    raw: object,
) -> WaterMaterialSource:
    if not isinstance(raw, Mapping):
        raise CompositionSourceError(
            "export contract water_material_source must be an object"
        )
    lods = raw.get("lods")
    if not isinstance(lods, Mapping):
        raise CompositionSourceError(
            "water_material_source.lods is required"
        )
    return WaterMaterialSource(
        lods={
            level: _artifact_source_from_record(
                lods.get(level),
                label=f"water_material_source.lods.{level}",
            )
            for level in LOD_LEVELS
        },
        pbr_validation_state=str(raw.get("pbr_validation_state", "")),
        pbr_validation_evidence=_artifact_source_from_record(
            raw.get("pbr_validation_evidence"),
            label="water_material_source.pbr_validation_evidence",
        ),
    )


def _load_height_field_source(
    raw: object,
    *,
    volume_root: Path,
) -> HeightFieldSource:
    artifact_source = _artifact_source_from_record(
        raw, label="height_field_source"
    )
    artifact = _resolve_artifact(
        artifact_source,
        volume_root=volume_root,
        label="height_field_source",
    )
    if artifact.physical_path.stat().st_size > MAX_HEIGHT_FIELD_SOURCE_BYTES:
        raise CompositionSourceError(
            "height field source exceeds the 64 MiB portable input bound"
        )
    value = _read_json(artifact.physical_path, label="height field source")
    samples = value.get("samples")
    if not isinstance(samples, list):
        raise CompositionSourceError(
            "height field source must contain a samples array"
        )
    return HeightFieldSource(
        origin_x=value.get("origin_x"),  # type: ignore[arg-type]
        origin_y=value.get("origin_y"),  # type: ignore[arg-type]
        spacing_m=value.get("spacing_m"),  # type: ignore[arg-type]
        samples=samples,
    )


def _native_artifacts_from_contract(raw: object) -> NativeArtifactsSource:
    if not isinstance(raw, Mapping):
        raise CompositionSourceError(
            "export contract native_artifacts must be an object"
        )
    terrain = raw.get("terrain_payloads")
    water = raw.get("water_payloads")
    water_validation = raw.get("water_validation")
    if (
        not isinstance(terrain, list)
        or not isinstance(water, list)
        or not isinstance(water_validation, Mapping)
    ):
        raise CompositionSourceError(
            "native artifacts need terrain, water and water validation records"
        )
    return NativeArtifactsSource(
        native_build_receipt=_artifact_source_from_record(
            raw.get("native_build_receipt"),
            label="native_artifacts.native_build_receipt",
        ),
        scene_auto_validation=_artifact_source_from_record(
            raw.get("scene_auto_validation"),
            label="native_artifacts.scene_auto_validation",
        ),
        root_usd=_artifact_source_from_record(
            raw.get("root_usd"), label="native_artifacts.root_usd"
        ),
        terrain_payloads=tuple(
            TerrainPayloadSource(
                artifact=_artifact_source_from_record(
                    record,
                    label=f"native_artifacts.terrain_payloads[{index}]",
                ),
                tile_ref=str(
                    record.get("tile_ref", "")
                    if isinstance(record, Mapping)
                    else ""
                ),
                local_bounds=(
                    record.get("local_bounds", {})
                    if isinstance(record, Mapping)
                    else {}
                ),
                epsg2154_bounds=(
                    record.get("epsg2154_bounds", {})
                    if isinstance(record, Mapping)
                    else {}
                ),
                instance_namespace=(
                    record.get("instance_namespace", 0)
                    if isinstance(record, Mapping)
                    else 0
                ),  # type: ignore[arg-type]
                terrain_lods=(
                    record.get("terrain_lods", [])
                    if isinstance(record, Mapping)
                    else []
                ),  # type: ignore[arg-type]
                collision_lods=(
                    record.get("collision_lods", [])
                    if isinstance(record, Mapping)
                    else []
                ),  # type: ignore[arg-type]
            )
            for index, record in enumerate(terrain)
        ),
        water_payloads=tuple(
            _artifact_source_from_record(
                record,
                label=f"native_artifacts.water_payloads[{index}]",
            )
            for index, record in enumerate(water)
        ),
        water_validation_state=str(water_validation.get("state", "")),
        water_validation_evidence=_artifact_source_from_record(
            water_validation.get("evidence"),
            label="native_artifacts.water_validation.evidence",
        ),
    )


def _ground_surface_from_contract(raw: object) -> GroundSurfaceSource:
    if not isinstance(raw, Mapping):
        raise CompositionSourceError(
            "export contract ground_surface must be an object"
        )
    removed = raw.get("removed_object_classes", [])
    if not isinstance(removed, list) or any(
        not isinstance(value, str) for value in removed
    ):
        raise CompositionSourceError(
            "ground_surface.removed_object_classes must be a string list"
        )
    payloads = raw.get("tile_material_payloads")
    if (
        raw.get("topology")
        != "payload_tiled_materials_shared_pbr_library"
        or not isinstance(payloads, list)
    ):
        raise CompositionSourceError(
            "ground_surface requires tiled shared-PBR payloads"
        )
    return GroundSurfaceSource(
        material=_artifact_source_from_record(
            raw.get("material"), label="ground_surface.material"
        ),
        tile_material_payloads=tuple(
            GroundMaterialPayloadSource(
                artifact=_artifact_source_from_record(
                    record,
                    label=(
                        f"ground_surface.tile_material_payloads[{index}]"
                    ),
                ),
                tile_id=(
                    str(record.get("tile_id", ""))
                    if isinstance(record, Mapping)
                    else ""
                ),
                tile_ref=(
                    str(record.get("tile_ref", ""))
                    if isinstance(record, Mapping)
                    else ""
                ),
                tile_bounds_m=(
                    record.get("tile_bounds_m", [])
                    if isinstance(record, Mapping)
                    else []
                ),  # type: ignore[arg-type]
            )
            for index, record in enumerate(payloads)
        ),
        validation_evidence=_artifact_source_from_record(
            raw.get("validation_evidence"),
            label="ground_surface.validation_evidence",
        ),
        validation_state=str(raw.get("validation_state", "")),
        kind=str(raw.get("kind", "")),
        removed_object_classes=tuple(removed),
    )


def _placement_height_sources_from_contract(
    raw: object,
) -> tuple[PlacementHeightTileSource, ...]:
    if not isinstance(raw, list):
        raise CompositionSourceError(
            "export contract placement_height_tiles must be an array"
        )
    result: list[PlacementHeightTileSource] = []
    for index, record in enumerate(raw):
        label = f"placement_height_tiles[{index}]"
        if not isinstance(record, Mapping):
            raise CompositionSourceError(f"{label} is malformed")
        raw_x = record.get("x_coordinates")
        raw_y = record.get("y_coordinates")
        if not isinstance(raw_x, list) or not isinstance(raw_y, list):
            raise CompositionSourceError(f"{label} axes must be arrays")
        result.append(
            PlacementHeightTileSource(
                artifact=_artifact_source_from_record(
                    record, label=label
                ),
                tile_ref=str(record.get("tile_ref", "")),
                local_bounds=(
                    record.get("local_bounds", {})
                    if isinstance(record.get("local_bounds"), Mapping)
                    else {}
                ),
                width=record.get("width", 0),  # type: ignore[arg-type]
                height=record.get("height", 0),  # type: ignore[arg-type]
                x_coordinates=raw_x,
                y_coordinates=raw_y,
                format=str(record.get("format", "")),
            )
        )
    return tuple(result)


def _detail_extraction_from_contract(
    raw: object,
) -> DetailPayloadExtractionSource:
    if not isinstance(raw, Mapping):
        raise CompositionSourceError(
            "export contract native_detail_extraction must be an object"
        )
    origin = raw.get("root_origin_epsg2154")
    if not isinstance(origin, list):
        raise CompositionSourceError(
            "native_detail_extraction.root_origin_epsg2154 must be an array"
        )
    return DetailPayloadExtractionSource(
        coordinate_space=str(raw.get("coordinate_space", "")),
        root_origin_epsg2154=origin,
        stable_id_primvar=str(
            raw.get("stable_id_primvar", "fireviewer_stable_id")
        ),
        footprint_radius_primvar=str(
            raw.get(
                "footprint_radius_primvar",
                "fireviewer_footprint_radius_m",
            )
        ),
        group_id_primvar=str(
            raw.get("group_id_primvar", "fireviewer_group_id")
        ),
        tile_bounds_tolerance_m=raw.get(  # type: ignore[arg-type]
            "tile_bounds_tolerance_m", 20.0
        ),
    )


def _verified_artifact_inventory(
    value: object,
    *,
    volume_root: Path,
    label: str,
) -> dict[str, str]:
    """Hash every artifact-shaped record in a contract or manifest."""

    inventory: dict[str, str] = {}
    record_count = 0

    def visit(node: object, node_label: str) -> None:
        nonlocal record_count
        if isinstance(node, Mapping):
            has_path = "path" in node
            has_sha = "sha256" in node
            if has_path or has_sha:
                if not (has_path and has_sha):
                    raise CompositionSourceError(
                        f"{node_label} has a partial artifact binding"
                    )
                path, expected = _receipt_artifact_tuple(
                    node, label=node_label
                )
                physical = (volume_root / path).resolve()
                if (
                    not _is_below(volume_root, physical)
                    or not physical.is_file()
                    or physical.is_symlink()
                    or _sha256(physical) != expected
                ):
                    raise CompositionSourceError(
                        f"{node_label} artifact is absent, unsafe or stale"
                    )
                previous = inventory.get(path)
                if previous is not None and previous != expected:
                    raise CompositionSourceError(
                        f"artifact {path} has divergent SHA-256 bindings"
                    )
                inventory[path] = expected
                record_count += 1
                if record_count > 50_000:
                    raise CompositionSourceError(
                        f"{label} artifact inventory exceeds 50,000 records"
                    )
            for key, child in node.items():
                visit(child, f"{node_label}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{node_label}[{index}]")

    visit(value, label)
    if not inventory:
        raise CompositionSourceError(f"{label} has no hash-bound artifacts")
    return inventory


def _placement_fingerprint_from_records(records: object) -> str:
    if not isinstance(records, list) or len(records) != 400:
        raise CompositionSourceError(
            "placement height verification requires exactly 400 records"
        )
    if any(not isinstance(record, Mapping) for record in records):
        raise CompositionSourceError(
            "placement height verification contains malformed records"
        )
    ordered = sorted(
        (dict(record) for record in records),
        key=lambda record: str(record.get("tile_ref", "")),
    )
    return hashlib.sha256(
        json.dumps(
            ordered,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _native_road_fragment_count_from_contract(
    contract: Mapping[str, object],
    *,
    volume_root: Path,
) -> int:
    native = contract.get("native_artifacts")
    build_record = (
        native.get("native_build_receipt")
        if isinstance(native, Mapping)
        else None
    )
    path, expected = _receipt_artifact_tuple(
        build_record, label="native_artifacts.native_build_receipt"
    )
    physical = (volume_root / path).resolve()
    if (
        not _is_below(volume_root, physical)
        or not physical.is_file()
        or physical.is_symlink()
        or _sha256(physical) != expected
    ):
        raise CompositionSourceError(
            "native build receipt is absent or stale during verification"
        )
    build = _read_json(physical, label="native build receipt")
    layers = build.get("layers")
    roads = (
        layers.get("roads")
        if isinstance(layers, Mapping)
        else None
    )
    rendered_count = (
        roads.get("prim_count") if isinstance(roads, Mapping) else None
    )
    source_line_count = (
        roads.get("source_line_count", roads.get("source_feature_count", rendered_count))
        if isinstance(roads, Mapping)
        else None
    )
    if (
        isinstance(rendered_count, bool)
        or not isinstance(rendered_count, int)
        or rendered_count < 0
        or isinstance(source_line_count, bool)
        or not isinstance(source_line_count, int)
        or source_line_count < 1
    ):
        raise CompositionSourceError(
            "native build receipt has no source-backed route line count"
        )
    if rendered_count == 0 and (
        roads.get("visible_representation")
        != "orthophoto_derived_terrain_material"
        or roads.get("geometry_authoring") != "disabled"
        or roads.get("asset_dependencies") != []
    ):
        raise CompositionSourceError(
            "native build receipt road layer lacks the orthophoto-only contract"
        )
    return source_line_count


def _validate_prepared_contract_payload(
    contract: Mapping[str, object],
    *,
    volume_root: Path,
) -> dict[str, object]:
    if (
        contract.get("schema_version") != 1
        or contract.get("state")
        != "NATIVE_COMPOSITION_EXPORT_INPUT_READY"
        or contract.get("object_source") != "native_hero_detail_payloads"
        or contract.get("fire_simulation_status")
        != "blocked_pending_editor_review"
    ):
        raise CompositionSourceError(
            "composition export contract is not the blocked native schema-1 "
            "tile-streaming input"
        )
    native = contract.get("native_artifacts")
    ground = contract.get("ground_surface")
    terrain = (
        native.get("terrain_payloads")
        if isinstance(native, Mapping)
        else None
    )
    ground_tiles = (
        ground.get("tile_material_payloads")
        if isinstance(ground, Mapping)
        else None
    )
    if (
        not isinstance(terrain, list)
        or len(terrain) != 400
        or not isinstance(ground_tiles, list)
        or len(ground_tiles) != 400
    ):
        raise CompositionSourceError(
            "prepared contract does not contain exact 400 terrain/ground tiles"
        )
    placement = contract.get("placement_height_tiles")
    placement_fingerprint = _placement_fingerprint_from_records(placement)
    if contract.get("placement_height_fingerprint") != placement_fingerprint:
        raise CompositionSourceError(
            "prepared contract placement height fingerprint is stale"
        )
    routes = contract.get("routes")
    waters = contract.get("waters")
    suitability = contract.get("suitability_zones")
    constraints = contract.get("variant_constraints")
    if (
        not isinstance(routes, list)
        or not routes
        or not isinstance(waters, list)
        or not waters
        or not isinstance(suitability, list)
        or not suitability
        or not isinstance(constraints, Mapping)
    ):
        raise CompositionSourceError(
            "prepared contract route/water/suitability data is incomplete"
        )
    topology = _route_topology_contract(
        routes,  # type: ignore[arg-type]
        tolerance_m=_positive(
            constraints.get("road_connectivity_tolerance_m"),
            label="variant_constraints.road_connectivity_tolerance_m",
        ),
    )
    if contract.get("route_topology") != topology:
        raise CompositionSourceError(
            "prepared contract route topology fingerprint is stale"
        )
    native_fragment_count = _native_road_fragment_count_from_contract(
        contract, volume_root=volume_root
    )
    route_source = contract.get("route_source")
    if not isinstance(route_source, Mapping):
        raise CompositionSourceError(
            "prepared contract lacks continuous route source evidence"
        )
    normalized_route_source = _validated_route_source_evidence(
        route_source,
        volume_root=volume_root,
        prepared_route_count=len(routes),
        native_fragment_count=native_fragment_count,
    )
    inventory = _verified_artifact_inventory(
        contract,
        volume_root=volume_root,
        label="composition export contract",
    )
    return {
        "base_scene_id": str(contract.get("base_scene_id", "")),
        "artifact_inventory": inventory,
        "placement_height_fingerprint": placement_fingerprint,
        "route_topology": topology,
        "route_source": normalized_route_source,
        "native_road_fragment_count": native_fragment_count,
    }


def _load_verified_prepared_contract(
    *,
    volume_root: Path,
    contract_path: Path,
    contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    volume = volume_root.expanduser().resolve()
    if not volume.is_dir():
        raise CompositionSourceError(
            f"persistent volume root is absent: {volume}"
        )
    path = contract_path.expanduser().resolve()
    if (
        not _is_below(volume, path)
        or not path.is_file()
        or path.is_symlink()
    ):
        raise CompositionSourceError(
            "composition export contract must be a regular file below the "
            "persistent volume"
        )
    if not _SHA256.fullmatch(contract_sha256):
        raise CompositionSourceError(
            "contract_sha256 must be lowercase SHA-256"
        )
    if _sha256(path) != contract_sha256:
        raise CompositionSourceError(
            "composition export contract hash mismatch"
        )
    if path.stat().st_size > MAX_COMPOSITION_CONTRACT_BYTES:
        raise CompositionSourceError(
            "composition export contract exceeds the 256 MiB bound"
        )
    contract = _read_json(path, label="composition export contract")
    verification = _validate_prepared_contract_payload(
        contract, volume_root=volume
    )
    return contract, verification


def verify_prepared_contract(
    *,
    volume_root: Path,
    contract_path: Path,
    contract_sha256: str,
) -> dict[str, object]:
    """Verify a published preparation directory without rewriting it."""

    volume = volume_root.expanduser().resolve()
    path = contract_path.expanduser().resolve()
    contract, verification = _load_verified_prepared_contract(
        volume_root=volume,
        contract_path=path,
        contract_sha256=contract_sha256,
    )
    prepared_root = path.parent
    if prepared_root.is_symlink():
        raise CompositionSourceError(
            "prepared composition directory must not be a symlink"
        )
    inventory = verification["artifact_inventory"]
    if not isinstance(inventory, Mapping):
        raise CompositionSourceError(
            "prepared artifact inventory verification is malformed"
        )
    expected = {path.relative_to(volume).as_posix()}
    expected.update(
        portable
        for portable in inventory
        if _is_below(prepared_root, (volume / portable).resolve())
    )
    actual: set[str] = set()
    for candidate in prepared_root.rglob("*"):
        if candidate.is_symlink():
            raise CompositionSourceError(
                "prepared composition contains a symbolic link"
            )
        if candidate.is_file():
            actual.add(candidate.relative_to(volume).as_posix())
    if actual != expected:
        raise CompositionSourceError(
            "prepared composition file inventory is incomplete or contains "
            f"unreferenced files (missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)})"
        )
    return {
        "state": "NATIVE_COMPOSITION_EXPORT_INPUT_VERIFIED",
        "base_scene_id": verification["base_scene_id"],
        "contract": {
            "path": path.relative_to(volume).as_posix(),
            "sha256": contract_sha256,
        },
        "artifact_count": len(inventory),
        "placement_height_fingerprint": verification[
            "placement_height_fingerprint"
        ],
        "route_topology": verification["route_topology"],
        "route_source": verification["route_source"],
        "contract_payload": contract,
    }


def _verified_jsonl_count(
    *,
    physical_path: Path,
    expected_count: object,
    label: str,
) -> int:
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise CompositionSourceError(f"{label}.count must be positive")
    count = 0
    try:
        with physical_path.open("rb") as stream:
            for line in stream:
                if not line.strip():
                    raise CompositionSourceError(
                        f"{label} contains an empty JSONL record"
                    )
                count += 1
    except OSError as error:
        raise CompositionSourceError(
            f"{label} cannot be streamed for verification"
        ) from error
    if count != expected_count:
        raise CompositionSourceError(
            f"{label} count mismatch: expected {expected_count}, found {count}"
        )
    return count


def verify_composition_output(
    *,
    volume_root: Path,
    contract_path: Path,
    contract_sha256: str,
    output_root: Path,
) -> dict[str, object]:
    """Verify an existing export and its immutable prepared-contract binding."""

    volume = volume_root.expanduser().resolve()
    prepared = verify_prepared_contract(
        volume_root=volume,
        contract_path=contract_path,
        contract_sha256=contract_sha256,
    )
    contract = prepared.get("contract_payload")
    if not isinstance(contract, Mapping):
        raise CompositionSourceError(
            "prepared contract verification did not return its payload"
        )
    output = output_root.expanduser().resolve()
    if (
        not _is_below(volume, output)
        or output == volume
        or not output.is_dir()
        or output.is_symlink()
    ):
        raise CompositionSourceError(
            "composition output must be a regular dedicated directory below "
            "the persistent volume"
        )
    manifest_path = output / "composition-source.json"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or manifest_path.stat().st_size > MAX_COMPOSITION_CONTRACT_BYTES
    ):
        raise CompositionSourceError(
            "composition output manifest is absent, unsafe or oversized"
        )
    manifest = _read_json(
        manifest_path, label="composition output manifest"
    )
    expected_contract = {
        "path": contract_path.expanduser().resolve().relative_to(
            volume
        ).as_posix(),
        "sha256": contract_sha256,
    }
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("state") != "COMPOSITION_SOURCE_READY"
        or manifest.get("source_contract") != expected_contract
        or manifest.get("base_scene_id") != contract.get("base_scene_id")
        or manifest.get("fire_simulation_status")
        != "blocked_pending_editor_review"
    ):
        raise CompositionSourceError(
            "existing composition output is not bound to the exact prepared "
            "contract"
        )
    for field in (
        "routes",
        "waters",
        "variant_constraints",
        "placement_height_fingerprint",
    ):
        if manifest.get(field) != contract.get(field):
            raise CompositionSourceError(
                f"existing composition output {field} diverges from its "
                "prepared contract"
            )
    topology = _route_topology_contract(
        manifest.get("routes", []),  # type: ignore[arg-type]
        tolerance_m=_positive(
            (
                manifest.get("variant_constraints", {}).get(
                    "road_connectivity_tolerance_m"
                )
                if isinstance(
                    manifest.get("variant_constraints"), Mapping
                )
                else None
            ),
            label=(
                "composition output "
                "variant_constraints.road_connectivity_tolerance_m"
            ),
        ),
    )
    if manifest.get("route_topology") != topology:
        raise CompositionSourceError(
            "existing composition output route topology is stale"
        )
    placement_fingerprint = _placement_fingerprint_from_records(
        manifest.get("placement_height_tiles")
    )
    if (
        manifest.get("placement_height_fingerprint")
        != placement_fingerprint
    ):
        raise CompositionSourceError(
            "existing composition output placement fingerprint is stale"
        )
    native_fragment_count = _native_road_fragment_count_from_contract(
        contract, volume_root=volume
    )
    route_source = manifest.get("route_source")
    if not isinstance(route_source, Mapping):
        raise CompositionSourceError(
            "existing composition output lacks route source evidence"
        )
    normalized_route_source = _validated_route_source_evidence(
        route_source,
        volume_root=volume,
        prepared_route_count=len(manifest.get("routes", [])),
        native_fragment_count=native_fragment_count,
    )
    if normalized_route_source != prepared.get("route_source"):
        raise CompositionSourceError(
            "existing composition route evidence differs from preparation"
        )
    inventory = _verified_artifact_inventory(
        manifest,
        volume_root=volume,
        label="composition output manifest",
    )
    expected_files = {
        manifest_path.relative_to(volume).as_posix()
    }
    expected_files.update(
        portable
        for portable in inventory
        if _is_below(output, (volume / portable).resolve())
    )
    actual_files: set[str] = set()
    for candidate in output.rglob("*"):
        if candidate.is_symlink():
            raise CompositionSourceError(
                "composition output contains a symbolic link"
            )
        if candidate.is_file():
            actual_files.add(candidate.relative_to(volume).as_posix())
    if actual_files != expected_files:
        raise CompositionSourceError(
            "composition output file inventory is incomplete or contains "
            f"unreferenced files (missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)})"
        )
    object_counts: dict[str, int] = {}
    for category in OBJECT_CATEGORIES:
        record = manifest.get(category)
        if not isinstance(record, Mapping):
            raise CompositionSourceError(
                f"composition output {category} inventory is malformed"
            )
        portable, _sha = _receipt_artifact_tuple(
            record, label=f"composition output {category}"
        )
        object_counts[category] = _verified_jsonl_count(
            physical_path=(volume / portable).resolve(),
            expected_count=record.get("count"),
            label=f"composition output {category}",
        )
    return {
        "state": "COMPOSITION_SOURCE_VERIFIED",
        "base_scene_id": manifest["base_scene_id"],
        "contract": expected_contract,
        "composition_source": {
            "path": manifest_path.relative_to(volume).as_posix(),
            "sha256": _sha256(manifest_path),
        },
        "artifact_count": len(inventory),
        "trees": object_counts["trees"],
        "buildings": object_counts["buildings"],
        "route_topology": topology,
        "route_source": normalized_route_source,
        "manifest": manifest,
    }


def export_from_contract(
    *,
    volume_root: Path,
    contract_path: Path,
    contract_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    """Execute the pod-facing, tile-streamed composition export contract."""

    volume = volume_root.expanduser().resolve()
    path = contract_path.expanduser().resolve()
    prepared = verify_prepared_contract(
        volume_root=volume,
        contract_path=path,
        contract_sha256=contract_sha256,
    )
    contract = prepared.get("contract_payload")
    if not isinstance(contract, dict):
        raise CompositionSourceError(
            "verified prepared contract payload is unavailable"
        )
    output = output_root.expanduser().resolve()
    if output.exists():
        verified = verify_composition_output(
            volume_root=volume,
            contract_path=path,
            contract_sha256=contract_sha256,
            output_root=output,
        )
        manifest = verified.get("manifest")
        if not isinstance(manifest, dict):
            raise CompositionSourceError(
                "verified existing composition has no manifest"
            )
        return manifest
    return export_composition_source(
        volume_root=volume,
        output_root=output,
        base_scene_id=str(contract.get("base_scene_id", "")),
        coordinate_contract=str(contract.get("coordinate_contract", "")),
        epsg2154_origin=(
            contract.get("epsg2154_origin")
            if isinstance(contract.get("epsg2154_origin"), list)
            else []
        ),
        native_artifacts=_native_artifacts_from_contract(
            contract.get("native_artifacts")
        ),
        bounds=contract.get("bounds", {}),
        height_field=_load_height_field_source(
            contract.get("height_field_source"),
            volume_root=volume,
        ),
        placement_height_tiles=_placement_height_sources_from_contract(
            contract.get("placement_height_tiles")
        ),
        ground_surface=_ground_surface_from_contract(
            contract.get("ground_surface")
        ),
        asset_library=_asset_sources_from_contract(
            contract.get("asset_library")
        ),
        water_material_lods=_water_material_source_from_contract(
            contract.get("water_material_source")
        ),
        trees=None,
        buildings=None,
        routes=contract["routes"],
        waters=contract["waters"],
        suitability_zones=contract["suitability_zones"],
        variant_constraints=(
            contract.get("variant_constraints")
            if isinstance(contract.get("variant_constraints"), Mapping)
            else None
        ),
        native_detail_extraction=_detail_extraction_from_contract(
            contract.get("native_detail_extraction")
        ),
        expected_placement_height_fingerprint=(
            str(contract.get("placement_height_fingerprint", ""))
        ),
        expected_route_topology=(
            contract.get("route_topology")
            if isinstance(contract.get("route_topology"), Mapping)
            else {}
        ),
        route_source_evidence=(
            contract.get("route_source")
            if isinstance(contract.get("route_source"), Mapping)
            else None
        ),
        source_contract_artifact={
            "path": path.relative_to(volume).as_posix(),
            "sha256": contract_sha256,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    commands = {"prepare-contract", "export", "verify", "build"}
    if (
        raw_argv
        and raw_argv[0] not in commands
        and raw_argv[0] not in {"-h", "--help"}
    ):
        # Preserve the original pod command while exposing explicit subcommands.
        raw_argv.insert(0, "export")
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and export a native build into a variant composition "
            "source without retaining the full object inventory in RAM"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser(
        "export",
        help="export an immutable prepared contract",
    )
    export_parser.add_argument("--volume-root", required=True)
    export_parser.add_argument("--contract", required=True)
    export_parser.add_argument("--contract-sha256", required=True)
    export_parser.add_argument("--output", required=True)
    verify_parser = subparsers.add_parser(
        "verify",
        help=(
            "verify a prepared contract and, when --output is supplied, its "
            "existing composition export"
        ),
    )
    verify_parser.add_argument("--volume-root", required=True)
    verify_parser.add_argument("--contract", required=True)
    verify_parser.add_argument("--contract-sha256", required=True)
    verify_parser.add_argument("--output")

    def add_preparation_arguments(
        command_parser: argparse.ArgumentParser,
    ) -> None:
        command_parser.add_argument("--volume-root", required=True)
        command_parser.add_argument("--zone-root", required=True)
        command_parser.add_argument(
            "--scene-auto-validation", required=True
        )
        command_parser.add_argument("--asset-manifest", required=True)
        command_parser.add_argument(
            "--asset-lod-validation", required=True
        )
        command_parser.add_argument(
            "--asset-pbr-validation", required=True
        )
        command_parser.add_argument(
            "--ground-artifact-root", required=True
        )
        command_parser.add_argument(
            "--ground-authoring-receipt", required=True
        )
        command_parser.add_argument("--prepared-output", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-contract",
        help=(
            "derive routes, waters, suitability and height data directly "
            "from accepted native payloads/locks"
        ),
    )
    add_preparation_arguments(prepare_parser)
    build_parser = subparsers.add_parser(
        "build",
        help="prepare the immutable contract and export it in one command",
    )
    add_preparation_arguments(build_parser)
    build_parser.add_argument("--output", required=True)
    args = parser.parse_args(raw_argv)
    if args.command == "verify":
        if args.output:
            verification = verify_composition_output(
                volume_root=Path(args.volume_root),
                contract_path=Path(args.contract),
                contract_sha256=args.contract_sha256,
                output_root=Path(args.output),
            )
        else:
            verification = verify_prepared_contract(
                volume_root=Path(args.volume_root),
                contract_path=Path(args.contract),
                contract_sha256=args.contract_sha256,
            )
        print(
            json.dumps(
                {
                    key: value
                    for key, value in verification.items()
                    if key not in {"contract_payload", "manifest"}
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if args.command in {"prepare-contract", "build"}:
        prepared = prepare_native_contract(
            volume_root=Path(args.volume_root),
            output_root=Path(args.prepared_output),
            source=NativePreparationSource(
                zone_root=args.zone_root,
                scene_auto_validation=args.scene_auto_validation,
                asset_manifest=args.asset_manifest,
                asset_lod_validation=args.asset_lod_validation,
                asset_pbr_validation=args.asset_pbr_validation,
                ground_artifact_root=args.ground_artifact_root,
                ground_authoring_receipt=args.ground_authoring_receipt,
            ),
        )
        if args.command == "prepare-contract":
            print(
                json.dumps(prepared, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
            return 0
        contract_path = (
            Path(args.prepared_output).expanduser().resolve()
            / "composition-export-input.json"
        )
        result = export_from_contract(
            volume_root=Path(args.volume_root),
            contract_path=contract_path,
            contract_sha256=prepared["contract"]["sha256"],
            output_root=Path(args.output),
        )
    else:
        result = export_from_contract(
            volume_root=Path(args.volume_root),
            contract_path=Path(args.contract),
            contract_sha256=args.contract_sha256,
            output_root=Path(args.output),
        )
    print(
        json.dumps(
            {
                "state": result["state"],
                "base_scene_id": result["base_scene_id"],
                "composition_source": (
                    Path(args.output).expanduser().resolve()
                    / "composition-source.json"
                ).as_posix(),
                "trees": result["trees"]["count"],
                "buildings": result["buildings"]["count"],
                "memory_contract": result["streaming_memory_contract"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


__all__ = [
    "ArtifactSource",
    "AssetSource",
    "CompositionSourceError",
    "DetailPayloadExtractionSource",
    "GroundSurfaceSource",
    "HeightFieldSource",
    "MAX_HEIGHT_FIELD_SAMPLES",
    "MAX_HEIGHT_FIELD_SIDE",
    "NativeArtifactsSource",
    "NativePreparationSource",
    "ROOT_LOCAL_COORDINATE_CONTRACT",
    "TerrainPayloadSource",
    "WaterMaterialSource",
    "export_composition_source",
    "export_from_contract",
    "main",
    "prepare_native_contract",
    "verify_composition_output",
    "verify_prepared_contract",
]


if __name__ == "__main__":
    raise SystemExit(main())

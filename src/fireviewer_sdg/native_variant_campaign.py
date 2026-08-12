"""Fail-closed bridge from accepted native zone layouts to 20 OpenUSD variants.

``scene_variants`` owns the deterministic spatial composition.  This module
owns the portable interchange and the native USD authoring boundary:

* exactly four hash-bound, automatically validated native base builds;
* exactly five variants per base;
* exact preservation of object stable IDs, numeric instance IDs and counts;
* exact reuse of the 400 terrain payloads and isolated water payloads;
* real HERO/MID/FAR USD asset references for every tree and building;
* object-free PBR ground material overriding source orthophoto bindings;
* source-backed route topology retained for placement and annotations; road
  appearance comes exclusively from the orthophoto draped over the terrain.

The base layout is deliberately stricter than the legacy build receipt.  A
receipt alone cannot reconstruct a global road network, instance identities,
asset LOD lineage or an isolated water layer.  Missing information therefore
raises :class:`NativeVariantContractError`; it is never replaced by procedural
objects, primitives, guessed geometry or empty zones.

``pxr`` is imported only inside the authoring backend.  Planning and validation
remain import-safe on ordinary CI machines.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fireviewer_sdg.scene_variants import (
    ALGORITHM_ID,
    BASE_SCENE_COUNT,
    PORTFOLIO_SCENE_COUNT,
    VARIANTS_PER_BASE,
    BaseScene,
    Bounds,
    BridgeSpan,
    FamilySuitability,
    GroundSurfaceContract,
    HeightField,
    PlacementHeightTile,
    SceneAsset,
    SceneRoute,
    SceneVariant,
    SuitabilityZone,
    TiledHeightField,
    VariantConstraints,
    Vec2,
    Vec3,
    WaterFeature,
    family_counts,
    validate_scene_variant,
)
from fireviewer_sdg import scene_variants as _scene_variants
from fireviewer_sdg.campaign_asset_bundle import (
    REQUIRED_ACTOR_CLASSES,
    SELECTED_ACTOR_GROUP_ID,
    SELECTED_ACTOR_GROUP_IDS,
    SELECTED_ACTOR_GROUP_PLACEMENT_BY_ID,
    SELECTED_ACTOR_GROUP_SOURCE_BY_ID,
    SELECTED_ENVIRONMENT_GROUP_ID,
    SELECTED_ENVIRONMENT_GROUP_IDS,
    SELECTED_ENVIRONMENT_TARGET_BY_ID,
)


SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
AUTHORING_SCHEMA_VERSION = 1
LOD_LEVELS = ("HERO", "MID", "FAR")
OBJECT_CATEGORIES = ("trees", "buildings")
ACTORS_PER_SCENE = len(SELECTED_ACTOR_GROUP_IDS)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_USD_SUFFIXES = {".usd", ".usda", ".usdc"}
_SAFE_ID = re.compile(r"[^A-Za-z0-9_]")
_FORBIDDEN_ASSET = re.compile(
    r"(?:^|[/_.-])(cube|cone|cylinder|sphere|capsule|primitive|placeholder)"
    r"(?:$|[/_.-])",
    re.IGNORECASE,
)
_MAX_INT64 = 2**63 - 1
ROAD_VISUAL_CONTRACT = {
    "visible_representation": "orthophoto_derived_terrain_material",
    "geometry_authoring": "disabled",
    "asset_dependencies": [],
    "route_vectors_retained_for": [
        "topology",
        "actor_placement",
        "annotations",
        "composition_constraints",
    ],
}
NETWORK_GEOMETRY_POLICY = (
    "hydrology_hash_bound_20m_fragments_all_lods;"
    "roads_orthophoto_derived_terrain_material_no_route_meshes;"
    "measured_vertex_budget"
)


class NativeVariantContractError(RuntimeError):
    """Raised when a faithful variant cannot be planned or authored."""


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    path: str
    sha256: str
    prim_path: str = ""
    isolated_content_roles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TerrainTile:
    tile_ref: str
    local_bounds: Bounds
    epsg2154_bounds: Bounds
    instance_namespace: int
    terrain_lods: tuple[str, ...]
    collision_lods: tuple[str, ...]
    artifact: ArtifactRef


@dataclass(frozen=True, slots=True)
class GroundMaterialTile:
    tile_id: str
    local_bounds: Bounds
    artifact: ArtifactRef


@dataclass(frozen=True, slots=True)
class GroundMaterialBinding:
    topology: str
    index: ArtifactRef
    tile_material_payloads: tuple[GroundMaterialTile, ...]


@dataclass(frozen=True, slots=True)
class AssetBinding:
    key: str
    category: str
    family: str
    lods: Mapping[str, ArtifactRef]
    grounding_offsets_m: Mapping[str, float]
    lineage: str


@dataclass(frozen=True, slots=True)
class ActorBinding:
    selection_id: str
    asset_id: str
    family: str
    placement_class: str
    source_url: str
    lods: Mapping[str, ArtifactRef]
    ground_anchor_m: Vec3
    semantic_roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SupplementalEnvironmentBinding:
    selection_id: str
    asset_id: str
    environment_kind: str
    environment_family: str
    lods: Mapping[str, ArtifactRef]
    ground_anchor_m: Vec3


@dataclass(frozen=True, slots=True)
class ObjectBinding:
    stable_id: str
    numeric_id: int
    asset_key: str


@dataclass(frozen=True, slots=True)
class RouteBinding:
    stable_id: str
    numeric_id: int
    # Source surface classification only.  It is retained for topology and
    # annotations and never resolves to a rendered asset or USD material.
    surface_class: str


@dataclass(frozen=True, slots=True)
class NativeBaseLayout:
    layout_path: Path
    layout_sha256: str
    scene: BaseScene
    build_receipt: ArtifactRef
    auto_validation: ArtifactRef
    root_usd: ArtifactRef
    asset_lock: ArtifactRef
    asset_lock_assets: tuple[Mapping[str, Any], ...]
    shared_asset_manifest: ArtifactRef
    asset_content_sha256: str
    review_cameras: ArtifactRef
    review_camera_count: int
    epsg2154_origin: Vec3
    terrain_payloads: tuple[TerrainTile, ...]
    preview_height_field: ArtifactRef
    placement_height_fingerprint: str
    water_payloads: tuple[ArtifactRef, ...]
    ground_material: GroundMaterialBinding
    assets: Mapping[str, AssetBinding]
    selected_actors: Mapping[str, ActorBinding]
    supplemental_environment: Mapping[str, SupplementalEnvironmentBinding]
    object_bindings: Mapping[str, ObjectBinding]
    water_materials: Mapping[str, ArtifactRef]
    route_bindings: Mapping[str, RouteBinding]
    route_component_count: int
    route_membership_sha256: str
    variant_constraints: VariantConstraints


@dataclass(frozen=True, slots=True)
class NativeBuildBindings:
    root_usd: ArtifactRef
    asset_lock: ArtifactRef
    asset_lock_assets: tuple[Mapping[str, Any], ...]
    shared_asset_manifest: ArtifactRef
    asset_content_sha256: str
    review_cameras: ArtifactRef
    review_camera_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeVariantContractError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(payload, dict):
        raise NativeVariantContractError(f"{label} must be a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _is_below(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _portable_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise NativeVariantContractError(f"{label}.path must be a non-empty string")
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise NativeVariantContractError(
            f"{label}.path must be relative to the persistent volume"
        )
    return normalized


def _artifact(
    payload: object,
    *,
    volume_root: Path,
    label: str,
    require_usd: bool = False,
    require_prim: bool = False,
    roles: tuple[str, ...] | None = None,
) -> ArtifactRef:
    if not isinstance(payload, dict):
        raise NativeVariantContractError(f"{label} must be an artifact object")
    portable = _portable_path(payload.get("path"), label=label)
    expected = payload.get("sha256")
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        raise NativeVariantContractError(f"{label}.sha256 must be lowercase SHA-256")
    resolved = (volume_root / Path(portable)).resolve()
    if not _is_below(volume_root, resolved) or not resolved.is_file():
        raise NativeVariantContractError(f"{label} file is absent: {portable}")
    actual = _sha256(resolved)
    if actual != expected:
        raise NativeVariantContractError(
            f"{label} hash mismatch: expected {expected}, found {actual}"
        )
    if require_usd and resolved.suffix.casefold() not in _USD_SUFFIXES:
        raise NativeVariantContractError(f"{label} must reference a USD layer")
    if require_usd and _FORBIDDEN_ASSET.search(portable):
        raise NativeVariantContractError(f"{label} references a forbidden primitive")
    prim_path = payload.get("prim_path", "")
    if not isinstance(prim_path, str):
        raise NativeVariantContractError(f"{label}.prim_path must be a string")
    if require_prim and (not prim_path.startswith("/") or prim_path == "/"):
        raise NativeVariantContractError(
            f"{label}.prim_path must identify a concrete absolute USD prim"
        )
    raw_roles = payload.get("isolated_content_roles", [])
    if not isinstance(raw_roles, list) or any(
        not isinstance(role, str) or not role.strip() for role in raw_roles
    ):
        raise NativeVariantContractError(
            f"{label}.isolated_content_roles must be a string list"
        )
    normalized_roles = tuple(sorted(set(raw_roles)))
    if roles is not None and normalized_roles != tuple(sorted(roles)):
        raise NativeVariantContractError(
            f"{label} must isolate only these content roles: {', '.join(roles)}"
        )
    return ArtifactRef(
        path=portable,
        sha256=expected,
        prim_path=prim_path,
        isolated_content_roles=normalized_roles,
    )


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeVariantContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise NativeVariantContractError(f"{label} must be finite")
    return result


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NativeVariantContractError(f"{label} must be an integer >= {minimum}")
    return value


def _vec2(payload: object, *, label: str) -> Vec2:
    if not isinstance(payload, list) or len(payload) != 2:
        raise NativeVariantContractError(f"{label} must contain two coordinates")
    return Vec2(
        _number(payload[0], label=f"{label}[0]"),
        _number(payload[1], label=f"{label}[1]"),
    )


def _vec3(payload: object, *, label: str) -> Vec3:
    if not isinstance(payload, list) or len(payload) != 3:
        raise NativeVariantContractError(f"{label} must contain three coordinates")
    return Vec3(
        _number(payload[0], label=f"{label}[0]"),
        _number(payload[1], label=f"{label}[1]"),
        _number(payload[2], label=f"{label}[2]"),
    )


def _bounds(payload: object) -> Bounds:
    if not isinstance(payload, dict):
        raise NativeVariantContractError("bounds must be an object")
    return Bounds(
        _number(payload.get("min_x"), label="bounds.min_x"),
        _number(payload.get("min_y"), label="bounds.min_y"),
        _number(payload.get("max_x"), label="bounds.max_x"),
        _number(payload.get("max_y"), label="bounds.max_y"),
    )


def _terrain_tile(
    payload: object,
    *,
    volume_root: Path,
    base_id: str,
    index: int,
) -> TerrainTile:
    label = f"{base_id}.terrain_payloads[{index}]"
    if not isinstance(payload, dict):
        raise NativeVariantContractError(f"{label} must be an object")
    tile_ref = payload.get("tile_ref")
    if not isinstance(tile_ref, str) or not tile_ref.strip():
        raise NativeVariantContractError(f"{label}.tile_ref is required")
    namespace = _integer(
        payload.get("instance_namespace"),
        label=f"{label}.instance_namespace",
        minimum=1,
    )
    if namespace >= 1 << 20:
        raise NativeVariantContractError(
            f"{label}.instance_namespace exceeds the stable-ID contract"
        )
    raw_lods = payload.get("terrain_lods")
    if (
        not isinstance(raw_lods, list)
        or any(not isinstance(item, str) for item in raw_lods)
        or len(raw_lods) != len(set(raw_lods))
        or not {"LOD1", "LOD2", "LOD3"}.issubset(raw_lods)
        or not set(raw_lods).issubset(
            {"LOD0", "LOD1", "LOD2", "LOD3"}
        )
    ):
        raise NativeVariantContractError(
            f"{label}.terrain_lods must expose real LOD1/LOD2/LOD3 "
            "and optional real LOD0"
        )
    raw_collision_lods = payload.get("collision_lods")
    if raw_collision_lods != ["NEAR", "FAR"]:
        raise NativeVariantContractError(
            f"{label}.collision_lods must be the real NEAR/FAR pair"
        )
    try:
        local_bounds = _bounds(payload.get("local_bounds"))
    except (ValueError, NativeVariantContractError) as error:
        raise NativeVariantContractError(
            f"{label}.local_bounds is invalid: {error}"
        ) from error
    try:
        epsg2154_bounds = _bounds(payload.get("epsg2154_bounds"))
    except (ValueError, NativeVariantContractError) as error:
        raise NativeVariantContractError(
            f"{label}.epsg2154_bounds is invalid: {error}"
        ) from error
    artifact = _artifact(
        payload,
        volume_root=volume_root,
        label=label,
        require_usd=True,
        require_prim=True,
        roles=("terrain",),
    )
    return TerrainTile(
        tile_ref,
        local_bounds,
        epsg2154_bounds,
        namespace,
        tuple(raw_lods),
        tuple(raw_collision_lods),
        artifact,
    )


def _validate_tile_partition(
    tiles: Sequence[TerrainTile],
    *,
    scene_bounds: Bounds,
    epsg2154_origin: Vec3,
    base_id: str,
) -> None:
    if len(tiles) != 400:
        raise NativeVariantContractError(
            f"{base_id} requires exactly 400 terrain/detail streaming tiles"
        )
    if len({tile.tile_ref for tile in tiles}) != len(tiles):
        raise NativeVariantContractError(f"{base_id} tile_ref values repeat")
    if len({tile.instance_namespace for tile in tiles}) != len(tiles):
        raise NativeVariantContractError(
            f"{base_id} instance namespaces repeat"
        )
    if len({tile.artifact.path for tile in tiles}) != len(tiles):
        raise NativeVariantContractError(
            f"{base_id} terrain payload paths repeat"
        )
    if not any("LOD0" in tile.terrain_lods for tile in tiles):
        raise NativeVariantContractError(
            f"{base_id} review-camera working set exposes no real LOD0 terrain"
        )
    total_area = 0.0
    for index, tile in enumerate(tiles):
        bounds = tile.local_bounds
        epsg = tile.epsg2154_bounds
        if not all(
            math.isclose(actual, expected, abs_tol=0.01)
            for actual, expected in (
                (epsg.min_x, epsg2154_origin.x + bounds.min_x),
                (epsg.min_y, epsg2154_origin.y + bounds.min_y),
                (epsg.max_x, epsg2154_origin.x + bounds.max_x),
                (epsg.max_y, epsg2154_origin.y + bounds.max_y),
            )
        ):
            raise NativeVariantContractError(
                f"{base_id} tile {tile.tile_ref} local/EPSG:2154 bounds diverge"
            )
        if (
            bounds.min_x < scene_bounds.min_x - 1.0e-6
            or bounds.min_y < scene_bounds.min_y - 1.0e-6
            or bounds.max_x > scene_bounds.max_x + 1.0e-6
            or bounds.max_y > scene_bounds.max_y + 1.0e-6
        ):
            raise NativeVariantContractError(
                f"{base_id} tile {tile.tile_ref} leaves scene bounds"
            )
        total_area += bounds.width * bounds.height
        for other in tiles[index + 1 :]:
            overlap_x = min(bounds.max_x, other.local_bounds.max_x) - max(
                bounds.min_x, other.local_bounds.min_x
            )
            overlap_y = min(bounds.max_y, other.local_bounds.max_y) - max(
                bounds.min_y, other.local_bounds.min_y
            )
            if overlap_x > 1.0e-6 and overlap_y > 1.0e-6:
                raise NativeVariantContractError(
                    f"{base_id} tiles {tile.tile_ref} and "
                    f"{other.tile_ref} overlap"
                )
    scene_area = scene_bounds.width * scene_bounds.height
    if not math.isclose(total_area, scene_area, rel_tol=0.0, abs_tol=1.0e-3):
        raise NativeVariantContractError(
            f"{base_id} tile partition covers {total_area:.3f} m2, "
            f"expected {scene_area:.3f} m2"
        )


def _ground_material_binding(
    payload: object,
    *,
    volume_root: Path,
    base_id: str,
) -> GroundMaterialBinding:
    label = f"{base_id}.ground_material"
    if not isinstance(payload, dict):
        raise NativeVariantContractError(f"{label} must be an object")
    topology = payload.get("topology")
    if topology != "payload_tiled_materials_shared_pbr_library":
        raise NativeVariantContractError(
            f"{label} must use the tiled shared-PBR topology"
        )
    index = _artifact(
        payload,
        volume_root=volume_root,
        label=label,
        require_usd=True,
        require_prim=True,
        roles=("object_free_pbr_ground",),
    )
    raw_tiles = payload.get("tile_material_payloads")
    if not isinstance(raw_tiles, list) or len(raw_tiles) != 400:
        raise NativeVariantContractError(
            f"{label} requires exactly 400 terrain-tile material payloads"
        )
    records: list[GroundMaterialTile] = []
    identifiers: set[str] = set()
    for index_number, raw in enumerate(raw_tiles):
        item_label = f"{label}.tile_material_payloads[{index_number}]"
        if not isinstance(raw, dict):
            raise NativeVariantContractError(f"{item_label} is malformed")
        tile_id = raw.get("tile_id")
        if (
            not isinstance(tile_id, str)
            or not tile_id.strip()
            or raw.get("tile_ref") != tile_id
            or tile_id in identifiers
        ):
            raise NativeVariantContractError(
                f"{item_label}.tile_id/tile_ref is missing, divergent or duplicated"
            )
        identifiers.add(tile_id)
        raw_bounds = raw.get("tile_bounds_m")
        try:
            if isinstance(raw_bounds, list) and len(raw_bounds) == 4:
                bounds = Bounds(
                    *(
                        _number(
                            value,
                            label=f"{item_label}.tile_bounds_m[{offset}]",
                        )
                        for offset, value in enumerate(raw_bounds)
                    )
                )
            else:
                bounds = _bounds(raw_bounds)
        except (ValueError, NativeVariantContractError) as error:
            raise NativeVariantContractError(
                f"{item_label}.tile_bounds_m is invalid: {error}"
            ) from error
        artifact = _artifact(
            raw,
            volume_root=volume_root,
            label=item_label,
            require_usd=True,
            require_prim=True,
        )
        if artifact.prim_path != "/Ground":
            raise NativeVariantContractError(
                f"{item_label}.prim_path must be /Ground"
            )
        records.append(GroundMaterialTile(tile_id, bounds, artifact))
    return GroundMaterialBinding(
        topology=topology,
        index=index,
        tile_material_payloads=tuple(records),
    )


def _ground_material_for_terrain(
    terrain: TerrainTile,
    *,
    ground: GroundMaterialBinding,
    base_id: str,
) -> GroundMaterialTile:
    matches = [
        material
        for material in ground.tile_material_payloads
        if material.tile_id == terrain.tile_ref
    ]
    if len(matches) != 1:
        raise NativeVariantContractError(
            f"{base_id} terrain tile {terrain.tile_ref} maps to "
            f"{len(matches)} ground-material regions"
        )
    material = matches[0]
    if not all(
        math.isclose(actual, expected, abs_tol=0.01)
        for actual, expected in (
            (material.local_bounds.min_x, terrain.local_bounds.min_x),
            (material.local_bounds.min_y, terrain.local_bounds.min_y),
            (material.local_bounds.max_x, terrain.local_bounds.max_x),
            (material.local_bounds.max_y, terrain.local_bounds.max_y),
        )
    ):
        raise NativeVariantContractError(
            f"{base_id} terrain/material bounds diverge for {terrain.tile_ref}"
        )
    return material


def _height_field(payload: object) -> HeightField:
    if not isinstance(payload, dict):
        raise NativeVariantContractError("height field must be an object")
    rows = payload.get("samples")
    if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
        raise NativeVariantContractError("height field samples must be a 2D array")
    return HeightField(
        origin_x=_number(payload.get("origin_x"), label="height_field.origin_x"),
        origin_y=_number(payload.get("origin_y"), label="height_field.origin_y"),
        spacing_m=_number(payload.get("spacing_m"), label="height_field.spacing_m"),
        samples=tuple(
            tuple(
                _number(value, label=f"height_field.samples[{row_index}]")
                for value in row
            )
            for row_index, row in enumerate(rows)
        ),
    )


def _placement_height_field(
    payload: object,
    *,
    fingerprint: object,
    terrain: Sequence[TerrainTile],
    volume_root: Path,
    base_id: str,
) -> TiledHeightField:
    if not isinstance(payload, list) or len(payload) != 400:
        raise NativeVariantContractError(
            f"{base_id} requires exactly 400 placement height tiles"
        )
    if not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint):
        raise NativeVariantContractError(
            f"{base_id}.placement_height_fingerprint is invalid"
        )
    terrain_by_ref = {tile.tile_ref: tile for tile in terrain}
    placements: list[PlacementHeightTile] = []
    normalized_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload):
        label = f"{base_id}.placement_height_tiles[{index}]"
        if not isinstance(raw, dict):
            raise NativeVariantContractError(f"{label} is malformed")
        tile_ref = raw.get("tile_ref")
        if (
            not isinstance(tile_ref, str)
            or tile_ref not in terrain_by_ref
            or tile_ref in seen
        ):
            raise NativeVariantContractError(
                f"{label}.tile_ref is unknown or duplicated"
            )
        seen.add(tile_ref)
        try:
            bounds = _bounds(raw.get("local_bounds"))
        except (ValueError, NativeVariantContractError) as error:
            raise NativeVariantContractError(
                f"{label}.local_bounds is invalid: {error}"
            ) from error
        terrain_bounds = terrain_by_ref[tile_ref].local_bounds
        if not all(
            math.isclose(actual, expected, abs_tol=0.01)
            for actual, expected in (
                (bounds.min_x, terrain_bounds.min_x),
                (bounds.min_y, terrain_bounds.min_y),
                (bounds.max_x, terrain_bounds.max_x),
                (bounds.max_y, terrain_bounds.max_y),
            )
        ):
            raise NativeVariantContractError(
                f"{label} bounds differ from its terrain payload"
            )
        if raw.get("format") != "float32-le-row-major-south-to-north":
            raise NativeVariantContractError(
                f"{label}.format is unsupported"
            )
        width = _integer(raw.get("width"), label=f"{label}.width", minimum=2)
        height = _integer(raw.get("height"), label=f"{label}.height", minimum=2)
        raw_x = raw.get("x_coordinates")
        raw_y = raw.get("y_coordinates")
        if (
            not isinstance(raw_x, list)
            or not isinstance(raw_y, list)
            or len(raw_x) != width
            or len(raw_y) != height
        ):
            raise NativeVariantContractError(
                f"{label} axes do not match width/height"
            )
        x_coordinates = tuple(
            _number(value, label=f"{label}.x_coordinates")
            for value in raw_x
        )
        y_coordinates = tuple(
            _number(value, label=f"{label}.y_coordinates")
            for value in raw_y
        )
        artifact = _artifact(
            raw,
            volume_root=volume_root,
            label=label,
        )
        physical_path = (volume_root / artifact.path).resolve()
        expected_bytes = width * height * 4
        if physical_path.stat().st_size != expected_bytes:
            raise NativeVariantContractError(
                f"{label} byte size differs from its float32 grid shape"
            )

        def load_samples(
            *,
            path: Path = physical_path,
            columns: int = width,
            rows: int = height,
            expected_size: int = expected_bytes,
            source_label: str = label,
        ) -> Sequence[Sequence[float]]:
            if path.stat().st_size != expected_size:
                raise NativeVariantContractError(
                    f"{source_label} changed size after layout validation"
                )
            values = array("f")
            try:
                with path.open("rb") as stream:
                    values.fromfile(stream, columns * rows)
                    if stream.read(1):
                        raise NativeVariantContractError(
                            f"{source_label} contains trailing bytes"
                        )
            except (EOFError, OSError) as error:
                raise NativeVariantContractError(
                    f"{source_label} sample grid cannot be loaded"
                ) from error
            if sys.byteorder != "little":
                values.byteswap()
            return tuple(
                values[row * columns : (row + 1) * columns]
                for row in range(rows)
            )

        try:
            placements.append(
                PlacementHeightTile(
                    tile_ref=tile_ref,
                    bounds=bounds,
                    x_coordinates=x_coordinates,
                    y_coordinates=y_coordinates,
                    sample_sha256=artifact.sha256,
                    sample_loader=load_samples,
                )
            )
        except ValueError as error:
            raise NativeVariantContractError(str(error)) from error
        normalized_records.append(
            {
                "tile_ref": tile_ref,
                "local_bounds": {
                    "min_x": bounds.min_x,
                    "min_y": bounds.min_y,
                    "max_x": bounds.max_x,
                    "max_y": bounds.max_y,
                },
                "path": artifact.path,
                "sha256": artifact.sha256,
                "format": "float32-le-row-major-south-to-north",
                "width": width,
                "height": height,
                "x_coordinates": list(x_coordinates),
                "y_coordinates": list(y_coordinates),
            }
        )
    if seen != set(terrain_by_ref):
        raise NativeVariantContractError(
            f"{base_id} placement/terrain tile identities differ"
        )
    computed = hashlib.sha256(
        json.dumps(
            sorted(normalized_records, key=lambda item: item["tile_ref"]),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if computed != fingerprint:
        raise NativeVariantContractError(
            f"{base_id}.placement_height_fingerprint is stale"
        )
    try:
        return TiledHeightField(
            tiles=tuple(placements),
            content_fingerprint=fingerprint,
            expected_tile_count=400,
            cache_tile_limit=2,
        )
    except ValueError as error:
        raise NativeVariantContractError(str(error)) from error


def _ground_surface(payload: object, *, ground_material: ArtifactRef) -> GroundSurfaceContract:
    if not isinstance(payload, dict):
        raise NativeVariantContractError("ground_surface must be an object")
    removed = payload.get("removed_object_classes", [])
    if not isinstance(removed, list) or any(not isinstance(item, str) for item in removed):
        raise NativeVariantContractError(
            "ground_surface.removed_object_classes must be a string list"
        )
    fingerprint = payload.get("content_fingerprint")
    if fingerprint != ground_material.sha256:
        raise NativeVariantContractError(
            "ground surface fingerprint must equal the bound PBR material hash"
        )
    try:
        return GroundSurfaceContract(
            kind=str(payload.get("kind", "")),
            material_ref=ground_material.path,
            content_fingerprint=str(fingerprint),
            removed_object_classes=frozenset(removed),
        )
    except ValueError as error:
        raise NativeVariantContractError(str(error)) from error


def _bridge_span(payload: object, *, label: str) -> BridgeSpan:
    if not isinstance(payload, dict):
        raise NativeVariantContractError(f"{label} must be an object")
    try:
        return BridgeSpan(
            stable_id=str(payload.get("stable_id", "")),
            start_fraction=_number(
                payload.get("start_fraction"), label=f"{label}.start_fraction"
            ),
            water_start_fraction=_number(
                payload.get("water_start_fraction"),
                label=f"{label}.water_start_fraction",
            ),
            water_end_fraction=_number(
                payload.get("water_end_fraction"),
                label=f"{label}.water_end_fraction",
            ),
            end_fraction=_number(
                payload.get("end_fraction"), label=f"{label}.end_fraction"
            ),
            minimum_deck_clearance_m=_number(
                payload.get("minimum_deck_clearance_m"),
                label=f"{label}.minimum_deck_clearance_m",
            ),
        )
    except ValueError as error:
        raise NativeVariantContractError(str(error)) from error


def _route(payload: object, *, index: int) -> tuple[SceneRoute, RouteBinding]:
    label = f"routes[{index}]"
    if not isinstance(payload, dict):
        raise NativeVariantContractError(f"{label} must be an object")
    raw_points = payload.get("points")
    raw_bridges = payload.get("bridge_spans", [])
    if not isinstance(raw_points, list) or not isinstance(raw_bridges, list):
        raise NativeVariantContractError(f"{label} geometry must be arrays")
    stable_id = str(payload.get("stable_id", ""))
    try:
        route = SceneRoute(
            stable_id=stable_id,
            family=str(payload.get("family", "")),
            points=tuple(
                _vec3(point, label=f"{label}.points[{point_index}]")
                for point_index, point in enumerate(raw_points)
            ),
            width_m=_number(payload.get("width_m"), label=f"{label}.width_m"),
            bridge_spans=tuple(
                _bridge_span(bridge, label=f"{label}.bridge_spans[{bridge_index}]")
                for bridge_index, bridge in enumerate(raw_bridges)
            ),
        )
    except ValueError as error:
        raise NativeVariantContractError(str(error)) from error
    numeric_id = _integer(
        payload.get("numeric_id"), label=f"{label}.numeric_id", minimum=1
    )
    if numeric_id > _MAX_INT64:
        raise NativeVariantContractError(f"{label}.numeric_id exceeds signed int64")
    surface_class = payload.get("surface_class", payload.get("material_key"))
    if not isinstance(surface_class, str) or not surface_class.strip():
        raise NativeVariantContractError(
            f"{label}.surface_class is required"
        )
    return route, RouteBinding(stable_id, numeric_id, surface_class)


def _water(payload: object, *, index: int) -> WaterFeature:
    label = f"waters[{index}]"
    if not isinstance(payload, dict):
        raise NativeVariantContractError(f"{label} must be an object")
    outlines = payload.get("outline")
    centrelines = payload.get("centreline", [])
    surfaces = payload.get("surface_profile_m")
    if (
        not isinstance(outlines, list)
        or not isinstance(centrelines, list)
        or not isinstance(surfaces, list)
    ):
        raise NativeVariantContractError(f"{label} geometry must be arrays")
    try:
        return WaterFeature(
            stable_id=str(payload.get("stable_id", "")),
            family=str(payload.get("family", "")),
            outline=tuple(
                _vec2(point, label=f"{label}.outline[{point_index}]")
                for point_index, point in enumerate(outlines)
            ),
            kind=str(payload.get("kind", "")),
            centreline=tuple(
                _vec2(point, label=f"{label}.centreline[{point_index}]")
                for point_index, point in enumerate(centrelines)
            ),
            surface_profile_m=tuple(
                _number(value, label=f"{label}.surface_profile_m")
                for value in surfaces
            ),
        )
    except ValueError as error:
        raise NativeVariantContractError(str(error)) from error


def _suitability(payload: object, *, index: int) -> SuitabilityZone:
    label = f"suitability_zones[{index}]"
    if not isinstance(payload, dict):
        raise NativeVariantContractError(f"{label} must be an object")
    outline = payload.get("outline")
    families = payload.get("tree_families")
    if not isinstance(outline, list) or not isinstance(families, list):
        raise NativeVariantContractError(f"{label} outline/families must be arrays")
    try:
        return SuitabilityZone(
            stable_id=str(payload.get("stable_id", "")),
            outline=tuple(
                _vec2(point, label=f"{label}.outline[{point_index}]")
                for point_index, point in enumerate(outline)
            ),
            biome=str(payload.get("biome", "")),
            soil=str(payload.get("soil", "")),
            tree_families=frozenset(str(value) for value in families),
            buildable=payload.get("buildable") is True,
        )
    except ValueError as error:
        raise NativeVariantContractError(str(error)) from error


def _layout_constraints(
    payload: object,
    *,
    zones: Sequence[SuitabilityZone],
    tree_families: set[str],
) -> VariantConstraints:
    if not isinstance(payload, dict):
        raise NativeVariantContractError(
            "variant_constraints must explicitly define every numeric limit"
        )
    expected = {
        field.name
        for field in dataclasses.fields(VariantConstraints)
        if field.name != "tree_suitability"
    }
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise NativeVariantContractError(
            "variant_constraints fields differ from the fixed contract "
            f"(missing={missing}, extra={extra})"
        )
    suitability: list[FamilySuitability] = []
    for family in sorted(tree_families):
        habitats = [zone for zone in zones if family in zone.tree_families]
        if not habitats:
            raise NativeVariantContractError(
                f"tree family {family} has no biome/soil habitat"
            )
        suitability.append(
            FamilySuitability(
                family=family,
                allowed_biomes=frozenset(zone.biome for zone in habitats),
                allowed_soils=frozenset(zone.soil for zone in habitats),
            )
        )
    try:
        return VariantConstraints(
            **payload,
            tree_suitability=tuple(suitability),
        )
    except (TypeError, ValueError) as error:
        raise NativeVariantContractError(
            f"variant_constraints are invalid: {error}"
        ) from error


def _asset_library(
    payload: object, *, volume_root: Path
) -> dict[str, AssetBinding]:
    if not isinstance(payload, dict) or not payload:
        raise NativeVariantContractError("asset_library must be a non-empty object")
    result: dict[str, AssetBinding] = {}
    for key, raw in payload.items():
        label = f"asset_library.{key}"
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(raw, dict)
        ):
            raise NativeVariantContractError(f"{label} is malformed")
        category = raw.get("category")
        family = raw.get("family")
        if category not in OBJECT_CATEGORIES:
            raise NativeVariantContractError(f"{label}.category must be trees/buildings")
        if not isinstance(family, str) or not family.strip():
            raise NativeVariantContractError(f"{label}.family is required")
        validation = raw.get("simready_validation")
        if not isinstance(validation, dict) or validation.get("state") != "SIMREADY_VALIDATED":
            raise NativeVariantContractError(
                f"{label} is not bound to SIMREADY_VALIDATED asset evidence"
            )
        lineage = validation.get("lod_lineage")
        if not isinstance(lineage, str) or not lineage.strip():
            raise NativeVariantContractError(f"{label} lacks a common LOD lineage")
        raw_offsets = validation.get("grounding_offsets_m")
        if not isinstance(raw_offsets, dict) or set(raw_offsets) != set(LOD_LEVELS):
            raise NativeVariantContractError(
                f"{label} requires grounding offsets for HERO, MID and FAR"
            )
        offsets = {
            level: _number(
                raw_offsets[level],
                label=f"{label}.simready_validation.grounding_offsets_m.{level}",
            )
            for level in LOD_LEVELS
        }
        raw_lods = raw.get("lods")
        if not isinstance(raw_lods, dict) or set(raw_lods) != set(LOD_LEVELS):
            raise NativeVariantContractError(
                f"{label} requires distinct HERO, MID and FAR USD assets"
            )
        lods = {
            level: _artifact(
                raw_lods[level],
                volume_root=volume_root,
                label=f"{label}.lods.{level}",
                require_usd=True,
                require_prim=True,
            )
            for level in LOD_LEVELS
        }
        if len({entry.path for entry in lods.values()}) != len(LOD_LEVELS):
            raise NativeVariantContractError(
                f"{label} HERO, MID and FAR must be distinct files"
            )
        result[key] = AssetBinding(
            key=key,
            category=category,
            family=family,
            lods=lods,
            grounding_offsets_m=offsets,
            lineage=lineage,
        )
    return result


def _selected_actor_library(
    manifest: ArtifactRef,
    *,
    volume_root: Path,
) -> dict[str, ActorBinding]:
    """Load the exact Chrome actor group from the shared locked manifest."""

    path = (volume_root / manifest.path).resolve()
    payload = _read_json(path, label="shared selected-actor manifest")
    group = payload.get("selected_actor_group")
    assets = group.get("assets") if isinstance(group, dict) else None
    expected_ids = set(SELECTED_ACTOR_GROUP_IDS)
    actual_ids = set(assets) if isinstance(assets, dict) else set()
    if (
        not isinstance(group, dict)
        or group.get("group_id") != SELECTED_ACTOR_GROUP_ID
        or group.get("selection_count") != len(SELECTED_ACTOR_GROUP_IDS)
        or group.get("selection_order") != list(SELECTED_ACTOR_GROUP_IDS)
        or not isinstance(assets, dict)
        or actual_ids != expected_ids
        or group.get("usage_contract")
        != "all_selected_assets_must_be_placed_across_the_20_scene_campaign"
    ):
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise NativeVariantContractError(
            "shared manifest does not contain the exact locked selected "
            f"Chrome actor group (missing={missing}, unexpected={unexpected})"
        )
    raw_roles = payload.get("actors")
    role_keys = set(raw_roles) if isinstance(raw_roles, dict) else set()
    if (
        not isinstance(raw_roles, dict)
        or role_keys != set(REQUIRED_ACTOR_CLASSES)
        or any(not isinstance(value, dict) for value in raw_roles.values())
    ):
        raise NativeVariantContractError(
            "shared manifest does not retain the independent seven-role "
            "semantic actor minimum"
        )

    result: dict[str, ActorBinding] = {}
    seen_asset_ids: set[str] = set()
    for selection_id in SELECTED_ACTOR_GROUP_IDS:
        raw = assets[selection_id]
        label = f"selected_actor_group.assets.{selection_id}"
        if not isinstance(raw, dict):
            raise NativeVariantContractError(f"{label} must be an object")
        asset_id = str(raw.get("asset_id", "")).strip()
        family = str(raw.get("family", "")).strip()
        source_url = str(raw.get("selection_source_url", "")).strip()
        placement_class = str(raw.get("placement_class", "")).strip()
        if (
            not asset_id
            or asset_id in seen_asset_ids
            or not family
            or raw.get("selection_id") != selection_id
            or source_url != SELECTED_ACTOR_GROUP_SOURCE_BY_ID[selection_id]
            or placement_class
            != SELECTED_ACTOR_GROUP_PLACEMENT_BY_ID[selection_id]
        ):
            raise NativeVariantContractError(
                f"{label} identity, source URL or placement class changed"
            )
        seen_asset_ids.add(asset_id)
        raw_lods = raw.get("lod_paths")
        if not isinstance(raw_lods, dict) or set(raw_lods) != set(LOD_LEVELS):
            raise NativeVariantContractError(
                f"{label} requires real HERO, MID and FAR USD stages"
            )
        lods = {
            level: _artifact(
                raw_lods[level],
                volume_root=volume_root,
                label=f"{label}.lod_paths.{level}",
                require_usd=True,
                require_prim=True,
            )
            for level in LOD_LEVELS
        }
        if len({record.path for record in lods.values()}) != len(LOD_LEVELS):
            raise NativeVariantContractError(
                f"{label} reuses one file for multiple LOD levels"
            )
        anchor = _vec3(raw.get("ground_anchor_m"), label=f"{label}.ground_anchor_m")
        result[selection_id] = ActorBinding(
            selection_id=selection_id,
            asset_id=asset_id,
            family=family,
            placement_class=placement_class,
            source_url=source_url,
            lods=lods,
            ground_anchor_m=anchor,
            semantic_roles=(),
        )
    return result


def _actor_library_dict(
    actors: Mapping[str, ActorBinding],
) -> dict[str, dict[str, Any]]:
    return {
        selection_id: {
            "selection_id": actor.selection_id,
            "asset_id": actor.asset_id,
            "family": actor.family,
            "placement_class": actor.placement_class,
            "selection_source_url": actor.source_url,
            "lods": {
                level: _artifact_dict(actor.lods[level])
                for level in LOD_LEVELS
            },
            "ground_anchor_m": [
                actor.ground_anchor_m.x,
                actor.ground_anchor_m.y,
                actor.ground_anchor_m.z,
            ],
            "semantic_roles": list(actor.semantic_roles),
        }
        for selection_id, actor in actors.items()
    }


def _supplemental_environment_library(
    manifest: ArtifactRef,
    *,
    volume_root: Path,
) -> dict[str, SupplementalEnvironmentBinding]:
    payload = _read_json(
        (volume_root / manifest.path).resolve(),
        label="shared supplemental-environment manifest",
    )
    group = payload.get("selected_environment_group")
    assets = group.get("assets") if isinstance(group, dict) else None
    expected = set(SELECTED_ENVIRONMENT_GROUP_IDS)
    actual = set(assets) if isinstance(assets, dict) else set()
    if (
        not isinstance(group, dict)
        or group.get("group_id") != SELECTED_ENVIRONMENT_GROUP_ID
        or group.get("selection_count") != len(SELECTED_ENVIRONMENT_GROUP_IDS)
        or group.get("selection_order")
        != list(SELECTED_ENVIRONMENT_GROUP_IDS)
        or group.get("usage_contract")
        != "all_four_assets_are_additive_and_used_in_every_variant"
        or not isinstance(assets, dict)
        or actual != expected
    ):
        raise NativeVariantContractError(
            "shared manifest lacks the exact four acquired supplemental "
            "environment assets"
        )
    result: dict[str, SupplementalEnvironmentBinding] = {}
    for selection_id in SELECTED_ENVIRONMENT_GROUP_IDS:
        raw = assets[selection_id]
        label = f"selected_environment_group.assets.{selection_id}"
        kind, family = SELECTED_ENVIRONMENT_TARGET_BY_ID[selection_id]
        if (
            not isinstance(raw, dict)
            or raw.get("selection_id") != selection_id
            or raw.get("environment_kind") != kind
            or raw.get("environment_family") != family
            or not isinstance(raw.get("asset_id"), str)
            or not raw["asset_id"].strip()
        ):
            raise NativeVariantContractError(
                f"{label} identity or target family changed"
            )
        raw_lods = raw.get("lod_paths")
        if not isinstance(raw_lods, dict) or set(raw_lods) != set(LOD_LEVELS):
            raise NativeVariantContractError(
                f"{label} requires distinct HERO, MID and FAR stages"
            )
        lods = {
            level: _artifact(
                raw_lods[level],
                volume_root=volume_root,
                label=f"{label}.lod_paths.{level}",
                require_usd=True,
                require_prim=True,
            )
            for level in LOD_LEVELS
        }
        if len({record.path for record in lods.values()}) != len(LOD_LEVELS):
            raise NativeVariantContractError(
                f"{label} reuses one LOD file"
            )
        result[selection_id] = SupplementalEnvironmentBinding(
            selection_id=selection_id,
            asset_id=str(raw["asset_id"]),
            environment_kind=kind,
            environment_family=family,
            lods=lods,
            ground_anchor_m=_vec3(
                raw.get("ground_anchor_m"),
                label=f"{label}.ground_anchor_m",
            ),
        )
    return result


def _supplemental_environment_dict(
    assets: Mapping[str, SupplementalEnvironmentBinding],
) -> dict[str, dict[str, Any]]:
    return {
        selection_id: {
            "selection_id": selected.selection_id,
            "asset_id": selected.asset_id,
            "environment_kind": selected.environment_kind,
            "environment_family": selected.environment_family,
            "lods": {
                level: _artifact_dict(selected.lods[level])
                for level in LOD_LEVELS
            },
            "ground_anchor_m": [
                selected.ground_anchor_m.x,
                selected.ground_anchor_m.y,
                selected.ground_anchor_m.z,
            ],
        }
        for selection_id, selected in assets.items()
    }


def _water_materials(
    payload: object, *, volume_root: Path
) -> dict[str, ArtifactRef]:
    if not isinstance(payload, dict) or set(payload) != set(LOD_LEVELS):
        raise NativeVariantContractError(
            "water_material_lods requires HERO, MID and FAR materials"
        )
    return {
        level: _artifact(
            payload[level],
            volume_root=volume_root,
            label=f"water_material_lods.{level}",
            require_usd=True,
            require_prim=True,
        )
        for level in LOD_LEVELS
    }


def _object_stream(
    payload: object,
    *,
    volume_root: Path,
    category: str,
    assets: Mapping[str, AssetBinding],
) -> tuple[tuple[SceneAsset, ...], dict[str, ObjectBinding]]:
    artifact = _artifact(
        payload,
        volume_root=volume_root,
        label=category,
    )
    raw_count = payload.get("count") if isinstance(payload, dict) else None
    expected_count = _integer(raw_count, label=f"{category}.count", minimum=1)
    path = (volume_root / artifact.path).resolve()
    items: list[SceneAsset] = []
    bindings: dict[str, ObjectBinding] = {}
    numeric_ids: set[int] = set()
    try:
        lines = path.open("r", encoding="utf-8")
    except OSError as error:
        raise NativeVariantContractError(f"{category} stream cannot be opened") from error
    with lines:
        for line_number, text in enumerate(lines, start=1):
            if not text.strip():
                raise NativeVariantContractError(
                    f"{category} stream contains a blank line at {line_number}"
                )
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as error:
                raise NativeVariantContractError(
                    f"{category} stream line {line_number} is invalid JSON"
                ) from error
            if not isinstance(raw, dict):
                raise NativeVariantContractError(
                    f"{category} stream line {line_number} is not an object"
                )
            stable_id = raw.get("stable_id")
            asset_key = raw.get("asset_key")
            if not isinstance(stable_id, str) or not stable_id.strip():
                raise NativeVariantContractError(
                    f"{category} line {line_number} has no stable_id"
                )
            if stable_id in bindings:
                raise NativeVariantContractError(
                    f"{category} contains duplicate stable ID {stable_id}"
                )
            if not isinstance(asset_key, str) or asset_key not in assets:
                raise NativeVariantContractError(
                    f"{category} {stable_id} references an unknown asset_key"
                )
            asset = assets[asset_key]
            if asset.category != category:
                raise NativeVariantContractError(
                    f"{category} {stable_id} references a {asset.category} asset"
                )
            numeric_id = _integer(
                raw.get("numeric_id"),
                label=f"{category}.{stable_id}.numeric_id",
                minimum=1,
            )
            if numeric_id > _MAX_INT64 or numeric_id in numeric_ids:
                raise NativeVariantContractError(
                    f"{category} has an invalid/duplicate numeric ID {numeric_id}"
                )
            numeric_ids.add(numeric_id)
            position = _vec3(
                raw.get("position"), label=f"{category}.{stable_id}.position"
            )
            try:
                scene_asset = SceneAsset(
                    stable_id=stable_id,
                    family=asset.family,
                    asset_ref=asset.lods["HERO"].path,
                    position=position,
                    heading_degrees=_number(
                        raw.get("heading_degrees"),
                        label=f"{category}.{stable_id}.heading_degrees",
                    ),
                    uniform_scale=_number(
                        raw.get("uniform_scale"),
                        label=f"{category}.{stable_id}.uniform_scale",
                    ),
                    footprint_radius_m=_number(
                        raw.get("footprint_radius_m"),
                        label=f"{category}.{stable_id}.footprint_radius_m",
                    ),
                    group_id=str(raw.get("group_id", "")),
                )
            except ValueError as error:
                raise NativeVariantContractError(str(error)) from error
            items.append(scene_asset)
            bindings[stable_id] = ObjectBinding(stable_id, numeric_id, asset_key)
    if len(items) != expected_count:
        raise NativeVariantContractError(
            f"{category} count mismatch: expected {expected_count}, found {len(items)}"
        )
    return tuple(items), bindings


def _verify_build_contract(
    *,
    build: Mapping[str, Any],
    auto: Mapping[str, Any],
    build_ref: ArtifactRef,
    base_id: str,
    terrain_payloads: Sequence[TerrainTile],
    volume_root: Path,
) -> NativeBuildBindings:
    if build.get("schema_version") != 2 or build.get("zone_id") != base_id:
        raise NativeVariantContractError(
            f"{base_id} build receipt is not the expected native schema"
        )
    if (
        build.get("source_profile") != "full"
        or build.get("fire_simulation_status")
        != "blocked_pending_editor_review"
    ):
        raise NativeVariantContractError(
            f"{base_id} must use a full native build blocked before fire"
        )
    raw_root = build.get("root_usd")
    if not isinstance(raw_root, dict):
        raise NativeVariantContractError(f"{base_id} build has no root_usd")
    root_path = raw_root.get("path")
    root_sha = raw_root.get("sha256")
    if not isinstance(root_path, str) or not isinstance(root_sha, str):
        raise NativeVariantContractError(f"{base_id} root_usd is malformed")
    receipt_payloads = build.get("payloads")
    if not isinstance(receipt_payloads, list) or len(receipt_payloads) != 400:
        raise NativeVariantContractError(
            f"{base_id} build must expose exactly 400 terrain payloads"
        )
    build_path = (volume_root / build_ref.path).resolve()
    if build_path.parent.name != "build":
        raise NativeVariantContractError(
            f"{base_id} native build receipt must live below zone_root/build"
        )
    zone_root = build_path.parent.parent

    def normalized_receipt_artifact(item: object, *, label: str) -> tuple[str, str]:
        if not isinstance(item, dict):
            raise NativeVariantContractError(f"{label} is malformed")
        raw_path = item.get("path")
        sha = item.get("sha256")
        if (
            not isinstance(raw_path, str)
            or not raw_path.strip()
            or Path(raw_path).is_absolute()
            or ".." in Path(raw_path).parts
            or not isinstance(sha, str)
            or not _SHA256.fullmatch(sha)
        ):
            raise NativeVariantContractError(f"{label} is malformed")
        resolved = (zone_root / raw_path).resolve()
        if not _is_below(volume_root, resolved):
            raise NativeVariantContractError(f"{label} escapes the persistent volume")
        return resolved.relative_to(volume_root).as_posix(), sha

    raw_asset_lock = build.get("asset_lock")
    asset_lock_path, asset_lock_sha = normalized_receipt_artifact(
        raw_asset_lock, label=f"{base_id}.asset_lock"
    )
    asset_lock = _artifact(
        {
            "path": asset_lock_path,
            "sha256": asset_lock_sha,
        },
        volume_root=volume_root,
        label=f"{base_id}.asset_lock",
    )
    asset_lock_payload = _read_json(
        (volume_root / asset_lock.path).resolve(),
        label=f"{base_id} asset lock",
    )
    raw_asset_entries = (
        raw_asset_lock.get("assets")
        if isinstance(raw_asset_lock, dict)
        else None
    )
    if not isinstance(raw_asset_entries, list) or not raw_asset_entries:
        raw_asset_entries = asset_lock_payload.get("assets")
    if not isinstance(raw_asset_entries, list) or not raw_asset_entries:
        raise NativeVariantContractError(
            f"{base_id} build has no locked asset inventory"
        )
    asset_entries = tuple(
        dict(item)
        for item in raw_asset_entries
        if isinstance(item, dict)
    )
    if len(asset_entries) != len(raw_asset_entries):
        raise NativeVariantContractError(
            f"{base_id} asset lock contains a malformed entry"
        )
    shared_asset = next(
        (
            item
            for item in asset_entries
            if item.get("id")
            == "shared-materialized-simready-environment"
        ),
        None,
    )
    if not isinstance(shared_asset, dict):
        raise NativeVariantContractError(
            f"{base_id} build is not bound to the shared SimReady asset manifest"
        )
    manifest_path = shared_asset.get("manifest")
    manifest_sha = shared_asset.get("manifest_sha256")
    if (
        not isinstance(manifest_path, str)
        or not isinstance(manifest_sha, str)
        or not _SHA256.fullmatch(manifest_sha)
    ):
        raise NativeVariantContractError(
            f"{base_id} shared asset manifest lock is malformed"
        )
    shared_manifest = _artifact(
        {
            "path": manifest_path,
            "sha256": manifest_sha,
        },
        volume_root=volume_root,
        label=f"{base_id}.shared_asset_manifest",
    )
    raw_asset_validation = shared_asset.get("validation")
    asset_content_sha = (
        raw_asset_validation.get("asset_content_sha256")
        if isinstance(raw_asset_validation, dict)
        else None
    )
    if (
        not isinstance(asset_content_sha, str)
        or not _SHA256.fullmatch(asset_content_sha)
    ):
        raise NativeVariantContractError(
            f"{base_id} shared asset content lock is absent"
        )

    raw_cameras = build.get("cameras")
    camera_path, camera_sha = normalized_receipt_artifact(
        raw_cameras, label=f"{base_id}.cameras"
    )
    camera_count = (
        raw_cameras.get("count")
        if isinstance(raw_cameras, dict)
        else None
    )
    if (
        not isinstance(camera_count, int)
        or isinstance(camera_count, bool)
        or camera_count < 6
    ):
        raise NativeVariantContractError(
            f"{base_id} build must expose at least six review cameras"
        )
    review_cameras = _artifact(
        {
            "path": camera_path,
            "sha256": camera_sha,
            "prim_path": "/ReviewCameras",
        },
        volume_root=volume_root,
        label=f"{base_id}.review_cameras",
        require_usd=True,
        require_prim=True,
    )

    receipt_set = {
        normalized_receipt_artifact(
            item, label=f"{base_id}.build.payloads[{index}]"
        )
        for index, item in enumerate(receipt_payloads)
    }
    layout_set = {
        (item.artifact.path, item.artifact.sha256) for item in terrain_payloads
    }
    if len(receipt_set) != 400 or receipt_set != layout_set:
        raise NativeVariantContractError(
            f"{base_id} layout does not reuse the exact 400 receipt terrain payloads"
        )
    for field in ("detail_payloads", "detail_mid_payloads", "detail_far_payloads"):
        details = build.get(field)
        if not isinstance(details, list) or len(details) != 400:
            raise NativeVariantContractError(
                f"{base_id} build lacks the exact 400 {field} records"
            )
    coverage = build.get("tile_coverage")
    if not isinstance(coverage, list) or len(coverage) != 400:
        raise NativeVariantContractError(
            f"{base_id} build lacks exact 400-tile coverage metadata"
        )
    expected_coverage = {
        (
            tile.tile_ref,
            tile.artifact.path,
            tile.instance_namespace,
            tile.terrain_lods,
            tile.collision_lods,
        )
        for tile in terrain_payloads
    }
    receipt_coverage: set[
        tuple[str, str, int, tuple[str, ...], tuple[str, ...]]
    ] = set()
    for index, record in enumerate(coverage):
        if not isinstance(record, dict):
            raise NativeVariantContractError(
                f"{base_id}.tile_coverage[{index}] is malformed"
            )
        terrain_path, _terrain_sha = normalized_receipt_artifact(
            {
                "path": record.get("terrain_payload"),
                "sha256": next(
                    (
                        item.get("sha256")
                        for item in receipt_payloads
                        if isinstance(item, dict)
                        and item.get("path") == record.get("terrain_payload")
                    ),
                    "",
                ),
            },
            label=f"{base_id}.tile_coverage[{index}].terrain_payload",
        )
        namespace = record.get("instance_namespace")
        if not isinstance(namespace, int) or isinstance(namespace, bool):
            raise NativeVariantContractError(
                f"{base_id}.tile_coverage[{index}] has no instance namespace"
            )
        raw_lods = record.get("terrain_lods")
        raw_collision_lods = record.get("collision_lods")
        if (
            not isinstance(raw_lods, list)
            or any(not isinstance(item, str) for item in raw_lods)
            or raw_collision_lods != ["NEAR", "FAR"]
        ):
            raise NativeVariantContractError(
                f"{base_id}.tile_coverage[{index}] has no real terrain/collision LOD contract"
            )
        receipt_coverage.add(
            (
                str(record.get("tile_ref", "")),
                terrain_path,
                namespace,
                tuple(raw_lods),
                tuple(raw_collision_lods),
            )
        )
    if receipt_coverage != expected_coverage:
        raise NativeVariantContractError(
            f"{base_id} composition source tile identities do not match the build"
        )
    layers = build.get("layers")
    if not isinstance(layers, dict):
        raise NativeVariantContractError(f"{base_id} build has no layer inventory")
    for layer in ("vegetation", "buildings", "hydrology"):
        record = layers.get(layer)
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("prim_count"), int)
            or record["prim_count"] < 1
        ):
            raise NativeVariantContractError(
                f"{base_id} validated base is empty for required layer {layer}"
            )
    roads = layers.get("roads")
    if not isinstance(roads, dict) or not isinstance(roads.get("prim_count"), int):
        raise NativeVariantContractError(
            f"{base_id} build has no road topology layer summary"
        )
    if (
        auto.get("state") != "AUTO_VALIDATED"
        or auto.get("fire_simulation_status")
        != "blocked_pending_editor_review"
        or auto.get("build_receipt_sha256") != build_ref.sha256
        or auto.get("root_usd_sha256") != root_sha
    ):
        raise NativeVariantContractError(
            f"{base_id} automatic scene validation is absent or stale"
        )
    normalized_root, _ = normalized_receipt_artifact(
        raw_root, label=f"{base_id}.root_usd"
    )
    return NativeBuildBindings(
        root_usd=ArtifactRef(path=normalized_root, sha256=root_sha),
        asset_lock=asset_lock,
        asset_lock_assets=asset_entries,
        shared_asset_manifest=shared_manifest,
        asset_content_sha256=asset_content_sha,
        review_cameras=review_cameras,
        review_camera_count=camera_count,
    )


def load_native_base_layout(
    layout_path: Path,
    *,
    volume_root: Path,
) -> NativeBaseLayout:
    """Load one hash-bound native layout without importing USD."""

    root = volume_root.expanduser().resolve()
    if not root.is_dir():
        raise NativeVariantContractError(f"persistent volume root is absent: {root}")
    path = layout_path.expanduser().resolve()
    if not _is_below(root, path) or not path.is_file():
        raise NativeVariantContractError(
            "base layout must be an existing file below the persistent volume"
        )
    layout = _read_json(path, label="native base layout")
    if layout.get("schema_version") != SCHEMA_VERSION:
        raise NativeVariantContractError("unsupported native base layout schema")
    base_id = layout.get("base_scene_id")
    if not isinstance(base_id, str) or not base_id.strip():
        raise NativeVariantContractError("base_scene_id is required")
    try:
        epsg2154_origin = _vec3(
            layout.get("epsg2154_origin"),
            label=f"{base_id}.epsg2154_origin",
        )
    except (ValueError, NativeVariantContractError) as error:
        raise NativeVariantContractError(
            f"{base_id}.epsg2154_origin is invalid: {error}"
        ) from error

    build_ref = _artifact(
        layout.get("native_build_receipt"),
        volume_root=root,
        label=f"{base_id}.native_build_receipt",
    )
    auto_ref = _artifact(
        layout.get("scene_auto_validation"),
        volume_root=root,
        label=f"{base_id}.scene_auto_validation",
    )
    raw_terrain = layout.get("terrain_payloads")
    raw_water = layout.get("water_payloads")
    if not isinstance(raw_terrain, list) or len(raw_terrain) != 400:
        raise NativeVariantContractError(
            f"{base_id} requires exactly 400 isolated terrain payloads"
        )
    if not isinstance(raw_water, list) or not raw_water:
        raise NativeVariantContractError(
            f"{base_id} requires at least one isolated water payload; "
            "the legacy build receipt alone is insufficient"
        )
    terrain = tuple(
        _terrain_tile(
            record,
            volume_root=root,
            base_id=base_id,
            index=index,
        )
        for index, record in enumerate(raw_terrain)
    )
    water_payloads = tuple(
        _artifact(
            record,
            volume_root=root,
            label=f"{base_id}.water_payloads[{index}]",
            require_usd=True,
            require_prim=True,
            roles=("water",),
        )
        for index, record in enumerate(raw_water)
    )
    ground_material = _ground_material_binding(
        layout.get("ground_material"),
        volume_root=root,
        base_id=base_id,
    )
    height_ref = _artifact(
        layout.get("height_field"),
        volume_root=root,
        label=f"{base_id}.height_field",
    )
    height_payload = _read_json(
        (root / height_ref.path).resolve(), label=f"{base_id} height field"
    )
    # The global grid is retained only as a cheap overview/provenance artifact.
    # Every placement and slope query must traverse the hash-bound tiled
    # provider below; silently falling back to this overview would flatten
    # terrain-dependent composition.
    _height_field(height_payload)
    placement_height = _placement_height_field(
        layout.get("placement_height_tiles"),
        fingerprint=layout.get("placement_height_fingerprint"),
        terrain=terrain,
        volume_root=root,
        base_id=base_id,
    )
    assets = _asset_library(layout.get("asset_library"), volume_root=root)
    water_materials = _water_materials(
        layout.get("water_material_lods"), volume_root=root
    )
    trees, tree_bindings = _object_stream(
        layout.get("trees"),
        volume_root=root,
        category="trees",
        assets=assets,
    )
    buildings, building_bindings = _object_stream(
        layout.get("buildings"),
        volume_root=root,
        category="buildings",
        assets=assets,
    )
    raw_routes = layout.get("routes")
    raw_waters = layout.get("waters")
    raw_zones = layout.get("suitability_zones")
    if (
        not isinstance(raw_routes, list)
        or not isinstance(raw_waters, list)
        or not isinstance(raw_zones, list)
    ):
        raise NativeVariantContractError(
            "routes, waters and suitability_zones must be arrays"
        )
    route_pairs = tuple(
        _route(raw, index=index) for index, raw in enumerate(raw_routes)
    )
    routes = tuple(pair[0] for pair in route_pairs)
    route_bindings = {pair[1].stable_id: pair[1] for pair in route_pairs}
    if len(route_bindings) != len(routes):
        raise NativeVariantContractError(f"{base_id} route stable IDs repeat")
    if len({binding.numeric_id for binding in route_bindings.values()}) != len(routes):
        raise NativeVariantContractError(f"{base_id} route numeric IDs repeat")
    waters = tuple(_water(raw, index=index) for index, raw in enumerate(raw_waters))
    zones = tuple(
        _suitability(raw, index=index) for index, raw in enumerate(raw_zones)
    )
    variant_constraints = _layout_constraints(
        layout.get("variant_constraints"),
        zones=zones,
        tree_families={tree.family for tree in trees},
    )
    try:
        scene = BaseScene(
            stable_id=base_id,
            bounds=_bounds(layout.get("bounds")),
            terrain=placement_height,
            ground_surface=_ground_surface(
                layout.get("ground_surface"),
                ground_material=ground_material.index,
            ),
            trees=trees,
            buildings=buildings,
            routes=routes,
            waters=waters,
            suitability_zones=zones,
        )
    except ValueError as error:
        raise NativeVariantContractError(str(error)) from error
    raw_route_topology = layout.get("route_topology")
    if not isinstance(raw_route_topology, dict):
        raise NativeVariantContractError(
            f"{base_id}.route_topology is required"
        )
    topology_tolerance = _number(
        raw_route_topology.get("tolerance_m"),
        label=f"{base_id}.route_topology.tolerance_m",
    )
    if (
        raw_route_topology.get("algorithm")
        != "segment-connectivity-components-v1"
        or not math.isclose(
            topology_tolerance,
            variant_constraints.road_connectivity_tolerance_m,
            abs_tol=1.0e-9,
        )
    ):
        raise NativeVariantContractError(
            f"{base_id}.route_topology tolerance/algorithm differs from constraints"
        )
    source_component_count, source_membership_sha = (
        _scene_variants.route_topology(
            scene.routes,
            variant_constraints.road_connectivity_tolerance_m,
        )
    )
    if (
        raw_route_topology.get("source_component_count")
        != source_component_count
        or raw_route_topology.get("source_membership_sha256")
        != source_membership_sha
        or variant_constraints.maximum_road_components
        != source_component_count
    ):
        raise NativeVariantContractError(
            f"{base_id}.route_topology is stale or its component ceiling is loose"
        )
    _validate_tile_partition(
        terrain,
        scene_bounds=scene.bounds,
        epsg2154_origin=epsg2154_origin,
        base_id=base_id,
    )
    for tile in terrain:
        _ground_material_for_terrain(
            tile,
            ground=ground_material,
            base_id=base_id,
        )
    build = _read_json(
        (root / build_ref.path).resolve(), label=f"{base_id} build receipt"
    )
    auto = _read_json(
        (root / auto_ref.path).resolve(), label=f"{base_id} auto validation"
    )
    build_bindings = _verify_build_contract(
        build=build,
        auto=auto,
        build_ref=build_ref,
        base_id=base_id,
        terrain_payloads=terrain,
        volume_root=root,
    )
    root_artifact = _artifact(
        {
            "path": build_bindings.root_usd.path,
            "sha256": build_bindings.root_usd.sha256,
            "prim_path": "/World",
        },
        volume_root=root,
        label=f"{base_id}.root_usd",
        require_usd=True,
        require_prim=True,
    )
    selected_actors = _selected_actor_library(
        build_bindings.shared_asset_manifest,
        volume_root=root,
    )
    supplemental_environment = _supplemental_environment_library(
        build_bindings.shared_asset_manifest,
        volume_root=root,
    )
    bindings = {**tree_bindings, **building_bindings}
    if len(bindings) != len(tree_bindings) + len(building_bindings):
        raise NativeVariantContractError(
            f"{base_id} object stable IDs overlap across categories"
        )
    numeric_ids = [entry.numeric_id for entry in bindings.values()]
    if len(numeric_ids) != len(set(numeric_ids)):
        raise NativeVariantContractError(
            f"{base_id} object numeric IDs overlap across categories"
        )
    return NativeBaseLayout(
        layout_path=path,
        layout_sha256=_sha256(path),
        scene=scene,
        build_receipt=build_ref,
        auto_validation=auto_ref,
        root_usd=root_artifact,
        asset_lock=build_bindings.asset_lock,
        asset_lock_assets=build_bindings.asset_lock_assets,
        shared_asset_manifest=build_bindings.shared_asset_manifest,
        asset_content_sha256=build_bindings.asset_content_sha256,
        review_cameras=build_bindings.review_cameras,
        review_camera_count=build_bindings.review_camera_count,
        epsg2154_origin=epsg2154_origin,
        terrain_payloads=terrain,
        preview_height_field=height_ref,
        placement_height_fingerprint=placement_height.content_fingerprint,
        water_payloads=water_payloads,
        ground_material=ground_material,
        assets=assets,
        selected_actors=selected_actors,
        supplemental_environment=supplemental_environment,
        object_bindings=bindings,
        water_materials=water_materials,
        route_bindings=route_bindings,
        route_component_count=source_component_count,
        route_membership_sha256=source_membership_sha,
        variant_constraints=variant_constraints,
    )


def _artifact_dict(value: ArtifactRef) -> dict[str, Any]:
    payload: dict[str, Any] = {"path": value.path, "sha256": value.sha256}
    if value.prim_path:
        payload["prim_path"] = value.prim_path
    if value.isolated_content_roles:
        payload["isolated_content_roles"] = list(value.isolated_content_roles)
    return payload


def _terrain_tile_dict(value: TerrainTile) -> dict[str, Any]:
    return {
        **_artifact_dict(value.artifact),
        "tile_ref": value.tile_ref,
        "instance_namespace": value.instance_namespace,
        "terrain_lods": list(value.terrain_lods),
        "collision_lods": list(value.collision_lods),
        "local_bounds": {
            "min_x": value.local_bounds.min_x,
            "min_y": value.local_bounds.min_y,
            "max_x": value.local_bounds.max_x,
            "max_y": value.local_bounds.max_y,
        },
        "epsg2154_bounds": {
            "min_x": value.epsg2154_bounds.min_x,
            "min_y": value.epsg2154_bounds.min_y,
            "max_x": value.epsg2154_bounds.max_x,
            "max_y": value.epsg2154_bounds.max_y,
        },
    }


def _ground_material_dict(value: GroundMaterialBinding) -> dict[str, Any]:
    return {
        **_artifact_dict(value.index),
        "topology": value.topology,
        "tile_material_payloads": [
            {
                **_artifact_dict(item.artifact),
                "tile_id": item.tile_id,
                "tile_ref": item.tile_id,
                "tile_bounds_m": [
                    item.local_bounds.min_x,
                    item.local_bounds.min_y,
                    item.local_bounds.max_x,
                    item.local_bounds.max_y,
                ],
            }
            for item in value.tile_material_payloads
        ],
    }


def _asset_library_dict(
    assets: Mapping[str, AssetBinding],
) -> dict[str, Any]:
    return {
        key: {
            "category": value.category,
            "family": value.family,
            "lineage": value.lineage,
            "grounding_offsets_m": dict(value.grounding_offsets_m),
            "lods": {
                level: _artifact_dict(value.lods[level]) for level in LOD_LEVELS
            },
        }
        for key, value in sorted(assets.items())
    }


def _water_materials_dict(
    materials: Mapping[str, ArtifactRef],
) -> dict[str, Any]:
    return {
        level: _artifact_dict(materials[level]) for level in LOD_LEVELS
    }


def _contract_dict(variant: SceneVariant) -> dict[str, Any]:
    return dataclasses.asdict(variant.contract)


def _constraints_dict(constraints: VariantConstraints) -> dict[str, Any]:
    payload = dataclasses.asdict(constraints)
    payload["tree_suitability"] = [
        {
            "family": item.family,
            "allowed_biomes": sorted(item.allowed_biomes),
            "allowed_soils": sorted(item.allowed_soils),
        }
        for item in constraints.tree_suitability
    ]
    return payload


def _constraints_equivalent(
    first: VariantConstraints, second: VariantConstraints
) -> bool:
    first_payload = _constraints_dict(first)
    second_payload = _constraints_dict(second)
    first_payload["tree_suitability"] = sorted(
        first_payload["tree_suitability"], key=lambda item: item["family"]
    )
    second_payload["tree_suitability"] = sorted(
        second_payload["tree_suitability"], key=lambda item: item["family"]
    )
    return first_payload == second_payload


def constraints_from_json(path: Path) -> VariantConstraints:
    payload = _read_json(path, label="variant constraints")
    raw_suitability = payload.pop("tree_suitability", None)
    if not isinstance(raw_suitability, list) or not raw_suitability:
        raise NativeVariantContractError(
            "variant constraints require explicit tree_suitability entries"
        )
    try:
        suitability = tuple(
            FamilySuitability(
                family=str(item["family"]),
                allowed_biomes=frozenset(str(value) for value in item["allowed_biomes"]),
                allowed_soils=frozenset(str(value) for value in item["allowed_soils"]),
            )
            for item in raw_suitability
            if isinstance(item, dict)
        )
        if len(suitability) != len(raw_suitability):
            raise ValueError("malformed tree suitability entry")
        return VariantConstraints(**payload, tree_suitability=suitability)
    except (KeyError, TypeError, ValueError) as error:
        raise NativeVariantContractError(
            f"variant constraints are invalid: {error}"
        ) from error


def _tile_for_point(
    point: Vec2,
    *,
    tiles: Sequence[TerrainTile],
    scene_bounds: Bounds,
) -> TerrainTile:
    matches = []
    for tile in tiles:
        bounds = tile.local_bounds
        inside_x = bounds.min_x <= point.x < bounds.max_x or (
            math.isclose(point.x, scene_bounds.max_x)
            and math.isclose(bounds.max_x, scene_bounds.max_x)
        )
        inside_y = bounds.min_y <= point.y < bounds.max_y or (
            math.isclose(point.y, scene_bounds.max_y)
            and math.isclose(bounds.max_y, scene_bounds.max_y)
        )
        if inside_x and inside_y:
            matches.append(tile)
    if len(matches) != 1:
        raise NativeVariantContractError(
            f"point ({point.x:.3f}, {point.y:.3f}) maps to "
            f"{len(matches)} streaming tiles"
        )
    return matches[0]


def _clip_segment(
    first: Vec3, second: Vec3, bounds: Bounds
) -> tuple[Vec3, Vec3] | None:
    """Liang-Barsky clipping with elevation interpolated along the source edge."""

    dx = second.x - first.x
    dy = second.y - first.y
    low, high = 0.0, 1.0
    for p, q in (
        (-dx, first.x - bounds.min_x),
        (dx, bounds.max_x - first.x),
        (-dy, first.y - bounds.min_y),
        (dy, bounds.max_y - first.y),
    ):
        if abs(p) <= 1.0e-12:
            if q < 0.0:
                return None
            continue
        ratio = q / p
        if p < 0.0:
            low = max(low, ratio)
        else:
            high = min(high, ratio)
        if low > high:
            return None

    def interpolate(t: float) -> Vec3:
        return Vec3(
            first.x + dx * t,
            first.y + dy * t,
            first.z + (second.z - first.z) * t,
        )

    start, end = interpolate(low), interpolate(high)
    if math.hypot(end.x - start.x, end.y - start.y) <= 1.0e-6:
        return None
    return start, end


def _fragment_numeric_id(
    source_numeric_id: int, *, tile_ref: str, segment_index: int
) -> int:
    digest = hashlib.blake2b(
        f"{source_numeric_id}\0{tile_ref}\0{segment_index}".encode("utf-8"),
        digest_size=8,
    ).digest()
    value = int.from_bytes(digest, "big") & _MAX_INT64
    return value or 1


def _route_fragments_for_tile(
    routes: Sequence[SceneRoute],
    *,
    base: NativeBaseLayout,
    tile: TerrainTile,
) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    fragment_ids: set[int] = set()
    for route in routes:
        binding = base.route_bindings[route.stable_id]
        for segment_index, (first, second) in enumerate(
            zip(route.points, route.points[1:])
        ):
            clipped = _clip_segment(first, second, tile.local_bounds)
            if clipped is None:
                continue
            numeric_id = _fragment_numeric_id(
                binding.numeric_id,
                tile_ref=tile.tile_ref,
                segment_index=segment_index,
            )
            if numeric_id in fragment_ids:
                raise NativeVariantContractError(
                    f"{route.stable_id} route fragment ID collision in {tile.tile_ref}"
                )
            fragment_ids.add(numeric_id)
            fragments.append(
                {
                    "stable_id": route.stable_id,
                    "fragment_id": (
                        f"{route.stable_id}@{tile.tile_ref}@{segment_index:04d}"
                    ),
                    "source_numeric_id": binding.numeric_id,
                    "numeric_id": numeric_id,
                    "family": route.family,
                    "surface_class": binding.surface_class,
                    "width_m": route.width_m,
                    "points": [
                        [clipped[0].x, clipped[0].y, clipped[0].z],
                        [clipped[1].x, clipped[1].y, clipped[1].z],
                    ],
                }
            )
    return fragments


def _clip_polygon_to_bounds(
    points: Sequence[Vec2], bounds: Bounds
) -> tuple[Vec2, ...]:
    result = list(points)

    def clip(
        values: list[Vec2],
        *,
        inside: Any,
        intersect: Any,
    ) -> list[Vec2]:
        if not values:
            return []
        output: list[Vec2] = []
        previous = values[-1]
        previous_inside = inside(previous)
        for current in values:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersect(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersect(previous, current))
            previous, previous_inside = current, current_inside
        return output

    def vertical(first: Vec2, second: Vec2, x: float) -> Vec2:
        denominator = second.x - first.x
        if abs(denominator) <= 1.0e-12:
            return Vec2(x, first.y)
        ratio = (x - first.x) / denominator
        return Vec2(x, first.y + (second.y - first.y) * ratio)

    def horizontal(first: Vec2, second: Vec2, y: float) -> Vec2:
        denominator = second.y - first.y
        if abs(denominator) <= 1.0e-12:
            return Vec2(first.x, y)
        ratio = (y - first.y) / denominator
        return Vec2(first.x + (second.x - first.x) * ratio, y)

    result = clip(
        result,
        inside=lambda point: point.x >= bounds.min_x,
        intersect=lambda first, second: vertical(first, second, bounds.min_x),
    )
    result = clip(
        result,
        inside=lambda point: point.x <= bounds.max_x,
        intersect=lambda first, second: vertical(first, second, bounds.max_x),
    )
    result = clip(
        result,
        inside=lambda point: point.y >= bounds.min_y,
        intersect=lambda first, second: horizontal(first, second, bounds.min_y),
    )
    result = clip(
        result,
        inside=lambda point: point.y <= bounds.max_y,
        intersect=lambda first, second: horizontal(first, second, bounds.max_y),
    )
    if len(result) < 3:
        return ()
    area = abs(
        sum(
            first.x * second.y - second.x * first.y
            for first, second in zip(result, result[1:] + result[:1])
        )
    ) * 0.5
    return tuple(result) if area > 1.0e-4 else ()


def _water_surface_elevation(water: WaterFeature, point: Vec2) -> float:
    if water.kind == "standing":
        return water.surface_profile_m[0]
    best_distance = math.inf
    best_elevation = water.surface_profile_m[0]
    for index, (first, second) in enumerate(
        zip(water.centreline, water.centreline[1:])
    ):
        dx, dy = second.x - first.x, second.y - first.y
        denominator = dx * dx + dy * dy
        if denominator <= 1.0e-12:
            continue
        ratio = min(
            1.0,
            max(
                0.0,
                ((point.x - first.x) * dx + (point.y - first.y) * dy)
                / denominator,
            ),
        )
        projected = Vec2(first.x + dx * ratio, first.y + dy * ratio)
        distance = math.hypot(point.x - projected.x, point.y - projected.y)
        if distance < best_distance:
            best_distance = distance
            best_elevation = (
                water.surface_profile_m[index] * (1.0 - ratio)
                + water.surface_profile_m[index + 1] * ratio
            )
    return best_elevation


def _water_fragments_for_tile(
    waters: Sequence[WaterFeature], *, tile: TerrainTile
) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    for water in waters:
        outline = _clip_polygon_to_bounds(water.outline, tile.local_bounds)
        if not outline:
            continue
        fragments.append(
            {
                "stable_id": water.stable_id,
                "fragment_id": f"{water.stable_id}@{tile.tile_ref}",
                "family": water.family,
                "kind": water.kind,
                "outline": [
                    [
                        point.x,
                        point.y,
                        _water_surface_elevation(water, point) + 0.02,
                    ]
                    for point in outline
                ],
            }
        )
    return fragments


class _IdentityDigest:
    """Order-independent, bounded-memory digest of stable/numeric identities."""

    __slots__ = ("_count", "_sum", "_xor")

    def __init__(self) -> None:
        self._count = 0
        self._sum = 0
        self._xor = 0

    def update(self, *, category: str, stable_id: str, numeric_id: int) -> None:
        entry = hashlib.sha256(
            b"fireviewer-fictive-variant-object-v1\0"
            + category.encode("utf-8")
            + b"\0"
            + stable_id.encode("utf-8")
            + b"\0"
            + str(numeric_id).encode("ascii")
        ).digest()
        value = int.from_bytes(entry, "big")
        self._count += 1
        self._sum = (self._sum + value) % (1 << 256)
        self._xor ^= value

    def hexdigest(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"fireviewer-fictive-variant-object-multiset-v1\0")
        digest.update(self._count.to_bytes(8, "big"))
        digest.update(self._sum.to_bytes(32, "big"))
        digest.update(self._xor.to_bytes(32, "big"))
        return digest.hexdigest()


def _identity_sha256(
    *,
    trees: Iterable[SceneAsset],
    buildings: Iterable[SceneAsset],
    bindings: Mapping[str, ObjectBinding],
) -> str:
    """Hash the exact stable/numeric object identity stream without retaining it."""

    digest = _IdentityDigest()
    for category, values in (("trees", trees), ("buildings", buildings)):
        for item in values:
            binding = bindings.get(item.stable_id)
            if binding is None:
                raise NativeVariantContractError(
                    f"object identity binding is absent: {item.stable_id}"
                )
            digest.update(
                category=category,
                stable_id=item.stable_id,
                numeric_id=binding.numeric_id,
            )
    return digest.hexdigest()


def _authored_identity_contract(
    plan_identity: object,
) -> dict[str, Any]:
    if not isinstance(plan_identity, dict):
        raise NativeVariantContractError(
            "planned identity contract is malformed"
        )
    source = plan_identity.get("source_identity_sha256")
    result = plan_identity.get("result_identity_sha256")
    if (
        plan_identity.get("version") != 1
        or plan_identity.get("numeric_ids_preserved") is not True
        or plan_identity.get("stable_ids_preserved") is not True
        or plan_identity.get(
            "source_namespace_may_differ_from_destination_tile"
        )
        is not True
        or not isinstance(source, str)
        or not _SHA256.fullmatch(source)
        or result != source
    ):
        raise NativeVariantContractError(
            "planned source/result identity contract is inconsistent"
        )
    return {
        "version": 1,
        "numeric_ids_preserved": True,
        "stable_ids_preserved": True,
        "source_namespace_may_differ_from_destination_tile": True,
        "source_identity_sha256": source,
        "authored_identity_sha256": result,
    }


def _write_tiled_variant_plan(
    *,
    variant_dir: Path,
    variant: SceneVariant,
    base: NativeBaseLayout,
) -> tuple[list[dict[str, Any]], int, int, int, str]:
    tile_dirs: dict[str, Path] = {}
    streams: dict[str, Any] = {}
    digests = {
        tile.tile_ref: hashlib.sha256() for tile in base.terrain_payloads
    }
    counts = {
        tile.tile_ref: {"trees": 0, "buildings": 0}
        for tile in base.terrain_payloads
    }
    try:
        for index, tile in enumerate(base.terrain_payloads):
            directory = variant_dir / "tiles" / f"{index:04d}"
            directory.mkdir(parents=True)
            tile_dirs[tile.tile_ref] = directory
            streams[tile.tile_ref] = (directory / "objects.jsonl").open("wb")
        object_count = 0
        for category in OBJECT_CATEGORIES:
            for item in getattr(variant, category):
                binding = base.object_bindings.get(item.stable_id)
                if binding is None:
                    raise NativeVariantContractError(
                        f"{variant.stable_id} lost object binding {item.stable_id}"
                    )
                asset = base.assets[binding.asset_key]
                if asset.category != category or asset.family != item.family:
                    raise NativeVariantContractError(
                        f"{variant.stable_id} changed asset family for {item.stable_id}"
                    )
                tile = _tile_for_point(
                    item.position.xy,
                    tiles=base.terrain_payloads,
                    scene_bounds=base.scene.bounds,
                )
                record = {
                    "category": category,
                    "stable_id": item.stable_id,
                    "numeric_id": binding.numeric_id,
                    "asset_key": binding.asset_key,
                    "family": item.family,
                    "position": [item.position.x, item.position.y, item.position.z],
                    "heading_degrees": item.heading_degrees,
                    "uniform_scale": item.uniform_scale,
                    "footprint_radius_m": item.footprint_radius_m,
                    "group_id": item.group_id,
                }
                encoded = (
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                streams[tile.tile_ref].write(encoded)
                digests[tile.tile_ref].update(encoded)
                counts[tile.tile_ref][category] += 1
                object_count += 1
    finally:
        for stream in streams.values():
            stream.close()

    coverage: list[dict[str, Any]] = []
    route_fragment_count = 0
    hydrology_fragment_count = 0
    for tile in base.terrain_payloads:
        directory = tile_dirs[tile.tile_ref]
        routes = _route_fragments_for_tile(
            variant.routes, base=base, tile=tile
        )
        hydrology = _water_fragments_for_tile(
            variant.waters, tile=tile
        )
        route_fragment_count += len(routes)
        hydrology_fragment_count += len(hydrology)
        routes_path = directory / "routes.json"
        _write_json(
            routes_path,
            {
                "schema_version": PLAN_SCHEMA_VERSION,
                "variant_id": variant.stable_id,
                "tile_ref": tile.tile_ref,
                "routes": routes,
                "hydrology": hydrology,
            },
        )
        objects_path = directory / "objects.jsonl"
        coverage.append(
            {
                **_terrain_tile_dict(tile),
                "objects": {
                    "path": objects_path.relative_to(variant_dir).as_posix(),
                    "sha256": digests[tile.tile_ref].hexdigest(),
                    "count": sum(counts[tile.tile_ref].values()),
                    "family_counts": dict(counts[tile.tile_ref]),
                },
                "routes": {
                    "path": routes_path.relative_to(variant_dir).as_posix(),
                    "sha256": _sha256(routes_path),
                    "fragment_count": len(routes),
                    "hydrology_fragment_count": len(hydrology),
                },
                "detail_lod_counts": {
                    level: {
                        "buildings": counts[tile.tile_ref]["buildings"],
                        # Routes remain as source-backed topology in routes.json.
                        # Their appearance is already carried by each tile's
                        # orthophoto ground material, so there is no duplicate
                        # road ribbon geometry in a detail payload.
                        "roads": 0,
                        "hydrology": len(hydrology),
                        "vegetation": counts[tile.tile_ref]["trees"],
                    }
                    for level in LOD_LEVELS
                },
            }
        )
        if not any(
            coverage[-1]["detail_lod_counts"]["HERO"].values()
        ) and not routes:
            raise NativeVariantContractError(
                f"{variant.stable_id} tile {tile.tile_ref} has no real "
                "tree, building, orthophoto route or water representation"
            )
    return (
        coverage,
        object_count,
        route_fragment_count,
        hydrology_fragment_count,
        _identity_sha256(
            trees=variant.trees,
            buildings=variant.buildings,
            bindings=base.object_bindings,
        ),
    )


def _ensure_new_output(output_root: Path, *, volume_root: Path) -> Path:
    root = output_root.expanduser().resolve()
    if not _is_below(volume_root, root) or root == volume_root:
        raise NativeVariantContractError(
            "campaign output must be a dedicated directory below the persistent volume"
        )
    if root.exists():
        raise NativeVariantContractError(
            f"campaign output already exists; refusing to overwrite: {root}"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.with_name(f".{root.name}.partial-{uuid.uuid4().hex}")
    staging.mkdir(parents=False, exist_ok=False)
    return staging


@dataclass(frozen=True, slots=True)
class _CompactDiversitySnapshot:
    tree_xy: array
    building_xy: array
    route_xy: array

    @classmethod
    def from_variant(cls, variant: SceneVariant) -> "_CompactDiversitySnapshot":
        tree_xy = array("d")
        building_xy = array("d")
        route_xy = array("d")
        for item in variant.trees:
            tree_xy.extend((item.position.x, item.position.y))
        for item in variant.buildings:
            building_xy.extend((item.position.x, item.position.y))
        for route in variant.routes:
            for point in _scene_variants._sample_route(route):
                route_xy.extend((point.x, point.y))
        return cls(tree_xy, building_xy, route_xy)


def _mean_xy_distance(values: Iterable[tuple[float, float]], previous: array) -> float:
    total = 0.0
    count = 0
    iterator = iter(previous)
    for (x, y), old_x, old_y in zip(values, iterator, iterator):
        total += math.hypot(x - old_x, y - old_y)
        count += 1
    if count * 2 != len(previous) or count == 0:
        raise NativeVariantContractError(
            "compact diversity snapshot count changed"
        )
    return total / count


def _compact_diversity_is_valid(
    candidate: SceneVariant,
    previous: Sequence[_CompactDiversitySnapshot],
    constraints: VariantConstraints,
) -> bool:
    for snapshot in previous:
        if (
            _mean_xy_distance(
                (
                    (item.position.x, item.position.y)
                    for item in candidate.trees
                ),
                snapshot.tree_xy,
            )
            < constraints.minimum_tree_intervariant_distance_m
            or _mean_xy_distance(
                (
                    (item.position.x, item.position.y)
                    for item in candidate.buildings
                ),
                snapshot.building_xy,
            )
            < constraints.minimum_building_intervariant_distance_m
            or _mean_xy_distance(
                (
                    (point.x, point.y)
                    for route in candidate.routes
                    for point in _scene_variants._sample_route(route)
                ),
                snapshot.route_xy,
            )
            < constraints.minimum_route_intervariant_distance_m
        ):
            return False
    return True


def _iter_base_variants(
    source: BaseScene,
    *,
    master_seed: int,
    constraints: VariantConstraints,
) -> Iterable[SceneVariant]:
    """Yield one full variant at a time, retaining only compact diversity arrays."""

    _scene_variants._validate_base_scene(source, constraints)
    previous: list[_CompactDiversitySnapshot] = []
    for variant_index in range(1, VARIANTS_PER_BASE + 1):
        seed = _scene_variants._derived_seed(
            master_seed, source.stable_id, variant_index
        )
        accepted: SceneVariant | None = None
        last_error: Exception | None = None
        for composition_attempt in range(constraints.maximum_variant_attempts):
            try:
                candidate = _scene_variants._compose_one(
                    source,
                    seed=seed,
                    variant_index=variant_index,
                    composition_attempt=composition_attempt,
                    constraints=constraints,
                )
            except _scene_variants.SceneVariantError as error:
                last_error = error
                continue
            if _compact_diversity_is_valid(candidate, previous, constraints):
                accepted = candidate
                break
        if accepted is None:
            detail = f": {last_error}" if last_error is not None else ""
            raise NativeVariantContractError(
                f"{source.stable_id} variant {variant_index} cannot satisfy "
                f"pairwise diversity after "
                f"{constraints.maximum_variant_attempts} attempts{detail}"
            )
        previous.append(_CompactDiversitySnapshot.from_variant(accepted))
        yield accepted
        accepted = None


def _actor_deployments(
    *,
    variant: SceneVariant,
    actors: Mapping[str, ActorBinding],
    simulation_sequence: int,
) -> list[dict[str, Any]]:
    """Place a balanced five-actor response group in one coherent scene."""

    if set(actors) != set(SELECTED_ACTOR_GROUP_IDS):
        raise NativeVariantContractError(
            "scene actor library differs from the exact acquired actor group"
        )
    ground_ids = [
        selection_id
        for selection_id in SELECTED_ACTOR_GROUP_IDS
        if actors[selection_id].placement_class == "ground"
    ]
    aerial_ids = [
        selection_id
        for selection_id in SELECTED_ACTOR_GROUP_IDS
        if actors[selection_id].placement_class == "aerial"
    ]
    if not ground_ids or not aerial_ids:
        raise NativeVariantContractError(
            "selected actor group must retain ground and aerial responders"
        )
    selected_ground = ground_ids
    selected_aerial = aerial_ids
    route_samples = [
        point
        for route in variant.routes
        for point in _scene_variants._sample_route(route, samples=48)
    ]
    if len(route_samples) < 12:
        raise NativeVariantContractError(
            f"{variant.stable_id} has too little road geometry for responders"
        )
    deployments: list[dict[str, Any]] = []
    for slot, selection_id in enumerate(selected_ground):
        sample_index = round(
            (slot + 1) * (len(route_samples) - 1)
            / (len(selected_ground) + 1)
        )
        point = route_samples[sample_index]
        next_point = route_samples[min(sample_index + 1, len(route_samples) - 1)]
        if next_point == point:
            next_point = route_samples[max(0, sample_index - 1)]
        heading = math.degrees(
            math.atan2(next_point.y - point.y, next_point.x - point.x)
        )
        deployments.append(
            {
                "stable_id": (
                    f"SIM-{simulation_sequence:02d}-ACTOR-{slot + 1:02d}"
                ),
                "selection_id": selection_id,
                "placement_class": "ground",
                "position": [
                    point.x,
                    point.y,
                    variant.terrain.elevation(point),
                ],
                "heading_degrees": heading,
                "uniform_scale": 1.0,
                "placement_evidence": "sampled_variant_road_centreline",
            }
        )
    centre_x = (variant.bounds.min_x + variant.bounds.max_x) * 0.5
    centre_y = (variant.bounds.min_y + variant.bounds.max_y) * 0.5
    width = variant.bounds.max_x - variant.bounds.min_x
    height = variant.bounds.max_y - variant.bounds.min_y
    for aerial_slot, selection_id in enumerate(selected_aerial):
        angle = (
            (2.0 * math.pi * aerial_slot / len(selected_aerial))
            + simulation_sequence * 0.17
        )
        x = centre_x + math.cos(angle) * width * 0.12
        y = centre_y + math.sin(angle) * height * 0.12
        point = Vec2(x, y)
        altitude = 120.0 + aerial_slot * 70.0
        deployments.append(
            {
                "stable_id": (
                    f"SIM-{simulation_sequence:02d}-ACTOR-"
                    f"{len(selected_ground) + aerial_slot + 1:02d}"
                ),
                "selection_id": selection_id,
                "placement_class": "aerial",
                "position": [
                    x,
                    y,
                    variant.terrain.elevation(point) + altitude,
                ],
                "heading_degrees": (
                    math.degrees(math.atan2(centre_y - y, centre_x - x))
                ),
                "uniform_scale": 1.0,
                "altitude_agl_m": altitude,
                "placement_evidence": "scene_centre_response_orbit",
            }
        )
    if (
        len(deployments) != ACTORS_PER_SCENE
        or {item["selection_id"] for item in deployments}
        != set(SELECTED_ACTOR_GROUP_IDS)
    ):
        raise NativeVariantContractError(
            f"{variant.stable_id} actor deployment is incomplete or duplicated"
        )
    return deployments


def _actor_usage_from_metadata(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    contract = metadata.get("actor_group_contract")
    deployments = metadata.get("actor_deployments")
    library = metadata.get("selected_actor_library")
    if (
        not isinstance(contract, dict)
        or contract.get("group_id") != SELECTED_ACTOR_GROUP_ID
        or contract.get("selection_count") != len(SELECTED_ACTOR_GROUP_IDS)
        or contract.get("selection_order") != list(SELECTED_ACTOR_GROUP_IDS)
        or contract.get("actors_per_scene") != ACTORS_PER_SCENE
        or contract.get("all_selected_assets_used_across_campaign") is not True
        or not isinstance(library, dict)
        or set(library) != set(SELECTED_ACTOR_GROUP_IDS)
        or not isinstance(deployments, list)
        or len(deployments) != ACTORS_PER_SCENE
    ):
        raise NativeVariantContractError(
            "scene actor group/deployment contract is malformed"
        )
    used: list[str] = []
    stable_ids: set[str] = set()
    for index, deployment in enumerate(deployments):
        if not isinstance(deployment, dict):
            raise NativeVariantContractError(
                f"actor_deployments[{index}] must be an object"
            )
        selection_id = deployment.get("selection_id")
        stable_id = deployment.get("stable_id")
        placement_class = deployment.get("placement_class")
        position = deployment.get("position")
        scale = deployment.get("uniform_scale")
        if (
            selection_id not in library
            or not isinstance(stable_id, str)
            or not stable_id
            or stable_id in stable_ids
            or placement_class
            != library[selection_id].get("placement_class")
            or not isinstance(position, list)
            or len(position) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in position
            )
            or isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) <= 0.0
        ):
            raise NativeVariantContractError(
                f"actor_deployments[{index}] is not a finite selected actor placement"
            )
        stable_ids.add(stable_id)
        used.append(str(selection_id))
    if len(set(used)) != ACTORS_PER_SCENE:
        raise NativeVariantContractError(
            "one scene must place five distinct selected actor assets"
        )
    return tuple(used)


def _supplemental_environment_deployments(
    *,
    variant: SceneVariant,
    assets: Mapping[str, SupplementalEnvironmentBinding],
) -> list[dict[str, Any]]:
    if set(assets) != set(SELECTED_ENVIRONMENT_GROUP_IDS):
        raise NativeVariantContractError(
            "supplemental environment differs from the acquired four-asset group"
        )
    route_samples = [
        point
        for route in variant.routes
        for point in _scene_variants._sample_route(route, samples=48)
    ]
    deployments: list[dict[str, Any]] = []
    vegetation_index = 0
    building_index = 0
    for index, selection_id in enumerate(SELECTED_ENVIRONMENT_GROUP_IDS):
        selected = assets[selection_id]
        if selected.environment_kind == "vegetation":
            vegetation_index += 1
            ratio_x = 0.22 if vegetation_index == 1 else 0.78
            ratio_y = 0.72 if vegetation_index == 1 else 0.28
            x = variant.bounds.min_x + variant.bounds.width * ratio_x
            y = variant.bounds.min_y + variant.bounds.height * ratio_y
            point = Vec2(x, y)
            evidence = "forest_quadrant_additive_to_base_vegetation"
            heading = float(45 + vegetation_index * 90)
        else:
            building_index += 1
            sample_index = round(
                building_index * (len(route_samples) - 1) / 3.0
            )
            point = route_samples[sample_index]
            next_point = route_samples[min(sample_index + 1, len(route_samples) - 1)]
            x, y = point.x, point.y
            evidence = "road_accessible_additive_building_site"
            heading = math.degrees(
                math.atan2(next_point.y - point.y, next_point.x - point.x)
            )
        deployments.append(
            {
                "stable_id": f"{variant.stable_id}-ENV-{index + 1:02d}",
                "selection_id": selection_id,
                "environment_kind": selected.environment_kind,
                "environment_family": selected.environment_family,
                "position": [x, y, variant.terrain.elevation(point)],
                "heading_degrees": heading,
                "uniform_scale": 1.0,
                "placement_evidence": evidence,
            }
        )
    return deployments


def _supplemental_environment_usage(
    metadata: Mapping[str, Any],
) -> tuple[str, ...]:
    contract = metadata.get("supplemental_environment_contract")
    library = metadata.get("supplemental_environment_library")
    deployments = metadata.get("supplemental_environment_deployments")
    if (
        not isinstance(contract, dict)
        or contract.get("group_id") != SELECTED_ENVIRONMENT_GROUP_ID
        or contract.get("selection_order")
        != list(SELECTED_ENVIRONMENT_GROUP_IDS)
        or contract.get("additive_to_existing_minima") is not True
        or contract.get("all_assets_used_in_every_scene") is not True
        or contract.get("placeholder_substitution") is not False
        or not isinstance(library, dict)
        or set(library) != set(SELECTED_ENVIRONMENT_GROUP_IDS)
        or not isinstance(deployments, list)
        or len(deployments) != len(SELECTED_ENVIRONMENT_GROUP_IDS)
    ):
        raise NativeVariantContractError(
            "supplemental environment deployment contract is malformed"
        )
    used = tuple(
        str(item.get("selection_id"))
        for item in deployments
        if isinstance(item, dict)
    )
    if used != SELECTED_ENVIRONMENT_GROUP_IDS:
        raise NativeVariantContractError(
            "one or more acquired environment assets were not placed"
        )
    return used


def _layout_identity(path: Path, *, volume_root: Path) -> tuple[str, Path]:
    resolved = path.expanduser().resolve()
    if not _is_below(volume_root, resolved) or not resolved.is_file():
        raise NativeVariantContractError(
            "base layout must be an existing file below the persistent volume"
        )
    payload = _read_json(resolved, label="native base layout")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise NativeVariantContractError("unsupported native base layout schema")
    base_id = payload.get("base_scene_id")
    if not isinstance(base_id, str) or not base_id.strip():
        raise NativeVariantContractError("base_scene_id is required")
    return base_id, resolved


def prepare_variant_campaign(
    *,
    layout_paths: Sequence[Path],
    volume_root: Path,
    output_root: Path,
    master_seed: int,
    constraints: VariantConstraints | None = None,
) -> dict[str, Any]:
    """Generate a portable 4 x 5 composition plan atomically."""

    volume = volume_root.expanduser().resolve()
    if len(layout_paths) != BASE_SCENE_COUNT:
        raise NativeVariantContractError(
            f"exactly {BASE_SCENE_COUNT} base layouts are required"
        )
    identities = tuple(
        sorted(
            (
                _layout_identity(path, volume_root=volume)
                for path in layout_paths
            ),
            key=lambda item: item[0],
        )
    )
    if len({base_id for base_id, _path in identities}) != BASE_SCENE_COUNT:
        raise NativeVariantContractError("base layout stable IDs must be unique")
    staging = _ensure_new_output(output_root, volume_root=volume)
    output = output_root.expanduser().resolve()
    try:
        variant_records: list[dict[str, Any]] = []
        constraints_by_base: dict[str, Any] = {}
        shared_asset_contract: dict[str, str] | None = None
        actor_usage_counts = {
            selection_id: 0 for selection_id in SELECTED_ACTOR_GROUP_IDS
        }
        sequence = 0
        for expected_base_id, layout_path in identities:
            base = load_native_base_layout(
                layout_path, volume_root=volume
            )
            if base.scene.stable_id != expected_base_id:
                raise NativeVariantContractError(
                    "composition source identity changed during planning"
                )
            effective_constraints = base.variant_constraints
            if constraints is not None and not _constraints_equivalent(
                constraints, effective_constraints
            ):
                raise NativeVariantContractError(
                    f"{expected_base_id} composition-source constraints differ "
                    "from the external constraints; refusing silent relaxation"
                )
            constraints_by_base[expected_base_id] = _constraints_dict(
                effective_constraints
            )
            current_asset_contract = {
                "manifest_path": base.shared_asset_manifest.path,
                "manifest_sha256": base.shared_asset_manifest.sha256,
                "content_sha256": base.asset_content_sha256,
            }
            if shared_asset_contract is None:
                shared_asset_contract = current_asset_contract
            elif current_asset_contract != shared_asset_contract:
                raise NativeVariantContractError(
                    "the four base scenes do not share one exact materialized "
                    "SimReady asset manifest/content lock"
                )
            source_identity_sha = _identity_sha256(
                trees=base.scene.trees,
                buildings=base.scene.buildings,
                bindings=base.object_bindings,
            )
            for variant in _iter_base_variants(
                base.scene,
                master_seed=master_seed,
                constraints=effective_constraints,
            ):
                sequence += 1
                validate_scene_variant(
                    base.scene,
                    variant,
                    constraints=effective_constraints,
                )
                (
                    variant_component_count,
                    variant_membership_sha,
                ) = _scene_variants.route_topology(
                    variant.routes,
                    effective_constraints.road_connectivity_tolerance_m,
                )
                if (
                    variant_component_count != base.route_component_count
                    or variant_membership_sha
                    != base.route_membership_sha256
                ):
                    raise NativeVariantContractError(
                        f"{variant.stable_id} changed route component membership"
                    )
                variant_dir = staging / f"SIM-{sequence:02d}"
                variant_dir.mkdir()
                (
                    tile_coverage,
                    object_count,
                    route_fragment_count,
                    hydrology_fragment_count,
                    result_identity_sha,
                ) = (
                    _write_tiled_variant_plan(
                        variant_dir=variant_dir,
                        variant=variant,
                        base=base,
                    )
                )
                metadata_path = variant_dir / "variant.json"
                expected_count = len(variant.trees) + len(variant.buildings)
                if object_count != expected_count:
                    raise AssertionError(
                        "object stream count changed during serialization"
                    )
                if result_identity_sha != source_identity_sha:
                    raise NativeVariantContractError(
                        f"{variant.stable_id} changed stable/numeric object identity"
                    )
                actor_deployments = _actor_deployments(
                    variant=variant,
                    actors=base.selected_actors,
                    simulation_sequence=sequence,
                )
                for deployment in actor_deployments:
                    actor_usage_counts[str(deployment["selection_id"])] += 1
                supplemental_environment_deployments = (
                    _supplemental_environment_deployments(
                        variant=variant,
                        assets=base.supplemental_environment,
                    )
                )
                metadata = {
                "schema_version": PLAN_SCHEMA_VERSION,
                "simulation_id": f"SIM-{sequence:02d}",
                "variant_id": variant.stable_id,
                "base_scene_id": variant.base_scene_id,
                "variant_index": variant.variant_index,
                "algorithm": ALGORITHM_ID,
                "composition_contract": _contract_dict(variant),
                "family_counts": family_counts(variant),
                "object_count": object_count,
                "source_route_count": len(variant.routes),
                "route_fragment_count": route_fragment_count,
                "hydrology_fragment_count": hydrology_fragment_count,
                "tile_coverage": tile_coverage,
                "epsg2154_origin": [
                    base.epsg2154_origin.x,
                    base.epsg2154_origin.y,
                    base.epsg2154_origin.z,
                ],
                "identity_contract": {
                    "version": 1,
                    "numeric_ids_preserved": True,
                    "stable_ids_preserved": True,
                    "source_namespace_may_differ_from_destination_tile": True,
                    "source_identity_sha256": source_identity_sha,
                    "result_identity_sha256": result_identity_sha,
                },
                "route_topology": {
                    "algorithm": "segment-connectivity-components-v1",
                    "tolerance_m": (
                        effective_constraints.road_connectivity_tolerance_m
                    ),
                    "source_component_count": base.route_component_count,
                    "source_membership_sha256": (
                        base.route_membership_sha256
                    ),
                    "result_component_count": variant_component_count,
                    "result_membership_sha256": variant_membership_sha,
                    "exact_membership_preserved": True,
                },
                "base_bindings": {
                    "layout_path": base.layout_path.relative_to(volume).as_posix(),
                    "layout_sha256": base.layout_sha256,
                    "native_build_receipt": _artifact_dict(base.build_receipt),
                    "scene_auto_validation": _artifact_dict(base.auto_validation),
                    "source_root_usd": _artifact_dict(base.root_usd),
                    "asset_lock": {
                        **_artifact_dict(base.asset_lock),
                        "assets": [
                            dict(item) for item in base.asset_lock_assets
                        ],
                    },
                    "shared_asset_manifest": _artifact_dict(
                        base.shared_asset_manifest
                    ),
                    "asset_content_sha256": base.asset_content_sha256,
                    "review_cameras": {
                        **_artifact_dict(base.review_cameras),
                        "count": base.review_camera_count,
                    },
                    "water_payloads": [
                        _artifact_dict(item) for item in base.water_payloads
                    ],
                    "preview_height_field": _artifact_dict(
                        base.preview_height_field
                    ),
                    "placement_height": {
                        "provider": "hash_bound_tiled_float32",
                        "tile_count": 400,
                        "cache_tile_limit": 2,
                        "content_fingerprint": (
                            base.placement_height_fingerprint
                        ),
                    },
                    "ground_material": _ground_material_dict(
                        base.ground_material
                    ),
                },
                "asset_library": _asset_library_dict(base.assets),
                "selected_actor_library": _actor_library_dict(
                    base.selected_actors
                ),
                "actor_deployments": actor_deployments,
                "actor_group_contract": {
                    "group_id": SELECTED_ACTOR_GROUP_ID,
                    "selection_count": len(SELECTED_ACTOR_GROUP_IDS),
                    "selection_order": list(SELECTED_ACTOR_GROUP_IDS),
                    "actors_per_scene": ACTORS_PER_SCENE,
                    "ground_actors_per_scene": sum(
                        actor.placement_class == "ground"
                        for actor in base.selected_actors.values()
                    ),
                    "aerial_actors_per_scene": sum(
                        actor.placement_class == "aerial"
                        for actor in base.selected_actors.values()
                    ),
                    "all_selected_assets_used_across_campaign": True,
                    "placeholder_substitution": False,
                },
                "supplemental_environment_library": (
                    _supplemental_environment_dict(
                        base.supplemental_environment
                    )
                ),
                "supplemental_environment_deployments": (
                    supplemental_environment_deployments
                ),
                "supplemental_environment_contract": {
                    "group_id": SELECTED_ENVIRONMENT_GROUP_ID,
                    "selection_order": list(
                        SELECTED_ENVIRONMENT_GROUP_IDS
                    ),
                    "asset_count": len(SELECTED_ENVIRONMENT_GROUP_IDS),
                    "additive_to_existing_minima": True,
                    "all_assets_used_in_every_scene": True,
                    "placeholder_substitution": False,
                },
                "road_visual_contract": dict(ROAD_VISUAL_CONTRACT),
                "water_material_lods": _water_materials_dict(
                    base.water_materials
                ),
                "lod_contract": {
                    "levels": list(LOD_LEVELS),
                    "streaming_tile_count": 400,
                    "object_payload_count_per_scene": 1200,
                    "same_stable_ids_and_instance_counts_at_every_level": True,
                    "primitive_or_placeholder_fallbacks": False,
                    "monolithic_object_payloads": False,
                },
                "water_contract": {
                    "source_feature_ids": [
                        water.stable_id for water in variant.waters
                    ],
                    "geometry_rearranged": False,
                    "visible_representation": "tiled_detail_lods",
                    "source_payloads_composed": False,
                    "source_payloads_provenance_only": True,
                    "tiled_fragments_are_the_only_visible_geometry": True,
                },
                "terrain_contract": {
                    "height_field_fingerprint": (
                        variant.contract.terrain_fingerprint
                    ),
                    "placement_height_provider": (
                        "hash_bound_tiled_float32"
                    ),
                    "placement_height_tile_count": 400,
                    "placement_height_cache_tile_limit": 2,
                    "placement_height_fingerprint": (
                        base.placement_height_fingerprint
                    ),
                    "preview_height_field": _artifact_dict(
                        base.preview_height_field
                    ),
                    "source_payloads_reused": True,
                    "object_free_ground_override": True,
                },
                "authoring_status": "planned_not_authored",
                "fire_simulation_status": "blocked_pending_editor_review",
                }
                _write_json(metadata_path, metadata)
                variant_records.append(
                    {
                        "simulation_id": f"SIM-{sequence:02d}",
                        "variant_id": variant.stable_id,
                        "base_scene_id": variant.base_scene_id,
                        "variant_index": variant.variant_index,
                        "metadata": {
                            "path": f"SIM-{sequence:02d}/variant.json",
                            "sha256": _sha256(metadata_path),
                        },
                    }
                )
                del variant
            del base
        if sequence != PORTFOLIO_SCENE_COUNT:
            raise AssertionError("streamed planner did not write exactly 20 variants")
        if any(count <= 0 for count in actor_usage_counts.values()):
            missing = sorted(
                selection_id
                for selection_id, count in actor_usage_counts.items()
                if count <= 0
            )
            raise NativeVariantContractError(
                "the 20-scene plan did not place every selected actor asset: "
                + ", ".join(missing)
            )
        campaign = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "state": "VARIANT_PLAN_READY",
            "algorithm": ALGORITHM_ID,
            "master_seed": master_seed,
            "base_scene_count": BASE_SCENE_COUNT,
            "variants_per_base": VARIANTS_PER_BASE,
            "simulation_count": PORTFOLIO_SCENE_COUNT,
            "base_scene_ids": [base_id for base_id, _path in identities],
            "simulation_base_bindings": {
                item["simulation_id"]: {
                    "base_scene_id": item["base_scene_id"],
                    "variant_id": item["variant_id"],
                    "variant_index": item["variant_index"],
                }
                for item in variant_records
            },
            "constraints_by_base": constraints_by_base,
            "shared_asset_contract": shared_asset_contract,
            "actor_usage_contract": {
                "group_id": SELECTED_ACTOR_GROUP_ID,
                "selection_count": len(SELECTED_ACTOR_GROUP_IDS),
                "selection_order": list(SELECTED_ACTOR_GROUP_IDS),
                "actors_per_scene": ACTORS_PER_SCENE,
                "total_actor_placements": sum(actor_usage_counts.values()),
                "usage_counts": actor_usage_counts,
                "all_selected_assets_used": True,
                "placeholder_substitution": False,
            },
            "variants": variant_records,
            "planning_memory_contract": {
                "base_scenes_live": 1,
                "full_variants_live": 1,
                "prior_variants": "compact_float64_xy_snapshots_only",
            },
            "authoring_status": "planned_not_authored",
            "fire_simulation_status": "blocked_pending_editor_review",
        }
        _write_json(staging / "campaign-plan.json", campaign)
        os.replace(staging, output)
        return _read_json(output / "campaign-plan.json", label="campaign plan")
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _resolve_plan_artifact(
    payload: object,
    *,
    root: Path,
    label: str,
) -> Path:
    if not isinstance(payload, dict):
        raise NativeVariantContractError(f"{label} must be an artifact")
    portable = _portable_path(payload.get("path"), label=label)
    expected = payload.get("sha256")
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        raise NativeVariantContractError(f"{label}.sha256 is invalid")
    path = (root / portable).resolve()
    if not _is_below(root, path) or not path.is_file():
        raise NativeVariantContractError(f"{label} is absent: {portable}")
    if _sha256(path) != expected:
        raise NativeVariantContractError(f"{label} changed after planning")
    return path


def _read_object_rows(
    object_path: Path, *, expected_count: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stable_ids: set[str] = set()
    numeric_ids: set[int] = set()
    with object_path.open("r", encoding="utf-8") as stream:
        for number, text in enumerate(stream, start=1):
            try:
                row = json.loads(text)
            except json.JSONDecodeError as error:
                raise NativeVariantContractError(
                    f"planned object line {number} is invalid JSON"
                ) from error
            if not isinstance(row, dict):
                raise NativeVariantContractError(
                    f"planned object line {number} is not an object"
                )
            stable_id = row.get("stable_id")
            numeric_id = row.get("numeric_id")
            category = row.get("category")
            if (
                not isinstance(stable_id, str)
                or stable_id in stable_ids
                or category not in OBJECT_CATEGORIES
                or not isinstance(numeric_id, int)
                or isinstance(numeric_id, bool)
                or numeric_id < 1
                or numeric_id > _MAX_INT64
                or numeric_id in numeric_ids
            ):
                raise NativeVariantContractError(
                    "planned objects changed stable/numeric identity"
                )
            stable_ids.add(stable_id)
            numeric_ids.add(numeric_id)
            rows.append(row)
    if len(rows) != expected_count:
        raise NativeVariantContractError(
            f"planned tile object count changed: expected {expected_count}, "
            f"found {len(rows)}"
        )
    return rows


def _read_route_fragments(
    routes_path: Path,
    *,
    expected_count: int,
    expected_hydrology_count: int,
    expected_tile_ref: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    routes_payload = _read_json(routes_path, label="planned routes")
    if (
        expected_tile_ref is not None
        and routes_payload.get("tile_ref") != expected_tile_ref
    ):
        raise NativeVariantContractError(
            "planned route artifact is bound to another streaming tile"
        )
    routes = routes_payload.get("routes")
    if not isinstance(routes, list) or len(routes) != expected_count:
        raise NativeVariantContractError(
            "planned route fragment count changed"
        )
    fragment_ids: set[str] = set()
    numeric_ids: set[int] = set()
    for route in routes:
        if not isinstance(route, dict):
            raise NativeVariantContractError("planned route fragment is malformed")
        fragment_id = route.get("fragment_id")
        numeric_id = route.get("numeric_id")
        if (
            not isinstance(fragment_id, str)
            or not fragment_id.strip()
            or fragment_id in fragment_ids
            or not isinstance(numeric_id, int)
            or isinstance(numeric_id, bool)
            or numeric_id < 1
            or numeric_id > _MAX_INT64
            or numeric_id in numeric_ids
        ):
            raise NativeVariantContractError(
                "planned route fragment changed stable/numeric identity"
            )
        fragment_ids.add(fragment_id)
        numeric_ids.add(numeric_id)
    hydrology = routes_payload.get("hydrology")
    if (
        not isinstance(hydrology, list)
        or len(hydrology) != expected_hydrology_count
    ):
        raise NativeVariantContractError(
            "planned hydrology fragment count changed"
        )
    water_fragment_ids: set[str] = set()
    for water in hydrology:
        if not isinstance(water, dict):
            raise NativeVariantContractError(
                "planned hydrology fragment is malformed"
            )
        fragment_id = water.get("fragment_id")
        outline = water.get("outline")
        if (
            not isinstance(fragment_id, str)
            or not fragment_id.strip()
            or fragment_id in water_fragment_ids
            or not isinstance(outline, list)
            or len(outline) < 3
        ):
            raise NativeVariantContractError(
                "planned hydrology geometry/identity changed"
            )
        water_fragment_ids.add(fragment_id)
    return routes, hydrology


def _load_variant_tiles(
    metadata: Mapping[str, Any],
    *,
    variant_dir: Path,
) -> list[dict[str, Any]]:
    coverage = metadata.get("tile_coverage")
    if not isinstance(coverage, list) or len(coverage) != 400:
        raise NativeVariantContractError(
            "planned variant must contain exactly 400 streaming tiles"
        )
    tiles: list[dict[str, Any]] = []
    tile_refs: set[str] = set()
    namespaces: set[int] = set()
    global_stable_ids: set[str] = set()
    global_numeric_ids: set[int] = set()
    global_fragment_ids: set[str] = set()
    global_water_fragment_ids: set[str] = set()
    object_total = 0
    fragment_total = 0
    hydrology_total = 0
    identity_digest = _IdentityDigest()
    raw_origin = metadata.get("epsg2154_origin")
    origin = _vec3(raw_origin, label="variant.epsg2154_origin")
    for index, raw in enumerate(coverage):
        if not isinstance(raw, dict):
            raise NativeVariantContractError(
                f"tile_coverage[{index}] is malformed"
            )
        tile_ref = raw.get("tile_ref")
        namespace = raw.get("instance_namespace")
        if (
            not isinstance(tile_ref, str)
            or not tile_ref.strip()
            or tile_ref in tile_refs
            or not isinstance(namespace, int)
            or isinstance(namespace, bool)
            or namespace < 1
            or namespace >= 1 << 20
            or namespace in namespaces
        ):
            raise NativeVariantContractError(
                "planned tile identity/namespace changed"
            )
        tile_refs.add(tile_ref)
        namespaces.add(namespace)
        objects = raw.get("objects")
        routes = raw.get("routes")
        if not isinstance(objects, dict) or not isinstance(routes, dict):
            raise NativeVariantContractError(
                f"tile {tile_ref} has no object/route artifacts"
            )
        object_count = _integer(
            objects.get("count"),
            label=f"{tile_ref}.objects.count",
            minimum=0,
        )
        fragment_count = _integer(
            routes.get("fragment_count"),
            label=f"{tile_ref}.routes.fragment_count",
            minimum=0,
        )
        hydrology_count = _integer(
            routes.get("hydrology_fragment_count"),
            label=f"{tile_ref}.routes.hydrology_fragment_count",
            minimum=0,
        )
        object_path = _resolve_plan_artifact(
            objects,
            root=variant_dir,
            label=f"{tile_ref}.objects",
        )
        routes_path = _resolve_plan_artifact(
            routes,
            root=variant_dir,
            label=f"{tile_ref}.routes",
        )
        rows = _read_object_rows(
            object_path, expected_count=object_count
        )
        fragments, hydrology = _read_route_fragments(
            routes_path,
            expected_count=fragment_count,
            expected_hydrology_count=hydrology_count,
            expected_tile_ref=tile_ref,
        )
        local_bounds = _bounds(raw.get("local_bounds"))
        epsg2154_bounds = _bounds(raw.get("epsg2154_bounds"))
        if not all(
            math.isclose(actual, expected, abs_tol=0.01)
            for actual, expected in (
                (epsg2154_bounds.min_x, origin.x + local_bounds.min_x),
                (epsg2154_bounds.min_y, origin.y + local_bounds.min_y),
                (epsg2154_bounds.max_x, origin.x + local_bounds.max_x),
                (epsg2154_bounds.max_y, origin.y + local_bounds.max_y),
            )
        ):
            raise NativeVariantContractError(
                f"tile {tile_ref} local/EPSG:2154 bounds changed after planning"
            )
        for row in rows:
            stable_id = str(row["stable_id"])
            numeric_id = int(row["numeric_id"])
            position = _vec3(
                row.get("position"),
                label=f"{tile_ref}.{stable_id}.position",
            )
            if not local_bounds.contains(position.xy):
                raise NativeVariantContractError(
                    f"object {stable_id} falls outside its streaming tile {tile_ref}"
                )
            if (
                stable_id in global_stable_ids
                or numeric_id in global_numeric_ids
            ):
                raise NativeVariantContractError(
                    "an object appears in more than one streaming tile"
                )
            global_stable_ids.add(stable_id)
            global_numeric_ids.add(numeric_id)
            identity_digest.update(
                category=str(row.get("category", "")),
                stable_id=stable_id,
                numeric_id=numeric_id,
            )
        for fragment in fragments:
            fragment_id = str(fragment["fragment_id"])
            if fragment_id in global_fragment_ids:
                raise NativeVariantContractError(
                    "a route fragment appears in more than one streaming tile"
                )
            global_fragment_ids.add(fragment_id)
        for water in hydrology:
            fragment_id = str(water["fragment_id"])
            if fragment_id in global_water_fragment_ids:
                raise NativeVariantContractError(
                    "a water fragment appears in more than one streaming tile"
                )
            global_water_fragment_ids.add(fragment_id)
        object_total += object_count
        fragment_total += fragment_count
        hydrology_total += hydrology_count
        tile = dict(raw)
        tile["_objects_path"] = object_path
        tile["_routes_path"] = routes_path
        tiles.append(tile)
    if object_total != metadata.get("object_count"):
        raise NativeVariantContractError(
            "sum of streaming tile objects differs from the variant contract"
        )
    if fragment_total != metadata.get("route_fragment_count"):
        raise NativeVariantContractError(
            "sum of route fragments differs from the variant contract"
        )
    if hydrology_total != metadata.get("hydrology_fragment_count"):
        raise NativeVariantContractError(
            "sum of water fragments differs from the variant contract"
        )
    identity_contract = metadata.get("identity_contract")
    result_identity_sha = identity_digest.hexdigest()
    if (
        not isinstance(identity_contract, dict)
        or identity_contract.get("numeric_ids_preserved") is not True
        or identity_contract.get("stable_ids_preserved") is not True
        or identity_contract.get(
            "source_namespace_may_differ_from_destination_tile"
        )
        is not True
        or identity_contract.get("source_identity_sha256")
        != result_identity_sha
        or identity_contract.get("result_identity_sha256")
        != result_identity_sha
    ):
        raise NativeVariantContractError(
            "planned stable/numeric identity signature changed"
        )
    return tiles


def _usd_name(prefix: str, value: str) -> str:
    normalized = _SAFE_ID.sub("_", value)
    if not normalized or normalized[0].isdigit():
        normalized = f"_{normalized}"
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{normalized[:48]}_{suffix}"


def _relative_asset(layer_path: Path, asset_path: Path) -> str:
    return Path(
        os.path.relpath(asset_path, start=layer_path.parent)
    ).as_posix()


def _ribbon_vertices(
    points: Sequence[Sequence[float]], width_m: float
) -> tuple[list[tuple[float, float, float]], list[int], list[int], list[tuple[float, float]]]:
    if len(points) < 2 or width_m <= 0.0:
        raise NativeVariantContractError("route ribbon requires valid geometry")
    half = width_m * 0.5
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    cumulative = 0.0
    for index, raw in enumerate(points):
        point = (float(raw[0]), float(raw[1]), float(raw[2]))
        before = points[max(0, index - 1)]
        after = points[min(len(points) - 1, index + 1)]
        dx = float(after[0]) - float(before[0])
        dy = float(after[1]) - float(before[1])
        length = math.hypot(dx, dy)
        if length <= 1.0e-9:
            raise NativeVariantContractError("route ribbon has repeated points")
        nx, ny = -dy / length, dx / length
        if index:
            previous = points[index - 1]
            cumulative += math.hypot(
                point[0] - float(previous[0]), point[1] - float(previous[1])
            )
        z = point[2] + 0.03
        vertices.extend(
            [
                (point[0] + nx * half, point[1] + ny * half, z),
                (point[0] - nx * half, point[1] - ny * half, z),
            ]
        )
        u = cumulative / max(width_m, 0.1)
        uvs.extend([(u, 0.0), (u, 1.0)])
    counts = [4] * (len(points) - 1)
    indices: list[int] = []
    for index in range(len(points) - 1):
        left = index * 2
        indices.extend([left, left + 1, left + 3, left + 2])
    return vertices, counts, indices, uvs


def _mesh_vertex_normals(
    vertices: Sequence[tuple[float, float, float]],
    face_counts: Sequence[int],
    face_indices: Sequence[int],
) -> list[tuple[float, float, float]]:
    if sum(face_counts) != len(face_indices):
        raise NativeVariantContractError(
            "mesh face counts and indices diverge"
        )
    accumulated = [[0.0, 0.0, 0.0] for _ in vertices]
    offset = 0
    for count in face_counts:
        if count < 3:
            raise NativeVariantContractError("mesh face has fewer than 3 vertices")
        face = list(face_indices[offset : offset + count])
        offset += count
        if any(index < 0 or index >= len(vertices) for index in face):
            raise NativeVariantContractError("mesh face index is out of range")
        anchor = vertices[face[0]]
        for triangle_index in range(1, count - 1):
            second = vertices[face[triangle_index]]
            third = vertices[face[triangle_index + 1]]
            ax = second[0] - anchor[0]
            ay = second[1] - anchor[1]
            az = second[2] - anchor[2]
            bx = third[0] - anchor[0]
            by = third[1] - anchor[1]
            bz = third[2] - anchor[2]
            normal = (
                ay * bz - az * by,
                az * bx - ax * bz,
                ax * by - ay * bx,
            )
            length = math.sqrt(sum(value * value for value in normal))
            if length <= 1.0e-12:
                raise NativeVariantContractError(
                    "mesh contains a degenerate triangle"
                )
            for vertex_index in (
                face[0],
                face[triangle_index],
                face[triangle_index + 1],
            ):
                accumulated[vertex_index][0] += normal[0]
                accumulated[vertex_index][1] += normal[1]
                accumulated[vertex_index][2] += normal[2]
    result: list[tuple[float, float, float]] = []
    for value in accumulated:
        length = math.sqrt(sum(component * component for component in value))
        if length <= 1.0e-12:
            raise NativeVariantContractError(
                "mesh vertex has no non-degenerate adjacent face"
            )
        result.append(
            (
                value[0] / length,
                value[1] / length,
                value[2] / length,
            )
        )
    return result


def _triangulated_surface(
    raw_points: Sequence[Sequence[float]],
    *,
    uv_scale_m: float = 4.0,
) -> tuple[
    list[tuple[float, float, float]],
    list[int],
    list[int],
    list[tuple[float, float]],
    list[tuple[float, float, float]],
]:
    """Triangulate one simple concave XY polygon deterministically.

    Ear clipping is deliberately local and dependency-free so the exact same
    mesh is authored in every Kit image.  World-metric UVs remain continuous
    across streaming tile boundaries.
    """

    if not math.isfinite(uv_scale_m) or uv_scale_m <= 0.0:
        raise NativeVariantContractError("surface UV scale must be positive")
    points: list[tuple[float, float, float]] = []
    for index, raw in enumerate(raw_points):
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != 3
        ):
            raise NativeVariantContractError(
                f"surface point {index} is malformed"
            )
        point = (float(raw[0]), float(raw[1]), float(raw[2]))
        if any(not math.isfinite(value) for value in point):
            raise NativeVariantContractError(
                f"surface point {index} is not finite"
            )
        if points and math.hypot(
            point[0] - points[-1][0],
            point[1] - points[-1][1],
        ) <= 1.0e-9:
            continue
        points.append(point)
    if len(points) > 1 and math.hypot(
        points[0][0] - points[-1][0],
        points[0][1] - points[-1][1],
    ) <= 1.0e-9:
        points.pop()
    if len(points) < 3:
        raise NativeVariantContractError(
            "surface polygon has fewer than three distinct points"
        )
    signed_area = 0.5 * sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(points, points[1:] + points[:1])
    )
    if abs(signed_area) <= 1.0e-9:
        raise NativeVariantContractError("surface polygon has zero XY area")
    if signed_area < 0.0:
        points.reverse()

    def cross(
        first: tuple[float, float, float],
        second: tuple[float, float, float],
        third: tuple[float, float, float],
    ) -> float:
        return (
            (second[0] - first[0]) * (third[1] - first[1])
            - (second[1] - first[1]) * (third[0] - first[0])
        )

    def inside_triangle(
        point: tuple[float, float, float],
        first: tuple[float, float, float],
        second: tuple[float, float, float],
        third: tuple[float, float, float],
    ) -> bool:
        return (
            cross(first, second, point) >= -1.0e-10
            and cross(second, third, point) >= -1.0e-10
            and cross(third, first, point) >= -1.0e-10
        )

    remaining = list(range(len(points)))
    indices: list[int] = []
    while len(remaining) > 3:
        ear_found = False
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            if cross(
                points[previous], points[current], points[following]
            ) <= 1.0e-10:
                continue
            if any(
                candidate not in {previous, current, following}
                and inside_triangle(
                    points[candidate],
                    points[previous],
                    points[current],
                    points[following],
                )
                for candidate in remaining
            ):
                continue
            indices.extend((previous, current, following))
            del remaining[position]
            ear_found = True
            break
        if not ear_found:
            raise NativeVariantContractError(
                "surface polygon is self-intersecting or cannot be triangulated"
            )
    indices.extend(remaining)
    face_counts = [3] * (len(points) - 2)
    if len(indices) != 3 * (len(points) - 2):
        raise AssertionError("ear clipping did not produce n-2 triangles")
    uvs = [
        (point[0] / uv_scale_m, point[1] / uv_scale_m)
        for point in points
    ]
    normals = _mesh_vertex_normals(points, face_counts, indices)
    return points, face_counts, indices, uvs, normals


class _PXRVariantAuthor:
    """Small native backend; construction is the only pxr import boundary."""

    def __init__(self) -> None:
        try:
            from pxr import Gf, Sdf, Semantics, Usd, UsdGeom, UsdShade, Vt
        except (ImportError, ModuleNotFoundError) as error:
            raise NativeVariantContractError(
                "native variant authoring requires Kit/Isaac pxr with Semantics"
            ) from error
        self.Gf = Gf
        self.Sdf = Sdf
        self.Semantics = Semantics
        self.Usd = Usd
        self.UsdGeom = UsdGeom
        self.UsdShade = UsdShade
        self.Vt = Vt
        self._reference_cache: dict[tuple[str, str, str], Path] = {}

    def _semantic(self, prim: Any, semantic: str) -> None:
        api = self.Semantics.SemanticsAPI.Apply(prim, "Semantics")
        api.CreateSemanticTypeAttr().Set("class")
        api.CreateSemanticDataAttr().Set(semantic)
        prim.SetCustomDataByKey("fireviewer:semantic_class", semantic)

    def _validate_reference_prim(
        self, *, volume_root: Path, artifact: Mapping[str, Any], label: str
    ) -> Path:
        if isinstance(artifact, dict):
            cache_key = (
                str(artifact.get("path", "")),
                str(artifact.get("sha256", "")),
                str(artifact.get("prim_path", "")),
            )
            cached = self._reference_cache.get(cache_key)
            if cached is not None:
                return cached
        ref = _artifact(
            artifact,
            volume_root=volume_root,
            label=label,
            require_usd=True,
            require_prim=True,
        )
        path = (volume_root / ref.path).resolve()
        stage = self.Usd.Stage.Open(str(path), load=self.Usd.Stage.LoadNone)
        if stage is None or not stage.GetPrimAtPath(ref.prim_path).IsValid():
            raise NativeVariantContractError(
                f"{label} does not contain prim {ref.prim_path}"
            )
        self._reference_cache[(ref.path, ref.sha256, ref.prim_path)] = path
        return path

    def _author_objects(
        self,
        *,
        output_path: Path,
        level: str,
        tile_ref: str,
        instance_namespace: int,
        local_bounds: Mapping[str, Any],
        epsg2154_bounds: Mapping[str, Any],
        layer_counts: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
        routes: Sequence[Mapping[str, Any]],
        hydrology: Sequence[Mapping[str, Any]],
        assets: Mapping[str, Any],
        water_materials: Mapping[str, Any],
        volume_root: Path,
    ) -> dict[str, int]:
        stage = self.Usd.Stage.CreateNew(str(output_path))
        if stage is None:
            raise NativeVariantContractError(f"cannot create USD layer {output_path}")
        self.UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        self.UsdGeom.SetStageUpAxis(stage, self.UsdGeom.Tokens.z)
        root = self.UsdGeom.Xform.Define(stage, "/Detail")
        stage.SetDefaultPrim(root.GetPrim())
        root.GetPrim().SetCustomDataByKey("fireviewer:tile_ref", tile_ref)
        root.GetPrim().SetCustomDataByKey(
            "fireviewer:instance_namespace", int(instance_namespace)
        )
        root.GetPrim().SetCustomDataByKey("fireviewer:detail_level", level)
        root.GetPrim().SetCustomDataByKey(
            "fireviewer:role",
            f"camera_streamed_photoreal_detail_{level.lower()}",
        )
        root.GetPrim().SetCustomDataByKey(
            "fireviewer:local_bounds",
            ",".join(
                str(local_bounds[key])
                for key in ("min_x", "min_y", "max_x", "max_y")
            ),
        )
        root.GetPrim().SetCustomDataByKey(
            "fireviewer:epsg2154_bounds",
            ",".join(
                str(epsg2154_bounds[key])
                for key in ("min_x", "min_y", "max_x", "max_y")
            ),
        )
        normalized_counts = {
            key: int(layer_counts[key])
            for key in ("buildings", "roads", "hydrology", "vegetation")
        }
        root.GetPrim().SetCustomDataByKey(
            "fireviewer:layer_counts",
            json.dumps(normalized_counts, sort_keys=True, separators=(",", ":")),
        )
        for name in (
            "Materials",
            "Buildings",
            "RouteTopology",
            "Hydrology",
            "Vegetation",
            "Semantics",
        ):
            scope = self.UsdGeom.Scope.Define(stage, f"/Detail/{name}")
            scope.GetPrim().SetCustomDataByKey(
                "fireviewer:layer", name.lower()
            )
        groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            category = row.get("category")
            asset_key = row.get("asset_key")
            if category not in OBJECT_CATEGORIES or not isinstance(asset_key, str):
                raise NativeVariantContractError("planned object category/asset is invalid")
            groups[(category, asset_key)].append(row)
        actual_counts = {
            "buildings": sum(
                1 for row in rows if row.get("category") == "buildings"
            ),
            "roads": 0,
            "hydrology": len(hydrology),
            "vegetation": sum(
                1 for row in rows if row.get("category") == "trees"
            ),
        }
        if actual_counts != normalized_counts:
            raise NativeVariantContractError(
                f"{tile_ref}.{level} layer counts changed before USD authoring"
            )
        network_metrics = {
            "route_vertices": 0,
            "route_faces": 0,
            "hydrology_vertices": 0,
            "hydrology_faces": 0,
        }
        root.GetPrim().SetCustomDataByKey(
            "fireviewer:network_lod_policy",
            NETWORK_GEOMETRY_POLICY,
        )
        root.GetPrim().SetCustomDataByKey(
            "fireviewer:road_visualization",
            json.dumps(
                {
                    "visible_representation": (
                        "orthophoto_derived_terrain_material"
                    ),
                    "geometry_authoring": "disabled",
                    "route_fragment_count": len(routes),
                    "asset_dependencies": [],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        for (category, asset_key), members in sorted(groups.items()):
            asset = assets.get(asset_key)
            if not isinstance(asset, dict) or asset.get("category") != category:
                raise NativeVariantContractError(
                    f"planned asset binding is absent: {asset_key}"
                )
            lods = asset.get("lods")
            offsets = asset.get("grounding_offsets_m")
            if not isinstance(lods, dict) or not isinstance(offsets, dict):
                raise NativeVariantContractError(f"asset LOD contract is absent: {asset_key}")
            asset_path = self._validate_reference_prim(
                volume_root=volume_root,
                artifact=lods.get(level),
                label=f"{asset_key}.{level}",
            )
            asset_prim = lods[level]["prim_path"]
            collection = "Vegetation" if category == "trees" else "Buildings"
            instancer_path = (
                f"/Detail/{collection}/{_usd_name('Asset', asset_key)}"
            )
            instancer = self.UsdGeom.PointInstancer.Define(stage, instancer_path)
            prototype = self.UsdGeom.Xform.Define(
                stage, f"{instancer_path}/Prototypes/Asset"
            )
            prototype.GetPrim().GetReferences().AddReference(
                _relative_asset(output_path, asset_path), asset_prim
            )
            prototype.GetPrim().SetInstanceable(True)
            asset_family = str(asset.get("family", "")).strip()
            lineage = str(asset.get("lineage", "")).strip()
            if not asset_family or not lineage:
                raise NativeVariantContractError(
                    f"asset family/LOD lineage is absent: {asset_key}"
                )
            prototype.GetPrim().SetCustomDataByKey(
                "fireviewer:source_asset", asset_path.name
            )
            prototype.GetPrim().SetCustomDataByKey(
                "fireviewer:asset_family",
                (
                    f"vegetation.trees.{asset_family}"
                    if category == "trees"
                    else f"buildings.{asset_family}"
                ),
            )
            prototype.GetPrim().SetCustomDataByKey(
                "fireviewer:lod_role", "source_identity_lod_chain"
            )
            prototype.GetPrim().SetCustomDataByKey(
                "fireviewer:lod_lineage", lineage
            )
            instancer.GetPrototypesRel().SetTargets([prototype.GetPath()])
            positions = []
            orientations = []
            scales = []
            numeric_ids = []
            stable_ids = []
            footprint_radii = []
            group_ids = []
            source_namespaces = []
            grounding = float(offsets[level])
            for member in members:
                position = member.get("position")
                if not isinstance(position, list) or len(position) != 3:
                    raise NativeVariantContractError("planned object position is invalid")
                positions.append(
                    self.Gf.Vec3f(
                        float(position[0]),
                        float(position[1]),
                        float(position[2]) + grounding,
                    )
                )
                half = math.radians(float(member["heading_degrees"])) * 0.5
                orientations.append(
                    self.Gf.Quath(
                        math.cos(half),
                        self.Gf.Vec3h(0.0, 0.0, math.sin(half)),
                    )
                )
                scale = float(member["uniform_scale"])
                scales.append(self.Gf.Vec3f(scale, scale, scale))
                numeric_id = int(member["numeric_id"])
                stable_id = str(member["stable_id"])
                numeric_ids.append(numeric_id)
                stable_ids.append(stable_id)
                footprint = float(member["footprint_radius_m"])
                if not math.isfinite(footprint) or footprint <= 0.0:
                    raise NativeVariantContractError(
                        f"planned object {stable_id} has an invalid footprint"
                    )
                footprint_radii.append(footprint)
                source_group = str(member.get("group_id", "")).strip()
                group_ids.append(
                    source_group or f"ungrouped:{stable_id}"
                )
                source_namespaces.append(numeric_id >> 43)
            instancer.GetPositionsAttr().Set(self.Vt.Vec3fArray(positions))
            instancer.GetOrientationsAttr().Set(self.Vt.QuathArray(orientations))
            instancer.GetScalesAttr().Set(self.Vt.Vec3fArray(scales))
            instancer.GetProtoIndicesAttr().Set(self.Vt.IntArray([0] * len(members)))
            instancer.GetIdsAttr().Set(self.Vt.Int64Array(numeric_ids))
            stable_primvar = self.UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "fireviewer_stable_id",
                self.Sdf.ValueTypeNames.StringArray,
                self.UsdGeom.Tokens.vertex,
            )
            stable_primvar.Set(self.Vt.StringArray(stable_ids))
            footprint_primvar = self.UsdGeom.PrimvarsAPI(
                instancer
            ).CreatePrimvar(
                "fireviewer_footprint_radius_m",
                self.Sdf.ValueTypeNames.FloatArray,
                self.UsdGeom.Tokens.vertex,
            )
            footprint_primvar.Set(self.Vt.FloatArray(footprint_radii))
            group_primvar = self.UsdGeom.PrimvarsAPI(
                instancer
            ).CreatePrimvar(
                "fireviewer_group_id",
                self.Sdf.ValueTypeNames.StringArray,
                self.UsdGeom.Tokens.vertex,
            )
            group_primvar.Set(self.Vt.StringArray(group_ids))
            source_namespace_primvar = self.UsdGeom.PrimvarsAPI(
                instancer
            ).CreatePrimvar(
                "fireviewer_source_instance_namespace",
                self.Sdf.ValueTypeNames.Int64Array,
                self.UsdGeom.Tokens.vertex,
            )
            source_namespace_primvar.Set(
                self.Vt.Int64Array(source_namespaces)
            )
            instancer.GetPrim().SetCustomDataByKey(
                "fireviewer:instance_identity_contract",
                (
                    "ids+stable_id+footprint_radius_m+group_id"
                    "+source_instance_namespace"
                ),
            )
            self._semantic(
                instancer.GetPrim(),
                (
                    "vegetation_trees_fictive_variant"
                    if category == "trees"
                    else "building_fictive_variant"
                ),
            )
        if hydrology:
            water_path = self._validate_reference_prim(
                volume_root=volume_root,
                artifact=water_materials.get(level),
                label=f"water material {level}",
            )
            water_material = self.UsdShade.Material.Define(
                stage, "/Detail/Materials/Water"
            )
            water_material.GetPrim().GetReferences().AddReference(
                _relative_asset(output_path, water_path),
                water_materials[level]["prim_path"],
            )
            for feature in hydrology:
                outline = feature.get("outline")
                if not isinstance(outline, list) or len(outline) < 3:
                    raise NativeVariantContractError(
                        "hydrology fragment outline is invalid"
                    )
                mesh = self.UsdGeom.Mesh.Define(
                    stage,
                    f"/Detail/Hydrology/"
                    f"{_usd_name('Water', str(feature['fragment_id']))}",
                )
                (
                    vertices,
                    counts,
                    indices,
                    uvs,
                    normals,
                ) = _triangulated_surface(outline)
                mesh.CreatePointsAttr().Set(
                    self.Vt.Vec3fArray(
                        [self.Gf.Vec3f(*value) for value in vertices]
                    )
                )
                mesh.CreateFaceVertexCountsAttr().Set(
                    self.Vt.IntArray(counts)
                )
                mesh.CreateFaceVertexIndicesAttr().Set(
                    self.Vt.IntArray(indices)
                )
                mesh.CreateSubdivisionSchemeAttr().Set(self.UsdGeom.Tokens.none)
                mesh.CreateNormalsAttr().Set(
                    self.Vt.Vec3fArray(
                        [self.Gf.Vec3f(*value) for value in normals]
                    )
                )
                mesh.SetNormalsInterpolation(self.UsdGeom.Tokens.vertex)
                st = self.UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                    "st",
                    self.Sdf.ValueTypeNames.TexCoord2fArray,
                    self.UsdGeom.Tokens.vertex,
                )
                st.Set(
                    self.Vt.Vec2fArray(
                        [self.Gf.Vec2f(*value) for value in uvs]
                    )
                )
                mesh.GetPrim().SetCustomDataByKey(
                    "fireviewer:stable_id", str(feature["stable_id"])
                )
                mesh.GetPrim().SetCustomDataByKey(
                    "fireviewer:fragment_id", str(feature["fragment_id"])
                )
                self.UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                    water_material
                )
                self._semantic(mesh.GetPrim(), "water")
                network_metrics["hydrology_vertices"] += len(vertices)
                network_metrics["hydrology_faces"] += len(counts)
        total_network_vertices = network_metrics["hydrology_vertices"]
        if total_network_vertices > 262_144:
            raise NativeVariantContractError(
                f"{tile_ref}.{level} exceeds the measured network mesh "
                "budget of 262144 vertices"
            )
        root.GetPrim().SetCustomDataByKey(
            "fireviewer:network_geometry_metrics",
            json.dumps(
                network_metrics, sort_keys=True, separators=(",", ":")
            ),
        )
        stage.GetRootLayer().Save()
        reopened = self.Usd.Stage.Open(
            str(output_path), load=self.Usd.Stage.LoadNone
        )
        if reopened is None or not reopened.GetPrimAtPath("/Detail").IsValid():
            raise NativeVariantContractError(
                f"{tile_ref}.{level} detail layer cannot be reopened"
            )
        reopened_hydrology = 0
        for prim in reopened.Traverse():
            path = str(prim.GetPath())
            if not prim.IsA(self.UsdGeom.Mesh):
                continue
            if path.startswith("/Detail/Routes/"):
                raise NativeVariantContractError(
                    f"{tile_ref}.{level} contains forbidden rendered road geometry"
                )
            if path.startswith("/Detail/Hydrology/"):
                expected_face_size = 3
                reopened_hydrology += 1
            else:
                continue
            mesh = self.UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            normals = mesh.GetNormalsAttr().Get()
            st = self.UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            st_values = st.Get() if st else None
            if (
                points is None
                or counts is None
                or indices is None
                or normals is None
                or st_values is None
                or any(int(count) != expected_face_size for count in counts)
                or sum(int(count) for count in counts) != len(indices)
                or len(normals) != len(points)
                or len(st_values) != len(points)
            ):
                raise NativeVariantContractError(
                    f"{tile_ref}.{level} network mesh failed native reopen validation"
                )
        if reopened_hydrology != len(hydrology):
            raise NativeVariantContractError(
                f"{tile_ref}.{level} network prim count changed after reopen"
            )
        return network_metrics

    def _author_selected_actors(
        self,
        *,
        stage: Any,
        root_path: Path,
        metadata: Mapping[str, Any],
        volume_root: Path,
    ) -> list[str]:
        used = _actor_usage_from_metadata(metadata)
        library = metadata["selected_actor_library"]
        deployments = metadata["actor_deployments"]
        actors_root = self.UsdGeom.Xform.Define(stage, "/World/Actors")
        actors_root.GetPrim().SetCustomDataByKey(
            "fireviewer:selected_actor_group_id", SELECTED_ACTOR_GROUP_ID
        )
        actors_root.GetPrim().SetCustomDataByKey(
            "fireviewer:selection_count", len(SELECTED_ACTOR_GROUP_IDS)
        )
        actors_root.GetPrim().SetCustomDataByKey(
            "fireviewer:placed_actor_count", len(deployments)
        )
        for index, deployment in enumerate(deployments, start=1):
            selection_id = str(deployment["selection_id"])
            selected = library[selection_id]
            prim_path = f"/World/Actors/Actor_{index:02d}"
            actor = self.UsdGeom.Xform.Define(stage, prim_path)
            prim = actor.GetPrim()
            position = deployment["position"]
            scale = float(deployment["uniform_scale"])
            anchor = selected["ground_anchor_m"]
            self.UsdGeom.Xformable(prim).AddTranslateOp().Set(
                self.Gf.Vec3d(
                    float(position[0]) - float(anchor[0]) * scale,
                    float(position[1]) - float(anchor[1]) * scale,
                    float(position[2]) - float(anchor[2]) * scale,
                )
            )
            self.UsdGeom.Xformable(prim).AddRotateZOp().Set(
                float(deployment["heading_degrees"])
            )
            self.UsdGeom.Xformable(prim).AddScaleOp().Set(
                self.Gf.Vec3f(scale, scale, scale)
            )
            variants = prim.GetVariantSets().AddVariantSet("detail")
            lods = selected["lods"]
            for level in LOD_LEVELS:
                asset_path = self._validate_reference_prim(
                    volume_root=volume_root,
                    artifact=lods[level],
                    label=f"selected actor {selection_id}.{level}",
                )
                variants.AddVariant(level)
                variants.SetVariantSelection(level)
                with variants.GetVariantEditContext():
                    prim.GetPayloads().AddPayload(
                        _relative_asset(root_path, asset_path),
                        lods[level]["prim_path"],
                    )
            variants.SetVariantSelection("HERO")
            prim.SetCustomDataByKey(
                "fireviewer:stable_id", str(deployment["stable_id"])
            )
            prim.SetCustomDataByKey(
                "fireviewer:selected_actor_id", selection_id
            )
            prim.SetCustomDataByKey(
                "fireviewer:asset_id", str(selected["asset_id"])
            )
            prim.SetCustomDataByKey(
                "fireviewer:placement_class",
                str(deployment["placement_class"]),
            )
            prim.SetCustomDataByKey(
                "fireviewer:placement_evidence",
                str(deployment["placement_evidence"]),
            )
            prim.SetCustomDataByKey(
                "fireviewer:semantic_roles",
                ",".join(str(role) for role in selected["semantic_roles"]),
            )
            prim.SetCustomDataByKey(
                "fireviewer:lod_contract",
                "same_selected_asset_hero_mid_far_no_placeholder",
            )
            self._semantic(prim, "wildfire_response_actor")
        return list(used)

    def _author_supplemental_environment(
        self,
        *,
        stage: Any,
        root_path: Path,
        metadata: Mapping[str, Any],
        volume_root: Path,
    ) -> list[str]:
        used = _supplemental_environment_usage(metadata)
        library = metadata["supplemental_environment_library"]
        deployments = metadata["supplemental_environment_deployments"]
        root = self.UsdGeom.Xform.Define(
            stage, "/World/SupplementalEnvironment"
        )
        root.GetPrim().SetCustomDataByKey(
            "fireviewer:selected_environment_group_id",
            SELECTED_ENVIRONMENT_GROUP_ID,
        )
        for index, deployment in enumerate(deployments, start=1):
            selection_id = str(deployment["selection_id"])
            selected = library[selection_id]
            prim_path = f"/World/SupplementalEnvironment/Asset_{index:02d}"
            asset = self.UsdGeom.Xform.Define(stage, prim_path)
            prim = asset.GetPrim()
            scale = float(deployment["uniform_scale"])
            position = deployment["position"]
            anchor = selected["ground_anchor_m"]
            self.UsdGeom.Xformable(prim).AddTranslateOp().Set(
                self.Gf.Vec3d(
                    float(position[0]) - float(anchor[0]) * scale,
                    float(position[1]) - float(anchor[1]) * scale,
                    float(position[2]) - float(anchor[2]) * scale,
                )
            )
            self.UsdGeom.Xformable(prim).AddRotateZOp().Set(
                float(deployment["heading_degrees"])
            )
            self.UsdGeom.Xformable(prim).AddScaleOp().Set(
                self.Gf.Vec3f(scale, scale, scale)
            )
            variants = prim.GetVariantSets().AddVariantSet("detail")
            for level in LOD_LEVELS:
                lod = selected["lods"][level]
                source = self._validate_reference_prim(
                    volume_root=volume_root,
                    artifact=lod,
                    label=f"supplemental environment {selection_id}.{level}",
                )
                variants.AddVariant(level)
                variants.SetVariantSelection(level)
                with variants.GetVariantEditContext():
                    prim.GetPayloads().AddPayload(
                        _relative_asset(root_path, source),
                        lod["prim_path"],
                    )
            variants.SetVariantSelection("HERO")
            prim.SetCustomDataByKey(
                "fireviewer:stable_id", str(deployment["stable_id"])
            )
            prim.SetCustomDataByKey(
                "fireviewer:selected_environment_id", selection_id
            )
            prim.SetCustomDataByKey(
                "fireviewer:additive_to_existing_minima", True
            )
            prim.SetCustomDataByKey(
                "fireviewer:lod_contract",
                "same_selected_asset_hero_mid_far_no_placeholder",
            )
            self._semantic(
                prim,
                (
                    "vegetation_supplemental"
                    if selected["environment_kind"] == "vegetation"
                    else "building_supplemental"
                ),
            )
        return list(used)

    def author_variant(
        self,
        *,
        variant_dir: Path,
        metadata: Mapping[str, Any],
        tile_records: Sequence[Mapping[str, Any]],
        volume_root: Path,
    ) -> dict[str, Any]:
        variant_dir.mkdir(parents=True, exist_ok=False)
        if len(tile_records) != 400:
            raise NativeVariantContractError(
                "native authoring requires exactly 400 streaming tiles"
            )
        detail_layers: dict[str, list[Path]] = {
            level: [] for level in LOD_LEVELS
        }
        network_metrics_by_lod: dict[str, dict[str, int]] = {
            level: {
                "route_vertices": 0,
                "route_faces": 0,
                "hydrology_vertices": 0,
                "hydrology_faces": 0,
            }
            for level in LOD_LEVELS
        }
        authored_coverage: list[dict[str, Any]] = []
        for index, tile_record in enumerate(tile_records):
            object_count = int(tile_record["objects"]["count"])
            fragment_count = int(tile_record["routes"]["fragment_count"])
            hydrology_count = int(
                tile_record["routes"]["hydrology_fragment_count"]
            )
            rows = _read_object_rows(
                Path(tile_record["_objects_path"]),
                expected_count=object_count,
            )
            routes, hydrology = _read_route_fragments(
                Path(tile_record["_routes_path"]),
                expected_count=fragment_count,
                expected_hydrology_count=hydrology_count,
                expected_tile_ref=str(tile_record["tile_ref"]),
            )
            authored_lods: dict[str, dict[str, Any]] = {}
            for level in LOD_LEVELS:
                path = (
                    variant_dir
                    / "details"
                    / level.lower()
                    / f"tile_{index:04d}.usdc"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                metrics = self._author_objects(
                    output_path=path,
                    level=level,
                    tile_ref=str(tile_record["tile_ref"]),
                    instance_namespace=int(
                        tile_record["instance_namespace"]
                    ),
                    local_bounds=tile_record["local_bounds"],
                    epsg2154_bounds=tile_record["epsg2154_bounds"],
                    layer_counts=tile_record["detail_lod_counts"][level],
                    rows=rows,
                    routes=routes,
                    hydrology=hydrology,
                    assets=metadata["asset_library"],
                    water_materials=metadata["water_material_lods"],
                    volume_root=volume_root,
                )
                for key, value in metrics.items():
                    network_metrics_by_lod[level][key] += value
                detail_layers[level].append(path)
                authored_lods[level] = {
                    "path": path.relative_to(variant_dir).as_posix(),
                    "sha256": _sha256(path),
                }
            authored_coverage.append(
                {
                    "tile_ref": tile_record["tile_ref"],
                    "instance_namespace": tile_record["instance_namespace"],
                    "local_bounds": tile_record["local_bounds"],
                    "epsg2154_bounds": tile_record["epsg2154_bounds"],
                    "terrain_payload": {
                        key: tile_record[key]
                        for key in (
                            "path",
                            "sha256",
                            "prim_path",
                            "isolated_content_roles",
                        )
                    },
                    "object_count": object_count,
                    "route_fragment_count": fragment_count,
                    "hydrology_fragment_count": hydrology_count,
                    "detail_lod_counts": tile_record[
                        "detail_lod_counts"
                    ],
                    "detail_lods": authored_lods,
                }
            )
        root_path = variant_dir / "build" / "root.usdc"
        root_path.parent.mkdir(parents=True, exist_ok=True)
        stage = self.Usd.Stage.CreateNew(str(root_path))
        if stage is None:
            raise NativeVariantContractError(f"cannot create root USD {root_path}")
        self.UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        self.UsdGeom.SetStageUpAxis(stage, self.UsdGeom.Tokens.z)
        world = self.UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        world.GetPrim().SetCustomDataByKey(
            "fireviewer:variant_id", str(metadata["variant_id"])
        )
        world.GetPrim().SetCustomDataByKey(
            "fireviewer:scene_kind", "fictive_variant"
        )
        world.GetPrim().SetCustomDataByKey(
            "fireviewer:object_count", int(metadata["object_count"])
        )
        world.GetPrim().SetCustomDataByKey(
            "fireviewer:streaming_policy",
            "open_load_none_far_visible_mid_guard_hero_near",
        )
        raw_origin = metadata.get("epsg2154_origin")
        if (
            not isinstance(raw_origin, list)
            or len(raw_origin) != 3
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in raw_origin
            )
        ):
            raise NativeVariantContractError(
                "variant has no finite EPSG:2154 origin"
            )
        world.GetPrim().SetCustomDataByKey(
            "fireviewer:epsg2154_origin",
            ",".join(str(float(value)) for value in raw_origin),
        )
        world.GetPrim().SetCustomDataByKey(
            "fireviewer:coordinate_convention",
            "usd_z_up_meters_lambert93",
        )
        base = metadata.get("base_bindings")
        if not isinstance(base, dict):
            raise NativeVariantContractError("variant has no base bindings")
        terrain_root = self.UsdGeom.Xform.Define(stage, "/World/Terrain")
        terrain_root.GetPrim().SetCustomDataByKey(
            "fireviewer:role", "streaming_tile_headers_below_world_tiles"
        )
        fire = self.UsdGeom.Xform.Define(stage, "/World/FireAndSmoke")
        self.UsdGeom.Imageable(fire.GetPrim()).MakeInvisible()
        fire.GetPrim().SetCustomDataByKey(
            "fireviewer:default_visibility",
            "uncomposed_until_editor_review_acceptance",
        )
        fire.GetPrim().SetCustomDataByKey(
            "fireviewer:scene_kind", "fictive_variant"
        )
        # Keep simulation truth and the visible fire representation separate
        # from the base scene onward.  They stay intentionally empty and
        # invisible before the Editor gate; the post-acceptance fire worker
        # authors them together for each captured simulation state.
        fire_truth = self.UsdGeom.Xform.Define(stage, "/World/FireTruth")
        self.UsdGeom.Imageable(fire_truth.GetPrim()).MakeInvisible()
        fire_truth.GetPrim().SetCustomDataByKey(
            "fireviewer:role", "simulation_truth_pre_simulation"
        )
        fire_truth.GetPrim().SetCustomDataByKey(
            "fireviewer:state_policy",
            "must_be_authored_per_fire_state_before_capture",
        )
        fire_visual = self.UsdGeom.Xform.Define(stage, "/World/FireVisual")
        self.UsdGeom.Imageable(fire_visual.GetPrim()).MakeInvisible()
        fire_visual.GetPrim().SetCustomDataByKey(
            "fireviewer:role", "rendered_fire_visual_pre_simulation"
        )
        fire_visual.GetPrim().SetCustomDataByKey(
            "fireviewer:state_policy",
            "must_match_FireTruth_fire_state_before_capture",
        )
        review_camera_record = base.get("review_cameras")
        review_camera_path = self._validate_reference_prim(
            volume_root=volume_root,
            artifact=review_camera_record,
            label="review cameras",
        )
        review_cameras = self.UsdGeom.Xform.Define(
            stage, "/World/ReviewCameras"
        )
        review_cameras.GetPrim().GetReferences().AddReference(
            _relative_asset(root_path, review_camera_path),
            review_camera_record["prim_path"],
        )
        placed_actor_ids = self._author_selected_actors(
            stage=stage,
            root_path=root_path,
            metadata=metadata,
            volume_root=volume_root,
        )
        placed_environment_ids = self._author_supplemental_environment(
            stage=stage,
            root_path=root_path,
            metadata=metadata,
            volume_root=volume_root,
        )
        ground_record = base.get("ground_material")
        ground_binding = _ground_material_binding(
            ground_record,
            volume_root=volume_root,
            base_id=str(metadata["base_scene_id"]),
        )
        # Validate the hash-bound Scope index as evidence.  Every real Material
        # stays a payload below its terrain header so root LoadNone composes
        # zero Shader graphs.
        self._validate_reference_prim(
            volume_root=volume_root,
            artifact=ground_record,
            label="object-free tiled ground index",
        )
        ground_by_tile = {
            item.tile_id: item
            for item in ground_binding.tile_material_payloads
        }
        tiles_root = self.UsdGeom.Xform.Define(stage, "/World/Tiles")
        for index, tile_record in enumerate(tile_records):
            source_path = self._validate_reference_prim(
                volume_root=volume_root,
                artifact=tile_record,
                label=f"terrain payload {index}",
            )
            header_path = f"/World/Tiles/Tile_{index:04d}"
            header = self.UsdGeom.Xform.Define(
                stage, header_path
            )
            header.GetPrim().SetCustomDataByKey(
                "fireviewer:tile_ref", str(tile_record["tile_ref"])
            )
            header.GetPrim().SetCustomDataByKey(
                "fireviewer:instance_namespace",
                int(tile_record["instance_namespace"]),
            )
            bounds = tile_record["local_bounds"]
            header.GetPrim().SetCustomDataByKey(
                "fireviewer:local_bounds",
                ",".join(
                    str(bounds[key])
                    for key in ("min_x", "min_y", "max_x", "max_y")
                ),
            )
            terrain = self.UsdGeom.Xform.Define(
                stage, f"{header_path}/Terrain"
            )
            terrain.GetPrim().GetPayloads().AddPayload(
                _relative_asset(root_path, source_path),
                tile_record["prim_path"],
            )
            local_bounds = _bounds(tile_record["local_bounds"])
            selected_ground = ground_by_tile.get(
                str(tile_record["tile_ref"])
            )
            if selected_ground is None or not all(
                math.isclose(actual, expected, abs_tol=0.01)
                for actual, expected in (
                    (
                        selected_ground.local_bounds.min_x,
                        local_bounds.min_x,
                    ),
                    (
                        selected_ground.local_bounds.min_y,
                        local_bounds.min_y,
                    ),
                    (
                        selected_ground.local_bounds.max_x,
                        local_bounds.max_x,
                    ),
                    (
                        selected_ground.local_bounds.max_y,
                        local_bounds.max_y,
                    ),
                )
            ):
                raise NativeVariantContractError(
                    f"terrain tile {tile_record['tile_ref']} has no exact "
                    "tiled ground material"
                )
            selected_ground_path = self._validate_reference_prim(
                volume_root=volume_root,
                artifact=_artifact_dict(selected_ground.artifact),
                label=f"ground material {selected_ground.tile_id}",
            )
            ground_material = self.UsdShade.Material.Define(
                stage, f"{header_path}/Terrain/GroundMaterial"
            )
            ground_material.GetPrim().GetPayloads().AddPayload(
                _relative_asset(root_path, selected_ground_path),
                selected_ground.artifact.prim_path,
            )
            ground_material.GetPrim().SetCustomDataByKey(
                "fireviewer:ground_material_tile_id",
                selected_ground.tile_id,
            )
            self.UsdShade.MaterialBindingAPI.Apply(
                terrain.GetPrim()
            ).Bind(
                ground_material,
                bindingStrength=(
                    self.UsdShade.Tokens.strongerThanDescendants
                ),
            )
            terrain.GetPrim().SetCustomDataByKey(
                "fireviewer:ground_material_tile_id",
                selected_ground.tile_id,
            )
            for level in LOD_LEVELS:
                detail = self.UsdGeom.Xform.Define(
                    stage,
                    f"{header_path}/"
                    f"{'Details' if level == 'HERO' else 'Details' + level.title()}",
                )
                detail.GetPrim().GetPayloads().AddPayload(
                    _relative_asset(root_path, detail_layers[level][index]),
                    "/Detail",
                )
        water = self.UsdGeom.Scope.Define(stage, "/World/Water")
        raw_water = base.get("water_payloads")
        if not isinstance(raw_water, list) or not raw_water:
            raise NativeVariantContractError("variant root has no isolated water payload")
        for index, artifact in enumerate(raw_water):
            # Source water layers remain hash-bound provenance only.  The
            # visible representation is the tile-clipped HERO/MID/FAR mesh in
            # /Detail/Hydrology; composing both here would duplicate every
            # feature and produce z-fighting.
            self._validate_reference_prim(
                volume_root=volume_root,
                artifact=artifact,
                label=f"water payload {index}",
            )
        water.GetPrim().SetCustomDataByKey(
            "fireviewer:visible_geometry",
            "tiled_detail_hydrology_only",
        )
        water.GetPrim().SetCustomDataByKey(
            "fireviewer:source_payloads_provenance_count",
            len(raw_water),
        )
        stage.GetRootLayer().Save()
        reopened = self.Usd.Stage.Open(str(root_path), load=self.Usd.Stage.LoadNone)
        if (
            reopened is None
            or not reopened.GetPrimAtPath("/World/Terrain").IsValid()
            or not reopened.GetPrimAtPath("/World/FireAndSmoke").IsValid()
            or not reopened.GetPrimAtPath("/World/FireTruth").IsValid()
            or not reopened.GetPrimAtPath("/World/FireVisual").IsValid()
            or reopened.GetPrimAtPath(
                "/World/FireAndSmoke"
            ).HasAuthoredPayloads()
            or not reopened.GetPrimAtPath(
                "/World/ReviewCameras/Review06"
            ).IsValid()
            or not reopened.GetPrimAtPath("/World/Tiles").IsValid()
            or not reopened.GetPrimAtPath("/World/Water").IsValid()
            or not reopened.GetPrimAtPath("/World/Actors").IsValid()
            or not reopened.GetPrimAtPath(
                "/World/SupplementalEnvironment"
            ).IsValid()
            or any(
                not reopened.GetPrimAtPath(
                    f"/World/Actors/Actor_{index:02d}"
                ).IsValid()
                for index in range(1, ACTORS_PER_SCENE + 1)
            )
            or reopened.GetPrimAtPath("/World/Water").HasAuthoredPayloads()
            or list(
                reopened.GetPrimAtPath("/World/Water").GetChildren()
            )
            or not reopened.GetPrimAtPath(
                "/World/Tiles/Tile_0000/Terrain"
            ).IsValid()
            or not reopened.GetPrimAtPath(
                "/World/Tiles/Tile_0000/Details"
            ).IsValid()
            or not reopened.GetPrimAtPath(
                "/World/Tiles/Tile_0000/DetailsMid"
            ).IsValid()
            or not reopened.GetPrimAtPath(
                "/World/Tiles/Tile_0000/DetailsFar"
            ).IsValid()
        ):
            raise NativeVariantContractError(
                f"authored variant cannot be reopened: {metadata['variant_id']}"
            )
        root_shader_count = sum(
            1
            for prim in reopened.TraverseAll()
            if prim.IsA(self.UsdShade.Shader)
        )
        ground_payload_count = 0
        for index in range(400):
            terrain_path = (
                f"/World/Tiles/Tile_{index:04d}/Terrain"
            )
            material_path = f"{terrain_path}/GroundMaterial"
            material_prim = reopened.GetPrimAtPath(material_path)
            terrain_prim = reopened.GetPrimAtPath(terrain_path)
            binding = terrain_prim.GetRelationship("material:binding")
            if (
                not material_prim.IsValid()
                or not material_prim.IsA(self.UsdShade.Material)
                or not material_prim.HasAuthoredPayloads()
                or not binding
                or list(binding.GetTargets())
                != [self.Sdf.Path(material_path)]
            ):
                raise NativeVariantContractError(
                    f"terrain tile {index} lost its payload-streamed ground material"
                )
            ground_payload_count += 1
        if root_shader_count != 0 or ground_payload_count != 400:
            raise NativeVariantContractError(
                "root LoadNone must expose 400 material payload headers and "
                "zero eager Shader graphs"
            )
        terrain_catalog: list[dict[str, Any]] = []
        detail_catalogs: dict[str, list[dict[str, Any]]] = {
            level: [] for level in LOD_LEVELS
        }
        build_coverage: list[dict[str, Any]] = []
        for index, tile_record in enumerate(tile_records):
            terrain_source = (
                volume_root / str(tile_record["path"])
            ).resolve()
            terrain_artifact = {
                "path": Path(
                    os.path.relpath(terrain_source, start=variant_dir)
                ).as_posix(),
                "sha256": str(tile_record["sha256"]),
            }
            terrain_catalog.append(terrain_artifact)
            detail_lods: dict[str, str] = {}
            for level in LOD_LEVELS:
                detail_path = detail_layers[level][index]
                detail_artifact = {
                    "path": detail_path.relative_to(variant_dir).as_posix(),
                    "sha256": _sha256(detail_path),
                }
                detail_catalogs[level].append(detail_artifact)
                detail_lods[level] = detail_artifact["path"]
            counts = tile_record["detail_lod_counts"]
            build_coverage.append(
                {
                    "tile_ref": tile_record["tile_ref"],
                    "terrain_payload": terrain_artifact["path"],
                    "detail_payload": detail_lods["HERO"],
                    "detail_lods": detail_lods,
                    "terrain_lods": list(tile_record["terrain_lods"]),
                    "collision_lods": list(
                        tile_record["collision_lods"]
                    ),
                    "detail_counts": counts["HERO"],
                    "detail_lod_counts": counts,
                    "instance_namespace": tile_record["instance_namespace"],
                    "local_bounds": tile_record["local_bounds"],
                    "epsg2154_bounds": tile_record["epsg2154_bounds"],
                }
            )
        asset_lock_record = base.get("asset_lock")
        asset_lock_ref = _artifact(
            asset_lock_record,
            volume_root=volume_root,
            label="variant source asset lock",
        )
        shared_manifest_record = base.get("shared_asset_manifest")
        shared_manifest_ref = _artifact(
            shared_manifest_record,
            volume_root=volume_root,
            label="variant shared asset manifest",
        )
        asset_content_sha = base.get("asset_content_sha256")
        if (
            not isinstance(asset_content_sha, str)
            or not _SHA256.fullmatch(asset_content_sha)
        ):
            raise NativeVariantContractError(
                "variant shared asset content lock is malformed"
            )
        camera_count = review_camera_record.get("count")
        if (
            not isinstance(camera_count, int)
            or isinstance(camera_count, bool)
            or camera_count < 6
        ):
            raise NativeVariantContractError(
                "variant review-camera inventory is incomplete"
            )
        ground_material_receipt = {
            "topology": ground_binding.topology,
            "index": {
                "path": Path(
                    os.path.relpath(
                        (
                            volume_root
                            / ground_binding.index.path
                        ).resolve(),
                        start=variant_dir,
                    )
                ).as_posix(),
                "sha256": ground_binding.index.sha256,
                "prim_path": ground_binding.index.prim_path,
            },
            "tile_material_payloads": [
                {
                    "tile_id": item.tile_id,
                    "tile_bounds_m": [
                        item.local_bounds.min_x,
                        item.local_bounds.min_y,
                        item.local_bounds.max_x,
                        item.local_bounds.max_y,
                    ],
                    "path": Path(
                        os.path.relpath(
                            (
                                volume_root / item.artifact.path
                            ).resolve(),
                            start=variant_dir,
                        )
                    ).as_posix(),
                    "sha256": item.artifact.sha256,
                    "prim_path": item.artifact.prim_path,
                }
                for item in ground_binding.tile_material_payloads
            ],
            "binding_scope": "per_terrain_tile_stronger_than_descendants",
        }
        authored_identity = _authored_identity_contract(
            metadata.get("identity_contract")
        )
        water_payload_catalog: list[dict[str, Any]] = []
        for index, raw_artifact in enumerate(raw_water):
            water_ref = _artifact(
                raw_artifact,
                volume_root=volume_root,
                label=f"variant water payload {index}",
                require_usd=True,
                require_prim=True,
                roles=("water",),
            )
            water_payload_catalog.append(
                {
                    "path": Path(
                        os.path.relpath(
                            (volume_root / water_ref.path).resolve(),
                            start=variant_dir,
                        )
                    ).as_posix(),
                    "sha256": water_ref.sha256,
                    "prim_path": water_ref.prim_path,
                    "isolated_content_roles": list(
                        water_ref.isolated_content_roles
                    ),
                }
            )
        build_receipt_path = variant_dir / "build" / "build-receipt.json"
        build_receipt = {
            "schema_version": 2,
            "zone_id": metadata["simulation_id"],
            "variant_id": metadata["variant_id"],
            "base_scene_id": metadata["base_scene_id"],
            "variant_index": metadata["variant_index"],
            "scene_kind": "fictive_variant",
            "source_profile": "full",
            "coordinate_convention": "usd_z_up_meters_lambert93",
            "epsg2154_origin": list(raw_origin),
            "root_usd": {
                "path": root_path.relative_to(variant_dir).as_posix(),
                "sha256": _sha256(root_path),
            },
            "payloads": terrain_catalog,
            "detail_payloads": detail_catalogs["HERO"],
            "detail_mid_payloads": detail_catalogs["MID"],
            "detail_far_payloads": detail_catalogs["FAR"],
            "water_payloads": water_payload_catalog,
            "water_contract": dict(metadata["water_contract"]),
            "tile_coverage": build_coverage,
            "cameras": {
                "path": Path(
                    os.path.relpath(review_camera_path, start=variant_dir)
                ).as_posix(),
                "sha256": review_camera_record["sha256"],
                "count": camera_count,
                "root_prim": "/ReviewCameras",
            },
            "asset_lock": {
                "path": Path(
                    os.path.relpath(
                        (volume_root / asset_lock_ref.path).resolve(),
                        start=variant_dir,
                    )
                ).as_posix(),
                "sha256": asset_lock_ref.sha256,
                "assets": [
                    dict(item)
                    for item in asset_lock_record["assets"]
                ],
                "shared_manifest": {
                    "path": Path(
                        os.path.relpath(
                            (
                                volume_root
                                / shared_manifest_ref.path
                            ).resolve(),
                            start=variant_dir,
                        )
                    ).as_posix(),
                    "sha256": shared_manifest_ref.sha256,
                    "content_sha256": asset_content_sha,
                },
            },
            "identity_contract": authored_identity,
            "route_topology": dict(metadata["route_topology"]),
            "ground_material": ground_material_receipt,
            "layers": {
                "terrain": {
                    "prim_count": 400,
                    "ground_material_topology": ground_binding.topology,
                    "ground_material_payload_count": len(
                        ground_binding.tile_material_payloads
                    ),
                    "global_ground_material_binding": False,
                },
                "vegetation": {
                    "prim_count": sum(
                        metadata["family_counts"]["trees"].values()
                    )
                },
                "buildings": {
                    "prim_count": sum(
                        metadata["family_counts"]["buildings"].values()
                    )
                },
                "actors": {
                    "prim_count": len(placed_actor_ids),
                    "selected_actor_ids": placed_actor_ids,
                    "group_id": SELECTED_ACTOR_GROUP_ID,
                    "lod_levels": list(LOD_LEVELS),
                    "placeholder_substitution": False,
                },
                "supplemental_environment": {
                    "prim_count": len(placed_environment_ids),
                    "selected_environment_ids": placed_environment_ids,
                    "group_id": SELECTED_ENVIRONMENT_GROUP_ID,
                    "additive_to_existing_minima": True,
                    "placeholder_substitution": False,
                },
                "roads": {
                    "prim_count": 0,
                    "source_feature_count": int(
                        metadata["source_route_count"]
                    ),
                    "route_fragment_count": int(
                        metadata["route_fragment_count"]
                    ),
                    "visible_representation": (
                        "orthophoto_derived_terrain_material"
                    ),
                    "geometry_authoring": "disabled",
                    "asset_dependencies": [],
                    "vertices_by_lod": {
                        level: network_metrics_by_lod[level][
                            "route_vertices"
                        ]
                        for level in LOD_LEVELS
                    },
                    "faces_by_lod": {
                        level: network_metrics_by_lod[level][
                            "route_faces"
                        ]
                        for level in LOD_LEVELS
                    },
                },
                "hydrology": {
                    "prim_count": int(
                        metadata["hydrology_fragment_count"]
                    ),
                    "source_feature_count": len(
                        metadata["water_contract"][
                            "source_feature_ids"
                        ]
                    ),
                    "vertices_by_lod": {
                        level: network_metrics_by_lod[level][
                            "hydrology_vertices"
                        ]
                        for level in LOD_LEVELS
                    },
                    "faces_by_lod": {
                        level: network_metrics_by_lod[level][
                            "hydrology_faces"
                        ]
                        for level in LOD_LEVELS
                    },
                },
                "collisions": {
                    "prim_count": 400,
                    "levels": ["NEAR", "FAR"],
                    "near_spacing_m": 4.0,
                    "far_spacing_m": 32.0,
                },
                "detail_streaming": {
                    "prim_count": 400,
                    "levels": list(LOD_LEVELS),
                    "delivery": (
                        "far_all_tiles_mid_visible_hero_near_camera_working_set"
                    ),
                    "terrain_is_never_unloaded_for_detail_streaming": True,
                    "network_geometry_policy": (
                        NETWORK_GEOMETRY_POLICY
                    ),
                    "network_vertex_budget_per_tile": 262_144,
                },
                "fire": {
                    "truth_root": "/World/FireTruth",
                    "visual_root": "/World/FireVisual",
                    "state": "pre_simulation_editor_gate_pending",
                    "capture_writer": "FireViewerReplicatorWriter",
                },
            },
            "fire_simulation_status": "blocked_pending_editor_review",
        }
        _write_json(build_receipt_path, build_receipt)
        return {
            "root_usd": {
                "path": root_path.relative_to(variant_dir).as_posix(),
                "sha256": _sha256(root_path),
            },
            "streaming_tile_count": 400,
            "object_lod_payload_count": 1200,
            "object_lod_payloads": {
                level: [
                    {
                        "path": path.relative_to(variant_dir).as_posix(),
                        "sha256": _sha256(path),
                    }
                    for path in detail_layers[level]
                ]
                for level in LOD_LEVELS
            },
            "tile_coverage": authored_coverage,
            "scene_kind": "fictive_variant",
            "identity_contract": authored_identity,
            "actor_usage": {
                "group_id": SELECTED_ACTOR_GROUP_ID,
                "placed_actor_count": len(placed_actor_ids),
                "selected_actor_ids": placed_actor_ids,
                "placeholder_substitution": False,
            },
            "supplemental_environment_usage": {
                "group_id": SELECTED_ENVIRONMENT_GROUP_ID,
                "placed_asset_count": len(placed_environment_ids),
                "selected_environment_ids": placed_environment_ids,
                "additive_to_existing_minima": True,
                "placeholder_substitution": False,
            },
            "review_cameras": {
                "path": build_receipt["cameras"]["path"],
                "sha256": build_receipt["cameras"]["sha256"],
                "count": build_receipt["cameras"]["count"],
            },
            "monolithic_object_payloads": False,
            "composer_build_receipt": {
                "path": build_receipt_path.relative_to(variant_dir).as_posix(),
                "sha256": _sha256(build_receipt_path),
            },
        }


def _validate_variant_checkpoint_artifacts(
    *,
    simulation_id: str,
    target: Path,
    metadata: Mapping[str, Any],
    tile_records: Sequence[Mapping[str, Any]],
    artifacts: object,
    volume_root: Path,
) -> None:
    """Rehash one completed SIM before checkpoint reuse or publication."""

    authored_identity = _authored_identity_contract(
        metadata.get("identity_contract")
    )
    expected_actor_ids = list(_actor_usage_from_metadata(metadata))
    expected_actor_usage = {
        "group_id": SELECTED_ACTOR_GROUP_ID,
        "placed_actor_count": ACTORS_PER_SCENE,
        "selected_actor_ids": expected_actor_ids,
        "placeholder_substitution": False,
    }
    expected_environment_ids = list(
        _supplemental_environment_usage(metadata)
    )
    expected_environment_usage = {
        "group_id": SELECTED_ENVIRONMENT_GROUP_ID,
        "placed_asset_count": len(SELECTED_ENVIRONMENT_GROUP_IDS),
        "selected_environment_ids": expected_environment_ids,
        "additive_to_existing_minima": True,
        "placeholder_substitution": False,
    }
    if (
        not isinstance(artifacts, dict)
        or artifacts.get("scene_kind") != "fictive_variant"
        or artifacts.get("streaming_tile_count") != 400
        or artifacts.get("object_lod_payload_count") != 1200
        or artifacts.get("monolithic_object_payloads") is not False
        or artifacts.get("identity_contract") != authored_identity
        or artifacts.get("actor_usage") != expected_actor_usage
        or artifacts.get("supplemental_environment_usage")
        != expected_environment_usage
        or not isinstance(artifacts.get("tile_coverage"), list)
        or len(artifacts["tile_coverage"]) != 400
    ):
        raise NativeVariantContractError(
            f"{simulation_id} checkpoint artifact contract is malformed"
        )
    rehasher = _StreamingRehasher()
    rehasher.verify(
        artifacts.get("root_usd"),
        anchor=target,
        volume_root=volume_root,
        label=f"{simulation_id}.checkpoint.root_usd",
    )
    build_path = rehasher.verify(
        artifacts.get("composer_build_receipt"),
        anchor=target,
        volume_root=volume_root,
        label=f"{simulation_id}.checkpoint.build_receipt",
    )
    build = _read_json(
        build_path, label=f"{simulation_id} checkpoint build receipt"
    )
    if (
        build.get("schema_version") != 2
        or build.get("zone_id") != simulation_id
        or build.get("variant_id") != metadata.get("variant_id")
        or build.get("base_scene_id") != metadata.get("base_scene_id")
        or build.get("variant_index") != metadata.get("variant_index")
        or build.get("scene_kind") != "fictive_variant"
        or build.get("source_profile") != "full"
        or build.get("identity_contract") != authored_identity
        or build.get("route_topology") != metadata.get("route_topology")
        or build.get("water_contract") != metadata.get("water_contract")
        or build.get("fire_simulation_status")
        != "blocked_pending_editor_review"
    ):
        raise NativeVariantContractError(
            f"{simulation_id} checkpoint build contract is stale"
        )
    terrain_catalog = build.get("payloads")
    build_coverage = build.get("tile_coverage")
    artifact_coverage = artifacts.get("tile_coverage")
    lod_inventory = artifacts.get("object_lod_payloads")
    build_lods = {
        "HERO": build.get("detail_payloads"),
        "MID": build.get("detail_mid_payloads"),
        "FAR": build.get("detail_far_payloads"),
    }
    if (
        not isinstance(terrain_catalog, list)
        or len(terrain_catalog) != 400
        or not isinstance(build_coverage, list)
        or len(build_coverage) != 400
        or not isinstance(artifact_coverage, list)
        or len(artifact_coverage) != 400
        or not isinstance(lod_inventory, dict)
        or set(lod_inventory) != set(LOD_LEVELS)
    ):
        raise NativeVariantContractError(
            f"{simulation_id} checkpoint catalogs are incomplete"
        )
    unique_details: set[Path] = set()
    for level in LOD_LEVELS:
        authored_lods = lod_inventory.get(level)
        receipt_lods = build_lods[level]
        if (
            not isinstance(authored_lods, list)
            or len(authored_lods) != 400
            or not isinstance(receipt_lods, list)
            or len(receipt_lods) != 400
        ):
            raise NativeVariantContractError(
                f"{simulation_id} checkpoint {level} catalog is incomplete"
            )
        for index, (artifact, receipt_artifact) in enumerate(
            zip(authored_lods, receipt_lods)
        ):
            if _anchored_artifact_identity(
                artifact,
                anchor=target,
                volume_root=volume_root,
                label=f"{simulation_id}.checkpoint.{level}[{index}]",
            ) != _anchored_artifact_identity(
                receipt_artifact,
                anchor=target,
                volume_root=volume_root,
                label=f"{simulation_id}.checkpoint.build.{level}[{index}]",
            ):
                raise NativeVariantContractError(
                    f"{simulation_id} checkpoint {level}[{index}] diverges"
                )
            detail_path = rehasher.verify(
                artifact,
                anchor=target,
                volume_root=volume_root,
                label=f"{simulation_id}.checkpoint.{level}[{index}]",
            )
            if detail_path in unique_details:
                raise NativeVariantContractError(
                    f"{simulation_id} checkpoint reuses one detail layer"
                )
            unique_details.add(detail_path)
    if len(unique_details) != 1200:
        raise NativeVariantContractError(
            f"{simulation_id} checkpoint does not own 1200 detail layers"
        )
    lod0_count = 0
    for index, (tile, terrain, coverage, authored_coverage) in enumerate(
        zip(
            tile_records,
            terrain_catalog,
            build_coverage,
            artifact_coverage,
        )
    ):
        terrain_path = rehasher.verify(
            terrain,
            anchor=target,
            volume_root=volume_root,
            label=f"{simulation_id}.checkpoint.terrain[{index}]",
            shared=True,
        )
        expected_terrain_path = (
            volume_root / str(tile.get("path", ""))
        ).resolve()
        if (
            terrain_path != expected_terrain_path
            or terrain.get("sha256") != tile.get("sha256")
            or not isinstance(coverage, dict)
            or coverage.get("tile_ref") != tile.get("tile_ref")
            or coverage.get("terrain_lods") != tile.get("terrain_lods")
            or coverage.get("collision_lods")
            != tile.get("collision_lods")
            or not isinstance(authored_coverage, dict)
            or authored_coverage.get("tile_ref") != tile.get("tile_ref")
            or any(
                coverage.get("detail_lods", {}).get(level)
                != build_lods[level][index].get("path")
                or authored_coverage.get("detail_lods", {}).get(level)
                != lod_inventory[level][index]
                for level in LOD_LEVELS
            )
        ):
            raise NativeVariantContractError(
                f"{simulation_id} checkpoint tile {index} changed binding"
            )
        if "LOD0" in coverage["terrain_lods"]:
            lod0_count += 1
    if lod0_count <= 0:
        raise NativeVariantContractError(
            f"{simulation_id} checkpoint exposes no real LOD0"
        )
    expected_water = metadata.get("base_bindings", {}).get(
        "water_payloads"
    )
    water_catalog = build.get("water_payloads")
    if (
        not isinstance(expected_water, list)
        or not isinstance(water_catalog, list)
        or not expected_water
        or len(expected_water) != len(water_catalog)
    ):
        raise NativeVariantContractError(
            f"{simulation_id} checkpoint water provenance is incomplete"
        )
    for index, (actual, expected) in enumerate(
        zip(water_catalog, expected_water)
    ):
        path = rehasher.verify(
            actual,
            anchor=target,
            volume_root=volume_root,
            label=f"{simulation_id}.checkpoint.water[{index}]",
            shared=True,
        )
        if (
            path
            != (volume_root / str(expected.get("path", ""))).resolve()
            or actual.get("sha256") != expected.get("sha256")
            or actual.get("prim_path") != expected.get("prim_path")
            or actual.get("isolated_content_roles")
            != expected.get("isolated_content_roles")
        ):
            raise NativeVariantContractError(
                f"{simulation_id} checkpoint water source changed"
            )
    expected_ground = metadata.get("base_bindings", {}).get(
        "ground_material"
    )
    ground = build.get("ground_material")
    if (
        not isinstance(expected_ground, dict)
        or not isinstance(ground, dict)
        or ground.get("topology") != expected_ground.get("topology")
        or ground.get("binding_scope")
        != "per_terrain_tile_stronger_than_descendants"
        or not isinstance(ground.get("tile_material_payloads"), list)
        or len(ground["tile_material_payloads"]) != 400
    ):
        raise NativeVariantContractError(
            f"{simulation_id} checkpoint ground catalog is incomplete"
        )
    expected_ground_by_id = {
        str(item["tile_id"]): item
        for item in expected_ground["tile_material_payloads"]
    }
    for index, actual in enumerate(ground["tile_material_payloads"]):
        expected = expected_ground_by_id.get(str(actual.get("tile_id")))
        if not isinstance(expected, dict):
            raise NativeVariantContractError(
                f"{simulation_id} checkpoint ground tile {index} is unknown"
            )
        path = rehasher.verify(
            actual,
            anchor=target,
            volume_root=volume_root,
            label=f"{simulation_id}.checkpoint.ground[{index}]",
            shared=True,
        )
        if (
            path
            != (volume_root / str(expected.get("path", ""))).resolve()
            or actual.get("sha256") != expected.get("sha256")
            or actual.get("prim_path") != expected.get("prim_path")
            or actual.get("tile_bounds_m")
            != expected.get("tile_bounds_m")
        ):
            raise NativeVariantContractError(
                f"{simulation_id} checkpoint ground tile {index} changed"
            )


def author_variant_campaign(
    *,
    plan_path: Path,
    volume_root: Path,
    output_root: Path,
    _backend: Any | None = None,
) -> dict[str, Any]:
    """Author all 20 variants with native pxr, atomically and without fire."""

    volume = volume_root.expanduser().resolve()
    plan = plan_path.expanduser().resolve()
    if not _is_below(volume, plan) or not plan.is_file():
        raise NativeVariantContractError(
            "campaign plan must be below the persistent volume"
        )
    campaign = _read_json(plan, label="campaign plan")
    if (
        campaign.get("schema_version") != PLAN_SCHEMA_VERSION
        or campaign.get("state") != "VARIANT_PLAN_READY"
        or campaign.get("base_scene_count") != BASE_SCENE_COUNT
        or campaign.get("variants_per_base") != VARIANTS_PER_BASE
        or campaign.get("simulation_count") != PORTFOLIO_SCENE_COUNT
        or campaign.get("fire_simulation_status")
        != "blocked_pending_editor_review"
    ):
        raise NativeVariantContractError("campaign plan is not the fixed blocked 4 x 5 plan")
    variants = campaign.get("variants")
    if not isinstance(variants, list) or len(variants) != PORTFOLIO_SCENE_COUNT:
        raise NativeVariantContractError("campaign plan does not contain 20 variants")
    output = output_root.expanduser().resolve()
    if not _is_below(volume, output) or output == volume:
        raise NativeVariantContractError(
            "authoring output must be a dedicated directory below the volume"
        )
    plan_root = plan.parent
    if output.exists():
        receipt_path = output / "authoring-receipt.json"
        if not receipt_path.is_file():
            raise NativeVariantContractError(
                "existing authoring output has no final receipt"
            )
        verify_variant_campaign(
            plan_path=plan,
            layout_paths=_campaign_layout_paths(
                campaign=campaign,
                plan_root=plan_root,
                volume_root=volume,
            ),
            volume_root=volume,
            authoring_receipt_path=receipt_path,
        )
        return _read_json(receipt_path, label="authoring receipt")
    backend = _backend if _backend is not None else _PXRVariantAuthor()
    plan_sha = _sha256(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(
        f".{output.name}.authoring-{plan_sha[:16]}.partial"
    )
    campaign_checkpoint_path = staging / "campaign-checkpoint.json"
    expected_campaign_checkpoint = {
        "schema_version": AUTHORING_SCHEMA_VERSION,
        "state": "VARIANT_AUTHORING_IN_PROGRESS",
        "plan": {
            "path": plan.relative_to(volume).as_posix(),
            "sha256": plan_sha,
        },
        "simulation_count": PORTFOLIO_SCENE_COUNT,
        "fire_simulation_status": "blocked_pending_editor_review",
    }
    if staging.exists():
        if not campaign_checkpoint_path.is_file() or _read_json(
            campaign_checkpoint_path,
            label="authoring campaign checkpoint",
        ) != expected_campaign_checkpoint:
            raise NativeVariantContractError(
                "existing authoring checkpoint belongs to another plan"
            )
    else:
        staging.mkdir(parents=False, exist_ok=False)
        _write_json(
            campaign_checkpoint_path, expected_campaign_checkpoint
        )
    try:
        authored: list[dict[str, Any]] = []
        for expected_sequence, record in enumerate(variants, start=1):
            expected_simulation = f"SIM-{expected_sequence:02d}"
            if (
                not isinstance(record, dict)
                or record.get("simulation_id") != expected_simulation
            ):
                raise NativeVariantContractError(
                    "campaign simulation ordering is not SIM-01..SIM-20"
                )
            metadata_path = _resolve_plan_artifact(
                record.get("metadata"),
                root=plan_root,
                label=f"{expected_simulation} metadata",
            )
            metadata = _read_json(metadata_path, label=f"{expected_simulation} metadata")
            if (
                metadata.get("simulation_id") != expected_simulation
                or metadata.get("variant_id") != record.get("variant_id")
                or metadata.get("authoring_status") != "planned_not_authored"
                or metadata.get("fire_simulation_status")
                != "blocked_pending_editor_review"
            ):
                raise NativeVariantContractError(
                    f"{expected_simulation} metadata is stale or already mutated"
                )
            expected_actor_ids = list(_actor_usage_from_metadata(metadata))
            expected_actor_usage = {
                "group_id": SELECTED_ACTOR_GROUP_ID,
                "placed_actor_count": ACTORS_PER_SCENE,
                "selected_actor_ids": expected_actor_ids,
                "placeholder_substitution": False,
            }
            expected_environment_ids = list(
                _supplemental_environment_usage(metadata)
            )
            expected_environment_usage = {
                "group_id": SELECTED_ENVIRONMENT_GROUP_ID,
                "placed_asset_count": len(SELECTED_ENVIRONMENT_GROUP_IDS),
                "selected_environment_ids": expected_environment_ids,
                "additive_to_existing_minima": True,
                "placeholder_substitution": False,
            }
            tile_records = _load_variant_tiles(
                metadata, variant_dir=metadata_path.parent
            )
            authored_identity = _authored_identity_contract(
                metadata.get("identity_contract")
            )
            final_target = staging / expected_simulation
            if final_target.exists():
                variant_checkpoint_path = (
                    final_target / "variant-checkpoint.json"
                )
                variant_checkpoint = _read_json(
                    variant_checkpoint_path,
                    label=f"{expected_simulation} variant checkpoint",
                )
                checkpoint_record = variant_checkpoint.get(
                    "authored_record"
                )
                if (
                    variant_checkpoint.get("schema_version")
                    != AUTHORING_SCHEMA_VERSION
                    or variant_checkpoint.get("state")
                    != "VARIANT_USD_CHECKPOINTED"
                    or variant_checkpoint.get("simulation_id")
                    != expected_simulation
                    or variant_checkpoint.get("metadata_sha256")
                    != record.get("metadata", {}).get("sha256")
                    or not isinstance(checkpoint_record, dict)
                ):
                    raise NativeVariantContractError(
                        f"{expected_simulation} checkpoint is stale"
                    )
                _validate_variant_checkpoint_artifacts(
                    simulation_id=expected_simulation,
                    target=final_target,
                    metadata=metadata,
                    tile_records=tile_records,
                    artifacts=checkpoint_record.get("artifacts"),
                    volume_root=volume,
                )
                if (
                    checkpoint_record.get("variant_id")
                    != metadata.get("variant_id")
                    or checkpoint_record.get("base_scene_id")
                    != metadata.get("base_scene_id")
                    or checkpoint_record.get("variant_index")
                    != metadata.get("variant_index")
                    or checkpoint_record.get("identity_contract")
                    != authored_identity
                ):
                    raise NativeVariantContractError(
                        f"{expected_simulation} checkpoint identity is stale"
                    )
                authored.append(checkpoint_record)
                continue
            target = staging / f".{expected_simulation}.incomplete"
            if target.exists():
                shutil.rmtree(target)
            artifacts = backend.author_variant(
                variant_dir=target,
                metadata=metadata,
                tile_records=tile_records,
                volume_root=volume,
            )
            if (
                not isinstance(artifacts, dict)
                or artifacts.get("scene_kind") != "fictive_variant"
                or artifacts.get("streaming_tile_count") != 400
                or artifacts.get("object_lod_payload_count") != 1200
                or artifacts.get("monolithic_object_payloads") is not False
                or not isinstance(artifacts.get("tile_coverage"), list)
                or len(artifacts["tile_coverage"]) != 400
                or artifacts.get("identity_contract")
                != authored_identity
                or artifacts.get("actor_usage") != expected_actor_usage
                or artifacts.get("supplemental_environment_usage")
                != expected_environment_usage
            ):
                raise NativeVariantContractError(
                    f"{expected_simulation} authoring did not produce "
                    "400 tiled HERO/MID/FAR payload chains"
                )
            _resolve_plan_artifact(
                artifacts.get("root_usd"),
                root=target,
                label=f"{expected_simulation}.root_usd",
            )
            composer_receipt_path = _resolve_plan_artifact(
                artifacts.get("composer_build_receipt"),
                root=target,
                label=f"{expected_simulation}.composer_build_receipt",
            )
            composer_receipt = _read_json(
                composer_receipt_path,
                label=f"{expected_simulation} Composer build receipt",
            )
            if (
                composer_receipt.get("schema_version") != 2
                or composer_receipt.get("zone_id") != expected_simulation
                or composer_receipt.get("variant_id")
                != metadata.get("variant_id")
                or composer_receipt.get("base_scene_id")
                != metadata.get("base_scene_id")
                or composer_receipt.get("variant_index")
                != metadata.get("variant_index")
                or composer_receipt.get("scene_kind")
                != "fictive_variant"
                or composer_receipt.get("source_profile") != "full"
                or composer_receipt.get("identity_contract")
                != authored_identity
                or composer_receipt.get("water_contract")
                != metadata.get("water_contract")
                or any(
                    not isinstance(composer_receipt.get(field), list)
                    or len(composer_receipt[field]) != 400
                    for field in (
                        "payloads",
                        "detail_payloads",
                        "detail_mid_payloads",
                        "detail_far_payloads",
                        "tile_coverage",
                    )
                )
            ):
                raise NativeVariantContractError(
                    f"{expected_simulation} is not directly consumable by "
                    "open-zone-scene-in-composer.py"
                )
            actor_layer = composer_receipt.get("layers", {}).get("actors")
            if (
                not isinstance(actor_layer, dict)
                or actor_layer.get("prim_count") != ACTORS_PER_SCENE
                or actor_layer.get("selected_actor_ids") != expected_actor_ids
                or actor_layer.get("group_id") != SELECTED_ACTOR_GROUP_ID
                or actor_layer.get("lod_levels") != list(LOD_LEVELS)
                or actor_layer.get("placeholder_substitution") is not False
            ):
                raise NativeVariantContractError(
                    f"{expected_simulation} did not author its five selected actors"
                )
            environment_layer = composer_receipt.get("layers", {}).get(
                "supplemental_environment"
            )
            if (
                not isinstance(environment_layer, dict)
                or environment_layer.get("prim_count")
                != len(SELECTED_ENVIRONMENT_GROUP_IDS)
                or environment_layer.get("selected_environment_ids")
                != expected_environment_ids
                or environment_layer.get("group_id")
                != SELECTED_ENVIRONMENT_GROUP_ID
                or environment_layer.get("additive_to_existing_minima")
                is not True
                or environment_layer.get("placeholder_substitution")
                is not False
            ):
                raise NativeVariantContractError(
                    f"{expected_simulation} did not author all four additive "
                    "environment assets"
                )
            composer_coverage = composer_receipt["tile_coverage"]
            lod0_count = 0
            for tile_index, (coverage_record, tile_record) in enumerate(
                zip(composer_coverage, tile_records)
            ):
                if (
                    not isinstance(coverage_record, dict)
                    or coverage_record.get("tile_ref")
                    != tile_record.get("tile_ref")
                    or coverage_record.get("terrain_lods")
                    != tile_record.get("terrain_lods")
                    or coverage_record.get("collision_lods")
                    != tile_record.get("collision_lods")
                ):
                    raise NativeVariantContractError(
                        f"{expected_simulation} terrain/collision LOD "
                        f"binding {tile_index} differs from the native base"
                    )
                if "LOD0" in coverage_record["terrain_lods"]:
                    lod0_count += 1
            if lod0_count <= 0:
                raise NativeVariantContractError(
                    f"{expected_simulation} exposes no real review-camera LOD0"
                )
            receipt_water = composer_receipt.get("water_payloads")
            expected_water = metadata.get("base_bindings", {}).get(
                "water_payloads"
            )
            if (
                not isinstance(receipt_water, list)
                or not isinstance(expected_water, list)
                or len(receipt_water) != len(expected_water)
                or not receipt_water
            ):
                raise NativeVariantContractError(
                    f"{expected_simulation} isolated water catalog is incomplete"
                )
            for water_index, (actual, expected) in enumerate(
                zip(receipt_water, expected_water)
            ):
                actual_path, actual_sha = _anchored_artifact_identity(
                    actual,
                    anchor=target,
                    volume_root=volume,
                    label=(
                        f"{expected_simulation}.water[{water_index}]"
                    ),
                )
                expected_path = (
                    volume / str(expected.get("path", ""))
                ).resolve()
                if (
                    actual_path != expected_path
                    or actual_sha != expected.get("sha256")
                    or actual.get("prim_path")
                    != expected.get("prim_path")
                    or actual.get("isolated_content_roles")
                    != expected.get("isolated_content_roles")
                ):
                    raise NativeVariantContractError(
                        f"{expected_simulation} water payload "
                        f"{water_index} changed binding"
                    )
            ground_receipt = composer_receipt.get("ground_material")
            if (
                not isinstance(ground_receipt, dict)
                or not isinstance(
                    ground_receipt.get("tile_material_payloads"), list
                )
                or len(ground_receipt["tile_material_payloads"]) != 400
                or ground_receipt.get("binding_scope")
                != "per_terrain_tile_stronger_than_descendants"
            ):
                raise NativeVariantContractError(
                    f"{expected_simulation} ground payload catalog is incomplete"
                )
            build_layers = composer_receipt.get("layers")
            roads_layer = (
                build_layers.get("roads")
                if isinstance(build_layers, dict)
                else None
            )
            hydrology_layer = (
                build_layers.get("hydrology")
                if isinstance(build_layers, dict)
                else None
            )
            if (
                not isinstance(roads_layer, dict)
                or roads_layer.get("prim_count") != 0
                or roads_layer.get("source_feature_count")
                != metadata.get("source_route_count")
                or roads_layer.get("route_fragment_count")
                != metadata.get("route_fragment_count")
                or roads_layer.get("visible_representation")
                != "orthophoto_derived_terrain_material"
                or roads_layer.get("geometry_authoring") != "disabled"
                or roads_layer.get("asset_dependencies") != []
                or not isinstance(hydrology_layer, dict)
                or hydrology_layer.get("prim_count")
                != metadata.get("hydrology_fragment_count")
                or hydrology_layer.get("source_feature_count")
                != len(
                    metadata.get("water_contract", {}).get(
                        "source_feature_ids", []
                    )
                )
            ):
                raise NativeVariantContractError(
                    f"{expected_simulation} network fragment counts are stale"
                )
            lod_payloads = artifacts.get("object_lod_payloads")
            if not isinstance(lod_payloads, dict) or set(lod_payloads) != set(
                LOD_LEVELS
            ):
                raise NativeVariantContractError(
                    f"{expected_simulation} has no complete LOD payload inventory"
                )
            unique_lod_paths: set[Path] = set()
            for level in LOD_LEVELS:
                records_for_level = lod_payloads[level]
                if (
                    not isinstance(records_for_level, list)
                    or len(records_for_level) != 400
                ):
                    raise NativeVariantContractError(
                        f"{expected_simulation}.{level} is not 400 tiled payloads"
                    )
                for tile_index, artifact in enumerate(records_for_level):
                    path = _resolve_plan_artifact(
                        artifact,
                        root=target,
                        label=(
                            f"{expected_simulation}.{level}[{tile_index}]"
                        ),
                    )
                    if path in unique_lod_paths:
                        raise NativeVariantContractError(
                            f"{expected_simulation} reuses one file for multiple LOD tiles"
                        )
                    unique_lod_paths.add(path)
            if len(unique_lod_paths) != 1200:
                raise NativeVariantContractError(
                    f"{expected_simulation} does not own 1200 distinct object payloads"
                )
            for tile_index, coverage_record in enumerate(
                artifacts["tile_coverage"]
            ):
                if not isinstance(coverage_record, dict):
                    raise NativeVariantContractError(
                        f"{expected_simulation} tile coverage is malformed"
                    )
                detail_lods = coverage_record.get("detail_lods")
                if not isinstance(detail_lods, dict) or any(
                    detail_lods.get(level)
                    != lod_payloads[level][tile_index]
                    for level in LOD_LEVELS
                ):
                    raise NativeVariantContractError(
                        f"{expected_simulation} tile {tile_index} LOD bindings diverge"
                    )
            authored_record = {
                "simulation_id": expected_simulation,
                "variant_id": metadata["variant_id"],
                "base_scene_id": metadata["base_scene_id"],
                "variant_index": metadata["variant_index"],
                "artifacts": artifacts,
                "object_count": metadata["object_count"],
                "family_counts": metadata["family_counts"],
                "scene_kind": "fictive_variant",
                "identity_contract": authored_identity,
                "actor_usage": expected_actor_usage,
                "supplemental_environment_usage": (
                    expected_environment_usage
                ),
                "review_cameras": artifacts.get("review_cameras"),
                "streaming_tile_count": artifacts.get(
                    "streaming_tile_count"
                ),
                "object_lod_payload_count": artifacts.get(
                    "object_lod_payload_count"
                ),
                "monolithic_object_payloads": artifacts.get(
                    "monolithic_object_payloads"
                ),
                "fire_simulation_status": "blocked_pending_editor_review",
            }
            _validate_variant_checkpoint_artifacts(
                simulation_id=expected_simulation,
                target=target,
                metadata=metadata,
                tile_records=tile_records,
                artifacts=artifacts,
                volume_root=volume,
            )
            _write_json(
                target / "variant-checkpoint.json",
                {
                    "schema_version": AUTHORING_SCHEMA_VERSION,
                    "state": "VARIANT_USD_CHECKPOINTED",
                    "simulation_id": expected_simulation,
                    "metadata_sha256": record["metadata"]["sha256"],
                    "authored_record": authored_record,
                },
            )
            os.replace(target, final_target)
            authored.append(authored_record)
        receipt = {
            "schema_version": AUTHORING_SCHEMA_VERSION,
            "state": "VARIANT_USD_AUTHORED",
            "plan": {
                "path": plan.relative_to(volume).as_posix(),
                "sha256": _sha256(plan),
            },
            "simulation_count": PORTFOLIO_SCENE_COUNT,
            "variants": authored,
            "review_target": {
                "simulation_id": "SIM-01",
                "root_usd": "SIM-01/build/root.usdc",
                "composer_build_receipt": (
                    "SIM-01/build/build-receipt.json"
                ),
                "must_be_reviewed_before_fire": True,
            },
            "manual_editor_review": "required",
            "fire_simulation_status": "blocked_pending_editor_review",
        }
        _write_json(staging / "authoring-receipt.json", receipt)
        os.replace(staging, output)
        return _read_json(
            output / "authoring-receipt.json", label="authoring receipt"
        )
    except BaseException:
        # Keep every hash-verified SIM checkpoint.  A subsequent identical
        # invocation revalidates and reuses them, while rebuilding only the
        # one deterministic .SIM-xx.incomplete directory interrupted in flight.
        raise


def _campaign_layout_paths(
    *,
    campaign: Mapping[str, Any],
    plan_root: Path,
    volume_root: Path,
) -> tuple[Path, ...]:
    variants = campaign.get("variants")
    if not isinstance(variants, list):
        raise NativeVariantContractError(
            "campaign has no variant inventory for idempotent verification"
        )
    by_base: dict[str, Path] = {}
    for record in variants:
        if not isinstance(record, dict):
            raise NativeVariantContractError(
                "campaign variant record is malformed"
            )
        base_id = record.get("base_scene_id")
        if not isinstance(base_id, str) or base_id in by_base:
            continue
        metadata_path = _resolve_plan_artifact(
            record.get("metadata"),
            root=plan_root,
            label=f"{base_id} idempotent metadata",
        )
        metadata = _read_json(
            metadata_path, label=f"{base_id} idempotent metadata"
        )
        bindings = metadata.get("base_bindings")
        raw_path = (
            bindings.get("layout_path")
            if isinstance(bindings, dict)
            else None
        )
        portable = _portable_path(
            raw_path, label=f"{base_id}.layout_path"
        )
        path = (volume_root / portable).resolve()
        if not _is_below(volume_root, path) or not path.is_file():
            raise NativeVariantContractError(
                f"{base_id} idempotent layout is absent"
            )
        by_base[base_id] = path
    if len(by_base) != BASE_SCENE_COUNT:
        raise NativeVariantContractError(
            "campaign does not bind exactly four native layouts"
        )
    return tuple(by_base[key] for key in sorted(by_base))


def _expected_plan_base_bindings(
    base: NativeBaseLayout,
    *,
    volume_root: Path,
) -> dict[str, Any]:
    return {
        "layout_path": base.layout_path.relative_to(volume_root).as_posix(),
        "layout_sha256": base.layout_sha256,
        "native_build_receipt": _artifact_dict(base.build_receipt),
        "scene_auto_validation": _artifact_dict(base.auto_validation),
        "source_root_usd": _artifact_dict(base.root_usd),
        "asset_lock": {
            **_artifact_dict(base.asset_lock),
            "assets": [dict(item) for item in base.asset_lock_assets],
        },
        "shared_asset_manifest": _artifact_dict(
            base.shared_asset_manifest
        ),
        "asset_content_sha256": base.asset_content_sha256,
        "review_cameras": {
            **_artifact_dict(base.review_cameras),
            "count": base.review_camera_count,
        },
        "water_payloads": [
            _artifact_dict(item) for item in base.water_payloads
        ],
        "preview_height_field": _artifact_dict(
            base.preview_height_field
        ),
        "placement_height": {
            "provider": "hash_bound_tiled_float32",
            "tile_count": 400,
            "cache_tile_limit": 2,
            "content_fingerprint": base.placement_height_fingerprint,
        },
        "ground_material": _ground_material_dict(base.ground_material),
    }


def _verify_plan_against_layouts(
    *,
    plan_path: Path,
    layout_paths: Sequence[Path],
    volume_root: Path,
) -> dict[str, Any]:
    """Revalidate a persisted plan against its four current native layouts.

    The verifier intentionally does not regenerate variants.  It rehashes all
    source/layout and plan artifacts, recomputes source identity/topology, and
    streams each tile object/route file through the same fail-closed reader
    used by authoring.  This makes setup resumption deterministic without
    retaining four base scenes or one complete variant in memory.
    """

    volume = volume_root.expanduser().resolve()
    plan = plan_path.expanduser().resolve()
    if not _is_below(volume, plan) or not plan.is_file():
        raise NativeVariantContractError(
            "campaign plan must be an existing file below the persistent volume"
        )
    if len(layout_paths) != BASE_SCENE_COUNT:
        raise NativeVariantContractError(
            f"verify requires exactly {BASE_SCENE_COUNT} base layouts"
        )
    identities = tuple(
        sorted(
            (
                _layout_identity(path, volume_root=volume)
                for path in layout_paths
            ),
            key=lambda item: item[0],
        )
    )
    if len({base_id for base_id, _path in identities}) != BASE_SCENE_COUNT:
        raise NativeVariantContractError(
            "verify base layout stable IDs must be unique"
        )
    campaign = _read_json(plan, label="campaign plan")
    if (
        campaign.get("schema_version") != PLAN_SCHEMA_VERSION
        or campaign.get("state") != "VARIANT_PLAN_READY"
        or campaign.get("algorithm") != ALGORITHM_ID
        or campaign.get("base_scene_count") != BASE_SCENE_COUNT
        or campaign.get("variants_per_base") != VARIANTS_PER_BASE
        or campaign.get("simulation_count") != PORTFOLIO_SCENE_COUNT
        or campaign.get("authoring_status") != "planned_not_authored"
        or campaign.get("fire_simulation_status")
        != "blocked_pending_editor_review"
    ):
        raise NativeVariantContractError(
            "campaign plan is not the fixed blocked 4 x 5 plan"
        )
    expected_base_ids = [base_id for base_id, _path in identities]
    if campaign.get("base_scene_ids") != expected_base_ids:
        raise NativeVariantContractError(
            "campaign base ordering differs from the four supplied layouts"
        )

    snapshots: dict[str, dict[str, Any]] = {}
    expected_constraints: dict[str, Any] = {}
    shared_asset_contract: dict[str, str] | None = None
    for base_id, layout_path in identities:
        base = load_native_base_layout(layout_path, volume_root=volume)
        if base.scene.stable_id != base_id:
            raise NativeVariantContractError(
                "composition source identity changed during verification"
            )
        source_identity_sha = _identity_sha256(
            trees=base.scene.trees,
            buildings=base.scene.buildings,
            bindings=base.object_bindings,
        )
        expected_constraints[base_id] = _constraints_dict(
            base.variant_constraints
        )
        current_asset_contract = {
            "manifest_path": base.shared_asset_manifest.path,
            "manifest_sha256": base.shared_asset_manifest.sha256,
            "content_sha256": base.asset_content_sha256,
        }
        if shared_asset_contract is None:
            shared_asset_contract = current_asset_contract
        elif current_asset_contract != shared_asset_contract:
            raise NativeVariantContractError(
                "the four verified layouts no longer share one asset lock"
            )
        snapshots[base_id] = {
            "base_bindings": _expected_plan_base_bindings(
                base, volume_root=volume
            ),
            "asset_library": _asset_library_dict(base.assets),
            "selected_actor_library": _actor_library_dict(
                base.selected_actors
            ),
            "supplemental_environment_library": (
                _supplemental_environment_dict(
                    base.supplemental_environment
                )
            ),
            "road_visual_contract": dict(ROAD_VISUAL_CONTRACT),
            "water_material_lods": _water_materials_dict(
                base.water_materials
            ),
            "identity_sha256": source_identity_sha,
            "route_component_count": base.route_component_count,
            "route_membership_sha256": base.route_membership_sha256,
            "placement_height_fingerprint": (
                base.placement_height_fingerprint
            ),
        }
        del base
    if campaign.get("constraints_by_base") != expected_constraints:
        raise NativeVariantContractError(
            "campaign constraints differ from the supplied layouts"
        )
    if campaign.get("shared_asset_contract") != shared_asset_contract:
        raise NativeVariantContractError(
            "campaign shared asset contract differs from the supplied layouts"
        )

    variants = campaign.get("variants")
    if not isinstance(variants, list) or len(variants) != PORTFOLIO_SCENE_COUNT:
        raise NativeVariantContractError(
            "campaign plan does not contain exactly 20 variants"
        )
    expected_bindings: dict[str, dict[str, Any]] = {}
    actor_usage_counts = {
        selection_id: 0 for selection_id in SELECTED_ACTOR_GROUP_IDS
    }
    plan_root = plan.parent
    for sequence, record in enumerate(variants, start=1):
        simulation_id = f"SIM-{sequence:02d}"
        base_id = expected_base_ids[
            (sequence - 1) // VARIANTS_PER_BASE
        ]
        variant_index = (sequence - 1) % VARIANTS_PER_BASE + 1
        if (
            not isinstance(record, dict)
            or record.get("simulation_id") != simulation_id
            or record.get("base_scene_id") != base_id
            or record.get("variant_index") != variant_index
            or not isinstance(record.get("variant_id"), str)
        ):
            raise NativeVariantContractError(
                "campaign simulation binding/order is not canonical"
            )
        metadata_path = _resolve_plan_artifact(
            record.get("metadata"),
            root=plan_root,
            label=f"{simulation_id} metadata",
        )
        metadata = _read_json(
            metadata_path, label=f"{simulation_id} metadata"
        )
        snapshot = snapshots[base_id]
        identity_sha = snapshot["identity_sha256"]
        expected_identity = {
            "version": 1,
            "numeric_ids_preserved": True,
            "stable_ids_preserved": True,
            "source_namespace_may_differ_from_destination_tile": True,
            "source_identity_sha256": identity_sha,
            "result_identity_sha256": identity_sha,
        }
        route_topology_contract = metadata.get("route_topology")
        if (
            metadata.get("schema_version") != PLAN_SCHEMA_VERSION
            or metadata.get("simulation_id") != simulation_id
            or metadata.get("variant_id") != record.get("variant_id")
            or metadata.get("base_scene_id") != base_id
            or metadata.get("variant_index") != variant_index
            or metadata.get("algorithm") != ALGORITHM_ID
            or metadata.get("authoring_status") != "planned_not_authored"
            or metadata.get("fire_simulation_status")
            != "blocked_pending_editor_review"
            or metadata.get("identity_contract") != expected_identity
            or metadata.get("base_bindings")
            != snapshot["base_bindings"]
            or metadata.get("asset_library")
            != snapshot["asset_library"]
            or metadata.get("selected_actor_library")
            != snapshot["selected_actor_library"]
            or metadata.get("supplemental_environment_library")
            != snapshot["supplemental_environment_library"]
            or metadata.get("road_visual_contract")
            != snapshot["road_visual_contract"]
            or metadata.get("water_material_lods")
            != snapshot["water_material_lods"]
            or not isinstance(route_topology_contract, dict)
            or route_topology_contract.get("source_component_count")
            != snapshot["route_component_count"]
            or route_topology_contract.get("result_component_count")
            != snapshot["route_component_count"]
            or route_topology_contract.get("source_membership_sha256")
            != snapshot["route_membership_sha256"]
            or route_topology_contract.get("result_membership_sha256")
            != snapshot["route_membership_sha256"]
            or route_topology_contract.get("exact_membership_preserved")
            is not True
        ):
            raise NativeVariantContractError(
                f"{simulation_id} metadata differs from its current native base"
            )
        for selection_id in _actor_usage_from_metadata(metadata):
            actor_usage_counts[selection_id] += 1
        _supplemental_environment_usage(metadata)
        terrain_contract = metadata.get("terrain_contract")
        if (
            not isinstance(terrain_contract, dict)
            or terrain_contract.get("height_field_fingerprint")
            != snapshot["placement_height_fingerprint"]
            or terrain_contract.get("placement_height_fingerprint")
            != snapshot["placement_height_fingerprint"]
            or terrain_contract.get("placement_height_provider")
            != "hash_bound_tiled_float32"
            or terrain_contract.get("placement_height_tile_count") != 400
            or terrain_contract.get("placement_height_cache_tile_limit") != 2
        ):
            raise NativeVariantContractError(
                f"{simulation_id} placement-height contract is stale"
            )
        tiles = _load_variant_tiles(
            metadata, variant_dir=metadata_path.parent
        )
        if len(tiles) != 400:
            raise AssertionError("streaming tile verifier lost coverage")
        del tiles
        expected_bindings[simulation_id] = {
            "base_scene_id": base_id,
            "variant_id": record["variant_id"],
            "variant_index": variant_index,
        }
    if campaign.get("simulation_base_bindings") != expected_bindings:
        raise NativeVariantContractError(
            "campaign simulation/base binding index is stale"
        )
    expected_actor_contract = {
        "group_id": SELECTED_ACTOR_GROUP_ID,
        "selection_count": len(SELECTED_ACTOR_GROUP_IDS),
        "selection_order": list(SELECTED_ACTOR_GROUP_IDS),
        "actors_per_scene": ACTORS_PER_SCENE,
        "total_actor_placements": PORTFOLIO_SCENE_COUNT * ACTORS_PER_SCENE,
        "usage_counts": actor_usage_counts,
        "all_selected_assets_used": True,
        "placeholder_substitution": False,
    }
    if (
        any(count <= 0 for count in actor_usage_counts.values())
        or campaign.get("actor_usage_contract") != expected_actor_contract
    ):
        raise NativeVariantContractError(
            "campaign actor usage contract is incomplete or stale"
        )
    return campaign


def _anchored_artifact_identity(
    payload: object,
    *,
    anchor: Path,
    volume_root: Path,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(payload, dict):
        raise NativeVariantContractError(f"{label} must be an artifact")
    raw_path = payload.get("path")
    if (
        not isinstance(raw_path, str)
        or not raw_path.strip()
        or raw_path != raw_path.strip()
    ):
        raise NativeVariantContractError(
            f"{label}.path must be a non-empty relative path"
        )
    relative = Path(raw_path.replace("\\", "/"))
    if relative.is_absolute():
        raise NativeVariantContractError(
            f"{label}.path must be relative to its receipt"
        )
    expected = payload.get("sha256")
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        raise NativeVariantContractError(
            f"{label}.sha256 must be lowercase SHA-256"
        )
    resolved = (anchor / relative).resolve()
    if not _is_below(volume_root, resolved) or not resolved.is_file():
        raise NativeVariantContractError(
            f"{label} is absent or escapes the persistent volume"
        )
    return resolved, expected


class _StreamingRehasher:
    """Hash files one at a time; cache only explicitly shared payloads."""

    __slots__ = ("_shared", "bytes_hashed", "files_hashed")

    def __init__(self) -> None:
        self._shared: dict[Path, str] = {}
        self.bytes_hashed = 0
        self.files_hashed = 0

    def verify(
        self,
        payload: object,
        *,
        anchor: Path,
        volume_root: Path,
        label: str,
        shared: bool = False,
    ) -> Path:
        path, expected = _anchored_artifact_identity(
            payload,
            anchor=anchor,
            volume_root=volume_root,
            label=label,
        )
        actual = self._shared.get(path) if shared else None
        if actual is None:
            actual = _sha256(path)
            self.files_hashed += 1
            self.bytes_hashed += path.stat().st_size
            if shared:
                self._shared[path] = actual
        if actual != expected:
            raise NativeVariantContractError(
                f"{label} hash mismatch: expected {expected}, found {actual}"
            )
        return path


def _verify_authored_campaign(
    *,
    receipt_path: Path,
    campaign: Mapping[str, Any],
    plan_path: Path,
    volume_root: Path,
) -> dict[str, Any]:
    volume = volume_root.expanduser().resolve()
    receipt_file = receipt_path.expanduser().resolve()
    if not _is_below(volume, receipt_file) or not receipt_file.is_file():
        raise NativeVariantContractError(
            "authoring receipt must be an existing file below the volume"
        )
    receipt = _read_json(receipt_file, label="authoring receipt")
    if (
        receipt.get("schema_version") != AUTHORING_SCHEMA_VERSION
        or receipt.get("state") != "VARIANT_USD_AUTHORED"
        or receipt.get("simulation_count") != PORTFOLIO_SCENE_COUNT
        or receipt.get("manual_editor_review") != "required"
        or receipt.get("fire_simulation_status")
        != "blocked_pending_editor_review"
    ):
        raise NativeVariantContractError(
            "authoring receipt is not the blocked 20-scene result"
        )
    expected_plan = {
        "path": plan_path.relative_to(volume).as_posix(),
        "sha256": _sha256(plan_path),
    }
    if receipt.get("plan") != expected_plan:
        raise NativeVariantContractError(
            "authoring receipt is bound to another or changed plan"
        )
    plan_variants = campaign.get("variants")
    authored_variants = receipt.get("variants")
    if (
        not isinstance(plan_variants, list)
        or not isinstance(authored_variants, list)
        or len(plan_variants) != PORTFOLIO_SCENE_COUNT
        or len(authored_variants) != PORTFOLIO_SCENE_COUNT
    ):
        raise NativeVariantContractError(
            "plan/authoring receipt variant inventory is incomplete"
        )

    rehasher = _StreamingRehasher()
    author_root = receipt_file.parent
    plan_root = plan_path.parent
    terrain_paths: set[Path] = set()
    ground_paths: set[Path] = set()
    water_paths: set[Path] = set()
    root_count = 0
    build_receipt_count = 0
    detail_count = 0
    terrain_reference_count = 0
    ground_reference_count = 0
    water_reference_count = 0
    for sequence, (plan_record, authored_record) in enumerate(
        zip(plan_variants, authored_variants), start=1
    ):
        simulation_id = f"SIM-{sequence:02d}"
        if (
            not isinstance(plan_record, dict)
            or not isinstance(authored_record, dict)
            or plan_record.get("simulation_id") != simulation_id
            or authored_record.get("simulation_id") != simulation_id
            or authored_record.get("variant_id")
            != plan_record.get("variant_id")
            or authored_record.get("base_scene_id")
            != plan_record.get("base_scene_id")
            or authored_record.get("variant_index")
            != plan_record.get("variant_index")
            or authored_record.get("scene_kind") != "fictive_variant"
            or authored_record.get("fire_simulation_status")
            != "blocked_pending_editor_review"
        ):
            raise NativeVariantContractError(
                "authored simulation binding/order differs from the plan"
            )
        metadata_path = _resolve_plan_artifact(
            plan_record.get("metadata"),
            root=plan_root,
            label=f"{simulation_id} metadata",
        )
        metadata = _read_json(
            metadata_path, label=f"{simulation_id} metadata"
        )
        tiles = _load_variant_tiles(
            metadata, variant_dir=metadata_path.parent
        )
        authored_identity = _authored_identity_contract(
            metadata.get("identity_contract")
        )
        expected_actor_ids = list(_actor_usage_from_metadata(metadata))
        expected_actor_usage = {
            "group_id": SELECTED_ACTOR_GROUP_ID,
            "placed_actor_count": ACTORS_PER_SCENE,
            "selected_actor_ids": expected_actor_ids,
            "placeholder_substitution": False,
        }
        expected_environment_ids = list(
            _supplemental_environment_usage(metadata)
        )
        expected_environment_usage = {
            "group_id": SELECTED_ENVIRONMENT_GROUP_ID,
            "placed_asset_count": len(SELECTED_ENVIRONMENT_GROUP_IDS),
            "selected_environment_ids": expected_environment_ids,
            "additive_to_existing_minima": True,
            "placeholder_substitution": False,
        }
        artifacts = authored_record.get("artifacts")
        if (
            not isinstance(artifacts, dict)
            or artifacts.get("scene_kind") != "fictive_variant"
            or artifacts.get("streaming_tile_count") != 400
            or artifacts.get("object_lod_payload_count") != 1200
            or artifacts.get("monolithic_object_payloads") is not False
            or artifacts.get("identity_contract")
            != authored_identity
            or authored_record.get("identity_contract")
            != authored_identity
            or artifacts.get("actor_usage") != expected_actor_usage
            or authored_record.get("actor_usage") != expected_actor_usage
            or artifacts.get("supplemental_environment_usage")
            != expected_environment_usage
            or authored_record.get("supplemental_environment_usage")
            != expected_environment_usage
        ):
            raise NativeVariantContractError(
                f"{simulation_id} authored identity/streaming contract is stale"
            )
        simulation_root = author_root / simulation_id
        root_path = rehasher.verify(
            artifacts.get("root_usd"),
            anchor=simulation_root,
            volume_root=volume,
            label=f"{simulation_id}.root_usd",
        )
        root_count += 1
        build_receipt_path = rehasher.verify(
            artifacts.get("composer_build_receipt"),
            anchor=simulation_root,
            volume_root=volume,
            label=f"{simulation_id}.composer_build_receipt",
        )
        build_receipt_count += 1
        build = _read_json(
            build_receipt_path,
            label=f"{simulation_id} Composer build receipt",
        )
        if (
            build.get("schema_version") != 2
            or build.get("zone_id") != simulation_id
            or build.get("variant_id") != metadata.get("variant_id")
            or build.get("base_scene_id") != metadata.get("base_scene_id")
            or build.get("variant_index") != metadata.get("variant_index")
            or build.get("scene_kind") != "fictive_variant"
            or build.get("source_profile") != "full"
            or build.get("identity_contract")
            != authored_identity
            or build.get("route_topology")
            != metadata.get("route_topology")
            or build.get("water_contract")
            != metadata.get("water_contract")
            or build.get("fire_simulation_status")
            != "blocked_pending_editor_review"
        ):
            raise NativeVariantContractError(
                f"{simulation_id} Composer build receipt is stale"
            )
        build_root_path, build_root_sha = _anchored_artifact_identity(
            build.get("root_usd"),
            anchor=simulation_root,
            volume_root=volume,
            label=f"{simulation_id}.build.root_usd",
        )
        root_identity = _anchored_artifact_identity(
            artifacts.get("root_usd"),
            anchor=simulation_root,
            volume_root=volume,
            label=f"{simulation_id}.artifacts.root_usd",
        )
        if (build_root_path, build_root_sha) != root_identity or (
            root_path != build_root_path
        ):
            raise NativeVariantContractError(
                f"{simulation_id} root USD bindings diverge"
            )
        base_bindings = metadata.get("base_bindings")
        if not isinstance(base_bindings, dict):
            raise NativeVariantContractError(
                f"{simulation_id} has no verified base bindings"
            )
        expected_cameras = base_bindings.get("review_cameras")
        build_cameras = build.get("cameras")
        artifact_cameras = artifacts.get("review_cameras")
        if (
            not isinstance(expected_cameras, dict)
            or not isinstance(build_cameras, dict)
            or not isinstance(artifact_cameras, dict)
            or build_cameras.get("count") != expected_cameras.get("count")
            or artifact_cameras.get("count")
            != expected_cameras.get("count")
            or build_cameras.get("root_prim") != "/ReviewCameras"
        ):
            raise NativeVariantContractError(
                f"{simulation_id} review-camera contract is stale"
            )
        camera_path = rehasher.verify(
            build_cameras,
            anchor=simulation_root,
            volume_root=volume,
            label=f"{simulation_id}.cameras",
            shared=True,
        )
        artifact_camera_identity = _anchored_artifact_identity(
            artifact_cameras,
            anchor=simulation_root,
            volume_root=volume,
            label=f"{simulation_id}.artifact_cameras",
        )
        expected_camera_path = (
            volume / str(expected_cameras.get("path", ""))
        ).resolve()
        if (
            camera_path != expected_camera_path
            or build_cameras.get("sha256")
            != expected_cameras.get("sha256")
            or artifact_camera_identity
            != (camera_path, build_cameras.get("sha256"))
        ):
            raise NativeVariantContractError(
                f"{simulation_id} review-camera artifact changed binding"
            )
        expected_lock = base_bindings.get("asset_lock")
        build_lock = build.get("asset_lock")
        if (
            not isinstance(expected_lock, dict)
            or not isinstance(build_lock, dict)
            or build_lock.get("assets") != expected_lock.get("assets")
        ):
            raise NativeVariantContractError(
                f"{simulation_id} asset-lock inventory is stale"
            )
        lock_path = rehasher.verify(
            build_lock,
            anchor=simulation_root,
            volume_root=volume,
            label=f"{simulation_id}.asset_lock",
            shared=True,
        )
        expected_lock_path = (
            volume / str(expected_lock.get("path", ""))
        ).resolve()
        expected_manifest = base_bindings.get("shared_asset_manifest")
        build_manifest = build_lock.get("shared_manifest")
        if (
            lock_path != expected_lock_path
            or build_lock.get("sha256") != expected_lock.get("sha256")
            or not isinstance(expected_manifest, dict)
            or not isinstance(build_manifest, dict)
            or build_manifest.get("content_sha256")
            != base_bindings.get("asset_content_sha256")
        ):
            raise NativeVariantContractError(
                f"{simulation_id} shared asset lock changed binding"
            )
        manifest_path = rehasher.verify(
            build_manifest,
            anchor=simulation_root,
            volume_root=volume,
            label=f"{simulation_id}.shared_asset_manifest",
            shared=True,
        )
        if (
            manifest_path
            != (volume / str(expected_manifest.get("path", ""))).resolve()
            or build_manifest.get("sha256")
            != expected_manifest.get("sha256")
        ):
            raise NativeVariantContractError(
                f"{simulation_id} shared asset manifest changed binding"
            )

        terrain_records = build.get("payloads")
        build_coverage = build.get("tile_coverage")
        artifact_coverage = artifacts.get("tile_coverage")
        lod_inventory = artifacts.get("object_lod_payloads")
        if (
            not isinstance(terrain_records, list)
            or len(terrain_records) != 400
            or not isinstance(build_coverage, list)
            or len(build_coverage) != 400
            or not isinstance(artifact_coverage, list)
            or len(artifact_coverage) != 400
            or not isinstance(lod_inventory, dict)
            or set(lod_inventory) != set(LOD_LEVELS)
        ):
            raise NativeVariantContractError(
                f"{simulation_id} authored catalogs are incomplete"
            )
        detail_paths: set[Path] = set()
        build_detail_by_level = {
            "HERO": build.get("detail_payloads"),
            "MID": build.get("detail_mid_payloads"),
            "FAR": build.get("detail_far_payloads"),
        }
        for level in LOD_LEVELS:
            inventory = lod_inventory.get(level)
            build_inventory = build_detail_by_level[level]
            if (
                not isinstance(inventory, list)
                or len(inventory) != 400
                or not isinstance(build_inventory, list)
                or len(build_inventory) != 400
            ):
                raise NativeVariantContractError(
                    f"{simulation_id}.{level} does not contain 400 payloads"
                )
            for tile_index, (artifact, build_artifact) in enumerate(
                zip(inventory, build_inventory)
            ):
                artifact_identity = _anchored_artifact_identity(
                    artifact,
                    anchor=simulation_root,
                    volume_root=volume,
                    label=(
                        f"{simulation_id}.{level}[{tile_index}]"
                    ),
                )
                build_identity = _anchored_artifact_identity(
                    build_artifact,
                    anchor=simulation_root,
                    volume_root=volume,
                    label=(
                        f"{simulation_id}.build.{level}[{tile_index}]"
                    ),
                )
                if artifact_identity != build_identity:
                    raise NativeVariantContractError(
                        f"{simulation_id}.{level}[{tile_index}] bindings diverge"
                    )
                detail_path = rehasher.verify(
                    artifact,
                    anchor=simulation_root,
                    volume_root=volume,
                    label=(
                        f"{simulation_id}.{level}[{tile_index}]"
                    ),
                )
                if detail_path in detail_paths:
                    raise NativeVariantContractError(
                        f"{simulation_id} reuses one detail payload"
                    )
                detail_paths.add(detail_path)
                detail_count += 1
        if len(detail_paths) != 1200:
            raise NativeVariantContractError(
                f"{simulation_id} does not own 1200 distinct detail payloads"
            )

        lod0_count = 0
        for tile_index, (
            tile,
            terrain_artifact,
            coverage,
            authored_coverage,
        ) in enumerate(
            zip(
                tiles,
                terrain_records,
                build_coverage,
                artifact_coverage,
            )
        ):
            terrain_path = rehasher.verify(
                terrain_artifact,
                anchor=simulation_root,
                volume_root=volume,
                label=f"{simulation_id}.terrain[{tile_index}]",
                shared=True,
            )
            terrain_paths.add(terrain_path)
            terrain_reference_count += 1
            expected_terrain = (
                volume / str(tile["path"])
            ).resolve()
            if (
                terrain_path != expected_terrain
                or terrain_artifact.get("sha256") != tile.get("sha256")
                or not isinstance(coverage, dict)
                or coverage.get("tile_ref") != tile.get("tile_ref")
                or coverage.get("instance_namespace")
                != tile.get("instance_namespace")
                or coverage.get("terrain_payload")
                != terrain_artifact.get("path")
                or coverage.get("terrain_lods")
                != tile.get("terrain_lods")
                or coverage.get("collision_lods")
                != tile.get("collision_lods")
                or not isinstance(authored_coverage, dict)
                or authored_coverage.get("tile_ref")
                != tile.get("tile_ref")
                or authored_coverage.get("terrain_payload")
                != {
                    key: tile[key]
                    for key in (
                        "path",
                        "sha256",
                        "prim_path",
                        "isolated_content_roles",
                    )
                }
            ):
                raise NativeVariantContractError(
                    f"{simulation_id} terrain tile {tile_index} changed binding"
                )
            if "LOD0" in coverage["terrain_lods"]:
                lod0_count += 1
            coverage_lods = coverage.get("detail_lods")
            authored_lods = authored_coverage.get("detail_lods")
            if (
                not isinstance(coverage_lods, dict)
                or not isinstance(authored_lods, dict)
                or any(
                    coverage_lods.get(level)
                    != build_detail_by_level[level][tile_index].get("path")
                    or authored_lods.get(level)
                    != lod_inventory[level][tile_index]
                    for level in LOD_LEVELS
                )
            ):
                raise NativeVariantContractError(
                    f"{simulation_id} detail tile {tile_index} changed binding"
                )
        if lod0_count <= 0:
            raise NativeVariantContractError(
                f"{simulation_id} exposes no real review-camera LOD0"
            )

        water_catalog = build.get("water_payloads")
        expected_water = metadata.get("base_bindings", {}).get(
            "water_payloads"
        )
        if (
            not isinstance(water_catalog, list)
            or not isinstance(expected_water, list)
            or not water_catalog
            or len(water_catalog) != len(expected_water)
        ):
            raise NativeVariantContractError(
                f"{simulation_id} isolated water catalog is incomplete"
            )
        for water_index, (actual, expected) in enumerate(
            zip(water_catalog, expected_water)
        ):
            water_path = rehasher.verify(
                actual,
                anchor=simulation_root,
                volume_root=volume,
                label=f"{simulation_id}.water[{water_index}]",
                shared=True,
            )
            expected_path = (
                volume / str(expected.get("path", ""))
            ).resolve()
            if (
                water_path != expected_path
                or actual.get("sha256") != expected.get("sha256")
                or actual.get("prim_path") != expected.get("prim_path")
                or actual.get("isolated_content_roles")
                != expected.get("isolated_content_roles")
            ):
                raise NativeVariantContractError(
                    f"{simulation_id} water payload {water_index} changed binding"
                )
            water_paths.add(water_path)
            water_reference_count += 1

        ground = build.get("ground_material")
        expected_ground = base_bindings.get("ground_material")
        if not isinstance(ground, dict) or not isinstance(
            expected_ground, dict
        ):
            raise NativeVariantContractError(
                f"{simulation_id} ground material contract is absent"
            )
        ground_tiles = ground.get("tile_material_payloads")
        expected_ground_tiles = expected_ground.get(
            "tile_material_payloads"
        )
        expected_tile_refs = {str(tile["tile_ref"]) for tile in tiles}
        if (
            ground.get("topology")
            != "payload_tiled_materials_shared_pbr_library"
            or ground.get("topology") != expected_ground.get("topology")
            or ground.get("binding_scope")
            != "per_terrain_tile_stronger_than_descendants"
            or not isinstance(ground_tiles, list)
            or not isinstance(expected_ground_tiles, list)
            or len(ground_tiles) != 400
            or len(expected_ground_tiles) != 400
            or {item.get("tile_id") for item in ground_tiles if isinstance(item, dict)}
            != expected_tile_refs
        ):
            raise NativeVariantContractError(
                f"{simulation_id} PBR ground mapping is incomplete"
            )
        ground_index_path = rehasher.verify(
            ground.get("index"),
            anchor=simulation_root,
            volume_root=volume,
            label=f"{simulation_id}.ground.index",
            shared=True,
        )
        if (
            ground_index_path
            != (
                volume / str(expected_ground.get("path", ""))
            ).resolve()
            or ground.get("index", {}).get("sha256")
            != expected_ground.get("sha256")
            or ground.get("index", {}).get("prim_path")
            != expected_ground.get("prim_path")
        ):
            raise NativeVariantContractError(
                f"{simulation_id} ground index changed binding"
            )
        for tile_index, (ground_artifact, expected_artifact) in enumerate(
            zip(ground_tiles, expected_ground_tiles)
        ):
            path = rehasher.verify(
                ground_artifact,
                anchor=simulation_root,
                volume_root=volume,
                label=f"{simulation_id}.ground[{tile_index}]",
                shared=True,
            )
            if (
                path
                != (
                    volume / str(expected_artifact.get("path", ""))
                ).resolve()
                or ground_artifact.get("tile_id")
                != expected_artifact.get("tile_id")
                or ground_artifact.get("tile_bounds_m")
                != expected_artifact.get("tile_bounds_m")
                or ground_artifact.get("sha256")
                != expected_artifact.get("sha256")
                or ground_artifact.get("prim_path")
                != expected_artifact.get("prim_path")
            ):
                raise NativeVariantContractError(
                    f"{simulation_id} ground tile {tile_index} changed binding"
                )
            ground_paths.add(path)
            ground_reference_count += 1
        terrain_layer = (
            build.get("layers", {}).get("terrain")
            if isinstance(build.get("layers"), dict)
            else None
        )
        roads_layer = (
            build.get("layers", {}).get("roads")
            if isinstance(build.get("layers"), dict)
            else None
        )
        hydrology_layer = (
            build.get("layers", {}).get("hydrology")
            if isinstance(build.get("layers"), dict)
            else None
        )
        actor_layer = (
            build.get("layers", {}).get("actors")
            if isinstance(build.get("layers"), dict)
            else None
        )
        environment_layer = (
            build.get("layers", {}).get("supplemental_environment")
            if isinstance(build.get("layers"), dict)
            else None
        )
        if (
            not isinstance(terrain_layer, dict)
            or terrain_layer.get("prim_count") != 400
            or terrain_layer.get("ground_material_payload_count") != 400
            or terrain_layer.get("global_ground_material_binding") is not False
            or not isinstance(roads_layer, dict)
            or roads_layer.get("prim_count") != 0
            or roads_layer.get("source_feature_count")
            != metadata.get("source_route_count")
            or roads_layer.get("route_fragment_count")
            != metadata.get("route_fragment_count")
            or roads_layer.get("visible_representation")
            != "orthophoto_derived_terrain_material"
            or roads_layer.get("geometry_authoring") != "disabled"
            or roads_layer.get("asset_dependencies") != []
            or not isinstance(hydrology_layer, dict)
            or hydrology_layer.get("prim_count")
            != metadata.get("hydrology_fragment_count")
            or hydrology_layer.get("source_feature_count")
            != len(
                metadata.get("water_contract", {}).get(
                    "source_feature_ids", []
                )
            )
            or not isinstance(actor_layer, dict)
            or actor_layer.get("selected_actor_ids") != expected_actor_ids
            or actor_layer.get("prim_count") != ACTORS_PER_SCENE
            or not isinstance(environment_layer, dict)
            or environment_layer.get("selected_environment_ids")
            != expected_environment_ids
            or environment_layer.get("prim_count")
            != len(SELECTED_ENVIRONMENT_GROUP_IDS)
        ):
            raise NativeVariantContractError(
                f"{simulation_id} terrain/PBR/fragment layer summary is stale"
            )
        for layer_name, layer in (
            ("roads", roads_layer),
            ("hydrology", hydrology_layer),
        ):
            for metric_name in ("vertices_by_lod", "faces_by_lod"):
                metrics = layer.get(metric_name)
                if (
                    not isinstance(metrics, dict)
                    or set(metrics) != set(LOD_LEVELS)
                    or any(
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value < 0
                        for value in metrics.values()
                    )
                    or len(set(metrics.values())) != 1
                ):
                    raise NativeVariantContractError(
                        f"{simulation_id} {layer_name} {metric_name} "
                        "does not measure the bounded same-geometry LOD policy"
                    )
        detail_streaming = build["layers"].get("detail_streaming")
        if (
            not isinstance(detail_streaming, dict)
            or detail_streaming.get("network_geometry_policy")
            != NETWORK_GEOMETRY_POLICY
            or detail_streaming.get("network_vertex_budget_per_tile")
            != 262_144
        ):
            raise NativeVariantContractError(
                f"{simulation_id} network LOD budget contract is absent"
            )
        del tiles

    return {
        "state": "VARIANT_CAMPAIGN_VERIFIED",
        "plan_sha256": _sha256(plan_path),
        "authoring_receipt_sha256": _sha256(receipt_file),
        "layout_count": BASE_SCENE_COUNT,
        "simulation_count": PORTFOLIO_SCENE_COUNT,
        "root_usd_rehashed": root_count,
        "build_receipts_rehashed": build_receipt_count,
        "terrain_payload_references_verified": terrain_reference_count,
        "terrain_payload_unique_files_rehashed": len(terrain_paths),
        "object_lod_payloads_rehashed": detail_count,
        "ground_material_references_verified": ground_reference_count,
        "ground_material_unique_files_rehashed": len(ground_paths),
        "water_payload_references_verified": water_reference_count,
        "water_payload_unique_files_rehashed": len(water_paths),
        "identity_contracts_verified": PORTFOLIO_SCENE_COUNT,
        "hash_operations": rehasher.files_hashed,
        "bytes_hashed": rehasher.bytes_hashed,
        "memory_contract": {
            "layout_scenes_live": 1,
            "variant_metadata_live": 1,
            "tile_object_files_live": 1,
            "shared_hash_cache": (
                "terrain_ground_and_shared_metadata_paths_only"
            ),
        },
        "manual_editor_review": "required",
        "fire_simulation_status": "blocked_pending_editor_review",
    }


def verify_variant_campaign(
    *,
    plan_path: Path,
    layout_paths: Sequence[Path],
    volume_root: Path,
    authoring_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Import-safe resume verifier for a plan and optional authored campaign."""

    volume = volume_root.expanduser().resolve()
    plan = plan_path.expanduser().resolve()
    campaign = _verify_plan_against_layouts(
        plan_path=plan,
        layout_paths=layout_paths,
        volume_root=volume,
    )
    if authoring_receipt_path is None:
        return {
            "state": "VARIANT_PLAN_VERIFIED",
            "plan_sha256": _sha256(plan),
            "layout_count": BASE_SCENE_COUNT,
            "simulation_count": PORTFOLIO_SCENE_COUNT,
            "plan_tile_streams_verified": (
                PORTFOLIO_SCENE_COUNT * 400
            ),
            "authoring_status": "planned_not_authored",
            "manual_editor_review": "not_started",
            "fire_simulation_status": "blocked_pending_editor_review",
        }
    return _verify_authored_campaign(
        receipt_path=authoring_receipt_path,
        campaign=campaign,
        plan_path=plan,
        volume_root=volume,
    )


def _main_plan(args: argparse.Namespace) -> int:
    constraints = (
        constraints_from_json(Path(args.constraints))
        if args.constraints
        else None
    )
    result = prepare_variant_campaign(
        layout_paths=[Path(path) for path in args.layout],
        volume_root=Path(args.volume_root),
        output_root=Path(args.output),
        master_seed=args.master_seed,
        constraints=constraints,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _main_author(args: argparse.Namespace) -> int:
    result = author_variant_campaign(
        plan_path=Path(args.plan),
        volume_root=Path(args.volume_root),
        output_root=Path(args.output),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _main_verify(args: argparse.Namespace) -> int:
    result = verify_variant_campaign(
        plan_path=Path(args.plan),
        layout_paths=[Path(path) for path in args.layout],
        volume_root=Path(args.volume_root),
        authoring_receipt_path=(
            Path(args.authoring_receipt)
            if args.authoring_receipt
            else None
        ),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan and author the fixed four-by-five native USD portfolio"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--volume-root", required=True)
    plan_parser.add_argument("--layout", action="append", required=True)
    plan_parser.add_argument(
        "--constraints",
        help=(
            "optional exact cross-check; biology and numeric constraints are "
            "normally derived from each composition-source"
        ),
    )
    plan_parser.add_argument("--master-seed", type=int, required=True)
    plan_parser.add_argument("--output", required=True)
    plan_parser.set_defaults(handler=_main_plan)
    author_parser = subparsers.add_parser("author")
    author_parser.add_argument("--volume-root", required=True)
    author_parser.add_argument("--plan", required=True)
    author_parser.add_argument("--output", required=True)
    author_parser.set_defaults(handler=_main_author)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--volume-root", required=True)
    verify_parser.add_argument("--plan", required=True)
    verify_parser.add_argument("--layout", action="append", required=True)
    verify_parser.add_argument(
        "--authoring-receipt",
        help=(
            "optional completed authoring-receipt.json; omit to verify only "
            "the resumable plan"
        ),
    )
    verify_parser.set_defaults(handler=_main_verify)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command in {"plan", "verify"} and len(args.layout) != BASE_SCENE_COUNT:
        parser.error(
            f"{args.command} requires exactly {BASE_SCENE_COUNT} "
            "--layout arguments"
        )
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

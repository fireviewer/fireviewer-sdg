"""Portable and native object-free PBR terrain composition for scene variants.

Planning and validation verify an already installed, hash-locked material
bundle without importing USD or creating raster data.  The explicit
``author-native`` path then lazily loads Kit/OpenUSD, MaterialX, GDAL and NumPy
on the pod to author one bounded material graph and one lightweight payload
per terrain tile.  Every graph shares the same seven locked PBR assets and
only its own classification/relief inputs.  Spatial sources are never reused
as colour imagery.

The contract has three important properties:

* all seven approved material roles are present and byte-locked;
* UVs are projected from one world-metric origin, so adjacent tiles cannot
  restart their texture coordinates at a seam;
* forest, water, roads and artificial ground only drive blend weights.  Raw
  orthophotos, baked objects and per-tile diffuse imagery are not accepted.

No ``pxr`` module is imported at module-import time, which keeps planning and
control-plane validation available before the pinned Kit runtime is started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol, Sequence

from fireviewer_sdg.asset_bundle import (
    INSTALL_MARKER,
    MATERIAL_SUFFIXES,
    MIN_PBR_TEXTURE_DIMENSION,
    PBR_MATERIAL_ROLES,
    PBR_OPTIONAL_TEXTURES,
    PBR_REQUIRED_TEXTURES,
    TEXTURE_SUFFIXES,
)


SCHEMA_VERSION = 2
ALGORITHM_ID = "fireviewer-object-free-terrain-pbr-v1"
COMPOSITE_AUTHORING_SCHEMA_VERSION = 2
NATIVE_COMPOSITE_INSPECTOR_ID = (
    "fireviewer-native-payload-ground-inspector-v2"
)
NATIVE_GROUND_STATE = "COMPOSITE_GROUND_MATERIAL_NATIVE_VALIDATED"
NATIVE_PREPARATION_STATE = "TERRAIN_PBR_NATIVE_REQUEST_PREPARED"
NATIVE_ZONE_TILE_COUNT = 400
NATIVE_ZONE_CRS = "EPSG:2154"
NATIVE_ZONE_VERTICAL_DATUM = "IGN69"
VECTOR_CLASSIFICATION_RESOLUTION_M = 0.5
NATIVE_ANALYSIS_SUBDIVISIONS = 2
EVIDENCE_SEMANTICS = (
    "elevation",
    "forest",
    "water",
    "roads",
    "artificial_ground",
)
CLASSIFICATION_SEMANTICS = EVIDENCE_SEMANTICS[1:]
MASK_PRIORITY = ("water", "roads", "artificial_ground", "forest")
MAX_DERIVED_TILE_EDGE_PIXELS = 4_096
MAX_REACHABLE_SHADER_PRIMS_PER_TILE = 256
MAX_SPATIAL_IMAGE_NODES_PER_TILE = 6
MASK_EDGE_SAMPLE_COUNT = 9
MASK_EDGE_CONTINUITY_TOLERANCE = 0.02
_TRANSITION_WIDTH_M = {
    "forest": 2.50,
    "water": 0.75,
    "roads": 0.45,
    "artificial_ground": 0.80,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ALLOWED_RASTER_SUFFIXES = frozenset(
    {".tif", ".tiff", ".png", ".exr", ".vrt", ".npy"}
)
_ALLOWED_VECTOR_SUFFIXES = frozenset(
    {".gpkg", ".geojson", ".json", ".fgb", ".parquet"}
)
_SOURCE_KINDS = {
    "elevation": frozenset({"heightfield"}),
    "forest": frozenset({"classified_mask", "classified_vector"}),
    "water": frozenset({"classified_mask", "classified_vector"}),
    "roads": frozenset({"classified_mask", "classified_vector"}),
    "artificial_ground": frozenset(
        {"classified_mask", "classified_vector"}
    ),
}
_USAGE_BY_SEMANTIC = {
    "elevation": "height_only",
    "forest": "blend_weights_only",
    "water": "blend_weights_only",
    "roads": "blend_weights_only",
    "artificial_ground": "blend_weights_only",
}
_USD_TEXTURE_COLOR_SPACES = {
    "srgb": "srgb_texture",
    "raw": "none",
    "linear": "none",
}


class TerrainPbrContractError(ValueError):
    """Raised when a terrain plan cannot be proved portable and object-free."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
    raw = str(value)
    digest = raw.strip().lower()
    if raw != digest:
        raise TerrainPbrContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    if not _SHA256.fullmatch(digest):
        raise TerrainPbrContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return digest


def _require_stable_id(value: str, *, label: str) -> str:
    identifier = str(value).strip()
    if not _STABLE_ID.fullmatch(identifier):
        raise TerrainPbrContractError(f"{label} has an invalid stable identifier")
    return identifier


def _safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    raw = str(value)
    if (
        not raw
        or raw != raw.strip()
        or "\\" in raw
        or "\x00" in raw
        or any(ord(character) < 32 for character in raw)
    ):
        raise TerrainPbrContractError(
            f"{label} must be a normalized relative POSIX path"
        )
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] in {"", "."}
        or ".." in path.parts
        or path.as_posix() != raw
    ):
        raise TerrainPbrContractError(f"{label} escapes its declared root")
    return path


def _inside(root: Path, candidate: Path) -> bool:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    return (
        resolved_candidate == resolved_root
        or resolved_root in resolved_candidate.parents
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TerrainPbrContractError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise TerrainPbrContractError(f"{label} must be a JSON object")
    return value


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_locked_file(
    *,
    root: Path,
    parent: Path,
    record: object,
    label: str,
    suffixes: frozenset[str],
) -> tuple[Path, PurePosixPath, str, int]:
    if not isinstance(record, dict):
        raise TerrainPbrContractError(f"{label} lock must be an object")
    relative = _safe_relative_path(str(record.get("path", "")), label=label)
    path = parent.joinpath(*relative.parts)
    if (
        not _inside(root, path)
        or not path.is_file()
        or path.is_symlink()
        or path.suffix.casefold() not in suffixes
    ):
        raise TerrainPbrContractError(
            f"{label} must resolve to a supported regular file inside its root"
        )
    actual_sha256 = _sha256_file(path)
    actual_size = path.stat().st_size
    expected_sha256 = _require_sha256(
        str(record.get("sha256", "")),
        label=f"{label} lock",
    )
    if expected_sha256 != actual_sha256 or record.get("size_bytes") != actual_size:
        raise TerrainPbrContractError(
            f"{label} SHA-256 or size lock does not match"
        )
    return path, relative, actual_sha256, actual_size


@dataclass(frozen=True, slots=True)
class Bounds2d:
    min_x_m: float
    min_y_m: float
    max_x_m: float
    max_y_m: float

    def __post_init__(self) -> None:
        values = (
            self.min_x_m,
            self.min_y_m,
            self.max_x_m,
            self.max_y_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise TerrainPbrContractError("bounds must contain finite metres")
        if self.max_x_m <= self.min_x_m or self.max_y_m <= self.min_y_m:
            raise TerrainPbrContractError("bounds must have a positive area")

    @property
    def area_m2(self) -> float:
        return (self.max_x_m - self.min_x_m) * (
            self.max_y_m - self.min_y_m
        )

    def contains(self, other: "Bounds2d", *, tolerance_m: float = 1.0e-6) -> bool:
        return (
            other.min_x_m >= self.min_x_m - tolerance_m
            and other.min_y_m >= self.min_y_m - tolerance_m
            and other.max_x_m <= self.max_x_m + tolerance_m
            and other.max_y_m <= self.max_y_m + tolerance_m
        )

    def overlap_area_m2(self, other: "Bounds2d") -> float:
        width = max(
            0.0,
            min(self.max_x_m, other.max_x_m)
            - max(self.min_x_m, other.min_x_m),
        )
        height = max(
            0.0,
            min(self.max_y_m, other.max_y_m)
            - max(self.min_y_m, other.min_y_m),
        )
        return width * height

    def as_list(self) -> list[float]:
        return [
            self.min_x_m,
            self.min_y_m,
            self.max_x_m,
            self.max_y_m,
        ]


@dataclass(frozen=True, slots=True)
class FileLock:
    """Portable immutable reference resolved below a caller-owned root."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _safe_relative_path(self.path, label="file lock path")
        _require_sha256(self.sha256, label="file lock")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise TerrainPbrContractError(
                "file lock size_bytes must be a positive integer"
            )


@dataclass(frozen=True, slots=True)
class MaterialTexture:
    role: str
    lock: FileLock
    width_px: int
    height_px: int
    color_space: str

    def __post_init__(self) -> None:
        if self.role not in (*PBR_REQUIRED_TEXTURES, *PBR_OPTIONAL_TEXTURES):
            raise TerrainPbrContractError("unsupported material texture role")
        if (
            isinstance(self.width_px, bool)
            or isinstance(self.height_px, bool)
            or not isinstance(self.width_px, int)
            or not isinstance(self.height_px, int)
            or self.width_px < MIN_PBR_TEXTURE_DIMENSION
            or self.height_px < MIN_PBR_TEXTURE_DIMENSION
            or self.width_px != self.height_px
        ):
            raise TerrainPbrContractError(
                "PBR textures must be square and meet the bundle resolution floor"
            )
        expected = {"srgb"} if self.role == "base_color" else {"raw", "linear"}
        if self.color_space.casefold() not in expected:
            raise TerrainPbrContractError(
                f"{self.role} has an invalid texture color space"
            )


@dataclass(frozen=True, slots=True)
class LockedMaterial:
    role: str
    material_id: str
    material_file: FileLock
    material_prim_path: str
    metres_per_uv_tile: float
    textures: tuple[MaterialTexture, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "textures", tuple(self.textures))
        if self.role not in PBR_MATERIAL_ROLES:
            raise TerrainPbrContractError("unsupported terrain material role")
        _require_stable_id(self.material_id, label=f"{self.role} material")
        if (
            not self.material_prim_path.startswith("/")
            or "//" in self.material_prim_path
            or ".." in self.material_prim_path.split("/")
        ):
            raise TerrainPbrContractError(
                f"{self.role} material prim path must be absolute and normalized"
            )
        if (
            not math.isfinite(self.metres_per_uv_tile)
            or not 0.1 <= self.metres_per_uv_tile <= 100.0
        ):
            raise TerrainPbrContractError(
                f"{self.role} material has an implausible metric repeat"
            )
        texture_roles = tuple(texture.role for texture in self.textures)
        if not set(PBR_REQUIRED_TEXTURES).issubset(texture_roles):
            raise TerrainPbrContractError(
                f"{self.role} material lacks a required texture"
            )
        if len(texture_roles) != len(set(texture_roles)):
            raise TerrainPbrContractError(
                f"{self.role} material repeats a texture role"
            )
        if len(
            {
                (texture.width_px, texture.height_px)
                for texture in self.textures
            }
        ) != 1:
            raise TerrainPbrContractError(
                f"{self.role} material texture dimensions must match"
            )


@dataclass(frozen=True, slots=True)
class LockedMaterialLibrary:
    bundle_sha256: str
    manifest_path: str
    manifest_sha256: str
    materials: tuple[LockedMaterial, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "materials", tuple(self.materials))
        _require_sha256(self.bundle_sha256, label="bundle")
        _safe_relative_path(self.manifest_path, label="bundle manifest")
        _require_sha256(self.manifest_sha256, label="bundle manifest")
        roles = tuple(material.role for material in self.materials)
        if roles != PBR_MATERIAL_ROLES:
            raise TerrainPbrContractError(
                "material library must contain the seven roles in contract order"
            )

    def by_role(self, role: str) -> LockedMaterial:
        for material in self.materials:
            if material.role == role:
                return material
        raise KeyError(role)


@dataclass(frozen=True, slots=True)
class SpatialEvidence:
    """A measured height source or classified spatial layer.

    ``content_kind`` deliberately excludes colour imagery.  Classification
    inputs may have been derived upstream from imagery, but only their class
    masks/geometries can be consumed by this contract.
    """

    stable_id: str
    semantic: str
    content_kind: str
    usage: str
    lock: FileLock
    crs: str
    bounds: Bounds2d
    resolution_m: float
    feature_count: int | None = None

    def __post_init__(self) -> None:
        _require_stable_id(self.stable_id, label="spatial evidence")
        if self.semantic not in EVIDENCE_SEMANTICS:
            raise TerrainPbrContractError("unsupported evidence semantic")
        if self.content_kind not in _SOURCE_KINDS[self.semantic]:
            raise TerrainPbrContractError(
                f"{self.semantic} evidence cannot consume {self.content_kind}; "
                "raw orthophotos and colour imagery are forbidden"
            )
        if self.usage != _USAGE_BY_SEMANTIC[self.semantic]:
            raise TerrainPbrContractError(
                f"{self.semantic} evidence has an unsafe appearance usage"
            )
        if not self.crs.strip() or self.crs != self.crs.strip():
            raise TerrainPbrContractError("spatial evidence requires a CRS")
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0.0:
            raise TerrainPbrContractError(
                "spatial evidence resolution must be positive metres"
            )
        if self.content_kind == "classified_vector":
            if (
                isinstance(self.feature_count, bool)
                or self.feature_count is None
                or not isinstance(self.feature_count, int)
                or self.feature_count <= 0
            ):
                raise TerrainPbrContractError(
                    "classified vectors require a positive feature count"
                )
        elif self.feature_count is not None:
            raise TerrainPbrContractError(
                "raster evidence must not claim a vector feature count"
            )


@dataclass(frozen=True, slots=True)
class TerrainSubzoneEvidence:
    stable_id: str
    bounds: Bounds2d
    mean_elevation_m: float
    mean_slope_degrees: float
    roughness_m: float
    coverage: Mapping[str, float]
    evidence_ids: frozenset[str]

    def __post_init__(self) -> None:
        _require_stable_id(self.stable_id, label="terrain subzone")
        if not all(
            math.isfinite(value)
            for value in (
                self.mean_elevation_m,
                self.mean_slope_degrees,
                self.roughness_m,
            )
        ):
            raise TerrainPbrContractError(
                "terrain subzone relief metrics must be finite"
            )
        if not 0.0 <= self.mean_slope_degrees <= 90.0:
            raise TerrainPbrContractError(
                "terrain subzone slope must be within 0..90 degrees"
            )
        if self.roughness_m < 0.0:
            raise TerrainPbrContractError(
                "terrain subzone roughness cannot be negative"
            )
        if not isinstance(self.coverage, Mapping):
            raise TerrainPbrContractError(
                "terrain subzone coverage must be a mapping"
            )
        normalized_coverage: dict[str, float] = {}
        for key, raw_value in self.coverage.items():
            if isinstance(raw_value, bool):
                raise TerrainPbrContractError(
                    "terrain subzone coverage cannot contain booleans"
                )
            try:
                normalized_coverage[str(key)] = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise TerrainPbrContractError(
                    "terrain subzone coverage must contain numeric fractions"
                ) from exc
        frozen_coverage = MappingProxyType(normalized_coverage)
        object.__setattr__(self, "coverage", frozen_coverage)
        if set(frozen_coverage) != set(CLASSIFICATION_SEMANTICS):
            raise TerrainPbrContractError(
                "terrain subzone requires forest, water, roads and "
                "artificial-ground coverage"
            )
        for semantic, value in frozen_coverage.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise TerrainPbrContractError(
                    f"{semantic} subzone coverage must be within 0..1"
                )
        if len(self.evidence_ids) != len(EVIDENCE_SEMANTICS):
            raise TerrainPbrContractError(
                "terrain subzone must cite all five spatial evidence layers"
            )


@dataclass(frozen=True, slots=True)
class TerrainTileEvidence:
    stable_id: str
    bounds: Bounds2d
    evidence: tuple[SpatialEvidence, ...]
    subzones: tuple[TerrainSubzoneEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "subzones", tuple(self.subzones))
        _require_stable_id(self.stable_id, label="terrain tile")
        if not self.subzones:
            raise TerrainPbrContractError(
                "terrain tile requires analysed subzones"
            )
        semantics = tuple(source.semantic for source in self.evidence)
        if set(semantics) != set(EVIDENCE_SEMANTICS) or len(semantics) != len(
            EVIDENCE_SEMANTICS
        ):
            raise TerrainPbrContractError(
                "terrain tile requires exactly elevation, forest, water, roads "
                "and artificial-ground evidence"
            )
        source_ids = {source.stable_id for source in self.evidence}
        if len(source_ids) != len(self.evidence):
            raise TerrainPbrContractError(
                "terrain tile evidence identifiers must be unique"
            )
        crs = {source.crs for source in self.evidence}
        if len(crs) != 1:
            raise TerrainPbrContractError(
                "terrain tile evidence must use one coherent CRS"
            )
        for source in self.evidence:
            if not source.bounds.contains(self.bounds):
                raise TerrainPbrContractError(
                    f"{source.semantic} evidence does not cover its terrain tile"
                )
        subzone_ids: set[str] = set()
        for subzone in self.subzones:
            if subzone.stable_id in subzone_ids:
                raise TerrainPbrContractError(
                    "terrain subzone identifiers must be unique within a tile"
                )
            subzone_ids.add(subzone.stable_id)
            if not self.bounds.contains(subzone.bounds):
                raise TerrainPbrContractError(
                    "terrain subzone escapes its terrain tile"
                )
            if subzone.evidence_ids != source_ids:
                raise TerrainPbrContractError(
                    "terrain subzone evidence set is incomplete or stale"
                )
        subzones = list(self.subzones)
        for index, left in enumerate(subzones):
            for right in subzones[index + 1 :]:
                if left.bounds.overlap_area_m2(right.bounds) > 1.0e-5:
                    raise TerrainPbrContractError(
                        "terrain subzones must not overlap"
                    )
        covered_area = math.fsum(subzone.bounds.area_m2 for subzone in subzones)
        tolerance = max(1.0e-4, self.bounds.area_m2 * 1.0e-9)
        if not math.isclose(
            covered_area,
            self.bounds.area_m2,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise TerrainPbrContractError(
                "terrain subzones must partition their complete tile"
            )


@dataclass(frozen=True, slots=True)
class MetricUv:
    role: str
    origin_x_m: float
    origin_y_m: float
    metres_per_uv_tile: float

    def uv_at(self, world_x_m: float, world_y_m: float) -> tuple[float, float]:
        if not math.isfinite(world_x_m) or not math.isfinite(world_y_m):
            raise TerrainPbrContractError("UV query must contain finite metres")
        return (
            (world_x_m - self.origin_x_m) / self.metres_per_uv_tile,
            (world_y_m - self.origin_y_m) / self.metres_per_uv_tile,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "projection": "world_xy_metric",
            "origin_m": [self.origin_x_m, self.origin_y_m],
            "metres_per_uv_tile": self.metres_per_uv_tile,
            "tile_local_reset": False,
        }


@dataclass(frozen=True, slots=True)
class SubzoneMaterialPlan:
    stable_id: str
    bounds: Bounds2d
    evidence_ids: tuple[str, ...]
    mean_elevation_m: float
    mean_slope_degrees: float
    roughness_m: float
    coverage: tuple[tuple[str, float], ...]
    mean_weights: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "subzone_id": self.stable_id,
            "bounds_m": self.bounds.as_list(),
            "evidence_ids": list(self.evidence_ids),
            "relief_analysis": {
                "mean_elevation_m": self.mean_elevation_m,
                "mean_slope_degrees": self.mean_slope_degrees,
                "roughness_m": self.roughness_m,
            },
            "classification_coverage": {
                semantic: value for semantic, value in self.coverage
            },
            "mean_weights": {
                role: weight for role, weight in self.mean_weights
            },
            "mean_weights_sum": math.fsum(
                weight for _role, weight in self.mean_weights
            ),
            "authoring_note": (
                "mean weights are QA summaries only; native authoring evaluates "
                "the locked masks, vectors and relief graph spatially"
            ),
        }


@dataclass(frozen=True, slots=True)
class TileMaterialPlan:
    stable_id: str
    bounds: Bounds2d
    evidence: tuple[SpatialEvidence, ...]
    subzones: tuple[SubzoneMaterialPlan, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "tile_id": self.stable_id,
            "bounds_m": self.bounds.as_list(),
            "evidence": [
                {
                    "stable_id": source.stable_id,
                    "semantic": source.semantic,
                    "content_kind": source.content_kind,
                    "usage": source.usage,
                    "path": source.lock.path,
                    "sha256": source.lock.sha256,
                    "size_bytes": source.lock.size_bytes,
                    "crs": source.crs,
                    "bounds_m": source.bounds.as_list(),
                    "resolution_m": source.resolution_m,
                    **(
                        {"feature_count": source.feature_count}
                        if source.feature_count is not None
                        else {}
                    ),
                }
                for source in self.evidence
            ],
            "subzones": [subzone.as_dict() for subzone in self.subzones],
        }


@dataclass(frozen=True, slots=True)
class TerrainPbrPlan:
    scene_id: str
    material_library: LockedMaterialLibrary
    metric_uv: tuple[MetricUv, ...]
    tiles: tuple[TileMaterialPlan, ...]
    evidence_inventory_sha256: str
    scene_origin_source_m: tuple[float, float]

    def __post_init__(self) -> None:
        _require_stable_id(self.scene_id, label="terrain PBR scene")
        object.__setattr__(self, "metric_uv", tuple(self.metric_uv))
        object.__setattr__(self, "tiles", tuple(self.tiles))
        object.__setattr__(
            self,
            "scene_origin_source_m",
            tuple(self.scene_origin_source_m),
        )
        if (
            len(self.scene_origin_source_m) != 2
            or any(
                not math.isfinite(value)
                for value in self.scene_origin_source_m
            )
        ):
            raise TerrainPbrContractError(
                "scene origin must contain two finite source-CRS metres"
            )
        _require_sha256(
            self.evidence_inventory_sha256,
            label="terrain evidence inventory",
        )

    def uv_for(
        self,
        role: str,
        world_x_m: float,
        world_y_m: float,
    ) -> tuple[float, float]:
        for contract in self.metric_uv:
            if contract.role == role:
                return contract.uv_at(world_x_m, world_y_m)
        raise KeyError(role)

    def as_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "algorithm_id": ALGORITHM_ID,
            "scene_id": self.scene_id,
            "appearance_contract": {
                "ground_kind": "object_free_pbr",
                "raw_orthophoto_allowed": False,
                "baked_object_imagery_allowed": False,
                "spatial_evidence_usage": "blend_weights_and_height_only",
                "generated_texture_assets": [],
                "authoring_mode": "runtime_evidence_driven",
                "macro_variation": {
                    "method": "rotated_world_metric_secondary_scale",
                    "repeat_m": 73.0,
                    "primary_weight": 0.82,
                    "macro_weight": 0.18,
                },
                "steep_slope_mapping": {
                    "method": "relief_weighted_xy_xz_yz_projection",
                    "material_roles": ["rock"],
                },
            },
            "coordinate_frames": {
                "source": "declared_metric_evidence_crs",
                "scene": "source_minus_scene_origin",
                "scene_origin_source_m": list(
                    self.scene_origin_source_m
                ),
            },
            "material_manifest": {
                "path": self.material_library.manifest_path,
                "sha256": self.material_library.manifest_sha256,
                "bundle_sha256": self.material_library.bundle_sha256,
            },
            "materials": {
                material.role: {
                    "material_id": material.material_id,
                    "material_file": {
                        "path": material.material_file.path,
                        "sha256": material.material_file.sha256,
                        "size_bytes": material.material_file.size_bytes,
                    },
                    "material_prim_path": material.material_prim_path,
                    "metres_per_uv_tile": material.metres_per_uv_tile,
                    "textures": {
                        texture.role: {
                            "path": texture.lock.path,
                            "sha256": texture.lock.sha256,
                            "size_bytes": texture.lock.size_bytes,
                            "width_px": texture.width_px,
                            "height_px": texture.height_px,
                            "color_space": texture.color_space,
                        }
                        for texture in material.textures
                    },
                }
                for material in self.material_library.materials
            },
            "metric_uv": {
                contract.role: contract.as_dict() for contract in self.metric_uv
            },
            "blend_graph": _blend_graph(),
            "mask_priority": list(MASK_PRIORITY),
            "evidence_inventory_sha256": self.evidence_inventory_sha256,
            "tiles": [tile.as_dict() for tile in self.tiles],
        }
        value["plan_sha256"] = _canonical_sha256(value)
        return value

    @property
    def fingerprint(self) -> str:
        return str(self.as_dict()["plan_sha256"])


@dataclass(frozen=True, slots=True)
class CompositeGroundMaterialSpec:
    """Portable interface a native tiled USD/MaterialX author must implement."""

    plan_sha256: str
    material_prim_path: str
    material_roles: tuple[str, ...]
    spatial_bindings: tuple[Mapping[str, object], ...]
    metric_uv_sha256: str
    blend_graph_sha256: str
    material_bindings_sha256: str
    evidence_bindings_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.plan_sha256, "terrain PBR plan"),
            (self.metric_uv_sha256, "metric UV contract"),
            (self.blend_graph_sha256, "terrain blend graph"),
            (self.material_bindings_sha256, "material bindings"),
            (self.evidence_bindings_sha256, "evidence bindings"),
        ):
            _require_sha256(value, label=label)
        object.__setattr__(self, "material_roles", tuple(self.material_roles))
        frozen_bindings: list[Mapping[str, object]] = []
        for original in self.spatial_bindings:
            binding = dict(original)
            for key in (
                "tile_bounds_m",
                "source_tile_bounds_m",
                "sampling_bounds_m",
                "source_sampling_bounds_m",
                "source_bounds_m",
                "scene_origin_source_m",
                "halo_edges_m",
            ):
                bounds = binding.get(key)
                if isinstance(bounds, list):
                    binding[key] = tuple(bounds)
            frozen_bindings.append(MappingProxyType(binding))
        object.__setattr__(
            self,
            "spatial_bindings",
            tuple(frozen_bindings),
        )
        if self.material_roles != PBR_MATERIAL_ROLES:
            raise TerrainPbrContractError(
                "composite ground material must expose all seven material roles"
            )
        if (
            not self.material_prim_path.startswith("/")
            or self.material_prim_path == "/"
            or "//" in self.material_prim_path
            or ".." in self.material_prim_path.split("/")
        ):
            raise TerrainPbrContractError(
                "composite ground material prim path must be concrete and absolute"
            )
        binding_ids = [
            str(binding.get("binding_id", ""))
            for binding in self.spatial_bindings
        ]
        if len(binding_ids) != len(set(binding_ids)) or any(
            not binding_id for binding_id in binding_ids
        ):
            raise TerrainPbrContractError(
                "composite ground spatial bindings must have unique identifiers"
            )
        bindings_by_tile: dict[str, list[Mapping[str, object]]] = {}
        for binding in self.spatial_bindings:
            tile_id = str(binding.get("tile_id", ""))
            bindings_by_tile.setdefault(tile_id, []).append(binding)
        if any(
            not tile_id
            or len(bindings) != len(EVIDENCE_SEMANTICS)
            or {str(binding.get("semantic", "")) for binding in bindings}
            != set(EVIDENCE_SEMANTICS)
            or len(
                {
                    tuple(binding.get("tile_bounds_m", ()))
                    for binding in bindings
                }
            )
            != 1
            for tile_id, bindings in bindings_by_tile.items()
        ):
            raise TerrainPbrContractError(
                "each payload tile requires one coherent binding per evidence "
                "semantic"
            )
        for binding in self.spatial_bindings:
            if binding.get("tile_ref") != binding.get("tile_id"):
                raise TerrainPbrContractError(
                    "terrain material tile_ref must equal its source tile_id"
                )
            local = Bounds2d(*binding["tile_bounds_m"])
            source_tile = Bounds2d(*binding["source_tile_bounds_m"])
            local_sampling = Bounds2d(*binding["sampling_bounds_m"])
            source_sampling = Bounds2d(
                *binding["source_sampling_bounds_m"]
            )
            source = Bounds2d(*binding["source_bounds_m"])
            origin = tuple(binding["scene_origin_source_m"])
            translated = Bounds2d(
                source_tile.min_x_m - float(origin[0]),
                source_tile.min_y_m - float(origin[1]),
                source_tile.max_x_m - float(origin[0]),
                source_tile.max_y_m - float(origin[1]),
            )
            translated_sampling = Bounds2d(
                source_sampling.min_x_m - float(origin[0]),
                source_sampling.min_y_m - float(origin[1]),
                source_sampling.max_x_m - float(origin[0]),
                source_sampling.max_y_m - float(origin[1]),
            )
            halo_m = float(binding.get("halo_m", 0.0))
            halo_edges = tuple(binding.get("halo_edges_m", ()))
            if (
                local != translated
                or local_sampling != translated_sampling
                or not source.contains(source_tile)
                or not source_sampling.contains(source_tile)
                or not math.isfinite(halo_m)
                or halo_m < max(_TRANSITION_WIDTH_M.values())
                or len(halo_edges) != 4
                or any(
                    not math.isfinite(float(value))
                    or float(value) < 0.0
                    or float(value) > halo_m + 1.0e-9
                    for value in halo_edges
                )
            ):
                raise TerrainPbrContractError(
                    "payload tile coordinate frames are inconsistent"
                )

    def as_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": COMPOSITE_AUTHORING_SCHEMA_VERSION,
            "contract": "object_free_composite_ground_material",
            "terrain_pbr_plan_sha256": self.plan_sha256,
            "material_prim_path": self.material_prim_path,
            "shader_interface": {
                "root_prim_type": "UsdGeom.Scope",
                "tile_material_prim_type": "UsdShade.Material",
                "allowed_render_contexts": ["mdl", "mtlx"],
                "preview_surface_only_allowed": False,
                "tile_surface_output_must_be_connected": True,
                "all_tile_branches_must_be_surface_reachable": True,
                "uniform_fallback_allowed": False,
            },
            "binding_topology": {
                "mode": "payload_tiled_materials_shared_pbr_library",
                "shared_pbr_library_count": 1,
                "material_graph_count": len(self.tile_ids),
                "tile_payload_count": len(self.tile_ids),
                "spatial_bindings_per_tile": len(EVIDENCE_SEMANTICS),
                "maximum_spatial_image_nodes_per_tile": (
                    MAX_SPATIAL_IMAGE_NODES_PER_TILE
                ),
                "maximum_reachable_shader_prims_per_tile": (
                    MAX_REACHABLE_SHADER_PRIMS_PER_TILE
                ),
                "single_graph_for_all_tiles_allowed": False,
                "one_monolithic_mask_atlas_allowed": False,
                "mask_storage": (
                    "locked_source_tiles_or_authorer_derived_tiled_masks"
                ),
                "mask_address_mode": "clamp",
                "minimum_halo_m": max(_TRANSITION_WIDTH_M.values()),
                "material_uv_space": "world_metres",
                "source_colour_can_feed_base_color": False,
                "source_geometry_can_create_rendered_objects": False,
            },
            "coordinate_frames": {
                "spatial_source_bounds": "declared_metric_evidence_crs",
                "tile_payload_bounds": "scene_local_metres",
                "translation": "scene_equals_source_minus_scene_origin",
            },
            "material_roles": list(self.material_roles),
            "spatial_bindings": [
                dict(binding) for binding in self.spatial_bindings
            ],
            "metric_uv_sha256": self.metric_uv_sha256,
            "blend_graph_sha256": self.blend_graph_sha256,
            "material_bindings_sha256": self.material_bindings_sha256,
            "evidence_bindings_sha256": self.evidence_bindings_sha256,
            "required_native_metadata": {
                "fireviewer:terrainPbrPlanSha256": self.plan_sha256,
                "fireviewer:metricUvSha256": self.metric_uv_sha256,
                "fireviewer:blendGraphSha256": self.blend_graph_sha256,
                "fireviewer:materialBindingsSha256": (
                    self.material_bindings_sha256
                ),
                "fireviewer:evidenceBindingsSha256": (
                    self.evidence_bindings_sha256
                ),
                "fireviewer:uniformFallbackPresent": False,
            },
        }
        value["specification_sha256"] = _canonical_sha256(value)
        return value

    @property
    def fingerprint(self) -> str:
        return str(self.as_dict()["specification_sha256"])

    @property
    def mask_binding_ids(self) -> tuple[str, ...]:
        return tuple(
            str(binding["binding_id"])
            for binding in self.spatial_bindings
            if binding["semantic"] in CLASSIFICATION_SEMANTICS
        )

    @property
    def relief_binding_ids(self) -> tuple[str, ...]:
        return tuple(
            str(binding["binding_id"])
            for binding in self.spatial_bindings
            if binding["semantic"] == "elevation"
        )

    @property
    def tile_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    str(binding["tile_id"])
                    for binding in self.spatial_bindings
                }
            )
        )

    def bindings_for_tile(
        self,
        tile_id: str,
    ) -> tuple[Mapping[str, object], ...]:
        return tuple(
            binding
            for binding in self.spatial_bindings
            if binding["tile_id"] == tile_id
        )


@dataclass(frozen=True, slots=True)
class CompositeGroundMaterialArtifact:
    """A native-validated payload index and its per-tile materials."""

    ground_material: FileLock
    material_prim_path: str
    authoring_receipt: FileLock
    plan_sha256: str
    specification_sha256: str
    render_context: str
    tile_material_payloads: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.material_prim_path.startswith("/")
            or self.material_prim_path == "/"
        ):
            raise TerrainPbrContractError(
                "ground artifact requires a concrete material prim path"
            )
        _require_sha256(self.plan_sha256, label="ground artifact plan")
        _require_sha256(
            self.specification_sha256,
            label="ground artifact specification",
        )
        if self.render_context not in {"mdl", "mtlx"}:
            raise TerrainPbrContractError(
                "ground artifact must use the MDL or MaterialX render context"
            )
        object.__setattr__(
            self,
            "tile_material_payloads",
            tuple(
                MappingProxyType(dict(payload))
                for payload in self.tile_material_payloads
            ),
        )
        if not self.tile_material_payloads:
            raise TerrainPbrContractError(
                "ground artifact requires at least one tile material payload"
            )

    def as_layout_artifact(self) -> dict[str, object]:
        """Return the exact tile-to-material mapping consumed by layouts."""

        return {
            "path": self.ground_material.path,
            "sha256": self.ground_material.sha256,
            "prim_path": self.material_prim_path,
            "topology": "payload_tiled_materials_shared_pbr_library",
            "isolated_content_roles": ["object_free_pbr_ground"],
            "tile_material_payloads": [
                dict(payload) for payload in self.tile_material_payloads
            ],
            "terrain_pbr_plan_sha256": self.plan_sha256,
            "terrain_pbr_specification_sha256": self.specification_sha256,
            "terrain_pbr_authoring_receipt": {
                "path": self.authoring_receipt.path,
                "sha256": self.authoring_receipt.sha256,
            },
        }


class CompositeGroundAuthoringBackend(Protocol):
    """Injectable native authoring boundary used by the pod and unit tests."""

    def author(
        self,
        *,
        plan: TerrainPbrPlan,
        specification: CompositeGroundMaterialSpec,
        bundle_root: Path,
        evidence_root: Path,
        artifact_root: Path,
        output_path: Path,
        final_output_path: Path,
        derived_output_root: Path,
        final_derived_output_root: Path,
    ) -> Mapping[str, object]:
        """Write a real USD layer and return inspected native metrics."""


def build_composite_ground_material_spec(
    plan: TerrainPbrPlan,
    *,
    material_prim_path: str = "/Ground",
) -> CompositeGroundMaterialSpec:
    """Describe the shared graph and independently loadable tile payloads."""

    portable = plan.as_dict()
    metric_uv = portable["metric_uv"]
    blend_graph = portable["blend_graph"]
    materials = portable["materials"]
    scene_origin_x, scene_origin_y = plan.scene_origin_source_m
    scene_min_x = min(tile.bounds.min_x_m for tile in plan.tiles)
    scene_min_y = min(tile.bounds.min_y_m for tile in plan.tiles)
    scene_max_x = max(tile.bounds.max_x_m for tile in plan.tiles)
    scene_max_y = max(tile.bounds.max_y_m for tile in plan.tiles)
    edge_tolerance_m = 1.0e-9
    spatial_bindings: list[dict[str, object]] = []
    for tile in plan.tiles:
        tile_halo_m = max(_TRANSITION_WIDTH_M.values()) + 2.0 * max(
            source.resolution_m for source in tile.evidence
        )
        sampling_min_x = tile.bounds.min_x_m - tile_halo_m
        sampling_min_y = tile.bounds.min_y_m - tile_halo_m
        sampling_max_x = tile.bounds.max_x_m + tile_halo_m
        sampling_max_y = tile.bounds.max_y_m + tile_halo_m
        if abs(tile.bounds.min_x_m - scene_min_x) <= edge_tolerance_m:
            sampling_min_x = max(
                sampling_min_x,
                max(source.bounds.min_x_m for source in tile.evidence),
            )
        if abs(tile.bounds.min_y_m - scene_min_y) <= edge_tolerance_m:
            sampling_min_y = max(
                sampling_min_y,
                max(source.bounds.min_y_m for source in tile.evidence),
            )
        if abs(tile.bounds.max_x_m - scene_max_x) <= edge_tolerance_m:
            sampling_max_x = min(
                sampling_max_x,
                min(source.bounds.max_x_m for source in tile.evidence),
            )
        if abs(tile.bounds.max_y_m - scene_max_y) <= edge_tolerance_m:
            sampling_max_y = min(
                sampling_max_y,
                min(source.bounds.max_y_m for source in tile.evidence),
            )
        sampling_bounds = Bounds2d(
            sampling_min_x,
            sampling_min_y,
            sampling_max_x,
            sampling_max_y,
        )
        halo_edges = [
            tile.bounds.min_x_m - sampling_min_x,
            tile.bounds.min_y_m - sampling_min_y,
            sampling_max_x - tile.bounds.max_x_m,
            sampling_max_y - tile.bounds.max_y_m,
        ]
        for source in tile.evidence:
            spatial_bindings.append(
                {
                    "binding_id": f"{tile.stable_id}:{source.semantic}",
                    "tile_id": tile.stable_id,
                    "tile_ref": tile.stable_id,
                    "semantic": source.semantic,
                    "content_kind": source.content_kind,
                    "usage": source.usage,
                    "path": source.lock.path,
                    "sha256": source.lock.sha256,
                    "size_bytes": source.lock.size_bytes,
                    "crs": source.crs,
                    "tile_bounds_m": [
                        tile.bounds.min_x_m - scene_origin_x,
                        tile.bounds.min_y_m - scene_origin_y,
                        tile.bounds.max_x_m - scene_origin_x,
                        tile.bounds.max_y_m - scene_origin_y,
                    ],
                    "source_tile_bounds_m": tile.bounds.as_list(),
                    "sampling_bounds_m": [
                        sampling_bounds.min_x_m - scene_origin_x,
                        sampling_bounds.min_y_m - scene_origin_y,
                        sampling_bounds.max_x_m - scene_origin_x,
                        sampling_bounds.max_y_m - scene_origin_y,
                    ],
                    "source_sampling_bounds_m": (
                        sampling_bounds.as_list()
                    ),
                    "halo_m": tile_halo_m,
                    "halo_edges_m": halo_edges,
                    "source_bounds_m": source.bounds.as_list(),
                    "scene_origin_source_m": [
                        scene_origin_x,
                        scene_origin_y,
                    ],
                    "resolution_m": source.resolution_m,
                    "shader_inputs": (
                        ["relief_slope", "relief_roughness"]
                        if source.semantic == "elevation"
                        else [f"mask_{source.semantic}"]
                    ),
                }
            )
    spatial_bindings.sort(key=lambda binding: str(binding["binding_id"]))
    return CompositeGroundMaterialSpec(
        plan_sha256=plan.fingerprint,
        material_prim_path=material_prim_path,
        material_roles=PBR_MATERIAL_ROLES,
        spatial_bindings=tuple(spatial_bindings),
        metric_uv_sha256=_canonical_sha256(metric_uv),
        blend_graph_sha256=_canonical_sha256(blend_graph),
        material_bindings_sha256=_canonical_sha256(materials),
        evidence_bindings_sha256=_canonical_sha256(spatial_bindings),
    )


def _measure_metric_uv_continuity(
    plan: TerrainPbrPlan,
) -> dict[str, object]:
    """Numerically compare world-metric UV values on every shared tile edge."""

    seams: list[dict[str, object]] = []
    tolerance_m = 1.0e-9
    for left_index, left in enumerate(plan.tiles):
        for right in plan.tiles[left_index + 1 :]:
            axis = ""
            coordinate = 0.0
            overlap_min = 0.0
            overlap_max = 0.0
            if (
                abs(left.bounds.max_x_m - right.bounds.min_x_m)
                <= tolerance_m
                or abs(right.bounds.max_x_m - left.bounds.min_x_m)
                <= tolerance_m
            ):
                axis = "x"
                coordinate = (
                    left.bounds.max_x_m
                    if abs(left.bounds.max_x_m - right.bounds.min_x_m)
                    <= tolerance_m
                    else right.bounds.max_x_m
                )
                overlap_min = max(
                    left.bounds.min_y_m,
                    right.bounds.min_y_m,
                )
                overlap_max = min(
                    left.bounds.max_y_m,
                    right.bounds.max_y_m,
                )
            elif (
                abs(left.bounds.max_y_m - right.bounds.min_y_m)
                <= tolerance_m
                or abs(right.bounds.max_y_m - left.bounds.min_y_m)
                <= tolerance_m
            ):
                axis = "y"
                coordinate = (
                    left.bounds.max_y_m
                    if abs(left.bounds.max_y_m - right.bounds.min_y_m)
                    <= tolerance_m
                    else right.bounds.max_y_m
                )
                overlap_min = max(
                    left.bounds.min_x_m,
                    right.bounds.min_x_m,
                )
                overlap_max = min(
                    left.bounds.max_x_m,
                    right.bounds.max_x_m,
                )
            if not axis or overlap_max - overlap_min <= tolerance_m:
                continue
            maximum_error = 0.0
            sample_count = 0
            for role in PBR_MATERIAL_ROLES:
                for along in (
                    overlap_min,
                    (overlap_min + overlap_max) / 2.0,
                    overlap_max,
                ):
                    world_x, world_y = (
                        (coordinate, along)
                        if axis == "x"
                        else (along, coordinate)
                    )
                    left_uv = plan.uv_for(role, world_x, world_y)
                    right_uv = plan.uv_for(role, world_x, world_y)
                    maximum_error = max(
                        maximum_error,
                        abs(left_uv[0] - right_uv[0]),
                        abs(left_uv[1] - right_uv[1]),
                    )
                    sample_count += 1
            seams.append(
                {
                    "tile_a": left.stable_id,
                    "tile_b": right.stable_id,
                    "axis": axis,
                    "coordinate_m": coordinate,
                    "overlap_m": [overlap_min, overlap_max],
                    "role_sample_count": sample_count,
                    "maximum_absolute_uv_error": maximum_error,
                }
            )
    seams.sort(key=lambda seam: (str(seam["tile_a"]), str(seam["tile_b"])))
    result: dict[str, object] = {
        "algorithm": "world_metric_shared_edge_samples_v1",
        "adjacent_pair_count": len(seams),
        "samples_per_role_per_edge": 3,
        "tolerance_uv": 1.0e-12,
        "maximum_absolute_uv_error": max(
            (
                float(seam["maximum_absolute_uv_error"])
                for seam in seams
            ),
            default=0.0,
        ),
        "seams": seams,
    }
    result["measurement_sha256"] = _canonical_sha256(result)
    return result


def _safe_usd_identifier(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not normalized or normalized[0].isdigit():
        normalized = f"N_{normalized}"
    return normalized[:80]


def _relative_asset_path(*, layer_path: Path, asset_path: Path) -> str:
    return os.path.relpath(
        asset_path.resolve(),
        start=layer_path.resolve().parent,
    ).replace("\\", "/")


def _write_float_geotiff(
    *,
    gdal: Any,
    path: Path,
    array: Any,
    geotransform: tuple[float, ...],
    projection: str,
) -> None:
    height, width = array.shape
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path),
        int(width),
        int(height),
        1,
        gdal.GDT_Float32,
        options=[
            "COMPRESS=DEFLATE",
            "PREDICTOR=3",
            "TILED=YES",
            "BIGTIFF=IF_SAFER",
        ],
    )
    if dataset is None:
        raise TerrainPbrContractError(
            f"GDAL could not create derived terrain input: {path.name}"
        )
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(array)
    band.FlushCache()
    dataset.FlushCache()
    dataset = None
    if not path.is_file() or path.stat().st_size <= 0:
        raise TerrainPbrContractError(
            f"GDAL produced no derived terrain input: {path.name}"
        )


def _smoothstep_array(np: Any, values: Any, low: float, high: float) -> Any:
    unit = np.clip((values - low) / (high - low), 0.0, 1.0)
    return unit * unit * (3.0 - 2.0 * unit)


def _finite_array_metrics(
    np: Any,
    values: Any,
    *,
    label: str,
) -> dict[str, object]:
    array = np.asarray(values)
    sample_count = int(array.size)
    finite = np.isfinite(array)
    finite_count = int(np.count_nonzero(finite))
    nodata_count = sample_count - finite_count
    if sample_count <= 0 or nodata_count != 0:
        raise TerrainPbrContractError(
            f"{label} contains missing or non-finite samples"
        )
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    return {
        "sample_count": sample_count,
        "finite_count": finite_count,
        "nodata_count": 0,
        "finite_fraction": 1.0,
        "minimum": minimum,
        "maximum": maximum,
    }


def _feather_classification_mask(
    *,
    gdal: Any,
    np: Any,
    source_dataset: Any,
    output_path: Path,
    transition_width_m: float,
) -> dict[str, object]:
    source_band = source_dataset.GetRasterBand(1)
    values = source_band.ReadAsArray()
    if values is None:
        raise TerrainPbrContractError(
            "GDAL could not read a classified terrain mask"
        )
    source_metrics = _finite_array_metrics(
        np,
        values,
        label="classified terrain mask",
    )
    binary = np.asarray(values > 0, dtype=np.uint8)
    width = source_dataset.RasterXSize
    height = source_dataset.RasterYSize
    transform = source_dataset.GetGeoTransform()
    projection = source_dataset.GetProjection()
    memory = gdal.GetDriverByName("MEM")

    def proximity(mask: Any) -> Any:
        mask_dataset = memory.Create("", width, height, 1, gdal.GDT_Byte)
        mask_dataset.SetGeoTransform(transform)
        mask_dataset.SetProjection(projection)
        mask_dataset.GetRasterBand(1).WriteArray(mask)
        distance_dataset = memory.Create(
            "",
            width,
            height,
            1,
            gdal.GDT_Float32,
        )
        distance_dataset.SetGeoTransform(transform)
        distance_dataset.SetProjection(projection)
        result = gdal.ComputeProximity(
            mask_dataset.GetRasterBand(1),
            distance_dataset.GetRasterBand(1),
            options=["VALUES=1", "DISTUNITS=GEO"],
        )
        if result not in (None, 0):
            raise TerrainPbrContractError(
                "GDAL could not compute a metric terrain-mask transition"
            )
        distances = distance_dataset.GetRasterBand(1).ReadAsArray()
        if distances is None:
            raise TerrainPbrContractError(
                "GDAL returned an empty terrain-mask distance field"
            )
        return np.asarray(distances, dtype=np.float32)

    distance_to_inside = proximity(binary)
    distance_to_outside = proximity(1 - binary)
    signed_distance = distance_to_outside - distance_to_inside
    normalized = np.clip(
        signed_distance / max(transition_width_m, 1.0e-6) + 0.5,
        0.0,
        1.0,
    )
    feathered = normalized * normalized * (3.0 - 2.0 * normalized)
    feathered_metrics = _finite_array_metrics(
        np,
        feathered,
        label="feathered terrain mask",
    )
    _write_float_geotiff(
        gdal=gdal,
        path=output_path,
        array=np.asarray(feathered, dtype=np.float32),
        geotransform=transform,
        projection=projection,
    )
    return {
        "classified_source": source_metrics,
        "feathered_mask": feathered_metrics,
    }


def _prepare_native_spatial_inputs(
    *,
    specification: CompositeGroundMaterialSpec,
    evidence_root: Path,
    artifact_root: Path,
    derived_output_root: Path,
) -> tuple[
    tuple[dict[str, object], ...],
    dict[str, dict[str, Path]],
]:
    """Rasterize bounded mask tiles and derive slope/roughness on the pod."""

    try:
        import numpy as np
        from osgeo import gdal
    except ImportError as exc:
        raise RuntimeError(
            "native composite terrain authoring requires NumPy and GDAL/osgeo"
        ) from exc
    gdal.UseExceptions()
    derived_output_root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    assets_by_binding: dict[str, dict[str, Path]] = {}
    for binding in specification.spatial_bindings:
        binding_id = str(binding["binding_id"])
        semantic = str(binding["semantic"])
        source_relative = _safe_relative_path(
            str(binding["path"]),
            label=f"{binding_id} source",
        )
        source_path = evidence_root.joinpath(*source_relative.parts)
        local_bounds = tuple(
            float(value) for value in binding["tile_bounds_m"]
        )
        local_sampling_bounds = tuple(
            float(value) for value in binding["sampling_bounds_m"]
        )
        source_tile_bounds = tuple(
            float(value) for value in binding["source_tile_bounds_m"]
        )
        bounds = tuple(
            float(value)
            for value in binding["source_sampling_bounds_m"]
        )
        source_bounds = tuple(
            float(value) for value in binding["source_bounds_m"]
        )
        if not Bounds2d(*source_bounds).contains(Bounds2d(*bounds)):
            raise TerrainPbrContractError(
                f"{binding_id} source lacks the required metric tile halo"
            )
        min_x, min_y, max_x, max_y = bounds
        resolution = float(binding["resolution_m"])
        width = int(math.ceil((max_x - min_x) / resolution))
        height = int(math.ceil((max_y - min_y) / resolution))
        if (
            width <= 1
            or height <= 1
            or width > MAX_DERIVED_TILE_EDGE_PIXELS
            or height > MAX_DERIVED_TILE_EDGE_PIXELS
        ):
            raise TerrainPbrContractError(
                f"{binding_id} cannot be authored at its declared resolution "
                "inside the bounded per-tile mask contract"
            )
        stem = (
            f"{_safe_usd_identifier(binding_id)}-"
            f"{hashlib.sha256(binding_id.encode()).hexdigest()[:12]}"
        )
        temporary = derived_output_root / f".{stem}.source.tif"
        creation_options = [
            "COMPRESS=DEFLATE",
            "TILED=YES",
            "BIGTIFF=IF_SAFER",
        ]
        content_kind = str(binding["content_kind"])
        if semantic == "elevation":
            warped = gdal.Warp(
                str(temporary),
                str(source_path),
                options=gdal.WarpOptions(
                    format="GTiff",
                    outputBounds=[min_x, min_y, max_x, max_y],
                    width=width,
                    height=height,
                    dstSRS=str(binding["crs"]),
                    creationOptions=creation_options,
                    outputType=gdal.GDT_Float32,
                    resampleAlg="bilinear",
                    dstNodata=float("nan"),
                ),
            )
            if warped is None:
                raise TerrainPbrContractError(
                    f"GDAL could not tile elevation input {binding_id}"
                )
            elevation_array = warped.GetRasterBand(1).ReadAsArray()
            if elevation_array is None:
                raise TerrainPbrContractError(
                    f"GDAL returned no elevation samples for {binding_id}"
                )
            elevation_metrics = _finite_array_metrics(
                np,
                elevation_array,
                label=f"{binding_id} elevation halo",
            )
            warped = None
            slope_temporary = derived_output_root / f".{stem}.slope.tif"
            roughness_temporary = (
                derived_output_root / f".{stem}.roughness.tif"
            )
            slope_dataset = gdal.DEMProcessing(
                str(slope_temporary),
                str(temporary),
                "slope",
                options=gdal.DEMProcessingOptions(
                    format="GTiff",
                    computeEdges=True,
                    slopeFormat="degree",
                    creationOptions=[
                        "COMPRESS=DEFLATE",
                        "TILED=YES",
                    ],
                ),
            )
            roughness_dataset = gdal.DEMProcessing(
                str(roughness_temporary),
                str(temporary),
                "TRI",
                options=gdal.DEMProcessingOptions(
                    format="GTiff",
                    computeEdges=True,
                    alg="Riley",
                    creationOptions=[
                        "COMPRESS=DEFLATE",
                        "TILED=YES",
                    ],
                ),
            )
            if slope_dataset is None or roughness_dataset is None:
                raise TerrainPbrContractError(
                    f"GDAL could not derive relief masks for {binding_id}"
                )
            slope_array = slope_dataset.GetRasterBand(1).ReadAsArray()
            roughness_array = roughness_dataset.GetRasterBand(1).ReadAsArray()
            transform = slope_dataset.GetGeoTransform()
            projection = slope_dataset.GetProjection()
            if slope_array is None or roughness_array is None:
                raise TerrainPbrContractError(
                    f"GDAL returned empty relief masks for {binding_id}"
                )
            slope_source_metrics = _finite_array_metrics(
                np,
                slope_array,
                label=f"{binding_id} slope",
            )
            roughness_source_metrics = _finite_array_metrics(
                np,
                roughness_array,
                label=f"{binding_id} roughness",
            )
            slope = _smoothstep_array(
                np,
                slope_array,
                20.0,
                42.0,
            )
            roughness = _smoothstep_array(
                np,
                roughness_array,
                0.15,
                1.25,
            )
            slope_metrics = _finite_array_metrics(
                np,
                slope,
                label=f"{binding_id} normalized slope",
            )
            roughness_metrics = _finite_array_metrics(
                np,
                roughness,
                label=f"{binding_id} normalized roughness",
            )
            slope_path = derived_output_root / f"{stem}-relief-slope.tif"
            roughness_path = (
                derived_output_root / f"{stem}-relief-roughness.tif"
            )
            _write_float_geotiff(
                gdal=gdal,
                path=slope_path,
                array=np.asarray(slope, dtype=np.float32),
                geotransform=transform,
                projection=projection,
            )
            _write_float_geotiff(
                gdal=gdal,
                path=roughness_path,
                array=np.asarray(roughness, dtype=np.float32),
                geotransform=transform,
                projection=projection,
            )
            slope_dataset = None
            roughness_dataset = None
            temporary.unlink(missing_ok=True)
            slope_temporary.unlink(missing_ok=True)
            roughness_temporary.unlink(missing_ok=True)
            asset_paths = {
                "relief_slope": slope_path,
                "relief_roughness": roughness_path,
            }
            representation = (
                "slope_and_roughness_from_locked_heightfield"
            )
            coverage_metrics = {
                "elevation_source": elevation_metrics,
                "slope_source": slope_source_metrics,
                "roughness_source": roughness_source_metrics,
                "relief_slope": slope_metrics,
                "relief_roughness": roughness_metrics,
            }
        else:
            if content_kind == "classified_vector":
                dataset = gdal.Rasterize(
                    str(temporary),
                    str(source_path),
                    options=gdal.RasterizeOptions(
                        format="GTiff",
                        outputBounds=[min_x, min_y, max_x, max_y],
                        width=width,
                        height=height,
                        outputSRS=str(binding["crs"]),
                        creationOptions=creation_options,
                        outputType=gdal.GDT_Byte,
                        burnValues=[1],
                        initValues=[0],
                        allTouched=True,
                        noData=0,
                    ),
                )
            elif content_kind == "classified_mask":
                dataset = gdal.Warp(
                    str(temporary),
                    str(source_path),
                    options=gdal.WarpOptions(
                        format="GTiff",
                        outputBounds=[min_x, min_y, max_x, max_y],
                        width=width,
                        height=height,
                        dstSRS=str(binding["crs"]),
                        creationOptions=creation_options,
                        outputType=gdal.GDT_Byte,
                        resampleAlg="near",
                        dstNodata=0,
                    ),
                )
            else:
                raise TerrainPbrContractError(
                    f"{binding_id} is not a classified mask or vector"
                )
            if dataset is None:
                raise TerrainPbrContractError(
                    f"GDAL could not tile classified input {binding_id}"
                )
            mask_path = derived_output_root / f"{stem}-mask.tif"
            coverage_metrics = _feather_classification_mask(
                gdal=gdal,
                np=np,
                source_dataset=dataset,
                output_path=mask_path,
                transition_width_m=_TRANSITION_WIDTH_M[semantic],
            )
            dataset = None
            temporary.unlink(missing_ok=True)
            asset_paths = {"mask": mask_path}
            representation = "feathered_classification_mask"
        assets_by_binding[binding_id] = asset_paths
        record_assets: dict[str, object] = {}
        for role, path in sorted(asset_paths.items()):
            record_assets[role] = {
                "path": path.relative_to(artifact_root).as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        records.append(
            {
                "binding_id": binding_id,
                "tile_id": binding["tile_id"],
                "tile_ref": binding["tile_ref"],
                "semantic": semantic,
                "source_path": binding["path"],
                "source_sha256": binding["sha256"],
                "tile_bounds_m": list(local_bounds),
                "source_tile_bounds_m": list(source_tile_bounds),
                "sampling_bounds_m": list(local_sampling_bounds),
                "source_sampling_bounds_m": list(bounds),
                "halo_m": binding["halo_m"],
                "halo_edges_m": list(binding["halo_edges_m"]),
                "source_bounds_m": list(source_bounds),
                "scene_origin_source_m": list(
                    binding["scene_origin_source_m"]
                ),
                "representation": representation,
                "coverage_metrics": coverage_metrics,
                "assets": record_assets,
            }
        )
    records.sort(key=lambda record: str(record["binding_id"]))
    return tuple(records), assets_by_binding


def _sample_north_up_raster(
    *,
    np: Any,
    dataset: Any,
    world_x_m: float,
    world_y_m: float,
) -> float:
    transform = dataset.GetGeoTransform()
    if (
        len(transform) != 6
        or transform[1] <= 0.0
        or transform[5] >= 0.0
        or abs(transform[2]) > 1.0e-12
        or abs(transform[4]) > 1.0e-12
    ):
        raise TerrainPbrContractError(
            "edge validation requires a north-up metric raster"
        )
    column = (world_x_m - transform[0]) / transform[1] - 0.5
    row = (world_y_m - transform[3]) / transform[5] - 0.5
    x0 = int(math.floor(column))
    y0 = int(math.floor(row))
    if (
        x0 < 0
        or y0 < 0
        or x0 + 1 >= dataset.RasterXSize
        or y0 + 1 >= dataset.RasterYSize
    ):
        raise TerrainPbrContractError(
            "tile halo does not contain a seam-validation sample"
        )
    values = dataset.GetRasterBand(1).ReadAsArray(x0, y0, 2, 2)
    if values is None:
        raise TerrainPbrContractError(
            "GDAL could not read a seam-validation sample"
        )
    _finite_array_metrics(np, values, label="terrain seam sample")
    dx = column - x0
    dy = row - y0
    return float(
        values[0, 0] * (1.0 - dx) * (1.0 - dy)
        + values[0, 1] * dx * (1.0 - dy)
        + values[1, 0] * (1.0 - dx) * dy
        + values[1, 1] * dx * dy
    )


def _measure_native_mask_edge_continuity(
    *,
    plan: TerrainPbrPlan,
    artifact_root: Path,
    derived_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Sample both halo rasters at identical source-CRS seam coordinates."""

    try:
        import numpy as np
        from osgeo import gdal
    except ImportError as exc:
        raise RuntimeError(
            "native mask-edge inspection requires NumPy and GDAL/osgeo"
        ) from exc
    records_by_tile_semantic = {
        (str(record["tile_id"]), str(record["semantic"])): record
        for record in derived_records
        if record["semantic"] in CLASSIFICATION_SEMANTICS
    }
    uv_measurement = _measure_metric_uv_continuity(plan)
    measurements: list[dict[str, object]] = []
    maximum_error = 0.0
    for seam in uv_measurement["seams"]:
        axis = str(seam["axis"])
        coordinate = float(seam["coordinate_m"])
        overlap_min, overlap_max = (
            float(value) for value in seam["overlap_m"]
        )
        for semantic in CLASSIFICATION_SEMANTICS:
            left_record = records_by_tile_semantic[
                (str(seam["tile_a"]), semantic)
            ]
            right_record = records_by_tile_semantic[
                (str(seam["tile_b"]), semantic)
            ]
            datasets: list[Any] = []
            for record in (left_record, right_record):
                assets = record["assets"]
                if not isinstance(assets, Mapping):
                    raise TerrainPbrContractError(
                        "mask edge inspection lacks derived assets"
                    )
                asset = assets.get("mask")
                if not isinstance(asset, Mapping):
                    raise TerrainPbrContractError(
                        "mask edge inspection lacks its halo mask"
                    )
                relative = _safe_relative_path(
                    str(asset.get("path", "")),
                    label="mask edge inspection asset",
                )
                dataset = gdal.Open(
                    str(artifact_root.joinpath(*relative.parts)),
                    gdal.GA_ReadOnly,
                )
                if dataset is None:
                    raise TerrainPbrContractError(
                        "GDAL could not open a halo mask for edge inspection"
                    )
                datasets.append(dataset)
            errors: list[float] = []
            for sample_index in range(MASK_EDGE_SAMPLE_COUNT):
                fraction = (sample_index + 0.5) / MASK_EDGE_SAMPLE_COUNT
                along = overlap_min + (
                    overlap_max - overlap_min
                ) * fraction
                x, y = (
                    (coordinate, along)
                    if axis == "x"
                    else (along, coordinate)
                )
                values = [
                    _sample_north_up_raster(
                        np=np,
                        dataset=dataset,
                        world_x_m=x,
                        world_y_m=y,
                    )
                    for dataset in datasets
                ]
                errors.append(abs(values[0] - values[1]))
            datasets = []
            edge_error = max(errors, default=0.0)
            maximum_error = max(maximum_error, edge_error)
            measurements.append(
                {
                    "tile_a": seam["tile_a"],
                    "tile_b": seam["tile_b"],
                    "semantic": semantic,
                    "axis": axis,
                    "sample_count": MASK_EDGE_SAMPLE_COUNT,
                    "maximum_absolute_mask_error": edge_error,
                }
            )
    if maximum_error > MASK_EDGE_CONTINUITY_TOLERANCE:
        raise TerrainPbrContractError(
            "halo-derived classification masks are discontinuous at a tile "
            "edge"
        )
    measurements.sort(
        key=lambda value: (
            str(value["tile_a"]),
            str(value["tile_b"]),
            str(value["semantic"]),
        )
    )
    result: dict[str, object] = {
        "algorithm": "gdal_halo_bilinear_shared_edge_v1",
        "adjacent_pair_count": uv_measurement["adjacent_pair_count"],
        "semantic_count_per_edge": len(CLASSIFICATION_SEMANTICS),
        "samples_per_semantic_edge": MASK_EDGE_SAMPLE_COUNT,
        "tolerance": MASK_EDGE_CONTINUITY_TOLERANCE,
        "maximum_absolute_mask_error": maximum_error,
        "measurements": measurements,
    }
    result["measurement_sha256"] = _canonical_sha256(result)
    return result


class _NativeSingleTileMaterialXBackend:
    """Author one bounded MaterialX tile layer for the payload backend."""

    _NODE_PORTS = {
        "ND_position_vector3": ({"space"}, {"out"}),
        "ND_separate3_vector3": ({"in"}, {"outx", "outy", "outz"}),
        "ND_combine2_vector2": ({"in1", "in2"}, {"out"}),
        "ND_subtract_float": ({"in1", "in2"}, {"out"}),
        "ND_divide_float": ({"in1", "in2"}, {"out"}),
        "ND_multiply_float": ({"in1", "in2"}, {"out"}),
        "ND_add_float": ({"in1", "in2"}, {"out"}),
        "ND_clamp_float": ({"in", "low", "high"}, {"out"}),
        "ND_ifgreater_float": (
            {"value1", "value2", "in1", "in2"},
            {"out"},
        ),
        "ND_image_float": (
            {"file", "texcoord", "uaddressmode", "vaddressmode"},
            {"out"},
        ),
        "ND_image_color3": (
            {"file", "texcoord", "uaddressmode", "vaddressmode"},
            {"out"},
        ),
        "ND_image_vector3": (
            {"file", "texcoord", "uaddressmode", "vaddressmode"},
            {"out"},
        ),
        "ND_normalmap_float": ({"in", "scale"}, {"out"}),
        "ND_multiply_color3": ({"in1", "in2"}, {"out"}),
        "ND_add_color3": ({"in1", "in2"}, {"out"}),
        "ND_combine3_color3": ({"in1", "in2", "in3"}, {"out"}),
        "ND_multiply_vector3": ({"in1", "in2"}, {"out"}),
        "ND_add_vector3": ({"in1", "in2"}, {"out"}),
        "ND_combine3_vector3": ({"in1", "in2", "in3"}, {"out"}),
        "ND_normalize_vector3": ({"in"}, {"out"}),
        "ND_standard_surface_surfaceshader": (
            {"base_color", "specular_roughness", "normal"},
            {"out"},
        ),
    }
    _NODE_PORT_TYPES = {
        "ND_position_vector3": (
            {"space": "String"},
            {"out": "Vector3f"},
        ),
        "ND_separate3_vector3": (
            {"in": "Vector3f"},
            {"outx": "Float", "outy": "Float", "outz": "Float"},
        ),
        "ND_combine2_vector2": (
            {"in1": "Float", "in2": "Float"},
            {"out": "Float2"},
        ),
        "ND_subtract_float": (
            {"in1": "Float", "in2": "Float"},
            {"out": "Float"},
        ),
        "ND_divide_float": (
            {"in1": "Float", "in2": "Float"},
            {"out": "Float"},
        ),
        "ND_multiply_float": (
            {"in1": "Float", "in2": "Float"},
            {"out": "Float"},
        ),
        "ND_add_float": (
            {"in1": "Float", "in2": "Float"},
            {"out": "Float"},
        ),
        "ND_clamp_float": (
            {"in": "Float", "low": "Float", "high": "Float"},
            {"out": "Float"},
        ),
        "ND_ifgreater_float": (
            {
                "value1": "Float",
                "value2": "Float",
                "in1": "Float",
                "in2": "Float",
            },
            {"out": "Float"},
        ),
        "ND_image_float": (
            {
                "file": "Asset",
                "texcoord": "Float2",
                "uaddressmode": "String",
                "vaddressmode": "String",
            },
            {"out": "Float"},
        ),
        "ND_image_color3": (
            {
                "file": "Asset",
                "texcoord": "Float2",
                "uaddressmode": "String",
                "vaddressmode": "String",
            },
            {"out": "Color3f"},
        ),
        "ND_image_vector3": (
            {
                "file": "Asset",
                "texcoord": "Float2",
                "uaddressmode": "String",
                "vaddressmode": "String",
            },
            {"out": "Vector3f"},
        ),
        "ND_normalmap_float": (
            {"in": "Vector3f", "scale": "Float"},
            {"out": "Vector3f"},
        ),
        "ND_multiply_color3": (
            {"in1": "Color3f", "in2": "Color3f"},
            {"out": "Color3f"},
        ),
        "ND_add_color3": (
            {"in1": "Color3f", "in2": "Color3f"},
            {"out": "Color3f"},
        ),
        "ND_combine3_color3": (
            {"in1": "Float", "in2": "Float", "in3": "Float"},
            {"out": "Color3f"},
        ),
        "ND_multiply_vector3": (
            {"in1": "Vector3f", "in2": "Vector3f"},
            {"out": "Vector3f"},
        ),
        "ND_add_vector3": (
            {"in1": "Vector3f", "in2": "Vector3f"},
            {"out": "Vector3f"},
        ),
        "ND_combine3_vector3": (
            {"in1": "Float", "in2": "Float", "in3": "Float"},
            {"out": "Vector3f"},
        ),
        "ND_normalize_vector3": (
            {"in": "Vector3f"},
            {"out": "Vector3f"},
        ),
        "ND_standard_surface_surfaceshader": (
            {
                "base_color": "Color3f",
                "specular_roughness": "Float",
                "normal": "Vector3f",
            },
            {"out": "Token"},
        ),
    }

    @staticmethod
    def _validate_sdr_nodes(Sdr: Any, Sdf: Any) -> None:
        registry = Sdr.Registry()
        for identifier, (required_inputs, required_outputs) in (
            _NativeSingleTileMaterialXBackend._NODE_PORTS.items()
        ):
            node = registry.GetShaderNodeByIdentifier(identifier)
            if node is None:
                raise RuntimeError(
                    f"Kit MaterialX registry lacks required node {identifier}"
                )
            inputs = {str(name) for name in node.GetInputNames()}
            outputs = {str(name) for name in node.GetOutputNames()}
            if not required_inputs.issubset(inputs) or not required_outputs.issubset(
                outputs
            ):
                raise RuntimeError(
                    f"Kit MaterialX node {identifier} lacks required ports"
                )
            expected_inputs, expected_outputs = (
                _NativeSingleTileMaterialXBackend._NODE_PORT_TYPES[identifier]
            )
            for direction, expected, getter in (
                ("input", expected_inputs, node.GetInput),
                ("output", expected_outputs, node.GetOutput),
            ):
                for port_name, sdf_type_name in expected.items():
                    shader_property = getter(port_name)
                    if shader_property is None or not hasattr(
                        shader_property,
                        "GetTypeAsSdfType",
                    ):
                        raise RuntimeError(
                            f"Kit MaterialX {identifier} {direction} "
                            f"{port_name} exposes no Sdf type"
                        )
                    converted = shader_property.GetTypeAsSdfType()
                    actual = (
                        converted[0]
                        if isinstance(converted, tuple)
                        else converted
                    )
                    expected_sdf = getattr(
                        Sdf.ValueTypeNames,
                        sdf_type_name,
                    )
                    if actual != expected_sdf:
                        raise RuntimeError(
                            f"Kit MaterialX {identifier} {direction} "
                            f"{port_name} type is {actual}, expected "
                            f"{expected_sdf}"
                        )

    @staticmethod
    def _reachable_shader_metadata(
        *,
        stage: Any,
        material: Any,
        render_context: str,
        UsdShade: Any,
    ) -> tuple[set[str], set[str], set[str], int, int]:
        output = material.GetSurfaceOutput(render_context)
        if not output or not output.GetAttr().GetConnections():
            raise RuntimeError(
                "authored MaterialX ground has no connected surface output"
            )
        pending = list(output.GetAttr().GetConnections())
        visited_properties: set[str] = set()
        visited_prims: set[str] = set()
        roles: set[str] = set()
        bindings: set[str] = set()
        quality_features: set[str] = set()
        texture_color_space_count = 0
        while pending:
            property_path = pending.pop()
            property_key = str(property_path)
            if property_key in visited_properties:
                continue
            visited_properties.add(property_key)
            prim_path = property_path.GetPrimPath()
            prim_key = str(prim_path)
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(
                    "MaterialX ground graph contains a broken connection"
                )
            if prim_key not in visited_prims:
                visited_prims.add(prim_key)
                role = prim.GetCustomDataByKey("fireviewer:materialRole")
                binding = prim.GetCustomDataByKey(
                    "fireviewer:spatialBindingId"
                )
                if isinstance(role, str) and role:
                    roles.add(role)
                if isinstance(binding, str) and binding:
                    bindings.add(binding)
                feature = prim.GetCustomDataByKey(
                    "fireviewer:qualityFeature"
                )
                if isinstance(feature, str) and feature:
                    quality_features.add(feature)
                texture_role = prim.GetCustomDataByKey(
                    "fireviewer:textureRole"
                )
                expected_color_space = prim.GetCustomDataByKey(
                    "fireviewer:sourceColorSpace"
                )
                if isinstance(texture_role, str) and texture_role:
                    shader = UsdShade.Shader(prim)
                    file_input = shader.GetInput("file")
                    if (
                        not file_input
                        or not isinstance(expected_color_space, str)
                        or not expected_color_space
                        or file_input.GetAttr().GetColorSpace()
                        != expected_color_space
                    ):
                        raise RuntimeError(
                            "MaterialX texture input lost its explicit source "
                            "color space after stage reopen"
                        )
                    texture_color_space_count += 1
            connectable = UsdShade.ConnectableAPI(prim)
            for shader_input in connectable.GetInputs():
                pending.extend(shader_input.GetAttr().GetConnections())
        return (
            roles,
            bindings,
            quality_features,
            len(visited_prims),
            texture_color_space_count,
        )

    def author(
        self,
        *,
        plan: TerrainPbrPlan,
        specification: CompositeGroundMaterialSpec,
        bundle_root: Path,
        evidence_root: Path,
        artifact_root: Path,
        output_path: Path,
        final_output_path: Path,
        derived_output_root: Path,
        final_derived_output_root: Path,
    ) -> Mapping[str, object]:
        try:
            from pxr import Sdf, Sdr, Usd, UsdShade
        except ImportError as exc:
            raise RuntimeError(
                "native composite ground authoring requires the pinned Kit "
                "Python with pxr, Sdr and MaterialX discovery"
            ) from exc
        self._validate_sdr_nodes(Sdr, Sdf)
        derived_records, derived_assets = _prepare_native_spatial_inputs(
            specification=specification,
            evidence_root=evidence_root,
            artifact_root=artifact_root,
            derived_output_root=derived_output_root,
        )
        stage = Usd.Stage.CreateNew(str(output_path))
        if stage is None:
            raise RuntimeError("Kit could not create the composite ground USD")
        material = UsdShade.Material.Define(
            stage,
            specification.material_prim_path,
        )
        stage.SetDefaultPrim(material.GetPrim())
        metadata = specification.as_dict()["required_native_metadata"]
        for key, value in metadata.items():
            material.GetPrim().SetCustomDataByKey(key, value)
        material.GetPrim().SetCustomDataByKey(
            "fireviewer:authoringBackend",
            "materialx_native_tiled_evidence_v1",
        )
        shader_root = f"{specification.material_prim_path}/Shaders"
        counter = 0

        def create_shader(
            *,
            identifier: str,
            label: str,
            outputs: Mapping[str, Any],
            values: Mapping[str, tuple[Any, object]] | None = None,
            connections: Mapping[
                str,
                tuple[Any, tuple[Any, str]],
            ]
            | None = None,
            tags: Mapping[str, object] | None = None,
        ) -> Any:
            nonlocal counter
            counter += 1
            name = f"{counter:06d}_{_safe_usd_identifier(label)}"
            shader = UsdShade.Shader.Define(
                stage,
                f"{shader_root}/{name}",
            )
            shader.CreateIdAttr(identifier)
            for output_name, output_type in outputs.items():
                shader.CreateOutput(output_name, output_type)
            for input_name, (input_type, value) in (values or {}).items():
                shader.CreateInput(input_name, input_type).Set(value)
            for input_name, (input_type, source) in (
                connections or {}
            ).items():
                source_shader, source_output = source
                shader.CreateInput(input_name, input_type).ConnectToSource(
                    source_shader.ConnectableAPI(),
                    source_output,
                )
            for key, value in (tags or {}).items():
                shader.GetPrim().SetCustomDataByKey(key, value)
            return shader

        float_type = Sdf.ValueTypeNames.Float
        color_type = Sdf.ValueTypeNames.Color3f
        vector_type = Sdf.ValueTypeNames.Vector3f
        float2_type = Sdf.ValueTypeNames.Float2
        string_type = Sdf.ValueTypeNames.String
        asset_type = Sdf.ValueTypeNames.Asset
        token_type = Sdf.ValueTypeNames.Token

        position = create_shader(
            identifier="ND_position_vector3",
            label="WorldPosition",
            outputs={"out": vector_type},
            values={"space": (string_type, "world")},
        )
        separate = create_shader(
            identifier="ND_separate3_vector3",
            label="WorldPositionXY",
            outputs={
                "outx": float_type,
                "outy": float_type,
                "outz": float_type,
            },
            connections={"in": (vector_type, (position, "out"))},
        )
        world_x = (separate, "outx")
        world_y = (separate, "outy")
        world_z = (separate, "outz")

        def binary_float(
            identifier: str,
            label: str,
            left: tuple[Any, str] | float,
            right: tuple[Any, str] | float,
        ) -> tuple[Any, str]:
            values: dict[str, tuple[Any, object]] = {}
            connections: dict[str, tuple[Any, tuple[Any, str]]] = {}
            for input_name, value in (("in1", left), ("in2", right)):
                if isinstance(value, tuple):
                    connections[input_name] = (float_type, value)
                else:
                    values[input_name] = (float_type, float(value))
            node = create_shader(
                identifier=identifier,
                label=label,
                outputs={"out": float_type},
                values=values,
                connections=connections,
            )
            return node, "out"

        def subtract(
            left: tuple[Any, str] | float,
            right: tuple[Any, str] | float,
            label: str,
        ) -> tuple[Any, str]:
            return binary_float(
                "ND_subtract_float",
                label,
                left,
                right,
            )

        def multiply(
            left: tuple[Any, str] | float,
            right: tuple[Any, str] | float,
            label: str,
        ) -> tuple[Any, str]:
            return binary_float(
                "ND_multiply_float",
                label,
                left,
                right,
            )

        def add(
            left: tuple[Any, str] | float,
            right: tuple[Any, str] | float,
            label: str,
        ) -> tuple[Any, str]:
            return binary_float("ND_add_float", label, left, right)

        def clamp(
            source: tuple[Any, str],
            label: str,
        ) -> tuple[Any, str]:
            node = create_shader(
                identifier="ND_clamp_float",
                label=label,
                outputs={"out": float_type},
                values={
                    "low": (float_type, 0.0),
                    "high": (float_type, 1.0),
                },
                connections={"in": (float_type, source)},
            )
            return node, "out"

        def if_greater(
            left: tuple[Any, str],
            right: float,
            yes: float,
            no: float,
            label: str,
        ) -> tuple[Any, str]:
            node = create_shader(
                identifier="ND_ifgreater_float",
                label=label,
                outputs={"out": float_type},
                values={
                    "value2": (float_type, right),
                    "in1": (float_type, yes),
                    "in2": (float_type, no),
                },
                connections={"value1": (float_type, left)},
            )
            return node, "out"

        def combine_uv(
            u: tuple[Any, str],
            v: tuple[Any, str],
            label: str,
        ) -> tuple[Any, str]:
            node = create_shader(
                identifier="ND_combine2_vector2",
                label=label,
                outputs={"out": float2_type},
                connections={
                    "in1": (float_type, u),
                    "in2": (float_type, v),
                },
            )
            return node, "out"

        def add_many(
            values: Sequence[tuple[Any, str]],
            label: str,
        ) -> tuple[Any, str]:
            if not values:
                raise TerrainPbrContractError(
                    f"MaterialX graph lacks values for {label}"
                )
            result = values[0]
            for index, value in enumerate(values[1:], start=1):
                result = add(result, value, f"{label}_{index}")
            return result

        source_x = add(
            world_x,
            plan.scene_origin_source_m[0],
            "SourceCrsX",
        )
        source_y = add(
            world_y,
            plan.scene_origin_source_m[1],
            "SourceCrsY",
        )
        macro_cos = 0.8910065241883679
        macro_sin = 0.45399049973954675
        macro_u = binary_float(
            "ND_divide_float",
            "MacroRotatedU",
            add(
                multiply(source_x, macro_cos, "MacroXC"),
                multiply(source_y, -macro_sin, "MacroYMinusS"),
                "MacroRotatedX",
            ),
            73.0,
        )
        macro_v = binary_float(
            "ND_divide_float",
            "MacroRotatedV",
            add(
                multiply(source_x, macro_sin, "MacroXS"),
                multiply(source_y, macro_cos, "MacroYC"),
                "MacroRotatedY",
            ),
            73.0,
        )
        macro_uv = combine_uv(macro_u, macro_v, "WorldMacroUV")
        rock_repeat = plan.material_library.by_role(
            "rock"
        ).metres_per_uv_tile
        rock_source_x = binary_float(
            "ND_divide_float",
            "RockSourceX",
            source_x,
            rock_repeat,
        )
        rock_source_y = binary_float(
            "ND_divide_float",
            "RockSourceY",
            source_y,
            rock_repeat,
        )
        rock_world_z = binary_float(
            "ND_divide_float",
            "RockWorldZ",
            world_z,
            rock_repeat,
        )
        rock_xz_uv = combine_uv(
            rock_source_x,
            rock_world_z,
            "RockSlopeXZUV",
        )
        rock_yz_uv = combine_uv(
            rock_source_y,
            rock_world_z,
            "RockSlopeYZUV",
        )

        contributions: dict[str, list[tuple[Any, str]]] = {
            "forest": [],
            "water": [],
            "roads": [],
            "artificial_ground": [],
            "relief_slope": [],
            "relief_roughness": [],
        }
        bindings_by_tile: dict[str, list[Mapping[str, object]]] = {}
        for binding in specification.spatial_bindings:
            bindings_by_tile.setdefault(str(binding["tile_id"]), []).append(
                binding
            )
        for tile_id in sorted(bindings_by_tile):
            bindings = bindings_by_tile[tile_id]
            bounds = tuple(
                float(value) for value in bindings[0]["sampling_bounds_m"]
            )
            min_x, min_y, max_x, max_y = bounds
            width_m = max_x - min_x
            height_m = max_y - min_y
            local_u = binary_float(
                "ND_divide_float",
                f"{tile_id}_LocalU",
                subtract(world_x, min_x, f"{tile_id}_OffsetX"),
                width_m,
            )
            local_v = binary_float(
                "ND_divide_float",
                f"{tile_id}_LocalV",
                subtract(world_y, min_y, f"{tile_id}_OffsetY"),
                height_m,
            )
            local_uv = combine_uv(local_u, local_v, f"{tile_id}_LocalUV")
            epsilon = min(width_m, height_m) * 1.0e-7
            inside_left = if_greater(
                world_x,
                min_x - epsilon,
                1.0,
                0.0,
                f"{tile_id}_InsideLeft",
            )
            inside_right = if_greater(
                world_x,
                max_x + epsilon,
                0.0,
                1.0,
                f"{tile_id}_InsideRight",
            )
            inside_bottom = if_greater(
                world_y,
                min_y - epsilon,
                1.0,
                0.0,
                f"{tile_id}_InsideBottom",
            )
            inside_top = if_greater(
                world_y,
                max_y + epsilon,
                0.0,
                1.0,
                f"{tile_id}_InsideTop",
            )
            inside = multiply(
                multiply(
                    inside_left,
                    inside_right,
                    f"{tile_id}_InsideX",
                ),
                multiply(
                    inside_bottom,
                    inside_top,
                    f"{tile_id}_InsideY",
                ),
                f"{tile_id}_Inside",
            )
            for binding in sorted(
                bindings,
                key=lambda value: str(value["binding_id"]),
            ):
                binding_id = str(binding["binding_id"])
                semantic = str(binding["semantic"])
                assets = derived_assets[binding_id]
                for asset_role, asset_path in sorted(assets.items()):
                    final_asset_path = final_derived_output_root / (
                        asset_path.relative_to(derived_output_root)
                    )
                    image = create_shader(
                        identifier="ND_image_float",
                        label=f"{binding_id}_{asset_role}",
                        outputs={"out": float_type},
                        values={
                            "file": (
                                asset_type,
                                Sdf.AssetPath(
                                    _relative_asset_path(
                                        layer_path=final_output_path,
                                        asset_path=final_asset_path,
                                    )
                                ),
                            ),
                            "uaddressmode": (string_type, "clamp"),
                            "vaddressmode": (string_type, "clamp"),
                        },
                        connections={
                            "texcoord": (float2_type, local_uv),
                        },
                        tags={
                            "fireviewer:spatialBindingId": binding_id,
                            "fireviewer:spatialSemantic": semantic,
                        },
                    )
                    contribution = multiply(
                        (image, "out"),
                        inside,
                        f"{binding_id}_{asset_role}_Inside",
                    )
                    key = (
                        asset_role
                        if semantic == "elevation"
                        else semantic
                    )
                    contributions[key].append(contribution)
        fields = {
            key: clamp(add_many(values, f"Aggregate_{key}"), f"Clamp_{key}")
            for key, values in contributions.items()
        }

        water = fields["water"]
        remaining = subtract(1.0, water, "RemainingAfterWater")
        roads = multiply(remaining, fields["roads"], "RoadWeight")
        remaining = subtract(remaining, roads, "RemainingAfterRoads")
        artificial = multiply(
            remaining,
            fields["artificial_ground"],
            "ArtificialWeight",
        )
        remaining = subtract(
            remaining,
            artificial,
            "RemainingAfterArtificial",
        )
        forest = multiply(remaining, fields["forest"], "ForestWeight")
        remaining = subtract(remaining, forest, "RemainingAfterForest")
        rock = multiply(remaining, fields["relief_slope"], "RockWeight")
        remaining = subtract(remaining, rock, "RemainingAfterRock")
        exposed_soil = multiply(
            multiply(
                remaining,
                fields["relief_roughness"],
                "RoughSoilBase",
            ),
            0.65,
            "RoughSoilWeight",
        )
        grass = subtract(remaining, exposed_soil, "GrassWeight")
        role_weights = {
            "forest_floor": forest,
            "grass": grass,
            "soil": add(
                exposed_soil,
                multiply(artificial, 0.35, "ArtificialSoil"),
                "SoilWeight",
            ),
            "rock": rock,
            "asphalt": add(
                roads,
                multiply(artificial, 0.20, "ArtificialAsphalt"),
                "AsphaltWeight",
            ),
            "gravel": multiply(artificial, 0.45, "GravelWeight"),
            "water": water,
        }

        role_base_colors: list[tuple[Any, str]] = []
        role_roughness: list[tuple[Any, str]] = []
        role_normals: list[tuple[Any, str]] = []
        for material_record in plan.material_library.materials:
            role = material_record.role
            uv_contract = next(
                contract
                for contract in plan.metric_uv
                if contract.role == role
            )
            u = binary_float(
                "ND_divide_float",
                f"{role}_MetricU",
                subtract(
                    world_x,
                    (
                        uv_contract.origin_x_m
                        - plan.scene_origin_source_m[0]
                    ),
                    f"{role}_MetricOffsetX",
                ),
                uv_contract.metres_per_uv_tile,
            )
            v = binary_float(
                "ND_divide_float",
                f"{role}_MetricV",
                subtract(
                    world_y,
                    (
                        uv_contract.origin_y_m
                        - plan.scene_origin_source_m[1]
                    ),
                    f"{role}_MetricOffsetY",
                ),
                uv_contract.metres_per_uv_tile,
            )
            uv = combine_uv(u, v, f"{role}_MetricUV")
            textures = {
                texture.role: texture
                for texture in material_record.textures
            }

            def texture_image(
                texture_role: str,
                identifier: str,
                output_type: Any,
                *,
                texcoord: tuple[Any, str] = uv,
                label_suffix: str = "",
                quality_feature: str | None = None,
            ) -> Any:
                texture = textures[texture_role]
                texture_path = bundle_root.joinpath(
                    *_safe_relative_path(
                        texture.lock.path,
                        label=f"{role} {texture_role}",
                    ).parts
                )
                expected_color_space = _USD_TEXTURE_COLOR_SPACES[
                    texture.color_space.casefold()
                ]
                image = create_shader(
                    identifier=identifier,
                    label=f"{role}_{texture_role}{label_suffix}",
                    outputs={"out": output_type},
                    values={
                        "file": (
                            asset_type,
                            Sdf.AssetPath(
                                _relative_asset_path(
                                    layer_path=final_output_path,
                                    asset_path=texture_path,
                                )
                            ),
                        ),
                        "uaddressmode": (string_type, "periodic"),
                        "vaddressmode": (string_type, "periodic"),
                    },
                    connections={"texcoord": (float2_type, texcoord)},
                    tags={
                        "fireviewer:materialRole": role,
                        "fireviewer:textureRole": texture_role,
                        "fireviewer:sourceColorSpace": (
                            expected_color_space
                        ),
                        **(
                            {
                                "fireviewer:qualityFeature": (
                                    quality_feature
                                )
                            }
                            if quality_feature is not None
                            else {}
                        ),
                    },
                )
                file_input = image.GetInput("file")
                if not file_input:
                    raise RuntimeError(
                        "MaterialX image node has no authored file input"
                    )
                file_input.GetAttr().SetColorSpace(expected_color_space)
                return image

            base = texture_image(
                "base_color",
                "ND_image_color3",
                color_type,
            )
            macro_base = texture_image(
                "base_color",
                "ND_image_color3",
                color_type,
                texcoord=macro_uv,
                label_suffix="_WorldMacro",
                quality_feature="world_macro_variation",
            )

            def scale_color(
                source: tuple[Any, str],
                factor: float,
                label: str,
            ) -> tuple[Any, str]:
                weight = create_shader(
                    identifier="ND_combine3_color3",
                    label=f"{label}_Weight",
                    outputs={"out": color_type},
                    values={
                        "in1": (float_type, factor),
                        "in2": (float_type, factor),
                        "in3": (float_type, factor),
                    },
                )
                scaled = create_shader(
                    identifier="ND_multiply_color3",
                    label=label,
                    outputs={"out": color_type},
                    connections={
                        "in1": (color_type, source),
                        "in2": (color_type, (weight, "out")),
                    },
                )
                return scaled, "out"

            primary_color = scale_color(
                (base, "out"),
                0.82,
                f"{role}_PrimaryColor",
            )
            macro_color = scale_color(
                (macro_base, "out"),
                0.18,
                f"{role}_MacroColor",
            )
            blended_base = create_shader(
                identifier="ND_add_color3",
                label=f"{role}_PrimaryMacroBlend",
                outputs={"out": color_type},
                connections={
                    "in1": (color_type, primary_color),
                    "in2": (color_type, macro_color),
                },
            )
            base = blended_base
            if role == "rock":
                rock_xz = texture_image(
                    "base_color",
                    "ND_image_color3",
                    color_type,
                    texcoord=rock_xz_uv,
                    label_suffix="_SlopeXZ",
                    quality_feature="slope_projection",
                )
                rock_yz = texture_image(
                    "base_color",
                    "ND_image_color3",
                    color_type,
                    texcoord=rock_yz_uv,
                    label_suffix="_SlopeYZ",
                    quality_feature="slope_projection",
                )
                side_x = scale_color(
                    (rock_xz, "out"),
                    0.5,
                    "RockSlopeXZHalf",
                )
                side_y = scale_color(
                    (rock_yz, "out"),
                    0.5,
                    "RockSlopeYZHalf",
                )
                side_base = create_shader(
                    identifier="ND_add_color3",
                    label="RockSlopeSideBlend",
                    outputs={"out": color_type},
                    connections={
                        "in1": (color_type, side_x),
                        "in2": (color_type, side_y),
                    },
                )
                slope_weight = create_shader(
                    identifier="ND_combine3_color3",
                    label="RockSlopeProjectionWeight",
                    outputs={"out": color_type},
                    connections={
                        "in1": (float_type, fields["relief_slope"]),
                        "in2": (float_type, fields["relief_slope"]),
                        "in3": (float_type, fields["relief_slope"]),
                    },
                )
                flat_factor = subtract(
                    1.0,
                    fields["relief_slope"],
                    "RockFlatProjectionFactor",
                )
                flat_weight = create_shader(
                    identifier="ND_combine3_color3",
                    label="RockFlatProjectionWeight",
                    outputs={"out": color_type},
                    connections={
                        "in1": (float_type, flat_factor),
                        "in2": (float_type, flat_factor),
                        "in3": (float_type, flat_factor),
                    },
                )
                flat_color = create_shader(
                    identifier="ND_multiply_color3",
                    label="RockFlatProjectionColor",
                    outputs={"out": color_type},
                    connections={
                        "in1": (color_type, (base, "out")),
                        "in2": (color_type, (flat_weight, "out")),
                    },
                )
                side_color = create_shader(
                    identifier="ND_multiply_color3",
                    label="RockSlopeProjectionColor",
                    outputs={"out": color_type},
                    connections={
                        "in1": (color_type, (side_base, "out")),
                        "in2": (color_type, (slope_weight, "out")),
                    },
                )
                base = create_shader(
                    identifier="ND_add_color3",
                    label="RockSlopeAwareBaseColor",
                    outputs={"out": color_type},
                    connections={
                        "in1": (color_type, (flat_color, "out")),
                        "in2": (color_type, (side_color, "out")),
                    },
                )
            rough = texture_image(
                "roughness",
                "ND_image_float",
                float_type,
            )
            normal_texture = texture_image(
                "normal",
                "ND_image_vector3",
                vector_type,
            )
            normal = create_shader(
                identifier="ND_normalmap_float",
                label=f"{role}_NormalMap",
                outputs={"out": vector_type},
                values={"scale": (float_type, 1.0)},
                connections={
                    "in": (vector_type, (normal_texture, "out")),
                },
                tags={"fireviewer:materialRole": role},
            )
            color_weight = create_shader(
                identifier="ND_combine3_color3",
                label=f"{role}_ColorWeight",
                outputs={"out": color_type},
                connections={
                    "in1": (float_type, role_weights[role]),
                    "in2": (float_type, role_weights[role]),
                    "in3": (float_type, role_weights[role]),
                },
            )
            vector_weight = create_shader(
                identifier="ND_combine3_vector3",
                label=f"{role}_VectorWeight",
                outputs={"out": vector_type},
                connections={
                    "in1": (float_type, role_weights[role]),
                    "in2": (float_type, role_weights[role]),
                    "in3": (float_type, role_weights[role]),
                },
            )
            base_weighted = create_shader(
                identifier="ND_multiply_color3",
                label=f"{role}_WeightedBaseColor",
                outputs={"out": color_type},
                connections={
                    "in1": (color_type, (base, "out")),
                    "in2": (color_type, (color_weight, "out")),
                },
            )
            rough_weighted = create_shader(
                identifier="ND_multiply_float",
                label=f"{role}_WeightedRoughness",
                outputs={"out": float_type},
                connections={
                    "in1": (float_type, (rough, "out")),
                    "in2": (float_type, role_weights[role]),
                },
            )
            normal_weighted = create_shader(
                identifier="ND_multiply_vector3",
                label=f"{role}_WeightedNormal",
                outputs={"out": vector_type},
                connections={
                    "in1": (vector_type, (normal, "out")),
                    "in2": (vector_type, (vector_weight, "out")),
                },
            )
            role_base_colors.append((base_weighted, "out"))
            role_roughness.append((rough_weighted, "out"))
            role_normals.append((normal_weighted, "out"))

        def add_typed_many(
            values: Sequence[tuple[Any, str]],
            *,
            identifier: str,
            value_type: Any,
            label: str,
        ) -> tuple[Any, str]:
            result = values[0]
            for index, value in enumerate(values[1:], start=1):
                node = create_shader(
                    identifier=identifier,
                    label=f"{label}_{index}",
                    outputs={"out": value_type},
                    connections={
                        "in1": (value_type, result),
                        "in2": (value_type, value),
                    },
                )
                result = node, "out"
            return result

        combined_base = add_typed_many(
            role_base_colors,
            identifier="ND_add_color3",
            value_type=color_type,
            label="CombinedBaseColor",
        )
        combined_roughness = add_many(
            role_roughness,
            "CombinedRoughness",
        )
        combined_normal_raw = add_typed_many(
            role_normals,
            identifier="ND_add_vector3",
            value_type=vector_type,
            label="CombinedNormal",
        )
        combined_normal = create_shader(
            identifier="ND_normalize_vector3",
            label="NormalizedCombinedNormal",
            outputs={"out": vector_type},
            connections={"in": (vector_type, combined_normal_raw)},
        )
        surface = create_shader(
            identifier="ND_standard_surface_surfaceshader",
            label="TerrainStandardSurface",
            outputs={"out": token_type},
            connections={
                "base_color": (color_type, combined_base),
                "specular_roughness": (
                    float_type,
                    combined_roughness,
                ),
                "normal": (vector_type, (combined_normal, "out")),
            },
        )
        material.CreateSurfaceOutput("mtlx").ConnectToSource(
            surface.ConnectableAPI(),
            "out",
        )
        stage.GetRootLayer().Save()
        stage = None
        reopened = Usd.Stage.Open(str(output_path), load=Usd.Stage.LoadNone)
        if reopened is None:
            raise RuntimeError(
                "Kit could not reopen the authored MaterialX ground layer"
            )
        reopened_material = UsdShade.Material(
            reopened.GetPrimAtPath(specification.material_prim_path)
        )
        (
            roles,
            bindings,
            quality_features,
            reachable_count,
            texture_color_space_count,
        ) = self._reachable_shader_metadata(
            stage=reopened,
            material=reopened_material,
            render_context="mtlx",
            UsdShade=UsdShade,
        )
        expected_bindings = {
            str(binding["binding_id"])
            for binding in specification.spatial_bindings
        }
        if (
            roles != set(PBR_MATERIAL_ROLES)
            or bindings != expected_bindings
            or quality_features
            != {"world_macro_variation", "slope_projection"}
            or texture_color_space_count != 30
        ):
            raise RuntimeError(
                "authored MaterialX graph does not connect every material and "
                "spatial evidence branch to its surface"
            )
        derived_sha256 = _canonical_sha256(derived_records)
        native_validation = {
            "inspector_id": NATIVE_COMPOSITE_INSPECTOR_ID,
            "render_context": "mtlx",
            "material_prim_path": specification.material_prim_path,
            "material_prim_type": "UsdShade.Material",
            "connected_material_roles": list(PBR_MATERIAL_ROLES),
            "connected_spatial_binding_ids": [
                str(binding["binding_id"])
                for binding in specification.spatial_bindings
            ],
            "connected_mask_binding_ids": list(
                specification.mask_binding_ids
            ),
            "connected_relief_binding_ids": list(
                specification.relief_binding_ids
            ),
            "world_metric_uv_roles": list(PBR_MATERIAL_ROLES),
            "reachable_quality_features": sorted(quality_features),
            "texture_color_space_contract": {
                "base_color": "srgb_texture",
                "normal": "none",
                "roughness": "none",
                "assignment_count": texture_color_space_count,
                "verified_after_reopen": True,
            },
            "metric_uv_sha256": specification.metric_uv_sha256,
            "blend_graph_sha256": specification.blend_graph_sha256,
            "material_bindings_sha256": (
                specification.material_bindings_sha256
            ),
            "evidence_bindings_sha256": (
                specification.evidence_bindings_sha256
            ),
            "derived_spatial_inputs_sha256": derived_sha256,
            "derived_spatial_input_count": len(derived_records),
            "native_stage_reopen_succeeded": True,
            "surface_output_connected": True,
            "all_required_branches_surface_reachable": True,
            "uniform_fallback_present": False,
            "material_metric_uv_uses_world_position": True,
            "spatial_mask_uv_uses_halo_sampling_bounds": True,
            "spatial_mask_address_mode": "clamp",
            "monolithic_generated_mask_atlas_present": False,
            "source_colour_feeds_base_color": False,
            "source_geometry_creates_rendered_objects": False,
            "reachable_shader_prim_count": reachable_count,
        }
        return {
            "derived_spatial_inputs": list(derived_records),
            "native_validation": native_validation,
        }


def _native_file_record(
    *,
    artifact_root: Path,
    path: Path,
) -> dict[str, object]:
    if (
        not _inside(artifact_root, path)
        or not path.is_file()
        or path.is_symlink()
        or path.suffix.casefold() not in MATERIAL_SUFFIXES
    ):
        raise TerrainPbrContractError(
            "native terrain payload must be a regular USD layer"
        )
    return {
        "path": path.resolve().relative_to(
            artifact_root.resolve()
        ).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


class _NativeMaterialXGroundBackend:
    """Author a bounded MaterialX layer per tile and a payload-only index.

    Each surface graph sees exactly five evidence bindings for one tile.  The
    seven byte-locked PBR roles remain shared external assets.  Consequently,
    opening the index with payloads unloaded does not compile a scene-wide
    shader graph, regardless of tile count.
    """

    def author(
        self,
        *,
        plan: TerrainPbrPlan,
        specification: CompositeGroundMaterialSpec,
        bundle_root: Path,
        evidence_root: Path,
        artifact_root: Path,
        output_path: Path,
        final_output_path: Path,
        derived_output_root: Path,
        final_derived_output_root: Path,
    ) -> Mapping[str, object]:
        try:
            from pxr import Sdf, Usd, UsdGeom
        except ImportError as exc:
            raise RuntimeError(
                "native tiled ground authoring requires the pinned Kit "
                "Python with OpenUSD payload support"
            ) from exc

        derived_output_root.mkdir(parents=True, exist_ok=False)
        tile_layer_root = derived_output_root / "tile-materials"
        tile_layer_root.mkdir()
        tile_input_root = derived_output_root / "spatial-inputs"
        tile_input_root.mkdir()
        single_tile_backend = _NativeSingleTileMaterialXBackend()
        derived_records: list[dict[str, object]] = []
        payload_records: list[dict[str, object]] = []
        tile_validations: list[dict[str, object]] = []
        root_payloads: list[tuple[str, Path, Bounds2d]] = []

        for tile_id in specification.tile_ids:
            tile_bindings = specification.bindings_for_tile(tile_id)
            if (
                len(tile_bindings) != len(EVIDENCE_SEMANTICS)
                or {str(binding["semantic"]) for binding in tile_bindings}
                != set(EVIDENCE_SEMANTICS)
            ):
                raise TerrainPbrContractError(
                    f"{tile_id} payload does not have exactly five evidence "
                    "bindings"
                )
            tile_bounds_values = {
                tuple(float(value) for value in binding["tile_bounds_m"])
                for binding in tile_bindings
            }
            if len(tile_bounds_values) != 1:
                raise TerrainPbrContractError(
                    f"{tile_id} payload contains conflicting tile bounds"
                )
            tile_bounds = Bounds2d(*next(iter(tile_bounds_values)))
            ordered_bindings = tuple(
                sorted(
                    tile_bindings,
                    key=lambda binding: str(binding["binding_id"]),
                )
            )
            tile_specification = CompositeGroundMaterialSpec(
                plan_sha256=specification.plan_sha256,
                material_prim_path="/Ground",
                material_roles=specification.material_roles,
                spatial_bindings=ordered_bindings,
                metric_uv_sha256=specification.metric_uv_sha256,
                blend_graph_sha256=specification.blend_graph_sha256,
                material_bindings_sha256=(
                    specification.material_bindings_sha256
                ),
                evidence_bindings_sha256=_canonical_sha256(
                    [dict(binding) for binding in ordered_bindings]
                ),
            )
            token = (
                f"{_safe_usd_identifier(tile_id)}-"
                f"{hashlib.sha256(tile_id.encode()).hexdigest()[:10]}"
            )
            tile_layer_path = tile_layer_root / f"{token}.usdc"
            tile_derived_root = tile_input_root / token
            final_tile_layer_path = (
                final_derived_output_root
                / "tile-materials"
                / f"{token}.usdc"
            )
            final_tile_derived_root = (
                final_derived_output_root / "spatial-inputs" / token
            )
            tile_result = single_tile_backend.author(
                plan=plan,
                specification=tile_specification,
                bundle_root=bundle_root,
                evidence_root=evidence_root,
                artifact_root=artifact_root,
                output_path=tile_layer_path,
                final_output_path=final_tile_layer_path,
                derived_output_root=tile_derived_root,
                final_derived_output_root=final_tile_derived_root,
            )
            tile_derived = tile_result.get("derived_spatial_inputs")
            tile_native = tile_result.get("native_validation")
            if not isinstance(tile_derived, list) or not isinstance(
                tile_native,
                Mapping,
            ):
                raise TerrainPbrContractError(
                    f"{tile_id} native payload inspection is incomplete"
                )
            expected_binding_ids = [
                str(binding["binding_id"])
                for binding in ordered_bindings
            ]
            reachable = tile_native.get("reachable_shader_prim_count")
            spatial_images = sum(
                2 if binding["semantic"] == "elevation" else 1
                for binding in ordered_bindings
            )
            if (
                tile_native.get("connected_material_roles")
                != list(PBR_MATERIAL_ROLES)
                or tile_native.get("reachable_quality_features")
                != ["slope_projection", "world_macro_variation"]
                or tile_native.get("texture_color_space_contract")
                != {
                    "base_color": "srgb_texture",
                    "normal": "none",
                    "roughness": "none",
                    "assignment_count": 30,
                    "verified_after_reopen": True,
                }
                or tile_native.get("connected_spatial_binding_ids")
                != expected_binding_ids
                or tile_native.get("surface_output_connected") is not True
                or tile_native.get("uniform_fallback_present") is not False
                or isinstance(reachable, bool)
                or not isinstance(reachable, int)
                or reachable <= 0
                or reachable > MAX_REACHABLE_SHADER_PRIMS_PER_TILE
                or spatial_images > MAX_SPATIAL_IMAGE_NODES_PER_TILE
            ):
                raise TerrainPbrContractError(
                    f"{tile_id} MaterialX payload is incomplete or too large"
                )
            layer_record = {
                "tile_id": tile_id,
                "tile_ref": tile_id,
                "tile_bounds_m": tile_bounds.as_list(),
                "prim_path": "/Ground",
                **_native_file_record(
                    artifact_root=artifact_root,
                    path=tile_layer_path,
                ),
            }
            payload_records.append(layer_record)
            derived_records.extend(tile_derived)
            tile_validations.append(
                {
                    "tile_id": tile_id,
                    "tile_ref": tile_id,
                    "tile_bounds_m": tile_bounds.as_list(),
                    "material_prim_path": "/Ground",
                    "connected_material_roles": list(PBR_MATERIAL_ROLES),
                    "reachable_quality_features": [
                        "slope_projection",
                        "world_macro_variation",
                    ],
                    "texture_color_space_contract": dict(
                        tile_native["texture_color_space_contract"]
                    ),
                    "connected_spatial_binding_ids": expected_binding_ids,
                    "connected_mask_binding_ids": [
                        binding_id
                        for binding_id in expected_binding_ids
                        if not binding_id.endswith(":elevation")
                    ],
                    "connected_relief_binding_ids": [
                        binding_id
                        for binding_id in expected_binding_ids
                        if binding_id.endswith(":elevation")
                    ],
                    "surface_output_connected": True,
                    "all_required_branches_surface_reachable": True,
                    "material_metric_uv_uses_world_position": True,
                    "spatial_mask_uv_uses_halo_sampling_bounds": True,
                    "spatial_mask_address_mode": "clamp",
                    "spatial_image_node_count": spatial_images,
                    "reachable_shader_prim_count": reachable,
                    "uniform_fallback_present": False,
                }
            )
            root_payloads.append((token, tile_layer_path, tile_bounds))

        root_stage = Usd.Stage.CreateNew(str(output_path))
        if root_stage is None:
            raise RuntimeError(
                "Kit could not create the payload-index ground layer"
            )
        root_prim = UsdGeom.Scope.Define(
            root_stage,
            specification.material_prim_path,
        )
        root_stage.SetDefaultPrim(root_prim.GetPrim())
        tiles_scope = UsdGeom.Scope.Define(
            root_stage,
            f"{specification.material_prim_path}/Tiles",
        )
        root_prim.GetPrim().SetCustomDataByKey(
            "fireviewer:authoringBackend",
            "materialx_payload_tiles_shared_pbr_library_v2",
        )
        root_prim.GetPrim().SetCustomDataByKey(
            "fireviewer:terrainPbrPlanSha256",
            plan.fingerprint,
        )
        root_prim.GetPrim().SetCustomDataByKey(
            "fireviewer:tilePayloadCount",
            len(root_payloads),
        )
        for token, tile_layer_path, bounds in root_payloads:
            payload_prim = root_stage.DefinePrim(
                f"{tiles_scope.GetPath()}/{token}"
            )
            payload_prim.GetPayloads().AddPayload(
                Sdf.Payload(
                    _relative_asset_path(
                        layer_path=final_output_path,
                        asset_path=(
                            final_derived_output_root
                            / "tile-materials"
                            / tile_layer_path.name
                        ),
                    ),
                    "/Ground",
                )
            )
            payload_prim.SetCustomDataByKey(
                "fireviewer:tileBoundsM",
                bounds.as_list(),
            )
        root_stage.GetRootLayer().Save()
        root_stage = None

        reopened = Usd.Stage.Open(str(output_path), load=Usd.Stage.LoadNone)
        if reopened is None:
            raise RuntimeError(
                "Kit could not reopen the payload-index ground layer"
            )
        reopened_root = reopened.GetPrimAtPath(
            specification.material_prim_path
        )
        reopened_tiles = reopened.GetPrimAtPath(
            f"{specification.material_prim_path}/Tiles"
        )
        payload_prims = list(reopened_tiles.GetChildren())
        root_shader_count = sum(
            1
            for prim in reopened.Traverse()
            if prim.GetTypeName() == "Shader"
        )
        if (
            not reopened_root.IsValid()
            or reopened_root.GetTypeName() != "Scope"
            or len(payload_prims) != len(root_payloads)
            or any(not prim.HasPayload() for prim in payload_prims)
            or root_shader_count != 0
        ):
            raise RuntimeError(
                "ground index is not an unloaded one-payload-per-tile stage"
            )

        derived_records.sort(key=lambda record: str(record["binding_id"]))
        payload_records.sort(key=lambda record: str(record["tile_id"]))
        tile_validations.sort(key=lambda record: str(record["tile_id"]))
        uv_measurement = _measure_metric_uv_continuity(plan)
        mask_edge_measurement = _measure_native_mask_edge_continuity(
            plan=plan,
            artifact_root=artifact_root,
            derived_records=derived_records,
        )
        derived_sha256 = _canonical_sha256(derived_records)
        payload_sha256 = _canonical_sha256(payload_records)
        native_validation = {
            "inspector_id": NATIVE_COMPOSITE_INSPECTOR_ID,
            "render_context": "mtlx",
            "ground_index_prim_path": specification.material_prim_path,
            "ground_index_prim_type": "UsdGeom.Scope",
            "topology": "payload_tiled_materials_shared_pbr_library",
            "shared_pbr_library_count": 1,
            "tile_payload_count": len(payload_records),
            "material_graph_count": len(payload_records),
            "root_shader_prim_count_with_payloads_unloaded": root_shader_count,
            "connected_material_roles": list(PBR_MATERIAL_ROLES),
            "connected_spatial_binding_ids": [
                str(binding["binding_id"])
                for binding in specification.spatial_bindings
            ],
            "connected_mask_binding_ids": list(
                specification.mask_binding_ids
            ),
            "connected_relief_binding_ids": list(
                specification.relief_binding_ids
            ),
            "world_metric_uv_roles": list(PBR_MATERIAL_ROLES),
            "reachable_quality_features": [
                "slope_projection",
                "world_macro_variation",
            ],
            "texture_color_space_contract": {
                "base_color": "srgb_texture",
                "normal": "none",
                "roughness": "none",
                "assignments_per_tile": 30,
                "all_tiles_verified_after_reopen": True,
            },
            "metric_uv_sha256": specification.metric_uv_sha256,
            "metric_uv_continuity_measurement": uv_measurement,
            "spatial_mask_edge_continuity_measurement": (
                mask_edge_measurement
            ),
            "blend_graph_sha256": specification.blend_graph_sha256,
            "material_bindings_sha256": (
                specification.material_bindings_sha256
            ),
            "evidence_bindings_sha256": (
                specification.evidence_bindings_sha256
            ),
            "derived_spatial_inputs_sha256": derived_sha256,
            "derived_spatial_input_count": len(derived_records),
            "tile_payload_layers_sha256": payload_sha256,
            "native_stage_reopen_succeeded": True,
            "all_tile_surface_outputs_connected": True,
            "all_required_branches_surface_reachable": True,
            "uniform_fallback_present": False,
            "single_graph_for_all_tiles_present": False,
            "monolithic_generated_mask_atlas_present": False,
            "source_colour_feeds_base_color": False,
            "source_geometry_creates_rendered_objects": False,
            "maximum_spatial_image_nodes_per_tile": max(
                (
                    int(record["spatial_image_node_count"])
                    for record in tile_validations
                ),
                default=0,
            ),
            "maximum_reachable_shader_prims_per_tile": max(
                (
                    int(record["reachable_shader_prim_count"])
                    for record in tile_validations
                ),
                default=0,
            ),
            "tile_validations": tile_validations,
        }
        return {
            "derived_spatial_inputs": derived_records,
            "tile_payload_layers": payload_records,
            "native_validation": native_validation,
        }


def author_composite_ground_material(
    *,
    plan: TerrainPbrPlan,
    artifact_root: Path,
    bundle_root: Path,
    evidence_root: Path,
    output_relative_path: str,
    receipt_relative_path: str,
    backend: CompositeGroundAuthoringBackend | None = None,
) -> CompositeGroundMaterialArtifact:
    """Author the real composite MaterialX USD and its native receipt.

    The default backend imports Kit/OpenUSD/GDAL only when called.  An injected
    backend exists solely for deterministic CI tests or another native backend
    implementing the same fail-closed interface.
    """

    root = artifact_root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise TerrainPbrContractError(
            "composite ground artifact root must already exist"
        )
    output_relative = _safe_relative_path(
        output_relative_path,
        label="composite ground output",
    )
    receipt_relative = _safe_relative_path(
        receipt_relative_path,
        label="composite ground receipt",
    )
    output_path = root.joinpath(*output_relative.parts)
    receipt_path = root.joinpath(*receipt_relative.parts)
    if (
        not _inside(root, output_path)
        or not _inside(root, receipt_path)
        or output_path.suffix.casefold() not in MATERIAL_SUFFIXES
        or receipt_path.suffix.casefold() != ".json"
        or output_path == receipt_path
    ):
        raise TerrainPbrContractError(
            "composite ground output must be USD and receipt must be JSON "
            "inside artifact root"
        )
    if (
        output_path.parent != receipt_path.parent
        or output_path.parent == root
    ):
        raise TerrainPbrContractError(
            "composite ground output and receipt must share one dedicated "
            "package directory below the artifact root"
        )
    if output_path.exists() or receipt_path.exists():
        if output_path.is_file() and receipt_path.is_file():
            return validate_composite_ground_material_artifact(
                plan=plan,
                artifact_root=root,
                bundle_root=bundle_root,
                evidence_root=evidence_root,
                authoring_receipt_path=receipt_path,
            )
        package_entries = (
            set(output_path.parent.iterdir())
            if output_path.parent.is_dir()
            and not output_path.parent.is_symlink()
            else set()
        )
        allowed_partial = {
            output_path,
            output_path.with_name(f"{output_path.stem}.inputs"),
        }
        if (
            package_entries
            and package_entries.issubset(allowed_partial)
            and not receipt_path.exists()
        ):
            quarantine = output_path.parent.with_name(
                f".{output_path.parent.name}.incomplete-{uuid.uuid4().hex}"
            )
            os.replace(output_path.parent, quarantine)
        else:
            raise TerrainPbrContractError(
                "composite ground package is partial or contains unowned "
                "content; refusing to overwrite it"
            )
    _validate_plan_dependency_files(
        plan=plan,
        bundle_root=bundle_root,
        evidence_root=evidence_root,
    )
    specification = build_composite_ground_material_spec(plan)
    output_path.parent.parent.mkdir(parents=True, exist_ok=True)
    final_derived_output_root = output_path.with_name(
        f"{output_path.stem}.inputs"
    )
    if output_path.parent.exists() or final_derived_output_root.exists():
        raise TerrainPbrContractError(
            "composite ground package directory already exists"
        )
    staging_artifact_root = root / (
        f".terrain-pbr-{uuid.uuid4().hex}.staging"
    )
    staging_output = staging_artifact_root.joinpath(*output_relative.parts)
    staging_receipt = staging_artifact_root.joinpath(
        *receipt_relative.parts
    )
    derived_output_root = staging_output.with_name(
        f"{staging_output.stem}.inputs"
    )
    staging_output.parent.mkdir(parents=True)
    authoring_backend = (
        backend if backend is not None else _NativeMaterialXGroundBackend()
    )
    try:
        result = authoring_backend.author(
            plan=plan,
            specification=specification,
            bundle_root=bundle_root.resolve(),
            evidence_root=evidence_root.resolve(),
            artifact_root=staging_artifact_root,
            output_path=staging_output,
            final_output_path=output_path,
            derived_output_root=derived_output_root,
            final_derived_output_root=final_derived_output_root,
        )
        if (
            not isinstance(result, Mapping)
            or not staging_output.is_file()
            or staging_output.is_symlink()
            or staging_output.stat().st_size <= 0
        ):
            raise TerrainPbrContractError(
                "native backend produced no composite ground USD"
            )
        derived_records = _validate_derived_spatial_inputs(
            artifact_root=staging_artifact_root,
            specification=specification,
            payload=result.get("derived_spatial_inputs"),
        )
        tile_payloads = _validate_tile_material_payloads(
            artifact_root=staging_artifact_root,
            specification=specification,
            payload=result.get("tile_payload_layers"),
        )
        native_validation = result.get("native_validation")
        if not isinstance(native_validation, Mapping):
            raise TerrainPbrContractError(
                "native backend produced no inspected graph metrics"
            )
        derived_sha256 = _canonical_sha256(derived_records)
        tile_payloads_sha256 = _canonical_sha256(tile_payloads)
        if (
            native_validation.get("derived_spatial_inputs_sha256")
            != derived_sha256
            or native_validation.get("tile_payload_layers_sha256")
            != tile_payloads_sha256
        ):
            raise TerrainPbrContractError(
                "native backend inspection is stale against its tiled inputs "
                "or material payloads"
            )
        receipt = {
            "schema_version": COMPOSITE_AUTHORING_SCHEMA_VERSION,
            "state": NATIVE_GROUND_STATE,
            "terrain_pbr_plan_sha256": plan.fingerprint,
            "specification_sha256": specification.fingerprint,
            "metric_uv_sha256": specification.metric_uv_sha256,
            "blend_graph_sha256": specification.blend_graph_sha256,
            "material_bindings_sha256": (
                specification.material_bindings_sha256
            ),
            "evidence_bindings_sha256": (
                specification.evidence_bindings_sha256
            ),
            "derived_spatial_inputs": list(derived_records),
            "derived_spatial_inputs_sha256": derived_sha256,
            "tile_material_payloads": list(tile_payloads),
            "tile_material_payloads_sha256": tile_payloads_sha256,
            "ground_material": {
                "path": output_relative.as_posix(),
                "sha256": _sha256_file(staging_output),
                "size_bytes": staging_output.stat().st_size,
                "prim_path": specification.material_prim_path,
            },
            "native_validation": dict(native_validation),
        }
        _atomic_write_json(staging_receipt, receipt)
        validate_composite_ground_material_artifact(
            plan=plan,
            artifact_root=staging_artifact_root,
            bundle_root=bundle_root,
            evidence_root=evidence_root,
            authoring_receipt_path=staging_receipt,
        )
        staging_package = staging_output.parent
        os.replace(staging_package, output_path.parent)
        shutil.rmtree(staging_artifact_root)
        return validate_composite_ground_material_artifact(
            plan=plan,
            artifact_root=root,
            bundle_root=bundle_root,
            evidence_root=evidence_root,
            authoring_receipt_path=receipt_path,
        )
    except BaseException:
        if (
            staging_artifact_root.exists()
            and _inside(root, staging_artifact_root)
        ):
            shutil.rmtree(staging_artifact_root, ignore_errors=True)
        raise


def _validate_plan_dependency_files(
    *,
    plan: TerrainPbrPlan,
    bundle_root: Path,
    evidence_root: Path,
) -> None:
    bundle = bundle_root.resolve()
    evidence = evidence_root.resolve()
    if not bundle.is_dir() or not evidence.is_dir():
        raise TerrainPbrContractError(
            "ground material validation requires bundle and evidence roots"
        )
    current_library = load_locked_material_library(
        bundle_root=bundle,
        manifest_path=bundle.joinpath(
            *_safe_relative_path(
                plan.material_library.manifest_path,
                label="ground material manifest",
            ).parts
        ),
    )
    if current_library != plan.material_library:
        raise TerrainPbrContractError(
            "ground material library changed after terrain planning"
        )
    for material in plan.material_library.materials:
        for lock in (
            material.material_file,
            *(texture.lock for texture in material.textures),
        ):
            relative = _safe_relative_path(
                lock.path,
                label=f"{material.role} dependency",
            )
            path = bundle.joinpath(*relative.parts)
            if (
                not _inside(bundle, path)
                or not path.is_file()
                or path.is_symlink()
                or _sha256_file(path) != lock.sha256
                or path.stat().st_size != lock.size_bytes
            ):
                raise TerrainPbrContractError(
                    f"{material.role} ground-material dependency drifted"
                )
    for tile in plan.tiles:
        for source in tile.evidence:
            _validate_evidence_file(root=evidence, source=source)


def _validate_coverage_metrics(
    *,
    semantic: str,
    payload: object,
) -> dict[str, object]:
    expected_keys = (
        {
            "elevation_source",
            "slope_source",
            "roughness_source",
            "relief_slope",
            "relief_roughness",
        }
        if semantic == "elevation"
        else {"classified_source", "feathered_mask"}
    )
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise TerrainPbrContractError(
            "derived spatial input lacks complete finite-coverage metrics"
        )
    normalized: dict[str, object] = {}
    for role in sorted(expected_keys):
        metric = payload[role]
        if not isinstance(metric, dict):
            raise TerrainPbrContractError(
                "finite-coverage metric must be an object"
            )
        sample_count = metric.get("sample_count")
        finite_count = metric.get("finite_count")
        nodata_count = metric.get("nodata_count")
        try:
            minimum = float(metric.get("minimum", float("nan")))
            maximum = float(metric.get("maximum", float("nan")))
            finite_fraction = float(
                metric.get("finite_fraction", float("nan"))
            )
        except (TypeError, ValueError) as exc:
            raise TerrainPbrContractError(
                "finite-coverage metric contains non-numeric values"
            ) from exc
        normalized_role = role in {
            "relief_slope",
            "relief_roughness",
            "feathered_mask",
        }
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count <= 0
            or finite_count != sample_count
            or nodata_count != 0
            or finite_fraction != 1.0
            or not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or maximum < minimum
            or (
                normalized_role
                and (minimum < 0.0 or maximum > 1.0)
            )
        ):
            raise TerrainPbrContractError(
                "derived spatial input contains nodata or invalid values"
            )
        normalized[role] = {
            "sample_count": sample_count,
            "finite_count": finite_count,
            "nodata_count": 0,
            "finite_fraction": 1.0,
            "minimum": minimum,
            "maximum": maximum,
        }
    return normalized


def _validate_derived_spatial_inputs(
    *,
    artifact_root: Path,
    specification: CompositeGroundMaterialSpec,
    payload: object,
) -> tuple[dict[str, object], ...]:
    if not isinstance(payload, list):
        raise TerrainPbrContractError(
            "native composite receipt lacks derived tiled spatial inputs"
        )
    expected = {
        str(binding["binding_id"]): binding
        for binding in specification.spatial_bindings
    }
    records: dict[str, dict[str, object]] = {}
    seen_paths: set[str] = set()
    for raw_record in payload:
        if not isinstance(raw_record, dict):
            raise TerrainPbrContractError(
                "derived spatial input receipt entry must be an object"
            )
        binding_id = str(raw_record.get("binding_id", ""))
        binding = expected.get(binding_id)
        if binding is None or binding_id in records:
            raise TerrainPbrContractError(
                "derived spatial input binding is unknown or duplicated"
            )
        if (
            raw_record.get("tile_id") != binding["tile_id"]
            or raw_record.get("tile_ref") != binding["tile_ref"]
            or raw_record.get("tile_bounds_m")
            != list(binding["tile_bounds_m"])
            or raw_record.get("source_tile_bounds_m")
            != list(binding["source_tile_bounds_m"])
            or raw_record.get("sampling_bounds_m")
            != list(binding["sampling_bounds_m"])
            or raw_record.get("source_sampling_bounds_m")
            != list(binding["source_sampling_bounds_m"])
            or raw_record.get("halo_m") != binding["halo_m"]
            or raw_record.get("halo_edges_m")
            != list(binding["halo_edges_m"])
            or raw_record.get("source_bounds_m")
            != list(binding["source_bounds_m"])
            or raw_record.get("scene_origin_source_m")
            != list(binding["scene_origin_source_m"])
            or raw_record.get("source_sha256") != binding["sha256"]
            or raw_record.get("source_path") != binding["path"]
            or raw_record.get("semantic") != binding["semantic"]
        ):
            raise TerrainPbrContractError(
                "derived spatial input is stale against its source lock"
            )
        assets = raw_record.get("assets")
        required_assets = (
            {"relief_slope", "relief_roughness"}
            if binding["semantic"] == "elevation"
            else {"mask"}
        )
        if not isinstance(assets, dict) or set(assets) != required_assets:
            raise TerrainPbrContractError(
                "derived spatial input has an incomplete tiled asset set"
            )
        normalized_assets: dict[str, object] = {}
        for asset_role in sorted(required_assets):
            asset = assets[asset_role]
            if not isinstance(asset, dict):
                raise TerrainPbrContractError(
                    "derived spatial input asset lock must be an object"
                )
            relative = _safe_relative_path(
                str(asset.get("path", "")),
                label=f"{binding_id} {asset_role}",
            )
            path = artifact_root.joinpath(*relative.parts)
            if (
                not _inside(artifact_root, path)
                or not path.is_file()
                or path.is_symlink()
                or path.suffix.casefold() not in {".tif", ".tiff", ".exr"}
            ):
                raise TerrainPbrContractError(
                    "derived spatial input must be a regular tiled float image"
                )
            lock = FileLock(
                path=relative.as_posix(),
                sha256=_require_sha256(
                    str(asset.get("sha256", "")),
                    label=f"{binding_id} {asset_role}",
                ),
                size_bytes=asset.get("size_bytes"),
            )
            if (
                lock.path.casefold() in seen_paths
                or _sha256_file(path) != lock.sha256
                or path.stat().st_size != lock.size_bytes
            ):
                raise TerrainPbrContractError(
                    "derived spatial input hash, size or uniqueness check failed"
                )
            seen_paths.add(lock.path.casefold())
            normalized_assets[asset_role] = {
                "path": lock.path,
                "sha256": lock.sha256,
                "size_bytes": lock.size_bytes,
            }
        expected_representation = (
            "slope_and_roughness_from_locked_heightfield"
            if binding["semantic"] == "elevation"
            else "feathered_classification_mask"
        )
        if raw_record.get("representation") != expected_representation:
            raise TerrainPbrContractError(
                "derived spatial input representation is not authorable"
            )
        coverage_metrics = _validate_coverage_metrics(
            semantic=str(binding["semantic"]),
            payload=raw_record.get("coverage_metrics"),
        )
        records[binding_id] = {
            "binding_id": binding_id,
            "tile_id": binding["tile_id"],
            "tile_ref": binding["tile_ref"],
            "semantic": binding["semantic"],
            "source_path": binding["path"],
            "source_sha256": binding["sha256"],
            "tile_bounds_m": list(binding["tile_bounds_m"]),
            "source_tile_bounds_m": list(binding["source_tile_bounds_m"]),
            "sampling_bounds_m": list(binding["sampling_bounds_m"]),
            "source_sampling_bounds_m": list(
                binding["source_sampling_bounds_m"]
            ),
            "halo_m": binding["halo_m"],
            "halo_edges_m": list(binding["halo_edges_m"]),
            "source_bounds_m": list(binding["source_bounds_m"]),
            "scene_origin_source_m": list(
                binding["scene_origin_source_m"]
            ),
            "representation": expected_representation,
            "coverage_metrics": coverage_metrics,
            "assets": normalized_assets,
        }
    if set(records) != set(expected):
        raise TerrainPbrContractError(
            "derived spatial inputs do not cover every terrain evidence binding"
        )
    return tuple(records[binding_id] for binding_id in sorted(records))


def _validate_tile_material_payloads(
    *,
    artifact_root: Path,
    specification: CompositeGroundMaterialSpec,
    payload: object,
) -> tuple[dict[str, object], ...]:
    if not isinstance(payload, list):
        raise TerrainPbrContractError(
            "native ground receipt lacks tile material payload layers"
        )
    expected_bounds = {
        tile_id: list(
            specification.bindings_for_tile(tile_id)[0]["tile_bounds_m"]
        )
        for tile_id in specification.tile_ids
    }
    records: dict[str, dict[str, object]] = {}
    seen_paths: set[str] = set()
    for raw_record in payload:
        if not isinstance(raw_record, dict):
            raise TerrainPbrContractError(
                "tile material payload lock must be an object"
            )
        tile_id = str(raw_record.get("tile_id", ""))
        if tile_id not in expected_bounds or tile_id in records:
            raise TerrainPbrContractError(
                "tile material payload is unknown or duplicated"
            )
        relative = _safe_relative_path(
            str(raw_record.get("path", "")),
            label=f"{tile_id} material payload",
        )
        path = artifact_root.joinpath(*relative.parts)
        if (
            not _inside(artifact_root, path)
            or not path.is_file()
            or path.is_symlink()
            or path.suffix.casefold() not in MATERIAL_SUFFIXES
        ):
            raise TerrainPbrContractError(
                "tile material payload must be a regular USD layer"
            )
        lock = FileLock(
            path=relative.as_posix(),
            sha256=_require_sha256(
                str(raw_record.get("sha256", "")),
                label=f"{tile_id} material payload",
            ),
            size_bytes=raw_record.get("size_bytes"),
        )
        if (
            lock.path.casefold() in seen_paths
            or _sha256_file(path) != lock.sha256
            or path.stat().st_size != lock.size_bytes
            or raw_record.get("tile_bounds_m") != expected_bounds[tile_id]
            or raw_record.get("tile_ref") != tile_id
            or raw_record.get("prim_path") != "/Ground"
        ):
            raise TerrainPbrContractError(
                "tile material payload lock, bounds or prim path drifted"
            )
        seen_paths.add(lock.path.casefold())
        records[tile_id] = {
            "tile_id": tile_id,
            "tile_ref": tile_id,
            "tile_bounds_m": expected_bounds[tile_id],
            "path": lock.path,
            "sha256": lock.sha256,
            "size_bytes": lock.size_bytes,
            "prim_path": "/Ground",
        }
    if set(records) != set(expected_bounds):
        raise TerrainPbrContractError(
            "tile material payloads do not cover every terrain tile"
        )
    return tuple(records[tile_id] for tile_id in sorted(records))


def _validate_mask_edge_measurement(
    *,
    plan: TerrainPbrPlan,
    payload: object,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TerrainPbrContractError(
            "native receipt lacks measured mask-edge continuity"
        )
    uv_measurement = _measure_metric_uv_continuity(plan)
    expected_keys = {
        (
            str(seam["tile_a"]),
            str(seam["tile_b"]),
            semantic,
        )
        for seam in uv_measurement["seams"]
        for semantic in CLASSIFICATION_SEMANTICS
    }
    raw_measurements = payload.get("measurements")
    if not isinstance(raw_measurements, list):
        raise TerrainPbrContractError(
            "mask-edge continuity measurements must be a list"
        )
    observed: set[tuple[str, str, str]] = set()
    errors: list[float] = []
    for measurement in raw_measurements:
        if not isinstance(measurement, dict):
            raise TerrainPbrContractError(
                "mask-edge continuity entry must be an object"
            )
        key = (
            str(measurement.get("tile_a", "")),
            str(measurement.get("tile_b", "")),
            str(measurement.get("semantic", "")),
        )
        try:
            error = float(
                measurement.get(
                    "maximum_absolute_mask_error",
                    float("nan"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise TerrainPbrContractError(
                "mask-edge continuity error must be numeric"
            ) from exc
        if (
            key not in expected_keys
            or key in observed
            or measurement.get("axis") not in {"x", "y"}
            or measurement.get("sample_count") != MASK_EDGE_SAMPLE_COUNT
            or not math.isfinite(error)
            or error < 0.0
            or error > MASK_EDGE_CONTINUITY_TOLERANCE
        ):
            raise TerrainPbrContractError(
                "mask-edge continuity measurement is incomplete or failed"
            )
        observed.add(key)
        errors.append(error)
    without_sha = {
        key: value
        for key, value in payload.items()
        if key != "measurement_sha256"
    }
    maximum = max(errors, default=0.0)
    if (
        observed != expected_keys
        or payload.get("algorithm")
        != "gdal_halo_bilinear_shared_edge_v1"
        or payload.get("adjacent_pair_count")
        != uv_measurement["adjacent_pair_count"]
        or payload.get("semantic_count_per_edge")
        != len(CLASSIFICATION_SEMANTICS)
        or payload.get("samples_per_semantic_edge")
        != MASK_EDGE_SAMPLE_COUNT
        or payload.get("tolerance") != MASK_EDGE_CONTINUITY_TOLERANCE
        or payload.get("maximum_absolute_mask_error") != maximum
        or payload.get("measurement_sha256")
        != _canonical_sha256(without_sha)
    ):
        raise TerrainPbrContractError(
            "mask-edge continuity measurement is stale or unverified"
        )
    return dict(payload)


def validate_composite_ground_material_artifact(
    *,
    plan: TerrainPbrPlan,
    artifact_root: Path,
    bundle_root: Path,
    evidence_root: Path,
    authoring_receipt_path: Path,
) -> CompositeGroundMaterialArtifact:
    """Verify a native receipt, payload index and every tile material layer.

    The receipt must come from the native graph inspector identified by
    :data:`NATIVE_COMPOSITE_INSPECTOR_ID`.  This pure-Python gate rechecks all
    file locks and rejects a scene-wide graph, oversized tile graph, uniform
    fallback, disconnected role, missing mask or monolithic atlas.
    """

    root = artifact_root.resolve()
    receipt_path = authoring_receipt_path.resolve()
    if (
        not root.is_dir()
        or not receipt_path.is_file()
        or receipt_path.is_symlink()
        or not _inside(root, receipt_path)
    ):
        raise TerrainPbrContractError(
            "composite ground receipt must be a regular file inside artifact root"
        )
    _validate_plan_dependency_files(
        plan=plan,
        bundle_root=bundle_root,
        evidence_root=evidence_root,
    )
    spec = build_composite_ground_material_spec(plan)
    receipt = _read_json(receipt_path, label="composite ground authoring receipt")
    native = receipt.get("native_validation")
    if not isinstance(native, dict):
        raise TerrainPbrContractError(
            "composite ground receipt lacks native validation"
        )
    if (
        receipt.get("schema_version") != COMPOSITE_AUTHORING_SCHEMA_VERSION
        or receipt.get("state")
        != NATIVE_GROUND_STATE
        or receipt.get("terrain_pbr_plan_sha256") != plan.fingerprint
        or receipt.get("specification_sha256") != spec.fingerprint
        or receipt.get("metric_uv_sha256") != spec.metric_uv_sha256
        or receipt.get("blend_graph_sha256") != spec.blend_graph_sha256
        or receipt.get("material_bindings_sha256")
        != spec.material_bindings_sha256
        or receipt.get("evidence_bindings_sha256")
        != spec.evidence_bindings_sha256
    ):
        raise TerrainPbrContractError(
            "composite ground receipt is stale or bound to another plan"
        )
    derived_spatial_inputs = _validate_derived_spatial_inputs(
        artifact_root=root,
        specification=spec,
        payload=receipt.get("derived_spatial_inputs"),
    )
    derived_sha256 = _canonical_sha256(derived_spatial_inputs)
    if (
        receipt.get("derived_spatial_inputs_sha256") != derived_sha256
        or native.get("derived_spatial_inputs_sha256") != derived_sha256
    ):
        raise TerrainPbrContractError(
            "derived spatial input inventory drifted from native validation"
        )
    tile_payloads = _validate_tile_material_payloads(
        artifact_root=root,
        specification=spec,
        payload=receipt.get("tile_material_payloads"),
    )
    tile_payloads_sha256 = _canonical_sha256(tile_payloads)
    if (
        receipt.get("tile_material_payloads_sha256")
        != tile_payloads_sha256
        or native.get("tile_payload_layers_sha256")
        != tile_payloads_sha256
    ):
        raise TerrainPbrContractError(
            "tile material payload inventory drifted from native validation"
        )
    material_record = receipt.get("ground_material")
    if not isinstance(material_record, dict):
        raise TerrainPbrContractError(
            "composite ground receipt lacks its payload-index artifact"
        )
    relative = _safe_relative_path(
        str(material_record.get("path", "")),
        label="composite ground payload index",
    )
    material_path = root.joinpath(*relative.parts)
    if (
        not _inside(root, material_path)
        or not material_path.is_file()
        or material_path.is_symlink()
        or material_path.suffix.casefold() not in MATERIAL_SUFFIXES
    ):
        raise TerrainPbrContractError(
            "composite ground payload index must be a regular USD artifact"
        )
    material_lock = FileLock(
        path=relative.as_posix(),
        sha256=_require_sha256(
            str(material_record.get("sha256", "")),
            label="composite ground payload index",
        ),
        size_bytes=material_record.get("size_bytes"),
    )
    if (
        _sha256_file(material_path) != material_lock.sha256
        or material_path.stat().st_size != material_lock.size_bytes
    ):
        raise TerrainPbrContractError(
            "composite ground payload index SHA-256 or size lock does not match"
        )
    prim_path = str(material_record.get("prim_path", ""))
    expected_binding_ids = tuple(
        str(binding["binding_id"]) for binding in spec.spatial_bindings
    )
    render_context = str(native.get("render_context", ""))
    expected_uv_measurement = _measure_metric_uv_continuity(plan)
    _validate_mask_edge_measurement(
        plan=plan,
        payload=native.get(
            "spatial_mask_edge_continuity_measurement"
        ),
    )
    raw_tile_validations = native.get("tile_validations")
    if not isinstance(raw_tile_validations, list):
        raise TerrainPbrContractError(
            "native receipt lacks per-tile material inspections"
        )
    expected_by_tile = {
        tile_id: spec.bindings_for_tile(tile_id)
        for tile_id in spec.tile_ids
    }
    observed_by_tile: dict[str, Mapping[str, object]] = {}
    reachable_counts: list[int] = []
    spatial_image_counts: list[int] = []
    for record in raw_tile_validations:
        if not isinstance(record, Mapping):
            raise TerrainPbrContractError(
                "native tile material inspection must be an object"
            )
        tile_id = str(record.get("tile_id", ""))
        bindings = expected_by_tile.get(tile_id)
        if bindings is None or tile_id in observed_by_tile:
            raise TerrainPbrContractError(
                "native tile material inspection is unknown or duplicated"
            )
        binding_ids = [
            str(binding["binding_id"]) for binding in bindings
        ]
        mask_ids = [
            binding_id
            for binding_id in binding_ids
            if not binding_id.endswith(":elevation")
        ]
        relief_ids = [
            binding_id
            for binding_id in binding_ids
            if binding_id.endswith(":elevation")
        ]
        reachable = record.get("reachable_shader_prim_count")
        spatial_images = record.get("spatial_image_node_count")
        if (
            record.get("tile_ref") != tile_id
            or record.get("tile_bounds_m")
            != list(bindings[0]["tile_bounds_m"])
            or record.get("material_prim_path") != "/Ground"
            or record.get("connected_material_roles")
            != list(PBR_MATERIAL_ROLES)
            or record.get("reachable_quality_features")
            != ["slope_projection", "world_macro_variation"]
            or record.get("texture_color_space_contract")
            != {
                "base_color": "srgb_texture",
                "normal": "none",
                "roughness": "none",
                "assignment_count": 30,
                "verified_after_reopen": True,
            }
            or record.get("connected_spatial_binding_ids") != binding_ids
            or record.get("connected_mask_binding_ids") != mask_ids
            or record.get("connected_relief_binding_ids") != relief_ids
            or record.get("surface_output_connected") is not True
            or record.get("all_required_branches_surface_reachable")
            is not True
            or record.get("material_metric_uv_uses_world_position")
            is not True
            or record.get("spatial_mask_uv_uses_halo_sampling_bounds")
            is not True
            or record.get("spatial_mask_address_mode") != "clamp"
            or record.get("uniform_fallback_present") is not False
            or isinstance(reachable, bool)
            or not isinstance(reachable, int)
            or reachable
            < len(PBR_MATERIAL_ROLES) + len(EVIDENCE_SEMANTICS) + 1
            or reachable > MAX_REACHABLE_SHADER_PRIMS_PER_TILE
            or isinstance(spatial_images, bool)
            or not isinstance(spatial_images, int)
            or spatial_images != MAX_SPATIAL_IMAGE_NODES_PER_TILE
        ):
            raise TerrainPbrContractError(
                "native tile graph is incomplete, uniform or unsafe: "
                "oversized or uses wrong UVs"
            )
        observed_by_tile[tile_id] = record
        reachable_counts.append(reachable)
        spatial_image_counts.append(spatial_images)
    if set(observed_by_tile) != set(expected_by_tile):
        raise TerrainPbrContractError(
            "native tile inspections do not cover every terrain tile"
        )
    exact_checks = (
        native.get("inspector_id") == NATIVE_COMPOSITE_INSPECTOR_ID,
        native.get("ground_index_prim_path")
        == prim_path
        == spec.material_prim_path,
        native.get("ground_index_prim_type") == "UsdGeom.Scope",
        native.get("topology")
        == "payload_tiled_materials_shared_pbr_library",
        native.get("shared_pbr_library_count") == 1,
        native.get("tile_payload_count") == len(spec.tile_ids),
        native.get("material_graph_count") == len(spec.tile_ids),
        native.get("root_shader_prim_count_with_payloads_unloaded") == 0,
        native.get("connected_material_roles") == list(PBR_MATERIAL_ROLES),
        native.get("connected_spatial_binding_ids")
        == list(expected_binding_ids),
        native.get("connected_mask_binding_ids")
        == list(spec.mask_binding_ids),
        native.get("connected_relief_binding_ids")
        == list(spec.relief_binding_ids),
        native.get("world_metric_uv_roles") == list(PBR_MATERIAL_ROLES),
        native.get("reachable_quality_features")
        == ["slope_projection", "world_macro_variation"],
        native.get("texture_color_space_contract")
        == {
            "base_color": "srgb_texture",
            "normal": "none",
            "roughness": "none",
            "assignments_per_tile": 30,
            "all_tiles_verified_after_reopen": True,
        },
        native.get("metric_uv_sha256") == spec.metric_uv_sha256,
        native.get("metric_uv_continuity_measurement")
        == expected_uv_measurement,
        native.get("blend_graph_sha256") == spec.blend_graph_sha256,
        native.get("material_bindings_sha256")
        == spec.material_bindings_sha256,
        native.get("evidence_bindings_sha256")
        == spec.evidence_bindings_sha256,
        native.get("derived_spatial_input_count")
        == len(spec.spatial_bindings),
        native.get("native_stage_reopen_succeeded") is True,
        native.get("all_tile_surface_outputs_connected") is True,
        native.get("all_required_branches_surface_reachable") is True,
        native.get("uniform_fallback_present") is False,
        native.get("single_graph_for_all_tiles_present") is False,
        native.get("monolithic_generated_mask_atlas_present") is False,
        native.get("source_colour_feeds_base_color") is False,
        native.get("source_geometry_creates_rendered_objects") is False,
        native.get("maximum_spatial_image_nodes_per_tile")
        == max(spatial_image_counts),
        native.get("maximum_reachable_shader_prims_per_tile")
        == max(reachable_counts),
        render_context in {"mdl", "mtlx"},
    )
    if not all(exact_checks):
        raise TerrainPbrContractError(
            "native composite ground graph is incomplete, uniform or unsafe"
        )
    receipt_relative = receipt_path.relative_to(root).as_posix()
    return CompositeGroundMaterialArtifact(
        ground_material=material_lock,
        material_prim_path=prim_path,
        authoring_receipt=FileLock(
            path=receipt_relative,
            sha256=_sha256_file(receipt_path),
            size_bytes=receipt_path.stat().st_size,
        ),
        plan_sha256=plan.fingerprint,
        specification_sha256=spec.fingerprint,
        render_context=render_context,
        tile_material_payloads=tile_payloads,
    )


def _locked_file_record(
    *,
    bundle_root: Path,
    manifest_parent: Path,
    record: object,
    label: str,
    suffixes: frozenset[str],
) -> tuple[FileLock, dict[str, object]]:
    path, relative, digest, size = _resolve_locked_file(
        root=bundle_root,
        parent=manifest_parent,
        record=record,
        label=label,
        suffixes=suffixes,
    )
    portable = path.resolve().relative_to(bundle_root.resolve()).as_posix()
    lock = FileLock(path=portable, sha256=digest, size_bytes=size)
    structural = {
        "path": relative.as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }
    return lock, structural


def load_locked_material_library(
    *,
    bundle_root: Path,
    manifest_path: Path,
) -> LockedMaterialLibrary:
    """Load the seven installed materials and prove their byte locks.

    The installer marker is mandatory.  This prevents callers from presenting
    an otherwise plausible standalone manifest that never passed the curated
    asset-bundle installation gate.
    """

    root = bundle_root.resolve()
    manifest = manifest_path.resolve()
    marker_path = root / INSTALL_MARKER
    if (
        not root.is_dir()
        or root.is_symlink()
        or not manifest.is_file()
        or manifest.is_symlink()
        or not _inside(root, manifest)
        or not marker_path.is_file()
        or marker_path.is_symlink()
    ):
        raise TerrainPbrContractError(
            "material manifest and install marker must be regular files inside "
            "the installed bundle"
        )
    payload = _read_json(manifest, label="material manifest")
    marker = _read_json(marker_path, label="asset bundle install marker")
    manifest_sha256 = _sha256_file(manifest)
    manifest_relative = manifest.relative_to(root).as_posix()
    bundle_sha256 = _require_sha256(
        str(marker.get("bundle_sha256", "")),
        label="installed bundle",
    )
    if (
        marker.get("state") != "ASSET_BUNDLE_INSTALLED"
        or marker.get("manifest_relative") != manifest_relative
        or marker.get("runtime_manifest_sha256") != manifest_sha256
        or marker.get("pbr_material_roles") != list(PBR_MATERIAL_ROLES)
    ):
        raise TerrainPbrContractError(
            "asset bundle install marker is stale or incomplete"
        )
    materials_payload = payload.get("pbr_materials")
    if not isinstance(materials_payload, dict) or set(materials_payload) != set(
        PBR_MATERIAL_ROLES
    ):
        raise TerrainPbrContractError(
            "material manifest must contain exactly the seven terrain PBR roles"
        )

    materials: list[LockedMaterial] = []
    structural: dict[str, object] = {}
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for role in PBR_MATERIAL_ROLES:
        record = materials_payload.get(role)
        if not isinstance(record, dict):
            raise TerrainPbrContractError(
                f"pbr_materials.{role} must be an object"
            )
        material_id = _require_stable_id(
            str(record.get("material_id", "")),
            label=f"{role} material",
        )
        if material_id in seen_ids:
            raise TerrainPbrContractError("material identifiers must be unique")
        seen_ids.add(material_id)
        material_lock, material_structural = _locked_file_record(
            bundle_root=root,
            manifest_parent=manifest.parent,
            record=record.get("material_file"),
            label=f"pbr_materials.{role}.material_file",
            suffixes=MATERIAL_SUFFIXES,
        )
        prim_path = str(record.get("material_prim_path", "")).strip()
        try:
            metric_repeat = float(record["metres_per_uv_tile"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TerrainPbrContractError(
                f"{role} material metric repeat is invalid"
            ) from exc
        textures_payload = record.get("textures")
        if (
            not isinstance(textures_payload, dict)
            or not set(PBR_REQUIRED_TEXTURES).issubset(textures_payload)
            or set(textures_payload)
            - set((*PBR_REQUIRED_TEXTURES, *PBR_OPTIONAL_TEXTURES))
        ):
            raise TerrainPbrContractError(
                f"{role} material requires base_color, normal and roughness "
                "with optional displacement"
            )
        textures: list[MaterialTexture] = []
        texture_structural: dict[str, object] = {}
        for texture_role in (
            *PBR_REQUIRED_TEXTURES,
            *(
                optional
                for optional in PBR_OPTIONAL_TEXTURES
                if optional in textures_payload
            ),
        ):
            texture_record = textures_payload[texture_role]
            texture_lock, locked_summary = _locked_file_record(
                bundle_root=root,
                manifest_parent=manifest.parent,
                record=texture_record,
                label=f"pbr_materials.{role}.textures.{texture_role}",
                suffixes=TEXTURE_SUFFIXES,
            )
            if not isinstance(texture_record, dict):
                raise TerrainPbrContractError(
                    f"{role} {texture_role} texture lock is invalid"
                )
            try:
                width_px = int(texture_record["width_px"])
                height_px = int(texture_record["height_px"])
            except (KeyError, TypeError, ValueError) as exc:
                raise TerrainPbrContractError(
                    f"{role} {texture_role} dimensions are invalid"
                ) from exc
            texture = MaterialTexture(
                role=texture_role,
                lock=texture_lock,
                width_px=width_px,
                height_px=height_px,
                color_space=str(texture_record.get("color_space", "")).casefold(),
            )
            textures.append(texture)
            texture_structural[texture_role] = {
                **locked_summary,
                "width_px": width_px,
                "height_px": height_px,
                "color_space": texture.color_space,
            }
        material = LockedMaterial(
            role=role,
            material_id=material_id,
            material_file=material_lock,
            material_prim_path=prim_path,
            metres_per_uv_tile=metric_repeat,
            textures=tuple(textures),
        )
        materials.append(material)
        structural[role] = {
            "material_id": material_id,
            "material_file": material_structural["path"],
            "material_file_sha256": material_structural["sha256"],
            "material_prim_path": prim_path,
            "metres_per_uv_tile": metric_repeat,
            "textures": texture_structural,
        }
        for lock in (
            material.material_file,
            *(texture.lock for texture in material.textures),
        ):
            key = lock.path.casefold()
            if key in seen_paths or lock.sha256 in seen_hashes:
                raise TerrainPbrContractError(
                    "material and texture files must be globally unique"
                )
            seen_paths.add(key)
            seen_hashes.add(lock.sha256)
    if marker.get("pbr_materials_sha256") != _canonical_sha256(structural):
        raise TerrainPbrContractError(
            "installed PBR material summary drifted from its bundle marker"
        )
    return LockedMaterialLibrary(
        bundle_sha256=bundle_sha256,
        manifest_path=manifest_relative,
        manifest_sha256=manifest_sha256,
        materials=tuple(materials),
    )


def _validate_evidence_file(*, root: Path, source: SpatialEvidence) -> None:
    relative = _safe_relative_path(
        source.lock.path,
        label=f"{source.semantic} evidence path",
    )
    path = root.joinpath(*relative.parts)
    suffixes = (
        _ALLOWED_VECTOR_SUFFIXES
        if source.content_kind == "classified_vector"
        else _ALLOWED_RASTER_SUFFIXES
    )
    if (
        not _inside(root, path)
        or not path.is_file()
        or path.is_symlink()
        or path.suffix.casefold() not in suffixes
    ):
        raise TerrainPbrContractError(
            f"{source.semantic} evidence is not a supported regular spatial file"
        )
    if (
        _sha256_file(path) != source.lock.sha256
        or path.stat().st_size != source.lock.size_bytes
    ):
        raise TerrainPbrContractError(
            f"{source.semantic} evidence SHA-256 or size lock does not match"
        )


def _smoothstep(edge_low: float, edge_high: float, value: float) -> float:
    if edge_high <= edge_low:
        raise TerrainPbrContractError("smoothstep thresholds are invalid")
    unit = min(1.0, max(0.0, (value - edge_low) / (edge_high - edge_low)))
    return unit * unit * (3.0 - 2.0 * unit)


def _mean_material_weights(
    subzone: TerrainSubzoneEvidence,
) -> tuple[tuple[str, float], ...]:
    """Compute a QA summary using the same priority as the authoring graph."""

    coverage = subzone.coverage
    remaining = 1.0
    water = remaining * float(coverage["water"])
    remaining -= water
    roads = remaining * float(coverage["roads"])
    remaining -= roads
    artificial = remaining * float(coverage["artificial_ground"])
    remaining -= artificial
    forest = remaining * float(coverage["forest"])
    remaining -= forest

    rock_factor = _smoothstep(20.0, 42.0, subzone.mean_slope_degrees)
    rough_soil_factor = _smoothstep(0.15, 1.25, subzone.roughness_m)
    rock = remaining * rock_factor
    residual = remaining - rock
    exposed_soil = residual * rough_soil_factor * 0.65
    grass = residual - exposed_soil

    weights = {
        "forest_floor": forest,
        "grass": grass,
        "soil": exposed_soil + artificial * 0.35,
        "rock": rock,
        "asphalt": roads + artificial * 0.20,
        "gravel": artificial * 0.45,
        "water": water,
    }
    total = math.fsum(weights.values())
    if total <= 0.0:
        raise TerrainPbrContractError("terrain blend produced an empty material set")
    normalized = {
        role: max(0.0, value / total) for role, value in weights.items()
    }
    correction = 1.0 - math.fsum(normalized.values())
    normalized["grass"] += correction
    return tuple((role, round(normalized[role], 12)) for role in PBR_MATERIAL_ROLES)


def _blend_graph() -> dict[str, object]:
    return {
        "evaluation_space": "world_metres",
        "source_resampling": "native_source_or_authorer_filtered",
        "normalization": "priority_masks_then_sum_to_one",
        "operations": [
            {
                "input": "water",
                "outputs": {"water": 1.0},
                "operator": "signed_distance_feather",
                "transition_width_m": 0.75,
                "priority": 0,
            },
            {
                "input": "roads",
                "outputs": {"asphalt": 1.0},
                "operator": "vector_or_mask_metric_core_and_edge",
                "transition_width_m": 0.45,
                "priority": 1,
            },
            {
                "input": "artificial_ground",
                "outputs": {"gravel": 0.45, "soil": 0.35, "asphalt": 0.20},
                "operator": "signed_distance_feather",
                "transition_width_m": 0.80,
                "priority": 2,
            },
            {
                "input": "forest",
                "outputs": {"forest_floor": 1.0},
                "operator": "signed_distance_feather",
                "transition_width_m": 2.50,
                "priority": 3,
            },
            {
                "input": "elevation",
                "outputs": {"rock": 1.0},
                "operator": "slope_smoothstep",
                "threshold_degrees": [20.0, 42.0],
                "priority": 4,
            },
            {
                "input": "elevation",
                "outputs": {"soil": 0.65, "grass": 0.35},
                "operator": "roughness_smoothstep",
                "threshold_m": [0.15, 1.25],
                "priority": 5,
            },
            {
                "input": "unclassified_residual",
                "outputs": {"grass": 1.0},
                "operator": "remainder",
                "priority": 6,
            },
        ],
        "transition_policy": {
            "method": "metric_signed_distance_or_relief_smoothstep",
            "classification_sampling": "bounded_payload_tile",
            "classification_address_mode": "clamp",
            "material_metric_uv_continuity": "shared_world_position",
            "height_blend_material_normals": True,
            "base_color_from_spatial_evidence": False,
            "spatial_evidence_can_create_rendered_objects": False,
        },
    }


def _evidence_inventory(tiles: Iterable[TerrainTileEvidence]) -> list[dict[str, object]]:
    records: dict[tuple[str, str], dict[str, object]] = {}
    for tile in tiles:
        for source in tile.evidence:
            key = (source.semantic, source.stable_id)
            value = {
                "stable_id": source.stable_id,
                "semantic": source.semantic,
                "path": source.lock.path,
                "sha256": source.lock.sha256,
                "size_bytes": source.lock.size_bytes,
                "content_kind": source.content_kind,
                "usage": source.usage,
                "crs": source.crs,
                "bounds_m": source.bounds.as_list(),
                "resolution_m": source.resolution_m,
                "feature_count": source.feature_count,
            }
            previous = records.get(key)
            if previous is not None and previous != value:
                raise TerrainPbrContractError(
                    "a spatial evidence identifier resolves to conflicting locks"
                )
            records[key] = value
    return [records[key] for key in sorted(records)]


def build_terrain_pbr_plan(
    *,
    scene_id: str,
    bundle_root: Path,
    material_manifest_path: Path,
    evidence_root: Path,
    tiles: Sequence[TerrainTileEvidence],
    world_uv_origin_m: tuple[float, float] = (0.0, 0.0),
    scene_origin_source_m: tuple[float, float] | None = None,
) -> TerrainPbrPlan:
    """Build a deterministic, authorable terrain-material plan.

    All inputs are verified in-place.  No file is created and no texture is
    loaded into memory beyond the streaming SHA-256 pass.
    """

    identifier = _require_stable_id(scene_id, label="terrain PBR scene")
    if not tiles:
        raise TerrainPbrContractError(
            "terrain PBR plan requires at least one terrain tile"
        )
    origin_x_m, origin_y_m = world_uv_origin_m
    if not math.isfinite(origin_x_m) or not math.isfinite(origin_y_m):
        raise TerrainPbrContractError("world UV origin must contain finite metres")
    evidence = evidence_root.resolve()
    if not evidence.is_dir() or evidence.is_symlink():
        raise TerrainPbrContractError(
            "evidence root must be a regular directory"
        )
    library = load_locked_material_library(
        bundle_root=bundle_root,
        manifest_path=material_manifest_path,
    )
    tile_ids: set[str] = set()
    scene_crs: set[str] = set()
    for tile in tiles:
        if tile.stable_id in tile_ids:
            raise TerrainPbrContractError(
                "terrain tile identifiers must be globally unique"
            )
        tile_ids.add(tile.stable_id)
        for source in tile.evidence:
            scene_crs.add(source.crs)
            _validate_evidence_file(root=evidence, source=source)
    if len(scene_crs) != 1:
        raise TerrainPbrContractError(
            "all terrain tiles must use one coherent metric source CRS"
        )

    ordered_tiles = tuple(sorted(tiles, key=lambda tile: tile.stable_id))
    if scene_origin_source_m is None:
        scene_origin = (
            min(tile.bounds.min_x_m for tile in ordered_tiles),
            min(tile.bounds.min_y_m for tile in ordered_tiles),
        )
    else:
        scene_origin = tuple(scene_origin_source_m)
    if (
        len(scene_origin) != 2
        or any(not math.isfinite(value) for value in scene_origin)
    ):
        raise TerrainPbrContractError(
            "scene origin must contain two finite source-CRS metres"
        )
    tile_plans = tuple(
        TileMaterialPlan(
            stable_id=tile.stable_id,
            bounds=tile.bounds,
            evidence=tuple(
                sorted(tile.evidence, key=lambda source: source.semantic)
            ),
            subzones=tuple(
                SubzoneMaterialPlan(
                    stable_id=subzone.stable_id,
                    bounds=subzone.bounds,
                    evidence_ids=tuple(sorted(subzone.evidence_ids)),
                    mean_elevation_m=subzone.mean_elevation_m,
                    mean_slope_degrees=subzone.mean_slope_degrees,
                    roughness_m=subzone.roughness_m,
                    coverage=tuple(
                        (
                            semantic,
                            float(subzone.coverage[semantic]),
                        )
                        for semantic in CLASSIFICATION_SEMANTICS
                    ),
                    mean_weights=_mean_material_weights(subzone),
                )
                for subzone in sorted(
                    tile.subzones,
                    key=lambda value: value.stable_id,
                )
            ),
        )
        for tile in ordered_tiles
    )
    inventory = _evidence_inventory(ordered_tiles)
    metric_uv = tuple(
        MetricUv(
            role=material.role,
            origin_x_m=origin_x_m,
            origin_y_m=origin_y_m,
            metres_per_uv_tile=material.metres_per_uv_tile,
        )
        for material in library.materials
    )
    return TerrainPbrPlan(
        scene_id=identifier,
        material_library=library,
        metric_uv=metric_uv,
        tiles=tile_plans,
        evidence_inventory_sha256=_canonical_sha256(inventory),
        scene_origin_source_m=(scene_origin[0], scene_origin[1]),
    )


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise TerrainPbrContractError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TerrainPbrContractError(
            f"{label} must be a finite number"
        ) from exc
    if not math.isfinite(result):
        raise TerrainPbrContractError(f"{label} must be a finite number")
    return result


def _locked_zone_artifact(
    *,
    zone_root: Path,
    volume_root: Path,
    record: object,
    label: str,
    suffixes: frozenset[str] | None = None,
) -> tuple[Path, dict[str, object]]:
    if not isinstance(record, Mapping):
        raise TerrainPbrContractError(f"{label} lock is missing")
    relative = _safe_relative_path(
        str(record.get("path", "")),
        label=f"{label} path",
    )
    path = zone_root.joinpath(*relative.parts).resolve()
    expected_sha256 = _require_sha256(
        str(record.get("sha256", "")),
        label=label,
    )
    if (
        not _inside(zone_root, path)
        or not _inside(volume_root, path)
        or not path.is_file()
        or path.is_symlink()
        or (
            suffixes is not None
            and path.suffix.casefold() not in suffixes
        )
    ):
        raise TerrainPbrContractError(
            f"{label} is absent, unsafe or has an unsupported type"
        )
    actual_size = path.stat().st_size
    if actual_size <= 0 or _sha256_file(path) != expected_sha256:
        raise TerrainPbrContractError(f"{label} SHA-256 lock drifted")
    return path, {
        "path": relative.as_posix(),
        "sha256": expected_sha256,
        "size_bytes": actual_size,
    }


def _source_lock_download(
    *,
    volume_root: Path,
    zone_root: Path,
    raw_root: Path,
    record: Mapping[str, object],
    dataset: str | None,
    label: str,
) -> tuple[Path, dict[str, object]]:
    download = record.get("download")
    if not isinstance(download, Mapping) or download.get("state") not in {
        "downloaded",
        "verified_existing",
    }:
        raise TerrainPbrContractError(
            f"{label} has no completed source-lock download"
        )
    relative = _safe_relative_path(
        str(download.get("relpath", "")),
        label=f"{label} download path",
    )
    base = raw_root / dataset if dataset else raw_root
    path = base.joinpath(*relative.parts).resolve()
    expected_sha256 = _require_sha256(
        str(download.get("sha256", "")),
        label=f"{label} download",
    )
    expected_bytes = download.get("bytes")
    if (
        not _inside(raw_root, path)
        or not _inside(zone_root, path)
        or not _inside(volume_root, path)
        or not path.is_file()
        or path.is_symlink()
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or path.stat().st_size != expected_bytes
        or _sha256_file(path) != expected_sha256
    ):
        raise TerrainPbrContractError(
            f"{label} source file is absent, unsafe or stale"
        )
    return path, {
        "path": path.relative_to(volume_root).as_posix(),
        "sha256": expected_sha256,
        "size_bytes": expected_bytes,
    }


def _native_preparation_inputs(
    *,
    volume_root: Path,
    zone_root: Path,
    scene_auto_validation_path: Path,
) -> dict[str, object]:
    """Validate immutable Znn receipts before loading pxr/GDAL."""

    volume = volume_root.resolve()
    zone = zone_root.resolve()
    auto_path = scene_auto_validation_path.resolve()
    if (
        not volume.is_dir()
        or volume.is_symlink()
        or not zone.is_dir()
        or zone.is_symlink()
        or not _inside(volume, zone)
        or not auto_path.is_file()
        or auto_path.is_symlink()
        or not _inside(volume, auto_path)
    ):
        raise TerrainPbrContractError(
            "native preparation roots or scene validation are unsafe"
        )
    build_path = zone / "build" / "build-receipt.json"
    if (
        not build_path.is_file()
        or build_path.is_symlink()
        or not _inside(zone, build_path)
    ):
        raise TerrainPbrContractError(
            "native preparation requires build/build-receipt.json"
        )
    build = _read_json(build_path, label="native zone build receipt")
    zone_id = _require_stable_id(
        str(build.get("zone_id", "")),
        label="native zone",
    )
    if (
        build.get("schema_version") != 2
        or build.get("source_profile") != "full"
        or build.get("fire_simulation_status")
        != "blocked_pending_editor_review"
    ):
        raise TerrainPbrContractError(
            "native preparation requires a full schema-2 build awaiting "
            "Editor review"
        )
    root_usd, root_record = _locked_zone_artifact(
        zone_root=zone,
        volume_root=volume,
        record=build.get("root_usd"),
        label="native root USD",
        suffixes=frozenset(MATERIAL_SUFFIXES),
    )
    auto = _read_json(auto_path, label="scene auto-validation")
    if (
        auto.get("state") != "AUTO_VALIDATED"
        or auto.get("fire_simulation_status")
        != "blocked_pending_editor_review"
        or auto.get("build_receipt_sha256") != _sha256_file(build_path)
        or auto.get("root_usd_sha256") != root_record["sha256"]
    ):
        raise TerrainPbrContractError(
            "scene auto-validation is stale or bound to another build"
        )
    source_lock_path, source_lock_record = _locked_zone_artifact(
        zone_root=zone,
        volume_root=volume,
        record=build.get("source_lock"),
        label="native source-lock",
        suffixes=frozenset({".json"}),
    )
    source_lock = _read_json(
        source_lock_path,
        label="native source-lock",
    )
    if source_lock.get("zone_id") != zone_id:
        raise TerrainPbrContractError(
            "native source-lock belongs to another zone"
        )
    lidar_path, lidar_record = _locked_zone_artifact(
        zone_root=zone,
        volume_root=volume,
        record=build.get("lidar_quality"),
        label="PDAL LiDAR quality receipt",
        suffixes=frozenset({".json"}),
    )
    if build.get("lidar_quality", {}).get("source_count") != (
        NATIVE_ZONE_TILE_COUNT
    ):
        raise TerrainPbrContractError(
            "PDAL LiDAR evidence does not cover all 400 source tiles"
        )
    georeference_path = zone / "build" / "metadata" / "georeference.json"
    if (
        not georeference_path.is_file()
        or georeference_path.is_symlink()
        or not _inside(zone, georeference_path)
    ):
        raise TerrainPbrContractError(
            "native georeference receipt is missing"
        )
    georeference = _read_json(
        georeference_path,
        label="native georeference",
    )
    raw_origin = georeference.get("local_origin_epsg2154")
    if (
        georeference.get("zone_id") != zone_id
        or georeference.get("crs") != NATIVE_ZONE_CRS
        or georeference.get("vertical_datum")
        != NATIVE_ZONE_VERTICAL_DATUM
        or not isinstance(raw_origin, list)
        or len(raw_origin) < 2
    ):
        raise TerrainPbrContractError(
            "native georeference is incomplete or incoherent"
        )
    scene_origin = (
        _finite_number(raw_origin[0], label="scene origin X"),
        _finite_number(raw_origin[1], label="scene origin Y"),
    )
    payloads = build.get("payloads")
    coverage = build.get("tile_coverage")
    if (
        not isinstance(payloads, list)
        or len(payloads) != NATIVE_ZONE_TILE_COUNT
        or not isinstance(coverage, list)
        or len(coverage) != NATIVE_ZONE_TILE_COUNT
    ):
        raise TerrainPbrContractError(
            "native build must expose exactly 400 payloads and coverage rows"
        )
    coverage_by_path: dict[str, Mapping[str, object]] = {}
    for index, raw_coverage in enumerate(coverage):
        if not isinstance(raw_coverage, Mapping):
            raise TerrainPbrContractError(
                f"tile_coverage[{index}] is malformed"
            )
        relative = _safe_relative_path(
            str(raw_coverage.get("terrain_payload", "")),
            label=f"tile_coverage[{index}].terrain_payload",
        ).as_posix()
        tile_ref = _require_stable_id(
            str(raw_coverage.get("tile_ref", "")),
            label=f"tile_coverage[{index}].tile_ref",
        )
        namespace = raw_coverage.get("instance_namespace")
        if (
            relative in coverage_by_path
            or isinstance(namespace, bool)
            or not isinstance(namespace, int)
            or namespace <= 0
        ):
            raise TerrainPbrContractError(
                "terrain coverage paths/namespaces must be unique and valid"
            )
        coverage_by_path[relative] = {
            **dict(raw_coverage),
            "tile_ref": tile_ref,
        }
    payload_records: list[dict[str, object]] = []
    seen_tiles: set[str] = set()
    seen_namespaces: set[int] = set()
    for index, raw_payload in enumerate(payloads):
        path, lock = _locked_zone_artifact(
            zone_root=zone,
            volume_root=volume,
            record=raw_payload,
            label=f"terrain payload {index}",
            suffixes=frozenset(MATERIAL_SUFFIXES),
        )
        coverage_record = coverage_by_path.get(str(lock["path"]))
        if coverage_record is None:
            raise TerrainPbrContractError(
                "terrain payload has no exact tile coverage record"
            )
        tile_ref = str(coverage_record["tile_ref"])
        namespace = int(coverage_record["instance_namespace"])
        if tile_ref in seen_tiles or namespace in seen_namespaces:
            raise TerrainPbrContractError(
                "terrain tile refs and namespaces must be unique"
            )
        seen_tiles.add(tile_ref)
        seen_namespaces.add(namespace)
        payload_records.append(
            {
                "tile_ref": tile_ref,
                "instance_namespace": namespace,
                "physical_path": path,
                **lock,
            }
        )
    entries = source_lock.get("entries")
    if not isinstance(entries, list):
        raise TerrainPbrContractError(
            "native source-lock entries are malformed"
        )
    raw_root = zone / "raw"
    mnt_sources: dict[str, dict[str, object]] = {}
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            raise TerrainPbrContractError(
                f"source-lock entry {index} is malformed"
            )
        if raw_entry.get("dataset") != "mnt":
            continue
        tile_ref = _require_stable_id(
            str(raw_entry.get("tile_ref", "")),
            label=f"MNT source-lock entry {index}",
        )
        path, lock = _source_lock_download(
            volume_root=volume,
            zone_root=zone,
            raw_root=raw_root,
            record=raw_entry,
            dataset="mnt",
            label=f"MNT {tile_ref}",
        )
        if tile_ref in mnt_sources:
            raise TerrainPbrContractError(
                f"MNT source repeats tile {tile_ref}"
            )
        mnt_sources[tile_ref] = {
            "physical_path": path,
            **lock,
        }
    if set(mnt_sources) != seen_tiles:
        raise TerrainPbrContractError(
            "MNT source-lock coverage does not match the 400 terrain tiles"
        )
    vector_sources = source_lock.get("vector_sources")
    if not isinstance(vector_sources, Mapping):
        raise TerrainPbrContractError(
            "source-lock lacks classified BDTOPO vectors"
        )
    semantic_categories = {
        "forest": "vegetation",
        "water": "hydrology",
        "roads": "roads",
        "artificial_ground": "buildings",
    }
    prepared_vectors: dict[str, list[dict[str, object]]] = {}
    for semantic, category in semantic_categories.items():
        raw_records = vector_sources.get(category)
        if not isinstance(raw_records, list) or not raw_records:
            raise TerrainPbrContractError(
                f"source-lock lacks {semantic} classified vectors"
            )
        records: list[dict[str, object]] = []
        for index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, Mapping):
                raise TerrainPbrContractError(
                    f"{semantic} vector source {index} is malformed"
                )
            bbox = raw_record.get("bbox_epsg2154")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise TerrainPbrContractError(
                    f"{semantic} vector source has no EPSG:2154 coverage"
                )
            bounds = Bounds2d(
                *(
                    _finite_number(
                        value,
                        label=f"{semantic} vector source bound",
                    )
                    for value in bbox
                )
            )
            path, lock = _source_lock_download(
                volume_root=volume,
                zone_root=zone,
                raw_root=raw_root,
                record=raw_record,
                dataset=None,
                label=f"{semantic} vector source {index}",
            )
            records.append(
                {
                    "physical_path": path,
                    "bounds": bounds,
                    "declared_feature_count": raw_record.get(
                        "feature_count"
                    ),
                    **lock,
                }
            )
        prepared_vectors[semantic] = records
    return {
        "zone_id": zone_id,
        "volume_root": volume,
        "zone_root": zone,
        "build_path": build_path,
        "build_sha256": _sha256_file(build_path),
        "auto_path": auto_path,
        "auto_sha256": _sha256_file(auto_path),
        "root_usd": root_usd,
        "root_usd_record": root_record,
        "source_lock_path": source_lock_path,
        "source_lock_record": source_lock_record,
        "lidar_path": lidar_path,
        "lidar_record": lidar_record,
        "georeference_path": georeference_path,
        "georeference_sha256": _sha256_file(georeference_path),
        "scene_origin_source_m": scene_origin,
        "payloads": payload_records,
        "mnt_sources": mnt_sources,
        "vector_sources": prepared_vectors,
    }


def _parse_epsg2154_bounds(value: object, *, label: str) -> Bounds2d:
    if not isinstance(value, str):
        raise TerrainPbrContractError(
            f"{label} lacks fireviewer:epsg2154_bounds"
        )
    parts = value.split(",")
    if len(parts) != 4:
        raise TerrainPbrContractError(
            f"{label} has malformed EPSG:2154 bounds"
        )
    return Bounds2d(
        *(
            _finite_number(part, label=f"{label} EPSG:2154 bound")
            for part in parts
        )
    )


def _gdal_is_epsg2154(*, osr: Any, projection: str) -> bool:
    if not projection.strip():
        return False
    actual = osr.SpatialReference()
    expected = osr.SpatialReference()
    if actual.SetFromUserInput(projection) != 0:
        return False
    if expected.ImportFromEPSG(2154) != 0:
        return False
    return bool(actual.IsSame(expected))


def _gdal_raster_contract(
    *,
    dataset: Any,
    osr: Any,
    label: str,
) -> tuple[Bounds2d, float]:
    if dataset is None:
        raise TerrainPbrContractError(f"{label} cannot be opened by GDAL")
    transform = dataset.GetGeoTransform()
    if (
        len(transform) != 6
        or transform[1] <= 0.0
        or transform[5] >= 0.0
        or abs(transform[2]) > 1.0e-12
        or abs(transform[4]) > 1.0e-12
        or not _gdal_is_epsg2154(
            osr=osr,
            projection=str(dataset.GetProjection()),
        )
    ):
        raise TerrainPbrContractError(
            f"{label} lacks a north-up EPSG:2154 grid"
        )
    resolution_x = float(transform[1])
    resolution_y = abs(float(transform[5]))
    if (
        not math.isclose(
            resolution_x,
            resolution_y,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        or resolution_x < 0.1
        or resolution_x > 2.0
    ):
        raise TerrainPbrContractError(
            f"{label} has an incomplete or unsuitable metric resolution"
        )
    return (
        Bounds2d(
            float(transform[0]),
            float(transform[3] + transform[5] * dataset.RasterYSize),
            float(transform[0] + transform[1] * dataset.RasterXSize),
            float(transform[3]),
        ),
        resolution_x,
    )


def _aligned_native_halo_m(resolution_m: float) -> float:
    """Round the required physical halo outward to an exact source pixel."""

    if not math.isfinite(resolution_m) or resolution_m <= 0.0:
        raise TerrainPbrContractError(
            "native halo requires a positive finite source resolution"
        )
    minimum_halo_m = max(_TRANSITION_WIDTH_M.values()) + 2.0 * max(
        resolution_m,
        VECTOR_CLASSIFICATION_RESOLUTION_M,
    )
    pixel_count = math.ceil(
        minimum_halo_m / resolution_m - 1.0e-12
    )
    return pixel_count * resolution_m


def _prepared_file_lock(
    *,
    physical_path: Path,
    final_path: Path,
    volume_root: Path,
) -> FileLock:
    if (
        not physical_path.is_file()
        or physical_path.is_symlink()
        or physical_path.stat().st_size <= 0
        or not _inside(volume_root, final_path)
    ):
        raise TerrainPbrContractError(
            "prepared spatial evidence is absent or unsafe"
        )
    return FileLock(
        path=final_path.resolve().relative_to(
            volume_root.resolve()
        ).as_posix(),
        sha256=_sha256_file(physical_path),
        size_bytes=physical_path.stat().st_size,
    )


def _merge_native_classified_vectors(
    *,
    context: Mapping[str, object],
    physical_root: Path,
    final_root: Path,
) -> tuple[
    dict[str, SpatialEvidence],
    dict[str, dict[str, object]],
]:
    """Materialize four immutable class layers, never a colour/mask atlas."""

    volume_root = Path(context["volume_root"])
    vector_sources = context["vector_sources"]
    if not isinstance(vector_sources, Mapping):
        raise TerrainPbrContractError(
            "native vector preparation context is malformed"
        )
    shared_physical = physical_root / "classified-vectors"
    shared_final = final_root / "classified-vectors"
    shared_physical.mkdir(parents=True)
    spatial: dict[str, SpatialEvidence] = {}
    lineage: dict[str, dict[str, object]] = {}
    for semantic in CLASSIFICATION_SEMANTICS:
        raw_sources = vector_sources.get(semantic)
        if not isinstance(raw_sources, list) or not raw_sources:
            raise TerrainPbrContractError(
                f"native preparation lacks {semantic} source vectors"
            )
        features: list[object] = []
        source_records: list[dict[str, object]] = []
        coverage_bounds: list[Bounds2d] = []
        for source_index, raw_source in enumerate(raw_sources):
            if not isinstance(raw_source, Mapping):
                raise TerrainPbrContractError(
                    f"{semantic} source vector {source_index} is malformed"
                )
            path = Path(raw_source["physical_path"])
            payload = _read_json(
                path,
                label=f"{semantic} classified vector {source_index}",
            )
            raw_features = payload.get("features")
            if (
                payload.get("type") != "FeatureCollection"
                or not isinstance(raw_features, list)
            ):
                raise TerrainPbrContractError(
                    f"{semantic} source is not a GeoJSON FeatureCollection"
                )
            declared_count = raw_source.get("declared_feature_count")
            if (
                isinstance(declared_count, bool)
                or not isinstance(declared_count, int)
                or declared_count < 0
                or declared_count != len(raw_features)
            ):
                raise TerrainPbrContractError(
                    f"{semantic} feature count drifted from source-lock"
                )
            features.extend(raw_features)
            coverage_bounds.append(raw_source["bounds"])
            source_records.append(
                {
                    "path": raw_source["path"],
                    "sha256": raw_source["sha256"],
                    "size_bytes": raw_source["size_bytes"],
                    "feature_count": declared_count,
                }
            )
        if not features:
            raise TerrainPbrContractError(
                f"{semantic} classified evidence contains no real feature"
            )
        common_bounds = Bounds2d(
            max(value.min_x_m for value in coverage_bounds),
            max(value.min_y_m for value in coverage_bounds),
            min(value.max_x_m for value in coverage_bounds),
            min(value.max_y_m for value in coverage_bounds),
        )
        output_physical = shared_physical / f"{semantic}.geojson"
        output_final = shared_final / f"{semantic}.geojson"
        _atomic_write_json(
            output_physical,
            {
                "type": "FeatureCollection",
                "name": f"fireviewer-{semantic}-classified",
                "crs": {
                    "type": "name",
                    "properties": {"name": NATIVE_ZONE_CRS},
                },
                "source_locks": source_records,
                "features": features,
            },
        )
        lock = _prepared_file_lock(
            physical_path=output_physical,
            final_path=output_final,
            volume_root=volume_root,
        )
        spatial[semantic] = SpatialEvidence(
            stable_id=f"{context['zone_id']}:{semantic}",
            semantic=semantic,
            content_kind="classified_vector",
            usage="blend_weights_only",
            lock=lock,
            crs=NATIVE_ZONE_CRS,
            bounds=common_bounds,
            resolution_m=VECTOR_CLASSIFICATION_RESOLUTION_M,
            feature_count=len(features),
        )
        lineage[semantic] = {
            "prepared_path": lock.path,
            "prepared_sha256": lock.sha256,
            "prepared_size_bytes": lock.size_bytes,
            "feature_count": len(features),
            "source_locks": source_records,
        }
    return spatial, lineage


def _subzone_array_slices(
    *,
    bounds: Bounds2d,
    height: int,
    width: int,
) -> tuple[
    tuple[str, Bounds2d, slice, slice],
    ...,
]:
    if height < 4 or width < 4:
        raise TerrainPbrContractError(
            "terrain analysis grid is too small for measured subzones"
        )
    row_mid = height // 2
    column_mid = width // 2
    mid_x = (bounds.min_x_m + bounds.max_x_m) * 0.5
    mid_y = (bounds.min_y_m + bounds.max_y_m) * 0.5
    return (
        (
            "NW",
            Bounds2d(bounds.min_x_m, mid_y, mid_x, bounds.max_y_m),
            slice(0, row_mid),
            slice(0, column_mid),
        ),
        (
            "NE",
            Bounds2d(mid_x, mid_y, bounds.max_x_m, bounds.max_y_m),
            slice(0, row_mid),
            slice(column_mid, width),
        ),
        (
            "SW",
            Bounds2d(bounds.min_x_m, bounds.min_y_m, mid_x, mid_y),
            slice(row_mid, height),
            slice(0, column_mid),
        ),
        (
            "SE",
            Bounds2d(mid_x, bounds.min_y_m, bounds.max_x_m, mid_y),
            slice(row_mid, height),
            slice(column_mid, width),
        ),
    )


def _derive_native_zone_terrain_tiles(
    *,
    context: Mapping[str, object],
    physical_evidence_root: Path,
    final_evidence_root: Path,
) -> tuple[
    tuple[TerrainTileEvidence, ...],
    dict[str, object],
]:
    """Create measured, halo-complete elevation evidence for all 400 tiles."""

    try:
        import numpy as np
        from osgeo import gdal, osr
        from pxr import Usd
    except ImportError as exc:
        raise RuntimeError(
            "prepare-native requires the pinned Kit pxr runtime plus "
            "NumPy and GDAL/osgeo"
        ) from exc
    gdal.UseExceptions()
    volume_root = Path(context["volume_root"])
    scene_origin = tuple(context["scene_origin_source_m"])
    payloads = context["payloads"]
    mnt_sources = context["mnt_sources"]
    if (
        not isinstance(payloads, list)
        or not isinstance(mnt_sources, Mapping)
    ):
        raise TerrainPbrContractError(
            "native terrain preparation context is malformed"
        )
    tile_records: list[dict[str, object]] = []
    for raw_payload in payloads:
        if not isinstance(raw_payload, Mapping):
            raise TerrainPbrContractError(
                "native terrain payload context is malformed"
            )
        payload_path = Path(raw_payload["physical_path"])
        stage = Usd.Stage.Open(str(payload_path), load=Usd.Stage.LoadNone)
        if stage is None:
            raise TerrainPbrContractError(
                f"terrain payload cannot be opened: {payload_path.name}"
            )
        tile = stage.GetPrimAtPath("/Tile")
        tile_ref = str(raw_payload["tile_ref"])
        if (
            not tile
            or not tile.IsValid()
            or tile.GetCustomDataByKey("fireviewer:tile_ref") != tile_ref
        ):
            raise TerrainPbrContractError(
                f"terrain payload identity differs from coverage: {tile_ref}"
            )
        source_bounds = _parse_epsg2154_bounds(
            tile.GetCustomDataByKey("fireviewer:epsg2154_bounds"),
            label=tile_ref,
        )
        local_bounds = Bounds2d(
            source_bounds.min_x_m - float(scene_origin[0]),
            source_bounds.min_y_m - float(scene_origin[1]),
            source_bounds.max_x_m - float(scene_origin[0]),
            source_bounds.max_y_m - float(scene_origin[1]),
        )
        if (
            not math.isclose(
                source_bounds.area_m2,
                1_000_000.0,
                rel_tol=0.0,
                abs_tol=0.01,
            )
            or not math.isclose(
                source_bounds.max_x_m - source_bounds.min_x_m,
                1_000.0,
                rel_tol=0.0,
                abs_tol=0.01,
            )
            or not math.isclose(
                source_bounds.max_y_m - source_bounds.min_y_m,
                1_000.0,
                rel_tol=0.0,
                abs_tol=0.01,
            )
        ):
            raise TerrainPbrContractError(
                f"{tile_ref} is not an exact 1 km native terrain tile"
            )
        source = mnt_sources.get(tile_ref)
        if not isinstance(source, Mapping):
            raise TerrainPbrContractError(
                f"{tile_ref} has no locked MNT source"
            )
        dataset = gdal.Open(str(source["physical_path"]), gdal.GA_ReadOnly)
        mnt_bounds, resolution = _gdal_raster_contract(
            dataset=dataset,
            osr=osr,
            label=f"{tile_ref} MNT",
        )
        if not mnt_bounds.contains(
            source_bounds,
            tolerance_m=resolution + 1.0e-6,
        ):
            raise TerrainPbrContractError(
                f"{tile_ref} MNT does not cover its payload bounds"
            )
        dataset = None
        tile_records.append(
            {
                **dict(raw_payload),
                "source_bounds": source_bounds,
                "local_bounds": local_bounds,
                "mnt_path": Path(source["physical_path"]),
                "mnt_bounds": mnt_bounds,
                "mnt_resolution_m": resolution,
            }
        )
    tile_records.sort(key=lambda value: str(value["tile_ref"]))
    resolutions = {
        round(float(record["mnt_resolution_m"]), 9)
        for record in tile_records
    }
    if len(resolutions) != 1:
        raise TerrainPbrContractError(
            "native MNT sources do not share one exact metric resolution"
        )
    mnt_resolution = next(iter(resolutions))
    scene_bounds = Bounds2d(
        min(record["source_bounds"].min_x_m for record in tile_records),
        min(record["source_bounds"].min_y_m for record in tile_records),
        max(record["source_bounds"].max_x_m for record in tile_records),
        max(record["source_bounds"].max_y_m for record in tile_records),
    )
    if not math.isclose(
        scene_bounds.area_m2,
        400_000_000.0,
        rel_tol=0.0,
        abs_tol=0.1,
    ):
        raise TerrainPbrContractError(
            "400 native tiles do not form one complete 20 km square"
        )
    for index, left in enumerate(tile_records):
        for right in tile_records[index + 1 :]:
            if (
                left["source_bounds"].overlap_area_m2(
                    right["source_bounds"]
                )
                > 0.01
            ):
                raise TerrainPbrContractError(
                    "native terrain payload bounds overlap"
                )
    vector_spatial, vector_lineage = _merge_native_classified_vectors(
        context=context,
        physical_root=physical_evidence_root,
        final_root=final_evidence_root,
    )
    for semantic, evidence in vector_spatial.items():
        if not evidence.bounds.contains(scene_bounds, tolerance_m=0.01):
            raise TerrainPbrContractError(
                f"{semantic} vectors do not cover the complete native zone"
            )
        vector_path = volume_root.joinpath(
            *_safe_relative_path(
                evidence.lock.path,
                label=f"{semantic} prepared vector",
            ).parts
        )
        physical_vector = (
            physical_evidence_root
            / vector_path.relative_to(final_evidence_root)
        )
        dataset = gdal.OpenEx(str(physical_vector), gdal.OF_VECTOR)
        if dataset is None or dataset.GetLayerCount() <= 0:
            raise TerrainPbrContractError(
                f"{semantic} prepared vector cannot be opened by GDAL"
            )
        actual_count = 0
        for layer_index in range(dataset.GetLayerCount()):
            layer = dataset.GetLayerByIndex(layer_index)
            if layer is None:
                raise TerrainPbrContractError(
                    f"{semantic} prepared vector lacks layer {layer_index}"
                )
            spatial_reference = layer.GetSpatialRef()
            if (
                spatial_reference is None
                or not _gdal_is_epsg2154(
                    osr=osr,
                    projection=spatial_reference.ExportToWkt(),
                )
            ):
                raise TerrainPbrContractError(
                    f"{semantic} prepared vector CRS is not EPSG:2154"
                )
            actual_count += int(layer.GetFeatureCount())
        if actual_count != evidence.feature_count:
            raise TerrainPbrContractError(
                f"{semantic} prepared vector feature count drifted"
            )
        dataset = None

    elevation_physical_root = physical_evidence_root / "elevation"
    elevation_final_root = final_evidence_root / "elevation"
    elevation_physical_root.mkdir()
    halo_m = _aligned_native_halo_m(mnt_resolution)
    tiles: list[TerrainTileEvidence] = []
    elevation_records: list[dict[str, object]] = []
    for tile_index, record in enumerate(tile_records):
        tile_ref = str(record["tile_ref"])
        bounds = record["source_bounds"]
        sampling = Bounds2d(
            max(scene_bounds.min_x_m, bounds.min_x_m - halo_m),
            max(scene_bounds.min_y_m, bounds.min_y_m - halo_m),
            min(scene_bounds.max_x_m, bounds.max_x_m + halo_m),
            min(scene_bounds.max_y_m, bounds.max_y_m + halo_m),
        )
        candidate_paths = [
            str(other["mnt_path"])
            for other in tile_records
            if other["mnt_bounds"].overlap_area_m2(sampling) > 0.0
            or other["mnt_bounds"].contains(
                sampling,
                tolerance_m=mnt_resolution,
            )
        ]
        if not candidate_paths:
            raise TerrainPbrContractError(
                f"{tile_ref} has no MNT source for its required halo"
            )
        width_float = (
            sampling.max_x_m - sampling.min_x_m
        ) / mnt_resolution
        height_float = (
            sampling.max_y_m - sampling.min_y_m
        ) / mnt_resolution
        width = int(round(width_float))
        height = int(round(height_float))
        if (
            width <= 1
            or height <= 1
            or width > MAX_DERIVED_TILE_EDGE_PIXELS
            or height > MAX_DERIVED_TILE_EDGE_PIXELS
            or not math.isclose(
                width_float,
                width,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
            or not math.isclose(
                height_float,
                height,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
        ):
            raise TerrainPbrContractError(
                f"{tile_ref} MNT grid cannot preserve its exact resolution"
            )
        token = (
            f"{tile_index:04d}-{_safe_usd_identifier(tile_ref)}-"
            f"{hashlib.sha256(tile_ref.encode()).hexdigest()[:10]}"
        )
        elevation_physical = elevation_physical_root / f"{token}.tif"
        elevation_final = elevation_final_root / f"{token}.tif"
        warped = gdal.Warp(
            str(elevation_physical),
            candidate_paths,
            options=gdal.WarpOptions(
                format="GTiff",
                outputBounds=sampling.as_list(),
                width=width,
                height=height,
                dstSRS=NATIVE_ZONE_CRS,
                outputType=gdal.GDT_Float32,
                resampleAlg="bilinear",
                dstNodata=float("nan"),
                multithread=True,
                creationOptions=[
                    "COMPRESS=DEFLATE",
                    "PREDICTOR=3",
                    "TILED=YES",
                    "BIGTIFF=IF_SAFER",
                ],
            ),
        )
        if warped is None:
            raise TerrainPbrContractError(
                f"GDAL could not prepare halo MNT for {tile_ref}"
            )
        elevation_values = warped.GetRasterBand(1).ReadAsArray()
        _finite_array_metrics(
            np,
            elevation_values,
            label=f"{tile_ref} prepared elevation",
        )
        core = gdal.Translate(
            "",
            warped,
            options=gdal.TranslateOptions(
                format="MEM",
                projWin=[
                    bounds.min_x_m,
                    bounds.max_y_m,
                    bounds.max_x_m,
                    bounds.min_y_m,
                ],
            ),
        )
        if core is None:
            raise TerrainPbrContractError(
                f"GDAL could not extract {tile_ref} analysis core"
            )
        core_elevation = core.GetRasterBand(1).ReadAsArray()
        slope_dataset = gdal.DEMProcessing(
            "",
            core,
            "slope",
            options=gdal.DEMProcessingOptions(
                format="MEM",
                computeEdges=True,
                slopeFormat="degree",
            ),
        )
        roughness_dataset = gdal.DEMProcessing(
            "",
            core,
            "TRI",
            options=gdal.DEMProcessingOptions(
                format="MEM",
                computeEdges=True,
                alg="Riley",
            ),
        )
        if slope_dataset is None or roughness_dataset is None:
            raise TerrainPbrContractError(
                f"GDAL could not analyse relief for {tile_ref}"
            )
        slope_values = slope_dataset.GetRasterBand(1).ReadAsArray()
        roughness_values = roughness_dataset.GetRasterBand(1).ReadAsArray()
        for name, values in (
            ("elevation", core_elevation),
            ("slope", slope_values),
            ("roughness", roughness_values),
        ):
            _finite_array_metrics(
                np,
                values,
                label=f"{tile_ref} {name} analysis",
            )
        slices = _subzone_array_slices(
            bounds=bounds,
            height=int(core.RasterYSize),
            width=int(core.RasterXSize),
        )
        coverage_by_subzone = [
            {
                semantic: 0.0
                for semantic in CLASSIFICATION_SEMANTICS
            }
            for _ in slices
        ]
        mask_width = int(
            round(
                (bounds.max_x_m - bounds.min_x_m)
                / VECTOR_CLASSIFICATION_RESOLUTION_M
            )
        )
        mask_height = int(
            round(
                (bounds.max_y_m - bounds.min_y_m)
                / VECTOR_CLASSIFICATION_RESOLUTION_M
            )
        )
        for semantic, evidence in vector_spatial.items():
            final_vector = volume_root.joinpath(
                *_safe_relative_path(
                    evidence.lock.path,
                    label=f"{semantic} prepared vector",
                ).parts
            )
            physical_vector = physical_evidence_root / (
                final_vector.relative_to(final_evidence_root)
            )
            mask_dataset = gdal.Rasterize(
                "",
                str(physical_vector),
                options=gdal.RasterizeOptions(
                    format="MEM",
                    outputBounds=bounds.as_list(),
                    width=mask_width,
                    height=mask_height,
                    outputSRS=NATIVE_ZONE_CRS,
                    outputType=gdal.GDT_Byte,
                    burnValues=[1],
                    initValues=[0],
                    allTouched=True,
                ),
            )
            if mask_dataset is None:
                raise TerrainPbrContractError(
                    f"GDAL could not analyse {semantic} for {tile_ref}"
                )
            mask = mask_dataset.GetRasterBand(1).ReadAsArray()
            _finite_array_metrics(
                np,
                mask,
                label=f"{tile_ref} {semantic} classification",
            )
            mask_slices = _subzone_array_slices(
                bounds=bounds,
                height=mask_height,
                width=mask_width,
            )
            for subzone_index, (
                _name,
                _bounds,
                rows,
                columns,
            ) in enumerate(mask_slices):
                coverage_by_subzone[subzone_index][semantic] = float(
                    np.mean(mask[rows, columns] > 0)
                )
            mask_dataset = None
        elevation_lock = _prepared_file_lock(
            physical_path=elevation_physical,
            final_path=elevation_final,
            volume_root=volume_root,
        )
        elevation = SpatialEvidence(
            stable_id=f"{tile_ref}:elevation",
            semantic="elevation",
            content_kind="heightfield",
            usage="height_only",
            lock=elevation_lock,
            crs=NATIVE_ZONE_CRS,
            bounds=sampling,
            resolution_m=mnt_resolution,
        )
        evidence = (
            elevation,
            *(vector_spatial[semantic] for semantic in CLASSIFICATION_SEMANTICS),
        )
        evidence_ids = frozenset(
            source.stable_id for source in evidence
        )
        subzones: list[TerrainSubzoneEvidence] = []
        for subzone_index, (
            name,
            subzone_bounds,
            rows,
            columns,
        ) in enumerate(slices):
            subzones.append(
                TerrainSubzoneEvidence(
                    stable_id=f"{tile_ref}:{name}",
                    bounds=subzone_bounds,
                    mean_elevation_m=float(
                        np.mean(core_elevation[rows, columns])
                    ),
                    mean_slope_degrees=float(
                        np.mean(slope_values[rows, columns])
                    ),
                    roughness_m=float(
                        np.mean(roughness_values[rows, columns])
                    ),
                    coverage=coverage_by_subzone[subzone_index],
                    evidence_ids=evidence_ids,
                )
            )
        tiles.append(
            TerrainTileEvidence(
                stable_id=tile_ref,
                bounds=bounds,
                evidence=tuple(evidence),
                subzones=tuple(subzones),
            )
        )
        elevation_records.append(
            {
                "tile_ref": tile_ref,
                "prepared_path": elevation_lock.path,
                "prepared_sha256": elevation_lock.sha256,
                "prepared_size_bytes": elevation_lock.size_bytes,
                "source_bounds_m": bounds.as_list(),
                "sampling_bounds_m": sampling.as_list(),
                "resolution_m": mnt_resolution,
                "source_mnt_locks": [
                    {
                        "path": other["path"],
                        "sha256": other["sha256"],
                        "size_bytes": other["size_bytes"],
                    }
                    for other in context["mnt_sources"].values()
                    if str(other["physical_path"]) in candidate_paths
                ],
            }
        )
        warped = None
        core = None
        slope_dataset = None
        roughness_dataset = None
    provenance: dict[str, object] = {
        "tile_count": len(tiles),
        "crs": NATIVE_ZONE_CRS,
        "vertical_datum": NATIVE_ZONE_VERTICAL_DATUM,
        "scene_bounds_source_m": scene_bounds.as_list(),
        "scene_origin_source_m": list(scene_origin),
        "mnt_resolution_m": mnt_resolution,
        "classification_resolution_m": (
            VECTOR_CLASSIFICATION_RESOLUTION_M
        ),
        "analysis_subzones_per_tile": (
            NATIVE_ANALYSIS_SUBDIVISIONS**2
        ),
        "elevation_tiles": elevation_records,
        "classified_vectors": vector_lineage,
        "global_mask_atlas_created": False,
        "uniform_mask_substitution_allowed": False,
    }
    provenance["prepared_evidence_sha256"] = _canonical_sha256(
        provenance
    )
    return tuple(tiles), provenance


def _native_preparation_input_contract(
    *,
    context: Mapping[str, object],
    bundle_root: Path,
    material_manifest_path: Path,
) -> dict[str, object]:
    """Return the canonical immutable inputs that make a request reusable."""

    volume_root = Path(context["volume_root"])
    marker_path = bundle_root / INSTALL_MARKER
    if (
        not bundle_root.is_dir()
        or bundle_root.is_symlink()
        or not _inside(volume_root, bundle_root)
        or not material_manifest_path.is_file()
        or material_manifest_path.is_symlink()
        or not _inside(bundle_root, material_manifest_path)
        or not marker_path.is_file()
        or marker_path.is_symlink()
    ):
        raise TerrainPbrContractError(
            "native material bundle or manifest is absent or unsafe"
        )

    payloads = context.get("payloads")
    mnt_sources = context.get("mnt_sources")
    vector_sources = context.get("vector_sources")
    if (
        not isinstance(payloads, list)
        or not isinstance(mnt_sources, Mapping)
        or not isinstance(vector_sources, Mapping)
    ):
        raise TerrainPbrContractError(
            "native preparation context lacks locked spatial inputs"
        )

    def volume_record(path: Path) -> dict[str, object]:
        resolved = path.resolve()
        if (
            not resolved.is_file()
            or resolved.is_symlink()
            or not _inside(volume_root, resolved)
        ):
            raise TerrainPbrContractError(
                "native preparation input is absent or unsafe"
            )
        return {
            "path": resolved.relative_to(volume_root).as_posix(),
            "sha256": _sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        }

    locked_payloads = sorted(
        (
            {
                "tile_ref": str(record["tile_ref"]),
                "instance_namespace": int(
                    record["instance_namespace"]
                ),
                "path": str(
                    Path(record["physical_path"])
                    .resolve()
                    .relative_to(volume_root)
                    .as_posix()
                ),
                "sha256": str(record["sha256"]),
                "size_bytes": int(record["size_bytes"]),
            }
            for record in payloads
            if isinstance(record, Mapping)
        ),
        key=lambda value: value["tile_ref"],
    )
    if len(locked_payloads) != NATIVE_ZONE_TILE_COUNT:
        raise TerrainPbrContractError(
            "native preparation payload lock set is incomplete"
        )

    locked_mnt = [
        {
            "tile_ref": str(tile_ref),
            "path": str(record["path"]),
            "sha256": str(record["sha256"]),
            "size_bytes": int(record["size_bytes"]),
        }
        for tile_ref, record in sorted(mnt_sources.items())
        if isinstance(record, Mapping)
    ]
    if len(locked_mnt) != NATIVE_ZONE_TILE_COUNT:
        raise TerrainPbrContractError(
            "native preparation MNT lock set is incomplete"
        )

    locked_vectors: dict[str, list[dict[str, object]]] = {}
    for semantic in CLASSIFICATION_SEMANTICS:
        records = vector_sources.get(semantic)
        if not isinstance(records, list) or not records:
            raise TerrainPbrContractError(
                f"native preparation lacks {semantic} vector locks"
            )
        locked_vectors[semantic] = [
            {
                "path": str(record["path"]),
                "sha256": str(record["sha256"]),
                "size_bytes": int(record["size_bytes"]),
                "bounds_m": record["bounds"].as_list(),
                "feature_count": record["declared_feature_count"],
            }
            for record in records
            if isinstance(record, Mapping)
        ]
        if len(locked_vectors[semantic]) != len(records):
            raise TerrainPbrContractError(
                f"native preparation {semantic} vector locks are malformed"
            )

    contract: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "zone_id": context["zone_id"],
        "crs": NATIVE_ZONE_CRS,
        "vertical_datum": NATIVE_ZONE_VERTICAL_DATUM,
        "scene_origin_source_m": list(
            context["scene_origin_source_m"]
        ),
        "native_build_receipt": volume_record(
            Path(context["build_path"])
        ),
        "scene_auto_validation": volume_record(
            Path(context["auto_path"])
        ),
        "root_usd": volume_record(Path(context["root_usd"])),
        "source_lock": volume_record(
            Path(context["source_lock_path"])
        ),
        "lidar_quality": volume_record(Path(context["lidar_path"])),
        "georeference": volume_record(
            Path(context["georeference_path"])
        ),
        "bundle_install_marker": volume_record(marker_path),
        "material_manifest": volume_record(material_manifest_path),
        "terrain_payloads": locked_payloads,
        "mnt_sources": locked_mnt,
        "classified_vector_sources": locked_vectors,
    }
    contract["input_contract_sha256"] = _canonical_sha256(contract)
    return contract


def _native_preparation_receipt_path(request_path: Path) -> Path:
    return request_path.with_name(
        f"{request_path.stem}.preparation-receipt.json"
    )


def _quarantine_native_preparation_paths(
    *,
    artifact_root: Path,
    paths: Sequence[Path],
) -> Path | None:
    """Move only the exact owned partial outputs to a recoverable folder."""

    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    recovery = (
        artifact_root
        / ".terrain-pbr-recovery"
        / uuid.uuid4().hex
    )
    recovery.mkdir(parents=True)
    for index, path in enumerate(existing):
        if (
            path.is_symlink()
            or not _inside(artifact_root, path)
            or path == artifact_root
        ):
            raise TerrainPbrContractError(
                "partial native preparation path is unsafe to quarantine"
            )
        destination = recovery / f"{index:02d}-{path.name}"
        os.replace(path, destination)
    return recovery


def terrain_pbr_authoring_request(
    *,
    plan: TerrainPbrPlan,
    bundle_root: Path,
    evidence_root: Path,
    artifact_root: Path,
    output_relative_path: str,
    receipt_relative_path: str,
) -> dict[str, object]:
    """Serialize every planned tile into the native CLI request contract."""

    output_relative = _safe_relative_path(
        output_relative_path,
        label="terrain PBR request output",
    )
    receipt_relative = _safe_relative_path(
        receipt_relative_path,
        label="terrain PBR request receipt",
    )
    origins = {
        (contract.origin_x_m, contract.origin_y_m)
        for contract in plan.metric_uv
    }
    if len(origins) != 1:
        raise TerrainPbrContractError(
            "terrain PBR plan has inconsistent world UV origins"
        )
    origin = next(iter(origins))
    return {
        "schema_version": SCHEMA_VERSION,
        "scene_id": plan.scene_id,
        "bundle_root": str(bundle_root.resolve()),
        "material_manifest_path": plan.material_library.manifest_path,
        "evidence_root": str(evidence_root.resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "output_relative_path": output_relative.as_posix(),
        "receipt_relative_path": receipt_relative.as_posix(),
        "world_uv_origin_m": [origin[0], origin[1]],
        "scene_origin_source_m": list(plan.scene_origin_source_m),
        "plan_sha256": plan.fingerprint,
        "evidence_inventory_sha256": plan.evidence_inventory_sha256,
        "tiles": [
            {
                "stable_id": tile.stable_id,
                "bounds_m": tile.bounds.as_list(),
                "evidence": [
                    {
                        "stable_id": source.stable_id,
                        "semantic": source.semantic,
                        "content_kind": source.content_kind,
                        "usage": source.usage,
                        "lock": {
                            "path": source.lock.path,
                            "sha256": source.lock.sha256,
                            "size_bytes": source.lock.size_bytes,
                        },
                        "crs": source.crs,
                        "bounds_m": source.bounds.as_list(),
                        "resolution_m": source.resolution_m,
                        **(
                            {"feature_count": source.feature_count}
                            if source.feature_count is not None
                            else {}
                        ),
                    }
                    for source in tile.evidence
                ],
                "subzones": [
                    {
                        "stable_id": subzone.stable_id,
                        "bounds_m": subzone.bounds.as_list(),
                        "mean_elevation_m": subzone.mean_elevation_m,
                        "mean_slope_degrees": (
                            subzone.mean_slope_degrees
                        ),
                        "roughness_m": subzone.roughness_m,
                        "coverage": {
                            semantic: value
                            for semantic, value in subzone.coverage
                        },
                        "evidence_ids": list(subzone.evidence_ids),
                    }
                    for subzone in tile.subzones
                ],
            }
            for tile in plan.tiles
        ],
    }


def write_terrain_pbr_authoring_request(
    path: Path,
    *,
    plan: TerrainPbrPlan,
    bundle_root: Path,
    evidence_root: Path,
    artifact_root: Path,
    output_relative_path: str,
    receipt_relative_path: str,
) -> dict[str, object]:
    """Atomically write the complete request; no per-tile manual input."""

    request = terrain_pbr_authoring_request(
        plan=plan,
        bundle_root=bundle_root,
        evidence_root=evidence_root,
        artifact_root=artifact_root,
        output_relative_path=output_relative_path,
        receipt_relative_path=receipt_relative_path,
    )
    if path.exists():
        existing = _read_json(path, label="terrain PBR authoring request")
        if existing != request:
            raise TerrainPbrContractError(
                "terrain PBR authoring request already exists with other inputs"
            )
        return request
    _atomic_write_json(path, request)
    return request


def _bounds_from_json(value: object, *, label: str) -> Bounds2d:
    if not isinstance(value, list) or len(value) != 4:
        raise TerrainPbrContractError(
            f"{label} must contain four metric bounds"
        )
    try:
        coordinates = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TerrainPbrContractError(
            f"{label} contains a non-numeric bound"
        ) from exc
    return Bounds2d(*coordinates)


def _file_lock_from_json(value: object, *, label: str) -> FileLock:
    if not isinstance(value, dict):
        raise TerrainPbrContractError(f"{label} lock must be an object")
    return FileLock(
        path=str(value.get("path", "")),
        sha256=str(value.get("sha256", "")),
        size_bytes=value.get("size_bytes"),
    )


def _tile_from_json(value: object, *, index: int) -> TerrainTileEvidence:
    if not isinstance(value, dict):
        raise TerrainPbrContractError(
            f"tiles[{index}] must be an object"
        )
    tile_bounds = _bounds_from_json(
        value.get("bounds_m"),
        label=f"tiles[{index}].bounds_m",
    )
    raw_evidence = value.get("evidence")
    if not isinstance(raw_evidence, list):
        raise TerrainPbrContractError(
            f"tiles[{index}].evidence must be a list"
        )
    evidence: list[SpatialEvidence] = []
    for source_index, source in enumerate(raw_evidence):
        if not isinstance(source, dict):
            raise TerrainPbrContractError(
                f"tiles[{index}].evidence[{source_index}] must be an object"
            )
        evidence.append(
            SpatialEvidence(
                stable_id=str(source.get("stable_id", "")),
                semantic=str(source.get("semantic", "")),
                content_kind=str(source.get("content_kind", "")),
                usage=str(source.get("usage", "")),
                lock=_file_lock_from_json(
                    source.get("lock"),
                    label=(
                        f"tiles[{index}].evidence[{source_index}]"
                    ),
                ),
                crs=str(source.get("crs", "")),
                bounds=_bounds_from_json(
                    source.get("bounds_m"),
                    label=(
                        f"tiles[{index}].evidence[{source_index}].bounds_m"
                    ),
                ),
                resolution_m=float(source.get("resolution_m", 0.0)),
                feature_count=source.get("feature_count"),
            )
        )
    raw_subzones = value.get("subzones")
    if not isinstance(raw_subzones, list):
        raise TerrainPbrContractError(
            f"tiles[{index}].subzones must be a list"
        )
    subzones: list[TerrainSubzoneEvidence] = []
    for subzone_index, subzone in enumerate(raw_subzones):
        if not isinstance(subzone, dict):
            raise TerrainPbrContractError(
                f"tiles[{index}].subzones[{subzone_index}] must be an object"
            )
        coverage = subzone.get("coverage")
        evidence_ids = subzone.get("evidence_ids")
        if not isinstance(coverage, dict) or not isinstance(
            evidence_ids,
            list,
        ):
            raise TerrainPbrContractError(
                f"tiles[{index}].subzones[{subzone_index}] is incomplete"
            )
        subzones.append(
            TerrainSubzoneEvidence(
                stable_id=str(subzone.get("stable_id", "")),
                bounds=_bounds_from_json(
                    subzone.get("bounds_m"),
                    label=(
                        f"tiles[{index}].subzones[{subzone_index}].bounds_m"
                    ),
                ),
                mean_elevation_m=float(
                    subzone.get("mean_elevation_m", float("nan"))
                ),
                mean_slope_degrees=float(
                    subzone.get("mean_slope_degrees", float("nan"))
                ),
                roughness_m=float(
                    subzone.get("roughness_m", float("nan"))
                ),
                coverage=coverage,
                evidence_ids=frozenset(str(item) for item in evidence_ids),
            )
        )
    return TerrainTileEvidence(
        stable_id=str(value.get("stable_id", "")),
        bounds=tile_bounds,
        evidence=tuple(evidence),
        subzones=tuple(subzones),
    )


def _terrain_pbr_plan_from_request(
    request: Mapping[str, object],
) -> tuple[TerrainPbrPlan, Path, Path, Path]:
    """Rebuild and revalidate every lock represented by a request."""

    if request.get("schema_version") != SCHEMA_VERSION:
        raise TerrainPbrContractError(
            "terrain PBR authoring request schema_version is unsupported"
        )
    raw_roots = {
        "bundle": Path(str(request.get("bundle_root", ""))),
        "evidence": Path(str(request.get("evidence_root", ""))),
        "artifact": Path(str(request.get("artifact_root", ""))),
    }
    if any(not path.is_absolute() for path in raw_roots.values()):
        raise TerrainPbrContractError(
            "terrain PBR request roots must be absolute"
        )
    bundle_root = raw_roots["bundle"].resolve()
    evidence_root = raw_roots["evidence"].resolve()
    artifact_root = raw_roots["artifact"].resolve()
    if (
        not bundle_root.is_dir()
        or bundle_root.is_symlink()
        or not evidence_root.is_dir()
        or evidence_root.is_symlink()
        or not artifact_root.is_dir()
        or artifact_root.is_symlink()
    ):
        raise TerrainPbrContractError(
            "terrain PBR request roots are absent or unsafe"
        )
    manifest = Path(str(request.get("material_manifest_path", "")))
    if not manifest.is_absolute():
        manifest = bundle_root / manifest
    origin = request.get("world_uv_origin_m", [0.0, 0.0])
    if not isinstance(origin, list) or len(origin) != 2:
        raise TerrainPbrContractError(
            "terrain PBR request world_uv_origin_m must contain two metres"
        )
    scene_origin = request.get("scene_origin_source_m")
    if not isinstance(scene_origin, list) or len(scene_origin) != 2:
        raise TerrainPbrContractError(
            "terrain PBR request scene_origin_source_m must contain two metres"
        )
    raw_tiles = request.get("tiles")
    if not isinstance(raw_tiles, list):
        raise TerrainPbrContractError(
            "terrain PBR authoring request tiles must be a list"
        )
    tiles = tuple(
        _tile_from_json(tile, index=index)
        for index, tile in enumerate(raw_tiles)
    )
    plan = build_terrain_pbr_plan(
        scene_id=str(request.get("scene_id", "")),
        bundle_root=bundle_root,
        material_manifest_path=manifest,
        evidence_root=evidence_root,
        tiles=tiles,
        world_uv_origin_m=(float(origin[0]), float(origin[1])),
        scene_origin_source_m=(
            float(scene_origin[0]),
            float(scene_origin[1]),
        ),
    )
    if (
        request.get("plan_sha256") != plan.fingerprint
        or request.get("evidence_inventory_sha256")
        != plan.evidence_inventory_sha256
    ):
        raise TerrainPbrContractError(
            "terrain PBR authoring request drifted from its planned tiles"
        )
    return plan, bundle_root, evidence_root, artifact_root


def _validated_existing_native_preparation(
    *,
    request_path: Path,
    preparation_receipt_path: Path,
    prepared_evidence_root: Path,
    artifact_root: Path,
    expected_input_contract: Mapping[str, object],
    expected_bundle_root: Path,
    expected_material_manifest_path: Path,
    expected_volume_root: Path,
    expected_output_relative_path: PurePosixPath,
    expected_receipt_relative_path: PurePosixPath,
) -> dict[str, object]:
    """Validate a completed preparation without regenerating spatial data."""

    request = _read_json(
        request_path,
        label="terrain PBR authoring request",
    )
    receipt = _read_json(
        preparation_receipt_path,
        label="terrain PBR preparation receipt",
    )
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("state") != NATIVE_PREPARATION_STATE
        or receipt.get("input_contract") != expected_input_contract
        or receipt.get("input_contract_sha256")
        != expected_input_contract.get("input_contract_sha256")
        or receipt.get("tile_count") != NATIVE_ZONE_TILE_COUNT
        or receipt.get("global_mask_atlas_created") is not False
        or receipt.get("uniform_mask_substitution_allowed") is not False
    ):
        raise TerrainPbrContractError(
            "existing native preparation receipt is stale or incomplete"
        )
    request_record = receipt.get("authoring_request")
    if not isinstance(request_record, Mapping):
        raise TerrainPbrContractError(
            "existing native preparation lacks its request lock"
        )
    try:
        request_relative = _safe_relative_path(
            str(request_record.get("path", "")),
            label="prepared authoring request",
        )
    except TerrainPbrContractError:
        raise
    locked_request_path = artifact_root.joinpath(
        *request_relative.parts
    ).resolve()
    if (
        locked_request_path != request_path
        or request_path.is_symlink()
        or request_record.get("sha256") != _sha256_file(request_path)
        or request_record.get("size_bytes") != request_path.stat().st_size
    ):
        raise TerrainPbrContractError(
            "existing native preparation request lock drifted"
        )
    provenance_record = receipt.get("prepared_evidence_provenance")
    if not isinstance(provenance_record, Mapping):
        raise TerrainPbrContractError(
            "existing native preparation lacks provenance"
        )
    provenance_relative = _safe_relative_path(
        str(provenance_record.get("path", "")),
        label="prepared evidence provenance",
    )
    provenance_path = expected_volume_root.joinpath(
        *provenance_relative.parts
    ).resolve()
    if (
        not _inside(prepared_evidence_root, provenance_path)
        or not provenance_path.is_file()
        or provenance_path.is_symlink()
        or provenance_record.get("sha256")
        != _sha256_file(provenance_path)
        or provenance_record.get("size_bytes")
        != provenance_path.stat().st_size
    ):
        raise TerrainPbrContractError(
            "existing native preparation provenance lock drifted"
        )
    provenance = _read_json(
        provenance_path,
        label="prepared evidence provenance",
    )
    if (
        provenance.get("tile_count") != NATIVE_ZONE_TILE_COUNT
        or provenance.get("global_mask_atlas_created") is not False
        or provenance.get("uniform_mask_substitution_allowed") is not False
        or provenance.get("prepared_evidence_sha256")
        != receipt.get("prepared_evidence_sha256")
    ):
        raise TerrainPbrContractError(
            "existing native preparation provenance is incomplete"
        )
    if (
        request.get("bundle_root") != str(expected_bundle_root)
        or request.get("evidence_root") != str(expected_volume_root)
        or request.get("artifact_root") != str(artifact_root)
        or request.get("material_manifest_path")
        != expected_material_manifest_path.relative_to(
            expected_bundle_root
        ).as_posix()
        or request.get("output_relative_path")
        != expected_output_relative_path.as_posix()
        or request.get("receipt_relative_path")
        != expected_receipt_relative_path.as_posix()
        or not isinstance(request.get("tiles"), list)
        or len(request["tiles"]) != NATIVE_ZONE_TILE_COUNT
    ):
        raise TerrainPbrContractError(
            "existing native preparation request differs from its command"
        )
    plan, bundle_root, evidence_root, request_artifact_root = (
        _terrain_pbr_plan_from_request(request)
    )
    if (
        bundle_root != expected_bundle_root
        or evidence_root != expected_volume_root
        or request_artifact_root != artifact_root
        or receipt.get("terrain_pbr_plan_sha256") != plan.fingerprint
        or receipt.get("evidence_inventory_sha256")
        != plan.evidence_inventory_sha256
    ):
        raise TerrainPbrContractError(
            "existing native preparation plan lock drifted"
        )
    return dict(request)


def prepare_native_terrain_pbr_request(
    *,
    volume_root: Path,
    zone_root: Path,
    scene_auto_validation_path: Path,
    bundle_root: Path,
    material_manifest_path: Path,
    artifact_root: Path,
    request_path: Path,
    output_relative_path: str = "authored/ground.usdc",
    receipt_relative_path: str = (
        "authored/ground-authoring-receipt.json"
    ),
) -> dict[str, object]:
    """Prepare the complete locked 400-tile request for native authoring.

    The accepted Znn build remains immutable.  Derived elevation is produced
    as one halo-complete raster per terrain payload, while the four semantic
    classes remain explicit source-backed vectors.  A crash can leave only a
    bounded staging folder or an exact partial package, which is quarantined
    rather than overwritten on the next run.
    """

    volume = volume_root.resolve()
    if (
        not volume.is_dir()
        or volume.is_symlink()
    ):
        raise TerrainPbrContractError(
            "persistent volume root must be a real directory"
        )
    artifacts = artifact_root.resolve()
    if (
        artifacts == volume
        or not _inside(volume, artifacts)
        or artifacts.is_symlink()
    ):
        raise TerrainPbrContractError(
            "terrain PBR artifact root must stay below the volume root"
        )
    if not artifacts.exists():
        parent = artifacts.parent.resolve()
        if (
            not parent.is_dir()
            or parent.is_symlink()
            or not _inside(volume, parent)
        ):
            raise TerrainPbrContractError(
                "terrain PBR artifact parent is absent or unsafe"
            )
        artifacts.mkdir()
    if not artifacts.is_dir() or artifacts.is_symlink():
        raise TerrainPbrContractError(
            "terrain PBR artifact root must be a real directory"
        )

    bundle = bundle_root.resolve()
    manifest = material_manifest_path
    if not manifest.is_absolute():
        manifest = bundle / manifest
    manifest = manifest.resolve()
    request = request_path.resolve()
    if (
        request.suffix.casefold() != ".json"
        or not _inside(artifacts, request)
        or request.parent.is_symlink()
    ):
        raise TerrainPbrContractError(
            "terrain PBR request output must be JSON below its artifact root"
        )
    output_relative = _safe_relative_path(
        output_relative_path,
        label="terrain PBR request output",
    )
    authoring_receipt_relative = _safe_relative_path(
        receipt_relative_path,
        label="terrain PBR request receipt",
    )
    if (
        output_relative.parent == PurePosixPath(".")
        or output_relative.parent != authoring_receipt_relative.parent
        or output_relative == authoring_receipt_relative
        or output_relative.parts[0]
        in {"prepared-evidence", ".terrain-pbr-recovery"}
    ):
        raise TerrainPbrContractError(
            "native output and receipt require one dedicated package folder"
        )

    context = _native_preparation_inputs(
        volume_root=volume,
        zone_root=zone_root,
        scene_auto_validation_path=scene_auto_validation_path,
    )
    input_contract = _native_preparation_input_contract(
        context=context,
        bundle_root=bundle,
        material_manifest_path=manifest,
    )
    preparation_receipt = _native_preparation_receipt_path(request)
    prepared_evidence = artifacts / "prepared-evidence"
    authoring_package = artifacts.joinpath(
        *output_relative.parent.parts
    ).resolve()
    if (
        authoring_package == prepared_evidence
        or _inside(authoring_package, request)
        or _inside(prepared_evidence, request)
    ):
        raise TerrainPbrContractError(
            "request, prepared evidence and authored package must be isolated"
        )
    completed = (
        request.is_file()
        and preparation_receipt.is_file()
        and prepared_evidence.is_dir()
    )
    partial_exists = any(
        path.exists()
        for path in (
            request,
            preparation_receipt,
            prepared_evidence,
        )
    )
    if completed:
        return _validated_existing_native_preparation(
            request_path=request,
            preparation_receipt_path=preparation_receipt,
            prepared_evidence_root=prepared_evidence,
            artifact_root=artifacts,
            expected_input_contract=input_contract,
            expected_bundle_root=bundle,
            expected_material_manifest_path=manifest,
            expected_volume_root=volume,
            expected_output_relative_path=output_relative,
            expected_receipt_relative_path=authoring_receipt_relative,
        )
    if partial_exists:
        _quarantine_native_preparation_paths(
            artifact_root=artifacts,
            paths=(
                request,
                preparation_receipt,
                prepared_evidence,
            ),
        )

    staging_root = (
        artifacts
        / f".terrain-pbr-{uuid.uuid4().hex}.staging"
    )
    physical_evidence = staging_root / "prepared-evidence"
    staging_root.mkdir()
    promoted = False
    try:
        tiles, provenance = _derive_native_zone_terrain_tiles(
            context=context,
            physical_evidence_root=physical_evidence,
            final_evidence_root=prepared_evidence,
        )
        if len(tiles) != NATIVE_ZONE_TILE_COUNT:
            raise TerrainPbrContractError(
                "native derivation did not produce exactly 400 terrain tiles"
            )
        provenance_physical = (
            physical_evidence / "preparation-provenance.json"
        )
        provenance_final = (
            prepared_evidence / "preparation-provenance.json"
        )
        _atomic_write_json(provenance_physical, provenance)
        provenance_lock = _prepared_file_lock(
            physical_path=provenance_physical,
            final_path=provenance_final,
            volume_root=volume,
        )
        os.replace(physical_evidence, prepared_evidence)
        promoted = True

        plan = build_terrain_pbr_plan(
            scene_id=str(context["zone_id"]),
            bundle_root=bundle,
            material_manifest_path=manifest,
            evidence_root=volume,
            tiles=tiles,
            world_uv_origin_m=(0.0, 0.0),
            scene_origin_source_m=tuple(
                context["scene_origin_source_m"]
            ),
        )
        authored_request = write_terrain_pbr_authoring_request(
            request,
            plan=plan,
            bundle_root=bundle,
            evidence_root=volume,
            artifact_root=artifacts,
            output_relative_path=output_relative.as_posix(),
            receipt_relative_path=(
                authoring_receipt_relative.as_posix()
            ),
        )
        request_lock = {
            "path": request.relative_to(artifacts).as_posix(),
            "sha256": _sha256_file(request),
            "size_bytes": request.stat().st_size,
        }
        receipt_payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "state": NATIVE_PREPARATION_STATE,
            "zone_id": context["zone_id"],
            "tile_count": len(tiles),
            "crs": NATIVE_ZONE_CRS,
            "vertical_datum": NATIVE_ZONE_VERTICAL_DATUM,
            "scene_origin_source_m": list(
                context["scene_origin_source_m"]
            ),
            "input_contract": input_contract,
            "input_contract_sha256": input_contract[
                "input_contract_sha256"
            ],
            "prepared_evidence_provenance": {
                "path": provenance_lock.path,
                "sha256": provenance_lock.sha256,
                "size_bytes": provenance_lock.size_bytes,
            },
            "prepared_evidence_sha256": provenance[
                "prepared_evidence_sha256"
            ],
            "terrain_pbr_plan_sha256": plan.fingerprint,
            "evidence_inventory_sha256": (
                plan.evidence_inventory_sha256
            ),
            "authoring_request": request_lock,
            "global_mask_atlas_created": False,
            "uniform_mask_substitution_allowed": False,
            "bounded_topology": {
                "terrain_payload_count": NATIVE_ZONE_TILE_COUNT,
                "elevation_raster_per_payload": True,
                "classified_vector_layers": list(
                    CLASSIFICATION_SEMANTICS
                ),
                "monolithic_scene_raster": False,
            },
        }
        _atomic_write_json(preparation_receipt, receipt_payload)
        validated = _validated_existing_native_preparation(
            request_path=request,
            preparation_receipt_path=preparation_receipt,
            prepared_evidence_root=prepared_evidence,
            artifact_root=artifacts,
            expected_input_contract=input_contract,
            expected_bundle_root=bundle,
            expected_material_manifest_path=manifest,
            expected_volume_root=volume,
            expected_output_relative_path=output_relative,
            expected_receipt_relative_path=authoring_receipt_relative,
        )
        if validated != authored_request:
            raise TerrainPbrContractError(
                "prepared native authoring request changed after validation"
            )
        return authored_request
    except Exception:
        if promoted or request.exists() or preparation_receipt.exists():
            try:
                _quarantine_native_preparation_paths(
                    artifact_root=artifacts,
                    paths=(
                        request,
                        preparation_receipt,
                        prepared_evidence,
                    ),
                )
            except Exception:
                pass
        raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def _author_from_request(
    *,
    request_path: Path,
    backend: CompositeGroundAuthoringBackend | None,
) -> CompositeGroundMaterialArtifact:
    request = _read_json(request_path, label="terrain PBR authoring request")
    plan, bundle_root, evidence_root, artifact_root = (
        _terrain_pbr_plan_from_request(request)
    )
    return author_composite_ground_material(
        plan=plan,
        artifact_root=artifact_root,
        bundle_root=bundle_root,
        evidence_root=evidence_root,
        output_relative_path=str(request.get("output_relative_path", "")),
        receipt_relative_path=str(request.get("receipt_relative_path", "")),
        backend=backend,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    backend: CompositeGroundAuthoringBackend | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Author a native object-free composite terrain material under "
            "Kit/OpenUSD"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    author_parser = subparsers.add_parser(
        "author-native",
        help="author and inspect the real tiled MaterialX ground USD",
    )
    author_parser.add_argument(
        "--request",
        type=Path,
        required=True,
        help="JSON request containing locked bundle, evidence and output roots",
    )
    prepare_parser = subparsers.add_parser(
        "prepare-native",
        help=(
            "derive the locked 400-tile request from an accepted native zone"
        ),
    )
    prepare_parser.add_argument(
        "--volume-root",
        type=Path,
        required=True,
        help="persistent pod volume containing all immutable inputs",
    )
    prepare_parser.add_argument(
        "--zone-root",
        type=Path,
        required=True,
        help="accepted Znn native-zone root below the persistent volume",
    )
    prepare_parser.add_argument(
        "--scene-auto-validation",
        type=Path,
        required=True,
        help="AUTO_VALIDATED receipt bound to the accepted native build",
    )
    prepare_parser.add_argument(
        "--bundle-root",
        type=Path,
        required=True,
        help="installed and hash-locked native material bundle",
    )
    prepare_parser.add_argument(
        "--material-manifest",
        type=Path,
        required=True,
        help="locked seven-role PBR material manifest",
    )
    prepare_parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="dedicated ground preparation and authoring artifact root",
    )
    prepare_parser.add_argument(
        "--request-output",
        type=Path,
        required=True,
        help="JSON request to pass unchanged to author-native",
    )
    prepare_parser.add_argument(
        "--output-relative-path",
        default="authored/ground.usdc",
        help="final ground index path relative to the artifact root",
    )
    prepare_parser.add_argument(
        "--receipt-relative-path",
        default="authored/ground-authoring-receipt.json",
        help="final authoring receipt path relative to the artifact root",
    )
    args = parser.parse_args(argv)
    if args.command == "prepare-native":
        request = prepare_native_terrain_pbr_request(
            volume_root=args.volume_root,
            zone_root=args.zone_root,
            scene_auto_validation_path=args.scene_auto_validation,
            bundle_root=args.bundle_root,
            material_manifest_path=args.material_manifest,
            artifact_root=args.artifact_root,
            request_path=args.request_output,
            output_relative_path=args.output_relative_path,
            receipt_relative_path=args.receipt_relative_path,
        )
        print(
            json.dumps(
                {
                    "state": NATIVE_PREPARATION_STATE,
                    "request_path": str(
                        args.request_output.resolve()
                    ),
                    "scene_id": request["scene_id"],
                    "tile_count": len(request["tiles"]),
                    "plan_sha256": request["plan_sha256"],
                    "evidence_inventory_sha256": request[
                        "evidence_inventory_sha256"
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command != "author-native":
        parser.error("unsupported terrain PBR command")
    artifact = _author_from_request(
        request_path=args.request.resolve(),
        backend=backend,
    )
    print(
        json.dumps(
            artifact.as_layout_artifact(),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ALGORITHM_ID",
    "Bounds2d",
    "CLASSIFICATION_SEMANTICS",
    "COMPOSITE_AUTHORING_SCHEMA_VERSION",
    "CompositeGroundMaterialArtifact",
    "CompositeGroundAuthoringBackend",
    "CompositeGroundMaterialSpec",
    "EVIDENCE_SEMANTICS",
    "FileLock",
    "LockedMaterial",
    "LockedMaterialLibrary",
    "MASK_PRIORITY",
    "MaterialTexture",
    "MetricUv",
    "NATIVE_COMPOSITE_INSPECTOR_ID",
    "NATIVE_GROUND_STATE",
    "NATIVE_PREPARATION_STATE",
    "SCHEMA_VERSION",
    "SpatialEvidence",
    "SubzoneMaterialPlan",
    "TerrainPbrContractError",
    "TerrainPbrPlan",
    "TerrainSubzoneEvidence",
    "TerrainTileEvidence",
    "TileMaterialPlan",
    "author_composite_ground_material",
    "build_composite_ground_material_spec",
    "build_terrain_pbr_plan",
    "load_locked_material_library",
    "main",
    "prepare_native_terrain_pbr_request",
    "terrain_pbr_authoring_request",
    "validate_composite_ground_material_artifact",
    "write_terrain_pbr_authoring_request",
]


if __name__ == "__main__":
    raise SystemExit(main())

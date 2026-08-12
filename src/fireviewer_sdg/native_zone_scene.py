"""Native Isaac/OpenUSD builder for one fully locked geographic zone.

This worker is intentionally launched by the Windows Isaac Sim Python runtime.
It turns downloaded, hash-verified MNT and BDTOPO GeoJSON inputs into real USD
payloads; it never writes the supplied external catalogue and it refuses to
manufacture source-free scene layers.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from fireviewer_sdg.artifacts import sha256, write_json
from fireviewer_sdg.zone_scenes import (
    BASELINE_DATASETS,
    VECTOR_SOURCE_LAYERS,
    _is_below,
    _load_source_lock,
    _read_json,
    _review_camera_lod0_tiles,
    _zone_manifest,
    _zone_rows,
)


COORDINATE_CONVENTION = "usd_z_up_meters_lambert93"
TERRAIN_LOD0_SAMPLES = 2001  # Native 0.5 m LiDAR grid around review cameras.
TERRAIN_LOD1_SAMPLES = 257  # ~4 m interactive mid-distance representation.
TERRAIN_LOD2_SAMPLES = 129  # ~8 m over a one-kilometre payload.
TERRAIN_LOD3_SAMPLES = 65  # ~16 m distant representation.
COLLISION_SAMPLES = 32  # ~32 m, separate from the render mesh.
HERO_TREE_INSTANCE_LIMIT = 360
FOREST_CANOPY_INSTANCE_BUDGET = 36_000
PHOTOREAL_FOREST_INSTANCE_BUDGET = 2_500_000
PHOTOREAL_FOREST_AREA_PER_INSTANCE_M2 = 50.0
PHOTOREAL_BUILDING_INSTANCE_LIMIT = 20_000
CONTINUOUS_TERRAIN_SAMPLES = 769  # ~26 m across the 20 km visual terrain.
# The native RTX path resolves this JPEG reliably at 4096 but returns a black
# texture at 8192 on the installed Composer.  Keep the overview renderer
# proven instead of publishing a higher-resolution black viewport; dedicated
# close-camera imagery remains separately locked for the next LOD pass.
CONTINUOUS_ORTHO_PIXELS = 4096
MAX_RASTER_NODATA_RATIO = 0.01
MIN_CANOPY_HEIGHT_METRES = 3.0
CANOPY_NMS_RADIUS_METRES = 2.5
LOCKED_SOURCE_FINAL_STATES = frozenset(
    {
        "downloaded",
        "downloaded_segmented",
        "verified_existing",
        "recovered_complete_partial",
    }
)
_INSTANCE_FAMILY_CODES = {
    "buildings": 1,
    "trees": 2,
    "shrubs": 3,
    "understory": 4,
}


def _positive_int_environment(name: str, default: int, *, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    value = int(raw) if raw else default
    if value <= 0 or value > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _stable_instance_id(
    *, tile_namespace: int, family: str, local_index: int
) -> int:
    """Encode an auditable globally unique signed-int64 instance ID."""

    if not 1 <= tile_namespace < (1 << 20):
        raise ValueError("tile namespace must be between 1 and 1048575")
    family_code = _INSTANCE_FAMILY_CODES.get(family)
    if family_code is None:
        raise ValueError(f"unsupported instance family: {family}")
    if not 0 <= local_index < (1 << 39):
        raise ValueError("local instance index exceeds the signed-int64 contract")
    return (tile_namespace << 43) | (family_code << 39) | local_index


def _author_instance_identity_primvars(
    *,
    instancer: Any,
    stable_ids: list[str],
    footprint_radii_m: list[float],
    group_ids: list[str],
    usd_geom: Any,
    sdf: Any,
) -> None:
    """Expose the exact per-instance contract used by variant authoring.

    Numeric IDs remain on ``PointInstancer.ids``.  These three vertex
    primvars add portable string identity, the real ground footprint and a
    deterministic spatial group without loading referenced prototype meshes.
    """

    count = len(stable_ids)
    if (
        count < 1
        or len(footprint_radii_m) != count
        or len(group_ids) != count
        or len(set(stable_ids)) != count
        or any(not value for value in stable_ids)
        or any(not value for value in group_ids)
        or any(
            not math.isfinite(value) or value <= 0.0
            for value in footprint_radii_m
        )
    ):
        raise RuntimeError("invalid per-instance identity primvar contract")
    api = usd_geom.PrimvarsAPI(instancer)
    stable = api.CreatePrimvar(
        "fireviewer_stable_id",
        sdf.ValueTypeNames.StringArray,
        usd_geom.Tokens.vertex,
    )
    radius = api.CreatePrimvar(
        "fireviewer_footprint_radius_m",
        sdf.ValueTypeNames.FloatArray,
        usd_geom.Tokens.vertex,
    )
    group = api.CreatePrimvar(
        "fireviewer_group_id",
        sdf.ValueTypeNames.StringArray,
        usd_geom.Tokens.vertex,
    )
    stable.Set(stable_ids)
    radius.Set(footprint_radii_m)
    group.Set(group_ids)
    prim = instancer.GetPrim()
    prim.SetCustomDataByKey(
        "fireviewer:instance_identity_contract",
        "ids+stable_id+footprint_radius_m+group_id",
    )


def _apply_training_semantics(prim: Any, semantic: str) -> None:
    """Author the Omniverse SemanticsAPI consumed by Replicator annotators."""

    try:
        from pxr import Semantics
    except ImportError as exc:  # pragma: no cover - native Kit gate
        raise RuntimeError(
            "the native Isaac runtime has no pxr.Semantics schema"
        ) from exc
    api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    if not api:
        raise RuntimeError(f"could not apply SemanticsAPI to {prim.GetPath()}")
    api.CreateSemanticTypeAttr().Set("class")
    api.CreateSemanticDataAttr().Set(semantic)
    prim.SetCustomDataByKey("fireviewer:semantic_class", semantic)


def _photoreal_asset_contract(
    *,
    workspace_root: Path,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any] | None]:
    """Resolve the fully materialized shared USD asset lock for pod builds."""

    configured = os.getenv("FW_SDG_SIMREADY_ASSET_MANIFEST", "").strip()
    required = os.getenv("FW_SDG_PHOTOREAL_ASSETS_REQUIRED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not configured:
        if required:
            raise RuntimeError(
                "photoreal pod build requires FW_SDG_SIMREADY_ASSET_MANIFEST"
            )
        return {"vegetation": {}, "buildings": {}}, None
    manifest = Path(configured).expanduser().resolve()
    volume = Path(
        os.getenv("FW_OMNI_VOLUME_ROOT", str(workspace_root))
    ).expanduser().resolve()
    skip_content_validation = os.getenv(
        "FW_SDG_SKIP_ASSET_CONTENT_VALIDATION", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if skip_content_validation:
        validation = {
            "state": "SKIPPED_FOR_EDITOR_PREVIEW",
            "reason": "explicit fast-preview mode; wrapper and LOD paths remain checked",
        }
    else:
        from fireviewer_sdg.omniverse_pod import validate_materialized_assets

        validation = validate_materialized_assets(
            manifest_path=manifest,
            volume_root=volume,
        )
    payload = _read_json(manifest, label="SimReady asset manifest")
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("SimReady asset manifest has no environment object")
    vegetation = environment.get("vegetation")
    buildings = environment.get("buildings")
    if not isinstance(vegetation, dict) or not isinstance(buildings, dict):
        raise ValueError("SimReady environment assets are malformed")

    def wrapper(item: dict[str, Any], *, label: str) -> Path:
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{label} wrapper must use a safe relative path")
        path = (manifest.parent / relative).resolve()
        if not _is_below(volume, path) or not path.is_file():
            raise RuntimeError(f"{label} wrapper is absent from the shared volume")
        return path

    def lod_wrappers(item: dict[str, Any], *, label: str) -> dict[str, Path]:
        records = item.get("lod_paths")
        if not isinstance(records, dict) or set(records) != {
            "HERO",
            "MID",
            "FAR",
        }:
            raise ValueError(
                f"{label} must expose distinct HERO/MID/FAR USD wrappers"
            )
        result: dict[str, Path] = {}
        for level, record in records.items():
            raw = record.get("path") if isinstance(record, dict) else record
            relative = Path(str(raw or ""))
            if (
                not str(raw or "").strip()
                or relative.is_absolute()
                or ".." in relative.parts
            ):
                raise ValueError(f"{label}.{level} wrapper path is unsafe")
            path = (manifest.parent / relative).resolve()
            if not _is_below(volume, path) or not path.is_file():
                raise RuntimeError(
                    f"{label}.{level} wrapper is absent from the shared volume"
                )
            result[str(level)] = path
        if len({path.resolve() for path in result.values()}) != 3:
            raise ValueError(f"{label} LOD wrappers must be three distinct files")
        return result

    resolved: dict[str, dict[str, list[dict[str, Any]]]] = {
        "vegetation": {},
        "buildings": {},
    }
    for kind, families in (("vegetation", vegetation), ("buildings", buildings)):
        for family, entries in families.items():
            if not isinstance(entries, list):
                raise ValueError(f"SimReady environment {kind}.{family} is malformed")
            resolved_entries: list[dict[str, Any]] = []
            for index, item in enumerate(entries):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"SimReady environment {kind}.{family}[{index}] is malformed"
                    )
                dimensions = item.get("native_dimensions_m")
                ground_anchor = item.get("ground_anchor_m")
                placement = item.get("placement")
                if (
                    not isinstance(dimensions, dict)
                    or not isinstance(ground_anchor, list)
                    or len(ground_anchor) != 3
                    or not isinstance(placement, dict)
                ):
                    if skip_content_validation:
                        continue
                    raise ValueError(
                        f"SimReady environment {kind}.{family}[{index}] "
                        "has no placement dimensions or ground anchor"
                    )
                normalized_anchor = [float(value) for value in ground_anchor]
                if any(not math.isfinite(value) for value in normalized_anchor):
                    raise ValueError(
                        f"SimReady environment {kind}.{family}[{index}] "
                        "has a non-finite ground anchor"
                    )
                normalized_dimensions = {
                    axis: float(dimensions[axis])
                    for axis in ("x", "y", "z")
                }
                intrinsic_uniform_scale = 1.0
                if (
                    kind == "buildings"
                    and min(normalized_dimensions.values()) >= 500.0
                    and max(
                        normalized_dimensions["x"],
                        normalized_dimensions["y"],
                    )
                    * 0.01
                    <= 100.0
                    and 2.0
                    <= normalized_dimensions["z"] * 0.01
                    <= 100.0
                ):
                    # Some official rural NVIDIA assets declare metre units
                    # while their geometry is authored in centimetres. Keep
                    # the model and normalize it at the prototype boundary.
                    intrinsic_uniform_scale = 0.01
                    normalized_dimensions = {
                        axis: value * intrinsic_uniform_scale
                        for axis, value in normalized_dimensions.items()
                    }
                resolved_entries.append(
                    {
                        "asset_id": str(item.get("asset_id", "")),
                        "family": str(item.get("family", "")),
                        "path": wrapper(
                            item, label=f"{kind}.{family}[{index}]"
                        ),
                        "lod_paths": lod_wrappers(
                            item, label=f"{kind}.{family}[{index}]"
                        ),
                        "native_dimensions_m": normalized_dimensions,
                        "intrinsic_uniform_scale": intrinsic_uniform_scale,
                        "ground_anchor_m": normalized_anchor,
                        "minimum_uniform_scale": float(
                            placement.get("minimum_uniform_scale", 0.8)
                        ),
                        "maximum_uniform_scale": float(
                            placement.get("maximum_uniform_scale", 1.25)
                        ),
                    }
                )
            resolved[kind][str(family)] = resolved_entries
    return resolved, {
        "manifest": manifest.relative_to(volume).as_posix(),
        "manifest_sha256": sha256(manifest),
        "validation": validation,
    }


def _build_lock_owner_is_live(owner: object) -> bool:
    if not isinstance(owner, dict):
        return False
    try:
        pid = int(owner.get("pid", 0))
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but belongs to another security context.  Do not
        # risk overlapping writes to a shared package in that situation.
        return True
    except OSError:
        return False
    return True


@contextlib.contextmanager
def _exclusive_zone_build(zone_root: Path) -> Iterable[None]:
    """Prevent two native builders from mutating one zone package at once.

    Build receipts hash every payload.  Concurrent local invocations used to
    overwrite the same ``build/payloads`` files while the other process was
    recording hashes, yielding a real checksum crash after an otherwise valid
    build.  The owner record also lets an interrupted build resume safely once
    its process no longer exists.
    """

    lock_path = zone_root / ".native-zone-build.lock"
    owner = {
        "pid": os.getpid(),
        "started_unix_ns": time.time_ns(),
    }
    owner_bytes = (json.dumps(owner, sort_keys=True) + "\n").encode("utf-8")
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(
                str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError:
            try:
                previous = json.loads(lock_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = None
            if _build_lock_owner_is_live(previous):
                pid = previous.get("pid") if isinstance(previous, dict) else "unknown"
                raise RuntimeError(
                    f"native zone build is already active for {zone_root.name} (pid={pid})"
                )
            # The creator has exited, so no file handle can still be writing
            # this package.  Removing only this zone-local stale marker makes
            # resumptions possible without touching generated source data.
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
        else:
            try:
                os.write(descriptor, owner_bytes)
                os.fsync(descriptor)
            except BaseException:
                os.close(descriptor)
                descriptor = None
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                raise
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            persisted = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            persisted = None
        if persisted == owner:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _absolute_directory(value: str, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{label} is absent or is not a directory: {path}")
    return path


def _relative_asset(target: Path, base: Path) -> str:
    return os.path.relpath(target, start=base).replace("\\", "/")


def _as_float(value: object, *, default: float) -> float:
    try:
        candidate = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return candidate if math.isfinite(candidate) else default


def _feature_polygons(geometry: object) -> Iterable[list[list[float]]]:
    if not isinstance(geometry, dict):
        return []
    coordinates = geometry.get("coordinates")
    geometry_type = geometry.get("type")
    if geometry_type == "Polygon" and isinstance(coordinates, list):
        candidates = [coordinates]
    elif geometry_type == "MultiPolygon" and isinstance(coordinates, list):
        candidates = coordinates
    else:
        return []
    result: list[list[list[float]]] = []
    for polygon in candidates:
        if not isinstance(polygon, list) or not polygon or not isinstance(polygon[0], list):
            continue
        ring = polygon[0]
        points = [
            [float(item[0]), float(item[1])]
            for item in ring
            if isinstance(item, list)
            and len(item) >= 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], (int, float))
        ]
        if len(points) >= 4:
            if points[0] == points[-1]:
                points.pop()
            if len(points) >= 3:
                result.append(points)
    return result


def _feature_lines(geometry: object) -> Iterable[list[list[float]]]:
    if not isinstance(geometry, dict):
        return []
    coordinates = geometry.get("coordinates")
    geometry_type = geometry.get("type")
    if geometry_type == "LineString" and isinstance(coordinates, list):
        candidates = [coordinates]
    elif geometry_type == "MultiLineString" and isinstance(coordinates, list):
        candidates = coordinates
    else:
        return []
    result: list[list[list[float]]] = []
    for line in candidates:
        if not isinstance(line, list):
            continue
        points = [
            [float(item[0]), float(item[1])]
            for item in line
            if isinstance(item, list)
            and len(item) >= 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], (int, float))
        ]
        if len(points) >= 2:
            result.append(points)
    return result


def _polygon_centroid(points: list[list[float]]) -> tuple[float, float]:
    twice_area = 0.0
    x_sum = 0.0
    y_sum = 0.0
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        cross = point[0] * following[1] - following[0] * point[1]
        twice_area += cross
        x_sum += (point[0] + following[0]) * cross
        y_sum += (point[1] + following[1]) * cross
    if abs(twice_area) > 1e-6:
        return x_sum / (3.0 * twice_area), y_sum / (3.0 * twice_area)
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _mesh_grid(
    *,
    values: np.ndarray,
    xmin: int,
    ymax: int,
    origin_x: int,
    origin_y: int,
    samples: int,
    gf: Any,
    width_metres: int = 1000,
    height_metres: int = 1000,
) -> tuple[list[Any], list[int], list[int]]:
    """Sample a Float32 MNT deterministically into a local-metres grid mesh."""

    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("MNT must be a two-dimensional raster")
    actual_samples = min(samples, values.shape[0], values.shape[1])
    if actual_samples < 2:
        raise ValueError("terrain grid needs at least two source samples")
    rows = np.linspace(0, values.shape[0] - 1, actual_samples, dtype=np.intp)
    columns = np.linspace(0, values.shape[1] - 1, actual_samples, dtype=np.intp)
    sampled = values[np.ix_(rows, columns)].astype(np.float32, copy=False)
    finite = sampled[np.isfinite(sampled)]
    if finite.size == 0:
        raise ValueError("MNT has no finite elevation sample")
    sampled = np.where(np.isfinite(sampled), sampled, float(np.median(finite)))
    x_scale = float(width_metres) / float(values.shape[1] - 1)
    y_scale = float(height_metres) / float(values.shape[0] - 1)
    points = [
        gf.Vec3f(
            float(xmin + int(column) * x_scale - origin_x),
            float(ymax - int(row) * y_scale - origin_y),
            float(sampled[row_index, column_index]),
        )
        for row_index, row in enumerate(rows)
        for column_index, column in enumerate(columns)
    ]
    counts = [4] * ((actual_samples - 1) * (actual_samples - 1))
    indices: list[int] = []
    for row in range(actual_samples - 1):
        for column in range(actual_samples - 1):
            base = row * actual_samples + column
            # Raster row zero is the northern edge.  The local +Y direction is
            # north, hence the face must run north-west -> south-west ->
            # south-east -> north-east to have an upward (+Z) normal.  The old
            # winding was reversed, which made terrain illumination unreliable
            # in RTX review.
            indices.extend(
                (base, base + actual_samples, base + actual_samples + 1, base + 1)
            )
    return points, counts, indices


def _terrain_vertex_normals(
    *, points: list[Any], samples: int, gf: Any
) -> list[Any]:
    if samples < 2 or len(points) != samples * samples:
        raise ValueError("terrain normal grid dimensions are inconsistent")
    grid = np.asarray(
        [tuple(float(point[index]) for index in range(3)) for point in points],
        dtype=np.float64,
    ).reshape(samples, samples, 3)
    tangent_x = np.empty_like(grid)
    tangent_x[:, 1:-1] = grid[:, 2:] - grid[:, :-2]
    tangent_x[:, 0] = grid[:, 1] - grid[:, 0]
    tangent_x[:, -1] = grid[:, -1] - grid[:, -2]
    tangent_y = np.empty_like(grid)
    tangent_y[1:-1] = grid[:-2] - grid[2:]
    tangent_y[0] = grid[0] - grid[1]
    tangent_y[-1] = grid[-2] - grid[-1]
    normal_grid = np.cross(tangent_x, tangent_y)
    lengths = np.linalg.norm(normal_grid, axis=2)
    valid = np.isfinite(lengths) & (lengths > 1e-9)
    normal_grid[valid] /= lengths[valid, np.newaxis]
    normal_grid[~valid] = (0.0, 0.0, 1.0)
    normal_grid[normal_grid[:, :, 2] < 0.0] *= -1.0
    return [
        gf.Vec3f(float(normal[0]), float(normal[1]), float(normal[2]))
        for normal in normal_grid.reshape(-1, 3)
    ]


def _write_terrain_pbr_maps(
    *, values: np.ndarray, output_root: Path, tile_ref: str
) -> tuple[Path, Path]:
    """Create deterministic metric normal and roughness maps from the MNT."""

    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("terrain PBR maps require a two-dimensional MNT")
    finite = np.isfinite(values)
    if not bool(finite.all()):
        raise ValueError("terrain PBR maps refuse non-finite elevation")
    output_root.mkdir(parents=True, exist_ok=True)
    pixel_x = 1000.0 / float(values.shape[1] - 1)
    pixel_y = 1000.0 / float(values.shape[0] - 1)
    dz_drow, dz_dcolumn = np.gradient(
        values.astype(np.float64, copy=False), pixel_y, pixel_x
    )
    # Raster rows increase southward, so +Y north is the negative row gradient.
    dz_dy = -dz_drow
    nx = -dz_dcolumn
    ny = -dz_dy
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = np.stack((nx / norm, ny / norm, nz / norm), axis=-1)
    encoded_normal = np.clip(
        np.rint((normal * 0.5 + 0.5) * 255.0), 0, 255
    ).astype(np.uint8)
    slope = np.sqrt(dz_dcolumn * dz_dcolumn + dz_dy * dz_dy)
    roughness = np.clip(0.76 + np.minimum(slope, 1.0) * 0.18, 0.0, 1.0)
    encoded_roughness = np.rint(roughness * 255.0).astype(np.uint8)
    normal_path = output_root / f"{tile_ref}_normal.png"
    roughness_path = output_root / f"{tile_ref}_roughness.png"
    Image.fromarray(encoded_normal, mode="RGB").save(
        normal_path, optimize=True
    )
    Image.fromarray(encoded_roughness, mode="L").save(
        roughness_path, optimize=True
    )
    return normal_path, roughness_path


class _ElevationGrid:
    """Two-metre raster samples retained for source-derived object placement."""

    def __init__(self, *, samples: int = 513, label: str = "raster") -> None:
        if samples < 2:
            raise ValueError("elevation grid needs at least two samples per tile")
        self._samples = samples
        self._label = label
        self._tiles: dict[tuple[int, int, int, int], np.ndarray] = {}

    def add(self, *, xmin: int, ymin: int, xmax: int, ymax: int, values: np.ndarray) -> None:
        sampled = values[
            np.ix_(
                np.linspace(0, values.shape[0] - 1, self._samples, dtype=np.intp),
                np.linspace(0, values.shape[1] - 1, self._samples, dtype=np.intp),
            )
        ].astype(np.float32, copy=True)
        finite = sampled[np.isfinite(sampled)]
        if finite.size == 0:
            raise ValueError(f"{self._label} tile has no finite sample")
        missing_ratio = 1.0 - float(finite.size) / float(sampled.size)
        if missing_ratio > MAX_RASTER_NODATA_RATIO:
            raise ValueError(
                f"{self._label} tile has {missing_ratio:.2%} missing samples; "
                f"{MAX_RASTER_NODATA_RATIO:.2%} maximum"
            )
        sampled[~np.isfinite(sampled)] = float(np.median(finite))
        self._tiles[(xmin, ymin, xmax, ymax)] = sampled

    def elevation(self, x: float, y: float, *, fallback: float) -> float:
        for (xmin, ymin, xmax, ymax), values in self._tiles.items():
            if xmin <= x <= xmax and ymin <= y <= ymax:
                u = min(1.0, max(0.0, (x - xmin) / float(xmax - xmin)))
                # Image row zero is the northern edge.
                v = min(1.0, max(0.0, (ymax - y) / float(ymax - ymin)))
                row_position = v * (values.shape[0] - 1)
                column_position = u * (values.shape[1] - 1)
                row0 = int(math.floor(row_position))
                column0 = int(math.floor(column_position))
                row1 = min(values.shape[0] - 1, row0 + 1)
                column1 = min(values.shape[1] - 1, column0 + 1)
                row_weight = row_position - row0
                column_weight = column_position - column0
                north_west = float(values[row0, column0])
                north_east = float(values[row0, column1])
                south_west = float(values[row1, column0])
                south_east = float(values[row1, column1])
                north = (
                    north_west * (1.0 - column_weight)
                    + north_east * column_weight
                )
                south = (
                    south_west * (1.0 - column_weight)
                    + south_east * column_weight
                )
                return north * (1.0 - row_weight) + south * row_weight
        return fallback


def _entry_path(zone_root: Path, entry: dict[str, Any]) -> Path:
    download = entry.get("download")
    if (
        not isinstance(download, dict)
        or download.get("state") not in LOCKED_SOURCE_FINAL_STATES
    ):
        raise RuntimeError(f"source has not been downloaded: {entry.get('id')}")
    relpath = str(download.get("relpath", ""))
    dataset = str(entry.get("dataset", ""))
    path = (zone_root / "raw" / dataset / relpath).resolve()
    if not _is_below(zone_root / "raw", path) or not path.is_file():
        raise RuntimeError(f"locked source is absent: {entry.get('id')}")
    identity = download.get("file_identity")
    stat = path.stat()
    identity_matches = (
        isinstance(identity, dict)
        and all(
            int(identity.get(key, -1)) == actual
            for key, actual in (
                ("bytes", stat.st_size),
                ("mtime_ns", stat.st_mtime_ns),
                ("device", stat.st_dev),
                ("inode", stat.st_ino),
            )
        )
    )
    skip_content_validation = os.getenv(
        "FW_SDG_SKIP_SOURCE_CONTENT_VALIDATION", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not (skip_content_validation and identity_matches) and sha256(path) != str(
        download.get("sha256", "")
    ):
        raise RuntimeError(f"locked source checksum changed: {entry.get('id')}")
    return path


def _source_index(zone_root: Path, lock: dict[str, Any]) -> dict[tuple[str, str], Path]:
    entries = lock.get("entries")
    if not isinstance(entries, list):
        raise ValueError("source lock entries are malformed")
    result: dict[tuple[str, str], Path] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("source lock entry is malformed")
        dataset = str(item.get("dataset", ""))
        tile_ref = str(item.get("tile_ref", ""))
        if dataset in BASELINE_DATASETS or dataset == "lidar":
            result[(tile_ref, dataset)] = _entry_path(zone_root, item)
        elif dataset == "ortho20":
            download = item.get("download")
            if (
                isinstance(download, dict)
                and download.get("state") in LOCKED_SOURCE_FINAL_STATES
            ):
                result[(tile_ref, dataset)] = _entry_path(zone_root, item)
    return result


def _read_raster_values(path: Path, *, label: str) -> np.ndarray:
    """Read one locked raster and convert declared/obvious NoData to NaN.

    A broad median fill can turn a missing LiDAR tile into a convincing but
    false plateau.  The downstream grid/mesh checks therefore tolerate only a
    bounded number of missing samples and reject material holes.
    """

    image = Image.open(path)
    try:
        values = np.asarray(image, dtype=np.float32)
        nodata: float | None = None
        tags = getattr(image, "tag_v2", None)
        if tags is not None:
            raw_nodata = tags.get(42113)
            if isinstance(raw_nodata, (tuple, list)) and raw_nodata:
                raw_nodata = raw_nodata[0]
            if raw_nodata not in (None, ""):
                try:
                    nodata = float(raw_nodata)
                except (TypeError, ValueError):
                    nodata = None
    finally:
        image.close()
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError(f"{label} must be a two-dimensional raster")
    values = values.astype(np.float32, copy=True)
    if nodata is not None and math.isfinite(nodata):
        values[np.isclose(values, nodata, rtol=0.0, atol=1e-5)] = np.nan
    # These finite sentinels are common in elevation products and must never
    # become real terrain/canopy heights when the metadata tag was stripped.
    values[(values <= -9990.0) | (values >= 1.0e20)] = np.nan
    finite = np.isfinite(values)
    if not bool(finite.any()):
        raise ValueError(f"{label} has no finite sample")
    missing_ratio = 1.0 - float(np.count_nonzero(finite)) / float(values.size)
    if missing_ratio > MAX_RASTER_NODATA_RATIO:
        raise ValueError(
            f"{label} has {missing_ratio:.2%} NoData; "
            f"{MAX_RASTER_NODATA_RATIO:.2%} maximum"
        )
    return values


def _vector_paths(zone_root: Path, lock: dict[str, Any]) -> dict[str, list[Path]]:
    sources = lock.get("vector_sources")
    if not isinstance(sources, dict):
        raise RuntimeError("BDTOPO vector sources were not acquired and locked")
    result: dict[str, list[Path]] = {}
    for name in VECTOR_SOURCE_LAYERS:
        records = sources.get(name)
        if not isinstance(records, list) or len(records) != len(VECTOR_SOURCE_LAYERS[name]):
            raise RuntimeError(f"locked vector coverage is incomplete for {name}")
        paths: list[Path] = []
        for record in records:
            if not isinstance(record, dict) or record.get("license") != "Licence Ouverte / Etalab 2.0":
                raise RuntimeError(f"locked vector provenance is incomplete for {name}")
            download = record.get("download")
            if not isinstance(download, dict) or download.get("state") != "downloaded":
                raise RuntimeError(f"locked vector source is not downloaded for {name}")
            path = (zone_root / "raw" / str(download.get("relpath", ""))).resolve()
            if not _is_below(zone_root / "raw", path) or not path.is_file():
                raise RuntimeError(f"locked vector source is absent for {name}")
            if sha256(path) != str(download.get("sha256", "")):
                raise RuntimeError(f"locked vector checksum changed for {name}")
            paths.append(path)
        result[name] = paths
    return result


def _geojson_features(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        collection = _read_json(path, label="locked vector GeoJSON")
        features = collection.get("features")
        if not isinstance(features, list):
            raise ValueError(f"locked vector GeoJSON has no features: {path}")
        for feature in features:
            if isinstance(feature, dict):
                yield feature


def _create_mesh(*, stage: Any, path: str, points: list[Any], counts: list[int], indices: list[int], usd_geom: Any, semantic: str, purpose: str) -> Any:
    mesh = usd_geom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(usd_geom.Tokens.none)
    mesh.CreatePurposeAttr(purpose)
    _apply_training_semantics(mesh.GetPrim(), semantic)
    return mesh


def _set_color(*, prim: Any, color: tuple[float, float, float], usd_geom: Any) -> None:
    attribute = prim.CreateDisplayColorAttr()
    attribute.Set([color])
    prim.GetPrim().SetCustomDataByKey(
        "fireviewer:procedural_color", ",".join(str(value) for value in color)
    )


def _bind_preview_material(*, prim: Any, material: Any, usd_shade: Any) -> None:
    usd_shade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)


def _textured_preview_material(
    *,
    stage: Any,
    path: str,
    texture_path: Path,
    base_path: Path,
    usd_shade: Any,
    sdf: Any,
    normal_texture_path: Path | None = None,
    roughness_texture_path: Path | None = None,
) -> Any:
    """Create a PreviewSurface material with an explicit `st` primvar reader.

    Implicit UVs are not portable between the native Isaac build and Composer.
    Every textured terrain/roof material therefore declares the same reader;
    its meshes author a matching vertex `st` primvar.
    """

    material = usd_shade.Material.Define(stage, path)
    preview = usd_shade.Shader.Define(stage, f"{path}/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    # Orthophotography is an albedo reference, not a shiny painted surface.
    # A high roughness keeps the terrain legible under the review sun instead
    # of washing it out with a broad specular highlight.
    roughness_input = preview.CreateInput(
        "roughness", sdf.ValueTypeNames.Float
    )
    roughness_input.Set(0.92)
    preview.CreateInput("metallic", sdf.ValueTypeNames.Float).Set(0.0)
    reader = usd_shade.Shader.Define(stage, f"{path}/ST")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", sdf.ValueTypeNames.Float2)
    texture = usd_shade.Shader.Define(stage, f"{path}/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", sdf.ValueTypeNames.Asset).Set(
        sdf.AssetPath(_relative_asset(texture_path, base_path))
    )
    texture.CreateInput("sourceColorSpace", sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("st", sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.GetOutput("result")
    )
    texture.CreateOutput("rgb", sdf.ValueTypeNames.Float3)
    preview.CreateInput("diffuseColor", sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.GetOutput("rgb")
    )
    if normal_texture_path is not None:
        normal = usd_shade.Shader.Define(stage, f"{path}/NormalTexture")
        normal.CreateIdAttr("UsdUVTexture")
        normal.CreateInput("file", sdf.ValueTypeNames.Asset).Set(
            sdf.AssetPath(
                _relative_asset(normal_texture_path, base_path)
            )
        )
        normal.CreateInput("sourceColorSpace", sdf.ValueTypeNames.Token).Set(
            "raw"
        )
        normal.CreateInput("scale", sdf.ValueTypeNames.Float4).Set(
            (2.0, 2.0, 2.0, 1.0)
        )
        normal.CreateInput("bias", sdf.ValueTypeNames.Float4).Set(
            (-1.0, -1.0, -1.0, 0.0)
        )
        normal.CreateInput("st", sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.GetOutput("result")
        )
        normal.CreateOutput("rgb", sdf.ValueTypeNames.Float3)
        preview.CreateInput(
            "normal", sdf.ValueTypeNames.Normal3f
        ).ConnectToSource(normal.GetOutput("rgb"))
    if roughness_texture_path is not None:
        roughness = usd_shade.Shader.Define(
            stage, f"{path}/RoughnessTexture"
        )
        roughness.CreateIdAttr("UsdUVTexture")
        roughness.CreateInput("file", sdf.ValueTypeNames.Asset).Set(
            sdf.AssetPath(
                _relative_asset(roughness_texture_path, base_path)
            )
        )
        roughness.CreateInput(
            "sourceColorSpace", sdf.ValueTypeNames.Token
        ).Set("raw")
        roughness.CreateInput("st", sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.GetOutput("result")
        )
        roughness.CreateOutput("r", sdf.ValueTypeNames.Float)
        roughness_input.ConnectToSource(roughness.GetOutput("r"))
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    material.GetPrim().SetCustomDataByKey(
        "fireviewer:pbr_channels",
        "baseColor,normal,roughness"
        if normal_texture_path is not None
        and roughness_texture_path is not None
        else "baseColor",
    )
    return material


def _colored_preview_material(
    *, stage: Any, path: str, color: tuple[float, float, float], usd_shade: Any, sdf: Any
) -> Any:
    material = usd_shade.Material.Define(stage, path)
    preview = usd_shade.Shader.Define(stage, f"{path}/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", sdf.ValueTypeNames.Color3f).Set(color)
    preview.CreateInput("roughness", sdf.ValueTypeNames.Float).Set(0.88)
    preview.CreateInput("metallic", sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    return material


def _set_planar_uv(*, mesh: Any, points: list[Any], gf: Any, usd_geom: Any, sdf: Any) -> None:
    """Author stable vertex UVs over a mesh's local XY extent."""

    if not points:
        return
    xy = np.asarray(
        [(float(point[0]), float(point[1])) for point in points],
        dtype=np.float64,
    )
    min_x, min_y = np.min(xy, axis=0)
    max_x, max_y = np.max(xy, axis=0)
    span_x = max(0.01, max_x - min_x)
    span_y = max(0.01, max_y - min_y)
    uv = np.empty_like(xy)
    uv[:, 0] = (xy[:, 0] - min_x) / span_x
    uv[:, 1] = (xy[:, 1] - min_y) / span_y
    values = [
        gf.Vec2f(float(value[0]), float(value[1]))
        for value in uv
    ]
    primvars = usd_geom.PrimvarsAPI(mesh)
    primvar = primvars.CreatePrimvar(
        "st", sdf.ValueTypeNames.TexCoord2fArray, usd_geom.Tokens.vertex
    )
    primvar.Set(values)


def _payload_paths(build_root: Path, tile_ref: str) -> Path:
    return build_root / "payloads" / f"{tile_ref}.usdc"


def _detail_payload_path(
    build_root: Path, tile_ref: str, detail_level: str = "HERO"
) -> Path:
    level = detail_level.upper()
    if level not in {"HERO", "MID", "FAR"}:
        raise ValueError(f"unsupported detail level: {detail_level}")
    if level == "HERO":
        return build_root / "details" / "hero" / f"{tile_ref}_details.usdc"
    return (
        build_root
        / "details"
        / level.lower()
        / f"{tile_ref}_{level.lower()}_details.usdc"
    )


def _detail_lod_candidates(
    candidates: Iterable[Any], detail_level: str
) -> list[Any]:
    """Reduce the same observed canopy deterministically for MID/FAR.

    The coarser levels retain real source-derived tree locations and real USD
    tree assets.  They never replace vegetation with cubes, cones or an empty
    tile.  A non-empty HERO set therefore always retains at least one
    representative in MID and FAR.
    """

    values = list(candidates)
    level = detail_level.upper()
    strides = {"HERO": 1, "MID": 4, "FAR": 16}
    if level not in strides:
        raise ValueError(f"unsupported detail level: {detail_level}")
    if not values:
        return []
    selected = values[:: strides[level]]
    return selected or [values[0]]


def _flow_preset_source() -> Path:
    configured = os.getenv("FW_SDG_FLOW_PRESET", "").strip()
    candidates = [Path(configured)] if configured else []
    runtime_root = Path(os.getenv("FW_SDG_RUNTIME_ROOT", sys.prefix))
    for root in (
        runtime_root / "lib" / "python3.12" / "site-packages",
        runtime_root / "Lib" / "site-packages",
        Path(sys.prefix),
    ):
        if root.is_dir():
            candidates.extend(
                root.glob("**/omni.flowusd-*/data/presets/Fire/Fire.usda")
            )
    candidates.extend(
        Path("C:/isaacsim/extscache").glob(
            "omni.flowusd-*/data/presets/Fire/Fire.usda"
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError("the installed native Isaac runtime has no Flow Fire preset")


def _lock_flow_preset(*, build_root: Path) -> dict[str, Any]:
    """Copy the installed Flow preset without falsifying its coordinate metadata."""

    source = _flow_preset_source()
    source_text = source.read_text(encoding="utf-8")
    if "metersPerUnit = 0.01" not in source_text or 'upAxis = "Y"' not in source_text:
        raise RuntimeError("installed Flow Fire preset has an unsupported coordinate contract")
    asset_root = build_root / "assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    target = asset_root / "flow-fire-source-yup-centimetres.usda"
    # The preset is local to the installed Isaac runtime, but the package must
    # remain reproducible after that runtime moves.  Stage metadata alone does
    # not rescale or rotate composed geometry, so rewriting centimetres/Y-up as
    # metres/Z-up would make the fire roughly 100x too large.  Preserve the
    # source layer byte-for-byte and author the real conversion on its parent
    # Xform when the post-review simulation layer composes it.
    shutil.copy2(source, target)
    source_version = next(
        (parent.name for parent in source.parents if parent.name.startswith("omni.flowusd-")),
        "unknown",
    )
    return {
        "id": "nvidia-flow-fire-preset",
        "source_path": str(source),
        "source_sha256": sha256(source),
        "source_version": source_version,
        "license": "NVIDIA Isaac Sim installed-runtime terms",
        "attribution": "NVIDIA Flow Fire preset bundled with native Isaac Sim",
        "packaged_path": target.relative_to(build_root).as_posix(),
        "packaged_sha256": sha256(target),
        "coordinate_normalization": {
            "source_meters_per_unit": 0.01,
            "source_up_axis": "Y",
            "parent_uniform_scale": 0.01,
            "parent_rotation_xyz_degrees": [90, 0, 0],
            "result_meters_per_unit": 1,
            "result_up_axis": "Z",
        },
    }


def _write_payload(
    *,
    payload_path: Path,
    tile: dict[str, str],
    values: np.ndarray,
    ortho_path: Path | None,
    ortho_lod0_path: Path | None,
    origin_x: int,
    origin_y: int,
    usd: Any,
    usd_geom: Any,
    usd_shade: Any,
    sdf: Any,
    gf: Any,
    elevation_grid: _ElevationGrid,
    render_visible: bool = True,
) -> dict[str, int]:
    fast_preview = os.getenv(
        "FW_SDG_FAST_EDITOR_PREVIEW", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = (int(tile[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
    elevation_grid.add(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, values=values)
    stage = usd.Stage.CreateNew(str(payload_path))
    usd_geom.SetStageMetersPerUnit(stage, 1.0)
    usd_geom.SetStageUpAxis(stage, usd_geom.Tokens.z)
    tile_prim = usd_geom.Xform.Define(stage, "/Tile")
    stage.SetDefaultPrim(tile_prim.GetPrim())
    tile_prim.GetPrim().SetCustomDataByKey("fireviewer:tile_ref", tile["tile_ref"])
    tile_prim.GetPrim().SetCustomDataByKey(
        "fireviewer:epsg2154_bounds", f"{xmin},{ymin},{xmax},{ymax}"
    )

    materials: dict[str, Any] = {}
    normal_texture_path: Path | None = None
    roughness_texture_path: Path | None = None
    skip_terrain_pbr_generation = os.getenv(
        "FW_SDG_SKIP_TERRAIN_PBR_GENERATION", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if (
        render_visible
        and ortho_path is not None
        and not skip_terrain_pbr_generation
    ):
        normal_texture_path, roughness_texture_path = _write_terrain_pbr_maps(
            values=values,
            output_root=payload_path.parent.parent
            / "textures"
            / "terrain-pbr",
            tile_ref=tile["tile_ref"],
        )
    if not render_visible:
        tile_prim.GetPrim().SetCustomDataByKey(
            "fireviewer:role", "non_render_collision_payload"
        )
    elif ortho_path is not None:
        materials["context"] = _textured_preview_material(
            stage=stage,
            path="/Tile/Materials/TerrainContext",
            texture_path=ortho_path,
            base_path=payload_path.parent,
            usd_shade=usd_shade,
            sdf=sdf,
            normal_texture_path=normal_texture_path,
            roughness_texture_path=roughness_texture_path,
        )
        if ortho_lod0_path is not None:
            materials["near"] = _textured_preview_material(
                stage=stage,
                path="/Tile/Materials/TerrainNear",
                texture_path=ortho_lod0_path,
                base_path=payload_path.parent,
                usd_shade=usd_shade,
                sdf=sdf,
                normal_texture_path=normal_texture_path,
                roughness_texture_path=roughness_texture_path,
            )
    else:
        # The lightweight profile intentionally does not download a 20 km
        # orthophoto.  Retain a deterministic, zone-local procedural palette
        # rather than pretending a texture source exists.
        variation = ((xmin // 1000) * 17 + (ymin // 1000) * 31) % 5
        colors = (
            (0.29, 0.36, 0.20),
            (0.34, 0.39, 0.23),
            (0.39, 0.36, 0.22),
            (0.31, 0.33, 0.18),
            (0.42, 0.39, 0.27),
        )
        material = usd_shade.Material.Define(stage, "/Tile/Materials/TerrainContext")
        preview = usd_shade.Shader.Define(
            stage, "/Tile/Materials/TerrainContext/Preview"
        )
        preview.CreateIdAttr("UsdPreviewSurface")
        preview.CreateInput("roughness", sdf.ValueTypeNames.Float).Set(0.85)
        preview.CreateInput("metallic", sdf.ValueTypeNames.Float).Set(0.0)
        preview.CreateInput("diffuseColor", sdf.ValueTypeNames.Color3f).Set(
            gf.Vec3f(*colors[variation])
        )
        material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
        material.GetPrim().SetCustomDataByKey(
            "fireviewer:imagery", "procedural_zone_palette"
        )
        materials["context"] = material
    terrain_counts: dict[str, int] = {}
    collision_lod = tile_prim.GetPrim().GetVariantSets().AddVariantSet(
        "collisionLOD"
    )
    collision_variants = (
        (("NEAR", TERRAIN_LOD1_SAMPLES),)
        if fast_preview
        else (
            ("NEAR", TERRAIN_LOD1_SAMPLES),
            ("FAR", COLLISION_SAMPLES),
        )
    )
    for collision_label, collision_samples in collision_variants:
        collision_lod.AddVariant(collision_label)
        collision_lod.SetVariantSelection(collision_label)
        with collision_lod.GetVariantEditContext():
            collision_points, collision_counts, collision_indices = _mesh_grid(
                values=values,
                xmin=xmin,
                ymax=ymax,
                origin_x=origin_x,
                origin_y=origin_y,
                samples=collision_samples,
                gf=gf,
            )
            collision = _create_mesh(
                stage=stage,
                path="/Tile/Collision",
                points=collision_points,
                counts=collision_counts,
                indices=collision_indices,
                usd_geom=usd_geom,
                semantic="terrain_collision",
                purpose=usd_geom.Tokens.guide,
            )
            collision.GetPrim().SetCustomDataByKey(
                "fireviewer:collision_lod", collision_label
            )
            terrain_counts[f"Collision{collision_label}"] = len(
                collision_points
            )
    selected_collision = "NEAR" if fast_preview else "FAR"
    collision_lod.SetVariantSelection(selected_collision)
    terrain_counts["Collision"] = terrain_counts[f"Collision{selected_collision}"]
    if not render_visible:
        stage.GetRootLayer().Save()
        return terrain_counts

    # All display levels are authored as variants of the same source-backed
    # terrain mesh.  The Editor can therefore switch quality without composing
    # overlapping meshes, z-fighting, or replacing the place with a proxy box.
    variants: list[tuple[str, int, Any, Path | None, str]] = []
    if ortho_lod0_path is not None and not fast_preview:
        variants.append(
            ("LOD0", TERRAIN_LOD0_SAMPLES, usd_geom.Tokens.render, ortho_lod0_path, "near")
        )
    preview_samples = (
        TERRAIN_LOD1_SAMPLES,
        TERRAIN_LOD2_SAMPLES,
        TERRAIN_LOD3_SAMPLES,
    )
    variants.extend(
        (
            ("LOD1", preview_samples[0], usd_geom.Tokens.render, ortho_path, "context"),
        )
        if fast_preview
        else (
            ("LOD1", preview_samples[0], usd_geom.Tokens.render, ortho_path, "context"),
            ("LOD2", preview_samples[1], usd_geom.Tokens.render, ortho_path, "context"),
            ("LOD3", preview_samples[2], usd_geom.Tokens.proxy, ortho_path, "context"),
        )
    )
    lod_set = tile_prim.GetPrim().GetVariantSets().AddVariantSet("terrainLOD")
    for label, samples, purpose, imagery_path, material_key in variants:
        lod_set.AddVariant(label)
        lod_set.SetVariantSelection(label)
        with lod_set.GetVariantEditContext():
            points, counts, indices = _mesh_grid(
                values=values,
                xmin=xmin,
                ymax=ymax,
                origin_x=origin_x,
                origin_y=origin_y,
                samples=samples,
                gf=gf,
            )
            mesh = _create_mesh(
                stage=stage,
                path="/Tile/Terrain",
                points=points,
                counts=counts,
                indices=indices,
                usd_geom=usd_geom,
                semantic="terrain",
                purpose=purpose,
            )
            actual_samples = int(round(math.sqrt(len(points))))
            mesh.CreateNormalsAttr(
                _terrain_vertex_normals(
                    points=points,
                    samples=actual_samples,
                    gf=gf,
                )
            )
            mesh.SetNormalsInterpolation(usd_geom.Tokens.vertex)
            mesh.GetPrim().SetCustomDataByKey("fireviewer:terrain_lod", label)
            material = materials.get(material_key)
            if material is not None:
                _bind_preview_material(
                    prim=mesh, material=material, usd_shade=usd_shade
                )
            _set_planar_uv(
                mesh=mesh,
                points=points,
                gf=gf,
                usd_geom=usd_geom,
                sdf=sdf,
            )
            if imagery_path is not None:
                resolution_label = "ortho20" if label == "LOD0" else "ortho50"
                mesh.GetPrim().CreateAttribute(
                    f"fireviewer:{resolution_label}",
                    sdf.ValueTypeNames.Asset,
                ).Set(
                    sdf.AssetPath(
                        _relative_asset(imagery_path, payload_path.parent)
                    )
                )
            else:
                mesh.GetPrim().SetCustomDataByKey(
                    "fireviewer:imagery", "procedural_zone_palette"
                )
        terrain_counts[f"Terrain{label}"] = len(points)
    default_lod = "LOD1"
    lod_set.SetVariantSelection(default_lod)
    tile_prim.GetPrim().SetCustomDataByKey(
        "fireviewer:terrain_lods", ",".join(item[0] for item in variants)
    )
    tile_prim.GetPrim().SetCustomDataByKey(
        "fireviewer:default_terrain_lod", default_lod
    )
    stage.GetRootLayer().Save()
    return terrain_counts


def _read_mnt_values(path: Path) -> np.ndarray:
    return _read_raster_values(path, label="MNT")


def _validate_height_products(
    *,
    mnt: np.ndarray,
    mns: np.ndarray,
    mnh: np.ndarray,
    label: str,
    maximum_p95_residual_metres: float = 2.0,
) -> dict[str, float]:
    """Check that MNH describes the same surface-minus-ground signal."""

    if any(values.ndim != 2 for values in (mnt, mns, mnh)):
        raise ValueError(f"{label} height products must be two-dimensional")
    samples = min(65, *(min(values.shape) for values in (mnt, mns, mnh)))
    normalized = []
    for values in (mnt, mns, mnh):
        rows = np.linspace(0, values.shape[0] - 1, samples, dtype=np.intp)
        columns = np.linspace(0, values.shape[1] - 1, samples, dtype=np.intp)
        normalized.append(values[np.ix_(rows, columns)].astype(np.float32, copy=False))
    residual = (normalized[1] - normalized[0]) - normalized[2]
    finite = np.abs(residual[np.isfinite(residual)])
    if finite.size == 0:
        raise ValueError(f"{label} has no comparable MNT/MNS/MNH samples")
    p95 = float(np.percentile(finite, 95))
    median = float(np.median(finite))
    if p95 > maximum_p95_residual_metres:
        raise ValueError(
            f"{label} MNH disagrees with MNS-MNT: p95 residual {p95:.2f} m"
        )
    return {
        "median_absolute_residual_metres": median,
        "p95_absolute_residual_metres": p95,
    }


def _mosaic_tile_values(
    *, values: np.ndarray, tile: dict[str, str], zone: dict[str, Any]
) -> np.ndarray:
    """Extract a deterministic 129×129 one-kilometre grid from a zone MNT."""

    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("light terrain MNT must be a two-dimensional raster")
    xmin, ymin, xmax, ymax = (int(tile[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
    zone_xmin, zone_ymin, zone_xmax, zone_ymax = (
        int(zone[key]) for key in ("xmin", "ymin", "xmax", "ymax")
    )
    if not (
        zone_xmin <= xmin < xmax <= zone_xmax
        and zone_ymin <= ymin < ymax <= zone_ymax
    ):
        raise ValueError("tile is outside the light terrain MNT coverage")
    columns = np.linspace(
        round((xmin - zone_xmin) / (zone_xmax - zone_xmin) * (values.shape[1] - 1)),
        round((xmax - zone_xmin) / (zone_xmax - zone_xmin) * (values.shape[1] - 1)),
        129,
        dtype=np.intp,
    )
    rows = np.linspace(
        round((zone_ymax - ymax) / (zone_ymax - zone_ymin) * (values.shape[0] - 1)),
        round((zone_ymax - ymin) / (zone_ymax - zone_ymin) * (values.shape[0] - 1)),
        129,
        dtype=np.intp,
    )
    return values[np.ix_(rows, columns)].astype(np.float32, copy=True)


def _light_orthophoto_mosaic(
    *, zone_root: Path, lock: dict[str, Any], zone: dict[str, Any]
) -> np.ndarray:
    """Compose the four locked 4 m WMS images into a north-up RGB mosaic."""

    entries = lock.get("entries")
    if not isinstance(entries, list):
        raise ValueError("light source lock entries are malformed")
    xmin, ymin, xmax, ymax = (
        int(zone[key]) for key in ("xmin", "ymin", "xmax", "ymax")
    )
    resolution = int(
        lock.get("light_profile", {})
        .get("imagery", {})
        .get("resolution_metres", 0)
    )
    if resolution <= 0 or (xmax - xmin) % resolution or (ymax - ymin) % resolution:
        raise RuntimeError("light orthophoto metadata has an invalid resolution")
    mosaic = np.zeros(
        ((ymax - ymin) // resolution, (xmax - xmin) // resolution, 3),
        dtype=np.uint8,
    )
    covered = np.zeros(mosaic.shape[:2], dtype=bool)
    images = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("dataset") == "ortho_lod2"
    ]
    if len(images) != 4:
        raise RuntimeError("light source lock must contain four orthophoto context images")
    for entry in images:
        bounds = entry.get("bbox_epsg2154")
        if not isinstance(bounds, list) or len(bounds) != 4:
            raise RuntimeError("orthophoto source has no EPSG:2154 bounds")
        left, bottom, right, top = (int(value) for value in bounds)
        if not (xmin <= left < right <= xmax and ymin <= bottom < top <= ymax):
            raise RuntimeError("orthophoto source bounds escape the zone")
        image = Image.open(_entry_path(zone_root, entry))
        try:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        finally:
            image.close()
        width = (right - left) // resolution
        height = (top - bottom) // resolution
        if rgb.shape[:2] != (height, width):
            rgb = np.asarray(
                Image.fromarray(rgb, mode="RGB").resize(
                    (width, height), Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            )
        row = (ymax - top) // resolution
        column = (left - xmin) // resolution
        mosaic[row : row + height, column : column + width] = rgb
        covered[row : row + height, column : column + width] = True
    if not bool(covered.all()):
        raise RuntimeError("light orthophoto mosaic does not cover the full zone")
    return mosaic


def _write_light_orthophoto_tile(
    *, mosaic: np.ndarray, tile: dict[str, str], zone: dict[str, Any], path: Path
) -> Path:
    """Write one portable context-texture crop for a one-kilometre payload."""

    xmin, ymin, xmax, ymax = (
        int(zone[key]) for key in ("xmin", "ymin", "xmax", "ymax")
    )
    tile_xmin, _tile_ymin, tile_xmax, tile_ymax = (
        int(tile[key]) for key in ("xmin", "ymin", "xmax", "ymax")
    )
    resolution = (xmax - xmin) // mosaic.shape[1]
    if resolution <= 0 or (tile_xmax - tile_xmin) % resolution:
        raise RuntimeError("light orthophoto mosaic is incompatible with one-kilometre tiles")
    size = (tile_xmax - tile_xmin) // resolution
    row = (ymax - tile_ymax) // resolution
    column = (tile_xmin - xmin) // resolution
    pixels = mosaic[row : row + size, column : column + size]
    if pixels.shape != (size, size, 3):
        raise RuntimeError("light orthophoto crop escapes the zone mosaic")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path, quality=90, optimize=True)
    return path


def _write_continuous_orthophoto(*, mosaic: np.ndarray, path: Path) -> Path:
    """Package one color map for the visible 20 × 20 km terrain surface.

    The one-kilometre source tiles remain available as data payloads, but they
    must not define the visible color layout: that was the source of the grid
    of small visual squares in Composer.
    """

    if mosaic.ndim != 3 or mosaic.shape[2] != 3:
        raise ValueError("continuous orthophoto needs an RGB mosaic")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(mosaic, mode="RGB")
    try:
        image = image.resize(
            (CONTINUOUS_ORTHO_PIXELS, CONTINUOUS_ORTHO_PIXELS),
            Image.Resampling.LANCZOS,
        )
        image.save(path, quality=94, optimize=True, subsampling=0)
    finally:
        image.close()
    return path


def _write_continuous_terrain(
    *,
    terrain_path: Path,
    values: np.ndarray,
    ortho_path: Path,
    zone: dict[str, Any],
    origin_x: int,
    origin_y: int,
    usd: Any,
    usd_geom: Any,
    usd_shade: Any,
    sdf: Any,
    gf: Any,
) -> int:
    """Author the sole visible terrain surface for the whole geographic zone."""

    terrain_path.parent.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = (
        int(zone[key]) for key in ("xmin", "ymin", "xmax", "ymax")
    )
    stage = usd.Stage.CreateNew(str(terrain_path))
    usd_geom.SetStageMetersPerUnit(stage, 1.0)
    usd_geom.SetStageUpAxis(stage, usd_geom.Tokens.z)
    root = usd_geom.Xform.Define(stage, "/Terrain")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("fireviewer:role", "continuous_visual_terrain")
    root.GetPrim().SetCustomDataByKey(
        "fireviewer:epsg2154_bounds", f"{xmin},{ymin},{xmax},{ymax}"
    )
    material = _textured_preview_material(
        stage=stage,
        path="/Terrain/Materials/Orthophoto",
        texture_path=ortho_path,
        base_path=terrain_path.parent,
        usd_shade=usd_shade,
        sdf=sdf,
    )
    points, counts, indices = _mesh_grid(
        values=values,
        xmin=xmin,
        ymax=ymax,
        origin_x=origin_x,
        origin_y=origin_y,
        samples=CONTINUOUS_TERRAIN_SAMPLES,
        width_metres=xmax - xmin,
        height_metres=ymax - ymin,
        gf=gf,
    )
    mesh = _create_mesh(
        stage=stage,
        path="/Terrain/Surface",
        points=points,
        counts=counts,
        indices=indices,
        usd_geom=usd_geom,
        semantic="terrain",
        purpose=usd_geom.Tokens.render,
    )
    # Smooth per-vertex normals remove the faceted, patchwork illumination of
    # the earlier individual-tile meshes while retaining actual MNT relief.
    rows = np.linspace(0, values.shape[0] - 1, CONTINUOUS_TERRAIN_SAMPLES, dtype=np.intp)
    columns = np.linspace(0, values.shape[1] - 1, CONTINUOUS_TERRAIN_SAMPLES, dtype=np.intp)
    sampled = values[np.ix_(rows, columns)].astype(np.float32, copy=False)
    finite = sampled[np.isfinite(sampled)]
    if finite.size == 0:
        raise ValueError("continuous terrain MNT has no finite elevation sample")
    sampled = np.where(np.isfinite(sampled), sampled, float(np.median(finite)))
    south_gradient, east_gradient = np.gradient(
        sampled,
        float(ymax - ymin) / float(CONTINUOUS_TERRAIN_SAMPLES - 1),
        float(xmax - xmin) / float(CONTINUOUS_TERRAIN_SAMPLES - 1),
    )
    normals = []
    for south, east in zip(south_gradient.ravel(), east_gradient.ravel(), strict=True):
        normal = np.array((-float(east), float(south), 1.0), dtype=np.float32)
        normal /= max(1e-6, float(np.linalg.norm(normal)))
        normals.append(gf.Vec3f(float(normal[0]), float(normal[1]), float(normal[2])))
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(usd_geom.Tokens.vertex)
    _set_planar_uv(mesh=mesh, points=points, gf=gf, usd_geom=usd_geom, sdf=sdf)
    _bind_preview_material(prim=mesh, material=material, usd_shade=usd_shade)
    mesh.GetPrim().CreateAttribute("fireviewer:orthophoto", sdf.ValueTypeNames.Asset).Set(
        sdf.AssetPath(_relative_asset(ortho_path, terrain_path.parent))
    )
    stage.GetRootLayer().Save()
    return len(points)


def _light_lod0_textures(*, zone_root: Path, lock: dict[str, Any]) -> dict[str, Path]:
    """Return the hash-verified 20 cm texture for each hero kilometre tile."""

    entries = lock.get("entries")
    if not isinstance(entries, list):
        raise ValueError("light source lock entries are malformed")
    result: dict[str, Path] = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("dataset") != "ortho_lod0":
            continue
        tile_ref = str(entry.get("tile_ref", ""))
        if not tile_ref:
            raise RuntimeError("20 cm orthophoto source has no tile reference")
        result[tile_ref] = _entry_path(zone_root, entry)
    if len(result) != 4:
        raise RuntimeError("light source lock must contain four 20 cm hero orthophotos")
    return result


def _light_hero_bounds(lock: dict[str, Any]) -> tuple[int, int, int, int]:
    entries = lock.get("entries")
    if not isinstance(entries, list):
        raise ValueError("light source lock entries are malformed")
    bounds = [
        entry.get("bbox_epsg2154")
        for entry in entries
        if isinstance(entry, dict) and entry.get("dataset") == "ortho_lod0"
    ]
    if len(bounds) != 4 or any(not isinstance(item, list) or len(item) != 4 for item in bounds):
        raise RuntimeError("light source lock has no complete camera LOD0 bounds")
    return (
        min(int(item[0]) for item in bounds),
        min(int(item[1]) for item in bounds),
        max(int(item[2]) for item in bounds),
        max(int(item[3]) for item in bounds),
    )


def _package_light_external_assets(
    *, zone_root: Path, build_root: Path, lock: dict[str, Any]
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """Copy every locked CC0 input into the portable scene package.

    Each glTF sidecar keeps its relative location.  This prevents Composer from
    resolving a texture from the mutable raw cache after an accepted package is
    archived or after raw sources are cleaned.
    """

    entries = lock.get("entries")
    if not isinstance(entries, list):
        raise ValueError("light source lock entries are malformed")
    result: dict[str, Path] = {}
    receipt_assets: list[dict[str, Any]] = []
    target_root = build_root / "assets" / "external"
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("dataset") != "assets":
            continue
        source = _entry_path(zone_root, entry)
        relative = Path(str(entry.get("relative_path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("external asset lock has an unsafe relative path")
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        asset_id = str(entry.get("id", ""))
        result[asset_id] = target
        receipt_assets.append(
            {
                "id": asset_id,
                "asset_id": str(entry.get("asset_id", "")),
                "source_url": str(entry.get("url", "")),
                "license": str(entry.get("license", "")),
                "version": str(entry.get("version", "")),
                "source_sha256": sha256(source),
                "packaged_path": target.relative_to(build_root).as_posix(),
                "packaged_sha256": sha256(target),
            }
        )
    if not result:
        raise RuntimeError("light source lock has no external assets")
    return result, receipt_assets


def _convert_gltf_to_usd(*, source: Path, target: Path) -> Path:
    """Convert a packaged glTF asset with Kit's native converter.

    A failed conversion is a hard failure: falling back to a cone or a generic
    placeholder would hide a missing production asset from the reviewer.
    """

    import asyncio
    import omni.kit.asset_converter as asset_converter

    target.parent.mkdir(parents=True, exist_ok=True)
    context = asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.embed_textures = False
    task = asset_converter.get_instance().create_converter_task(
        str(source), str(target), context
    )
    loop = asyncio.get_event_loop()
    converted = loop.run_until_complete(task.wait_until_finished())
    if not converted or not target.is_file():
        raise RuntimeError(f"Kit could not convert locked glTF asset: {source.name}")
    return target


def _write_aggregate(
    *,
    aggregate_path: Path,
    payload_records: list[
        tuple[Path, dict[str, Path] | None, dict[str, str]]
    ],
    origin_x: int,
    origin_y: int,
    usd: Any,
    usd_geom: Any,
) -> None:
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    stage = usd.Stage.CreateNew(str(aggregate_path))
    usd_geom.SetStageMetersPerUnit(stage, 1.0)
    usd_geom.SetStageUpAxis(stage, usd_geom.Tokens.z)
    aggregate = usd_geom.Xform.Define(stage, "/Aggregate")
    stage.SetDefaultPrim(aggregate.GetPrim())
    for terrain_path, detail_paths, tile in payload_records:
        child = usd_geom.Xform.Define(
            stage, f"/Aggregate/{tile['tile_ref']}"
        )
        xmin, ymin, xmax, ymax = (
            int(tile[key]) for key in ("xmin", "ymin", "xmax", "ymax")
        )
        child.GetPrim().SetCustomDataByKey(
            "fireviewer:local_bounds",
            f"{xmin - origin_x},{ymin - origin_y},"
            f"{xmax - origin_x},{ymax - origin_y}",
        )
        child.GetPrim().SetCustomDataByKey(
            "fireviewer:tile_ref", tile["tile_ref"]
        )
        terrain = usd_geom.Xform.Define(
            stage, f"{child.GetPath()}/Terrain"
        )
        terrain.GetPrim().GetPayloads().AddPayload(
            _relative_asset(terrain_path, aggregate_path.parent)
        )
        if detail_paths is not None:
            if set(detail_paths) != {"HERO", "MID", "FAR"}:
                raise ValueError(
                    f"{tile['tile_ref']} aggregate detail LODs are incomplete"
                )
            for level, child_name in (
                ("HERO", "Details"),
                ("MID", "DetailsMid"),
                ("FAR", "DetailsFar"),
            ):
                details = usd_geom.Xform.Define(
                    stage, f"{child.GetPath()}/{child_name}"
                )
                details.GetPrim().SetCustomDataByKey(
                    "fireviewer:detail_level", level
                )
                details.GetPrim().GetPayloads().AddPayload(
                    _relative_asset(
                        detail_paths[level], aggregate_path.parent
                    )
                )
    stage.GetRootLayer().Save()


def _write_cameras(
    *,
    cameras_path: Path,
    origin_x: int,
    origin_y: int,
    elevation: float,
    usd: Any,
    usd_geom: Any,
    gf: Any,
) -> list[dict[str, Any]]:
    stage = usd.Stage.CreateNew(str(cameras_path))
    usd_geom.SetStageMetersPerUnit(stage, 1.0)
    usd_geom.SetStageUpAxis(stage, usd_geom.Tokens.z)
    root = usd_geom.Xform.Define(stage, "/ReviewCameras")
    stage.SetDefaultPrim(root.GetPrim())
    offsets = ((-7000, -7000), (-3000, -7000), (1000, -7000), (5000, -7000), (-7000, -2000), (-3000, -2000), (1000, -2000), (5000, -2000), (-7000, 3000), (-3000, 3000), (1000, 3000), (5000, 3000))
    result: list[dict[str, Any]] = []
    for index, (x, y) in enumerate(offsets, start=1):
        camera = usd_geom.Camera.Define(stage, f"/ReviewCameras/Review{index:02d}")
        eye = gf.Vec3d(float(x), float(y - 1800), elevation + 1200.0)
        target = gf.Vec3d(float(x), float(y), elevation)
        # USD cameras look down local -Z with +Y as the image up direction.
        # SetLookAt returns the view matrix; its inverse is the camera's world
        # transform.  A translation-only camera silently kept the default
        # orientation and missed the prescribed review target.
        transform = gf.Matrix4d().SetLookAt(
            eye,
            target,
            gf.Vec3d(0.0, 0.0, 1.0),
        ).GetInverse()
        camera.AddTransformOp().Set(transform)
        camera.CreateFocalLengthAttr(35.0)
        camera.CreateClippingRangeAttr((0.25, 50_000.0))
        camera.CreateFocusDistanceAttr(float((eye - target).GetLength()))
        camera.GetPrim().SetCustomDataByKey(
            "fireviewer:look_at_local", f"{float(x)},{float(y)},{elevation}"
        )
        result.append({"name": f"Review{index:02d}", "local_target": [x, y, elevation], "epsg2154_target": [origin_x + x, origin_y + y, elevation]})
    stage.GetRootLayer().Save()
    return result


def _write_root(
    *,
    root_path: Path,
    aggregate_paths: list[Path],
    visual_terrain_path: Path | None,
    cameras_path: Path,
    zone: dict[str, Any],
    source_lock: Path,
    flow_asset: Path,
    flow_asset_lock: dict[str, Any],
    origin_x: int,
    origin_y: int,
    flow_elevation: float,
    usd: Any,
    usd_geom: Any,
    usd_lux: Any,
) -> None:
    stage = usd.Stage.CreateNew(str(root_path))
    usd_geom.SetStageMetersPerUnit(stage, 1.0)
    usd_geom.SetStageUpAxis(stage, usd_geom.Tokens.z)
    world = usd_geom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    world.GetPrim().SetCustomDataByKey("fireviewer:coordinate_convention", COORDINATE_CONVENTION)
    world.GetPrim().SetCustomDataByKey(
        "fireviewer:epsg2154_origin", f"{origin_x},{origin_y},0"
    )
    world.GetPrim().SetCustomDataByKey("fireviewer:vertical_datum", "IGN69")
    world.GetPrim().SetCustomDataByKey("fireviewer:source_lock_sha256", sha256(source_lock))
    world.GetPrim().SetCustomDataByKey("fireviewer:flow_asset_sha256", str(flow_asset_lock["packaged_sha256"]))
    metadata = usd_geom.Xform.Define(stage, "/World/Metadata")
    metadata.GetPrim().SetCustomDataByKey("fireviewer:zone_id", zone["id"])
    metadata.GetPrim().SetCustomDataByKey("fireviewer:zone_name", zone["name"])
    lighting = usd_geom.Xform.Define(stage, "/World/Lighting")
    lighting.GetPrim().SetCustomDataByKey("fireviewer:role", "review_lighting")
    dome = usd_lux.DomeLight.Define(stage, "/World/Lighting/Sky")
    # RTX renders the aerial/orthophoto context under the dome contribution as
    # well as under the distant sun.  A value of 0.35 is night-level fill for
    # this renderer, so it produces a black viewport even with valid terrain
    # texture payloads.  The 250 / 1000 sky/sun pair is calibrated against the
    # packaged Z16 orthophoto in native RTX: it keeps the satellite texture
    # legible without bleaching fields and forest canopy.
    dome.CreateIntensityAttr(250.0)
    dome.CreateColorAttr((0.72, 0.80, 0.92))
    sun = usd_lux.DistantLight.Define(stage, "/World/Lighting/Sun")
    # DistantLight intensity is expressed in daylight-scale units.  The old
    # value of 3 gave the orthophoto only a few lux, which makes a valid RGB
    # terrain read as black in RTX.  1000 is the paired daylight setting for
    # the authored dome above; it restores readable colour without clipping
    # the imagery.
    sun.CreateIntensityAttr(1000.0)
    sun.CreateAngleAttr(0.55)
    sun.AddRotateXYZOp().Set((35.0, -25.0, 35.0))
    terrain = usd_geom.Xform.Define(stage, "/World/Terrain")
    if visual_terrain_path is not None:
        # One geographic surface is the visible cartographic contract.  The
        # 1 km payloads below are still retained in the package for collision,
        # provenance and later near-camera work, but they must never show up as
        # 400 independently lit squares in a Composer overview.
        terrain.GetPrim().SetCustomDataByKey(
            "fireviewer:visual_composition", "continuous_20km_surface"
        )
        surface = usd_geom.Xform.Define(stage, "/World/Terrain/VisualSurface")
        surface.GetPrim().GetPayloads().AddPayload(
            _relative_asset(visual_terrain_path, root_path.parent)
        )
        terrain_data = usd_geom.Scope.Define(stage, "/World/TerrainData")
        terrain_data.GetPrim().SetCustomDataByKey(
            "fireviewer:role", "non_render_tile_payload_catalog"
        )
        terrain_data.GetPrim().SetCustomDataByKey(
            "fireviewer:aggregate_count", len(aggregate_paths)
        )
    else:
        for aggregate_path in aggregate_paths:
            aggregate = usd_geom.Xform.Define(stage, f"/World/Terrain/{aggregate_path.stem}")
            # Aggregate layers contain only lightweight tile headers and
            # payload arcs.  Compose them as references so LoadNone can discover
            # every tile before selecting the camera working set.
            aggregate.GetPrim().GetReferences().AddReference(
                _relative_asset(aggregate_path, root_path.parent)
            )
    for name in ("Imagery", "Hydrology", "Roads", "Buildings", "Vegetation", "Collisions", "Semantics"):
        scope = usd_geom.Scope.Define(stage, f"/World/{name}")
        scope.GetPrim().SetCustomDataByKey("fireviewer:layer", name.lower())
    # Fire/smoke is retained as a locked, packaged future simulation asset, but
    # the review root does not compose its payload at all.  Visibility alone
    # would still make stage.Load() allocate Flow resources before the human
    # gate.  The post-acceptance simulation layer is the only component allowed
    # to attach this payload.
    flow = usd_geom.Xform.Define(stage, "/World/FireAndSmoke")
    flow.AddTranslateOp().Set((0.0, 0.0, flow_elevation + 2.0))
    flow.AddRotateXOp().Set(90.0)
    flow.AddScaleOp().Set((0.01, 0.01, 0.01))
    usd_geom.Imageable(flow.GetPrim()).MakeInvisible()
    _apply_training_semantics(flow.GetPrim(), "fire_and_smoke")
    flow.GetPrim().SetCustomDataByKey("fireviewer:asset_lock_id", str(flow_asset_lock["id"]))
    flow.GetPrim().SetCustomDataByKey(
        "fireviewer:staged_payload",
        _relative_asset(flow_asset, root_path.parent),
    )
    flow.GetPrim().SetCustomDataByKey(
        "fireviewer:default_visibility",
        "uncomposed_until_editor_review_acceptance",
    )
    cameras = usd_geom.Xform.Define(stage, "/World/ReviewCameras")
    cameras.GetPrim().GetReferences().AddReference(_relative_asset(cameras_path, root_path.parent))
    stage.GetRootLayer().Save()


def _building_family(properties: dict[str, Any]) -> str:
    normalized = " ".join(
        str(properties.get(key, "")).lower()
        for key in ("nature", "usage_1", "usage_2", "etat_de_l_objet")
    )
    if any(
        term in normalized
        for term in ("agric", "hangar", "silo", "serre", "élevage", "elevage")
    ):
        return "agricultural"
    if any(
        term in normalized
        for term in ("industri", "usine", "entrepôt", "entrepot", "commercial")
    ):
        return "industrial"
    if any(
        term in normalized
        for term in ("annexe", "garage", "cabane", "abri", "dépendance", "dependance")
    ):
        return "annex"
    return "habitat"


def _oriented_footprint(
    polygon: list[list[float]],
) -> tuple[float, float, float]:
    """Return principal length, width and Z rotation for a source footprint."""

    points = np.asarray(polygon, dtype=np.float64)
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered.T, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    perpendicular = np.array((-principal[1], principal[0]), dtype=np.float64)
    primary_projection = centered @ principal
    secondary_projection = centered @ perpendicular
    length = max(0.1, float(primary_projection.max() - primary_projection.min()))
    width = max(0.1, float(secondary_projection.max() - secondary_projection.min()))
    angle = math.atan2(float(principal[1]), float(principal[0]))
    if width > length:
        length, width = width, length
        angle += math.pi * 0.5
    return length, width, angle


def _append_buildings(
    *,
    stage: Any,
    features: Iterable[dict[str, Any]],
    origin_x: int,
    origin_y: int,
    elevation_grid: _ElevationGrid,
    surface_grid: _ElevationGrid,
    height_grid: _ElevationGrid,
    fallback_elevation: float,
    usd_geom: Any,
    usd_shade: Any,
    sdf: Any,
    gf: Any,
    roof_texture: Path | None,
    asset_base_path: Path,
    building_assets: dict[str, list[dict[str, Any]]] | None = None,
    hero_bounds: tuple[int, int, int, int] | None = None,
    overview_visible: bool = True,
    scene_root: str = "/World",
    instance_namespace: int = 0,
    asset_lod: str = "HERO",
) -> int:
    """Place family-compatible photoreal buildings with uniform-only scale."""

    materials = {
        "limestone": _colored_preview_material(
            stage=stage, path=f"{scene_root}/Materials/FacadeLimestone", color=(0.68, 0.61, 0.50), usd_shade=usd_shade, sdf=sdf
        ),
        "plaster": _colored_preview_material(
            stage=stage, path=f"{scene_root}/Materials/FacadePlaster", color=(0.78, 0.73, 0.63), usd_shade=usd_shade, sdf=sdf
        ),
        "farm": _colored_preview_material(
            stage=stage, path=f"{scene_root}/Materials/FacadeFarm", color=(0.51, 0.43, 0.33), usd_shade=usd_shade, sdf=sdf
        ),
    }
    roof_material = (
        _textured_preview_material(
            stage=stage,
            path=f"{scene_root}/Materials/RoofTerracotta",
            texture_path=roof_texture,
            base_path=asset_base_path,
            usd_shade=usd_shade,
            sdf=sdf,
        )
        if roof_texture is not None
        else _colored_preview_material(
            stage=stage,
            path=f"{scene_root}/Materials/RoofTerracotta",
            color=(0.43, 0.19, 0.10),
            usd_shade=usd_shade,
            sdf=sdf,
        )
    )
    library = building_assets or {}
    hero_instancer = None
    hero_positions: list[Any] = []
    hero_indices: list[int] = []
    hero_scales: list[Any] = []
    hero_orientations: list[Any] = []
    hero_ids: list[int] = []
    hero_stable_ids: list[str] = []
    hero_footprint_radii_m: list[float] = []
    hero_group_ids: list[str] = []
    prototype_records: list[dict[str, Any]] = []
    prototype_usage: list[int] = []
    hero_limit = _positive_int_environment(
        "FW_SDG_PHOTOREAL_BUILDING_INSTANCE_LIMIT",
        PHOTOREAL_BUILDING_INSTANCE_LIMIT,
        maximum=100_000,
    )
    if library:
        hero_instancer = usd_geom.PointInstancer.Define(
            stage, f"{scene_root}/Buildings/HeroBuildings"
        )
        prototype_targets = []
        for family in ("habitat", "agricultural", "industrial", "annex"):
            for asset in library.get(family, []):
                lod_paths = asset.get("lod_paths")
                if not isinstance(lod_paths, dict) or asset_lod not in lod_paths:
                    raise RuntimeError(
                        f"building asset has no materialized {asset_lod} LOD"
                    )
                path = Path(lod_paths[asset_lod])
                prototype_path = (
                    f"{scene_root}/Buildings/HeroBuildings/Prototypes/"
                    f"Building{len(prototype_records):02d}"
                )
                prototype = usd_geom.Xform.Define(stage, prototype_path)
                anchor = [float(value) for value in asset["ground_anchor_m"]]
                prototype.AddTranslateOp().Set(
                    (-anchor[0], -anchor[1], -anchor[2])
                )
                intrinsic_uniform_scale = float(
                    asset.get("intrinsic_uniform_scale", 1.0)
                )
                if intrinsic_uniform_scale != 1.0:
                    prototype.AddScaleOp().Set(
                        (
                            intrinsic_uniform_scale,
                            intrinsic_uniform_scale,
                            intrinsic_uniform_scale,
                        )
                    )
                prototype.GetPrim().GetReferences().AddReference(
                    _relative_asset(path, asset_base_path)
                )
                prototype.GetPrim().SetCustomDataByKey(
                    "fireviewer:source_asset", path.name
                )
                prototype.GetPrim().SetCustomDataByKey(
                    "fireviewer:asset_family", f"buildings.{family}"
                )
                prototype.GetPrim().SetCustomDataByKey(
                    "fireviewer:lod_role", "source_identity_lod_chain"
                )
                _apply_training_semantics(
                    prototype.GetPrim(), "building_photoreal"
                )
                prototype_targets.append(prototype_path)
                prototype_records.append(asset)
                prototype_usage.append(0)
        if not prototype_targets:
            raise RuntimeError("photoreal building library has no usable prototypes")
        hero_instancer.CreatePrototypesRel().SetTargets(prototype_targets)
        if instance_namespace <= 0:
            raise RuntimeError(
                "photoreal buildings require a positive tile instance namespace"
            )
        _apply_training_semantics(
            hero_instancer.GetPrim(), "building_photoreal"
        )
    effective_hero_bounds = hero_bounds or (
        origin_x - 10_000,
        origin_y - 10_000,
        origin_x + 10_000,
        origin_y + 10_000,
    )
    hero_xmin, hero_ymin, hero_xmax, hero_ymax = effective_hero_bounds
    count = 0
    unmatched: list[str] = []
    for feature in features:
        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        nature = str(properties.get("nature", ""))
        wall_material = materials[
            "farm" if "agricole" in nature.lower() else "plaster" if count % 3 else "limestone"
        ]
        for polygon in _feature_polygons(feature.get("geometry")):
            centroid_x, centroid_y = _polygon_centroid(polygon)
            terrain_elevation = elevation_grid.elevation(
                centroid_x,
                centroid_y,
                fallback=fallback_elevation,
            )
            surface_elevation = surface_grid.elevation(
                centroid_x,
                centroid_y,
                fallback=terrain_elevation + 6.0,
            )
            normalized_height = height_grid.elevation(
                centroid_x,
                centroid_y,
                fallback=max(2.5, surface_elevation - terrain_elevation),
            )
            lidar_height = max(
                2.5,
                min(
                    80.0,
                    max(
                        normalized_height,
                        surface_elevation - terrain_elevation,
                    ),
                ),
            )
            base = _as_float(
                properties.get("altitude_minimale_sol"),
                default=terrain_elevation,
            )
            height = max(
                2.5,
                min(
                    80.0,
                    _as_float(properties.get("hauteur"), default=lidar_height),
                ),
            )
            if hero_instancer is not None:
                if len(hero_positions) >= hero_limit:
                    raise RuntimeError(
                        "photoreal building count exceeds the configured scene limit"
                    )
                if not (
                    hero_xmin <= centroid_x <= hero_xmax
                    and hero_ymin <= centroid_y <= hero_ymax
                ):
                    continue
                family = _building_family(properties)
                length, width, angle = _oriented_footprint(polygon)
                compatible: list[tuple[float, int, float, float]] = []
                for prototype_index, asset in enumerate(prototype_records):
                    if str(asset.get("family", "")) != f"buildings.{family}":
                        continue
                    dimensions = asset["native_dimensions_m"]
                    native_x = float(dimensions["x"])
                    native_y = float(dimensions["y"])
                    native_z = float(dimensions["z"])
                    options = (
                        (length / native_x, width / native_y, angle),
                        (length / native_y, width / native_x, angle + math.pi * 0.5),
                    )
                    for ratio_x, ratio_y, asset_angle in options:
                        ratio_z = height / native_z
                        scale = (ratio_x * ratio_y * ratio_z) ** (1.0 / 3.0)
                        minimum_scale = float(asset["minimum_uniform_scale"])
                        maximum_scale = float(asset["maximum_uniform_scale"])
                        if not minimum_scale <= scale <= maximum_scale:
                            continue
                        distortion = max(
                            abs(math.log(max(1e-6, ratio / scale)))
                            for ratio in (ratio_x, ratio_y, ratio_z)
                        )
                        if distortion > math.log(1.8):
                            continue
                        usage_penalty = prototype_usage[prototype_index] * 0.02
                        compatible.append(
                            (
                                distortion + usage_penalty,
                                prototype_index,
                                asset_angle,
                                scale,
                            )
                        )
                if not compatible:
                    # Fictional scene variants must still use real downloaded
                    # photoreal assets when a source footprint falls outside
                    # the preferred family envelope. Select the closest real
                    # prototype and clamp its uniform scale instead of
                    # substituting procedural geometry or leaving a hole.
                    for prototype_index, asset in enumerate(prototype_records):
                        dimensions = asset["native_dimensions_m"]
                        native_x = float(dimensions["x"])
                        native_y = float(dimensions["y"])
                        native_z = float(dimensions["z"])
                        family_penalty = (
                            0.0
                            if str(asset.get("family", ""))
                            == f"buildings.{family}"
                            else 1.5
                        )
                        for ratio_x, ratio_y, asset_angle in (
                            (length / native_x, width / native_y, angle),
                            (
                                length / native_y,
                                width / native_x,
                                angle + math.pi * 0.5,
                            ),
                        ):
                            ratio_z = height / native_z
                            requested_scale = (
                                ratio_x * ratio_y * ratio_z
                            ) ** (1.0 / 3.0)
                            scale = max(
                                float(asset["minimum_uniform_scale"]),
                                min(
                                    float(asset["maximum_uniform_scale"]),
                                    requested_scale,
                                ),
                            )
                            distortion = max(
                                abs(math.log(max(1e-6, ratio / scale)))
                                for ratio in (ratio_x, ratio_y, ratio_z)
                            )
                            usage_penalty = (
                                prototype_usage[prototype_index] * 0.02
                            )
                            compatible.append(
                                (
                                    family_penalty
                                    + distortion
                                    + usage_penalty,
                                    prototype_index,
                                    asset_angle,
                                    scale,
                                )
                            )
                (
                    _score,
                    selected_index,
                    selected_angle,
                    uniform_scale,
                ) = min(compatible)
                selected = prototype_records[selected_index]
                hero_positions.append(
                    gf.Vec3f(
                        float(centroid_x - origin_x),
                        float(centroid_y - origin_y),
                        base,
                    )
                )
                hero_indices.append(selected_index)
                hero_scales.append(
                    gf.Vec3f(uniform_scale, uniform_scale, uniform_scale)
                )
                hero_orientations.append(
                    gf.Quath(
                        float(math.cos(selected_angle * 0.5)),
                        gf.Vec3h(
                            0.0,
                            0.0,
                            float(math.sin(selected_angle * 0.5)),
                        ),
                    )
                )
                numeric_id = _stable_instance_id(
                    tile_namespace=instance_namespace,
                    family="buildings",
                    local_index=count,
                )
                hero_ids.append(numeric_id)
                hero_stable_ids.append(
                    f"tile-{instance_namespace}:buildings:{numeric_id}"
                )
                native_dimensions = selected["native_dimensions_m"]
                hero_footprint_radii_m.append(
                    max(
                        0.1,
                        max(
                            float(native_dimensions["x"]),
                            float(native_dimensions["y"]),
                        )
                        * uniform_scale
                        * 0.5,
                    )
                )
                local_x = float(centroid_x - origin_x)
                local_y = float(centroid_y - origin_y)
                hero_group_ids.append(
                    "settlement:"
                    f"{math.floor(local_x / 125.0)}:"
                    f"{math.floor(local_y / 125.0)}"
                )
                prototype_usage[selected_index] += 1
                # Preserve the exact source footprint as a guide/semantic layer
                # without overlaying a procedural shell on the visual asset.
                footprint_points = [
                    gf.Vec3f(float(x - origin_x), float(y - origin_y), base)
                    for x, y in polygon
                ]
                footprint = _create_mesh(
                    stage=stage,
                    path=f"{scene_root}/Semantics/BuildingFootprint_{count:05d}",
                    points=footprint_points,
                    counts=[len(footprint_points)],
                    indices=list(range(len(footprint_points))),
                    usd_geom=usd_geom,
                    semantic=f"building_footprint_{family}",
                    purpose=usd_geom.Tokens.guide,
                )
                footprint.GetPrim().SetCustomDataByKey(
                    "fireviewer:asset_id", str(selected["asset_id"])
                )
                count += 1
                continue
            # Legacy non-photoreal builds retain the source footprint; the pod
            # profile always supplies the family library and cannot enter here.
            points = [gf.Vec3f(float(x - origin_x), float(y - origin_y), base) for x, y in polygon]
            points += [gf.Vec3f(float(x - origin_x), float(y - origin_y), base + height) for x, y in polygon]
            size = len(polygon)
            counts = [size, size] + [4] * size
            indices = list(reversed(range(size))) + list(range(size, size * 2))
            for index in range(size):
                next_index = (index + 1) % size
                indices.extend((index, next_index, next_index + size, index + size))
            mesh = _create_mesh(stage=stage, path=f"{scene_root}/Buildings/Building_{count:05d}", points=points, counts=counts, indices=indices, usd_geom=usd_geom, semantic="building", purpose=usd_geom.Tokens.render)
            _bind_preview_material(prim=mesh, material=wall_material, usd_shade=usd_shade)
            mesh.GetPrim().SetCustomDataByKey("fireviewer:building_typology", nature or "local_habitat")
            roof_height = max(1.5, min(5.5, height * 0.20))
            roof_points = [
                gf.Vec3f(float(x - origin_x), float(y - origin_y), base + height)
                for x, y in polygon
            ]
            roof_points.append(
                gf.Vec3f(
                    float(centroid_x - origin_x),
                    float(centroid_y - origin_y),
                    base + height + roof_height,
                )
            )
            roof_indices: list[int] = []
            for index in range(size):
                roof_indices.extend((index, (index + 1) % size, size))
            roof = _create_mesh(
                stage=stage,
                path=f"{scene_root}/Buildings/Building_{count:05d}/Roof",
                points=roof_points,
                counts=[3] * size,
                indices=roof_indices,
                usd_geom=usd_geom,
                semantic="building_roof",
                purpose=usd_geom.Tokens.render,
            )
            _set_planar_uv(mesh=roof, points=roof_points, gf=gf, usd_geom=usd_geom, sdf=sdf)
            _bind_preview_material(prim=roof, material=roof_material, usd_shade=usd_shade)
            count += 1
    if hero_instancer is not None and hero_positions:
        if unmatched:
            raise RuntimeError(
                f"{len(unmatched)} buildings have no dimension-compatible "
                "photoreal family asset"
            )
        hero_instancer.CreatePositionsAttr(hero_positions)
        hero_instancer.CreateProtoIndicesAttr(hero_indices)
        hero_instancer.CreateScalesAttr(hero_scales)
        hero_instancer.CreateOrientationsAttr(hero_orientations)
        hero_instancer.CreateIdsAttr(hero_ids)
        _author_instance_identity_primvars(
            instancer=hero_instancer,
            stable_ids=hero_stable_ids,
            footprint_radii_m=hero_footprint_radii_m,
            group_ids=hero_group_ids,
            usd_geom=usd_geom,
            sdf=sdf,
        )
        hero_instancer.GetPrim().SetCustomDataByKey(
            "fireviewer:instance_count", len(hero_positions)
        )
        if len(hero_positions) >= 8 and not os.getenv(
            "FW_SDG_FAST_EDITOR_PREVIEW", ""
        ).strip().lower() in {"1", "true", "yes", "on"}:
            dominance = max(prototype_usage) / float(len(hero_positions))
            if dominance > 0.25 + 1e-9:
                raise RuntimeError(
                    f"one building prototype dominates {dominance:.1%} of the tile"
                )
    elif hero_instancer is not None and unmatched:
        raise RuntimeError(
            "tile buildings have no dimension-compatible photoreal family asset"
        )
    buildings = stage.GetPrimAtPath(f"{scene_root}/Buildings")
    if buildings and buildings.IsValid() and not overview_visible:
        # At the scale of a 20 km geographic overview, simplified pyramidal
        # roofs turn correct BDTOPO footprints into white visual noise.  The
        # locked 2 m orthophoto is the authoritative, readable overview of the
        # real buildings; the generated geometry remains in the stage as an
        # explicit LOD0 layer that can be enabled for a close inspection.
        usd_geom.Imageable(buildings).MakeInvisible()
        buildings.SetCustomDataByKey("fireviewer:overview", "hidden_use_orthophoto")
    return count


def _append_road_ribbons(*, stage: Any, features: Iterable[dict[str, Any]], origin_x: int, origin_y: int, elevation_grid: _ElevationGrid, fallback_elevation: float, usd_geom: Any, gf: Any, scene_root: str = "/World", clip_bounds: tuple[float, float, float, float] | None = None, render_visible: bool = True) -> int:
    """Legacy route-ribbon helper; native production must not call it.

    Roads are visible through the orthophoto already bound to the terrain.
    Source vectors are retained separately for topology, annotations and actor
    placement, never as generated USD road meshes.
    """

    raise RuntimeError(
        "route ribbon authoring is prohibited; use the orthophoto-derived "
        "terrain material and retained route topology"
    )

    count = 0
    for feature in features:
        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        width = max(2.0, min(16.0, _as_float(properties.get("largeur_de_chaussee"), default=4.5)))
        source_lines = _feature_lines(feature.get("geometry"))
        for source_line in source_lines:
            lines = (
                _clip_line_to_bounds(source_line, clip_bounds)
                if clip_bounds is not None
                else [source_line]
            )
            for line in lines:
                if len(line) < 2:
                    continue
                points: list[Any] = []
                indices: list[int] = []
                for index, (x, y) in enumerate(line):
                    before = line[max(0, index - 1)]
                    after = line[min(len(line) - 1, index + 1)]
                    dx = float(after[0] - before[0])
                    dy = float(after[1] - before[1])
                    length = math.hypot(dx, dy)
                    if length <= 1e-4:
                        continue
                    normal_x = -dy / length * width * 0.5
                    normal_y = dx / length * width * 0.5
                    elevation = elevation_grid.elevation(x, y, fallback=fallback_elevation) + 0.08
                    points.extend(
                        (
                            gf.Vec3f(float(x - origin_x + normal_x), float(y - origin_y + normal_y), elevation),
                            gf.Vec3f(float(x - origin_x - normal_x), float(y - origin_y - normal_y), elevation),
                        )
                    )
                if len(points) < 4:
                    continue
                for index in range(0, len(points) - 2, 2):
                    indices.extend((index, index + 1, index + 3, index + 2))
                ribbon = _create_mesh(
                    stage=stage,
                    path=f"{scene_root}/Roads/Road_{count:05d}",
                    points=points,
                    counts=[4] * ((len(points) // 2) - 1),
                    indices=indices,
                    usd_geom=usd_geom,
                    semantic="road",
                    purpose=(
                        usd_geom.Tokens.render
                        if render_visible
                        else usd_geom.Tokens.guide
                    ),
                )
                _set_color(prim=ribbon, color=(0.18, 0.18, 0.17), usd_geom=usd_geom)
                if not render_visible:
                    usd_geom.Imageable(ribbon).MakeInvisible()
                    ribbon.GetPrim().SetCustomDataByKey(
                        "fireviewer:visual_source",
                        "orthophoto_until_pbr_road_material_is_validated",
                    )
                count += 1
    return count


def _append_lines(*, stage: Any, root: str, features: Iterable[dict[str, Any]], origin_x: int, origin_y: int, elevation_grid: _ElevationGrid, fallback_elevation: float, usd_geom: Any, gf: Any, semantic: str, color: tuple[float, float, float], default_width: float, clip_bounds: tuple[float, float, float, float] | None = None, render_visible: bool = True) -> int:
    count = 0
    for feature in features:
        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        width = max(1.0, min(30.0, _as_float(properties.get("largeur_de_chaussee"), default=default_width)))
        for source_line in _feature_lines(feature.get("geometry")):
            lines = (
                _clip_line_to_bounds(source_line, clip_bounds)
                if clip_bounds is not None
                else [source_line]
            )
            for line in lines:
                if len(line) < 2:
                    continue
                points = [
                    gf.Vec3f(float(x - origin_x), float(y - origin_y), elevation_grid.elevation(x, y, fallback=fallback_elevation) + 0.25)
                    for x, y in line
                ]
                curve = usd_geom.BasisCurves.Define(stage, f"{root}/Feature_{count:05d}")
                curve.CreateTypeAttr(usd_geom.Tokens.linear)
                curve.CreateCurveVertexCountsAttr([len(points)])
                curve.CreatePointsAttr(points)
                curve.CreateWidthsAttr([width] * len(points))
                _apply_training_semantics(curve.GetPrim(), semantic)
                _set_color(prim=curve, color=color, usd_geom=usd_geom)
                if not render_visible:
                    usd_geom.Imageable(curve).CreatePurposeAttr(
                        usd_geom.Tokens.guide
                    )
                    usd_geom.Imageable(curve).MakeInvisible()
                    curve.GetPrim().SetCustomDataByKey(
                        "fireviewer:visual_source",
                        "orthophoto_until_pbr_water_material_is_validated",
                    )
                count += 1
    return count


def _append_hydro_surfaces(*, stage: Any, features: Iterable[dict[str, Any]], origin_x: int, origin_y: int, elevation_grid: _ElevationGrid, fallback_elevation: float, usd_geom: Any, gf: Any, scene_root: str = "/World", clip_bounds: tuple[float, float, float, float] | None = None, render_visible: bool = True) -> int:
    count = 0
    for feature in features:
        for source_polygon in _feature_polygons(feature.get("geometry")):
            polygon = (
                _clip_polygon_to_bounds(source_polygon, clip_bounds)
                if clip_bounds is not None
                else source_polygon
            )
            if len(polygon) < 3:
                continue
            centroid_x, centroid_y = _polygon_centroid(polygon)
            elevation = elevation_grid.elevation(centroid_x, centroid_y, fallback=fallback_elevation) + 0.15
            points = [gf.Vec3f(float(x - origin_x), float(y - origin_y), elevation) for x, y in polygon]
            mesh = _create_mesh(
                stage=stage,
                path=f"{scene_root}/Hydrology/Surface_{count:05d}",
                points=points,
                counts=[len(points)],
                indices=list(range(len(points))),
                usd_geom=usd_geom,
                semantic="water",
                purpose=(
                    usd_geom.Tokens.render
                    if render_visible
                    else usd_geom.Tokens.guide
                ),
            )
            _set_color(prim=mesh, color=(0.08, 0.25, 0.42), usd_geom=usd_geom)
            if not render_visible:
                usd_geom.Imageable(mesh).MakeInvisible()
                mesh.GetPrim().SetCustomDataByKey(
                    "fireviewer:visual_source",
                    "orthophoto_until_pbr_water_material_is_validated",
                )
            count += 1
    return count


def _polygon_area(polygon: list[list[float]]) -> float:
    return abs(
        sum(
            point[0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * point[1]
            for index, point in enumerate(polygon)
        )
    ) * 0.5


def _point_in_polygon(*, x: float, y: float, polygon: list[list[float]]) -> bool:
    crossings = 0
    for index, point in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        if (point[1] > y) != (following[1] > y):
            intersection = (following[0] - point[0]) * (y - point[1]) / (following[1] - point[1]) + point[0]
            if x < intersection:
                crossings += 1
    return crossings % 2 == 1


def _sample_polygon(*, polygon: list[list[float]], seed: int, attempts: int = 16) -> tuple[float, float]:
    """Deterministically sample inside a possibly concave source polygon."""

    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)
    state = seed & 0x7FFFFFFF
    for _ in range(attempts):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        x = min_x + (max_x - min_x) * (state / 0x7FFFFFFF)
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        y = min_y + (max_y - min_y) * (state / 0x7FFFFFFF)
        if _point_in_polygon(x=x, y=y, polygon=polygon):
            return x, y
    return _polygon_centroid(polygon)


def _clip_polygon_to_bounds(
    polygon: list[list[float]],
    bounds: tuple[float, float, float, float],
) -> list[list[float]]:
    """Clip one source polygon to a kilometre tile (Sutherland-Hodgman)."""

    xmin, ymin, xmax, ymax = bounds
    output = [list(point) for point in polygon]

    def clip(
        points: list[list[float]],
        *,
        inside: Any,
        intersection: Any,
    ) -> list[list[float]]:
        if not points:
            return []
        result: list[list[float]] = []
        previous = points[-1]
        previous_inside = inside(previous)
        for current in points:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    result.append(intersection(previous, current))
                result.append(current)
            elif previous_inside:
                result.append(intersection(previous, current))
            previous = current
            previous_inside = current_inside
        return result

    def vertical(a: list[float], b: list[float], x: float) -> list[float]:
        delta = b[0] - a[0]
        ratio = 0.0 if abs(delta) < 1e-12 else (x - a[0]) / delta
        return [x, a[1] + (b[1] - a[1]) * ratio]

    def horizontal(a: list[float], b: list[float], y: float) -> list[float]:
        delta = b[1] - a[1]
        ratio = 0.0 if abs(delta) < 1e-12 else (y - a[1]) / delta
        return [a[0] + (b[0] - a[0]) * ratio, y]

    output = clip(
        output,
        inside=lambda point: point[0] >= xmin,
        intersection=lambda a, b: vertical(a, b, xmin),
    )
    output = clip(
        output,
        inside=lambda point: point[0] <= xmax,
        intersection=lambda a, b: vertical(a, b, xmax),
    )
    output = clip(
        output,
        inside=lambda point: point[1] >= ymin,
        intersection=lambda a, b: horizontal(a, b, ymin),
    )
    output = clip(
        output,
        inside=lambda point: point[1] <= ymax,
        intersection=lambda a, b: horizontal(a, b, ymax),
    )
    return output if len(output) >= 3 else []


def _clip_segment_to_bounds(
    start: list[float],
    end: list[float],
    bounds: tuple[float, float, float, float],
) -> tuple[list[float], list[float]] | None:
    """Clip a segment to an axis-aligned tile with Liang-Barsky."""

    xmin, ymin, xmax, ymax = bounds
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    entering = 0.0
    leaving = 1.0
    for p, q in (
        (-dx, start[0] - xmin),
        (dx, xmax - start[0]),
        (-dy, start[1] - ymin),
        (dy, ymax - start[1]),
    ):
        if abs(p) < 1e-12:
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
        [start[0] + entering * dx, start[1] + entering * dy],
        [start[0] + leaving * dx, start[1] + leaving * dy],
    )


def _clip_line_to_bounds(
    line: list[list[float]],
    bounds: tuple[float, float, float, float],
) -> list[list[list[float]]]:
    """Return contiguous source-line fragments inside one kilometre tile."""

    fragments: list[list[list[float]]] = []
    current: list[list[float]] = []
    for start, end in zip(line, line[1:]):
        clipped = _clip_segment_to_bounds(start, end, bounds)
        if clipped is None:
            if len(current) >= 2:
                fragments.append(current)
            current = []
            continue
        clipped_start, clipped_end = clipped
        if current and math.dist(current[-1], clipped_start) <= 1e-6:
            current.append(clipped_end)
        else:
            if len(current) >= 2:
                fragments.append(current)
            current = [clipped_start, clipped_end]
    if len(current) >= 2:
        fragments.append(current)
    return fragments


def _forest_polygons_for_tile(
    features: Iterable[dict[str, Any]],
    *,
    bounds: tuple[float, float, float, float],
) -> list[list[list[float]]]:
    polygons: list[list[list[float]]] = []
    for feature in features:
        properties = (
            feature.get("properties")
            if isinstance(feature.get("properties"), dict)
            else {}
        )
        nature = str(properties.get("nature", ""))
        if not (
            nature == "Bois"
            or "Forêt" in nature
            or nature == "Lande ligneuse"
        ):
            continue
        for polygon in _feature_polygons(feature.get("geometry")):
            clipped = _clip_polygon_to_bounds(polygon, bounds)
            if clipped:
                polygons.append(clipped)
    return polygons


def _building_features_for_tile(
    features: Iterable[dict[str, Any]],
    *,
    bounds: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    """Assign each building exactly once by source-footprint centroid."""

    xmin, ymin, xmax, ymax = bounds
    selected: list[dict[str, Any]] = []
    for feature in features:
        polygons = list(_feature_polygons(feature.get("geometry")))
        if not polygons:
            continue
        if any(
            xmin <= (centroid := _polygon_centroid(polygon))[0] < xmax
            and ymin <= centroid[1] < ymax
            for polygon in polygons
        ):
            selected.append(feature)
    return selected


def _append_vegetation(*, stage: Any, features: Iterable[dict[str, Any]], origin_x: int, origin_y: int, elevation_grid: _ElevationGrid, fallback_elevation: float, usd_geom: Any, gf: Any, hero_bounds: tuple[int, int, int, int], tree_assets: list[Path], asset_base_path: Path, assets_z_up: bool = False, overview_visible: bool = True) -> int:
    """Use detailed locked trees near cameras; reserve simple fill for dense forest."""

    scope = usd_geom.Xform.Define(stage, "/World/Vegetation")
    _apply_training_semantics(scope.GetPrim(), "vegetation")
    if not overview_visible:
        # Do not scatter cone proxies across a photo-backed 20 km scene.  They
        # read as debug pins rather than forest.  The source canopy is already
        # accurately visible in the orthophoto; forest polygons are retained
        # as semantic metadata until a dedicated close-camera canopy LOD uses
        # proper mature-tree assets.
        forest_features = 0
        for feature in features:
            properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
            nature = str(properties.get("nature", ""))
            if nature == "Bois" or "Forêt" in nature:
                forest_features += sum(1 for _ in _feature_polygons(feature.get("geometry")))
        scope.GetPrim().SetCustomDataByKey(
            "fireviewer:overview", "orthophoto_canopy_no_proxy_instances"
        )
        return forest_features

    # A PointInstancer's prototype geometry must be a child of the consuming
    # instancer.  Keeping it as an ordinary sibling under /Vegetation makes
    # Composer render a second copy at the local origin: the black artificial
    # grove reported during review.  The standard layout also lets USD/Hydra
    # prune raw prototype traversal correctly.
    dense_instancer = usd_geom.PointInstancer.Define(stage, "/World/Vegetation/DenseForest")
    dense_prototypes = usd_geom.Scope.Define(stage, "/World/Vegetation/DenseForest/Prototypes")
    hero_instancer = usd_geom.PointInstancer.Define(stage, "/World/Vegetation/HeroTrees")
    hero_prototypes = usd_geom.Scope.Define(stage, "/World/Vegetation/HeroTrees/Prototypes")

    detailed_paths: list[str] = []
    for index, asset in enumerate(tree_assets):
        root = f"{hero_prototypes.GetPath()}/LockedTree{index}"
        prototype = usd_geom.Xform.Define(stage, root)
        # Legacy glTF assets are Y-up; the materialized NVIDIA wrappers are
        # already validated Z-up and must not receive a second axis rotation.
        if not assets_z_up:
            prototype.AddRotateXOp().Set(90.0)
        asset_scale = (
            1.0
            if assets_z_up or "pine" in asset.stem
            else 0.58
        )
        prototype.AddScaleOp().Set((asset_scale, asset_scale, asset_scale))
        prototype.GetPrim().GetReferences().AddReference(_relative_asset(asset, asset_base_path))
        prototype.GetPrim().SetCustomDataByKey("fireviewer:source_asset", asset.name)
        prototype.GetPrim().SetCustomDataByKey(
            "fireviewer:axis_normalization",
            "source_already_z_up" if assets_z_up else "gltf_y_up_to_usd_z_up",
        )
        detailed_paths.append(root)

    dense_paths: list[str] = []
    if tree_assets:
        # The pod profile uses the same locked photoreal USD library throughout
        # the forest.  The PointInstancer shares prototypes, so increasing tree
        # count does not duplicate geometry or textures.
        for index, asset in enumerate(tree_assets):
            root = f"{dense_prototypes.GetPath()}/LockedForestTree{index}"
            prototype = usd_geom.Xform.Define(stage, root)
            if not assets_z_up:
                prototype.AddRotateXOp().Set(90.0)
            prototype.GetPrim().GetReferences().AddReference(
                _relative_asset(asset, asset_base_path)
            )
            prototype.GetPrim().SetCustomDataByKey(
                "fireviewer:source_asset", asset.name
            )
            prototype.GetPrim().SetCustomDataByKey(
                "fireviewer:lod_role", "dense_forest_photoreal_instance"
            )
            dense_paths.append(root)
    else:
        # Kept only for the legacy non-pod builder.  The pod setup sets
        # FW_SDG_PHOTOREAL_ASSETS_REQUIRED and therefore cannot enter this
        # simple closed-forest fill path.
        foliage = ((0.08, 0.20, 0.06), (0.12, 0.29, 0.09), (0.15, 0.34, 0.11))
        for index, color in enumerate(foliage):
            root = f"{dense_prototypes.GetPath()}/DenseForest{index}"
            prototype = usd_geom.Xform.Define(stage, root)
            trunk = usd_geom.Cylinder.Define(stage, f"{root}/Trunk")
            trunk.CreateRadiusAttr(0.22)
            trunk.CreateHeightAttr(4.2)
            trunk.AddTranslateOp().Set((0.0, 0.0, 2.1))
            _set_color(prim=trunk, color=(0.20, 0.12, 0.06), usd_geom=usd_geom)
            for tier, radius in enumerate((2.8, 2.25, 1.65)):
                canopy = usd_geom.Cone.Define(stage, f"{root}/Canopy{tier}")
                canopy.CreateRadiusAttr(radius + index * 0.18)
                canopy.CreateHeightAttr(5.2)
                canopy.AddTranslateOp().Set((0.0, 0.0, 4.5 + tier * 2.3))
                _set_color(prim=canopy, color=color, usd_geom=usd_geom)
            prototype.GetPrim().SetCustomDataByKey(
                "fireviewer:lod_role", "dense_forest_fill_only"
            )
            dense_paths.append(root)
    dense_instancer.CreatePrototypesRel().SetTargets(dense_paths)
    hero_instancer.CreatePrototypesRel().SetTargets(detailed_paths)

    hero_positions: list[Any] = []
    hero_indices: list[int] = []
    hero_scales: list[Any] = []
    dense_positions: list[Any] = []
    dense_indices: list[int] = []
    dense_scales: list[Any] = []
    hero_xmin, hero_ymin, hero_xmax, hero_ymax = hero_bounds
    dense_candidates: list[tuple[int, list[list[float]], float]] = []
    hero_candidates: list[tuple[int, list[list[float]], float]] = []
    for feature_index, feature in enumerate(features):
        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        nature = str(properties.get("nature", ""))
        is_dense = nature == "Bois" or "Forêt fermée" in nature
        is_tree_cover = is_dense or nature in {"Forêt ouverte", "Lande ligneuse"}
        if not is_tree_cover:
            # Vines and agricultural patterns remain correctly visible in the
            # orthophoto; planting conifers over them would be a data error.
            continue
        for polygon in _feature_polygons(feature.get("geometry")):
            min_x = min(point[0] for point in polygon)
            max_x = max(point[0] for point in polygon)
            min_y = min(point[1] for point in polygon)
            max_y = max(point[1] for point in polygon)
            area = _polygon_area(polygon)
            if is_dense:
                dense_candidates.append((feature_index, polygon, area))
            if (
                not detailed_paths
                or max_x < hero_xmin
                or min_x > hero_xmax
                or max_y < hero_ymin
                or min_y > hero_ymax
            ):
                continue
            # Haies and vergers stay in the photo; only forest cover receives
            # 3D trees.  Estimate the camera overlap from its bounding box so
            # small fragmented polygons can no longer monopolize the budget.
            overlap_area = max(0.0, min(max_x, hero_xmax) - max(min_x, hero_xmin)) * max(
                0.0, min(max_y, hero_ymax) - max(min_y, hero_ymin)
            )
            if overlap_area > 0.0:
                hero_candidates.append((feature_index, polygon, overlap_area))

    # Forest canopy is geographically distributed over every closed-forest
    # polygon.  The budget keeps a 20 km scene interactive while a two-pass
    # density calculation avoids the prior camera-tile-only concentration.
    dense_area = sum(area for _feature_index, _polygon, area in dense_candidates)
    photoreal_forest = bool(tree_assets)
    forest_budget = _positive_int_environment(
        "FW_SDG_FOREST_INSTANCE_BUDGET",
        (
            PHOTOREAL_FOREST_INSTANCE_BUDGET
            if photoreal_forest
            else FOREST_CANOPY_INSTANCE_BUDGET
        ),
        maximum=1_000_000,
    )
    area_per_instance = (
        PHOTOREAL_FOREST_AREA_PER_INSTANCE_M2
        if photoreal_forest
        else 11_500.0
    )
    canopy_target = min(
        forest_budget,
        max(25_000 if photoreal_forest else 4_000, int(dense_area / area_per_instance)),
    )
    dense_density = canopy_target / dense_area if dense_area > 0.0 else 0.0
    for feature_index, polygon, area in dense_candidates:
        samples = min(
            4096 if photoreal_forest else 256,
            max(1, round(area * dense_density)),
        )
        for sample in range(samples):
            seed = (feature_index + 1) * 1000003 + sample * 9176
            x, y = _sample_polygon(polygon=polygon, seed=seed)
            dense_positions.append(
                gf.Vec3f(
                    float(x - origin_x),
                    float(y - origin_y),
                    elevation_grid.elevation(x, y, fallback=fallback_elevation),
                )
            )
            dense_indices.append(seed % len(dense_paths))
            scale = 0.72 + ((seed >> 7) % 85) / 100.0
            dense_scales.append(gf.Vec3f(scale, scale, scale))

    for feature_index, polygon, overlap_area in hero_candidates:
        samples = min(12, max(1, round(overlap_area / 30_000.0)))
        for sample in range(samples):
            if len(hero_positions) >= HERO_TREE_INSTANCE_LIMIT:
                break
            seed = (feature_index + 1) * 2000003 + sample * 7919
            x, y = _sample_polygon(polygon=polygon, seed=seed)
            if not (hero_xmin <= x <= hero_xmax and hero_ymin <= y <= hero_ymax):
                continue
            hero_positions.append(
                gf.Vec3f(
                    float(x - origin_x),
                    float(y - origin_y),
                    elevation_grid.elevation(x, y, fallback=fallback_elevation),
                )
            )
            hero_indices.append(seed % len(detailed_paths))
            scale = 0.62 + ((seed >> 5) % 45) / 100.0
            hero_scales.append(gf.Vec3f(scale, scale, scale))
    if dense_positions:
        dense_instancer.CreatePositionsAttr(dense_positions)
        dense_instancer.CreateProtoIndicesAttr(dense_indices)
        dense_instancer.CreateScalesAttr(dense_scales)
        _apply_training_semantics(
            dense_instancer.GetPrim(), "vegetation_dense_forest"
        )
    if hero_positions:
        hero_instancer.CreatePositionsAttr(hero_positions)
        hero_instancer.CreateProtoIndicesAttr(hero_indices)
        hero_instancer.CreateScalesAttr(hero_scales)
        _apply_training_semantics(
            hero_instancer.GetPrim(), "vegetation_hero_tree"
        )
    return len(dense_positions) + len(hero_positions)


def _append_mnh_vegetation(
    *,
    stage: Any,
    candidates: Iterable[Any],
    origin_x: int,
    origin_y: int,
    elevation_grid: _ElevationGrid,
    fallback_elevation: float,
    usd_geom: Any,
    sdf: Any,
    gf: Any,
    vegetation_assets: dict[str, list[dict[str, Any]]],
    asset_base_path: Path,
    deterministic_seed: int,
    instance_namespace: int,
    asset_lod: str,
    scene_root: str = "/Detail",
) -> int:
    """Author one tile of photoreal trees at LiDAR-MNH canopy summits."""

    items = list(candidates)
    if not items:
        return 0
    for family in ("trees", "shrubs", "understory"):
        if not vegetation_assets.get(family):
            raise RuntimeError(
                f"MNH vegetation requires photoreal {family} assets"
            )
    vegetation = usd_geom.Xform.Define(stage, f"{scene_root}/Vegetation")
    _apply_training_semantics(
        vegetation.GetPrim(), "vegetation_lidar_mnh"
    )

    tree_items = [item for item in items if float(item.height) >= 5.0]
    shrub_items = [item for item in items if float(item.height) < 5.0]
    # Understory is anchored close to observed canopy centres instead of being
    # sprayed over the tile.  It remains a separate family/instancer so the
    # renderer and annotations can control it independently.
    understory_items = tree_items[::4]

    def author_family(
        *,
        family: str,
        family_items: list[Any],
        target_height: bool,
        offset_understory: bool = False,
    ) -> int:
        if not family_items:
            return 0
        records = vegetation_assets[family]
        title = {
            "trees": "CanopyTrees",
            "shrubs": "Shrubs",
            "understory": "Understory",
        }[family]
        instancer = usd_geom.PointInstancer.Define(
            stage, f"{scene_root}/Vegetation/{title}"
        )
        prototypes = usd_geom.Scope.Define(
            stage, f"{scene_root}/Vegetation/{title}/Prototypes"
        )
        prototype_paths = []
        for index, record in enumerate(records):
            lod_paths = record.get("lod_paths")
            if not isinstance(lod_paths, dict) or asset_lod not in lod_paths:
                raise RuntimeError(
                    f"vegetation asset has no materialized {asset_lod} LOD"
                )
            asset = Path(lod_paths[asset_lod])
            prototype_path = f"{prototypes.GetPath()}/{title}{index:02d}"
            prototype = usd_geom.Xform.Define(stage, prototype_path)
            anchor = [float(value) for value in record["ground_anchor_m"]]
            prototype.AddTranslateOp().Set(
                (-anchor[0], -anchor[1], -anchor[2])
            )
            prototype.GetPrim().GetReferences().AddReference(
                _relative_asset(asset, asset_base_path)
            )
            prototype.GetPrim().SetCustomDataByKey(
                "fireviewer:source_asset", asset.name
            )
            prototype.GetPrim().SetCustomDataByKey(
                "fireviewer:asset_family", f"vegetation.{family}"
            )
            prototype.GetPrim().SetCustomDataByKey(
                "fireviewer:lod_role", "source_identity_lod_chain"
            )
            _apply_training_semantics(
                prototype.GetPrim(), f"vegetation_{family}_mnh"
            )
            prototype_paths.append(prototype_path)
        instancer.CreatePrototypesRel().SetTargets(prototype_paths)

        positions: list[Any] = []
        indices: list[int] = []
        scales: list[Any] = []
        orientations: list[Any] = []
        ids: list[int] = []
        stable_ids: list[str] = []
        footprint_radii_m: list[float] = []
        group_ids: list[str] = []
        heights: list[float] = []
        usage = [0] * len(records)
        unmatched = 0
        for item_index, item in enumerate(family_items):
            x = float(item.x)
            y = float(item.y)
            height = max(0.1, min(60.0, float(item.height)))
            source_index = int(item.source_index)
            choices: list[tuple[float, int, float]] = []
            for prototype_index, record in enumerate(records):
                if target_height:
                    native_height = float(record["native_dimensions_m"]["z"])
                    scale = height / native_height
                else:
                    minimum_scale = float(record["minimum_uniform_scale"])
                    maximum_scale = float(record["maximum_uniform_scale"])
                    scale = minimum_scale + (
                        (
                            source_index * 1_103_515_245
                            + deterministic_seed
                            + prototype_index * 97
                        )
                        & 0xFFFF
                    ) / 65535.0 * (maximum_scale - minimum_scale)
                if not (
                    float(record["minimum_uniform_scale"])
                    <= scale
                    <= float(record["maximum_uniform_scale"])
                ):
                    continue
                choices.append((usage[prototype_index] * 0.05, prototype_index, scale))
            if not choices:
                # Keep real vegetation assets when a LiDAR observation falls
                # just outside every asset's declared scale envelope. Select
                # the closest native height and clamp only the uniform scale;
                # never drop the observed tree or replace it with geometry.
                for prototype_index, record in enumerate(records):
                    native_height = float(record["native_dimensions_m"]["z"])
                    minimum_scale = float(record["minimum_uniform_scale"])
                    maximum_scale = float(record["maximum_uniform_scale"])
                    requested_scale = height / native_height
                    scale = max(
                        minimum_scale,
                        min(maximum_scale, requested_scale),
                    )
                    represented_height = max(0.1, native_height * scale)
                    height_error = abs(
                        math.log(max(1.0e-6, height / represented_height))
                    )
                    choices.append(
                        (
                            height_error + usage[prototype_index] * 0.05,
                            prototype_index,
                            scale,
                        )
                    )
            _score, selected_index, uniform_scale = min(
                choices,
                key=lambda value: (
                    value[0],
                    (
                        value[1] * 1_103_515_245
                        + source_index
                        + deterministic_seed
                    )
                    & 0xFFFFFFFF,
                ),
            )
            if offset_understory:
                offset_angle = (
                    ((source_index * 2_654_435_761) & 0xFFFF)
                    / 65535.0
                    * math.tau
                )
                x += math.cos(offset_angle) * 1.25
                y += math.sin(offset_angle) * 1.25
            positions.append(
                gf.Vec3f(
                    float(x - origin_x),
                    float(y - origin_y),
                    elevation_grid.elevation(x, y, fallback=fallback_elevation),
                )
            )
            indices.append(selected_index)
            scales.append(gf.Vec3f(uniform_scale, uniform_scale, uniform_scale))
            angle = (
                ((source_index * 2_654_435_761 + deterministic_seed) & 0xFFFF)
                / 65535.0
                * math.tau
            )
            orientations.append(
                gf.Quath(
                    float(math.cos(angle * 0.5)),
                    gf.Vec3h(0.0, 0.0, float(math.sin(angle * 0.5))),
                )
            )
            numeric_id = _stable_instance_id(
                tile_namespace=instance_namespace,
                family=family,
                local_index=item_index,
            )
            ids.append(numeric_id)
            stable_ids.append(
                f"tile-{instance_namespace}:{family}:{numeric_id}"
            )
            selected_record = records[selected_index]
            native_dimensions = selected_record["native_dimensions_m"]
            footprint_radii_m.append(
                max(
                    0.1,
                    max(
                        float(native_dimensions["x"]),
                        float(native_dimensions["y"]),
                    )
                    * uniform_scale
                    * 0.5,
                )
            )
            local_x = float(x - origin_x)
            local_y = float(y - origin_y)
            group_ids.append(
                "forest:"
                f"{math.floor(local_x / 25.0)}:"
                f"{math.floor(local_y / 25.0)}"
            )
            heights.append(height)
            usage[selected_index] += 1
        if unmatched:
            raise RuntimeError(
                f"{unmatched} {family} observations have no height-compatible "
                "photoreal asset"
            )
        if not positions:
            return 0
        instancer.CreatePositionsAttr(positions)
        instancer.CreateProtoIndicesAttr(indices)
        instancer.CreateScalesAttr(scales)
        instancer.CreateOrientationsAttr(orientations)
        instancer.CreateIdsAttr(ids)
        _author_instance_identity_primvars(
            instancer=instancer,
            stable_ids=stable_ids,
            footprint_radii_m=footprint_radii_m,
            group_ids=group_ids,
            usd_geom=usd_geom,
            sdf=sdf,
        )
        _apply_training_semantics(
            instancer.GetPrim(), f"vegetation_{family}_mnh"
        )
        instancer.GetPrim().SetCustomDataByKey(
            "fireviewer:instance_count", len(positions)
        )
        instancer.GetPrim().SetCustomDataByKey(
            "fireviewer:height_min_metres", float(min(heights))
        )
        instancer.GetPrim().SetCustomDataByKey(
            "fireviewer:height_max_metres", float(max(heights))
        )
        fast_editor_preview = os.getenv(
            "FW_SDG_FAST_EDITOR_PREVIEW", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not fast_editor_preview and len(positions) >= len(records) * 4:
            dominance = max(usage) / float(len(positions))
            if dominance > 0.25 + 1e-9:
                raise RuntimeError(
                    f"one {family} prototype dominates {dominance:.1%} of the tile"
                )
        return len(positions)

    return (
        author_family(family="trees", family_items=tree_items, target_height=True)
        + author_family(family="shrubs", family_items=shrub_items, target_height=True)
        + author_family(
            family="understory",
            family_items=understory_items,
            target_height=False,
            offset_understory=True,
        )
    )


def _write_detail_payload(
    *,
    detail_path: Path,
    tile: dict[str, str],
    building_features: list[dict[str, Any]],
    road_features: list[dict[str, Any]],
    hydrology_features: list[dict[str, Any]],
    canopy_candidates: list[Any],
    asset_library: dict[str, dict[str, list[dict[str, Any]]]],
    origin_x: int,
    origin_y: int,
    elevation_grid: _ElevationGrid,
    surface_grid: _ElevationGrid,
    height_grid: _ElevationGrid,
    fallback_elevation: float,
    usd: Any,
    usd_geom: Any,
    usd_shade: Any,
    sdf: Any,
    gf: Any,
    deterministic_seed: int,
    instance_namespace: int,
    detail_level: str,
) -> dict[str, int]:
    """Write one streamable kilometre of full visual/semantic scene detail."""

    detail_path.parent.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = (
        int(tile[key]) for key in ("xmin", "ymin", "xmax", "ymax")
    )
    bounds = (float(xmin), float(ymin), float(xmax), float(ymax))
    stage = usd.Stage.CreateNew(str(detail_path))
    usd_geom.SetStageMetersPerUnit(stage, 1.0)
    usd_geom.SetStageUpAxis(stage, usd_geom.Tokens.z)
    detail = usd_geom.Xform.Define(stage, "/Detail")
    stage.SetDefaultPrim(detail.GetPrim())
    detail.GetPrim().SetCustomDataByKey("fireviewer:tile_ref", tile["tile_ref"])
    detail.GetPrim().SetCustomDataByKey(
        "fireviewer:epsg2154_bounds", f"{xmin},{ymin},{xmax},{ymax}"
    )
    if detail_level not in {"HERO", "MID", "FAR"}:
        raise ValueError(f"unsupported detail level: {detail_level}")
    detail.GetPrim().SetCustomDataByKey(
        "fireviewer:role",
        f"camera_streamed_photoreal_detail_{detail_level.lower()}",
    )
    detail.GetPrim().SetCustomDataByKey(
        "fireviewer:detail_level", detail_level
    )
    detail.GetPrim().SetCustomDataByKey(
        "fireviewer:instance_namespace", int(instance_namespace)
    )
    for name in (
        "Materials",
        "Buildings",
        "Roads",
        "Hydrology",
        "Vegetation",
        "Semantics",
    ):
        scope = usd_geom.Scope.Define(stage, f"/Detail/{name}")
        scope.GetPrim().SetCustomDataByKey("fireviewer:layer", name.lower())

    selected_buildings = _building_features_for_tile(
        building_features, bounds=bounds
    )
    building_count = (
        _append_buildings(
            stage=stage,
            features=selected_buildings,
            origin_x=origin_x,
            origin_y=origin_y,
            elevation_grid=elevation_grid,
            surface_grid=surface_grid,
            height_grid=height_grid,
            fallback_elevation=fallback_elevation,
            usd_geom=usd_geom,
            usd_shade=usd_shade,
            sdf=sdf,
            gf=gf,
            roof_texture=None,
            asset_base_path=detail_path.parent,
            building_assets=asset_library.get("buildings", {}),
            hero_bounds=(xmin, ymin, xmax, ymax),
            overview_visible=True,
            scene_root="/Detail",
            instance_namespace=instance_namespace,
            asset_lod=detail_level,
        )
        if selected_buildings
        else 0
    )
    # Route vectors stay in the composition contract.  Do not create a mesh
    # that duplicates or masks the road pixels visible on the orthophoto.
    road_count = 0
    hydro_surface_count = _append_hydro_surfaces(
        stage=stage,
        features=hydrology_features,
        origin_x=origin_x,
        origin_y=origin_y,
        elevation_grid=elevation_grid,
        fallback_elevation=fallback_elevation,
        usd_geom=usd_geom,
        gf=gf,
        scene_root="/Detail",
        clip_bounds=bounds,
        render_visible=False,
    )
    hydro_line_count = _append_lines(
        stage=stage,
        root="/Detail/Hydrology",
        features=hydrology_features,
        origin_x=origin_x,
        origin_y=origin_y,
        elevation_grid=elevation_grid,
        fallback_elevation=fallback_elevation,
        usd_geom=usd_geom,
        gf=gf,
        semantic="watercourse",
        color=(0.08, 0.25, 0.42),
        default_width=2.0,
        clip_bounds=bounds,
        render_visible=False,
    )
    vegetation_count = _append_mnh_vegetation(
        stage=stage,
        candidates=canopy_candidates,
        origin_x=origin_x,
        origin_y=origin_y,
        elevation_grid=elevation_grid,
        fallback_elevation=fallback_elevation,
        usd_geom=usd_geom,
        sdf=sdf,
        gf=gf,
        vegetation_assets=asset_library.get("vegetation", {}),
        asset_base_path=detail_path.parent,
        deterministic_seed=deterministic_seed,
        instance_namespace=instance_namespace,
        asset_lod=detail_level,
        scene_root="/Detail",
    )
    counts = {
        "buildings": building_count,
        "roads": road_count,
        "hydrology": hydro_surface_count + hydro_line_count,
        "vegetation": vegetation_count,
    }
    detail.GetPrim().SetCustomDataByKey(
        "fireviewer:layer_counts",
        json.dumps(counts, sort_keys=True, separators=(",", ":")),
    )
    detail.GetPrim().SetCustomDataByKey(
        "fireviewer:road_visualization",
        json.dumps(
            {
                "visible_representation": (
                    "orthophoto_derived_terrain_material"
                ),
                "geometry_authoring": "disabled",
                "route_vector_feature_count": len(road_features),
                "asset_dependencies": [],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    stage.GetRootLayer().Save()
    return counts


def build_zone(*, catalog_root: Path, workspace_root: Path, zone_id: str) -> dict[str, Any]:
    """Build and receipt one native USD package from verified Z16-like inputs."""

    zone_root = (workspace_root / "zone-scenes" / zone_id).resolve()
    if not _is_below(workspace_root / "zone-scenes", zone_root):
        raise ValueError("zone workspace escapes zone-scenes root")
    lock_path = zone_root / "source-lock.json"
    lock = _load_source_lock(zone_root)
    vector_paths = _vector_paths(zone_root, lock)
    _manifest_path, manifest = _zone_manifest(catalog_root, zone_id)
    zone = manifest.get("zone")
    if not isinstance(zone, dict):
        raise ValueError("zone manifest is missing zone metadata")
    rows = _zone_rows(catalog_root, zone_id)
    if len(rows) != 400:
        raise ValueError("native builder requires exactly 400 tile rows")
    origin_x = int(zone["center_x"])
    origin_y = int(zone["center_y"])
    source_profile = str(lock.get("source_profile", "full"))
    if source_profile == "light":
        entries = lock.get("entries")
        if not isinstance(entries, list):
            raise ValueError("light source lock entries are malformed")
        terrain_entry = next(
            (
                entry
                for entry in entries
                if isinstance(entry, dict) and entry.get("dataset") == "terrain_lod3"
            ),
            None,
        )
        if not isinstance(terrain_entry, dict):
            raise RuntimeError("light source lock has no terrain_lod3 MNT")
        mosaic_values = _read_mnt_values(_entry_path(zone_root, terrain_entry))
        ortho_mosaic = _light_orthophoto_mosaic(
            zone_root=zone_root, lock=lock, zone=zone
        )
        lod0_textures = _light_lod0_textures(zone_root=zone_root, lock=lock)
        hero_bounds = _light_hero_bounds(lock)
        source_index: dict[tuple[str, str], Path] = {}
    else:
        source_index = _source_index(zone_root, lock)
        for row in rows:
            for dataset in ("lidar", "mnt", "ortho50", "ortho20", "mns", "mnh"):
                if (row["tile_ref"], dataset) not in source_index:
                    raise RuntimeError(f"missing locked {dataset} for {row['tile_ref']}")
        mosaic_values = None
        ortho_mosaic = None
        lod0_textures = {}
        hero_bounds = (int(zone["xmin"]), int(zone["ymin"]), int(zone["xmax"]), int(zone["ymax"]))

    lidar_evidence: dict[str, Any] | None = None
    lidar_evidence_path: Path | None = None
    if source_profile != "light":
        configured_evidence = os.getenv(
            "FW_SDG_LIDAR_EVIDENCE_RECEIPT", ""
        ).strip()
        if not configured_evidence:
            raise RuntimeError(
                "full photoreal build requires the PDAL LiDAR quality receipt"
            )
        lidar_evidence_path = Path(configured_evidence).expanduser().resolve()
        if (
            not _is_below(zone_root, lidar_evidence_path)
            or not lidar_evidence_path.is_file()
        ):
            raise RuntimeError("PDAL LiDAR quality receipt is outside the zone build")
        lidar_evidence = _read_json(
            lidar_evidence_path, label="PDAL LiDAR quality receipt"
        )

    # Import pxr only after Isaac's SimulationApp exists.  Direct pxr imports
    # before Kit initialization are not a supported native Isaac workflow.
    from isaacsim.simulation_app import SimulationApp

    SimulationApp({"headless": True})
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    vector_features = {
        name: list(_geojson_features(paths))
        for name, paths in vector_paths.items()
    }
    road_source_line_count = sum(
        len(_feature_lines(feature.get("geometry")))
        for feature in vector_features["roads"]
    )
    if road_source_line_count < 1:
        raise RuntimeError("locked route vector source has no usable line")
    build_root = zone_root / "build"
    payload_root = build_root / "payloads"
    detail_root = build_root / "details"
    aggregate_root = build_root / "aggregates"
    metadata_root = build_root / "metadata"
    texture_root = build_root / "textures"
    for directory in (
        payload_root,
        detail_root,
        aggregate_root,
        metadata_root,
        texture_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    flow_asset_lock = _lock_flow_preset(build_root=build_root)
    flow_asset = build_root / str(flow_asset_lock["packaged_path"])
    asset_library, photoreal_asset_lock = (
        _photoreal_asset_contract(workspace_root=workspace_root)
    )
    if source_profile == "light" and photoreal_asset_lock is not None:
        raise RuntimeError(
            "photoreal pod scene build requires the full LiDAR source profile"
        )
    continuous_ortho_path: Path | None = None
    visual_terrain_path: Path | None = None
    continuous_terrain_vertices = 0
    if source_profile == "light":
        external_assets, external_asset_locks = _package_light_external_assets(
            zone_root=zone_root, build_root=build_root, lock=lock
        )
        # Saplings are locked for the future close-camera vegetation pass, but
        # are intentionally not converted into a misleading whole-zone forest.
        tree_usds: list[Path] = []
        roof_texture = external_assets["polyhaven-red-slate-roof-diffuse"]
    else:
        external_asset_locks = []
        if photoreal_asset_lock is None:
            raise RuntimeError(
                "full pod scene requires the materialized photoreal asset library"
            )
        tree_usds = []
        roof_texture = None
    if source_profile == "light":
        if mosaic_values is None or ortho_mosaic is None:
            raise RuntimeError("light terrain or orthophoto context was not loaded")
        continuous_ortho_path = _write_continuous_orthophoto(
            mosaic=ortho_mosaic,
            path=texture_root / f"{zone_id}_orthophoto_context_5m.jpg",
        )
        visual_terrain_path = build_root / "terrain" / f"{zone_id}_visual_terrain.usdc"
        continuous_terrain_vertices = _write_continuous_terrain(
            terrain_path=visual_terrain_path,
            values=mosaic_values,
            ortho_path=continuous_ortho_path,
            zone=zone,
            origin_x=origin_x,
            origin_y=origin_y,
            usd=Usd,
            usd_geom=UsdGeom,
            usd_shade=UsdShade,
            sdf=Sdf,
            gf=Gf,
        )
    fallback_elevation = _as_float(zone.get("elev_min"), default=0.0)
    # 513 samples per kilometre preserve roughly two-metre placement fidelity
    # while keeping all three grids below ~1.3 GiB for a 400-tile zone.  The
    # previous 129² nearest-neighbour cache displaced close buildings and trees
    # by up to an eight-metre cell on steep ground.
    elevation_grid = _ElevationGrid(samples=513, label="MNT")
    surface_grid = _ElevationGrid(samples=513, label="MNS")
    height_grid = _ElevationGrid(samples=513, label="MNH")
    payloads: list[Path] = []
    detail_payloads_by_level: dict[str, list[Path]] = {
        "HERO": [],
        "MID": [],
        "FAR": [],
    }
    terrain_points = {
        "lod0": 0,
        "lod1": 0,
        "lod2": 0,
        "lod3": 0,
        "collision": 0,
    }
    lidar_lod0_tiles = (
        _review_camera_lod0_tiles(rows) if source_profile != "light" else set()
    )
    canopy_by_tile: dict[str, list[Any]] = {}
    forest_area_by_tile: dict[str, float] = {}
    height_product_quality: dict[str, dict[str, float]] = {}
    if source_profile != "light":
        from fireviewer_sdg.canopy import detect_canopy_candidates

    for row in rows:
        tile_ref = row["tile_ref"]
        payload = _payload_paths(build_root, tile_ref)
        if source_profile == "light":
            if mosaic_values is None or ortho_mosaic is None:
                raise RuntimeError("light terrain or orthophoto context was not loaded")
            terrain_values = _mosaic_tile_values(
                values=mosaic_values, tile=row, zone=zone
            )
            # The verified LOD0 source remains locked for a later close-camera
            # pass.  It is not pasted into every 1 km render mesh: the overview
            # has exactly one continuous orthophoto surface.
            _ = lod0_textures.get(tile_ref)
            ortho_path = None
            ortho_lod0_path = None
            surface_values = terrain_values
            height_values = np.zeros_like(terrain_values, dtype=np.float32)
        else:
            terrain_values = _read_raster_values(
                source_index[(tile_ref, "mnt")], label=f"{tile_ref} MNT"
            )
            surface_values = _read_raster_values(
                source_index[(tile_ref, "mns")], label=f"{tile_ref} MNS"
            )
            height_values = _read_raster_values(
                source_index[(tile_ref, "mnh")], label=f"{tile_ref} MNH"
            )
            height_product_quality[tile_ref] = _validate_height_products(
                mnt=terrain_values,
                mns=surface_values,
                mnh=height_values,
                label=tile_ref,
            )
            ortho_path = source_index[(tile_ref, "ortho50")]
            ortho_lod0_path = (
                source_index.get((tile_ref, "ortho20"))
                if tile_ref in lidar_lod0_tiles
                else None
            )
            xmin, ymin, xmax, ymax = (
                int(row[key]) for key in ("xmin", "ymin", "xmax", "ymax")
            )
            surface_grid.add(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                values=surface_values,
            )
            height_grid.add(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                values=height_values,
            )
            forest_polygons = _forest_polygons_for_tile(
                vector_features["vegetation"],
                bounds=(float(xmin), float(ymin), float(xmax), float(ymax)),
            )
            forest_area = sum(_polygon_area(polygon) for polygon in forest_polygons)
            canopy_values = height_values
            candidates = detect_canopy_candidates(
                values=canopy_values,
                bounds=(float(xmin), float(ymin), float(xmax), float(ymax)),
                forest_polygons=forest_polygons,
                minimum_height_metres=MIN_CANOPY_HEIGHT_METRES,
                nms_radius_metres=CANOPY_NMS_RADIUS_METRES,
            )
            if forest_area >= 10_000.0 and not candidates:
                raise RuntimeError(
                    f"{tile_ref} has {forest_area:.0f} m2 of source forest but "
                    "no MNH canopy candidate"
                )
            canopy_by_tile[tile_ref] = candidates
            forest_area_by_tile[tile_ref] = forest_area
        reuse_complete_terrain = os.getenv(
            "FW_SDG_REUSE_COMPLETE_TERRAIN", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if (
            reuse_complete_terrain
            and payload.is_file()
            and payload.stat().st_size >= 2_000_000
        ):
            elevation_grid.add(
                xmin=int(row["xmin"]),
                ymin=int(row["ymin"]),
                xmax=int(row["xmax"]),
                ymax=int(row["ymax"]),
                values=terrain_values,
            )
            terrain = {
                "TerrainLOD0": 0,
                "TerrainLOD1": TERRAIN_LOD1_SAMPLES**2,
                "TerrainLOD2": 0,
                "TerrainLOD3": 0,
                "Collision": TERRAIN_LOD1_SAMPLES**2,
            }
        else:
            terrain = _write_payload(
                payload_path=payload,
                tile=row,
                values=terrain_values,
                ortho_path=ortho_path,
                ortho_lod0_path=ortho_lod0_path,
                origin_x=origin_x,
                origin_y=origin_y,
                usd=Usd,
                usd_geom=UsdGeom,
                usd_shade=UsdShade,
                sdf=Sdf,
                gf=Gf,
                elevation_grid=elevation_grid,
                render_visible=source_profile != "light",
            )
        terrain_points["lod0"] += terrain.get("TerrainLOD0", 0)
        terrain_points["lod1"] += terrain.get("TerrainLOD1", 0)
        terrain_points["lod2"] += terrain.get("TerrainLOD2", 0)
        terrain_points["lod3"] += terrain.get("TerrainLOD3", 0)
        terrain_points["collision"] += terrain["Collision"]
        payloads.append(payload)

    layer_counts = {"buildings": 0, "roads": 0, "hydrology": 0, "vegetation": 0}
    detail_counts_by_tile: dict[str, dict[str, dict[str, int]]] = {}
    detail_paths_by_tile: dict[str, dict[str, Path]] = {}
    selected_canopy_count = 0
    canopy_candidate_count = sum(len(items) for items in canopy_by_tile.values())
    if source_profile != "light":
        from fireviewer_sdg.canopy import select_canopy_instances

        forest_budget = _positive_int_environment(
            "FW_SDG_FOREST_INSTANCE_BUDGET",
            PHOTOREAL_FOREST_INSTANCE_BUDGET,
            maximum=10_000_000,
        )
        target = min(forest_budget, canopy_candidate_count)
        allocations: dict[str, int] = {tile_ref: 0 for tile_ref in canopy_by_tile}
        if canopy_candidate_count:
            weighted = {
                tile_ref: target * len(items) / canopy_candidate_count
                for tile_ref, items in canopy_by_tile.items()
            }
            allocations = {
                tile_ref: int(math.floor(value))
                for tile_ref, value in weighted.items()
            }
            remainder = target - sum(allocations.values())
            for tile_ref in sorted(
                weighted,
                key=lambda key: (
                    -(weighted[key] - allocations[key]),
                    key,
                ),
            )[:remainder]:
                allocations[tile_ref] += 1
        zone_seed = sum(
            (index + 1) * ord(character)
            for index, character in enumerate(zone_id)
        )
        for tile_index, row in enumerate(rows):
            tile_ref = row["tile_ref"]
            selected = select_canopy_instances(
                canopy_by_tile.get(tile_ref, []),
                budget=allocations.get(tile_ref, 0),
                deterministic_seed=zone_seed + tile_index,
            )
            selected_canopy_count += len(selected)
            level_counts: dict[str, dict[str, int]] = {}
            level_paths: dict[str, Path] = {}
            detail_levels = ("HERO",) if os.getenv(
                "FW_SDG_FAST_EDITOR_PREVIEW", ""
            ).strip().lower() in {"1", "true", "yes", "on"} else (
                "HERO",
                "MID",
                "FAR",
            )
            for detail_level in detail_levels:
                detail_path = _detail_payload_path(
                    build_root, tile_ref, detail_level
                )
                counts = _write_detail_payload(
                    detail_path=detail_path,
                    tile=row,
                    building_features=vector_features["buildings"],
                    road_features=vector_features["roads"],
                    hydrology_features=vector_features["hydrology"],
                    canopy_candidates=(
                        selected
                        if len(detail_levels) == 1
                        else _detail_lod_candidates(selected, detail_level)
                    ),
                    asset_library=asset_library,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    elevation_grid=elevation_grid,
                    surface_grid=surface_grid,
                    height_grid=height_grid,
                    fallback_elevation=fallback_elevation,
                    usd=Usd,
                    usd_geom=UsdGeom,
                    usd_shade=UsdShade,
                    sdf=Sdf,
                    gf=Gf,
                    deterministic_seed=zone_seed + tile_index,
                    instance_namespace=tile_index + 1,
                    detail_level=detail_level,
                )
                if detail_level == "HERO":
                    for layer, count in counts.items():
                        layer_counts[layer] += count
                level_counts[detail_level] = counts
                level_paths[detail_level] = detail_path
                detail_payloads_by_level[detail_level].append(detail_path)
            if len(detail_levels) == 1:
                source_level = detail_levels[0]
                for alias_level in ("HERO", "MID", "FAR"):
                    if alias_level == source_level:
                        continue
                    level_counts[alias_level] = level_counts[source_level]
                    level_paths[alias_level] = level_paths[source_level]
                    detail_payloads_by_level[alias_level].append(
                        level_paths[source_level]
                    )
            detail_counts_by_tile[tile_ref] = level_counts
            detail_paths_by_tile[tile_ref] = level_paths

    grouped: dict[
        tuple[int, int],
        list[tuple[Path, dict[str, Path] | None, dict[str, str]]],
    ] = {}
    for row, payload in zip(rows, payloads, strict=True):
        key = ((int(row["xmin"]) - int(zone["xmin"])) // 5000, (int(row["ymin"]) - int(zone["ymin"])) // 5000)
        grouped.setdefault(key, []).append(
            (payload, detail_paths_by_tile.get(row["tile_ref"]), row)
        )
    aggregate_paths: list[Path] = []
    for (column, row), members in sorted(grouped.items()):
        aggregate = aggregate_root / f"aggregate_5km_{column}_{row}.usdc"
        _write_aggregate(
            aggregate_path=aggregate,
            payload_records=members,
            origin_x=origin_x,
            origin_y=origin_y,
            usd=Usd,
            usd_geom=UsdGeom,
        )
        aggregate_paths.append(aggregate)

    root_path = build_root / f"{zone_id}_root.usdc"
    cameras_path = build_root / "review-cameras.usda"
    origin_elevation = elevation_grid.elevation(
        float(origin_x), float(origin_y), fallback=fallback_elevation
    )
    cameras = _write_cameras(
        cameras_path=cameras_path,
        origin_x=origin_x,
        origin_y=origin_y,
        elevation=origin_elevation,
        usd=Usd,
        usd_geom=UsdGeom,
        gf=Gf,
    )
    _write_root(root_path=root_path, aggregate_paths=aggregate_paths, visual_terrain_path=visual_terrain_path, cameras_path=cameras_path, zone=zone, source_lock=lock_path, flow_asset=flow_asset, flow_asset_lock=flow_asset_lock, origin_x=origin_x, origin_y=origin_y, flow_elevation=origin_elevation, usd=Usd, usd_geom=UsdGeom, usd_lux=UsdLux)

    if source_profile == "light":
        root_stage = Usd.Stage.Open(str(root_path))
        if root_stage is None:
            raise RuntimeError("written root USD cannot be reopened")
        building_count = _append_buildings(
            stage=root_stage,
            features=vector_features["buildings"],
            origin_x=origin_x,
            origin_y=origin_y,
            elevation_grid=elevation_grid,
            surface_grid=elevation_grid,
            height_grid=elevation_grid,
            fallback_elevation=fallback_elevation,
            usd_geom=UsdGeom,
            usd_shade=UsdShade,
            sdf=Sdf,
            gf=Gf,
            roof_texture=roof_texture,
            asset_base_path=root_path.parent,
            building_assets=None,
            hero_bounds=hero_bounds,
            overview_visible=False,
        )
        road_count = 0
        hydro_count = _append_hydro_surfaces(
            stage=root_stage,
            features=vector_features["hydrology"],
            origin_x=origin_x,
            origin_y=origin_y,
            elevation_grid=elevation_grid,
            fallback_elevation=fallback_elevation,
            usd_geom=UsdGeom,
            gf=Gf,
        ) + _append_lines(
            stage=root_stage,
            root="/World/Hydrology",
            features=vector_features["hydrology"],
            origin_x=origin_x,
            origin_y=origin_y,
            elevation_grid=elevation_grid,
            fallback_elevation=fallback_elevation,
            usd_geom=UsdGeom,
            gf=Gf,
            semantic="watercourse",
            color=(0.08, 0.25, 0.42),
            default_width=2.0,
        )
        vegetation_count = _append_vegetation(
            stage=root_stage,
            features=vector_features["vegetation"],
            origin_x=origin_x,
            origin_y=origin_y,
            elevation_grid=elevation_grid,
            fallback_elevation=fallback_elevation,
            usd_geom=UsdGeom,
            gf=Gf,
            hero_bounds=hero_bounds,
            tree_assets=tree_usds,
            asset_base_path=root_path.parent,
            assets_z_up=False,
            overview_visible=False,
        )
        layer_counts = {
            "buildings": building_count,
            "roads": road_count,
            "hydrology": hydro_count,
            "vegetation": vegetation_count,
        }
        root_stage.GetRootLayer().Save()

    georeference_path = metadata_root / "georeference.json"
    write_json(georeference_path, {"zone_id": zone_id, "crs": "EPSG:2154", "vertical_datum": "IGN69", "local_origin_epsg2154": [origin_x, origin_y, 0], "coordinate_convention": COORDINATE_CONVENTION})
    sources_path = metadata_root / "sources-lock.json"
    write_json(sources_path, {"source_lock": "../../source-lock.json", "source_lock_sha256": sha256(lock_path), "vector_source_layers": VECTOR_SOURCE_LAYERS})
    asset_lock_path = metadata_root / "asset-lock.json"
    packaged_tree_locks = (
        [
            {
                "id": f"converted-{tree_path.stem}",
                "source": "locked Poly Haven glTF",
                "packaged_path": tree_path.relative_to(build_root).as_posix(),
                "packaged_sha256": sha256(tree_path),
            }
            for tree_path in tree_usds
        ]
        if photoreal_asset_lock is None
        else []
    )
    shared_photoreal_locks = (
        [
            {
                "id": "shared-materialized-simready-environment",
                **photoreal_asset_lock,
            }
        ]
        if photoreal_asset_lock is not None
        else []
    )
    write_json(asset_lock_path, {"zone_id": zone_id, "assets": [flow_asset_lock, *external_asset_locks, *packaged_tree_locks, *shared_photoreal_locks]})
    receipt_path = build_root / "build-receipt.json"
    lidar_evidence_record = (
        {
            "path": lidar_evidence_path.relative_to(zone_root).as_posix(),
            "sha256": sha256(lidar_evidence_path),
            "source_count": len(lidar_evidence.get("sources", [])),
        }
        if lidar_evidence_path is not None and lidar_evidence is not None
        else None
    )
    tile_coverage = [
        {
            "tile_ref": row["tile_ref"],
            "terrain_payload": payload.relative_to(zone_root).as_posix(),
            "detail_payload": (
                detail_paths_by_tile[row["tile_ref"]]["HERO"]
                .relative_to(zone_root)
                .as_posix()
                if row["tile_ref"] in detail_paths_by_tile
                else None
            ),
            "detail_lods": (
                {
                    level: path.relative_to(zone_root).as_posix()
                    for level, path in detail_paths_by_tile[
                        row["tile_ref"]
                    ].items()
                }
                if row["tile_ref"] in detail_paths_by_tile
                else {}
            ),
            "terrain_lods": (
                ["LOD0", "LOD1", "LOD2", "LOD3"]
                if row["tile_ref"] in lidar_lod0_tiles
                else ["LOD1", "LOD2", "LOD3"]
            ),
            "collision_lods": ["NEAR", "FAR"],
            "detail_counts": detail_counts_by_tile.get(
                row["tile_ref"], {}
            ).get("HERO", {}),
            "detail_lod_counts": detail_counts_by_tile.get(
                row["tile_ref"], {}
            ),
            "instance_namespace": (
                tile_index + 1 if source_profile != "light" else None
            ),
            "forest_area_m2": round(
                forest_area_by_tile.get(row["tile_ref"], 0.0), 3
            ),
        }
        for tile_index, (row, payload) in enumerate(
            zip(rows, payloads, strict=True)
        )
    ]
    receipt = {
        "schema_version": 2,
        "zone_id": zone_id,
        "built_at": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
        "coordinate_convention": COORDINATE_CONVENTION,
        "root_usd": {"path": root_path.relative_to(zone_root).as_posix(), "sha256": sha256(root_path)},
        "payloads": [{"path": path.relative_to(zone_root).as_posix(), "sha256": sha256(path)} for path in payloads],
        "detail_payloads": [
            {
                "path": path.relative_to(zone_root).as_posix(),
                "sha256": sha256(path),
            }
            for path in detail_payloads_by_level["HERO"]
        ],
        "detail_mid_payloads": [
            {
                "path": path.relative_to(zone_root).as_posix(),
                "sha256": sha256(path),
            }
            for path in detail_payloads_by_level["MID"]
        ],
        "detail_far_payloads": [
            {
                "path": path.relative_to(zone_root).as_posix(),
                "sha256": sha256(path),
            }
            for path in detail_payloads_by_level["FAR"]
        ],
        "tile_coverage": tile_coverage,
        "aggregates_5km": [{"path": path.relative_to(zone_root).as_posix(), "sha256": sha256(path)} for path in aggregate_paths],
        "cameras": {"path": cameras_path.relative_to(zone_root).as_posix(), "sha256": sha256(cameras_path), "count": len(cameras)},
        "source_lock": {"path": lock_path.relative_to(zone_root).as_posix(), "sha256": sha256(lock_path)},
        "lidar_quality": lidar_evidence_record,
        "source_profile": source_profile,
        "asset_lock": {"path": asset_lock_path.relative_to(zone_root).as_posix(), "sha256": sha256(asset_lock_path), "assets": [flow_asset_lock, *external_asset_locks, *packaged_tree_locks, *shared_photoreal_locks]},
        "fire_simulation_status": "blocked_pending_editor_review",
        "layers": {
            "terrain": {
                "prim_count": 1 if visual_terrain_path is not None else len(payloads),
                "visual_surface": (
                    {
                        "path": visual_terrain_path.relative_to(zone_root).as_posix(),
                        "sha256": sha256(visual_terrain_path),
                        "vertices": continuous_terrain_vertices,
                    }
                    if visual_terrain_path is not None
                    else None
                ),
                "non_render_tile_payload_count": len(payloads) if visual_terrain_path is not None else 0,
                "lod0_vertices": terrain_points["lod0"],
                "lod1_vertices": terrain_points["lod1"],
                "lod2_vertices": terrain_points["lod2"],
                "lod3_vertices": terrain_points["lod3"],
                "height_product_p95_residual_max_metres": (
                    max(
                        (
                            value["p95_absolute_residual_metres"]
                            for value in height_product_quality.values()
                        ),
                        default=0.0,
                    )
                ),
            },
            "imagery": {
                "prim_count": 1 if continuous_ortho_path is not None else len(payloads),
                "orthophoto_product": (
                    "BD ORTHO 20cm review LOD0 + 50cm full-zone context"
                    if source_profile != "light"
                    else "IGN BD ORTHO WMS: 2 m full zone + 20 cm camera LOD0"
                ),
                "continuous_texture": (
                    {
                        "path": continuous_ortho_path.relative_to(zone_root).as_posix(),
                        "sha256": sha256(continuous_ortho_path),
                    }
                    if continuous_ortho_path is not None
                    else None
                ),
            },
            "hydrology": {"prim_count": layer_counts["hydrology"]},
            "roads": {
                "prim_count": layer_counts["roads"],
                "source_feature_count": len(vector_features["roads"]),
                "source_line_count": road_source_line_count,
                "visible_representation": (
                    "orthophoto_derived_terrain_material"
                ),
                "geometry_authoring": "disabled",
                "asset_dependencies": [],
            },
            "buildings": {"prim_count": layer_counts["buildings"]},
            "vegetation": {
                "prim_count": layer_counts["vegetation"],
                "mnh_candidate_count": canopy_candidate_count,
                "mnh_selected_tree_or_shrub_count": selected_canopy_count,
                "instance_budget": (
                    forest_budget if source_profile != "light" else None
                ),
                "overview": (
                    "orthophoto_canopy"
                    if source_profile == "light"
                    else "camera_streamed_photoreal_geometry"
                ),
            },
            "collisions": {
                "prim_count": len(payloads),
                "vertices": terrain_points["collision"],
                "levels": ["NEAR", "FAR"],
                "near_spacing_m": 4.0,
                "far_spacing_m": 32.0,
            },
            "semantics": {
                "prim_count": len(payloads) * 3 + sum(layer_counts.values())
            },
            "detail_streaming": {
                "prim_count": len(detail_payloads_by_level["HERO"]),
                "levels": ["HERO", "MID", "FAR"],
                "delivery": (
                    "far_all_tiles_mid_visible_hero_near_camera_working_set"
                ),
                "terrain_is_never_unloaded_for_detail_streaming": True,
            },
            "fire_and_smoke": {
                "prim_count": 1,
                "asset": flow_asset_lock["packaged_path"],
                "payload_composed": False,
                "default_visibility": "uncomposed_until_editor_review_acceptance",
            },
        },
    }
    write_json(receipt_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build one native Isaac/OpenUSD geographic scene")
    parser.add_argument("--catalog-root", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--zone", required=True)
    args = parser.parse_args(argv)
    catalog_root = _absolute_directory(args.catalog_root, label="catalog root")
    workspace_root = _absolute_directory(args.workspace_root, label="workspace root")
    zone_root = (workspace_root / "zone-scenes" / args.zone).resolve()
    if not _is_below(workspace_root / "zone-scenes", zone_root):
        raise ValueError("zone workspace escapes zone-scenes root")
    with _exclusive_zone_build(zone_root):
        receipt = build_zone(
            catalog_root=catalog_root,
            workspace_root=workspace_root,
            zone_id=args.zone,
        )
    print(json.dumps({"zone_id": args.zone, "root_usd": receipt["root_usd"], "payloads": len(receipt["payloads"]), "layers": receipt["layers"]}, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    # Keep the same guarded native shutdown policy as the RTX probe: Kit can
    # otherwise abort while tearing down live extension task groups on Windows.
    os._exit(exit_code)

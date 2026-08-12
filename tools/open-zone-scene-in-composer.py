"""Open a complete FireViewer zone in USD Composer without eager payload load.

The scene builder delivers one lightweight header per 1 km tile.  Each header
exposes two independent payload families:

* ``Terrain`` is always loaded.  Its ``terrainLOD`` and ``collisionLOD``
  selections are authored only in the anonymous session layer and change with
  the active camera.
* ``Details`` (HERO), ``DetailsMid`` and ``DetailsFar`` contain progressively
  lighter representations of the same vegetation, buildings, roads and
  hydrology.  Every tile always keeps exactly one level loaded: FAR outside
  the frustum, MID across the complete visible footprint, and HERO only in a
  bounded near-camera set.

No source layer is saved or rewritten by this script.  A root checksum is
verified before and after the initial Editor setup so an accidental source
mutation cannot be mistaken for a successful review open.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import carb
import omni.kit.app
import omni.kit.viewport.utility
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


LOD_VARIANT_SET = "terrainLOD"
DISTANT_LOD = "LOD3"
REQUIRED_TERRAIN_LODS = frozenset({"LOD1", "LOD2", "LOD3"})
COLLISION_VARIANT_SET = "collisionLOD"
COLLISION_LEVELS = ("NEAR", "FAR")
DETAIL_LEVELS = ("HERO", "MID", "FAR")
DETAIL_COUNT_KEYS = frozenset(
    {"buildings", "roads", "hydrology", "vegetation"}
)
SESSION_CAMERA_PATH = "/FireViewerSession/StreamingReviewCamera"
DEFAULT_REVIEW_CAMERA_PATH = "/World/ReviewCameras/Review06"


@dataclass(frozen=True)
class TileHeader:
    """One lightweight 1 km tile interface discovered with payloads unloaded."""

    tile_ref: str
    tile_path: str
    terrain_path: str
    hero_detail_path: str
    mid_detail_path: str
    far_detail_path: str
    bounds: tuple[float, float, float, float]
    terrain_lods: tuple[str, ...]
    collision_lods: tuple[str, ...]
    ground_z: float = 0.0
    minimum_z: float = 0.0
    maximum_z: float = 0.0


@dataclass(frozen=True)
class CameraView:
    """Small renderer-independent camera description used by the planner."""

    eye: tuple[float, float, float]
    forward: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    horizontal_half_tangent: float
    vertical_half_tangent: float
    ground_z: float
    focus_xy: tuple[float, float]
    altitude: float


@dataclass(frozen=True)
class WorkingSetPlan:
    """A complete terrain LOD plan plus the bounded detail working set."""

    terrain_lods: dict[str, str]
    collision_lods: dict[str, str]
    collision_paths: dict[str, str]
    detail_levels: dict[str, str]
    detail_paths: dict[str, str]
    visible_tile_refs: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain one JSON object")
    return payload


def _bounded_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < minimum or value > maximum:
        raise RuntimeError(
            f"{name} must be between {minimum} and {maximum}, got {value}"
        )
    return value


def _approximately(raw: Any, expected: float, tolerance: float) -> bool:
    return (
        isinstance(raw, (int, float))
        and not isinstance(raw, bool)
        and math.isfinite(float(raw))
        and math.isclose(
            float(raw),
            expected,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    )


def _parse_bounds(raw: Any, *, tile_ref: str) -> tuple[float, float, float, float]:
    if not isinstance(raw, str):
        raise RuntimeError(f"{tile_ref} has no local bounds in its USD header")
    try:
        values = tuple(float(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise RuntimeError(f"{tile_ref} has invalid local bounds") from exc
    if len(values) != 4:
        raise RuntimeError(f"{tile_ref} local bounds must contain four values")
    xmin, ymin, xmax, ymax = values
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{tile_ref} local bounds are not finite")
    if xmax <= xmin or ymax <= ymin:
        raise RuntimeError(f"{tile_ref} local bounds are empty")
    return xmin, ymin, xmax, ymax


def _read_build_contract(
    *,
    receipt_path: Path,
    usd_path: Path,
    zone_id: str,
    expected_tile_count: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate the schema-2 full-scene contract before Composer opens it."""

    receipt = _read_json(receipt_path, label="scene build receipt")
    if receipt.get("schema_version") != 2:
        raise RuntimeError("scene build receipt must use schema version 2")
    if receipt.get("zone_id") != zone_id:
        raise RuntimeError("scene build receipt belongs to another zone")
    if receipt.get("source_profile") != "full":
        raise RuntimeError(
            "progressive photoreal review requires the full source profile"
        )
    root_artifact = receipt.get("root_usd")
    if not isinstance(root_artifact, dict):
        raise RuntimeError("scene build receipt has no root USD artifact")
    zone_root = receipt_path.parent.parent.resolve()
    declared_root = (zone_root / str(root_artifact.get("path", ""))).resolve()
    if declared_root != usd_path:
        raise RuntimeError("scene build receipt points to a different root USD")
    if root_artifact.get("sha256") != _sha256(usd_path):
        raise RuntimeError("root USD checksum does not match the build receipt")

    coverage = receipt.get("tile_coverage")
    terrain_payloads = receipt.get("payloads")
    hero_payloads = receipt.get("detail_payloads")
    mid_payloads = receipt.get("detail_mid_payloads")
    far_payloads = receipt.get("detail_far_payloads")
    if not isinstance(coverage, list):
        raise RuntimeError("scene build receipt has no tile coverage")
    if not isinstance(terrain_payloads, list) or not isinstance(hero_payloads, list):
        raise RuntimeError("scene build receipt has no complete payload catalogs")
    if not isinstance(mid_payloads, list) or not isinstance(far_payloads, list):
        raise RuntimeError(
            "full scene has no MID/FAR detail payload catalogs; refusing "
            "camera streaming that could leave visible tiles empty"
        )
    if (
        len(coverage) != expected_tile_count
        or len(terrain_payloads) != expected_tile_count
        or len(hero_payloads) != expected_tile_count
        or len(mid_payloads) != expected_tile_count
        or len(far_payloads) != expected_tile_count
    ):
        raise RuntimeError(
            "progressive review requires exactly "
            f"{expected_tile_count} terrain, HERO, MID and FAR tile payloads"
        )
    layers = receipt.get("layers")
    detail_streaming = (
        layers.get("detail_streaming") if isinstance(layers, dict) else None
    )
    collisions = layers.get("collisions") if isinstance(layers, dict) else None
    if (
        not isinstance(detail_streaming, dict)
        or detail_streaming.get("prim_count") != expected_tile_count
        or detail_streaming.get("levels") != list(DETAIL_LEVELS)
        or detail_streaming.get(
            "terrain_is_never_unloaded_for_detail_streaming"
        )
        is not True
    ):
        raise RuntimeError(
            "scene build receipt does not guarantee complete detail delivery "
            "with persistent terrain"
        )
    if (
        not isinstance(collisions, dict)
        or collisions.get("prim_count") != expected_tile_count
        or collisions.get("levels") != list(COLLISION_LEVELS)
        or not _approximately(
            collisions.get("near_spacing_m"),
            4.0,
            0.25,
        )
        or not _approximately(
            collisions.get("far_spacing_m"),
            32.0,
            1.0,
        )
    ):
        raise RuntimeError(
            "scene build receipt does not guarantee NEAR/FAR collision LODs "
            "at approximately 4 m and 32 m"
        )

    by_ref: dict[str, dict[str, Any]] = {}
    terrain_paths = {
        str(item.get("path", ""))
        for item in terrain_payloads
        if isinstance(item, dict)
    }
    hero_paths = {
        str(item.get("path", ""))
        for item in hero_payloads
        if isinstance(item, dict)
    }
    mid_paths = {
        str(item.get("path", ""))
        for item in mid_payloads
        if isinstance(item, dict)
    }
    far_paths = {
        str(item.get("path", ""))
        for item in far_payloads
        if isinstance(item, dict)
    }
    if len(terrain_paths) != expected_tile_count:
        raise RuntimeError("terrain payload catalog contains duplicates")
    if any(
        len(paths) != expected_tile_count
        for paths in (hero_paths, mid_paths, far_paths)
    ):
        raise RuntimeError("HERO/MID/FAR detail payload catalogs contain duplicates")
    if hero_paths & mid_paths or hero_paths & far_paths or mid_paths & far_paths:
        raise RuntimeError(
            "HERO/MID/FAR detail payload catalogs must use distinct artifacts"
        )
    for item in coverage:
        if not isinstance(item, dict):
            raise RuntimeError("tile coverage contains a non-object entry")
        tile_ref = str(item.get("tile_ref", "")).strip()
        if not tile_ref or tile_ref in by_ref:
            raise RuntimeError("tile coverage contains a missing or duplicate tile_ref")
        terrain_path = str(item.get("terrain_payload", ""))
        detail_lods = item.get("detail_lods")
        if not isinstance(detail_lods, dict) or set(detail_lods) != set(DETAIL_LEVELS):
            raise RuntimeError(
                f"{tile_ref} does not declare HERO, MID and FAR detail levels"
            )
        detail_lod_counts = item.get("detail_lod_counts")
        if (
            not isinstance(detail_lod_counts, dict)
            or set(detail_lod_counts) != set(DETAIL_LEVELS)
        ):
            raise RuntimeError(
                f"{tile_ref} has no HERO/MID/FAR representation counts"
            )
        for level in DETAIL_LEVELS:
            counts = detail_lod_counts.get(level)
            if (
                not isinstance(counts, dict)
                or set(counts) != DETAIL_COUNT_KEYS
            ):
                raise RuntimeError(
                    f"{tile_ref} {level} representation counts are invalid"
                )
            if (
                any(
                    not isinstance(counts[key], int)
                    or isinstance(counts[key], bool)
                    or counts[key] < 0
                    for key in DETAIL_COUNT_KEYS
                )
                or sum(counts[key] for key in DETAIL_COUNT_KEYS) < 1
            ):
                raise RuntimeError(
                    f"{tile_ref} {level} has no authored detail representation"
                )
        hero_path = str(detail_lods.get("HERO", ""))
        mid_path = str(detail_lods.get("MID", ""))
        far_path = str(detail_lods.get("FAR", ""))
        if (
            terrain_path not in terrain_paths
            or hero_path not in hero_paths
            or mid_path not in mid_paths
            or far_path not in far_paths
        ):
            raise RuntimeError(f"{tile_ref} payload paths disagree with their catalogs")
        lods = item.get("terrain_lods")
        if not isinstance(lods, list) or not REQUIRED_TERRAIN_LODS.issubset(
            str(value) for value in lods
        ):
            raise RuntimeError(f"{tile_ref} does not declare LOD1, LOD2 and LOD3")
        collision_lods = item.get("collision_lods")
        if collision_lods != list(COLLISION_LEVELS):
            raise RuntimeError(
                f"{tile_ref} does not declare NEAR and FAR collision LODs"
            )
        by_ref[tile_ref] = item
    if len(by_ref) != expected_tile_count:
        raise RuntimeError("scene build receipt does not cover every tile once")
    return receipt, by_ref


def _custom_data(prim: Any, key: str) -> Any:
    value = prim.GetCustomDataByKey(key)
    if value is not None:
        return value
    # Some USD versions expose namespaced customData as a nested dictionary.
    namespace, _, child = key.partition(":")
    nested = prim.GetCustomData().get(namespace)
    if isinstance(nested, dict):
        return nested.get(child)
    return None


def _discover_tile_headers(
    stage: Usd.Stage,
    coverage_by_ref: dict[str, dict[str, Any]],
) -> list[TileHeader]:
    """Discover payload interfaces without composing either heavy payload."""

    discovered: dict[str, TileHeader] = {}
    for prim in stage.Traverse():
        raw_ref = _custom_data(prim, "fireviewer:tile_ref")
        if raw_ref is None:
            continue
        tile_ref = str(raw_ref)
        if tile_ref not in coverage_by_ref:
            raise RuntimeError(f"USD exposes undeclared tile header {tile_ref}")
        if tile_ref in discovered:
            raise RuntimeError(f"USD exposes duplicate tile header {tile_ref}")
        bounds = _parse_bounds(
            _custom_data(prim, "fireviewer:local_bounds"),
            tile_ref=tile_ref,
        )
        terrain_path = str(prim.GetPath().AppendChild("Terrain"))
        hero_detail_path = str(prim.GetPath().AppendChild("Details"))
        mid_detail_path = str(prim.GetPath().AppendChild("DetailsMid"))
        far_detail_path = str(prim.GetPath().AppendChild("DetailsFar"))
        terrain = stage.GetPrimAtPath(terrain_path)
        hero_details = stage.GetPrimAtPath(hero_detail_path)
        mid_details = stage.GetPrimAtPath(mid_detail_path)
        far_details = stage.GetPrimAtPath(far_detail_path)
        if not terrain.IsValid() or not terrain.HasPayload():
            raise RuntimeError(f"{tile_ref} has no terrain payload arc")
        for level, details in (
            ("HERO", hero_details),
            ("MID", mid_details),
            ("FAR", far_details),
        ):
            if not details.IsValid() or not details.HasPayload():
                raise RuntimeError(
                    f"{tile_ref} has no {level} detail payload arc"
                )
        lods = tuple(
            str(value) for value in coverage_by_ref[tile_ref]["terrain_lods"]
        )
        collision_lods = tuple(
            str(value)
            for value in coverage_by_ref[tile_ref]["collision_lods"]
        )
        discovered[tile_ref] = TileHeader(
            tile_ref=tile_ref,
            tile_path=str(prim.GetPath()),
            terrain_path=terrain_path,
            hero_detail_path=hero_detail_path,
            mid_detail_path=mid_detail_path,
            far_detail_path=far_detail_path,
            bounds=bounds,
            terrain_lods=lods,
            collision_lods=collision_lods,
        )
    if set(discovered) != set(coverage_by_ref):
        missing = sorted(set(coverage_by_ref) - set(discovered))
        raise RuntimeError(
            "root USD does not expose every declared tile header: "
            + ", ".join(missing[:8])
        )
    return [discovered[key] for key in sorted(discovered)]


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _point_to_bounds_distance(
    point: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> float:
    xmin, ymin, xmax, ymax = bounds
    dx = max(xmin - point[0], 0.0, point[0] - xmax)
    dy = max(ymin - point[1], 0.0, point[1] - ymax)
    return math.hypot(dx, dy)


def _cross_2d(
    origin: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (
        first[1] - origin[1]
    ) * (second[0] - origin[0])


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    epsilon = 1e-8
    if (
        max(min(first_start[0], first_end[0]), min(second_start[0], second_end[0]))
        > min(max(first_start[0], first_end[0]), max(second_start[0], second_end[0]))
        + epsilon
        or max(
            min(first_start[1], first_end[1]),
            min(second_start[1], second_end[1]),
        )
        > min(
            max(first_start[1], first_end[1]),
            max(second_start[1], second_end[1]),
        )
        + epsilon
    ):
        return False
    first_a = _cross_2d(first_start, first_end, second_start)
    first_b = _cross_2d(first_start, first_end, second_end)
    second_a = _cross_2d(second_start, second_end, first_start)
    second_b = _cross_2d(second_start, second_end, first_end)
    return (
        min(first_a, first_b) <= epsilon
        and max(first_a, first_b) >= -epsilon
        and min(second_a, second_b) <= epsilon
        and max(second_a, second_b) >= -epsilon
    )


def _point_in_convex_polygon(
    point: tuple[float, float],
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    signs = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        cross = _cross_2d(start, end, point)
        if abs(cross) > 1e-8:
            signs.append(cross > 0.0)
    return not signs or all(sign == signs[0] for sign in signs)


def _rectangle_intersects_polygon(
    bounds: tuple[float, float, float, float],
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    xmin, ymin, xmax, ymax = bounds
    rectangle = (
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    )
    if any(
        xmin <= point[0] <= xmax and ymin <= point[1] <= ymax
        for point in polygon
    ):
        return True
    if any(_point_in_convex_polygon(point, polygon) for point in rectangle):
        return True
    rectangle_edges = tuple(
        (rectangle[index], rectangle[(index + 1) % len(rectangle)])
        for index in range(len(rectangle))
    )
    polygon_edges = tuple(
        (polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )
    return any(
        _segments_intersect(rect_start, rect_end, poly_start, poly_end)
        for rect_start, rect_end in rectangle_edges
        for poly_start, poly_end in polygon_edges
    )


def _ground_frustum_polygon(
    view: CameraView,
    *,
    guard: float,
    ground_z: float,
) -> tuple[tuple[float, float], ...]:
    footprint = []
    for horizontal, vertical in (
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
    ):
        direction = tuple(
            view.forward[index]
            + horizontal
            * view.horizontal_half_tangent
            * guard
            * view.right[index]
            + vertical
            * view.vertical_half_tangent
            * guard
            * view.up[index]
            for index in range(3)
        )
        if direction[2] >= -1e-6:
            continue
        distance = (ground_z - view.eye[2]) / direction[2]
        if distance <= 0.0:
            continue
        footprint.append(
            (
                view.eye[0] + direction[0] * distance,
                view.eye[1] + direction[1] * distance,
            )
        )
    return tuple(footprint)


def _aabb_intersects_view_frustum(
    tile: TileHeader,
    view: CameraView,
    *,
    guard: float,
) -> bool:
    """Conservatively test the complete terrain AABB against the camera frustum."""

    xmin, ymin, xmax, ymax = tile.bounds
    corners = tuple(
        _subtract((x, y, z), view.eye)
        for x in (xmin, xmax)
        for y in (ymin, ymax)
        for z in (tile.minimum_z, tile.maximum_z)
    )
    horizontal_tangent = view.horizontal_half_tangent * guard
    vertical_tangent = view.vertical_half_tangent * guard

    def coordinates(
        offset: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return (
            _dot(offset, view.forward),
            _dot(offset, view.right),
            _dot(offset, view.up),
        )

    projected = tuple(coordinates(corner) for corner in corners)
    # Reject only when all AABB corners lie outside one frustum plane.  This
    # admits some harmless false positives but cannot omit a terrain volume
    # merely because steep relief crosses the view above/below its midpoint.
    outside_planes = (
        lambda depth, horizontal, vertical: depth < 0.1,
        lambda depth, horizontal, vertical: horizontal < -depth * horizontal_tangent,
        lambda depth, horizontal, vertical: horizontal > depth * horizontal_tangent,
        lambda depth, horizontal, vertical: vertical < -depth * vertical_tangent,
        lambda depth, horizontal, vertical: vertical > depth * vertical_tangent,
    )
    return not any(
        all(plane(depth, horizontal, vertical) for depth, horizontal, vertical in projected)
        for plane in outside_planes
    )


def _tile_visible(tile: TileHeader, view: CameraView, *, guard: float) -> bool:
    return _aabb_intersects_view_frustum(tile, view, guard=guard)


def _plan_working_set(
    *,
    tiles: Iterable[TileHeader],
    view: CameraView,
    hero_cap: int,
    hero_guard_minimum: int,
    lod0_cap: int,
    lod1_cap: int,
    lod2_cap: int,
    retained_detail_levels: dict[str, str] | None = None,
) -> WorkingSetPlan:
    """Plan visible detail streaming while retaining terrain for every tile."""

    ordered_tiles = sorted(
        tiles,
        key=lambda tile: (
            _point_to_bounds_distance(view.focus_xy, tile.bounds),
            tile.tile_ref,
        ),
    )
    visible = [tile for tile in ordered_tiles if _tile_visible(tile, view, guard=1.18)]
    visible_guard = [
        tile for tile in ordered_tiles if _tile_visible(tile, view, guard=1.48)
    ]
    if not visible_guard:
        visible_guard = ordered_tiles[: min(4, len(ordered_tiles))]
    ranked = visible_guard + [
        tile for tile in ordered_tiles if tile not in visible_guard
    ]

    retained_detail_levels = retained_detail_levels or {}
    hero_tiles = visible_guard[:hero_cap]
    # Retain already loaded tiles inside a wider frustum to prevent payload
    # thrashing when the camera sits on a tile boundary.
    retained_hero_guard = [
        tile
        for tile in ordered_tiles
        if retained_detail_levels.get(tile.tile_ref) == "HERO"
        and _tile_visible(tile, view, guard=1.72)
        and tile not in hero_tiles
    ]
    hero_tiles.extend(
        retained_hero_guard[: max(0, hero_cap - len(hero_tiles))]
    )
    # A low, inclined camera can have a mathematically narrow ground footprint.
    # Preloading a bounded guard ring prevents the old "single detail tile"
    # failure while the user starts moving through the review scene.
    if len(hero_tiles) < hero_guard_minimum:
        hero_tiles.extend(
            tile
            for tile in ranked
            if tile not in hero_tiles
        )
        hero_tiles = hero_tiles[:hero_guard_minimum]
    retained_mid_guard = [
        tile
        for tile in ordered_tiles
        if retained_detail_levels.get(tile.tile_ref) == "MID"
        and _tile_visible(tile, view, guard=1.72)
        and tile not in hero_tiles
        and tile not in visible_guard
    ]
    mid_tiles = [
        tile for tile in visible_guard if tile not in hero_tiles
    ] + retained_mid_guard
    lod1_tiles = ranked[:lod1_cap]
    lod2_tiles = ranked[:lod2_cap]
    lod0_tiles = [
        tile for tile in lod1_tiles if "LOD0" in tile.terrain_lods
    ][:lod0_cap]
    lod0_refs = {tile.tile_ref for tile in lod0_tiles}
    lod1_refs = {tile.tile_ref for tile in lod1_tiles}
    lod2_refs = {tile.tile_ref for tile in lod2_tiles}
    hero_refs = {tile.tile_ref for tile in hero_tiles}
    mid_refs = {tile.tile_ref for tile in mid_tiles}
    visible_refs = {tile.tile_ref for tile in visible}
    if not visible_refs.issubset(hero_refs | mid_refs):
        raise RuntimeError(
            "detail planner left a frustum-visible tile without HERO/MID coverage"
        )
    terrain_lods: dict[str, str] = {}
    collision_lods: dict[str, str] = {}
    collision_paths: dict[str, str] = {}
    detail_levels: dict[str, str] = {}
    detail_paths: dict[str, str] = {}
    for tile in ordered_tiles:
        if tile.tile_ref in lod0_refs:
            lod = "LOD0"
        elif tile.tile_ref in lod1_refs:
            lod = "LOD1"
        elif tile.tile_ref in lod2_refs:
            lod = "LOD2"
        else:
            lod = DISTANT_LOD
        if lod not in tile.terrain_lods:
            raise RuntimeError(f"{tile.tile_ref} cannot select declared terrain {lod}")
        terrain_lods[tile.terrain_path] = lod
        collision_lod = (
            "NEAR"
            if tile.tile_ref in hero_refs or tile.tile_ref in mid_refs
            else "FAR"
        )
        if collision_lod not in tile.collision_lods:
            raise RuntimeError(
                f"{tile.tile_ref} cannot select declared collision {collision_lod}"
            )
        collision_lods[tile.tile_ref] = collision_lod
        collision_paths[tile.tile_ref] = tile.terrain_path
        if tile.tile_ref in hero_refs:
            detail_level = "HERO"
            detail_path = tile.hero_detail_path
        elif tile.tile_ref in mid_refs:
            detail_level = "MID"
            detail_path = tile.mid_detail_path
        else:
            detail_level = "FAR"
            detail_path = tile.far_detail_path
        detail_levels[tile.tile_ref] = detail_level
        detail_paths[tile.tile_ref] = detail_path
    return WorkingSetPlan(
        terrain_lods=terrain_lods,
        collision_lods=collision_lods,
        collision_paths=collision_paths,
        detail_levels=detail_levels,
        detail_paths=detail_paths,
        visible_tile_refs=tuple(tile.tile_ref for tile in visible),
    )


def _look_at(eye: Gf.Vec3d, target: Gf.Vec3d) -> Gf.Matrix4d:
    forward = (target - eye).GetNormalized()
    world_up = Gf.Vec3d(0.0, 0.0, 1.0)
    right = Gf.Cross(forward, world_up).GetNormalized()
    if right.GetLength() < 1e-6:
        right = Gf.Vec3d(1.0, 0.0, 0.0)
    up = Gf.Cross(right, forward).GetNormalized()
    back = -forward
    return Gf.Matrix4d(
        right[0],
        right[1],
        right[2],
        0.0,
        up[0],
        up[1],
        up[2],
        0.0,
        back[0],
        back[1],
        back[2],
        0.0,
        eye[0],
        eye[1],
        eye[2],
        1.0,
    )


def _parse_target(prim: Usd.Prim) -> Gf.Vec3d:
    raw = _custom_data(prim, "fireviewer:look_at_local")
    if not isinstance(raw, str):
        raise RuntimeError("initial review camera has no look-at target")
    try:
        values = tuple(float(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise RuntimeError("initial review camera look-at target is invalid") from exc
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise RuntimeError("initial review camera look-at target is invalid")
    return Gf.Vec3d(*values)


def _create_session_camera(
    *,
    stage: Usd.Stage,
    source_path: str,
    scene_span: float,
) -> tuple[str, float]:
    """Create an inclined review camera exclusively in the session layer."""

    source_prim = stage.GetPrimAtPath(source_path)
    if not source_prim.IsValid() or not source_prim.IsA(UsdGeom.Camera):
        raise RuntimeError("review USD is missing its initial review camera")
    source = UsdGeom.Camera(source_prim)
    target = _parse_target(source_prim)
    eye = UsdGeom.XformCache().GetLocalToWorldTransform(
        source_prim
    ).ExtractTranslation()
    focal = float(source.GetFocalLengthAttr().Get() or 35.0)
    horizontal_aperture = float(source.GetHorizontalApertureAttr().Get() or 20.955)
    vertical_aperture = float(source.GetVerticalApertureAttr().Get() or 15.2908)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Xform.Define(stage, "/FireViewerSession")
        camera = UsdGeom.Camera.Define(stage, SESSION_CAMERA_PATH)
        xformable = UsdGeom.Xformable(camera)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(_look_at(eye, target))
        camera.CreateFocalLengthAttr().Set(focal)
        camera.CreateHorizontalApertureAttr().Set(horizontal_aperture)
        camera.CreateVerticalApertureAttr().Set(vertical_aperture)
        camera.CreateClippingRangeAttr().Set(
            Gf.Vec2f(0.1, float(max(scene_span * 8.0, 100_000.0)))
        )
        camera.GetPrim().SetCustomDataByKey(
            "fireviewer:session_only", "progressive_review_camera"
        )
    return SESSION_CAMERA_PATH, float(target[2])


def _camera_view(
    stage: Usd.Stage,
    *,
    camera_path: str,
    fallback_ground_z: float,
) -> CameraView:
    prim = stage.GetPrimAtPath(camera_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"viewport camera is unavailable in the stage: {camera_path}")
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    eye_value = matrix.ExtractTranslation()
    eye = (float(eye_value[0]), float(eye_value[1]), float(eye_value[2]))
    raw_right = (float(matrix[0][0]), float(matrix[0][1]), float(matrix[0][2]))
    raw_up = (float(matrix[1][0]), float(matrix[1][1]), float(matrix[1][2]))
    raw_forward = (
        -float(matrix[2][0]),
        -float(matrix[2][1]),
        -float(matrix[2][2]),
    )
    right_length = math.sqrt(_dot(raw_right, raw_right))
    up_length = math.sqrt(_dot(raw_up, raw_up))
    forward_length = math.sqrt(_dot(raw_forward, raw_forward))
    if min(right_length, up_length, forward_length) < 1e-6:
        raise RuntimeError(f"viewport camera has a degenerate transform: {camera_path}")
    right = tuple(value / right_length for value in raw_right)
    up = tuple(value / up_length for value in raw_up)
    forward = tuple(value / forward_length for value in raw_forward)
    camera = UsdGeom.Camera(prim)
    focal = max(1e-6, float(camera.GetFocalLengthAttr().Get() or 35.0))
    horizontal_aperture = max(
        1e-6, float(camera.GetHorizontalApertureAttr().Get() or 20.955)
    )
    vertical_aperture = max(
        1e-6, float(camera.GetVerticalApertureAttr().Get() or 15.2908)
    )
    if forward[2] < -1e-4:
        ray_distance = (fallback_ground_z - eye[2]) / forward[2]
        if ray_distance <= 0.0:
            ray_distance = 1_000.0
    else:
        ray_distance = 1_000.0
    focus = (
        eye[0] + forward[0] * ray_distance,
        eye[1] + forward[1] * ray_distance,
    )
    return CameraView(
        eye=eye,
        forward=forward,
        right=right,
        up=up,
        horizontal_half_tangent=horizontal_aperture / (2.0 * focal),
        vertical_half_tangent=vertical_aperture / (2.0 * focal),
        ground_z=fallback_ground_z,
        focus_xy=focus,
        altitude=max(0.0, eye[2] - fallback_ground_z),
    )


def _camera_view_for_tiles(
    stage: Usd.Stage,
    *,
    camera_path: str,
    fallback_ground_z: float,
    tiles: list[TileHeader],
) -> CameraView:
    """Refine the ground intersection with the nearest real terrain tile."""

    provisional = _camera_view(
        stage,
        camera_path=camera_path,
        fallback_ground_z=fallback_ground_z,
    )
    nearest = min(
        tiles,
        key=lambda tile: (
            _point_to_bounds_distance(provisional.focus_xy, tile.bounds),
            tile.tile_ref,
        ),
    )
    return _camera_view(
        stage,
        camera_path=camera_path,
        fallback_ground_z=nearest.ground_z,
    )


def _session_select_variant(
    stage: Usd.Stage,
    prim_path: str,
    variant_set: str,
    selection: str,
) -> None:
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        prim = stage.OverridePrim(prim_path)
        if not prim.GetVariantSets().SetSelection(variant_set, selection):
            raise RuntimeError(
                f"could not select {variant_set}={selection} on {prim_path}"
            )


def _session_select_lod(stage: Usd.Stage, terrain_path: str, lod: str) -> None:
    _session_select_variant(
        stage,
        terrain_path,
        LOD_VARIANT_SET,
        lod,
    )


def _session_select_collision(
    stage: Usd.Stage,
    terrain_path: str,
    collision_lod: str,
) -> None:
    _session_select_variant(
        stage,
        terrain_path,
        COLLISION_VARIANT_SET,
        collision_lod,
    )


async def _load_paths(
    stage: Usd.Stage,
    paths: Iterable[str],
    *,
    batch_size: int,
    label: str,
) -> None:
    bounded_paths = tuple(paths)
    for index, path in enumerate(bounded_paths, start=1):
        stage.Load(Sdf.Path(path), Usd.LoadWithDescendants)
        if index % batch_size == 0:
            await omni.kit.app.get_app().next_update_async()
        if index % (batch_size * 4) == 0 or index == len(bounded_paths):
            carb.log_info(
                f"FireViewer {label}: {index}/{len(bounded_paths)} payloads loaded"
            )


async def _wait_for_stage_settle(
    context: Any,
    *,
    maximum_updates: int,
    stable_updates: int,
) -> int:
    stable = 0
    for update in range(1, maximum_updates + 1):
        await omni.kit.app.get_app().next_update_async()
        message, loaded, total = context.get_stage_loading_status()
        streaming = bool(context.get_stage_streaming_status())
        if int(loaded) >= int(total) and not streaming:
            stable += 1
            if stable >= stable_updates:
                return update
        else:
            stable = 0
        if update == maximum_updates:
            raise RuntimeError(
                "initial terrain/detail streaming did not settle: "
                f"{message}; loaded={loaded}; total={total}; streaming={streaming}"
            )
    raise AssertionError("unreachable stage settle loop")


async def _wait_for_detail_composition(
    context: Any,
    stage: Usd.Stage,
    paths: Iterable[str],
    *,
    maximum_updates: int,
    stable_updates: int = 2,
) -> int:
    """Wait for authored payloads and USD streaming to settle before replacement."""

    expected_paths = tuple(paths)
    stable = 0
    for update in range(1, maximum_updates + 1):
        await omni.kit.app.get_app().next_update_async()
        if context.get_stage() is not stage:
            raise RuntimeError(
                "review stage changed while detail replacements were loading"
            )
        payloads_loaded = all(
            stage.GetPrimAtPath(path).IsValid()
            and stage.GetPrimAtPath(path).IsLoaded()
            for path in expected_paths
        )
        message, loaded, total = context.get_stage_loading_status()
        streaming = bool(context.get_stage_streaming_status())
        if (
            payloads_loaded
            and int(loaded) >= int(total)
            and not streaming
        ):
            stable += 1
            if stable >= stable_updates:
                return update
        else:
            stable = 0
        if update == maximum_updates:
            raise RuntimeError(
                "replacement detail payloads did not settle before unload: "
                f"{message}; loaded={loaded}; total={total}; "
                f"streaming={streaming}; paths={expected_paths}"
            )
    raise AssertionError("unreachable detail settle loop")


async def _apply_plan(
    *,
    context: Any,
    stage: Usd.Stage,
    plan: WorkingSetPlan,
    current_lods: dict[str, str],
    current_collision_lods: dict[str, str],
    active_detail_levels: dict[str, str],
    active_detail_paths: dict[str, str],
    detail_transition_cap: int,
    detail_settle_maximum_updates: int,
    lod_transition_cap: int,
    initial: bool,
) -> None:
    lod_changes = [
        (path, lod)
        for path, lod in plan.terrain_lods.items()
        if current_lods.get(path) != lod
    ]
    lod_limit = len(lod_changes) if initial else lod_transition_cap
    for index, (path, lod) in enumerate(lod_changes[:lod_limit], start=1):
        _session_select_lod(stage, path, lod)
        current_lods[path] = lod
        if index % 8 == 0:
            await omni.kit.app.get_app().next_update_async()

    collision_changes = [
        (
            tile_ref,
            plan.collision_paths[tile_ref],
            collision_lod,
        )
        for tile_ref, collision_lod in plan.collision_lods.items()
        if current_collision_lods.get(tile_ref) != collision_lod
    ]
    collision_promotions = [
        change for change in collision_changes if change[2] == "NEAR"
    ]
    # Promote collision first so an arriving HERO/MID representation never
    # becomes interactive against the coarser FAR surface.
    for index, (tile_ref, path, collision_lod) in enumerate(
        collision_promotions,
        start=1,
    ):
        _session_select_collision(stage, path, collision_lod)
        current_collision_lods[tile_ref] = collision_lod
        if index % 8 == 0:
            await omni.kit.app.get_app().next_update_async()

    if set(active_detail_levels) != set(plan.detail_levels):
        raise RuntimeError(
            "active detail coverage does not contain exactly one level per tile"
        )
    transitions = [
        (
            tile_ref,
            active_detail_paths[tile_ref],
            plan.detail_paths[tile_ref],
            plan.detail_levels[tile_ref],
        )
        for tile_ref in plan.detail_levels
        if active_detail_levels[tile_ref] != plan.detail_levels[tile_ref]
    ]
    detail_rank = {"FAR": 0, "MID": 1, "HERO": 2}
    promotions = [
        transition
        for transition in transitions
        if detail_rank[transition[3]]
        > detail_rank[active_detail_levels[transition[0]]]
    ]
    demotions = [
        transition
        for transition in transitions
        if detail_rank[transition[3]]
        < detail_rank[active_detail_levels[transition[0]]]
    ]
    selected = (
        transitions
        if initial
        else promotions + demotions[:detail_transition_cap]
    )
    # Compose every quality promotion first and wait until USD reports the
    # replacement payloads loaded and stable.  Only then unload the previous
    # representation.  The transition cap throttles demotions only: a camera
    # teleport cannot leave newly visible tiles at FAR for dozens of ticks.
    for offset in range(0, len(selected), 4):
        batch = selected[offset : offset + 4]
        for _, _, desired_path, _ in batch:
            stage.Load(Sdf.Path(desired_path), Usd.LoadWithDescendants)
        if batch:
            await _wait_for_detail_composition(
                context,
                stage,
                (desired_path for _, _, desired_path, _ in batch),
                maximum_updates=detail_settle_maximum_updates,
            )
        for tile_ref, old_path, desired_path, desired_level in batch:
            stage.Unload(Sdf.Path(old_path))
            active_detail_levels[tile_ref] = desired_level
            active_detail_paths[tile_ref] = desired_path

    # Demote collision only after the matching detail transition has succeeded.
    # If loading/settling raises, the old HERO/MID representation therefore
    # retains NEAR collision and remains internally coherent.
    collision_demotions = [
        change
        for change in collision_changes
        if change[2] == "FAR"
        and active_detail_levels[change[0]] == "FAR"
    ]
    selected_collision_demotions = (
        collision_demotions
        if initial
        else collision_demotions[:lod_transition_cap]
    )
    for index, (tile_ref, path, collision_lod) in enumerate(
        selected_collision_demotions,
        start=1,
    ):
        _session_select_collision(stage, path, collision_lod)
        current_collision_lods[tile_ref] = collision_lod
        if index % 8 == 0:
            await omni.kit.app.get_app().next_update_async()


def _active_camera_path(viewport: Any, *, fallback: str) -> str:
    raw = str(viewport.camera_path)
    return raw if raw and raw != "." else fallback


async def _stream_camera_working_set(
    *,
    context: Any,
    stage: Usd.Stage,
    viewport: Any,
    tiles: list[TileHeader],
    fallback_camera_path: str,
    ground_z: float,
    hero_cap: int,
    hero_guard_minimum: int,
    lod0_cap: int,
    lod1_cap: int,
    lod2_cap: int,
    detail_transition_cap: int,
    detail_settle_maximum_updates: int,
    lod_transition_cap: int,
    active_detail_levels: dict[str, str],
    active_detail_paths: dict[str, str],
    current_lods: dict[str, str],
    current_collision_lods: dict[str, str],
    reevaluation_updates: int,
) -> None:
    consecutive_failures = 0
    while context.get_stage() is stage:
        try:
            camera_path = _active_camera_path(
                viewport, fallback=fallback_camera_path
            )
            try:
                view = _camera_view_for_tiles(
                    stage,
                    camera_path=camera_path,
                    fallback_ground_z=ground_z,
                    tiles=tiles,
                )
            except RuntimeError:
                view = _camera_view_for_tiles(
                    stage,
                    camera_path=fallback_camera_path,
                    fallback_ground_z=ground_z,
                    tiles=tiles,
                )
            plan = _plan_working_set(
                tiles=tiles,
                view=view,
                hero_cap=hero_cap,
                hero_guard_minimum=hero_guard_minimum,
                lod0_cap=lod0_cap,
                lod1_cap=lod1_cap,
                lod2_cap=lod2_cap,
                retained_detail_levels=active_detail_levels,
            )
            await _apply_plan(
                context=context,
                stage=stage,
                plan=plan,
                current_lods=current_lods,
                current_collision_lods=current_collision_lods,
                active_detail_levels=active_detail_levels,
                active_detail_paths=active_detail_paths,
                detail_transition_cap=detail_transition_cap,
                detail_settle_maximum_updates=detail_settle_maximum_updates,
                lod_transition_cap=lod_transition_cap,
                initial=False,
            )
            consecutive_failures = 0
        except asyncio.CancelledError:
            return
        except Exception as exc:
            consecutive_failures += 1
            carb.log_error(
                "FireViewer camera working-set update failed "
                f"({consecutive_failures}/3): {exc}"
            )
            if consecutive_failures >= 3:
                return
        # One reassessment per configured-rate second avoids composition churn
        # while the user moves the camera.
        for _ in range(reevaluation_updates):
            if context.get_stage() is not stage:
                return
            await omni.kit.app.get_app().next_update_async()


def _task_result(task: asyncio.Task[Any], *, label: str) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as exc:
        carb.log_error(f"{label} failed: {exc}")


async def _open_review_stage() -> None:
    raw_usd = os.getenv("FW_SDG_REVIEW_USD", "").strip()
    raw_receipt = os.getenv("FW_SDG_REVIEW_OPENED_RECEIPT", "").strip()
    raw_pending = os.getenv("FW_SDG_REVIEW_PENDING_RECEIPT", "").strip()
    raw_build = os.getenv("FW_SDG_REVIEW_BUILD_RECEIPT", "").strip()
    zone_id = os.getenv("FW_SDG_REVIEW_ZONE", "").strip()
    if not raw_usd or not raw_receipt or not zone_id:
        raise RuntimeError("review editor launch is missing its bounded environment")
    usd_path = Path(raw_usd).resolve()
    receipt_path = Path(raw_receipt).resolve()
    pending_path = Path(raw_pending).resolve() if raw_pending else None
    build_receipt_path = (
        Path(raw_build).resolve()
        if raw_build
        else usd_path.parent / "build-receipt.json"
    )
    if not usd_path.is_file():
        raise RuntimeError(f"review root USD is absent: {usd_path}")
    if pending_path is not None and not pending_path.is_file():
        raise RuntimeError(f"pending review receipt is absent: {pending_path}")
    if not build_receipt_path.is_file():
        raise RuntimeError(f"scene build receipt is absent: {build_receipt_path}")

    target_fps = _bounded_int(
        "FW_OMNI_EDITOR_TARGET_FPS", 60, minimum=10, maximum=60
    )
    expected_tile_count = _bounded_int(
        "FW_OMNI_EDITOR_EXPECTED_TILE_COUNT", 400, minimum=1, maximum=4_096
    )
    hero_cap = _bounded_int(
        "FW_OMNI_EDITOR_DETAIL_TILE_CAP", 48, minimum=1, maximum=96
    )
    hero_guard_minimum = _bounded_int(
        "FW_OMNI_EDITOR_DETAIL_GUARD_MIN_TILES",
        16,
        minimum=1,
        maximum=96,
    )
    lod0_cap = _bounded_int(
        "FW_OMNI_EDITOR_LOD0_TILE_CAP", 12, minimum=0, maximum=48
    )
    lod1_cap = _bounded_int(
        "FW_OMNI_EDITOR_LOD1_TILE_CAP", 64, minimum=1, maximum=160
    )
    lod2_cap = _bounded_int(
        "FW_OMNI_EDITOR_LOD2_TILE_CAP", 196, minimum=1, maximum=400
    )
    detail_transition_cap = _bounded_int(
        "FW_OMNI_EDITOR_DETAIL_TRANSITIONS_PER_TICK",
        8,
        minimum=1,
        maximum=32,
    )
    lod_transition_cap = _bounded_int(
        "FW_OMNI_EDITOR_LOD_TRANSITIONS_PER_TICK",
        32,
        minimum=1,
        maximum=96,
    )
    if lod1_cap > lod2_cap:
        raise RuntimeError("LOD1 tile cap must not exceed the LOD2 tile cap")
    if hero_guard_minimum > hero_cap:
        raise RuntimeError(
            "HERO guard minimum must not exceed the HERO tile cap"
        )
    settle_maximum_updates = _bounded_int(
        "FW_OMNI_EDITOR_SETTLE_MAX_UPDATES",
        3_600,
        minimum=300,
        maximum=18_000,
    )

    build_receipt, coverage_by_ref = _read_build_contract(
        receipt_path=build_receipt_path,
        usd_path=usd_path,
        zone_id=zone_id,
        expected_tile_count=expected_tile_count,
    )
    root_hash_before = _sha256(usd_path)
    settings = carb.settings.get_settings()
    settings.set("/app/runLoops/main/rateLimitEnabled", True)
    settings.set("/app/runLoops/main/rateLimitFrequency", target_fps)
    await omni.kit.app.get_app().next_update_async()

    context = omni.usd.get_context()
    result, error = await context.open_stage_async(
        str(usd_path),
        load_set=omni.usd.UsdContextInitialLoadSet.LOAD_NONE,
    )
    if not result:
        raise RuntimeError(f"FireViewer USD Composer could not open review USD: {error}")
    for _ in range(4):
        await omni.kit.app.get_app().next_update_async()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Composer did not expose the opened USD stage")
    # Keep subsequent Composer interactions on the anonymous session layer as
    # well; a visual review must not dirty the packaged root by default.
    stage.SetEditTarget(stage.GetSessionLayer())
    tiles = _discover_tile_headers(stage, coverage_by_ref)

    # Author LOD3 opinions before loading terrain.  This avoids transiently
    # composing the default LOD1 mesh for all 400 tiles during initial open.
    current_lods: dict[str, str] = {}
    current_collision_lods: dict[str, str] = {}
    for index, tile in enumerate(tiles, start=1):
        _session_select_lod(stage, tile.terrain_path, DISTANT_LOD)
        current_lods[tile.terrain_path] = DISTANT_LOD
        _session_select_collision(stage, tile.terrain_path, "FAR")
        current_collision_lods[tile.tile_ref] = "FAR"
        if index % 64 == 0:
            carb.log_info(
                "FireViewer distant terrain LOD selection: "
                f"{index}/{len(tiles)} tiles"
            )
            await omni.kit.app.get_app().next_update_async()
    await _load_paths(
        stage,
        (tile.terrain_path for tile in tiles),
        batch_size=16,
        label="terrain",
    )
    for tile in tiles:
        terrain = stage.GetPrimAtPath(tile.terrain_path)
        if not terrain.IsValid() or not terrain.IsLoaded():
            raise RuntimeError(f"{tile.tile_ref} terrain did not load")
        variants = terrain.GetVariantSets().GetVariantSet(LOD_VARIANT_SET)
        names = {str(name) for name in variants.GetVariantNames()}
        if not set(tile.terrain_lods).issubset(names):
            raise RuntimeError(
                f"{tile.tile_ref} terrain variants disagree with the build receipt"
            )
        collision_variants = terrain.GetVariantSets().GetVariantSet(
            COLLISION_VARIANT_SET
        )
        collision_names = {
            str(name) for name in collision_variants.GetVariantNames()
        }
        if collision_names != set(tile.collision_lods):
            raise RuntimeError(
                f"{tile.tile_ref} collision variants disagree with the build receipt"
            )
        collision = stage.GetPrimAtPath(f"{tile.terrain_path}/Collision")
        if not collision.IsValid():
            raise RuntimeError(
                f"{tile.tile_ref} selected FAR collision representation is absent"
            )
    bounds_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        True,
    )
    terrain_elevation: dict[str, tuple[float, float, float]] = {}
    for tile in tiles:
        aligned = bounds_cache.ComputeWorldBound(
            stage.GetPrimAtPath(tile.terrain_path)
        ).ComputeAlignedRange()
        if aligned.IsEmpty():
            raise RuntimeError(f"{tile.tile_ref} loaded terrain has no world bounds")
        minimum = aligned.GetMin()
        maximum = aligned.GetMax()
        minimum_z = float(minimum[2])
        maximum_z = float(maximum[2])
        ground = (minimum_z + maximum_z) * 0.5
        if (
            not math.isfinite(minimum_z)
            or not math.isfinite(maximum_z)
            or maximum_z < minimum_z
        ):
            raise RuntimeError(f"{tile.tile_ref} loaded terrain bounds are not finite")
        terrain_elevation[tile.tile_ref] = (ground, minimum_z, maximum_z)
    tiles = [
        replace(
            tile,
            ground_z=terrain_elevation[tile.tile_ref][0],
            minimum_z=terrain_elevation[tile.tile_ref][1],
            maximum_z=terrain_elevation[tile.tile_ref][2],
        )
        for tile in tiles
    ]
    # FAR is deliberately lightweight and provides uninterrupted detail
    # coverage for every tile before the review camera is shown.  Higher levels
    # replace it per tile only after their payload has composed.
    await _load_paths(
        stage,
        (tile.far_detail_path for tile in tiles),
        batch_size=32,
        label="FAR detail coverage",
    )
    active_detail_levels = {tile.tile_ref: "FAR" for tile in tiles}
    active_detail_paths = {
        tile.tile_ref: tile.far_detail_path for tile in tiles
    }

    scene_xmin = min(tile.bounds[0] for tile in tiles)
    scene_ymin = min(tile.bounds[1] for tile in tiles)
    scene_xmax = max(tile.bounds[2] for tile in tiles)
    scene_ymax = max(tile.bounds[3] for tile in tiles)
    scene_span = max(scene_xmax - scene_xmin, scene_ymax - scene_ymin)
    camera_path, ground_z = _create_session_camera(
        stage=stage,
        source_path=DEFAULT_REVIEW_CAMERA_PATH,
        scene_span=scene_span,
    )
    viewport = omni.kit.viewport.utility.get_active_viewport()
    if viewport is None:
        raise RuntimeError("FireViewer USD Composer has no active viewport")
    viewport.camera_path = camera_path
    await omni.kit.app.get_app().next_update_async()

    view = _camera_view_for_tiles(
        stage,
        camera_path=camera_path,
        fallback_ground_z=ground_z,
        tiles=tiles,
    )
    initial_plan = _plan_working_set(
        tiles=tiles,
        view=view,
        hero_cap=hero_cap,
        hero_guard_minimum=hero_guard_minimum,
        lod0_cap=lod0_cap,
        lod1_cap=lod1_cap,
        lod2_cap=lod2_cap,
    )
    await _apply_plan(
        context=context,
        stage=stage,
        plan=initial_plan,
        current_lods=current_lods,
        current_collision_lods=current_collision_lods,
        active_detail_levels=active_detail_levels,
        active_detail_paths=active_detail_paths,
        detail_transition_cap=detail_transition_cap,
        detail_settle_maximum_updates=settle_maximum_updates,
        lod_transition_cap=lod_transition_cap,
        initial=True,
    )
    settled_updates = await _wait_for_stage_settle(
        context,
        maximum_updates=settle_maximum_updates,
        stable_updates=8,
    )
    if _sha256(usd_path) != root_hash_before:
        raise RuntimeError("root USD changed while Composer prepared the review session")

    controller = asyncio.ensure_future(
        _stream_camera_working_set(
            context=context,
            stage=stage,
            viewport=viewport,
            tiles=tiles,
            fallback_camera_path=camera_path,
            ground_z=ground_z,
            hero_cap=hero_cap,
            hero_guard_minimum=hero_guard_minimum,
            lod0_cap=lod0_cap,
            lod1_cap=lod1_cap,
            lod2_cap=lod2_cap,
            detail_transition_cap=detail_transition_cap,
            detail_settle_maximum_updates=min(settle_maximum_updates, 600),
            lod_transition_cap=lod_transition_cap,
            active_detail_levels=active_detail_levels,
            active_detail_paths=active_detail_paths,
            current_lods=current_lods,
            current_collision_lods=current_collision_lods,
            reevaluation_updates=target_fps,
        )
    )
    controller.add_done_callback(
        lambda task: _task_result(task, label="FireViewer camera working-set controller")
    )

    lod_counts: dict[str, int] = {}
    for lod in current_lods.values():
        lod_counts[lod] = lod_counts.get(lod, 0) + 1
    detail_level_counts: dict[str, int] = {}
    for level in active_detail_levels.values():
        detail_level_counts[level] = detail_level_counts.get(level, 0) + 1
    collision_lod_counts: dict[str, int] = {}
    for collision_lod in current_collision_lods.values():
        collision_lod_counts[collision_lod] = (
            collision_lod_counts.get(collision_lod, 0) + 1
        )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "zone_id": zone_id,
                "opened_at": datetime.now(UTC).isoformat(),
                "editor": "fireviewer_usd_composer",
                "root_usd": str(usd_path),
                "root_usd_sha256": root_hash_before,
                "build_receipt": str(build_receipt_path),
                "build_receipt_sha256": _sha256(build_receipt_path),
                "build_schema_version": build_receipt.get("schema_version"),
                "pending_review_sha256": (
                    _sha256(pending_path) if pending_path is not None else None
                ),
                "target_fps": target_fps,
                "viewport_camera": camera_path,
                "session_layer_only_edits": True,
                "source_layers_saved": False,
                "terrain_delivery": "all_declared_tiles_loaded",
                "terrain_loaded_tile_count": len(tiles),
                "terrain_lod_counts": lod_counts,
                "collision_delivery": (
                    "session_NEAR_for_HERO_MID_and_FAR_elsewhere"
                ),
                "collision_lod_counts": collision_lod_counts,
                "detail_delivery": "exactly_one_HERO_MID_or_FAR_level_per_tile",
                "detail_delivered_tile_count": len(tiles),
                "initial_active_detail_tile_count": len(active_detail_levels),
                "initial_detail_level_counts": detail_level_counts,
                "initial_visible_tile_count": len(initial_plan.visible_tile_refs),
                "hero_detail_tile_cap": hero_cap,
                "hero_guard_minimum_tiles": hero_guard_minimum,
                "settled_updates": settled_updates,
                "editor_load_policy": (
                    "LOAD_NONE_then_all_terrain_distance_lod_"
                    "plus_gapless_HERO_MID_FAR_detail_working_set"
                ),
                "state": "opened_for_human_review",
                "human_review": "pending",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt_path)


def _schedule_open() -> None:
    task = asyncio.ensure_future(_open_review_stage())
    task.add_done_callback(
        lambda completed: _task_result(
            completed, label="FireViewer progressive review open"
        )
    )


if os.getenv("FW_SDG_REVIEW_DISABLE_AUTOSTART", "").strip() != "1":
    _schedule_open()

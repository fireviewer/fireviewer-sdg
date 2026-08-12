"""Deterministic, fail-closed composition of four base scenes into 20 variants.

The module deliberately contains no USD, filesystem, network or renderer code.
It transforms already accepted scene data while preserving the terrain, water
features, stable identifiers, asset references and exact per-family counts.
Consumers can therefore author the returned immutable scene into OpenUSD
without having to reproduce any placement logic.

The placement contract is intentionally strict:

* every tree and building keeps its real USD asset reference;
* roads are warped as connected, terrain-draped networks;
* trees respect habitat, soil, slope, water, road and building buffers;
* buildings are placed as road-accessible groups on viable foundations;
* water courses keep their accepted geometry and must follow the terrain;
* all five variants of each of the four bases must be measurably distinct.

No constraint is silently relaxed.  An impossible scene raises
``SceneVariantError`` and produces no partial portfolio.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Sequence


ALGORITHM_ID = "fireviewer-photoreal-scene-variants-v1"
BASE_SCENE_COUNT = 4
VARIANTS_PER_BASE = 5
PORTFOLIO_SCENE_COUNT = BASE_SCENE_COUNT * VARIANTS_PER_BASE
_EPSILON = 1.0e-9
_PRIMITIVE_ASSET = re.compile(
    r"(?:^|[/_.-])(cube|cone|cylinder|sphere|capsule|primitive|placeholder)"
    r"(?:$|[/_.-])",
    re.IGNORECASE,
)
_USD_SUFFIXES = (".usd", ".usda", ".usdc")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SceneVariantError(RuntimeError):
    """Raised when a requested composition cannot satisfy its contract."""


class InvalidBaseScene(ValueError):
    """Raised when an accepted base scene is structurally unsuitable."""


@dataclass(frozen=True, slots=True)
class Vec2:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("2D coordinates must be finite")


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x, self.y, self.z)):
            raise ValueError("3D coordinates must be finite")

    @property
    def xy(self) -> Vec2:
        return Vec2(self.x, self.y)


@dataclass(frozen=True, slots=True)
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.min_x, self.min_y, self.max_x, self.max_y)
        ):
            raise ValueError("scene bounds must be finite")
        if self.max_x <= self.min_x or self.max_y <= self.min_y:
            raise ValueError("scene bounds must have positive area")

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def centre(self) -> Vec2:
        return Vec2(
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
        )

    def contains(self, point: Vec2, margin: float = 0.0) -> bool:
        return (
            self.min_x + margin <= point.x <= self.max_x - margin
            and self.min_y + margin <= point.y <= self.max_y - margin
        )


@dataclass(frozen=True, slots=True)
class HeightField:
    """Regular terrain samples used for elevation, slope and road draping."""

    origin_x: float
    origin_y: float
    spacing_m: float
    samples: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.origin_x)
            or not math.isfinite(self.origin_y)
            or not math.isfinite(self.spacing_m)
            or self.spacing_m <= 0.0
        ):
            raise ValueError("height field origin and spacing must be valid")
        if len(self.samples) < 2 or len(self.samples[0]) < 2:
            raise ValueError("height field needs at least a 2 by 2 sample grid")
        width = len(self.samples[0])
        for row in self.samples:
            if len(row) != width:
                raise ValueError("height field rows must have a constant width")
            if any(not math.isfinite(value) for value in row):
                raise ValueError("height field samples must be finite")

    @property
    def width(self) -> int:
        return len(self.samples[0])

    @property
    def height(self) -> int:
        return len(self.samples)

    @property
    def bounds(self) -> Bounds:
        return Bounds(
            self.origin_x,
            self.origin_y,
            self.origin_x + (self.width - 1) * self.spacing_m,
            self.origin_y + (self.height - 1) * self.spacing_m,
        )

    def elevation(self, point: Vec2) -> float:
        """Return a bilinearly interpolated terrain elevation."""

        u = (point.x - self.origin_x) / self.spacing_m
        v = (point.y - self.origin_y) / self.spacing_m
        if u < -_EPSILON or v < -_EPSILON:
            raise SceneVariantError("terrain query falls outside the height field")
        if u > self.width - 1 + _EPSILON or v > self.height - 1 + _EPSILON:
            raise SceneVariantError("terrain query falls outside the height field")
        u = min(max(u, 0.0), self.width - 1.0)
        v = min(max(v, 0.0), self.height - 1.0)
        x0 = min(int(math.floor(u)), self.width - 2)
        y0 = min(int(math.floor(v)), self.height - 2)
        fx = u - x0
        fy = v - y0
        z00 = self.samples[y0][x0]
        z10 = self.samples[y0][x0 + 1]
        z01 = self.samples[y0 + 1][x0]
        z11 = self.samples[y0 + 1][x0 + 1]
        return (
            z00 * (1.0 - fx) * (1.0 - fy)
            + z10 * fx * (1.0 - fy)
            + z01 * (1.0 - fx) * fy
            + z11 * fx * fy
        )

    def slope_percent(self, point: Vec2) -> float:
        """Return the local gradient magnitude as a percentage."""

        step = self.spacing_m
        x0 = max(self.origin_x, point.x - step)
        x1 = min(self.bounds.max_x, point.x + step)
        y0 = max(self.origin_y, point.y - step)
        y1 = min(self.bounds.max_y, point.y + step)
        if x1 - x0 <= _EPSILON or y1 - y0 <= _EPSILON:
            raise SceneVariantError("height field is too small for a slope query")
        dz_dx = (
            self.elevation(Vec2(x1, point.y))
            - self.elevation(Vec2(x0, point.y))
        ) / (x1 - x0)
        dz_dy = (
            self.elevation(Vec2(point.x, y1))
            - self.elevation(Vec2(point.x, y0))
        ) / (y1 - y0)
        return math.hypot(dz_dx, dz_dy) * 100.0


@dataclass(frozen=True, slots=True)
class PlacementHeightTile:
    """One hash-bound high-resolution terrain tile loaded on demand."""

    tile_ref: str
    bounds: Bounds
    x_coordinates: tuple[float, ...]
    y_coordinates: tuple[float, ...]
    sample_sha256: str
    sample_loader: Callable[[], Sequence[Sequence[float]]] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_stable_id(self.tile_ref, label="placement height tile")
        if (
            len(self.x_coordinates) < 2
            or len(self.y_coordinates) < 2
            or any(
                not math.isfinite(value)
                for value in (*self.x_coordinates, *self.y_coordinates)
            )
            or any(
                second <= first
                for first, second in zip(
                    self.x_coordinates, self.x_coordinates[1:]
                )
            )
            or any(
                second <= first
                for first, second in zip(
                    self.y_coordinates, self.y_coordinates[1:]
                )
            )
        ):
            raise ValueError(
                f"placement height tile {self.tile_ref} has invalid axes"
            )
        if not all(
            math.isclose(actual, expected, abs_tol=0.01, rel_tol=0.0)
            for actual, expected in (
                (self.x_coordinates[0], self.bounds.min_x),
                (self.x_coordinates[-1], self.bounds.max_x),
                (self.y_coordinates[0], self.bounds.min_y),
                (self.y_coordinates[-1], self.bounds.max_y),
            )
        ):
            raise ValueError(
                f"placement height tile {self.tile_ref} axes/bounds diverge"
            )
        if not _SHA256.fullmatch(self.sample_sha256):
            raise ValueError(
                f"placement height tile {self.tile_ref} lacks sample SHA-256"
            )
        if not callable(self.sample_loader):
            raise ValueError(
                f"placement height tile {self.tile_ref} has no sample loader"
            )


@dataclass(slots=True)
class TiledHeightField:
    """High-resolution terrain authority with a bounded tile sample cache.

    The tile loader is supplied by the interchange consumer.  This keeps the
    composition algorithm independent of USD and file formats while ensuring
    that placement, slopes and foundation relief never use the coarse global
    overview grid.
    """

    tiles: tuple[PlacementHeightTile, ...]
    content_fingerprint: str
    expected_tile_count: int = 400
    cache_tile_limit: int = 2
    _by_origin: dict[tuple[float, float], PlacementHeightTile] = field(
        init=False,
        repr=False,
    )
    _x_starts: tuple[float, ...] = field(init=False, repr=False)
    _y_starts: tuple[float, ...] = field(init=False, repr=False)
    _bounds: Bounds = field(init=False, repr=False)
    _cache: OrderedDict[str, tuple[tuple[float, ...], ...]] = field(
        init=False,
        repr=False,
    )
    _lock: threading.RLock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.expected_tile_count, bool)
            or self.expected_tile_count < 1
            or len(self.tiles) != self.expected_tile_count
        ):
            raise ValueError(
                "tiled height field does not contain the exact expected tile count"
            )
        if (
            isinstance(self.cache_tile_limit, bool)
            or not 1 <= self.cache_tile_limit <= 8
        ):
            raise ValueError(
                "tiled height field cache must hold between one and eight tiles"
            )
        if not _SHA256.fullmatch(self.content_fingerprint):
            raise ValueError(
                "tiled height field requires a canonical content fingerprint"
            )
        if len({tile.tile_ref for tile in self.tiles}) != len(self.tiles):
            raise ValueError("placement height tile refs repeat")
        self._x_starts = tuple(
            sorted({tile.bounds.min_x for tile in self.tiles})
        )
        self._y_starts = tuple(
            sorted({tile.bounds.min_y for tile in self.tiles})
        )
        self._by_origin = {
            (tile.bounds.min_x, tile.bounds.min_y): tile
            for tile in self.tiles
        }
        if len(self._by_origin) != len(self.tiles) or (
            len(self._x_starts) * len(self._y_starts) != len(self.tiles)
        ):
            raise ValueError(
                "placement height tiles do not form a complete lattice"
            )
        self._bounds = Bounds(
            min(tile.bounds.min_x for tile in self.tiles),
            min(tile.bounds.min_y for tile in self.tiles),
            max(tile.bounds.max_x for tile in self.tiles),
            max(tile.bounds.max_y for tile in self.tiles),
        )
        total_area = sum(
            tile.bounds.width * tile.bounds.height for tile in self.tiles
        )
        if not math.isclose(
            total_area,
            self._bounds.width * self._bounds.height,
            abs_tol=1.0e-3,
            rel_tol=0.0,
        ):
            raise ValueError(
                "placement height tiles do not cover the complete terrain"
            )
        for x_index, x in enumerate(self._x_starts):
            for y_index, y in enumerate(self._y_starts):
                tile = self._by_origin.get((x, y))
                if tile is None:
                    raise ValueError(
                        "placement height tile lattice contains a gap"
                    )
                expected_max_x = (
                    self._x_starts[x_index + 1]
                    if x_index + 1 < len(self._x_starts)
                    else self._bounds.max_x
                )
                expected_max_y = (
                    self._y_starts[y_index + 1]
                    if y_index + 1 < len(self._y_starts)
                    else self._bounds.max_y
                )
                if not (
                    math.isclose(
                        tile.bounds.max_x,
                        expected_max_x,
                        abs_tol=0.01,
                        rel_tol=0.0,
                    )
                    and math.isclose(
                        tile.bounds.max_y,
                        expected_max_y,
                        abs_tol=0.01,
                        rel_tol=0.0,
                    )
                ):
                    raise ValueError(
                        f"placement height tile {tile.tile_ref} overlaps or gaps"
                    )
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    @property
    def origin_x(self) -> float:
        return self._bounds.min_x

    @property
    def origin_y(self) -> float:
        return self._bounds.min_y

    @property
    def cached_tile_count(self) -> int:
        with self._lock:
            return len(self._cache)

    def _tile_for(self, point: Vec2) -> PlacementHeightTile:
        if not self._bounds.contains(point):
            raise SceneVariantError(
                "terrain query falls outside the tiled height field"
            )
        import bisect

        x_index = min(
            max(bisect.bisect_right(self._x_starts, point.x) - 1, 0),
            len(self._x_starts) - 1,
        )
        y_index = min(
            max(bisect.bisect_right(self._y_starts, point.y) - 1, 0),
            len(self._y_starts) - 1,
        )
        tile = self._by_origin.get(
            (self._x_starts[x_index], self._y_starts[y_index])
        )
        if tile is None:
            raise SceneVariantError(
                "terrain query falls in an uncovered placement tile"
            )
        return tile

    def _samples(
        self, tile: PlacementHeightTile
    ) -> tuple[tuple[float, ...], ...]:
        with self._lock:
            cached = self._cache.get(tile.tile_ref)
            if cached is not None:
                self._cache.move_to_end(tile.tile_ref)
                return cached
            raw = tile.sample_loader()
            if (
                len(raw) != len(tile.y_coordinates)
                or any(len(row) != len(tile.x_coordinates) for row in raw)
            ):
                raise SceneVariantError(
                    f"placement height tile {tile.tile_ref} sample shape changed"
                )
            samples = tuple(
                tuple(float(value) for value in row) for row in raw
            )
            if any(
                not math.isfinite(value)
                for row in samples
                for value in row
            ):
                raise SceneVariantError(
                    f"placement height tile {tile.tile_ref} has non-finite samples"
                )
            self._cache[tile.tile_ref] = samples
            self._cache.move_to_end(tile.tile_ref)
            while len(self._cache) > self.cache_tile_limit:
                self._cache.popitem(last=False)
            return samples

    def elevation(self, point: Vec2) -> float:
        tile = self._tile_for(point)
        samples = self._samples(tile)
        xs = tile.x_coordinates
        ys = tile.y_coordinates
        import bisect

        x = min(max(point.x, xs[0]), xs[-1])
        y = min(max(point.y, ys[0]), ys[-1])
        column = min(
            max(bisect.bisect_right(xs, x) - 1, 0), len(xs) - 2
        )
        row = min(
            max(bisect.bisect_right(ys, y) - 1, 0), len(ys) - 2
        )
        x0, x1 = xs[column], xs[column + 1]
        y0, y1 = ys[row], ys[row + 1]
        fx = (x - x0) / (x1 - x0)
        fy = (y - y0) / (y1 - y0)
        z00 = samples[row][column]
        z10 = samples[row][column + 1]
        z01 = samples[row + 1][column]
        z11 = samples[row + 1][column + 1]
        return (
            z00 * (1.0 - fx) * (1.0 - fy)
            + z10 * fx * (1.0 - fy)
            + z01 * (1.0 - fx) * fy
            + z11 * fx * fy
        )

    def slope_percent(self, point: Vec2) -> float:
        tile = self._tile_for(point)
        xs = tile.x_coordinates
        ys = tile.y_coordinates
        local_step = min(
            min(second - first for first, second in zip(xs, xs[1:])),
            min(second - first for first, second in zip(ys, ys[1:])),
        )
        x0 = max(self._bounds.min_x, point.x - local_step)
        x1 = min(self._bounds.max_x, point.x + local_step)
        y0 = max(self._bounds.min_y, point.y - local_step)
        y1 = min(self._bounds.max_y, point.y + local_step)
        if x1 - x0 <= _EPSILON or y1 - y0 <= _EPSILON:
            raise SceneVariantError(
                "tiled height field is too small for a slope query"
            )
        dz_dx = (
            self.elevation(Vec2(x1, point.y))
            - self.elevation(Vec2(x0, point.y))
        ) / (x1 - x0)
        dz_dy = (
            self.elevation(Vec2(point.x, y1))
            - self.elevation(Vec2(point.x, y0))
        ) / (y1 - y0)
        return math.hypot(dz_dx, dz_dy) * 100.0


@dataclass(frozen=True, slots=True)
class SuitabilityZone:
    """Land-cover/soil polygon used by tree and foundation placement."""

    stable_id: str
    outline: tuple[Vec2, ...]
    biome: str
    soil: str
    tree_families: frozenset[str]
    buildable: bool = False

    def __post_init__(self) -> None:
        _require_stable_id(self.stable_id, label="suitability zone")
        _validate_polygon(self.outline, label=f"suitability zone {self.stable_id}")
        if not self.biome.strip() or not self.soil.strip():
            raise ValueError("suitability zones require biome and soil labels")
        if any(not family.strip() for family in self.tree_families):
            raise ValueError("tree family labels must be non-empty")


@dataclass(frozen=True, slots=True)
class FamilySuitability:
    family: str
    allowed_biomes: frozenset[str]
    allowed_soils: frozenset[str]

    def __post_init__(self) -> None:
        if (
            not self.family.strip()
            or not self.allowed_biomes
            or not self.allowed_soils
        ):
            raise ValueError("family suitability must define family, biome and soil")


@dataclass(frozen=True, slots=True)
class GroundSurfaceContract:
    """Object-free ground appearance required before moving scene objects."""

    kind: str
    material_ref: str
    content_fingerprint: str
    removed_object_classes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.kind not in {"object_free_pbr", "object_removed_orthomosaic"}:
            raise ValueError(
                "ground surface must be object-free PBR or an object-removed "
                "orthomosaic; raw orthophotos are forbidden"
            )
        if not self.material_ref.strip() or self.material_ref != self.material_ref.strip():
            raise ValueError("ground surface material reference is required")
        if not _SHA256.fullmatch(self.content_fingerprint):
            raise ValueError("ground surface requires a lowercase SHA-256 fingerprint")
        if self.kind == "object_removed_orthomosaic":
            required = {"trees", "buildings", "routes"}
            if not required.issubset(self.removed_object_classes):
                raise ValueError(
                    "object-removed orthomosaic must remove trees, buildings and routes"
                )
        elif self.removed_object_classes:
            raise ValueError(
                "object-free PBR must not claim orthomosaic object removals"
            )


@dataclass(frozen=True, slots=True)
class BridgeSpan:
    """Normalized route interval containing approaches and a water deck."""

    stable_id: str
    start_fraction: float
    water_start_fraction: float
    water_end_fraction: float
    end_fraction: float
    minimum_deck_clearance_m: float

    def __post_init__(self) -> None:
        _require_stable_id(self.stable_id, label="bridge span")
        values = (
            self.start_fraction,
            self.water_start_fraction,
            self.water_end_fraction,
            self.end_fraction,
            self.minimum_deck_clearance_m,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("bridge span values must be finite")
        if not (
            0.0
            <= self.start_fraction
            < self.water_start_fraction
            <= self.water_end_fraction
            < self.end_fraction
            <= 1.0
        ):
            raise ValueError(
                "bridge fractions must order approach, water deck and exit in [0, 1]"
            )
        if self.minimum_deck_clearance_m <= 0.0:
            raise ValueError("bridge deck clearance must be positive")


@dataclass(frozen=True, slots=True)
class SceneAsset:
    """A real USD-backed tree or building instance."""

    stable_id: str
    family: str
    asset_ref: str
    position: Vec3
    heading_degrees: float
    uniform_scale: float
    footprint_radius_m: float
    group_id: str = ""

    def __post_init__(self) -> None:
        _require_stable_id(self.stable_id, label="scene asset")
        if not self.family.strip():
            raise ValueError("scene assets require a family")
        _validate_asset_ref(self.asset_ref)
        if not math.isfinite(self.heading_degrees):
            raise ValueError("asset heading must be finite")
        if not math.isfinite(self.uniform_scale) or self.uniform_scale <= 0.0:
            raise ValueError("asset scale must be positive")
        if (
            not math.isfinite(self.footprint_radius_m)
            or self.footprint_radius_m <= 0.0
        ):
            raise ValueError("asset footprint radius must be positive")


@dataclass(frozen=True, slots=True)
class SceneRoute:
    stable_id: str
    family: str
    points: tuple[Vec3, ...]
    width_m: float
    bridge_spans: tuple[BridgeSpan, ...] = ()

    def __post_init__(self) -> None:
        _require_stable_id(self.stable_id, label="route")
        if not self.family.strip() or len(self.points) < 2:
            raise ValueError("routes require a family and at least two points")
        if not math.isfinite(self.width_m) or self.width_m <= 0.0:
            raise ValueError("route width must be positive")
        if all(
            _distance(first.xy, second.xy) <= _EPSILON
            for first, second in zip(self.points, self.points[1:])
        ):
            raise ValueError("route geometry must have non-zero length")
        bridge_ids = [span.stable_id for span in self.bridge_spans]
        if len(bridge_ids) != len(set(bridge_ids)):
            raise ValueError("route bridge span stable IDs must be unique")
        ordered = sorted(self.bridge_spans, key=lambda span: span.start_fraction)
        if any(
            first.end_fraction > second.start_fraction
            for first, second in zip(ordered, ordered[1:])
        ):
            raise ValueError("route bridge spans must not overlap")


@dataclass(frozen=True, slots=True)
class WaterFeature:
    stable_id: str
    family: str
    outline: tuple[Vec2, ...]
    kind: str
    centreline: tuple[Vec2, ...] = ()
    surface_profile_m: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _require_stable_id(self.stable_id, label="water feature")
        if not self.family.strip():
            raise ValueError("water features require a family")
        _validate_polygon(self.outline, label=f"water feature {self.stable_id}")
        if self.kind not in {"standing", "watercourse"}:
            raise ValueError("water kind must be 'standing' or 'watercourse'")
        if self.kind == "watercourse" and len(self.centreline) < 2:
            raise ValueError("watercourses require an upstream-to-downstream centreline")
        if self.kind == "standing" and self.centreline:
            raise ValueError("standing water must not declare a drainage centreline")
        expected_profile = len(self.centreline) if self.kind == "watercourse" else 1
        if len(self.surface_profile_m) != expected_profile:
            raise ValueError(
                f"{self.kind} water surface profile must contain "
                f"{expected_profile} elevations"
            )
        if any(not math.isfinite(value) for value in self.surface_profile_m):
            raise ValueError("water surface elevations must be finite")


@dataclass(frozen=True, slots=True)
class BaseScene:
    stable_id: str
    bounds: Bounds
    terrain: HeightField | TiledHeightField
    ground_surface: GroundSurfaceContract
    trees: tuple[SceneAsset, ...]
    buildings: tuple[SceneAsset, ...]
    routes: tuple[SceneRoute, ...]
    waters: tuple[WaterFeature, ...]
    suitability_zones: tuple[SuitabilityZone, ...]

    def __post_init__(self) -> None:
        _require_stable_id(self.stable_id, label="base scene")


@dataclass(frozen=True, slots=True)
class VariantConstraints:
    edge_buffer_m: float = 2.0
    tree_water_buffer_m: float = 3.0
    building_water_buffer_m: float = 8.0
    road_water_buffer_m: float = 0.5
    tree_road_buffer_m: float = 1.5
    tree_building_buffer_m: float = 2.0
    minimum_tree_spacing_m: float = 0.5
    minimum_building_spacing_m: float = 2.0
    maximum_tree_slope_percent: float = 65.0
    maximum_building_slope_percent: float = 18.0
    maximum_road_grade_percent: float = 18.0
    maximum_foundation_relief_m: float = 0.75
    maximum_building_road_distance_m: float = 30.0
    maximum_building_group_radius_m: float = 250.0
    building_group_spacing_m: float = 22.0
    building_front_setback_m: float = 3.0
    road_sample_spacing_m: float = 20.0
    route_warp_amplitude_m: float = 30.0
    route_rotation_degrees: float = 5.0
    route_scale_delta: float = 0.015
    tree_relocation_radius_m: float = 100.0
    tree_cluster_cell_m: float = 25.0
    maximum_tree_density_per_hectare: float = 3_000.0
    minimum_moved_fraction: float = 0.75
    movement_epsilon_m: float = 2.0
    minimum_tree_intervariant_distance_m: float = 3.0
    minimum_building_intervariant_distance_m: float = 5.0
    minimum_route_intervariant_distance_m: float = 1.5
    water_uphill_tolerance_m: float = 0.75
    road_connectivity_tolerance_m: float = 0.25
    minimum_bridge_deck_clearance_m: float = 3.0
    maximum_road_components: int = 1
    maximum_route_attempts: int = 128
    maximum_building_attempts: int = 256
    maximum_tree_attempts: int = 96
    maximum_variant_attempts: int = 24
    tree_suitability: tuple[FamilySuitability, ...] = ()

    def __post_init__(self) -> None:
        positive = {
            "tree_cluster_cell_m": self.tree_cluster_cell_m,
            "maximum_tree_density_per_hectare": (
                self.maximum_tree_density_per_hectare
            ),
            "maximum_building_road_distance_m": (
                self.maximum_building_road_distance_m
            ),
            "maximum_building_group_radius_m": (
                self.maximum_building_group_radius_m
            ),
            "building_group_spacing_m": self.building_group_spacing_m,
            "road_sample_spacing_m": self.road_sample_spacing_m,
            "tree_relocation_radius_m": self.tree_relocation_radius_m,
            "movement_epsilon_m": self.movement_epsilon_m,
            "road_connectivity_tolerance_m": self.road_connectivity_tolerance_m,
            "minimum_bridge_deck_clearance_m": (
                self.minimum_bridge_deck_clearance_m
            ),
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in positive.values()):
            raise ValueError("positive placement constraints must be finite")
        nonnegative = (
            self.edge_buffer_m,
            self.tree_water_buffer_m,
            self.building_water_buffer_m,
            self.road_water_buffer_m,
            self.tree_road_buffer_m,
            self.tree_building_buffer_m,
            self.minimum_tree_spacing_m,
            self.minimum_building_spacing_m,
            self.maximum_tree_slope_percent,
            self.maximum_building_slope_percent,
            self.maximum_road_grade_percent,
            self.maximum_foundation_relief_m,
            self.building_front_setback_m,
            self.route_warp_amplitude_m,
            self.route_rotation_degrees,
            self.route_scale_delta,
            self.minimum_tree_intervariant_distance_m,
            self.minimum_building_intervariant_distance_m,
            self.minimum_route_intervariant_distance_m,
            self.water_uphill_tolerance_m,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("placement constraints cannot be negative")
        if not 0.0 < self.minimum_moved_fraction <= 1.0:
            raise ValueError("minimum moved fraction must be in (0, 1]")
        if self.route_scale_delta >= 0.25:
            raise ValueError("route scale delta must remain below 0.25")
        if self.maximum_road_components < 1:
            raise ValueError("maximum road components must be positive")
        attempts = (
            self.maximum_route_attempts,
            self.maximum_building_attempts,
            self.maximum_tree_attempts,
            self.maximum_variant_attempts,
        )
        if any(value < 1 for value in attempts):
            raise ValueError("placement attempt limits must be positive")
        families = [entry.family for entry in self.tree_suitability]
        if len(families) != len(set(families)):
            raise ValueError("tree suitability families must be unique")


@dataclass(frozen=True, slots=True)
class DisplacementMetrics:
    count: int
    moved_fraction: float
    mean_distance_m: float
    maximum_distance_m: float


@dataclass(frozen=True, slots=True)
class TransformationContract:
    algorithm: str
    base_scene_id: str
    variant_id: str
    variant_index: int
    seed: int
    composition_attempt: int
    terrain_fingerprint: str
    source_fingerprint: str
    result_fingerprint: str
    stable_id_policy: str
    stable_id_hashes: tuple[tuple[str, str], ...]
    source_family_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    result_family_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    route_warp_parameters: tuple[tuple[str, float], ...]
    displacement: tuple[tuple[str, DisplacementMetrics], ...]


@dataclass(frozen=True, slots=True)
class SceneVariant:
    stable_id: str
    base_scene_id: str
    variant_index: int
    terrain: HeightField | TiledHeightField
    ground_surface: GroundSurfaceContract
    bounds: Bounds
    trees: tuple[SceneAsset, ...]
    buildings: tuple[SceneAsset, ...]
    routes: tuple[SceneRoute, ...]
    waters: tuple[WaterFeature, ...]
    suitability_zones: tuple[SuitabilityZone, ...]
    contract: TransformationContract


@dataclass(frozen=True, slots=True)
class _Warp:
    angle_radians: float
    scale: float
    phase_x: float
    phase_y: float
    amplitude_m: float
    translate_x: float
    translate_y: float

    def contract_values(self) -> tuple[tuple[str, float], ...]:
        return (
            ("angle_radians", self.angle_radians),
            ("scale", self.scale),
            ("phase_x", self.phase_x),
            ("phase_y", self.phase_y),
            ("amplitude_m", self.amplitude_m),
            ("translate_x", self.translate_x),
            ("translate_y", self.translate_y),
        )


class _PointGrid:
    """Small deterministic spatial hash for collision queries."""

    def __init__(self, cell_size: float) -> None:
        self._cell_size = max(cell_size, 0.05)
        self._cells: dict[tuple[int, int], list[tuple[Vec2, float]]] = defaultdict(list)

    def _key(self, point: Vec2) -> tuple[int, int]:
        return (
            math.floor(point.x / self._cell_size),
            math.floor(point.y / self._cell_size),
        )

    def add(self, point: Vec2, radius: float) -> None:
        self._cells[self._key(point)].append((point, radius))

    def overlaps(self, point: Vec2, radius: float, extra: float = 0.0) -> bool:
        reach = radius + extra + self._cell_size
        cells = max(1, math.ceil(reach / self._cell_size))
        key_x, key_y = self._key(point)
        for offset_x in range(-cells, cells + 1):
            for offset_y in range(-cells, cells + 1):
                for other, other_radius in self._cells.get(
                    (key_x + offset_x, key_y + offset_y), ()
                ):
                    if _distance(point, other) < (
                        radius + other_radius + extra - _EPSILON
                    ):
                        return True
        return False


class _RouteSpatialIndex:
    """Bounded-distance road lookup used by million-instance forests."""

    def __init__(
        self,
        routes: Sequence[SceneRoute],
        *,
        maximum_query_distance_m: float,
    ) -> None:
        self._cell_size = max(10.0, maximum_query_distance_m)
        self._segments: list[tuple[Vec2, Vec2, float, float]] = []
        self._cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for first, second, width in _route_segments(routes):
            heading = math.degrees(
                math.atan2(second.y - first.y, second.x - first.x)
            )
            index = len(self._segments)
            self._segments.append((first, second, width, heading))
            reach = maximum_query_distance_m + width * 0.5
            min_x = math.floor((min(first.x, second.x) - reach) / self._cell_size)
            max_x = math.floor((max(first.x, second.x) + reach) / self._cell_size)
            min_y = math.floor((min(first.y, second.y) - reach) / self._cell_size)
            max_y = math.floor((max(first.y, second.y) + reach) / self._cell_size)
            for cell_x in range(min_x, max_x + 1):
                for cell_y in range(min_y, max_y + 1):
                    self._cells[(cell_x, cell_y)].append(index)

    def nearest(self, point: Vec2) -> tuple[float, float]:
        key = (
            math.floor(point.x / self._cell_size),
            math.floor(point.y / self._cell_size),
        )
        best_distance = math.inf
        best_heading = 0.0
        for index in self._cells.get(key, ()):
            first, second, width, heading = self._segments[index]
            distance = max(
                0.0,
                _point_segment_distance(point, first, second) - width * 0.5,
            )
            if distance < best_distance:
                best_distance = distance
                best_heading = heading
        return best_distance, best_heading


def _require_stable_id(value: str, *, label: str) -> None:
    if not value.strip() or value != value.strip():
        raise ValueError(f"{label} stable ID must be non-empty and trimmed")


def _validate_asset_ref(asset_ref: str) -> None:
    value = asset_ref.strip()
    if not value or value != asset_ref:
        raise ValueError("asset references must be non-empty and trimmed")
    lowered = value.casefold()
    if _PRIMITIVE_ASSET.search(lowered):
        raise ValueError(f"procedural or placeholder asset is forbidden: {asset_ref}")
    clean = lowered.split("?", 1)[0].split("#", 1)[0]
    if not clean.endswith(_USD_SUFFIXES):
        raise ValueError(f"scene asset must reference a USD file: {asset_ref}")


def _validate_polygon(points: Sequence[Vec2], *, label: str) -> None:
    if len(points) < 3:
        raise ValueError(f"{label} needs at least three vertices")
    area = 0.0
    for first, second in zip(points, points[1:] + points[:1]):
        area += first.x * second.y - second.x * first.y
    if abs(area) <= _EPSILON:
        raise ValueError(f"{label} has zero area")


def _distance(first: Vec2, second: Vec2) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def _point_segment_distance(point: Vec2, first: Vec2, second: Vec2) -> float:
    dx = second.x - first.x
    dy = second.y - first.y
    length_squared = dx * dx + dy * dy
    if length_squared <= _EPSILON:
        return _distance(point, first)
    t = (
        (point.x - first.x) * dx + (point.y - first.y) * dy
    ) / length_squared
    t = min(max(t, 0.0), 1.0)
    return math.hypot(
        point.x - (first.x + t * dx),
        point.y - (first.y + t * dy),
    )


def _orientation(first: Vec2, second: Vec2, third: Vec2) -> float:
    return (
        (second.x - first.x) * (third.y - first.y)
        - (second.y - first.y) * (third.x - first.x)
    )


def _on_segment(first: Vec2, point: Vec2, second: Vec2, tolerance: float) -> bool:
    return (
        min(first.x, second.x) - tolerance
        <= point.x
        <= max(first.x, second.x) + tolerance
        and min(first.y, second.y) - tolerance
        <= point.y
        <= max(first.y, second.y) + tolerance
        and abs(_orientation(first, second, point)) <= tolerance
    )


def _segments_intersect(
    first_a: Vec2,
    first_b: Vec2,
    second_a: Vec2,
    second_b: Vec2,
    tolerance: float = _EPSILON,
) -> bool:
    o1 = _orientation(first_a, first_b, second_a)
    o2 = _orientation(first_a, first_b, second_b)
    o3 = _orientation(second_a, second_b, first_a)
    o4 = _orientation(second_a, second_b, first_b)
    if (o1 > tolerance and o2 < -tolerance or o1 < -tolerance and o2 > tolerance) and (
        o3 > tolerance and o4 < -tolerance or o3 < -tolerance and o4 > tolerance
    ):
        return True
    return any(
        (
            abs(value) <= tolerance
            and _on_segment(start, point, end, tolerance)
        )
        for value, start, point, end in (
            (o1, first_a, second_a, first_b),
            (o2, first_a, second_b, first_b),
            (o3, second_a, first_a, second_b),
            (o4, second_a, first_b, second_b),
        )
    )


def _segment_distance(
    first_a: Vec2, first_b: Vec2, second_a: Vec2, second_b: Vec2
) -> float:
    if _segments_intersect(first_a, first_b, second_a, second_b):
        return 0.0
    return min(
        _point_segment_distance(first_a, second_a, second_b),
        _point_segment_distance(first_b, second_a, second_b),
        _point_segment_distance(second_a, first_a, first_b),
        _point_segment_distance(second_b, first_a, first_b),
    )


def _point_in_polygon(point: Vec2, polygon: Sequence[Vec2]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_segment_distance(point, previous, current) <= _EPSILON:
            return True
        if (current.y > point.y) != (previous.y > point.y):
            intersect_x = (
                (previous.x - current.x)
                * (point.y - current.y)
                / (previous.y - current.y)
                + current.x
            )
            if point.x < intersect_x:
                inside = not inside
        previous = current
    return inside


def _point_polygon_distance(point: Vec2, polygon: Sequence[Vec2]) -> float:
    if _point_in_polygon(point, polygon):
        return 0.0
    return min(
        _point_segment_distance(point, first, second)
        for first, second in zip(polygon, polygon[1:] + polygon[:1])
    )


def _water_surface_elevation(water: WaterFeature, point: Vec2) -> float:
    if water.kind == "standing":
        return water.surface_profile_m[0]
    best_distance = math.inf
    best_elevation = water.surface_profile_m[0]
    for index, (first, second) in enumerate(
        zip(water.centreline, water.centreline[1:])
    ):
        dx = second.x - first.x
        dy = second.y - first.y
        length_squared = dx * dx + dy * dy
        if length_squared <= _EPSILON:
            continue
        t = (
            (point.x - first.x) * dx + (point.y - first.y) * dy
        ) / length_squared
        t = min(max(t, 0.0), 1.0)
        projected = Vec2(first.x + t * dx, first.y + t * dy)
        distance = _distance(point, projected)
        if distance < best_distance:
            best_distance = distance
            best_elevation = (
                water.surface_profile_m[index] * (1.0 - t)
                + water.surface_profile_m[index + 1] * t
            )
    return best_elevation


def _segment_polygon_distance(
    first: Vec2, second: Vec2, polygon: Sequence[Vec2]
) -> float:
    if _point_in_polygon(first, polygon) or _point_in_polygon(second, polygon):
        return 0.0
    return min(
        _segment_distance(first, second, edge_a, edge_b)
        for edge_a, edge_b in zip(polygon, polygon[1:] + polygon[:1])
    )


def _route_segments(routes: Sequence[SceneRoute]) -> Iterable[tuple[Vec2, Vec2, float]]:
    for route in routes:
        for first, second in zip(route.points, route.points[1:]):
            yield first.xy, second.xy, route.width_m


def _family_counts(scene: BaseScene | SceneVariant) -> tuple[
    tuple[str, tuple[tuple[str, int], ...]], ...
]:
    def count(values: Iterable[object]) -> tuple[tuple[str, int], ...]:
        return tuple(
            sorted(Counter(getattr(value, "family") for value in values).items())
        )

    return (
        ("trees", count(scene.trees)),
        ("buildings", count(scene.buildings)),
        ("routes", count(scene.routes)),
        ("waters", count(scene.waters)),
    )


def family_counts(
    scene: BaseScene | SceneVariant,
) -> dict[str, dict[str, int]]:
    """Return exact category/family counts as a consumer-friendly dictionary."""

    return {category: dict(values) for category, values in _family_counts(scene)}


def global_stable_id(base_scene_id: str, object_stable_id: str) -> str:
    """Return the portfolio-wide identity used by USD authoring consumers."""

    _require_stable_id(base_scene_id, label="base scene")
    _require_stable_id(object_stable_id, label="scene object")
    return f"{base_scene_id}:{object_stable_id}"


def _stable_ids(scene: BaseScene | SceneVariant, category: str) -> tuple[str, ...]:
    return tuple(
        value.stable_id
        for value in getattr(scene, category)
    )


def _stable_id_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _height_field_fingerprint(
    terrain: HeightField | TiledHeightField,
) -> str:
    if isinstance(terrain, TiledHeightField):
        return terrain.content_fingerprint
    digest = hashlib.sha256()
    digest.update(
        f"{terrain.origin_x:.17g}|{terrain.origin_y:.17g}|"
        f"{terrain.spacing_m:.17g}|{terrain.width}|{terrain.height}\n".encode()
    )
    for row in terrain.samples:
        digest.update("|".join(f"{value:.17g}" for value in row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _scene_fingerprint(scene: BaseScene | SceneVariant) -> str:
    # Stream directly into the digest.  A portfolio can legitimately contain
    # millions of trees, so constructing a second JSON-sized object graph here
    # would be an avoidable multi-gigabyte memory spike.
    digest = hashlib.sha256()

    def emit(*values: str | float | int) -> None:
        for value in values:
            text = f"{value:.17g}" if isinstance(value, float) else str(value)
            encoded = text.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)

    emit("scene", scene.stable_id, _height_field_fingerprint(scene.terrain))
    emit(
        "ground",
        scene.ground_surface.kind,
        scene.ground_surface.material_ref,
        scene.ground_surface.content_fingerprint,
    )
    for value in sorted(scene.ground_surface.removed_object_classes):
        emit("ground-removed", value)
    for category in ("trees", "buildings"):
        emit("category", category, len(getattr(scene, category)))
        for item in getattr(scene, category):
            emit(
                "asset",
                item.stable_id,
                item.family,
                item.asset_ref,
                item.position.x,
                item.position.y,
                item.position.z,
                item.heading_degrees,
                item.uniform_scale,
                item.footprint_radius_m,
                item.group_id,
            )
    emit("category", "routes", len(scene.routes))
    for item in scene.routes:
        emit("route", item.stable_id, item.family, item.width_m, len(item.points))
        for point in item.points:
            emit("route-point", point.x, point.y, point.z)
        emit("bridge-count", len(item.bridge_spans))
        for span in item.bridge_spans:
            emit(
                "bridge",
                span.stable_id,
                span.start_fraction,
                span.water_start_fraction,
                span.water_end_fraction,
                span.end_fraction,
                span.minimum_deck_clearance_m,
            )
    emit("category", "waters", len(scene.waters))
    for item in scene.waters:
        emit(
            "water",
            item.stable_id,
            item.family,
            item.kind,
            len(item.outline),
            len(item.centreline),
        )
        for point in item.outline:
            emit("water-outline", point.x, point.y)
        for point, elevation in zip(item.centreline, item.surface_profile_m):
            emit("water-centreline", point.x, point.y, elevation)
        if item.kind == "standing":
            emit("water-surface", item.surface_profile_m[0])
    return digest.hexdigest()


def _derived_seed(master_seed: int, base_id: str, variant_index: int) -> int:
    if not 0 <= master_seed <= 2**63 - 1:
        raise ValueError("master seed must be between 0 and 2^63-1")
    digest = hashlib.blake2b(
        f"{ALGORITHM_ID}\0{master_seed}\0{base_id}\0{variant_index}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") & (2**63 - 1)


def _validate_base_scene(
    scene: BaseScene,
    constraints: VariantConstraints,
) -> None:
    for category in ("trees", "buildings", "routes", "waters"):
        if not getattr(scene, category):
            raise InvalidBaseScene(
                f"{scene.stable_id} must contain at least one {category} feature"
            )
    if not scene.suitability_zones:
        raise InvalidBaseScene(f"{scene.stable_id} has no suitability zones")
    if not (
        scene.terrain.bounds.contains(
            Vec2(scene.bounds.min_x, scene.bounds.min_y)
        )
        and scene.terrain.bounds.contains(
            Vec2(scene.bounds.max_x, scene.bounds.max_y)
        )
    ):
        raise InvalidBaseScene(
            f"{scene.stable_id} terrain does not cover its scene bounds"
        )
    identifiers: list[str] = []
    for category in ("trees", "buildings", "routes", "waters"):
        identifiers.extend(_stable_ids(scene, category))
    if len(identifiers) != len(set(identifiers)):
        raise InvalidBaseScene(f"{scene.stable_id} contains duplicate stable IDs")
    if any(not item.group_id.strip() for item in scene.buildings):
        raise InvalidBaseScene(
            f"{scene.stable_id} buildings require stable settlement group IDs"
        )
    for category in ("trees", "buildings"):
        for item in getattr(scene, category):
            if not scene.bounds.contains(item.position.xy, item.footprint_radius_m):
                raise InvalidBaseScene(
                    f"{scene.stable_id} {category[:-1]} {item.stable_id} "
                    "falls outside scene bounds"
                )
    for route in scene.routes:
        last_index = len(route.points) - 1
        invalid_point = False
        for index, point in enumerate(route.points):
            if not scene.bounds.contains(point.xy):
                invalid_point = True
                break
            if scene.bounds.contains(point.xy, route.width_m * 0.5):
                continue
            is_endpoint = index in {0, last_index}
            is_clipped_boundary = is_endpoint and (
                math.isclose(point.x, scene.bounds.min_x, abs_tol=1.0e-6)
                or math.isclose(point.x, scene.bounds.max_x, abs_tol=1.0e-6)
                or math.isclose(point.y, scene.bounds.min_y, abs_tol=1.0e-6)
                or math.isclose(point.y, scene.bounds.max_y, abs_tol=1.0e-6)
            )
            if not is_clipped_boundary:
                invalid_point = True
                break
        if invalid_point:
            raise InvalidBaseScene(
                f"{scene.stable_id} route {route.stable_id} falls outside bounds"
            )
    bridge_ids = [
        span.stable_id
        for route in scene.routes
        for span in route.bridge_spans
    ]
    if len(bridge_ids) != len(set(bridge_ids)):
        raise InvalidBaseScene(
            f"{scene.stable_id} contains duplicate bridge span stable IDs"
        )
    for route in scene.routes:
        if not _source_bridge_layout_is_valid(route, scene.waters, constraints):
            raise InvalidBaseScene(
                f"{scene.stable_id} route {route.stable_id} has an "
                "undeclared water crossing or a bridge span over dry land"
            )
    for water in scene.waters:
        if water.kind == "watercourse" and any(
            not _point_in_polygon(point, water.outline)
            for point in water.centreline
        ):
            raise InvalidBaseScene(
                f"{scene.stable_id} watercourse {water.stable_id} centreline "
                "leaves its water outline"
            )
    requirements = {
        requirement.family: requirement
        for requirement in constraints.tree_suitability
    }
    tree_families = {tree.family for tree in scene.trees}
    missing = sorted(tree_families - requirements.keys())
    if missing:
        raise InvalidBaseScene(
            f"{scene.stable_id} has no biome/soil contract for tree families: "
            + ", ".join(missing)
        )
    zone_families = set().union(
        *(zone.tree_families for zone in scene.suitability_zones)
    )
    missing_zones = sorted(tree_families - zone_families)
    if missing_zones:
        raise InvalidBaseScene(
            f"{scene.stable_id} has no habitat polygon for tree families: "
            + ", ".join(missing_zones)
        )
    if not any(zone.buildable for zone in scene.suitability_zones):
        raise InvalidBaseScene(f"{scene.stable_id} has no buildable land polygon")
    _validate_water_drainage(scene, constraints)
    components = _route_component_count(
        scene.routes, constraints.road_connectivity_tolerance_m
    )
    if components > constraints.maximum_road_components:
        raise InvalidBaseScene(
            f"{scene.stable_id} source road network has {components} components"
        )


def _validate_water_drainage(
    scene: BaseScene | SceneVariant,
    constraints: VariantConstraints,
) -> None:
    for water in scene.waters:
        if water.kind != "watercourse":
            if (
                water.surface_profile_m[0]
                + constraints.water_uphill_tolerance_m
                < min(
                    scene.terrain.elevation(point)
                    for point in water.outline
                )
            ):
                raise InvalidBaseScene(
                    f"standing water {water.stable_id} surface is below terrain"
                )
            continue
        previous = water.surface_profile_m[0]
        for index, point in enumerate(water.centreline):
            current = water.surface_profile_m[index]
            if (
                current + constraints.water_uphill_tolerance_m
                < scene.terrain.elevation(point)
            ):
                raise InvalidBaseScene(
                    f"watercourse {water.stable_id} surface is below terrain"
                )
            if index == 0:
                continue
            if current > previous + constraints.water_uphill_tolerance_m:
                raise InvalidBaseScene(
                    f"watercourse {water.stable_id} runs uphill by "
                    f"{current - previous:.3f} m"
                )
            previous = current


def _resample_route(
    route: SceneRoute, spacing_m: float
) -> tuple[tuple[Vec2, float], ...]:
    total = sum(
        _distance(first.xy, second.xy)
        for first, second in zip(route.points, route.points[1:])
    )
    if total <= _EPSILON:
        raise SceneVariantError(f"route {route.stable_id} has zero length")
    result: list[tuple[Vec2, float]] = [(route.points[0].xy, 0.0)]
    travelled = 0.0
    for first3, second3 in zip(route.points, route.points[1:]):
        first = first3.xy
        second = second3.xy
        length = _distance(first, second)
        pieces = max(1, math.ceil(length / spacing_m))
        for index in range(1, pieces + 1):
            t = index / pieces
            result.append(
                (
                    Vec2(
                        first.x + (second.x - first.x) * t,
                        first.y + (second.y - first.y) * t,
                    ),
                    (travelled + length * t) / total,
                )
            )
        travelled += length
    return tuple(result)


def _source_bridge_layout_is_valid(
    route: SceneRoute,
    waters: Sequence[WaterFeature],
    constraints: VariantConstraints,
) -> bool:
    cumulative, total = _polyline_lengths(route)
    crossed_bridges: set[str] = set()
    for segment_index, (first, second) in enumerate(
        zip(route.points, route.points[1:])
    ):
        crossing = any(
            _segment_polygon_distance(first.xy, second.xy, water.outline)
            < constraints.road_water_buffer_m + route.width_m * 0.5
            for water in waters
        )
        if not crossing:
            continue
        start_fraction = cumulative[segment_index] / total
        end_fraction = cumulative[segment_index + 1] / total
        span = next(
            (
                candidate
                for candidate in route.bridge_spans
                if candidate.water_start_fraction
                <= start_fraction
                <= end_fraction
                <= candidate.water_end_fraction
            ),
            None,
        )
        if span is None:
            return False
        crossed_bridges.add(span.stable_id)
    return crossed_bridges == {span.stable_id for span in route.bridge_spans}


def _bridge_clearance_at_fraction(route: SceneRoute, fraction: float) -> float:
    for span in route.bridge_spans:
        if not span.start_fraction <= fraction <= span.end_fraction:
            continue
        if fraction < span.water_start_fraction:
            factor = (fraction - span.start_fraction) / (
                span.water_start_fraction - span.start_fraction
            )
        elif fraction <= span.water_end_fraction:
            factor = 1.0
        else:
            factor = (span.end_fraction - fraction) / (
                span.end_fraction - span.water_end_fraction
            )
        return span.minimum_deck_clearance_m * max(0.0, min(1.0, factor))
    return 0.0


def _bridge_span_at_fraction(
    route: SceneRoute, fraction: float
) -> BridgeSpan | None:
    return next(
        (
            span
            for span in route.bridge_spans
            if span.start_fraction <= fraction <= span.end_fraction
        ),
        None,
    )


def _draped_route_elevation(
    scene: BaseScene,
    route: SceneRoute,
    point: Vec2,
    fraction: float,
) -> float:
    terrain_elevation = scene.terrain.elevation(point)
    clearance = _bridge_clearance_at_fraction(route, fraction)
    if clearance <= 0.0:
        return terrain_elevation
    span = _bridge_span_at_fraction(route, fraction)
    on_deck = (
        span is not None
        and span.water_start_fraction <= fraction <= span.water_end_fraction
    )
    if on_deck:
        nearest_water = min(
            scene.waters,
            key=lambda water: _point_polygon_distance(point, water.outline),
        )
        water_elevation = _water_surface_elevation(nearest_water, point)
    else:
        water_elevation = terrain_elevation
    return max(terrain_elevation, water_elevation) + clearance


def _warp_point(point: Vec2, bounds: Bounds, warp: _Warp) -> Vec2:
    centre = bounds.centre
    local_x = point.x - centre.x
    local_y = point.y - centre.y
    cosine = math.cos(warp.angle_radians)
    sine = math.sin(warp.angle_radians)
    rotated_x = warp.scale * (local_x * cosine - local_y * sine)
    rotated_y = warp.scale * (local_x * sine + local_y * cosine)
    base_x = centre.x + rotated_x
    base_y = centre.y + rotated_y
    u = (point.x - bounds.min_x) / bounds.width
    v = (point.y - bounds.min_y) / bounds.height
    boundary_weight = min(1.0, 5.0 * min(u, 1.0 - u, v, 1.0 - v))
    displacement_x = (
        warp.amplitude_m
        * boundary_weight
        * math.sin(2.0 * math.pi * v + warp.phase_x)
        * math.sin(math.pi * u)
    )
    displacement_y = (
        warp.amplitude_m
        * boundary_weight
        * math.sin(2.0 * math.pi * u + warp.phase_y)
        * math.sin(math.pi * v)
    )
    return Vec2(
        base_x + displacement_x + warp.translate_x,
        base_y + displacement_y + warp.translate_y,
    )


def _random_warp(
    rng: random.Random,
    bounds: Bounds,
    constraints: VariantConstraints,
) -> _Warp:
    return _Warp(
        angle_radians=math.radians(
            rng.uniform(
                -constraints.route_rotation_degrees,
                constraints.route_rotation_degrees,
            )
        ),
        scale=1.0
        + rng.uniform(-constraints.route_scale_delta, constraints.route_scale_delta),
        phase_x=rng.uniform(0.0, 2.0 * math.pi),
        phase_y=rng.uniform(0.0, 2.0 * math.pi),
        amplitude_m=rng.uniform(
            constraints.route_warp_amplitude_m * 0.65,
            constraints.route_warp_amplitude_m,
        ),
        translate_x=rng.uniform(-0.015, 0.015) * bounds.width,
        translate_y=rng.uniform(-0.015, 0.015) * bounds.height,
    )


def _route_is_valid(
    route: SceneRoute,
    scene: BaseScene,
    constraints: VariantConstraints,
) -> bool:
    for point in route.points:
        if not scene.bounds.contains(point.xy, route.width_m * 0.5):
            return False
    cumulative, total = _polyline_lengths(route)
    crossed_bridges: set[str] = set()
    for segment_index, (first, second) in enumerate(
        zip(route.points, route.points[1:])
    ):
        horizontal = _distance(first.xy, second.xy)
        if horizontal <= _EPSILON:
            return False
        if (
            abs(second.z - first.z) / horizontal * 100.0
            > constraints.maximum_road_grade_percent
        ):
            return False
        for water in scene.waters:
            collision_clearance = (
                constraints.road_water_buffer_m + route.width_m * 0.5
            )
            if (
                _segment_polygon_distance(first.xy, second.xy, water.outline)
                >= collision_clearance
            ):
                continue
            start_fraction = cumulative[segment_index] / total
            end_fraction = cumulative[segment_index + 1] / total
            span = next(
                (
                    candidate
                    for candidate in route.bridge_spans
                    if candidate.water_start_fraction
                    <= start_fraction
                    <= end_fraction
                    <= candidate.water_end_fraction
                ),
                None,
            )
            if span is None:
                return False
            midpoint = Vec2(
                (first.x + second.x) * 0.5,
                (first.y + second.y) * 0.5,
            )
            deck_elevation = (first.z + second.z) * 0.5
            if (
                deck_elevation - _water_surface_elevation(water, midpoint)
                < span.minimum_deck_clearance_m - 1.0e-6
            ):
                return False
            crossed_bridges.add(span.stable_id)
    if crossed_bridges != {span.stable_id for span in route.bridge_spans}:
        return False
    return True


def _compose_routes(
    scene: BaseScene,
    rng: random.Random,
    constraints: VariantConstraints,
) -> tuple[tuple[SceneRoute, ...], _Warp]:
    source_topology = route_topology(
        scene.routes, constraints.road_connectivity_tolerance_m
    )
    for _ in range(constraints.maximum_route_attempts):
        warp = _random_warp(rng, scene.bounds, constraints)
        routes: list[SceneRoute] = []
        for source in scene.routes:
            parameterized = tuple(
                (
                    _warp_point(point, scene.bounds, warp),
                    fraction,
                )
                for point, fraction in _resample_route(
                    source, constraints.road_sample_spacing_m
                )
            )
            points3 = tuple(
                Vec3(
                    point.x,
                    point.y,
                    _draped_route_elevation(
                        scene, source, point, fraction
                    ),
                )
                for point, fraction in parameterized
            )
            candidate = replace(source, points=points3)
            if not _route_is_valid(candidate, scene, constraints):
                break
            routes.append(candidate)
        if len(routes) != len(scene.routes):
            continue
        result = tuple(routes)
        if route_topology(
            result, constraints.road_connectivity_tolerance_m
        ) != source_topology:
            continue
        movement = _route_distance(scene.routes, result)
        if movement < constraints.movement_epsilon_m:
            continue
        return result, warp
    raise SceneVariantError(
        f"{scene.stable_id}: road network cannot be warped within grade, "
        "water, bounds and connectivity constraints"
    )


def _polyline_lengths(route: SceneRoute) -> tuple[tuple[float, ...], float]:
    lengths = [0.0]
    for first, second in zip(route.points, route.points[1:]):
        lengths.append(lengths[-1] + _distance(first.xy, second.xy))
    return tuple(lengths), lengths[-1]


def _point_along_route(route: SceneRoute, distance_m: float) -> tuple[Vec2, float]:
    cumulative, total = _polyline_lengths(route)
    if total <= _EPSILON:
        raise SceneVariantError(f"route {route.stable_id} has zero length")
    distance_m = min(max(distance_m, 0.0), total)
    for index, (start, end) in enumerate(zip(cumulative, cumulative[1:])):
        if distance_m <= end or index == len(cumulative) - 2:
            length = end - start
            t = 0.0 if length <= _EPSILON else (distance_m - start) / length
            first = route.points[index]
            second = route.points[index + 1]
            return (
                Vec2(
                    first.x + (second.x - first.x) * t,
                    first.y + (second.y - first.y) * t,
                ),
                math.atan2(second.y - first.y, second.x - first.x),
            )
    raise AssertionError("unreachable route interpolation")


def _clear_of_water(
    point: Vec2,
    waters: Sequence[WaterFeature],
    clearance_m: float,
) -> bool:
    return all(
        _point_polygon_distance(point, water.outline) >= clearance_m
        for water in waters
    )


def _buildable(point: Vec2, zones: Sequence[SuitabilityZone]) -> bool:
    return any(zone.buildable and _point_in_polygon(point, zone.outline) for zone in zones)


def _foundation_is_valid(
    point: Vec2,
    radius: float,
    scene: BaseScene,
    constraints: VariantConstraints,
) -> bool:
    if scene.terrain.slope_percent(point) > constraints.maximum_building_slope_percent:
        return False
    samples = (
        Vec2(point.x - radius, point.y - radius),
        Vec2(point.x + radius, point.y - radius),
        Vec2(point.x + radius, point.y + radius),
        Vec2(point.x - radius, point.y + radius),
        point,
    )
    if any(not scene.bounds.contains(sample) for sample in samples):
        return False
    elevations = [scene.terrain.elevation(sample) for sample in samples]
    return (
        max(elevations) - min(elevations)
        <= constraints.maximum_foundation_relief_m
    )


def _compose_buildings(
    scene: BaseScene,
    routes: tuple[SceneRoute, ...],
    rng: random.Random,
    constraints: VariantConstraints,
) -> tuple[SceneAsset, ...]:
    groups: dict[str, list[SceneAsset]] = defaultdict(list)
    for building in scene.buildings:
        groups[building.group_id].append(building)
    route_lengths = [_polyline_lengths(route)[1] for route in routes]
    usable_routes = [
        (route, length)
        for route, length in zip(routes, route_lengths)
        if length > 2.0 * constraints.edge_buffer_m
    ]
    if not usable_routes:
        raise SceneVariantError(f"{scene.stable_id}: no route can serve buildings")
    route_index = _RouteSpatialIndex(
        routes,
        maximum_query_distance_m=constraints.maximum_building_road_distance_m,
    )
    placed: list[SceneAsset] = []
    grid = _PointGrid(
        max(
            constraints.minimum_building_spacing_m,
            max(item.footprint_radius_m for item in scene.buildings) * 2.0,
        )
    )
    for group_id in sorted(groups):
        source_group = sorted(groups[group_id], key=lambda item: item.stable_id)
        group_placed: list[SceneAsset] | None = None
        for _ in range(constraints.maximum_building_attempts):
            route, route_length = rng.choice(usable_routes)
            anchor = rng.uniform(0.1, 0.9) * route_length
            local: list[SceneAsset] = []
            local_grid = _PointGrid(
                max(
                    constraints.minimum_building_spacing_m,
                    max(item.footprint_radius_m for item in source_group) * 2.0,
                )
            )
            span = min(
                constraints.maximum_building_group_radius_m * 1.6,
                route_length * 0.65,
                max(0, len(source_group) - 1)
                * constraints.building_group_spacing_m,
            )
            step = 0.0 if len(source_group) == 1 else span / (len(source_group) - 1)
            failed = False
            for index, source in enumerate(source_group):
                along = anchor + (index - (len(source_group) - 1) * 0.5) * step
                along += rng.uniform(-0.18, 0.18) * max(step, source.footprint_radius_m)
                if not 0.0 <= along <= route_length:
                    failed = True
                    break
                road_point, tangent = _point_along_route(route, along)
                minimum_offset = (
                    route.width_m * 0.5
                    + constraints.building_front_setback_m
                    + source.footprint_radius_m
                )
                maximum_offset = (
                    route.width_m * 0.5
                    + constraints.maximum_building_road_distance_m
                )
                if minimum_offset >= maximum_offset:
                    failed = True
                    break
                offset = rng.uniform(minimum_offset, maximum_offset)
                if rng.random() < 0.5:
                    offset = -offset
                point = Vec2(
                    road_point.x - math.sin(tangent) * offset,
                    road_point.y + math.cos(tangent) * offset,
                )
                clearance = (
                    source.footprint_radius_m + constraints.edge_buffer_m
                )
                if not scene.bounds.contains(point, clearance):
                    failed = True
                    break
                if not _buildable(point, scene.suitability_zones):
                    failed = True
                    break
                if not _clear_of_water(
                    point,
                    scene.waters,
                    source.footprint_radius_m
                    + constraints.building_water_buffer_m,
                ):
                    failed = True
                    break
                if not _foundation_is_valid(
                    point, source.footprint_radius_m, scene, constraints
                ):
                    failed = True
                    break
                if grid.overlaps(
                    point,
                    source.footprint_radius_m,
                    constraints.minimum_building_spacing_m,
                ) or local_grid.overlaps(
                    point,
                    source.footprint_radius_m,
                    constraints.minimum_building_spacing_m,
                ):
                    failed = True
                    break
                road_distance, road_heading = route_index.nearest(point)
                if road_distance > constraints.maximum_building_road_distance_m:
                    failed = True
                    break
                heading = (road_heading + rng.uniform(-8.0, 8.0)) % 360.0
                local.append(
                    replace(
                        source,
                        position=Vec3(
                            point.x,
                            point.y,
                            scene.terrain.elevation(point),
                        ),
                        heading_degrees=heading,
                    )
                )
                local_grid.add(point, source.footprint_radius_m)
            if failed or len(local) != len(source_group):
                continue
            centroid = Vec2(
                sum(item.position.x for item in local) / len(local),
                sum(item.position.y for item in local) / len(local),
            )
            if any(
                _distance(item.position.xy, centroid)
                > constraints.maximum_building_group_radius_m
                for item in local
            ):
                continue
            group_placed = local
            break
        if group_placed is None:
            raise SceneVariantError(
                f"{scene.stable_id}: settlement group {group_id} cannot be "
                "placed with road access, viable foundations and buffers"
            )
        for item in group_placed:
            placed.append(item)
            grid.add(item.position.xy, item.footprint_radius_m)
    by_id = {item.stable_id: item for item in placed}
    return tuple(by_id[source.stable_id] for source in scene.buildings)


def _tree_habitat_is_valid(
    tree: SceneAsset,
    point: Vec2,
    zones: Sequence[SuitabilityZone],
    requirements: dict[str, FamilySuitability],
) -> bool:
    requirement = requirements[tree.family]
    return any(
        tree.family in zone.tree_families
        and zone.biome in requirement.allowed_biomes
        and zone.soil in requirement.allowed_soils
        and _point_in_polygon(point, zone.outline)
        for zone in zones
    )


def _compose_trees(
    scene: BaseScene,
    routes: tuple[SceneRoute, ...],
    buildings: tuple[SceneAsset, ...],
    warp: _Warp,
    rng: random.Random,
    constraints: VariantConstraints,
) -> tuple[SceneAsset, ...]:
    requirements = {
        requirement.family: requirement
        for requirement in constraints.tree_suitability
    }
    max_radius = max(item.footprint_radius_m for item in scene.trees)
    tree_grid = _PointGrid(
        max(constraints.minimum_tree_spacing_m, max_radius * 2.0)
    )
    building_grid = _PointGrid(
        max(item.footprint_radius_m for item in buildings) * 2.0
    )
    for building in buildings:
        building_grid.add(building.position.xy, building.footprint_radius_m)
    route_index = _RouteSpatialIndex(
        routes,
        maximum_query_distance_m=max(
            constraints.maximum_building_road_distance_m,
            max_radius + constraints.tree_road_buffer_m,
        ),
    )
    density_counts: Counter[tuple[int, int]] = Counter()
    cell_area_hectares = constraints.tree_cluster_cell_m**2 / 10_000.0
    maximum_cell_count = max(
        1,
        math.ceil(
            constraints.maximum_tree_density_per_hectare * cell_area_hectares
        ),
    )
    result: list[SceneAsset] = []
    for source in scene.trees:
        warped = _warp_point(source.position.xy, scene.bounds, warp)
        placed: SceneAsset | None = None
        for attempt in range(constraints.maximum_tree_attempts):
            radius = (
                constraints.tree_relocation_radius_m
                * math.sqrt((attempt + rng.random()) / constraints.maximum_tree_attempts)
            )
            angle = rng.uniform(0.0, 2.0 * math.pi)
            point = Vec2(
                warped.x + math.cos(angle) * radius,
                warped.y + math.sin(angle) * radius,
            )
            clearance = source.footprint_radius_m + constraints.edge_buffer_m
            if not scene.bounds.contains(point, clearance):
                continue
            if not _tree_habitat_is_valid(
                source, point, scene.suitability_zones, requirements
            ):
                continue
            if (
                scene.terrain.slope_percent(point)
                > constraints.maximum_tree_slope_percent
            ):
                continue
            if not _clear_of_water(
                point,
                scene.waters,
                source.footprint_radius_m + constraints.tree_water_buffer_m,
            ):
                continue
            road_distance, _ = route_index.nearest(point)
            if road_distance < (
                source.footprint_radius_m + constraints.tree_road_buffer_m
            ):
                continue
            if building_grid.overlaps(
                point,
                source.footprint_radius_m,
                constraints.tree_building_buffer_m,
            ):
                continue
            if tree_grid.overlaps(
                point,
                source.footprint_radius_m,
                constraints.minimum_tree_spacing_m,
            ):
                continue
            density_key = (
                math.floor(point.x / constraints.tree_cluster_cell_m),
                math.floor(point.y / constraints.tree_cluster_cell_m),
            )
            if density_counts[density_key] >= maximum_cell_count:
                continue
            placed = replace(
                source,
                position=Vec3(
                    point.x,
                    point.y,
                    scene.terrain.elevation(point),
                ),
                heading_degrees=(
                    source.heading_degrees + rng.uniform(-35.0, 35.0)
                )
                % 360.0,
            )
            density_counts[density_key] += 1
            tree_grid.add(point, source.footprint_radius_m)
            break
        if placed is None:
            raise SceneVariantError(
                f"{scene.stable_id}: tree {source.stable_id} cannot be placed "
                "without violating habitat, slope, density or exclusion buffers"
            )
        result.append(placed)
    return tuple(result)


def _asset_displacement(
    source: Sequence[SceneAsset], result: Sequence[SceneAsset], epsilon_m: float
) -> DisplacementMetrics:
    result_by_id = {item.stable_id: item for item in result}
    distances = [
        _distance(item.position.xy, result_by_id[item.stable_id].position.xy)
        for item in source
    ]
    return DisplacementMetrics(
        count=len(distances),
        moved_fraction=sum(value >= epsilon_m for value in distances) / len(distances),
        mean_distance_m=sum(distances) / len(distances),
        maximum_distance_m=max(distances),
    )


def _sample_route(route: SceneRoute, samples: int = 32) -> tuple[Vec2, ...]:
    _, total = _polyline_lengths(route)
    if total <= _EPSILON:
        raise SceneVariantError(f"route {route.stable_id} has zero length")
    return tuple(
        _point_along_route(route, total * index / (samples - 1))[0]
        for index in range(samples)
    )


def _route_distance(
    source: Sequence[SceneRoute], result: Sequence[SceneRoute]
) -> float:
    result_by_id = {route.stable_id: route for route in result}
    distances: list[float] = []
    for route in source:
        source_samples = _sample_route(route)
        result_samples = _sample_route(result_by_id[route.stable_id])
        distances.extend(
            _distance(first, second)
            for first, second in zip(source_samples, result_samples)
        )
    return sum(distances) / len(distances)


def _route_displacement(
    source: Sequence[SceneRoute],
    result: Sequence[SceneRoute],
    epsilon_m: float,
) -> DisplacementMetrics:
    result_by_id = {route.stable_id: route for route in result}
    distances: list[float] = []
    for route in source:
        source_samples = _sample_route(route)
        result_samples = _sample_route(result_by_id[route.stable_id])
        distances.extend(
            _distance(first, second)
            for first, second in zip(source_samples, result_samples)
        )
    return DisplacementMetrics(
        count=len(source),
        moved_fraction=sum(value >= epsilon_m for value in distances) / len(distances),
        mean_distance_m=sum(distances) / len(distances),
        maximum_distance_m=max(distances),
    )


def _route_components(
    routes: Sequence[SceneRoute], tolerance_m: float
) -> tuple[tuple[str, ...], ...]:
    """Return exact route-component membership through a spatial segment index.

    Every segment is inserted into all grid cells touched by its
    tolerance-expanded axis-aligned bounds.  Therefore any pair whose exact
    segment distance is within ``tolerance_m`` is guaranteed to share at least
    one candidate cell.  The grid only removes impossible pairs; connectivity
    is still decided by the same exact ``_segment_distance`` predicate used by
    the former quadratic oracle.
    """

    if not routes:
        return ()
    if not math.isfinite(tolerance_m) or tolerance_m <= 0.0:
        raise ValueError("road connectivity tolerance must be finite and positive")
    stable_ids = [route.stable_id for route in routes]
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError("road topology requires unique stable route IDs")
    parent = list(range(len(routes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    segments: list[tuple[int, Vec2, Vec2]] = []
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for route_index, route in enumerate(routes):
        for first, second in zip(route.points, route.points[1:]):
            first_xy = first.xy
            second_xy = second.xy
            segments.append((route_index, first_xy, second_xy))
            min_x = min(min_x, first_xy.x, second_xy.x)
            min_y = min(min_y, first_xy.y, second_xy.y)
            max_x = max(max_x, first_xy.x, second_xy.x)
            max_y = max(max_y, first_xy.y, second_xy.y)
    if not segments:
        raise ValueError("road topology requires at least one route segment")

    # Use a deterministic bounded sample of native segment spans. Geographic
    # extents can be hundreds of kilometres while individual road segments are
    # short; an extent-derived cell would collapse many unrelated segments into
    # one bucket and reintroduce quadratic candidate counts.
    sample_step = max(1, len(segments) // 4096)
    sampled_spans = sorted(
        max(
            abs(second.x - first.x),
            abs(second.y - first.y),
            tolerance_m * 2.0,
        )
        for _route_index, first, second in segments[::sample_step]
    )
    cell_size = max(
        tolerance_m * 2.0,
        sampled_spans[len(sampled_spans) // 2],
    )
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    indexed_segments: list[tuple[int, Vec2, Vec2]] = []
    for route_index, first, second in segments:
        cell_min_x = math.floor(
            (min(first.x, second.x) - tolerance_m - min_x) / cell_size
        )
        cell_max_x = math.floor(
            (max(first.x, second.x) + tolerance_m - min_x) / cell_size
        )
        cell_min_y = math.floor(
            (min(first.y, second.y) - tolerance_m - min_y) / cell_size
        )
        cell_max_y = math.floor(
            (max(first.y, second.y) + tolerance_m - min_y) / cell_size
        )
        candidates: set[int] = set()
        for cell_x in range(cell_min_x, cell_max_x + 1):
            for cell_y in range(cell_min_y, cell_max_y + 1):
                candidates.update(cells.get((cell_x, cell_y), ()))
        for candidate_index in candidates:
            other_route_index, other_first, other_second = indexed_segments[
                candidate_index
            ]
            if other_route_index == route_index:
                continue
            if (
                _segment_distance(
                    first,
                    second,
                    other_first,
                    other_second,
                )
                <= tolerance_m
            ):
                union(route_index, other_route_index)
        segment_index = len(indexed_segments)
        indexed_segments.append((route_index, first, second))
        for cell_x in range(cell_min_x, cell_max_x + 1):
            for cell_y in range(cell_min_y, cell_max_y + 1):
                cells[(cell_x, cell_y)].append(segment_index)
    grouped: dict[int, list[str]] = defaultdict(list)
    for index, stable_id in enumerate(stable_ids):
        grouped[find(index)].append(stable_id)
    return tuple(
        sorted(
            tuple(sorted(component))
            for component in grouped.values()
        )
    )


def route_topology(
    routes: Sequence[SceneRoute], tolerance_m: float
) -> tuple[int, str]:
    """Return the exact stable-ID membership of the route components.

    The digest is independent of route order and component root selection. It
    lets portable/native consumers prove that a variant kept the accepted
    source network topology instead of merely staying below a loose component
    ceiling.
    """

    components = _route_components(routes, tolerance_m)
    membership_sha256 = hashlib.sha256(
        json.dumps(
            components,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return len(components), membership_sha256


def _route_component_count(
    routes: Sequence[SceneRoute], tolerance_m: float
) -> int:
    return route_topology(routes, tolerance_m)[0]


def _validate_identity_and_counts(
    source: BaseScene, result: SceneVariant
) -> None:
    if result.terrain is not source.terrain:
        raise SceneVariantError("variant replaced the accepted base terrain")
    if result.ground_surface is not source.ground_surface:
        raise SceneVariantError("variant replaced the accepted ground surface")
    if result.waters is not source.waters:
        raise SceneVariantError("variant replaced accepted water geometry")
    if _family_counts(source) != _family_counts(result):
        raise SceneVariantError("variant changed exact per-family object counts")
    for category in ("trees", "buildings", "routes", "waters"):
        source_ids = _stable_ids(source, category)
        result_ids = _stable_ids(result, category)
        if source_ids != result_ids:
            raise SceneVariantError(
                f"variant changed stable ID order for {category}"
            )
    for category in ("trees", "buildings"):
        source_assets = getattr(source, category)
        result_assets = getattr(result, category)
        for before, after in zip(source_assets, result_assets):
            if (
                before.asset_ref != after.asset_ref
                or before.uniform_scale != after.uniform_scale
                or before.family != after.family
            ):
                raise SceneVariantError(
                    f"variant changed asset identity or scale for {before.stable_id}"
                )


def _validate_output_constraints(
    source: BaseScene,
    result: SceneVariant,
    constraints: VariantConstraints,
) -> None:
    _validate_identity_and_counts(source, result)
    requirements = {
        requirement.family: requirement
        for requirement in constraints.tree_suitability
    }
    route_index = _RouteSpatialIndex(
        result.routes,
        maximum_query_distance_m=max(
            constraints.maximum_building_road_distance_m,
            max(item.footprint_radius_m for item in result.trees)
            + constraints.tree_road_buffer_m,
        ),
    )
    building_grid = _PointGrid(
        max(item.footprint_radius_m for item in result.buildings) * 2.0
    )
    for building in result.buildings:
        point = building.position.xy
        if not _buildable(point, result.suitability_zones):
            raise SceneVariantError(f"building {building.stable_id} is not buildable")
        if not _foundation_is_valid(
            point, building.footprint_radius_m, source, constraints
        ):
            raise SceneVariantError(
                f"building {building.stable_id} has an invalid foundation"
            )
        if not _clear_of_water(
            point,
            result.waters,
            building.footprint_radius_m + constraints.building_water_buffer_m,
        ):
            raise SceneVariantError(f"building {building.stable_id} enters water")
        road_distance, _ = route_index.nearest(point)
        if road_distance > constraints.maximum_building_road_distance_m:
            raise SceneVariantError(
                f"building {building.stable_id} has no road access"
            )
        if building_grid.overlaps(
            point,
            building.footprint_radius_m,
            constraints.minimum_building_spacing_m,
        ):
            raise SceneVariantError(f"building {building.stable_id} overlaps another")
        building_grid.add(point, building.footprint_radius_m)
    tree_grid = _PointGrid(
        max(item.footprint_radius_m for item in result.trees) * 2.0
    )
    density: Counter[tuple[int, int]] = Counter()
    max_density = max(
        1,
        math.ceil(
            constraints.maximum_tree_density_per_hectare
            * constraints.tree_cluster_cell_m**2
            / 10_000.0
        ),
    )
    for tree in result.trees:
        point = tree.position.xy
        if not _tree_habitat_is_valid(
            tree, point, result.suitability_zones, requirements
        ):
            raise SceneVariantError(f"tree {tree.stable_id} is outside its habitat")
        if source.terrain.slope_percent(point) > constraints.maximum_tree_slope_percent:
            raise SceneVariantError(f"tree {tree.stable_id} exceeds its slope")
        if not _clear_of_water(
            point,
            result.waters,
            tree.footprint_radius_m + constraints.tree_water_buffer_m,
        ):
            raise SceneVariantError(f"tree {tree.stable_id} enters water")
        road_distance, _ = route_index.nearest(point)
        if road_distance < tree.footprint_radius_m + constraints.tree_road_buffer_m:
            raise SceneVariantError(f"tree {tree.stable_id} enters a road buffer")
        if building_grid.overlaps(
            point,
            tree.footprint_radius_m,
            constraints.tree_building_buffer_m,
        ):
            raise SceneVariantError(f"tree {tree.stable_id} enters a building buffer")
        if tree_grid.overlaps(
            point,
            tree.footprint_radius_m,
            constraints.minimum_tree_spacing_m,
        ):
            raise SceneVariantError(f"tree {tree.stable_id} overlaps another tree")
        tree_grid.add(point, tree.footprint_radius_m)
        key = (
            math.floor(point.x / constraints.tree_cluster_cell_m),
            math.floor(point.y / constraints.tree_cluster_cell_m),
        )
        density[key] += 1
        if density[key] > max_density:
            raise SceneVariantError("tree distribution contains a pathological cluster")
    source_topology = route_topology(
        source.routes, constraints.road_connectivity_tolerance_m
    )
    result_topology = route_topology(
        result.routes, constraints.road_connectivity_tolerance_m
    )
    if result_topology != source_topology:
        raise SceneVariantError(
            "variant road network changed source component membership"
        )
    if any(
        not _route_is_valid(route, source, constraints)
        for route in result.routes
    ):
        raise SceneVariantError("variant road network violates terrain or water")


def _validate_transformation_contract(
    source: BaseScene,
    variant: SceneVariant,
    constraints: VariantConstraints,
) -> None:
    contract = variant.contract
    if contract.algorithm != ALGORITHM_ID:
        raise SceneVariantError("variant transformation algorithm is unsupported")
    if (
        contract.base_scene_id != source.stable_id
        or contract.variant_id != variant.stable_id
        or contract.variant_index != variant.variant_index
    ):
        raise SceneVariantError("variant transformation identity is inconsistent")
    if contract.stable_id_policy != "base-scene-qualified-identity-v1":
        raise SceneVariantError("variant stable ID policy is inconsistent")
    expected_hashes = tuple(
        (
            category,
            _stable_id_hash(_stable_ids(source, category)),
        )
        for category in ("trees", "buildings", "routes", "waters")
    )
    if contract.stable_id_hashes != expected_hashes:
        raise SceneVariantError("variant stable ID contract is stale")
    if (
        contract.source_family_counts != _family_counts(source)
        or contract.result_family_counts != _family_counts(variant)
    ):
        raise SceneVariantError("variant count contract is stale")
    if contract.terrain_fingerprint != _height_field_fingerprint(source.terrain):
        raise SceneVariantError("variant terrain fingerprint is stale")
    if contract.source_fingerprint != _scene_fingerprint(source):
        raise SceneVariantError("variant source fingerprint is stale")
    if contract.result_fingerprint != _scene_fingerprint(variant):
        raise SceneVariantError("variant result fingerprint is stale")
    if tuple(name for name, _ in contract.route_warp_parameters) != (
        "angle_radians",
        "scale",
        "phase_x",
        "phase_y",
        "amplitude_m",
        "translate_x",
        "translate_y",
    ) or any(
        not math.isfinite(value) for _, value in contract.route_warp_parameters
    ):
        raise SceneVariantError("variant road warp contract is malformed")
    expected_displacement = (
        (
            "trees",
            _asset_displacement(
                source.trees, variant.trees, constraints.movement_epsilon_m
            ),
        ),
        (
            "buildings",
            _asset_displacement(
                source.buildings,
                variant.buildings,
                constraints.movement_epsilon_m,
            ),
        ),
        (
            "routes",
            _route_displacement(
                source.routes, variant.routes, constraints.movement_epsilon_m
            ),
        ),
    )
    if contract.displacement != expected_displacement:
        raise SceneVariantError("variant displacement contract is stale")


def _variant_distance(
    first: SceneVariant,
    second: SceneVariant,
    category: str,
) -> float:
    if category in {"trees", "buildings"}:
        second_by_id = {
            item.stable_id: item
            for item in getattr(second, category)
        }
        distances = [
            _distance(item.position.xy, second_by_id[item.stable_id].position.xy)
            for item in getattr(first, category)
        ]
        return sum(distances) / len(distances)
    if category == "routes":
        return _route_distance(first.routes, second.routes)
    raise ValueError(f"unsupported diversity category: {category}")


def _validate_diversity(
    candidate: SceneVariant,
    previous: Sequence[SceneVariant],
    constraints: VariantConstraints,
) -> bool:
    minimums = {
        "trees": constraints.minimum_tree_intervariant_distance_m,
        "buildings": constraints.minimum_building_intervariant_distance_m,
        "routes": constraints.minimum_route_intervariant_distance_m,
    }
    return all(
        _variant_distance(candidate, other, category) >= minimum
        for other in previous
        for category, minimum in minimums.items()
    )


def _build_contract(
    source: BaseScene,
    result_stub: SceneVariant,
    *,
    seed: int,
    variant_index: int,
    composition_attempt: int,
    warp: _Warp,
    constraints: VariantConstraints,
) -> TransformationContract:
    displacement = (
        (
            "trees",
            _asset_displacement(
                source.trees, result_stub.trees, constraints.movement_epsilon_m
            ),
        ),
        (
            "buildings",
            _asset_displacement(
                source.buildings,
                result_stub.buildings,
                constraints.movement_epsilon_m,
            ),
        ),
        (
            "routes",
            _route_displacement(
                source.routes, result_stub.routes, constraints.movement_epsilon_m
            ),
        ),
    )
    for category, metric in displacement:
        if metric.moved_fraction < constraints.minimum_moved_fraction:
            raise SceneVariantError(
                f"{source.stable_id}: {category} rearrangement moved only "
                f"{metric.moved_fraction:.1%} of the source geometry"
            )
    stable_hashes = tuple(
        (
            category,
            _stable_id_hash(_stable_ids(source, category)),
        )
        for category in ("trees", "buildings", "routes", "waters")
    )
    return TransformationContract(
        algorithm=ALGORITHM_ID,
        base_scene_id=source.stable_id,
        variant_id=result_stub.stable_id,
        variant_index=variant_index,
        seed=seed,
        composition_attempt=composition_attempt,
        terrain_fingerprint=_height_field_fingerprint(source.terrain),
        source_fingerprint=_scene_fingerprint(source),
        result_fingerprint=_scene_fingerprint(result_stub),
        stable_id_policy="base-scene-qualified-identity-v1",
        stable_id_hashes=stable_hashes,
        source_family_counts=_family_counts(source),
        result_family_counts=_family_counts(result_stub),
        route_warp_parameters=warp.contract_values(),
        displacement=displacement,
    )


def _compose_one(
    source: BaseScene,
    *,
    seed: int,
    variant_index: int,
    composition_attempt: int,
    constraints: VariantConstraints,
) -> SceneVariant:
    rng = random.Random(
        seed ^ ((composition_attempt + 1) * 0x9E3779B97F4A7C15)
    )
    routes, warp = _compose_routes(source, rng, constraints)
    buildings = _compose_buildings(source, routes, rng, constraints)
    trees = _compose_trees(source, routes, buildings, warp, rng, constraints)
    stable_id = f"{source.stable_id}-variant-{variant_index:02d}"
    placeholder_contract = TransformationContract(
        algorithm=ALGORITHM_ID,
        base_scene_id=source.stable_id,
        variant_id=stable_id,
        variant_index=variant_index,
        seed=seed,
        composition_attempt=composition_attempt,
        terrain_fingerprint="",
        source_fingerprint="",
        result_fingerprint="",
        stable_id_policy="",
        stable_id_hashes=(),
        source_family_counts=(),
        result_family_counts=(),
        route_warp_parameters=(),
        displacement=(),
    )
    stub = SceneVariant(
        stable_id=stable_id,
        base_scene_id=source.stable_id,
        variant_index=variant_index,
        terrain=source.terrain,
        ground_surface=source.ground_surface,
        bounds=source.bounds,
        trees=trees,
        buildings=buildings,
        routes=routes,
        waters=source.waters,
        suitability_zones=source.suitability_zones,
        contract=placeholder_contract,
    )
    contract = _build_contract(
        source,
        stub,
        seed=seed,
        variant_index=variant_index,
        composition_attempt=composition_attempt,
        warp=warp,
        constraints=constraints,
    )
    result = replace(stub, contract=contract)
    _validate_output_constraints(source, result, constraints)
    return result


def generate_scene_variants(
    base_scenes: Sequence[BaseScene],
    *,
    master_seed: int,
    constraints: VariantConstraints,
) -> tuple[SceneVariant, ...]:
    """Generate exactly five deterministic variants for exactly four bases.

    Bases are sorted by stable ID, so caller ordering does not affect output.
    Generation is atomic from the caller's perspective: any impossible
    constraint raises and no partial tuple is returned.
    """

    if len(base_scenes) != BASE_SCENE_COUNT:
        raise ValueError(
            f"exactly {BASE_SCENE_COUNT} accepted base scenes are required"
        )
    if len({scene.stable_id for scene in base_scenes}) != BASE_SCENE_COUNT:
        raise ValueError("base scene stable IDs must be unique")
    ordered = tuple(sorted(base_scenes, key=lambda scene: scene.stable_id))
    global_ids: set[str] = set()
    for scene in ordered:
        _validate_base_scene(scene, constraints)
        for category in ("trees", "buildings", "routes", "waters"):
            for stable_id in _stable_ids(scene, category):
                global_id = global_stable_id(scene.stable_id, stable_id)
                if global_id in global_ids:
                    raise InvalidBaseScene(
                        f"duplicate portfolio stable ID: {global_id}"
                    )
                global_ids.add(global_id)
    portfolio: list[SceneVariant] = []
    for source in ordered:
        base_variants: list[SceneVariant] = []
        for variant_index in range(1, VARIANTS_PER_BASE + 1):
            seed = _derived_seed(master_seed, source.stable_id, variant_index)
            accepted: SceneVariant | None = None
            last_error: SceneVariantError | None = None
            for composition_attempt in range(constraints.maximum_variant_attempts):
                try:
                    candidate = _compose_one(
                        source,
                        seed=seed,
                        variant_index=variant_index,
                        composition_attempt=composition_attempt,
                        constraints=constraints,
                    )
                except SceneVariantError as error:
                    last_error = error
                    continue
                if _validate_diversity(candidate, base_variants, constraints):
                    accepted = candidate
                    break
            if accepted is None:
                detail = f": {last_error}" if last_error is not None else ""
                raise SceneVariantError(
                    f"{source.stable_id} variant {variant_index} cannot satisfy "
                    f"pairwise diversity after "
                    f"{constraints.maximum_variant_attempts} attempts{detail}"
                )
            base_variants.append(accepted)
            portfolio.append(accepted)
    if len(portfolio) != PORTFOLIO_SCENE_COUNT:
        raise AssertionError("internal error: variant portfolio is not 4 x 5")
    return tuple(portfolio)


def validate_scene_variant(
    source: BaseScene,
    variant: SceneVariant,
    *,
    constraints: VariantConstraints,
) -> None:
    """Re-run all structural and spatial checks on an authored variant."""

    _validate_base_scene(source, constraints)
    if variant.base_scene_id != source.stable_id:
        raise SceneVariantError("variant references a different base scene")
    if not 1 <= variant.variant_index <= VARIANTS_PER_BASE:
        raise SceneVariantError("variant index is outside the fixed 1..5 range")
    _validate_output_constraints(source, variant, constraints)
    _validate_transformation_contract(source, variant, constraints)

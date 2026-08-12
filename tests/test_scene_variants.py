from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import unittest
from unittest import mock

import fireviewer_sdg.scene_variants as scene_variants_module
from fireviewer_sdg.scene_variants import (
    ALGORITHM_ID,
    PORTFOLIO_SCENE_COUNT,
    BaseScene,
    Bounds,
    BridgeSpan,
    FamilySuitability,
    GroundSurfaceContract,
    HeightField,
    InvalidBaseScene,
    PlacementHeightTile,
    SceneAsset,
    SceneRoute,
    SceneVariantError,
    SuitabilityZone,
    TiledHeightField,
    VariantConstraints,
    Vec2,
    Vec3,
    WaterFeature,
    _validate_base_scene,
    family_counts,
    generate_scene_variants,
    global_stable_id,
    route_topology,
    validate_scene_variant,
)


def _terrain() -> HeightField:
    # A gentle southward descent: water and roads remain physically plausible.
    rows = tuple(
        tuple(100.0 - y * 0.5 + x * 0.02 for x in range(21))
        for y in range(21)
    )
    return HeightField(0.0, 0.0, 20.0, rows)


def _asset(
    stable_id: str,
    family: str,
    x: float,
    y: float,
    *,
    building: bool = False,
    group_id: str = "",
) -> SceneAsset:
    return SceneAsset(
        stable_id=stable_id,
        family=family,
        asset_ref=(
            f"omniverse://assets/buildings/{family}/{stable_id}.usdc"
            if building
            else f"omniverse://assets/trees/{family}/{stable_id}.usdc"
        ),
        position=Vec3(x, y, _terrain().elevation(Vec2(x, y))),
        heading_degrees=12.0,
        uniform_scale=1.0,
        footprint_radius_m=3.0 if building else 1.2,
        group_id=group_id,
    )


def _base(prefix: str) -> BaseScene:
    terrain = _terrain()
    trees = (
        _asset(f"{prefix}-tree-01", "pine", 45.0, 55.0),
        _asset(f"{prefix}-tree-02", "pine", 70.0, 80.0),
        _asset(f"{prefix}-tree-03", "pine", 95.0, 110.0),
        _asset(f"{prefix}-tree-04", "pine", 125.0, 135.0),
        _asset(f"{prefix}-tree-05", "oak", 275.0, 50.0),
        _asset(f"{prefix}-tree-06", "oak", 310.0, 85.0),
        _asset(f"{prefix}-tree-07", "oak", 285.0, 320.0),
        _asset(f"{prefix}-tree-08", "oak", 325.0, 345.0),
    )
    buildings = (
        _asset(
            f"{prefix}-house-01",
            "stone_house",
            55.0,
            185.0,
            building=True,
            group_id=f"{prefix}-hamlet-a",
        ),
        _asset(
            f"{prefix}-house-02",
            "stone_house",
            75.0,
            205.0,
            building=True,
            group_id=f"{prefix}-hamlet-a",
        ),
        _asset(
            f"{prefix}-barn-01",
            "barn",
            110.0,
            235.0,
            building=True,
            group_id=f"{prefix}-hamlet-a",
        ),
        _asset(
            f"{prefix}-house-03",
            "stone_house",
            300.0,
            195.0,
            building=True,
            group_id=f"{prefix}-hamlet-b",
        ),
        _asset(
            f"{prefix}-barn-02",
            "barn",
            320.0,
            225.0,
            building=True,
            group_id=f"{prefix}-hamlet-b",
        ),
    )
    # The two left-side roads share an explicit junction.  The east branch is
    # connected by a declared, elevated bridge crossing the central river.
    routes = (
        SceneRoute(
            stable_id=f"{prefix}-road-west",
            family="local_road",
            points=(
                Vec3(25.0, 40.0, terrain.elevation(Vec2(25.0, 40.0))),
                Vec3(100.0, 180.0, terrain.elevation(Vec2(100.0, 180.0))),
                Vec3(145.0, 350.0, terrain.elevation(Vec2(145.0, 350.0))),
            ),
            width_m=5.0,
        ),
        SceneRoute(
            stable_id=f"{prefix}-road-link",
            family="local_road",
            points=(
                Vec3(100.0, 180.0, terrain.elevation(Vec2(100.0, 180.0))),
                Vec3(145.0, 180.0, terrain.elevation(Vec2(145.0, 180.0))),
            ),
            width_m=4.0,
        ),
        SceneRoute(
            stable_id=f"{prefix}-road-east",
            family="local_road",
            points=(
                Vec3(255.0, 350.0, terrain.elevation(Vec2(255.0, 350.0))),
                Vec3(300.0, 180.0, terrain.elevation(Vec2(300.0, 180.0))),
                Vec3(370.0, 45.0, terrain.elevation(Vec2(370.0, 45.0))),
            ),
            width_m=5.0,
        ),
        SceneRoute(
            stable_id=f"{prefix}-road-north",
            family="bridge_road",
            points=(
                Vec3(145.0, 350.0, terrain.elevation(Vec2(145.0, 350.0))),
                Vec3(175.0, 350.0, terrain.elevation(Vec2(175.0, 350.0))),
                Vec3(225.0, 350.0, terrain.elevation(Vec2(225.0, 350.0))),
                Vec3(255.0, 350.0, terrain.elevation(Vec2(255.0, 350.0))),
            ),
            width_m=5.0,
            bridge_spans=(
                BridgeSpan(
                    stable_id=f"{prefix}-bridge-north",
                    start_fraction=0.0,
                    water_start_fraction=0.2,
                    water_end_fraction=0.8,
                    end_fraction=1.0,
                    minimum_deck_clearance_m=3.0,
                ),
            ),
        ),
    )
    water = WaterFeature(
        stable_id=f"{prefix}-river",
        family="river",
        outline=(
            Vec2(185.0, 10.0),
            Vec2(215.0, 10.0),
            Vec2(215.0, 390.0),
            Vec2(185.0, 390.0),
        ),
        kind="watercourse",
        centreline=(Vec2(200.0, 10.0), Vec2(200.0, 390.0)),
        surface_profile_m=(
            terrain.elevation(Vec2(200.0, 10.0)) + 1.0,
            terrain.elevation(Vec2(200.0, 390.0)) + 1.0,
        ),
    )
    zones = (
        SuitabilityZone(
            stable_id=f"{prefix}-pine-zone",
            outline=(
                Vec2(10.0, 10.0),
                Vec2(180.0, 10.0),
                Vec2(180.0, 390.0),
                Vec2(10.0, 390.0),
            ),
            biome="montane_conifer",
            soil="siliceous",
            tree_families=frozenset({"pine"}),
            buildable=True,
        ),
        SuitabilityZone(
            stable_id=f"{prefix}-oak-zone",
            outline=(
                Vec2(220.0, 10.0),
                Vec2(390.0, 10.0),
                Vec2(390.0, 390.0),
                Vec2(220.0, 390.0),
            ),
            biome="dry_deciduous",
            soil="calcareous",
            tree_families=frozenset({"oak"}),
            buildable=True,
        ),
    )
    return BaseScene(
        stable_id=prefix,
        bounds=Bounds(0.0, 0.0, 400.0, 400.0),
        terrain=terrain,
        ground_surface=GroundSurfaceContract(
            kind="object_free_pbr",
            material_ref=f"omniverse://materials/{prefix}/ground.mdl",
            content_fingerprint=("1" * 64),
        ),
        trees=trees,
        buildings=buildings,
        routes=routes,
        waters=(water,),
        suitability_zones=zones,
    )


def _constraints(**changes: object) -> VariantConstraints:
    values = {
        "edge_buffer_m": 1.0,
        "tree_water_buffer_m": 2.0,
        "building_water_buffer_m": 4.0,
        # The accepted base contains one explicit bridge.  Roads otherwise stay
        # out of water; this zero value permits the bridge crossing itself.
        "road_water_buffer_m": 0.0,
        "tree_road_buffer_m": 1.0,
        "tree_building_buffer_m": 1.0,
        "minimum_tree_spacing_m": 0.2,
        "minimum_building_spacing_m": 1.0,
        "maximum_building_road_distance_m": 35.0,
        "maximum_building_group_radius_m": 180.0,
        "maximum_foundation_relief_m": 1.5,
        "maximum_road_grade_percent": 20.0,
        "road_sample_spacing_m": 12.0,
        "route_warp_amplitude_m": 10.0,
        "route_rotation_degrees": 1.0,
        "route_scale_delta": 0.005,
        "tree_relocation_radius_m": 55.0,
        "minimum_moved_fraction": 0.65,
        "movement_epsilon_m": 1.0,
        "minimum_tree_intervariant_distance_m": 2.0,
        "minimum_building_intervariant_distance_m": 3.0,
        "minimum_route_intervariant_distance_m": 0.8,
        "maximum_route_attempts": 300,
        "maximum_building_attempts": 400,
        "maximum_tree_attempts": 150,
        "maximum_variant_attempts": 30,
        "tree_suitability": (
            FamilySuitability(
                "pine",
                frozenset({"montane_conifer"}),
                frozenset({"siliceous"}),
            ),
            FamilySuitability(
                "oak",
                frozenset({"dry_deciduous"}),
                frozenset({"calcareous"}),
            ),
        ),
    }
    values.update(changes)
    return VariantConstraints(**values)


class SceneVariantsTests(unittest.TestCase):
    def test_route_topology_hashes_exact_component_membership(self) -> None:
        def route(stable_id: str, start: tuple[float, float], end: tuple[float, float]):
            return SceneRoute(
                stable_id=stable_id,
                family="local_road",
                points=(
                    Vec3(start[0], start[1], 0.0),
                    Vec3(end[0], end[1], 0.0),
                ),
                width_m=4.0,
            )

        source = (
            route("a", (0.0, 0.0), (10.0, 0.0)),
            route("b", (10.0, 0.0), (20.0, 0.0)),
            route("c", (0.0, 100.0), (10.0, 100.0)),
            route("d", (10.0, 100.0), (20.0, 100.0)),
        )
        changed_membership = (
            route("a", (0.0, 0.0), (10.0, 0.0)),
            route("c", (10.0, 0.0), (20.0, 0.0)),
            route("b", (0.0, 100.0), (10.0, 100.0)),
            route("d", (10.0, 100.0), (20.0, 100.0)),
        )
        source_topology = route_topology(source, 0.25)
        self.assertEqual(source_topology, route_topology(tuple(reversed(source)), 0.25))
        self.assertEqual(source_topology[0], 2)
        self.assertEqual(route_topology(changed_membership, 0.25)[0], 2)
        self.assertNotEqual(
            source_topology[1],
            route_topology(changed_membership, 0.25)[1],
        )

    def test_spatial_route_topology_matches_quadratic_oracle(self) -> None:
        rng = random.Random(735_991)
        routes = tuple(
            SceneRoute(
                stable_id=f"route-{index:03d}",
                family="local_road",
                points=(
                    Vec3(
                        rng.uniform(-250.0, 250.0),
                        rng.uniform(-250.0, 250.0),
                        0.0,
                    ),
                    Vec3(
                        rng.uniform(-250.0, 250.0),
                        rng.uniform(-250.0, 250.0),
                        0.0,
                    ),
                ),
                width_m=4.0,
            )
            for index in range(80)
        )
        tolerance = 0.75
        parent = list(range(len(routes)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first: int, second: int) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                parent[second_root] = first_root

        for first_index, first_route in enumerate(routes):
            for second_index in range(first_index + 1, len(routes)):
                second_route = routes[second_index]
                if any(
                    scene_variants_module._segment_distance(
                        first_a.xy,
                        first_b.xy,
                        second_a.xy,
                        second_b.xy,
                    )
                    <= tolerance
                    for first_a, first_b in zip(
                        first_route.points,
                        first_route.points[1:],
                    )
                    for second_a, second_b in zip(
                        second_route.points,
                        second_route.points[1:],
                    )
                ):
                    union(first_index, second_index)
        grouped: dict[int, list[str]] = {}
        for index, route in enumerate(routes):
            grouped.setdefault(find(index), []).append(route.stable_id)
        components = tuple(
            sorted(
                tuple(sorted(component))
                for component in grouped.values()
            )
        )
        expected = (
            len(components),
            hashlib.sha256(
                json.dumps(
                    components,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )

        self.assertEqual(route_topology(routes, tolerance), expected)
        self.assertEqual(
            route_topology(tuple(reversed(routes)), tolerance),
            expected,
        )

    def test_spatial_route_topology_avoids_quadratic_sparse_comparisons(self) -> None:
        routes = tuple(
            SceneRoute(
                stable_id=f"sparse-{index:05d}",
                family="local_road",
                points=(
                    Vec3(index * 1000.0, 0.0, 0.0),
                    Vec3(index * 1000.0 + 10.0, 0.0, 0.0),
                ),
                width_m=4.0,
            )
            for index in range(5000)
        )
        real_distance = scene_variants_module._segment_distance
        with mock.patch.object(
            scene_variants_module,
            "_segment_distance",
            wraps=real_distance,
        ) as measured:
            component_count, _membership = route_topology(routes, 0.25)

        self.assertEqual(component_count, len(routes))
        self.assertLess(measured.call_count, len(routes) * 10)

    def test_tiled_height_field_preserves_local_relief_with_bounded_cache(self) -> None:
        loads: list[str] = []

        def tile(
            tile_ref: str,
            bounds: Bounds,
            samples: tuple[tuple[float, ...], ...],
        ) -> PlacementHeightTile:
            def load() -> tuple[tuple[float, ...], ...]:
                loads.append(tile_ref)
                return samples

            return PlacementHeightTile(
                tile_ref=tile_ref,
                bounds=bounds,
                x_coordinates=(
                    bounds.min_x,
                    (bounds.min_x + bounds.max_x) * 0.5,
                    bounds.max_x,
                ),
                y_coordinates=(
                    bounds.min_y,
                    (bounds.min_y + bounds.max_y) * 0.5,
                    bounds.max_y,
                ),
                sample_sha256=hashlib.sha256(
                    tile_ref.encode("utf-8")
                ).hexdigest(),
                sample_loader=load,
            )

        flat = (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        terrain = TiledHeightField(
            tiles=(
                tile(
                    "height-00",
                    Bounds(0.0, 0.0, 10.0, 10.0),
                    (
                        (0.0, 0.0, 0.0),
                        (0.0, 8.0, 0.0),
                        (0.0, 0.0, 0.0),
                    ),
                ),
                tile("height-10", Bounds(10.0, 0.0, 20.0, 10.0), flat),
                tile("height-01", Bounds(0.0, 10.0, 10.0, 20.0), flat),
                tile("height-11", Bounds(10.0, 10.0, 20.0, 20.0), flat),
            ),
            content_fingerprint="a" * 64,
            expected_tile_count=4,
            cache_tile_limit=2,
        )
        coarse_overview = HeightField(
            0.0,
            0.0,
            20.0,
            ((0.0, 0.0), (0.0, 0.0)),
        )

        self.assertEqual(coarse_overview.elevation(Vec2(5.0, 5.0)), 0.0)
        self.assertEqual(terrain.elevation(Vec2(5.0, 5.0)), 8.0)
        for point in (
            Vec2(15.0, 5.0),
            Vec2(5.0, 15.0),
            Vec2(15.0, 15.0),
        ):
            self.assertEqual(terrain.elevation(point), 0.0)
            self.assertLessEqual(terrain.cached_tile_count, 2)
        self.assertEqual(len(loads), 4)

    def test_source_route_may_end_exactly_on_a_clipped_scene_boundary(self) -> None:
        base = _base("boundary")
        route = base.routes[0]
        clipped = dataclasses.replace(
            route,
            points=(
                Vec3(
                    0.0,
                    route.points[0].y,
                    base.terrain.elevation(Vec2(0.0, route.points[0].y)),
                ),
                *route.points[1:],
            ),
        )
        accepted = dataclasses.replace(
            base, routes=(clipped, *base.routes[1:])
        )
        _validate_base_scene(accepted, _constraints())

        near_boundary = dataclasses.replace(
            clipped,
            points=(
                Vec3(
                    1.0,
                    clipped.points[0].y,
                    base.terrain.elevation(Vec2(1.0, clipped.points[0].y)),
                ),
                *clipped.points[1:],
            ),
        )
        rejected = dataclasses.replace(
            base, routes=(near_boundary, *base.routes[1:])
        )
        with self.assertRaisesRegex(InvalidBaseScene, "falls outside bounds"):
            _validate_base_scene(rejected, _constraints())

    def test_exact_four_by_five_is_deterministic_and_preserves_contract(self) -> None:
        bases = tuple(_base(f"base-{index}") for index in range(4))
        constraints = _constraints()

        first = generate_scene_variants(
            tuple(reversed(bases)),
            master_seed=0xF17E2026,
            constraints=constraints,
        )
        second = generate_scene_variants(
            bases,
            master_seed=0xF17E2026,
            constraints=constraints,
        )

        self.assertEqual(len(first), PORTFOLIO_SCENE_COUNT)
        self.assertEqual(first, second)
        self.assertEqual(len({item.contract.seed for item in first}), 20)
        portfolio_tree_ids = {
            global_stable_id(variant.base_scene_id, tree.stable_id)
            for variant in first
            for tree in variant.trees
        }
        self.assertEqual(
            len(portfolio_tree_ids),
            sum(len(base.trees) for base in bases),
        )
        self.assertEqual(
            [(item.base_scene_id, item.variant_index) for item in first],
            [
                (f"base-{base_index}", variant_index)
                for base_index in range(4)
                for variant_index in range(1, 6)
            ],
        )
        by_base = {base.stable_id: base for base in bases}
        for variant in first:
            source = by_base[variant.base_scene_id]
            self.assertIs(variant.terrain, source.terrain)
            self.assertIs(variant.ground_surface, source.ground_surface)
            self.assertIs(variant.waters, source.waters)
            self.assertEqual(family_counts(variant), family_counts(source))
            self.assertEqual(
                [item.stable_id for item in variant.trees],
                [item.stable_id for item in source.trees],
            )
            self.assertEqual(
                [item.asset_ref for item in variant.trees],
                [item.asset_ref for item in source.trees],
            )
            self.assertEqual(
                [item.asset_ref for item in variant.buildings],
                [item.asset_ref for item in source.buildings],
            )
            self.assertEqual(variant.contract.algorithm, ALGORITHM_ID)
            self.assertEqual(
                variant.contract.source_family_counts,
                variant.contract.result_family_counts,
            )
            self.assertEqual(
                variant.contract.stable_id_policy,
                "base-scene-qualified-identity-v1",
            )
            validate_scene_variant(source, variant, constraints=constraints)

    def test_every_variant_has_real_spatial_rearrangement_and_pairwise_diversity(self) -> None:
        bases = tuple(_base(f"diverse-{index}") for index in range(4))
        variants = generate_scene_variants(
            bases,
            master_seed=918273,
            constraints=_constraints(),
        )
        for base in bases:
            selected = [item for item in variants if item.base_scene_id == base.stable_id]
            for variant in selected:
                displacement = dict(variant.contract.displacement)
                self.assertGreaterEqual(displacement["trees"].moved_fraction, 0.65)
                self.assertGreaterEqual(displacement["buildings"].moved_fraction, 0.65)
                self.assertGreaterEqual(displacement["routes"].moved_fraction, 0.65)
                self.assertGreater(displacement["trees"].mean_distance_m, 1.0)
                self.assertGreater(displacement["buildings"].mean_distance_m, 1.0)
                self.assertGreater(displacement["routes"].mean_distance_m, 0.8)
            tree_layouts = {
                tuple((round(tree.position.x, 4), round(tree.position.y, 4)) for tree in item.trees)
                for item in selected
            }
            building_layouts = {
                tuple(
                    (round(building.position.x, 4), round(building.position.y, 4))
                    for building in item.buildings
                )
                for item in selected
            }
            road_layouts = {
                tuple(
                    (round(point.x, 4), round(point.y, 4))
                    for route in item.routes
                    for point in route.points
                )
                for item in selected
            }
            self.assertEqual(len(tree_layouts), 5)
            self.assertEqual(len(building_layouts), 5)
            self.assertEqual(len(road_layouts), 5)

    def test_output_has_water_road_building_and_habitat_exclusions(self) -> None:
        bases = tuple(_base(f"spatial-{index}") for index in range(4))
        constraints = _constraints()
        variants = generate_scene_variants(
            bases,
            master_seed=123456,
            constraints=constraints,
        )
        for variant in variants:
            for tree in variant.trees:
                # Public validation traverses all exact geometric constraints.
                self.assertGreater(tree.position.z, 0.0)
            for building in variant.buildings:
                self.assertGreater(building.position.z, 0.0)
            validate_scene_variant(
                next(base for base in bases if base.stable_id == variant.base_scene_id),
                variant,
                constraints=constraints,
            )

    def test_bridge_is_explicit_crosses_water_and_has_vertical_clearance(self) -> None:
        bases = tuple(_base(f"bridge-{index}") for index in range(4))
        constraints = _constraints()
        variant = generate_scene_variants(
            bases,
            master_seed=9876,
            constraints=constraints,
        )[0]
        bridge_route = next(
            route for route in variant.routes if route.bridge_spans
        )
        deck_points = [
            point
            for point in bridge_route.points
            if 185.0 <= point.x <= 215.0 and 10.0 <= point.y <= 390.0
        ]
        self.assertTrue(deck_points, "bridge must actually cross the water polygon")
        for point in deck_points:
            water_t = (point.y - 10.0) / 380.0
            water_z = (
                variant.waters[0].surface_profile_m[0] * (1.0 - water_t)
                + variant.waters[0].surface_profile_m[1] * water_t
            )
            self.assertGreaterEqual(point.z - water_z, 3.0 - 1.0e-6)

        north_index = next(
            index
            for index, route in enumerate(bases[0].routes)
            if route.bridge_spans
        )
        undeclared = dataclasses.replace(
            bases[0].routes[north_index],
            bridge_spans=(),
        )
        invalid_base = dataclasses.replace(
            bases[0],
            routes=(
                bases[0].routes[:north_index]
                + (undeclared,)
                + bases[0].routes[north_index + 1 :]
            ),
        )
        with self.assertRaisesRegex(InvalidBaseScene, "undeclared water crossing"):
            generate_scene_variants(
                (invalid_base,) + bases[1:],
                master_seed=9876,
                constraints=constraints,
            )

    def test_raw_or_incompletely_cleaned_ground_imagery_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "raw orthophotos are forbidden"):
            GroundSurfaceContract(
                kind="raw_orthophoto",
                material_ref="omniverse://materials/raw-ground.mdl",
                content_fingerprint="2" * 64,
            )
        with self.assertRaisesRegex(
            ValueError, "remove trees, buildings and routes"
        ):
            GroundSurfaceContract(
                kind="object_removed_orthomosaic",
                material_ref="omniverse://materials/partly-clean-ground.mdl",
                content_fingerprint="3" * 64,
                removed_object_classes=frozenset({"trees"}),
            )

    def test_wrong_base_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 4"):
            generate_scene_variants(
                (_base("only-one"),),
                master_seed=1,
                constraints=_constraints(),
            )

    def test_primitive_or_placeholder_asset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden"):
            _asset("bad", "pine", 20.0, 20.0).__class__(
                stable_id="bad",
                family="pine",
                asset_ref="omniverse://assets/primitive/Cube.usd",
                position=Vec3(20.0, 20.0, 90.0),
                heading_degrees=0.0,
                uniform_scale=1.0,
                footprint_radius_m=1.0,
            )

    def test_missing_biome_contract_fails_closed(self) -> None:
        bases = tuple(_base(f"biome-{index}") for index in range(4))
        constraints = _constraints(
            tree_suitability=(
                FamilySuitability(
                    "pine",
                    frozenset({"montane_conifer"}),
                    frozenset({"siliceous"}),
                ),
            )
        )
        with self.assertRaisesRegex(InvalidBaseScene, "oak"):
            generate_scene_variants(
                bases,
                master_seed=4,
                constraints=constraints,
            )

    def test_impossible_foundation_fails_without_partial_result(self) -> None:
        bases = tuple(_base(f"steep-{index}") for index in range(4))
        constraints = _constraints(maximum_building_slope_percent=0.01)
        with self.assertRaises(SceneVariantError):
            generate_scene_variants(
                bases,
                master_seed=55,
                constraints=constraints,
            )

    def test_mutated_variant_fingerprint_is_rejected(self) -> None:
        bases = tuple(_base(f"mutate-{index}") for index in range(4))
        constraints = _constraints()
        variant = generate_scene_variants(
            bases,
            master_seed=77,
            constraints=constraints,
        )[0]
        moved = dataclasses.replace(
            variant.trees[0],
            position=Vec3(
                variant.trees[0].position.x + 1.0,
                variant.trees[0].position.y,
                variant.trees[0].position.z,
            ),
        )
        mutated = dataclasses.replace(
            variant,
            trees=(moved,) + variant.trees[1:],
        )
        with self.assertRaises(SceneVariantError):
            validate_scene_variant(
                bases[0],
                mutated,
                constraints=constraints,
            )


if __name__ == "__main__":
    unittest.main()

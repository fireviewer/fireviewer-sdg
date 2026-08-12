from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest import mock

from fireviewer_sdg.native_variant_campaign import (
    NativeVariantContractError,
    _actor_deployments,
    _authored_identity_contract,
    _iter_base_variants,
    _mesh_vertex_normals,
    _ribbon_vertices,
    _triangulated_surface,
    _supplemental_environment_deployments,
    author_variant_campaign,
    load_native_base_layout,
    prepare_variant_campaign,
    verify_variant_campaign,
)
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
from tests.test_scene_variants import _base, _constraints
from fireviewer_sdg.scene_variants import (
    SceneAsset,
    VariantConstraints,
    Vec2,
    Vec3,
    route_topology,
)


def _campaign_constraints() -> VariantConstraints:
    # Coverage trees start near each 20 m test-tile centre.  A four-metre
    # relocation radius keeps every real tree in its streaming tile while
    # preserving the required inter-variant movement.
    return _constraints(
        tree_relocation_radius_m=4.0,
        route_warp_amplitude_m=1.0,
        route_rotation_degrees=0.1,
        route_scale_delta=0.001,
    )


def _segment_distance(
    point: Vec2, first: Vec2, second: Vec2
) -> float:
    dx, dy = second.x - first.x, second.y - first.y
    denominator = dx * dx + dy * dy
    if denominator <= 1.0e-12:
        return math.hypot(point.x - first.x, point.y - first.y)
    ratio = min(
        1.0,
        max(
            0.0,
            ((point.x - first.x) * dx + (point.y - first.y) * dy)
            / denominator,
        ),
    )
    return math.hypot(
        point.x - (first.x + ratio * dx),
        point.y - (first.y + ratio * dy),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact(
    volume: Path,
    path: Path,
    *,
    prim_path: str = "",
    roles: list[str] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "path": path.relative_to(volume).as_posix(),
        "sha256": _sha256(path),
    }
    if prim_path:
        result["prim_path"] = prim_path
    if roles is not None:
        result["isolated_content_roles"] = roles
    return result


def _touch_usd(path: Path, prim_name: str = "Asset") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'#usda 1.0\n\ndef Xform "{prim_name}" {{\n}}\n',
        encoding="utf-8",
    )


def _touch_review_cameras(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    children = "\n".join(
        (
            f'    def Camera "Review{index:02d}"\n'
            "    {\n"
            '        custom string "fireviewer:look_at_local" = "200,200,0"\n'
            "    }"
        )
        for index in range(1, 13)
    )
    path.write_text(
        '#usda 1.0\n\ndef Xform "ReviewCameras"\n{\n'
        + children
        + "\n}\n",
        encoding="utf-8",
    )


def _asset_library(volume: Path, base_root: Path) -> dict[str, object]:
    records: dict[str, object] = {}
    families = (
        ("tree-pine", "trees", "pine"),
        ("tree-oak", "trees", "oak"),
        ("building-house", "buildings", "stone_house"),
        ("building-barn", "buildings", "barn"),
    )
    for key, category, family in families:
        lods: dict[str, object] = {}
        for level in ("HERO", "MID", "FAR"):
            path = base_root / "assets" / f"{key}_{level}.usdc"
            _touch_usd(path)
            lods[level] = _artifact(volume, path, prim_path="/Asset")
        records[key] = {
            "category": category,
            "family": family,
            "lods": lods,
            "simready_validation": {
                "state": "SIMREADY_VALIDATED",
                "lod_lineage": f"lineage-{key}",
                "grounding_offsets_m": {
                    "HERO": 0.0,
                    "MID": 0.0,
                    "FAR": 0.0,
                },
            },
        }
    return records


def _water_materials(volume: Path, base_root: Path) -> dict[str, object]:
    levels: dict[str, object] = {}
    for level in ("HERO", "MID", "FAR"):
        path = base_root / "materials" / f"water_{level}.usdc"
        _touch_usd(path, "Material")
        levels[level] = _artifact(volume, path, prim_path="/Material")
    return levels


def _height_payload(scene: object) -> dict[str, object]:
    terrain = scene.terrain
    return {
        "origin_x": terrain.origin_x,
        "origin_y": terrain.origin_y,
        "spacing_m": terrain.spacing_m,
        "samples": [list(row) for row in terrain.samples],
    }


def _routes(scene: object) -> list[dict[str, object]]:
    return [
        {
            "stable_id": route.stable_id,
            "numeric_id": 10_000 + index,
            "family": route.family,
            "surface_class": (
                "bridge" if route.bridge_spans else "local"
            ),
            "width_m": route.width_m,
            "points": [
                [point.x, point.y, point.z] for point in route.points
            ],
            "bridge_spans": [
                dataclasses.asdict(span) for span in route.bridge_spans
            ],
        }
        for index, route in enumerate(scene.routes)
    ]


def _waters(scene: object) -> list[dict[str, object]]:
    return [
        {
            "stable_id": water.stable_id,
            "family": water.family,
            "outline": [[point.x, point.y] for point in water.outline],
            "kind": water.kind,
            "centreline": [
                [point.x, point.y] for point in water.centreline
            ],
            "surface_profile_m": list(water.surface_profile_m),
        }
        for water in scene.waters
    ]


def _zones(scene: object) -> list[dict[str, object]]:
    return [
        {
            "stable_id": zone.stable_id,
            "outline": [[point.x, point.y] for point in zone.outline],
            "biome": zone.biome,
            "soil": zone.soil,
            "tree_families": sorted(zone.tree_families),
            "buildable": zone.buildable,
        }
        for zone in scene.suitability_zones
    ]


def _object_asset_key(category: str, family: str) -> str:
    if category == "trees":
        return f"tree-{family}"
    return "building-barn" if family == "barn" else "building-house"


def _write_objects(
    *,
    volume: Path,
    base_root: Path,
    category: str,
    values: object,
    numeric_start: int,
) -> dict[str, object]:
    path = base_root / f"{category}.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for index, item in enumerate(values):
            stream.write(
                json.dumps(
                    {
                        "stable_id": item.stable_id,
                        "numeric_id": numeric_start + index,
                        "asset_key": _object_asset_key(
                            category, item.family
                        ),
                        "position": [
                            item.position.x,
                            item.position.y,
                            item.position.z,
                        ],
                        "heading_degrees": item.heading_degrees,
                        "uniform_scale": item.uniform_scale,
                        "footprint_radius_m": item.footprint_radius_m,
                        "group_id": item.group_id,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    result = _artifact(volume, path)
    result["count"] = len(values)
    return result


def _write_base_layout(volume: Path, base_id: str) -> Path:
    scene = _base(base_id)
    coverage_trees: list[SceneAsset] = []
    for index in range(400):
        column = index % 20
        row = index // 20
        if column in {9, 10}:
            continue
        family = "pine" if column < 9 else "oak"
        minimum_x = column * 20.0 + 3.0
        maximum_x = (column + 1) * 20.0 - 3.0
        minimum_y = row * 20.0 + 3.0
        maximum_y = (row + 1) * 20.0 - 3.0
        if family == "pine":
            minimum_x = max(minimum_x, 12.0)
            maximum_x = min(maximum_x, 178.0)
        else:
            minimum_x = max(minimum_x, 222.0)
            maximum_x = min(maximum_x, 388.0)
        candidates: list[Vec2] = []
        for x_step in range(8):
            for y_step in range(8):
                x = minimum_x + (maximum_x - minimum_x) * x_step / 7.0
                y = minimum_y + (maximum_y - minimum_y) * y_step / 7.0
                point = Vec2(x, y)
                route_distance = min(
                    _segment_distance(point, first.xy, second.xy)
                    - route.width_m * 0.5
                    for route in scene.routes
                    for first, second in zip(route.points, route.points[1:])
                )
                building_distance = min(
                    math.hypot(
                        point.x - building.position.x,
                        point.y - building.position.y,
                    )
                    - building.footprint_radius_m
                    for building in scene.buildings
                )
                if route_distance < 5.0 or building_distance < 6.0:
                    continue
                if any(
                    math.hypot(point.x - other.x, point.y - other.y) < 3.0
                    for other in candidates
                ):
                    continue
                candidates.append(point)
        tile_centre = Vec2(column * 20.0 + 10.0, row * 20.0 + 10.0)
        candidates.sort(
            key=lambda point: (
                math.hypot(
                    point.x - tile_centre.x,
                    point.y - tile_centre.y,
                ),
                point.x,
                point.y,
            )
        )
        for local_index, point in enumerate(candidates[:1]):
            x, y = point.x, point.y
            point = Vec2(x, y)
            coverage_trees.append(
                SceneAsset(
                    stable_id=(
                        f"{base_id}-coverage-tree-{index:04d}-"
                        f"{local_index:02d}"
                    ),
                    family=family,
                    asset_ref=(
                        f"omniverse://assets/trees/{family}/coverage.usdc"
                    ),
                    position=Vec3(x, y, scene.terrain.elevation(point)),
                    heading_degrees=float((index * 4 + local_index) % 360),
                    uniform_scale=1.0,
                    footprint_radius_m=0.35,
                    group_id=f"{base_id}-forest-{column:02d}-{row:02d}",
                )
            )
        if not candidates:
            # A genuinely occupied road/building tile remains represented by
            # that real feature; no synthetic tree is introduced.
            continue
    scene = dataclasses.replace(
        scene, trees=scene.trees + tuple(coverage_trees)
    )
    constraint_payload = dataclasses.asdict(_campaign_constraints())
    constraint_payload.pop("tree_suitability")
    route_component_count, route_membership_sha = route_topology(
        scene.routes,
        _campaign_constraints().road_connectivity_tolerance_m,
    )
    base_root = volume / "bases" / base_id
    base_root.mkdir(parents=True)
    epsg_origin = (700_000.0, 6_600_000.0, 0.0)
    terrain_records: list[dict[str, object]] = []
    for index in range(400):
        column = index % 20
        row = index // 20
        terrain = base_root / "terrain" / f"tile_{index:04d}.usdc"
        _touch_usd(terrain, "Tile")
        record = _artifact(
            volume,
            terrain,
            prim_path="/Tile",
            roles=["terrain"],
        )
        record.update(
            {
                "tile_ref": f"{base_id}-tile-{index:04d}",
                "instance_namespace": index + 1,
                "local_bounds": {
                    "min_x": column * 20.0,
                    "min_y": row * 20.0,
                    "max_x": (column + 1) * 20.0,
                    "max_y": (row + 1) * 20.0,
                },
                "epsg2154_bounds": {
                    "min_x": epsg_origin[0] + column * 20.0,
                    "min_y": epsg_origin[1] + row * 20.0,
                    "max_x": epsg_origin[0] + (column + 1) * 20.0,
                    "max_y": epsg_origin[1] + (row + 1) * 20.0,
                },
                "terrain_lods": (
                    ["LOD0", "LOD1", "LOD2", "LOD3"]
                    if index < 12
                    else ["LOD1", "LOD2", "LOD3"]
                ),
                "collision_lods": ["NEAR", "FAR"],
            }
        )
        terrain_records.append(record)
    water = base_root / "water.usdc"
    _touch_usd(water, "Water")
    water_record = _artifact(
        volume, water, prim_path="/Water", roles=["water"]
    )
    ground = base_root / "materials" / "ground-index.usdc"
    _touch_usd(ground, "Ground")
    ground_record = _artifact(
        volume,
        ground,
        prim_path="/Ground",
        roles=["object_free_pbr_ground"],
    )
    ground_record["topology"] = (
        "payload_tiled_materials_shared_pbr_library"
    )
    ground_payloads: list[dict[str, object]] = []
    for index, terrain_record in enumerate(terrain_records):
        material = (
            base_root / "materials" / "ground" / f"region_{index:02d}.usdc"
        )
        _touch_usd(material, "Ground")
        material_record = _artifact(
            volume,
            material,
            prim_path="/Ground",
        )
        material_record.update(
            {
                "tile_id": terrain_record["tile_ref"],
                "tile_ref": terrain_record["tile_ref"],
                "tile_bounds_m": [
                    terrain_record["local_bounds"]["min_x"],
                    terrain_record["local_bounds"]["min_y"],
                    terrain_record["local_bounds"]["max_x"],
                    terrain_record["local_bounds"]["max_y"],
                ],
            }
        )
        ground_payloads.append(material_record)
    ground_record["tile_material_payloads"] = ground_payloads
    height = base_root / "height-field.json"
    _write_json(height, _height_payload(scene))
    placement_height_records: list[dict[str, object]] = []
    for index, terrain_record in enumerate(terrain_records):
        bounds = terrain_record["local_bounds"]
        x_coordinates = [
            float(bounds["min_x"]),
            (float(bounds["min_x"]) + float(bounds["max_x"])) * 0.5,
            float(bounds["max_x"]),
        ]
        y_coordinates = [
            float(bounds["min_y"]),
            (float(bounds["min_y"]) + float(bounds["max_y"])) * 0.5,
            float(bounds["max_y"]),
        ]
        samples = array(
            "f",
            (
                scene.terrain.elevation(Vec2(x, y))
                for y in y_coordinates
                for x in x_coordinates
            ),
        )
        if sys.byteorder != "little":
            samples.byteswap()
        sample_path = (
            base_root
            / "placement-height"
            / f"tile_{index:04d}.f32"
        )
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_bytes(samples.tobytes())
        placement_height_records.append(
            {
                "tile_ref": terrain_record["tile_ref"],
                "local_bounds": dict(bounds),
                "path": sample_path.relative_to(volume).as_posix(),
                "sha256": _sha256(sample_path),
                "format": "float32-le-row-major-south-to-north",
                "width": 3,
                "height": 3,
                "x_coordinates": x_coordinates,
                "y_coordinates": y_coordinates,
            }
        )
    placement_height_fingerprint = hashlib.sha256(
        json.dumps(
            sorted(
                placement_height_records,
                key=lambda item: item["tile_ref"],
            ),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    root = base_root / "build" / "root.usdc"
    _touch_usd(root, "World")
    cameras = base_root / "build" / "review-cameras.usda"
    _touch_review_cameras(cameras)
    asset_manifest = volume / "shared" / "assets" / "manifest.json"
    if not asset_manifest.exists():
        selected_actors: dict[str, object] = {}
        for selection_id in SELECTED_ACTOR_GROUP_IDS:
            lods: dict[str, object] = {}
            for level in ("HERO", "MID", "FAR"):
                actor_path = (
                    volume
                    / "shared"
                    / "assets"
                    / "selected-actors"
                    / selection_id
                    / f"{level.casefold()}.usdc"
                )
                _touch_usd(actor_path)
                lods[level] = _artifact(
                    volume, actor_path, prim_path="/Asset"
                )
            selected_actors[selection_id] = {
                "selection_id": selection_id,
                "selection_source_url": (
                    SELECTED_ACTOR_GROUP_SOURCE_BY_ID[selection_id]
                ),
                "placement_class": (
                    SELECTED_ACTOR_GROUP_PLACEMENT_BY_ID[selection_id]
                ),
                "asset_id": f"selected-actor.{selection_id}",
                "family": f"selected-actor.{selection_id}",
                "lod_paths": lods,
                "ground_anchor_m": [0.0, 0.0, 0.0],
            }
        supplemental_environment: dict[str, object] = {}
        for selection_id in SELECTED_ENVIRONMENT_GROUP_IDS:
            kind, family = SELECTED_ENVIRONMENT_TARGET_BY_ID[selection_id]
            lods = {}
            for level in ("HERO", "MID", "FAR"):
                environment_path = (
                    volume
                    / "shared"
                    / "assets"
                    / "supplemental-environment"
                    / selection_id
                    / f"{level.casefold()}.usdc"
                )
                _touch_usd(environment_path)
                lods[level] = _artifact(
                    volume, environment_path, prim_path="/Asset"
                )
            supplemental_environment[selection_id] = {
                "selection_id": selection_id,
                "asset_id": f"selected-environment.{selection_id}",
                "environment_kind": kind,
                "environment_family": family,
                "lod_paths": lods,
                "ground_anchor_m": [0.0, 0.0, 0.0],
            }
        _write_json(
            asset_manifest,
            {
                "schema_version": 1,
                "profile": "test-photoreal",
                "assets": ["trees", "buildings"],
                "actors": {
                    role: {"asset_id": f"semantic-role.{role}"}
                    for role in REQUIRED_ACTOR_CLASSES
                },
                "selected_actor_group": {
                    "group_id": SELECTED_ACTOR_GROUP_ID,
                    "selection_count": len(SELECTED_ACTOR_GROUP_IDS),
                    "selection_order": list(SELECTED_ACTOR_GROUP_IDS),
                    "assets": selected_actors,
                    "usage_contract": (
                        "all_selected_assets_must_be_placed_across_the_20_scene_campaign"
                    ),
                },
                "selected_environment_group": {
                    "group_id": SELECTED_ENVIRONMENT_GROUP_ID,
                    "selection_count": len(
                        SELECTED_ENVIRONMENT_GROUP_IDS
                    ),
                    "selection_order": list(
                        SELECTED_ENVIRONMENT_GROUP_IDS
                    ),
                    "assets": supplemental_environment,
                    "usage_contract": (
                        "all_four_assets_are_additive_and_used_in_every_variant"
                    ),
                },
            },
        )
    asset_content_sha = hashlib.sha256(
        b"test-shared-materialized-assets"
    ).hexdigest()
    asset_lock = base_root / "build" / "metadata" / "asset-lock.json"
    shared_asset_lock = {
        "id": "shared-materialized-simready-environment",
        "manifest": asset_manifest.relative_to(volume).as_posix(),
        "manifest_sha256": _sha256(asset_manifest),
        "validation": {
            "state": "ASSETS_LOCKED",
            "asset_content_sha256": asset_content_sha,
        },
    }
    _write_json(
        asset_lock,
        {
            "zone_id": base_id,
            "assets": [shared_asset_lock],
        },
    )
    fake_details = [
        {"path": f"bases/{base_id}/details/{index:04d}.usdc", "sha256": "d" * 64}
        for index in range(400)
    ]
    build_receipt = base_root / "build" / "build-receipt.json"
    _write_json(
        build_receipt,
        {
            "schema_version": 2,
            "zone_id": base_id,
            "source_profile": "full",
            "root_usd": {
                "path": root.relative_to(base_root).as_posix(),
                "sha256": _sha256(root),
            },
            "cameras": {
                "path": cameras.relative_to(base_root).as_posix(),
                "sha256": _sha256(cameras),
                "count": 12,
            },
            "asset_lock": {
                "path": asset_lock.relative_to(base_root).as_posix(),
                "sha256": _sha256(asset_lock),
                "assets": [shared_asset_lock],
            },
            "payloads": [
                {
                    "path": (
                        volume / item["path"]
                    ).relative_to(base_root).as_posix(),
                    "sha256": item["sha256"],
                }
                for item in terrain_records
            ],
            "detail_payloads": fake_details,
            "detail_mid_payloads": fake_details,
            "detail_far_payloads": fake_details,
            "layers": {
                "vegetation": {"prim_count": len(scene.trees)},
                "buildings": {"prim_count": len(scene.buildings)},
                "roads": {
                    "prim_count": 0,
                    "source_feature_count": len(scene.routes),
                    "visible_representation": (
                        "orthophoto_derived_terrain_material"
                    ),
                    "geometry_authoring": "disabled",
                    "asset_dependencies": [],
                },
                "hydrology": {"prim_count": len(scene.waters)},
            },
            "tile_coverage": [
                {
                    "tile_ref": item["tile_ref"],
                    "terrain_payload": (
                        volume / item["path"]
                    ).relative_to(base_root).as_posix(),
                    "instance_namespace": item["instance_namespace"],
                    "terrain_lods": item["terrain_lods"],
                    "collision_lods": item["collision_lods"],
                }
                for item in terrain_records
            ],
            "fire_simulation_status": "blocked_pending_editor_review",
        },
    )
    auto = base_root / "auto-validation.json"
    _write_json(
        auto,
        {
            "state": "AUTO_VALIDATED",
            "fire_simulation_status": "blocked_pending_editor_review",
            "build_receipt_sha256": _sha256(build_receipt),
            "root_usd_sha256": _sha256(root),
        },
    )
    layout = base_root / "variant-layout.json"
    _write_json(
        layout,
        {
            "schema_version": 1,
            "base_scene_id": base_id,
            "epsg2154_origin": list(epsg_origin),
            "native_build_receipt": _artifact(volume, build_receipt),
            "scene_auto_validation": _artifact(volume, auto),
            "terrain_payloads": terrain_records,
            "water_payloads": [water_record],
            "ground_material": ground_record,
            "height_field": _artifact(volume, height),
            "placement_height_tiles": placement_height_records,
            "placement_height_fingerprint": (
                placement_height_fingerprint
            ),
            "ground_surface": {
                "kind": "object_free_pbr",
                "content_fingerprint": ground_record["sha256"],
                "removed_object_classes": [],
            },
            "bounds": {
                "min_x": scene.bounds.min_x,
                "min_y": scene.bounds.min_y,
                "max_x": scene.bounds.max_x,
                "max_y": scene.bounds.max_y,
            },
            "asset_library": _asset_library(volume, base_root),
            "water_material_lods": _water_materials(volume, base_root),
            "trees": _write_objects(
                volume=volume,
                base_root=base_root,
                category="trees",
                values=scene.trees,
                numeric_start=1,
            ),
            "buildings": _write_objects(
                volume=volume,
                base_root=base_root,
                category="buildings",
                values=scene.buildings,
                numeric_start=1_000_000,
            ),
            "routes": _routes(scene),
            "route_topology": {
                "algorithm": "segment-connectivity-components-v1",
                "tolerance_m": (
                    _campaign_constraints().road_connectivity_tolerance_m
                ),
                "source_component_count": route_component_count,
                "source_membership_sha256": route_membership_sha,
            },
            "waters": _waters(scene),
            "suitability_zones": _zones(scene),
            "variant_constraints": constraint_payload,
        },
    )
    return layout


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def author_variant(
        self,
        *,
        variant_dir: Path,
        metadata: dict[str, object],
        tile_records: list[dict[str, object]],
        volume_root: Path,
    ) -> dict[str, object]:
        variant_dir.mkdir(parents=True, exist_ok=False)
        self.calls.append(str(metadata["simulation_id"]))
        self.asserted_tiles = len(tile_records)
        root = variant_dir / "build" / "root.usdc"
        root.parent.mkdir(parents=True, exist_ok=True)
        root.write_bytes(b"native-usd-test-double")
        lods: dict[str, list[dict[str, str]]] = {
            level: [] for level in ("HERO", "MID", "FAR")
        }
        coverage: list[dict[str, object]] = []
        build_coverage: list[dict[str, object]] = []
        terrain_catalog: list[dict[str, str]] = []
        for index, tile in enumerate(tile_records):
            detail_lods: dict[str, dict[str, str]] = {}
            for level in ("HERO", "MID", "FAR"):
                path = (
                    variant_dir
                    / "details"
                    / level.lower()
                    / f"tile_{index:04d}.usdc"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{level}-{index}".encode("ascii"))
                artifact = {
                    "path": path.relative_to(variant_dir).as_posix(),
                    "sha256": _sha256(path),
                }
                lods[level].append(artifact)
                detail_lods[level] = artifact
            coverage.append(
                {
                    "tile_ref": tile["tile_ref"],
                    "instance_namespace": tile["instance_namespace"],
                    "local_bounds": tile["local_bounds"],
                    "epsg2154_bounds": tile["epsg2154_bounds"],
                    "terrain_payload": {
                        key: tile[key]
                        for key in (
                            "path",
                            "sha256",
                            "prim_path",
                            "isolated_content_roles",
                        )
                    },
                    "object_count": tile["objects"]["count"],
                    "route_fragment_count": tile["routes"][
                        "fragment_count"
                    ],
                    "hydrology_fragment_count": tile["routes"][
                        "hydrology_fragment_count"
                    ],
                    "detail_lod_counts": tile["detail_lod_counts"],
                    "detail_lods": detail_lods,
                }
            )
            terrain_path = (volume_root / tile["path"]).resolve()
            terrain_artifact = {
                "path": Path(
                    os.path.relpath(terrain_path, start=variant_dir)
                ).as_posix(),
                "sha256": tile["sha256"],
            }
            terrain_catalog.append(terrain_artifact)
            build_detail_lods = {
                level: lods[level][index]["path"]
                for level in ("HERO", "MID", "FAR")
            }
            build_coverage.append(
                {
                    "tile_ref": tile["tile_ref"],
                    "terrain_payload": terrain_artifact["path"],
                    "detail_payload": build_detail_lods["HERO"],
                    "detail_lods": build_detail_lods,
                    "terrain_lods": tile["terrain_lods"],
                    "collision_lods": tile["collision_lods"],
                    "detail_counts": tile["detail_lod_counts"]["HERO"],
                    "detail_lod_counts": tile["detail_lod_counts"],
                    "instance_namespace": tile["instance_namespace"],
                    "local_bounds": tile["local_bounds"],
                    "epsg2154_bounds": tile["epsg2154_bounds"],
                }
            )
        base_bindings = metadata["base_bindings"]

        def shared_artifact(record: dict[str, object]) -> dict[str, object]:
            source = (volume_root / str(record["path"])).resolve()
            result: dict[str, object] = {
                "path": Path(
                    os.path.relpath(source, start=variant_dir)
                ).as_posix(),
                "sha256": record["sha256"],
            }
            for key in ("prim_path", "isolated_content_roles"):
                if key in record:
                    result[key] = record[key]
            return result

        water_catalog = [
            shared_artifact(record)
            for record in base_bindings["water_payloads"]
        ]
        ground_source = base_bindings["ground_material"]
        ground_receipt = {
            "topology": ground_source["topology"],
            "index": shared_artifact(ground_source),
            "tile_material_payloads": [
                {
                    **shared_artifact(record),
                    "tile_id": record["tile_id"],
                    "tile_bounds_m": record["tile_bounds_m"],
                }
                for record in ground_source["tile_material_payloads"]
            ],
            "binding_scope": "per_terrain_tile_stronger_than_descendants",
        }
        authored_identity = _authored_identity_contract(
            metadata["identity_contract"]
        )
        actor_ids = [
            item["selection_id"]
            for item in metadata["actor_deployments"]
        ]
        environment_ids = [
            item["selection_id"]
            for item in metadata["supplemental_environment_deployments"]
        ]
        cameras = shared_artifact(base_bindings["review_cameras"])
        cameras["count"] = base_bindings["review_cameras"]["count"]
        cameras["root_prim"] = "/ReviewCameras"
        asset_lock = shared_artifact(base_bindings["asset_lock"])
        asset_lock["assets"] = base_bindings["asset_lock"]["assets"]
        asset_lock["shared_manifest"] = {
            **shared_artifact(base_bindings["shared_asset_manifest"]),
            "content_sha256": base_bindings["asset_content_sha256"],
        }
        build_receipt = variant_dir / "build" / "build-receipt.json"
        _write_json(
            build_receipt,
            {
                "schema_version": 2,
                "zone_id": metadata["simulation_id"],
                "variant_id": metadata["variant_id"],
                "base_scene_id": metadata["base_scene_id"],
                "variant_index": metadata["variant_index"],
                "scene_kind": "fictive_variant",
                "source_profile": "full",
                "root_usd": {
                    "path": root.relative_to(variant_dir).as_posix(),
                    "sha256": _sha256(root),
                },
                "payloads": terrain_catalog,
                "detail_payloads": lods["HERO"],
                "detail_mid_payloads": lods["MID"],
                "detail_far_payloads": lods["FAR"],
                "water_payloads": water_catalog,
                "water_contract": metadata["water_contract"],
                "tile_coverage": build_coverage,
                "cameras": cameras,
                "asset_lock": asset_lock,
                "identity_contract": authored_identity,
                "route_topology": metadata["route_topology"],
                "ground_material": ground_receipt,
                "layers": {
                    "terrain": {
                        "prim_count": 400,
                        "ground_material_payload_count": 400,
                        "global_ground_material_binding": False,
                    },
                    "actors": {
                        "prim_count": len(actor_ids),
                        "selected_actor_ids": actor_ids,
                        "group_id": SELECTED_ACTOR_GROUP_ID,
                        "lod_levels": ["HERO", "MID", "FAR"],
                        "placeholder_substitution": False,
                    },
                    "supplemental_environment": {
                        "prim_count": len(environment_ids),
                        "selected_environment_ids": environment_ids,
                        "group_id": SELECTED_ENVIRONMENT_GROUP_ID,
                        "additive_to_existing_minima": True,
                        "placeholder_substitution": False,
                    },
                    "roads": {
                        "prim_count": 0,
                        "source_feature_count": metadata[
                            "source_route_count"
                        ],
                        "route_fragment_count": metadata[
                            "route_fragment_count"
                        ],
                        "visible_representation": (
                            "orthophoto_derived_terrain_material"
                        ),
                        "geometry_authoring": "disabled",
                        "asset_dependencies": [],
                        "vertices_by_lod": {
                            level: 0
                            for level in ("HERO", "MID", "FAR")
                        },
                        "faces_by_lod": {
                            level: 0
                            for level in ("HERO", "MID", "FAR")
                        },
                    },
                    "hydrology": {
                        "prim_count": metadata[
                            "hydrology_fragment_count"
                        ],
                        "source_feature_count": len(
                            metadata["water_contract"][
                                "source_feature_ids"
                            ]
                        ),
                        "vertices_by_lod": {
                            level: metadata[
                                "hydrology_fragment_count"
                            ]
                            * 4
                            for level in ("HERO", "MID", "FAR")
                        },
                        "faces_by_lod": {
                            level: metadata[
                                "hydrology_fragment_count"
                            ]
                            * 2
                            for level in ("HERO", "MID", "FAR")
                        },
                    },
                    "detail_streaming": {
                        "network_geometry_policy": (
                            "hydrology_hash_bound_20m_fragments_all_lods;"
                            "roads_orthophoto_derived_terrain_material_no_route_meshes;"
                            "measured_vertex_budget"
                        ),
                        "network_vertex_budget_per_tile": 262_144,
                    },
                },
                "fire_simulation_status": (
                    "blocked_pending_editor_review"
                ),
            },
        )
        return {
            "root_usd": {
                "path": root.relative_to(variant_dir).as_posix(),
                "sha256": _sha256(root),
            },
            "streaming_tile_count": 400,
            "object_lod_payload_count": 1200,
            "scene_kind": "fictive_variant",
            "identity_contract": authored_identity,
            "actor_usage": {
                "group_id": SELECTED_ACTOR_GROUP_ID,
                "placed_actor_count": len(actor_ids),
                "selected_actor_ids": actor_ids,
                "placeholder_substitution": False,
            },
            "supplemental_environment_usage": {
                "group_id": SELECTED_ENVIRONMENT_GROUP_ID,
                "placed_asset_count": len(environment_ids),
                "selected_environment_ids": environment_ids,
                "additive_to_existing_minima": True,
                "placeholder_substitution": False,
            },
            "review_cameras": {
                "path": cameras["path"],
                "sha256": cameras["sha256"],
                "count": cameras["count"],
            },
            "object_lod_payloads": lods,
            "tile_coverage": coverage,
            "monolithic_object_payloads": False,
            "composer_build_receipt": {
                "path": build_receipt.relative_to(variant_dir).as_posix(),
                "sha256": _sha256(build_receipt),
            },
        }


class _InterruptingFakeBackend(_FakeBackend):
    def __init__(self, *, fail_on: str) -> None:
        super().__init__()
        self.fail_on = fail_on

    def author_variant(self, **kwargs: object) -> dict[str, object]:
        metadata = kwargs["metadata"]
        simulation_id = str(metadata["simulation_id"])
        if simulation_id == self.fail_on:
            self.calls.append(simulation_id)
            raise RuntimeError("simulated pod interruption")
        return super().author_variant(**kwargs)


class NativeVariantCampaignTests(unittest.TestCase):
    def _fixtures(self, root: Path) -> tuple[Path, ...]:
        return tuple(
            _write_base_layout(root, f"base-{index}") for index in range(4)
        )

    def test_acquired_actor_and_environment_groups_are_all_deployed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            layout = _write_base_layout(volume, "base-selected-groups")
            base = load_native_base_layout(layout, volume_root=volume)
            variant = next(
                iter(
                    _iter_base_variants(
                        base.scene,
                        master_seed=0xF17E2026,
                        constraints=base.variant_constraints,
                    )
                )
            )

            actors = _actor_deployments(
                variant=variant,
                actors=base.selected_actors,
                simulation_sequence=1,
            )
            environment = _supplemental_environment_deployments(
                variant=variant,
                assets=base.supplemental_environment,
            )

            self.assertEqual(
                {item["selection_id"] for item in actors},
                set(SELECTED_ACTOR_GROUP_IDS),
            )
            self.assertEqual(
                [item["placement_class"] for item in actors].count("ground"),
                2,
            )
            self.assertEqual(
                [item["placement_class"] for item in actors].count("aerial"),
                3,
            )
            self.assertEqual(
                tuple(item["selection_id"] for item in environment),
                SELECTED_ENVIRONMENT_GROUP_IDS,
            )
            self.assertEqual(
                [item["environment_kind"] for item in environment].count(
                    "vegetation"
                ),
                2,
            )
            self.assertEqual(
                [item["environment_kind"] for item in environment].count(
                    "buildings"
                ),
                2,
            )

    def test_plan_is_exact_four_by_five_and_preserves_all_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            layouts = self._fixtures(volume)
            result = prepare_variant_campaign(
                layout_paths=tuple(reversed(layouts)),
                volume_root=volume,
                output_root=volume / "plans" / "campaign",
                master_seed=0xF17E2026,
                constraints=_campaign_constraints(),
            )

            self.assertEqual(result["state"], "VARIANT_PLAN_READY")
            self.assertEqual(result["simulation_count"], 20)
            self.assertEqual(
                [
                    (item["simulation_id"], item["base_scene_id"], item["variant_index"])
                    for item in result["variants"]
                ],
                [
                    (f"SIM-{sequence:02d}", f"base-{base}", variant)
                    for sequence, (base, variant) in enumerate(
                        (
                            (base, variant)
                            for base in range(4)
                            for variant in range(1, 6)
                        ),
                        start=1,
                    )
                ],
            )
            for record in result["variants"]:
                metadata_path = (
                    volume / "plans" / "campaign" / record["metadata"]["path"]
                )
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.assertGreater(metadata["object_count"], 350)
                self.assertEqual(
                    set(metadata["asset_library"]["tree-pine"]["lods"]),
                    {"HERO", "MID", "FAR"},
                )
                self.assertTrue(
                    metadata["terrain_contract"]["source_payloads_reused"]
                )
                self.assertEqual(
                    metadata["water_contract"]["visible_representation"],
                    "tiled_detail_lods",
                )
                self.assertFalse(
                    metadata["water_contract"]["source_payloads_composed"]
                )
                self.assertEqual(len(metadata["tile_coverage"]), 400)
                self.assertTrue(
                    any(
                        "LOD0" in tile["terrain_lods"]
                        for tile in metadata["tile_coverage"]
                    )
                )
                self.assertGreater(
                    metadata["route_fragment_count"],
                    metadata["source_route_count"],
                )
                self.assertEqual(
                    metadata["terrain_contract"][
                        "placement_height_fingerprint"
                    ],
                    metadata["base_bindings"]["placement_height"][
                        "content_fingerprint"
                    ],
                )
                self.assertEqual(
                    metadata["lod_contract"]["object_payload_count_per_scene"],
                    1200,
                )
                self.assertFalse(
                    metadata["lod_contract"]["monolithic_object_payloads"]
                )
                self.assertFalse(
                    (metadata_path.parent / "objects.jsonl").exists()
                )
                rows: list[dict[str, object]] = []
                for tile in metadata["tile_coverage"]:
                    object_path = metadata_path.parent / tile["objects"]["path"]
                    tile_rows = [
                        json.loads(line)
                        for line in object_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ]
                    self.assertEqual(
                        len(tile_rows), tile["objects"]["count"]
                    )
                    rows.extend(tile_rows)
                self.assertEqual(len(rows), metadata["object_count"])
                self.assertEqual(
                    len({row["stable_id"] for row in rows}), len(rows)
                )
                self.assertEqual(
                    len({row["numeric_id"] for row in rows}), len(rows)
                )
            verification = verify_variant_campaign(
                plan_path=volume / "plans" / "campaign" / "campaign-plan.json",
                layout_paths=layouts,
                volume_root=volume,
            )
            self.assertEqual(
                verification["state"], "VARIANT_PLAN_VERIFIED"
            )
            self.assertEqual(
                verification["plan_tile_streams_verified"], 8000
            )

    def test_concave_water_and_sloped_route_meshes_have_real_uvs_and_normals(
        self,
    ) -> None:
        outline = [
            [0.0, 0.0, 10.0],
            [4.0, 0.0, 10.1],
            [4.0, 1.0, 10.2],
            [1.0, 1.0, 10.15],
            [1.0, 4.0, 10.1],
            [0.0, 4.0, 10.0],
        ]
        vertices, counts, indices, uvs, normals = (
            _triangulated_surface(outline)
        )
        self.assertEqual(counts, [3, 3, 3, 3])
        self.assertEqual(len(indices), 12)
        self.assertEqual(len(uvs), len(vertices))
        self.assertEqual(len(normals), len(vertices))
        self.assertTrue(any(abs(normal[0]) > 1.0e-4 for normal in normals))
        self.assertTrue(all(normal[2] > 0.9 for normal in normals))

        route_vertices, route_counts, route_indices, route_uvs = (
            _ribbon_vertices(
                [[0.0, 0.0, 1.0], [5.0, 0.0, 2.0]],
                2.0,
            )
        )
        route_normals = _mesh_vertex_normals(
            route_vertices, route_counts, route_indices
        )
        self.assertEqual(len(route_uvs), len(route_vertices))
        self.assertTrue(
            any(abs(normal[0]) > 1.0e-4 for normal in route_normals)
        )

    def test_layout_rejects_a_campaign_without_real_lod0(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            layout = _write_base_layout(volume, "base-no-lod0")
            payload = json.loads(layout.read_text(encoding="utf-8"))
            for tile in payload["terrain_payloads"]:
                tile["terrain_lods"] = ["LOD1", "LOD2", "LOD3"]
            _write_json(layout, payload)
            with self.assertRaisesRegex(
                NativeVariantContractError,
                "no real LOD0",
            ):
                load_native_base_layout(layout, volume_root=volume)

    def test_layout_rejects_missing_isolated_water_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            layout = _write_base_layout(volume, "base-water")
            payload = json.loads(layout.read_text(encoding="utf-8"))
            payload["water_payloads"] = []
            _write_json(layout, payload)

            with self.assertRaisesRegex(
                NativeVariantContractError,
                "isolated water payload",
            ):
                load_native_base_layout(layout, volume_root=volume)

    def test_layout_rejects_stale_auto_validation_and_changed_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            layout = _write_base_layout(volume, "base-stale")
            payload = json.loads(layout.read_text(encoding="utf-8"))
            auto = volume / payload["scene_auto_validation"]["path"]
            auto.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                NativeVariantContractError,
                "hash mismatch",
            ):
                load_native_base_layout(layout, volume_root=volume)

            layout = _write_base_layout(volume, "base-asset")
            payload = json.loads(layout.read_text(encoding="utf-8"))
            asset_path = (
                volume
                / payload["asset_library"]["tree-pine"]["lods"]["HERO"]["path"]
            )
            asset_path.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(
                NativeVariantContractError,
                "hash mismatch",
            ):
                load_native_base_layout(layout, volume_root=volume)

    def test_authoring_orchestration_is_twenty_blocked_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            layouts = self._fixtures(volume)
            plan_root = volume / "plans" / "campaign"
            prepare_variant_campaign(
                layout_paths=layouts,
                volume_root=volume,
                output_root=plan_root,
                master_seed=0xF17E2026,
                constraints=_campaign_constraints(),
            )
            interrupted = _InterruptingFakeBackend(fail_on="SIM-04")
            with self.assertRaisesRegex(
                RuntimeError, "simulated pod interruption"
            ):
                author_variant_campaign(
                    plan_path=plan_root / "campaign-plan.json",
                    volume_root=volume,
                    output_root=volume / "authored",
                    _backend=interrupted,
                )
            partial = next(
                (volume).glob(
                    ".authored.authoring-*.partial"
                )
            )
            self.assertTrue(
                (partial / "SIM-01" / "variant-checkpoint.json").is_file()
            )
            self.assertFalse((volume / "authored").exists())
            backend = _FakeBackend()
            receipt = author_variant_campaign(
                plan_path=plan_root / "campaign-plan.json",
                volume_root=volume,
                output_root=volume / "authored",
                _backend=backend,
            )

            self.assertEqual(receipt["state"], "VARIANT_USD_AUTHORED")
            self.assertEqual(receipt["simulation_count"], 20)
            self.assertEqual(
                interrupted.calls,
                ["SIM-01", "SIM-02", "SIM-03", "SIM-04"],
            )
            self.assertEqual(
                backend.calls,
                [f"SIM-{index:02d}" for index in range(4, 21)],
            )
            self.assertEqual(
                {
                    item["fire_simulation_status"]
                    for item in receipt["variants"]
                },
                {"blocked_pending_editor_review"},
            )
            self.assertEqual(receipt["manual_editor_review"], "required")
            self.assertEqual(
                {
                    item["streaming_tile_count"]
                    for item in receipt["variants"]
                },
                {400},
            )
            self.assertEqual(
                {
                    item["object_lod_payload_count"]
                    for item in receipt["variants"]
                },
                {1200},
            )
            self.assertEqual(
                {
                    item["monolithic_object_payloads"]
                    for item in receipt["variants"]
                },
                {False},
            )
            self.assertFalse((volume / "authored" / "fire").exists())
            verification = verify_variant_campaign(
                plan_path=plan_root / "campaign-plan.json",
                layout_paths=layouts,
                volume_root=volume,
                authoring_receipt_path=(
                    volume / "authored" / "authoring-receipt.json"
                ),
            )
            self.assertEqual(
                verification["state"], "VARIANT_CAMPAIGN_VERIFIED"
            )
            self.assertEqual(
                verification["root_usd_rehashed"], 20
            )
            self.assertEqual(
                verification["terrain_payload_references_verified"],
                8_000,
            )
            self.assertEqual(
                verification["object_lod_payloads_rehashed"], 24_000
            )
            first_build = json.loads(
                (
                    volume
                    / "authored"
                    / "SIM-01"
                    / "build"
                    / "build-receipt.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                first_build["layers"]["roads"]["prim_count"], 0
            )
            self.assertEqual(
                first_build["layers"]["roads"]["visible_representation"],
                "orthophoto_derived_terrain_material",
            )
            self.assertEqual(
                first_build["water_contract"][
                    "visible_representation"
                ],
                "tiled_detail_lods",
            )
            self.assertFalse(
                first_build["water_contract"]["source_payloads_composed"]
            )
            idempotent_backend = _FakeBackend()
            repeated = author_variant_campaign(
                plan_path=plan_root / "campaign-plan.json",
                volume_root=volume,
                output_root=volume / "authored",
                _backend=idempotent_backend,
            )
            self.assertEqual(repeated, receipt)
            self.assertEqual(idempotent_backend.calls, [])

    def test_resume_does_not_reauthor_one_valid_sim_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            layout = _write_base_layout(volume, "base-bounded-resume")
            plan_root = volume / "plans" / "bounded"
            with (
                mock.patch(
                    "fireviewer_sdg.native_variant_campaign.BASE_SCENE_COUNT",
                    1,
                ),
                mock.patch(
                    "fireviewer_sdg.native_variant_campaign.VARIANTS_PER_BASE",
                    2,
                ),
                mock.patch(
                    "fireviewer_sdg.native_variant_campaign.PORTFOLIO_SCENE_COUNT",
                    2,
                ),
            ):
                prepare_variant_campaign(
                    layout_paths=[layout],
                    volume_root=volume,
                    output_root=plan_root,
                    master_seed=0xB0A1DED,
                    constraints=_campaign_constraints(),
                )
                interrupted = _InterruptingFakeBackend(
                    fail_on="SIM-02"
                )
                with self.assertRaisesRegex(
                    RuntimeError, "simulated pod interruption"
                ):
                    author_variant_campaign(
                        plan_path=plan_root / "campaign-plan.json",
                        volume_root=volume,
                        output_root=volume / "authored-bounded",
                        _backend=interrupted,
                    )
                partial = next(
                    volume.glob(
                        ".authored-bounded.authoring-*.partial"
                    )
                )
                preserved_root = partial / "SIM-01" / "build" / "root.usdc"
                preserved_sha = _sha256(preserved_root)
                resumed = _FakeBackend()
                receipt = author_variant_campaign(
                    plan_path=plan_root / "campaign-plan.json",
                    volume_root=volume,
                    output_root=volume / "authored-bounded",
                    _backend=resumed,
                )

            self.assertEqual(receipt["simulation_count"], 2)
            self.assertEqual(interrupted.calls, ["SIM-01", "SIM-02"])
            self.assertEqual(resumed.calls, ["SIM-02"])
            self.assertEqual(
                _sha256(
                    volume
                    / "authored-bounded"
                    / "SIM-01"
                    / "build"
                    / "root.usdc"
                ),
                preserved_sha,
            )

    def test_default_authoring_fails_closed_without_native_pxr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            layouts = self._fixtures(volume)
            plan_root = volume / "plans" / "campaign"
            prepare_variant_campaign(
                layout_paths=layouts,
                volume_root=volume,
                output_root=plan_root,
                master_seed=0xF17E2026,
                constraints=_campaign_constraints(),
            )
            with mock.patch(
                "fireviewer_sdg.native_variant_campaign._PXRVariantAuthor",
                side_effect=NativeVariantContractError(
                    "native variant authoring requires Kit/Isaac pxr with Semantics"
                ),
            ):
                with self.assertRaisesRegex(
                    NativeVariantContractError,
                    "requires Kit/Isaac pxr",
                ):
                    author_variant_campaign(
                        plan_path=plan_root / "campaign-plan.json",
                        volume_root=volume,
                        output_root=volume / "authored",
                    )
            self.assertFalse((volume / "authored").exists())

    def test_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            layouts = self._fixtures(volume)
            output = volume / "plans" / "campaign"
            output.mkdir(parents=True)
            sentinel = output / "keep.txt"
            sentinel.write_text("user-data", encoding="utf-8")

            with self.assertRaisesRegex(
                NativeVariantContractError,
                "refusing to overwrite",
            ):
                prepare_variant_campaign(
                    layout_paths=layouts,
                    volume_root=volume,
                    output_root=output,
                    master_seed=0xF17E2026,
                    constraints=_campaign_constraints(),
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "user-data")


if __name__ == "__main__":
    unittest.main()

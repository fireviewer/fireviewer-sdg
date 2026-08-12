from __future__ import annotations

import builtins
import contextlib
import dataclasses
import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from fireviewer_sdg.composition_source import (
    ArtifactSource,
    AssetSource,
    CompositionSourceError,
    DetailPayloadExtractionSource,
    GroundMaterialPayloadSource,
    GroundSurfaceSource,
    HeightFieldSource,
    NativeArtifactsSource,
    PlacementHeightTileSource,
    ROOT_LOCAL_COORDINATE_CONTRACT,
    TerrainPayloadSource,
    WaterMaterialSource,
    _derive_native_bridge_spans,
    _placement_height_records,
    _PreparedPlacementHeight,
    _source_backed_routes,
    _validate_no_route_segment_overlap,
    export_composition_source,
    export_from_contract,
    main,
    verify_composition_output,
)
from fireviewer_sdg.native_variant_campaign import load_native_base_layout
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _touch_usd(path: Path, prim_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'#usda 1.0\n\ndef Xform "{prim_name}" {{\n}}\n',
        encoding="utf-8",
    )


def _portable(volume: Path, path: Path) -> str:
    return path.relative_to(volume).as_posix()


def _source(
    path: Path,
    *,
    prim_path: str = "",
) -> ArtifactSource:
    return ArtifactSource(
        path=path,
        prim_path=prim_path,
        expected_sha256=_sha256(path),
    )


def _artifact(volume: Path, path: Path) -> dict[str, str]:
    return {"path": _portable(volume, path), "sha256": _sha256(path)}


@dataclasses.dataclass
class _Fixture:
    volume: Path
    native: NativeArtifactsSource
    ground: GroundSurfaceSource
    placement_heights: tuple[PlacementHeightTileSource, ...]
    assets: tuple[AssetSource, ...]
    water_materials: WaterMaterialSource
    trees: list[dict[str, object]]
    buildings: list[dict[str, object]]
    routes: list[dict[str, object]]
    waters: list[dict[str, object]]
    suitability: list[dict[str, object]]

    def export(
        self,
        output: Path,
        **overrides: Any,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {
            "volume_root": self.volume,
            "output_root": output,
            "base_scene_id": "base-native",
            "coordinate_contract": ROOT_LOCAL_COORDINATE_CONTRACT,
            "epsg2154_origin": [0.0, 0.0, 0.0],
            "native_artifacts": self.native,
            "bounds": {
                "min_x": 0.0,
                "min_y": 0.0,
                "max_x": 100.0,
                "max_y": 100.0,
            },
            "height_field": HeightFieldSource(
                origin_x=0.0,
                origin_y=0.0,
                spacing_m=100.0,
                samples=((0.0, 1.0), (2.0, 3.0)),
            ),
            "placement_height_tiles": self.placement_heights,
            "ground_surface": self.ground,
            "asset_library": self.assets,
            "water_material_lods": self.water_materials,
            "trees": self.trees,
            "buildings": self.buildings,
            "routes": self.routes,
            "waters": self.waters,
            "suitability_zones": self.suitability,
        }
        values.update(overrides)
        return export_composition_source(**values)


def _fixture(volume: Path) -> _Fixture:
    root = volume / "native" / "base-native"
    root_usd = root / "build" / "root.usdc"
    _touch_usd(root_usd, "World")
    terrain_paths: list[Path] = []
    for index in range(400):
        path = root / "build" / "terrain" / f"tile_{index:04d}.usdc"
        _touch_usd(path, "Tile")
        terrain_paths.append(path)
    water_payload = root / "build" / "water" / "water.usdc"
    _touch_usd(water_payload, "Water")
    water_evidence = root / "validation" / "water.json"
    _write_json(
        water_evidence,
        {
            "state": "ISOLATED_WATER_VALIDATED",
            "visible": True,
            "content_roles": ["water"],
            "payloads": [_artifact(volume, water_payload)],
        },
    )
    ground = root / "materials" / "ground.usdc"
    _touch_usd(ground, "GroundMaterials")
    ground_payloads: list[GroundMaterialPayloadSource] = []
    placement_heights: list[PlacementHeightTileSource] = []
    for index in range(400):
        column = index % 20
        row = index // 20
        tile_ref = f"tile-{index:04d}"
        bounds = {
            "min_x": column * 5.0,
            "min_y": row * 5.0,
            "max_x": (column + 1) * 5.0,
            "max_y": (row + 1) * 5.0,
        }
        ground_payload = (
            root / "materials" / "ground-tiles" / f"{tile_ref}.usdc"
        )
        _touch_usd(ground_payload, "Ground")
        ground_payloads.append(
            GroundMaterialPayloadSource(
                artifact=_source(ground_payload, prim_path="/Ground"),
                tile_id=tile_ref,
                tile_ref=tile_ref,
                tile_bounds_m=bounds,
            )
        )
        x_coordinates = (bounds["min_x"], bounds["max_x"])
        y_coordinates = (bounds["min_y"], bounds["max_y"])
        samples = tuple(
            0.01 * x + 0.02 * y
            for y in y_coordinates
            for x in x_coordinates
        )
        placement = (
            root / "placement-height" / f"{tile_ref}.f32"
        )
        placement.parent.mkdir(parents=True, exist_ok=True)
        placement.write_bytes(struct.pack("<4f", *samples))
        placement_heights.append(
            PlacementHeightTileSource(
                artifact=_source(placement),
                tile_ref=tile_ref,
                local_bounds=bounds,
                width=2,
                height=2,
                x_coordinates=x_coordinates,
                y_coordinates=y_coordinates,
            )
        )
    ground_evidence = root / "validation" / "ground.json"
    _write_json(
        ground_evidence,
        {
            "state": "OBJECT_FREE_PBR_VALIDATED",
            "material": _artifact(volume, ground),
            "contains_object_imagery": False,
        },
    )
    simready_evidence = root / "validation" / "simready.json"
    _write_json(simready_evidence, {"state": "SIMREADY_VALIDATED"})
    water_pbr_evidence = root / "validation" / "water-pbr.json"
    _write_json(water_pbr_evidence, {"state": "PBR_VALIDATED"})

    assets: list[AssetSource] = []
    for key, category, family in (
        ("tree-pine", "trees", "pine"),
        ("building-stone", "buildings", "stone_house"),
    ):
        lods: dict[str, ArtifactSource] = {}
        for level in ("HERO", "MID", "FAR"):
            path = root / "assets" / f"{key}_{level.lower()}.usdc"
            _touch_usd(path, "Asset")
            lods[level] = _source(path, prim_path="/Asset")
        assets.append(
            AssetSource(
                key=key,
                category=category,
                family=family,
                lods=lods,
                lod_lineage=f"lineage-{key}",
                grounding_offsets_m={
                    "HERO": 0.0,
                    "MID": 0.0,
                    "FAR": 0.0,
                },
                simready_validation_state="SIMREADY_VALIDATED",
                simready_validation_evidence=_source(simready_evidence),
            )
        )
    water_lods: dict[str, ArtifactSource] = {}
    for level in ("HERO", "MID", "FAR"):
        path = root / "materials" / f"water_{level.lower()}.usdc"
        _touch_usd(path, "Material")
        water_lods[level] = _source(path, prim_path="/Material")
    water_materials = WaterMaterialSource(
        lods=water_lods,
        pbr_validation_state="PBR_VALIDATED",
        pbr_validation_evidence=_source(water_pbr_evidence),
    )

    fake_details = [
        {
            "path": f"native/base-native/build/details/{index:04d}.usdc",
            "sha256": "d" * 64,
        }
        for index in range(400)
    ]
    selected_actor_assets: dict[str, dict[str, object]] = {}
    for selection_id in SELECTED_ACTOR_GROUP_IDS:
        lod_paths: dict[str, dict[str, str]] = {}
        for level in ("HERO", "MID", "FAR"):
            path = (
                root
                / "assets"
                / "selected-actors"
                / selection_id
                / f"{level.lower()}.usdc"
            )
            _touch_usd(path, "Actor")
            lod_paths[level] = {
                **_artifact(volume, path),
                "prim_path": "/Actor",
            }
        selected_actor_assets[selection_id] = {
            "selection_id": selection_id,
            "asset_id": f"selected-actor.{selection_id}:test",
            "family": f"selected-actors.{selection_id}",
            "selection_source_url": SELECTED_ACTOR_GROUP_SOURCE_BY_ID[
                selection_id
            ],
            "placement_class": SELECTED_ACTOR_GROUP_PLACEMENT_BY_ID[
                selection_id
            ],
            "lod_paths": lod_paths,
            "ground_anchor_m": [0.0, 0.0, 0.0],
        }

    selected_environment_assets: dict[str, dict[str, object]] = {}
    for selection_id in SELECTED_ENVIRONMENT_GROUP_IDS:
        environment_kind, environment_family = (
            SELECTED_ENVIRONMENT_TARGET_BY_ID[selection_id]
        )
        lod_paths: dict[str, dict[str, str]] = {}
        for level in ("HERO", "MID", "FAR"):
            path = (
                root
                / "assets"
                / "selected-environment"
                / selection_id
                / f"{level.lower()}.usdc"
            )
            _touch_usd(path, "EnvironmentAsset")
            lod_paths[level] = {
                **_artifact(volume, path),
                "prim_path": "/EnvironmentAsset",
            }
        selected_environment_assets[selection_id] = {
            "selection_id": selection_id,
            "asset_id": f"selected-environment.{selection_id}:test",
            "environment_kind": environment_kind,
            "environment_family": environment_family,
            "lod_paths": lod_paths,
            "ground_anchor_m": [0.0, 0.0, 0.0],
        }

    shared_manifest = root / "assets" / "shared-asset-manifest.json"
    _write_json(
        shared_manifest,
        {
            "schema_version": 1,
            "state": "SIMREADY_ASSET_LIBRARY_READY",
            "selected_actor_group": {
                "group_id": SELECTED_ACTOR_GROUP_ID,
                "selection_count": len(SELECTED_ACTOR_GROUP_IDS),
                "selection_order": list(SELECTED_ACTOR_GROUP_IDS),
                "assets": selected_actor_assets,
                "usage_contract": (
                    "all_selected_assets_must_be_placed_across_the_20_scene_campaign"
                ),
            },
            "actors": {
                role: {"asset_id": f"semantic-role.{role}:test"}
                for role in sorted(REQUIRED_ACTOR_CLASSES)
            },
            "selected_environment_group": {
                "group_id": SELECTED_ENVIRONMENT_GROUP_ID,
                "selection_count": len(SELECTED_ENVIRONMENT_GROUP_IDS),
                "selection_order": list(SELECTED_ENVIRONMENT_GROUP_IDS),
                "assets": selected_environment_assets,
                "usage_contract": (
                    "all_four_assets_are_additive_and_used_in_every_variant"
                ),
            },
        },
    )
    asset_lock = root / "build" / "asset-lock.json"
    locked_assets = [
        {
            "id": "shared-materialized-simready-environment",
            "manifest": _portable(volume, shared_manifest),
            "manifest_sha256": _sha256(shared_manifest),
            "validation": {"asset_content_sha256": "a" * 64},
        }
    ]
    _write_json(asset_lock, {"assets": locked_assets})
    review_cameras = root / "build" / "review-cameras.usda"
    _touch_usd(review_cameras, "ReviewCameras")
    build_receipt = root / "build" / "build-receipt.json"
    _write_json(
        build_receipt,
        {
            "schema_version": 2,
            "zone_id": "base-native",
            "source_profile": "full",
            "fire_simulation_status": "blocked_pending_editor_review",
            "root_usd": {
                "path": root_usd.relative_to(root).as_posix(),
                "sha256": _sha256(root_usd),
            },
            "asset_lock": {
                "path": asset_lock.relative_to(root).as_posix(),
                "sha256": _sha256(asset_lock),
                "assets": locked_assets,
            },
            "cameras": {
                "path": review_cameras.relative_to(root).as_posix(),
                "sha256": _sha256(review_cameras),
                "count": 6,
            },
            "payloads": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256(path),
                }
                for path in terrain_paths
            ],
            "detail_payloads": fake_details,
            "detail_mid_payloads": fake_details,
            "detail_far_payloads": fake_details,
            "layers": {
                "vegetation": {"prim_count": 1},
                "buildings": {"prim_count": 1},
                "roads": {"prim_count": 1},
                "hydrology": {"prim_count": 1},
            },
            "tile_coverage": [
                {
                    "tile_ref": f"tile-{index:04d}",
                    "terrain_payload": path.relative_to(root).as_posix(),
                    "instance_namespace": index + 1,
                    "terrain_lods": (
                        ["LOD0", "LOD1", "LOD2", "LOD3"]
                        if index == 0
                        else ["LOD1", "LOD2", "LOD3"]
                    ),
                    "collision_lods": ["NEAR", "FAR"],
                }
                for index, path in enumerate(terrain_paths)
            ],
        },
    )
    auto_validation = root / "validation" / "auto.json"
    _write_json(
        auto_validation,
        {
            "state": "AUTO_VALIDATED",
            "fire_simulation_status": "blocked_pending_editor_review",
            "build_receipt_sha256": _sha256(build_receipt),
            "root_usd_sha256": _sha256(root_usd),
        },
    )
    native = NativeArtifactsSource(
        native_build_receipt=_source(build_receipt),
        scene_auto_validation=_source(auto_validation),
        root_usd=_source(root_usd, prim_path="/World"),
        terrain_payloads=tuple(
            TerrainPayloadSource(
                artifact=_source(path, prim_path="/Tile"),
                tile_ref=f"tile-{index:04d}",
                local_bounds={
                    "min_x": (index % 20) * 5.0,
                    "min_y": (index // 20) * 5.0,
                    "max_x": (index % 20 + 1) * 5.0,
                    "max_y": (index // 20 + 1) * 5.0,
                },
                epsg2154_bounds={
                    "min_x": (index % 20) * 5.0,
                    "min_y": (index // 20) * 5.0,
                    "max_x": (index % 20 + 1) * 5.0,
                    "max_y": (index // 20 + 1) * 5.0,
                },
                instance_namespace=index + 1,
                terrain_lods=(
                    ("LOD0", "LOD1", "LOD2", "LOD3")
                    if index == 0
                    else ("LOD1", "LOD2", "LOD3")
                ),
                collision_lods=("NEAR", "FAR"),
            )
            for index, path in enumerate(terrain_paths)
        ),
        water_payloads=(_source(water_payload, prim_path="/Water"),),
        water_validation_state="ISOLATED_WATER_VALIDATED",
        water_validation_evidence=_source(water_evidence),
    )
    trees = [
        {
            "stable_id": "tree-0001",
            "numeric_id": 1,
            "asset_key": "tree-pine",
            "position": [20.0, 20.0, 0.5],
            "heading_degrees": 12.5,
            "uniform_scale": 1.1,
            "footprint_radius_m": 2.0,
            "group_id": "forest-a",
        }
    ]
    buildings = [
        {
            "stable_id": "building-0001",
            "numeric_id": 2,
            "asset_key": "building-stone",
            "position": [80.0, 80.0, 2.5],
            "heading_degrees": 45.0,
            "uniform_scale": 0.9,
            "footprint_radius_m": 5.0,
            "group_id": "village-a",
        }
    ]
    routes = [
        {
            "stable_id": "route-0001",
            "numeric_id": 3,
            "family": "local",
            "surface_class": "bridge",
            "width_m": 4.0,
            "points": [[10.0, 50.0, 1.0], [90.0, 50.0, 1.0]],
            "bridge_spans": [
                {
                    "stable_id": "bridge-0001",
                    "start_fraction": 0.35,
                    "water_start_fraction": 0.45,
                    "water_end_fraction": 0.55,
                    "end_fraction": 0.65,
                    "minimum_deck_clearance_m": 2.0,
                }
            ],
        }
    ]
    waters = [
        {
            "stable_id": "water-0001",
            "family": "pond",
            "outline": [
                [40.0, 40.0],
                [60.0, 40.0],
                [60.0, 60.0],
                [40.0, 60.0],
            ],
            "kind": "standing",
            "centreline": [],
            "surface_profile_m": [1.0],
        }
    ]
    suitability = [
        {
            "stable_id": "suitability-0001",
            "outline": [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
            ],
            "biome": "temperate_forest",
            "soil": "loam",
            "tree_families": ["pine"],
            "buildable": True,
        }
    ]
    return _Fixture(
        volume=volume,
        native=native,
        ground=GroundSurfaceSource(
            material=_source(ground, prim_path="/GroundMaterials"),
            tile_material_payloads=tuple(ground_payloads),
            validation_evidence=_source(ground_evidence),
            validation_state="OBJECT_FREE_PBR_VALIDATED",
        ),
        placement_heights=tuple(placement_heights),
        assets=tuple(assets),
        water_materials=water_materials,
        trees=trees,
        buildings=buildings,
        routes=routes,
        waters=waters,
        suitability=suitability,
    )


def _bind_existing_export_to_prepared_contract(
    *,
    fixture: _Fixture,
    output: Path,
    prepared: Path,
) -> tuple[Path, str]:
    manifest_path = output / "composition-source.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    build_path = Path(fixture.native.native_build_receipt.path)
    route_source = {
        "route_geometry_authority": "locked_continuous_bdtopo_lines",
        "locked_source_artifacts": [
            _artifact(fixture.volume, build_path)
        ],
        "source_feature_count": 1,
        "source_line_count": 1,
        "prepared_route_count": len(manifest["routes"]),
        "placement_height_cache_tile_limit": 2,
        "placement_height_peak_cached_tiles": 2,
        "overlap_validation": {
            "algorithm": "spatial-collinear-interior-overlap-gate-v1",
            "segment_count": 1,
            "candidate_comparison_count": 0,
            "maximum_interroute_collinear_overlap_m": 0.0,
            "maximum_allowed_interroute_collinear_overlap_m": 0.01,
        },
        "native_hero_fragment_proof_count": 1,
        "native_hero_receipt_fragment_count": 1,
    }
    contract = {
        "schema_version": 1,
        "state": "NATIVE_COMPOSITION_EXPORT_INPUT_READY",
        "base_scene_id": manifest["base_scene_id"],
        "coordinate_contract": manifest["coordinate_contract"],
        "epsg2154_origin": manifest["epsg2154_origin"],
        "object_source": "native_hero_detail_payloads",
        "native_artifacts": {
            "native_build_receipt": manifest["native_build_receipt"],
            "scene_auto_validation": manifest[
                "scene_auto_validation"
            ],
            "root_usd": manifest["root_usd"],
            "terrain_payloads": manifest["terrain_payloads"],
            "water_payloads": manifest["water_payloads"],
            "water_validation": {
                "state": "ISOLATED_WATER_VALIDATED",
                "evidence": manifest["validation_evidence"]["water"],
            },
        },
        "ground_surface": {
            "material": {
                key: value
                for key, value in manifest["ground_material"].items()
                if key
                in {
                    "path",
                    "sha256",
                    "prim_path",
                    "isolated_content_roles",
                }
            },
            "topology": manifest["ground_material"]["topology"],
            "tile_material_payloads": manifest["ground_material"][
                "tile_material_payloads"
            ],
            "validation_evidence": manifest["validation_evidence"][
                "ground_surface"
            ],
            "validation_state": "OBJECT_FREE_PBR_VALIDATED",
            "kind": "object_free_pbr",
            "removed_object_classes": [],
        },
        "height_field_source": manifest["height_field"],
        "placement_height_tiles": manifest["placement_height_tiles"],
        "placement_height_fingerprint": manifest[
            "placement_height_fingerprint"
        ],
        "routes": manifest["routes"],
        "route_topology": manifest["route_topology"],
        "route_source": route_source,
        "waters": manifest["waters"],
        "suitability_zones": manifest["suitability_zones"],
        "variant_constraints": manifest["variant_constraints"],
        "fire_simulation_status": "blocked_pending_editor_review",
    }
    contract_path = prepared / "composition-export-input.json"
    _write_json(contract_path, contract)
    contract_sha = _sha256(contract_path)
    manifest["source_contract"] = {
        "path": _portable(fixture.volume, contract_path),
        "sha256": contract_sha,
    }
    manifest["route_source"] = route_source
    _write_json(manifest_path, manifest)
    return contract_path, contract_sha


class CompositionSourceTests(unittest.TestCase):
    def test_continuous_bdtopo_route_crosses_tile_boundary_without_overlap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            zone = volume / "zone"
            roads = zone / "raw" / "roads" / "roads.geojson"
            _write_json(
                roads,
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "id": "road.boundary-crossing",
                            "properties": {"largeur_de_chaussee": 4.0},
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [
                                    [0.0, 5.0],
                                    [5.0, 5.0],
                                    [10.0, 5.0],
                                ],
                            },
                        }
                    ],
                },
            )
            source_lock = {
                "vector_sources": {
                    "roads": [
                        {
                            "license": "Licence Ouverte / Etalab 2.0",
                            "download": {
                                "state": "downloaded",
                                "relpath": "roads/roads.geojson",
                                "sha256": _sha256(roads),
                            },
                        }
                    ]
                }
            }
            placements: list[_PreparedPlacementHeight] = []
            for index, (min_x, max_x) in enumerate(
                ((0.0, 5.0), (5.0, 10.0))
            ):
                path = zone / "placement" / f"{index}.f32"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(struct.pack("<4f", 0.0, 0.0, 0.0, 0.0))
                placements.append(
                    _PreparedPlacementHeight(
                        tile_ref=f"tile-{index}",
                        local_bounds={
                            "min_x": min_x,
                            "min_y": 0.0,
                            "max_x": max_x,
                            "max_y": 10.0,
                        },
                        x_coordinates=(min_x, max_x),
                        y_coordinates=(0.0, 10.0),
                        physical_path=path,
                        final_path=path,
                        sha256=_sha256(path),
                        south_edge=(0.0, 0.0),
                        north_edge=(0.0, 0.0),
                        west_edge=(0.0, 0.0),
                        east_edge=(0.0, 0.0),
                    )
                )

            routes, source_metrics = _source_backed_routes(
                volume_root=volume,
                zone_root=zone,
                source_lock=source_lock,
                epsg2154_bounds=(0.0, 0.0, 10.0, 10.0),
                root_origin=(0.0, 0.0),
                placement_tiles=placements,
            )
            water = {
                "stable_id": "water-boundary",
                "family": "surface",
                "outline": [
                    [4.0, 4.0],
                    [6.0, 4.0],
                    [6.0, 6.0],
                    [4.0, 6.0],
                ],
                "kind": "standing",
                "centreline": [],
                "surface_profile_m": [1.0],
            }
            bridge_metrics = _derive_native_bridge_spans(
                routes=routes,
                waters=[water],
            )

            self.assertEqual(len(routes), 1)
            self.assertTrue(
                any(point[0] == 5.0 for point in routes[0]["points"])
            )
            self.assertEqual(bridge_metrics["bridge_span_count"], 1)
            self.assertEqual(
                bridge_metrics["source_interpolated_approaches"], 2
            )
            self.assertEqual(
                routes[0]["bridge_spans"][0][
                    "minimum_deck_clearance_m"
                ],
                3.0,
            )
            self.assertEqual(
                source_metrics["route_geometry_authority"],
                "locked_continuous_bdtopo_lines",
            )
            self.assertEqual(
                _validate_no_route_segment_overlap(routes)[
                    "maximum_interroute_collinear_overlap_m"
                ],
                0.0,
            )

    def test_parallel_route_near_bank_is_not_reclassified_as_bridge(
        self,
    ) -> None:
        routes = [
            {
                "stable_id": "parallel-road",
                "family": "bdtopo_road",
                "material_key": "asphalt",
                "width_m": 4.0,
                "points": [[0.0, 1.25, 0.0], [10.0, 1.25, 0.0]],
                "bridge_spans": [],
            }
        ]
        water = {
            "stable_id": "water",
            "outline": [
                [4.0, -1.0],
                [6.0, -1.0],
                [6.0, 1.0],
                [4.0, 1.0],
            ],
            "kind": "standing",
            "centreline": [],
            "surface_profile_m": [1.0],
        }

        metrics = _derive_native_bridge_spans(
            routes=routes, waters=[water]
        )

        self.assertEqual(metrics["bridge_span_count"], 0)
        self.assertEqual(routes[0]["bridge_spans"], [])

    def test_route_overlap_gate_rejects_duplicated_fragment_interior(
        self,
    ) -> None:
        routes = [
            {"points": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]},
            {"points": [[5.0, 0.0, 0.0], [15.0, 0.0, 0.0]]},
        ]
        with self.assertRaisesRegex(
            CompositionSourceError, "duplicated collinear interiors"
        ):
            _validate_no_route_segment_overlap(routes)

    def test_route_overlap_gate_rejects_self_retracing_source_line(
        self,
    ) -> None:
        routes = [
            {
                "points": [
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ]
            }
        ]
        with self.assertRaisesRegex(
            CompositionSourceError, "duplicated collinear interiors"
        ):
            _validate_no_route_segment_overlap(routes)

    def test_near_height_seam_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            fixture = _fixture(volume)
            placements: list[_PreparedPlacementHeight] = []
            for source in fixture.placement_heights:
                path = Path(source.artifact.path)
                values = struct.unpack("<4f", path.read_bytes())
                placements.append(
                    _PreparedPlacementHeight(
                        tile_ref=source.tile_ref,
                        local_bounds=dict(source.local_bounds),
                        x_coordinates=tuple(source.x_coordinates),
                        y_coordinates=tuple(source.y_coordinates),
                        physical_path=path,
                        final_path=path,
                        sha256=_sha256(path),
                        south_edge=(values[0], values[1]),
                        north_edge=(values[2], values[3]),
                        west_edge=(values[0], values[2]),
                        east_edge=(values[1], values[3]),
                    )
                )
            placements[0] = dataclasses.replace(
                placements[0],
                east_edge=tuple(
                    value + 0.1 for value in placements[0].east_edge
                ),
            )

            with self.assertRaisesRegex(
                CompositionSourceError,
                "adjacent placement height seam",
            ):
                _placement_height_records(
                    placements=placements,
                    volume_root=volume,
                )

    def test_existing_export_resumes_only_after_full_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            fixture = _fixture(volume)
            output = volume / "composition" / "resume"
            fixture.export(output)
            contract_path, contract_sha = (
                _bind_existing_export_to_prepared_contract(
                    fixture=fixture,
                    output=output,
                    prepared=volume / "prepared",
                )
            )
            manifest_path = output / "composition-source.json"
            before_sha = _sha256(manifest_path)

            verified = verify_composition_output(
                volume_root=volume,
                contract_path=contract_path,
                contract_sha256=contract_sha,
                output_root=output,
            )
            resumed = export_from_contract(
                volume_root=volume,
                contract_path=contract_path,
                contract_sha256=contract_sha,
                output_root=output,
            )

            self.assertEqual(
                verified["state"], "COMPOSITION_SOURCE_VERIFIED"
            )
            self.assertEqual(resumed["state"], "COMPOSITION_SOURCE_READY")
            self.assertEqual(_sha256(manifest_path), before_sha)

    def test_existing_export_corruption_fails_closed_without_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            fixture = _fixture(volume)
            output = volume / "composition" / "corrupt"
            fixture.export(output)
            contract_path, contract_sha = (
                _bind_existing_export_to_prepared_contract(
                    fixture=fixture,
                    output=output,
                    prepared=volume / "prepared",
                )
            )
            trees = output / "trees.jsonl"
            trees.write_bytes(trees.read_bytes() + b"{}\n")
            corrupted_sha = _sha256(trees)

            with self.assertRaisesRegex(
                CompositionSourceError, "absent, unsafe or stale"
            ):
                export_from_contract(
                    volume_root=volume,
                    contract_path=contract_path,
                    contract_sha256=contract_sha,
                    output_root=output,
                )

            self.assertTrue(output.is_dir())
            self.assertEqual(_sha256(trees), corrupted_sha)

    def test_verify_cli_checks_prepared_and_existing_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            fixture = _fixture(volume)
            output = volume / "composition" / "verify-cli"
            fixture.export(output)
            contract_path, contract_sha = (
                _bind_existing_export_to_prepared_contract(
                    fixture=fixture,
                    output=output,
                    prepared=volume / "prepared",
                )
            )
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exit_code = main(
                    [
                        "verify",
                        "--volume-root",
                        str(volume),
                        "--contract",
                        str(contract_path),
                        "--contract-sha256",
                        contract_sha,
                        "--output",
                        str(output),
                    ]
                )
            payload = json.loads(stream.getvalue())

            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["state"], "COMPOSITION_SOURCE_VERIFIED")

    def test_atomic_export_is_directly_loadable_by_variant_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            fixture = _fixture(volume)
            output = volume / "composition" / "base-native"

            result = fixture.export(output)
            loaded = load_native_base_layout(
                output / "composition-source.json",
                volume_root=volume,
            )

            self.assertEqual(result["state"], "COMPOSITION_SOURCE_READY")
            self.assertEqual(result["trees"]["count"], 1)
            self.assertEqual(result["buildings"]["count"], 1)
            self.assertEqual(result["coordinate_contract"], ROOT_LOCAL_COORDINATE_CONTRACT)
            self.assertEqual(loaded.scene.stable_id, "base-native")
            self.assertEqual(len(loaded.terrain_payloads), 400)
            self.assertEqual(len(loaded.scene.trees), 1)
            self.assertEqual(len(loaded.scene.buildings), 1)
            self.assertEqual(len(loaded.scene.routes), 1)
            self.assertEqual(len(loaded.scene.waters), 1)
            self.assertEqual(
                result["road_visual_contract"]["visible_representation"],
                "orthophoto_derived_terrain_material",
            )
            self.assertEqual(
                result["road_visual_contract"]["geometry_authoring"],
                "disabled",
            )
            self.assertEqual(
                result["road_visual_contract"]["asset_dependencies"], [],
            )
            self.assertNotIn("road_material_library", result)
            self.assertEqual(
                result["identity_contract"]["object_source"],
                "native_build_in_memory_records",
            )
            self.assertFalse(
                result["streaming_memory_contract"][
                    "full_object_inventory_retained_in_ram"
                ]
            )
            self.assertFalse(
                list(output.parent.glob(".base-native.*.staging"))
            )

    def test_duplicate_native_identity_fails_without_partial_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            fixture = _fixture(volume)
            duplicate = [dict(fixture.buildings[0], numeric_id=1)]
            output = volume / "composition" / "duplicate"

            with self.assertRaisesRegex(
                CompositionSourceError, "numeric ID 1 repeats"
            ):
                fixture.export(output, buildings=duplicate)

            self.assertFalse(output.exists())
            self.assertFalse(list(output.parent.glob(".duplicate.*.staging")))

    def test_unvalidated_ground_or_water_is_never_labeled_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            fixture = _fixture(volume)
            output = volume / "composition" / "unvalidated"
            ground = dataclasses.replace(
                fixture.ground,
                validation_state="NOT_VALIDATED",
            )

            with self.assertRaisesRegex(
                CompositionSourceError, "OBJECT_FREE_PBR_VALIDATED"
            ):
                fixture.export(output, ground_surface=ground)
            self.assertFalse(output.exists())

            native = dataclasses.replace(
                fixture.native,
                water_validation_state="NOT_VALIDATED",
            )
            with self.assertRaisesRegex(
                CompositionSourceError, "ISOLATED_WATER_VALIDATED"
            ):
                fixture.export(output, native_artifacts=native)
            self.assertFalse(output.exists())

    def test_height_field_bound_is_enforced_without_resampling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            fixture = _fixture(volume)
            output = volume / "composition" / "too-tall"
            height = HeightFieldSource(
                origin_x=0.0,
                origin_y=0.0,
                spacing_m=1.0,
                samples=([0.0, 0.0] for _ in range(1026)),
            )

            with self.assertRaisesRegex(
                CompositionSourceError, "1025-row portable bound"
            ):
                fixture.export(output, height_field=height)

            self.assertFalse(output.exists())

    def test_lazy_pxr_backend_is_optional_and_fails_closed_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            fixture = _fixture(volume)
            output = volume / "composition" / "lazy"
            extraction = DetailPayloadExtractionSource(
                coordinate_space="root_local_xy_ign69_z",
                root_origin_epsg2154=(700_000.0, 6_500_000.0),
            )
            real_import = builtins.__import__

            def deny_pxr(name: str, *args: object, **kwargs: object) -> object:
                if name == "pxr":
                    raise ImportError("pxr deliberately absent in unit test")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=deny_pxr):
                with self.assertRaisesRegex(
                    CompositionSourceError, "requires native Kit/Isaac pxr"
                ):
                    fixture.export(
                        output,
                        trees=None,
                        buildings=None,
                        native_detail_extraction=extraction,
                    )

            self.assertFalse(output.exists())

    def test_cli_is_a_real_module_entrypoint(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()

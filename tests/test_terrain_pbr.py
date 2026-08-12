from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fireviewer_sdg import terrain_pbr  # noqa: E402
from fireviewer_sdg.asset_bundle import (  # noqa: E402
    INSTALL_MARKER,
    PBR_MATERIAL_ROLES,
    PBR_REQUIRED_TEXTURES,
)
from fireviewer_sdg.terrain_pbr import (  # noqa: E402
    Bounds2d,
    FileLock,
    SpatialEvidence,
    TerrainPbrContractError,
    TerrainSubzoneEvidence,
    TerrainTileEvidence,
    build_composite_ground_material_spec,
    build_terrain_pbr_plan,
    load_locked_material_library,
    validate_composite_ground_material_artifact,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _record(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_tile_material_payloads(
    *,
    artifact_root: Path,
    specification: terrain_pbr.CompositeGroundMaterialSpec,
    payload_root: Path,
) -> list[dict[str, object]]:
    payload_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, tile_id in enumerate(specification.tile_ids):
        layer = payload_root / f"{index:04d}-{tile_id}.usdc"
        layer.write_bytes(f"materialx-payload:{tile_id}".encode())
        records.append(
            {
                "tile_id": tile_id,
                "tile_ref": tile_id,
                "tile_bounds_m": list(
                    specification.bindings_for_tile(tile_id)[0][
                        "tile_bounds_m"
                    ]
                ),
                **_record(layer, root=artifact_root),
                "prim_path": "/Ground",
            }
        )
    return records


def _native_validation(
    *,
    plan: terrain_pbr.TerrainPbrPlan,
    specification: terrain_pbr.CompositeGroundMaterialSpec,
    derived_inputs: list[dict[str, object]],
    tile_payloads: list[dict[str, object]],
    uniform_fallback: bool = False,
) -> dict[str, object]:
    uv_continuity = terrain_pbr._measure_metric_uv_continuity(plan)
    edge_measurements = [
        {
            "tile_a": seam["tile_a"],
            "tile_b": seam["tile_b"],
            "semantic": semantic,
            "axis": seam["axis"],
            "sample_count": terrain_pbr.MASK_EDGE_SAMPLE_COUNT,
            "maximum_absolute_mask_error": 0.0,
        }
        for seam in uv_continuity["seams"]
        for semantic in terrain_pbr.CLASSIFICATION_SEMANTICS
    ]
    mask_continuity: dict[str, object] = {
        "algorithm": "gdal_halo_bilinear_shared_edge_v1",
        "adjacent_pair_count": uv_continuity["adjacent_pair_count"],
        "semantic_count_per_edge": len(
            terrain_pbr.CLASSIFICATION_SEMANTICS
        ),
        "samples_per_semantic_edge": (
            terrain_pbr.MASK_EDGE_SAMPLE_COUNT
        ),
        "tolerance": terrain_pbr.MASK_EDGE_CONTINUITY_TOLERANCE,
        "maximum_absolute_mask_error": 0.0,
        "measurements": edge_measurements,
    }
    mask_continuity["measurement_sha256"] = _canonical_sha256(
        mask_continuity
    )
    tile_validations: list[dict[str, object]] = []
    for tile_id in specification.tile_ids:
        bindings = specification.bindings_for_tile(tile_id)
        binding_ids = [
            str(binding["binding_id"]) for binding in bindings
        ]
        tile_validations.append(
            {
                "tile_id": tile_id,
                "tile_ref": tile_id,
                "tile_bounds_m": list(bindings[0]["tile_bounds_m"]),
                "material_prim_path": "/Ground",
                "connected_material_roles": list(PBR_MATERIAL_ROLES),
                "reachable_quality_features": [
                    "slope_projection",
                    "world_macro_variation",
                ],
                "texture_color_space_contract": {
                    "base_color": "srgb_texture",
                    "normal": "none",
                    "roughness": "none",
                    "assignment_count": 30,
                    "verified_after_reopen": True,
                },
                "connected_spatial_binding_ids": binding_ids,
                "connected_mask_binding_ids": [
                    binding_id
                    for binding_id in binding_ids
                    if not binding_id.endswith(":elevation")
                ],
                "connected_relief_binding_ids": [
                    binding_id
                    for binding_id in binding_ids
                    if binding_id.endswith(":elevation")
                ],
                "surface_output_connected": True,
                "all_required_branches_surface_reachable": True,
                "material_metric_uv_uses_world_position": True,
                "spatial_mask_uv_uses_halo_sampling_bounds": True,
                "spatial_mask_address_mode": "clamp",
                "spatial_image_node_count": (
                    terrain_pbr.MAX_SPATIAL_IMAGE_NODES_PER_TILE
                ),
                "reachable_shader_prim_count": 64,
                "uniform_fallback_present": uniform_fallback,
            }
        )
    binding_ids = [
        str(binding["binding_id"])
        for binding in specification.spatial_bindings
    ]
    return {
        "inspector_id": terrain_pbr.NATIVE_COMPOSITE_INSPECTOR_ID,
        "render_context": "mtlx",
        "ground_index_prim_path": specification.material_prim_path,
        "ground_index_prim_type": "UsdGeom.Scope",
        "topology": "payload_tiled_materials_shared_pbr_library",
        "shared_pbr_library_count": 1,
        "tile_payload_count": len(specification.tile_ids),
        "material_graph_count": len(specification.tile_ids),
        "root_shader_prim_count_with_payloads_unloaded": 0,
        "connected_material_roles": list(PBR_MATERIAL_ROLES),
        "connected_spatial_binding_ids": binding_ids,
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
        "metric_uv_continuity_measurement": uv_continuity,
        "spatial_mask_edge_continuity_measurement": mask_continuity,
        "blend_graph_sha256": specification.blend_graph_sha256,
        "material_bindings_sha256": (
            specification.material_bindings_sha256
        ),
        "evidence_bindings_sha256": (
            specification.evidence_bindings_sha256
        ),
        "derived_spatial_inputs_sha256": _canonical_sha256(
            derived_inputs
        ),
        "derived_spatial_input_count": len(derived_inputs),
        "tile_payload_layers_sha256": _canonical_sha256(tile_payloads),
        "native_stage_reopen_succeeded": True,
        "all_tile_surface_outputs_connected": True,
        "all_required_branches_surface_reachable": True,
        "uniform_fallback_present": uniform_fallback,
        "single_graph_for_all_tiles_present": False,
        "monolithic_generated_mask_atlas_present": False,
        "source_colour_feeds_base_color": False,
        "source_geometry_creates_rendered_objects": False,
        "maximum_spatial_image_nodes_per_tile": (
            terrain_pbr.MAX_SPATIAL_IMAGE_NODES_PER_TILE
        ),
        "maximum_reachable_shader_prims_per_tile": 64,
        "tile_validations": tile_validations,
    }


def _coverage_metrics(semantic: str) -> dict[str, object]:
    def metric(minimum: float, maximum: float) -> dict[str, object]:
        return {
            "sample_count": 64,
            "finite_count": 64,
            "nodata_count": 0,
            "finite_fraction": 1.0,
            "minimum": minimum,
            "maximum": maximum,
        }

    if semantic == "elevation":
        return {
            "elevation_source": metric(120.0, 280.0),
            "slope_source": metric(0.0, 42.0),
            "roughness_source": metric(0.0, 1.25),
            "relief_slope": metric(0.0, 1.0),
            "relief_roughness": metric(0.0, 1.0),
        }
    return {
        "classified_source": metric(0.0, 1.0),
        "feathered_mask": metric(0.0, 1.0),
    }


def _write_locked_bundle(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    payload: dict[str, object] = {"pbr_materials": {}}
    structural: dict[str, object] = {}
    materials = payload["pbr_materials"]
    assert isinstance(materials, dict)
    for role_index, role in enumerate(PBR_MATERIAL_ROLES):
        material = root / "materials" / role / f"{role}.usda"
        material.parent.mkdir(parents=True)
        material.write_text(
            f'#usda 1.0\ndef Material "Material_{role_index}" {{}}\n',
            encoding="utf-8",
        )
        material_record = _record(material, root=root)
        textures: dict[str, dict[str, object]] = {}
        structural_textures: dict[str, dict[str, object]] = {}
        for texture_index, texture_role in enumerate(PBR_REQUIRED_TEXTURES):
            texture = (
                root
                / "materials"
                / role
                / f"{role}-{texture_role}.png"
            )
            texture.write_bytes(
                (
                    f"locked-{role_index}-{texture_index}-"
                    f"{role}-{texture_role}"
                ).encode("utf-8")
            )
            texture_record = {
                **_record(texture, root=root),
                "width_px": 2048,
                "height_px": 2048,
                "color_space": (
                    "sRGB" if texture_role == "base_color" else "raw"
                ),
            }
            textures[texture_role] = texture_record
            structural_textures[texture_role] = {
                **texture_record,
                "color_space": str(texture_record["color_space"]).casefold(),
            }
        material_id = f"fireviewer.pbr.{role}"
        materials[role] = {
            "material_id": material_id,
            "material_file": material_record,
            "material_prim_path": "/Material",
            "metres_per_uv_tile": float(2 + role_index),
            "textures": textures,
        }
        structural[role] = {
            "material_id": material_id,
            "material_file": material_record["path"],
            "material_file_sha256": material_record["sha256"],
            "material_prim_path": "/Material",
            "metres_per_uv_tile": float(2 + role_index),
            "textures": structural_textures,
        }
    manifest = root / "manifest-v3.json"
    manifest.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    marker = {
        "schema_version": 1,
        "state": "ASSET_BUNDLE_INSTALLED",
        "bundle_sha256": hashlib.sha256(b"curated-bundle").hexdigest(),
        "manifest_relative": manifest.relative_to(root).as_posix(),
        "runtime_manifest_sha256": _sha256(manifest),
        "pbr_material_roles": list(PBR_MATERIAL_ROLES),
        "pbr_materials_sha256": _canonical_sha256(structural),
    }
    (root / INSTALL_MARKER).write_text(
        json.dumps(marker, sort_keys=True),
        encoding="utf-8",
    )
    return root, manifest


def _write_native_preparation_fixture(
    root: Path,
) -> dict[str, Path]:
    volume = root.resolve()
    zone = volume / "zones" / "Z-NATIVE-400"
    build_root = zone / "build"
    terrain_root = build_root / "terrain"
    metadata_root = build_root / "metadata"
    raw_root = zone / "raw"
    mnt_root = raw_root / "mnt"
    for path in (
        terrain_root,
        metadata_root,
        mnt_root,
        raw_root / "vectors",
    ):
        path.mkdir(parents=True, exist_ok=True)

    payloads: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    for index in range(terrain_pbr.NATIVE_ZONE_TILE_COUNT):
        tile_ref = f"T{index:03d}"
        payload = terrain_root / f"{tile_ref}.usda"
        payload.write_text(
            f'#usda 1.0\ndef Xform "Tile_{index}" {{}}\n',
            encoding="utf-8",
        )
        payload_record = _record(payload, root=zone)
        payloads.append(payload_record)
        coverage.append(
            {
                "tile_ref": tile_ref,
                "terrain_payload": payload_record["path"],
                "instance_namespace": index + 1,
            }
        )
        mnt = mnt_root / f"{tile_ref}.tif"
        mnt.write_bytes(f"mnt:{tile_ref}".encode())
        entries.append(
            {
                "dataset": "mnt",
                "tile_ref": tile_ref,
                "download": {
                    "state": "downloaded",
                    "relpath": mnt.name,
                    "bytes": mnt.stat().st_size,
                    "sha256": _sha256(mnt),
                },
            }
        )

    vector_sources: dict[str, list[dict[str, object]]] = {}
    for category in ("vegetation", "hydrology", "roads", "buildings"):
        vector = raw_root / "vectors" / f"{category}.geojson"
        vector.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "crs": {
                        "type": "name",
                        "properties": {"name": "EPSG:2154"},
                    },
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"class": category},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [700000.0, 6600000.0],
                                        [720000.0, 6600000.0],
                                        [720000.0, 6620000.0],
                                        [700000.0, 6620000.0],
                                        [700000.0, 6600000.0],
                                    ]
                                ],
                            },
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        vector_sources[category] = [
            {
                "bbox_epsg2154": [
                    700000.0,
                    6600000.0,
                    720000.0,
                    6620000.0,
                ],
                "feature_count": 1,
                "download": {
                    "state": "downloaded",
                    "relpath": vector.relative_to(raw_root).as_posix(),
                    "bytes": vector.stat().st_size,
                    "sha256": _sha256(vector),
                },
            }
        ]

    source_lock = zone / "source-lock.json"
    source_lock.write_text(
        json.dumps(
            {
                "zone_id": "Z-NATIVE-400",
                "entries": entries,
                "vector_sources": vector_sources,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lidar = metadata_root / "lidar-quality.json"
    lidar.write_text(
        json.dumps({"state": "PDAL_VALIDATED", "source_count": 400}),
        encoding="utf-8",
    )
    georeference = metadata_root / "georeference.json"
    georeference.write_text(
        json.dumps(
            {
                "zone_id": "Z-NATIVE-400",
                "crs": "EPSG:2154",
                "vertical_datum": "IGN69",
                "local_origin_epsg2154": [700000.0, 6600000.0],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    root_usd = build_root / "scene.usda"
    root_usd.write_text(
        '#usda 1.0\ndef Xform "World" {}\n',
        encoding="utf-8",
    )
    build_receipt = build_root / "build-receipt.json"
    build_receipt.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "zone_id": "Z-NATIVE-400",
                "source_profile": "full",
                "fire_simulation_status": (
                    "blocked_pending_editor_review"
                ),
                "root_usd": _record(root_usd, root=zone),
                "source_lock": _record(source_lock, root=zone),
                "lidar_quality": {
                    **_record(lidar, root=zone),
                    "source_count": 400,
                },
                "payloads": payloads,
                "tile_coverage": coverage,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    auto_validation = zone / "scene-auto-validation.json"
    auto_validation.write_text(
        json.dumps(
            {
                "state": "AUTO_VALIDATED",
                "fire_simulation_status": (
                    "blocked_pending_editor_review"
                ),
                "build_receipt_sha256": _sha256(build_receipt),
                "root_usd_sha256": _sha256(root_usd),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    bundle, manifest = _write_locked_bundle(volume / "bundle")
    artifacts = volume / "artifacts" / "ground"
    artifacts.parent.mkdir()
    return {
        "volume": volume,
        "zone": zone,
        "build": build_receipt,
        "auto": auto_validation,
        "source_lock": source_lock,
        "georeference": georeference,
        "bundle": bundle,
        "manifest": manifest,
        "artifacts": artifacts,
        "request": artifacts / "terrain-pbr-request.json",
        "first_mnt": mnt_root / "T000.tif",
    }


def _fake_native_terrain_derivation(
    *,
    context: dict[str, object],
    physical_evidence_root: Path,
    final_evidence_root: Path,
) -> tuple[tuple[TerrainTileEvidence, ...], dict[str, object]]:
    volume = Path(context["volume_root"])
    scene_bounds = Bounds2d(
        700000.0,
        6600000.0,
        720000.0,
        6620000.0,
    )
    vector_physical_root = (
        physical_evidence_root / "classified-vectors"
    )
    vector_final_root = final_evidence_root / "classified-vectors"
    vector_physical_root.mkdir(parents=True)
    vectors: dict[str, SpatialEvidence] = {}
    for semantic in terrain_pbr.CLASSIFICATION_SEMANTICS:
        physical = vector_physical_root / f"{semantic}.geojson"
        final = vector_final_root / f"{semantic}.geojson"
        physical.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"semantic": semantic},
                            "geometry": None,
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        lock = terrain_pbr._prepared_file_lock(
            physical_path=physical,
            final_path=final,
            volume_root=volume,
        )
        vectors[semantic] = SpatialEvidence(
            stable_id=f"Z-NATIVE-400:{semantic}",
            semantic=semantic,
            content_kind="classified_vector",
            usage="blend_weights_only",
            lock=lock,
            crs="EPSG:2154",
            bounds=scene_bounds,
            resolution_m=0.5,
            feature_count=1,
        )

    elevation_physical_root = physical_evidence_root / "elevation"
    elevation_final_root = final_evidence_root / "elevation"
    elevation_physical_root.mkdir()
    tiles: list[TerrainTileEvidence] = []
    for index in range(terrain_pbr.NATIVE_ZONE_TILE_COUNT):
        column = index % 20
        row = index // 20
        tile_ref = f"T{index:03d}"
        bounds = Bounds2d(
            700000.0 + column * 1000.0,
            6600000.0 + row * 1000.0,
            701000.0 + column * 1000.0,
            6601000.0 + row * 1000.0,
        )
        sampling = Bounds2d(
            max(scene_bounds.min_x_m, bounds.min_x_m - 3.5),
            max(scene_bounds.min_y_m, bounds.min_y_m - 3.5),
            min(scene_bounds.max_x_m, bounds.max_x_m + 3.5),
            min(scene_bounds.max_y_m, bounds.max_y_m + 3.5),
        )
        elevation_physical = (
            elevation_physical_root / f"{tile_ref}.tif"
        )
        elevation_final = elevation_final_root / f"{tile_ref}.tif"
        elevation_physical.write_bytes(f"elevation:{tile_ref}".encode())
        elevation_lock = terrain_pbr._prepared_file_lock(
            physical_path=elevation_physical,
            final_path=elevation_final,
            volume_root=volume,
        )
        elevation = SpatialEvidence(
            stable_id=f"{tile_ref}:elevation",
            semantic="elevation",
            content_kind="heightfield",
            usage="height_only",
            lock=elevation_lock,
            crs="EPSG:2154",
            bounds=sampling,
            resolution_m=0.5,
        )
        evidence = (
            elevation,
            *(vectors[name] for name in terrain_pbr.CLASSIFICATION_SEMANTICS),
        )
        evidence_ids = frozenset(
            item.stable_id for item in evidence
        )
        mid_x = (bounds.min_x_m + bounds.max_x_m) * 0.5
        mid_y = (bounds.min_y_m + bounds.max_y_m) * 0.5
        subzone_bounds = (
            Bounds2d(bounds.min_x_m, mid_y, mid_x, bounds.max_y_m),
            Bounds2d(mid_x, mid_y, bounds.max_x_m, bounds.max_y_m),
            Bounds2d(bounds.min_x_m, bounds.min_y_m, mid_x, mid_y),
            Bounds2d(mid_x, bounds.min_y_m, bounds.max_x_m, mid_y),
        )
        subzones = tuple(
            TerrainSubzoneEvidence(
                stable_id=f"{tile_ref}:Q{subzone_index}",
                bounds=subzone,
                mean_elevation_m=100.0 + row + column,
                mean_slope_degrees=4.0 + subzone_index,
                roughness_m=0.2 + subzone_index * 0.01,
                coverage={
                    "forest": 0.65,
                    "water": 0.04,
                    "roads": 0.08,
                    "artificial_ground": 0.12,
                },
                evidence_ids=evidence_ids,
            )
            for subzone_index, subzone in enumerate(subzone_bounds)
        )
        tiles.append(
            TerrainTileEvidence(
                stable_id=tile_ref,
                bounds=bounds,
                evidence=evidence,
                subzones=subzones,
            )
        )
    provenance: dict[str, object] = {
        "tile_count": len(tiles),
        "crs": "EPSG:2154",
        "vertical_datum": "IGN69",
        "scene_bounds_source_m": scene_bounds.as_list(),
        "global_mask_atlas_created": False,
        "uniform_mask_substitution_allowed": False,
    }
    provenance["prepared_evidence_sha256"] = _canonical_sha256(
        provenance
    )
    return tuple(tiles), provenance


def _source(
    *,
    evidence_root: Path,
    tile_id: str,
    semantic: str,
    bounds: Bounds2d,
) -> SpatialEvidence:
    vector = semantic in {"water", "roads", "artificial_ground"}
    suffix = ".geojson" if vector else ".tif"
    path = evidence_root / f"{tile_id}-{semantic}{suffix}"
    path.write_bytes(f"measured:{tile_id}:{semantic}".encode("utf-8"))
    return SpatialEvidence(
        stable_id=f"{tile_id}:{semantic}",
        semantic=semantic,
        content_kind="classified_vector" if vector else (
            "heightfield" if semantic == "elevation" else "classified_mask"
        ),
        usage="height_only" if semantic == "elevation" else "blend_weights_only",
        lock=FileLock(
            path=path.relative_to(evidence_root).as_posix(),
            sha256=_sha256(path),
            size_bytes=path.stat().st_size,
        ),
        crs="EPSG:2154",
        bounds=bounds,
        resolution_m=0.5 if semantic == "elevation" else 1.0,
        feature_count=17 if vector else None,
    )


def _tile(
    *,
    evidence_root: Path,
    tile_id: str,
    bounds: Bounds2d,
    coverage: dict[str, float] | None = None,
    source_bounds: Bounds2d | None = None,
) -> TerrainTileEvidence:
    sources = tuple(
        _source(
            evidence_root=evidence_root,
            tile_id=tile_id,
            semantic=semantic,
            bounds=source_bounds or bounds,
        )
        for semantic in terrain_pbr.EVIDENCE_SEMANTICS
    )
    subzone = TerrainSubzoneEvidence(
        stable_id=f"{tile_id}:S0",
        bounds=bounds,
        mean_elevation_m=242.0,
        mean_slope_degrees=18.0,
        roughness_m=0.42,
        coverage=coverage
        or {
            "forest": 0.65,
            "water": 0.03,
            "roads": 0.08,
            "artificial_ground": 0.12,
        },
        evidence_ids=frozenset(source.stable_id for source in sources),
    )
    return TerrainTileEvidence(
        stable_id=tile_id,
        bounds=bounds,
        evidence=sources,
        subzones=(subzone,),
    )


def _write_native_ground_receipt(
    *,
    artifact_root: Path,
    plan: terrain_pbr.TerrainPbrPlan,
) -> tuple[Path, Path]:
    spec = build_composite_ground_material_spec(plan)
    ground = artifact_root / "authored" / "ground-composite.usdc"
    ground.parent.mkdir(parents=True, exist_ok=True)
    ground.write_bytes(b"native-usd-composite-ground-material")
    derived_root = artifact_root / "authored" / "ground-composite.inputs"
    derived_root.mkdir()
    derived_inputs: list[dict[str, object]] = []
    for index, binding in enumerate(spec.spatial_bindings):
        semantic = str(binding["semantic"])
        required_assets = (
            ("relief_slope", "relief_roughness")
            if semantic == "elevation"
            else ("mask",)
        )
        assets: dict[str, object] = {}
        for asset_index, asset_role in enumerate(required_assets):
            asset = (
                derived_root
                / f"{index:04d}-{asset_index}-{asset_role}.tif"
            )
            asset.write_bytes(
                f"native:{binding['binding_id']}:{asset_role}".encode()
            )
            assets[asset_role] = _record(asset, root=artifact_root)
        derived_inputs.append(
            {
                "binding_id": binding["binding_id"],
                "tile_id": binding["tile_id"],
                "tile_ref": binding["tile_ref"],
                "semantic": semantic,
                "source_path": binding["path"],
                "source_sha256": binding["sha256"],
                "tile_bounds_m": list(binding["tile_bounds_m"]),
                "source_tile_bounds_m": list(
                    binding["source_tile_bounds_m"]
                ),
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
                "representation": (
                    "slope_and_roughness_from_locked_heightfield"
                    if semantic == "elevation"
                    else "feathered_classification_mask"
                ),
                "coverage_metrics": _coverage_metrics(semantic),
                "assets": assets,
            }
        )
    derived_inputs.sort(key=lambda value: str(value["binding_id"]))
    derived_sha256 = _canonical_sha256(derived_inputs)
    tile_payloads = _write_tile_material_payloads(
        artifact_root=artifact_root,
        specification=spec,
        payload_root=derived_root / "tile-materials",
    )
    tile_payloads_sha256 = _canonical_sha256(tile_payloads)
    receipt = artifact_root / "authored" / "ground-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": (
                    terrain_pbr.COMPOSITE_AUTHORING_SCHEMA_VERSION
                ),
                "state": terrain_pbr.NATIVE_GROUND_STATE,
                "terrain_pbr_plan_sha256": plan.fingerprint,
                "specification_sha256": spec.fingerprint,
                "metric_uv_sha256": spec.metric_uv_sha256,
                "blend_graph_sha256": spec.blend_graph_sha256,
                "material_bindings_sha256": spec.material_bindings_sha256,
                "evidence_bindings_sha256": spec.evidence_bindings_sha256,
                "derived_spatial_inputs": derived_inputs,
                "derived_spatial_inputs_sha256": derived_sha256,
                "tile_material_payloads": tile_payloads,
                "tile_material_payloads_sha256": tile_payloads_sha256,
                "ground_material": {
                    **_record(ground, root=artifact_root),
                    "prim_path": "/Ground",
                },
                "native_validation": _native_validation(
                    plan=plan,
                    specification=spec,
                    derived_inputs=derived_inputs,
                    tile_payloads=tile_payloads,
                ),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return ground, receipt


class _FakeNativeBackend:
    def __init__(self, *, uniform_fallback: bool = False) -> None:
        self.calls = 0
        self.uniform_fallback = uniform_fallback

    def author(
        self,
        *,
        plan: terrain_pbr.TerrainPbrPlan,
        specification: terrain_pbr.CompositeGroundMaterialSpec,
        bundle_root: Path,
        evidence_root: Path,
        artifact_root: Path,
        output_path: Path,
        final_output_path: Path,
        derived_output_root: Path,
        final_derived_output_root: Path,
    ) -> dict[str, object]:
        del (
            bundle_root,
            evidence_root,
            final_output_path,
            final_derived_output_root,
        )
        self.calls += 1
        output_path.write_bytes(b"native-materialx-usd-stage")
        derived_output_root.mkdir()
        derived_inputs: list[dict[str, object]] = []
        for index, binding in enumerate(specification.spatial_bindings):
            semantic = str(binding["semantic"])
            roles = (
                ("relief_slope", "relief_roughness")
                if semantic == "elevation"
                else ("mask",)
            )
            assets: dict[str, object] = {}
            for asset_index, role in enumerate(roles):
                path = (
                    derived_output_root
                    / f"{index:04d}-{asset_index}-{role}.tif"
                )
                path.write_bytes(
                    f"derived:{binding['binding_id']}:{role}".encode()
                )
                assets[role] = _record(path, root=artifact_root)
            derived_inputs.append(
                {
                    "binding_id": binding["binding_id"],
                    "tile_id": binding["tile_id"],
                    "tile_ref": binding["tile_ref"],
                    "semantic": semantic,
                    "source_path": binding["path"],
                    "source_sha256": binding["sha256"],
                    "tile_bounds_m": list(binding["tile_bounds_m"]),
                    "source_tile_bounds_m": list(
                        binding["source_tile_bounds_m"]
                    ),
                    "sampling_bounds_m": list(
                        binding["sampling_bounds_m"]
                    ),
                    "source_sampling_bounds_m": list(
                        binding["source_sampling_bounds_m"]
                    ),
                    "halo_m": binding["halo_m"],
                    "halo_edges_m": list(binding["halo_edges_m"]),
                    "source_bounds_m": list(binding["source_bounds_m"]),
                    "scene_origin_source_m": list(
                        binding["scene_origin_source_m"]
                    ),
                    "representation": (
                        "slope_and_roughness_from_locked_heightfield"
                        if semantic == "elevation"
                        else "feathered_classification_mask"
                    ),
                    "coverage_metrics": _coverage_metrics(semantic),
                    "assets": assets,
                }
            )
        derived_inputs.sort(key=lambda value: str(value["binding_id"]))
        tile_payloads = _write_tile_material_payloads(
            artifact_root=artifact_root,
            specification=specification,
            payload_root=derived_output_root / "tile-materials",
        )
        return {
            "derived_spatial_inputs": derived_inputs,
            "tile_payload_layers": tile_payloads,
            "native_validation": _native_validation(
                plan=plan,
                specification=specification,
                derived_inputs=derived_inputs,
                tile_payloads=tile_payloads,
                uniform_fallback=self.uniform_fallback,
            ),
        }


class TerrainPbrTests(unittest.TestCase):
    def test_loads_exact_seven_byte_locked_roles_without_pxr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle, manifest = _write_locked_bundle(
                Path(directory) / "bundle"
            )
            library = load_locked_material_library(
                bundle_root=bundle,
                manifest_path=manifest,
            )

        self.assertEqual(
            tuple(material.role for material in library.materials),
            PBR_MATERIAL_ROLES,
        )
        self.assertEqual(
            {
                texture.role
                for material in library.materials
                for texture in material.textures
            },
            set(PBR_REQUIRED_TEXTURES),
        )
        self.assertNotIn("pxr", terrain_pbr.__dict__)

    def test_uvs_are_world_metric_and_continuous_across_adjacent_tiles(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            evidence = root / "evidence"
            evidence.mkdir()
            left = _tile(
                evidence_root=evidence,
                tile_id="T000",
                bounds=Bounds2d(1000.0, 2000.0, 1100.0, 2100.0),
            )
            right = _tile(
                evidence_root=evidence,
                tile_id="T001",
                bounds=Bounds2d(1100.0, 2000.0, 1200.0, 2100.0),
            )
            before = sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
            )
            plan = build_terrain_pbr_plan(
                scene_id="SIM-01",
                bundle_root=bundle,
                material_manifest_path=manifest,
                evidence_root=evidence,
                tiles=(left, right),
                world_uv_origin_m=(500.0, 1500.0),
            )
            after = sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
            )

        for role in PBR_MATERIAL_ROLES:
            before_edge = plan.uv_for(role, 1100.0 - 1.0e-6, 2042.0)
            on_edge = plan.uv_for(role, 1100.0, 2042.0)
            after_edge = plan.uv_for(role, 1100.0 + 1.0e-6, 2042.0)
            self.assertLess(before_edge[0], on_edge[0])
            self.assertLess(on_edge[0], after_edge[0])
            self.assertAlmostEqual(
                after_edge[0] - before_edge[0],
                2.0e-6
                / plan.material_library.by_role(role).metres_per_uv_tile,
                places=12,
            )
            self.assertEqual(before_edge[1], on_edge[1])
            self.assertEqual(on_edge[1], after_edge[1])
            self.assertFalse(
                plan.as_dict()["metric_uv"][role]["tile_local_reset"]
            )
        self.assertEqual(before, after, "planning must not generate textures")
        appearance = plan.as_dict()["appearance_contract"]
        self.assertFalse(appearance["raw_orthophoto_allowed"])
        self.assertFalse(appearance["baked_object_imagery_allowed"])
        self.assertEqual(appearance["generated_texture_assets"], [])
        continuity = terrain_pbr._measure_metric_uv_continuity(plan)
        self.assertEqual(continuity["adjacent_pair_count"], 1)
        self.assertEqual(continuity["maximum_absolute_uv_error"], 0.0)
        self.assertEqual(
            continuity["measurement_sha256"],
            _canonical_sha256(
                {
                    key: value
                    for key, value in continuity.items()
                    if key != "measurement_sha256"
                }
            ),
        )

    def test_plan_is_deterministic_when_tiles_and_sources_are_reordered(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            evidence = root / "evidence"
            evidence.mkdir()
            tile_a = _tile(
                evidence_root=evidence,
                tile_id="T-A",
                bounds=Bounds2d(0.0, 0.0, 100.0, 100.0),
            )
            tile_b = _tile(
                evidence_root=evidence,
                tile_id="T-B",
                bounds=Bounds2d(100.0, 0.0, 200.0, 100.0),
            )
            reordered_a = TerrainTileEvidence(
                stable_id=tile_a.stable_id,
                bounds=tile_a.bounds,
                evidence=tuple(reversed(tile_a.evidence)),
                subzones=tile_a.subzones,
            )
            first = build_terrain_pbr_plan(
                scene_id="SIM-02",
                bundle_root=bundle,
                material_manifest_path=manifest,
                evidence_root=evidence,
                tiles=(tile_b, reordered_a),
            )
            second = build_terrain_pbr_plan(
                scene_id="SIM-02",
                bundle_root=bundle,
                material_manifest_path=manifest,
                evidence_root=evidence,
                tiles=(tile_a, tile_b),
            )

        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.as_dict(), second.as_dict())

    def test_blend_uses_all_roles_and_normalizes_without_object_imagery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            evidence = root / "evidence"
            evidence.mkdir()
            tile = _tile(
                evidence_root=evidence,
                tile_id="MIXED",
                bounds=Bounds2d(0.0, 0.0, 200.0, 200.0),
                coverage={
                    "forest": 0.70,
                    "water": 0.08,
                    "roads": 0.10,
                    "artificial_ground": 0.20,
                },
            )
            plan = build_terrain_pbr_plan(
                scene_id="SIM-03",
                bundle_root=bundle,
                material_manifest_path=manifest,
                evidence_root=evidence,
                tiles=(tile,),
            )

        portable = plan.as_dict()
        weights = portable["tiles"][0]["subzones"][0]["mean_weights"]
        self.assertEqual(set(weights), set(PBR_MATERIAL_ROLES))
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=10)
        self.assertGreater(weights["forest_floor"], weights["water"])
        transition = portable["blend_graph"]["transition_policy"]
        self.assertEqual(
            transition["material_metric_uv_continuity"],
            "shared_world_position",
        )
        self.assertEqual(
            transition["classification_address_mode"],
            "clamp",
        )
        self.assertFalse(transition["base_color_from_spatial_evidence"])
        self.assertFalse(
            transition["spatial_evidence_can_create_rendered_objects"]
        )

    def test_rejects_raw_orthophoto_and_unsafe_appearance_usage(self) -> None:
        common = {
            "stable_id": "T0:forest",
            "semantic": "forest",
            "lock": FileLock(
                path="forest.tif",
                sha256="a" * 64,
                size_bytes=10,
            ),
            "crs": "EPSG:2154",
            "bounds": Bounds2d(0.0, 0.0, 10.0, 10.0),
            "resolution_m": 0.5,
        }
        with self.assertRaisesRegex(
            TerrainPbrContractError,
            "raw orthophotos",
        ):
            SpatialEvidence(
                **common,
                content_kind="raw_orthophoto",
                usage="blend_weights_only",
            )
        with self.assertRaisesRegex(
            TerrainPbrContractError,
            "unsafe appearance usage",
        ):
            SpatialEvidence(
                **common,
                content_kind="classified_mask",
                usage="base_color",
            )

    def test_rejects_tampered_material_bytes_and_stale_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle, manifest = _write_locked_bundle(
                Path(directory) / "bundle"
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            forest = payload["pbr_materials"]["forest_floor"]["material_file"]
            material_path = manifest.parent / forest["path"]
            material_path.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(
                TerrainPbrContractError,
                "SHA-256 or size lock",
            ):
                load_locked_material_library(
                    bundle_root=bundle,
                    manifest_path=manifest,
                )

        with tempfile.TemporaryDirectory() as directory:
            bundle, manifest = _write_locked_bundle(
                Path(directory) / "bundle"
            )
            marker_path = bundle / INSTALL_MARKER
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["runtime_manifest_sha256"] = "f" * 64
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(
                TerrainPbrContractError,
                "stale or incomplete",
            ):
                load_locked_material_library(
                    bundle_root=bundle,
                    manifest_path=manifest,
                )

    def test_rejects_escaping_paths_and_tampered_spatial_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["pbr_materials"]["water"]["material_file"]["path"] = (
                "../outside.usda"
            )
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            marker_path = bundle / INSTALL_MARKER
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["runtime_manifest_sha256"] = _sha256(manifest)
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(
                TerrainPbrContractError,
                "escapes",
            ):
                load_locked_material_library(
                    bundle_root=bundle,
                    manifest_path=manifest,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            evidence = root / "evidence"
            evidence.mkdir()
            tile = _tile(
                evidence_root=evidence,
                tile_id="TAMPER",
                bounds=Bounds2d(0.0, 0.0, 50.0, 50.0),
            )
            tampered = evidence / tile.evidence[0].lock.path
            tampered.write_bytes(b"not-the-locked-heightfield")
            with self.assertRaisesRegex(
                TerrainPbrContractError,
                "SHA-256 or size lock",
            ):
                build_terrain_pbr_plan(
                    scene_id="SIM-04",
                    bundle_root=bundle,
                    material_manifest_path=manifest,
                    evidence_root=evidence,
                    tiles=(tile,),
                )

    def test_rejects_missing_evidence_and_subzone_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence"
            evidence.mkdir()
            complete = _tile(
                evidence_root=evidence,
                tile_id="GAP",
                bounds=Bounds2d(0.0, 0.0, 100.0, 100.0),
            )
            with self.assertRaisesRegex(
                TerrainPbrContractError,
                "exactly elevation",
            ):
                TerrainTileEvidence(
                    stable_id="MISSING",
                    bounds=complete.bounds,
                    evidence=complete.evidence[:-1],
                    subzones=complete.subzones,
                )
            subzone = complete.subzones[0]
            with self.assertRaisesRegex(
                TerrainPbrContractError,
                "partition",
            ):
                TerrainTileEvidence(
                    stable_id="GAPPED",
                    bounds=complete.bounds,
                    evidence=complete.evidence,
                    subzones=(
                        TerrainSubzoneEvidence(
                            stable_id=subzone.stable_id,
                            bounds=Bounds2d(0.0, 0.0, 90.0, 100.0),
                            mean_elevation_m=subzone.mean_elevation_m,
                            mean_slope_degrees=subzone.mean_slope_degrees,
                            roughness_m=subzone.roughness_m,
                            coverage=subzone.coverage,
                            evidence_ids=subzone.evidence_ids,
                        ),
                    ),
                )

    def test_composite_spec_requires_tiled_masks_and_all_reachable_roles(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            evidence = root / "evidence"
            evidence.mkdir()
            tiles = (
                _tile(
                    evidence_root=evidence,
                    tile_id="T100",
                    bounds=Bounds2d(0.0, 0.0, 100.0, 100.0),
                ),
                _tile(
                    evidence_root=evidence,
                    tile_id="T101",
                    bounds=Bounds2d(100.0, 0.0, 200.0, 100.0),
                ),
            )
            plan = build_terrain_pbr_plan(
                scene_id="SIM-05",
                bundle_root=bundle,
                material_manifest_path=manifest,
                evidence_root=evidence,
                tiles=tiles,
            )
            spec = build_composite_ground_material_spec(plan)

        portable = spec.as_dict()
        self.assertEqual(portable["material_prim_path"], "/Ground")
        self.assertEqual(
            portable["material_roles"],
            list(PBR_MATERIAL_ROLES),
        )
        self.assertEqual(
            len(portable["spatial_bindings"]),
            len(tiles) * len(terrain_pbr.EVIDENCE_SEMANTICS),
        )
        interface = portable["shader_interface"]
        self.assertTrue(
            interface["all_tile_branches_must_be_surface_reachable"]
        )
        self.assertFalse(interface["uniform_fallback_allowed"])
        topology = portable["binding_topology"]
        self.assertFalse(topology["one_monolithic_mask_atlas_allowed"])
        self.assertFalse(topology["single_graph_for_all_tiles_allowed"])
        self.assertEqual(topology["tile_payload_count"], len(tiles))
        self.assertFalse(topology["source_colour_can_feed_base_color"])
        self.assertEqual(
            portable["required_native_metadata"][
                "fireviewer:terrainPbrPlanSha256"
            ],
            plan.fingerprint,
        )
        appearance = plan.as_dict()["appearance_contract"]
        self.assertEqual(
            appearance["macro_variation"]["method"],
            "rotated_world_metric_secondary_scale",
        )
        self.assertEqual(
            appearance["steep_slope_mapping"]["material_roles"],
            ["rock"],
        )

    def test_composite_spec_crops_global_sources_to_local_tile_bounds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            evidence = root / "evidence"
            evidence.mkdir()
            tile = _tile(
                evidence_root=evidence,
                tile_id="CROP",
                bounds=Bounds2d(
                    700_000.0,
                    6_600_000.0,
                    700_100.0,
                    6_600_100.0,
                ),
                source_bounds=Bounds2d(
                    699_000.0,
                    6_599_000.0,
                    701_000.0,
                    6_601_000.0,
                ),
            )
            plan = build_terrain_pbr_plan(
                scene_id="SIM-CROP",
                bundle_root=bundle,
                material_manifest_path=manifest,
                evidence_root=evidence,
                tiles=(tile,),
            )
            spec = build_composite_ground_material_spec(plan)

        self.assertEqual(
            plan.scene_origin_source_m,
            (700_000.0, 6_600_000.0),
        )
        for binding in spec.spatial_bindings:
            self.assertEqual(
                binding["tile_bounds_m"],
                (0.0, 0.0, 100.0, 100.0),
            )
            self.assertEqual(
                binding["source_tile_bounds_m"],
                (700_000.0, 6_600_000.0, 700_100.0, 6_600_100.0),
            )
            self.assertEqual(
                binding["source_bounds_m"],
                (699_000.0, 6_599_000.0, 701_000.0, 6_601_000.0),
            )

    def test_halo_is_full_at_internal_edge_and_clipped_at_scene_edge(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            evidence = root / "evidence"
            evidence.mkdir()
            scene_bounds = Bounds2d(0.0, 0.0, 200.0, 100.0)
            tiles = (
                _tile(
                    evidence_root=evidence,
                    tile_id="EDGE-L",
                    bounds=Bounds2d(0.0, 0.0, 100.0, 100.0),
                    source_bounds=scene_bounds,
                ),
                _tile(
                    evidence_root=evidence,
                    tile_id="EDGE-R",
                    bounds=Bounds2d(100.0, 0.0, 200.0, 100.0),
                    source_bounds=scene_bounds,
                ),
            )
            plan = build_terrain_pbr_plan(
                scene_id="SIM-EDGE-HALO",
                bundle_root=bundle,
                material_manifest_path=manifest,
                evidence_root=evidence,
                tiles=tiles,
            )
            spec = build_composite_ground_material_spec(plan)

        left = spec.bindings_for_tile("EDGE-L")[0]
        right = spec.bindings_for_tile("EDGE-R")[0]
        halo = float(left["halo_m"])
        self.assertEqual(
            left["source_sampling_bounds_m"],
            (0.0, 0.0, 100.0 + halo, 100.0),
        )
        self.assertEqual(
            right["source_sampling_bounds_m"],
            (100.0 - halo, 0.0, 200.0, 100.0),
        )
        self.assertEqual(left["halo_edges_m"], (0.0, 0.0, halo, 0.0))
        self.assertEqual(right["halo_edges_m"], (halo, 0.0, 0.0, 0.0))

    def test_halo_sampling_matches_crossing_mask_and_rejects_no_halo(
        self,
    ) -> None:
        import numpy as np

        class _Band:
            def __init__(self, values: object) -> None:
                self.values = values

            def ReadAsArray(
                self,
                xoff: int,
                yoff: int,
                xsize: int,
                ysize: int,
            ) -> object:
                return self.values[
                    yoff : yoff + ysize,
                    xoff : xoff + xsize,
                ]

        class _Dataset:
            def __init__(
                self,
                *,
                minimum_x: float,
                maximum_y: float,
                width: int,
                height: int,
            ) -> None:
                self.RasterXSize = width
                self.RasterYSize = height
                self.transform = (
                    minimum_x,
                    1.0,
                    0.0,
                    maximum_y,
                    0.0,
                    -1.0,
                )
                values = np.empty((height, width), dtype=np.float32)
                for row in range(height):
                    y = maximum_y - row - 0.5
                    for column in range(width):
                        x = minimum_x + column + 0.5
                        values[row, column] = 0.2 + 0.001 * x + 0.001 * y
                self.band = _Band(values)

            def GetGeoTransform(self) -> tuple[float, ...]:
                return self.transform

            def GetRasterBand(self, index: int) -> _Band:
                self.assert_band = index
                return self.band

        left = _Dataset(
            minimum_x=-10.0,
            maximum_y=110.0,
            width=120,
            height=120,
        )
        right = _Dataset(
            minimum_x=90.0,
            maximum_y=110.0,
            width=120,
            height=120,
        )
        left_value = terrain_pbr._sample_north_up_raster(
            np=np,
            dataset=left,
            world_x_m=100.0,
            world_y_m=50.0,
        )
        right_value = terrain_pbr._sample_north_up_raster(
            np=np,
            dataset=right,
            world_x_m=100.0,
            world_y_m=50.0,
        )
        self.assertAlmostEqual(left_value, right_value, places=7)
        no_halo = _Dataset(
            minimum_x=100.0,
            maximum_y=100.0,
            width=100,
            height=100,
        )
        with self.assertRaisesRegex(
            TerrainPbrContractError,
            "halo",
        ):
            terrain_pbr._sample_north_up_raster(
                np=np,
                dataset=no_halo,
                world_x_m=100.0,
                world_y_m=50.0,
            )

    def test_finite_elevation_gate_rejects_nodata(self) -> None:
        import numpy as np

        with self.assertRaisesRegex(
            TerrainPbrContractError,
            "missing or non-finite",
        ):
            terrain_pbr._finite_array_metrics(
                np,
                np.asarray([[120.0, float("nan")]], dtype=np.float32),
                label="test elevation",
            )
        metrics = terrain_pbr._finite_array_metrics(
            np,
            np.asarray([[120.0, 121.0]], dtype=np.float32),
            label="test elevation",
        )
        self.assertEqual(metrics["finite_fraction"], 1.0)
        self.assertEqual(metrics["nodata_count"], 0)
        self.assertEqual(terrain_pbr._aligned_native_halo_m(1.0), 5.0)
        self.assertEqual(terrain_pbr._aligned_native_halo_m(0.5), 3.5)
        self.assertAlmostEqual(
            terrain_pbr._aligned_native_halo_m(0.2),
            3.6,
        )

    def test_validates_real_composite_artifact_for_variant_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            evidence = root / "evidence"
            evidence.mkdir()
            tile = _tile(
                evidence_root=evidence,
                tile_id="T200",
                bounds=Bounds2d(0.0, 0.0, 100.0, 100.0),
            )
            plan = build_terrain_pbr_plan(
                scene_id="SIM-06",
                bundle_root=bundle,
                material_manifest_path=manifest,
                evidence_root=evidence,
                tiles=(tile,),
            )
            ground, receipt = _write_native_ground_receipt(
                artifact_root=root,
                plan=plan,
            )
            ground_sha256 = _sha256(ground)
            artifact = validate_composite_ground_material_artifact(
                plan=plan,
                artifact_root=root,
                bundle_root=bundle,
                evidence_root=evidence,
                authoring_receipt_path=receipt,
            )

        layout = artifact.as_layout_artifact()
        self.assertEqual(layout["sha256"], ground_sha256)
        self.assertEqual(layout["prim_path"], "/Ground")
        self.assertEqual(
            layout["isolated_content_roles"],
            ["object_free_pbr_ground"],
        )
        self.assertEqual(
            layout["topology"],
            "payload_tiled_materials_shared_pbr_library",
        )
        self.assertEqual(len(layout["tile_material_payloads"]), 1)
        self.assertEqual(
            layout["tile_material_payloads"][0]["tile_ref"],
            "T200",
        )
        self.assertEqual(
            layout["terrain_pbr_plan_sha256"],
            plan.fingerprint,
        )

    def test_rejects_uniform_or_tampered_composite_ground_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            evidence = root / "evidence"
            evidence.mkdir()
            tile = _tile(
                evidence_root=evidence,
                tile_id="T300",
                bounds=Bounds2d(0.0, 0.0, 100.0, 100.0),
            )
            plan = build_terrain_pbr_plan(
                scene_id="SIM-07",
                bundle_root=bundle,
                material_manifest_path=manifest,
                evidence_root=evidence,
                tiles=(tile,),
            )
            _ground, receipt = _write_native_ground_receipt(
                artifact_root=root,
                plan=plan,
            )
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["native_validation"]["uniform_fallback_present"] = True
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                TerrainPbrContractError,
                "incomplete, uniform or unsafe",
            ):
                validate_composite_ground_material_artifact(
                    plan=plan,
                    artifact_root=root,
                    bundle_root=bundle,
                    evidence_root=evidence,
                    authoring_receipt_path=receipt,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            evidence = root / "evidence"
            evidence.mkdir()
            tile = _tile(
                evidence_root=evidence,
                tile_id="T302",
                bounds=Bounds2d(0.0, 0.0, 100.0, 100.0),
            )
            plan = build_terrain_pbr_plan(
                scene_id="SIM-08-COLOR",
                bundle_root=bundle,
                material_manifest_path=manifest,
                evidence_root=evidence,
                tiles=(tile,),
            )
            _ground, receipt = _write_native_ground_receipt(
                artifact_root=root,
                plan=plan,
            )
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["native_validation"][
                "texture_color_space_contract"
            ]["base_color"] = "none"
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                TerrainPbrContractError,
                "incomplete, uniform or unsafe",
            ):
                validate_composite_ground_material_artifact(
                    plan=plan,
                    artifact_root=root,
                    bundle_root=bundle,
                    evidence_root=evidence,
                    authoring_receipt_path=receipt,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            evidence = root / "evidence"
            evidence.mkdir()
            tile = _tile(
                evidence_root=evidence,
                tile_id="T301",
                bounds=Bounds2d(0.0, 0.0, 100.0, 100.0),
            )
            plan = build_terrain_pbr_plan(
                scene_id="SIM-08",
                bundle_root=bundle,
                material_manifest_path=manifest,
                evidence_root=evidence,
                tiles=(tile,),
            )
            ground, receipt = _write_native_ground_receipt(
                artifact_root=root,
                plan=plan,
            )
            ground.write_bytes(b"tampered-ground")
            with self.assertRaisesRegex(
                TerrainPbrContractError,
                "SHA-256 or size lock",
            ):
                validate_composite_ground_material_artifact(
                    plan=plan,
                    artifact_root=root,
                    bundle_root=bundle,
                    evidence_root=evidence,
                    authoring_receipt_path=receipt,
                )

    def test_native_authoring_backend_writes_reuses_and_validates_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            evidence = root / "evidence"
            evidence.mkdir()
            tile = _tile(
                evidence_root=evidence,
                tile_id="AUTHOR",
                bounds=Bounds2d(0.0, 0.0, 100.0, 100.0),
            )
            plan = build_terrain_pbr_plan(
                scene_id="SIM-09",
                bundle_root=bundle,
                material_manifest_path=manifest,
                evidence_root=evidence,
                tiles=(tile,),
            )
            backend = _FakeNativeBackend()
            artifact = terrain_pbr.author_composite_ground_material(
                plan=plan,
                artifact_root=root,
                bundle_root=bundle,
                evidence_root=evidence,
                output_relative_path="native/ground.usdc",
                receipt_relative_path="native/ground-receipt.json",
                backend=backend,
            )
            reused = terrain_pbr.author_composite_ground_material(
                plan=plan,
                artifact_root=root,
                bundle_root=bundle,
                evidence_root=evidence,
                output_relative_path="native/ground.usdc",
                receipt_relative_path="native/ground-receipt.json",
                backend=backend,
            )
            receipt = json.loads(
                (root / "native" / "ground-receipt.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(backend.calls, 1)
        self.assertEqual(artifact, reused)
        self.assertEqual(artifact.material_prim_path, "/Ground")
        self.assertEqual(artifact.render_context, "mtlx")
        self.assertEqual(
            receipt["state"],
            terrain_pbr.NATIVE_GROUND_STATE,
        )
        self.assertEqual(
            receipt["derived_spatial_inputs_sha256"],
            receipt["native_validation"][
                "derived_spatial_inputs_sha256"
            ],
        )

    def test_native_authoring_rejects_uniform_backend_and_cleans_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            evidence = root / "evidence"
            evidence.mkdir()
            tile = _tile(
                evidence_root=evidence,
                tile_id="UNIFORM",
                bounds=Bounds2d(0.0, 0.0, 100.0, 100.0),
            )
            plan = build_terrain_pbr_plan(
                scene_id="SIM-10",
                bundle_root=bundle,
                material_manifest_path=manifest,
                evidence_root=evidence,
                tiles=(tile,),
            )
            with self.assertRaisesRegex(
                TerrainPbrContractError,
                "incomplete, uniform or unsafe",
            ):
                terrain_pbr.author_composite_ground_material(
                    plan=plan,
                    artifact_root=root,
                    bundle_root=bundle,
                    evidence_root=evidence,
                    output_relative_path="bad/ground.usdc",
                    receipt_relative_path="bad/ground-receipt.json",
                    backend=_FakeNativeBackend(uniform_fallback=True),
                )
            self.assertFalse((root / "bad" / "ground.usdc").exists())
            self.assertFalse(
                (root / "bad" / "ground-receipt.json").exists()
            )
            self.assertFalse((root / "bad" / "ground.inputs").exists())

    def test_native_authoring_crash_leaves_no_final_partial_and_retries(
        self,
    ) -> None:
        class _CrashingBackend:
            def author(self, **kwargs: object) -> dict[str, object]:
                output = Path(kwargs["output_path"])
                derived = Path(kwargs["derived_output_root"])
                output.write_bytes(b"partial-native-stage")
                derived.mkdir()
                (derived / "partial.tif").write_bytes(b"partial")
                raise RuntimeError("simulated native process failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            evidence = root / "evidence"
            evidence.mkdir()
            tile = _tile(
                evidence_root=evidence,
                tile_id="ATOMIC",
                bounds=Bounds2d(0.0, 0.0, 100.0, 100.0),
            )
            plan = build_terrain_pbr_plan(
                scene_id="SIM-ATOMIC",
                bundle_root=bundle,
                material_manifest_path=manifest,
                evidence_root=evidence,
                tiles=(tile,),
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated native process failure",
            ):
                terrain_pbr.author_composite_ground_material(
                    plan=plan,
                    artifact_root=root,
                    bundle_root=bundle,
                    evidence_root=evidence,
                    output_relative_path="atomic/ground.usdc",
                    receipt_relative_path=(
                        "atomic/ground-authoring-receipt.json"
                    ),
                    backend=_CrashingBackend(),
                )
            self.assertFalse((root / "atomic").exists())
            self.assertFalse(
                any(root.glob(".terrain-pbr-*.staging"))
            )
            working = _FakeNativeBackend()
            artifact = terrain_pbr.author_composite_ground_material(
                plan=plan,
                artifact_root=root,
                bundle_root=bundle,
                evidence_root=evidence,
                output_relative_path="atomic/ground.usdc",
                receipt_relative_path=(
                    "atomic/ground-authoring-receipt.json"
                ),
                backend=working,
            )

        self.assertEqual(working.calls, 1)
        self.assertEqual(artifact.ground_material.path, "atomic/ground.usdc")

    def test_prepare_native_builds_and_reuses_exact_400_tile_request(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _write_native_preparation_fixture(Path(directory))
            with mock.patch.object(
                terrain_pbr,
                "_derive_native_zone_terrain_tiles",
                side_effect=_fake_native_terrain_derivation,
            ) as derive:
                request = (
                    terrain_pbr.prepare_native_terrain_pbr_request(
                        volume_root=fixture["volume"],
                        zone_root=fixture["zone"],
                        scene_auto_validation_path=fixture["auto"],
                        bundle_root=fixture["bundle"],
                        material_manifest_path=fixture["manifest"],
                        artifact_root=fixture["artifacts"],
                        request_path=fixture["request"],
                    )
                )
                reused = (
                    terrain_pbr.prepare_native_terrain_pbr_request(
                        volume_root=fixture["volume"],
                        zone_root=fixture["zone"],
                        scene_auto_validation_path=fixture["auto"],
                        bundle_root=fixture["bundle"],
                        material_manifest_path=fixture["manifest"],
                        artifact_root=fixture["artifacts"],
                        request_path=fixture["request"],
                    )
                )
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    result = terrain_pbr.main(
                        [
                            "prepare-native",
                            "--volume-root",
                            str(fixture["volume"]),
                            "--zone-root",
                            str(fixture["zone"]),
                            "--scene-auto-validation",
                            str(fixture["auto"]),
                            "--bundle-root",
                            str(fixture["bundle"]),
                            "--material-manifest",
                            str(fixture["manifest"]),
                            "--artifact-root",
                            str(fixture["artifacts"]),
                            "--request-output",
                            str(fixture["request"]),
                        ]
                    )
            preparation_receipt = json.loads(
                terrain_pbr._native_preparation_receipt_path(
                    fixture["request"]
                ).read_text(encoding="utf-8")
            )
            cli_result = json.loads(stdout.getvalue())

        self.assertEqual(derive.call_count, 1)
        self.assertEqual(request, reused)
        self.assertEqual(result, 0)
        self.assertEqual(len(request["tiles"]), 400)
        self.assertEqual(
            request["scene_origin_source_m"],
            [700000.0, 6600000.0],
        )
        self.assertEqual(
            request["output_relative_path"],
            "authored/ground.usdc",
        )
        self.assertEqual(
            preparation_receipt["state"],
            terrain_pbr.NATIVE_PREPARATION_STATE,
        )
        self.assertFalse(
            preparation_receipt["global_mask_atlas_created"]
        )
        self.assertFalse(
            preparation_receipt["uniform_mask_substitution_allowed"]
        )
        self.assertEqual(cli_result["tile_count"], 400)
        self.assertEqual(
            cli_result["plan_sha256"],
            request["plan_sha256"],
        )

    def test_prepare_native_rejects_coverage_crs_semantics_and_resolution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _write_native_preparation_fixture(Path(directory))
            fixture["first_mnt"].unlink()
            with self.assertRaisesRegex(
                TerrainPbrContractError,
                "absent, unsafe or stale",
            ):
                terrain_pbr._native_preparation_inputs(
                    volume_root=fixture["volume"],
                    zone_root=fixture["zone"],
                    scene_auto_validation_path=fixture["auto"],
                )

        with tempfile.TemporaryDirectory() as directory:
            fixture = _write_native_preparation_fixture(Path(directory))
            georeference = json.loads(
                fixture["georeference"].read_text(encoding="utf-8")
            )
            georeference["crs"] = "EPSG:4326"
            fixture["georeference"].write_text(
                json.dumps(georeference),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                TerrainPbrContractError,
                "georeference is incomplete",
            ):
                terrain_pbr._native_preparation_inputs(
                    volume_root=fixture["volume"],
                    zone_root=fixture["zone"],
                    scene_auto_validation_path=fixture["auto"],
                )

        with tempfile.TemporaryDirectory() as directory:
            fixture = _write_native_preparation_fixture(Path(directory))
            source_lock = json.loads(
                fixture["source_lock"].read_text(encoding="utf-8")
            )
            del source_lock["vector_sources"]["buildings"]
            fixture["source_lock"].write_text(
                json.dumps(source_lock, sort_keys=True),
                encoding="utf-8",
            )
            build = json.loads(
                fixture["build"].read_text(encoding="utf-8")
            )
            build["source_lock"] = _record(
                fixture["source_lock"],
                root=fixture["zone"],
            )
            fixture["build"].write_text(
                json.dumps(build, sort_keys=True),
                encoding="utf-8",
            )
            auto = json.loads(
                fixture["auto"].read_text(encoding="utf-8")
            )
            auto["build_receipt_sha256"] = _sha256(fixture["build"])
            fixture["auto"].write_text(
                json.dumps(auto, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                TerrainPbrContractError,
                "lacks artificial_ground",
            ):
                terrain_pbr._native_preparation_inputs(
                    volume_root=fixture["volume"],
                    zone_root=fixture["zone"],
                    scene_auto_validation_path=fixture["auto"],
                )

        class _FourMetreDataset:
            RasterXSize = 100
            RasterYSize = 100

            @staticmethod
            def GetGeoTransform() -> tuple[float, ...]:
                return (700000.0, 4.0, 0.0, 6600400.0, 0.0, -4.0)

            @staticmethod
            def GetProjection() -> str:
                return "EPSG:2154"

        with mock.patch.object(
            terrain_pbr,
            "_gdal_is_epsg2154",
            return_value=True,
        ):
            with self.assertRaisesRegex(
                TerrainPbrContractError,
                "unsuitable metric resolution",
            ):
                terrain_pbr._gdal_raster_contract(
                    dataset=_FourMetreDataset(),
                    osr=object(),
                    label="coarse MNT",
                )

    def test_author_native_cli_consumes_locked_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = _write_locked_bundle(root / "bundle")
            evidence = root / "evidence"
            evidence.mkdir()
            tile = _tile(
                evidence_root=evidence,
                tile_id="CLI",
                bounds=Bounds2d(0.0, 0.0, 100.0, 100.0),
            )
            plan = build_terrain_pbr_plan(
                scene_id="SIM-11",
                bundle_root=bundle,
                material_manifest_path=manifest,
                evidence_root=evidence,
                tiles=(tile,),
            )
            request = root / "author-request.json"
            authored_request = (
                terrain_pbr.write_terrain_pbr_authoring_request(
                    request,
                    plan=plan,
                    bundle_root=bundle,
                    evidence_root=evidence,
                    artifact_root=root,
                    output_relative_path="cli/ground.usdc",
                    receipt_relative_path="cli/ground-receipt.json",
                )
            )
            backend = _FakeNativeBackend()
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = terrain_pbr.main(
                    ["author-native", "--request", str(request)],
                    backend=backend,
                )
            layout = json.loads(stdout.getvalue())

        self.assertEqual(result, 0)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(len(authored_request["tiles"]), 1)
        self.assertEqual(
            authored_request["plan_sha256"],
            plan.fingerprint,
        )
        self.assertEqual(layout["path"], "cli/ground.usdc")
        self.assertEqual(layout["prim_path"], "/Ground")
        self.assertEqual(
            layout["tile_material_payloads"][0]["tile_ref"],
            "CLI",
        )
        self.assertEqual(
            layout["isolated_content_roles"],
            ["object_free_pbr_ground"],
        )


if __name__ == "__main__":
    unittest.main()

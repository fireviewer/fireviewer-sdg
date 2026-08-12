from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fireviewer_sdg import sim01_quality_gate as gate


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@lru_cache(maxsize=8)
def _proof_png(index: int) -> tuple[bytes, dict[str, float | int]]:
    width = gate.PROOF_WIDTH_PX
    height = gate.PROOF_HEIGHT_PX
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    columns = 16
    rows = 12
    for row in range(rows):
        for column in range(columns):
            base = (column * 37 + row * 61 + index * 29) % 256
            colour = (
                base,
                (base * 3 + row * 17 + index * 11) % 256,
                (base * 5 + column * 23 + index * 7) % 256,
            )
            left = column * width // columns
            top = row * height // rows
            right = (column + 1) * width // columns
            bottom = (row + 1) * height // rows
            draw.rectangle((left, top, right, bottom), fill=colour)
    array = np.asarray(image, dtype=np.uint8)
    float_rgb = array.astype(np.float32)
    luminance = (
        float_rgb[:, :, 0] * 0.2126
        + float_rgb[:, :, 1] * 0.7152
        + float_rgb[:, :, 2] * 0.0722
    )
    metrics: dict[str, float | int] = {
        "width_px": width,
        "height_px": height,
        "rgb_stddev": float(float_rgb.std()),
        "luminance_stddev": float(luminance.std()),
        "dark_fraction": float((luminance <= 2.0).mean()),
        "bright_fraction": float((luminance >= 253.0).mean()),
        "edge_energy": (
            float(np.abs(np.diff(luminance, axis=1)).mean())
            + float(np.abs(np.diff(luminance, axis=0)).mean())
        )
        * 0.5,
    }
    encoded = io.BytesIO()
    image.save(encoded, format="PNG", compress_level=4)
    return encoded.getvalue(), metrics


def _record(path: Path, *, root: Path, include_size: bool = False) -> dict[str, object]:
    record: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": _digest(path),
    }
    if include_size:
        record["size_bytes"] = path.stat().st_size
    return record


def _camera(index: int) -> dict[str, object]:
    camera_id = f"VIEW-{index:02d}"
    core: dict[str, object] = {
        "camera_id": camera_id,
        "pose_local": {
            "position_m": [float(index * 25), float(index * 10), 850.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "intrinsics": {
            "model": "pinhole",
            "width_px": 3840,
            "height_px": 2160,
            "fx_px": 2600.0,
            "fy_px": 2600.0,
            "cx_px": 1920.0,
            "cy_px": 1080.0,
            "near_clip_m": 0.1,
            "far_clip_m": 40_000.0,
        },
    }
    core["camera_contract_sha256"] = _canonical_sha256(core)
    return core


def _build_fixture(root: Path) -> dict[str, Path]:
    volume = root / "volume"
    contracts = volume / "contracts"
    authored = volume / "authored"
    evidence = volume / "evidence"
    volume.mkdir(parents=True)

    runtime = contracts / "runtime-preflight.json"
    _write_json(
        runtime,
        {
            "schema_version": 1,
            "state": "SETUP_PREFLIGHT_PASSED",
            "gpu": {
                "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "driver_version": "570.158.01",
                "memory_mib": 96_000,
                "minimum_memory_mib": 90_000,
                "required_name_exact": (
                    "RTX PRO 6000 Blackwell Server Edition"
                ),
                "vulkan_summary_sha256": "1" * 64,
            },
            "system_memory": {
                "effective_mib": 142_000,
                "limit_bytes": 142_000 * 1024 * 1024,
                "minimum_effective_mib": 138_000,
                "measurement": "finite_container_cgroup_limit",
                "source": "/sys/fs/cgroup/memory.max",
                "host_proc_meminfo_used": False,
            },
            "storage": {
                "mode": "ephemeral-nvme",
                "capacity_bytes": 1_610_612_736_000,
                "minimum_capacity_gb_decimal": 1500.0,
                "free_bytes": 1_200_000_000_000,
                "automatic_stop_allowed": False,
                "durability": "ephemeral_until_explicit_pod_termination",
            },
            "proof_boundary": (
                "hardware and built runtime only; no human scene acceptance"
            ),
        },
    )

    authoring_plan = volume / "planning" / "campaign-plan.json"
    _write_json(
        authoring_plan,
        {
            "state": "VARIANT_CAMPAIGN_PLANNED",
            "simulation_count": 20,
        },
    )
    variants: list[dict[str, object]] = []
    scene_paths: dict[str, tuple[Path, Path]] = {}
    for index in range(1, 21):
        simulation_id = f"SIM-{index:02d}"
        scene_root = authored / simulation_id
        build_dir = scene_root / "build"
        root_usd = build_dir / "root.usdc"
        root_usd.parent.mkdir(parents=True, exist_ok=True)
        root_usd.write_bytes(
            f"PXR-USDC:{simulation_id}:native-full-scene\n".encode("ascii")
        )
        build_receipt = build_dir / "build-receipt.json"
        build_payload = {
            "schema_version": 2,
            "zone_id": simulation_id,
            "variant_id": f"variant-{index:02d}",
            "base_scene_id": f"BASE-{((index - 1) // 5) + 1:02d}",
            "variant_index": ((index - 1) % 5) + 1,
            "scene_kind": "fictive_variant",
            "source_profile": "full",
            "root_usd": {
                "path": root_usd.relative_to(scene_root).as_posix(),
                "sha256": _digest(root_usd),
            },
            "layers": {
                "terrain": {
                    "prim_count": 400,
                    "ground_material_payload_count": 400,
                    "global_ground_material_binding": False,
                },
                "vegetation": {"prim_count": 1_000},
                "buildings": {"prim_count": 100},
                "roads": {
                    "prim_count": 80,
                    "source_feature_count": 12,
                },
                "hydrology": {
                    "prim_count": 24,
                    "source_feature_count": 4,
                },
                "collisions": {
                    "prim_count": 400,
                    "levels": ["NEAR", "FAR"],
                },
            },
            "fire_simulation_status": "blocked_pending_editor_review",
        }
        _write_json(build_receipt, build_payload)
        scene_paths[simulation_id] = (root_usd, build_receipt)
        variants.append(
            {
                "simulation_id": simulation_id,
                "variant_id": f"variant-{index:02d}",
                "base_scene_id": f"BASE-{((index - 1) // 5) + 1:02d}",
                "variant_index": ((index - 1) % 5) + 1,
                "scene_kind": "fictive_variant",
                "artifacts": {
                    "root_usd": _record(root_usd, root=scene_root),
                    "composer_build_receipt": _record(
                        build_receipt,
                        root=scene_root,
                    ),
                    "scene_kind": "fictive_variant",
                    "streaming_tile_count": 400,
                    "object_lod_payload_count": 1200,
                    "monolithic_object_payloads": False,
                    "tile_coverage": [
                        {"tile_ref": f"TILE-{tile:03d}"}
                        for tile in range(400)
                    ],
                },
                "fire_simulation_status": "blocked_pending_editor_review",
            }
        )
    authoring = authored / "authoring-receipt.json"
    _write_json(
        authoring,
        {
            "schema_version": 1,
            "state": "VARIANT_USD_AUTHORED",
            "plan": _record(authoring_plan, root=volume),
            "simulation_count": 20,
            "variants": variants,
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
        },
    )

    verification = contracts / "variant-campaign-verification.json"
    _write_json(
        verification,
        {
            "state": "VARIANT_CAMPAIGN_VERIFIED",
            "plan_sha256": _digest(authoring_plan),
            "authoring_receipt_sha256": _digest(authoring),
            "layout_count": 4,
            "simulation_count": 20,
            "root_usd_rehashed": 20,
            "build_receipts_rehashed": 20,
            "terrain_payload_references_verified": 8_000,
            "terrain_payload_unique_files_rehashed": 1_600,
            "object_lod_payloads_rehashed": 24_000,
            "ground_material_references_verified": 8_000,
            "ground_material_unique_files_rehashed": 1_600,
            "water_payload_references_verified": 100,
            "water_payload_unique_files_rehashed": 20,
            "identity_contracts_verified": 20,
            "hash_operations": 27_000,
            "bytes_hashed": 2_000_000,
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
        },
    )

    sim01_root, sim01_build = scene_paths["SIM-01"]
    auto_validation = authored / "SIM-01" / "scene-auto-validation.json"
    _write_json(
        auto_validation,
        {
            "schema_version": 2,
            "state": "AUTO_VALIDATED",
            "scene_kind": "fictive_variant",
            "validation_scope": (
                "incremental_structural_geometric_gate_not_human_review"
            ),
            "root_usd": str(sim01_root),
            "root_usd_sha256": _digest(sim01_root),
            "build_receipt_sha256": _digest(sim01_build),
            "asset_manifest_sha256": "2" * 64,
            "fire_simulation_status": "blocked_pending_editor_review",
            "streaming": {
                "root_initial_load_set": "LoadNone",
                "terrain_payloads_inspected_incrementally": 400,
                "detail_payloads_inspected_incrementally": 400,
                "simultaneously_retained_detail_stages": 1,
                "root_terrain_payload_arcs": 400,
                "root_detail_payload_arcs": 400,
                "root_ground_material_payload_arcs": 400,
            },
            "terrain": {
                "payload_count": 400,
                "lod0_tile_count": 25,
                "tiles": [
                    {"tile_ref": f"TILE-{index:03d}"}
                    for index in range(400)
                ],
            },
            "details": {
                "payload_count": 400,
                "expected_totals": {
                    "vegetation": 1_000,
                    "buildings": 100,
                },
                "tiles": [
                    {"tile_ref": f"TILE-{index:03d}"}
                    for index in range(400)
                ],
            },
            "vegetation_instances": 1_000,
            "vegetation_family_instances": {
                "trees": 1_000,
                "shrubs": 0,
                "understory": 0,
            },
            "building_instances": 100,
            "forest_world_bounds": {
                "minimum": [-10_000.0, -10_000.0, 0.0],
                "maximum": [10_000.0, 10_000.0, 700.0],
                "span_metres": [20_000.0, 20_000.0, 700.0],
            },
            "forest_near_origin_instances": 0,
            "forest_unique_xy": 1_000,
            "spatial_signature": "3" * 64,
            "used_layers": [
                {
                    "path": str(sim01_root),
                    "sha256": _digest(sim01_root),
                }
            ],
        },
    )

    cameras = [_camera(index) for index in range(1, 41)]
    review_camera_plan = evidence / "sim01-review-camera-plan.json"
    review_payload: dict[str, object] = {
        "schema_version": 1,
        "state": gate.REVIEW_CAMERA_PLAN_STATE,
        "simulation_id": "SIM-01",
        "root_usd_sha256": _digest(sim01_root),
        "build_receipt_sha256": _digest(sim01_build),
        "scene_auto_validation_sha256": _digest(auto_validation),
        "camera_count": 40,
        "cameras": cameras,
        "camera_checks": [
            {
                "camera_id": camera["camera_id"],
                "camera_contract_sha256": camera[
                    "camera_contract_sha256"
                ],
                "status": "passed",
                "covered_tile_count": 12,
                "minimum_terrain_clearance_m": 40.0,
                "permanent_occlusion_fraction": 0.05,
                "inside_extent": True,
                "projection_finite": True,
            }
            for camera in cameras
        ],
        "coverage_gate": {
            "status": "passed",
            "covered_tile_count": 400,
            "occluded_view_count": 0,
            "below_terrain_view_count": 0,
            "out_of_bounds_view_count": 0,
            "duplicate_view_count": 0,
            "non_finite_projection_count": 0,
        },
        "fire_simulation_status": "blocked_pending_editor_review",
        "simulation_execution_performed": False,
        "render_execution_performed": False,
    }
    review_payload["plan_sha256"] = _canonical_sha256(review_payload)
    _write_json(review_camera_plan, review_payload)
    review_plan_sha = str(review_payload["plan_sha256"])

    proof_pack = evidence / "proof" / "proof-pack.json"
    proof_renders: list[dict[str, object]] = []
    proof_visual_metrics: list[dict[str, object]] = []
    roles = (
        "vertical",
        "low",
        "inclined",
        "oblique",
        "vertical",
        "low",
        "inclined",
        "oblique",
    )
    for index, role in enumerate(roles, start=1):
        render_id = f"PROOF-{index:02d}"
        camera = cameras[index - 1]
        image = proof_pack.parent / "images" / f"{render_id}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        encoded_image, visual_metrics = _proof_png(index)
        image.write_bytes(encoded_image)
        metadata = proof_pack.parent / "metadata" / f"{render_id}.json"
        _write_json(
            metadata,
            {
                "state": "NATIVE_PROOF_RENDER_CAPTURED",
                "simulation_id": "SIM-01",
                "render_id": render_id,
                "camera_id": camera["camera_id"],
                "view_role": role,
                "camera_contract_sha256": camera[
                    "camera_contract_sha256"
                ],
                "root_usd_sha256": _digest(sim01_root),
                "build_receipt_sha256": _digest(sim01_build),
                "review_camera_plan_sha256": review_plan_sha,
                "image_sha256": _digest(image),
                "width_px": 3840,
                "height_px": 2160,
                "renderer_backend": "kit_rtx_native",
                "visual_metrics": visual_metrics,
                "fire_simulation_status": "blocked_pending_editor_review",
                "timeline_advanced": False,
            },
        )
        image_record = _record(
            image,
            root=proof_pack.parent,
            include_size=True,
        )
        image_record.update({"width_px": 3840, "height_px": 2160})
        proof_renders.append(
            {
                "render_id": render_id,
                "camera_id": camera["camera_id"],
                "view_role": role,
                "camera_contract_sha256": camera[
                    "camera_contract_sha256"
                ],
                "image": image_record,
                "metadata": _record(
                    metadata,
                    root=proof_pack.parent,
                    include_size=True,
                ),
            }
        )
        proof_visual_metrics.append(
            {
                "render_id": render_id,
                **visual_metrics,
            }
        )
    _write_json(
        proof_pack,
        {
            "schema_version": 1,
            "state": gate.PROOF_PACK_STATE,
            "simulation_id": "SIM-01",
            "root_usd_sha256": _digest(sim01_root),
            "build_receipt_sha256": _digest(sim01_build),
            "review_camera_plan_sha256": review_plan_sha,
            "renderer": {
                "backend": "kit_rtx_native",
                "native_render": True,
                "screen_capture": False,
                "execution_mode": "headless_native_qa",
                "render_mode": "RayTracedLighting",
                "resolution_px": [3840, 2160],
                "rt_subframes": 16,
            },
            "inspection_decision": "passed",
            "inspection_scope": "internal_visual_qa",
            "human_editor_validation": {
                "state": "required_before_fire_simulation",
                "performed": False,
                "required": True,
            },
            "render_count": 8,
            "renders": proof_renders,
            "visual_metrics": proof_visual_metrics,
            "fire_simulation_status": "blocked_pending_editor_review",
            "simulation_execution_performed": False,
        },
    )

    quality = evidence / "sim01-quality-report.json"
    _write_json(
        quality,
        {
            "schema_version": 1,
            "state": gate.QUALITY_REPORT_STATE,
            "simulation_id": "SIM-01",
            "root_usd_sha256": _digest(sim01_root),
            "build_receipt_sha256": _digest(sim01_build),
            "scene_auto_validation_sha256": _digest(auto_validation),
            "status": "passed",
            "defect_count": 0,
            "checks": {
                "structure": {
                    "status": "passed",
                    "terrain_tile_count": 400,
                    "hero_payload_count": 400,
                    "mid_payload_count": 400,
                    "far_payload_count": 400,
                    "collision_tile_count": 400,
                    "forbidden_primitive_count": 0,
                    "placeholder_count": 0,
                    "empty_zone_count": 0,
                    "invalid_reference_count": 0,
                },
                "density": {
                    "status": "passed",
                    "occupied_terrain_tile_count": 400,
                    "vegetation_instance_count": 1_000,
                    "building_instance_count": 100,
                    "empty_tile_count": 0,
                    "failed_habitat_placement_count": 0,
                    "origin_pile_count": 0,
                },
                "lod": {
                    "status": "passed",
                    "terrain_payload_count": 400,
                    "terrain_lod0_tile_count": 25,
                    "hero_payload_count": 400,
                    "mid_payload_count": 400,
                    "far_payload_count": 400,
                    "collision_levels": ["NEAR", "FAR"],
                    "missing_transition_count": 0,
                    "missing_collision_count": 0,
                },
                "pbr": {
                    "status": "passed",
                    "ground_material_payload_count": 400,
                    "object_free_ground_tile_count": 400,
                    "unbound_material_count": 0,
                    "global_aggregate_material_count": 0,
                    "phantom_imagery_count": 0,
                    "material_discontinuity_count": 0,
                },
                "topology": {
                    "status": "passed",
                    "route_source_count": 12,
                    "route_fragment_count": 80,
                    "hydrology_source_count": 4,
                    "hydrology_fragment_count": 24,
                    "disconnected_route_count": 0,
                    "invalid_bridge_count": 0,
                    "water_overlap_count": 0,
                    "topology_violation_count": 0,
                },
                "native_visual": {
                    "status": "passed",
                    "render_count": 8,
                    "distinct_image_count": 8,
                    "minimum_rgb_stddev": min(
                        float(item["rgb_stddev"])
                        for item in proof_visual_metrics
                    ),
                    "minimum_edge_energy": min(
                        float(item["edge_energy"])
                        for item in proof_visual_metrics
                    ),
                    "renderer_backend": "kit_rtx_native",
                },
            },
        },
    )

    stability = evidence / "sim01-stability-report.json"
    _write_json(
        stability,
        {
            "schema_version": 1,
            "state": gate.STABILITY_REPORT_STATE,
            "simulation_id": "SIM-01",
            "root_usd_sha256": _digest(sim01_root),
            "build_receipt_sha256": _digest(sim01_build),
            "runtime_preflight_sha256": _digest(runtime),
            "workflow": "headless_native_qa",
            "execution_mode": "headless_native_qa",
            "status": "passed",
            "human_editor_validation": {
                "state": "required_before_fire_simulation",
                "performed": False,
                "required": True,
            },
            "stage_open_attempt_count": 1,
            "successful_stage_open_count": 1,
            "failed_stage_open_count": 0,
            "duration_seconds": 600.0,
            "stage_open_seconds": 22.0,
            "payload_settle_seconds": 18.0,
            "fps": {
                "status": "passed",
                "measurement_scope": (
                    "headless_kit_with_live_3840x2160_render_product"
                ),
                "camera_id": "VIEW-01",
                "render_product_resolution_px": [3840, 2160],
                "acceptance_threshold_fps": 30.0,
                "observed_minimum_fps": 38.0,
                "observed_mean_fps": 57.0,
                "sample_count": 1_800,
            },
            "vram": {
                "status": "passed",
                "measurement_scope": "live_4k_render_product",
                "total_mib": 96_000,
                "peak_mib": 72_000.0,
                "sample_count": 600,
                "oom_count": 0,
            },
            "ram": {
                "status": "passed",
                "measurement_scope": "live_4k_render_product",
                "cgroup_limit_mib": 142_000,
                "peak_mib": 118_000.0,
                "sample_count": 600,
                "oom_count": 0,
            },
            "crash_count": 0,
            "hang_count": 0,
            "device_lost_count": 0,
            "fatal_error_count": 0,
        },
    )

    return {
        "volume": volume,
        "runtime": runtime,
        "authoring": authoring,
        "verification": verification,
        "auto": auto_validation,
        "review_plan": review_camera_plan,
        "proof": proof_pack,
        "quality": quality,
        "stability": stability,
        "output": evidence / "SIM01_INTERNAL_QA_PASSED.json",
    }


class Sim01QualityGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.fixture = _build_fixture(Path(cls._temporary.name))
        cls._originals = {
            path: path.read_bytes()
            for path in cls.fixture["volume"].rglob("*")
            if path.is_file()
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def setUp(self) -> None:
        for path, content in self._originals.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.fixture["output"].unlink(missing_ok=True)

    def _evaluate(self) -> dict[str, object]:
        return gate.evaluate_sim01_internal_qa(
            volume_root=self.fixture["volume"],
            runtime_preflight_path=self.fixture["runtime"],
            authoring_receipt_path=self.fixture["authoring"],
            campaign_verification_path=self.fixture["verification"],
            scene_auto_validation_path=self.fixture["auto"],
            review_camera_plan_path=self.fixture["review_plan"],
            proof_pack_path=self.fixture["proof"],
            quality_report_path=self.fixture["quality"],
            stability_report_path=self.fixture["stability"],
            output_path=self.fixture["output"],
        )

    def _json(self, name: str) -> dict[str, object]:
        return json.loads(self.fixture[name].read_text(encoding="utf-8"))

    def _write_fixture_json(self, name: str, payload: object) -> None:
        _write_json(self.fixture[name], payload)

    def _replace_proof_image(self, index: int, content: bytes) -> None:
        proof = self._json("proof")
        render = proof["renders"][index]
        image_path = self.fixture["proof"].parent / render["image"]["path"]
        image_path.write_bytes(content)
        render["image"]["sha256"] = _digest(image_path)
        render["image"]["size_bytes"] = image_path.stat().st_size

        metadata_path = (
            self.fixture["proof"].parent / render["metadata"]["path"]
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["image_sha256"] = _digest(image_path)
        _write_json(metadata_path, metadata)
        render["metadata"] = _record(
            metadata_path,
            root=self.fixture["proof"].parent,
            include_size=True,
        )
        self._write_fixture_json("proof", proof)

    def _assert_gate_fails(self, pattern: str) -> None:
        with self.assertRaisesRegex(gate.Sim01QualityGateError, pattern):
            self._evaluate()
        self.assertFalse(self.fixture["output"].exists())

    def test_complete_evidence_writes_only_the_internal_pass_receipt(self) -> None:
        receipt = self._evaluate()

        self.assertEqual(receipt["state"], gate.INTERNAL_QA_STATE)
        self.assertTrue(receipt["review_handoff_ready"])
        self.assertEqual(
            receipt["fire_simulation_status"],
            "blocked_pending_editor_review",
        )
        self.assertFalse(receipt["rendering_performed_by_gate"])
        self.assertFalse(receipt["simulation_performed_by_gate"])
        self.assertEqual(receipt["counts"]["capture_cameras"], 40)
        self.assertEqual(receipt["counts"]["proof_renders"], 8)
        self.assertTrue(self.fixture["output"].is_file())

    def test_pre_review_camera_plan_has_no_post_acceptance_dependency(self) -> None:
        plan = self._json("review_plan")
        serialized = json.dumps(plan, sort_keys=True)
        self.assertNotIn("simulation_allowed", serialized)
        self.assertNotIn("editor_acceptance", serialized)

        plan["simulation_allowed_receipt"] = {
            "path": "contracts/simulation-allowed.json",
            "sha256": "9" * 64,
            "size_bytes": 10,
        }
        plan_without_hash = dict(plan)
        plan_without_hash.pop("plan_sha256")
        plan["plan_sha256"] = _canonical_sha256(plan_without_hash)
        self._write_fixture_json("review_plan", plan)

        self._assert_gate_fails("forbidden post-review dependency")

    def test_any_missing_or_sentinel_decision_fails_closed(self) -> None:
        proof = self._json("proof")
        proof["inspection_decision"] = "not_reviewed"
        self._write_fixture_json("proof", proof)
        self._assert_gate_fails("forbidden evidence sentinel")

        self.setUp()
        quality = self._json("quality")
        del quality["checks"]["pbr"]
        self._write_fixture_json("quality", quality)
        self._assert_gate_fails("missing the pbr section")

    def test_nominal_150_gb_pod_uses_its_finite_cgroup_limit(self) -> None:
        limit_bytes = 150_000_000_000
        effective_mib = limit_bytes // (1024 * 1024)
        self.assertEqual(effective_mib, 143_051)

        runtime = self._json("runtime")
        runtime["system_memory"].update(
            {
                "effective_mib": effective_mib,
                "limit_bytes": limit_bytes,
            }
        )
        self._write_fixture_json("runtime", runtime)

        stability = self._json("stability")
        stability["runtime_preflight_sha256"] = _digest(
            self.fixture["runtime"]
        )
        stability["ram"]["cgroup_limit_mib"] = effective_mib
        self._write_fixture_json("stability", stability)

        receipt = self._evaluate()
        self.assertEqual(receipt["state"], gate.INTERNAL_QA_STATE)
        self.assertEqual(
            receipt["stability"]["peak_ram_mib"],
            118_000.0,
        )

    def test_runtime_requires_exact_rtx_finite_cgroup_and_nvme(self) -> None:
        base = self._json("runtime")
        mutations = (
            (
                "GPU",
                lambda value: value["gpu"].update(
                    {"name": "NVIDIA A100-SXM4-80GB"}
                ),
                "runtime GPU",
            ),
            (
                "cgroup",
                lambda value: value["system_memory"].update(
                    {"measurement": "host_proc_meminfo"}
                ),
                "finite container cgroup",
            ),
            (
                "NVMe",
                lambda value: value["storage"].update(
                    {"mode": "persistent-volume"}
                ),
                "ephemeral NVMe",
            ),
        )
        for label, mutate, pattern in mutations:
            with self.subTest(label=label):
                payload = copy.deepcopy(base)
                mutate(payload)
                self._write_fixture_json("runtime", payload)
                self._assert_gate_fails(pattern)
                self.setUp()

    def test_authoring_and_verification_must_cover_all_twenty_scenes(self) -> None:
        authoring = self._json("authoring")
        authoring["simulation_count"] = 19
        self._write_fixture_json("authoring", authoring)
        self._assert_gate_fails("20-scene result")

        self.setUp()
        verification = self._json("verification")
        verification["object_lod_payloads_rehashed"] = 23_999
        self._write_fixture_json("verification", verification)
        self._assert_gate_fails(
            "object_lod_payloads_rehashed must equal 24000"
        )

    def test_auto_validation_and_camera_coverage_are_exact(self) -> None:
        auto = self._json("auto")
        auto["streaming"]["detail_payloads_inspected_incrementally"] = 399
        self._write_fixture_json("auto", auto)
        self._assert_gate_fails(
            "detail_payloads_inspected_incrementally must equal 400"
        )

        self.setUp()
        plan = self._json("review_plan")
        plan["cameras"].pop()
        plan["camera_count"] = 39
        core = dict(plan)
        core.pop("plan_sha256")
        plan["plan_sha256"] = _canonical_sha256(core)
        self._write_fixture_json("review_plan", plan)
        self._assert_gate_fails("autonomous SIM-01 plan")

        self.setUp()
        plan = self._json("review_plan")
        plan["coverage_gate"]["below_terrain_view_count"] = 1
        core = dict(plan)
        core.pop("plan_sha256")
        plan["plan_sha256"] = _canonical_sha256(core)
        self._write_fixture_json("review_plan", plan)
        self._assert_gate_fails("below_terrain_view_count must be zero")

        self.setUp()
        plan = self._json("review_plan")
        plan["camera_checks"][0]["projection_finite"] = False
        core = dict(plan)
        core.pop("plan_sha256")
        plan["plan_sha256"] = _canonical_sha256(core)
        self._write_fixture_json("review_plan", plan)
        self._assert_gate_fails("per-camera QA binding is stale")

        self.setUp()
        plan = self._json("review_plan")
        plan["scene_auto_validation_sha256"] = "f" * 64
        core = dict(plan)
        core.pop("plan_sha256")
        plan["plan_sha256"] = _canonical_sha256(core)
        self._write_fixture_json("review_plan", plan)
        self._assert_gate_fails("autonomous SIM-01 plan")

    def test_proof_pack_requires_eight_distinct_nonempty_native_renders(self) -> None:
        proof = self._json("proof")
        proof["renders"].pop()
        proof["render_count"] = 7
        self._write_fixture_json("proof", proof)
        self._assert_gate_fails("proof pack is stale or incomplete")

        self.setUp()
        proof = self._json("proof")
        first = proof["renders"][0]
        second = proof["renders"][1]
        second["camera_id"] = first["camera_id"]
        second["camera_contract_sha256"] = first[
            "camera_contract_sha256"
        ]
        metadata_path = (
            self.fixture["proof"].parent / second["metadata"]["path"]
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["camera_id"] = first["camera_id"]
        metadata["camera_contract_sha256"] = first[
            "camera_contract_sha256"
        ]
        _write_json(metadata_path, metadata)
        second["metadata"] = _record(
            metadata_path,
            root=self.fixture["proof"].parent,
            include_size=True,
        )
        self._write_fixture_json("proof", proof)
        self._assert_gate_fails("not distinct")

        self.setUp()
        proof = self._json("proof")
        image_path = (
            self.fixture["proof"].parent
            / proof["renders"][0]["image"]["path"]
        )
        image_path.write_bytes(b"")
        self._assert_gate_fails("is empty")

        self.setUp()
        self._replace_proof_image(
            0,
            b"\x89PNG\r\n\x1a\n" + b"not-a-real-png" * 256,
        )
        self._assert_gate_fails("not a decodable native PNG render")

        self.setUp()
        blank = io.BytesIO()
        Image.new(
            "RGB",
            (gate.PROOF_WIDTH_PX, gate.PROOF_HEIGHT_PX),
            (127, 127, 127),
        ).save(blank, format="PNG", compress_level=4)
        self._replace_proof_image(0, blank.getvalue())
        self._assert_gate_fails("blank, clipped, uniform")

        self.setUp()
        source, _ = _proof_png(1)
        wrong_size = io.BytesIO()
        with Image.open(io.BytesIO(source)) as image:
            image.resize((1920, 1080)).save(
                wrong_size,
                format="PNG",
                compress_level=4,
            )
        self._replace_proof_image(0, wrong_size.getvalue())
        self._assert_gate_fails("must decode as a 3840x2160 RGB PNG")

        self.setUp()
        proof = self._json("proof")
        proof["visual_metrics"][0]["edge_energy"] += 1.0
        self._write_fixture_json("proof", proof)
        self._assert_gate_fails("differs from the decoded image")

    def test_quality_report_rejects_placeholders_empty_zones_and_bad_topology(
        self,
    ) -> None:
        base = self._json("quality")
        mutations = (
            (
                "primitive",
                lambda value: value["checks"]["structure"].update(
                    {"forbidden_primitive_count": 1}
                ),
                "forbidden_primitive_count must equal 0",
            ),
            (
                "placeholder",
                lambda value: value["checks"]["structure"].update(
                    {"placeholder_count": 1}
                ),
                "placeholder_count must equal 0",
            ),
            (
                "empty",
                lambda value: value["checks"]["structure"].update(
                    {"empty_zone_count": 1}
                ),
                "empty_zone_count must equal 0",
            ),
            (
                "topology",
                lambda value: value["checks"]["topology"].update(
                    {"disconnected_route_count": 1}
                ),
                "disconnected_route_count must be zero",
            ),
            (
                "native visual",
                lambda value: value["checks"]["native_visual"].update(
                    {"minimum_edge_energy": 999.0}
                ),
                "differs from proof images",
            ),
        )
        for label, mutate, pattern in mutations:
            with self.subTest(label=label):
                payload = copy.deepcopy(base)
                mutate(payload)
                self._write_fixture_json("quality", payload)
                self._assert_gate_fails(pattern)
                self.setUp()

    def test_stability_requires_headless_4k_fps_memory_and_no_crash(self) -> None:
        base = self._json("stability")
        mutations = (
            (
                "stage open",
                lambda value: value.update(
                    {"successful_stage_open_count": 0}
                ),
                "stage open attempts",
            ),
            (
                "fabricated human review",
                lambda value: value["human_editor_validation"].update(
                    {"performed": True}
                ),
                "fabricated a human Editor decision",
            ),
            (
                "fps",
                lambda value: value["fps"].update(
                    {"observed_minimum_fps": 20.0}
                ),
                "measured FPS",
            ),
            (
                "weak FPS contract",
                lambda value: value["fps"].update(
                    {"acceptance_threshold_fps": 1.0}
                ),
                "live 4K product",
            ),
            (
                "wrong FPS scope",
                lambda value: value["fps"].update(
                    {"render_product_resolution_px": [1920, 1080]}
                ),
                "live 4K product",
            ),
            (
                "VRAM",
                lambda value: value["vram"].update({"peak_mib": 96_000.0}),
                "VRAM measurement",
            ),
            (
                "RAM",
                lambda value: value["ram"].update({"peak_mib": 142_000.0}),
                "RAM measurement",
            ),
            (
                "crash",
                lambda value: value.update({"crash_count": 1}),
                "crash_count must be zero",
            ),
            (
                "runtime binding",
                lambda value: value.update(
                    {"runtime_preflight_sha256": "f" * 64}
                ),
                "stability report is stale",
            ),
        )
        for label, mutate, pattern in mutations:
            with self.subTest(label=label):
                payload = copy.deepcopy(base)
                mutate(payload)
                self._write_fixture_json("stability", payload)
                self._assert_gate_fails(pattern)
                self.setUp()

    def test_existing_output_is_reverified_without_overwrite(self) -> None:
        first = self._evaluate()
        original = self.fixture["output"].read_bytes()
        self.assertEqual(first["state"], gate.INTERNAL_QA_STATE)

        second = self._evaluate()
        self.assertEqual(second, first)
        self.assertEqual(self.fixture["output"].read_bytes(), original)

        quality = self._json("quality")
        quality["checks"]["structure"]["placeholder_count"] = 1
        self._write_fixture_json("quality", quality)
        with self.assertRaisesRegex(
            gate.Sim01QualityGateError,
            "placeholder_count must equal 0",
        ):
            self._evaluate()
        self.assertEqual(self.fixture["output"].read_bytes(), original)

        self.fixture["quality"].write_bytes(
            self._originals[self.fixture["quality"]]
        )
        tampered = json.loads(original.decode("utf-8"))
        tampered["review_handoff_ready"] = False
        _write_json(self.fixture["output"], tampered)
        tampered_bytes = self.fixture["output"].read_bytes()
        with self.assertRaisesRegex(
            gate.Sim01QualityGateError,
            "receipt is stale",
        ):
            self._evaluate()
        self.assertEqual(self.fixture["output"].read_bytes(), tampered_bytes)

    def test_cli_uses_the_autonomous_review_camera_plan(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = gate.main(
                [
                    "--volume-root",
                    str(self.fixture["volume"]),
                    "--runtime-preflight",
                    str(self.fixture["runtime"]),
                    "--authoring-receipt",
                    str(self.fixture["authoring"]),
                    "--campaign-verification",
                    str(self.fixture["verification"]),
                    "--scene-auto-validation",
                    str(self.fixture["auto"]),
                    "--review-camera-plan",
                    str(self.fixture["review_plan"]),
                    "--proof-pack",
                    str(self.fixture["proof"]),
                    "--quality-report",
                    str(self.fixture["quality"]),
                    "--stability-report",
                    str(self.fixture["stability"]),
                    "--output",
                    str(self.fixture["output"]),
                ]
            )

        self.assertEqual(status, 0)
        self.assertEqual(
            json.loads(stdout.getvalue())["state"],
            gate.INTERNAL_QA_STATE,
        )


if __name__ == "__main__":
    unittest.main()

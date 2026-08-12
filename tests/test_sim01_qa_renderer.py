from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from fireviewer_sdg import sim01_qa_renderer as renderer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tiles() -> tuple[renderer.Tile, ...]:
    values: list[renderer.Tile] = []
    for index in range(400):
        column = index % 20
        row = index // 20
        values.append(
            renderer.Tile(
                index=index,
                tile_ref=f"TILE-{index:03d}",
                min_x=column * 1_000.0,
                min_y=row * 1_000.0,
                max_x=(column + 1) * 1_000.0,
                max_y=(row + 1) * 1_000.0,
                maximum_z=300.0 + column * 2.0 + row * 3.0,
            )
        )
    return tuple(values)


def _runtime() -> dict[str, object]:
    return {
        "state": "SETUP_PREFLIGHT_PASSED",
        "gpu": {
            "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "memory_mib": 96_000,
        },
        "system_memory": {
            "effective_mib": 142_000,
            "measurement": "finite_container_cgroup_limit",
            "source": "/sys/fs/cgroup/memory.max",
            "host_proc_meminfo_used": False,
        },
        "storage": {
            "mode": "ephemeral-nvme",
            "capacity_bytes": 1_610_612_736_000,
            "automatic_stop_allowed": False,
        },
    }


def _build(tiles: tuple[renderer.Tile, ...]) -> dict[str, object]:
    coverage = [
        {
            "tile_ref": tile.tile_ref,
            "local_bounds": {
                "min_x": tile.min_x,
                "min_y": tile.min_y,
                "max_x": tile.max_x,
                "max_y": tile.max_y,
            },
            "detail_counts": {
                "trees": 1,
                "buildings": 1 if tile.index % 8 == 0 else 0,
            },
        }
        for tile in tiles
    ]
    artifacts = [
        {"path": f"payload-{index:04d}.usdc", "sha256": "1" * 64}
        for index in range(400)
    ]
    return {
        "schema_version": 2,
        "zone_id": "SIM-01",
        "scene_kind": "fictive_variant",
        "source_profile": "full",
        "fire_simulation_status": "blocked_pending_editor_review",
        "payloads": artifacts,
        "detail_payloads": artifacts,
        "detail_mid_payloads": artifacts,
        "detail_far_payloads": artifacts,
        "tile_coverage": coverage,
        "ground_material": {
            "tile_material_payloads": artifacts,
        },
        "route_topology": {
            "exact_membership_preserved": True,
            "source_component_count": 3,
            "result_component_count": 3,
            "source_membership_sha256": "a" * 64,
            "result_membership_sha256": "a" * 64,
        },
        "layers": {
            "terrain": {
                "prim_count": 400,
                "ground_material_payload_count": 400,
            },
            "vegetation": {"prim_count": 1_000},
            "buildings": {"prim_count": 100},
            "roads": {"prim_count": 80, "source_feature_count": 12},
            "hydrology": {"prim_count": 24, "source_feature_count": 4},
            "collisions": {"prim_count": 400, "levels": ["NEAR", "FAR"]},
            "detail_streaming": {"prim_count": 400},
        },
    }


def _inputs(root: Path) -> renderer.Inputs:
    tiles = _tiles()
    runtime_path = root / "runtime.json"
    root_usd = root / "SIM-01" / "build" / "root.usdc"
    build_path = root / "SIM-01" / "build" / "build-receipt.json"
    auto_path = root / "SIM-01" / "scene-auto-validation.json"
    root_usd.parent.mkdir(parents=True, exist_ok=True)
    root_usd.write_bytes(b"PXR-USDC-real-SIM-01")
    _write_json(runtime_path, _runtime())
    build = _build(tiles)
    _write_json(build_path, build)
    auto = {
        "schema_version": 2,
        "state": "AUTO_VALIDATED",
        "scene_kind": "fictive_variant",
        "fire_simulation_status": "blocked_pending_editor_review",
        "vegetation_instances": 1_000,
        "building_instances": 100,
        "terrain": {"lod0_tile_count": 25},
    }
    _write_json(auto_path, auto)
    return renderer.Inputs(
        volume_root=root,
        runtime_preflight_path=runtime_path,
        root_usd_path=root_usd,
        build_receipt_path=build_path,
        scene_auto_validation_path=auto_path,
        runtime=_runtime(),
        build=build,
        auto_validation=auto,
        root_usd_sha256=_sha(root_usd),
        build_receipt_sha256=_sha(build_path),
        scene_auto_validation_sha256=_sha(auto_path),
        scene_root=build_path.parent.parent,
        tiles=tiles,
    )


def _streaming_snapshot(index: int) -> dict[str, object]:
    return {
        "state": "EXCLUSIVE_CAMERA_WORKING_SET_SETTLED",
        "planner_source": renderer.STREAMING_PLANNER_SOURCE,
        "transition_semantics_source": renderer.STREAMING_TRANSITION_SOURCE,
        "transition_mode": "headless_unload_then_load_no_overlap",
        "visible_tile_count": 120,
        "terrain_lod_counts": {"LOD0": 12, "LOD1": 52, "LOD2": 132, "LOD3": 204},
        "collision_lod_counts": {"NEAR": 148, "FAR": 252},
        "transition_counts": (
            {"UNLOADED_to_HERO": 48, "UNLOADED_to_MID": 100, "UNLOADED_to_FAR": 252}
            if index == 0
            else {"HERO_to_MID": 12, "MID_to_HERO": 12}
        ),
        "loaded_detail_payload_count": 400,
        "duplicate_detail_tile_count": 0,
        "unloaded_detail_tile_count": 0,
        "detail_level_counts": {"HERO": 48, "MID": 100, "FAR": 252},
        "active_detail_levels_sha256": "a" * 64,
        "camera_id": f"VIEW-{index + 1:02d}",
        "camera_contract_sha256": f"{index + 1:064x}",
    }


class Sim01QaRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()
        self.inputs = _inputs(self.root)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_camera_plan_has_exact_real_coverage_and_current_hashes(self) -> None:
        plan = renderer.generate_camera_plan(self.inputs, self.inputs.tiles)

        renderer._validate_plan(plan, inputs=self.inputs)
        self.assertEqual(plan["state"], renderer.REVIEW_CAMERA_PLAN_STATE)
        self.assertEqual(plan["camera_count"], 40)
        self.assertEqual(
            [camera["camera_id"] for camera in plan["cameras"]],
            list(renderer.CAMERA_IDS),
        )
        self.assertEqual(len(plan["camera_checks"]), 40)
        self.assertEqual(plan["coverage_gate"]["covered_tile_count"], 400)
        self.assertEqual(
            plan["scene_auto_validation_sha256"],
            self.inputs.scene_auto_validation_sha256,
        )
        self.assertFalse(plan["simulation_execution_performed"])
        self.assertFalse(plan["render_execution_performed"])
        self.assertEqual(
            len(
                {
                    camera["camera_contract_sha256"]
                    for camera in plan["cameras"]
                }
            ),
            40,
        )
        self.assertTrue(
            all(check["inside_extent"] for check in plan["camera_checks"])
        )
        self.assertTrue(
            all(
                check["minimum_terrain_clearance_m"] > 0
                and check["permanent_occlusion_fraction"] < 1.0
                for check in plan["camera_checks"]
            )
        )

    def test_plan_refuses_an_empty_camera_cell(self) -> None:
        sparse = tuple(
            renderer.Tile(
                index=index,
                tile_ref=f"TILE-{index:03d}",
                min_x=float(index),
                min_y=0.0,
                max_x=float(index + 1),
                max_y=1.0,
                maximum_z=1.0,
            )
            for index in range(400)
        )
        with self.assertRaisesRegex(
            renderer.Sim01QaRendererError,
            "8 x 5 camera grid",
        ):
            renderer.generate_camera_plan(self.inputs, sparse)

    def test_rgb_encoder_accepts_real_detail_and_rejects_blank_output(self) -> None:
        x = np.arange(640, dtype=np.uint16)[None, :]
        y = np.arange(360, dtype=np.uint16)[:, None]
        image = np.empty((360, 640, 3), dtype=np.uint8)
        image[:, :, 0] = (x + y) % 256
        image[:, :, 1] = (x * 3 + y * 2) % 256
        image[:, :, 2] = (x * 7 + y * 5) % 256
        path = self.root / "proof.png"

        metrics = renderer._write_rgb_png(path, image)

        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 1024)
        self.assertGreater(metrics["rgb_stddev"], 4.0)
        self.assertGreater(metrics["edge_energy"], 0.15)
        with self.assertRaisesRegex(
            renderer.Sim01QaRendererError,
            "blank, uniform",
        ):
            renderer._write_rgb_png(
                self.root / "blank.png",
                np.zeros((360, 640, 3), dtype=np.uint8),
            )

    def test_quality_report_is_bound_to_real_counts_and_eight_images(self) -> None:
        renders = [
            {
                "render_id": f"PROOF-{index:02d}",
                "image": {"sha256": f"{index:064x}"},
                "visual_metrics": {
                    "rgb_stddev": 30.0 + index,
                    "edge_energy": 4.0 + index,
                },
                "streaming_working_set": _streaming_snapshot(index - 1),
            }
            for index in range(1, 9)
        ]
        report = renderer._quality_report(
            inputs=self.inputs,
            runtime_scene={
                "composed_prim_count": 10_000,
                "authored_payload_prim_count": 1_600,
                "resolved_layer_count": 2_000,
                "resolved_asset_count": 30,
                "unresolved_dependency_count": 0,
                "forbidden_primitive_count": 0,
            },
            renders=renders,
        )

        self.assertEqual(report["state"], renderer.QUALITY_REPORT_STATE)
        self.assertEqual(report["defect_count"], 0)
        self.assertEqual(
            report["checks"]["structure"]["hero_payload_count"],
            400,
        )
        self.assertEqual(
            report["checks"]["density"]["vegetation_instance_count"],
            1_000,
        )
        self.assertEqual(
            report["checks"]["native_visual"]["distinct_image_count"],
            8,
        )

    def test_stability_report_uses_live_sample_contract_and_runtime_hash(self) -> None:
        sampler = renderer._MemorySampler()
        sampler.gpu_name = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        sampler.total_vram_mib = 96_000
        sampler.vram_samples = [30_000.0, 62_000.0]
        sampler.ram_samples = [70_000.0, 110_000.0]
        measurement = renderer.PostRenderMeasurement(
            camera_id="VIEW-01",
            resolution_px=(3840, 2160),
            frame_times=(1.0 / 60.0,) * 120,
            vram_samples=(30_000.0, 62_000.0),
            ram_samples=(70_000.0, 110_000.0),
        )
        with mock.patch.object(renderer, "_cgroup_limit_mib", return_value=142_000):
            report = renderer._stability_report(
                inputs=self.inputs,
                duration_seconds=120.0,
                stage_open_seconds=8.0,
                payload_settle_seconds=12.0,
                measurement=measurement,
                gpu_name=sampler.gpu_name,
                total_vram_mib=sampler.total_vram_mib,
                minimum_accepted_fps=30.0,
            )

        self.assertEqual(report["state"], renderer.STABILITY_REPORT_STATE)
        self.assertEqual(
            report["runtime_preflight_sha256"],
            _sha(self.inputs.runtime_preflight_path),
        )
        self.assertEqual(report["fps"]["sample_count"], 120)
        self.assertEqual(report["execution_mode"], "headless_native_qa")
        self.assertNotIn("editor_opened", report)
        self.assertEqual(
            report["human_editor_validation"]["state"],
            "required_before_fire_simulation",
        )
        self.assertEqual(
            report["fps"]["render_product_resolution_px"],
            [3840, 2160],
        )
        self.assertEqual(report["vram"]["peak_mib"], 62_000.0)
        self.assertEqual(report["ram"]["peak_mib"], 110_000.0)
        self.assertEqual(report["crash_count"], 0)

    def test_payload_catalog_fails_closed_on_a_missing_file(self) -> None:
        missing = self.root / "missing.usdc"
        records = [{"path": missing.name, "sha256": "0" * 64}]
        with self.assertRaisesRegex(
            renderer.Sim01QaRendererError,
            "regular non-symlink file",
        ):
            renderer._validate_catalog(
                records,
                expected_count=1,
                volume_root=self.root,
                scene_root=self.root,
                label="test payloads",
            )

    def test_payload_catalog_rehash_detects_mutation_after_verification(self) -> None:
        payload = self.root / "payload.usdc"
        payload.write_bytes(b"verified-payload")
        records = [{"path": payload.name, "sha256": _sha(payload)}]

        renderer._validate_catalog(
            records,
            expected_count=1,
            volume_root=self.root,
            scene_root=self.root,
            label="test payloads",
        )
        payload.write_bytes(b"mutated-after-verification")

        with self.assertRaisesRegex(
            renderer.Sim01QaRendererError,
            "SHA-256 is stale",
        ):
            renderer._validate_catalog(
                records,
                expected_count=1,
                volume_root=self.root,
                scene_root=self.root,
                label="test payloads",
            )

    def test_exclusive_detail_gate_rejects_two_lods_for_one_tile(self) -> None:
        class Prim:
            def __init__(self, loaded: bool) -> None:
                self.loaded = loaded

            def IsValid(self) -> bool:
                return True

            def HasAuthoredPayloads(self) -> bool:
                return True

            def IsLoaded(self) -> bool:
                return self.loaded

        headers = [
            SimpleNamespace(
                tile_ref=f"TILE-{index:03d}",
                hero_detail_path=f"/Tile{index}/Details",
                mid_detail_path=f"/Tile{index}/DetailsMid",
                far_detail_path=f"/Tile{index}/DetailsFar",
            )
            for index in range(400)
        ]
        prims: dict[str, Prim] = {}
        for header in headers:
            prims[header.hero_detail_path] = Prim(False)
            prims[header.mid_detail_path] = Prim(False)
            prims[header.far_detail_path] = Prim(True)
        prims[headers[17].hero_detail_path] = Prim(True)
        stage = SimpleNamespace(GetPrimAtPath=lambda path: prims[str(path)])

        with self.assertRaisesRegex(
            renderer.Sim01QaRendererError,
            "duplicate_tiles",
        ):
            renderer._assert_exclusive_detail_working_set(
                stage=stage,
                headers=headers,
            )

    def test_producer_uses_load_none_and_never_global_stage_load(self) -> None:
        source = inspect.getsource(renderer.produce)
        self.assertIn("UsdContextInitialLoadSet.LOAD_NONE", source)
        self.assertIn("_apply_exclusive_streaming_plan", source)
        self.assertNotIn("stage.Load()", source)

    def test_4k_measurement_occurs_while_render_product_is_alive(self) -> None:
        source = inspect.getsource(renderer._render_one)
        created = source.index("rep.create.render_product")
        measured = source.index("_measure_post_render_product")
        destroyed = source.index("render_product.destroy")
        self.assertLess(created, measured)
        self.assertLess(measured, destroyed)

    def test_runtime_rejects_wrong_gpu_or_host_ram(self) -> None:
        wrong_gpu = _runtime()
        wrong_gpu["gpu"] = {
            "name": "NVIDIA A100-SXM4-80GB",
            "memory_mib": 81_920,
        }
        with self.assertRaisesRegex(
            renderer.Sim01QaRendererError,
            "RTX PRO 6000",
        ):
            renderer._validate_runtime(wrong_gpu)
        host_ram = _runtime()
        host_ram["system_memory"] = {
            "effective_mib": 240_000,
            "measurement": "host_meminfo",
            "source": "/proc/meminfo",
            "host_proc_meminfo_used": True,
        }
        with self.assertRaisesRegex(
            renderer.Sim01QaRendererError,
            "cgroup",
        ):
            renderer._validate_runtime(host_ram)

    def test_producer_never_plays_timeline_or_authors_fire(self) -> None:
        source = inspect.getsource(renderer.produce)
        self.assertIn("timeline.stop()", source)
        self.assertNotIn("timeline.play", source)
        self.assertNotIn("FIRE_SIMULATION_ALLOWED", source)
        self.assertIn("simulation_execution_performed", source)


if __name__ == "__main__":
    unittest.main()

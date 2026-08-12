from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fireviewer_sdg import rgb_restyle as restyle  # noqa: E402


def _load_ordered_batch_module():
    path = PROJECT_ROOT / "tools" / "restyle-rgb-ordered-validated-batch.py"
    spec = importlib.util.spec_from_file_location("fireviewer_ordered_restyle_batch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import ordered restyle runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _write_npz(path: Path, value: np.ndarray) -> None:
    np.savez_compressed(path, data=value)


def _build_capture(root: Path, *, positive: bool = True) -> Path:
    capture = root / "day01" / "case01" / "point01" / "original"
    capture.mkdir(parents=True)
    height = width = 64
    yy, xx = np.mgrid[:height, :width]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :32] = (38, 105, 48)
    rgb[:, 32:] = (86, 132, 56)
    rgb[:32] = np.clip(rgb[:32].astype(np.int16) + (28, 25, 24), 0, 255)
    rgb[18:46, 27:39] = (160, 145, 118)
    rgb[27:34, 30:36] = (55, 48, 42)
    rgb = np.clip(rgb.astype(np.int16) + ((xx + yy) % 4)[..., None], 0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(capture / "rgb.png")

    depth = np.full((height, width), 150.0, dtype=np.float32)
    depth[:, 32:] = 310.0
    depth[:32] -= 45.0
    depth[18:46, 27:39] = 95.0
    normals = np.zeros((height, width, 4), dtype=np.float32)
    normals[..., 2] = 1.0
    normals[..., 3] = 1.0
    semantic = np.ones((height, width), dtype=np.uint32)
    semantic[:, 32:] = 2
    semantic[:32] += 2
    semantic[18:46, 27:39] = 5
    instance = semantic.copy()
    instance[18:46, 27:39] = 11
    _write_npz(capture / "depth_distance_to_camera_m.npz", depth)
    _write_npz(capture / "normals_replicator.npz", normals)
    _write_npz(capture / "semantic_ids.npz", semantic)
    _write_npz(capture / "instance_ids.npz", instance)

    masks: dict[str, np.ndarray] = {}
    for name in (
        "flame_mask.npz",
        "smoke_mask.npz",
        "smoke_source_mask.npz",
        "burned_area_mask.npz",
        "front_visible_mask.npz",
        "perimeter_mask.npz",
    ):
        masks[name] = np.zeros((height, width), dtype=np.uint8)
    if positive:
        masks["flame_mask.npz"][29:34, 31:35] = 1
        masks["smoke_mask.npz"][18:29, 29:38] = 1
        masks["smoke_source_mask.npz"][27:30, 31:36] = 1
        masks["front_visible_mask.npz"][33:35, 28:39] = 1
        masks["perimeter_mask.npz"][35:37, 24:43] = 1
    for name, value in masks.items():
        _write_npz(capture / name, value)

    camera = {"camera_params": {"width": width, "height": height}, "info": {}}
    geolocation = {
        "camera_id": "CAM_01",
        "position_local_m": [1.0, 2.0, 3.0],
        "position_epsg2154_ngf_ign69_m": [883001.0, 6408002.0, 703.0],
    }
    _write_json(capture / "camera_params.json", camera)
    _write_json(capture / "geolocation.json", geolocation)
    plan = {
        "capture_id": "day_01_state_001_view_01_CAM_01_zoom_02",
        "camera_id": "CAM_01",
        "image_resolution_px": [width, height],
        "expected_fire_visible": positive,
        "sample_kind": "positive_fire" if positive else "negative_no_fire",
        "camera_position_local_m": [1.0, 2.0, 3.0],
        "camera_position_l93_ngf_ign69_m": [883001.0, 6408002.0, 703.0],
        "camera_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "camera_orientation_yaw_pitch_roll_degrees": [0.0, -5.0, 0.0],
        "focal_length_mm": 36.0,
        "horizontal_fov_degrees": 55.0,
        "captured_at_utc": "2026-08-03T12:00:00Z",
    }
    _write_json(capture / "capture-plan.json", plan)
    modality_names = [
        "rgb.png",
        "camera_params.json",
        "geolocation.json",
        "depth_distance_to_camera_m.npz",
        "normals_replicator.npz",
        "semantic_ids.npz",
        "instance_ids.npz",
        *masks.keys(),
    ]
    targets = {
        "capture_id": plan["capture_id"],
        "camera_id": "CAM_01",
        "expected_fire_in_frame": positive,
        "sample_kind": plan["sample_kind"],
        "visibility": {
            "acceptance": {"visibility_validation_status": "accepted"}
        },
        "modality_sha256": {name: _sha256(capture / name) for name in modality_names},
        "nearest_flame": {
            "projection": {"in_frame": positive, "pixel_xy": [33.0, 31.0]}
        },
        "nearest_smoke": {
            "projection": {"in_frame": positive, "pixel_xy": [33.0, 24.0]}
        },
        "simulation_time": {"day_index": 1, "time_of_day_s": 43200},
        "visible_flame_points_local_m": [[10.0, 12.0, 3.0]] if positive else [],
        "smoke_source_points_local_m": [[10.0, 12.0, 4.0]] if positive else [],
    }
    _write_json(capture / "training-targets.json", targets)
    return capture


def _temporary_contract(root: Path, model_root: Path) -> Path:
    value = json.loads(restyle.DEFAULT_CONTRACT_PATH.read_text(encoding="utf-8"))
    for index, model in enumerate(value["engine"]["models"]):
        payload = f"test-model-{index}".encode("ascii")
        path = model_root / model["comfy_subdir"] / model["comfy_name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        model["size_bytes"] = len(payload)
        model["sha256"] = hashlib.sha256(payload).hexdigest()
    contract_path = root / "contract.json"
    _write_json(contract_path, value)
    return contract_path


class RgbRestyleTests(unittest.TestCase):
    def test_locked_contract_and_workflow_are_valid(self) -> None:
        contract = restyle.load_contract()
        self.assertEqual(contract["contract_id"], "rgb-photoreal-flux2-klein-4b-v1")
        self.assertFalse(contract["release_gate"]["batch_generation_allowed"])
        self.assertFalse(contract["release_gate"]["public_distribution_allowed"])
        self.assertEqual(contract["engine"]["control_adapters"], [])
        self.assertEqual(contract["engine"]["lora_adapters"], [])

    def test_inventory_binds_modalities_and_positive_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = _build_capture(Path(temporary))
            inventory = restyle.inventory_capture(capture)
            self.assertTrue(inventory.expected_fire_visible)
            self.assertGreater(inventory.fire_pixel_count, 0)
            self.assertGreater(inventory.smoke_pixel_count, 0)
            self.assertEqual(inventory.width_px, 64)
            self.assertEqual(inventory.height_px, 64)
            self.assertRegex(inventory.source_binding_sha256, r"^[0-9a-f]{64}$")

    def test_positive_infinity_depth_no_hit_pixels_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = _build_capture(Path(temporary))
            depth_path = capture / "depth_distance_to_camera_m.npz"
            with np.load(depth_path, allow_pickle=False) as archive:
                depth = archive[archive.files[0]].copy()
            depth[:4, :7] = np.inf
            _write_npz(depth_path, depth)
            targets_path = capture / "training-targets.json"
            targets = json.loads(targets_path.read_text(encoding="utf-8"))
            targets["modality_sha256"][depth_path.name] = _sha256(depth_path)
            _write_json(targets_path, targets)

            inventory = restyle.inventory_capture(capture)
            self.assertEqual(inventory.width_px, 64)
            source = np.asarray(Image.open(capture / "rgb.png").convert("RGB"), dtype=np.uint8)
            candidate = np.clip(source.astype(np.int16) + 18, 0, 255).astype(np.uint8)
            candidate_path = Path(temporary) / "candidate.png"
            Image.fromarray(candidate, mode="RGB").save(candidate_path)
            result = restyle.admit_candidate(
                capture,
                candidate_path,
                output_dir=Path(temporary) / "accepted",
            )
            qa = json.loads(Path(result["qa_path"]).read_text(encoding="utf-8"))
            self.assertTrue(qa["gates"]["depth_boundaries_preserved"])

    def test_tampered_modality_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = _build_capture(Path(temporary))
            (capture / "camera_params.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(restyle.RestyleContractError, "SHA-256 mismatch"):
                restyle.inventory_capture(capture)

    def test_seed_and_rendered_workflow_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = _build_capture(Path(temporary))
            contract = restyle.load_contract()
            inventory = restyle.inventory_capture(capture, contract)
            first, seed_one = restyle.render_workflow(
                contract,
                inventory,
                input_name="input/rgb.png",
                output_prefix="output/candidate",
            )
            second, seed_two = restyle.render_workflow(
                contract,
                inventory,
                input_name="input/rgb.png",
                output_prefix="output/candidate",
            )
            self.assertEqual(seed_one, seed_two)
            self.assertEqual(first, second)
            serialized = json.dumps(first)
            for placeholder in restyle.PLACEHOLDERS:
                self.assertNotIn(placeholder, serialized)
            self.assertEqual(first["11"]["inputs"]["noise_seed"], seed_one)

    def test_candidate_stems_and_retry_seeds_are_unique_and_repeatable(self) -> None:
        job_id = "a" * 24
        self.assertEqual(
            restyle.candidate_output_prefix(job_id),
            f"fireviewer_rgb_restyle/{job_id}/candidate-{job_id}-attempt-00",
        )
        self.assertNotEqual(
            restyle.candidate_output_prefix(job_id, 1),
            restyle.candidate_output_prefix(job_id),
        )
        self.assertEqual(restyle.derive_attempt_seed(42, 0), 42)
        self.assertEqual(restyle.derive_attempt_seed(42, 2), restyle.derive_attempt_seed(42, 2))
        self.assertNotEqual(restyle.derive_attempt_seed(42, 1), 42)

    def test_validated_contract_inherits_the_immutable_v1_engine(self) -> None:
        path = PROJECT_ROOT / "config" / "rgb-restyle-photoreal-v2-validated.json"
        contract = restyle.load_contract(path)
        self.assertEqual(contract["contract_id"], "rgb-restyle-photoreal-v2-validated")
        self.assertEqual(contract["engine"]["models"][0]["comfy_name"], "flux-2-klein-4b-nvfp4.safetensors")
        self.assertTrue(contract["release_gate"]["batch_generation_allowed"])
        self.assertGreater(contract["composition_lock"]["minimum_output_edge_precision"], 0.0)

    def test_v3_batch_selects_only_accepted_positive_originals(self) -> None:
        batch = _load_ordered_batch_module()
        contract_path = (
            PROJECT_ROOT / "config" / "rgb-restyle-photoreal-v3-img2img-series.json"
        )
        contract = restyle.load_contract(contract_path)
        self.assertEqual(contract["engine"]["minimum_version"], "0.26.2")
        self.assertEqual(contract["batch_selection"]["capture_member"], "original")
        self.assertIn("do not create a new scene", contract["prompt"]["positive"])
        self.assertIn("dirty forest ground", contract["prompt"]["positive"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            accepted = _build_capture(root)
            shutil.copytree(accepted, accepted.parent / "zoom01")
            rejected = root / "day01" / "case02" / "point01" / "original"
            shutil.copytree(accepted, rejected)
            rejected_targets = json.loads(
                (rejected / "training-targets.json").read_text(encoding="utf-8")
            )
            rejected_targets["visibility"]["acceptance"][
                "visibility_validation_status"
            ] = "rejected"
            _write_json(rejected / "training-targets.json", rejected_targets)

            records = batch.ordered_original_captures(root)
            self.assertEqual(len(records), 2)
            self.assertTrue(batch.capture_selection(accepted, contract)["selected"])
            self.assertFalse(batch.capture_selection(rejected, contract)["selected"])
            self.assertNotIn("zoom01", [capture.name for _key, capture in records])

    def test_v3_preflight_binds_event_projections_to_masks(self) -> None:
        batch = _load_ordered_batch_module()
        contract = restyle.load_contract(
            PROJECT_ROOT / "config" / "rgb-restyle-photoreal-v3-img2img-series.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            capture = _build_capture(Path(temporary))
            preflight = batch.validate_original_capture(capture, contract)
            self.assertEqual(preflight["status"], "passed")
            self.assertGreater(preflight["event_witnesses"][0]["pixel_count"], 0)
            targets_path = capture / "training-targets.json"
            targets = json.loads(targets_path.read_text(encoding="utf-8"))
            targets["nearest_flame"]["projection"]["pixel_xy"] = [0.0, 0.0]
            _write_json(targets_path, targets)
            with self.assertRaisesRegex(
                restyle.RestyleContractError, "not anchored to flame_mask"
            ):
                batch.validate_original_capture(capture, contract)

    def test_prepare_is_dry_run_and_does_not_change_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = _build_capture(root / "capture")
            before = {path.name: _sha256(path) for path in capture.iterdir() if path.is_file()}
            model_root = root / "models"
            contract = _temporary_contract(root, model_root)
            manifest_path, manifest = restyle.prepare_job(
                capture,
                root / "jobs",
                contract_path=contract,
                model_root=model_root,
            )
            after = {path.name: _sha256(path) for path in capture.iterdir() if path.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(manifest["state"], "prepared-not-submitted")
            self.assertFalse(manifest["gpu_workload_submitted"])
            self.assertTrue(manifest_path.is_file())
            self.assertTrue((manifest_path.parent / "prompt.api.json").is_file())
            with self.assertRaisesRegex(restyle.RestyleContractError, "already exists"):
                restyle.prepare_job(
                    capture,
                    root / "jobs",
                    contract_path=contract,
                    model_root=model_root,
                )

    def test_protected_pixels_are_exact_and_valid_style_is_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = _build_capture(root / "capture")
            source_hashes = {path.name: _sha256(path) for path in capture.iterdir() if path.is_file()}
            source = np.asarray(Image.open(capture / "rgb.png").convert("RGB"), dtype=np.uint8)
            candidate = np.clip(source.astype(np.int16) + 18, 0, 255).astype(np.uint8)
            candidate_path = root / "candidate.png"
            Image.fromarray(candidate, mode="RGB").save(candidate_path)
            output_dir = root / "accepted"
            result = restyle.admit_candidate(capture, candidate_path, output_dir=output_dir)
            self.assertEqual(result["admission_status"], "passed")
            admitted = np.asarray(Image.open(output_dir / "rgb.png").convert("RGB"), dtype=np.uint8)
            contract = restyle.load_contract()
            protected = restyle.build_protected_mask(capture, contract, (64, 64))
            np.testing.assert_array_equal(admitted[protected], source[protected])
            self.assertEqual(
                source_hashes,
                {path.name: _sha256(path) for path in capture.iterdir() if path.is_file()},
            )
            receipt = json.loads((output_dir / "restyle-receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(receipt["ai_provenance"]["ai_generated_or_modified"])
            self.assertEqual(receipt["source_files"], restyle.inventory_capture(capture).source_files)
            with self.assertRaisesRegex(restyle.RestyleContractError, "already exists"):
                restyle.admit_candidate(capture, candidate_path, output_dir=output_dir)

    def test_no_op_and_shifted_candidates_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = _build_capture(root / "capture")
            source = np.asarray(Image.open(capture / "rgb.png").convert("RGB"), dtype=np.uint8)
            no_op = root / "no-op.png"
            Image.fromarray(source, mode="RGB").save(no_op)
            with self.assertRaises(restyle.AdmissionRejected) as no_op_context:
                restyle.admit_candidate(capture, no_op, output_dir=root / "no-op-output")
            self.assertFalse(no_op_context.exception.report["gates"]["style_changed"])

            shifted = root / "shifted.png"
            Image.fromarray(np.roll(source, 10, axis=1), mode="RGB").save(shifted)
            with self.assertRaises(restyle.AdmissionRejected) as shifted_context:
                restyle.admit_candidate(capture, shifted, output_dir=root / "shift-output")
            gates = shifted_context.exception.report["gates"]
            self.assertTrue(
                not gates["source_edges_preserved"]
                or not gates["semantic_boundaries_preserved"]
                or not gates["depth_boundaries_preserved"]
                or not gates["coarse_composition_preserved"]
            )

    def test_positive_metadata_without_visible_mask_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = _build_capture(Path(temporary), positive=True)
            for name in ("flame_mask.npz", "smoke_mask.npz"):
                _write_npz(capture / name, np.zeros((64, 64), dtype=np.uint8))
            targets = json.loads((capture / "training-targets.json").read_text(encoding="utf-8"))
            targets["modality_sha256"]["flame_mask.npz"] = _sha256(capture / "flame_mask.npz")
            targets["modality_sha256"]["smoke_mask.npz"] = _sha256(capture / "smoke_mask.npz")
            _write_json(capture / "training-targets.json", targets)
            with self.assertRaisesRegex(restyle.RestyleContractError, "no flame or smoke pixels"):
                restyle.inventory_capture(capture)


if __name__ == "__main__":
    unittest.main()

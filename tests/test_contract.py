from __future__ import annotations

import json
import unittest
from pathlib import Path


from fireviewer_sdg.generate import build_pose_schedule, build_procedural_camera_poses


ROOT = Path(__file__).resolve().parents[1]


class ImageContractTests(unittest.TestCase):
    def test_build_context_contains_no_payload_binaries(self) -> None:
        forbidden = {".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}
        offenders = [
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in forbidden
        ]
        self.assertEqual(offenders, [])

    def test_default_manifest_has_no_runtime_payload(self) -> None:
        payload = json.loads((ROOT / "provision-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["artifacts"], [])

    def test_pose_schedule_maps_every_frame_deterministically(self) -> None:
        scenario = {
            "frame_count": 5,
            "camera_poses": [
                {"id": "road-day", "position": [0, 0, 2], "look_at": [1, 0, 2]},
                {"id": "building-night", "position": [4, 2, 8], "look_at": [0, 0, 3]},
            ],
        }
        schedule = build_pose_schedule(scenario)
        self.assertEqual(len(schedule), 5)
        self.assertEqual(
            [pose["id"] for pose in schedule],
            ["road-day", "building-night", "road-day", "building-night", "road-day"],
        )

    def test_procedural_camera_poses_are_deterministic(self) -> None:
        scenario = {"seed": 42, "viewpoint": "valley"}
        first = build_procedural_camera_poses(scenario)
        second = build_procedural_camera_poses(scenario)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(len({pose["id"] for pose in first}), 16)

    def test_dockerfile_bootstraps_runtime_without_local_payloads(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:", dockerfile)
        self.assertIn("fireviewer_sdg.runtime_bootstrap", dockerfile)
        self.assertIn("FW_SDG_RUNTIME_ROOT=/opt/fireviewer-runtime/", dockerfile)
        self.assertIn("COPY campaigns ./campaigns", dockerfile)
        self.assertNotIn("nvcr.io", dockerfile)
        self.assertNotIn("_install_runtime", dockerfile)
        self.assertNotIn("pip install", dockerfile)
        self.assertNotIn("COPY models", dockerfile)
        self.assertNotIn("COPY datasets", dockerfile)
        self.assertNotIn("COPY assets", dockerfile)
        omniverse = (ROOT / "config" / "omniverse.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn('cache_root = "/opt/fireviewer-cache/ov"', omniverse)
        self.assertIn('data_root = "/opt/fireviewer-data/ov"', omniverse)
        self.assertIn('logs_root = "/opt/fireviewer-logs/ov"', omniverse)

    def test_console_contains_no_mock_case_inventory(self) -> None:
        static_root = ROOT / "src" / "fireviewer_sdg" / "static"
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(static_root.iterdir())
            if path.is_file()
        ).lower()
        for forbidden in (
            "case_001024",
            "18446744073709551615",
            "1024 / 2048",
            "768 / 2048",
            "const mockcases",
            "mock_data",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn('request("/v1/console/status")', source)
        self.assertIn('request(`/v1/cases?', source)
        self.assertIn("setup en cours — pilote verrouillé", source)
        html = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "fireviewer_sdg"
            / "static"
            / "console.html"
        ).read_text(encoding="utf-8")
        self.assertIn('id="start-production" class="start-button" type="button" disabled', html)
        self.assertIn('id="accept" type="button" class="accept-button" disabled', html)

    def test_probe_exits_disposable_process_without_kit_shutdown(self) -> None:
        source = (
            ROOT / "src" / "fireviewer_sdg" / "isaac_probe.py"
        ).read_text(encoding="utf-8")
        self.assertIn("os._exit(0)", source)
        self.assertIn("os._exit(1)", source)
        self.assertNotIn("application.close", source)
        self.assertIn("_prepare_official_asset_lock()", source)
        self.assertIn("provision_official_nvidia_manifest", source)

    def test_ign_usd_validation_initializes_isaac_namespace_first(self) -> None:
        source = (
            ROOT / "src" / "fireviewer_sdg" / "ign_catalog.py"
        ).read_text(encoding="utf-8")
        validator = source[
            source.index("def _validate_usd_assets") :
            source.index("def _event_contract")
        ]
        self.assertLess(validator.index("import isaacsim"), validator.index("from pxr import"))
        self.assertNotIn("def _write_actor_assets", source)
        self.assertNotIn("def _actor_geometry", source)


if __name__ == "__main__":
    unittest.main()

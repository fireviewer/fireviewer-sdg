from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fireviewer_sdg import isaac_probe


class IsaacProbeTests(unittest.TestCase):
    def test_rtx_rgb_requires_expected_dimensions_and_contrast(self) -> None:
        class Rgb:
            shape = (64, 64, 4)
            size = 64 * 64 * 4

            @staticmethod
            def min() -> int:
                return 3

            @staticmethod
            def max() -> int:
                return 251

        result = isaac_probe._validate_rtx_rgb(Rgb(), resolution=(64, 64))

        self.assertEqual(result["resolution"], [64, 64])
        self.assertEqual(result["minimum"], 3.0)
        self.assertEqual(result["maximum"], 251.0)

    def test_rtx_rgb_rejects_uniform_or_malformed_output(self) -> None:
        class UniformRgb:
            shape = (64, 64, 3)
            size = 64 * 64 * 3

            @staticmethod
            def min() -> int:
                return 0

            @staticmethod
            def max() -> int:
                return 0

        with self.assertRaisesRegex(RuntimeError, "uniform"):
            isaac_probe._validate_rtx_rgb(UniformRgb(), resolution=(64, 64))
        with self.assertRaisesRegex(RuntimeError, "shape"):
            isaac_probe._validate_rtx_rgb(object(), resolution=(64, 64))

    def test_preflight_receipt_is_confined_to_volume_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            receipt = root / "zone-scenes" / "Z16" / "runtime-preflight.json"
            with mock.patch.dict(
                os.environ,
                {
                    "FW_SDG_VOLUME_ROOT": str(root),
                    "FW_SDG_GPU_PREFLIGHT_RECEIPT": str(receipt),
                },
                clear=False,
            ):
                written = isaac_probe._write_preflight_receipt(
                    {"rtx_render": {"maximum": 1.0}}
                )
            self.assertEqual(written, receipt.resolve())
            self.assertEqual(
                json.loads(receipt.read_text(encoding="utf-8")),
                {"rtx_render": {"maximum": 1.0}},
            )
            with mock.patch.dict(
                os.environ,
                {
                    "FW_SDG_VOLUME_ROOT": str(root),
                    "FW_SDG_GPU_PREFLIGHT_RECEIPT": str(
                        Path(directory) / "outside.json"
                    ),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "inside FW_SDG_VOLUME_ROOT"
                ):
                    isaac_probe._write_preflight_receipt({})

    def test_asset_lock_is_not_touched_without_catalog_preparation(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"FW_SDG_PREPARE_IGN_CATALOG": "0"},
            clear=False,
        ):
            self.assertIsNone(isaac_probe._prepare_official_asset_lock())

    def test_asset_lock_is_created_inside_probe_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            manifest = volume / "input" / "simready-assets-hd-v2.json"
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "FW_SDG_PREPARE_IGN_CATALOG": "1",
                        "FW_SDG_VOLUME_ROOT": str(volume),
                        "FW_SDG_SIMREADY_ASSET_MANIFEST": "",
                    },
                    clear=False,
                ),
                mock.patch(
                    "fireviewer_sdg.simready_assets.provision_official_nvidia_manifest",
                    return_value={
                        "manifest": manifest,
                        "candidate_count": 12,
                        "missing_environment": [],
                        "missing_actor_classes": ["canadair"],
                    },
                ) as provision,
            ):
                result = isaac_probe._prepare_official_asset_lock()

        self.assertEqual(result["state"], "prepared")
        self.assertEqual(result["candidate_count"], 12)
        self.assertEqual(result["missing_actor_classes"], ["canadair"])
        provision.assert_called_once()

    def test_existing_asset_lock_is_not_crawled_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            manifest = volume / "input" / "simready-assets-hd-v2.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                '{"discovery":{"mode":"official_nvidia_materialized_lock_v2"}}',
                encoding="utf-8",
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "FW_SDG_PREPARE_IGN_CATALOG": "1",
                        "FW_SDG_VOLUME_ROOT": str(volume),
                        "FW_SDG_SIMREADY_ASSET_MANIFEST": "",
                    },
                    clear=False,
                ),
                mock.patch(
                    "fireviewer_sdg.simready_assets.provision_official_nvidia_manifest"
                ) as provision,
            ):
                result = isaac_probe._prepare_official_asset_lock()

        self.assertEqual(result["state"], "existing")
        provision.assert_not_called()

    def test_remote_only_asset_lock_is_rebuilt_as_materialized_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            manifest = volume / "input" / "simready-assets-hd-v2.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                '{"discovery":{"mode":"official_nvidia_remote_reference_lock"}}',
                encoding="utf-8",
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "FW_SDG_PREPARE_IGN_CATALOG": "1",
                        "FW_SDG_VOLUME_ROOT": str(volume),
                        "FW_SDG_SIMREADY_ASSET_MANIFEST": "",
                    },
                    clear=False,
                ),
                mock.patch(
                    "fireviewer_sdg.simready_assets.provision_official_nvidia_manifest",
                    return_value={
                        "manifest": manifest,
                        "candidate_count": 42,
                        "missing_environment": [],
                        "missing_actor_classes": [],
                    },
                ) as provision,
            ):
                result = isaac_probe._prepare_official_asset_lock()

        self.assertEqual(result["state"], "prepared")
        self.assertEqual(result["candidate_count"], 42)
        provision.assert_called_once()


if __name__ == "__main__":
    unittest.main()

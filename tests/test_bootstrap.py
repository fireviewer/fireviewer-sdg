from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg import bootstrap


class BootstrapTests(unittest.TestCase):
    def test_compatibility_checker_explicitly_allows_container_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checker = Path(directory) / "isaacsim"
            checker.write_text("", encoding="ascii")
            with (
                mock.patch.object(bootstrap.sys, "executable", str(checker)),
                mock.patch.object(bootstrap.sys, "platform", "linux"),
                mock.patch.object(os, "geteuid", return_value=0, create=True),
                mock.patch.object(bootstrap.subprocess, "run") as run,
            ):
                bootstrap._run_compatibility_checker()

        command = run.call_args.args[0]
        self.assertIn("isaacsim.exp.compatibility_check", command)
        self.assertIn("--allow-root", command)
        self.assertEqual(run.call_args.kwargs["timeout"], 180)
        self.assertTrue(run.call_args.kwargs["check"])

    def test_compatibility_checker_uses_windows_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "python.exe"
            checker = Path(directory) / "isaacsim.exe"
            checker.write_text("", encoding="ascii")
            with (
                mock.patch.object(bootstrap.sys, "executable", str(python)),
                mock.patch.object(bootstrap.sys, "platform", "win32"),
                mock.patch.object(bootstrap.subprocess, "run") as run,
            ):
                bootstrap._run_compatibility_checker()

        self.assertEqual(Path(run.call_args.args[0][0]), checker)

    def test_compatibility_checker_accepts_native_workstation_script(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checker = Path(directory) / "isaac-sim.compatibility_check.bat"
            checker.write_text("", encoding="ascii")
            with (
                mock.patch.dict(
                    os.environ,
                    {"FW_SDG_ISAAC_COMPATIBILITY_CHECKER": str(checker)},
                    clear=False,
                ),
                mock.patch.object(bootstrap.subprocess, "run") as run,
            ):
                bootstrap._run_compatibility_checker()

        command = run.call_args.args[0]
        self.assertEqual(Path(command[0]), checker)
        self.assertNotIn("isaacsim.exp.compatibility_check", command)

    def test_probe_allows_first_run_asset_discovery_to_finish(self) -> None:
        with (
            mock.patch.object(bootstrap.subprocess, "run") as run,
        ):
            bootstrap._run_probe(prepare_assets=True)

        self.assertEqual(run.call_args.kwargs["timeout"], 1800)
        self.assertEqual(
            run.call_args.kwargs["env"]["FW_SDG_PREPARE_IGN_CATALOG"],
            "1",
        )

    def test_existing_zone_preflight_reuses_only_a_complete_workspace_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "zone-scenes" / "Z16" / "runtime-preflight.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(
                '{"isaac_sim":true,"replicator":true,"flow":true,'
                '"rtx_render":{"resolution":[64,64],"shape":[64,64,4]}}',
                encoding="utf-8",
            )
            settings = mock.Mock(run_mode="zone_scenes", volume_root=root)
            with mock.patch.dict(
                os.environ,
                {"FW_SDG_GPU_PREFLIGHT_RECEIPT": str(receipt)},
                clear=False,
            ):
                result = bootstrap._existing_zone_gpu_preflight(settings)
        self.assertIsNotNone(result)

    def test_existing_zone_preflight_rejects_incomplete_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "runtime-preflight.json"
            receipt.write_text('{"isaac_sim":true}', encoding="utf-8")
            settings = mock.Mock(run_mode="zone_scenes", volume_root=root)
            with mock.patch.dict(
                os.environ,
                {"FW_SDG_GPU_PREFLIGHT_RECEIPT": str(receipt)},
                clear=False,
            ):
                result = bootstrap._existing_zone_gpu_preflight(settings)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

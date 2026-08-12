from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "tools" / "hunyuan3d" / "production_batches.py"
DOCKER_ROOT = MODULE_PATH.parent / "docker"
SPEC = importlib.util.spec_from_file_location("production_batches", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

FREEZER_PATH = MODULE_PATH.parent / "freeze_completed_initial.py"
FREEZER_SPEC = importlib.util.spec_from_file_location("freeze_completed_initial", FREEZER_PATH)
assert FREEZER_SPEC and FREEZER_SPEC.loader
FREEZER = importlib.util.module_from_spec(FREEZER_SPEC)
sys.modules[FREEZER_SPEC.name] = FREEZER
FREEZER_SPEC.loader.exec_module(FREEZER)


class ProductionBatchTests(unittest.TestCase):
    def test_remaining_242_assets_are_split_into_49_batches(self) -> None:
        assets = [{"asset_id": f"asset-{index:03d}"} for index in range(294)]
        plans = MODULE.plan_batches(assets, start=52, batch_size=5)
        self.assertEqual(len(plans), 49)
        self.assertEqual(plans[0].name, "batch-0053-0057")
        self.assertEqual(len(plans[0].assets), 5)
        self.assertEqual(plans[-1].name, "batch-0293-0294")
        self.assertEqual(len(plans[-1].assets), 2)

    def test_remaining_assets_after_102_are_split_into_ten_batches_of_twenty(self) -> None:
        assets = [{"asset_id": f"asset-{index:03d}"} for index in range(294)]
        plans = MODULE.plan_batches(assets, start=102, batch_size=20)
        self.assertEqual(len(plans), 10)
        self.assertEqual(plans[0].name, "batch-0103-0122")
        self.assertEqual(len(plans[0].assets), 20)
        self.assertEqual(plans[-1].name, "batch-0283-0294")
        self.assertEqual(len(plans[-1].assets), 12)

    def test_generation_only_copies_both_initial_glbs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            textured = root / "source-textured.glb"
            untextured = root / "source-untextured.glb"
            textured.write_bytes(b"textured")
            untextured.write_bytes(b"untextured")
            plan = MODULE.BatchPlan(1, 102, 103, ({"asset_id": "asset-103"},))
            state = {
                "assets": {
                    "asset-103": {
                        "textured_50k": str(textured),
                        "untextured_50k": str(untextured),
                    }
                }
            }

            MODULE.copy_initial_glbs(state, plan, root / "glb", root / "untextured_50k")

            self.assertEqual((root / "glb" / "asset-103.glb").read_bytes(), b"textured")
            self.assertEqual((root / "untextured_50k" / "asset-103.glb").read_bytes(), b"untextured")

    def test_transfer_manifest_detects_payload_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "glb").mkdir()
            payload = root / "glb" / "asset.glb"
            payload.write_bytes(b"valid")
            manifest = MODULE.transfer_manifest(root)
            MODULE.verify_transfer_manifest(root, manifest)
            payload.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                MODULE.verify_transfer_manifest(root, manifest)

    def test_subset_manifest_contains_only_batch_assets(self) -> None:
        assets = tuple({"asset_id": f"asset-{index}", "route": "hunyuan3d"} for index in range(5))
        plan = MODULE.BatchPlan(1, 52, 57, assets)
        subset = MODULE.subset_manifest({"assets": [], "route_counts": {}}, plan)
        self.assertEqual(subset["asset_count"], 5)
        self.assertEqual(subset["route_counts"], {"hunyuan3d": 5})
        self.assertEqual(subset["production_batch"]["start_asset_number"], 53)

    def test_remote_output_retries_transient_ssh_failure(self) -> None:
        args = SimpleNamespace(
            port=22,
            identity=Path("identity"),
            known_hosts=Path("known_hosts"),
            user="root",
            host="pod.example",
            poll=0.0,
            transport_timeout=30.0,
        )
        failure = MODULE.subprocess.CalledProcessError(255, ["ssh"])
        success = MODULE.subprocess.CompletedProcess(["ssh"], 0, stdout="ready\n", stderr="")
        with (
            mock.patch.object(MODULE.subprocess, "run", side_effect=[failure, success]) as run,
            mock.patch.object(MODULE.time, "sleep") as sleep,
        ):
            self.assertEqual(MODULE.remote_output(args, "true"), "ready")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args.kwargs["creationflags"],
            getattr(MODULE.subprocess, "CREATE_NO_WINDOW", 0) if MODULE.os.name == "nt" else 0,
        )
        self.assertEqual(run.call_args.kwargs["timeout"], 30.0)
        sleep.assert_called_once_with(0.0)

    def test_remote_output_retries_transport_timeout(self) -> None:
        args = SimpleNamespace(
            port=22,
            identity=Path("identity"),
            known_hosts=Path("known_hosts"),
            user="root",
            host="pod.example",
            poll=0.0,
            transport_timeout=12.0,
        )
        failure = MODULE.subprocess.TimeoutExpired(cmd=["ssh"], timeout=12.0)
        success = MODULE.subprocess.CompletedProcess(
            args=["ssh"], returncode=0, stdout="ready\n", stderr=""
        )
        with (
            mock.patch.object(MODULE.subprocess, "run", side_effect=[failure, success]) as run,
            mock.patch.object(MODULE.time, "sleep") as sleep,
        ):
            self.assertEqual(MODULE.remote_output(args, "true"), "ready")

        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(0.0)

    def test_usd_runtime_environment_prepends_wheel_library_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            library_dir = (
                prefix
                / "lib"
                / f"python{MODULE.sys.version_info.major}.{MODULE.sys.version_info.minor}"
                / "site-packages"
                / "omni"
                / "converter"
                / "hoops"
            )
            library_dir.mkdir(parents=True)
            with mock.patch.object(MODULE.sys, "platform", "linux"), mock.patch.dict(
                MODULE.os.environ, {"LD_LIBRARY_PATH": "/system/libs"}
            ):
                env = MODULE.usd_runtime_environment(prefix)
        self.assertEqual(env["LD_LIBRARY_PATH"], f"{library_dir}{MODULE.os.pathsep}/system/libs")

    def test_complex_assets_can_promote_above_twenty_thousand_faces(self) -> None:
        self.assertEqual(MODULE.RETOPO_MAXIMUM_FACES, 50_000)

    def test_retriever_accepts_a_durable_log_file(self) -> None:
        args = MODULE.build_parser().parse_args(
            [
                "retrieve",
                "--host", "pod.example",
                "--port", "22",
                "--identity", "identity",
                "--known-hosts", "known_hosts",
                "--local-root", "output",
                "--log-file", "output/retriever.log",
            ]
        )
        self.assertEqual(args.log_file, Path("output/retriever.log"))

    def test_retriever_remote_probes_normalize_empty_state_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                local_root=Path(temporary),
                remote_outgoing="/workspace/outgoing-generation-only",
                poll=0.0,
            )
            complete = json.dumps({"schema_version": 1, "status": "complete"})
            with mock.patch.object(MODULE, "remote_output", side_effect=["", complete]) as remote:
                self.assertEqual(MODULE.retrieve(args), 0)

        ready_probe = remote.call_args_list[0].args[1]
        complete_probe = remote.call_args_list[1].args[1]
        self.assertIn("find /workspace/outgoing-generation-only", ready_probe)
        self.assertTrue(ready_probe.endswith("; exit 0"))
        self.assertIn("if test -f /workspace/outgoing-generation-only/PRODUCTION_COMPLETE.json", complete_probe)
        self.assertTrue(complete_probe.endswith("; exit 0"))

    def test_transfer_ack_survives_concurrent_remote_rotation(self) -> None:
        args = SimpleNamespace()
        ready = "/workspace/outgoing-generation-only/batch-0163-0182"
        with mock.patch.object(MODULE, "remote_output", return_value="") as remote:
            MODULE.acknowledge_transfer(args, ready)

        command = remote.call_args.args[1]
        self.assertIn(f"if test -d {ready}", command)
        self.assertIn(f"touch {ready}/TRANSFERRED", command)
        self.assertTrue(command.endswith("; exit 0"))

    def test_comfy_memory_release_unloads_models(self) -> None:
        response = mock.MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        with mock.patch.object(MODULE.urllib.request, "urlopen", return_value=response) as urlopen:
            MODULE.free_comfy_memory("http://127.0.0.1:8188/")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8188/free")
        self.assertEqual(request.data, b'{"unload_models":true,"free_memory":true}')

    def test_runtime_image_applies_linux_heap_trim_patch(self) -> None:
        dockerfile = (DOCKER_ROOT / "Dockerfile").read_text(encoding="utf-8")
        patch = (DOCKER_ROOT / "patches" / "comfyui-malloc-trim.patch").read_text(encoding="utf-8")
        self.assertIn("git -C /opt/ComfyUI apply --check", dockerfile)
        self.assertIn("comfyui-malloc-trim.patch", dockerfile)
        self.assertIn('ctypes.CDLL("libc.so.6")', patch)
        self.assertIn("libc.malloc_trim(0)", patch)

    def test_freezer_excludes_completed_assets_from_previous_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            untextured = root / "untextured"
            textured = root / "textured"
            for asset_id in ("current", "stale"):
                (untextured / asset_id).mkdir(parents=True)
                (textured / asset_id).mkdir(parents=True)
                (untextured / asset_id / f"{asset_id}.glb").write_bytes(b"raw")
                (textured / asset_id / f"{asset_id}.glb").write_bytes(b"painted")
            state = root / "state.json"
            state.write_text(json.dumps({"assets": {"current": {}}}), encoding="utf-8")

            pairs = FREEZER.completed_pairs(
                state,
                untextured,
                textured,
                allowed_asset_ids={"current"},
            )

        self.assertEqual([pair["asset_id"] for pair in pairs], ["current"])


if __name__ == "__main__":
    unittest.main()

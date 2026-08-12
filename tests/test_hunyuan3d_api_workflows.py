from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "tools" / "hunyuan3d" / "api_workflows.py"
SPEC = importlib.util.spec_from_file_location("api_workflows", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

BATCH_PATH = Path(__file__).parents[1] / "tools" / "hunyuan3d" / "comfyui_batch.py"
BATCH_SPEC = importlib.util.spec_from_file_location("comfyui_batch", BATCH_PATH)
assert BATCH_SPEC and BATCH_SPEC.loader
BATCH_MODULE = importlib.util.module_from_spec(BATCH_SPEC)
sys.path.insert(0, str(BATCH_PATH.parent))
sys.modules[BATCH_SPEC.name] = BATCH_MODULE
BATCH_SPEC.loader.exec_module(BATCH_MODULE)


class ApiWorkflowTests(unittest.TestCase):
    def test_geometry_workflow_keeps_raw_mesh_for_global_budgeting(self) -> None:
        prompt = MODULE.geometry_prompt("asset4sim/reference/test.png", "raw/test", 123)
        self.assertEqual(prompt["7"]["class_type"], "Hy3DVAEDecode")
        self.assertEqual(prompt["8"]["inputs"]["trimesh"], ["7", 0])
        self.assertFalse(any(node["class_type"] == "Asset4SimAdaptiveRepairRetopo" for node in prompt.values()))

    def test_texture_workflow_loads_retopo_before_uv_and_paint(self) -> None:
        prompt = MODULE.texture_prompt("asset4sim/reference/test.png", "/tmp/test.glb", "textured/test", 456)
        self.assertEqual(prompt["11"]["inputs"]["glb_path"], "/tmp/test.glb")
        self.assertEqual(prompt["12"]["inputs"]["trimesh"], ["11", 0])
        self.assertEqual(prompt["23"]["inputs"]["trimesh"], ["22", 0])
        self.assertEqual(prompt["15"]["inputs"]["model"], "hunyuan3d-paint-v2-0")

    def test_saved_glb_is_resolved_when_comfy_history_has_no_ui_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            expected = output / "asset4sim" / "raw" / "test" / "test_00001_.glb"
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"glTF-test")

            relative, absolute = BATCH_MODULE.locate_exported_glb(
                {}, output, "asset4sim/raw/test/test", not_before=0.0
            )

            self.assertEqual(relative, "asset4sim/raw/test/test_00001_.glb")
            self.assertEqual(absolute, expected.resolve())

    def test_history_timeout_is_retried_while_prompt_keeps_running(self) -> None:
        completed = {
            "prompt-1": {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {},
            }
        }
        with (
            mock.patch.object(
                BATCH_MODULE,
                "api_json",
                side_effect=[{"prompt_id": "prompt-1"}, TimeoutError("busy"), completed],
            ) as api,
            mock.patch.object(BATCH_MODULE.time, "monotonic", side_effect=[0.0, 0.0, 1.0]),
            mock.patch.object(BATCH_MODULE.time, "sleep") as sleep,
        ):
            prompt_id, record = BATCH_MODULE.submit_and_wait(
                "http://127.0.0.1:8188",
                {"1": {"class_type": "Test", "inputs": {}}},
                timeout_seconds=10,
                poll_seconds=2,
            )

        self.assertEqual(prompt_id, "prompt-1")
        self.assertIs(record, completed["prompt-1"])
        self.assertEqual(api.call_count, 3)
        sleep.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()

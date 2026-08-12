from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "hunyuan3d" / "canonical_workflow.py"
SPEC = importlib.util.spec_from_file_location("canonical_workflow", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
WORKFLOW_PATH = MODULE_PATH.parent / "workflows" / "hy3d_example_01.user-supplied.api.json"


class CanonicalWorkflowTests(unittest.TestCase):
    def test_supplied_workflow_hash_and_required_nodes_are_locked(self) -> None:
        workflow = MODULE.load_canonical(WORKFLOW_PATH)
        self.assertEqual(workflow["59"]["inputs"]["max_facenum"], 50_000)
        self.assertEqual(workflow["79"]["inputs"]["texture_size"], 2_048)
        self.assertEqual(workflow["88"]["inputs"]["steps"], 25)

    def test_initial_binding_changes_only_image_and_prefixes(self) -> None:
        workflow = MODULE.load_canonical(WORKFLOW_PATH)
        prompt, untextured, textured = MODULE.bind_initial(
            workflow,
            image_name="asset4sim/reference/test.png",
            asset_id="test",
        )
        self.assertEqual(prompt["13"]["inputs"]["image"], "asset4sim/reference/test.png")
        self.assertEqual(prompt["17"]["inputs"]["filename_prefix"], untextured)
        self.assertEqual(prompt["99"]["inputs"]["filename_prefix"], textured)
        self.assertEqual(prompt["10"]["inputs"]["model"], "hy3dgen/hunyuan3d-dit-v2-0-fp16.safetensors")
        self.assertEqual(workflow["10"]["inputs"]["model"], "hy3dgen\\hunyuan3d-dit-v2-0-fp16.safetensors")
        self.assertEqual(prompt["141"]["inputs"], workflow["141"]["inputs"])
        self.assertEqual(prompt["88"]["inputs"], workflow["88"]["inputs"])

    def test_retexture_replaces_only_mesh_source_and_output_bindings(self) -> None:
        workflow = MODULE.load_canonical(WORKFLOW_PATH)
        prompt, _, final = MODULE.bind_retexture(
            workflow,
            image_name="asset4sim/reference/test.png",
            corrected_mesh_path="/workspace/test.glb",
            asset_id="test",
        )
        self.assertEqual(prompt["59"]["class_type"], "Hy3DLoadMesh")
        self.assertEqual(prompt["59"]["inputs"]["glb_path"], "/workspace/test.glb")
        self.assertEqual(prompt["99"]["inputs"]["filename_prefix"], final)
        for node_id in ("28", "35", "61", "79", "85", "88", "92", "104", "117", "129"):
            self.assertEqual(prompt[node_id], workflow[node_id])


if __name__ == "__main__":
    unittest.main()

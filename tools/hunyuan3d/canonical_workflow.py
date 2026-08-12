"""Strict bindings for the user-supplied canonical Hunyuan3D API workflow."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


CANONICAL_SHA256 = "8f19292458c3bbc103e5a9b142a8affbe6a3082329f82c4a7f216be9ff7a0489"

REQUIRED_NODE_TYPES = {
    "13": "LoadImage",
    "17": "Hy3DExportMesh",
    "28": "DownloadAndLoadHy3DDelightModel",
    "59": "Hy3DPostprocessMesh",
    "79": "Hy3DRenderMultiView",
    "83": "Hy3DMeshUVWrap",
    "85": "DownloadAndLoadHy3DPaintModel",
    "88": "Hy3DSampleMultiView",
    "99": "Hy3DExportMesh",
    "140": "Hy3DVAEDecode",
    "141": "Hy3DGenerateMesh",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_canonical(path: Path) -> dict[str, dict[str, Any]]:
    actual_sha256 = sha256(path)
    if actual_sha256 != CANONICAL_SHA256:
        raise ValueError(
            f"Canonical workflow SHA-256 mismatch: expected {CANONICAL_SHA256}, got {actual_sha256}"
        )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(workflow, dict):
        raise ValueError("Canonical workflow must be a ComfyUI API prompt object")
    for node_id, expected_type in REQUIRED_NODE_TYPES.items():
        actual_type = workflow.get(node_id, {}).get("class_type")
        if actual_type != expected_type:
            raise ValueError(f"Node {node_id} must be {expected_type}, got {actual_type}")
    return workflow


def bind_initial(
    canonical: dict[str, dict[str, Any]],
    *,
    image_name: str,
    asset_id: str,
) -> tuple[dict[str, dict[str, Any]], str, str]:
    """Bind only the per-asset image and export prefixes."""

    prompt = copy.deepcopy(canonical)
    untextured_prefix = f"asset4sim/canonical_initial/untextured/{asset_id}/{asset_id}"
    textured_prefix = f"asset4sim/canonical_initial/textured/{asset_id}/{asset_id}"
    prompt["13"]["inputs"]["image"] = image_name
    # The supplied workflow was exported on Windows. ComfyUI model choices on
    # the Linux pod use POSIX separators; normalize only this environment path.
    prompt["10"]["inputs"]["model"] = prompt["10"]["inputs"]["model"].replace("\\", "/")
    prompt["17"]["inputs"]["filename_prefix"] = untextured_prefix
    prompt["99"]["inputs"]["filename_prefix"] = textured_prefix
    return prompt, untextured_prefix, textured_prefix


def bind_retexture(
    canonical: dict[str, dict[str, Any]],
    *,
    image_name: str,
    corrected_mesh_path: str,
    asset_id: str,
) -> tuple[dict[str, dict[str, Any]], str, str]:
    """Reuse the canonical texture graph on a corrected mesh.

    Node 59 is the only topology substitution: the canonical postprocess output
    is replaced by a GLB loader.  Every image, delight, camera, paint, bake,
    inpaint, and texture parameter remains byte-for-byte derived from the
    supplied workflow.
    """

    prompt = copy.deepcopy(canonical)
    corrected_prefix = f"asset4sim/retexture/corrected/{asset_id}/{asset_id}"
    final_prefix = f"asset4sim/retexture/final/{asset_id}/{asset_id}"
    prompt["13"]["inputs"]["image"] = image_name
    prompt["10"]["inputs"]["model"] = prompt["10"]["inputs"]["model"].replace("\\", "/")
    prompt["59"] = {
        "class_type": "Hy3DLoadMesh",
        "inputs": {"glb_path": corrected_mesh_path},
    }
    prompt["17"]["inputs"]["filename_prefix"] = corrected_prefix
    prompt["99"]["inputs"]["filename_prefix"] = final_prefix
    return prompt, corrected_prefix, final_prefix

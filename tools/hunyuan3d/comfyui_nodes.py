"""ComfyUI nodes for the Asset4Sim Hunyuan3D post-generation pipeline."""

from __future__ import annotations

import logging
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .asset4sim_mesh import (
    as_trimesh,
    compare_geometry_quality,
    measure_mesh,
    repair_and_retopologize,
    repair_mesh,
    simplify_with_quality_gate,
)


LOG = logging.getLogger("asset4sim.hunyuan3d.comfyui")


class Asset4SimAdaptiveRepairRetopo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_average": ("INT", {"default": 5000, "min": 500, "max": 100000, "step": 100}),
                "minimum_faces": ("INT", {"default": 2500, "min": 100, "max": 100000, "step": 100}),
                "maximum_faces": ("INT", {"default": 50000, "min": 500, "max": 250000, "step": 100}),
                "fill_holes": ("BOOLEAN", {"default": True}),
                "preserve_topology": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Asset4Sim/Hunyuan3D"
    DESCRIPTION = (
        "Repairs non-manifold geometry, closes holes, then performs an adaptive "
        "quality-weighted simplification before UV generation. The batch runner "
        "normalizes all targets to 5k faces on average."
    )

    def process(
        self,
        trimesh,
        target_average: int,
        minimum_faces: int,
        maximum_faces: int,
        fill_holes: bool,
        preserve_topology: bool,
    ):
        result, report = repair_and_retopologize(
            trimesh,
            target_average=target_average,
            minimum_faces=minimum_faces,
            maximum_faces=maximum_faces,
            fill_holes=fill_holes,
            preserve_topology=preserve_topology,
        )
        final = report["final"]
        LOG.info(
            "Asset4Sim repair/retopo: %s -> %s faces, target=%s, watertight=%s",
            report["before"]["faces"],
            final["faces"],
            report["target_faces"],
            final["watertight"],
        )
        return {
            "ui": {
                "text": [
                    f"target={report['target_faces']:,} final={final['faces']:,} "
                    f"watertight={final['watertight']}"
                ]
            },
            "result": (result,),
        }


class Asset4SimRepairMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "fill_holes": ("BOOLEAN", {"default": True}),
                "maximum_hole_edges": ("INT", {"default": 100000, "min": 3, "max": 1000000}),
                "floater_face_ratio": (
                    "FLOAT",
                    {"default": 0.0005, "min": 0.0, "max": 0.05, "step": 0.0001},
                ),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("repaired_mesh", "repair_report")
    FUNCTION = "process"
    CATEGORY = "Asset4Sim/Hunyuan3D/Post Generation"
    DESCRIPTION = "Cleans non-manifold geometry, removes microscopic floaters and closes holes."

    def process(self, trimesh, fill_holes: bool, maximum_hole_edges: int, floater_face_ratio: float):
        before = measure_mesh(trimesh)
        result = repair_mesh(
            trimesh,
            fill_holes=fill_holes,
            maximum_hole_edges=maximum_hole_edges,
            floater_face_ratio=floater_face_ratio,
        )
        after = measure_mesh(result)
        report = {
            "before": asdict(before),
            "after": asdict(after),
            "holes_closed": max(0, before.boundary_edges - after.boundary_edges),
        }
        summary = json.dumps(report, ensure_ascii=False)
        return {"ui": {"text": [summary]}, "result": (result, summary)}


class Asset4SimQualityRetopo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_faces": ("INT", {"default": 5000, "min": 100, "max": 250000, "step": 100}),
                "maximum_faces": ("INT", {"default": 50000, "min": 500, "max": 250000, "step": 100}),
                "preserve_topology": ("BOOLEAN", {"default": True}),
                "quality_samples": ("INT", {"default": 20000, "min": 1000, "max": 100000, "step": 1000}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("retopologized_mesh", "quality_report")
    FUNCTION = "process"
    CATEGORY = "Asset4Sim/Hunyuan3D/Post Generation"
    DESCRIPTION = (
        "Progressive QEM retopology. Starts at the requested face count and promotes complex assets "
        "up to the maximum only when the geometry quality gate fails."
    )

    def process(
        self,
        trimesh,
        target_faces: int,
        maximum_faces: int,
        preserve_topology: bool,
        quality_samples: int,
    ):
        if maximum_faces < target_faces:
            raise ValueError("maximum_faces must be greater than or equal to target_faces")
        result, attempts = simplify_with_quality_gate(
            trimesh,
            target_faces,
            maximum_faces=maximum_faces,
            preserve_topology=preserve_topology,
            quality_samples=quality_samples,
        )
        report = {
            "target_faces": target_faces,
            "accepted_faces": int(len(result.faces)),
            "attempts": attempts,
            "final": asdict(measure_mesh(result)),
        }
        summary = json.dumps(report, ensure_ascii=False)
        return {"ui": {"text": [summary]}, "result": (result, summary)}


class Asset4SimValidateRetopo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_mesh": ("TRIMESH",),
                "candidate_mesh": ("TRIMESH",),
                "quality_samples": ("INT", {"default": 20000, "min": 1000, "max": 100000, "step": 1000}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "BOOLEAN", "STRING")
    RETURN_NAMES = ("candidate_mesh", "passed", "quality_report")
    FUNCTION = "process"
    CATEGORY = "Asset4Sim/Hunyuan3D/Post Generation"

    def process(self, reference_mesh, candidate_mesh, quality_samples: int):
        quality = compare_geometry_quality(reference_mesh, candidate_mesh, sample_count=quality_samples)
        report = asdict(quality)
        summary = json.dumps(report, ensure_ascii=False)
        return {"ui": {"text": [summary]}, "result": (candidate_mesh, quality.passed, summary)}


def _safe_output_prefix(filename_prefix: str) -> Path:
    relative = Path(filename_prefix.replace("\\", "/"))
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe output prefix: {filename_prefix!r}")
    return relative


class Asset4SimExportGLBUSD:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "asset4sim/final_asset"}),
                "export_usd": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("glb_path", "usd_path", "export_report")
    FUNCTION = "process"
    OUTPUT_NODE = True
    CATEGORY = "Asset4Sim/Hunyuan3D/Post Generation"
    DESCRIPTION = "Exports the textured GLB and an Omniverse-compatible textured USD with UV validation."

    def process(self, trimesh, filename_prefix: str, export_usd: bool):
        import folder_paths

        relative = _safe_output_prefix(filename_prefix)
        stem = Path(folder_paths.get_output_directory()).resolve() / relative
        stem.parent.mkdir(parents=True, exist_ok=True)
        glb_path = stem.with_suffix(".glb")
        usd_path = stem.with_suffix(".usd")
        report_path = stem.with_name(stem.name + "-export.json")
        mesh = as_trimesh(trimesh)
        mesh.export(glb_path, file_type="glb")
        report = {
            "glb": str(glb_path),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "usd": None,
            "passed": True,
        }
        if export_usd:
            converter = shutil.which("usd-convert-cad")
            if not converter:
                candidate = Path(sys.prefix) / "bin" / "usd-convert-cad"
                converter = str(candidate) if candidate.is_file() else None
            if not converter:
                raise RuntimeError("usd-convert-cad is not installed in the ComfyUI environment")
            subprocess.run(
                [
                    converter,
                    "-i", str(glb_path),
                    "-o", str(usd_path),
                    "--material-type", "preview-surface",
                    "--instancing-style", "none",
                    "--composition-style", "none",
                    "--creator", "FireViewer Asset4Sim Hunyuan3D 2.0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            restore_script = Path(__file__).with_name("restore_usd_glb_material.py")
            runtime_lib = (
                Path(sys.prefix)
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
                / "omni"
                / "converter"
                / "hoops"
            )
            restore_env = os.environ.copy()
            if runtime_lib.is_dir():
                previous = restore_env.get("LD_LIBRARY_PATH", "")
                restore_env["LD_LIBRARY_PATH"] = str(runtime_lib) + (os.pathsep + previous if previous else "")
            subprocess.run(
                [
                    sys.executable,
                    str(restore_script),
                    str(glb_path),
                    str(usd_path),
                    "--report", str(report_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=restore_env,
            )
            restored = json.loads(report_path.read_text(encoding="utf-8"))
            if not restored.get("passed"):
                raise RuntimeError(f"USD material restoration failed: {restored}")
            report["usd"] = str(usd_path)
            report["material_restore"] = restored
            report["usd_validation"] = restored["structural_validation"]
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary = json.dumps(report, ensure_ascii=False)
        return {
            "ui": {"text": [summary]},
            "result": (str(glb_path), str(usd_path) if export_usd else "", summary),
        }


NODE_CLASS_MAPPINGS = {
    "Asset4SimAdaptiveRepairRetopo": Asset4SimAdaptiveRepairRetopo,
    "Asset4SimRepairMesh": Asset4SimRepairMesh,
    "Asset4SimQualityRetopo": Asset4SimQualityRetopo,
    "Asset4SimValidateRetopo": Asset4SimValidateRetopo,
    "Asset4SimExportGLBUSD": Asset4SimExportGLBUSD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Asset4SimAdaptiveRepairRetopo": "Asset4Sim Adaptive Repair + Retopo",
    "Asset4SimRepairMesh": "Asset4Sim 1. Repair + Fill Holes",
    "Asset4SimQualityRetopo": "Asset4Sim 2. Quality-Gated Retopology",
    "Asset4SimValidateRetopo": "Asset4Sim 3. Validate Retopology",
    "Asset4SimExportGLBUSD": "Asset4Sim 4. Export Textured GLB + USD",
}

#!/usr/bin/env python3
"""Build an active, uniformly scaled and color-corrected Asset4Sim library."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageEnhance
from pygltflib import GLTF2, Node


REJECTION_REASONS = {
    26: "User-rejected: crumpled and fragmented guardrail termination.",
    28: "User-rejected: heavily distorted concrete bollard geometry.",
    52: "User-rejected: open/incomplete roof and building shell.",
    85: "User-rejected: severe spikes, floating panels and broken trench geometry.",
    87: "User-rejected: flat debris slab instead of a concrete retaining wall.",
    88: "User-rejected: thin floating grids without a usable gabion volume.",
    100: "User-rejected: long protruding mesh/texture artifact at the grass base.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def rgba_statistics(image: Image.Image) -> dict[str, float]:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    mask = rgba[:, :, 3] > 16
    pixels = rgba[:, :, :3][mask].astype(np.float32) / 255.0
    if not len(pixels):
        raise ValueError("embedded texture has no visible pixels")
    luminance = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    maximum = pixels.max(axis=1)
    minimum = pixels.min(axis=1)
    saturation = np.divide(maximum - minimum, maximum, out=np.zeros_like(maximum), where=maximum > 0)
    return {
        "luminance_mean": float(np.mean(luminance)),
        "luminance_median": float(np.median(luminance)),
        "saturation_mean": float(np.mean(saturation)),
        "saturation_median": float(np.median(saturation)),
    }


def correction_parameters(stats: dict[str, float]) -> dict[str, float]:
    luminance = stats["luminance_median"]
    saturation = stats["saturation_median"]
    gamma = 1.18 if luminance > 0.50 else 1.10 if luminance > 0.35 else 1.04
    saturation_factor = 1.15 if saturation < 0.25 else 1.08 if saturation < 0.55 else 1.02
    return {"gamma": gamma, "contrast": 1.10, "saturation": saturation_factor}


def adjust_image(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    mode = "RGBA" if "A" in image.getbands() else "RGB"
    source = image.convert(mode)
    before = rgba_statistics(source)
    parameters = correction_parameters(before)
    corrected = ImageEnhance.Color(source).enhance(parameters["saturation"])
    corrected = ImageEnhance.Contrast(corrected).enhance(parameters["contrast"])
    gamma_lut = [int(round(((value / 255.0) ** parameters["gamma"]) * 255.0)) for value in range(256)]
    if mode == "RGBA":
        red, green, blue, alpha = corrected.split()
        corrected = Image.merge("RGBA", (red.point(gamma_lut), green.point(gamma_lut), blue.point(gamma_lut), alpha))
    else:
        corrected = corrected.point(gamma_lut * 3)
    return corrected, {"parameters": parameters, "before": before, "after": rgba_statistics(corrected)}


def replace_embedded_image(gltf: GLTF2, image_index: int, replacement: bytes) -> None:
    image = gltf.images[image_index]
    if image.bufferView is None:
        raise RuntimeError("only bufferView-embedded GLB images are supported")
    blob = bytes(gltf.binary_blob() or b"")
    view_index = image.bufferView
    view = gltf.bufferViews[view_index]
    start = int(view.byteOffset or 0)
    following = sorted(
        int(other.byteOffset or 0)
        for index, other in enumerate(gltf.bufferViews)
        if index != view_index and int(other.byteOffset or 0) > start
    )
    region_end = following[0] if following else len(blob)
    if start + int(view.byteLength) > region_end:
        raise RuntimeError("embedded image bufferView overlaps the next bufferView")
    padded = replacement + b"\0" * ((-len(replacement)) % 4)
    new_blob = blob[:start] + padded + blob[region_end:]
    delta = len(padded) - (region_end - start)
    view.byteLength = len(replacement)
    for index, other in enumerate(gltf.bufferViews):
        if index != view_index and int(other.byteOffset or 0) >= region_end:
            other.byteOffset = int(other.byteOffset or 0) + delta
    gltf.buffers[0].byteLength = len(new_blob)
    gltf.set_binary_blob(new_blob)


def apply_uniform_root_scale(gltf: GLTF2, factor: float, metadata: dict[str, Any]) -> None:
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"invalid uniform scale factor: {factor}")
    scene_index = int(gltf.scene or 0)
    if scene_index >= len(gltf.scenes):
        raise RuntimeError(f"invalid default scene index: {scene_index}")
    roots = list(gltf.scenes[scene_index].nodes or [])
    if not roots:
        raise RuntimeError("GLB default scene has no root nodes")
    if len(roots) == 1 and gltf.nodes[roots[0]].matrix is None:
        node = gltf.nodes[roots[0]]
        existing = node.scale or [1.0, 1.0, 1.0]
        if max(existing) - min(existing) > 1e-8:
            raise RuntimeError(f"source root already has non-uniform scale: {existing}")
        node.scale = [float(value) * factor for value in existing]
        extras = dict(node.extras or {})
        extras["asset4simUniformScale"] = metadata
        node.extras = extras
        return
    wrapper = Node(
        name="Asset4Sim_UniformScale",
        children=roots,
        scale=[factor, factor, factor],
        extras={"asset4simUniformScale": metadata},
    )
    gltf.nodes.append(wrapper)
    gltf.scenes[scene_index].nodes = [len(gltf.nodes) - 1]


def scene_metrics(path: Path) -> dict[str, Any]:
    scene = trimesh.load(path, force="scene", process=False)
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)
    meshes = [mesh for mesh in scene.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
    if not meshes:
        raise RuntimeError(f"no mesh in {path}")
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    return {
        "meshes": len(meshes),
        "vertices": int(sum(len(mesh.vertices) for mesh in meshes)),
        "faces": int(sum(len(mesh.faces) for mesh in meshes)),
        "bounds": bounds.tolist(),
        "extents": (bounds[1] - bounds[0]).tolist(),
    }


def corrected_glb(
    source: Path,
    destination: Path,
    mode: str,
    target_m: float,
    index: int,
    asset_id: str,
) -> dict[str, Any]:
    before_geometry = scene_metrics(source)
    before_extents = np.asarray(before_geometry["extents"], dtype=np.float64)
    source_dimension = float(before_extents[1] if mode == "height_y" else np.max(before_extents))
    if source_dimension <= 1e-9:
        raise RuntimeError(f"source dimension is degenerate for {asset_id}")
    factor = float(target_m / source_dimension)

    gltf = GLTF2().load_binary(str(source))
    if len(gltf.images or []) != 1:
        raise RuntimeError(f"expected one embedded texture, found {len(gltf.images or [])}")
    image = gltf.images[0]
    view = gltf.bufferViews[image.bufferView]
    blob = bytes(gltf.binary_blob() or b"")
    raw = blob[int(view.byteOffset or 0) : int(view.byteOffset or 0) + int(view.byteLength)]
    with Image.open(BytesIO(raw)) as opened:
        corrected_texture, color_report = adjust_image(opened)
        stream = BytesIO()
        corrected_texture.save(stream, format="PNG", optimize=True)
    replace_embedded_image(gltf, 0, stream.getvalue())
    apply_uniform_root_scale(
        gltf,
        factor,
        {"factor": factor, "mode": mode, "targetMeters": target_m, "assetIndex": index},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(destination))

    after_geometry = scene_metrics(destination)
    after_extents = np.asarray(after_geometry["extents"], dtype=np.float64)
    ratios = after_extents / before_extents
    after_dimension = float(after_extents[1] if mode == "height_y" else np.max(after_extents))
    if before_geometry["faces"] != after_geometry["faces"] or before_geometry["vertices"] != after_geometry["vertices"]:
        raise RuntimeError("geometry topology changed during scale/color correction")
    if float(np.max(ratios) - np.min(ratios)) > 1e-4:
        raise RuntimeError(f"non-uniform corrected extents detected: {ratios.tolist()}")
    if not math.isclose(after_dimension, target_m, rel_tol=1e-4, abs_tol=1e-3):
        raise RuntimeError(f"target dimension mismatch: expected {target_m}, got {after_dimension}")
    return {
        "mode": mode,
        "target_m": target_m,
        "source_dimension": source_dimension,
        "uniform_scale_factor": factor,
        "axis_scale_ratios": ratios.tolist(),
        "before_geometry": before_geometry,
        "after_geometry": after_geometry,
        "color_correction": color_report,
        "source_sha256": sha256(source),
        "corrected_sha256": sha256(destination),
    }


def run_logged(command: list[str], log_path: Path, environment: dict[str, str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")


def load_targets(path: Path, expected_count: int) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = {int(item["index"]): item for item in payload["targets"]}
    expected = set(range(1, expected_count + 1))
    if set(targets) != expected:
        raise RuntimeError(f"scale target indices mismatch: missing={sorted(expected-set(targets))}, extra={sorted(set(targets)-expected)}")
    rejected = {index for index, item in targets.items() if item["mode"] == "reject"}
    if rejected != set(REJECTION_REASONS):
        raise RuntimeError(f"rejected index mismatch: config={sorted(rejected)}, expected={sorted(REJECTION_REASONS)}")
    return targets


def process_asset(
    asset: dict[str, Any],
    target: dict[str, Any],
    args: argparse.Namespace,
    environment: dict[str, str],
) -> dict[str, Any]:
    index = int(asset["index"])
    asset_id = asset["asset_id"]
    final_dir = args.output_root / "assets" / f"{index:03d}_{asset_id}"
    build_dir = args.output_root / ".building" / f"{index:03d}_{asset_id}"
    report_path = final_dir / "asset-report.json"
    if final_dir.is_dir() and not args.overwrite:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("passed") is not True:
            raise RuntimeError(f"existing asset report is not passed: {report_path}")
        return report
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    glb_path = build_dir / f"{asset_id}.glb"
    usd_path = build_dir / f"{asset_id}.usd"
    try:
        scale_color = corrected_glb(
            Path(asset["glb_path"]),
            glb_path,
            target["mode"],
            float(target["target_m"]),
            index,
            asset_id,
        )
        run_logged(
            [
                str(args.usd_converter), "-i", str(glb_path), "-o", str(usd_path),
                "--material-type", "preview-surface", "--up-axis", "file",
            ],
            build_dir / "reports" / "usd-convert.log",
            environment,
        )
        material_report = build_dir / "reports" / "usd-material-restore.json"
        run_logged(
            [
                str(args.usd_python), str(args.restore_material_script), str(glb_path), str(usd_path),
                "--report", str(material_report), "--meters-per-unit", "1.0", "--up-axis", "Z",
            ],
            build_dir / "reports" / "usd-material-restore.log",
            environment,
        )
        restored = json.loads(material_report.read_text(encoding="utf-8"))
        if (
            restored.get("passed") is not True
            or not math.isclose(float(restored["meters_per_unit"]), 1.0)
            or restored.get("up_axis") != "Z"
        ):
            raise RuntimeError(f"USD material/unit restoration did not pass: {restored}")
        for path_key in ("source_glb", "usd", "texture"):
            restored_path = Path(restored[path_key])
            try:
                restored[path_key] = str((final_dir / restored_path.relative_to(build_dir)).resolve())
            except ValueError:
                pass
        report = {
            "schema_version": 1,
            "index": index,
            "asset_id": asset_id,
            "source_reference": asset.get("source"),
            "source_glb": asset["glb_path"],
            "source_usd": asset.get("usd_path"),
            "glb": str((final_dir / glb_path.name).resolve()),
            "usd": str((final_dir / usd_path.name).resolve()),
            "texture": str((final_dir / "textures" / f"{asset_id}.png").resolve()),
            "scale_color": scale_color,
            "usd_material_restore": restored,
            "passed": True,
        }
        atomic_json(build_dir / "asset-report.json", report)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(build_dir, final_dir)
        return report
    except Exception:
        atomic_json(build_dir / "FAILED.json", {"index": index, "asset_id": asset_id, "failed": True})
        raise


def run(args: argparse.Namespace) -> int:
    review = json.loads(args.review_report.read_text(encoding="utf-8"))
    assets = review["assets"]
    targets = load_targets(args.scale_targets, len(assets))
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary_root = args.output_root / ".tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["TEMP"] = str(temporary_root.resolve())
    environment["TMP"] = str(temporary_root.resolve())

    rejected: list[dict[str, Any]] = []
    active_reports: list[dict[str, Any]] = []
    for asset in assets:
        index = int(asset["index"])
        target = targets[index]
        if target["mode"] == "reject":
            rejected.append({
                "index": index,
                "asset_id": asset["asset_id"],
                "reason": REJECTION_REASONS[index],
                "source_glb_archive": asset["glb_path"],
                "source_usd_archive": asset.get("usd_path"),
                "review_capture": asset.get("capture_path"),
                "active_library_included": False,
            })
            continue
        report = process_asset(asset, target, args, environment)
        active_reports.append(report)
        print(json.dumps({
            "index": index,
            "asset_id": asset["asset_id"],
            "status": "passed",
            "uniform_scale_factor": report["scale_color"]["uniform_scale_factor"],
            "target_m": report["scale_color"]["target_m"],
        }, ensure_ascii=False), flush=True)

    rejected_manifest = {
        "schema_version": 1,
        "status": "user_rejected",
        "asset_count": len(rejected),
        "archive_policy": "Excluded from the active corrected library; immutable production sources retained only for provenance.",
        "assets": rejected,
    }
    atomic_json(args.output_root / "rejected-assets.json", rejected_manifest)
    active_manifest = {
        "schema_version": 1,
        "status": "prepared",
        "property_assignment_intent": "skip",
        "asset_count": len(active_reports),
        "rejected_count": len(rejected),
        "meters_per_unit": 1.0,
        "scale_policy": "uniform_root_scale_only",
        "color_policy": "adaptive_sRGB_gamma_contrast_saturation",
        "assets": active_reports,
    }
    atomic_json(args.output_root / "active-assets.json", active_manifest)
    print(json.dumps({"active_count": len(active_reports), "rejected_count": len(rejected), "passed": True}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-report", type=Path, required=True)
    parser.add_argument("--scale-targets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--usd-converter", type=Path, required=True)
    parser.add_argument("--usd-python", type=Path, required=True)
    parser.add_argument("--restore-material-script", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

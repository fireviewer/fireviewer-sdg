#!/usr/bin/env python3
"""Apply explicit uniform metric scale targets to a reviewed Asset4Sim library."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from pygltflib import GLTF2, Node


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


def run_logged(command: list[str], log_path: Path) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")


def scene_metrics(path: Path) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    loaded = trimesh.load(path, force="scene", process=False)
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    meshes = [mesh for mesh in scene.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
    if len(meshes) != 1:
        raise RuntimeError(f"expected one mesh in {path}, found {len(meshes)}")
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)) or np.any(extents <= 0):
        raise RuntimeError(f"invalid scene bounds in {path}")
    mesh = meshes[0]
    return mesh, {
        "meshes": 1,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": bounds.tolist(),
        "extents": extents.tolist(),
    }


def texture_pixels(mesh: trimesh.Trimesh) -> np.ndarray:
    texture = getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None)
    if texture is None:
        raise RuntimeError("GLB has no embedded base-color texture")
    return np.asarray(texture.convert("RGBA"), dtype=np.uint8)


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


def scaled_glb(source: Path, destination: Path, target: dict[str, Any], index: int) -> dict[str, Any]:
    source_mesh, before = scene_metrics(source)
    source_pixels = texture_pixels(source_mesh)
    before_extents = np.asarray(before["extents"], dtype=np.float64)
    mode = str(target["mode"])
    target_m = float(target["target_m"])
    source_dimension = float(before_extents[1] if mode == "height_y" else np.max(before_extents))
    if source_dimension <= 1e-9:
        raise RuntimeError(f"source dimension is degenerate for index {index}")
    factor = target_m / source_dimension

    gltf = GLTF2().load_binary(str(source))
    apply_uniform_root_scale(
        gltf,
        factor,
        {"factor": factor, "mode": mode, "targetMeters": target_m, "assetIndex": index},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(destination))

    corrected_mesh, after = scene_metrics(destination)
    after_extents = np.asarray(after["extents"], dtype=np.float64)
    ratios = after_extents / before_extents
    corrected_dimension = float(after_extents[1] if mode == "height_y" else np.max(after_extents))
    if before["faces"] != after["faces"] or before["vertices"] != after["vertices"]:
        raise RuntimeError("topology changed while applying uniform scale")
    if float(np.max(ratios) - np.min(ratios)) > 1e-4:
        raise RuntimeError(f"non-uniform scale detected: {ratios.tolist()}")
    if not math.isclose(corrected_dimension, target_m, rel_tol=1e-4, abs_tol=1e-3):
        raise RuntimeError(f"target mismatch: expected {target_m}, got {corrected_dimension}")
    if not np.array_equal(source_pixels, texture_pixels(corrected_mesh)):
        raise RuntimeError("embedded texture pixels changed while applying scale")
    return {
        "mode": mode,
        "target_m": target_m,
        "source_dimension_m": source_dimension,
        "uniform_scale_factor": factor,
        "axis_scale_ratios": ratios.tolist(),
        "before_geometry": before,
        "after_geometry": after,
        "source_sha256": sha256(source),
        "corrected_sha256": sha256(destination),
        "texture_pixels_unchanged": True,
    }


def load_targets(path: Path, start: int, end: int, rejected: set[int]) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = {int(item["index"]): item for item in payload.get("targets", [])}
    expected = set(range(start, end + 1))
    if set(targets) != expected:
        raise RuntimeError(
            f"scale target indices mismatch: missing={sorted(expected-set(targets))}, extra={sorted(set(targets)-expected)}"
        )
    target_rejected = {index for index, target in targets.items() if target.get("mode") == "reject"}
    if target_rejected != rejected:
        raise RuntimeError(
            f"rejected index mismatch: targets={sorted(target_rejected)}, reviewed={sorted(rejected)}"
        )
    for index, target in targets.items():
        mode = target.get("mode")
        value = target.get("target_m")
        if mode not in {"height_y", "max_extent", "reject"}:
            raise RuntimeError(f"invalid target mode for {index}: {mode}")
        if mode == "reject" and value is not None:
            raise RuntimeError(f"rejected target must have null target_m: {index}")
        if mode != "reject" and (not isinstance(value, (int, float)) or float(value) <= 0):
            raise RuntimeError(f"invalid target_m for {index}: {value}")
    return targets


def final_path(final_root: Path, build_root: Path, path: Path) -> str:
    return str((final_root / path.relative_to(build_root)).resolve())


def process_asset(
    asset: dict[str, Any],
    target: dict[str, Any],
    output_root: Path,
    converter: Path,
    usd_python: Path,
    restore_script: Path,
) -> dict[str, Any]:
    index = int(asset["index"])
    asset_id = str(asset["asset_id"])
    # Keep the published directory short so OpenUSD can reopen long asset
    # filenames on Windows without exceeding its path-length limit.
    final_dir = output_root / "assets" / f"{index:03d}"
    # Keep converter paths short on Windows. Some complete asset IDs push the
    # HOOPS USD writer beyond MAX_PATH even though the final directory itself
    # is valid once moved into place.
    build_dir = output_root / ".building" / f"{index:03d}"
    report_path = final_dir / "asset-report.json"
    if final_dir.is_dir():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("passed") is not True:
            raise RuntimeError(f"existing asset report is not passed: {report_path}")
        if report.get("scale", {}).get("target_m") != target.get("target_m"):
            raise RuntimeError(f"existing asset target differs: {report_path}")
        return report
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    source_glb = Path(asset["glb"]).resolve()
    glb = build_dir / f"{asset_id}.glb"
    usd = build_dir / f"{asset_id}.usd"
    try:
        scale = scaled_glb(source_glb, glb, target, index)
        run_logged(
            [
                str(converter), "-i", str(glb), "-o", str(usd),
                "--material-type", "preview-surface", "--up-axis", "file",
            ],
            build_dir / "reports" / "usd-convert.log",
        )
        restore_report = build_dir / "reports" / "usd-material-restore.json"
        run_logged(
            [
                str(usd_python), str(restore_script), str(glb), str(usd),
                "--report", str(restore_report), "--meters-per-unit", "1.0", "--up-axis", "Z",
            ],
            build_dir / "reports" / "usd-material-restore.log",
        )
        restored = json.loads(restore_report.read_text(encoding="utf-8"))
        if (
            restored.get("passed") is not True
            or not math.isclose(float(restored["meters_per_unit"]), 1.0)
            or restored.get("up_axis") != "Z"
        ):
            raise RuntimeError(f"USD restoration did not pass: {restore_report}")
        for key in ("source_glb", "usd", "texture"):
            restored_path = Path(restored[key])
            try:
                restored[key] = final_path(
                    output_root,
                    output_root,
                    final_dir / restored_path.relative_to(build_dir),
                )
            except ValueError:
                pass
        report = {
            "schema_version": 1,
            "index": index,
            "asset_id": asset_id,
            "source_reference": asset.get("source_reference"),
            "source_glb": str(source_glb),
            "source_usd": asset.get("usd"),
            "glb": final_path(output_root, output_root, final_dir / glb.name),
            "usd": final_path(output_root, output_root, final_dir / usd.name),
            "texture": final_path(output_root, output_root, final_dir / "textures" / f"{asset_id}.png"),
            "review_capture": asset.get("review_capture"),
            "scale": scale,
            "color_policy": "unchanged_from_reviewed_source",
            "usd_material_restore": restored,
            "passed": True,
        }
        atomic_json(build_dir / "asset-report.json", report)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(build_dir, final_dir)
        return report
    except Exception:
        atomic_json(build_dir / "FAILED.json", {"index": index, "asset_id": asset_id, "failed": True})
        raise


def copy_tree_hardlink(source: Path, destination: Path) -> None:
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(path, target)
            except OSError:
                temporary = target.with_suffix(target.suffix + ".tmp")
                shutil.copy2(path, temporary)
                os.replace(temporary, target)


def run(args: argparse.Namespace) -> int:
    source_manifest = json.loads(args.active_manifest.read_text(encoding="utf-8"))
    rejected_manifest = json.loads(args.rejected_manifest.read_text(encoding="utf-8"))
    active_assets = list(source_manifest.get("assets", []))
    rejected_indices = {int(item["index"]) for item in rejected_manifest.get("assets", [])}
    start = int(args.start)
    end = int(args.end)
    expected_active = set(range(start, end + 1)) - rejected_indices
    active_indices = {int(item["index"]) for item in active_assets}
    if active_indices != expected_active:
        raise RuntimeError(
            f"reviewed active indices mismatch: missing={sorted(expected_active-active_indices)}, extra={sorted(active_indices-expected_active)}"
        )
    targets = load_targets(args.scale_targets, start, end, rejected_indices)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    for position, asset in enumerate(active_assets, start=1):
        index = int(asset["index"])
        report = process_asset(
            asset,
            targets[index],
            output_root,
            args.usd_converter.resolve(),
            args.usd_python.resolve(),
            args.restore_material_script.resolve(),
        )
        reports.append(report)
        print(json.dumps({
            "position": position,
            "count": len(active_assets),
            "index": index,
            "asset_id": asset["asset_id"],
            "target_m": targets[index]["target_m"],
            "status": "passed",
        }, ensure_ascii=False), flush=True)

    source_review = args.active_manifest.parent / "review"
    if source_review.is_dir():
        copy_tree_hardlink(source_review, output_root / "review")
    final_active = {
        "schema_version": 1,
        "status": "scaled_pending_independent_validation",
        "property_assignment_intent": "skip",
        "asset_count": len(reports),
        "rejected_count": len(rejected_indices),
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "scale_policy": "explicit_uniform_root_scale_only",
        "color_policy": "unchanged_from_reviewed_source",
        "scale_targets": str(args.scale_targets.resolve()),
        "assets": reports,
    }
    atomic_json(output_root / "active-assets.json", final_active)
    atomic_json(output_root / "rejected-assets.json", rejected_manifest)
    summary = {
        "schema_version": 1,
        "status": "prepared",
        "asset_count": len(reports),
        "rejected_count": len(rejected_indices),
        "start": start,
        "end": end,
        "property_assignment_intent": "skip",
        "scale_policy": "uniform; identical factor on X, Y and Z",
        "color_policy": "unchanged",
        "passed": True,
    }
    atomic_json(output_root / "scale-preparation.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument("--rejected-manifest", type=Path, required=True)
    parser.add_argument("--scale-targets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--usd-converter", type=Path, required=True)
    parser.add_argument("--usd-python", type=Path, required=True)
    parser.add_argument("--restore-material-script", type=Path, required=True)
    parser.add_argument("--start", type=int, default=103)
    parser.add_argument("--end", type=int, default=294)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Independently validate the corrected Asset4Sim GLB/USD library."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import usd_convert_cad  # Initializes the wheel-bundled OpenUSD runtime.
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade


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


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def scene_data(path: Path) -> tuple[trimesh.Scene, trimesh.Trimesh, dict[str, Any]]:
    loaded = trimesh.load(path, force="scene", process=False)
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    meshes = [mesh for mesh in scene.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
    if len(meshes) != 1:
        raise RuntimeError(f"expected one mesh in {path}, found {len(meshes)}")
    mesh = meshes[0]
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)) or np.any(extents <= 0):
        raise RuntimeError(f"invalid scene bounds in {path}")
    return scene, mesh, {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": bounds.tolist(),
        "extents": extents.tolist(),
    }


def rgba_pixels(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def usd_texture(stage: Usd.Stage, usd_path: Path) -> Path:
    textures: list[Path] = []
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != "UsdUVTexture":
            continue
        asset = shader.GetInput("file").Get()
        if asset is None or not asset.path:
            continue
        textures.append((usd_path.parent / asset.path).resolve())
    if len(textures) != 1:
        raise RuntimeError(f"expected one UsdUVTexture, found {len(textures)}")
    return textures[0]


def validate_asset(report: dict[str, Any], assets_root: Path) -> dict[str, Any]:
    index = int(report["index"])
    asset_id = report["asset_id"]
    errors: list[str] = []
    evidence: dict[str, Any] = {}
    try:
        source = Path(report["source_glb"])
        glb = Path(report["glb"])
        usd = Path(report["usd"])
        texture = Path(report["texture"])
        for label, path in (("source GLB", source), ("corrected GLB", glb), ("USD", usd), ("texture", texture)):
            if not path.is_file() or path.stat().st_size <= 0:
                errors.append(f"{label} missing or empty: {path}")
        for label, path in (("corrected GLB", glb), ("USD", usd), ("texture", texture)):
            if not inside(path, assets_root):
                errors.append(f"{label} outside active library: {path}")
        if errors:
            raise RuntimeError("; ".join(errors))

        _, source_mesh, source_metrics = scene_data(source)
        _, corrected_mesh, corrected_metrics = scene_data(glb)
        if source_metrics["faces"] != corrected_metrics["faces"]:
            errors.append("face count changed")
        if source_metrics["vertices"] != corrected_metrics["vertices"]:
            errors.append("vertex count changed")

        source_extents = np.asarray(source_metrics["extents"], dtype=np.float64)
        corrected_extents = np.asarray(corrected_metrics["extents"], dtype=np.float64)
        ratios = corrected_extents / source_extents
        if float(np.max(ratios) - np.min(ratios)) > 1e-4:
            errors.append(f"non-uniform scale ratios: {ratios.tolist()}")
        scale = report.get("scale") or report["scale_color"]
        mode = scale["mode"]
        target_m = float(scale["target_m"])
        glb_dimension = float(corrected_extents[1] if mode == "height_y" else np.max(corrected_extents))
        if not math.isclose(glb_dimension, target_m, rel_tol=1e-4, abs_tol=1e-3):
            errors.append(f"GLB target mismatch: {glb_dimension} != {target_m}")
        if sha256(source) != scale["source_sha256"]:
            errors.append("source GLB hash mismatch")
        if sha256(glb) != scale["corrected_sha256"]:
            errors.append("corrected GLB hash mismatch")

        uv = getattr(corrected_mesh.visual, "uv", None)
        embedded_image = getattr(getattr(corrected_mesh.visual, "material", None), "baseColorTexture", None)
        if uv is None or len(uv) != corrected_metrics["vertices"] or not np.all(np.isfinite(uv)):
            errors.append("corrected GLB UVs missing or invalid")
        if embedded_image is None:
            errors.append("corrected GLB embedded texture missing")
        else:
            with Image.open(texture) as external_image:
                if not np.array_equal(rgba_pixels(embedded_image), rgba_pixels(external_image)):
                    errors.append("GLB embedded texture differs from USD texture")

        stage = Usd.Stage.Open(str(usd))
        if stage is None:
            raise RuntimeError(f"unable to open USD: {usd}")
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not math.isclose(meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1e-12):
            errors.append(f"USD metersPerUnit is {meters_per_unit}, expected 1.0")
        up_axis = UsdGeom.GetStageUpAxis(stage)
        if up_axis != UsdGeom.Tokens.z:
            errors.append(f"USD upAxis is {up_axis}, expected Z")
        usd_meshes = [UsdGeom.Mesh(prim) for prim in stage.TraverseAll() if prim.IsA(UsdGeom.Mesh)]
        if len(usd_meshes) != 1:
            errors.append(f"USD mesh count is {len(usd_meshes)}, expected 1")
        else:
            target_mesh = usd_meshes[0]
            counts = np.asarray(target_mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
            if len(counts) != corrected_metrics["faces"] or not np.all(counts == 3):
                errors.append("USD triangle topology differs from GLB")
            st = UsdGeom.PrimvarsAPI(target_mesh.GetPrim()).GetPrimvar("st")
            values = st.Get() if st else None
            if not values or len(values) != corrected_metrics["faces"] * 3:
                errors.append("USD face-varying UV count is invalid")
            elif st.GetInterpolation() != UsdGeom.Tokens.faceVarying:
                errors.append(f"USD UV interpolation is {st.GetInterpolation()}")
            bound_material, _ = UsdShade.MaterialBindingAPI(target_mesh.GetPrim()).ComputeBoundMaterial()
            if not bound_material or not bound_material.GetPrim().IsValid():
                errors.append("USD mesh has no bound material")
            elif not bound_material.GetPath().HasPrefix(stage.GetDefaultPrim().GetPath()):
                errors.append("USD material is outside the default asset prim")

        authored_texture = usd_texture(stage, usd)
        if authored_texture != texture.resolve() or not authored_texture.is_file():
            errors.append(f"USD texture does not resolve to active texture: {authored_texture}")

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        aligned = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedBox()
        usd_extents = np.asarray(
            [aligned.GetMax()[axis] - aligned.GetMin()[axis] for axis in range(3)],
            dtype=np.float64,
        )
        usd_dimension = float(usd_extents[2] if mode == "height_y" else np.max(usd_extents))
        if not math.isclose(usd_dimension, target_m, rel_tol=1e-4, abs_tol=1e-3):
            errors.append(f"USD world target mismatch: {usd_dimension} != {target_m}")

        evidence = {
            "source_sha256": sha256(source),
            "glb_sha256": sha256(glb),
            "usd_sha256": sha256(usd),
            "texture_sha256": sha256(texture),
            "vertices": corrected_metrics["vertices"],
            "faces": corrected_metrics["faces"],
            "uniform_axis_ratios": ratios.tolist(),
            "target_mode": mode,
            "target_m": target_m,
            "glb_dimension_m": glb_dimension,
            "usd_dimension_m": usd_dimension,
            "usd_extents_m": usd_extents.tolist(),
            "meters_per_unit": meters_per_unit,
            "up_axis": up_axis,
            "usd_texture": str(authored_texture),
        }
    except Exception as exc:
        if not errors:
            errors.append(str(exc))
        elif str(exc) not in errors:
            errors.append(str(exc))
    return {
        "index": index,
        "asset_id": asset_id,
        "passed": not errors,
        "errors": errors,
        "evidence": evidence,
    }


def run(args: argparse.Namespace) -> int:
    active = json.loads(args.active_manifest.read_text(encoding="utf-8"))
    rejected = json.loads(args.rejected_manifest.read_text(encoding="utf-8"))
    assets_root = args.active_manifest.parent / "assets"
    active_indices = [int(item["index"]) for item in active["assets"]]
    rejected_indices = [int(item["index"]) for item in rejected["assets"]]
    expected_rejected = {
        int(value) for value in args.expected_rejected.split(",") if value.strip()
    }
    expected_indices = set(range(args.start, args.end + 1))
    expected_active_count = len(expected_indices - expected_rejected)
    library_errors: list[str] = []
    if len(active_indices) != len(set(active_indices)):
        library_errors.append("duplicate active indices")
    if len(rejected_indices) != len(set(rejected_indices)):
        library_errors.append("duplicate rejected indices")
    if set(rejected_indices) != expected_rejected:
        library_errors.append(f"rejected set mismatch: {sorted(rejected_indices)}")
    if set(active_indices) | set(rejected_indices) != expected_indices:
        library_errors.append(
            f"active/rejected index union is not exactly {args.start:03d}-{args.end:03d}"
        )
    if set(active_indices) & set(rejected_indices):
        library_errors.append("an index is both active and rejected")
    asset_dirs = [path for path in assets_root.iterdir() if path.is_dir()]
    directory_indices = {int(path.name[:3]) for path in asset_dirs}
    if directory_indices != set(active_indices):
        library_errors.append("active directory indices do not match active manifest")
    if directory_indices & expected_rejected:
        library_errors.append("a rejected asset directory exists in the active library")
    building_root = args.active_manifest.parent / ".building"
    if building_root.is_dir() and any(building_root.iterdir()):
        library_errors.append("unfinished building directories remain")

    results: list[dict[str, Any]] = []
    for asset in active["assets"]:
        result = validate_asset(asset, assets_root)
        results.append(result)
        print(json.dumps({"index": result["index"], "passed": result["passed"]}), flush=True)

    passed = sum(item["passed"] for item in results)
    report = {
        "schema_version": 1,
        "active_manifest": str(args.active_manifest.resolve()),
        "rejected_manifest": str(args.rejected_manifest.resolve()),
        "expected_active_count": expected_active_count,
        "expected_rejected_indices": sorted(expected_rejected),
        "asset_directory_count": len(asset_dirs),
        "asset_count": len(results),
        "passed_count": passed,
        "failed_count": len(results) - passed,
        "library_errors": library_errors,
        "passed": passed == expected_active_count and len(results) == expected_active_count and not library_errors,
        "assets": results,
    }
    atomic_json(args.report, report)
    print(json.dumps({key: report[key] for key in ("asset_count", "passed_count", "failed_count", "library_errors", "passed")}), flush=True)
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument("--rejected-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=102)
    parser.add_argument("--expected-rejected", default="26,28,52,85,87,88,100")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

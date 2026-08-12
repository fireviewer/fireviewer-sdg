#!/usr/bin/env python3
"""Restore source GLB UVs and texture on geometry converted by usd-convert-cad."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import usd_convert_cad  # Initializes the wheel-bundled OpenUSD runtime.
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    meshes = [mesh for mesh in loaded.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
    if len(meshes) != 1:
        raise RuntimeError(f"Expected one GLB mesh, found {len(meshes)}")
    mesh = meshes[0]
    uv = getattr(mesh.visual, "uv", None)
    texture = getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None)
    if uv is None or len(uv) != len(mesh.vertices):
        raise RuntimeError("Source GLB has no valid per-vertex UVs")
    if texture is None:
        raise RuntimeError("Source GLB has no embedded base-color texture")
    return mesh


def usd_mesh(stage: Usd.Stage) -> UsdGeom.Mesh:
    # usd-convert-cad authors instanced geometry beneath an abstract
    # ``Prototypes`` class. The default traversal predicate excludes abstract
    # prims, even though that mesh is the geometry used by the live instance.
    meshes = [UsdGeom.Mesh(prim) for prim in stage.TraverseAll() if prim.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise RuntimeError(f"Expected one USD mesh, found {len(meshes)}")
    return meshes[0]


def correspondence_error(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise RuntimeError(f"Corner shape mismatch: {source.shape} != {target.shape}")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    u, singular, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = u @ vt
    denominator = float(np.square(source_centered).sum())
    if denominator <= 0:
        raise RuntimeError("Source geometry has zero extent")
    scale = float(singular.sum() / denominator)
    fitted = source_centered @ rotation * scale + target_center
    diagonal = float(np.linalg.norm(target.max(axis=0) - target.min(axis=0)))
    errors = np.linalg.norm(fitted - target, axis=1) / diagonal
    return {
        "normalized_p95": float(np.quantile(errors, 0.95)),
        "normalized_p99": float(np.quantile(errors, 0.99)),
        "normalized_max": float(errors.max()),
        "similarity_scale": scale,
        "orthogonal_determinant": float(np.linalg.det(rotation)),
    }


def restore(
    source_glb: Path,
    usd_path: Path,
    report_path: Path,
    flip_v: bool,
    meters_per_unit: float | None = None,
    up_axis: str = "Z",
) -> dict[str, Any]:
    mesh = source_mesh(source_glb)
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {usd_path}")
    target_mesh = usd_mesh(stage)
    counts = np.asarray(target_mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(target_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    points = np.asarray(target_mesh.GetPointsAttr().Get(), dtype=np.float64)
    if len(counts) != len(mesh.faces) or not np.all(counts == 3):
        raise RuntimeError(f"Triangle topology mismatch: GLB={len(mesh.faces)}, USD={len(counts)}")
    source_corners = np.asarray(mesh.vertices[mesh.faces], dtype=np.float64).reshape((-1, 3))
    target_corners = points[indices]
    correspondence = correspondence_error(source_corners, target_corners)
    if correspondence["normalized_p99"] > 1e-4:
        raise RuntimeError(f"Triangle ordering/correspondence check failed: {correspondence}")

    uv = np.asarray(mesh.visual.uv, dtype=np.float32)[mesh.faces].reshape((-1, 2))
    if flip_v:
        uv[:, 1] = 1.0 - uv[:, 1]
    primvar = UsdGeom.PrimvarsAPI(target_mesh.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    primvar.Set(Vt.Vec2fArray([Gf.Vec2f(float(value[0]), float(value[1])) for value in uv]))

    texture_dir = usd_path.parent / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    texture_path = texture_dir / f"{source_glb.stem}.png"
    material = mesh.visual.material
    material.baseColorTexture.save(texture_path, format="PNG", optimize=True)

    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise RuntimeError("Converted USD has no valid default prim")
    legacy_looks_path = Sdf.Path("/Asset4SimLooks")
    if stage.GetPrimAtPath(legacy_looks_path).IsValid():
        stage.RemovePrim(legacy_looks_path)
    looks_path = default_prim.GetPath().AppendChild("Looks")
    UsdGeom.Scope.Define(stage, looks_path)
    material_path = looks_path.AppendChild(f"m_{source_glb.stem}_Material")
    usd_material = UsdShade.Material.Define(stage, material_path)
    preview = UsdShade.Shader.Define(stage, material_path.AppendChild("PreviewSurface"))
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    raw_factors = getattr(material, "baseColorFactor", None)
    factors = np.asarray(raw_factors if raw_factors is not None else [1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    if factors.max(initial=1.0) > 1.0:
        factors /= 255.0
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    roughness = getattr(material, "roughnessFactor", None)
    metallic = getattr(material, "metallicFactor", None)
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(1.0 if roughness is None else roughness))
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(0.0 if metallic is None else metallic))
    preview.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(factors[3]))

    st_reader = UsdShade.Shader.Define(stage, material_path.AppendChild("stReader"))
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    texture = UsdShade.Shader.Define(stage, material_path.AppendChild("BaseColorTexture"))
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f"textures/{texture_path.name}"))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    texture.CreateOutput("a", Sdf.ValueTypeNames.Float)
    preview.GetInput("diffuseColor").ConnectToSource(texture.ConnectableAPI(), "rgb")
    preview.GetInput("opacity").ConnectToSource(texture.ConnectableAPI(), "a")
    usd_material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(target_mesh.GetPrim()).Bind(usd_material)
    target_mesh.GetPrim().SetCustomDataByKey("asset4sim:sourceGlbSha256", sha256(source_glb))
    target_mesh.GetPrim().SetCustomDataByKey("asset4sim:uvFlipV", bool(flip_v))
    if meters_per_unit is not None:
        if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
            raise ValueError(f"Invalid meters-per-unit value: {meters_per_unit}")
        UsdGeom.SetStageMetersPerUnit(stage, float(meters_per_unit))
    requested_up_axis = up_axis.upper()
    if requested_up_axis not in {"Y", "Z"}:
        raise ValueError(f"Unsupported up axis: {up_axis}")
    previous_up_axis = UsdGeom.GetStageUpAxis(stage)
    geometry_rotation_degrees = 0.0
    op_name = "xformOp:rotateX:asset4simYUpToZUp"
    root_xform = UsdGeom.Xformable(default_prim)
    root_ops = root_xform.GetOrderedXformOps()
    legacy_root_rotation = next((op for op in root_ops if op.GetOpName() == op_name), None)
    if legacy_root_rotation is not None:
        root_xform.SetXformOpOrder([op for op in root_ops if op.GetOpName() != op_name])
        default_prim.RemoveProperty(op_name)
    needs_y_to_z = previous_up_axis == UsdGeom.Tokens.y or legacy_root_rotation is not None
    geometry_xform = UsdGeom.Xformable(target_mesh.GetPrim().GetParent())
    rotation_op = next((op for op in geometry_xform.GetOrderedXformOps() if op.GetOpName() == op_name), None)
    if requested_up_axis == "Z" and (needs_y_to_z or rotation_op is not None):
        if rotation_op is None:
            rotation_op = geometry_xform.AddRotateXOp(
                UsdGeom.XformOp.PrecisionDouble,
                "asset4simYUpToZUp",
            )
        rotation_op.Set(90.0)
        geometry_rotation_degrees = 90.0
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if requested_up_axis == "Z" else UsdGeom.Tokens.y)
    stage.GetRootLayer().Save()

    reloaded = Usd.Stage.Open(str(usd_path))
    if reloaded is None:
        raise RuntimeError(f"Unable to reopen restored USD: {usd_path}")
    verified_mesh = usd_mesh(reloaded)
    verified_st = UsdGeom.PrimvarsAPI(verified_mesh.GetPrim()).GetPrimvar("st")
    verified_values = verified_st.Get() if verified_st else None
    if not verified_values or len(verified_values) != len(mesh.faces) * 3:
        raise RuntimeError("Restored USD has missing or invalid face-varying UVs")
    if verified_st.GetInterpolation() != UsdGeom.Tokens.faceVarying:
        raise RuntimeError(f"Unexpected restored UV interpolation: {verified_st.GetInterpolation()}")
    bound_material, _ = UsdShade.MaterialBindingAPI(verified_mesh.GetPrim()).ComputeBoundMaterial()
    if not bound_material or not bound_material.GetPrim().IsValid():
        raise RuntimeError("Restored USD has no bound material")
    if not texture_path.is_file():
        raise RuntimeError(f"Restored USD texture is missing: {texture_path}")

    report = {
        "schema_version": 1,
        "source_glb": str(source_glb),
        "source_glb_sha256": sha256(source_glb),
        "usd": str(usd_path),
        "usd_sha256": sha256(usd_path),
        "faces": int(len(mesh.faces)),
        "uv_values": int(len(uv)),
        "flip_v": bool(flip_v),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(reloaded)),
        "up_axis": UsdGeom.GetStageUpAxis(reloaded),
        "source_up_axis": previous_up_axis,
        "root_rotation_x_degrees": 0.0,
        "geometry_rotation_x_degrees": geometry_rotation_degrees,
        "texture": str(texture_path),
        "texture_size": list(material.baseColorTexture.size),
        "correspondence": correspondence,
        "structural_validation": {
            "mesh_count": 1,
            "faces": int(len(mesh.faces)),
            "uv_values": int(len(verified_values)),
            "uv_interpolation": verified_st.GetInterpolation(),
            "material": str(bound_material.GetPath()),
            "texture_sha256": sha256(texture_path),
        },
        "passed": True,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_glb", type=Path)
    parser.add_argument("usd", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--no-flip-v", action="store_true")
    parser.add_argument("--meters-per-unit", type=float)
    parser.add_argument("--up-axis", choices=("Y", "Z"), default="Z")
    args = parser.parse_args()
    report = restore(
        args.source_glb.resolve(),
        args.usd.resolve(),
        args.report.resolve(),
        not args.no_flip_v,
        args.meters_per_unit,
        args.up_axis,
    )
    print(json.dumps({"faces": report["faces"], "texture_size": report["texture_size"], "passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

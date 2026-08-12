#!/usr/bin/env python3
"""Convert validated GLBs to textured OpenUSD with the official NVIDIA converter."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import usd_convert_cad  # noqa: F401 - initializes the wheel-bundled OpenUSD runtime
from pxr import Usd, UsdGeom, UsdShade

from restore_usd_glb_material import restore, sha256


def validate_usd(path: Path, expected_faces: int) -> dict[str, Any]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {path}")
    meshes = [UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise RuntimeError(f"Expected one USD mesh, found {len(meshes)}")
    mesh = meshes[0]
    counts = mesh.GetFaceVertexCountsAttr().Get()
    if len(counts) != expected_faces or any(value != 3 for value in counts):
        raise RuntimeError(f"Triangle topology mismatch: {len(counts)} != {expected_faces}")
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("st")
    st_values = st.Get() if st else None
    if not st_values or len(st_values) != expected_faces * 3:
        raise RuntimeError("Missing or invalid face-varying UVs")
    if st.GetInterpolation() != UsdGeom.Tokens.faceVarying:
        raise RuntimeError(f"Unexpected UV interpolation: {st.GetInterpolation()}")
    material, _ = UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()
    if not material or not material.GetPrim().IsValid():
        raise RuntimeError("USD mesh has no bound material")
    texture_paths: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != "UsdUVTexture":
            continue
        asset = shader.GetInput("file").Get()
        asset_path = getattr(asset, "path", "")
        resolved = (path.parent / asset_path).resolve()
        if not asset_path or not resolved.is_file():
            raise RuntimeError(f"Unresolved USD texture: {asset_path}")
        texture_paths.append(str(resolved))
    if len(texture_paths) != 1:
        raise RuntimeError(f"Expected one USD texture, found {len(texture_paths)}")
    return {
        "mesh_count": 1,
        "faces": len(counts),
        "uv_values": len(st_values),
        "material": str(material.GetPath()),
        "texture": texture_paths[0],
        "texture_sha256": sha256(Path(texture_paths[0])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--converter", type=Path)
    parser.add_argument("--creator", default="FireViewer Asset4Sim Hunyuan3D 2.0")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    reports_dir = args.reports_dir.resolve()
    converter = (args.converter or (Path(sys.prefix) / "Scripts" / "usd-convert-cad.exe")).resolve()
    sources = sorted(input_dir.glob("*.glb"))
    if len(sources) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} GLBs, found {len(sources)}")
    if not converter.is_file():
        raise RuntimeError(f"Official converter not found: {converter}")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        usd_path = output_dir / f"{source.stem}.usd"
        report_path = reports_dir / f"{source.stem}-usd.json"
        print(f"[{index}/{len(sources)}] {source.stem}", flush=True)
        try:
            completed = subprocess.run(
                [
                    str(converter),
                    "-i", str(source),
                    "-o", str(usd_path),
                    "--material-type", "preview-surface",
                    "--instancing-style", "none",
                    "--composition-style", "none",
                    "--creator", args.creator,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            restored = restore(source, usd_path, report_path, flip_v=True)
            structural = validate_usd(usd_path, restored["faces"])
            result = {
                "asset": source.stem,
                "source_glb": str(source),
                "source_glb_sha256": restored["source_glb_sha256"],
                "usd": str(usd_path),
                "usd_sha256": sha256(usd_path),
                "converter": str(converter),
                "converter_stdout_tail": completed.stdout.splitlines()[-10:],
                "restore": restored,
                "structural_validation": structural,
                "passed": True,
            }
        except Exception as exc:  # keep the batch auditable even if one asset fails
            result = {
                "asset": source.stem,
                "source_glb": str(source),
                "usd": str(usd_path),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "passed": False,
            }
        report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        results.append(result)

    passed = sum(bool(item["passed"]) for item in results)
    aggregate = {
        "schema_version": 1,
        "converter": str(converter),
        "usd_convert_cad_version": "0.2.0",
        "asset_count": len(results),
        "passed_count": passed,
        "failed_count": len(results) - passed,
        "passed": passed == len(results),
        "results": results,
    }
    aggregate_path = reports_dir / "usd-conversion-manifest.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: aggregate[key] for key in ("asset_count", "passed_count", "failed_count", "passed")}))
    return 0 if aggregate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

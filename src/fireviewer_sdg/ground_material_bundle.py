"""Build the seven-role 4K PBR ground library used by native terrain authoring.

The source textures are CC0 Poly Haven materials.  Downloads are independent
and resumable so this can run alongside the much larger NVIDIA runtime setup.
The resulting directory follows the installed bundle contract consumed by
``terrain_pbr``; no archive or persistent volume is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from fireviewer_sdg.asset_bundle import (
    INSTALL_MARKER,
    PBR_MATERIAL_ROLES,
    _canonical_sha256,
    _sha256,
)

POLY_HAVEN_API = "https://api.polyhaven.com/files"
USER_AGENT = "FireViewer-Omniverse/1.0"
RESOLUTION = "4k"
TEXTURE_DIMENSION = 4096

ROLE_SOURCES: dict[str, tuple[str, float]] = {
    "forest_floor": ("forest_ground_04", 3.0),
    "grass": ("leafy_grass", 2.0),
    "soil": ("brown_mud_dry", 2.5),
    "rock": ("rocky_terrain_02", 5.0),
    "asphalt": ("asphalt_03", 4.0),
    "gravel": ("gravel_road", 3.0),
    # This is the physically visible bed below the dedicated water surface.
    "water": ("river_small_rocks", 2.0),
}

CHANNELS: dict[str, tuple[str, str, str]] = {
    "base_color": ("Diffuse", "jpg", "sRGB"),
    "normal": ("nor_gl", "exr", "raw"),
    "roughness": ("Rough", "exr", "raw"),
    "displacement": ("Displacement", "png", "raw"),
}


class GroundMaterialBundleError(RuntimeError):
    pass


def _read_json_url(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise GroundMaterialBundleError(f"unexpected JSON payload from {url}")
    return payload


def _download_file(destination: Path, url: str, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_size:
        return
    partial = destination.with_name(f".{destination.name}.partial")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": USER_AGENT}
    if 0 < offset < expected_size:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as stream:
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
    actual = partial.stat().st_size
    if actual != expected_size:
        raise GroundMaterialBundleError(
            f"incomplete material download {destination.name}: "
            f"{actual}/{expected_size} bytes"
        )
    os.replace(partial, destination)


def _record(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _material_usda(
    *,
    role: str,
    texture_names: dict[str, str],
    materialx_name: str,
) -> str:
    return f'''#usda 1.0
(
    defaultPrim = "Material"
    metersPerUnit = 1
)

def Material "Material" (
    customData = {{
        string fireviewer:source = "Poly Haven CC0"
        string fireviewer:sourceMaterialX = "{materialx_name}"
        string fireviewer:terrainRole = "{role}"
    }}
)
{{
    token outputs:surface.connect = </Material/PreviewSurface.outputs:surface>

    def Shader "PreviewSurface"
    {{
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor.connect = </Material/BaseColor.outputs:rgb>
        normal3f inputs:normal.connect = </Material/Normal.outputs:rgb>
        float inputs:roughness.connect = </Material/Roughness.outputs:r>
        token outputs:surface
    }}

    def Shader "TexCoord"
    {{
        uniform token info:id = "UsdPrimvarReader_float2"
        token inputs:varname = "st"
        float2 outputs:result
    }}

    def Shader "BaseColor"
    {{
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @{texture_names["base_color"]}@
        token inputs:sourceColorSpace = "sRGB"
        float2 inputs:st.connect = </Material/TexCoord.outputs:result>
        float3 outputs:rgb
    }}

    def Shader "Normal"
    {{
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @{texture_names["normal"]}@
        token inputs:sourceColorSpace = "raw"
        float2 inputs:st.connect = </Material/TexCoord.outputs:result>
        float3 outputs:rgb
    }}

    def Shader "Roughness"
    {{
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @{texture_names["roughness"]}@
        token inputs:sourceColorSpace = "raw"
        float2 inputs:st.connect = </Material/TexCoord.outputs:result>
        float outputs:r
    }}
}}
'''


def build_ground_material_bundle(
    *,
    output_root: Path,
    workers: int = 8,
    fetch_json: Callable[[str], dict[str, Any]] = _read_json_url,
    download_file: Callable[[Path, str, int], None] = _download_file,
) -> dict[str, object]:
    root = output_root.resolve()
    manifest = root / "manifest-v3.json"
    marker = root / INSTALL_MARKER
    if manifest.is_file() and marker.is_file():
        return {
            "state": "GROUND_PBR_BUNDLE_READY",
            "reused": True,
            "root": str(root),
            "manifest": str(manifest),
        }

    root.mkdir(parents=True, exist_ok=True)
    if workers < 1 or workers > 16:
        raise GroundMaterialBundleError("workers must be between 1 and 16")

    source_payloads = {
        role: fetch_json(f"{POLY_HAVEN_API}/{slug}")
        for role, (slug, _repeat) in ROLE_SOURCES.items()
    }
    jobs: list[tuple[Path, str, int]] = []
    resolved: dict[str, dict[str, tuple[Path, str]]] = {}
    for role, (slug, _repeat) in ROLE_SOURCES.items():
        source = source_payloads[role]
        role_root = root / "materials" / role
        role_textures: dict[str, tuple[Path, str]] = {}
        for texture_role, (api_channel, file_format, _color_space) in CHANNELS.items():
            try:
                entry = source[api_channel][RESOLUTION][file_format]
                url = str(entry["url"])
                size = int(entry["size"])
            except (KeyError, TypeError, ValueError) as exc:
                raise GroundMaterialBundleError(
                    f"{slug} lacks {RESOLUTION} {texture_role}"
                ) from exc
            suffix = Path(urllib.parse.urlparse(url).path).suffix
            destination = role_root / f"{slug}-{texture_role}{suffix}"
            jobs.append((destination, url, size))
            role_textures[texture_role] = (destination, _color_space)

        try:
            mtlx_entry = source["mtlx"][RESOLUTION]["mtlx"]
            mtlx_url = str(mtlx_entry["url"])
            mtlx_size = int(mtlx_entry["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GroundMaterialBundleError(
                f"{slug} lacks its {RESOLUTION} MaterialX graph"
            ) from exc
        mtlx = role_root / f"{slug}-{RESOLUTION}.mtlx"
        jobs.append((mtlx, mtlx_url, mtlx_size))
        role_textures["materialx"] = (mtlx, "raw")
        resolved[role] = role_textures

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(download_file, destination, url, size)
            for destination, url, size in jobs
        ]
        for future in as_completed(futures):
            future.result()

    materials: dict[str, object] = {}
    structural: dict[str, object] = {}
    for role in PBR_MATERIAL_ROLES:
        slug, repeat = ROLE_SOURCES[role]
        paths = resolved[role]
        role_root = root / "materials" / role
        texture_names = {
            texture_role: path.name
            for texture_role, (path, _space) in paths.items()
            if texture_role != "materialx"
        }
        materialx = paths["materialx"][0]
        material = role_root / f"{role}.usda"
        material.write_text(
            _material_usda(
                role=role,
                texture_names=texture_names,
                materialx_name=materialx.name,
            ),
            encoding="utf-8",
        )
        material_record = _record(material, root=root)
        textures: dict[str, object] = {}
        structural_textures: dict[str, object] = {}
        for texture_role in CHANNELS:
            path, color_space = paths[texture_role]
            lock = {
                **_record(path, root=root),
                "width_px": TEXTURE_DIMENSION,
                "height_px": TEXTURE_DIMENSION,
                "color_space": color_space,
            }
            textures[texture_role] = lock
            structural_textures[texture_role] = {
                **lock,
                "color_space": color_space.casefold(),
            }
        material_id = f"fireviewer.pbr.polyhaven.{role}"
        materials[role] = {
            "material_id": material_id,
            "material_file": material_record,
            "material_prim_path": "/Material",
            "metres_per_uv_tile": repeat,
            "textures": textures,
            "source": {
                "provider": "Poly Haven",
                "license": "CC0",
                "asset": slug,
                "materialx": _record(materialx, root=root),
            },
        }
        structural[role] = {
            "material_id": material_id,
            "material_file": material_record["path"],
            "material_file_sha256": material_record["sha256"],
            "material_prim_path": "/Material",
            "metres_per_uv_tile": repeat,
            "textures": structural_textures,
        }

    manifest_payload = {
        "schema_version": 3,
        "state": "GROUND_PBR_MATERIALS_INSTALLED",
        "resolution": RESOLUTION,
        "pbr_materials": materials,
    }
    temporary_manifest = manifest.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest)
    source_contract = {
        role: {"slug": slug, "resolution": RESOLUTION}
        for role, (slug, _repeat) in ROLE_SOURCES.items()
    }
    marker_payload = {
        "schema_version": 1,
        "state": "ASSET_BUNDLE_INSTALLED",
        "bundle_sha256": _canonical_sha256(source_contract),
        "manifest_relative": manifest.relative_to(root).as_posix(),
        "runtime_manifest_sha256": _sha256(manifest),
        "pbr_material_roles": list(PBR_MATERIAL_ROLES),
        "pbr_materials_sha256": _canonical_sha256(structural),
        "license": "CC0",
    }
    temporary_marker = marker.with_name(f"{marker.name}.tmp")
    temporary_marker.write_text(
        json.dumps(marker_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_marker, marker)
    return {
        "state": "GROUND_PBR_BUNDLE_READY",
        "reused": False,
        "root": str(root),
        "manifest": str(manifest),
        "material_count": len(materials),
        "download_count": len(jobs),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and install the seven-role 4K CC0 ground PBR bundle"
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_ground_material_bundle(
        output_root=args.output_root,
        workers=args.workers,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Materialize one already-downloaded photogrammetric scene scan for Z16.

This is deliberately separate from the nine-asset emergency library.  A rural
or village scan is a coherent environment payload, not a building prototype to
repeat across a landscape.  The command copies an existing local source into
the site, converts it with NVIDIA Kit, and records its real composed bounds.
No browser download is initiated and no primitive fallback exists.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


STATE = "Z16_PHOTOGRAMMETRIC_SCENE_SCAN_MATERIALIZED"
SCHEMA_VERSION = 1
_SCAN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


class SceneScanError(RuntimeError):
    """Raised when a local scene scan cannot be materialized faithfully."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp",
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative(source: Path, parent: Path) -> str:
    return PurePosixPath(os.path.relpath(source, parent)).as_posix()


def _validate_source(source: Path, *, repository_root: Path) -> Path:
    candidate = source.expanduser().resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise SceneScanError(f"scene scan source is absent or unsafe: {candidate}")
    try:
        candidate.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise SceneScanError("scene scan source must remain inside the FireViewer repository root") from exc
    if candidate.suffix.lower() not in {".glb", ".gltf"}:
        raise SceneScanError("scene scan source must be a local glTF/GLB file")
    return candidate


async def _convert(*, source: Path, destination: Path) -> None:
    try:
        import omni.kit.app
        import omni.kit.asset_converter as asset_converter
    except ImportError as exc:  # pragma: no cover - requires NVIDIA Kit
        raise SceneScanError("scene scan conversion requires an active NVIDIA Kit runtime") from exc
    app = omni.kit.app.get_app()
    manager = app.get_extension_manager()
    if not manager.is_extension_enabled("omni.kit.asset_converter"):
        manager.set_extension_enabled_immediate("omni.kit.asset_converter", True)
    converter = asset_converter.get_instance()
    if converter is None:
        raise SceneScanError("NVIDIA Kit Asset Converter is unavailable")
    context = asset_converter.AssetConverterContext()
    for name, value in {
        "ignore_materials": False,
        "ignore_animation": True,
        "ignore_animations": True,
        "ignore_cameras": False,
        "ignore_lights": False,
        "merge_all_meshes": False,
        "single_mesh": False,
        "smooth_normals": True,
        "export_preview_surface": True,
        "embed_textures": True,
        "use_meter_as_world_unit": True,
        "create_world_as_default_root_prim": True,
        "convert_stage_up_z": True,
        "baking_scales": True,
    }.items():
        if hasattr(context, name):
            setattr(context, name, value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    task = converter.create_converter_task(
        str(source), str(destination), lambda _current, _total: True, context
    )
    try:
        completed = bool(await asyncio.wait_for(task.wait_until_finished(), timeout=1800.0))
    except TimeoutError as exc:
        task.cancel()
        raise SceneScanError(f"NVIDIA Kit conversion timed out for {source.name}") from exc
    if not completed or not destination.is_file() or destination.stat().st_size == 0:
        detail = ""
        getter = getattr(task, "get_error_message", None)
        if callable(getter):
            detail = str(getter() or "").strip()
        raise SceneScanError(
            f"NVIDIA Kit conversion failed for {source.name}" + (f": {detail}" if detail else "")
        )


def _write_wrapper(*, wrapper: Path, source_usdc: Path) -> None:
    try:
        from pxr import Usd, UsdGeom  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - requires NVIDIA Kit
        raise SceneScanError("OpenUSD bindings are unavailable in this Kit runtime") from exc
    stage = Usd.Stage.CreateNew(str(wrapper))
    if stage is None:
        raise SceneScanError(f"could not create wrapper: {wrapper}")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/EnvironmentScan")
    root.GetPrim().GetReferences().AddReference(_relative(source_usdc, wrapper.parent))
    root.GetPrim().SetCustomDataByKey("fireviewer:role", "photogrammetric_environment_scan")
    root.GetPrim().SetCustomDataByKey("fireviewer:primitive_substitution", "forbidden")
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()


def _measure_wrapper(wrapper: Path) -> dict[str, Any]:
    try:
        from pxr import Usd, UsdGeom  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - requires NVIDIA Kit
        raise SceneScanError("OpenUSD bindings are unavailable in this Kit runtime") from exc
    stage = Usd.Stage.Open(str(wrapper))
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise SceneScanError("converted scene scan cannot be reopened")
    root = stage.GetDefaultPrim()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bounds = cache.ComputeWorldBound(root).ComputeAlignedRange()
    minimum = [float(value) for value in bounds.GetMin()]
    maximum = [float(value) for value in bounds.GetMax()]
    dimensions = [round(maximum[index] - minimum[index], 6) for index in range(3)]
    if any(value <= 0.0 for value in dimensions):
        raise SceneScanError("converted scene scan has empty world bounds")
    return {
        "default_prim": root.GetPath().pathString,
        "prim_count": sum(1 for _ in stage.TraverseAll()),
        "world_bounds": {"minimum": minimum, "maximum": maximum, "dimensions": dimensions},
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
    }


async def materialize(*, site_root: Path, source: Path, scan_id: str) -> dict[str, Any]:
    if not _SCAN_ID.fullmatch(scan_id):
        raise SceneScanError("scan identifier must be lowercase letters, digits and hyphens")
    root = site_root.expanduser().resolve()
    repository_root = Path(__file__).resolve().parents[3]
    source = _validate_source(source, repository_root=repository_root)
    output_root = root / "assets" / "scene-scans" / scan_id
    if output_root.exists() or output_root.is_symlink():
        raise SceneScanError(f"refusing to overwrite existing scene scan output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{scan_id}-", dir=output_root.parent))
    try:
        staged_source = staging_root / f"source{source.suffix.lower()}"
        shutil.copy2(source, staged_source)
        converted = staging_root / "source.usdc"
        await _convert(source=staged_source, destination=converted)
        wrapper = staging_root / "asset.usda"
        _write_wrapper(wrapper=wrapper, source_usdc=converted)
        validation = _measure_wrapper(wrapper)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "state": STATE,
            "scan_id": scan_id,
            "source_kind": "already_downloaded_local_photogrammetric_scene",
            "source": {
                "cached_repository_path": source.relative_to(repository_root).as_posix(),
                "sha256": _sha256(staged_source),
                "bytes": staged_source.stat().st_size,
            },
            "materialized": {
                "wrapper": "asset.usda",
                "wrapper_sha256": _sha256(wrapper),
                "layer": "source.usdc",
                "layer_sha256": _sha256(converted),
            },
            "validation": validation,
            "allowed_scene_usage": ["single coherent environment island", "review-stage payload"],
            "forbidden_scene_usage": ["stretched 4km terrain replacement", "repeated building prototype", "primitive fallback"],
            "next_required_step": "editor_semantic_and_visual_qc_before_composition",
        }
        _write_json(staging_root / "materialization-receipt.json", receipt)
        os.replace(staging_root, output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return {
        "state": STATE,
        "scan_id": scan_id,
        "receipt": (output_root / "materialization-receipt.json").relative_to(root).as_posix(),
        "wrapper": (output_root / "asset.usda").relative_to(root).as_posix(),
        "world_dimensions_m": validation["world_bounds"]["dimensions"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    environment_site_root = os.getenv("FW_SDG_SITE_ROOT", "").strip()
    environment_source = os.getenv("FW_SDG_SCENE_SCAN_SOURCE", "").strip()
    environment_scan_id = os.getenv("FW_SDG_SCENE_SCAN_ID", "").strip()
    parser.add_argument(
        "--site-root",
        required=not bool(environment_site_root),
        default=Path(environment_site_root) if environment_site_root else None,
        type=Path,
    )
    parser.add_argument(
        "--source",
        required=not bool(environment_source),
        default=Path(environment_source) if environment_source else None,
        type=Path,
    )
    parser.add_argument(
        "--scan-id",
        required=not bool(environment_scan_id),
        default=environment_scan_id or None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import omni.kit.app
        import omni.kit.async_engine
    except ImportError:
        payload = asyncio.run(materialize(site_root=args.site_root, source=args.source, scan_id=args.scan_id))
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    async def run_from_kit() -> None:
        exit_code = 0
        try:
            payload = await materialize(site_root=args.site_root, source=args.source, scan_id=args.scan_id)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        except Exception:
            exit_code = 1
            raise
        finally:
            omni.kit.app.get_app().post_quit(exit_code)

    omni.kit.async_engine.run_coroutine(run_from_kit())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

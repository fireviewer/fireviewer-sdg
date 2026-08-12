#!/usr/bin/env python3
"""Materialize the recovered Z16 source assets into directly referenceable USD.

The input receipt records the exact source archives recovered from the local
Sketchfab intake cache.  This command is intentionally a source-preparation
step: it does not alter the historical pod-only wrapper lock or claim that
native MID/FAR LODs have been generated.  Existing USDZ sources are referenced
from their unpacked USD layer; the one GLB source is converted with NVIDIA's
bundled Asset Converter before receiving the same wrapper structure.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


STATE = "Z16_RECOVERED_ASSETS_MATERIALIZED"
SCHEMA_VERSION = 1
RECOVERY_RECEIPT_NAME = "recovery-receipt.json"
MATERIALIZATION_RECEIPT_NAME = "materialization-receipt.json"


class MaterializationError(RuntimeError):
    """Raised when a recovered source cannot be materialized safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"invalid JSON receipt: {path}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"receipt must contain an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as temporary:
        json.dump(value, temporary, ensure_ascii=False, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _safe_id(raw: object) -> str:
    if not isinstance(raw, str) or len(raw) != 32:
        raise MaterializationError(f"asset identifier is invalid: {raw!r}")
    if any(character not in "0123456789abcdef" for character in raw.lower()):
        raise MaterializationError(f"asset identifier is invalid: {raw!r}")
    return raw.lower()


def _relative_asset_path(source: Path, wrapper_parent: Path) -> str:
    return PurePosixPath(os.path.relpath(source, wrapper_parent)).as_posix()


def _ensure_absent(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise MaterializationError(
            f"refusing to overwrite existing materialized output: {path}"
        )


def _set_context_option(context: object, name: str, value: object) -> None:
    if hasattr(context, name):
        setattr(context, name, value)


async def _convert_glb(*, source: Path, destination: Path) -> None:
    """Convert the recovered GLB through an active NVIDIA Kit runtime.

    Calling the private converter DLL through an ordinary Python process leaves
    its Kit dependencies unresolved.  The supported API is exposed by the
    ``omni.kit.asset_converter`` extension after Kit has initialized it.
    """

    try:
        import omni.kit.app
        import omni.kit.asset_converter as asset_converter
    except ImportError as exc:  # pragma: no cover - requires Kit
        raise MaterializationError(
            "GLB conversion requires an active FireViewer Kit runtime with "
            "omni.kit.asset_converter enabled"
        ) from exc

    app = omni.kit.app.get_app()
    manager = app.get_extension_manager()
    if not manager.is_extension_enabled("omni.kit.asset_converter"):
        manager.set_extension_enabled_immediate("omni.kit.asset_converter", True)
    converter = asset_converter.get_instance()
    if converter is None:
        raise MaterializationError("Kit Asset Converter instance is unavailable")

    destination.parent.mkdir(parents=True, exist_ok=True)
    context = asset_converter.AssetConverterContext()
    for name, value in {
        "ignore_materials": False,
        "ignore_animation": True,
        "ignore_animations": True,
        "ignore_cameras": True,
        "ignore_lights": True,
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
        _set_context_option(context, name, value)
    task = converter.create_converter_task(
        str(source), str(destination), lambda _current, _total: True, context
    )
    try:
        converted = bool(
            await asyncio.wait_for(task.wait_until_finished(), timeout=1800.0)
        )
    except TimeoutError as exc:
        task.cancel()
        raise MaterializationError(
            f"Kit Asset Converter timed out for {source.name}"
        ) from exc
    if not converted or not destination.is_file() or destination.stat().st_size == 0:
        detail = ""
        getter = getattr(task, "get_error_message", None)
        if callable(getter):
            detail = str(getter() or "").strip()
        raise MaterializationError(
            f"Kit Asset Converter failed for {source.name}"
            + (f": {detail}" if detail else "")
        )


def _write_reference_wrapper(*, destination: Path, source_layer: Path) -> None:
    try:
        from pxr import Usd, UsdGeom  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MaterializationError(
            "OpenUSD Python bindings are unavailable; launch through the "
            "FireViewer Kit Python runtime."
        ) from exc

    _ensure_absent(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(destination))
    if stage is None:
        raise MaterializationError(f"could not create USD wrapper: {destination}")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    root.GetPrim().GetReferences().AddReference(
        _relative_asset_path(source_layer, destination.parent)
    )
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()


def _validate_wrapper_reference(*, wrapper: Path, source_layer: Path) -> None:
    """Require a resumable wrapper to point to the exact recovered layer."""

    try:
        source_text = wrapper.read_text(encoding="utf-8")
    except OSError as exc:
        raise MaterializationError(f"could not read generated wrapper: {wrapper}") from exc
    match = re.search(r"references\s*=\s*@([^@]+)@", source_text)
    if match is None:
        raise MaterializationError(
            f"generated wrapper does not declare a source reference: {wrapper}"
        )
    reference = match.group(1).replace("/", os.sep).replace("\\", os.sep)
    resolved_reference = (wrapper.parent / reference).resolve()
    if resolved_reference != source_layer.resolve():
        raise MaterializationError(
            f"generated wrapper reference does not match recovered source: {wrapper}"
        )


def _validate_wrapper(path: Path) -> dict[str, Any]:
    try:
        from pxr import Usd, UsdGeom  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MaterializationError(
            "OpenUSD Python bindings are unavailable; launch through the "
            "FireViewer Kit Python runtime."
        ) from exc

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise MaterializationError(f"could not open generated USD wrapper: {path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise MaterializationError(f"generated wrapper has no valid default prim: {path}")
    prim_count = sum(1 for _ in stage.TraverseAll())
    if prim_count < 1:
        raise MaterializationError(f"generated wrapper has no composed prims: {path}")
    return {
        "default_prim": default_prim.GetPath().pathString,
        "prim_count": prim_count,
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
    }


async def materialize(*, site_root: Path) -> dict[str, Any]:
    root = site_root.resolve()
    recovery_root = root / "assets" / "source-recovery"
    receipt_path = recovery_root / RECOVERY_RECEIPT_NAME
    receipt = _read_json(receipt_path)
    if receipt.get("state") != "Z16_EXACT_ASSET_SOURCES_RECOVERED":
        raise MaterializationError(
            "recovery receipt is not the expected exact-source recovery state"
        )
    raw_assets = receipt.get("assets")
    if not isinstance(raw_assets, list) or len(raw_assets) != 9:
        raise MaterializationError("recovery receipt must contain exactly nine assets")

    output_root = root / "assets" / "materialized"
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, dict):
            raise MaterializationError("recovery asset entry must be an object")
        asset_id = _safe_id(raw_asset.get("id"))
        if asset_id in seen_ids:
            raise MaterializationError(f"duplicate recovered asset id: {asset_id}")
        seen_ids.add(asset_id)
        role = raw_asset.get("role")
        if role not in {"actor", "vegetation", "building"}:
            raise MaterializationError(f"invalid recovered asset role: {role!r}")
        source_file = raw_asset.get("source_file")
        if source_file not in {"source.usdz", "source.glb"}:
            raise MaterializationError(f"unsupported recovered source file: {source_file!r}")
        recovered_asset_root = recovery_root / asset_id
        original_source = recovered_asset_root / source_file
        expected_sha = raw_asset.get("source_sha256")
        if not isinstance(expected_sha, str) or _sha256(original_source) != expected_sha:
            raise MaterializationError(f"recovered source hash drifted: {asset_id}")

        materialized_root = output_root / asset_id
        wrapper = materialized_root / "asset.usda"
        if source_file == "source.usdz":
            source_layer = recovered_asset_root / "source" / "scene.usdc"
            if not source_layer.is_file():
                raise MaterializationError(
                    f"unpacked USD layer is missing for recovered asset: {asset_id}"
                )
            conversion = "existing_usdz_layer_referenced"
        else:
            source_layer = materialized_root / "native" / "scene.usdc"
            if source_layer.is_file():
                _validate_wrapper(source_layer)
                conversion = "existing_native_glb_conversion_revalidated"
            else:
                _ensure_absent(source_layer)
                await _convert_glb(source=original_source, destination=source_layer)
                conversion = "nvidia_asset_converter_glb_to_usdc"

        if wrapper.is_file():
            _validate_wrapper_reference(wrapper=wrapper, source_layer=source_layer)
            wrapper_state = "existing_wrapper_revalidated"
        else:
            _ensure_absent(wrapper)
            _write_reference_wrapper(destination=wrapper, source_layer=source_layer)
            wrapper_state = "new_wrapper"
        validation = _validate_wrapper(wrapper)
        results.append(
            {
                "asset_id": asset_id,
                "role": role,
                "source": _relative_asset_path(original_source, root),
                "source_sha256": expected_sha,
                "conversion": conversion,
                "wrapper_state": wrapper_state,
                "wrapper": _relative_asset_path(wrapper, root),
                "wrapper_sha256": _sha256(wrapper),
                "validation": validation,
                "lod_state": "HERO_SOURCE_READY_MID_FAR_NOT_YET_AUTHORED",
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "state": STATE,
        "site_root": str(root),
        "asset_count": len(results),
        "asset_roles": {
            "actors": sum(item["role"] == "actor" for item in results),
            "vegetation": sum(item["role"] == "vegetation" for item in results),
            "buildings": sum(item["role"] == "building" for item in results),
        },
        "historical_pod_wrapper_lock_replaced": False,
        "next_required_step": "author_native_mid_far_lods_then_build_compact_sim_01",
        "assets": results,
    }
    _write_json(output_root / MATERIALIZATION_RECEIPT_NAME, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    environment_site_root = os.getenv("FW_SDG_SITE_ROOT", "").strip()
    parser.add_argument(
        "--site-root",
        required=not bool(environment_site_root),
        default=Path(environment_site_root) if environment_site_root else None,
        type=Path,
        help=(
            "Z16 site root; when launched through Kit --exec this may be "
            "provided as FW_SDG_SITE_ROOT"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import omni.kit.app
        import omni.kit.async_engine
    except ImportError:
        payload = asyncio.run(materialize(site_root=args.site_root))
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    async def run_from_kit() -> None:
        exit_code = 0
        try:
            payload = await materialize(site_root=args.site_root)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        except Exception:
            exit_code = 1
            raise
        finally:
            omni.kit.app.get_app().post_quit(exit_code)

    omni.kit.async_engine.run_coroutine(run_from_kit())
    return 0


if __name__ == "__main__":
    main()

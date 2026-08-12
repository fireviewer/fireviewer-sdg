"""Isaac Sim, Replicator and Flow capability gate."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _enabled(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"1", "true", "yes", "on"})


def _write_preflight_receipt(result: dict[str, object]) -> Path | None:
    """Persist optional runtime evidence without allowing an arbitrary write."""

    configured = os.getenv("FW_SDG_GPU_PREFLIGHT_RECEIPT", "").strip()
    if not configured:
        return None
    volume_root = Path(
        os.getenv("FW_SDG_VOLUME_ROOT", "/workspace/fireviewer-sdg")
    ).resolve()
    receipt = Path(configured).resolve()
    try:
        receipt.relative_to(volume_root)
    except ValueError as exc:
        raise RuntimeError(
            "GPU preflight receipt must be written inside FW_SDG_VOLUME_ROOT"
        ) from exc
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_suffix(f"{receipt.suffix}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt)
    return receipt


def _prepare_official_asset_lock() -> dict[str, object] | None:
    if not _enabled(os.getenv("FW_SDG_PREPARE_IGN_CATALOG")):
        return None
    if os.getenv("FW_SDG_SIMREADY_ASSET_MANIFEST", "").strip():
        return {"state": "manual_manifest_override"}
    volume_root = Path(
        os.getenv("FW_SDG_VOLUME_ROOT", "/workspace/fireviewer-sdg")
    ).resolve()
    manifest_path = volume_root / "input" / "simready-assets-hd-v2.json"
    if manifest_path.is_file():
        try:
            locked = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            locked = {}
        discovery = (
            locked.get("discovery", {}) if isinstance(locked, dict) else {}
        )
        if (
            isinstance(discovery, dict)
            and discovery.get("mode")
            == "official_nvidia_materialized_lock_v2"
        ):
            return {
                "state": "existing",
                "manifest": str(manifest_path),
            }
    from fireviewer_sdg.config import Settings
    from fireviewer_sdg.simready_assets import (
        cache_official_nvidia_indexes,
        provision_official_nvidia_manifest,
    )

    settings = Settings.from_environment()
    if _enabled(os.getenv("FW_SDG_CACHE_NVIDIA_INDEXES_ONLY")):
        cached = cache_official_nvidia_indexes(
            volume_root=volume_root,
            asset_root=settings.nvidia_asset_root,
        )
        return {
            "state": "indexes_cached",
            "index_lock": str(cached["index_lock"]),
            "indexes": len(cached["indexes"]),
        }
    result = provision_official_nvidia_manifest(
        volume_root=volume_root,
        manifest_path=manifest_path,
        asset_root=settings.nvidia_asset_root,
    )
    return {
        "state": "prepared",
        "manifest": str(result["manifest"]),
        "candidate_count": int(result["candidate_count"]),
        "missing_environment": list(result["missing_environment"]),
        "missing_actor_classes": list(result["missing_actor_classes"]),
    }


def _prepare_input_catalog() -> dict[str, object] | None:
    if (
        not _enabled(os.getenv("FW_SDG_PREPARE_IGN_CATALOG"))
        or _enabled(os.getenv("FW_SDG_CACHE_NVIDIA_INDEXES_ONLY"))
    ):
        return None
    from fireviewer_sdg.config import Settings
    from fireviewer_sdg.ign_catalog import prepare_ign_catalog
    from fireviewer_sdg.production import load_production_plan

    settings = Settings.from_environment()
    production = load_production_plan(settings.campaign_path)
    result = prepare_ign_catalog(
        settings.volume_root,
        runtime_root=Path(sys.prefix),
        asset_manifest=settings.simready_asset_manifest,
        nvidia_asset_root=settings.nvidia_asset_root,
        include_response_assets=(
            "response_engagement" in production["active_categories"]
        ),
    )
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
        if key
        in {
            "state",
            "catalog",
            "fire_events",
            "site_count",
            "production_scope",
            "bulk_allowed",
            "readiness_report",
        }
    }


def _validate_rtx_rgb(
    data: Any,
    *,
    resolution: tuple[int, int],
) -> dict[str, object]:
    """Reject an Isaac start that has no functioning RTX render output.

    Importing Isaac and registering a Replicator writer is not enough: Kit can
    finish its startup while Vulkan has failed to create a GPU device.  The
    probe therefore requires a non-uniform RGB buffer from a lit USD cube.
    """

    width, height = resolution
    shape = tuple(int(value) for value in getattr(data, "shape", ()))
    if len(shape) != 3 or shape[0] != height or shape[1] != width or shape[2] < 3:
        raise RuntimeError(
            "RTX probe returned an invalid RGB buffer shape "
            f"{shape}; expected ({height}, {width}, 3+)."
        )
    size = int(getattr(data, "size", 0))
    if size < width * height * 3:
        raise RuntimeError("RTX probe returned an empty RGB buffer")
    minimum = float(data.min())
    maximum = float(data.max())
    if not maximum > minimum:
        raise RuntimeError("RTX probe rendered a uniform RGB buffer")
    return {
        "resolution": [width, height],
        "shape": list(shape),
        "minimum": minimum,
        "maximum": maximum,
    }


def _probe_rtx_render(rep: Any) -> dict[str, object]:
    """Render a tiny lit USD scene and return evidence of a real RTX buffer."""

    import carb.settings
    import omni.usd

    resolution = (64, 64)
    context = omni.usd.get_context()
    context.new_stage()
    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Keep the smoke scene on the supported Replicator functional API.  The
    # probe intentionally has no writer: it validates the pixels in memory,
    # and wait_until_complete() is only for flushing writer backends.
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(
        intensity=500,
        parent="/World",
        name="RtxProbeLight",
    )
    cube = rep.functional.create.cube(parent="/World", name="RtxProbeCube")
    rep.functional.modify.position(cube, (0.0, 0.0, 1.0))
    camera = rep.functional.create.camera(
        position=(5.0, 5.0, 5.0),
        look_at=(0.0, 0.0, 1.0),
        parent="/World",
        name="RtxProbeCamera",
    )
    product = rep.create.render_product(
        camera, resolution, name="RtxProbeRenderProduct"
    )
    rgb = rep.annotators.get("rgb")
    rgb.attach(product)
    try:
        rep.orchestrator.step(delta_time=0.0, rt_subframes=1)
        return _validate_rtx_rgb(rgb.get_data(), resolution=resolution)
    finally:
        rgb.detach()
        product.destroy()


def run_probe() -> dict[str, object]:
    import isaacsim
    from isaacsim.simulation_app import SimulationApp

    SimulationApp({"headless": True})
    import omni.kit.app
    import omni.replicator.core as rep

    manager = omni.kit.app.get_app().get_extension_manager()
    flow_enabled = manager.set_extension_enabled_immediate("omni.flowusd", True)
    if not manager.is_extension_enabled("omni.flowusd"):
        raise RuntimeError("omni.flowusd could not be enabled")
    rtx_render = _probe_rtx_render(rep)
    asset_lock = _prepare_official_asset_lock()
    input_catalog = _prepare_input_catalog()
    result = {
        "isaac_sim": True,
        "replicator": True,
        "flow": True,
        "flow_enable_result": bool(flow_enabled),
        "rtx_render": rtx_render,
        "asset_lock": asset_lock,
        "input_catalog": input_catalog,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    # This capability check intentionally owns a disposable child process.
    # Isaac/Flow can abort or hang while destroying live Kit task groups even
    # with skip_cleanup=True. Exit directly only after every gate above passed;
    # failures keep a non-zero status and their traceback without entering Kit
    # shutdown or Python finalization.
    try:
        result = run_probe()
        _write_preflight_receipt(result)
    except BaseException as exc:
        if _enabled(os.getenv("FW_SDG_PREPARE_IGN_CATALOG")):
            try:
                from fireviewer_sdg.preparation_progress import (
                    load_progress,
                    write_progress,
                )

                volume = Path(
                    os.getenv(
                        "FW_SDG_VOLUME_ROOT",
                        "/workspace/fireviewer-sdg",
                    )
                ).resolve()
                previous = load_progress(volume) or {}
                write_progress(
                    volume,
                    phase=str(previous.get("phase") or "input_preparation"),
                    state="blocked",
                    message=f"{type(exc).__name__}: {exc}",
                    error_type=type(exc).__name__,
                )
            except BaseException:
                pass
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)

"""Isaac Sim, Replicator and Flow capability gate."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def _enabled(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"1", "true", "yes", "on"})


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
    if rep.WriterRegistry.get("BasicWriter") is None:
        raise RuntimeError("Replicator BasicWriter is unavailable")
    asset_lock = _prepare_official_asset_lock()
    input_catalog = _prepare_input_catalog()
    result = {
        "isaac_sim": True,
        "replicator": True,
        "flow": True,
        "flow_enable_result": bool(flow_enabled),
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
        run_probe()
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

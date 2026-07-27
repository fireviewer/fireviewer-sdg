"""Fail-closed boot sequence for the FireViewer SDG image."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from fireviewer_sdg.config import Settings
from fireviewer_sdg.production import ProductionManager, load_production_plan
from fireviewer_sdg.provisioning import provision
from fireviewer_sdg.event_catalog import load_event_catalog
from fireviewer_sdg.preparation_progress import load_progress, write_progress
from fireviewer_sdg.review_store import CaseStore
from fireviewer_sdg.serve import serve
from fireviewer_sdg.storage import assert_storage_architecture


DIRECTORIES = (
    "provision/manifests",
    "provision/locks",
    "provision/receipts",
    "models",
    "assets",
    "scenes",
    "cache/isaac",
    "cache/shaders",
    "runs",
    "quarantine",
    "exports",
    "logs",
    "input",
    "training/releases",
)


def _prepare_volume(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for relative in DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)
    marker = root / ".write-test"
    marker.write_text("ok\n", encoding="ascii")
    marker.unlink()


def _run_compatibility_checker() -> None:
    checker = Path(sys.executable).parent / "isaacsim"
    if sys.platform == "win32":
        checker = checker.with_suffix(".exe")
    if not checker.is_file():
        raise RuntimeError("Isaac Sim compatibility checker entrypoint is absent")
    command = [
        str(checker),
        "isaacsim.exp.compatibility_check",
        "--/app/quitAfter=10",
        "--no-window",
    ]
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        command.append("--allow-root")
    subprocess.run(
        command,
        check=True,
        timeout=180,
    )


def _run_probe(*, prepare_assets: bool = False) -> None:
    environment = os.environ.copy()
    environment["FW_SDG_PREPARE_IGN_CATALOG"] = (
        "1" if prepare_assets else "0"
    )
    subprocess.run(
        [sys.executable, "-m", "fireviewer_sdg.isaac_probe"],
        check=True,
        timeout=1800 if prepare_assets else 240,
        env=environment,
    )


def _set_setup_stage(
    status: dict[str, object],
    stage: str,
    *,
    state: str,
    detail: str,
) -> None:
    setup = dict(status.get("setup", {}))
    setup[stage] = {"state": state, "detail": detail}
    status["setup"] = setup


def _load_real_world_status(
    *,
    settings: Settings,
    production: dict[str, object],
) -> dict[str, object]:
    try:
        catalog = load_event_catalog(
            Path(str(production["real_world_catalog"])),
            volume_root=settings.volume_root,
            target_per_category=int(production["target_per_category"]),
        )
    except (OSError, ValueError) as exc:
        return {
            "state": "blocked",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "state": "ready",
        "catalog": str(catalog["catalog_path"]),
        "fire_events": catalog["coverage"]["fire_events"],
        "fire_duration_days": catalog["coverage"]["fire_duration_days"],
        "case_slots_per_category": catalog["coverage"][
            "case_slots_per_category"
        ],
        "progression_profiles": catalog["coverage"][
            "progression_profiles"
        ],
    }


def _prepare_inputs_in_background(
    *,
    settings: Settings,
    production: dict[str, object],
    status: dict[str, object],
) -> None:
    try:
        catalog_path = Path(str(production["real_world_catalog"]))
        if not catalog_path.is_file():
            _set_setup_stage(
                status,
                "asset_lock",
                state="running",
                detail=(
                    "Inventaire officiel NVIDIA en cours dans Isaac; "
                    "aucun asset générique n'est accepté."
                ),
            )
            status["input_preparation"] = {
                "state": "preparing",
                "phase": "isaac_input_preparation",
                "reason": (
                    "Inventaire, scènes IGN et validation USD exécutés dans "
                    "le processus Isaac."
                ),
            }
            _run_probe(prepare_assets=True)
        _set_setup_stage(
            status,
            "asset_lock",
            state="ready",
            detail="Lockfile NVIDIA ou manifeste revu disponible.",
        )
        _set_setup_stage(
            status,
            "terrain_catalog",
            state="running",
            detail="Préparation des trois sites IGN et des 512 contrats.",
        )
        status["input_preparation"] = {
            "state": "preparing",
            "phase": "ign_terrain_and_event_catalog",
            "reason": "Préparation des terrains et contrats en cours.",
        }
        from fireviewer_sdg.ign_catalog import prepare_ign_catalog

        preparation = prepare_ign_catalog(
            settings.volume_root,
            runtime_root=Path(sys.prefix),
            asset_manifest=settings.simready_asset_manifest,
            nvidia_asset_root=settings.nvidia_asset_root,
            include_response_assets=(
                "response_engagement" in production["active_categories"]
            ),
        )
        status["input_preparation"] = preparation
        if preparation.get("state") in {"existing", "prepared"}:
            _set_setup_stage(
                status,
                "terrain_catalog",
                state="ready",
                detail="Catalogue pilote écrit; revue visuelle requise.",
            )
        else:
            _set_setup_stage(
                status,
                "terrain_catalog",
                state="blocked",
                detail=str(
                    preparation.get("reason")
                    or "Préparation des entrées incomplète."
                ),
            )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        existing_progress = load_progress(settings.volume_root)
        if (
            not isinstance(existing_progress, dict)
            or existing_progress.get("state") != "blocked"
        ):
            existing_progress = write_progress(
                settings.volume_root,
                phase="input_preparation",
                state="blocked",
                message=f"{type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
            )
        reason = str(
            existing_progress.get("message")
            or f"{type(exc).__name__}: {exc}"
        )
        status["input_preparation"] = {
            "state": "blocked",
            "phase": str(
                existing_progress.get("phase") or "input_preparation"
            ),
            "reason": reason,
        }
        _set_setup_stage(
            status,
            "terrain_catalog",
            state="blocked",
            detail=reason,
        )
    status["real_world_input"] = _load_real_world_status(
        settings=settings,
        production=production,
    )


def main() -> None:
    settings = Settings.from_environment()
    _prepare_volume(settings.volume_root)
    result = provision(
        manifest_path=settings.manifest_path,
        volume_root=settings.volume_root,
        allowed_hosts=settings.allowed_hosts,
    )
    if not settings.skip_gpu_preflight:
        _run_compatibility_checker()
        _run_probe()
    status: dict[str, object] = {
        "status": "ready",
        "mode": settings.run_mode,
        "gpu_preflight_skipped": settings.skip_gpu_preflight,
        "downloaded": len(result.downloaded),
        "cache_hits": len(result.cache_hits),
        "setup": {
            "runtime": {
                "state": "ready",
                "detail": "GPU, Isaac, Flow et Replicator validés.",
            },
            "asset_lock": {
                "state": "pending",
                "detail": "Inventaire NVIDIA non commencé.",
            },
            "terrain_catalog": {
                "state": "pending",
                "detail": "Aucun catalogue pilote validé.",
            },
        },
    }
    print(f"fireviewer sdg ready {json.dumps(status, sort_keys=True)}", flush=True)
    if settings.run_mode == "probe":
        return
    if settings.run_mode == "generate":
        scenario = os.getenv("FW_SDG_SCENARIO", "").strip()
        if not scenario:
            raise RuntimeError("FW_SDG_SCENARIO is required in generate mode")
        from fireviewer_sdg.generate import generate

        print(generate(Path(scenario).resolve()), flush=True)
        return
    production = load_production_plan(settings.campaign_path)
    status["production_id"] = production["production_id"]
    status["production_categories"] = len(production["categories"])
    status["nvidia_ngc_credentials_configured"] = bool(
        os.getenv("NGC_API_KEY", "").strip()
    )
    try:
        status["storage"] = assert_storage_architecture(
            settings.volume_root, production["storage"]
        )
        _set_setup_stage(
            status,
            "storage",
            state="ready",
            detail="Capacité et réserve du stockage validées.",
        )
    except RuntimeError as exc:
        status["storage"] = {
            "state": "blocked",
            "reason": f"{type(exc).__name__}: {exc}",
        }
        _set_setup_stage(
            status,
            "storage",
            state="blocked",
            detail=f"{type(exc).__name__}: {exc}",
        )
    status["input_preparation"] = {
        "state": "pending",
        "phase": "not_started",
        "reason": "Préparation des entrées non commencée.",
    }
    status["real_world_input"] = _load_real_world_status(
        settings=settings,
        production=production,
    )
    if settings.prepare_input_catalog:
        threading.Thread(
            target=_prepare_inputs_in_background,
            kwargs={
                "settings": settings,
                "production": production,
                "status": status,
            },
            name="fireviewer-sdg-input-preparation",
            daemon=True,
        ).start()
    case_store = CaseStore(
        settings.volume_root,
        target_per_category=production["target_per_category"],
        active_categories=production["active_categories"],
        render_revision=production["render_revision"],
    )
    serve(
        port=settings.port,
        auth_token=settings.auth_token,
        volume_root=settings.volume_root,
        status=status,
        campaign_path=settings.campaign_path,
        production_manager=ProductionManager(settings.volume_root, case_store),
        case_store=case_store,
    )


if __name__ == "__main__":
    main()

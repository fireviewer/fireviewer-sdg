"""Runtime configuration with fail-closed defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fireviewer_sdg.simready_assets import DEFAULT_NVIDIA_ASSET_ROOT


def _enabled(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    volume_root: Path
    manifest_path: Path
    campaign_path: Path
    run_mode: str
    port: int
    auth_token: str
    allowed_hosts: frozenset[str]
    skip_gpu_preflight: bool
    prepare_input_catalog: bool
    simready_asset_manifest: Path | None
    nvidia_asset_root: str

    @classmethod
    def from_environment(cls) -> "Settings":
        hosts = {
            host.strip().lower()
            for host in os.getenv(
                "FW_SDG_ALLOWED_HOSTS",
                "huggingface.co",
            ).split(",")
            if host.strip()
        }
        run_mode = os.getenv("FW_SDG_RUN_MODE", "service").strip().lower()
        if run_mode not in {"service", "probe", "generate"}:
            raise RuntimeError(f"unsupported FW_SDG_RUN_MODE: {run_mode}")
        port = int(os.getenv("FW_SDG_PORT", "8000"))
        if not 1 <= port <= 65535:
            raise RuntimeError("FW_SDG_PORT must be between 1 and 65535")
        manifest_override = os.getenv("FW_SDG_SIMREADY_ASSET_MANIFEST", "").strip()
        return cls(
            volume_root=Path(
                os.getenv("FW_SDG_VOLUME_ROOT", "/workspace/fireviewer-sdg")
            ).resolve(),
            manifest_path=Path(
                os.getenv(
                    "FW_SDG_PROVISION_MANIFEST",
                    "/opt/fireviewer-sdg/provision-manifest.json",
                )
            ).resolve(),
            campaign_path=Path(
                os.getenv(
                    "FW_SDG_CAMPAIGN",
                    "/opt/fireviewer-sdg/campaigns/fireviewer-new-synthetic-cases-v1.json",
                )
            ).resolve(),
            run_mode=run_mode,
            port=port,
            auth_token=os.getenv("FW_SDG_AUTH_TOKEN", "").strip(),
            allowed_hosts=frozenset(hosts),
            skip_gpu_preflight=_enabled(os.getenv("FW_SDG_SKIP_GPU_PREFLIGHT")),
            prepare_input_catalog=_enabled(
                os.getenv("FW_SDG_PREPARE_IGN_CATALOG")
            ),
            simready_asset_manifest=(
                Path(manifest_override).resolve() if manifest_override else None
            ),
            nvidia_asset_root=os.getenv(
                "FW_SDG_NVIDIA_ASSET_ROOT",
                DEFAULT_NVIDIA_ASSET_ROOT,
            ).strip(),
        )

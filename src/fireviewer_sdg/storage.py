"""Storage topology and capacity gates for RunPod and Windows production."""

from __future__ import annotations

import os
import shutil
from pathlib import Path, PureWindowsPath
from typing import Any


NETWORK_FILESYSTEMS = frozenset({"nfs", "nfs4", "ceph", "lustre", "cifs"})
GIB = 1024**3
GB = 1000**3


def _is_posix() -> bool:
    return os.name == "posix"


def _runtime_storage_mode() -> tuple[str, int | None]:
    mode = os.getenv("FW_SDG_STORAGE_MODE", "network_volume").strip().lower()
    if mode == "network_volume":
        return mode, None
    if mode != "ephemeral":
        raise RuntimeError(f"unsupported FW_SDG_STORAGE_MODE: {mode}")
    try:
        declared_capacity_gb = int(
            os.getenv("FW_SDG_EPHEMERAL_CAPACITY_GB", "0")
        )
    except ValueError as exc:
        raise RuntimeError(
            "FW_SDG_EPHEMERAL_CAPACITY_GB must be an integer"
        ) from exc
    if os.getenv("FW_SDG_EPHEMERAL_EXPORT_ACK", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise RuntimeError(
            "ephemeral production requires FW_SDG_EPHEMERAL_EXPORT_ACK=1"
        )
    return mode, declared_capacity_gb


def validate_storage_plan(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("production storage plan is required")
    plan = dict(payload)
    kind = str(plan.get("kind", "")).strip()
    mount_point = str(plan.get("mount_point", "")).strip()
    if kind not in {"network_volume", "local_windows"}:
        raise ValueError(
            "production storage kind must be network_volume or local_windows"
        )
    if kind == "network_volume" and mount_point != "/workspace":
        raise ValueError("production network volume must mount at /workspace")
    if kind == "local_windows" and not PureWindowsPath(mount_point).is_absolute():
        raise ValueError(
            "production local Windows storage mount_point must be absolute"
        )
    minimum_total_gb = int(plan.get("minimum_total_gb", 0))
    reserve_free_gb = int(plan.get("reserve_free_gb", 0))
    estimated_max_case_bytes = int(plan.get("estimated_max_case_bytes", 0))
    minimum_fire_events = int(plan.get("minimum_fire_events", 0))
    estimated_max_event_input_bytes = int(
        plan.get("estimated_max_event_input_bytes", 0)
    )
    if minimum_total_gb < 1000:
        raise ValueError("production storage must contain at least 1000 GB")
    if reserve_free_gb < 100:
        raise ValueError("production storage must reserve at least 100 GB free")
    if estimated_max_case_bytes < 12 * 1024**2:
        raise ValueError("production must budget at least 12 MiB per case")
    if minimum_fire_events < 512:
        raise ValueError("production storage must budget at least 512 fire inputs")
    if estimated_max_event_input_bytes < 512 * 1024**2:
        raise ValueError("production must budget at least 512 MiB per fire input")
    plan.update(
        kind=kind,
        mount_point=mount_point,
        minimum_total_gb=minimum_total_gb,
        reserve_free_gb=reserve_free_gb,
        estimated_max_case_bytes=estimated_max_case_bytes,
        minimum_fire_events=minimum_fire_events,
        estimated_max_event_input_bytes=estimated_max_event_input_bytes,
    )
    return plan


def _unescape_mount(value: str) -> str:
    return value.replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\")


def _linux_mount(path: Path) -> tuple[Path, str]:
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        raise RuntimeError("Linux mount inventory is absent")
    matches: list[tuple[Path, str]] = []
    target = path.resolve()
    for line in mountinfo.read_text(encoding="utf-8").splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        filesystem = right.split()[0] if right.split() else ""
        if len(fields) < 5:
            continue
        mount = Path(_unescape_mount(fields[4])).resolve()
        if target == mount or mount in target.parents:
            matches.append((mount, filesystem))
    if not matches:
        raise RuntimeError(f"no mount contains production volume: {target}")
    return max(matches, key=lambda item: len(item[0].parts))


def assert_storage_architecture(volume_root: Path, storage_plan: dict[str, Any]) -> dict[str, Any]:
    plan = validate_storage_plan(storage_plan)
    root = volume_root.resolve()
    usage = shutil.disk_usage(root)
    minimum_total = plan["minimum_total_gb"] * GB
    if not _is_posix():
        if plan["kind"] != "local_windows":
            raise RuntimeError(
                "Windows production requires storage kind local_windows"
            )
        workspace = Path(plan["mount_point"]).resolve()
        if root != workspace and workspace not in root.parents:
            raise RuntimeError(
                f"local Windows production root must remain below {workspace}: {root}"
            )
        if usage.total < minimum_total:
            raise RuntimeError(
                "production storage is undersized: "
                f"total_bytes={usage.total} required_bytes={minimum_total}"
            )
        return {
            "state": "ready",
            "mount": str(workspace),
            "filesystem": "windows_local",
            "total_bytes": usage.total,
            "free_bytes": usage.free,
            "durability": "local_windows",
            "export_required": False,
            "declared_capacity_gb": None,
        }
    if plan["kind"] != "network_volume":
        raise RuntimeError(
            "Linux production requires storage kind network_volume"
        )
    mode, declared_capacity_gb = _runtime_storage_mode()
    mount, filesystem = _linux_mount(root)
    if mode == "network_volume":
        if mount != Path(plan["mount_point"]):
            raise RuntimeError(
                f"production volume must be mounted from {plan['mount_point']}, got {mount}"
            )
        if filesystem not in NETWORK_FILESYSTEMS:
            raise RuntimeError(
                f"/workspace is not a supported network filesystem: {filesystem}"
            )
    else:
        workspace = Path(plan["mount_point"]).resolve()
        if root != workspace and workspace not in root.parents:
            raise RuntimeError(
                f"ephemeral production root must remain below {workspace}: {root}"
            )
        if declared_capacity_gb is None or declared_capacity_gb < plan["minimum_total_gb"]:
            raise RuntimeError(
                "ephemeral container disk is undersized: "
                f"declared_gb={declared_capacity_gb or 0} "
                f"required_gb={plan['minimum_total_gb']}"
            )
    if usage.total < minimum_total:
        raise RuntimeError(
            f"production storage is undersized: total_bytes={usage.total} required_bytes={minimum_total}"
        )
    return {
        "state": "ready",
        "mount": str(mount),
        "filesystem": filesystem,
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "durability": mode,
        "export_required": mode == "ephemeral",
        "declared_capacity_gb": declared_capacity_gb,
    }


def assert_remaining_capacity(
    volume_root: Path,
    storage_plan: dict[str, Any],
    *,
    remaining_cases: int,
) -> None:
    if remaining_cases < 0:
        raise ValueError("remaining_cases must be non-negative")
    plan = validate_storage_plan(storage_plan)
    usage = shutil.disk_usage(volume_root)
    required = (
        remaining_cases * plan["estimated_max_case_bytes"]
        + plan["reserve_free_gb"] * GIB
    )
    if usage.free < required:
        raise RuntimeError(
            "production storage guard refused generation: "
            f"free_bytes={usage.free} projected_required_bytes={required} "
            f"remaining_cases={remaining_cases}"
        )

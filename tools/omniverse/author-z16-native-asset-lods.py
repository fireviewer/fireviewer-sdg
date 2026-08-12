#!/usr/bin/env python3
"""Author a real HERO/MID/FAR chain for each recovered Z16 source asset.

This is deliberately a source-preparation command.  It consumes only the
materialized local assets proved by ``materialization-receipt.json`` and asks
NVIDIA Scene Optimizer to decimate isolated copies of the composed HERO mesh.
It never creates primitive substitutes, does not alter recovered sources, and
publishes the LOD directory only after every asset passed the native geometric
identity checks.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fireviewer_sdg import campaign_asset_bundle as campaign  # noqa: E402


MATERIALIZATION_RECEIPT = "materialization-receipt.json"
OUTPUT_DIRECTORY = "lod"
OUTPUT_RECEIPT = "lod-receipt.json"
STATE = "Z16_RECOVERED_ASSET_LODS_READY"
SCHEMA_VERSION = 1


class Z16LodError(RuntimeError):
    """Raised when an asset cannot form a real native LOD chain."""


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
        raise Z16LodError(f"invalid JSON document: {path}") from exc
    if not isinstance(value, dict):
        raise Z16LodError(f"JSON document must contain an object: {path}")
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


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _safe_relative(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise Z16LodError(f"{label} must be a non-empty relative path")
    normalized = raw.replace("\\", "/")
    value = PurePosixPath(normalized)
    if value.is_absolute() or ".." in value.parts:
        raise Z16LodError(f"{label} is an unsafe relative path")
    return Path(*value.parts)


def _asset_id(raw: object) -> str:
    if not isinstance(raw, str) or len(raw) != 32:
        raise Z16LodError(f"invalid recovered asset id: {raw!r}")
    normalized = raw.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise Z16LodError(f"invalid recovered asset id: {raw!r}")
    return normalized


def _validate_materialization(*, site_root: Path) -> tuple[Path, dict[str, Any]]:
    receipt_path = site_root / "assets" / "materialized" / MATERIALIZATION_RECEIPT
    receipt = _read_json(receipt_path)
    if receipt.get("state") != "Z16_RECOVERED_ASSETS_MATERIALIZED":
        raise Z16LodError("recovered assets have not been materialized")
    raw_assets = receipt.get("assets")
    if not isinstance(raw_assets, list) or len(raw_assets) != 9:
        raise Z16LodError("materialization receipt must declare exactly nine assets")
    if receipt.get("historical_pod_wrapper_lock_replaced") is not False:
        raise Z16LodError("materialization receipt must preserve the historical lock")
    return receipt_path, receipt


def _asset_wrapper(
    *, site_root: Path, item: dict[str, Any]
) -> tuple[str, Path, str]:
    asset_id = _asset_id(item.get("asset_id"))
    role = item.get("role")
    if role not in {"actor", "vegetation", "building"}:
        raise Z16LodError(f"{asset_id} has invalid recovered asset role: {role!r}")
    expected_sha = item.get("wrapper_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise Z16LodError(f"{asset_id} has no valid wrapper checksum")
    wrapper = (site_root / _safe_relative(item.get("wrapper"), label=f"{asset_id}.wrapper")).resolve()
    if not _inside(site_root, wrapper) or not wrapper.is_file() or wrapper.is_symlink():
        raise Z16LodError(f"{asset_id} wrapper is missing or unsafe")
    if _sha256(wrapper) != expected_sha:
        raise Z16LodError(f"{asset_id} wrapper checksum drifted since materialization")
    return asset_id, wrapper, str(role)


def _lod_record(*, root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
    }


def _normalize_metric_paths(
    value: object,
    *,
    site_root: Path,
    staging_root: Path,
    output_root: Path,
) -> object:
    """Replace local absolute paths with stable site-relative paths."""

    if isinstance(value, dict):
        return {
            str(key): _normalize_metric_paths(
                nested,
                site_root=site_root,
                staging_root=staging_root,
                output_root=output_root,
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_metric_paths(
                nested,
                site_root=site_root,
                staging_root=staging_root,
                output_root=output_root,
            )
            for nested in value
        ]
    if not isinstance(value, str):
        return value
    candidate = Path(value)
    if not candidate.is_absolute():
        return value
    if _inside(staging_root, candidate):
        candidate = output_root / candidate.resolve().relative_to(staging_root.resolve())
    else:
        parts = candidate.parts
        staging_index = next(
            (
                index
                for index, part in enumerate(parts)
                if part.startswith(".z16-lod-staging-")
            ),
            None,
        )
        if staging_index is not None:
            candidate = output_root.joinpath(*parts[staging_index + 1 :])
    if _inside(site_root, candidate):
        return candidate.resolve().relative_to(site_root.resolve()).as_posix()
    return value


def _validate_lod_record(*, lod_root: Path, level: str, raw: object) -> None:
    if not isinstance(raw, dict):
        raise Z16LodError(f"native LOD receipt has no {level} artifact")
    relative = _safe_relative(raw.get("path"), label=f"lod_paths.{level}.path")
    expected_sha = raw.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise Z16LodError(f"native LOD receipt has no {level} checksum")
    candidate = (lod_root / relative).resolve()
    if not _inside(lod_root, candidate) or not candidate.is_file() or candidate.is_symlink():
        raise Z16LodError(f"native {level} artifact is missing or unsafe")
    if _sha256(candidate) != expected_sha:
        raise Z16LodError(f"native {level} artifact checksum drifted")


def repair_existing_receipt(*, site_root: Path) -> dict[str, Any]:
    """Repair only old staging paths after a successful atomic directory move."""

    root = site_root.resolve()
    lod_root = root / "assets" / OUTPUT_DIRECTORY
    receipt_path = lod_root / OUTPUT_RECEIPT
    receipt = _read_json(receipt_path)
    if receipt.get("state") != STATE or receipt.get("schema_version") != SCHEMA_VERSION:
        raise Z16LodError("native LOD receipt is not an expected Z16 receipt")
    raw_assets = receipt.get("assets")
    if not isinstance(raw_assets, list) or len(raw_assets) != 9:
        raise Z16LodError("native LOD receipt must contain exactly nine assets")
    for asset in raw_assets:
        if not isinstance(asset, dict):
            raise Z16LodError("native LOD receipt contains a non-object asset")
        _asset_id(asset.get("asset_id"))
        lod_paths = asset.get("lod_paths")
        if not isinstance(lod_paths, dict):
            raise Z16LodError("native LOD receipt has no LOD path mapping")
        for level in campaign.LOD_LEVELS:
            _validate_lod_record(lod_root=lod_root, level=level, raw=lod_paths.get(level))
        metrics = asset.get("native_metrics")
        if not isinstance(metrics, dict):
            raise Z16LodError("native LOD receipt has no native metrics")
        asset["native_metrics"] = _normalize_metric_paths(
            metrics,
            site_root=root,
            staging_root=lod_root,
            output_root=lod_root,
        )
    _write_json(receipt_path, receipt)
    return {
        "state": "Z16_NATIVE_LOD_RECEIPT_PATHS_REPAIRED",
        "receipt": receipt_path.relative_to(root).as_posix(),
        "asset_count": len(raw_assets),
        "receipt_sha256": _sha256(receipt_path),
    }


def author_lods(*, site_root: Path) -> dict[str, Any]:
    root = site_root.resolve()
    receipt_path, materialization = _validate_materialization(site_root=root)
    output_root = root / "assets" / OUTPUT_DIRECTORY
    if output_root.exists() or output_root.is_symlink():
        raise Z16LodError(
            f"refusing to overwrite an existing native LOD output: {output_root}"
        )
    staging_root = Path(
        tempfile.mkdtemp(prefix=".z16-lod-staging-", dir=output_root.parent)
    )
    assets: list[dict[str, Any]] = []
    try:
        raw_assets = materialization["assets"]
        assert isinstance(raw_assets, list)
        seen_ids: set[str] = set()
        for item in raw_assets:
            if not isinstance(item, dict):
                raise Z16LodError("materialization receipt contains a non-object asset")
            asset_id, wrapper, role = _asset_wrapper(site_root=root, item=item)
            if asset_id in seen_ids:
                raise Z16LodError(f"duplicate recovered asset id: {asset_id}")
            seen_ids.add(asset_id)
            asset_root = staging_root / asset_id
            paths = {
                "HERO": asset_root / "hero.usdc",
                "MID": asset_root / "mid.usdc",
                "FAR": asset_root / "far.usdc",
            }
            try:
                metrics = campaign._build_scene_optimizer_lods(
                    official_wrapper=wrapper,
                    output_paths=paths,
                    bundle_root=root,
                )
            except campaign.CampaignAssetBundleError as exc:
                raise Z16LodError(
                    f"{asset_id} could not produce a real native LOD chain: {exc}"
                ) from exc
            if any(not path.is_file() or path.stat().st_size == 0 for path in paths.values()):
                raise Z16LodError(f"{asset_id} LOD generation emitted an empty stage")
            assets.append(
                {
                    "asset_id": asset_id,
                    "role": role,
                    "source_wrapper": wrapper.relative_to(root).as_posix(),
                    "lod_paths": {
                        level: _lod_record(root=staging_root, path=paths[level])
                        for level in campaign.LOD_LEVELS
                    },
                    "native_metrics": _normalize_metric_paths(
                        metrics,
                        site_root=root,
                        staging_root=staging_root,
                        output_root=output_root,
                    ),
                }
            )
        output_receipt = {
            "schema_version": SCHEMA_VERSION,
            "state": STATE,
            "source_materialization_receipt": receipt_path.relative_to(root).as_posix(),
            "source_materialization_receipt_sha256": _sha256(receipt_path),
            "asset_count": len(assets),
            "asset_roles": {
                "actors": sum(asset["role"] == "actor" for asset in assets),
                "vegetation": sum(asset["role"] == "vegetation" for asset in assets),
                "buildings": sum(asset["role"] == "building" for asset in assets),
            },
            "lod_levels": list(campaign.LOD_LEVELS),
            "generation": {
                "backend": "NVIDIA Scene Optimizer decimateMeshes",
                "mid_retained_percent": campaign.MID_RETAINED_PERCENT,
                "far_retained_percent_attempts": list(
                    campaign.FAR_RETAINED_PERCENT_ATTEMPTS
                ),
                "primitive_substitutions": "forbidden",
                "source_assets_mutated": False,
            },
            "next_required_step": "build_compact_sim_01_from_raster_terrain_and_native_asset_lods",
            "assets": assets,
        }
        _write_json(staging_root / OUTPUT_RECEIPT, output_receipt)
        os.replace(staging_root, output_root)
        return {
            **output_receipt,
            "lod_receipt": (output_root / OUTPUT_RECEIPT).relative_to(root).as_posix(),
        }
    except Exception:
        # Preserve an incomplete staging directory for forensic inspection and
        # keep the final output path absent, so no partial LOD set is reusable.
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    environment_site_root = os.getenv("FW_SDG_SITE_ROOT", "").strip()
    parser.add_argument(
        "--site-root",
        required=not bool(environment_site_root),
        default=Path(environment_site_root) if environment_site_root else None,
        type=Path,
    )
    parser.add_argument(
        "--repair-existing-receipt",
        action="store_true",
        help="validate artifacts and normalize only legacy temporary metric paths",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repair_existing_receipt:
        payload = repair_existing_receipt(site_root=args.site_root)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    try:
        import omni.kit.app
        import omni.kit.async_engine
    except ImportError:
        payload = author_lods(site_root=args.site_root)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    async def run_from_kit() -> None:
        exit_code = 0
        try:
            payload = author_lods(site_root=args.site_root)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        except Exception:
            exit_code = 1
            traceback.print_exc()
        finally:
            omni.kit.app.get_app().post_quit(exit_code)

    omni.kit.async_engine.run_coroutine(run_from_kit())
    return 0


if __name__ == "__main__":
    main()

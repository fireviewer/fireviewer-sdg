"""Assemble the local NVIDIA and ground-PBR inputs into one locked bundle.

The official NVIDIA materializer deliberately records provider metadata.  It
does not, however, emit the stricter ``HERO``/``MID``/``FAR`` wrapper contract
consumed by :mod:`fireviewer_sdg.asset_bundle` and
:mod:`fireviewer_sdg.composition_source`.  This module bridges that boundary
only when three real native representations already exist in the provider
asset or when NVIDIA Scene Optimizer can derive two validated native stages:

* a native LOD variant set, or
* a native ``LOD<N>`` prim hierarchy, or
* one native source decimated into distinct ``MID`` and ``FAR`` stages.

It never duplicates a wrapper, fabricates geometry, or substitutes a
primitive. An asset without either a real native chain or successful Scene
Optimizer decimation blocks the whole bundle with an asset-specific diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from fireviewer_sdg import asset_bundle as _asset_contract
from fireviewer_sdg.simready_assets import (
    MANIFEST_PROFILE,
    MANIFEST_SCHEMA_VERSION,
    PHOTOREAL_FAMILY_MINIMUMS,
    PHOTOREAL_LIBRARY_POLICY,
)


INSTALL_MARKER = _asset_contract.INSTALL_MARKER
LOD_LEVELS = _asset_contract.LOD_LEVELS
REQUIRED_ACTOR_CLASSES = _asset_contract.REQUIRED_ACTOR_CLASSES
PBR_MATERIAL_ROLES = _asset_contract.PBR_MATERIAL_ROLES
PBR_REQUIRED_TEXTURES = _asset_contract.PBR_REQUIRED_TEXTURES
USD_SUFFIXES = _asset_contract.USD_SUFFIXES
TEXTURE_SUFFIXES = _asset_contract.TEXTURE_SUFFIXES

ASSEMBLY_STATE = "CAMPAIGN_ASSET_BUNDLE_READY"
BASE_LANDSCAPE_ASSEMBLY_STATE = "BASE_LANDSCAPE_ENVIRONMENT_BUNDLE_READY"
ASSEMBLY_SCHEMA_VERSION = 1
DEFAULT_MANIFEST_NAME = "manifest-v3.json"
CAMPAIGN_BUNDLE_MODE = "campaign"
BASE_LANDSCAPE_BUNDLE_MODE = "base-landscape-environment"
SELECTED_ACTOR_GROUP_ID = "chrome-fire-response-group-2026-07-29"
SELECTED_ACTOR_GROUP_SOURCES = (
    (
        "94ef5c37c3c543fd9efbaa571a7a7590",
        "https://sketchfab.com/3d-models/"
        "po2-94ef5c37c3c543fd9efbaa571a7a7590",
        "aerial",
    ),
    (
        "8f62ab4eacbc430186d85a7029d7d156",
        "https://sketchfab.com/3d-models/"
        "an2-8f62ab4eacbc430186d85a7029d7d156",
        "aerial",
    ),
    (
        "6246617aeb874e4793b21d5861eea8c9",
        "https://sketchfab.com/3d-models/"
        "sikorsky-ch-53e-sea-stallion-6246617aeb874e4793b21d5861eea8c9",
        "aerial",
    ),
    (
        "fc2b5eb692ca40c2b44357b62eb149df",
        "https://sketchfab.com/3d-models/"
        "truck-fc2b5eb692ca40c2b44357b62eb149df",
        "ground",
    ),
    (
        "c573303be1f04e0c94cfa245c2f2ddcf",
        "https://sketchfab.com/3d-models/"
        "construction-truck-c573303be1f04e0c94cfa245c2f2ddcf",
        "ground",
    ),
)
SELECTED_ACTOR_GROUP_IDS = tuple(
    selection_id for selection_id, _url, _placement in SELECTED_ACTOR_GROUP_SOURCES
)
SELECTED_ACTOR_GROUP_SOURCE_BY_ID = {
    selection_id: source_url
    for selection_id, source_url, _placement in SELECTED_ACTOR_GROUP_SOURCES
}
SELECTED_ACTOR_GROUP_PLACEMENT_BY_ID = {
    selection_id: placement
    for selection_id, _source_url, placement in SELECTED_ACTOR_GROUP_SOURCES
}
SELECTED_ENVIRONMENT_GROUP_ID = "chrome-free-environment-group-2026-07-29"
SELECTED_ENVIRONMENT_GROUP = (
    ("9153c2b370934758bf14c395abe36b27", "vegetation", "trees"),
    ("cf138b8eb2d340cda643ed59f824989c", "vegetation", "trees"),
    ("b75d4fbbee614c4898ee5214b9fd04aa", "buildings", "habitat"),
    ("17861ac10c5e480d84339ed6d6cf8073", "buildings", "habitat"),
)
SELECTED_ENVIRONMENT_GROUP_IDS = tuple(
    selection_id
    for selection_id, _kind, _family in SELECTED_ENVIRONMENT_GROUP
)
SELECTED_ENVIRONMENT_TARGET_BY_ID = {
    selection_id: (kind, family)
    for selection_id, kind, family in SELECTED_ENVIRONMENT_GROUP
}
SCENE_OPTIMIZER_OPERATION = "decimateMeshes"
MID_RETAINED_PERCENT = 60.0
FAR_RETAINED_PERCENT = 20.0
FAR_RETAINED_PERCENT_ATTEMPTS = (20.0, 30.0, 40.0, 50.0)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOD_NUMBER_RE = re.compile(r"(?i)(?:^|[^a-z0-9])lod[_-]?(\d+)(?:$|[^0-9])")
_SAFE_ID_RE = re.compile(r"[^a-z0-9_-]+")


class CampaignAssetBundleError(RuntimeError):
    """Raised when the two input libraries cannot form the strict bundle."""


class _FarLodRetryableError(CampaignAssetBundleError):
    """Raised when only the isolated FAR stage can benefit from a safer retry."""


@dataclass(frozen=True, slots=True)
class _LockedSource:
    path: Path
    relative_to_manifest_parent: PurePosixPath
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _AssetPlan:
    kind: str
    family: str
    index: int
    entry: Mapping[str, Any]
    wrapper: _LockedSource
    source: _LockedSource
    dependencies: tuple[_LockedSource, ...]
    selected_levels: Mapping[str, str]
    lod_strategy: str

    @property
    def label(self) -> str:
        if self.kind == "actors":
            return f"actors.{self.family}"
        if self.kind == "selected_actor_group":
            return f"selected_actor_group.assets.{self.family}"
        if self.kind == "selected_environment_group":
            return f"selected_environment_group.assets.{self.family}"
        return f"environment.{self.kind}.{self.family}[{self.index}]"

    @property
    def asset_id(self) -> str:
        return str(self.entry.get("asset_id", "")).strip()

    @property
    def requires_decimation(self) -> bool:
        return self.lod_strategy == "scene_optimizer_decimateMeshes"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignAssetBundleError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise CampaignAssetBundleError(f"{label} must be a JSON object")
    return payload


def _inside(root: Path, candidate: Path) -> bool:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    return (
        resolved_candidate == resolved_root
        or resolved_root in resolved_candidate.parents
    )


def _relative_path(raw: object, *, label: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw.strip():
        raise CampaignAssetBundleError(f"{label} must be a non-empty relative path")
    if "\\" in raw or "\x00" in raw or any(ord(character) < 32 for character in raw):
        raise CampaignAssetBundleError(f"{label} contains forbidden characters")
    value = PurePosixPath(raw)
    if (
        value.is_absolute()
        or not value.parts
        or ".." in value.parts
        or value.parts[0] in {"", "."}
    ):
        raise CampaignAssetBundleError(f"{label} must stay below its manifest root")
    return value


def _input_reference_path(raw: object, *, label: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw.strip():
        raise CampaignAssetBundleError(f"{label} must be a non-empty relative path")
    if "\\" in raw or "\x00" in raw or any(ord(character) < 32 for character in raw):
        raise CampaignAssetBundleError(f"{label} contains forbidden characters")
    value = PurePosixPath(raw)
    if value.is_absolute() or not value.parts or value.parts[0] in {"", "."}:
        raise CampaignAssetBundleError(
            f"{label} must be relative to its manifest or volume root"
        )
    return value


def _symlink_free_volume_file(*, candidate: Path, volume_root: Path) -> Path | None:
    volume = volume_root.resolve()
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(volume)
    except ValueError:
        return None
    cursor = volume
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return None
    resolved = lexical.resolve()
    if not resolved.is_file() or not _inside(volume, resolved):
        return None
    return resolved


def _resolve_input_file(
    *,
    raw_path: object,
    manifest_parent: Path,
    volume_root: Path,
    label: str,
) -> tuple[Path, PurePosixPath]:
    relative = _input_reference_path(raw_path, label=label)
    candidates = (
        manifest_parent.joinpath(*relative.parts),
        volume_root.joinpath(*relative.parts),
    )
    existing = {
        resolved
        for candidate in candidates
        if (
            resolved := _symlink_free_volume_file(
                candidate=candidate,
                volume_root=volume_root,
            )
        )
        is not None
    }
    if not existing:
        raise CampaignAssetBundleError(
            f"{label} is absent below both the manifest parent and volume root: "
            f"{relative.as_posix()}"
        )
    if len(existing) > 1:
        raise CampaignAssetBundleError(
            f"{label} resolves ambiguously to multiple local files: "
            f"{relative.as_posix()}"
        )
    path = sorted(existing, key=str)[0]
    return path, PurePosixPath(path.relative_to(volume_root.resolve()).as_posix())


def _locked_source(
    *,
    record: Mapping[str, Any],
    manifest_parent: Path,
    volume_root: Path,
    label: str,
    allowed_suffixes: frozenset[str] | None = None,
    require_size: bool = True,
) -> _LockedSource:
    path, relative = _resolve_input_file(
        raw_path=record.get("path"),
        manifest_parent=manifest_parent,
        volume_root=volume_root,
        label=f"{label}.path",
    )
    if allowed_suffixes is not None and path.suffix.casefold() not in allowed_suffixes:
        raise CampaignAssetBundleError(
            f"{label} has unsupported file type {path.suffix or '<none>'}"
        )
    expected_sha = str(record.get("sha256", "")).strip().lower()
    if not _SHA256_RE.fullmatch(expected_sha) or _sha256(path) != expected_sha:
        raise CampaignAssetBundleError(f"{label} SHA-256 lock does not match")
    size = path.stat().st_size
    if require_size and record.get("size_bytes") != size:
        raise CampaignAssetBundleError(f"{label} size lock does not match")
    return _LockedSource(
        path=path,
        relative_to_manifest_parent=relative,
        sha256=expected_sha,
        size_bytes=size,
    )


def _source_record(source: _LockedSource, *, prefix: str) -> dict[str, object]:
    return {
        "path": (
            PurePosixPath(prefix) / source.relative_to_manifest_parent
        ).as_posix(),
        "sha256": source.sha256,
        "size_bytes": source.size_bytes,
    }


def _provider_lod_rank(level: str) -> tuple[int, int, str]:
    """Return a safe high-to-low rank for one provider-authored LOD label."""

    token = level.rsplit("=", 1)[-1].rsplit("/", 1)[-1].casefold()
    numeric = _LOD_NUMBER_RE.search(f" {token} ")
    if numeric:
        return (0, int(numeric.group(1)), token)
    normalized = "".join(character for character in token if character.isalnum())
    named = {
        "hero": 0,
        "high": 0,
        "highest": 0,
        "full": 0,
        "mid": 1,
        "medium": 1,
        "far": 2,
        "low": 2,
        "lowest": 2,
        "proxy": 2,
    }
    if normalized in named:
        return (1, named[normalized], normalized)
    raise CampaignAssetBundleError(
        f"native LOD level cannot be ordered without inventing quality: {level!r}"
    )


def _select_native_levels(
    *,
    entry: Mapping[str, Any],
    label: str,
) -> tuple[str, dict[str, str]]:
    lod = entry.get("lod")
    if not isinstance(lod, Mapping):
        raise CampaignAssetBundleError(f"{label}.lod is absent")
    strategy = str(lod.get("strategy", "")).strip()
    raw_levels = lod.get("levels")
    levels = (
        [str(level).strip() for level in raw_levels if str(level).strip()]
        if isinstance(raw_levels, list)
        else []
    )
    level_count = lod.get("level_count")
    asset_id = str(entry.get("asset_id", "")).strip() or "<missing>"
    structurally_valid = (
        lod.get("state") == "passed"
        and isinstance(level_count, int)
        and not isinstance(level_count, bool)
        and level_count == len(levels)
        and len(set(levels)) == len(levels)
    )
    if structurally_valid and strategy == "source_default_only" and len(levels) == 1:
        return (
            "scene_optimizer_decimateMeshes",
            {
                "HERO": levels[0],
                "MID": f"{SCENE_OPTIMIZER_OPERATION}:retain={MID_RETAINED_PERCENT:g}",
                "FAR": f"{SCENE_OPTIMIZER_OPERATION}:retain={FAR_RETAINED_PERCENT:g}",
            },
        )
    if (
        not structurally_valid
        or strategy not in {"native_variant_set", "native_prim_hierarchy"}
        or len(levels) < 3
    ):
        raise CampaignAssetBundleError(
            f"{label} asset_id={asset_id!r} has no real three-level LOD chain: "
            f"state={lod.get('state')!r}, strategy={strategy!r}, "
            f"level_count={level_count!r}, levels={levels!r}"
        )
    ranked = sorted((_provider_lod_rank(level), level) for level in levels)
    if len({rank for rank, _level in ranked}) != len(ranked):
        raise CampaignAssetBundleError(
            f"{label} asset_id={asset_id!r} has ambiguous native LOD ordering: "
            f"strategy={strategy!r}, levels={levels!r}"
        )
    ordered = [level for _rank, level in ranked]
    chosen = {
        "HERO": ordered[0],
        "MID": ordered[len(ordered) // 2],
        "FAR": ordered[-1],
    }
    if len(set(chosen.values())) != 3:
        raise CampaignAssetBundleError(
            f"{label} asset_id={asset_id!r} cannot provide three distinct LODs"
        )
    return strategy, chosen


def _validate_official_header(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise CampaignAssetBundleError(
            f"official NVIDIA schema_version must be {MANIFEST_SCHEMA_VERSION}"
        )
    if payload.get("profile") != MANIFEST_PROFILE:
        raise CampaignAssetBundleError(
            f"official NVIDIA profile must be {MANIFEST_PROFILE}"
        )
    if payload.get("family_minimums") != PHOTOREAL_FAMILY_MINIMUMS:
        raise CampaignAssetBundleError("official NVIDIA family minimums were weakened")
    if payload.get("library_policy") != PHOTOREAL_LIBRARY_POLICY:
        raise CampaignAssetBundleError("official NVIDIA library policy was weakened")
    discovery = payload.get("discovery")
    if (
        not isinstance(discovery, Mapping)
        or discovery.get("mode") != "materialized_photoreal_asset_library_v3"
        or list(discovery.get("missing_environment") or [])
    ):
        raise CampaignAssetBundleError(
            "official NVIDIA environment is not fully materialized"
        )


def _locked_asset_plan(
    *,
    entry: object,
    label: str,
    kind: str,
    family: str,
    index: int,
    manifest_parent: Path,
    volume_root: Path,
    seen_asset_ids: set[str],
) -> _AssetPlan:
    if not isinstance(entry, Mapping):
        raise CampaignAssetBundleError(f"{label} is not an object")
    asset_id = str(entry.get("asset_id", "")).strip()
    identity = entry.get("identity")
    source_identity = (
        str(identity.get("source_identity", "")).strip()
        if isinstance(identity, Mapping)
        else ""
    )
    if not asset_id or not source_identity:
        raise CampaignAssetBundleError(
            f"{label} requires asset_id and identity.source_identity"
        )
    if asset_id in seen_asset_ids:
        raise CampaignAssetBundleError(
            f"{label} repeats asset_id={asset_id!r}"
        )
    if (
        entry.get("quality_validation") != "native_metadata_passed"
        or not isinstance(entry.get("anchor_validation"), Mapping)
        or entry["anchor_validation"].get("state") != "passed"
        or not isinstance(entry.get("materials"), Mapping)
        or entry["materials"].get("state") != "passed"
    ):
        raise CampaignAssetBundleError(
            f"{label} asset_id={asset_id!r} did not pass native "
            "metadata, anchor and material validation"
        )
    wrapper = _locked_source(
        record={
            "path": entry.get("path"),
            "sha256": entry.get("sha256"),
        },
        manifest_parent=manifest_parent,
        volume_root=volume_root,
        label=f"{label}.path",
        allowed_suffixes=USD_SUFFIXES,
        require_size=False,
    )
    materialized = entry.get("materialized_files")
    if not isinstance(materialized, list) or not materialized:
        raise CampaignAssetBundleError(f"{label}.materialized_files is empty")
    dependencies: list[_LockedSource] = []
    for file_index, record in enumerate(materialized):
        if not isinstance(record, Mapping):
            raise CampaignAssetBundleError(
                f"{label}.materialized_files[{file_index}] is malformed"
            )
        dependencies.append(
            _locked_source(
                record=record,
                manifest_parent=manifest_parent,
                volume_root=volume_root,
                label=f"{label}.materialized_files[{file_index}]",
            )
        )
    expected_content_lock = hashlib.sha256(
        json.dumps(materialized, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if (
        str(entry.get("content_lock_sha256", "")).strip().lower()
        != expected_content_lock
    ):
        raise CampaignAssetBundleError(
            f"{label}.content_lock_sha256 does not bind its materialized_files"
        )
    source_path, source_relative = _resolve_input_file(
        raw_path=entry.get("source_cache_path"),
        manifest_parent=manifest_parent,
        volume_root=volume_root,
        label=f"{label}.source_cache_path",
    )
    source_dependency = next(
        (
            dependency
            for dependency in dependencies
            if dependency.path == source_path
        ),
        None,
    )
    if source_dependency is None:
        raise CampaignAssetBundleError(
            f"{label}.source_cache_path is absent from materialized_files"
        )
    source = _locked_source(
        record={
            "path": source_relative.as_posix(),
            "sha256": source_dependency.sha256,
            "size_bytes": source_dependency.size_bytes,
        },
        manifest_parent=manifest_parent,
        volume_root=volume_root,
        label=f"{label}.source_cache_path",
        allowed_suffixes=USD_SUFFIXES,
    )
    lod_strategy, selected = _select_native_levels(entry=entry, label=label)
    seen_asset_ids.add(asset_id)
    return _AssetPlan(
        kind=kind,
        family=family,
        index=index,
        entry=entry,
        wrapper=wrapper,
        source=source,
        dependencies=tuple(dependencies),
        selected_levels=selected,
        lod_strategy=lod_strategy,
    )


def _plan_environment_tree(
    *,
    payload: Mapping[str, Any],
    manifest_parent: Path,
    volume_root: Path,
    seen_asset_ids: set[str],
    issues: list[str],
    supplemental_family_counts: Mapping[tuple[str, str], int] | None = None,
) -> list[_AssetPlan]:
    environment = payload.get("environment")
    if not isinstance(environment, Mapping) or set(environment) != set(
        PHOTOREAL_FAMILY_MINIMUMS
    ):
        raise CampaignAssetBundleError(
            "official NVIDIA environment family tree is incomplete"
        )
    plans: list[_AssetPlan] = []
    for kind, family_minimums in PHOTOREAL_FAMILY_MINIMUMS.items():
        families = environment.get(kind)
        if not isinstance(families, Mapping) or set(families) != set(family_minimums):
            raise CampaignAssetBundleError(
                f"official NVIDIA environment.{kind} family tree is incomplete"
            )
        for family, minimum in family_minimums.items():
            entries = families.get(family)
            supplemental = int(
                (supplemental_family_counts or {}).get((kind, family), 0)
            )
            if (
                not isinstance(entries, list)
                or len(entries) + supplemental < minimum
            ):
                direct_count = len(entries) if isinstance(entries, list) else 0
                raise CampaignAssetBundleError(
                    f"official NVIDIA environment.{kind}.{family} requires "
                    f"at least {minimum} assets; direct={direct_count}, "
                    f"selected_environment_group={supplemental}, "
                    f"combined={direct_count + supplemental}"
                )
            for index, entry in enumerate(entries):
                label = f"environment.{kind}.{family}[{index}]"
                try:
                    plans.append(
                        _locked_asset_plan(
                            entry=entry,
                            label=label,
                            kind=kind,
                            family=family,
                            index=index,
                            manifest_parent=manifest_parent,
                            volume_root=volume_root,
                            seen_asset_ids=seen_asset_ids,
                        )
                    )
                except CampaignAssetBundleError as exc:
                    issues.append(str(exc))
    return plans


def _plan_selected_environment_group(
    *,
    payload: Mapping[str, Any],
    manifest_parent: Path,
    volume_root: Path,
    seen_asset_ids: set[str],
    issues: list[str],
) -> list[_AssetPlan]:
    environment_group = payload.get("selected_environment_group")
    selected_environment = (
        environment_group.get("assets")
        if isinstance(environment_group, Mapping)
        else None
    )
    selected_environment_keys = (
        set(selected_environment)
        if isinstance(selected_environment, Mapping)
        else set()
    )
    if (
        not isinstance(environment_group, Mapping)
        or environment_group.get("group_id")
        != SELECTED_ENVIRONMENT_GROUP_ID
        or environment_group.get("selection_order")
        != list(SELECTED_ENVIRONMENT_GROUP_IDS)
        or environment_group.get("selection_count")
        != len(SELECTED_ENVIRONMENT_GROUP_IDS)
        or not isinstance(selected_environment, Mapping)
        or selected_environment_keys != set(SELECTED_ENVIRONMENT_GROUP_IDS)
    ):
        missing = sorted(
            set(SELECTED_ENVIRONMENT_GROUP_IDS) - selected_environment_keys
        )
        unexpected = sorted(
            selected_environment_keys - set(SELECTED_ENVIRONMENT_GROUP_IDS)
        )
        raise CampaignAssetBundleError(
            "official manifest requires the exact four acquired free "
            f"environment assets: missing={missing}, unexpected={unexpected}"
        )
    plans: list[_AssetPlan] = []
    for index, selection_id in enumerate(SELECTED_ENVIRONMENT_GROUP_IDS):
        label = f"selected_environment_group.assets.{selection_id}"
        entry = selected_environment[selection_id]
        identity = entry.get("identity") if isinstance(entry, Mapping) else None
        source_identity = (
            str(identity.get("source_identity", "")).strip()
            if isinstance(identity, Mapping)
            else ""
        )
        if selection_id not in source_identity.casefold():
            issues.append(
                f"{label}.identity.source_identity does not bind acquired "
                f"model {selection_id}"
            )
            continue
        try:
            plans.append(
                _locked_asset_plan(
                    entry=entry,
                    label=label,
                    kind="selected_environment_group",
                    family=selection_id,
                    index=index,
                    manifest_parent=manifest_parent,
                    volume_root=volume_root,
                    seen_asset_ids=seen_asset_ids,
                )
            )
        except CampaignAssetBundleError as exc:
            issues.append(str(exc))
    return plans


def _plan_environment_assets(
    *,
    payload: Mapping[str, Any],
    manifest_parent: Path,
    volume_root: Path,
) -> tuple[_AssetPlan, ...]:
    """Plan only the complete base-landscape environment and its four additions."""

    _validate_official_header(payload)
    issues: list[str] = []
    seen_asset_ids: set[str] = set()
    selected_plans = _plan_selected_environment_group(
        payload=payload,
        manifest_parent=manifest_parent,
        volume_root=volume_root,
        seen_asset_ids=seen_asset_ids,
        issues=issues,
    )
    supplemental_family_counts: dict[tuple[str, str], int] = {}
    for plan in selected_plans:
        target = SELECTED_ENVIRONMENT_TARGET_BY_ID[plan.family]
        supplemental_family_counts[target] = (
            supplemental_family_counts.get(target, 0) + 1
        )
    plans = _plan_environment_tree(
        payload=payload,
        manifest_parent=manifest_parent,
        volume_root=volume_root,
        seen_asset_ids=seen_asset_ids,
        issues=issues,
        supplemental_family_counts=supplemental_family_counts,
    )
    plans.extend(selected_plans)
    if issues:
        raise CampaignAssetBundleError(
            "official NVIDIA assets cannot satisfy the strict HERO/MID/FAR "
            "base-landscape environment contract:\n- " + "\n- ".join(issues)
        )
    return tuple(plans)


def _plan_assets(
    *,
    payload: Mapping[str, Any],
    manifest_parent: Path,
    volume_root: Path,
) -> tuple[_AssetPlan, ...]:
    _validate_official_header(payload)
    issues: list[str] = []
    seen_asset_ids: set[str] = set()
    plans = _plan_environment_tree(
        payload=payload,
        manifest_parent=manifest_parent,
        volume_root=volume_root,
        seen_asset_ids=seen_asset_ids,
        issues=issues,
    )
    actor_group = payload.get("selected_actor_group")
    group_assets = (
        actor_group.get("assets")
        if isinstance(actor_group, Mapping)
        else None
    )
    group_keys = set(group_assets) if isinstance(group_assets, Mapping) else set()
    if (
        not isinstance(actor_group, Mapping)
        or actor_group.get("group_id") != SELECTED_ACTOR_GROUP_ID
        or actor_group.get("selection_order") != list(SELECTED_ACTOR_GROUP_IDS)
        or actor_group.get("selection_count") != len(SELECTED_ACTOR_GROUP_IDS)
        or not isinstance(group_assets, Mapping)
        or group_keys != set(SELECTED_ACTOR_GROUP_IDS)
    ):
        missing = sorted(set(SELECTED_ACTOR_GROUP_IDS) - group_keys)
        unexpected = sorted(group_keys - set(SELECTED_ACTOR_GROUP_IDS))
        raise CampaignAssetBundleError(
            "official manifest requires the exact selected Chrome actor group: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for index, selection_id in enumerate(SELECTED_ACTOR_GROUP_IDS):
        label = f"selected_actor_group.assets.{selection_id}"
        entry = group_assets[selection_id]
        identity = entry.get("identity") if isinstance(entry, Mapping) else None
        source_identity = (
            str(identity.get("source_identity", "")).strip()
            if isinstance(identity, Mapping)
            else ""
        )
        if selection_id not in source_identity.casefold():
            issues.append(
                f"{label}.identity.source_identity does not bind Sketchfab "
                f"model {selection_id}"
            )
            continue
        try:
            plans.append(
                _locked_asset_plan(
                    entry=entry,
                    label=label,
                    kind="selected_actor_group",
                    family=selection_id,
                    index=index,
                    manifest_parent=manifest_parent,
                    volume_root=volume_root,
                    seen_asset_ids=seen_asset_ids,
                )
            )
        except CampaignAssetBundleError as exc:
            issues.append(str(exc))
    actors = payload.get("actors")
    actor_keys = set(actors) if isinstance(actors, Mapping) else set()
    if not isinstance(actors, Mapping) or actor_keys != set(
        REQUIRED_ACTOR_CLASSES
    ):
        missing = sorted(set(REQUIRED_ACTOR_CLASSES) - actor_keys)
        unexpected = sorted(actor_keys - set(REQUIRED_ACTOR_CLASSES))
        raise CampaignAssetBundleError(
            "official manifest requires the exact seven semantic actor roles: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for class_id in REQUIRED_ACTOR_CLASSES:
        label = f"actors.{class_id}"
        try:
            plans.append(
                _locked_asset_plan(
                    entry=actors[class_id],
                    label=label,
                    kind="actors",
                    family=class_id,
                    index=0,
                    manifest_parent=manifest_parent,
                    volume_root=volume_root,
                    seen_asset_ids=seen_asset_ids,
                )
            )
        except CampaignAssetBundleError as exc:
            issues.append(str(exc))
    plans.extend(
        _plan_selected_environment_group(
            payload=payload,
            manifest_parent=manifest_parent,
            volume_root=volume_root,
            seen_asset_ids=seen_asset_ids,
            issues=issues,
        )
    )
    if issues:
        raise CampaignAssetBundleError(
            "official NVIDIA assets cannot satisfy the strict HERO/MID/FAR "
            "bundle contract:\n- " + "\n- ".join(issues)
        )
    return tuple(plans)


def _materialize_file(
    source: Path,
    destination: Path,
    *,
    allow_hardlink: bool = True,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (
            not destination.is_file()
            or destination.is_symlink()
            or destination.stat().st_size != source.stat().st_size
            or _sha256(destination) != _sha256(source)
        ):
            raise CampaignAssetBundleError(
                f"bundle path collision with different content: {destination}"
            )
        return
    try:
        if not allow_hardlink:
            raise OSError("isolated writable USD copy required")
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    if _sha256(destination) != _sha256(source):
        raise CampaignAssetBundleError(
            f"materialized bundle file changed during copy: {destination}"
        )


def _asset_relative_target(source: _LockedSource) -> PurePosixPath:
    return PurePosixPath("official") / source.relative_to_manifest_parent


def _safe_asset_directory(asset_id: str) -> str:
    slug = _SAFE_ID_RE.sub("-", asset_id.casefold()).strip("-")
    if not slug:
        slug = "asset"
    identity = hashlib.sha256(asset_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:80]}-{identity}"


def _relocated_prim_parts(native_path: str) -> tuple[str, ...]:
    parts = tuple(part for part in PurePosixPath(native_path).parts if part != "/")
    if not parts:
        raise CampaignAssetBundleError(
            f"native LOD prim path is invalid: {native_path!r}"
        )
    # The provider default prim is referenced onto /Asset/Source.  Its first
    # component is therefore replaced by Source in the composed wrapper.
    return ("Source", *parts[1:])


def _render_over_tree(
    tree: Mapping[str, Any],
    *,
    indent: int,
) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for name in sorted(tree):
        node = tree[name]
        metadata = node.get("metadata", [])
        body = node.get("body", [])
        lines.append(f'{prefix}over "{name}"')
        if metadata:
            lines.append(f"{prefix}(")
            lines.extend(f"{prefix}    {line}" for line in metadata)
            lines.append(f"{prefix})")
        lines.append(f"{prefix}{{")
        lines.extend(f"{prefix}    {line}" for line in body)
        lines.extend(
            _render_over_tree(node.get("children", {}), indent=indent + 4)
        )
        lines.append(f"{prefix}}}")
    return lines


def _tree_node(tree: dict[str, Any], parts: Sequence[str]) -> dict[str, Any]:
    cursor = tree
    node: dict[str, Any] | None = None
    for part in parts:
        node = cursor.setdefault(
            part,
            {"metadata": [], "body": [], "children": {}},
        )
        cursor = node["children"]
    if node is None:
        raise CampaignAssetBundleError("empty native LOD target path")
    return node


def _variant_selection(level: str) -> tuple[str, str, str]:
    try:
        prim_path, selection = level.rsplit(":", 1)
        variant_set, value = selection.split("=", 1)
    except ValueError as exc:
        raise CampaignAssetBundleError(
            f"native variant LOD descriptor is malformed: {level!r}"
        ) from exc
    if (
        not prim_path.startswith("/")
        or not variant_set.strip()
        or not value.strip()
        or any(character in variant_set + value for character in '"\r\n')
    ):
        raise CampaignAssetBundleError(
            f"native variant LOD descriptor is unsafe: {level!r}"
        )
    return prim_path, variant_set.strip(), value.strip()


def _lod_wrapper_text(
    *,
    official_wrapper_relative: PurePosixPath,
    wrapper_relative: PurePosixPath,
    strategy: str,
    selected_level: str,
    all_levels: Sequence[str],
    output_level: str,
    asset_id: str,
) -> str:
    reference = os.path.relpath(
        official_wrapper_relative.as_posix(),
        PurePosixPath(wrapper_relative).parent.as_posix(),
    ).replace("\\", "/")
    if "@" in reference or "\r" in reference or "\n" in reference:
        raise CampaignAssetBundleError(
            f"official wrapper path cannot be encoded safely: {reference}"
        )
    tree: dict[str, Any] = {}
    if strategy == "native_variant_set":
        prim_path, variant_set, value = _variant_selection(selected_level)
        node = _tree_node(tree, _relocated_prim_parts(prim_path))
        node["metadata"].extend(
            [
                "variants = {",
                f'    string {variant_set} = "{value}"',
                "}",
            ]
        )
    elif strategy == "native_prim_hierarchy":
        for level in all_levels:
            if not level.startswith("/"):
                raise CampaignAssetBundleError(
                    f"native hierarchy LOD path is malformed: {level!r}"
                )
            node = _tree_node(tree, _relocated_prim_parts(level))
            node["metadata"].append(
                f"active = {'true' if level == selected_level else 'false'}"
            )
            if level == selected_level:
                node["body"].append('token visibility = "inherited"')
    else:
        raise CampaignAssetBundleError(
            f"unsupported native LOD strategy: {strategy!r}"
        )
    tree_lines = _render_over_tree(tree, indent=4)
    safe_asset_id = asset_id.replace("\\", "\\\\").replace('"', '\\"')
    safe_selected = selected_level.replace("\\", "\\\\").replace('"', '\\"')
    return "\n".join(
        [
            "#usda 1.0",
            "(",
            '    defaultPrim = "Asset"',
            "    metersPerUnit = 1",
            '    upAxis = "Z"',
            "    customLayerData = {",
            f'        string fireviewer_asset_id = "{safe_asset_id}"',
            f'        string fireviewer_lod = "{output_level}"',
            f'        string fireviewer_native_lod = "{safe_selected}"',
            f'        string fireviewer_native_lod_strategy = "{strategy}"',
            "    }",
            ")",
            f'def Xform "Asset" (',
            f"    prepend references = @{reference}@",
            ")",
            "{",
            *tree_lines,
            "}",
            "",
        ]
    )


def _flatten_hero_stage(
    *,
    official_wrapper: Path,
    hero_path: Path,
    bundle_root: Path,
) -> None:
    try:
        from pxr import Usd, UsdUtils
    except ImportError as exc:
        raise CampaignAssetBundleError(
            "source_default_only requires the pinned Kit/Isaac OpenUSD runtime"
        ) from exc

    stage = Usd.Stage.Open(str(official_wrapper), load=Usd.Stage.LoadAll)
    if stage is None:
        raise CampaignAssetBundleError(
            f"unable to open the locked NVIDIA source for HERO flattening: "
            f"{official_wrapper}"
        )
    # NVIDIA's Kasa assets are authored as instanceable reference roots.  A
    # direct Stage.Flatten() preserves that instance boundary, so the exported
    # HERO contains no directly traversable meshes and its genuine material
    # bindings become invisible to the native quality gate.  Author the
    # de-instancing opinion only in the anonymous session layer before
    # flattening: the locked provider files stay untouched while the HERO
    # receives the exact composed meshes and their original material bindings.
    stage.SetEditTarget(stage.GetSessionLayer())
    for prim in stage.Traverse():
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
    flattened = stage.Flatten()
    if flattened is None:
        raise CampaignAssetBundleError(
            f"OpenUSD could not flatten the locked NVIDIA HERO source: "
            f"{official_wrapper}"
        )
    unresolved: list[str] = []

    def rebase_asset_path(raw: str) -> str:
        value = str(raw or "").strip()
        if not value:
            return value
        if "://" in value:
            unresolved.append(value)
            return value
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (official_wrapper.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or not _inside(bundle_root, candidate)
        ):
            unresolved.append(value)
            return value
        return os.path.relpath(candidate, hero_path.parent).replace("\\", "/")

    try:
        UsdUtils.ModifyAssetPaths(flattened, rebase_asset_path)
    except Exception as exc:
        raise CampaignAssetBundleError(
            f"failed to rebase flattened HERO dependencies for {official_wrapper}"
        ) from exc
    if unresolved:
        raise CampaignAssetBundleError(
            "flattened HERO retained non-local or unlocked asset dependencies: "
            + ", ".join(sorted(set(unresolved))[:20])
        )
    hero_path.parent.mkdir(parents=True, exist_ok=True)
    if not flattened.Export(str(hero_path)):
        raise CampaignAssetBundleError(
            f"OpenUSD failed to export flattened HERO stage: {hero_path}"
        )


def _scene_optimizer_arguments(retained_percent: float) -> dict[str, object]:
    """Return the supported arguments for one decimation target."""

    arguments = {
        "paths": [],
        # Scene Optimizer calls 100 a no-op.  This value is the retained
        # percentage, even though the operation argument is named factor.
        "reductionFactor": float(retained_percent),
        "maxMeanError": 0.0,
        "cpuVertexCountThreshold": 0,
        "gpuVertexCountThreshold": 0,
        # Match NVIDIA's supported decimation preset. Pinning every boundary
        # prevents assets made of open mesh sections from reaching distinct
        # MID and FAR targets; identity is guarded by the post-op bounds gate.
        "pinBoundaries": False,
    }
    return arguments


def _scene_optimizer_decimate(
    *,
    stage_path: Path,
    retained_percent: float,
) -> None:
    """Run NVIDIA's real ``decimateMeshes`` operation on one isolated stage."""

    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        if not manager.is_extension_enabled("omni.scene.optimizer.core"):
            manager.set_extension_enabled_immediate(
                "omni.scene.optimizer.core",
                True,
            )
        from omni.scene.optimizer.core import acquire_interface
        from omni.scene.optimizer.impl.core import ExecutionContext
        from pxr import Usd
    except (ImportError, AttributeError, RuntimeError) as exc:
        raise CampaignAssetBundleError(
            "source_default_only requires the enabled "
            "omni.scene.optimizer.core extension"
        ) from exc

    interface = acquire_interface()
    if interface is None:
        raise CampaignAssetBundleError(
            "omni.scene.optimizer.core.acquire_interface() returned no interface"
        )
    try:
        interface.load_plugins()
        operations = set(interface.get_operations())
    except Exception as exc:
        raise CampaignAssetBundleError(
            "Scene Optimizer could not load its native operation registry"
        ) from exc
    if SCENE_OPTIMIZER_OPERATION not in operations:
        raise CampaignAssetBundleError(
            "Scene Optimizer operation decimateMeshes is unavailable; "
            "source_default_only cannot be promoted to a real LOD chain"
        )
    stage = Usd.Stage.Open(str(stage_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise CampaignAssetBundleError(
            f"Scene Optimizer cannot open the isolated LOD stage: {stage_path}"
        )
    stage.SetEditTarget(stage.GetRootLayer())
    context = ExecutionContext()
    context.set_stage(stage)
    context.captureStats = 1
    context.generateReport = 0
    context.singleThreaded = 0
    arguments = _scene_optimizer_arguments(retained_percent)
    try:
        result = interface.execute_operation(
            SCENE_OPTIMIZER_OPERATION,
            context,
            json.dumps(arguments, sort_keys=True),
        )
        success = bool(result[0]) if isinstance(result, tuple) else bool(result)
        if not success:
            detail = result[1] if isinstance(result, tuple) and len(result) > 1 else result
            raise CampaignAssetBundleError(
                f"Scene Optimizer decimateMeshes failed for "
                f"{stage_path.name} at {retained_percent:g}% retained: {detail}"
            )
        if not stage.GetRootLayer().Save():
            raise CampaignAssetBundleError(
                f"Scene Optimizer could not save its authored LOD: {stage_path}"
            )
    finally:
        try:
            context.remove_stage()
        except Exception:
            pass


def _validate_generated_native_chain(
    *,
    paths: Mapping[str, Path],
    bundle_root: Path,
) -> dict[str, dict[str, Any]]:
    locked_paths = {
        candidate.resolve()
        for candidate in bundle_root.rglob("*")
        if candidate.is_file() and not candidate.is_symlink()
    }
    metrics: dict[str, dict[str, Any]] = {}
    try:
        for level in LOD_LEVELS:
            metrics[level] = _asset_contract._native_usd_metrics(
                paths[level],
                bundle_root=bundle_root,
                locked_paths=locked_paths,
            )
    except (RuntimeError, ValueError) as exc:
        raise CampaignAssetBundleError(
            f"generated Scene Optimizer LOD cannot pass native USD quality: {exc}"
        ) from exc
    # Scene Optimizer's reductionFactor targets vertices.  De-instancing an
    # NVIDIA source may triangulate authored n-gons, so polygon-face count can
    # legitimately rise while the actual vertex workload falls sharply (the
    # live Kasa asset measured 92,682 -> 55,221 -> 18,273 vertices).  Use the
    # native point/vertex metric for the strict LOD chain and retain face count
    # in the recorded metrics for audit; only fall back to faces for a source
    # whose USD metrics genuinely expose no point counts.
    point_complexities = [
        int(metrics[level].get("geometry_point_count", 0))
        for level in LOD_LEVELS
    ]
    complexities = (
        point_complexities
        if all(complexity > 0 for complexity in point_complexities)
        else [
            int(metrics[level]["geometry_face_count"])
            for level in LOD_LEVELS
        ]
    )
    if not complexities[0] > complexities[1]:
        raise CampaignAssetBundleError(
            "Scene Optimizer did not preserve the strict HERO > MID "
            f"geometric chain: complexities={complexities}"
        )
    if not complexities[1] > complexities[2] >= 4:
        raise _FarLodRetryableError(
            "Scene Optimizer FAR did not complete the strict HERO > MID > FAR "
            f"geometric chain: complexities={complexities}"
        )
    hero_dimensions = metrics["HERO"]["world_bounds"]["dimensions"]
    for level in ("MID", "FAR"):
        dimensions = metrics[level]["world_bounds"]["dimensions"]
        ratios = [
            dimensions[axis] / hero_dimensions[axis]
            for axis in range(3)
        ]
        if any(not 0.65 <= ratio <= 1.35 for ratio in ratios):
            error_type = (
                _FarLodRetryableError
                if level == "FAR"
                else CampaignAssetBundleError
            )
            raise error_type(
                f"Scene Optimizer {level} changed asset bounds beyond the "
                f"accepted identity envelope: ratios={ratios}"
            )
    return metrics


def _build_scene_optimizer_lods(
    *,
    official_wrapper: Path,
    output_paths: Mapping[str, Path],
    bundle_root: Path,
) -> dict[str, dict[str, Any]]:
    """Flatten HERO and derive two isolated, natively decimated USD stages."""

    _flatten_hero_stage(
        official_wrapper=official_wrapper,
        hero_path=output_paths["HERO"],
        bundle_root=bundle_root,
    )
    hero_sha256 = _sha256(output_paths["HERO"])
    shutil.copy2(output_paths["HERO"], output_paths["MID"])
    _scene_optimizer_decimate(
        stage_path=output_paths["MID"],
        retained_percent=MID_RETAINED_PERCENT,
    )
    if _sha256(output_paths["HERO"]) != hero_sha256:
        raise CampaignAssetBundleError(
            "Scene Optimizer modified HERO while deriving the isolated MID stage"
        )
    metrics: dict[str, dict[str, Any]] | None = None
    far_retained_percent: float | None = None
    last_far_error: _FarLodRetryableError | None = None
    for retained_percent in FAR_RETAINED_PERCENT_ATTEMPTS:
        # Every retry starts from the immutable flattened HERO.  Reusing the
        # previously decimated FAR would compound topology loss and invalidate
        # the retained-percentage evidence.
        shutil.copy2(output_paths["HERO"], output_paths["FAR"])
        _scene_optimizer_decimate(
            stage_path=output_paths["FAR"],
            retained_percent=retained_percent,
        )
        if _sha256(output_paths["HERO"]) != hero_sha256:
            raise CampaignAssetBundleError(
                "Scene Optimizer modified HERO while deriving the isolated FAR "
                "stage"
            )
        try:
            metrics = _validate_generated_native_chain(
                paths=output_paths,
                bundle_root=bundle_root,
            )
        except _FarLodRetryableError as exc:
            last_far_error = exc
            continue
        far_retained_percent = retained_percent
        break
    if metrics is None or far_retained_percent is None:
        attempted = ", ".join(
            f"{value:g}%" for value in FAR_RETAINED_PERCENT_ATTEMPTS
        )
        raise CampaignAssetBundleError(
            "Scene Optimizer FAR could not satisfy the unchanged identity and "
            f"strict complexity gates after pristine-HERO attempts [{attempted}]: "
            f"{last_far_error}"
        ) from last_far_error
    metrics["HERO"]["scene_optimizer_retained_percent"] = 100.0
    metrics["MID"]["scene_optimizer_retained_percent"] = MID_RETAINED_PERCENT
    metrics["FAR"]["scene_optimizer_retained_percent"] = far_retained_percent
    return metrics


def _copy_asset_plan(
    *,
    plan: _AssetPlan,
    staging_root: Path,
) -> dict[str, Any]:
    sources = {plan.wrapper.path: plan.wrapper}
    sources.update({dependency.path: dependency for dependency in plan.dependencies})
    for source in sources.values():
        relative = _asset_relative_target(source)
        _materialize_file(
            source.path,
            staging_root.joinpath(*relative.parts),
            allow_hardlink=not (
                plan.requires_decimation
                and source.path.suffix.casefold() in USD_SUFFIXES
            ),
        )

    asset_directory = _safe_asset_directory(plan.asset_id)
    lod_records: dict[str, dict[str, object]] = {}
    provider_strategy = str(plan.entry["lod"]["strategy"])
    all_levels = [str(level) for level in plan.entry["lod"]["levels"]]
    official_wrapper_relative = _asset_relative_target(plan.wrapper)
    identity = plan.entry.get("identity")
    source_identity = str(identity["source_identity"])
    lineage = hashlib.sha256(
        f"{plan.asset_id}\0{source_identity}".encode("utf-8")
    ).hexdigest()
    native_generation_metrics: dict[str, dict[str, Any]] | None = None
    generated_paths: dict[str, tuple[PurePosixPath, Path]] = {}
    if plan.requires_decimation:
        for output_level in LOD_LEVELS:
            relative = (
                PurePosixPath("lod")
                / asset_directory
                / f"{output_level.casefold()}.usdc"
            )
            generated_paths[output_level] = (
                relative,
                staging_root.joinpath(*relative.parts),
            )
        copied_official_paths = {
            staging_root.joinpath(*_asset_relative_target(source).parts): source.sha256
            for source in sources.values()
        }
        try:
            native_generation_metrics = _build_scene_optimizer_lods(
                official_wrapper=staging_root.joinpath(
                    *official_wrapper_relative.parts
                ),
                output_paths={
                    level: generated_paths[level][1] for level in LOD_LEVELS
                },
                bundle_root=staging_root,
            )
        except CampaignAssetBundleError as exc:
            raise CampaignAssetBundleError(
                f"{plan.label} asset_id={plan.asset_id!r} failed native "
                f"Scene Optimizer LOD generation: {exc}"
            ) from exc
        changed_sources = [
            str(path)
            for path, expected in copied_official_paths.items()
            if not path.is_file() or _sha256(path) != expected
        ]
        if changed_sources:
            raise CampaignAssetBundleError(
                "Scene Optimizer modified locked NVIDIA source layers instead "
                "of isolated LOD edit targets: "
                + ", ".join(changed_sources[:20])
            )
    else:
        for output_level in LOD_LEVELS:
            relative = (
                PurePosixPath("lod")
                / asset_directory
                / f"{output_level.casefold()}.usda"
            )
            path = staging_root.joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                _lod_wrapper_text(
                    official_wrapper_relative=official_wrapper_relative,
                    wrapper_relative=relative,
                    strategy=provider_strategy,
                    selected_level=plan.selected_levels[output_level],
                    all_levels=all_levels,
                    output_level=output_level,
                    asset_id=plan.asset_id,
                ),
                encoding="utf-8",
                newline="\n",
            )
            generated_paths[output_level] = (relative, path)
    for output_level in LOD_LEVELS:
        relative, path = generated_paths[output_level]
        retained_percent = (
            float(
                native_generation_metrics[output_level].get(
                    "scene_optimizer_retained_percent",
                    {
                        "HERO": 100.0,
                        "MID": MID_RETAINED_PERCENT,
                        "FAR": FAR_RETAINED_PERCENT,
                    }[output_level],
                )
            )
            if native_generation_metrics is not None
            else None
        )
        lod_records[output_level] = {
            "path": relative.as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "prim_path": "/Asset",
            "lineage_id": lineage,
            "native_level": (
                f"{SCENE_OPTIMIZER_OPERATION}:retain={retained_percent:g}"
                if retained_percent is not None
                else plan.selected_levels[output_level]
            ),
            "native_strategy": plan.lod_strategy,
        }
    if len({record["sha256"] for record in lod_records.values()}) != 3:
        raise CampaignAssetBundleError(
            f"{plan.label} generated duplicate LOD wrappers, refusing bundle"
        )
    materialized_records = [
        _source_record(source, prefix="official")
        for source in sorted(
            sources.values(),
            key=lambda item: item.relative_to_manifest_parent.as_posix(),
        )
    ]
    materialized_records.extend(
        {
            "path": record["path"],
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
        for record in lod_records.values()
    )
    entry = dict(plan.entry)
    entry.update(
        {
            "path": lod_records["HERO"]["path"],
            "sha256": lod_records["HERO"]["sha256"],
            "source_cache_path": _asset_relative_target(plan.source).as_posix(),
            "materialized_files": materialized_records,
            "content_lock_sha256": hashlib.sha256(
                json.dumps(materialized_records, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "lod_lineage_id": lineage,
            "lod_paths": lod_records,
            "ground_anchor_m": list(plan.entry.get("ground_anchor_m") or []),
            "quality_validation": "native_metadata_passed",
            "lod": (
                {
                    "state": "passed",
                    "strategy": "scene_optimizer_decimateMeshes",
                    "levels": list(LOD_LEVELS),
                    "level_count": len(LOD_LEVELS),
                }
                if plan.requires_decimation
                else entry["lod"]
            ),
            "campaign_lod_generation": {
                "strategy": plan.lod_strategy,
                "scene_optimizer_operation": (
                    SCENE_OPTIMIZER_OPERATION
                    if plan.requires_decimation
                    else None
                ),
                "retained_percent": (
                    {
                        level: float(
                            native_generation_metrics[level].get(
                                "scene_optimizer_retained_percent",
                                {
                                    "HERO": 100.0,
                                    "MID": MID_RETAINED_PERCENT,
                                    "FAR": FAR_RETAINED_PERCENT,
                                }[level],
                            )
                        )
                        for level in LOD_LEVELS
                    }
                    if plan.requires_decimation
                    else None
                ),
                "native_metrics_sha256": (
                    _canonical_sha256(native_generation_metrics)
                    if native_generation_metrics is not None
                    else None
                ),
            },
        }
    )
    if (
        len(entry["ground_anchor_m"]) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in entry["ground_anchor_m"]
        )
    ):
        raise CampaignAssetBundleError(
            f"{plan.label}.ground_anchor_m is missing or invalid"
        )
    entry["metadata_validation_sha256"] = _canonical_sha256(
        {
            key: entry.get(key)
            for key in (
                "native_dimensions_m",
                "ground_anchor_m",
                "anchor_validation",
                "lod",
                "materials",
                "placement",
            )
        }
    )
    return entry


def _copy_ground_materials(
    *,
    payload: Mapping[str, Any],
    manifest_parent: Path,
    volume_root: Path,
    staging_root: Path,
) -> dict[str, Any]:
    if (
        payload.get("schema_version") != 3
        or payload.get("state") != "GROUND_PBR_MATERIALS_INSTALLED"
    ):
        raise CampaignAssetBundleError(
            "ground PBR manifest is not a completed v3 installation"
        )
    materials = payload.get("pbr_materials")
    if not isinstance(materials, Mapping) or set(materials) != set(
        PBR_MATERIAL_ROLES
    ):
        raise CampaignAssetBundleError(
            "ground PBR manifest must contain exactly the seven required roles"
        )
    output: dict[str, Any] = {}
    for role in PBR_MATERIAL_ROLES:
        raw = materials[role]
        if not isinstance(raw, Mapping):
            raise CampaignAssetBundleError(f"pbr_materials.{role} is malformed")
        material_record = raw.get("material_file")
        if not isinstance(material_record, Mapping):
            raise CampaignAssetBundleError(
                f"pbr_materials.{role}.material_file is malformed"
            )
        material = _locked_source(
            record=material_record,
            manifest_parent=manifest_parent,
            volume_root=volume_root,
            label=f"pbr_materials.{role}.material_file",
            allowed_suffixes=USD_SUFFIXES,
        )
        textures = raw.get("textures")
        if not isinstance(textures, Mapping) or not set(
            PBR_REQUIRED_TEXTURES
        ).issubset(textures):
            raise CampaignAssetBundleError(
                f"pbr_materials.{role} lacks base_color, normal or roughness"
            )
        copied_textures: dict[str, dict[str, Any]] = {}
        for semantic, record in textures.items():
            if not isinstance(record, Mapping):
                raise CampaignAssetBundleError(
                    f"pbr_materials.{role}.textures.{semantic} is malformed"
                )
            source = _locked_source(
                record=record,
                manifest_parent=manifest_parent,
                volume_root=volume_root,
                label=f"pbr_materials.{role}.textures.{semantic}",
                allowed_suffixes=TEXTURE_SUFFIXES,
            )
            target = PurePosixPath("ground") / source.relative_to_manifest_parent
            _materialize_file(
                source.path,
                staging_root.joinpath(*target.parts),
            )
            if semantic in PBR_REQUIRED_TEXTURES:
                copied_textures[semantic] = {
                    **dict(record),
                    "path": target.as_posix(),
                    "sha256": source.sha256,
                    "size_bytes": source.size_bytes,
                }
        material_target = (
            PurePosixPath("ground") / material.relative_to_manifest_parent
        )
        _materialize_file(
            material.path,
            staging_root.joinpath(*material_target.parts),
        )
        source_metadata = dict(raw.get("source") or {})
        materialx_record = source_metadata.get("materialx")
        if isinstance(materialx_record, Mapping):
            materialx = _locked_source(
                record=materialx_record,
                manifest_parent=manifest_parent,
                volume_root=volume_root,
                label=f"pbr_materials.{role}.source.materialx",
            )
            materialx_target = (
                PurePosixPath("ground")
                / materialx.relative_to_manifest_parent
            )
            _materialize_file(
                materialx.path,
                staging_root.joinpath(*materialx_target.parts),
            )
            source_metadata["materialx"] = {
                **dict(materialx_record),
                "path": materialx_target.as_posix(),
                "sha256": materialx.sha256,
                "size_bytes": materialx.size_bytes,
            }
        output[role] = {
            **dict(raw),
            "material_file": {
                **dict(material_record),
                "path": material_target.as_posix(),
                "sha256": material.sha256,
                "size_bytes": material.size_bytes,
            },
            # The source ground graph currently connects these three branches.
            # An unconnected displacement download is retained on input but is
            # not claimed as native PBR evidence in the combined manifest.
            "textures": copied_textures,
            "source": source_metadata,
        }
    return output


def _validate_environment_bundle_manifest(
    manifest: Path,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, dict[str, object]]],
    dict[str, dict[str, dict[str, object]]],
]:
    """Validate the actor-free base bundle without invoking campaign gates."""

    payload = _read_json(manifest, label="base-landscape asset manifest")
    _validate_official_header(payload)
    if payload.get("actors") not in ({}, None):
        raise ValueError("base-landscape environment bundle must not contain actors")
    if "selected_actor_group" in payload:
        raise ValueError(
            "base-landscape environment bundle must not contain selected actors"
        )
    pbr_summary = _asset_contract._validate_pbr_materials(
        payload=payload,
        manifest_parent=manifest.parent,
    )
    lod_summary: dict[str, dict[str, dict[str, object]]] = {}
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for role, entry in _asset_contract._environment_entries(payload):
        summary = _asset_contract._validate_asset_lod_paths(
            role=role,
            entry=entry,
            manifest_parent=manifest.parent,
        )
        for level in LOD_LEVELS:
            path_key = str(summary[level]["path"]).casefold()
            digest = str(summary[level]["sha256"])
            if path_key in seen_paths or digest in seen_hashes:
                raise ValueError(
                    "every base-landscape HERO, MID and FAR representation "
                    "must be a unique local USD stage"
                )
            seen_paths.add(path_key)
            seen_hashes.add(digest)
        lod_summary[role] = summary

    selected_group = payload.get("selected_environment_group")
    selected_assets = (
        selected_group.get("assets")
        if isinstance(selected_group, Mapping)
        else None
    )
    selected_keys = set(selected_assets) if isinstance(selected_assets, Mapping) else set()
    if (
        not isinstance(selected_group, Mapping)
        or selected_group.get("group_id") != SELECTED_ENVIRONMENT_GROUP_ID
        or selected_group.get("selection_count")
        != len(SELECTED_ENVIRONMENT_GROUP_IDS)
        or selected_group.get("selection_order")
        != list(SELECTED_ENVIRONMENT_GROUP_IDS)
        or not isinstance(selected_assets, Mapping)
        or selected_keys != set(SELECTED_ENVIRONMENT_GROUP_IDS)
    ):
        raise ValueError(
            "base-landscape environment bundle does not retain the exact "
            "selected_environment_group contract"
        )
    selected_summary: dict[str, dict[str, dict[str, object]]] = {}
    environment = payload["environment"]
    for selection_id in SELECTED_ENVIRONMENT_GROUP_IDS:
        reference = selected_assets[selection_id]
        if not isinstance(reference, Mapping):
            raise ValueError(
                f"selected_environment_group.assets.{selection_id} is malformed"
            )
        kind, family = SELECTED_ENVIRONMENT_TARGET_BY_ID[selection_id]
        if (
            reference.get("selection_id") != selection_id
            or reference.get("environment_kind") != kind
            or reference.get("environment_family") != family
            or "lod_paths" in reference
        ):
            raise ValueError(
                f"selected environment {selection_id} must be a non-duplicating "
                f"reference into environment.{kind}.{family}"
            )
        try:
            environment_index = int(reference["environment_index"])
            candidate = environment[kind][family][environment_index]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"selected environment {selection_id} has an invalid "
                "environment_index"
            ) from exc
        if (
            not isinstance(candidate, Mapping)
            or candidate.get("asset_id") != reference.get("asset_id")
            or candidate.get("selection_id") != selection_id
        ):
            raise ValueError(
                f"selected environment {selection_id} does not resolve to its "
                f"single environment.{kind}.{family} entry"
            )
        role = f"{kind}.{family}[{environment_index}]"
        selected_summary[selection_id] = lod_summary[role]
    return pbr_summary, lod_summary, selected_summary


def _reuse_existing(
    *,
    destination: Path,
    official_manifest_sha256: str,
    ground_manifest_sha256: str,
    receipt_path: Path,
    bundle_mode: str = CAMPAIGN_BUNDLE_MODE,
) -> dict[str, Any]:
    marker_path = destination / INSTALL_MARKER
    marker = _read_json(marker_path, label="campaign asset install marker")
    bundle_sha = str(marker.get("bundle_sha256", "")).strip().lower()
    marker_mode = str(marker.get("bundle_mode") or CAMPAIGN_BUNDLE_MODE)
    if (
        marker.get("state") != "ASSET_BUNDLE_INSTALLED"
        or marker_mode != bundle_mode
        or not _SHA256_RE.fullmatch(bundle_sha)
        or marker.get("official_manifest_sha256") != official_manifest_sha256
        or marker.get("ground_manifest_sha256") != ground_manifest_sha256
    ):
        raise CampaignAssetBundleError(
            "existing campaign asset bundle is bound to different inputs"
        )
    manifest_relative = _relative_path(
        marker.get("manifest_relative"),
        label="asset bundle manifest",
    )
    try:
        if bundle_mode == CAMPAIGN_BUNDLE_MODE:
            validated = _asset_contract._validate_reuse(
                destination=destination,
                expected_sha256=bundle_sha,
                manifest_relative=manifest_relative,
            )
        else:
            manifest = destination.joinpath(*manifest_relative.parts)
            if (
                marker.get("runtime_manifest_sha256") != _sha256(manifest)
                or not manifest.is_file()
            ):
                raise RuntimeError(
                    "persisted base-landscape manifest drifted from its marker"
                )
            pbr_summary, lod_summary, selected_summary = (
                _validate_environment_bundle_manifest(manifest)
            )
            inventory, unpacked_bytes = _asset_contract._inventory(destination)
            if (
                marker.get("file_count") != len(inventory)
                or marker.get("unpacked_bytes") != unpacked_bytes
                or marker.get("content_inventory_sha256")
                != _canonical_sha256(inventory)
                or marker.get("pbr_materials_sha256")
                != _canonical_sha256(pbr_summary)
                or marker.get("asset_lod_library_sha256")
                != _canonical_sha256(lod_summary)
                or marker.get("selected_environment_lod_library_sha256")
                != _canonical_sha256(selected_summary)
            ):
                raise RuntimeError(
                    "persisted base-landscape bundle content drifted from its marker"
                )
            validated = marker
    except (RuntimeError, ValueError) as exc:
        raise CampaignAssetBundleError(
            "existing asset bundle failed its locked inventory contract"
        ) from exc
    _atomic_write_json(receipt_path, validated)
    return {
        "state": (
            ASSEMBLY_STATE
            if bundle_mode == CAMPAIGN_BUNDLE_MODE
            else BASE_LANDSCAPE_ASSEMBLY_STATE
        ),
        "reused": True,
        "bundle_root": str(destination),
        "manifest": str(
            destination.joinpath(
                *_relative_path(
                    marker["manifest_relative"],
                    label="campaign asset manifest",
                ).parts
            )
        ),
        "receipt": str(receipt_path),
        "asset_count": marker.get("asset_count"),
        "selected_actor_count": marker.get("selected_actor_count"),
        "selected_environment_count": marker.get(
            "selected_environment_count"
        ),
        "material_count": len(PBR_MATERIAL_ROLES),
    }


def assemble_campaign_asset_bundle(
    *,
    volume_root: Path,
    official_manifest_path: Path,
    ground_manifest_path: Path,
    destination_root: Path,
    receipt_path: Path,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
) -> dict[str, Any]:
    """Build one immutable campaign bundle from already local, locked inputs."""

    volume = volume_root.resolve()
    official_manifest = official_manifest_path.resolve()
    ground_manifest = ground_manifest_path.resolve()
    destination = destination_root.resolve()
    receipt = receipt_path.resolve()
    if (
        not volume.is_dir()
        or volume.is_symlink()
        or not official_manifest.is_file()
        or official_manifest.is_symlink()
        or not ground_manifest.is_file()
        or ground_manifest.is_symlink()
        or not _inside(volume, official_manifest)
        or not _inside(volume, ground_manifest)
        or not _inside(volume, destination)
        or destination == volume
        or not _inside(volume, receipt)
        or receipt == destination
    ):
        raise CampaignAssetBundleError(
            "volume, input manifests, destination and receipt must be safe local "
            "paths below the production volume"
        )
    manifest_relative = _relative_path(
        manifest_name, label="combined campaign manifest name"
    )
    if len(manifest_relative.parts) != 1 or Path(manifest_name).suffix != ".json":
        raise CampaignAssetBundleError(
            "combined campaign manifest must be one JSON filename"
        )

    official_sha = _sha256(official_manifest)
    ground_sha = _sha256(ground_manifest)
    if destination.exists():
        return _reuse_existing(
            destination=destination,
            official_manifest_sha256=official_sha,
            ground_manifest_sha256=ground_sha,
            receipt_path=receipt,
        )

    official_payload = _read_json(
        official_manifest, label="official NVIDIA manifest"
    )
    ground_payload = _read_json(ground_manifest, label="ground PBR manifest")
    plans = _plan_assets(
        payload=official_payload,
        manifest_parent=official_manifest.parent,
        volume_root=volume,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        environment: dict[str, dict[str, list[dict[str, Any]]]] = {
            kind: {family: [] for family in family_minimums}
            for kind, family_minimums in PHOTOREAL_FAMILY_MINIMUMS.items()
        }
        selected_actor_assets: dict[str, dict[str, Any]] = {}
        selected_environment_assets: dict[str, dict[str, Any]] = {}
        actors: dict[str, dict[str, Any]] = {}
        for plan in plans:
            copied = _copy_asset_plan(plan=plan, staging_root=staging)
            if plan.kind == "selected_actor_group":
                selected_actor_assets[plan.family] = {
                    **copied,
                    "selection_id": plan.family,
                    "selection_source_url": (
                        SELECTED_ACTOR_GROUP_SOURCE_BY_ID[plan.family]
                    ),
                    "placement_class": (
                        SELECTED_ACTOR_GROUP_PLACEMENT_BY_ID[plan.family]
                    ),
                }
            elif plan.kind == "actors":
                actors[plan.family] = copied
            elif plan.kind == "selected_environment_group":
                kind, family = SELECTED_ENVIRONMENT_TARGET_BY_ID[plan.family]
                selected = {
                    **copied,
                    "selection_id": plan.family,
                    "environment_kind": kind,
                    "environment_family": family,
                }
                selected_environment_assets[plan.family] = selected
                environment[kind][family].append(selected)
            else:
                environment[plan.kind][plan.family].append(copied)
        pbr_materials = _copy_ground_materials(
            payload=ground_payload,
            manifest_parent=ground_manifest.parent,
            volume_root=volume,
            staging_root=staging,
        )
        manifest_payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "profile": MANIFEST_PROFILE,
            "library_policy": dict(PHOTOREAL_LIBRARY_POLICY),
            "family_minimums": PHOTOREAL_FAMILY_MINIMUMS,
            "discovery": {
                "mode": "materialized_photoreal_asset_library_v3",
                "missing_environment": [],
                "source": "official_nvidia_plus_polyhaven_ground",
                "official_manifest_sha256": official_sha,
                "ground_manifest_sha256": ground_sha,
                "lod_contract": (
                    "provider_native_or_scene_optimizer_decimated_hero_mid_far"
                ),
            },
            "environment": environment,
            "actors": actors,
            "selected_actor_group": {
                "group_id": SELECTED_ACTOR_GROUP_ID,
                "selection_count": len(SELECTED_ACTOR_GROUP_IDS),
                "selection_order": list(SELECTED_ACTOR_GROUP_IDS),
                "assets": selected_actor_assets,
                "usage_contract": (
                    "all_selected_assets_must_be_placed_across_the_20_scene_campaign"
                ),
            },
            "selected_environment_group": {
                "group_id": SELECTED_ENVIRONMENT_GROUP_ID,
                "selection_count": len(SELECTED_ENVIRONMENT_GROUP_IDS),
                "selection_order": list(SELECTED_ENVIRONMENT_GROUP_IDS),
                "assets": selected_environment_assets,
                "usage_contract": (
                    "all_four_assets_are_additive_and_used_in_every_variant"
                ),
            },
            "pbr_materials": pbr_materials,
        }
        manifest = staging.joinpath(*manifest_relative.parts)
        _atomic_write_json(manifest, manifest_payload)
        try:
            normalized_payload = _asset_contract._read_manifest(manifest)
            pbr_summary = _asset_contract._validate_pbr_materials(
                payload=normalized_payload,
                manifest_parent=manifest.parent,
            )
            lod_summary = _asset_contract._validate_asset_lod_library(
                payload=normalized_payload,
                manifest_parent=manifest.parent,
            )
            selected_actor_lod_summary = {
                selection_id: _asset_contract._validate_asset_lod_paths(
                    role=f"selected_actor_group.assets.{selection_id}",
                    entry=normalized_payload["selected_actor_group"]["assets"][
                        selection_id
                    ],
                    manifest_parent=manifest.parent,
                )
                for selection_id in SELECTED_ACTOR_GROUP_IDS
            }
            main_lod_identities = {
                (
                    str(record["path"]).casefold(),
                    str(record["sha256"]),
                )
                for summary in lod_summary.values()
                for record in summary.values()
            }
            selected_lod_identities = [
                (
                    str(record["path"]).casefold(),
                    str(record["sha256"]),
                )
                for summary in selected_actor_lod_summary.values()
                for record in summary.values()
            ]
            if (
                len(set(selected_lod_identities))
                != len(selected_lod_identities)
                or any(
                    identity in main_lod_identities
                    for identity in selected_lod_identities
                )
            ):
                raise ValueError(
                    "selected actor HERO/MID/FAR stages must be unique and "
                    "independent from the seven semantic-role minima"
                )
        except (RuntimeError, ValueError) as exc:
            raise CampaignAssetBundleError(
                "combined manifest does not satisfy the native asset bundle "
                f"structural contract: {exc}"
            ) from exc
        inventory, unpacked_bytes = _asset_contract._inventory(staging)
        source_contract = {
            "official_manifest_sha256": official_sha,
            "ground_manifest_sha256": ground_sha,
            "asset_native_levels": {
                plan.asset_id: dict(plan.selected_levels) for plan in plans
            },
            "selected_actor_group_id": SELECTED_ACTOR_GROUP_ID,
            "selected_actor_group_ids": list(SELECTED_ACTOR_GROUP_IDS),
            "selected_environment_group_id": SELECTED_ENVIRONMENT_GROUP_ID,
            "selected_environment_group_ids": list(
                SELECTED_ENVIRONMENT_GROUP_IDS
            ),
        }
        marker = {
            "schema_version": 1,
            "state": "ASSET_BUNDLE_INSTALLED",
            "bundle_sha256": _canonical_sha256(source_contract),
            "manifest_relative": manifest_relative.as_posix(),
            "source_manifest_sha256": _canonical_sha256(
                {
                    "official": official_sha,
                    "ground": ground_sha,
                }
            ),
            "runtime_manifest_sha256": _sha256(manifest),
            "official_manifest_sha256": official_sha,
            "ground_manifest_sha256": ground_sha,
            "file_count": len(inventory),
            "unpacked_bytes": unpacked_bytes,
            "content_inventory_sha256": _canonical_sha256(inventory),
            "pbr_material_roles": list(PBR_MATERIAL_ROLES),
            "pbr_materials_sha256": _canonical_sha256(pbr_summary),
            "asset_lod_levels": list(LOD_LEVELS),
            "asset_lod_library_sha256": _canonical_sha256(lod_summary),
            "selected_actor_lod_library_sha256": _canonical_sha256(
                selected_actor_lod_summary
            ),
            "asset_count": len(plans),
            "selected_actor_count": len(SELECTED_ACTOR_GROUP_IDS),
            "selected_actor_group_id": SELECTED_ACTOR_GROUP_ID,
            "selected_environment_count": len(
                SELECTED_ENVIRONMENT_GROUP_IDS
            ),
            "selected_environment_group_id": (
                SELECTED_ENVIRONMENT_GROUP_ID
            ),
            "assembly_state": ASSEMBLY_STATE,
            "proof_boundary": (
                "local source locks and structurally distinct provider-native "
                "or Scene Optimizer-decimated LOD stages; native USD geometry "
                "and shader validation remain required before composition"
            ),
        }
        _atomic_write_json(staging / INSTALL_MARKER, marker)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _atomic_write_json(receipt, marker)
    return {
        "state": ASSEMBLY_STATE,
        "reused": False,
        "bundle_root": str(destination),
        "manifest": str(destination.joinpath(*manifest_relative.parts)),
        "receipt": str(receipt),
        "asset_count": len(plans),
        "selected_actor_count": len(SELECTED_ACTOR_GROUP_IDS),
        "selected_environment_count": len(SELECTED_ENVIRONMENT_GROUP_IDS),
        "material_count": len(PBR_MATERIAL_ROLES),
        "lod_wrapper_count": len(plans) * len(LOD_LEVELS),
        "unpacked_bytes": marker["unpacked_bytes"],
    }


def assemble_base_landscape_environment_bundle(
    *,
    volume_root: Path,
    official_manifest_path: Path,
    ground_manifest_path: Path,
    destination_root: Path,
    receipt_path: Path,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
) -> dict[str, Any]:
    """Build the actor-free landscape bundle used by the first scene.

    This explicit mode consumes only the complete environment tree, the four
    locked selected environment additions and the ground PBR library.  The
    stricter campaign assembler remains the only path that may consume actors.
    """

    volume = volume_root.resolve()
    official_manifest = official_manifest_path.resolve()
    ground_manifest = ground_manifest_path.resolve()
    destination = destination_root.resolve()
    receipt = receipt_path.resolve()
    if (
        not volume.is_dir()
        or volume.is_symlink()
        or not official_manifest.is_file()
        or official_manifest.is_symlink()
        or not ground_manifest.is_file()
        or ground_manifest.is_symlink()
        or not _inside(volume, official_manifest)
        or not _inside(volume, ground_manifest)
        or not _inside(volume, destination)
        or destination == volume
        or not _inside(volume, receipt)
        or receipt == destination
    ):
        raise CampaignAssetBundleError(
            "volume, input manifests, destination and receipt must be safe local "
            "paths below the production volume"
        )
    manifest_relative = _relative_path(
        manifest_name, label="combined base-landscape manifest name"
    )
    if len(manifest_relative.parts) != 1 or Path(manifest_name).suffix != ".json":
        raise CampaignAssetBundleError(
            "combined base-landscape manifest must be one JSON filename"
        )

    official_sha = _sha256(official_manifest)
    ground_sha = _sha256(ground_manifest)
    if destination.exists():
        return _reuse_existing(
            destination=destination,
            official_manifest_sha256=official_sha,
            ground_manifest_sha256=ground_sha,
            receipt_path=receipt,
            bundle_mode=BASE_LANDSCAPE_BUNDLE_MODE,
        )

    official_payload = _read_json(
        official_manifest, label="official NVIDIA manifest"
    )
    ground_payload = _read_json(ground_manifest, label="ground PBR manifest")
    plans = _plan_environment_assets(
        payload=official_payload,
        manifest_parent=official_manifest.parent,
        volume_root=volume,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        environment: dict[str, dict[str, list[dict[str, Any]]]] = {
            kind: {family: [] for family in family_minimums}
            for kind, family_minimums in PHOTOREAL_FAMILY_MINIMUMS.items()
        }
        selected_environment_assets: dict[str, dict[str, Any]] = {}
        for plan in plans:
            copied = _copy_asset_plan(plan=plan, staging_root=staging)
            if plan.kind == "selected_environment_group":
                kind, family = SELECTED_ENVIRONMENT_TARGET_BY_ID[plan.family]
                selected = {
                    **copied,
                    "selection_id": plan.family,
                    "environment_kind": kind,
                    "environment_family": family,
                }
                environment_index = len(environment[kind][family])
                environment[kind][family].append(selected)
                selected_environment_assets[plan.family] = {
                    "selection_id": plan.family,
                    "asset_id": selected["asset_id"],
                    "environment_kind": kind,
                    "environment_family": family,
                    "environment_index": environment_index,
                }
            else:
                environment[plan.kind][plan.family].append(copied)
        pbr_materials = _copy_ground_materials(
            payload=ground_payload,
            manifest_parent=ground_manifest.parent,
            volume_root=volume,
            staging_root=staging,
        )
        manifest_payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "profile": MANIFEST_PROFILE,
            "library_policy": dict(PHOTOREAL_LIBRARY_POLICY),
            "family_minimums": PHOTOREAL_FAMILY_MINIMUMS,
            "discovery": {
                "mode": "materialized_photoreal_asset_library_v3",
                "missing_environment": [],
                "source": "official_nvidia_plus_polyhaven_ground",
                "official_manifest_sha256": official_sha,
                "ground_manifest_sha256": ground_sha,
                "lod_contract": (
                    "provider_native_or_scene_optimizer_decimated_hero_mid_far"
                ),
            },
            "environment": environment,
            "selected_environment_group": {
                "group_id": SELECTED_ENVIRONMENT_GROUP_ID,
                "selection_count": len(SELECTED_ENVIRONMENT_GROUP_IDS),
                "selection_order": list(SELECTED_ENVIRONMENT_GROUP_IDS),
                "assets": selected_environment_assets,
                "usage_contract": (
                    "all_four_assets_are_additive_and_used_in_every_variant"
                ),
            },
            "pbr_materials": pbr_materials,
        }
        manifest = staging.joinpath(*manifest_relative.parts)
        _atomic_write_json(manifest, manifest_payload)
        try:
            pbr_summary, lod_summary, selected_summary = (
                _validate_environment_bundle_manifest(manifest)
            )
        except (RuntimeError, ValueError) as exc:
            raise CampaignAssetBundleError(
                "combined base-landscape manifest does not satisfy the native "
                f"environment bundle structural contract: {exc}"
            ) from exc
        inventory, unpacked_bytes = _asset_contract._inventory(staging)
        source_contract = {
            "bundle_mode": BASE_LANDSCAPE_BUNDLE_MODE,
            "official_manifest_sha256": official_sha,
            "ground_manifest_sha256": ground_sha,
            "asset_native_levels": {
                plan.asset_id: dict(plan.selected_levels) for plan in plans
            },
            "selected_environment_group_id": SELECTED_ENVIRONMENT_GROUP_ID,
            "selected_environment_group_ids": list(
                SELECTED_ENVIRONMENT_GROUP_IDS
            ),
        }
        marker = {
            "schema_version": 1,
            "state": "ASSET_BUNDLE_INSTALLED",
            "bundle_mode": BASE_LANDSCAPE_BUNDLE_MODE,
            "bundle_sha256": _canonical_sha256(source_contract),
            "manifest_relative": manifest_relative.as_posix(),
            "source_manifest_sha256": _canonical_sha256(
                {
                    "official": official_sha,
                    "ground": ground_sha,
                }
            ),
            "runtime_manifest_sha256": _sha256(manifest),
            "official_manifest_sha256": official_sha,
            "ground_manifest_sha256": ground_sha,
            "file_count": len(inventory),
            "unpacked_bytes": unpacked_bytes,
            "content_inventory_sha256": _canonical_sha256(inventory),
            "pbr_material_roles": list(PBR_MATERIAL_ROLES),
            "pbr_materials_sha256": _canonical_sha256(pbr_summary),
            "asset_lod_levels": list(LOD_LEVELS),
            "asset_lod_library_sha256": _canonical_sha256(lod_summary),
            "selected_environment_lod_library_sha256": (
                _canonical_sha256(selected_summary)
            ),
            "asset_count": len(plans),
            "selected_actor_count": 0,
            "selected_environment_count": len(
                SELECTED_ENVIRONMENT_GROUP_IDS
            ),
            "selected_environment_group_id": SELECTED_ENVIRONMENT_GROUP_ID,
            "assembly_state": BASE_LANDSCAPE_ASSEMBLY_STATE,
            "proof_boundary": (
                "actor-free landscape source locks and structurally distinct "
                "provider-native or Scene Optimizer-decimated LOD stages; "
                "native USD geometry and shader validation remain required "
                "before composition"
            ),
        }
        _atomic_write_json(staging / INSTALL_MARKER, marker)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _atomic_write_json(receipt, marker)
    return {
        "state": BASE_LANDSCAPE_ASSEMBLY_STATE,
        "reused": False,
        "bundle_root": str(destination),
        "manifest": str(destination.joinpath(*manifest_relative.parts)),
        "receipt": str(receipt),
        "asset_count": len(plans),
        "selected_actor_count": 0,
        "selected_environment_count": len(SELECTED_ENVIRONMENT_GROUP_IDS),
        "material_count": len(PBR_MATERIAL_ROLES),
        "lod_wrapper_count": len(plans) * len(LOD_LEVELS),
        "unpacked_bytes": marker["unpacked_bytes"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the materialized NVIDIA library and local ground PBR "
            "materials into one strict native campaign asset bundle"
        )
    )
    parser.add_argument("--volume-root", required=True, type=Path)
    parser.add_argument("--official-manifest", required=True, type=Path)
    parser.add_argument("--ground-manifest", required=True, type=Path)
    parser.add_argument("--destination-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    parser.add_argument(
        "--mode",
        choices=(CAMPAIGN_BUNDLE_MODE, BASE_LANDSCAPE_BUNDLE_MODE),
        default=CAMPAIGN_BUNDLE_MODE,
    )
    return parser


def _start_isaac_runtime() -> object:
    """Start the standalone Kit runtime required by Scene Optimizer."""

    try:
        from isaacsim import SimulationApp
    except ImportError as exc:  # pragma: no cover - pod-only dependency
        raise CampaignAssetBundleError(
            "campaign asset assembly requires Isaac Sim's packaged Python "
            "runtime so Scene Optimizer can start"
        ) from exc
    try:
        return SimulationApp(
            {
                "headless": True,
                "fast_shutdown": False,
            }
        )
    except Exception as exc:  # pragma: no cover - pod-only runtime failure
        raise CampaignAssetBundleError(
            "Isaac Sim could not start the headless Kit runtime required by "
            "Scene Optimizer"
        ) from exc


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    runtime = _start_isaac_runtime()
    try:
        assemble = (
            assemble_base_landscape_environment_bundle
            if args.mode == BASE_LANDSCAPE_BUNDLE_MODE
            else assemble_campaign_asset_bundle
        )
        result = assemble(
            volume_root=args.volume_root,
            official_manifest_path=args.official_manifest,
            ground_manifest_path=args.ground_manifest,
            destination_root=args.destination_root,
            receipt_path=args.receipt,
            manifest_name=args.manifest_name,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    finally:
        runtime.close()


__all__ = [
    "ASSEMBLY_STATE",
    "CampaignAssetBundleError",
    "SELECTED_ACTOR_GROUP_ID",
    "SELECTED_ACTOR_GROUP_IDS",
    "SELECTED_ACTOR_GROUP_PLACEMENT_BY_ID",
    "SELECTED_ACTOR_GROUP_SOURCES",
    "SELECTED_ENVIRONMENT_GROUP",
    "SELECTED_ENVIRONMENT_GROUP_ID",
    "SELECTED_ENVIRONMENT_GROUP_IDS",
    "SELECTED_ENVIRONMENT_TARGET_BY_ID",
    "assemble_base_landscape_environment_bundle",
    "assemble_campaign_asset_bundle",
]


if __name__ == "__main__":
    raise SystemExit(main())

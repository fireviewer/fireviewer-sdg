"""Merge the curated response assets with the corrected NVIDIA source manifest.

The curated archive remains immutable.  This module writes a distinct source
manifest next to the curated manifest, selects the corrected official
environment first, uses curated environment assets only as deterministic
backfill, and preserves the exact curated actor set.

The three reviewed Objaverse building families are deliberately left empty.
``community_building_assets`` is the only stage allowed to fill them.  The
verification path can therefore accept either the byte-exact base merge or the
one narrowly-defined community augmentation, while rejecting every other
mutation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from fireviewer_sdg.asset_bundle import (
    INSTALL_MARKER,
    REQUIRED_ACTOR_CLASSES,
)
from fireviewer_sdg.community_building_assets import (
    COMMUNITY_BUILDING_GLB_LOCKS,
    COMMUNITY_BUILDING_TITLES,
    INSTALL_SCHEMA_VERSION as COMMUNITY_INSTALL_SCHEMA_VERSION,
    INSTALL_STATE as COMMUNITY_INSTALL_STATE,
    UID_TO_FAMILY,
)
from fireviewer_sdg.simready_assets import (
    MANIFEST_PROFILE,
    MANIFEST_SCHEMA_VERSION,
    PHOTOREAL_FAMILY_MINIMUMS,
    PHOTOREAL_LIBRARY_POLICY,
)


SOURCE_MERGE_SCHEMA_VERSION = 1
SOURCE_MERGE_STATE = "SOURCE_MANIFESTS_MERGED"
COMMUNITY_BUILDING_FAMILIES = ("agricultural", "industrial", "annex")
COMMUNITY_MISSING_ENVIRONMENT = tuple(
    f"buildings.{family}" for family in COMMUNITY_BUILDING_FAMILIES
)
SELECTION_POLICY = "official_first_curated_backfill_to_family_minimum"
KASA_SOURCE_BASENAME = "kasa_house_01_inst.usd"
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SourceManifestMergeError(RuntimeError):
    """Raised when the locked source-manifest contract cannot be proven."""


@dataclass(frozen=True)
class _EntryLock:
    asset_id: str
    wrapper_sha256: str
    content_lock_sha256: str
    semantic_lock_sha256: str
    wrapper_path: Path
    source_path: Path


@dataclass(frozen=True)
class _MergeContext:
    volume: Path
    curated_manifest: Path
    official_manifest: Path
    output_manifest: Path
    receipt_path: Path
    curated_bundle_root: Path
    curated_bundle_sha256: str
    curated_manifest_sha256: str
    official_manifest_sha256: str
    curated_marker_sha256: str


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


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_fd = os.open(path.parent, flags)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _exclusive_merge_lock(path: Path) -> Any:
    """Hold a crash-released cross-platform lock for merge/read consistency."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise SourceManifestMergeError("source merge lock may not be a symlink")
    stream = path.open("a+b")
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"\0")
        stream.flush()
    stream.seek(0)
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise SourceManifestMergeError(
                    "another source-manifest merge is already running"
                ) from exc
            locked = True
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise SourceManifestMergeError(
                    "another source-manifest merge is already running"
                ) from exc
            locked = True
        yield
    finally:
        try:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(
    *,
    root: Path,
    candidate: Path,
    label: str,
) -> None:
    lexical_root = _absolute_without_resolving(root)
    lexical_candidate = _absolute_without_resolving(candidate)
    if not _inside(lexical_root, lexical_candidate):
        raise SourceManifestMergeError(f"{label} escaped the production volume")
    current = lexical_root
    if current.is_symlink():
        raise SourceManifestMergeError(f"{label} traverses a symlink")
    for part in lexical_candidate.relative_to(lexical_root).parts:
        current = current / part
        if current.is_symlink():
            raise SourceManifestMergeError(f"{label} traverses a symlink")


def _volume_path(
    *,
    volume: Path,
    path: Path,
    label: str,
    require_file: bool,
) -> Path:
    lexical = _absolute_without_resolving(path)
    if not _inside(volume, lexical):
        raise SourceManifestMergeError(f"{label} must stay inside the volume")
    _assert_no_symlink_components(root=volume, candidate=lexical, label=label)
    if require_file and not lexical.is_file():
        raise SourceManifestMergeError(f"{label} is not a regular file: {lexical}")
    return lexical


def _safe_posix_path(
    raw: object,
    *,
    label: str,
    allow_parent: bool,
) -> PurePosixPath:
    if not isinstance(raw, str) or not raw.strip():
        raise SourceManifestMergeError(f"{label} must be a non-empty POSIX path")
    if "\\" in raw or "\x00" in raw or any(ord(char) < 32 for char in raw):
        raise SourceManifestMergeError(f"{label} contains forbidden characters")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or not relative.parts:
        raise SourceManifestMergeError(f"{label} must be relative")
    if not allow_parent and ".." in relative.parts:
        raise SourceManifestMergeError(f"{label} may not contain parent traversal")
    return relative


def _manifest_relative_file(
    *,
    volume: Path,
    manifest_parent: Path,
    raw: object,
    label: str,
) -> Path:
    relative = _safe_posix_path(raw, label=label, allow_parent=True)
    candidate = manifest_parent.joinpath(*relative.parts)
    return _volume_path(
        volume=volume,
        path=candidate,
        label=label,
        require_file=True,
    )


def _volume_relative_file(
    *,
    volume: Path,
    raw: object,
    label: str,
) -> Path:
    relative = _safe_posix_path(raw, label=label, allow_parent=False)
    candidate = volume.joinpath(*relative.parts)
    return _volume_path(
        volume=volume,
        path=candidate,
        label=label,
        require_file=True,
    )


def _relative_to_volume(path: Path, volume: Path, *, label: str) -> str:
    if not _inside(volume, path):
        raise SourceManifestMergeError(f"{label} escaped the production volume")
    return path.relative_to(volume).as_posix()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceManifestMergeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SourceManifestMergeError(f"{label} must be a JSON object")
    return payload


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceManifestMergeError(
            f"{label} must be a non-empty JSON string"
        )
    return value.strip()


def _validate_manifest_header(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected_missing_environment: Sequence[str],
) -> None:
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SourceManifestMergeError(
            f"{label} schema_version must be {MANIFEST_SCHEMA_VERSION}"
        )
    if payload.get("profile") != MANIFEST_PROFILE:
        raise SourceManifestMergeError(
            f"{label} profile must be {MANIFEST_PROFILE}"
        )
    if payload.get("family_minimums") != PHOTOREAL_FAMILY_MINIMUMS:
        raise SourceManifestMergeError(f"{label} family minimums were weakened")
    if payload.get("library_policy") != PHOTOREAL_LIBRARY_POLICY:
        raise SourceManifestMergeError(f"{label} library policy was weakened")
    discovery = payload.get("discovery")
    if (
        not isinstance(discovery, Mapping)
        or discovery.get("mode") != "materialized_photoreal_asset_library_v3"
    ):
        raise SourceManifestMergeError(
            f"{label} is not a materialized photoreal v3 manifest"
        )
    missing = discovery.get("missing_environment")
    if list(missing or []) != list(expected_missing_environment):
        raise SourceManifestMergeError(
            f"{label} missing_environment must be exactly "
            f"{list(expected_missing_environment)}"
        )


def _environment_tree(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    environment = payload.get("environment")
    if not isinstance(environment, Mapping) or set(environment) != set(
        PHOTOREAL_FAMILY_MINIMUMS
    ):
        raise SourceManifestMergeError(
            f"{label} environment family tree is incomplete"
        )
    for kind, family_minimums in PHOTOREAL_FAMILY_MINIMUMS.items():
        families = environment.get(kind)
        if not isinstance(families, Mapping) or set(families) != set(
            family_minimums
        ):
            raise SourceManifestMergeError(
                f"{label} environment.{kind} family tree is incomplete"
            )
        for family in family_minimums:
            if not isinstance(families.get(family), list):
                raise SourceManifestMergeError(
                    f"{label} environment.{kind}.{family} must be a list"
                )
    return environment


def _locked_sha(
    *,
    path: Path,
    expected: object,
    label: str,
    expected_size: object | None = None,
) -> str:
    digest = _required_text(expected, label=f"{label}.sha256").lower()
    if not SHA256_RE.fullmatch(digest) or _sha256(path) != digest:
        raise SourceManifestMergeError(f"{label} SHA-256 lock drifted")
    if expected_size is not None and (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 0
        or path.stat().st_size != expected_size
    ):
        raise SourceManifestMergeError(f"{label} size lock drifted")
    return digest


def _validate_entry(
    entry: object,
    *,
    family: str,
    label: str,
    manifest_parent: Path,
    volume: Path,
) -> _EntryLock:
    if not isinstance(entry, Mapping):
        raise SourceManifestMergeError(f"{label} must be a JSON object")
    asset_id = _required_text(
        entry.get("asset_id"),
        label=f"{label}.asset_id",
    )
    entry_family = _required_text(
        entry.get("family"),
        label=f"{label}.family",
    )
    if entry_family != family:
        raise SourceManifestMergeError(
            f"{label}.family must be exactly {family!r}"
        )
    identity = entry.get("identity")
    if not isinstance(identity, Mapping):
        raise SourceManifestMergeError(
            f"{label}.identity requires source_name and source_identity"
        )
    _required_text(
        identity.get("source_name"),
        label=f"{label}.identity.source_name",
    )
    _required_text(
        identity.get("source_identity"),
        label=f"{label}.identity.source_identity",
    )

    wrapper = _manifest_relative_file(
        volume=volume,
        manifest_parent=manifest_parent,
        raw=entry.get("path"),
        label=f"{label}.path",
    )
    if wrapper.suffix.casefold() not in USD_SUFFIXES:
        raise SourceManifestMergeError(f"{label}.path is not a USD file")
    wrapper_sha = _locked_sha(
        path=wrapper,
        expected=entry.get("sha256"),
        expected_size=entry.get("size_bytes")
        if "size_bytes" in entry
        else None,
        label=f"{label}.path",
    )

    records = entry.get("materialized_files")
    if not isinstance(records, list) or not records:
        raise SourceManifestMergeError(f"{label}.materialized_files is empty")
    dependency_paths: dict[Path, tuple[str, int]] = {}
    normalized_records: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise SourceManifestMergeError(
                f"{label}.materialized_files[{index}] is malformed"
            )
        dependency = _volume_relative_file(
            volume=volume,
            raw=record.get("path"),
            label=f"{label}.materialized_files[{index}].path",
        )
        if dependency in dependency_paths:
            raise SourceManifestMergeError(
                f"{label}.materialized_files repeats {dependency}"
            )
        expected_size = record.get("size_bytes")
        digest = _locked_sha(
            path=dependency,
            expected=record.get("sha256"),
            expected_size=expected_size,
            label=f"{label}.materialized_files[{index}]",
        )
        dependency_paths[dependency] = (digest, int(expected_size))
        normalized_records.append(dict(record))
    content_lock = hashlib.sha256(
        json.dumps(normalized_records, sort_keys=True).encode("utf-8")
    ).hexdigest()
    stored_content_lock = _required_text(
        entry.get("content_lock_sha256"),
        label=f"{label}.content_lock_sha256",
    ).lower()
    if stored_content_lock != content_lock:
        raise SourceManifestMergeError(
            f"{label}.content_lock_sha256 does not bind materialized_files"
        )

    source = _manifest_relative_file(
        volume=volume,
        manifest_parent=manifest_parent,
        raw=entry.get("source_cache_path"),
        label=f"{label}.source_cache_path",
    )
    if source.suffix.casefold() not in USD_SUFFIXES:
        raise SourceManifestMergeError(
            f"{label}.source_cache_path is not a USD file"
        )
    if source not in dependency_paths:
        raise SourceManifestMergeError(
            f"{label}.source_cache_path is absent from materialized_files"
        )

    lod = entry.get("lod")
    levels = lod.get("levels") if isinstance(lod, Mapping) else None
    strategy = (
        lod.get("strategy")
        if isinstance(lod, Mapping)
        and isinstance(lod.get("strategy"), str)
        else ""
    )
    valid_lod = (
        isinstance(lod, Mapping)
        and lod.get("state") == "passed"
        and isinstance(levels, list)
        and all(isinstance(level, str) and level.strip() for level in levels)
        and len(set(levels)) == len(levels)
        and isinstance(lod.get("level_count"), int)
        and not isinstance(lod.get("level_count"), bool)
        and lod.get("level_count") == len(levels)
        and (
            (strategy == "source_default_only" and len(levels) == 1)
            or (
                strategy in {"native_variant_set", "native_prim_hierarchy"}
                and len(levels) >= 3
            )
        )
    )
    if (
        entry.get("quality_validation") != "native_metadata_passed"
        or not isinstance(entry.get("anchor_validation"), Mapping)
        or entry["anchor_validation"].get("state") != "passed"
        or not isinstance(entry.get("materials"), Mapping)
        or entry["materials"].get("state") != "passed"
        or not valid_lod
    ):
        raise SourceManifestMergeError(
            f"{label} did not pass native metadata, anchor, material and LOD "
            "validation"
        )

    provenance = entry.get("provenance")
    if not isinstance(provenance, Mapping):
        raise SourceManifestMergeError(f"{label}.provenance is incomplete")
    _required_text(
        provenance.get("provider"),
        label=f"{label}.provenance.provider",
    )
    provenance_uri = provenance.get("source_uri")
    if provenance_uri in (None, ""):
        source_uri = _required_text(
            entry.get("source_uri"),
            label=f"{label}.source_uri",
        )
    else:
        source_uri = _required_text(
            provenance_uri,
            label=f"{label}.provenance.source_uri",
        )
    licence = entry.get("license")
    if isinstance(licence, Mapping):
        licence_id = _required_text(
            licence.get("id"),
            label=f"{label}.license.id",
        )
        licence_uri = _required_text(
            licence.get("uri"),
            label=f"{label}.license.uri",
        )
    else:
        licence_id = _required_text(
            entry.get("license_id"),
            label=f"{label}.license_id",
        )
        licence_uri = _required_text(
            entry.get("license_uri"),
            label=f"{label}.license_uri",
        )
    semantic_lock = _canonical_sha256(
        {
            "family": family,
            "identity": dict(identity),
            "provenance": dict(provenance),
            "license": {"id": licence_id, "uri": licence_uri},
            "source_uri": source_uri,
        }
    )
    return _EntryLock(
        asset_id=asset_id,
        wrapper_sha256=wrapper_sha,
        content_lock_sha256=content_lock,
        semantic_lock_sha256=semantic_lock,
        wrapper_path=wrapper,
        source_path=source,
    )


def _rebase_path(path: Path, *, destination_parent: Path) -> str:
    return os.path.relpath(path, destination_parent).replace("\\", "/")


def _rebase_entry(
    entry: Mapping[str, Any],
    *,
    source_parent: Path,
    destination_parent: Path,
    volume: Path,
    label: str,
) -> dict[str, Any]:
    rebased = copy.deepcopy(dict(entry))
    for field in ("path", "source_cache_path", "thumbnail_path"):
        raw = rebased.get(field)
        if not raw:
            continue
        path = _manifest_relative_file(
            volume=volume,
            manifest_parent=source_parent,
            raw=raw,
            label=f"{label}.{field}",
        )
        rebased[field] = _rebase_path(
            path,
            destination_parent=destination_parent,
        )
    lod_paths = rebased.get("lod_paths")
    if isinstance(lod_paths, Mapping):
        for level, record in lod_paths.items():
            if not isinstance(record, dict):
                raise SourceManifestMergeError(
                    f"{label}.lod_paths.{level} is malformed"
                )
            path = _manifest_relative_file(
                volume=volume,
                manifest_parent=source_parent,
                raw=record.get("path"),
                label=f"{label}.lod_paths.{level}.path",
            )
            record["path"] = _rebase_path(
                path,
                destination_parent=destination_parent,
            )
    return rebased


def _validate_curated_bundle_marker(
    *,
    volume: Path,
    curated_manifest: Path,
    curated_bundle_root: Path,
    expected_bundle_sha256: str,
) -> str:
    expected = _required_text(
        expected_bundle_sha256,
        label="curated bundle SHA-256",
    ).lower()
    if not SHA256_RE.fullmatch(expected):
        raise SourceManifestMergeError(
            "curated bundle SHA-256 must be exactly 64 lowercase hex digits"
        )
    root = _volume_path(
        volume=volume,
        path=curated_bundle_root,
        label="curated bundle root",
        require_file=False,
    )
    if not root.is_dir() or root == volume or not _inside(root, curated_manifest):
        raise SourceManifestMergeError(
            "curated manifest must stay below its bundle root"
        )
    marker_path = _volume_path(
        volume=volume,
        path=root / INSTALL_MARKER,
        label="curated bundle marker",
        require_file=True,
    )
    marker = _read_json_object(marker_path, label="curated bundle marker")
    relative = curated_manifest.relative_to(root).as_posix()
    if (
        marker.get("state") != "ASSET_BUNDLE_INSTALLED"
        or marker.get("bundle_sha256") != expected
        or marker.get("manifest_relative") != relative
        or marker.get("runtime_manifest_sha256") != _sha256(curated_manifest)
    ):
        raise SourceManifestMergeError(
            "curated bundle marker does not bind the configured archive and "
            "runtime manifest"
        )
    return _sha256(marker_path)


def _prepare_context(
    *,
    volume_root: Path,
    curated_manifest: Path,
    official_manifest: Path,
    output_manifest: Path,
    receipt_path: Path,
    curated_bundle_root: Path,
    curated_bundle_sha256: str,
) -> _MergeContext:
    volume = _absolute_without_resolving(volume_root)
    if not volume.is_dir() or volume.is_symlink():
        raise SourceManifestMergeError(f"volume root is absent: {volume}")
    curated = _volume_path(
        volume=volume,
        path=curated_manifest,
        label="curated manifest",
        require_file=True,
    )
    official = _volume_path(
        volume=volume,
        path=official_manifest,
        label="official manifest",
        require_file=True,
    )
    output = _volume_path(
        volume=volume,
        path=output_manifest,
        label="merged output manifest",
        require_file=False,
    )
    receipt = _volume_path(
        volume=volume,
        path=receipt_path,
        label="source merge receipt",
        require_file=False,
    )
    bundle_root = _volume_path(
        volume=volume,
        path=curated_bundle_root,
        label="curated bundle root",
        require_file=False,
    )
    if curated == official or output in {curated, official}:
        raise SourceManifestMergeError(
            "curated, official and merged manifests must be distinct files"
        )
    if output.parent != curated.parent:
        raise SourceManifestMergeError(
            "merged manifest must be written next to the curated manifest"
        )
    if output.is_symlink() or receipt.is_symlink():
        raise SourceManifestMergeError("merge output and receipt may not be symlinks")
    marker_sha = _validate_curated_bundle_marker(
        volume=volume,
        curated_manifest=curated,
        curated_bundle_root=bundle_root,
        expected_bundle_sha256=curated_bundle_sha256,
    )
    return _MergeContext(
        volume=volume,
        curated_manifest=curated,
        official_manifest=official,
        output_manifest=output,
        receipt_path=receipt,
        curated_bundle_root=bundle_root,
        curated_bundle_sha256=_required_text(
            curated_bundle_sha256,
            label="curated bundle SHA-256",
        ).lower(),
        curated_manifest_sha256=_sha256(curated),
        official_manifest_sha256=_sha256(official),
        curated_marker_sha256=marker_sha,
    )


def _register_asset(
    lock: _EntryLock,
    *,
    origin: str,
    role: str,
    seen_asset_ids: dict[str, tuple[str, str, str, str, str]],
    seen_wrapper_hashes: dict[str, tuple[str, str, str, str, str]],
    seen_content_semantics: dict[
        tuple[str, str], tuple[str, str, str, str]
    ],
) -> str | None:
    by_id = seen_asset_ids.get(lock.asset_id)
    if by_id is not None:
        if (
            by_id[0] != lock.wrapper_sha256
            or by_id[1] != lock.content_lock_sha256
            or by_id[2] != lock.semantic_lock_sha256
        ):
            raise SourceManifestMergeError(
                f"asset_id conflict for {lock.asset_id!r}: "
                f"{by_id[3]} {by_id[4]} versus {origin} {role}"
            )
        return "asset_id"
    by_hash = seen_wrapper_hashes.get(lock.wrapper_sha256)
    if by_hash is not None:
        if (
            by_hash[1] != lock.content_lock_sha256
            or by_hash[2] != lock.semantic_lock_sha256
        ):
            raise SourceManifestMergeError(
                f"wrapper SHA-256 conflict for {lock.wrapper_sha256}: "
                f"{by_hash[3]} {by_hash[4]} and {origin} {role} bind "
                "different dependency or semantic identity"
            )
        return "sha256"
    content_semantic_key = (
        lock.content_lock_sha256,
        lock.semantic_lock_sha256,
    )
    by_content_semantic = seen_content_semantics.get(content_semantic_key)
    if by_content_semantic is not None:
        return "content_semantic"
    record = (
        lock.wrapper_sha256,
        lock.content_lock_sha256,
        lock.semantic_lock_sha256,
        origin,
        role,
    )
    seen_asset_ids[lock.asset_id] = record
    seen_wrapper_hashes[lock.wrapper_sha256] = (
        lock.asset_id,
        lock.content_lock_sha256,
        lock.semantic_lock_sha256,
        origin,
        role,
    )
    seen_content_semantics[content_semantic_key] = (
        lock.asset_id,
        lock.wrapper_sha256,
        origin,
        role,
    )
    return None


def _build_base_manifest(
    context: _MergeContext,
) -> tuple[dict[str, Any], dict[str, Any]]:
    curated = _read_json_object(
        context.curated_manifest,
        label="curated manifest",
    )
    official = _read_json_object(
        context.official_manifest,
        label="official manifest",
    )
    _validate_manifest_header(
        curated,
        label="curated manifest",
        expected_missing_environment=(),
    )
    _validate_manifest_header(
        official,
        label="official manifest",
        expected_missing_environment=COMMUNITY_MISSING_ENVIRONMENT,
    )
    curated_environment = _environment_tree(curated, label="curated manifest")
    official_environment = _environment_tree(official, label="official manifest")
    curated_actors = curated.get("actors")
    actor_keys = set(curated_actors) if isinstance(curated_actors, Mapping) else set()
    if actor_keys != set(REQUIRED_ACTOR_CLASSES):
        missing = sorted(set(REQUIRED_ACTOR_CLASSES) - actor_keys)
        unexpected = sorted(actor_keys - set(REQUIRED_ACTOR_CLASSES))
        raise SourceManifestMergeError(
            "curated manifest requires the exact seven reviewed actor classes: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for family in COMMUNITY_BUILDING_FAMILIES:
        if official_environment["buildings"][family]:
            raise SourceManifestMergeError(
                "official manifest must leave all three reviewed community "
                f"families empty; buildings.{family} is populated"
            )

    output = copy.deepcopy(curated)
    output_environment: dict[str, dict[str, list[dict[str, Any]]]] = {
        kind: {family: [] for family in family_minimums}
        for kind, family_minimums in PHOTOREAL_FAMILY_MINIMUMS.items()
    }
    output["environment"] = output_environment
    output["actors"] = copy.deepcopy(dict(curated_actors))
    output["discovery"] = {
        "mode": "materialized_photoreal_asset_library_v3",
        "missing_environment": list(COMMUNITY_MISSING_ENVIRONMENT),
        "missing_actor_classes": [],
        "semantic_policy": (
            "exact_response_identity_only_no_generic_vehicle_promotion"
        ),
        "source_manifest_merge": {
            "schema_version": SOURCE_MERGE_SCHEMA_VERSION,
            "state": SOURCE_MERGE_STATE,
            "selection_policy": SELECTION_POLICY,
            "deduplicate_by": ["asset_id", "sha256"],
            "reserved_community_families": list(
                COMMUNITY_MISSING_ENVIRONMENT
            ),
            "curated_manifest": _relative_to_volume(
                context.curated_manifest,
                context.volume,
                label="curated manifest",
            ),
            "curated_manifest_sha256": context.curated_manifest_sha256,
            "curated_bundle_sha256": context.curated_bundle_sha256,
            "official_manifest": _relative_to_volume(
                context.official_manifest,
                context.volume,
                label="official manifest",
            ),
            "official_manifest_sha256": context.official_manifest_sha256,
            "required_actor_classes": list(REQUIRED_ACTOR_CLASSES),
            "required_kasa_source_basename": KASA_SOURCE_BASENAME,
        },
    }

    seen_ids: dict[str, tuple[str, str, str, str, str]] = {}
    seen_hashes: dict[str, tuple[str, str, str, str, str]] = {}
    seen_content_semantics: dict[
        tuple[str, str], tuple[str, str, str, str]
    ] = {}
    actor_selection: list[dict[str, str]] = []
    for class_id in REQUIRED_ACTOR_CLASSES:
        role = f"actors.{class_id}"
        entry = output["actors"][class_id]
        lock = _validate_entry(
            entry,
            family=role,
            label=role,
            manifest_parent=context.output_manifest.parent,
            volume=context.volume,
        )
        duplicate = _register_asset(
            lock,
            origin="curated",
            role=role,
            seen_asset_ids=seen_ids,
            seen_wrapper_hashes=seen_hashes,
            seen_content_semantics=seen_content_semantics,
        )
        if duplicate is not None:
            raise SourceManifestMergeError(
                f"curated actor {role} duplicates another selected asset by "
                f"{duplicate}"
            )
        actor_selection.append(
            {
                "class_id": class_id,
                "asset_id": lock.asset_id,
                "sha256": lock.wrapper_sha256,
                "content_lock_sha256": lock.content_lock_sha256,
                "semantic_lock_sha256": lock.semantic_lock_sha256,
                "origin": "curated",
            }
        )

    family_selection: dict[str, dict[str, Any]] = {}
    total_duplicate_asset_ids = 0
    total_duplicate_hashes = 0
    total_duplicate_content_semantics = 0
    kasa_selected = False
    for kind, family_minimums in PHOTOREAL_FAMILY_MINIMUMS.items():
        for family, minimum in family_minimums.items():
            role = f"{kind}.{family}"
            curated_entries = curated_environment[kind][family]
            if len(curated_entries) < minimum:
                raise SourceManifestMergeError(
                    f"curated manifest {role} requires at least {minimum} assets"
                )
            for index, entry in enumerate(curated_entries):
                _validate_entry(
                    entry,
                    family=role,
                    label=f"curated {role}[{index}]",
                    manifest_parent=context.curated_manifest.parent,
                    volume=context.volume,
                )

            official_entries = official_environment[kind][family]
            if role in COMMUNITY_MISSING_ENVIRONMENT:
                output_environment[kind][family] = []
                family_selection[role] = {
                    "minimum": minimum,
                    "selected": [],
                    "selected_count": 0,
                    "reserved_for_community": True,
                }
                continue
            if len(official_entries) < minimum:
                raise SourceManifestMergeError(
                    f"official manifest {role} requires at least {minimum} "
                    "qualified assets"
                )

            official_candidates: list[
                tuple[int, dict[str, Any], _EntryLock]
            ] = []
            for index, entry in enumerate(official_entries):
                rebased = _rebase_entry(
                    entry,
                    source_parent=context.official_manifest.parent,
                    destination_parent=context.output_manifest.parent,
                    volume=context.volume,
                    label=f"official {role}[{index}]",
                )
                lock = _validate_entry(
                    rebased,
                    family=role,
                    label=f"official {role}[{index}]",
                    manifest_parent=context.output_manifest.parent,
                    volume=context.volume,
                )
                official_candidates.append((index, rebased, lock))
            if role == "buildings.habitat":
                official_candidates.sort(
                    key=lambda item: (
                        item[2].source_path.name.casefold()
                        != KASA_SOURCE_BASENAME,
                        item[0],
                    )
                )

            curated_candidates: list[
                tuple[int, dict[str, Any], _EntryLock]
            ] = []
            for index, entry in enumerate(curated_entries):
                copied = copy.deepcopy(dict(entry))
                lock = _validate_entry(
                    copied,
                    family=role,
                    label=f"curated {role}[{index}]",
                    manifest_parent=context.output_manifest.parent,
                    volume=context.volume,
                )
                curated_candidates.append((index, copied, lock))

            selected: list[dict[str, Any]] = []
            selected_records: list[dict[str, str]] = []
            duplicate_asset_ids = 0
            duplicate_hashes = 0
            duplicate_content_semantics = 0
            for origin, candidates in (
                ("official", official_candidates),
                ("curated", curated_candidates),
            ):
                for _index, entry, lock in candidates:
                    if len(selected) >= minimum:
                        break
                    duplicate = _register_asset(
                        lock,
                        origin=origin,
                        role=role,
                        seen_asset_ids=seen_ids,
                        seen_wrapper_hashes=seen_hashes,
                        seen_content_semantics=seen_content_semantics,
                    )
                    if duplicate == "asset_id":
                        duplicate_asset_ids += 1
                        total_duplicate_asset_ids += 1
                        continue
                    if duplicate == "sha256":
                        duplicate_hashes += 1
                        total_duplicate_hashes += 1
                        continue
                    if duplicate == "content_semantic":
                        duplicate_content_semantics += 1
                        total_duplicate_content_semantics += 1
                        continue
                    selected.append(entry)
                    selected_records.append(
                        {
                            "asset_id": lock.asset_id,
                            "sha256": lock.wrapper_sha256,
                            "content_lock_sha256": lock.content_lock_sha256,
                            "semantic_lock_sha256": lock.semantic_lock_sha256,
                            "origin": origin,
                        }
                    )
                    if (
                        origin == "official"
                        and lock.source_path.name.casefold()
                        == KASA_SOURCE_BASENAME
                    ):
                        kasa_selected = True
                if len(selected) >= minimum:
                    break
            if len(selected) != minimum:
                raise SourceManifestMergeError(
                    f"{role} cannot meet its minimum after global asset_id/SHA "
                    f"deduplication: selected={len(selected)} minimum={minimum}"
                )
            output_environment[kind][family] = selected
            family_selection[role] = {
                "minimum": minimum,
                "selected": selected_records,
                "selected_count": len(selected),
                "reserved_for_community": False,
                "deduplicated_asset_ids": duplicate_asset_ids,
                "deduplicated_sha256": duplicate_hashes,
                "deduplicated_content_semantics": (
                    duplicate_content_semantics
                ),
            }

    if not kasa_selected:
        raise SourceManifestMergeError(
            "corrected official Kasa source is absent from the selected "
            f"environment: expected basename {KASA_SOURCE_BASENAME}"
        )
    selection = {
        "policy": SELECTION_POLICY,
        "reserved_community_families": list(COMMUNITY_MISSING_ENVIRONMENT),
        "families": family_selection,
        "actors": actor_selection,
        "actor_classes": list(REQUIRED_ACTOR_CLASSES),
        "kasa": {
            "source_basename": KASA_SOURCE_BASENAME,
            "selected": True,
        },
        "deduplication": {
            "asset_id": total_duplicate_asset_ids,
            "sha256": total_duplicate_hashes,
            "content_semantic": total_duplicate_content_semantics,
        },
    }
    return output, selection


def _expected_receipt(
    context: _MergeContext,
    *,
    output_payload: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    output_sha = hashlib.sha256(_json_bytes(output_payload)).hexdigest()
    contract = {
        "schema_version": SOURCE_MERGE_SCHEMA_VERSION,
        "state": SOURCE_MERGE_STATE,
        "selection_policy": SELECTION_POLICY,
        "curated_manifest_sha256": context.curated_manifest_sha256,
        "official_manifest_sha256": context.official_manifest_sha256,
        "curated_bundle_sha256": context.curated_bundle_sha256,
        "curated_marker_sha256": context.curated_marker_sha256,
        "output_manifest_sha256": output_sha,
        "selection_sha256": _canonical_sha256(selection),
        "required_actor_classes": list(REQUIRED_ACTOR_CLASSES),
        "reserved_community_families": list(COMMUNITY_MISSING_ENVIRONMENT),
        "required_kasa_source_basename": KASA_SOURCE_BASENAME,
    }
    receipt: dict[str, Any] = {
        "schema_version": SOURCE_MERGE_SCHEMA_VERSION,
        "state": SOURCE_MERGE_STATE,
        "inputs": {
            "curated": {
                "manifest": _relative_to_volume(
                    context.curated_manifest,
                    context.volume,
                    label="curated manifest",
                ),
                "manifest_sha256": context.curated_manifest_sha256,
                "bundle_root": _relative_to_volume(
                    context.curated_bundle_root,
                    context.volume,
                    label="curated bundle root",
                ),
                "bundle_sha256": context.curated_bundle_sha256,
                "marker_sha256": context.curated_marker_sha256,
            },
            "official": {
                "manifest": _relative_to_volume(
                    context.official_manifest,
                    context.volume,
                    label="official manifest",
                ),
                "manifest_sha256": context.official_manifest_sha256,
            },
        },
        "output": {
            "manifest": _relative_to_volume(
                context.output_manifest,
                context.volume,
                label="merged output manifest",
            ),
            "manifest_sha256": output_sha,
        },
        "selection": copy.deepcopy(dict(selection)),
        "source_contract_sha256": _canonical_sha256(contract),
        "proof_boundary": (
            "hash-locked source selection and provenance only; the three "
            "reserved community families require their exact reviewed "
            "augmentation before campaign assembly"
        ),
    }
    return receipt


def _require_exact_receipt(
    context: _MergeContext,
    *,
    expected: Mapping[str, Any],
) -> None:
    if not context.receipt_path.is_file() or context.receipt_path.is_symlink():
        raise SourceManifestMergeError("source merge receipt is absent")
    actual = _read_json_object(
        context.receipt_path,
        label="source merge receipt",
    )
    if actual != expected or context.receipt_path.read_bytes() != _json_bytes(expected):
        raise SourceManifestMergeError("source merge receipt drifted")


def _validate_community_augmentation(
    context: _MergeContext,
    *,
    base_payload: Mapping[str, Any],
    current_payload: Mapping[str, Any],
) -> dict[str, Any]:
    current_environment = _environment_tree(
        current_payload,
        label="community-augmented merged manifest",
    )
    current_actors = current_payload.get("actors")
    if not isinstance(current_actors, Mapping) or set(current_actors) != set(
        REQUIRED_ACTOR_CLASSES
    ):
        raise SourceManifestMergeError(
            "community augmentation changed the exact actor-class contract"
        )
    expected_uids = set(UID_TO_FAMILY)
    manifest_entries: dict[str, dict[str, Any]] = {}
    manifest_metadata_sha256: dict[str, str] = {}
    observed_uids: set[str] = set()
    for family in COMMUNITY_BUILDING_FAMILIES:
        expected_family_uids = {
            uid for uid, mapped_family in UID_TO_FAMILY.items()
            if mapped_family == family
        }
        entries = current_environment["buildings"][family]
        if len(entries) != len(expected_family_uids):
            raise SourceManifestMergeError(
                f"community augmentation buildings.{family} requires exactly "
                f"{len(expected_family_uids)} reviewed entries"
            )
        for index, entry in enumerate(entries):
            lock = _validate_entry(
                entry,
                family=f"buildings.{family}",
                label=f"community buildings.{family}[{index}]",
                manifest_parent=context.output_manifest.parent,
                volume=context.volume,
            )
            identity = entry.get("identity")
            uid = (
                _required_text(
                    identity.get("objaverse_uid"),
                    label=(
                        f"community buildings.{family}[{index}]"
                        ".identity.objaverse_uid"
                    ),
                )
                if isinstance(identity, Mapping)
                else ""
            )
            if (
                uid not in expected_family_uids
                or lock.asset_id != f"buildings.{family}:objaverse-{uid}"
                or uid in observed_uids
            ):
                raise SourceManifestMergeError(
                    f"community buildings.{family}[{index}] is not one of the "
                    "exact reviewed Objaverse assets"
                )
            source_lock = COMMUNITY_BUILDING_GLB_LOCKS[uid]
            provenance = entry.get("provenance")
            source_uri = _required_text(
                entry.get("source_uri"),
                label=f"community buildings.{family}[{index}].source_uri",
            )
            parsed_source = urlparse(source_uri)
            licence = entry.get("license")
            attribution = entry.get("attribution")
            metadata_sha = (
                _required_text(
                    provenance.get("raw_metadata_sha256"),
                    label=(
                        f"community buildings.{family}[{index}]"
                        ".provenance.raw_metadata_sha256"
                    ),
                ).lower()
                if isinstance(provenance, Mapping)
                else ""
            )
            attribution_creator = (
                _required_text(
                    attribution.get("creator"),
                    label=(
                        f"community buildings.{family}[{index}]"
                        ".attribution.creator"
                    ),
                )
                if isinstance(attribution, Mapping)
                else ""
            )
            attribution_notice = (
                _required_text(
                    attribution.get("notice"),
                    label=(
                        f"community buildings.{family}[{index}]"
                        ".attribution.notice"
                    ),
                )
                if isinstance(attribution, Mapping)
                else ""
            )
            if (
                entry.get("source_glb_sha256") != source_lock["sha256"]
                or entry.get("source_glb_size_bytes")
                != source_lock["size_bytes"]
                or not isinstance(provenance, Mapping)
                or provenance.get("source_glb_sha256")
                != source_lock["sha256"]
                or provenance.get("source_glb_size_bytes")
                != source_lock["size_bytes"]
                or provenance.get("provider") != "Objaverse"
                or provenance.get("source_uri") != source_uri
                or provenance.get("reviewed_selection_name")
                != COMMUNITY_BUILDING_TITLES[uid]
                or not SHA256_RE.fullmatch(metadata_sha)
                or parsed_source.scheme != "https"
                or (
                    parsed_source.hostname or ""
                ).casefold() not in {"sketchfab.com", "www.sketchfab.com"}
                or uid not in source_uri.casefold()
                or not isinstance(licence, Mapping)
                or licence.get("id") != "CC-BY-4.0"
                or licence.get("uri")
                != "https://creativecommons.org/licenses/by/4.0/"
                or not isinstance(attribution, Mapping)
                or not attribution_creator
                or not attribution_notice
                or attribution.get("source_uri") != source_uri
            ):
                raise SourceManifestMergeError(
                    f"community buildings.{family}[{index}] is not bound to "
                    f"the reviewed GLB/metadata/attribution lock for {uid}"
                )
            observed_uids.add(uid)
            manifest_metadata_sha256[uid] = metadata_sha
            manifest_entries[lock.asset_id] = copy.deepcopy(dict(entry))
    if observed_uids != expected_uids:
        raise SourceManifestMergeError(
            "community augmentation does not contain the exact eight reviewed "
            "Objaverse UIDs"
        )

    discovery = current_payload.get("discovery")
    supplement = (
        discovery.get("community_building_supplement")
        if isinstance(discovery, Mapping)
        else None
    )
    if (
        not isinstance(discovery, Mapping)
        or list(discovery.get("missing_environment") or [])
        or not isinstance(supplement, Mapping)
        or supplement.get("state") != COMMUNITY_INSTALL_STATE
        or supplement.get("asset_count") != len(expected_uids)
        or set(supplement.get("objaverse_uids") or []) != expected_uids
    ):
        raise SourceManifestMergeError(
            "community supplement provenance is absent or incomplete"
        )
    receipt_path = _manifest_relative_file(
        volume=context.volume,
        manifest_parent=context.output_manifest.parent,
        raw=supplement.get("receipt"),
        label="community supplement receipt",
    )
    community_root = context.output_manifest.parent / "community-building-assets"
    resolved_community_root = _absolute_without_resolving(community_root)
    if (
        not _inside(resolved_community_root, receipt_path)
        or receipt_path.name != "install-receipt.json"
        or supplement.get("bundle_id") != receipt_path.parent.name
        or supplement.get("receipt_sha256") != _sha256(receipt_path)
    ):
        raise SourceManifestMergeError(
            "community supplement receipt escaped or drifted"
        )
    receipt = _read_json_object(
        receipt_path,
        label="community supplement receipt",
    )
    receipt_entries = receipt.get("entries")
    source_locks = receipt.get("source_locks")
    if (
        receipt.get("schema_version") != COMMUNITY_INSTALL_SCHEMA_VERSION
        or receipt.get("state") != COMMUNITY_INSTALL_STATE
        or receipt.get("bundle_id") != receipt_path.parent.name
        or receipt.get("source_count") != len(expected_uids)
        or not isinstance(receipt_entries, list)
        or not isinstance(source_locks, list)
        or len(receipt_entries) != len(expected_uids)
        or len(source_locks) != len(expected_uids)
    ):
        raise SourceManifestMergeError(
            "community supplement receipt has an invalid contract"
        )
    receipt_by_asset_id: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(receipt_entries):
        if not isinstance(entry, Mapping):
            raise SourceManifestMergeError(
                f"community receipt entries[{index}] is malformed"
            )
        asset_id = _required_text(
            entry.get("asset_id"),
            label=f"community receipt entries[{index}].asset_id",
        )
        receipt_by_asset_id[asset_id] = entry
    if (
        set(receipt_by_asset_id) != set(manifest_entries)
        or any(
            receipt_by_asset_id[asset_id] != manifest_entry
            for asset_id, manifest_entry in manifest_entries.items()
        )
    ):
        raise SourceManifestMergeError(
            "community receipt entries differ from the merged manifest"
        )
    source_by_uid: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(source_locks):
        if not isinstance(record, Mapping):
            raise SourceManifestMergeError(
                f"community source_locks[{index}] is malformed"
            )
        uid = _required_text(
            record.get("uid"),
            label=f"community source_locks[{index}].uid",
        )
        source_by_uid[uid] = record
    if set(source_by_uid) != expected_uids:
        raise SourceManifestMergeError(
            "community receipt source locks are incomplete"
        )
    for uid, record in source_by_uid.items():
        reviewed = COMMUNITY_BUILDING_GLB_LOCKS[uid]
        if (
            record.get("family") != UID_TO_FAMILY[uid]
            or record.get("sha256") != reviewed["sha256"]
            or record.get("size_bytes") != reviewed["size_bytes"]
            or record.get("metadata_sha256")
            != manifest_metadata_sha256[uid]
        ):
            raise SourceManifestMergeError(
                "community source lock differs from the reviewed GLB or "
                f"manifest metadata for {uid}"
            )

    seen_ids: dict[str, tuple[str, str, str, str, str]] = {}
    seen_hashes: dict[str, tuple[str, str, str, str, str]] = {}
    seen_content_semantics: dict[
        tuple[str, str], tuple[str, str, str, str]
    ] = {}
    all_roles: list[tuple[str, object]] = [
        *(
            (f"actors.{class_id}", current_actors[class_id])
            for class_id in REQUIRED_ACTOR_CLASSES
        ),
        *(
            (f"{kind}.{family}[{index}]", entry)
            for kind, families in current_environment.items()
            for family, entries in families.items()
            for index, entry in enumerate(entries)
        ),
    ]
    for label, entry in all_roles:
        family = label.split("[", 1)[0]
        lock = _validate_entry(
            entry,
            family=family,
            label=f"final merged {label}",
            manifest_parent=context.output_manifest.parent,
            volume=context.volume,
        )
        duplicate = _register_asset(
            lock,
            origin="final",
            role=label,
            seen_asset_ids=seen_ids,
            seen_wrapper_hashes=seen_hashes,
            seen_content_semantics=seen_content_semantics,
        )
        if duplicate is not None:
            raise SourceManifestMergeError(
                f"final merged source manifest repeats an asset by {duplicate}: "
                f"{label}"
            )

    normalized = copy.deepcopy(dict(current_payload))
    normalized_environment = normalized["environment"]
    for family in COMMUNITY_BUILDING_FAMILIES:
        normalized_environment["buildings"][family] = []
    normalized_discovery = normalized["discovery"]
    normalized_discovery.pop("community_building_supplement", None)
    normalized_discovery["missing_environment"] = list(
        COMMUNITY_MISSING_ENVIRONMENT
    )
    if normalized != base_payload:
        raise SourceManifestMergeError(
            "merged source manifest contains mutations outside the exact "
            "community supplement"
        )
    return {
        "community_asset_count": len(expected_uids),
        "community_bundle_id": receipt_path.parent.name,
        "community_receipt_sha256": _sha256(receipt_path),
    }


def _build_expected(
    context: _MergeContext,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload, selection = _build_base_manifest(context)
    receipt = _expected_receipt(
        context,
        output_payload=payload,
        selection=selection,
    )
    return payload, selection, receipt


def _merge_lock_path(context: _MergeContext) -> Path:
    return context.receipt_path.with_name(
        f".{context.receipt_path.name}.lock"
    )


def _merge_locked(context: _MergeContext) -> dict[str, Any]:
    payload, selection, receipt = _build_expected(context)
    expected_output = _json_bytes(payload)
    expected_receipt = _json_bytes(receipt)
    expected_output_sha = hashlib.sha256(expected_output).hexdigest()
    output_exists = context.output_manifest.exists()
    receipt_exists = context.receipt_path.exists()
    if output_exists:
        if (
            not context.output_manifest.is_file()
            or context.output_manifest.is_symlink()
            or context.output_manifest.read_bytes() != expected_output
        ):
            raise SourceManifestMergeError(
                "existing merged source manifest drifted; refusing to overwrite"
            )
    if receipt_exists:
        _require_exact_receipt(context, expected=receipt)
    if (
        _sha256(context.curated_manifest) != context.curated_manifest_sha256
        or _sha256(context.official_manifest) != context.official_manifest_sha256
    ):
        raise SourceManifestMergeError(
            "source manifests changed while the merge was being prepared"
        )

    recovered = output_exists != receipt_exists
    if not output_exists:
        _atomic_write_json(context.output_manifest, payload)
    if not receipt_exists:
        _atomic_write_json(context.receipt_path, receipt)
    if (
        context.output_manifest.read_bytes() != expected_output
        or context.receipt_path.read_bytes() != expected_receipt
    ):
        raise SourceManifestMergeError("source merge atomic write verification failed")
    return {
        "schema_version": SOURCE_MERGE_SCHEMA_VERSION,
        "state": SOURCE_MERGE_STATE,
        "manifest": str(context.output_manifest),
        "manifest_sha256": expected_output_sha,
        "receipt": str(context.receipt_path),
        "receipt_sha256": _sha256(context.receipt_path),
        "reused": output_exists and receipt_exists,
        "recovered_interrupted_write": recovered,
        "community_augmented": False,
        "selection_sha256": _canonical_sha256(selection),
    }


def merge_source_manifests(
    *,
    volume_root: Path,
    curated_manifest: Path,
    official_manifest: Path,
    output_manifest: Path,
    receipt_path: Path,
    curated_bundle_root: Path,
    curated_bundle_sha256: str,
) -> dict[str, Any]:
    """Create or exactly reuse the locked pre-community source merge."""

    context = _prepare_context(
        volume_root=volume_root,
        curated_manifest=curated_manifest,
        official_manifest=official_manifest,
        output_manifest=output_manifest,
        receipt_path=receipt_path,
        curated_bundle_root=curated_bundle_root,
        curated_bundle_sha256=curated_bundle_sha256,
    )
    with _exclusive_merge_lock(_merge_lock_path(context)):
        return _merge_locked(context)


def verify_source_manifest_merge(
    *,
    volume_root: Path,
    curated_manifest: Path,
    official_manifest: Path,
    output_manifest: Path,
    receipt_path: Path,
    curated_bundle_root: Path,
    curated_bundle_sha256: str,
    require_community: bool = False,
) -> dict[str, Any]:
    """Verify the base merge, or require its exact community augmentation."""

    context = _prepare_context(
        volume_root=volume_root,
        curated_manifest=curated_manifest,
        official_manifest=official_manifest,
        output_manifest=output_manifest,
        receipt_path=receipt_path,
        curated_bundle_root=curated_bundle_root,
        curated_bundle_sha256=curated_bundle_sha256,
    )
    with _exclusive_merge_lock(_merge_lock_path(context)):
        payload, selection, receipt = _build_expected(context)
        _require_exact_receipt(context, expected=receipt)
        if (
            not context.output_manifest.is_file()
            or context.output_manifest.is_symlink()
        ):
            raise SourceManifestMergeError("merged source manifest is absent")
        expected_bytes = _json_bytes(payload)
        current_bytes = context.output_manifest.read_bytes()
        community: dict[str, Any] | None = None
        if current_bytes == expected_bytes:
            if require_community:
                raise SourceManifestMergeError(
                    "the three reviewed community families are not installed"
                )
        else:
            current = _read_json_object(
                context.output_manifest,
                label="merged source manifest",
            )
            community = _validate_community_augmentation(
                context,
                base_payload=payload,
                current_payload=current,
            )
            if not require_community:
                raise SourceManifestMergeError(
                    "base source merge was already augmented; use the explicit "
                    "community verification gate"
                )
        return {
            "schema_version": SOURCE_MERGE_SCHEMA_VERSION,
            "state": (
                "SOURCE_MANIFESTS_MERGED_WITH_COMMUNITY"
                if community is not None
                else SOURCE_MERGE_STATE
            ),
            "manifest": str(context.output_manifest),
            "manifest_sha256": _sha256(context.output_manifest),
            "base_manifest_sha256": hashlib.sha256(expected_bytes).hexdigest(),
            "receipt": str(context.receipt_path),
            "receipt_sha256": _sha256(context.receipt_path),
            "reused": True,
            "community_augmented": community is not None,
            "selection_sha256": _canonical_sha256(selection),
            **(community or {}),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the immutable curated actors with the corrected NVIDIA "
            "environment before the reviewed community building stage."
        )
    )
    parser.add_argument("--volume-root", type=Path, required=True)
    parser.add_argument("--curated-manifest", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--curated-bundle-root", type=Path, required=True)
    parser.add_argument("--curated-bundle-sha256", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--require-community",
        action="store_true",
        help=(
            "require and verify the exact eight-asset community augmentation; "
            "valid only with --verify-only"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.require_community and not args.verify_only:
        raise SystemExit("--require-community requires --verify-only")
    kwargs = {
        "volume_root": args.volume_root,
        "curated_manifest": args.curated_manifest,
        "official_manifest": args.official_manifest,
        "output_manifest": args.output_manifest,
        "receipt_path": args.receipt,
        "curated_bundle_root": args.curated_bundle_root,
        "curated_bundle_sha256": args.curated_bundle_sha256,
    }
    try:
        if args.verify_only:
            result = verify_source_manifest_merge(
                **kwargs,
                require_community=args.require_community,
            )
        else:
            result = merge_source_manifests(**kwargs)
    except SourceManifestMergeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

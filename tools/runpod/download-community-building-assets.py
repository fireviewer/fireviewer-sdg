#!/usr/bin/env python3
"""Materialize the eight reviewed Objaverse GLBs for the RunPod Kit installer.

The downloader intentionally owns only source acquisition and attribution
validation.  Geometry conversion and manifest installation remain in
``fireviewer_sdg.community_building_assets`` and must run with Isaac Sim's
Python/Kit runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import struct
import sys
import uuid
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlparse, urlunparse

from fireviewer_sdg.community_building_assets import (
    COMMUNITY_BUILDING_GLB_LOCKS,
    COMMUNITY_BUILDING_TITLES,
    UID_TO_FAMILY,
)


OBJAVERSE_DISTRIBUTION = "objaverse"
OBJAVERSE_VERSION = "0.1.7"
DOWNLOAD_STATE = "OBJAVERSE_COMMUNITY_BUILDINGS_DOWNLOADED"
LOCKED_UIDS = tuple(sorted(UID_TO_FAMILY))
EXPECTED_GLB_LOCKS: Mapping[str, Mapping[str, object]] = (
    COMMUNITY_BUILDING_GLB_LOCKS
)


class CommunityBuildingDownloadError(RuntimeError):
    """Raised when a reviewed source cannot be acquired without substitution."""


class ObjaverseClient(Protocol):
    """Subset of the pinned Objaverse API used by this downloader."""

    def load_annotations(self, uids: list[str]) -> Mapping[str, object]: ...

    def load_objects(
        self,
        uids: list[str],
        download_processes: int = 1,
    ) -> Mapping[str, str]: ...


def _inside(root: Path, path: Path) -> bool:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _require_local_root(*, volume_root: Path, path: Path, label: str) -> Path:
    if volume_root.is_symlink():
        raise CommunityBuildingDownloadError(
            f"volume root may not be a symlink: {volume_root}"
        )
    if path.is_symlink():
        raise CommunityBuildingDownloadError(
            f"{label} may not be a symlink: {path}"
        )
    volume = volume_root.resolve()
    candidate = path.resolve()
    if not volume.is_dir():
        raise CommunityBuildingDownloadError(
            f"volume root must be a real directory: {volume}"
        )
    if not _inside(volume, candidate):
        raise CommunityBuildingDownloadError(
            f"{label} must remain inside the production volume: {candidate}"
        )
    if candidate.exists() and not candidate.is_dir():
        raise CommunityBuildingDownloadError(
            f"{label} must be a real directory: {candidate}"
        )
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_glb(path: Path, *, uid: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise CommunityBuildingDownloadError(f"{uid} GLB is absent: {path}")
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise CommunityBuildingDownloadError(
                f"{uid} GLB header is truncated"
            )
        magic, version, declared_size = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2 or declared_size != size:
            raise CommunityBuildingDownloadError(
                f"{uid} is not a complete binary glTF 2.0 asset"
            )
        remaining = size - 12
        chunks = 0
        first_chunk_type: int | None = None
        while remaining:
            chunk_header = stream.read(8)
            if len(chunk_header) != 8:
                raise CommunityBuildingDownloadError(
                    f"{uid} GLB chunk header is truncated"
                )
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            if chunk_length % 4 or chunk_length > remaining - 8:
                raise CommunityBuildingDownloadError(
                    f"{uid} GLB chunk length is invalid"
                )
            if chunks == 0:
                first_chunk_type = chunk_type
                encoded_document = stream.read(chunk_length)
                try:
                    document = json.loads(
                        encoded_document.rstrip(b" \t\r\n\x00").decode("utf-8")
                    )
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise CommunityBuildingDownloadError(
                        f"{uid} GLB JSON chunk is invalid"
                    ) from exc
                if (
                    not isinstance(document, Mapping)
                    or not isinstance(document.get("asset"), Mapping)
                    or str(document["asset"].get("version")) != "2.0"
                ):
                    raise CommunityBuildingDownloadError(
                        f"{uid} GLB JSON does not declare glTF 2.0"
                    )
            else:
                stream.seek(chunk_length, os.SEEK_CUR)
            remaining -= 8 + chunk_length
            chunks += 1
        if chunks == 0 or first_chunk_type != 0x4E4F534A:
            raise CommunityBuildingDownloadError(
                f"{uid} GLB has no leading JSON chunk"
            )
    return {
        "sha256": _sha256(path),
        "size_bytes": size,
        "glb_version": 2,
    }


def _locked_glb(path: Path, *, uid: str) -> dict[str, object]:
    lock = EXPECTED_GLB_LOCKS.get(uid)
    if not isinstance(lock, Mapping):
        raise CommunityBuildingDownloadError(
            f"{uid} has no reviewed GLB content lock"
        )
    expected_sha256 = str(lock.get("sha256", "")).strip().lower()
    expected_size = lock.get("size_bytes")
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size <= 0
    ):
        raise CommunityBuildingDownloadError(
            f"{uid} has an invalid reviewed GLB content lock"
        )
    actual = _validate_glb(path, uid=uid)
    if (
        actual["sha256"] != expected_sha256
        or actual["size_bytes"] != expected_size
    ):
        raise CommunityBuildingDownloadError(
            f"{uid} GLB differs from its reviewed SHA-256/size lock"
        )
    return actual


def _text(value: object) -> str:
    return str(value or "").strip()


def _first_text(record: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_https(value: object, *, label: str) -> str:
    text = _text(value)
    parsed = urlparse(text)
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
        or parsed.port not in {None, 80, 443}
    ):
        raise CommunityBuildingDownloadError(
            f"{label} must be a credential-free HTTP(S) URL"
        )
    port = parsed.port
    netloc = parsed.hostname.casefold()
    if port not in {None, 80, 443}:
        netloc = f"{netloc}:{port}"
    return urlunparse(
        ("https", netloc, parsed.path, parsed.params, parsed.query, "")
    )


def _creator(record: Mapping[str, Any], *, uid: str) -> str:
    creator = _first_text(record, "creator", "author", "artist")
    if creator:
        return creator
    for key in ("user", "creator", "author", "attribution"):
        nested = record.get(key)
        if not isinstance(nested, Mapping):
            continue
        creator = _first_text(
            nested,
            "displayName",
            "display_name",
            "username",
            "name",
            "creator",
            "author",
        )
        if creator:
            return creator
    raise CommunityBuildingDownloadError(
        f"{uid} Objaverse annotation has no creator"
    )


def _license(record: Mapping[str, Any], *, uid: str) -> tuple[str, str]:
    if record.get("license") != "by":
        raise CommunityBuildingDownloadError(
            f"{uid} must retain the exact Objaverse CC-BY token 'by'"
        )
    return (
        "CC-BY-4.0",
        "https://creativecommons.org/licenses/by/4.0/",
    )


def _downloadable(record: Mapping[str, Any], *, uid: str) -> None:
    candidates = (
        record.get("isDownloadable"),
        record.get("is_downloadable"),
        record.get("downloadable"),
    )
    if not any(value is True for value in candidates):
        raise CommunityBuildingDownloadError(
            f"{uid} Objaverse annotation is not explicitly downloadable"
        )


def _source_uri(record: Mapping[str, Any], *, uid: str) -> str:
    value = _first_text(
        record,
        "viewerUrl",
        "viewer_url",
        "source_uri",
        "uri",
        "url",
    )
    return _normalize_https(value, label=f"{uid} source URI")


def _provider_archive_size(record: Mapping[str, Any], *, uid: str) -> int:
    archives = record.get("archives")
    glb = archives.get("glb") if isinstance(archives, Mapping) else None
    size = glb.get("size") if isinstance(glb, Mapping) else None
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or size > 2_000_000_000
    ):
        raise CommunityBuildingDownloadError(
            f"{uid} Objaverse annotation has no bounded GLB archive size"
        )
    return size


def _annotation_records(payload: object) -> dict[str, Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise CommunityBuildingDownloadError(
            "Objaverse annotations must be keyed by UID"
        )
    records: dict[str, Mapping[str, Any]] = {}
    for uid in LOCKED_UIDS:
        raw = payload.get(uid)
        if not isinstance(raw, Mapping):
            raise CommunityBuildingDownloadError(
                f"Objaverse annotation is absent for locked UID {uid}"
            )
        records[uid] = raw
    return records


def _validated_attribution(
    annotations: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for uid in LOCKED_UIDS:
        annotation = annotations[uid]
        _downloadable(annotation, uid=uid)
        license_id, license_uri = _license(annotation, uid=uid)
        provider_title = _first_text(annotation, "name", "title")
        if not provider_title:
            raise CommunityBuildingDownloadError(
                f"{uid} Objaverse annotation has no title"
            )
        records[uid] = {
            "uid": uid,
            "family": UID_TO_FAMILY[uid],
            "name": COMMUNITY_BUILDING_TITLES[uid],
            "title": provider_title,
            "creator": _creator(annotation, uid=uid),
            "source_uri": _source_uri(annotation, uid=uid),
            "license_id": license_id,
            "license_uri": license_uri,
            "local_path": f"{uid}.glb",
            "downloadable": True,
            "provider_archive_size_bytes": _provider_archive_size(
                annotation,
                uid=uid,
            ),
        }
    return records


def _call_with_cache(
    function: object,
    *,
    cache_root: Path,
    kwargs: dict[str, object],
) -> object:
    if not callable(function):
        raise CommunityBuildingDownloadError(
            "pinned Objaverse client is missing a required callable"
        )
    signature = inspect.signature(function)
    if "download_dir" in signature.parameters:
        kwargs["download_dir"] = str(cache_root)
    return function(**kwargs)


def _configure_objaverse_cache(client: object, cache_root: Path) -> None:
    # Objaverse 0.1.7 exposes the legacy globals below.  Setting both keeps
    # annotations and GLBs on the pod NVMe instead of the root home directory.
    if hasattr(client, "BASE_PATH"):
        setattr(client, "BASE_PATH", str(cache_root))
    if hasattr(client, "_VERSIONED_PATH"):
        setattr(client, "_VERSIONED_PATH", str(cache_root / "hf-objaverse-v1"))


def _copy_glb_atomic(*, source: Path, destination: Path, uid: str) -> None:
    _locked_glb(source, uid=uid)
    temporary = destination.with_name(
        f".{destination.name}.partial-{uuid.uuid4().hex}"
    )
    try:
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as input_stream, temporary.open("xb") as output:
            while block := input_stream.read(4 * 1024 * 1024):
                output.write(block)
                digest.update(block)
                size += len(block)
            output.flush()
            os.fsync(output.fileno())
        staged = _locked_glb(temporary, uid=uid)
        if staged["sha256"] != digest.hexdigest() or staged["size_bytes"] != size:
            raise CommunityBuildingDownloadError(
                f"{uid} changed while its GLB was being staged"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: object) -> bool:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if path.is_file() and not path.is_symlink() and path.read_bytes() == encoded:
        return False
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def download_community_building_assets(
    *,
    client: ObjaverseClient,
    volume_root: Path,
    destination_root: Path,
    cache_root: Path,
    workers: int = 4,
) -> dict[str, object]:
    """Validate annotations and atomically materialize only the locked GLBs."""

    if not 1 <= workers <= 8:
        raise CommunityBuildingDownloadError(
            "Objaverse download workers must be between 1 and 8"
        )
    destination = _require_local_root(
        volume_root=volume_root,
        path=destination_root,
        label="Objaverse destination root",
    )
    cache = _require_local_root(
        volume_root=volume_root,
        path=cache_root,
        label="Objaverse cache root",
    )
    _configure_objaverse_cache(client, cache)
    raw_annotations = _call_with_cache(
        client.load_annotations,
        cache_root=cache,
        kwargs={"uids": list(LOCKED_UIDS)},
    )
    annotations = _annotation_records(raw_annotations)
    attribution = _validated_attribution(annotations)

    if set(EXPECTED_GLB_LOCKS) != set(LOCKED_UIDS):
        raise CommunityBuildingDownloadError(
            "reviewed GLB content locks must match the exact eight locked UIDs"
        )
    expected_hashes = [
        str(EXPECTED_GLB_LOCKS[uid].get("sha256", "")).strip().lower()
        for uid in LOCKED_UIDS
    ]
    if len(set(expected_hashes)) != len(expected_hashes):
        raise CommunityBuildingDownloadError(
            "reviewed Objaverse GLB locks contain duplicate source content"
        )

    metadata_path = destination / "metadata.json"
    if metadata_path.is_symlink() or (
        metadata_path.exists() and not metadata_path.is_file()
    ):
        raise CommunityBuildingDownloadError(
            "Objaverse metadata target must be a regular file"
        )

    missing: list[str] = []
    locks: dict[str, dict[str, object]] = {}
    for uid in LOCKED_UIDS:
        target = destination / f"{uid}.glb"
        try:
            locks[uid] = _locked_glb(target, uid=uid)
        except CommunityBuildingDownloadError:
            missing.append(uid)

    stale_metadata: Path | None = None
    if missing and metadata_path.is_file():
        stale_metadata = destination / (
            f".metadata.json.stale-{uuid.uuid4().hex}"
        )
        os.replace(metadata_path, stale_metadata)
    if missing:
        try:
            downloaded = _call_with_cache(
                client.load_objects,
                cache_root=cache,
                kwargs={
                    "uids": list(missing),
                    "download_processes": workers,
                },
            )
            if not isinstance(downloaded, Mapping):
                raise CommunityBuildingDownloadError(
                    "Objaverse object download result must be keyed by UID"
                )
            staging = destination / f".objaverse-stage-{uuid.uuid4().hex}"
            staging.mkdir(mode=0o750)
            try:
                for uid in missing:
                    raw_path = downloaded.get(uid)
                    if not isinstance(raw_path, (str, os.PathLike)):
                        raise CommunityBuildingDownloadError(
                            f"Objaverse did not return locked UID {uid}"
                        )
                    source = Path(raw_path).expanduser().resolve()
                    staged_target = staging / f"{uid}.glb"
                    _copy_glb_atomic(
                        source=source,
                        destination=staged_target,
                        uid=uid,
                    )
                    locks[uid] = _locked_glb(staged_target, uid=uid)

                for uid in missing:
                    target = destination / f"{uid}.glb"
                    os.replace(staging / f"{uid}.glb", target)
                    locks[uid] = _locked_glb(target, uid=uid)
            finally:
                for candidate in staging.iterdir():
                    if candidate.is_file() and not candidate.is_symlink():
                        candidate.unlink()
                staging.rmdir()
        except BaseException:
            if stale_metadata is not None and stale_metadata.is_file():
                stale_metadata.unlink()
            raise

    hashes = [str(locks[uid]["sha256"]) for uid in LOCKED_UIDS]
    if len(set(hashes)) != len(hashes):
        raise CommunityBuildingDownloadError(
            "reviewed Objaverse GLBs contain duplicate source content"
        )

    records: dict[str, dict[str, object]] = {}
    for uid in LOCKED_UIDS:
        records[uid] = {
            **attribution[uid],
            **locks[uid],
        }
    try:
        metadata_changed = _atomic_write_json(
            metadata_path,
            {"assets": records},
        )
    finally:
        if stale_metadata is not None and stale_metadata.is_file():
            stale_metadata.unlink()
    return {
        "state": DOWNLOAD_STATE,
        "asset_count": len(records),
        "uids": list(LOCKED_UIDS),
        "destination_root": str(destination),
        "metadata": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "downloaded_count": len(missing),
        "metadata_changed": metadata_changed,
    }


def _load_pinned_objaverse() -> ObjaverseClient:
    try:
        installed = importlib_metadata.version(OBJAVERSE_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError as exc:
        raise CommunityBuildingDownloadError(
            f"{OBJAVERSE_DISTRIBUTION}=={OBJAVERSE_VERSION} is not installed"
        ) from exc
    if installed != OBJAVERSE_VERSION:
        raise CommunityBuildingDownloadError(
            f"expected {OBJAVERSE_DISTRIBUTION}=={OBJAVERSE_VERSION}, "
            f"found {installed}"
        )
    try:
        import objaverse
    except ImportError as exc:
        raise CommunityBuildingDownloadError(
            "the pinned Objaverse client cannot be imported"
        ) from exc
    return objaverse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the eight reviewed CC-BY Objaverse building GLBs "
            "without selecting substitutes."
        )
    )
    parser.add_argument("--volume-root", required=True, type=Path)
    parser.add_argument("--destination-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = download_community_building_assets(
            client=_load_pinned_objaverse(),
            volume_root=args.volume_root,
            destination_root=args.destination_root,
            cache_root=args.cache_root,
            workers=args.workers,
        )
    except (CommunityBuildingDownloadError, OSError, ValueError) as exc:
        print(
            f"OBJAVERSE_COMMUNITY_BUILDINGS_FAILED: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

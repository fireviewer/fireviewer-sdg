"""Audited runtime provisioning for models, scenes and assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(frozen=True)
class ArtifactFile:
    relative_path: PurePosixPath
    url: str
    sha256: str
    size_bytes: int
    auth_env: str | None


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    kind: str
    revision: str
    destination: PurePosixPath
    files: tuple[ArtifactFile, ...]


@dataclass(frozen=True)
class ProvisionResult:
    downloaded: tuple[str, ...]
    cache_hits: tuple[str, ...]


def _safe_relative(value: str, *, field: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{field} must be a non-empty relative path")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_url(url: str, allowed_hosts: frozenset[str]) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("artifact URLs must use HTTPS")
    hostname = parsed.hostname.lower()
    if hostname not in allowed_hosts:
        raise ValueError(f"artifact host is not allowlisted: {hostname}")
    if parsed.username or parsed.password:
        raise ValueError("artifact URLs must not embed credentials")


def load_manifest(path: Path, allowed_hosts: frozenset[str]) -> tuple[Artifact, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported provision manifest schema_version")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ValueError("provision manifest artifacts must be a list")
    artifacts: list[Artifact] = []
    identifiers: set[str] = set()
    for raw_artifact in raw_artifacts:
        if not isinstance(raw_artifact, dict):
            raise ValueError("artifact entries must be objects")
        artifact_id = str(raw_artifact.get("artifact_id", "")).strip()
        if not artifact_id or artifact_id in identifiers:
            raise ValueError("artifact_id must be non-empty and unique")
        identifiers.add(artifact_id)
        revision = str(raw_artifact.get("revision", "")).strip()
        if not revision:
            raise ValueError(f"artifact {artifact_id} has no immutable revision")
        destination = _safe_relative(
            str(raw_artifact.get("destination", "")), field="destination"
        )
        raw_files = raw_artifact.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError(f"artifact {artifact_id} has no files")
        files: list[ArtifactFile] = []
        for raw_file in raw_files:
            if not isinstance(raw_file, dict):
                raise ValueError(f"artifact {artifact_id} file entries must be objects")
            url = str(raw_file.get("url", "")).strip()
            _validate_url(url, allowed_hosts)
            expected_hash = str(raw_file.get("sha256", "")).strip().lower()
            if len(expected_hash) != 64 or any(
                character not in "0123456789abcdef" for character in expected_hash
            ):
                raise ValueError(f"artifact {artifact_id} has an invalid SHA-256")
            size_bytes = int(raw_file.get("size_bytes", 0))
            if size_bytes <= 0:
                raise ValueError(f"artifact {artifact_id} has an invalid size")
            auth_env = str(raw_file.get("auth_env", "")).strip() or None
            files.append(
                ArtifactFile(
                    relative_path=_safe_relative(
                        str(raw_file.get("relative_path", "")),
                        field="relative_path",
                    ),
                    url=url,
                    sha256=expected_hash,
                    size_bytes=size_bytes,
                    auth_env=auth_env,
                )
            )
        artifacts.append(
            Artifact(
                artifact_id=artifact_id,
                kind=str(raw_artifact.get("kind", "asset")).strip() or "asset",
                revision=revision,
                destination=destination,
                files=tuple(files),
            )
        )
    return tuple(artifacts)


def _is_valid(path: Path, expected: ArtifactFile) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected.size_bytes
        and _sha256_file(path) == expected.sha256
    )


def _download(expected: ArtifactFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    headers = {"User-Agent": "FireViewer-SDG/0.1"}
    if expected.auth_env:
        token = os.getenv(expected.auth_env, "").strip()
        if not token:
            raise RuntimeError(f"missing credential environment: {expected.auth_env}")
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(expected.url, headers=headers)
    try:
        with urllib.request.urlopen(
            request, context=ssl.create_default_context(), timeout=120
        ) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
    except (OSError, urllib.error.URLError):
        partial.unlink(missing_ok=True)
        raise
    if not _is_valid(partial, expected):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded artifact failed validation: {expected.relative_path}")
    partial.replace(destination)


def provision(
    *, manifest_path: Path, volume_root: Path, allowed_hosts: frozenset[str]
) -> ProvisionResult:
    artifacts = load_manifest(manifest_path, allowed_hosts)
    downloaded: list[str] = []
    cache_hits: list[str] = []
    for artifact in artifacts:
        artifact_root = volume_root.joinpath(*artifact.destination.parts).resolve()
        if volume_root not in artifact_root.parents:
            raise ValueError("artifact destination escapes the volume root")
        for expected in artifact.files:
            destination = artifact_root.joinpath(*expected.relative_path.parts).resolve()
            if artifact_root != destination.parent and artifact_root not in destination.parents:
                raise ValueError("artifact file escapes its destination")
            label = f"{artifact.artifact_id}/{expected.relative_path.as_posix()}"
            if _is_valid(destination, expected):
                cache_hits.append(label)
                continue
            destination.unlink(missing_ok=True)
            _download(expected, destination)
            downloaded.append(label)
    receipt_path = volume_root / "provision" / "receipts" / "latest.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "manifest_sha256": _sha256_file(manifest_path),
        "downloaded": downloaded,
        "cache_hits": cache_hits,
    }
    temporary = receipt_path.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt_path)
    return ProvisionResult(tuple(downloaded), tuple(cache_hits))

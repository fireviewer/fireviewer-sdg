#!/usr/bin/env python3
"""Acquire reviewed-license actor source candidates without admitting them.

The output of this command is deliberately a candidate inventory, not a
FireViewer campaign asset manifest.  Native conversion, LOD generation,
rendered inspection and class-specific approval remain mandatory before any
entry can be promoted into ``campaign-assets/manifest-v3.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
STATE = "ACTOR_SOURCE_CANDIDATES_ACQUIRED"
INVENTORY_NAME = "candidate-source-inventory.json"

# These locks are the Objaverse v1 annotations observed on the production pod.
# They prevent a title/license/geometry downgrade from being silently accepted.
CANDIDATES: dict[str, dict[str, Any]] = {
    "canadair": {
        "uid": "85a01bcfe0834228bd5261ad5754ecca",
        "name": "Canadair CL-215",
        "creator": "AlessioPassera",
        "license": "by",
        "glb": {
            "size": 185_470_464,
            "faceCount": 55_006,
            "textureCount": 7,
            "textureMaxResolution": 8_192,
        },
        "materialized_glb": {
            "size": 7_730_332,
            "sha256": "9d172d46eec0e0438e815a879695e616c5cfcf719f0e8a5cf2dcda0f8c550810",
        },
    },
    "dash": {
        "uid": "140a2d7680504e9b98ec4def5d4a7bdf",
        "name": "Dash-8 q400 (with cockpit)ver IX)",
        "creator": "TonyWony",
        "license": "by",
        "glb": {
            "size": 181_191_624,
            "faceCount": 68_275,
            "textureCount": 23,
            "textureMaxResolution": 4_096,
        },
        "materialized_glb": {
            "size": 21_072_752,
            "sha256": "2360411d2e9bdd738a73dd1da08b180c25de8f5a17605141b69b989e610f9e2e",
        },
    },
    "hard_negative_construction_truck": {
        "uid": "fc2b5eb692ca40c2b44357b62eb149df",
        "name": "Truck",
        "creator": "MM",
        "license": "by",
        "glb": {
            "size": 55_635_832,
            "faceCount": 999_999,
            "textureCount": 1,
            "textureMaxResolution": 8_192,
        },
        "materialized_glb": {
            "size": 35_469_072,
            "sha256": "3c14ca9796a93b01e8df8dfd036adaa173274fd8b14430c37136fc284626bb61",
        },
    },
    "hard_negative_crop_duster": {
        "uid": "eb47ca6191b7438d88aaa1e44184ed1b",
        "name": "Crop Duster",
        "creator": "halflife1and2",
        "license": "by",
        "glb": {
            "size": 7_756_788,
            "faceCount": 10_565,
            "textureCount": 2,
            "textureMaxResolution": 2_048,
        },
        "materialized_glb": {
            "size": 3_428_920,
            "sha256": "9a8e431645128b5f1e8c5099a7c7c08daacef9417fe6295b3279fa1729d60715",
        },
    },
    "hard_negative_utility_helicopter": {
        "uid": "90bf99ba20f948a9838623311bd83da3",
        "name": "Bell UH-1N Twin Huey",
        "creator": "helijah",
        "license": "by",
        "glb": {
            "size": 7_592_324,
            "faceCount": 111_359,
            "textureCount": 21,
            "textureMaxResolution": 2_048,
        },
        "materialized_glb": {
            "size": 5_953_560,
            "sha256": "26b6207a69fd8319e944f04f9e8d7bfa1e2c873de1d079c1d15a4e6250570b57",
        },
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def _safe_root(volume_root: Path, path: Path, *, label: str) -> Path:
    volume = volume_root.resolve()
    candidate = path.resolve()
    if (
        not volume.is_dir()
        or volume.is_symlink()
        or candidate == volume
        or not _inside(volume, candidate)
        or path.is_symlink()
    ):
        raise RuntimeError(f"{label} must be a non-symlink child of the volume")
    return candidate


def _atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                (
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _glb_header(path: Path) -> dict[str, int]:
    with path.open("rb") as stream:
        header = stream.read(12)
    if len(header) != 12 or header[:4] != b"glTF":
        raise RuntimeError(f"Objaverse object is not a binary glTF: {path}")
    version = int.from_bytes(header[4:8], "little")
    declared_length = int.from_bytes(header[8:12], "little")
    if version != 2 or declared_length != path.stat().st_size:
        raise RuntimeError(
            f"GLB header/length is invalid: {path} "
            f"(version={version}, declared={declared_length})"
        )
    return {"version": version, "declared_length": declared_length}


def _configure_objaverse_cache(objaverse: Any, cache_root: Path) -> None:
    objaverse.BASE_PATH = str(cache_root)
    objaverse._VERSIONED_PATH = str(cache_root / "hf-objaverse-v1")


def _validate_annotation(
    *, role: str, expected: dict[str, Any], annotation: object
) -> dict[str, Any]:
    if not isinstance(annotation, dict):
        raise RuntimeError(f"Objaverse annotation is absent for {role}")
    user = annotation.get("user")
    archives = annotation.get("archives")
    glb = archives.get("glb") if isinstance(archives, dict) else None
    locked_glb = expected["glb"]
    observed_glb = (
        {
            "size": glb.get("size"),
            "faceCount": glb.get("faceCount"),
            "textureCount": glb.get("textureCount"),
            "textureMaxResolution": glb.get("textureMaxResolution"),
        }
        if isinstance(glb, dict)
        else None
    )
    if (
        annotation.get("uid") != expected["uid"]
        or annotation.get("name") != expected["name"]
        or not isinstance(user, dict)
        or user.get("displayName") != expected["creator"]
        or annotation.get("license") != expected["license"]
        or annotation.get("isDownloadable") is not True
        or observed_glb != locked_glb
    ):
        raise RuntimeError(
            f"Objaverse annotation lock drifted for {role}: "
            f"name={annotation.get('name')!r}, license={annotation.get('license')!r}, "
            f"downloadable={annotation.get('isDownloadable')!r}, glb={observed_glb!r}"
        )
    return {
        "uid": expected["uid"],
        "name": expected["name"],
        "creator": expected["creator"],
        "license": {
            "id": "CC-BY-4.0",
            "uri": "https://creativecommons.org/licenses/by/4.0/",
        },
        "source_uri": annotation.get("viewerUrl"),
        "objaverse_glb": observed_glb,
        "description": annotation.get("description") or "",
    }


def acquire(
    *,
    volume_root: Path,
    destination_root: Path,
    cache_root: Path,
    workers: int,
) -> dict[str, Any]:
    if workers < 1 or workers > 8:
        raise RuntimeError("Objaverse workers must be between 1 and 8")
    volume = volume_root.resolve()
    destination = _safe_root(
        volume_root,
        destination_root,
        label="actor candidate destination",
    )
    cache = _safe_root(
        volume_root,
        cache_root,
        label="Objaverse cache",
    )
    destination.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    import objaverse

    _configure_objaverse_cache(objaverse, cache)
    uids = [record["uid"] for record in CANDIDATES.values()]
    available = set(objaverse.load_uids())
    missing = sorted(set(uids) - available)
    if missing:
        raise RuntimeError(
            "locked actor candidate UIDs are absent from Objaverse v1: "
            + ", ".join(missing)
        )
    annotations = objaverse.load_annotations(uids)
    validated = {
        role: _validate_annotation(
            role=role,
            expected=expected,
            annotation=annotations.get(expected["uid"]),
        )
        for role, expected in CANDIDATES.items()
    }
    downloaded = objaverse.load_objects(
        uids=uids,
        download_processes=workers,
    )
    if set(downloaded) != set(uids):
        raise RuntimeError("Objaverse did not return every locked candidate UID")

    entries: dict[str, dict[str, Any]] = {}
    for role, expected in CANDIDATES.items():
        uid = expected["uid"]
        source = Path(downloaded[uid]).resolve()
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"Objaverse source is not a regular file: {source}")
        materialized_lock = expected["materialized_glb"]
        if source.stat().st_size != materialized_lock["size"]:
            raise RuntimeError(f"materialized Objaverse GLB size drifted for {role}")
        source_header = _glb_header(source)
        source_sha256 = _sha256(source)
        if source_sha256 != materialized_lock["sha256"]:
            raise RuntimeError(f"materialized Objaverse GLB SHA-256 drifted for {role}")
        target = destination / f"{role}--{uid}.glb"
        reused = target.is_file() and not target.is_symlink()
        if reused:
            if (
                target.stat().st_size != source.stat().st_size
                or _sha256(target) != source_sha256
            ):
                raise RuntimeError(
                    f"existing actor candidate drifted; refusing overwrite: {target}"
                )
        else:
            if target.exists():
                raise RuntimeError(
                    f"actor candidate target is not a regular file: {target}"
                )
            temporary = target.with_name(
                f".{target.name}.partial-{uuid.uuid4().hex}"
            )
            try:
                with source.open("rb") as input_stream, temporary.open("xb") as output:
                    shutil.copyfileobj(input_stream, output, 4 * 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                if (
                    temporary.stat().st_size != source.stat().st_size
                    or _sha256(temporary) != source_sha256
                ):
                    raise RuntimeError(f"actor candidate copy failed for {role}")
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        entries[role] = {
            **validated[role],
            "candidate_state": "acquired_unreviewed",
            "campaign_admitted": False,
            "path": target.relative_to(volume).as_posix(),
            "sha256": source_sha256,
            "size_bytes": target.stat().st_size,
            "glb_header": source_header,
            "reused": reused,
            "required_next_gates": [
                "native_usd_conversion",
                "dependency_and_material_validation",
                "metric_scale_and_anchor_validation",
                "distinct_hero_mid_far_lod_generation",
                "rtx_rendered_visual_review",
                "class_identity_approval",
            ],
        }

    inventory = {
        "schema_version": SCHEMA_VERSION,
        "state": STATE,
        "proof_boundary": (
            "source bytes, Objaverse metadata and CC BY attribution only; "
            "no candidate is admitted into the FireViewer campaign"
        ),
        "candidate_count": len(entries),
        "entries": entries,
    }
    inventory_path = destination / INVENTORY_NAME
    if inventory_path.exists():
        try:
            current = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("existing actor candidate inventory is invalid") from exc
        current_entries = current.get("entries") if isinstance(current, dict) else None
        if not isinstance(current_entries, dict):
            raise RuntimeError("existing actor candidate inventory is incomplete")
        for role, entry in entries.items():
            current_entry = current_entries.get(role)
            if (
                not isinstance(current_entry, dict)
                or current_entry.get("sha256") != entry["sha256"]
                or current_entry.get("size_bytes") != entry["size_bytes"]
            ):
                raise RuntimeError(
                    f"existing actor candidate inventory drifted for {role}"
                )
    _atomic_write_json(inventory_path, inventory)
    inventory["inventory"] = inventory_path.relative_to(volume).as_posix()
    inventory["inventory_sha256"] = _sha256(inventory_path)
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire the locked Objaverse actor source candidates"
    )
    parser.add_argument("--volume-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = acquire(
        volume_root=args.volume_root,
        destination_root=args.destination_root,
        cache_root=args.cache_root,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

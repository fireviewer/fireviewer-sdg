#!/usr/bin/env python3
"""Download the pinned Hunyuan3D shape, delight and paint model set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def already_verified(destination: Path, manifest: dict, manifest_sha256: str) -> bool:
    report = destination / "asset4sim-model-verification.json"
    if not report.is_file():
        return False
    try:
        previous = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if previous.get("manifest_sha256") != manifest_sha256:
        return False
    expected_files = manifest["critical_files"]
    recorded = {entry.get("path"): entry for entry in previous.get("verified", [])}
    for expected in expected_files:
        path = destination / expected["path"]
        entry = recorded.get(expected["path"])
        if (
            entry is None
            or entry.get("bytes") != expected["bytes"]
            or entry.get("sha256") != expected["sha256"]
            or not path.is_file()
            or path.stat().st_size != expected["bytes"]
        ):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if already_verified(destination, manifest, manifest_sha256):
        print(json.dumps({"verified_count": len(manifest["critical_files"]), "destination": str(destination), "reused": True}))
        return 0

    texture = manifest["repositories"]["texture"]
    snapshot_download(
        repo_id=texture["repo_id"],
        revision=texture["revision"],
        allow_patterns=texture["allow_patterns"],
        local_dir=destination / "diffusers",
    )
    shape = manifest["repositories"]["shape"]
    shape_dir = destination / "diffusion_models" / "hy3dgen"
    shape_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=shape["repo_id"],
        revision=shape["revision"],
        filename=shape["filename"],
        local_dir=shape_dir,
    )

    verified = []
    for expected in manifest["critical_files"]:
        path = destination / expected["path"]
        if not path.is_file():
            raise RuntimeError(f"Missing model file: {path}")
        size = path.stat().st_size
        digest = sha256(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise RuntimeError(f"Model integrity mismatch: {path}")
        verified.append({"path": expected["path"], "bytes": size, "sha256": digest})
    report = destination / "asset4sim-model-verification.json"
    report.write_text(
        json.dumps({"manifest_sha256": manifest_sha256, "verified": verified}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"verified_count": len(verified), "destination": str(destination), "reused": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

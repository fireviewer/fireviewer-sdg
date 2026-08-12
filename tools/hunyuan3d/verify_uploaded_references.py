#!/usr/bin/env python3
"""Verify uploaded reference files against the local SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("comfy_input", type=Path)
    parser.add_argument("--route", help="Only verify manifest entries assigned to this route")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assets = manifest["assets"]
    if args.route:
        assets = [asset for asset in assets if asset.get("route") == args.route]
    failures = []
    verified_bytes = 0
    for asset in assets:
        path = args.comfy_input / asset["input_name"]
        if not path.is_file():
            failures.append(f"missing:{asset['asset_id']}")
            continue
        if path.stat().st_size != asset["source_bytes"]:
            failures.append(f"size:{asset['asset_id']}")
            continue
        if digest(path) != asset["source_sha256"]:
            failures.append(f"sha256:{asset['asset_id']}")
            continue
        verified_bytes += path.stat().st_size
    print(f"verified_assets={len(assets) - len(failures)}")
    print(f"verified_bytes={verified_bytes}")
    if failures:
        print("failures=" + ",".join(failures[:20]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

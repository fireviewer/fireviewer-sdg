#!/usr/bin/env python3
"""Freeze complete canonical-initial GLB pairs into a post-process batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def single_latest_glb(directory: Path) -> Path | None:
    files = sorted(directory.glob("*.glb"), key=lambda path: (path.stat().st_mtime_ns, path.name))
    return files[-1].resolve() if files else None


def completed_pairs(
    state_path: Path,
    untextured_root: Path,
    textured_root: Path,
    allowed_asset_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state_assets = state.get("assets", {})
    discovered_asset_ids = (
        set(state_assets)
        | {path.name for path in untextured_root.iterdir() if path.is_dir()}
        | {path.name for path in textured_root.iterdir() if path.is_dir()}
    )
    asset_ids = sorted(
        discovered_asset_ids
        if allowed_asset_ids is None
        else discovered_asset_ids & allowed_asset_ids
    )
    pairs: list[dict[str, Any]] = []
    for asset_id in asset_ids:
        record = state_assets.get(asset_id, {})
        untextured = Path(record["untextured_50k"]).resolve() if record.get("untextured_50k") else None
        textured = Path(record["textured_50k"]).resolve() if record.get("textured_50k") else None
        if untextured is None or not untextured.is_file():
            untextured = single_latest_glb(untextured_root / asset_id)
        if textured is None or not textured.is_file():
            textured = single_latest_glb(textured_root / asset_id)
        if untextured is None or textured is None:
            continue
        pairs.append(
            {
                "asset_id": asset_id,
                "state_status": record.get("status", "reconciled_from_disk"),
                "untextured_50k": str(untextured),
                "textured_50k": str(textured),
                "untextured_bytes": untextured.stat().st_size,
                "textured_bytes": textured.stat().st_size,
                "untextured_sha256": sha256(untextured),
                "textured_sha256": sha256(textured),
            }
        )
    return pairs


def freeze(args: argparse.Namespace) -> int:
    reference = None
    allowed_asset_ids = None
    if args.reference_manifest:
        reference = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
        allowed_asset_ids = {item["asset_id"] for item in reference["assets"]}

    pairs = completed_pairs(
        args.state,
        args.untextured_root,
        args.textured_root,
        allowed_asset_ids,
    )
    if not pairs:
        raise RuntimeError("No complete canonical-initial GLB pairs were found")

    args.stage_dir.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        source = Path(pair["untextured_50k"])
        staged = args.stage_dir / pair["asset_id"] / source.name
        staged.parent.mkdir(parents=True, exist_ok=True)
        if staged.exists() or staged.is_symlink():
            if staged.resolve() != source:
                raise RuntimeError(f"Frozen path already points elsewhere: {staged}")
        else:
            os.symlink(source, staged)
        pair["staged_untextured"] = str(staged.resolve())

    manifest = {
        "schema_version": 1,
        "contract": "complete canonical-initial pairs only",
        "asset_count": len(pairs),
        "assets": pairs,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    subset_path = None
    if args.subset_reference_manifest and reference is None:
        raise ValueError("--subset-reference-manifest requires --reference-manifest")
    if args.subset_reference_manifest:
        assert reference is not None
        selected_ids = {pair["asset_id"] for pair in pairs}
        selected_assets = [item for item in reference["assets"] if item["asset_id"] in selected_ids]
        missing = selected_ids - {item["asset_id"] for item in selected_assets}
        if missing:
            raise RuntimeError(f"Frozen assets missing from reference manifest: {sorted(missing)}")
        subset = dict(reference)
        subset["assets"] = selected_assets
        subset["asset_count"] = len(selected_assets)
        subset["route_counts"] = {"hunyuan3d": len(selected_assets)}
        args.subset_reference_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.subset_reference_manifest.write_text(
            json.dumps(subset, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        subset_path = str(args.subset_reference_manifest)

    print(
        json.dumps(
            {"asset_count": len(pairs), "manifest": str(args.manifest), "subset_reference_manifest": subset_path},
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--untextured-root", type=Path, required=True)
    parser.add_argument("--textured-root", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--subset-reference-manifest", type=Path)
    return parser


def main() -> int:
    return freeze(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

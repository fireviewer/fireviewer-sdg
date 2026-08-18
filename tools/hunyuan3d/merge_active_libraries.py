#!/usr/bin/env python3
"""Merge validated Asset4Sim libraries into one final folder without modifying assets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        return "copy"


def active_record_paths(root: Path, index: int, asset_id: str) -> dict[str, Path]:
    asset_root = root / "assets" / f"{index:03d}_{asset_id}"
    glb = asset_root / f"{asset_id}.glb"
    usd = asset_root / f"{asset_id}.usd"
    texture = asset_root / "textures" / f"{asset_id}.png"
    required = {"glb": glb, "usd": usd, "texture": texture}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required active asset files: " + ", ".join(missing))
    return {"root": asset_root, **required}


def final_path(final_root: Path, stage_root: Path, path: Path) -> str:
    return str((final_root / path.relative_to(stage_root)).resolve())


def run(args: argparse.Namespace) -> int:
    final_root = args.output.resolve()
    stage = final_root.with_name(f".{final_root.name}.building")
    if final_root.exists():
        raise FileExistsError(f"final output already exists: {final_root}")
    if stage.exists():
        raise FileExistsError(f"staging output already exists: {stage}")

    active: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    storage = {"hardlink": 0, "copy": 0}
    sources: list[dict[str, Any]] = []
    stage.mkdir(parents=True)

    for source_root_arg in args.source:
        source_root = source_root_arg.resolve()
        active_manifest_path = source_root / "active-assets.json"
        rejected_manifest_path = source_root / "rejected-assets.json"
        active_manifest = json.loads(active_manifest_path.read_text(encoding="utf-8"))
        rejected_manifest = json.loads(rejected_manifest_path.read_text(encoding="utf-8"))
        source_active = list(active_manifest.get("assets", []))
        source_rejected = list(rejected_manifest.get("assets", []))
        if int(active_manifest.get("asset_count", -1)) != len(source_active):
            raise RuntimeError(f"active manifest count mismatch: {active_manifest_path}")
        if int(rejected_manifest.get("asset_count", -1)) != len(source_rejected):
            raise RuntimeError(f"rejected manifest count mismatch: {rejected_manifest_path}")
        sources.append({
            "root": str(source_root),
            "active_count": len(source_active),
            "rejected_count": len(source_rejected),
        })

        for record in source_active:
            index = int(record["index"])
            asset_id = str(record["asset_id"])
            if index in seen_indices:
                raise RuntimeError(f"duplicate active index across libraries: {index}")
            seen_indices.add(index)
            source_paths = active_record_paths(source_root, index, asset_id)
            destination_root = stage / "assets" / f"{index:03d}_{asset_id}"
            for source_file in sorted(path for path in source_paths["root"].rglob("*") if path.is_file()):
                if source_file.name == "asset-report.json":
                    continue
                destination = destination_root / source_file.relative_to(source_paths["root"])
                method = link_or_copy(source_file, destination)
                storage[method] += 1

            destination_glb = destination_root / f"{asset_id}.glb"
            destination_usd = destination_root / f"{asset_id}.usd"
            destination_texture = destination_root / "textures" / f"{asset_id}.png"
            final_record = {
                "schema_version": 1,
                "index": index,
                "asset_id": asset_id,
                "status": "retained",
                "source_library": str(source_root),
                "source_asset_report": str((source_paths["root"] / "asset-report.json").resolve()),
                "glb": final_path(final_root, stage, destination_glb),
                "usd": final_path(final_root, stage, destination_usd),
                "texture": final_path(final_root, stage, destination_texture),
                "scale_policy": active_manifest.get("scale_policy"),
                "color_policy": active_manifest.get("color_policy"),
                "passed": True,
            }
            write_json(destination_root / "asset-report.json", final_record)
            active.append(final_record)

        for record in source_rejected:
            index = int(record["index"])
            if index in seen_indices:
                raise RuntimeError(f"rejected index is also active: {index}")
            rejected.append({**record, "source_library": str(source_root)})

    active.sort(key=lambda item: int(item["index"]))
    rejected.sort(key=lambda item: int(item["index"]))
    active_indices = {int(item["index"]) for item in active}
    rejected_indices = {int(item["index"]) for item in rejected}
    if active_indices & rejected_indices:
        raise RuntimeError("active and rejected index sets overlap")
    expected = set(range(args.start, args.end + 1))
    if active_indices | rejected_indices != expected:
        raise RuntimeError(
            f"final index coverage mismatch: missing={sorted(expected-active_indices-rejected_indices)}, "
            f"extra={sorted((active_indices|rejected_indices)-expected)}"
        )

    active_manifest = {
        "schema_version": 1,
        "status": "final_merged",
        "range": {"start": args.start, "end": args.end},
        "asset_count": len(active),
        "rejected_count": len(rejected),
        "operation": "merge_only_no_asset_modification",
        "sources": sources,
        "assets": active,
    }
    rejected_manifest = {
        "schema_version": 1,
        "status": "excluded_from_final_library",
        "asset_count": len(rejected),
        "archive_policy": "Excluded assets remain only in their immutable source archives for provenance.",
        "assets": rejected,
    }
    validation = {
        "schema_version": 1,
        "status": "passed",
        "operation": "merge_only_no_asset_modification",
        "range_count": len(expected),
        "active_count": len(active),
        "rejected_count": len(rejected),
        "active_directory_count": len(list((stage / "assets").glob("*"))),
        "glb_count": len(list((stage / "assets").glob("*/*.glb"))),
        "usd_count": len(list((stage / "assets").glob("*/*.usd"))),
        "texture_count": len(list((stage / "assets").glob("*/textures/*.png"))),
        "active_indices": sorted(active_indices),
        "rejected_indices": sorted(rejected_indices),
        "storage_methods": storage,
    }
    for key in ("active_directory_count", "glb_count", "usd_count", "texture_count"):
        if validation[key] != len(active):
            raise RuntimeError(f"final file count mismatch: {key}={validation[key]}, expected={len(active)}")

    write_json(stage / "active-assets.json", active_manifest)
    write_json(stage / "rejected-assets.json", rejected_manifest)
    write_json(stage / "merge-validation.json", validation)
    rejected_text = ", ".join(f"{index:03d}" for index in sorted(rejected_indices))
    (stage / "README.md").write_text(
        "# Asset4Sim - bibliotheque finale 001-294\n\n"
        f"- {len(active)} assets retenus regroupes dans `assets/`.\n"
        f"- {len(rejected)} assets exclus : {rejected_text}.\n"
        "- Aucun GLB, USD ou fichier de texture n'a ete modifie durant ce regroupement.\n"
        "- Les echelles et corrections sont exactement celles des deux bibliotheques sources validees.\n"
        "- Les fichiers volumineux sont relies par hardlink lorsque le systeme de fichiers le permet.\n",
        encoding="utf-8",
    )
    os.replace(stage, final_root)
    print(json.dumps({
        "output": str(final_root),
        "active_count": len(active),
        "rejected_count": len(rejected),
        "rejected_indices": sorted(rejected_indices),
        "storage_methods": storage,
        "passed": True,
    }, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=294)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

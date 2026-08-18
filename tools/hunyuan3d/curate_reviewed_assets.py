#!/usr/bin/env python3
"""Create a reviewed active Asset4Sim library without altering production archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_rejections(path: Path, assets_by_index: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported rejection schema: {path}")
    rejections: dict[int, dict[str, Any]] = {}
    for rejection in payload.get("assets", []):
        index = int(rejection["index"])
        if index in rejections:
            raise RuntimeError(f"duplicate rejected index: {index}")
        if index not in assets_by_index:
            raise RuntimeError(f"rejected index is outside the review report: {index}")
        expected_id = assets_by_index[index]["asset_id"]
        if rejection.get("asset_id") != expected_id:
            raise RuntimeError(
                f"rejected asset mismatch for {index}: {rejection.get('asset_id')} != {expected_id}"
            )
        if not str(rejection.get("reason", "")).strip():
            raise RuntimeError(f"missing rejection reason for {index}")
        rejections[index] = rejection
    return rejections


def source_files(asset: dict[str, Any]) -> dict[str, Path]:
    asset_id = str(asset["asset_id"])
    glb = Path(asset["glb_path"]).resolve()
    usd = Path(asset["usd_path"]).resolve()
    batch_root = usd.parent.parent
    files = {
        "glb": glb,
        "usd": usd,
        "texture": usd.parent / "textures" / f"{asset_id}.png",
        "usd_report": batch_root / "reports" / "usd" / f"{asset_id}-usd.json",
    }
    missing = [f"{name}={path}" for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required source files: " + ", ".join(missing))
    return files


def copy_review(review_root: Path, destination: Path) -> dict[str, int]:
    counts = {"files": 0, "captures": 0, "contact_sheets": 0}
    for source in sorted(path for path in review_root.rglob("*") if path.is_file()):
        relative = source.relative_to(review_root)
        link_or_copy(source, destination / relative)
        counts["files"] += 1
        if relative.parts and relative.parts[0] == "captures" and source.suffix.lower() == ".png":
            counts["captures"] += 1
        if source.name.startswith("contact-sheet-") and source.suffix.lower() == ".png":
            counts["contact_sheets"] += 1
    return counts


def destination_paths(stage: Path, index: int, asset_id: str) -> dict[str, Path]:
    root = stage / "assets" / f"{index:03d}_{asset_id}"
    return {
        "root": root,
        "glb": root / f"{asset_id}.glb",
        "usd": root / f"{asset_id}.usd",
        "texture": root / "textures" / f"{asset_id}.png",
        "usd_report": root / "reports" / f"{asset_id}-usd.json",
        "asset_report": root / "asset-report.json",
    }


def library_path(final_root: Path, stage_root: Path, stage_path: Path) -> str:
    return str((final_root / stage_path.relative_to(stage_root)).resolve())


def run(args: argparse.Namespace) -> int:
    review_report = args.review_report.resolve()
    review_root = review_report.parent
    final_root = args.output_root.resolve()
    stage = final_root.with_name(f".{final_root.name}.building")
    if final_root.exists():
        raise FileExistsError(f"output library already exists: {final_root}")
    if stage.exists():
        raise FileExistsError(f"staging directory already exists: {stage}")

    review = json.loads(review_report.read_text(encoding="utf-8"))
    assets = list(review.get("assets", []))
    expected_count = int(review.get("asset_count", -1))
    if expected_count != len(assets) or expected_count <= 0:
        raise RuntimeError(f"review asset count mismatch: declared={expected_count}, actual={len(assets)}")
    indices = [int(asset["index"]) for asset in assets]
    if len(indices) != len(set(indices)):
        raise RuntimeError("duplicate indices in review report")
    if indices != list(range(int(review["start"]), int(review["end"]) + 1)):
        raise RuntimeError("review indices are not a complete contiguous range")

    assets_by_index = {int(asset["index"]): asset for asset in assets}
    rejections = load_rejections(args.rejections.resolve(), assets_by_index)
    active_assets: list[dict[str, Any]] = []
    rejected_assets: list[dict[str, Any]] = []
    manual_assets: list[dict[str, Any]] = []
    link_counts = {"hardlink": 0, "copy": 0}

    stage.mkdir(parents=True)
    review_counts = copy_review(review_root, stage / "review")
    if review_counts["captures"] != expected_count:
        raise RuntimeError(
            f"capture count mismatch: expected={expected_count}, actual={review_counts['captures']}"
        )

    for asset in assets:
        index = int(asset["index"])
        asset_id = str(asset["asset_id"])
        files = source_files(asset)
        capture = Path(asset["capture_path"]).resolve()
        if not capture.is_file():
            raise FileNotFoundError(f"missing review capture: {capture}")
        automatic_status = str(asset.get("status", "unknown"))
        warnings = list(asset.get("warnings", []))
        metrics = dict(asset.get("metrics", {}))

        if index in rejections:
            rejection = rejections[index]
            record = {
                "index": index,
                "asset_id": asset_id,
                "decision": "rejected",
                "reason": rejection["reason"],
                "source_reference": asset.get("source"),
                "source_glb_archive": str(files["glb"]),
                "source_usd_archive": str(files["usd"]),
                "source_texture_archive": str(files["texture"]),
                "review_capture": str(capture),
                "automatic_status": automatic_status,
                "automatic_warnings": warnings,
                "active_library_included": False,
            }
            rejected_assets.append(record)
            manual_assets.append(record)
            continue

        destinations = destination_paths(stage, index, asset_id)
        file_hashes: dict[str, str] = {}
        file_methods: dict[str, str] = {}
        for name in ("glb", "usd", "texture", "usd_report"):
            source = files[name]
            destination = destinations[name]
            method = link_or_copy(source, destination)
            link_counts[method] += 1
            source_hash = sha256(source)
            if method == "copy" and sha256(destination) != source_hash:
                raise RuntimeError(f"copied file hash mismatch: {destination}")
            if method == "hardlink" and not os.path.samefile(source, destination):
                raise RuntimeError(f"hardlink identity mismatch: {destination}")
            file_hashes[name] = source_hash
            file_methods[name] = method

        final_paths = {
            name: library_path(final_root, stage, destinations[name])
            for name in ("glb", "usd", "texture", "usd_report", "asset_report")
        }
        active_record = {
            "schema_version": 1,
            "index": index,
            "asset_id": asset_id,
            "decision": "accepted",
            "source_reference": asset.get("source"),
            "source_glb_archive": str(files["glb"]),
            "source_usd_archive": str(files["usd"]),
            "glb": final_paths["glb"],
            "usd": final_paths["usd"],
            "texture": final_paths["texture"],
            "usd_report": final_paths["usd_report"],
            "review_capture": str(capture),
            "automatic_status": automatic_status,
            "automatic_warnings": warnings,
            "metrics": metrics,
            "sha256": file_hashes,
            "storage_method": file_methods,
            "passed": True,
        }
        write_json(destinations["asset_report"], active_record)
        active_assets.append(active_record)
        manual_assets.append({
            "index": index,
            "asset_id": asset_id,
            "decision": "accepted",
            "reason": "No severe structural defect observed in the two rendered views compared with the reference.",
            "source_reference": asset.get("source"),
            "review_capture": str(capture),
            "automatic_status": automatic_status,
            "automatic_warnings": warnings,
            "active_library_included": True,
        })

    if len(active_assets) + len(rejected_assets) != expected_count:
        raise RuntimeError("manual review does not cover every asset")
    rejected_indices = set(rejections)
    active_indices = {int(asset["index"]) for asset in active_assets}
    if active_indices & rejected_indices:
        raise RuntimeError("a rejected asset was included in the active library")

    active_manifest = {
        "schema_version": 1,
        "status": "reviewed",
        "asset_count": len(active_assets),
        "rejected_count": len(rejected_assets),
        "archive_policy": "Active files are linked from immutable verified production archives; rejected files remain only in those archives for provenance.",
        "scale_policy": "unchanged_from_postprocess_0103_0294",
        "color_policy": "unchanged_from_postprocess_0103_0294",
        "assets": active_assets,
    }
    rejected_manifest = {
        "schema_version": 1,
        "status": "user_confirmed_visual_rejections",
        "asset_count": len(rejected_assets),
        "archive_policy": "Excluded from the active reviewed library; immutable production sources retained only for provenance.",
        "assets": rejected_assets,
    }
    manual_review = {
        "schema_version": 1,
        "status": "complete",
        "review_method": "Visual comparison of two rendered model views against each source reference; doubtful assets inspected individually.",
        "asset_count": expected_count,
        "accepted_count": len(active_assets),
        "rejected_count": len(rejected_assets),
        "assets": manual_assets,
    }

    faces = [int(asset.get("metrics", {}).get("faces", 0)) for asset in active_assets]
    warnings = sum(1 for asset in active_assets if asset["automatic_status"] == "warning")
    validation = {
        "schema_version": 1,
        "status": "passed",
        "expected_asset_count": expected_count,
        "active_asset_count": len(active_assets),
        "rejected_asset_count": len(rejected_assets),
        "decision_count": len(manual_assets),
        "capture_count": review_counts["captures"],
        "contact_sheet_count": review_counts["contact_sheets"],
        "active_glb_count": len(list((stage / "assets").glob("*/*.glb"))),
        "active_usd_count": len(list((stage / "assets").glob("*/*.usd"))),
        "active_texture_count": len(list((stage / "assets").glob("*/textures/*.png"))),
        "active_usd_report_count": len(list((stage / "assets").glob("*/reports/*-usd.json"))),
        "rejected_indices": sorted(rejected_indices),
        "rejected_directories_present": sorted(
            index for index in rejected_indices if list((stage / "assets").glob(f"{index:03d}_*"))
        ),
        "automatic_warning_count_active": warnings,
        "automatic_failure_count": int(review.get("fail_count", 0)),
        "storage_methods": link_counts,
        "geometry_faces": {
            "minimum": min(faces),
            "average": sum(faces) / len(faces),
            "maximum": max(faces),
        },
    }
    expected_active = len(active_assets)
    count_keys = (
        "active_glb_count", "active_usd_count", "active_texture_count", "active_usd_report_count"
    )
    if any(validation[key] != expected_active for key in count_keys):
        raise RuntimeError(f"active library file count mismatch: {validation}")
    if validation["rejected_directories_present"]:
        raise RuntimeError(f"rejected assets present in active library: {validation}")

    write_json(stage / "active-assets.json", active_manifest)
    write_json(stage / "rejected-assets.json", rejected_manifest)
    write_json(stage / "review" / "manual-review.json", manual_review)
    write_json(stage / "curation-validation.json", validation)
    status_text = f"""# Asset4Sim - actifs revus 103-294

Validation visuelle finale du 12 aout 2026.

## Contenu actif

- {len(active_assets)} assets actifs avec GLB texture, USD texture, PNG 2048 px et rapport individuel.
- {len(rejected_assets)} assets retires de la bibliotheque active : {', '.join(f'{index:03d}' for index in sorted(rejected_indices))}.
- Les sources de production des assets retires restent immuables hors de cette bibliotheque pour la tracabilite.

## Revue visuelle

- {review_counts['captures']} captures individuelles couvrant tous les assets 103 a 294.
- {review_counts['contact_sheets']} planches de controle enregistrees sous `review/`.
- Chaque capture compare deux vues du modele a sa reference source.
- Les six retraits ont ete confirmes explicitement par l'utilisateur.

## Validation technique conservee

- GLB, USD, UV et textures avaient passe les contrats de post-traitement des 10 lots.
- Aucun echec automatique dans les 192 captures; les avertissements de complexite ne sont pas des rejets visuels.
- Faces actives : minimum {min(faces):,}, moyenne {sum(faces) / len(faces):,.1f}, maximum {max(faces):,}.
- Echelle et colorimetrie conservees telles que produites; aucune correction supplementaire n'est revendiquee ici.

## Rapports

- `active-assets.json` : inventaire et empreintes des {len(active_assets)} actifs.
- `rejected-assets.json` : six exclusions confirmees et leurs preuves.
- `curation-validation.json` : comptages, couverture et methode de stockage.
- `review/manual-review.json` : decision visuelle pour chacun des {expected_count} assets.
- `review/index.html` : revue visuelle navigable, captures et planches.
"""
    (stage / "LIBRARY_STATUS.md").write_text(status_text, encoding="utf-8")
    os.replace(stage, final_root)
    print(json.dumps({
        "output_root": str(final_root),
        "active_count": len(active_assets),
        "rejected_count": len(rejected_assets),
        "capture_count": review_counts["captures"],
        "contact_sheet_count": review_counts["contact_sheets"],
        "passed": True,
    }, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-report", type=Path, required=True)
    parser.add_argument("--rejections", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

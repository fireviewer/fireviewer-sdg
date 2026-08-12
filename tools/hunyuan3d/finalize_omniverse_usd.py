#!/usr/bin/env python3
"""Transactionally finalize corrected USDs for Omniverse Z-up validation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from restore_usd_glb_material import restore


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> int:
    manifest = json.loads(args.active_manifest.read_text(encoding="utf-8"))
    for asset in manifest["assets"]:
        index = int(asset["index"])
        glb = Path(asset["glb"])
        usd = Path(asset["usd"])
        asset_report = usd.parent / "asset-report.json"
        reports_root = usd.parent / "reports"
        building_usd = usd.with_name(".__tmp.usd")
        building_report = reports_root / "usd-material-restore-building.json"
        if building_usd.exists():
            building_usd.unlink()
        if building_report.exists():
            building_report.unlink()
        shutil.copy2(usd, building_usd)
        try:
            restored = restore(
                glb.resolve(),
                building_usd.resolve(),
                building_report.resolve(),
                True,
                1.0,
                "Z",
            )
            if (
                restored.get("passed") is not True
                or restored.get("up_axis") != "Z"
                or float(restored.get("root_rotation_x_degrees", -1.0)) != 0.0
                or float(restored.get("geometry_rotation_x_degrees", 0.0)) != 90.0
            ):
                raise RuntimeError(f"Omniverse finalization did not pass: {restored}")
            os.replace(building_usd, usd)
            restored["usd"] = str(usd.resolve())
            restored["texture"] = str((usd.parent / "textures" / f"{glb.stem}.png").resolve())
            atomic_json(reports_root / "usd-material-restore.json", restored)
            asset["usd_material_restore"] = restored
            atomic_json(asset_report, asset)
            building_report.unlink(missing_ok=True)
            print(json.dumps({
                "index": index,
                "asset_id": asset["asset_id"],
                "up_axis": restored["up_axis"],
                "passed": True,
            }), flush=True)
        except Exception:
            building_usd.unlink(missing_ok=True)
            raise

    manifest["status"] = "omniverse_z_up_prepared"
    manifest["usd_up_axis"] = "Z"
    manifest["usd_root_rotation_degrees"] = 0.0
    manifest["usd_orientation_policy"] = "Y-up GLB retained; 90-degree X orientation authored below the USD asset root"
    atomic_json(args.active_manifest, manifest)
    print(json.dumps({"asset_count": len(manifest["assets"]), "up_axis": "Z", "passed": True}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-manifest", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

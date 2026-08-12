#!/usr/bin/env python3
"""Create the Asset4Sim adaptive Hunyuan3D workflow from Kijai's example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def configure(source: Path, destination: Path) -> None:
    workflow = json.loads(source.read_text(encoding="utf-8"))
    nodes = {int(node["id"]): node for node in workflow["nodes"]}

    repair = nodes[59]
    if repair["type"] not in {"Hy3DPostprocessMesh", "Asset4SimAdaptiveRepairRetopo"}:
        raise ValueError(f"Unexpected node 59 type: {repair['type']}")
    repair["type"] = "Asset4SimAdaptiveRepairRetopo"
    repair["size"] = [340, 230]
    repair["inputs"] = [item for item in repair["inputs"] if item["name"] == "trimesh"]
    repair["properties"]["Node name for S&R"] = "Asset4SimAdaptiveRepairRetopo"
    repair["widgets_values"] = [5000, 2500, 50000, True, True]

    nodes[17]["widgets_values"][0] = "asset4sim/Hy3D_adaptive"
    nodes[99]["widgets_values"][0] = "asset4sim/Hy3D_textured_adaptive"

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(workflow, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    configure(args.source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

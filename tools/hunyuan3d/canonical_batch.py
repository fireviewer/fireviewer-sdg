#!/usr/bin/env python3
"""Run the exact user-supplied Hunyuan workflow, then its retexture binding."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from canonical_workflow import CANONICAL_SHA256, bind_initial, bind_retexture, load_canonical
from comfyui_batch import atomic_json, locate_exported_glb, submit_and_wait


LOG = logging.getLogger("asset4sim.hunyuan3d.canonical")


def load_state(path: Path, phase: str) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("phase") != phase:
            raise ValueError(f"State phase mismatch: {state.get('phase')} != {phase}")
        if state.get("canonical_workflow_sha256") != CANONICAL_SHA256:
            raise ValueError("State was created with another canonical workflow")
        return state
    return {
        "schema_version": 1,
        "phase": phase,
        "canonical_workflow_sha256": CANONICAL_SHA256,
        "assets": {},
    }


def retopo_by_asset(path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {item["asset_id"]: item for item in manifest["assets"]}


def find_existing_pair(
    output_root: Path,
    first_prefix: str,
    second_prefix: str,
    not_before: float,
) -> tuple[tuple[str, Path], tuple[str, Path]]:
    first = locate_exported_glb({}, output_root, first_prefix, not_before=not_before)
    second = locate_exported_glb({}, output_root, second_prefix, not_before=not_before)
    return first, second


def run(args: argparse.Namespace) -> int:
    canonical = load_canonical(args.workflow)
    reference_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assets = [item for item in reference_manifest["assets"] if item["route"] == "hunyuan3d"]
    if args.asset_id:
        assets = [item for item in assets if item["asset_id"] == args.asset_id]
    if args.start:
        assets = assets[args.start :]
    if args.limit is not None:
        assets = assets[: args.limit]

    retopo = None
    if args.phase == "retexture":
        if not args.retopo_manifest:
            raise ValueError("--retopo-manifest is required for retexture")
        retopo = retopo_by_asset(args.retopo_manifest)

    phase_name = f"canonical_{args.phase}"
    state = load_state(args.state, phase_name)
    for index, asset in enumerate(assets, start=1):
        asset_id = asset["asset_id"]
        previous = state["assets"].get(asset_id, {})
        if args.phase == "initial":
            prompt, first_prefix, second_prefix = bind_initial(
                canonical,
                image_name=asset["input_name"],
                asset_id=asset_id,
            )
            first_key = "untextured_50k"
            second_key = "textured_50k"
        else:
            assert retopo is not None
            if asset_id not in retopo:
                raise KeyError(f"Missing corrected mesh for {asset_id}")
            prompt, first_prefix, second_prefix = bind_retexture(
                canonical,
                image_name=asset["input_name"],
                corrected_mesh_path=str(retopo[asset_id]["output"]),
                asset_id=asset_id,
            )
            first_key = "corrected_passthrough"
            second_key = "final_retextured"

        if previous.get("status") == "success" and not args.force:
            LOG.info("[%s/%s] skip %s", index, len(assets), asset_id)
            continue
        if previous.get("status") == "failed" and not args.force:
            try:
                first, second = find_existing_pair(
                    args.comfy_output,
                    first_prefix,
                    second_prefix,
                    float(previous.get("started_at", 0.0)),
                )
            except RuntimeError:
                pass
            else:
                state["assets"][asset_id] = {
                    "status": "success",
                    first_key: str(first[1]),
                    second_key: str(second[1]),
                    "source_sha256": asset["source_sha256"],
                    "reconciled_existing_exports": True,
                    "started_at": previous.get("started_at"),
                    "finished_at": time.time(),
                }
                atomic_json(args.state, state)
                continue

        started = time.time()
        state["assets"][asset_id] = {"status": "running", "started_at": started}
        atomic_json(args.state, state)
        LOG.info("[%s/%s] %s %s", index, len(assets), args.phase, asset_id)
        try:
            prompt_id, record = submit_and_wait(
                args.comfy_url,
                prompt,
                timeout_seconds=args.timeout,
                poll_seconds=args.poll,
            )
            first, second = find_existing_pair(
                args.comfy_output,
                first_prefix,
                second_prefix,
                started,
            )
            state["assets"][asset_id] = {
                "status": "success",
                "prompt_id": prompt_id,
                first_key: str(first[1]),
                second_key: str(second[1]),
                "source_sha256": asset["source_sha256"],
                "started_at": started,
                "finished_at": time.time(),
            }
        except Exception as exc:
            state["assets"][asset_id] = {
                "status": "failed",
                "error": str(exc),
                "source_sha256": asset["source_sha256"],
                "started_at": started,
                "finished_at": time.time(),
            }
            atomic_json(args.state, state)
            if not args.continue_on_error:
                raise
            LOG.exception("Asset failed: %s", asset_id)
        atomic_json(args.state, state)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("initial", "retexture"))
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--retopo-manifest", type=Path)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-output", type=Path, default=Path("/workspace/hunyuan3d/ComfyUI/output"))
    parser.add_argument("--asset-id")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=7_200)
    parser.add_argument("--poll", type=float, default=2.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

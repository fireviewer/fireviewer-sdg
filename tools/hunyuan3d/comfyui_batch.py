#!/usr/bin/env python3
"""Submit resumable Asset4Sim geometry or texture prompts to a local ComfyUI API."""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from api_workflows import geometry_prompt, texture_prompt


LOG = logging.getLogger("asset4sim.hunyuan3d.batch")


def api_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def submit_and_wait(
    base_url: str,
    prompt: dict[str, Any],
    *,
    timeout_seconds: int,
    poll_seconds: float,
) -> tuple[str, dict[str, Any]]:
    client_id = str(uuid.uuid4())
    reply = api_json(f"{base_url.rstrip('/')}/prompt", {"prompt": prompt, "client_id": client_id})
    if "prompt_id" not in reply:
        raise RuntimeError(f"ComfyUI rejected the prompt: {reply}")
    prompt_id = str(reply["prompt_id"])
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            history = api_json(f"{base_url.rstrip('/')}/history/{prompt_id}")
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            # Some Hunyuan3D nodes keep ComfyUI's HTTP loop busy for several
            # minutes while doing CPU-side mesh/texture work.  The prompt is
            # still running, so a transient history timeout must not abort the
            # resumable batch controller.
            LOG.warning("ComfyUI history temporarily unavailable for %s: %s", prompt_id, exc)
            time.sleep(poll_seconds)
            continue
        if prompt_id in history:
            record = history[prompt_id]
            status = record.get("status", {})
            if status.get("completed"):
                if status.get("status_str") != "success":
                    raise RuntimeError(f"Prompt {prompt_id} failed: {status}")
                return prompt_id, record
        time.sleep(poll_seconds)
    raise TimeoutError(f"Prompt {prompt_id} exceeded {timeout_seconds} seconds")


def exported_glb(record: dict[str, Any]) -> str:
    for node_output in record.get("outputs", {}).values():
        for key in ("glb_path", "string", "text"):
            values = node_output.get(key)
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str) and value.lower().endswith(".glb"):
                        return value
            elif isinstance(values, str) and values.lower().endswith(".glb"):
                return values
    raise RuntimeError(f"No GLB path found in ComfyUI outputs: {record.get('outputs', {})}")


def locate_exported_glb(
    record: dict[str, Any],
    output_root: Path,
    filename_prefix: str,
    *,
    not_before: float,
) -> tuple[str, Path]:
    """Resolve exporters that save correctly but expose no UI history output."""

    try:
        relative = Path(exported_glb(record))
        absolute = (output_root / relative).resolve()
        if absolute.is_file() and absolute.stat().st_size > 0:
            return relative.as_posix(), absolute
    except RuntimeError:
        pass

    prefix = Path(filename_prefix)
    directory = output_root / prefix.parent
    candidates = [
        path
        for path in directory.glob(f"{prefix.name}_*.glb")
        if path.is_file() and path.stat().st_size > 0 and path.stat().st_mtime >= not_before - 2.0
    ]
    if not candidates:
        raise RuntimeError(
            f"No non-empty GLB found for prefix {filename_prefix!r} after {not_before}"
        )
    absolute = max(candidates, key=lambda path: path.stat().st_mtime).resolve()
    return absolute.relative_to(output_root.resolve()).as_posix(), absolute


def load_state(path: Path, phase: str) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("phase") != phase:
            raise ValueError(f"State phase {state.get('phase')!r} does not match {phase!r}")
        return state
    return {"schema_version": 1, "phase": phase, "assets": {}}


def resolve_retopo_path(asset: dict[str, Any], retopo_manifest: dict[str, Any]) -> str:
    by_id = {item["asset_id"]: item for item in retopo_manifest["assets"]}
    item = by_id.get(asset["asset_id"])
    if not item or not item.get("output"):
        raise KeyError(f"No retopologized mesh for {asset['asset_id']}")
    return str(item["output"])


def run(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assets = [item for item in manifest["assets"] if item["route"] == "hunyuan3d"]
    if args.asset_id:
        assets = [item for item in assets if item["asset_id"] == args.asset_id]
    if args.start:
        assets = assets[args.start :]
    if args.limit is not None:
        assets = assets[: args.limit]

    retopo_manifest = None
    if args.phase == "texture":
        if not args.retopo_manifest:
            raise ValueError("--retopo-manifest is required for the texture phase")
        retopo_manifest = json.loads(args.retopo_manifest.read_text(encoding="utf-8"))

    state = load_state(args.state, args.phase)
    for index, asset in enumerate(assets, start=1):
        asset_id = asset["asset_id"]
        previous = state["assets"].get(asset_id, {})

        if args.phase == "geometry":
            prefix = f"asset4sim/raw/{asset_id}/{asset_id}"
            prompt = geometry_prompt(asset["input_name"], prefix, asset["seed"])
        else:
            assert retopo_manifest is not None
            mesh_path = resolve_retopo_path(asset, retopo_manifest)
            prefix = f"asset4sim/textured/{asset_id}/{asset_id}"
            prompt = texture_prompt(asset["input_name"], mesh_path, prefix, asset["seed"])

        if previous.get("status") == "success" and not args.force:
            LOG.info("[%s/%s] skip %s (already complete)", index, len(assets), asset_id)
            continue
        if previous.get("status") == "failed" and not args.force:
            try:
                relative_glb, absolute_glb = locate_exported_glb(
                    {},
                    args.comfy_output,
                    prefix,
                    not_before=float(previous.get("started_at", 0.0)),
                )
            except RuntimeError:
                pass
            else:
                state["assets"][asset_id] = {
                    "status": "success",
                    "output_relative": relative_glb,
                    "output": str(absolute_glb),
                    "source_sha256": asset["source_sha256"],
                    "reconciled_existing_export": True,
                    "started_at": previous.get("started_at"),
                    "finished_at": time.time(),
                }
                atomic_json(args.state, state)
                LOG.info("[%s/%s] reconciled existing GLB for %s", index, len(assets), asset_id)
                continue

        LOG.info("[%s/%s] %s %s", index, len(assets), args.phase, asset_id)
        started = time.time()
        state["assets"][asset_id] = {"status": "running", "started_at": started}
        atomic_json(args.state, state)
        try:
            prompt_id, record = submit_and_wait(
                args.comfy_url,
                prompt,
                timeout_seconds=args.timeout,
                poll_seconds=args.poll,
            )
            relative_glb, absolute_glb = locate_exported_glb(
                record,
                args.comfy_output,
                prefix,
                not_before=started,
            )
            state["assets"][asset_id] = {
                "status": "success",
                "prompt_id": prompt_id,
                "output_relative": relative_glb,
                "output": str(absolute_glb),
                "source_sha256": asset["source_sha256"],
                "started_at": started,
                "finished_at": time.time(),
            }
        except (OSError, RuntimeError, TimeoutError, urllib.error.URLError) as exc:
            state["assets"][asset_id] = {
                "status": "failed",
                "error": str(exc),
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
    parser.add_argument("phase", choices=("geometry", "texture"))
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

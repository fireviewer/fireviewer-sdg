#!/usr/bin/env python3
"""Submit composition-locked FireViewer RGB restyle jobs without waiting for renders."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fireviewer_sdg.rgb_restyle import (
    _load_json,
    _request_json,
    _upload_image,
    prepare_job,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--captures-root", type=Path, required=True)
    result.add_argument("--job-root", type=Path, required=True)
    result.add_argument("--contract", type=Path, required=True)
    result.add_argument("--model-root", type=Path, required=True)
    result.add_argument("--server", default="http://127.0.0.1:8188")
    result.add_argument("--confirm-gpu-workload", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if not args.confirm_gpu_workload:
        raise SystemExit("GPU submission requires --confirm-gpu-workload")

    captures = sorted(path.parent for path in args.captures_root.rglob("rgb.png"))
    if not captures:
        raise SystemExit(f"No RGB captures found below {args.captures_root}")

    args.job_root.mkdir(parents=True, exist_ok=True)
    receipt_path = args.job_root / "submissions.jsonl"
    queued_count = 0
    with receipt_path.open("a", encoding="utf-8") as receipt:
        for position, capture_dir in enumerate(captures, start=1):
            manifest_path, manifest = prepare_job(
                capture_dir,
                args.job_root,
                contract_path=args.contract,
                model_root=args.model_root,
                hash_models=False,
            )
            upload = manifest["comfy_upload"]
            uploaded = _upload_image(
                args.server,
                Path(manifest["source_rgb_path"]),
                filename=upload["filename"],
                subfolder=upload["subfolder"],
            )
            prompt = _load_json(Path(manifest["prompt_path"]))
            prompt["4"]["inputs"]["image"] = (
                f"{uploaded.get('subfolder', upload['subfolder'])}/"
                f"{uploaded.get('name', upload['filename'])}"
            )
            queued = _request_json(
                args.server.rstrip("/") + "/prompt",
                method="POST",
                payload={
                    "prompt": prompt,
                    "client_id": "fireviewer-rgb-restyle-" + manifest["job_id"],
                },
                timeout=30.0,
            )
            prompt_id = queued.get("prompt_id")
            if not isinstance(prompt_id, str):
                raise RuntimeError(f"ComfyUI did not return prompt_id: {queued}")
            receipt.write(
                json.dumps(
                    {
                        "position": position,
                        "job_id": manifest["job_id"],
                        "capture_dir": str(capture_dir),
                        "manifest_path": str(manifest_path),
                        "prompt_id": prompt_id,
                        "queued_at_unix": time.time(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            receipt.flush()
            queued_count += 1
            print(f"queued {position}/{len(captures)} prompt_id={prompt_id}", flush=True)
    print(json.dumps({"queued": queued_count, "receipt": str(receipt_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

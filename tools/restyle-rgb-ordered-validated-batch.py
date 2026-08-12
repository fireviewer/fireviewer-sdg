#!/usr/bin/env python3
"""Restyle only accepted positive-fire ``pointXX/original/rgb.png`` captures.

The source capture remains immutable.  Zoom directories and non-positive or
visibility-rejected captures are never submitted to ComfyUI.  Before GPU work,
the runner binds the flame/smoke projections to their co-registered masks.  A
generated candidate is admitted only through the composition lock, whose
protected core restores the exact source fire, smoke and perimeter pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fireviewer_sdg.rgb_restyle import (
    AdmissionRejected,
    RestyleContractError,
    _load_json,
    _write_json_atomic,
    admit_candidate,
    execute_single_job,
    inventory_capture,
    load_contract,
    prepare_job,
)


DAY_CASE_POINT = re.compile(r"^(day|case|point)(\d+)$", re.IGNORECASE)


def _number(value: str) -> int:
    match = DAY_CASE_POINT.match(value)
    return int(match.group(2)) if match else 10**9


def _series_key(capture: Path) -> tuple[str, str, str]:
    """Return the strict day/case/point key for an original capture."""
    if capture.name.casefold() != "original":
        raise RestyleContractError(f"capture is not an original framing: {capture}")
    point = capture.parent
    case = point.parent
    day = case.parent
    expected = ((day.name, "day"), (case.name, "case"), (point.name, "point"))
    if any(not re.fullmatch(rf"{prefix}\d+", value, re.I) for value, prefix in expected):
        raise RestyleContractError(f"capture path is not day/case/point/original scoped: {capture}")
    return (day.name, case.name, point.name)


def ordered_original_captures(captures_root: Path) -> list[tuple[tuple[str, str, str], Path]]:
    """Discover only the canonical, non-zoom RGB framing for every point."""
    records: list[tuple[tuple[str, str, str], Path]] = []
    for rgb in captures_root.rglob("rgb.png"):
        capture = rgb.parent
        if capture.name.casefold() != "original":
            continue
        try:
            key = _series_key(capture)
        except RestyleContractError:
            continue
        records.append((key, capture))
    if not records:
        raise RestyleContractError(
            f"no pointXX/original/rgb.png captures found below {captures_root}"
        )
    records.sort(key=lambda record: tuple(_number(part) for part in record[0]))
    return records


def _selection_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    selection = contract.get("batch_selection")
    if not isinstance(selection, Mapping):
        raise RestyleContractError("validated contract has no batch_selection")
    if selection.get("capture_member") != "original":
        raise RestyleContractError("batch_selection must be locked to original")
    if selection.get("sample_kind") != "positive_fire":
        raise RestyleContractError("batch_selection must be locked to positive_fire")
    if selection.get("visibility_validation_status") != "accepted":
        raise RestyleContractError("batch_selection visibility must be locked to accepted")
    return selection


def capture_selection(capture: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the exact training-target admission rule without touching a GPU."""
    selection = _selection_contract(contract)
    if capture.name.casefold() != str(selection["capture_member"]).casefold():
        return {"selected": False, "reason": "not-original"}
    targets = _load_json(capture / "training-targets.json")
    sample_kind = targets.get("sample_kind")
    visibility = targets.get("visibility")
    acceptance = visibility.get("acceptance") if isinstance(visibility, Mapping) else None
    status = (
        acceptance.get("visibility_validation_status")
        if isinstance(acceptance, Mapping)
        else None
    )
    selected = (
        sample_kind == selection["sample_kind"]
        and status == selection["visibility_validation_status"]
    )
    reason = "selected" if selected else "training-targets-selection-mismatch"
    return {
        "selected": selected,
        "reason": reason,
        "sample_kind": sample_kind,
        "visibility_validation_status": status,
    }


def _load_mask(capture: Path, name: str, shape: tuple[int, int]) -> np.ndarray:
    path = capture / name
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise RestyleContractError(f"event mask archive is empty: {name}")
            value = np.asarray(archive[archive.files[0]])
    except (OSError, ValueError) as exc:
        raise RestyleContractError(f"cannot read event mask {path}: {exc}") from exc
    if value.shape != shape:
        raise RestyleContractError(f"event mask is not co-registered: {name}")
    return value != 0


def _projection(targets: Mapping[str, Any], key: str) -> tuple[bool, tuple[float, float] | None]:
    witness = targets.get(key)
    projection = witness.get("projection") if isinstance(witness, Mapping) else None
    if not isinstance(projection, Mapping):
        return False, None
    in_frame = projection.get("in_frame") is True
    xy = projection.get("pixel_xy")
    if in_frame and (
        not isinstance(xy, list)
        or len(xy) != 2
        or not all(isinstance(item, (int, float)) for item in xy)
    ):
        raise RestyleContractError(f"{key} in-frame projection is invalid")
    return in_frame, (float(xy[0]), float(xy[1])) if in_frame else None


def _event_mask_witness(
    capture: Path,
    targets: Mapping[str, Any],
    *,
    mask_name: str,
    target_name: str,
    shape: tuple[int, int],
    tolerance_px: float,
) -> dict[str, Any]:
    mask = _load_mask(capture, mask_name, shape)
    yy, xx = np.nonzero(mask)
    count = int(xx.size)
    in_frame, xy = _projection(targets, target_name)
    if count and not in_frame:
        raise RestyleContractError(
            f"{mask_name} has visible pixels but {target_name} is not projected in frame"
        )
    if not count and in_frame:
        raise RestyleContractError(
            f"{target_name} is projected in frame but {mask_name} is empty"
        )
    bounds: list[int] | None = None
    if count:
        bounds = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
        assert xy is not None
        x, y = xy
        width, height = shape[1], shape[0]
        if not (0.0 <= x < width and 0.0 <= y < height):
            raise RestyleContractError(f"{target_name} projection lies outside the image")
        if not (
            bounds[0] - tolerance_px <= x <= bounds[2] + tolerance_px
            and bounds[1] - tolerance_px <= y <= bounds[3] + tolerance_px
        ):
            raise RestyleContractError(
                f"{target_name} projection is not anchored to {mask_name}"
            )
    return {
        "mask": mask_name,
        "target": target_name,
        "pixel_count": count,
        "bounds_xyxy": bounds,
        "projection_in_frame": in_frame,
        "projection_pixel_xy": list(xy) if xy is not None else None,
    }


def validate_original_capture(capture: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Bind selection, source hashes, event masks and event projections."""
    selection = capture_selection(capture, contract)
    if not selection["selected"]:
        raise RestyleContractError(
            "capture is not an accepted positive_fire original according to training-targets.json"
        )
    inventory = inventory_capture(capture, contract)
    if inventory.sample_kind != "positive_fire" or not inventory.expected_fire_visible:
        raise RestyleContractError("positive-fire selection disagrees with capture metadata")
    event_lock = contract.get("event_lock")
    if not isinstance(event_lock, Mapping):
        raise RestyleContractError("validated contract has no event_lock")
    tolerance_px = float(event_lock.get("projection_mask_tolerance_px", 0.0))
    if tolerance_px < 0:
        raise RestyleContractError("event projection tolerance must be non-negative")
    targets = _load_json(capture / "training-targets.json")
    shape = (inventory.height_px, inventory.width_px)
    witnesses = [
        _event_mask_witness(
            capture,
            targets,
            mask_name="flame_mask.npz",
            target_name="nearest_flame",
            shape=shape,
            tolerance_px=tolerance_px,
        ),
        _event_mask_witness(
            capture,
            targets,
            mask_name="smoke_mask.npz",
            target_name="nearest_smoke",
            shape=shape,
            tolerance_px=tolerance_px,
        ),
    ]
    if event_lock.get("require_visible_flame_mask") and witnesses[0]["pixel_count"] <= 0:
        raise RestyleContractError("accepted positive_fire capture has no visible flame mask")
    if event_lock.get("require_visible_smoke_mask") and witnesses[1]["pixel_count"] <= 0:
        raise RestyleContractError("accepted positive_fire capture has no visible smoke mask")
    return {
        "status": "passed",
        "capture": str(capture),
        "capture_id": inventory.capture_id,
        "selection": selection,
        "source_binding_sha256": inventory.source_binding_sha256,
        "protected_pixel_count": inventory.protected_pixel_count,
        "event_witnesses": witnesses,
    }


def _capture_seed(key: tuple[str, str, str]) -> int:
    digest = hashlib.sha256("/".join(key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _manifest_index(job_root: Path) -> dict[str, Path]:
    records: dict[str, Path] = {}
    for manifest_path in job_root.glob("*/job-manifest.json"):
        try:
            manifest = _load_json(manifest_path)
            source = manifest.get("source_rgb_path")
            if isinstance(source, str):
                records[str(Path(source).resolve())] = manifest_path
        except RestyleContractError:
            continue
    return records


def _job_manifest(
    capture: Path,
    *,
    index: dict[str, Path],
    job_root: Path,
    contract_path: Path,
    model_root: Path,
) -> Path:
    source = str((capture / "rgb.png").resolve())
    existing = index.get(source)
    if existing is not None:
        return existing
    manifest_path, _manifest = prepare_job(
        capture,
        job_root,
        contract_path=contract_path,
        model_root=model_root,
        hash_models=False,
    )
    index[source] = manifest_path
    return manifest_path


def _write_receipt(path: Path, value: Mapping[str, Any]) -> None:
    _write_json_atomic(path, value)


def _append_ledger(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--captures-root", type=Path, required=True)
    result.add_argument("--job-root", type=Path, required=True)
    result.add_argument("--contract", type=Path, required=True)
    result.add_argument("--model-root", type=Path, required=True)
    result.add_argument("--server", default="http://127.0.0.1:8188")
    result.add_argument("--timeout-seconds", type=float, default=1800.0)
    result.add_argument("--confirm-gpu-workload", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if not args.confirm_gpu_workload:
        raise SystemExit("GPU execution requires --confirm-gpu-workload")
    contract = load_contract(args.contract)
    if not contract["release_gate"].get("batch_generation_allowed"):
        raise SystemExit("validated contract does not allow batch generation")
    _selection_contract(contract)
    retry_count = int(contract["retry_policy"]["max_regenerations_after_rejection"])
    captures = ordered_original_captures(args.captures_root)
    index = _manifest_index(args.job_root)
    ledger = args.job_root / "ordered-validation-ledger.jsonl"
    selected_count = 0
    for capture_index, (key, capture) in enumerate(captures, start=1):
        selection = capture_selection(capture, contract)
        capture_dir = args.job_root / "captures" / "-".join(key)
        if not selection["selected"]:
            record = {
                "capture": str(capture),
                "series": key,
                "status": "skipped-selection",
                "selection": selection,
            }
            _write_receipt(capture_dir / "selection-receipt.json", record)
            _append_ledger(ledger, record)
            continue
        selected_count += 1
        try:
            preflight = validate_original_capture(capture, contract)
        except RestyleContractError as exc:
            record = {
                "capture": str(capture),
                "series": key,
                "status": "rejected-preflight",
                "error": str(exc),
            }
            _write_receipt(capture_dir / "capture-receipt.json", record)
            _append_ledger(ledger, record)
            continue
        _write_receipt(
            capture_dir / "capture-preflight.json", {"series": key, **preflight}
        )
        destination = capture / contract["source_contract"]["derived_output_relative_dir"]
        if (destination / "restyle-receipt.json").is_file():
            continue
        manifest_path = _job_manifest(
            capture,
            index=index,
            job_root=args.job_root,
            contract_path=args.contract,
            model_root=args.model_root,
        )
        attempts: list[dict[str, Any]] = []
        accepted = False
        for attempt in range(retry_count + 1):
            try:
                runtime = execute_single_job(
                    manifest_path,
                    server=args.server,
                    confirm_gpu_workload=True,
                    timeout_seconds=args.timeout_seconds,
                    attempt=attempt,
                    seed_override=_capture_seed(key),
                )
                admitted = admit_candidate(
                    capture,
                    Path(runtime["candidate_path"]),
                    contract_path=args.contract,
                )
                qa = _load_json(Path(admitted["qa_path"]))
                if not qa.get("gates", {}).get("protected_core_exact"):
                    raise RestyleContractError("admitted candidate violated the fire/smoke core")
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "accepted",
                        "runtime": runtime,
                        "admission": admitted,
                        "event_lock": preflight["event_witnesses"],
                    }
                )
                accepted = True
                break
            except AdmissionRejected as exc:
                attempts.append(
                    {"attempt": attempt, "status": "rejected-composition", "qa": exc.report}
                )
            except RestyleContractError as exc:
                attempts.append(
                    {"attempt": attempt, "status": "rejected-runtime", "error": str(exc)}
                )
                if "source" in str(exc).casefold() or "contract" in str(exc).casefold():
                    break
            time.sleep(1.0)
        record = {
            "series": key,
            "capture": str(capture),
            "status": "accepted" if accepted else "rejected-exhausted",
            "attempts": attempts,
        }
        _write_receipt(manifest_path.parent / "validation-receipt.json", record)
        _append_ledger(ledger, record)
        print(
            f"capture {capture_index}/{len(captures)} complete: {'/'.join(key)}/original",
            flush=True,
        )
    if selected_count == 0:
        raise SystemExit(
            "no accepted positive_fire original captures matched training-targets.json"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

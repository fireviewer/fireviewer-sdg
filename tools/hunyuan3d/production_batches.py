#!/usr/bin/env python3
"""Produce Hunyuan3D assets in transfer-gated batches and retrieve them safely.

The ``produce`` side runs on the GPU pod.  It completes shape + first texture,
hole repair, quality-gated retopology, corrected-mesh retexture, GLB checks and
OpenUSD conversion for one batch, then waits for a verified transfer receipt
before starting the next batch.

With ``--generation-only``, the pod stops after the canonical initial shape and
texture pass.  It transfers both the textured and untextured 50k source GLBs so
retopology, correction and USD publication can be deferred to the local host.

The ``retrieve`` side runs on Windows.  It downloads each ready bundle to D:,
verifies the archive and every payload hash, publishes the batch atomically,
then acknowledges the transfer to the pod.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


LOG = logging.getLogger("asset4sim.hunyuan3d.production")
RETOPO_MAXIMUM_FACES = 50_000
DEFAULT_TRANSPORT_TIMEOUT_SECONDS = 120.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_hunyuan_assets(manifest: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assets = [item for item in payload["assets"] if item.get("route") == "hunyuan3d"]
    if not assets:
        raise RuntimeError("Reference manifest has no Hunyuan3D assets")
    return payload, assets


@dataclass(frozen=True)
class BatchPlan:
    ordinal: int
    start_index: int
    end_index: int
    assets: tuple[dict[str, Any], ...]

    @property
    def name(self) -> str:
        return f"batch-{self.start_index + 1:04d}-{self.end_index:04d}"


def plan_batches(assets: list[dict[str, Any]], start: int, batch_size: int) -> list[BatchPlan]:
    if start < 0 or start > len(assets):
        raise ValueError(f"Invalid start index {start} for {len(assets)} assets")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    plans: list[BatchPlan] = []
    for ordinal, offset in enumerate(range(start, len(assets), batch_size), start=1):
        selected = tuple(assets[offset : offset + batch_size])
        plans.append(BatchPlan(ordinal, offset, offset + len(selected), selected))
    return plans


def run_checked(command: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("run %s", " ".join(command))
    with log_path.open("a", encoding="utf-8", errors="replace") as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, text=True, env=env)


def free_comfy_memory(comfy_url: str) -> None:
    """Unload cached models only when no ComfyUI prompt is running."""

    request = urllib.request.Request(
        f"{comfy_url.rstrip('/')}/free",
        data=b'{"unload_models":true,"free_memory":true}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status >= 300:
            raise RuntimeError(f"ComfyUI memory release failed with HTTP {response.status}")
    LOG.info("ComfyUI cached models and execution memory released")


def usd_runtime_environment(runtime_prefix: Path | None = None) -> dict[str, str]:
    """Expose the NVIDIA wheel's native USD libraries before Python starts."""

    env = os.environ.copy()
    if sys.platform.startswith("linux"):
        prefix = (runtime_prefix or Path(sys.prefix)).resolve()
        library_dir = (
            prefix
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
            / "omni"
            / "converter"
            / "hoops"
        )
        if not library_dir.is_dir():
            raise RuntimeError(f"usd-convert-cad native library directory is missing: {library_dir}")
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(library_dir) + (os.pathsep + current if current else "")
    return env


def ensure_success_state(state_path: Path, asset_ids: Iterable[str]) -> dict[str, Any]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    failed = [asset_id for asset_id in asset_ids if state.get("assets", {}).get(asset_id, {}).get("status") != "success"]
    if failed:
        raise RuntimeError(f"Incomplete batch state {state_path}: {failed}")
    return state


def subset_manifest(source: dict[str, Any], plan: BatchPlan) -> dict[str, Any]:
    payload = dict(source)
    payload["assets"] = list(plan.assets)
    payload["asset_count"] = len(plan.assets)
    payload["route_counts"] = {"hunyuan3d": len(plan.assets)}
    payload["production_batch"] = {
        "ordinal": plan.ordinal,
        "start_asset_number": plan.start_index + 1,
        "end_asset_number": plan.end_index,
    }
    return payload


def copy_final_glbs(state: dict[str, Any], plan: BatchPlan, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for asset in plan.assets:
        asset_id = asset["asset_id"]
        source = Path(state["assets"][asset_id]["final_retextured"])
        if not source.is_file():
            raise RuntimeError(f"Missing final retextured GLB: {source}")
        target = destination / f"{asset_id}.glb"
        shutil.copy2(source, target)
        outputs.append(target)
    return outputs


def copy_initial_glbs(
    state: dict[str, Any],
    plan: BatchPlan,
    textured_destination: Path,
    untextured_destination: Path,
) -> list[Path]:
    """Copy the canonical initial pair without running any post-processing."""

    textured_destination.mkdir(parents=True, exist_ok=True)
    untextured_destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for asset in plan.assets:
        asset_id = asset["asset_id"]
        record = state["assets"][asset_id]
        for key, destination in (
            ("textured_50k", textured_destination),
            ("untextured_50k", untextured_destination),
        ):
            source = Path(record[key])
            if not source.is_file():
                raise RuntimeError(f"Missing canonical initial {key} GLB: {source}")
            target = destination / f"{asset_id}.glb"
            shutil.copy2(source, target)
            outputs.append(target)
    return outputs


def transfer_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "transfer-manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {"schema_version": 1, "file_count": len(files), "files": files}


def verify_transfer_manifest(root: Path, manifest: dict[str, Any]) -> None:
    expected = manifest.get("files", [])
    if manifest.get("file_count") != len(expected):
        raise RuntimeError("Transfer manifest file_count mismatch")
    for item in expected:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe transfer path: {relative}")
        path = root.joinpath(*relative.parts)
        if not path.is_file():
            raise RuntimeError(f"Missing transferred file: {path}")
        if path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"Transferred size mismatch: {path}")
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Transferred SHA-256 mismatch: {path}")


def validate_batch_contract(
    plan: BatchPlan,
    retopo_manifest_path: Path,
    glb_report_path: Path,
    usd_report_path: Path,
) -> dict[str, Any]:
    retopo = json.loads(retopo_manifest_path.read_text(encoding="utf-8"))
    glb = json.loads(glb_report_path.read_text(encoding="utf-8"))
    usd = json.loads(usd_report_path.read_text(encoding="utf-8"))
    accepted = [int(item["accepted_faces"]) for item in retopo["assets"]]
    failures: list[str] = []
    if retopo.get("asset_count") != len(plan.assets):
        failures.append("retopology count mismatch")
    if any(value > RETOPO_MAXIMUM_FACES for value in accepted):
        failures.append("retopology maximum exceeded")
    if not glb.get("passed"):
        failures.append("GLB geometry/UV/texture validation failed")
    if not usd.get("passed"):
        failures.append("USD conversion or structural validation failed")
    return {
        "schema_version": 1,
        "batch": plan.name,
        "asset_count": len(plan.assets),
        "asset_ids": [item["asset_id"] for item in plan.assets],
        "target_average_faces": 5_000,
        "final_average_faces": (sum(accepted) / len(accepted)) if accepted else None,
        "maximum_faces": max(accepted) if accepted else None,
        "glb_passed": bool(glb.get("passed")),
        "usd_passed": bool(usd.get("passed")),
        "failures": failures,
        "passed": not failures,
    }


def safe_remove_tree(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"Refusing to remove path outside batch root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def produce(args: argparse.Namespace) -> int:
    source_manifest, assets = load_hunyuan_assets(args.manifest)
    plans = plan_batches(assets, args.start, args.batch_size)
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.outgoing_root.mkdir(parents=True, exist_ok=True)
    receipts_root = args.work_root / "receipts"
    receipts_root.mkdir(parents=True, exist_ok=True)

    python = str(args.python)
    tool_root = args.tool_root.resolve()
    canonical = str(tool_root / "canonical_batch.py")
    freezer = str(tool_root / "freeze_completed_initial.py")
    mesh_tool = str(tool_root / "asset4sim_mesh.py")
    glb_verifier = str(tool_root / "verify_glb_delivery.py")
    usd_converter = str(tool_root / "batch_convert_glb_to_usd.py")
    workflow = str(args.workflow.resolve())

    production_state = {
        "schema_version": 1,
        "mode": "generation_only" if args.generation_only else "simready",
        "manifest": str(args.manifest.resolve()),
        "hunyuan_asset_count": len(assets),
        "start_index": args.start,
        "remaining_count": len(assets) - args.start,
        "batch_size": args.batch_size,
        "batch_count": len(plans),
        "batches": [plan.name for plan in plans],
    }
    atomic_json(args.work_root / "production-plan.json", production_state)

    for plan in plans:
        receipt = receipts_root / f"{plan.name}.json"
        if receipt.is_file():
            LOG.info("skip transferred %s", plan.name)
            continue

        batch_root = args.work_root / "batches" / plan.name
        batch_root.mkdir(parents=True, exist_ok=True)
        manifest_path = batch_root / "reference-manifest.json"
        atomic_json(manifest_path, subset_manifest(source_manifest, plan))
        ids = [item["asset_id"] for item in plan.assets]
        log_path = batch_root / "pipeline.log"
        free_comfy_memory(args.comfy_url)

        initial_state = batch_root / "initial-state.json"
        run_checked(
            [
                python, canonical, "initial", "--workflow", workflow, "--manifest", str(manifest_path),
                "--state", str(initial_state), "--comfy-output", str(args.comfy_output),
                "--comfy-url", args.comfy_url,
                "--timeout", str(args.timeout), "--poll", "2", "--verbose",
            ],
            log_path,
        )
        initial = ensure_success_state(initial_state, ids)
        free_comfy_memory(args.comfy_url)

        outgoing = args.outgoing_root / plan.name
        temporary = args.outgoing_root / (plan.name + ".building")
        delivery = temporary / "delivery"
        glb_root = delivery / "glb"
        reports_root = delivery / "reports"

        if args.generation_only:
            safe_remove_tree(temporary, args.outgoing_root)
            copy_initial_glbs(initial, plan, glb_root, delivery / "untextured_50k")
            shutil.copy2(manifest_path, delivery / "reference-manifest.json")
            contract = {
                "schema_version": 1,
                "batch": plan.name,
                "mode": "generation_only",
                "asset_count": len(plan.assets),
                "asset_ids": ids,
                "outputs": ["glb", "untextured_50k"],
                "post_processing": "deferred_local",
                "content_verification": "skipped_at_user_request",
            }
            atomic_json(reports_root / "batch-contract.json", contract)
        else:
            frozen_root = batch_root / "frozen-untextured"
            frozen_manifest = batch_root / "frozen-manifest.json"
            # These directories are derived from the resumable initial state.  A
            # restarted batch must rebuild them so stale symlinks or retopology
            # outputs from a previously interrupted run cannot leak into the
            # current five-asset contract.
            safe_remove_tree(frozen_root, batch_root)
            run_checked(
                [
                    python, freezer, "--state", str(initial_state),
                    "--untextured-root", str(args.comfy_output / "asset4sim/canonical_initial/untextured"),
                    "--textured-root", str(args.comfy_output / "asset4sim/canonical_initial/textured"),
                    "--stage-dir", str(frozen_root), "--manifest", str(frozen_manifest),
                    "--reference-manifest", str(manifest_path),
                ],
                log_path,
            )

            corrected_root = batch_root / "corrected"
            retopo_manifest = batch_root / "retopo-manifest.json"
            safe_remove_tree(corrected_root, batch_root)
            run_checked(
                [
                    python, mesh_tool, "--input-dir", str(frozen_root), "--output-dir", str(corrected_root),
                    "--manifest", str(retopo_manifest), "--target-average", "5000",
                    "--minimum-faces", "2500", "--maximum-faces", str(RETOPO_MAXIMUM_FACES),
                    "--quality-samples", "20000", "--verbose",
                ],
                log_path,
            )

            retexture_state = batch_root / "retexture-state.json"
            run_checked(
                [
                    python, canonical, "retexture", "--workflow", workflow, "--manifest", str(manifest_path),
                    "--retopo-manifest", str(retopo_manifest), "--state", str(retexture_state),
                    "--comfy-output", str(args.comfy_output), "--comfy-url", args.comfy_url,
                    "--timeout", str(args.timeout),
                    "--poll", "2", "--verbose",
                ],
                log_path,
            )
            retexture = ensure_success_state(retexture_state, ids)
            free_comfy_memory(args.comfy_url)

            safe_remove_tree(temporary, args.outgoing_root)
            usd_root = delivery / "usd"
            copy_final_glbs(retexture, plan, glb_root)

            glb_report = reports_root / "glb-validation.json"
            run_checked(
                [python, glb_verifier, "--input-dir", str(glb_root), "--expected-count", str(len(plan.assets)),
                 "--report", str(glb_report)],
                log_path,
            )
            run_checked(
                [
                    python, usd_converter, "--input-dir", str(glb_root), "--output-dir", str(usd_root),
                    "--reports-dir", str(reports_root / "usd"), "--expected-count", str(len(plan.assets)),
                    "--converter", str(args.usd_converter),
                ],
                log_path,
                env=usd_runtime_environment(Path(sys.prefix)),
            )
            usd_report = reports_root / "usd" / "usd-conversion-manifest.json"
            contract = validate_batch_contract(plan, retopo_manifest, glb_report, usd_report)
            atomic_json(reports_root / "batch-contract.json", contract)
            if not contract["passed"]:
                raise RuntimeError(f"Batch contract failed: {contract['failures']}")

        manifest = transfer_manifest(delivery)
        atomic_json(delivery / "transfer-manifest.json", manifest)
        if not args.generation_only:
            verify_transfer_manifest(delivery, manifest)
        bundle = temporary / f"{plan.name}.tar"
        with tarfile.open(bundle, "w") as archive:
            archive.add(delivery, arcname="delivery")
        (temporary / f"{plan.name}.tar.sha256").write_text(f"{sha256(bundle)}  {bundle.name}\n", encoding="ascii")
        (temporary / "READY").write_text("ready\n", encoding="ascii")
        if outgoing.exists():
            raise RuntimeError(f"Outgoing batch already exists: {outgoing}")
        os.replace(temporary, outgoing)
        LOG.info("%s ready; waiting for verified transfer", plan.name)

        transferred = outgoing / "TRANSFERRED"
        while not transferred.is_file():
            time.sleep(args.transfer_poll)

        receipt_payload = {
            "schema_version": 1,
            "batch": plan.name,
            "asset_ids": ids,
            "transferred_at": time.time(),
            "bundle_sha256": sha256(outgoing / f"{plan.name}.tar"),
            "contract": contract,
        }
        atomic_json(receipt, receipt_payload)

        for asset_id in ids:
            cleanup_paths = [
                f"asset4sim/canonical_initial/untextured/{asset_id}",
                f"asset4sim/canonical_initial/textured/{asset_id}",
            ]
            if not args.generation_only:
                cleanup_paths.extend(
                    [
                        f"asset4sim/retexture/corrected/{asset_id}",
                        f"asset4sim/retexture/final/{asset_id}",
                    ]
                )
            for relative in cleanup_paths:
                safe_remove_tree(args.comfy_output / relative, args.comfy_output)
        safe_remove_tree(outgoing, args.outgoing_root)
        if not args.generation_only:
            safe_remove_tree(frozen_root, batch_root)
            safe_remove_tree(corrected_root, batch_root)
        LOG.info("%s transferred and rotated", plan.name)

    complete = {
        "schema_version": 1,
        "status": "complete",
        "mode": "generation_only" if args.generation_only else "simready",
        "hunyuan_asset_count": len(assets),
        "start_index": args.start,
        "produced_count": len(assets) - args.start,
        "batch_count": len(plans),
        "completed_at": time.time(),
    }
    atomic_json(args.outgoing_root / "PRODUCTION_COMPLETE.json", complete)
    return 0


def ssh_base(args: argparse.Namespace) -> list[str]:
    return [
        "ssh", "-p", str(args.port), "-i", str(args.identity),
        "-o", f"UserKnownHostsFile={args.known_hosts}", "-o", "StrictHostKeyChecking=yes",
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ConnectionAttempts=1",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
        f"{args.user}@{args.host}",
    ]


def scp_base(args: argparse.Namespace) -> list[str]:
    return [
        "scp", "-P", str(args.port), "-i", str(args.identity),
        "-o", f"UserKnownHostsFile={args.known_hosts}", "-o", "StrictHostKeyChecking=yes",
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ConnectionAttempts=1",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
    ]


def retry_transport(
    args: argparse.Namespace,
    command: list[str],
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Keep the gated transfer alive across transient SSH/RunPod endpoint failures."""

    attempt = 0
    while True:
        attempt += 1
        try:
            return subprocess.run(
                command,
                check=True,
                capture_output=capture_output,
                text=True,
                timeout=getattr(
                    args,
                    "transport_timeout",
                    DEFAULT_TRANSPORT_TIMEOUT_SECONDS,
                ),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            return_code = getattr(exc, "returncode", None)
            LOG.warning(
                "transport attempt %s failed%s; retrying in %.1fs",
                attempt,
                f" (exit {return_code})" if return_code is not None else "",
                args.poll,
            )
            time.sleep(args.poll)


def remote_output(args: argparse.Namespace, command: str) -> str:
    completed = retry_transport(args, ssh_base(args) + [command], capture_output=True)
    return completed.stdout.strip()


def acknowledge_transfer(args: argparse.Namespace, ready: str) -> None:
    """Acknowledge a batch even if the producer rotated it concurrently."""

    remote_batch = shlex.quote(ready)
    remote_output(
        args,
        f"if test -d {remote_batch}; then touch {remote_batch}/TRANSFERRED; fi; exit 0",
    )


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Archive links are not accepted: {member.name}")
        archive.extractall(destination)


def retrieve(args: argparse.Namespace) -> int:
    args.local_root.mkdir(parents=True, exist_ok=True)
    incoming_root = args.local_root / ".incoming"
    incoming_root.mkdir(parents=True, exist_ok=True)
    while True:
        ready = remote_output(
            args,
            f"(find {args.remote_outgoing} -mindepth 2 -maxdepth 2 -name READY -printf '%h\\n' 2>/dev/null || true) | sort | head -n 1; exit 0",
        )
        if not ready:
            complete = remote_output(
                args,
                f"if test -f {args.remote_outgoing}/PRODUCTION_COMPLETE.json; then "
                f"cat {args.remote_outgoing}/PRODUCTION_COMPLETE.json; fi; exit 0",
            )
            if complete:
                atomic_json(args.local_root / "PRODUCTION_COMPLETE.json", json.loads(complete))
                LOG.info("remote production complete")
                return 0
            time.sleep(args.poll)
            continue

        batch_name = PurePosixPath(ready).name
        final = args.local_root / batch_name
        if final.exists():
            manifest = json.loads((final / "transfer-manifest.json").read_text(encoding="utf-8"))
            verify_transfer_manifest(final, manifest)
            acknowledge_transfer(args, ready)
            LOG.info("reconciled already verified local batch %s", batch_name)
            time.sleep(1)
            continue
        incoming = incoming_root / batch_name
        safe_remove_tree(incoming, incoming_root)
        incoming.mkdir(parents=True)
        archive_name = f"{batch_name}.tar"
        checksum_name = archive_name + ".sha256"
        remote = f"{args.user}@{args.host}:{ready}"
        retry_transport(
            args,
            scp_base(args) + [f"{remote}/{archive_name}", f"{remote}/{checksum_name}", str(incoming)],
        )
        archive_path = incoming / archive_name
        checksum_line = (incoming / checksum_name).read_text(encoding="ascii").strip()
        expected_hash, expected_name = checksum_line.split(maxsplit=1)
        if expected_name != archive_name or sha256(archive_path) != expected_hash:
            raise RuntimeError(f"Bundle checksum mismatch for {batch_name}")

        extracted = incoming / "extracted"
        safe_extract(archive_path, extracted)
        delivery = extracted / "delivery"
        manifest = json.loads((delivery / "transfer-manifest.json").read_text(encoding="utf-8"))
        verify_transfer_manifest(delivery, manifest)
        os.replace(delivery, final)
        acknowledge_transfer(args, ready)
        shutil.rmtree(incoming)
        LOG.info("retrieved and verified %s", batch_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    producer = subparsers.add_parser("produce")
    producer.add_argument("--manifest", type=Path, required=True)
    producer.add_argument("--workflow", type=Path, required=True)
    producer.add_argument("--tool-root", type=Path, default=Path("/opt/ComfyUI/custom_nodes/Asset4Sim_Hunyuan3D"))
    producer.add_argument("--python", type=Path, default=Path("/opt/comfy-venv/bin/python"))
    producer.add_argument("--usd-converter", type=Path, default=Path("/opt/comfy-venv/bin/usd-convert-cad"))
    producer.add_argument("--comfy-output", type=Path, default=Path("/workspace/output"))
    producer.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    producer.add_argument("--work-root", type=Path, default=Path("/workspace/production"))
    producer.add_argument("--outgoing-root", type=Path, default=Path("/workspace/outgoing"))
    producer.add_argument("--start", type=int, default=52)
    producer.add_argument("--batch-size", type=int, default=5)
    producer.add_argument("--timeout", type=int, default=7_200)
    producer.add_argument("--transfer-poll", type=float, default=10.0)
    producer.add_argument(
        "--generation-only",
        action="store_true",
        help="transfer canonical initial textured/untextured GLBs and defer all post-processing",
    )
    producer.set_defaults(func=produce)

    retriever = subparsers.add_parser("retrieve")
    retriever.add_argument("--host", required=True)
    retriever.add_argument("--port", type=int, required=True)
    retriever.add_argument("--user", default="root")
    retriever.add_argument("--identity", type=Path, required=True)
    retriever.add_argument("--known-hosts", type=Path, required=True)
    retriever.add_argument("--remote-outgoing", default="/workspace/outgoing")
    retriever.add_argument("--local-root", type=Path, required=True)
    retriever.add_argument("--poll", type=float, default=10.0)
    retriever.add_argument(
        "--transport-timeout",
        type=float,
        default=DEFAULT_TRANSPORT_TIMEOUT_SECONDS,
        help="maximum seconds allowed for one SSH or SCP attempt before retrying",
    )
    retriever.add_argument("--log-file", type=Path)
    retriever.set_defaults(func=retrieve)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    handlers: list[logging.Handler] | None = None
    if getattr(args, "log_file", None):
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers = [logging.FileHandler(args.log_file, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

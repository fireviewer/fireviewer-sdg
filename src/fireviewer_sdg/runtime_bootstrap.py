"""Verify the image-baked NVIDIA runtime, then start the worker."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


ISAAC_SIM_VERSION = "6.0.1.0"
TORCH_VERSION = "2.11.0"
PILLOW_VERSION = "12.2.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
NVIDIA_INDEX = "https://pypi.nvidia.com"
HEARTBEAT_SECONDS = 30
DEFAULT_RUNTIME_ROOT = Path("/opt/fireviewer-runtime") / f"isaacsim-{ISAAC_SIM_VERSION}"


def _volume_root() -> Path:
    return Path(os.getenv("FW_SDG_VOLUME_ROOT", "/workspace/fireviewer-sdg")).resolve()


def _runtime_root() -> Path:
    configured = os.getenv("FW_SDG_RUNTIME_ROOT", "").strip()
    if configured:
        return Path(configured).resolve()
    return DEFAULT_RUNTIME_ROOT.resolve()


def _legacy_volume_runtime() -> Path:
    return _volume_root() / "runtime" / f"isaacsim-{ISAAC_SIM_VERSION}"


def _remove_legacy_volume_runtime(runtime_root: Path) -> None:
    """Remove the former version-specific runtime cache from production storage."""
    legacy = _legacy_volume_runtime().resolve()
    if legacy == runtime_root or not legacy.exists():
        return
    shutil.rmtree(legacy)
    try:
        legacy.parent.rmdir()
    except OSError:
        pass
    print(
        f"fireviewer sdg runtime: removed legacy workspace runtime root={legacy}",
        flush=True,
    )


def _marker(runtime_root: Path) -> Path:
    return runtime_root / ".fireviewer-runtime.json"


def _install_state(runtime_root: Path) -> Path:
    return runtime_root / ".fireviewer-install-state.json"


def _write_install_state(runtime_root: Path, *, phase: str, detail: str = "") -> None:
    runtime_root.mkdir(parents=True, exist_ok=True)
    _install_state(runtime_root).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": phase,
                "detail": detail,
                "updated_at": datetime.now(UTC).isoformat(),
            },
            sort_keys=True,
        )
        + chr(10),
        encoding="utf-8",
    )


def _runtime_ready(runtime_root: Path) -> bool:
    marker_path = _marker(runtime_root)
    python = runtime_root / "bin" / "python"
    if not marker_path.is_file() or not python.is_file():
        return False
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return payload == {
        "schema_version": 1,
        "isaacsim": ISAAC_SIM_VERSION,
        "torch": TORCH_VERSION,
        "pillow": PILLOW_VERSION,
    }


def _run(command: list[str], *, timeout: int) -> None:
    printable = " ".join(command[:4])
    print(f"fireviewer sdg runtime: running {printable}", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command)
    try:
        while True:
            return_code = process.poll()
            elapsed = int(time.monotonic() - started)
            if return_code is not None:
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command)
                print(
                    f"fireviewer sdg runtime: completed elapsed_seconds={elapsed} "
                    f"command={printable}",
                    flush=True,
                )
                return
            if elapsed >= timeout:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise subprocess.TimeoutExpired(command, timeout)
            print(
                f"fireviewer sdg runtime: heartbeat elapsed_seconds={elapsed} "
                f"command={printable}",
                flush=True,
            )
            time.sleep(min(HEARTBEAT_SECONDS, max(1, timeout - elapsed)))
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def _installed_version(python: Path, distribution: str) -> str | None:
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                "from importlib.metadata import version; "
                f"print(version({distribution!r}))",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _core_runtime_installed(python: Path) -> bool:
    isaacsim = _installed_version(python, "isaacsim")
    torch = _installed_version(python, "torch")
    return isaacsim == ISAAC_SIM_VERSION and bool(
        torch and torch.split("+", 1)[0] == TORCH_VERSION
    )


def _install_runtime(runtime_root: Path) -> None:
    parent = runtime_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    python = runtime_root / "bin" / "python"
    _marker(runtime_root).unlink(missing_ok=True)
    try:
        if not python.is_file():
            if runtime_root.exists():
                shutil.rmtree(runtime_root)
            _write_install_state(runtime_root, phase="creating_venv")
            _run([sys.executable, "-m", "venv", str(runtime_root)], timeout=180)
        else:
            print(
                "fireviewer sdg runtime: resuming existing partial environment",
                flush=True,
            )
        pip_base = [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
        ]
        _write_install_state(runtime_root, phase="upgrading_pip")
        _run([*pip_base, "--upgrade", "pip"], timeout=600)
        if _core_runtime_installed(python):
            print(
                "fireviewer sdg runtime: repairing compatible partial runtime "
                "without reinstalling Torch or Isaac Sim",
                flush=True,
            )
        else:
            _write_install_state(runtime_root, phase="installing_torch")
            _run(
                [
                    *pip_base,
                    f"torch=={TORCH_VERSION}",
                    "--index-url",
                    TORCH_INDEX,
                ],
                timeout=3600,
            )
            _write_install_state(runtime_root, phase="installing_isaacsim")
            _run(
                [
                    *pip_base,
                    f"isaacsim[all,extscache]=={ISAAC_SIM_VERSION}",
                    "--extra-index-url",
                    NVIDIA_INDEX,
                ],
                timeout=7200,
            )
        _write_install_state(runtime_root, phase="installing_image_codec")
        _run([*pip_base, f"Pillow=={PILLOW_VERSION}"], timeout=600)
        _write_install_state(runtime_root, phase="checking_dependencies")
        _run([str(python), "-m", "pip", "check"], timeout=600)
        _marker(runtime_root).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "isaacsim": ISAAC_SIM_VERSION,
                    "torch": TORCH_VERSION,
                    "pillow": PILLOW_VERSION,
                },
                sort_keys=True,
            )
            + chr(10),
            encoding="utf-8",
        )
        _write_install_state(runtime_root, phase="ready")
    except BaseException as exc:
        _write_install_state(
            runtime_root,
            phase="failed",
            detail=f"{type(exc).__name__}: {exc}",
        )
        raise

def ensure_runtime() -> Path:
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("the SDG runtime bootstrap requires Linux fcntl") from exc

    runtime_root = _runtime_root()
    _remove_legacy_volume_runtime(runtime_root)
    lock_path = runtime_root.parent / ".install.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not _runtime_ready(runtime_root):
            print(
                "fireviewer sdg runtime: installing pinned Isaac Sim and Replicator "
                f"on container storage root={runtime_root}",
                flush=True,
            )
            _install_runtime(runtime_root)
        else:
            print(
                f"fireviewer sdg runtime: image-baked cache hit root={runtime_root}",
                flush=True,
            )
    return runtime_root


def main() -> None:
    runtime_root = ensure_runtime()
    python = runtime_root / "bin" / "python"
    os.execv(str(python), [str(python), "-m", "fireviewer_sdg.bootstrap"])


if __name__ == "__main__":
    main()

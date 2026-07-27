from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fireviewer_sdg import runtime_bootstrap


class RuntimeBootstrapTests(unittest.TestCase):
    def test_runtime_ready_requires_exact_pinned_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            (root / "bin" / "python").write_text("", encoding="ascii")
            marker = root / ".fireviewer-runtime.json"
            marker.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "isaacsim": runtime_bootstrap.ISAAC_SIM_VERSION,
                        "torch": runtime_bootstrap.TORCH_VERSION,
                        "pillow": runtime_bootstrap.PILLOW_VERSION,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(runtime_bootstrap._runtime_ready(root))
            marker.write_text("{}", encoding="ascii")
            self.assertFalse(runtime_bootstrap._runtime_ready(root))

    def test_runtime_path_defaults_to_container_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {"FW_SDG_VOLUME_ROOT": directory},
                clear=False,
            ):
                os.environ.pop("FW_SDG_RUNTIME_ROOT", None)
                self.assertEqual(
                    runtime_bootstrap._runtime_root(),
                    runtime_bootstrap.DEFAULT_RUNTIME_ROOT.resolve(),
                )

    def test_legacy_workspace_runtime_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            runtime = Path(directory) / "container" / "isaac"
            legacy = (
                volume
                / "runtime"
                / f"isaacsim-{runtime_bootstrap.ISAAC_SIM_VERSION}"
            )
            legacy.mkdir(parents=True)
            (legacy / "partial.whl").write_text("partial", encoding="ascii")
            with mock.patch.dict(
                os.environ,
                {"FW_SDG_VOLUME_ROOT": str(volume)},
                clear=False,
            ):
                runtime_bootstrap._remove_legacy_volume_runtime(runtime.resolve())
            self.assertFalse(legacy.exists())
            self.assertFalse((volume / "runtime").exists())

    def test_install_uses_final_path_and_disables_pip_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"

            def fake_run(command: list[str], *, timeout: int) -> None:
                del timeout
                if command[1:3] == ["-m", "venv"]:
                    (root / "bin").mkdir(parents=True)
                    (root / "bin" / "python").write_text("", encoding="ascii")

            with mock.patch.object(runtime_bootstrap, "_run", side_effect=fake_run) as run:
                runtime_bootstrap._install_runtime(root)

            self.assertTrue(runtime_bootstrap._runtime_ready(root))
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(commands[0][-1], str(root))
            pip_installs = [
                command for command in commands if command[1:4] == ["-m", "pip", "install"]
            ]
            self.assertEqual(len(pip_installs), 4)
            self.assertTrue(
                any(
                    f"Pillow=={runtime_bootstrap.PILLOW_VERSION}" in command
                    for command in pip_installs
                )
            )
            self.assertTrue(all("--no-cache-dir" in command for command in pip_installs))
            self.assertFalse(any(".partial-" in " ".join(command) for command in commands))

    def test_incomplete_environment_is_preserved_and_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            (root / "bin").mkdir(parents=True)
            python = root / "bin" / "python"
            python.write_text("existing", encoding="ascii")

            def fake_run(command: list[str], *, timeout: int) -> None:
                del timeout
                if any("isaacsim[all,extscache]" in item for item in command):
                    raise subprocess.CalledProcessError(1, command)

            with mock.patch.object(runtime_bootstrap, "_run", side_effect=fake_run) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    runtime_bootstrap._install_runtime(root)

            self.assertEqual(python.read_text(encoding="ascii"), "existing")
            commands = [call.args[0] for call in run.call_args_list]
            self.assertFalse(any(command[1:3] == ["-m", "venv"] for command in commands))
            state = json.loads(
                (root / ".fireviewer-install-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["phase"], "failed")
            self.assertFalse((root / ".fireviewer-runtime.json").exists())

    def test_compatible_partial_runtime_skips_large_reinstall(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            (root / "bin").mkdir(parents=True)
            (root / "bin" / "python").write_text("existing", encoding="ascii")
            with mock.patch.object(
                runtime_bootstrap,
                "_core_runtime_installed",
                return_value=True,
            ):
                with mock.patch.object(runtime_bootstrap, "_run") as run:
                    runtime_bootstrap._install_runtime(root)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertFalse(any("torch==" in " ".join(command) for command in commands))
            self.assertFalse(any("isaacsim[" in " ".join(command) for command in commands))
            self.assertTrue(
                any(
                    f"Pillow=={runtime_bootstrap.PILLOW_VERSION}" in command
                    for command in commands
                )
            )
            self.assertTrue(runtime_bootstrap._runtime_ready(root))

if __name__ == "__main__":
    unittest.main()

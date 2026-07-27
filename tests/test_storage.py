from __future__ import annotations

import sys
import unittest
from os import environ
from pathlib import Path
from shutil import _ntuple_diskusage
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg.storage import (  # noqa: E402
    assert_storage_architecture,
    validate_storage_plan,
)


class StorageContractTests(unittest.TestCase):
    def _plan(self) -> dict[str, object]:
        return {
            "kind": "network_volume",
            "mount_point": "/workspace",
            "minimum_total_gb": 1000,
            "reserve_free_gb": 100,
            "estimated_max_case_bytes": 12 * 1024**2,
            "minimum_fire_events": 512,
            "estimated_max_event_input_bytes": 512 * 1024**2,
        }

    def test_network_volume_budget_supports_double_capacity(self) -> None:
        plan = validate_storage_plan(
            {
                "kind": "network_volume",
                "mount_point": "/workspace",
                "minimum_total_gb": 1000,
                "reserve_free_gb": 100,
                "estimated_max_case_bytes": 12 * 1024**2,
                "minimum_fire_events": 512,
                "estimated_max_event_input_bytes": 512 * 1024**2,
            }
        )
        projected = 4 * 8192 * plan["estimated_max_case_bytes"]
        event_inputs = 512 * plan["estimated_max_event_input_bytes"]
        self.assertLess(
            projected + event_inputs + plan["reserve_free_gb"] * 1024**3,
            1000 * 1000**3,
        )

    def test_unknown_or_undersized_storage_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "network_volume or local_windows"):
            validate_storage_plan(
                {
                    "kind": "volume_disk",
                    "mount_point": "/workspace",
                    "minimum_total_gb": 1000,
                    "reserve_free_gb": 100,
                    "estimated_max_case_bytes": 12 * 1024**2,
                    "minimum_fire_events": 512,
                    "estimated_max_event_input_bytes": 512 * 1024**2,
                }
            )
        with self.assertRaisesRegex(ValueError, "1000 GB"):
            validate_storage_plan(
                {
                    "kind": "network_volume",
                    "mount_point": "/workspace",
                    "minimum_total_gb": 200,
                    "reserve_free_gb": 100,
                    "estimated_max_case_bytes": 12 * 1024**2,
                    "minimum_fire_events": 512,
                    "estimated_max_event_input_bytes": 512 * 1024**2,
                }
            )

    @patch("fireviewer_sdg.storage._is_posix", return_value=False)
    @patch(
        "fireviewer_sdg.storage.shutil.disk_usage",
        return_value=_ntuple_diskusage(
            2_000_000_000_000,
            1_400_000_000_000,
            600_000_000_000,
        ),
    )
    def test_local_windows_storage_is_capacity_checked(
        self, _usage: object, _posix: object
    ) -> None:
        plan = self._plan()
        plan.update(
            kind="local_windows",
            mount_point="D:/FVS/workspace",
        )
        result = assert_storage_architecture(
            Path("D:/FVS/workspace/fireviewer-sdg"),
            plan,
        )
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["durability"], "local_windows")
        self.assertFalse(result["export_required"])

    @patch("fireviewer_sdg.storage._is_posix", return_value=False)
    @patch(
        "fireviewer_sdg.storage.shutil.disk_usage",
        return_value=_ntuple_diskusage(
            2_000_000_000_000,
            1_400_000_000_000,
            600_000_000_000,
        ),
    )
    def test_windows_rejects_network_volume_plan(
        self, _usage: object, _posix: object
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "local_windows"):
            assert_storage_architecture(
                Path("D:/FVS/workspace/fireviewer-sdg"),
                self._plan(),
            )

    @patch("fireviewer_sdg.storage._is_posix", return_value=True)
    @patch("fireviewer_sdg.storage._linux_mount", return_value=(Path("/"), "overlay"))
    @patch(
        "fireviewer_sdg.storage.shutil.disk_usage",
        return_value=_ntuple_diskusage(1_200_000_000_000, 200_000_000_000, 1_000_000_000_000),
    )
    def test_explicit_one_tb_ephemeral_mode_is_marked_for_export(
        self, _usage: object, _mount: object, _posix: object
    ) -> None:
        with patch.dict(
            environ,
            {
                "FW_SDG_STORAGE_MODE": "ephemeral",
                "FW_SDG_EPHEMERAL_CAPACITY_GB": "1000",
                "FW_SDG_EPHEMERAL_EXPORT_ACK": "1",
            },
            clear=False,
        ):
            result = assert_storage_architecture(
                Path("/workspace/fireviewer-sdg"), self._plan()
            )
        self.assertEqual(result["durability"], "ephemeral")
        self.assertTrue(result["export_required"])

    @patch("fireviewer_sdg.storage._is_posix", return_value=True)
    @patch("fireviewer_sdg.storage._linux_mount", return_value=(Path("/"), "overlay"))
    @patch(
        "fireviewer_sdg.storage.shutil.disk_usage",
        return_value=_ntuple_diskusage(1_200_000_000_000, 200_000_000_000, 1_000_000_000_000),
    )
    def test_ephemeral_mode_requires_capacity_and_export_ack(
        self, _usage: object, _mount: object, _posix: object
    ) -> None:
        with patch.dict(
            environ,
            {
                "FW_SDG_STORAGE_MODE": "ephemeral",
                "FW_SDG_EPHEMERAL_CAPACITY_GB": "999",
                "FW_SDG_EPHEMERAL_EXPORT_ACK": "1",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "undersized"):
                assert_storage_architecture(
                    Path("/workspace/fireviewer-sdg"), self._plan()
                )
        with patch.dict(
            environ,
            {
                "FW_SDG_STORAGE_MODE": "ephemeral",
                "FW_SDG_EPHEMERAL_CAPACITY_GB": "1000",
                "FW_SDG_EPHEMERAL_EXPORT_ACK": "0",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "EXPORT_ACK"):
                assert_storage_architecture(
                    Path("/workspace/fireviewer-sdg"), self._plan()
                )


if __name__ == "__main__":
    unittest.main()

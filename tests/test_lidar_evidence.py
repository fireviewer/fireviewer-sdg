from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fireviewer_sdg.lidar_evidence import (
    LidarEvidenceError,
    build_receipt,
    verify_receipt,
)


def _summary(*, points: int = 6, srs: object | None = None) -> dict[str, object]:
    return {
        "summary": {
            "bounds": {
                "native": {
                    "bbox": {
                        "minx": 700000.0,
                        "miny": 6600000.0,
                        "minz": 10.0,
                        "maxx": 701000.0,
                        "maxy": 6601000.0,
                        "maxz": 44.0,
                    }
                }
            },
            "dimensions": "X, Y, Z, Intensity, Classification",
            "num_points": points,
            "srs": (
                {"authority": "EPSG:2154", "wkt": 'PROJCRS["RGF93 v1 / Lambert-93"]'}
                if srs is None
                else srs
            ),
        }
    }


def _count_metadata(histogram: dict[int, int] | None = None) -> dict[str, object]:
    values = histogram if histogram is not None else {2: 2, 3: 1, 5: 1, 6: 1, 9: 1}
    return {
        "stages": {
            "filters.stats": {
                "statistic": [
                    {
                        "name": "Classification",
                        "count": sum(values.values()),
                        "bins": {
                            f"{value:.6f}": count
                            for value, count in values.items()
                        },
                        "counts": [
                            f"{value:.6f}/{count}"
                            for value, count in values.items()
                        ],
                    }
                ]
            }
        }
    }


def _completed(payload: object) -> subprocess.CompletedProcess[str]:
    output = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.CompletedProcess([], 0, stdout=output, stderr="")


def _write_pipeline_metadata(
    command: list[str],
    payload: object,
) -> subprocess.CompletedProcess[str]:
    metadata_argument = next(
        (value for value in command if value.startswith("--metadata=")),
        None,
    )
    if metadata_argument is None:
        raise AssertionError(f"PDAL pipeline has no metadata output: {command}")
    Path(metadata_argument.split("=", 1)[1]).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return _completed("")


class LidarEvidenceTests(unittest.TestCase):
    def _pdal_side_effect(
        self,
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if command[1:] == ["--version"]:
            return _completed("pdal 2.10.2 (Release)\n")
        if "--summary" in command:
            return _completed(_summary())
        if command[1:3] == ["pipeline", "--stdin"]:
            pipeline = json.loads(str(kwargs["input"]))
            stage = pipeline["pipeline"][1]
            if stage != {
                "type": "filters.stats",
                "dimensions": "Classification",
                "count": "Classification",
            }:
                raise AssertionError(f"unexpected count stage: {stage}")
            if "--stream" not in command:
                raise AssertionError(f"PDAL count pipeline is not streaming: {command}")
            return _write_pipeline_metadata(command, _count_metadata())
        raise AssertionError(f"unexpected PDAL command: {command}")

    def test_build_is_canonical_deterministic_and_inventories_every_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lidar"
            root.mkdir()
            second = root / "b.copc.laz"
            first = root / "a.laz"
            second.write_bytes(b"copc")
            first.write_bytes(b"laz")
            receipt_path = Path(directory) / "lidar-evidence.json"
            with patch(
                "fireviewer_sdg.lidar_evidence.subprocess.run",
                side_effect=self._pdal_side_effect,
            ) as run:
                receipt = build_receipt(
                    source_root=root,
                    output_path=receipt_path,
                    required_classes=(2, 5, 6),
                    pdal_bin="pdal-test",
                )
                first_bytes = receipt_path.read_bytes()
                build_receipt(
                    source_root=root,
                    output_path=receipt_path,
                    required_classes=(6, 2, 5, 5),
                    pdal_bin="pdal-test",
                )
            self.assertEqual(first_bytes, receipt_path.read_bytes())
            self.assertEqual(
                first_bytes,
                (
                    json.dumps(
                        receipt,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8"),
            )
            self.assertEqual(
                [entry["path"] for entry in receipt["sources"]],
                ["a.laz", "b.copc.laz"],
            )
            self.assertEqual(
                [entry["format"] for entry in receipt["sources"]],
                ["laz", "copc_laz"],
            )
            self.assertEqual(receipt["summary"]["source_count"], 2)
            self.assertEqual(receipt["summary"]["total_points"], 12)
            self.assertEqual(receipt["summary"]["classification_histogram"]["2"], 4)
            self.assertEqual(
                receipt["sources"][0]["sha256"],
                hashlib.sha256(b"laz").hexdigest(),
            )
            self.assertEqual(receipt["pdal"]["version"], "pdal 2.10.2 (Release)")
            self.assertEqual(run.call_count, 10)

    def test_pdal_2102_uses_exact_count_metadata_not_scalar_enumeration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lidar"
            root.mkdir()
            (root / "tile.laz").write_bytes(b"lidar")
            remote_histogram = {1: 7, 2: 31, 5: 13, 6: 1}

            def side_effect(
                command: list[str],
                **kwargs: object,
            ) -> subprocess.CompletedProcess[str]:
                if command[1:] == ["--version"]:
                    return _completed("pdal 2.10.2 (git-version: da2cd8)")
                if "--summary" in command:
                    return _completed(_summary(points=52))
                if "--stats" in command or "--enumerate=Classification" in command:
                    raise AssertionError(
                        "PDAL 2.10.2 scalar enumeration must not be used"
                    )
                if command[1:3] == ["pipeline", "--stdin"]:
                    pipeline = json.loads(str(kwargs["input"]))
                    self.assertEqual(
                        pipeline["pipeline"][1]["count"],
                        "Classification",
                    )
                    return _write_pipeline_metadata(
                        command,
                        _count_metadata(remote_histogram),
                    )
                raise AssertionError(f"unexpected PDAL command: {command}")

            with patch(
                "fireviewer_sdg.lidar_evidence.subprocess.run",
                side_effect=side_effect,
            ):
                receipt = build_receipt(
                    source_root=root,
                    output_path=Path(directory) / "receipt.json",
                    required_classes=(2, 5, 6),
                    pdal_bin="pdal-test",
                )
            self.assertEqual(
                receipt["sources"][0]["classification_histogram"],
                {"1": 7, "2": 31, "5": 13, "6": 1},
            )
            self.assertEqual(receipt["summary"]["total_points"], 52)

    def test_verify_reprobes_and_rejects_a_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lidar"
            root.mkdir()
            source = root / "tile.laz"
            source.write_bytes(b"stable")
            receipt_path = Path(directory) / "receipt.json"
            with patch(
                "fireviewer_sdg.lidar_evidence.subprocess.run",
                side_effect=self._pdal_side_effect,
            ) as run:
                expected = build_receipt(
                    source_root=root,
                    output_path=receipt_path,
                    pdal_bin="pdal-test",
                )
                verified = verify_receipt(
                    receipt_path=receipt_path,
                    source_root=root,
                    pdal_bin="pdal-test",
                )
            self.assertEqual(verified, expected)
            self.assertEqual(run.call_count, 6)
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(LidarEvidenceError, "size changed"):
                verify_receipt(
                    receipt_path=receipt_path,
                    source_root=root,
                    pdal_bin="pdal-test",
                )

    def test_missing_source_fails_before_pdal_is_executed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lidar"
            root.mkdir()
            with patch("fireviewer_sdg.lidar_evidence.subprocess.run") as run:
                with self.assertRaisesRegex(LidarEvidenceError, "source is absent"):
                    build_receipt(
                        source_root=root,
                        output_path=Path(directory) / "receipt.json",
                        sources=(Path("absent.laz"),),
                        pdal_bin="pdal-test",
                    )
            run.assert_not_called()

    def test_external_pdal_binary_is_taken_from_the_contract_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lidar"
            root.mkdir()
            (root / "tile.laz").write_bytes(b"lidar")
            with (
                patch.dict(os.environ, {"FW_SDG_PDAL_BIN": "pdal-from-env"}),
                patch(
                    "fireviewer_sdg.lidar_evidence.subprocess.run",
                    side_effect=self._pdal_side_effect,
                ) as run,
            ):
                build_receipt(
                    source_root=root,
                    output_path=Path(directory) / "receipt.json",
                )
            self.assertTrue(run.call_args_list)
            self.assertTrue(
                all(call.args[0][0] == "pdal-from-env" for call in run.call_args_list)
            )

    def test_missing_srs_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lidar"
            root.mkdir()
            (root / "tile.laz").write_bytes(b"lidar")

            def side_effect(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                if command[1:] == ["--version"]:
                    return _completed("pdal 2.10.2")
                if "--summary" in command:
                    return _completed(_summary(srs={}))
                return _write_pipeline_metadata(command, _count_metadata())

            with patch(
                "fireviewer_sdg.lidar_evidence.subprocess.run",
                side_effect=side_effect,
            ):
                with self.assertRaisesRegex(LidarEvidenceError, "no spatial reference"):
                    build_receipt(
                        source_root=root,
                        output_path=Path(directory) / "receipt.json",
                        pdal_bin="pdal-test",
                    )

    def test_missing_or_incomplete_classification_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lidar"
            root.mkdir()
            (root / "tile.laz").write_bytes(b"lidar")

            def missing_statistic(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                if command[1:] == ["--version"]:
                    return _completed("pdal 2.10.2")
                if "--summary" in command:
                    return _completed(_summary())
                return _write_pipeline_metadata(
                    command,
                    {"stages": {"filters.stats": {"statistic": []}}},
                )

            with patch(
                "fireviewer_sdg.lidar_evidence.subprocess.run",
                side_effect=missing_statistic,
            ):
                with self.assertRaisesRegex(
                    LidarEvidenceError, "no Classification statistic"
                ):
                    build_receipt(
                        source_root=root,
                        output_path=Path(directory) / "receipt.json",
                        pdal_bin="pdal-test",
                    )

            def missing_required_class(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                if command[1:] == ["--version"]:
                    return _completed("pdal 2.10.2")
                if "--summary" in command:
                    return _completed(_summary(points=6))
                return _write_pipeline_metadata(
                    command,
                    _count_metadata({2: 6}),
                )

            with patch(
                "fireviewer_sdg.lidar_evidence.subprocess.run",
                side_effect=missing_required_class,
            ):
                with self.assertRaisesRegex(
                    LidarEvidenceError, "missing required classifications: 5"
                ):
                    build_receipt(
                        source_root=root,
                        output_path=Path(directory) / "receipt.json",
                        required_classes=(2, 5),
                        pdal_bin="pdal-test",
                    )

    def test_worker_count_is_bounded_and_parallel_output_stays_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lidar"
            root.mkdir()
            for name in ("d.laz", "b.laz", "a.laz", "c.laz"):
                (root / name).write_bytes(name.encode("ascii"))
            barrier = threading.Barrier(3)
            active = 0
            peak_active = 0
            lock = threading.Lock()

            def probe(
                source: Path,
                *,
                relative_path: str,
                pdal_bin: str,
            ) -> dict[str, object]:
                nonlocal active, peak_active
                self.assertEqual(pdal_bin, "pdal-test")
                with lock:
                    active += 1
                    peak_active = max(peak_active, active)
                try:
                    if relative_path in {"a.laz", "b.laz", "c.laz"}:
                        barrier.wait(timeout=5)
                finally:
                    with lock:
                        active -= 1
                return {
                    "path": relative_path,
                    "format": "laz",
                    "bytes": source.stat().st_size,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "point_count": 1,
                    "bounds": {
                        "minx": 0.0,
                        "miny": 0.0,
                        "minz": 0.0,
                        "maxx": 1.0,
                        "maxy": 1.0,
                        "maxz": 1.0,
                    },
                    "dimensions": ["X", "Y", "Z", "Classification"],
                    "srs": {"authority": "EPSG:2154"},
                    "classification_histogram": {"2": 1},
                }

            def version_only(
                command: list[str],
                **_kwargs: object,
            ) -> subprocess.CompletedProcess[str]:
                if command[1:] != ["--version"]:
                    raise AssertionError(f"unexpected PDAL command: {command}")
                return _completed("pdal 2.10.2")

            with (
                patch(
                    "fireviewer_sdg.lidar_evidence.subprocess.run",
                    side_effect=version_only,
                ),
                patch(
                    "fireviewer_sdg.lidar_evidence.probe_source",
                    side_effect=probe,
                ),
            ):
                receipt = build_receipt(
                    source_root=root,
                    output_path=Path(directory) / "receipt.json",
                    pdal_bin="pdal-test",
                    workers=3,
                )
            self.assertEqual(peak_active, 3)
            self.assertEqual(
                [source["path"] for source in receipt["sources"]],
                ["a.laz", "b.laz", "c.laz", "d.laz"],
            )

            for invalid in (0, 33):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(
                        LidarEvidenceError,
                        "worker count must be an integer between 1 and 32",
                    ):
                        build_receipt(
                            source_root=root,
                            output_path=Path(directory) / "invalid.json",
                            pdal_bin="pdal-test",
                            workers=invalid,
                        )

    def test_noncanonical_or_tampered_receipt_is_rejected_without_pdal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lidar"
            root.mkdir()
            (root / "tile.laz").write_bytes(b"lidar")
            receipt_path = Path(directory) / "receipt.json"
            with patch(
                "fireviewer_sdg.lidar_evidence.subprocess.run",
                side_effect=self._pdal_side_effect,
            ):
                receipt = build_receipt(
                    source_root=root,
                    output_path=receipt_path,
                    pdal_bin="pdal-test",
                )
            receipt["summary"]["total_points"] += 1
            receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            with patch("fireviewer_sdg.lidar_evidence.subprocess.run") as run:
                with self.assertRaisesRegex(LidarEvidenceError, "not canonical"):
                    verify_receipt(
                        receipt_path=receipt_path,
                        source_root=root,
                        pdal_bin="pdal-test",
                    )
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

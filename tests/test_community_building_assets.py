from __future__ import annotations

import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg import campaign_asset_bundle  # noqa: E402
from fireviewer_sdg import community_building_assets as subject  # noqa: E402
from fireviewer_sdg.simready_assets import (  # noqa: E402
    MANIFEST_PROFILE,
    MANIFEST_SCHEMA_VERSION,
    PHOTOREAL_FAMILY_MINIMUMS,
    PHOTOREAL_LIBRARY_POLICY,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_glb(path: Path, *, uid: str) -> None:
    document = {
        "asset": {"version": "2.0", "generator": f"fixture-{uid}"},
        "meshes": [
            {
                "name": uid,
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [{"name": f"material-{uid}"}],
        "textures": [{"source": 0}],
        "images": [{"uri": f"{uid}.png"}],
    }
    payload = json.dumps(document, sort_keys=True).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    length = 12 + 8 + len(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, length)
        + struct.pack("<II", len(payload), 0x4E4F534A)
        + payload
    )


def _write_manifest(volume: Path) -> Path:
    environment: dict[str, dict[str, list[dict[str, object]]]] = {
        "vegetation": {},
        "buildings": {},
    }
    for kind, families in PHOTOREAL_FAMILY_MINIMUMS.items():
        for family, minimum in families.items():
            count = (
                0
                if kind == "buildings"
                and family in subject.COMMUNITY_BUILDING_UIDS
                else minimum
            )
            environment[kind][family] = [
                {"asset_id": f"{kind}.{family}:existing-{index}"}
                for index in range(count)
            ]
    manifest = volume / "input" / "simready-assets-hd-v3.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "profile": MANIFEST_PROFILE,
                "library_policy": PHOTOREAL_LIBRARY_POLICY,
                "family_minimums": PHOTOREAL_FAMILY_MINIMUMS,
                "discovery": {
                    "mode": "materialized_photoreal_asset_library_v3",
                    "missing_environment": [
                        "buildings.agricultural",
                        "buildings.industrial",
                        "buildings.annex",
                    ],
                },
                "environment": environment,
                "actors": {},
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _write_sources(volume: Path) -> tuple[Path, Path]:
    source_root = volume / "input" / "objaverse-buildings"
    records: dict[str, dict[str, str]] = {}
    for uid, family in subject.UID_TO_FAMILY.items():
        _write_glb(source_root / f"{uid}.glb", uid=uid)
        records[uid] = {
            "uid": uid,
            "family": family,
            "name": subject.COMMUNITY_BUILDING_TITLES[uid],
            "creator": f"Creator {uid[:6]}",
            "source_uri": f"https://sketchfab.com/3d-models/{uid}",
            "license_id": "CC-BY-4.0",
            "license_uri": "https://creativecommons.org/licenses/by/4.0/",
            "local_path": f"{uid}.glb",
        }
    metadata = source_root / "metadata.json"
    metadata.write_text(json.dumps({"assets": records}), encoding="utf-8")
    return source_root, metadata


def _fake_converter(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "#usda 1.0\n"
        '(\n    defaultPrim = "Building"\n    metersPerUnit = 1\n'
        '    upAxis = "Z"\n)\n'
        f'def Xform "Building" {{ string source = "{source.stem}" }}\n',
        encoding="utf-8",
    )
    (destination.parent / "albedo.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + source.stem.encode("ascii")
    )


def _fake_inspector(
    source_usd: Path,
    asset_root: Path,
    family: str,
) -> dict[str, object]:
    texture = asset_root / "albedo.png"
    uid = source_usd.parent.name
    dimensions = {
        "agricultural": [24.0, 13.0, 8.0],
        "industrial": [40.0, 22.0, 12.0],
        "annex": [8.0, 6.0, 4.0],
    }[family]
    anchor = [2.0, -3.0, 0.25]
    if uid in {
        "aab826498d5b42fabb46e8c53f69a66c",
        "70154ba65c7d4ba0b5f081fd0eb35f68",
    }:
        dimensions = [value / 0.0254 for value in dimensions]
        anchor = [value / 0.0254 for value in anchor]
    return {
        "source_meters_per_unit": 1.0,
        "source_up_axis": "Z",
        "default_prim": "/Building",
        "dimensions_m": dimensions,
        "anchor_m": anchor,
        "mesh_prim_count": 2,
        "geometry_point_count": 1800,
        "material_prim_count": 2,
        "bound_material_prim_count": 2,
        "dependency_paths": [source_usd, texture],
        "texture_paths": [texture],
        "unresolved_dependencies": [],
    }


class CommunityBuildingAssetTests(unittest.TestCase):
    def test_reviewed_mapping_excludes_unavailable_and_rejected_assets(self) -> None:
        self.assertEqual(
            subject.COMMUNITY_BUILDING_UIDS,
            {
                "agricultural": (
                    "bcdac4f85bb940aaa189f22f869ed11f",
                    "aab826498d5b42fabb46e8c53f69a66c",
                    "3447582ca28743d785d08f04ea710469",
                ),
                "industrial": (
                    "0e93ab7b05944087b4a19fb7262877fa",
                    "3ebb228ec3ad441c850e0fd298223348",
                ),
                "annex": (
                    "d790f2d3d93d4ad09c8c40f5331e5813",
                    "70154ba65c7d4ba0b5f081fd0eb35f68",
                    "cf8074f44da541189bb4936b05f4f434",
                ),
            },
        )
        self.assertEqual(
            subject.COMMUNITY_BUILDING_TITLES,
            {
                "bcdac4f85bb940aaa189f22f869ed11f": "Cow Shed",
                "aab826498d5b42fabb46e8c53f69a66c": "Abandoned Barn",
                "3447582ca28743d785d08f04ea710469": (
                    "Farm Out Building complex"
                ),
                "0e93ab7b05944087b4a19fb7262877fa": (
                    "Industrial Building drone"
                ),
                "3ebb228ec3ad441c850e0fd298223348": (
                    "sand-lime brick factory"
                ),
                "d790f2d3d93d4ad09c8c40f5331e5813": (
                    "Fergus Falls cottage"
                ),
                "70154ba65c7d4ba0b5f081fd0eb35f68": (
                    "Backyard Utility Shed"
                ),
                "cf8074f44da541189bb4936b05f4f434": (
                    "Wooden shed/open rural structure"
                ),
            },
        )
        rejected = {
            "fa420e6ca2c44b46bc4b6c88deae2b4b",
            "eb27ba0e85194d04936721b359a2eefd",
            "6fbb2f4999e84adc86687e6833322ff2",
            "ab9bbd10fed64b21bb60a134834a7689",
            "761c0879b18041e1a42bc10de88d2228",
        }
        self.assertTrue(rejected.isdisjoint(subject.UID_TO_FAMILY))
        self.assertNotIn(
            "a cold shed",
            {title.casefold() for title in subject.COMMUNITY_BUILDING_TITLES.values()},
        )

    def test_installs_reviewed_assets_atomically_with_family_minimums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            volume.mkdir()
            manifest = _write_manifest(volume)
            source_root, metadata = _write_sources(volume)

            result = subject.install_community_building_assets(
                volume_root=volume,
                manifest_path=manifest,
                source_root=source_root,
                metadata_path=metadata,
                converter=_fake_converter,
                inspector=_fake_inspector,
            )

            self.assertEqual(result["state"], subject.INSTALL_STATE)
            self.assertEqual(result["asset_count"], 8)
            self.assertEqual(
                result["family_counts"],
                {"agricultural": 3, "industrial": 2, "annex": 3},
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["discovery"]["missing_environment"], [])
            supplement = payload["discovery"]["community_building_supplement"]
            self.assertEqual(supplement["objaverse_uids"], sorted(subject.UID_TO_FAMILY))
            entries = [
                entry
                for family in subject.COMMUNITY_BUILDING_UIDS
                for entry in payload["environment"]["buildings"][family]
            ]
            self.assertEqual(len(entries), 8)
            self.assertEqual(
                len({entry["provider_hash"] for entry in entries}),
                8,
            )
            for entry in entries:
                uid = entry["identity"]["objaverse_uid"]
                family = subject.UID_TO_FAMILY[uid]
                self.assertEqual(entry["family"], f"buildings.{family}")
                self.assertEqual(entry["license_id"], "CC-BY-4.0")
                self.assertEqual(
                    entry["provenance"]["reviewed_selection_name"],
                    subject.COMMUNITY_BUILDING_TITLES[uid],
                )
                self.assertTrue(entry["attribution"]["creator"])
                self.assertEqual(entry["quality_validation"], "native_metadata_passed")
                self.assertEqual(entry["materials"]["state"], "passed")
                self.assertEqual(entry["lod"]["strategy"], "source_default_only")
                strategy, selected = campaign_asset_bundle._select_native_levels(
                    entry=entry,
                    label=entry["asset_id"],
                )
                self.assertEqual(strategy, "scene_optimizer_decimateMeshes")
                self.assertEqual(selected["HERO"], "/Building")
                self.assertEqual(entry["source_meters_per_unit"], 1.0)
                self.assertEqual(entry["source_up_axis"], "Z")
                expected_scale = (
                    0.0254
                    if uid
                    in {
                        "aab826498d5b42fabb46e8c53f69a66c",
                        "70154ba65c7d4ba0b5f081fd0eb35f68",
                    }
                    else 1.0
                )
                self.assertEqual(
                    entry["provider_unit_scale_correction"],
                    expected_scale,
                )
                self.assertEqual(
                    entry["provenance"]["unit_scale_correction"],
                    expected_scale,
                )
                wrapper = manifest.parent / entry["path"]
                self.assertEqual(_sha256(wrapper), entry["sha256"])
                source_usd = manifest.parent / entry["source_cache_path"]
                expected_reference = (
                    Path(
                        __import__("os").path.relpath(source_usd, wrapper.parent)
                    )
                    .as_posix()
                )
                self.assertIn(f"@{expected_reference}@", wrapper.read_text())
                expected_wrapper_scale = expected_scale
                self.assertIn(
                    f"float3 xformOp:scale = ({expected_wrapper_scale:.12g}, "
                    f"{expected_wrapper_scale:.12g}, "
                    f"{expected_wrapper_scale:.12g})",
                    wrapper.read_text(),
                )
                locked = entry["materialized_files"]
                self.assertEqual(
                    hashlib.sha256(
                        json.dumps(locked, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                    entry["content_lock_sha256"],
                )
                self.assertTrue(
                    any(
                        Path(record["path"]).suffix.casefold() == ".png"
                        for record in locked
                    )
                )

    def test_failed_material_validation_does_not_update_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            volume.mkdir()
            manifest = _write_manifest(volume)
            source_root, metadata = _write_sources(volume)
            original = manifest.read_bytes()
            output = volume / "input" / "community-building-assets" / "forced"

            def bad_inspector(
                source_usd: Path,
                asset_root: Path,
                family: str,
            ) -> dict[str, object]:
                result = _fake_inspector(source_usd, asset_root, family)
                result["bound_material_prim_count"] = 0
                return result

            with self.assertRaisesRegex(
                subject.CommunityBuildingAssetError,
                "mesh/material validation",
            ):
                subject.install_community_building_assets(
                    volume_root=volume,
                    manifest_path=manifest,
                    source_root=source_root,
                    metadata_path=metadata,
                    output_root=output,
                    converter=_fake_converter,
                    inspector=bad_inspector,
                )
            self.assertEqual(manifest.read_bytes(), original)
            self.assertFalse(output.exists())

    def test_rejects_non_cc_by_metadata_before_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            volume.mkdir()
            manifest = _write_manifest(volume)
            source_root, metadata = _write_sources(volume)
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            uid = next(iter(subject.UID_TO_FAMILY))
            payload["assets"][uid]["license_id"] = "CC-BY-SA-4.0"
            payload["assets"][uid][
                "license_uri"
            ] = "https://creativecommons.org/licenses/by-sa/4.0/"
            metadata.write_text(json.dumps(payload), encoding="utf-8")
            calls: list[Path] = []

            with self.assertRaisesRegex(
                subject.CommunityBuildingAssetError,
                "explicit CC BY",
            ):
                subject.install_community_building_assets(
                    volume_root=volume,
                    manifest_path=manifest,
                    source_root=source_root,
                    metadata_path=metadata,
                    converter=lambda source, destination: calls.append(source),
                    inspector=_fake_inspector,
                )
            self.assertEqual(calls, [])

    def test_cli_preserves_failure_exit_code_with_fast_kit_shutdown(self) -> None:
        runtime = mock.Mock()
        failure = subject.CommunityBuildingAssetError("invalid native bounds")
        stderr = io.StringIO()
        with (
            mock.patch.object(subject, "_start_isaac_runtime", return_value=runtime),
            mock.patch.object(
                subject,
                "install_community_building_assets",
                side_effect=failure,
            ),
            mock.patch.object(
                subject.os,
                "_exit",
                side_effect=SystemExit(1),
            ) as terminate,
            redirect_stderr(stderr),
            self.assertRaisesRegex(SystemExit, "1"),
        ):
            subject.main(
                [
                    "--volume-root",
                    "volume",
                    "--manifest",
                    "manifest.json",
                    "--source-root",
                    "sources",
                    "--metadata",
                    "metadata.json",
                ]
            )
        terminate.assert_called_once_with(1)
        runtime.close.assert_not_called()
        self.assertIn(
            "COMMUNITY_BUILDING_ASSETS_FAILED: invalid native bounds",
            stderr.getvalue(),
        )

    def test_complete_install_is_idempotent_and_revalidates_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            volume.mkdir()
            manifest = _write_manifest(volume)
            source_root, metadata = _write_sources(volume)
            first = subject.install_community_building_assets(
                volume_root=volume,
                manifest_path=manifest,
                source_root=source_root,
                metadata_path=metadata,
                converter=_fake_converter,
                inspector=_fake_inspector,
            )
            manifest_sha = _sha256(manifest)
            calls: list[Path] = []
            second = subject.install_community_building_assets(
                volume_root=volume,
                manifest_path=manifest,
                source_root=source_root,
                metadata_path=metadata,
                converter=lambda source, destination: calls.append(source),
                inspector=_fake_inspector,
            )
            self.assertEqual(calls, [])
            self.assertTrue(second["reused"])
            self.assertFalse(second["reattached"])
            self.assertFalse(second["manifest_changed"])
            self.assertEqual(_sha256(manifest), manifest_sha)
            self.assertEqual(second["bundle_id"], first["bundle_id"])

    def test_reuse_reattaches_to_reprovisioned_manifest_without_conversion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            volume.mkdir()
            manifest = _write_manifest(volume)
            source_root, metadata = _write_sources(volume)
            first = subject.install_community_building_assets(
                volume_root=volume,
                manifest_path=manifest,
                source_root=source_root,
                metadata_path=metadata,
                converter=_fake_converter,
                inspector=_fake_inspector,
            )
            receipt = Path(first["receipt"])
            receipt_sha = _sha256(receipt)

            _write_manifest(volume)
            calls: list[Path] = []
            second = subject.install_community_building_assets(
                volume_root=volume,
                manifest_path=manifest,
                source_root=source_root,
                metadata_path=metadata,
                converter=lambda source, destination: calls.append(source),
                inspector=_fake_inspector,
            )

            self.assertEqual(calls, [])
            self.assertTrue(second["reused"])
            self.assertTrue(second["reattached"])
            self.assertTrue(second["manifest_changed"])
            self.assertEqual(second["receipt_sha256"], receipt_sha)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["discovery"]["missing_environment"], [])
            self.assertEqual(
                {
                    family: len(payload["environment"]["buildings"][family])
                    for family in subject.COMMUNITY_BUILDING_UIDS
                },
                {"agricultural": 3, "industrial": 2, "annex": 3},
            )
            report = json.loads(
                manifest.with_name(
                    "community-building-install-report.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(report["manifest_sha256"], _sha256(manifest))
            self.assertTrue(report["reattached"])

    def test_reuse_rejects_drifted_source_locks_before_manifest_attachment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            volume.mkdir()
            manifest = _write_manifest(volume)
            source_root, metadata = _write_sources(volume)
            first = subject.install_community_building_assets(
                volume_root=volume,
                manifest_path=manifest,
                source_root=source_root,
                metadata_path=metadata,
                converter=_fake_converter,
                inspector=_fake_inspector,
            )
            receipt_path = Path(first["receipt"])
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["source_locks"][0]["metadata_sha256"] = "0" * 64
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            _write_manifest(volume)
            original = manifest.read_bytes()
            calls: list[Path] = []

            with self.assertRaisesRegex(
                subject.CommunityBuildingAssetError,
                "source locks drifted",
            ):
                subject.install_community_building_assets(
                    volume_root=volume,
                    manifest_path=manifest,
                    source_root=source_root,
                    metadata_path=metadata,
                    converter=lambda source, destination: calls.append(source),
                    inspector=_fake_inspector,
                )

            self.assertEqual(calls, [])
            self.assertEqual(manifest.read_bytes(), original)

    def test_inches_correction_is_strictly_allowlisted(self) -> None:
        self.assertEqual(
            subject._UNIT_SCALE_CORRECTIONS,
            {
                "aab826498d5b42fabb46e8c53f69a66c": 0.0254,
                "70154ba65c7d4ba0b5f081fd0eb35f68": 0.0254,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_usd = root / "source.usd"
            texture = root / "albedo.png"
            source_usd.write_text("#usda 1.0\n", encoding="utf-8")
            texture.write_bytes(b"texture")
            inspection = {
                "source_meters_per_unit": 0.01,
                "source_up_axis": "Y",
                "default_prim": "/World",
                "dimensions_m": [102.70820343, 155.40141338, 101.4014018],
                "anchor_m": [-47.99999985, 62.29929678, 0.0],
                "mesh_prim_count": 9,
                "geometry_point_count": 13198,
                "material_prim_count": 9,
                "bound_material_prim_count": 9,
                "dependency_paths": [source_usd, texture],
                "texture_paths": [texture],
                "unresolved_dependencies": [],
            }

            corrected = subject._validate_inspection(
                inspection,
                uid="70154ba65c7d4ba0b5f081fd0eb35f68",
                family="annex",
                asset_root=root,
                source_usd=source_usd,
            )
            self.assertEqual(corrected["unit_scale_correction"], 0.0254)
            self.assertAlmostEqual(corrected["dimensions_m"][0], 2.608789, 5)
            self.assertAlmostEqual(corrected["dimensions_m"][1], 3.947196, 5)
            self.assertAlmostEqual(corrected["dimensions_m"][2], 2.575596, 5)
            self.assertAlmostEqual(corrected["anchor_m"][0], -1.2192, 5)

            barn_inspection = dict(inspection)
            barn_inspection["dimensions_m"] = [
                220.68564577,
                226.02095131,
                187.75793333,
            ]
            corrected_barn = subject._validate_inspection(
                barn_inspection,
                uid="aab826498d5b42fabb46e8c53f69a66c",
                family="agricultural",
                asset_root=root,
                source_usd=source_usd,
            )
            self.assertAlmostEqual(
                corrected_barn["dimensions_m"][0],
                5.605416,
                5,
            )
            self.assertAlmostEqual(
                corrected_barn["dimensions_m"][2],
                4.769052,
                5,
            )

            with self.assertRaisesRegex(
                subject.CommunityBuildingAssetError,
                "dimensions are implausible",
            ):
                subject._validate_inspection(
                    inspection,
                    uid="cf8074f44da541189bb4936b05f4f434",
                    family="annex",
                    asset_root=root,
                    source_usd=source_usd,
                )


if __name__ == "__main__":
    unittest.main()

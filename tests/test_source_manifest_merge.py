from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg.asset_bundle import (  # noqa: E402
    INSTALL_MARKER,
    REQUIRED_ACTOR_CLASSES,
)
from fireviewer_sdg.community_building_assets import (  # noqa: E402
    COMMUNITY_BUILDING_GLB_LOCKS,
    COMMUNITY_BUILDING_TITLES,
    INSTALL_SCHEMA_VERSION as COMMUNITY_INSTALL_SCHEMA_VERSION,
    INSTALL_STATE as COMMUNITY_INSTALL_STATE,
    UID_TO_FAMILY,
)
from fireviewer_sdg.simready_assets import (  # noqa: E402
    MANIFEST_PROFILE,
    MANIFEST_SCHEMA_VERSION,
    PHOTOREAL_FAMILY_MINIMUMS,
    PHOTOREAL_LIBRARY_POLICY,
)
from fireviewer_sdg import source_manifest_merge as subject  # noqa: E402
from fireviewer_sdg import campaign_asset_bundle  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _record(path: Path, *, volume: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(volume).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _entry(
    *,
    volume: Path,
    manifest_parent: Path,
    family: str,
    stem: str,
    source_basename: str | None = None,
    wrapper_identity: str | None = None,
    objaverse_uid: str | None = None,
) -> dict[str, object]:
    root_name = manifest_parent.name.replace(".", "-")
    source = (
        manifest_parent
        / "sources"
        / stem
        / (source_basename or f"{stem}.usd")
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "#usda 1.0\n"
        f'def Xform "Source" {{ string identity = "{stem}" }}\n',
        encoding="utf-8",
    )
    wrapper = manifest_parent / "wrappers" / family.replace(".", "/") / f"{stem}.usda"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "#usda 1.0\n"
        f'def Xform "Asset" {{ string identity = "{wrapper_identity or stem}" }}\n',
        encoding="utf-8",
    )
    dependency = _record(source, volume=volume)
    identity: dict[str, str] = {
        "source_name": stem,
        "source_identity": f"fixture:{root_name}:{family}:{stem}",
    }
    if objaverse_uid is not None:
        identity["objaverse_uid"] = objaverse_uid
    entry: dict[str, object] = {
        "asset_id": (
            f"{family}:objaverse-{objaverse_uid}"
            if objaverse_uid is not None
            else f"{family}:{root_name}-{stem}"
        ),
        "family": family,
        "identity": identity,
        "path": wrapper.relative_to(manifest_parent).as_posix(),
        "sha256": _sha256(wrapper),
        "source_cache_path": source.relative_to(manifest_parent).as_posix(),
        "materialized_files": [dependency],
        "content_lock_sha256": hashlib.sha256(
            json.dumps([dependency], sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "quality_validation": "native_metadata_passed",
        "anchor_validation": {"state": "passed"},
        "materials": {"state": "passed"},
        "lod": {
            "state": "passed",
            "strategy": "source_default_only",
            "levels": ["/Source"],
            "level_count": 1,
        },
        "provenance": {
            "provider": "Objaverse" if objaverse_uid else "Fixture provider",
            "source_uri": (
                f"https://sketchfab.com/3d-models/{objaverse_uid}"
                if objaverse_uid
                else f"https://example.invalid/assets/{stem}"
            ),
        },
        "license": {
            "id": "CC-BY-4.0",
            "uri": "https://creativecommons.org/licenses/by/4.0/",
        },
    }
    if objaverse_uid is not None:
        source_lock = COMMUNITY_BUILDING_GLB_LOCKS[objaverse_uid]
        source_uri = f"https://sketchfab.com/3d-models/{objaverse_uid}"
        metadata_sha = hashlib.sha256(
            f"metadata:{objaverse_uid}".encode("utf-8")
        ).hexdigest()
        entry["source_glb_sha256"] = source_lock["sha256"]
        entry["source_glb_size_bytes"] = source_lock["size_bytes"]
        entry["source_uri"] = source_uri
        entry["provenance"]["source_glb_sha256"] = source_lock["sha256"]
        entry["provenance"]["source_glb_size_bytes"] = source_lock[
            "size_bytes"
        ]
        entry["provenance"]["source_uri"] = source_uri
        entry["provenance"]["raw_metadata_sha256"] = metadata_sha
        entry["provenance"]["reviewed_selection_name"] = (
            COMMUNITY_BUILDING_TITLES[objaverse_uid]
        )
        entry["attribution"] = {
            "creator": f"Creator {objaverse_uid[:8]}",
            "notice": f"Fixture attribution for {objaverse_uid}",
            "source_uri": source_uri,
        }
    return entry


class _Fixture:
    def __init__(
        self,
        root: Path,
        *,
        include_kasa: bool = True,
        missing_actor: bool = False,
        populate_reserved_official: bool = False,
    ) -> None:
        self.volume = root / "volume"
        self.volume.mkdir(parents=True)
        self.bundle_sha = "a" * 64
        self.curated_root = (
            self.volume / "input" / "asset-bundles" / self.bundle_sha
        )
        self.curated_root.mkdir(parents=True)
        self.curated = self.curated_root / "manifest-v3.json"
        self.official = self.volume / "input" / "simready-assets-hd-v3.json"
        self.output = self.curated_root / "merged-source-v3.json"
        self.receipt = self.volume / "contracts" / "source-assets-merged.json"

        curated_environment: dict[str, dict[str, list[dict[str, object]]]] = {
            "vegetation": {},
            "buildings": {},
        }
        official_environment: dict[str, dict[str, list[dict[str, object]]]] = {
            "vegetation": {},
            "buildings": {},
        }
        for kind, family_minimums in PHOTOREAL_FAMILY_MINIMUMS.items():
            for family, minimum in family_minimums.items():
                role = f"{kind}.{family}"
                curated_environment[kind][family] = [
                    _entry(
                        volume=self.volume,
                        manifest_parent=self.curated_root,
                        family=role,
                        stem=f"curated-{kind}-{family}-{index:02d}",
                    )
                    for index in range(minimum)
                ]
                if role in subject.COMMUNITY_MISSING_ENVIRONMENT:
                    count = 1 if populate_reserved_official else 0
                else:
                    count = minimum
                official_entries: list[dict[str, object]] = []
                for index in range(count):
                    source_basename = None
                    if (
                        role == "buildings.habitat"
                        and index == count - 1
                        and include_kasa
                    ):
                        source_basename = subject.KASA_SOURCE_BASENAME
                    wrapper_identity = None
                    if role == "vegetation.trees" and index == 1:
                        wrapper_identity = "official-vegetation-trees-00"
                    official_entries.append(
                        _entry(
                            volume=self.volume,
                            manifest_parent=self.official.parent,
                            family=role,
                            stem=f"official-{kind}-{family}-{index:02d}",
                            source_basename=source_basename,
                            wrapper_identity=wrapper_identity,
                        )
                    )
                    if role == "vegetation.trees" and index == 1:
                        previous = official_entries[0]
                        current = official_entries[-1]
                        current["source_cache_path"] = previous[
                            "source_cache_path"
                        ]
                        current["materialized_files"] = copy.deepcopy(
                            previous["materialized_files"]
                        )
                        current["content_lock_sha256"] = previous[
                            "content_lock_sha256"
                        ]
                        current["identity"] = copy.deepcopy(
                            previous["identity"]
                        )
                        current["provenance"] = copy.deepcopy(
                            previous["provenance"]
                        )
                        current["license"] = copy.deepcopy(
                            previous["license"]
                        )
                official_environment[kind][family] = official_entries

        actors = {
            class_id: _entry(
                volume=self.volume,
                manifest_parent=self.curated_root,
                family=f"actors.{class_id}",
                stem=f"curated-actor-{class_id}",
            )
            for class_id in REQUIRED_ACTOR_CLASSES
        }
        if missing_actor:
            actors.pop(REQUIRED_ACTOR_CLASSES[-1])
        selected_actor_assets = {
            selection_id: _entry(
                volume=self.volume,
                manifest_parent=self.curated_root,
                family="actors.selected_response",
                stem=f"selected-response-{selection_id}",
            )
            for selection_id in campaign_asset_bundle.SELECTED_ACTOR_GROUP_IDS
        }
        selected_environment_assets = {
            selection_id: _entry(
                volume=self.volume,
                manifest_parent=self.curated_root,
                family=f"{kind}.{family}",
                stem=f"selected-environment-{selection_id}",
            )
            for selection_id, kind, family in campaign_asset_bundle.SELECTED_ENVIRONMENT_GROUP
        }
        curated_payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "profile": MANIFEST_PROFILE,
            "library_policy": PHOTOREAL_LIBRARY_POLICY,
            "family_minimums": PHOTOREAL_FAMILY_MINIMUMS,
            "discovery": {
                "mode": "materialized_photoreal_asset_library_v3",
                "missing_environment": [],
                "missing_actor_classes": [],
            },
            "environment": curated_environment,
            "actors": actors,
            "selected_actor_group": {
                "group_id": campaign_asset_bundle.SELECTED_ACTOR_GROUP_ID,
                "selection_count": len(
                    campaign_asset_bundle.SELECTED_ACTOR_GROUP_IDS
                ),
                "selection_order": list(
                    campaign_asset_bundle.SELECTED_ACTOR_GROUP_IDS
                ),
                "assets": selected_actor_assets,
            },
            "selected_environment_group": {
                "group_id": campaign_asset_bundle.SELECTED_ENVIRONMENT_GROUP_ID,
                "selection_count": len(
                    campaign_asset_bundle.SELECTED_ENVIRONMENT_GROUP_IDS
                ),
                "selection_order": list(
                    campaign_asset_bundle.SELECTED_ENVIRONMENT_GROUP_IDS
                ),
                "assets": selected_environment_assets,
            },
            "pbr_materials": {"curated": "preserved"},
        }
        official_payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "profile": MANIFEST_PROFILE,
            "library_policy": PHOTOREAL_LIBRARY_POLICY,
            "family_minimums": PHOTOREAL_FAMILY_MINIMUMS,
            "discovery": {
                "mode": "materialized_photoreal_asset_library_v3",
                "missing_environment": list(
                    subject.COMMUNITY_MISSING_ENVIRONMENT
                ),
                "missing_actor_classes": list(REQUIRED_ACTOR_CLASSES),
            },
            "environment": official_environment,
            "actors": {},
        }
        _write_json(self.curated, curated_payload)
        _write_json(self.official, official_payload)
        _write_json(
            self.curated_root / INSTALL_MARKER,
            {
                "schema_version": 1,
                "state": "ASSET_BUNDLE_INSTALLED",
                "bundle_sha256": self.bundle_sha,
                "manifest_relative": self.curated.name,
                "runtime_manifest_sha256": _sha256(self.curated),
            },
        )

    def kwargs(self) -> dict[str, object]:
        return {
            "volume_root": self.volume,
            "curated_manifest": self.curated,
            "official_manifest": self.official,
            "output_manifest": self.output,
            "receipt_path": self.receipt,
            "curated_bundle_root": self.curated_root,
            "curated_bundle_sha256": self.bundle_sha,
        }

    def install_fake_community(self) -> None:
        payload = json.loads(self.output.read_text(encoding="utf-8"))
        final_root = self.curated_root / "community-building-assets" / "fixture"
        entries: list[dict[str, object]] = []
        source_locks: list[dict[str, object]] = []
        for uid, family in sorted(UID_TO_FAMILY.items()):
            entry = _entry(
                volume=self.volume,
                manifest_parent=self.curated_root,
                family=f"buildings.{family}",
                stem=f"community-{uid}",
                objaverse_uid=uid,
            )
            payload["environment"]["buildings"][family].append(entry)
            entries.append(copy.deepcopy(entry))
            source_locks.append(
                {
                    "uid": uid,
                    "family": family,
                    "sha256": COMMUNITY_BUILDING_GLB_LOCKS[uid]["sha256"],
                    "size_bytes": COMMUNITY_BUILDING_GLB_LOCKS[uid][
                        "size_bytes"
                    ],
                    "metadata_sha256": hashlib.sha256(
                        f"metadata:{uid}".encode("utf-8")
                    ).hexdigest(),
                }
            )
        receipt = {
            "schema_version": COMMUNITY_INSTALL_SCHEMA_VERSION,
            "state": COMMUNITY_INSTALL_STATE,
            "bundle_id": final_root.name,
            "source_count": len(entries),
            "entries": entries,
            "source_locks": source_locks,
        }
        receipt_path = final_root / "install-receipt.json"
        _write_json(receipt_path, receipt)
        payload["discovery"]["missing_environment"] = []
        payload["discovery"]["community_building_supplement"] = {
            "state": COMMUNITY_INSTALL_STATE,
            "provider": "Objaverse",
            "converter": "fixture",
            "bundle_id": final_root.name,
            "asset_count": len(entries),
            "objaverse_uids": sorted(UID_TO_FAMILY),
            "receipt": receipt_path.relative_to(self.curated_root).as_posix(),
            "receipt_sha256": _sha256(receipt_path),
        }
        _write_json(self.output, payload)


class SourceManifestMergeTests(unittest.TestCase):
    def test_merges_official_first_preserves_actors_and_reuses_exact_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))

            first = subject.merge_source_manifests(**fixture.kwargs())

            self.assertFalse(first["reused"])
            payload = json.loads(fixture.output.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["discovery"]["missing_environment"],
                list(subject.COMMUNITY_MISSING_ENVIRONMENT),
            )
            for family in subject.COMMUNITY_BUILDING_FAMILIES:
                self.assertEqual(
                    payload["environment"]["buildings"][family],
                    [],
                )
            curated_payload = json.loads(
                fixture.curated.read_text(encoding="utf-8")
            )
            self.assertEqual(payload["actors"], curated_payload["actors"])
            self.assertEqual(
                set(payload["actors"]),
                set(REQUIRED_ACTOR_CLASSES),
            )
            self.assertEqual(
                payload["pbr_materials"],
                curated_payload["pbr_materials"],
            )
            habitat = payload["environment"]["buildings"]["habitat"]
            self.assertEqual(
                Path(habitat[0]["source_cache_path"]).name.casefold(),
                subject.KASA_SOURCE_BASENAME,
            )
            self.assertTrue(habitat[0]["path"].startswith("../../"))
            selected_trees = payload["environment"]["vegetation"]["trees"]
            self.assertEqual(
                len(selected_trees),
                PHOTOREAL_FAMILY_MINIMUMS["vegetation"]["trees"],
            )
            self.assertTrue(
                any(
                    entry["identity"]["source_identity"].startswith(
                        "fixture:"
                        + fixture.curated_root.name
                        + ":vegetation.trees"
                    )
                    for entry in selected_trees
                )
            )
            all_entries = [
                *payload["actors"].values(),
                *(
                    entry
                    for kind, families in payload["environment"].items()
                    for family, entries in families.items()
                    if f"{kind}.{family}"
                    not in subject.COMMUNITY_MISSING_ENVIRONMENT
                    for entry in entries
                ),
            ]
            self.assertEqual(
                len({entry["asset_id"] for entry in all_entries}),
                len(all_entries),
            )
            self.assertEqual(
                len({entry["sha256"] for entry in all_entries}),
                len(all_entries),
            )
            receipt = json.loads(fixture.receipt.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["inputs"]["curated"]["manifest_sha256"],
                _sha256(fixture.curated),
            )
            self.assertEqual(
                receipt["inputs"]["official"]["manifest_sha256"],
                _sha256(fixture.official),
            )
            self.assertEqual(
                receipt["selection"]["deduplication"]["sha256"],
                1,
            )
            output_bytes = fixture.output.read_bytes()
            receipt_bytes = fixture.receipt.read_bytes()

            reused = subject.merge_source_manifests(**fixture.kwargs())

            self.assertTrue(reused["reused"])
            self.assertEqual(fixture.output.read_bytes(), output_bytes)
            self.assertEqual(fixture.receipt.read_bytes(), receipt_bytes)

            fixture.receipt.unlink()
            recovered_receipt = subject.merge_source_manifests(
                **fixture.kwargs()
            )
            self.assertTrue(
                recovered_receipt["recovered_interrupted_write"]
            )
            self.assertFalse(recovered_receipt["reused"])
            self.assertEqual(fixture.output.read_bytes(), output_bytes)
            self.assertEqual(fixture.receipt.read_bytes(), receipt_bytes)

            fixture.output.unlink()
            recovered_output = subject.merge_source_manifests(
                **fixture.kwargs()
            )
            self.assertTrue(
                recovered_output["recovered_interrupted_write"]
            )
            self.assertEqual(fixture.output.read_bytes(), output_bytes)
            self.assertEqual(fixture.receipt.read_bytes(), receipt_bytes)

    def test_rejects_output_or_input_drift_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            subject.merge_source_manifests(**fixture.kwargs())
            fixture.output.write_text('{"drifted":true}\n', encoding="utf-8")
            drifted = fixture.output.read_bytes()

            with self.assertRaisesRegex(
                subject.SourceManifestMergeError,
                "refusing to overwrite",
            ):
                subject.merge_source_manifests(**fixture.kwargs())
            self.assertEqual(fixture.output.read_bytes(), drifted)

            fixture = _Fixture(Path(temporary) / "second")
            subject.merge_source_manifests(**fixture.kwargs())
            fixture.official.write_text(
                fixture.official.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                subject.SourceManifestMergeError,
                "receipt drifted",
            ):
                subject.verify_source_manifest_merge(**fixture.kwargs())

    def test_fails_closed_without_kasa_three_gaps_or_seven_actors(self) -> None:
        scenarios = (
            (
                {"include_kasa": False},
                "corrected official Kasa source is absent",
            ),
            (
                {"missing_actor": True},
                "exact seven reviewed actor classes",
            ),
            (
                {"populate_reserved_official": True},
                "leave all three reviewed community families empty",
            ),
        )
        for options, message in scenarios:
            with self.subTest(options=options):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = _Fixture(Path(temporary), **options)
                    with self.assertRaisesRegex(
                        subject.SourceManifestMergeError,
                        message,
                    ):
                        subject.merge_source_manifests(**fixture.kwargs())
                    self.assertFalse(fixture.output.exists())
                    self.assertFalse(fixture.receipt.exists())

    def test_rejects_binary_duplicate_with_conflicting_semantic_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            payload = json.loads(fixture.official.read_text(encoding="utf-8"))
            duplicate = payload["environment"]["vegetation"]["trees"][1]
            duplicate["identity"]["source_identity"] = (
                "substituted:semantic-identity"
            )
            _write_json(fixture.official, payload)

            with self.assertRaisesRegex(
                subject.SourceManifestMergeError,
                "different dependency or semantic identity",
            ):
                subject.merge_source_manifests(**fixture.kwargs())

    def test_rejects_non_string_identity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            payload = json.loads(fixture.official.read_text(encoding="utf-8"))
            payload["environment"]["vegetation"]["trees"][0][
                "asset_id"
            ] = 123
            _write_json(fixture.official, payload)

            with self.assertRaisesRegex(
                subject.SourceManifestMergeError,
                "non-empty JSON string",
            ):
                subject.merge_source_manifests(**fixture.kwargs())

    def test_deduplicates_same_content_and_identity_with_another_wrapper(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            payload = json.loads(fixture.official.read_text(encoding="utf-8"))
            duplicate = payload["environment"]["vegetation"]["trees"][1]
            wrapper = fixture.official.parent / duplicate["path"]
            wrapper.write_text(
                "#usda 1.0\n"
                'def Xform "Asset" { string wrapper = "alternate" }\n',
                encoding="utf-8",
            )
            duplicate["sha256"] = _sha256(wrapper)
            _write_json(fixture.official, payload)

            subject.merge_source_manifests(**fixture.kwargs())

            receipt = json.loads(fixture.receipt.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["selection"]["deduplication"][
                    "content_semantic"
                ],
                1,
            )

    def test_interprocess_lock_rejects_a_concurrent_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            command = r"""
import sys,time
from pathlib import Path
from fireviewer_sdg import source_manifest_merge as subject
context=subject._prepare_context(
    volume_root=Path(sys.argv[1]),
    curated_manifest=Path(sys.argv[2]),
    official_manifest=Path(sys.argv[3]),
    output_manifest=Path(sys.argv[4]),
    receipt_path=Path(sys.argv[5]),
    curated_bundle_root=Path(sys.argv[6]),
    curated_bundle_sha256=sys.argv[7],
)
with subject._exclusive_merge_lock(subject._merge_lock_path(context)):
    print("LOCKED",flush=True)
    time.sleep(30)
"""
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    command,
                    str(fixture.volume),
                    str(fixture.curated),
                    str(fixture.official),
                    str(fixture.output),
                    str(fixture.receipt),
                    str(fixture.curated_root),
                    fixture.bundle_sha,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
            )
            try:
                self.assertIsNotNone(child.stdout)
                ready = child.stdout.readline().strip()
                if ready != "LOCKED":
                    details = (
                        child.stderr.read()
                        if child.poll() is not None and child.stderr
                        else ""
                    )
                    self.fail(f"lock helper did not start: {details}")
                context = subject._prepare_context(**fixture.kwargs())
                with self.assertRaisesRegex(
                    subject.SourceManifestMergeError,
                    "already running",
                ):
                    with subject._exclusive_merge_lock(
                        subject._merge_lock_path(context)
                    ):
                        self.fail("concurrent lock was unexpectedly acquired")
            finally:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=5)

            result = subject.merge_source_manifests(**fixture.kwargs())
            self.assertFalse(result["reused"])

    def test_accepts_only_the_exact_eight_asset_community_augmentation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            subject.merge_source_manifests(**fixture.kwargs())
            fixture.install_fake_community()

            verified = subject.verify_source_manifest_merge(
                **fixture.kwargs(),
                require_community=True,
            )

            self.assertEqual(
                verified["state"],
                "SOURCE_MANIFESTS_MERGED_WITH_COMMUNITY",
            )
            self.assertEqual(verified["community_asset_count"], 8)
            with self.assertRaisesRegex(
                subject.SourceManifestMergeError,
                "explicit community verification gate",
            ):
                subject.verify_source_manifest_merge(**fixture.kwargs())

            payload = json.loads(fixture.output.read_text(encoding="utf-8"))
            receipt_path = (
                fixture.output.parent
                / payload["discovery"]["community_building_supplement"][
                    "receipt"
                ]
            )
            original_receipt = json.loads(
                receipt_path.read_text(encoding="utf-8")
            )
            substituted_receipt = copy.deepcopy(original_receipt)
            substituted_receipt["source_locks"][0]["sha256"] = "0" * 64
            _write_json(receipt_path, substituted_receipt)
            payload["discovery"]["community_building_supplement"][
                "receipt_sha256"
            ] = _sha256(receipt_path)
            _write_json(fixture.output, payload)
            with self.assertRaisesRegex(
                subject.SourceManifestMergeError,
                "reviewed GLB",
            ):
                subject.verify_source_manifest_merge(
                    **fixture.kwargs(),
                    require_community=True,
                )

            substituted_receipt = copy.deepcopy(original_receipt)
            substituted_receipt["source_locks"][0][
                "metadata_sha256"
            ] = "f" * 64
            _write_json(receipt_path, substituted_receipt)
            payload["discovery"]["community_building_supplement"][
                "receipt_sha256"
            ] = _sha256(receipt_path)
            _write_json(fixture.output, payload)
            with self.assertRaisesRegex(
                subject.SourceManifestMergeError,
                "manifest metadata",
            ):
                subject.verify_source_manifest_merge(
                    **fixture.kwargs(),
                    require_community=True,
                )

            _write_json(receipt_path, original_receipt)
            payload["discovery"]["community_building_supplement"][
                "receipt_sha256"
            ] = _sha256(receipt_path)
            payload["actors"][REQUIRED_ACTOR_CLASSES[0]]["asset_id"] = "drifted"
            _write_json(fixture.output, payload)
            with self.assertRaisesRegex(
                subject.SourceManifestMergeError,
                "mutations outside",
            ):
                subject.verify_source_manifest_merge(
                    **fixture.kwargs(),
                    require_community=True,
                )

    def test_verified_merge_is_consumable_by_campaign_asset_planner(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            subject.merge_source_manifests(**fixture.kwargs())
            fixture.install_fake_community()
            subject.verify_source_manifest_merge(
                **fixture.kwargs(),
                require_community=True,
            )
            payload = json.loads(fixture.output.read_text(encoding="utf-8"))

            plans = campaign_asset_bundle._plan_assets(
                payload=payload,
                manifest_parent=fixture.output.parent,
                volume_root=fixture.volume,
            )

            expected_environment = sum(
                len(entries)
                for families in payload["environment"].values()
                for entries in families.values()
            )
            self.assertEqual(
                len(plans),
                expected_environment
                + len(REQUIRED_ACTOR_CLASSES)
                + len(campaign_asset_bundle.SELECTED_ACTOR_GROUP_IDS)
                + len(campaign_asset_bundle.SELECTED_ENVIRONMENT_GROUP_IDS),
            )
            official_plans = [
                plan
                for plan in plans
                if plan.source.path.is_relative_to(
                    fixture.official.parent
                )
                and not plan.source.path.is_relative_to(
                    fixture.curated_root
                )
            ]
            self.assertTrue(official_plans)
            self.assertTrue(
                any(
                    plan.source.path.name.casefold()
                    == subject.KASA_SOURCE_BASENAME
                    for plan in official_plans
                )
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TOOL = ROOT / "tools" / "runpod" / "download-community-building-assets.py"
SPEC = importlib.util.spec_from_file_location(
    "fireviewer_objaverse_downloader",
    TOOL,
)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


def _write_glb(path: Path, *, uid: str) -> None:
    document = {
        "asset": {"version": "2.0", "generator": f"fixture-{uid}"},
        "meshes": [{"name": uid, "primitives": [{"attributes": {}}]}],
    }
    payload = json.dumps(document, sort_keys=True).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    total = 12 + 8 + len(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(payload), 0x4E4F534A)
        + payload
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _annotations() -> dict[str, dict[str, Any]]:
    return {
        uid: {
            "name": f"Provider title {uid}",
            "viewerUrl": f"https://sketchfab.com/3d-models/{uid}",
            "user": {"displayName": f"Creator {uid[:8]}"},
            "license": "by",
            "isDownloadable": True,
            "archives": {
                "glb": {
                    "type": "glb",
                    "size": 100_000_000 + index,
                }
            },
        }
        for index, uid in enumerate(subject.LOCKED_UIDS)
    }


def _fixture_locks(sources: dict[str, Path]) -> dict[str, dict[str, object]]:
    return {
        uid: {
            "sha256": _sha256(source),
            "size_bytes": source.stat().st_size,
        }
        for uid, source in sources.items()
    }


class FakeObjaverse:
    def __init__(
        self,
        *,
        sources: dict[str, Path],
        annotations: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.sources = sources
        self.annotations = annotations or _annotations()
        self.annotation_calls: list[list[str]] = []
        self.object_calls: list[tuple[list[str], int]] = []
        self.BASE_PATH = "should-be-replaced"
        self._VERSIONED_PATH = "should-be-replaced"

    def load_annotations(self, uids: list[str]) -> dict[str, dict[str, Any]]:
        self.annotation_calls.append(list(uids))
        return {
            uid: self.annotations[uid]
            for uid in uids
            if uid in self.annotations
        }

    def load_objects(
        self,
        uids: list[str],
        download_processes: int = 1,
    ) -> dict[str, str]:
        self.object_calls.append((list(uids), download_processes))
        return {uid: str(self.sources[uid]) for uid in uids}


class ObjaverseCommunityDownloaderTests(unittest.TestCase):
    def _fixture(
        self,
        temporary: str,
    ) -> tuple[Path, Path, Path, dict[str, Path]]:
        volume = Path(temporary) / "volume"
        destination = volume / "input" / "objaverse-buildings"
        cache = volume / "cache" / "objaverse-0.1.7"
        downloaded = volume / "provider-downloads"
        volume.mkdir()
        sources: dict[str, Path] = {}
        for uid in subject.LOCKED_UIDS:
            source = downloaded / f"{uid}.glb"
            _write_glb(source, uid=uid)
            sources[uid] = source
        return volume, destination, cache, sources

    def test_production_content_locks_match_the_eight_reviewed_pod_glbs(
        self,
    ) -> None:
        self.assertEqual(
            subject.EXPECTED_GLB_LOCKS,
            {
                "0e93ab7b05944087b4a19fb7262877fa": {
                    "sha256": "390d20795f124f5157fb60dccc4803833a59f40d256b1386c6f481454fb0144f",
                    "size_bytes": 171_729_640,
                },
                "3447582ca28743d785d08f04ea710469": {
                    "sha256": "4c06f380f157097752b1c77e8465b0ed724299a939674f65f5170dd5137d4530",
                    "size_bytes": 4_522_804,
                },
                "3ebb228ec3ad441c850e0fd298223348": {
                    "sha256": "f5442862aa0560695899df72929f9da09b87bfc5f8087e3800a4726981842743",
                    "size_bytes": 290_421_108,
                },
                "70154ba65c7d4ba0b5f081fd0eb35f68": {
                    "sha256": "4ffb197402ddc2660ae22357d7a5cfd130f30520ba87e441a103daa942657e53",
                    "size_bytes": 15_852_908,
                },
                "aab826498d5b42fabb46e8c53f69a66c": {
                    "sha256": "c7fbfec4d5b1cf477c607b5b7c5c4281cd6eb075ac443d6533818c90fe03b7c9",
                    "size_bytes": 122_842_960,
                },
                "bcdac4f85bb940aaa189f22f869ed11f": {
                    "sha256": "0d0e475bd0295a025d95d2f11bab94075d8d5413dab60f6e01b373e4edeca8b1",
                    "size_bytes": 29_062_804,
                },
                "cf8074f44da541189bb4936b05f4f434": {
                    "sha256": "fbc7ea33b61b960ba68c8531000f5f4775922af85f1db3d069e145a59004e92a",
                    "size_bytes": 5_339_172,
                },
                "d790f2d3d93d4ad09c8c40f5331e5813": {
                    "sha256": "7ffc941ae3cbe8f335dd8e5872bc3e70ff39b6dc5b80a01eeb04786aaa8d60b1",
                    "size_bytes": 128_291_692,
                },
            },
        )
        self.assertEqual(
            set(subject.EXPECTED_GLB_LOCKS),
            set(subject.LOCKED_UIDS),
        )

    def test_downloads_only_locked_uids_and_writes_exact_installer_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume, destination, cache, sources = self._fixture(temporary)
            client = FakeObjaverse(sources=sources)

            with mock.patch.object(
                subject,
                "EXPECTED_GLB_LOCKS",
                _fixture_locks(sources),
            ):
                result = subject.download_community_building_assets(
                    client=client,
                    volume_root=volume,
                    destination_root=destination,
                    cache_root=cache,
                    workers=3,
                )

            self.assertEqual(result["state"], subject.DOWNLOAD_STATE)
            self.assertEqual(result["asset_count"], 8)
            self.assertEqual(result["downloaded_count"], 8)
            self.assertEqual(client.annotation_calls, [list(subject.LOCKED_UIDS)])
            self.assertEqual(
                client.object_calls,
                [(list(subject.LOCKED_UIDS), 3)],
            )
            self.assertEqual(client.BASE_PATH, str(cache.resolve()))
            self.assertEqual(
                client._VERSIONED_PATH,
                str((cache / "hf-objaverse-v1").resolve()),
            )

            metadata_path = destination / "metadata.json"
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(set(payload), {"assets"})
            self.assertEqual(set(payload["assets"]), set(subject.LOCKED_UIDS))
            for uid in subject.LOCKED_UIDS:
                target = destination / f"{uid}.glb"
                record = payload["assets"][uid]
                self.assertEqual(
                    record,
                    {
                        "uid": uid,
                        "family": subject.UID_TO_FAMILY[uid],
                        "name": subject.COMMUNITY_BUILDING_TITLES[uid],
                        "title": f"Provider title {uid}",
                        "creator": f"Creator {uid[:8]}",
                        "source_uri": (
                            f"https://sketchfab.com/3d-models/{uid}"
                        ),
                        "license_id": "CC-BY-4.0",
                        "license_uri": (
                            "https://creativecommons.org/licenses/by/4.0/"
                        ),
                        "local_path": f"{uid}.glb",
                        "downloadable": True,
                        "provider_archive_size_bytes": (
                            100_000_000
                            + list(subject.LOCKED_UIDS).index(uid)
                        ),
                        "sha256": _sha256(target),
                        "size_bytes": target.stat().st_size,
                        "glb_version": 2,
                    },
                )
                self.assertEqual(
                    list(destination.glob(f".{uid}.glb.partial-*")),
                    [],
                )

    def test_complete_destination_is_reused_without_object_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume, destination, cache, sources = self._fixture(temporary)
            first_client = FakeObjaverse(sources=sources)
            with mock.patch.object(
                subject,
                "EXPECTED_GLB_LOCKS",
                _fixture_locks(sources),
            ):
                first = subject.download_community_building_assets(
                    client=first_client,
                    volume_root=volume,
                    destination_root=destination,
                    cache_root=cache,
                )
                metadata_bytes = (destination / "metadata.json").read_bytes()
                second_client = FakeObjaverse(sources={})
                second = subject.download_community_building_assets(
                    client=second_client,
                    volume_root=volume,
                    destination_root=destination,
                    cache_root=cache,
                )

            self.assertEqual(first["downloaded_count"], 8)
            self.assertEqual(second["downloaded_count"], 0)
            self.assertFalse(second["metadata_changed"])
            self.assertEqual(second_client.object_calls, [])
            self.assertEqual(
                (destination / "metadata.json").read_bytes(),
                metadata_bytes,
            )

    def test_annotations_fail_closed_before_any_object_download(self) -> None:
        mutations = {
            "not downloadable": ("isDownloadable", False),
            "missing creator": ("user", {}),
            "license mapping is forbidden": (
                "license",
                {
                    "uid": "by",
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                },
            ),
            "license alias is forbidden": ("license", "ccby"),
            "non attribution license": ("license", "by-nc"),
        }
        for label, (key, value) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                volume, destination, cache, sources = self._fixture(temporary)
                annotations = _annotations()
                annotations[subject.LOCKED_UIDS[0]][key] = value
                client = FakeObjaverse(
                    sources=sources,
                    annotations=annotations,
                )

                with mock.patch.object(
                    subject,
                    "EXPECTED_GLB_LOCKS",
                    _fixture_locks(sources),
                ):
                    with self.assertRaises(
                        subject.CommunityBuildingDownloadError
                    ):
                        subject.download_community_building_assets(
                            client=client,
                            volume_root=volume,
                            destination_root=destination,
                            cache_root=cache,
                        )

                self.assertEqual(client.object_calls, [])
                self.assertFalse((destination / "metadata.json").exists())

    def test_truncated_provider_glb_never_replaces_target_or_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume, destination, cache, sources = self._fixture(temporary)
            locks = _fixture_locks(sources)
            bad_uid = subject.LOCKED_UIDS[0]
            sources[bad_uid].write_bytes(b"glTF")
            client = FakeObjaverse(sources=sources)

            with mock.patch.object(subject, "EXPECTED_GLB_LOCKS", locks):
                with self.assertRaises(subject.CommunityBuildingDownloadError):
                    subject.download_community_building_assets(
                        client=client,
                        volume_root=volume,
                        destination_root=destination,
                        cache_root=cache,
                    )

            self.assertFalse((destination / f"{bad_uid}.glb").exists())
            self.assertEqual(
                list(destination.glob(f".{bad_uid}.glb.partial-*")),
                [],
            )
            self.assertFalse((destination / "metadata.json").exists())

    def test_destination_and_cache_must_stay_inside_volume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume, destination, cache, sources = self._fixture(temporary)
            client = FakeObjaverse(sources=sources)
            outside = Path(temporary) / "outside"

            with mock.patch.object(
                subject,
                "EXPECTED_GLB_LOCKS",
                _fixture_locks(sources),
            ):
                with self.assertRaises(subject.CommunityBuildingDownloadError):
                    subject.download_community_building_assets(
                        client=client,
                        volume_root=volume,
                        destination_root=outside,
                        cache_root=cache,
                    )
                with self.assertRaises(subject.CommunityBuildingDownloadError):
                    subject.download_community_building_assets(
                        client=client,
                        volume_root=volume,
                        destination_root=destination,
                        cache_root=outside,
                    )

    def test_valid_but_substituted_glb_is_rejected_and_metadata_invalidated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume, destination, cache, sources = self._fixture(temporary)
            locks = _fixture_locks(sources)
            with mock.patch.object(subject, "EXPECTED_GLB_LOCKS", locks):
                subject.download_community_building_assets(
                    client=FakeObjaverse(sources=sources),
                    volume_root=volume,
                    destination_root=destination,
                    cache_root=cache,
                )
                bad_uid, substitute_uid = subject.LOCKED_UIDS[:2]
                (destination / f"{bad_uid}.glb").write_bytes(
                    sources[substitute_uid].read_bytes()
                )
                replacement_sources = dict(sources)
                replacement_sources[bad_uid] = sources[substitute_uid]
                client = FakeObjaverse(sources=replacement_sources)

                with self.assertRaises(subject.CommunityBuildingDownloadError):
                    subject.download_community_building_assets(
                        client=client,
                        volume_root=volume,
                        destination_root=destination,
                        cache_root=cache,
                    )

            self.assertEqual(client.object_calls, [([bad_uid], 4)])
            self.assertFalse((destination / "metadata.json").exists())
            self.assertEqual(
                list(destination.glob(".metadata.json.stale-*")),
                [],
            )
            self.assertEqual(
                list(destination.glob(".objaverse-stage-*")),
                [],
            )


if __name__ == "__main__":
    unittest.main()

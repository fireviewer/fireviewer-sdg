from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg.provisioning import load_manifest, provision  # noqa: E402


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.payload) - self.offset
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


class ProvisioningTests(unittest.TestCase):
    def _write_manifest(self, root: Path, payload: bytes) -> Path:
        manifest = {
            "schema_version": 1,
            "profile": "test",
            "artifacts": [
                {
                    "artifact_id": "scene",
                    "kind": "scene",
                    "revision": "commit-123",
                    "destination": "scenes/example",
                    "files": [
                        {
                            "relative_path": "scene.usd",
                            "url": "https://huggingface.co/example/resolve/commit-123/scene.usd",
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "size_bytes": len(payload),
                        }
                    ],
                }
            ],
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_download_then_cache_hit(self) -> None:
        payload = b"usd-scene"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_manifest(root, payload)
            volume = root / "volume"
            with patch(
                "fireviewer_sdg.provisioning.urllib.request.urlopen",
                return_value=_Response(payload),
            ):
                first = provision(
                    manifest_path=manifest,
                    volume_root=volume,
                    allowed_hosts=frozenset({"huggingface.co"}),
                )
            self.assertEqual(first.downloaded, ("scene/scene.usd",))
            with patch(
                "fireviewer_sdg.provisioning.urllib.request.urlopen",
                side_effect=AssertionError("cache hit must not access the network"),
            ):
                second = provision(
                    manifest_path=manifest,
                    volume_root=volume,
                    allowed_hosts=frozenset({"huggingface.co"}),
                )
            self.assertEqual(second.cache_hits, ("scene/scene.usd",))

    def test_rejects_non_https_or_non_allowlisted_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_manifest(root, b"payload")
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["artifacts"][0]["files"][0]["url"] = "http://127.0.0.1/model.bin"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                load_manifest(manifest, frozenset({"huggingface.co"}))

    def test_rejects_escape_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_manifest(root, b"payload")
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["artifacts"][0]["destination"] = "../outside"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "relative path"):
                load_manifest(manifest, frozenset({"huggingface.co"}))

    def test_missing_token_is_fail_closed(self) -> None:
        payload_bytes = b"payload"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_manifest(root, payload_bytes)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["artifacts"][0]["files"][0]["auth_env"] = "FW_TEST_TOKEN"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
                RuntimeError, "missing credential"
            ):
                provision(
                    manifest_path=manifest,
                    volume_root=root / "volume",
                    allowed_hosts=frozenset({"huggingface.co"}),
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fireviewer_sdg.ground_material_bundle import (
    CHANNELS,
    ROLE_SOURCES,
    build_ground_material_bundle,
)
from fireviewer_sdg.terrain_pbr import load_locked_material_library


class GroundMaterialBundleTests(unittest.TestCase):
    def test_builds_and_reuses_complete_locked_library(self) -> None:
        payloads: dict[str, dict[str, object]] = {}
        for slug, _repeat in ROLE_SOURCES.values():
            payload: dict[str, object] = {}
            for _texture_role, (channel, file_format, _space) in CHANNELS.items():
                content = f"{slug}-{channel}-{file_format}".encode()
                payload[channel] = {
                    "4k": {
                        file_format: {
                            "url": f"https://example.test/{slug}/{channel}.{file_format}",
                            "size": len(content),
                        }
                    }
                }
            mtlx = f"{slug}-mtlx".encode()
            payload["mtlx"] = {
                "4k": {
                    "mtlx": {
                        "url": f"https://example.test/{slug}/material.mtlx",
                        "size": len(mtlx),
                    }
                }
            }
            payloads[slug] = payload

        def fetch(url: str) -> dict[str, object]:
            return payloads[url.rsplit("/", 1)[-1]]

        def download(path: Path, url: str, expected: int) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            slug = url.split("/")[-2]
            leaf = url.rsplit("/", 1)[-1]
            if leaf == "material.mtlx":
                content = f"{slug}-mtlx".encode()
            else:
                channel, file_format = leaf.rsplit(".", 1)
                content = f"{slug}-{channel}-{file_format}".encode()
            self.assertEqual(len(content), expected)
            path.write_bytes(content)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ground"
            result = build_ground_material_bundle(
                output_root=root,
                workers=4,
                fetch_json=fetch,
                download_file=download,
            )
            self.assertFalse(result["reused"])
            manifest = root / "manifest-v3.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["pbr_materials"]), set(ROLE_SOURCES))
            locked = load_locked_material_library(
                bundle_root=root,
                manifest_path=manifest,
            )
            self.assertEqual(len(locked.materials), 7)

            reused = build_ground_material_bundle(
                output_root=root,
                fetch_json=lambda _url: self.fail("unexpected network request"),
                download_file=lambda *_args: self.fail("unexpected download"),
            )
            self.assertTrue(reused["reused"])


if __name__ == "__main__":
    unittest.main()

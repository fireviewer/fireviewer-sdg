from __future__ import annotations

import unittest
from collections import Counter
import hashlib
import os
import tempfile
from pathlib import Path

from fireviewer_sdg.omniverse_scene_gate import (
    _coverage_contract,
    _infer_zone_root,
    _nonnegative_count,
    _prototype_share_gate,
    _resolve_locked_artifact,
    _variant_build_contract,
)


def _receipt() -> dict[str, object]:
    terrain = [
        {"path": f"build/terrain/T{i:03d}.usdc", "sha256": "a" * 64}
        for i in range(400)
    ]
    hero_details = [
        {"path": f"build/details/T{i:03d}.usdc", "sha256": "b" * 64}
        for i in range(400)
    ]
    mid_details = [
        {"path": f"build/details-mid/T{i:03d}.usdc", "sha256": "c" * 64}
        for i in range(400)
    ]
    far_details = [
        {"path": f"build/details-far/T{i:03d}.usdc", "sha256": "d" * 64}
        for i in range(400)
    ]
    coverage = [
        {
            "tile_ref": f"T{i:03d}",
            "terrain_payload": terrain[i]["path"],
            "detail_payload": hero_details[i]["path"],
            "detail_lods": {
                "HERO": hero_details[i]["path"],
                "MID": mid_details[i]["path"],
                "FAR": far_details[i]["path"],
            },
            "terrain_lods": (
                ["LOD0", "LOD1", "LOD2", "LOD3"]
                if i < 12
                else ["LOD1", "LOD2", "LOD3"]
            ),
            "detail_counts": {
                "buildings": 1,
                "roads": 2,
                "hydrology": 3,
                "vegetation": 100,
            },
            "detail_lod_counts": {
                level: {
                    "buildings": 1,
                    "roads": 2,
                    "hydrology": 3,
                    "vegetation": (
                        100 if level == "HERO" else 25 if level == "MID" else 7
                    ),
                }
                for level in ("HERO", "MID", "FAR")
            },
            "instance_namespace": i + 1,
            "collision_lods": ["NEAR", "FAR"],
        }
        for i in range(400)
    ]
    return {
        "schema_version": 2,
        "source_profile": "full",
        "payloads": terrain,
        "detail_payloads": hero_details,
        "detail_mid_payloads": mid_details,
        "detail_far_payloads": far_details,
        "tile_coverage": coverage,
        "layers": {
            "buildings": {"prim_count": 400},
            "roads": {"prim_count": 800},
            "hydrology": {"prim_count": 1200},
            "vegetation": {"prim_count": 40_000},
            "collisions": {
                "prim_count": 400,
                "levels": ["NEAR", "FAR"],
                "near_spacing_m": 4.0,
                "far_spacing_m": 32.0,
            },
            "detail_streaming": {
                "prim_count": 400,
                "levels": ["HERO", "MID", "FAR"],
                "terrain_is_never_unloaded_for_detail_streaming": True,
            },
        },
    }


class OmniverseSceneGateContractTests(unittest.TestCase):
    def test_accepts_exact_400_tile_incremental_contract(self) -> None:
        contract = _coverage_contract(_receipt())
        self.assertEqual(len(contract["by_tile"]), 400)
        self.assertEqual(contract["lod0_tiles"], 12)
        self.assertEqual(
            contract["totals"],
            {
                "buildings": 400,
                "roads": 800,
                "hydrology": 1200,
                "vegetation": 40_000,
            },
        )

    def test_rejects_detail_count_that_differs_from_layer_total(self) -> None:
        receipt = _receipt()
        receipt["layers"]["vegetation"]["prim_count"] = 39_999
        with self.assertRaisesRegex(ValueError, "differs from tile total"):
            _coverage_contract(receipt)

    def test_rejects_missing_detail_payload_and_lod_chain(self) -> None:
        receipt = _receipt()
        receipt["detail_payloads"].pop()
        with self.assertRaisesRegex(ValueError, "400 HERO detail"):
            _coverage_contract(receipt)
        receipt = _receipt()
        receipt["tile_coverage"][0]["terrain_lods"] = ["LOD0", "LOD1"]
        with self.assertRaisesRegex(ValueError, "LOD contract"):
            _coverage_contract(receipt)

    def test_rejects_no_review_lod0_and_terrain_unload_contract(self) -> None:
        receipt = _receipt()
        for tile in receipt["tile_coverage"]:
            tile["terrain_lods"] = ["LOD1", "LOD2", "LOD3"]
        with self.assertRaisesRegex(ValueError, "no LOD0"):
            _coverage_contract(receipt)
        receipt = _receipt()
        receipt["layers"]["detail_streaming"][
            "terrain_is_never_unloaded_for_detail_streaming"
        ] = False
        with self.assertRaisesRegex(ValueError, "preserve full-zone terrain"):
            _coverage_contract(receipt)

    def test_prototype_gate_rejects_more_than_25_percent(self) -> None:
        accepted = _prototype_share_gate(
            Counter({"a": 25, "b": 25, "c": 25, "d": 25}),
            label="trees",
        )
        self.assertEqual(accepted["maximum_share"], 0.25)
        with self.assertRaisesRegex(RuntimeError, "dominates 26.0%"):
            _prototype_share_gate(
                Counter({"a": 26, "b": 25, "c": 25, "d": 24}),
                label="trees",
            )

    def test_integer_counts_reject_boolean_fraction_and_negative(self) -> None:
        for value in (True, -1, 1.5, "1.5"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _nonnegative_count(value, label="count")

    def test_locked_artifact_must_remain_inside_inferred_zone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            zone = Path(directory) / "zone"
            build = zone / "build"
            build.mkdir(parents=True)
            root = build / "review.usdc"
            root.write_bytes(b"root")
            receipt = build / "build-receipt.json"
            receipt.write_text("{}\n", encoding="utf-8")
            inferred = _infer_zone_root(
                build_receipt=receipt.resolve(),
                root_usd=root.resolve(),
                root_record={"path": "build/review.usdc"},
            )
            self.assertEqual(inferred, zone.resolve())
            record = {
                "path": "build/review.usdc",
                "sha256": hashlib.sha256(b"root").hexdigest(),
            }
            self.assertEqual(
                _resolve_locked_artifact(
                    zone_root=inferred,
                    record=record,
                    label="root",
                    suffixes=frozenset({".usdc"}),
                ),
                root.resolve(),
            )
            outside = Path(directory) / "outside.usdc"
            outside.write_bytes(b"outside")
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                _resolve_locked_artifact(
                    zone_root=inferred,
                    record={
                        "path": "../outside.usdc",
                        "sha256": hashlib.sha256(b"outside").hexdigest(),
                    },
                    label="outside",
                    suffixes=frozenset({".usdc"}),
                )

    def test_variant_contract_requires_exact_identity_and_route_membership(
        self,
    ) -> None:
        identity_sha = "1" * 64
        topology_sha = "2" * 64
        payload = {
            "scene_kind": "fictive_variant",
            "zone_id": "SIM-01",
            "base_scene_id": "Z16",
            "variant_index": 1,
            "identity_contract": {
                "numeric_ids_preserved": True,
                "stable_ids_preserved": True,
                "source_namespace_may_differ_from_destination_tile": True,
                "source_identity_sha256": identity_sha,
                "authored_identity_sha256": identity_sha,
            },
            "route_topology": {
                "algorithm": "segment-connectivity-components-v1",
                "source_component_count": 3,
                "result_component_count": 3,
                "source_membership_sha256": topology_sha,
                "result_membership_sha256": topology_sha,
                "exact_membership_preserved": True,
            },
        }
        self.assertEqual(
            _variant_build_contract(payload),
            {
                "scene_id": "SIM-01",
                "base_scene_id": "Z16",
                "variant_index": 1,
                "identity_sha256": identity_sha,
                "route_membership_sha256": topology_sha,
                "route_component_count": 3,
            },
        )
        payload["route_topology"]["result_membership_sha256"] = "3" * 64
        with self.assertRaisesRegex(ValueError, "component membership"):
            _variant_build_contract(payload)

    def test_shared_variant_artifact_may_leave_scene_but_not_volume(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            scene = volume / "variant-scenes" / "SIM-01"
            shared = volume / "composition-sources" / "Z16" / "terrain.usdc"
            scene.mkdir(parents=True)
            shared.parent.mkdir(parents=True)
            shared.write_bytes(b"shared")
            record = {
                "path": Path(
                    os.path.relpath(shared, start=scene)
                ).as_posix(),
                "sha256": hashlib.sha256(b"shared").hexdigest(),
            }
            self.assertEqual(
                _resolve_locked_artifact(
                    zone_root=scene,
                    allowed_root=volume,
                    record=record,
                    label="shared terrain",
                    suffixes=frozenset({".usdc"}),
                ),
                shared.resolve(),
            )


if __name__ == "__main__":
    unittest.main()

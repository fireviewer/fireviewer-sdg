from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg.real_world import (  # noqa: E402
    REQUIRED_ACTOR_CLASSES,
    _composition_provenance,
    load_real_world_contract,
    select_actor_variation,
    select_case_variation,
)


def _file(root: Path, name: str, content: bytes) -> tuple[str, str]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path.name, hashlib.sha256(content).hexdigest()


def _contract(volume: Path) -> Path:
    root = volume / "input"
    root.mkdir(parents=True)
    manifest, manifest_sha = _file(root, "capture.json", b'{"images":100}')
    scene, scene_sha = _file(root, "scene.usd", b"#usda 1.0\n")
    flow, flow_sha = _file(root, "flow.usd", b"#usda 1.0\n# flow\n")
    ortho, ortho_sha = _file(root, "ortho.tif", b"verified-ortho")
    mnt, mnt_sha = _file(root, "mnt.tif", b"verified-mnt")
    preview, preview_sha = _file(root, "mnt.png", b"verified-preview")
    poses = [
        {
            "id": f"pose-{index:02d}",
            "position": [float(index), -30.0, 6.0],
            "look_at": [0.0, 0.0, 2.0],
            "validation": "ncore_project_to_image_passed",
            "viewpoint": {
                "distance_band": ["near", "medium", "far", "very_far"][index],
                "occlusion": ["clear", "partial_building", "partial_mountain", "clear"][index],
                "azimuth_deg": float(index * 90),
                "elevation_deg": 5.0,
                "occlusion_fraction": [0.0, 0.25, 0.4, 0.0][index],
                "occluder_prim_path": (
                    "" if index in {0, 3} else f"/World/Occluders/occluder-{index}"
                ),
                "line_of_sight_validation": "usd_raycast_and_reference_render_passed",
                "required_anchors_visible": True,
                "reference_validation": (
                    "pending_console_review" if index == 2 else "reference_render_human_approved"
                ),
            },
        }
        for index in range(4)
    ]
    actors = []
    for index, class_id in enumerate(sorted(REQUIRED_ACTOR_CLASSES)):
        asset, asset_sha = _file(root, f"actor-{index}.usd", f"#usda 1.0\n# {class_id}\n".encode())
        actors.append(
            {
                "class_id": class_id,
                "asset": asset,
                "asset_sha256": asset_sha,
                "center_world_m": [float(index * 5), 5.0, 1.0],
                "aabb_min_world_m": [float(index * 5 - 1), 3.0, 0.0],
                "aabb_max_world_m": [float(index * 5 + 1), 7.0, 3.0],
                "translation_world_m": [float(index * 5), 5.0, 0.0],
                "rotation_xyz_deg": [0.0, 0.0, 0.0],
                "scale_xyz": [1.0, 1.0, 1.0],
                "camera_pose_ids": [f"pose-{index % 4:02d}"],
                "engagement_context": (
                    "hard_negative_not_engaged"
                    if class_id.startswith("hard_negative")
                    else "wildfire_response_engaged"
                ),
                "quality_validation": "simready_asset_human_approved",
            }
        )
    payload = {
        "schema_version": 1,
        "pipeline": "nvidia_omniverse_nurec_3dgut",
        "render_profile": "omniverse_realworld_hd_v1",
        "site_id": "fr-test-site-001",
        "event_id": "fr-test-event-001",
        "duration_days": 4,
        "capture": {
            "source": "new_real_world_capture",
            "capture_manifest": manifest,
            "capture_manifest_sha256": manifest_sha,
            "image_count": 100,
            "registered_image_count": 96,
            "minimum_source_resolution": [3840, 2160],
            "mean_reprojection_error_px": 0.5,
            "overlap_validated": True,
            "intrinsics_validated": True,
            "extrinsics_validated": True,
            "timestamps_validated": True,
            "coordinate_convention": "ncore_rig_and_camera_v4",
        },
        "reconstruction": {
            "trainer": "nv-tlabs/3dgrut",
            "format": "particle_field",
            "asset": scene,
            "asset_sha256": scene_sha,
            "metrics": {
                "psnr": 28.0,
                "ssim": 0.94,
                "held_out_evaluation": True,
                "held_out_view_count": 12,
            },
        },
        "composition": {
            "flow_asset": flow,
            "flow_asset_sha256": flow_sha,
            "flow_validation": {
                "preset_rendered_and_anchor_verified": True,
                "simulated_frame_count": 4,
            },
            "camera_poses": poses,
            "lighting_variants": [
                {
                    "id": f"light-{index}",
                    "prim_path": "/World/RealWorldScene",
                    "variant_set": "lighting",
                    "selection": f"light_{index}",
                    "time_of_day": ["day", "night", "dawn", "dusk"][index],
                    "validation": "reference_render_human_approved",
                }
                for index in range(4)
            ],
            "flow_states": [
                {
                    "id": f"flow-{index}",
                    "time_seconds": float(index + 1),
                    "event_day": index + 1,
                    "lighting_variant_id": f"light-{index % 4}",
                    "anchors_world_m": {
                        "active_fire_point": [0.0, 0.0, 0.2],
                        "visible_fire_front_point": [2.0 + index * 0.01, 0.5, 0.3],
                        "smoke_column_base": [0.5, 0.0, 1.0],
                    },
                    "progression": {
                        "phase": [
                            "initial_growth",
                            "advancing_flame_zone",
                            "front_split",
                            "reignition",
                        ][index],
                        "front_ids": (
                            ["front-a", "front-b"] if index >= 2 else ["front-a"]
                        ),
                        "parent_front_ids": ["front-a"] if index == 2 else [],
                        "advancing_zone_ids": ["zone-a"] if index == 1 else [],
                        "reignited_zone_ids": ["zone-b"] if index == 3 else [],
                        "burned_area_m2": float(100 + index * 25),
                        "active_flame_area_m2": float(20 + index * 5),
                    },
                    "validation": "flow_reference_render_human_approved",
                }
                for index in range(4)
            ],
            "diversity": {
                "selector": "operational_viewpoint_progression_v1",
                "capacity_per_category": 16,
            },
            "actors": actors,
        },
        "geospatial": {
            "crs": "EPSG:2154",
            "country_profile": "FR",
            "landscape_origin": "real_french_capture",
            "landscape_profile": "rural_mountain",
            "site_context_validation": "reference_render_human_approved",
            "world_axes_aligned_lambert93": True,
            "world_origin_lambert93_m": [700000.0, 6600000.0, 100.0],
            "orthophoto": ortho,
            "orthophoto_sha256": ortho_sha,
            "mnt": mnt,
            "mnt_sha256": mnt_sha,
            "mnt_preview": preview,
            "mnt_preview_sha256": preview_sha,
        },
    }
    path = root / "real-world-scene.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class RealWorldContractTests(unittest.TestCase):
    def test_v1_contract_can_explicitly_exclude_response_actors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            path = _contract(volume)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["scope"] = {
                "response_engagement": False,
                "humans": False,
            }
            payload["composition"]["actors"] = []
            path.write_text(json.dumps(payload), encoding="utf-8")
            contract = load_real_world_contract(path, volume_root=volume)
            self.assertFalse(contract["scope"]["response_engagement"])
            self.assertEqual(contract["composition"]["actors"], [])

    def test_composition_provenance_keeps_synthetic_quality_review(self) -> None:
        common = {
            "pipeline": "nvidia_omniverse_simready_flow",
            "render_profile": "omniverse_realworld_hd_v1",
            "site_id": "synthetic-fr",
            "capture": {
                "source": "new_synthetic_french_reference",
                "capture_manifest_sha256": "a" * 64,
            },
            "reconstruction": {
                "format": "review_gated_usd",
                "asset_sha256": "b" * 64,
                "metrics": {"quality_review": "pending_console_review"},
            },
            "composition": {"diversity": {"capacity_per_category": 16}},
        }
        provenance = _composition_provenance(common)
        self.assertEqual(provenance["quality_review"], "pending_console_review")
        self.assertNotIn("psnr", provenance)
        self.assertNotIn("ssim", provenance)

    def test_complete_contract_loads_verified_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            loaded = load_real_world_contract(_contract(volume), volume_root=volume)
            self.assertEqual(loaded["capture"]["registered_image_count"], 96)
            self.assertEqual(len(loaded["composition"]["camera_poses"]), 4)
            self.assertEqual(loaded["composition"]["diversity"]["capacity_per_category"], 16)
            self.assertEqual(
                {actor["class_id"] for actor in loaded["composition"]["actors"]},
                set(REQUIRED_ACTOR_CLASSES),
            )

    def test_contract_refuses_tampered_scene_asset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            path = _contract(volume)
            (path.parent / "scene.usd").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_real_world_contract(path, volume_root=volume)

    def test_contract_refuses_unvalidated_capture_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            path = _contract(volume)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["capture"]["intrinsics_validated"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "intrinsics_validated"):
                load_real_world_contract(path, volume_root=volume)

    def test_contract_refuses_low_held_out_quality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            path = _contract(volume)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["reconstruction"]["metrics"]["ssim"] = 0.89
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SSIM"):
                load_real_world_contract(path, volume_root=volume)

    def test_all_event_variations_are_unique_and_actor_capacity_is_sufficient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            loaded = load_real_world_contract(_contract(volume), volume_root=volume)
            signatures = {
                select_case_variation(loaded, index)["diversity_signature"]
                for index in range(16)
            }
            self.assertEqual(len(signatures), 16)
            for class_id in REQUIRED_ACTOR_CLASSES:
                actor_signatures = {
                    select_actor_variation(loaded, class_id, ordinal)["diversity_signature"]
                    for ordinal in range(4)
                }
                self.assertEqual(len(actor_signatures), 4)

    def test_contract_refuses_more_than_eight_camera_stations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            path = _contract(volume)
            payload = json.loads(path.read_text(encoding="utf-8"))
            for index in range(4, 9):
                payload["composition"]["camera_poses"].append(
                    {
                        "id": f"pose-{index:02d}",
                        "position": [float(index), -30.0, 6.0],
                        "look_at": [0.0, 0.0, 2.0],
                        "validation": "ncore_project_to_image_passed",
                        "viewpoint": {
                            "distance_band": "far",
                            "occlusion": "clear",
                            "azimuth_deg": float(index * 30),
                            "elevation_deg": 5.0,
                            "occlusion_fraction": 0.0,
                            "occluder_prim_path": "",
                            "line_of_sight_validation": "usd_raycast_and_reference_render_passed",
                            "required_anchors_visible": True,
                            "reference_validation": "reference_render_human_approved",
                        },
                    }
                )
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed validated camera stations"):
                load_real_world_contract(path, volume_root=volume)

    def test_contract_refuses_uncoupled_lighting_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            path = _contract(volume)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["composition"]["flow_states"][0]["lighting_variant_id"] = "absent"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "approved lighting variant"):
                load_real_world_contract(path, volume_root=volume)

    def test_contract_refuses_reignition_without_reignited_zone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            path = _contract(volume)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["composition"]["flow_states"][3]["progression"][
                "reignited_zone_ids"
            ] = []
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reignition requires"):
                load_real_world_contract(path, volume_root=volume)


if __name__ == "__main__":
    unittest.main()

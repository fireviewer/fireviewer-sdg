from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "open-zone-scene-in-composer.py"


def _load_module():
    omni = types.ModuleType("omni")
    omni.__path__ = []  # type: ignore[attr-defined]
    omni_kit = types.ModuleType("omni.kit")
    omni_kit.__path__ = []  # type: ignore[attr-defined]
    omni_kit_app = types.ModuleType("omni.kit.app")
    omni_viewport = types.ModuleType("omni.kit.viewport")
    omni_viewport.__path__ = []  # type: ignore[attr-defined]
    omni_viewport_utility = types.ModuleType("omni.kit.viewport.utility")
    omni_usd = types.ModuleType("omni.usd")
    omni.kit = omni_kit  # type: ignore[attr-defined]
    omni.usd = omni_usd  # type: ignore[attr-defined]
    omni_kit.app = omni_kit_app  # type: ignore[attr-defined]
    omni_kit.viewport = omni_viewport  # type: ignore[attr-defined]
    omni_viewport.utility = omni_viewport_utility  # type: ignore[attr-defined]
    pxr = types.ModuleType("pxr")
    for name in ("Gf", "Sdf", "Usd", "UsdGeom"):
        setattr(pxr, name, types.SimpleNamespace())
    carb = types.ModuleType("carb")
    modules = {
        "carb": carb,
        "omni": omni,
        "omni.kit": omni_kit,
        "omni.kit.app": omni_kit_app,
        "omni.kit.viewport": omni_viewport,
        "omni.kit.viewport.utility": omni_viewport_utility,
        "omni.usd": omni_usd,
        "pxr": pxr,
    }
    spec = importlib.util.spec_from_file_location(
        "fireviewer_open_zone_scene_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(sys.modules, modules),
        patch.dict(os.environ, {"FW_SDG_REVIEW_DISABLE_AUTOSTART": "1"}),
    ):
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


class ProgressiveEditorOpenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def _tiles(self):
        tiles = []
        for row in range(20):
            for column in range(20):
                xmin = float((column - 10) * 1_000)
                ymin = float((row - 10) * 1_000)
                ref = f"T{row:02d}_{column:02d}"
                lods = ("LOD0", "LOD1", "LOD2", "LOD3") if row < 3 else (
                    "LOD1",
                    "LOD2",
                    "LOD3",
                )
                tiles.append(
                    self.module.TileHeader(
                        tile_ref=ref,
                        tile_path=f"/Tiles/{ref}",
                        terrain_path=f"/Tiles/{ref}/Terrain",
                        hero_detail_path=f"/Tiles/{ref}/Details",
                        mid_detail_path=f"/Tiles/{ref}/DetailsMid",
                        far_detail_path=f"/Tiles/{ref}/DetailsFar",
                        bounds=(xmin, ymin, xmin + 1_000.0, ymin + 1_000.0),
                        terrain_lods=lods,
                        collision_lods=("NEAR", "FAR"),
                    )
                )
        return tiles

    def test_plan_keeps_all_terrain_and_bounds_heavy_details(self) -> None:
        inverse = 1.0 / math.sqrt(5.0)
        view = self.module.CameraView(
            eye=(0.0, -6_000.0, 3_000.0),
            forward=(0.0, 2.0 * inverse, -inverse),
            right=(1.0, 0.0, 0.0),
            up=(0.0, inverse, 2.0 * inverse),
            horizontal_half_tangent=0.55,
            vertical_half_tangent=0.4,
            ground_z=0.0,
            focus_xy=(0.0, 0.0),
            altitude=3_000.0,
        )
        plan = self.module._plan_working_set(
            tiles=self._tiles(),
            view=view,
            hero_cap=48,
            hero_guard_minimum=16,
            lod0_cap=12,
            lod1_cap=64,
            lod2_cap=196,
        )
        self.assertEqual(len(plan.terrain_lods), 400)
        self.assertEqual(len(plan.detail_levels), 400)
        self.assertEqual(len(plan.detail_paths), 400)
        self.assertGreater(len(plan.visible_tile_refs), 1)
        detail_counts = {
            level: list(plan.detail_levels.values()).count(level)
            for level in ("HERO", "MID", "FAR")
        }
        self.assertGreaterEqual(detail_counts["HERO"], 16)
        self.assertLessEqual(detail_counts["HERO"], 48)
        self.assertEqual(sum(detail_counts.values()), 400)
        for tile_ref in plan.visible_tile_refs:
            self.assertIn(plan.detail_levels[tile_ref], {"HERO", "MID"})
        for tile_ref, detail_level in plan.detail_levels.items():
            terrain_path = f"/Tiles/{tile_ref}/Terrain"
            expected_collision = (
                "NEAR" if detail_level in {"HERO", "MID"} else "FAR"
            )
            self.assertEqual(
                plan.collision_paths[tile_ref],
                terrain_path,
            )
            self.assertEqual(
                plan.collision_lods[tile_ref],
                expected_collision,
            )
        counts = {
            lod: list(plan.terrain_lods.values()).count(lod)
            for lod in ("LOD0", "LOD1", "LOD2", "LOD3")
        }
        self.assertLessEqual(counts["LOD0"], 12)
        self.assertLessEqual(counts["LOD1"], 64)
        self.assertGreater(counts["LOD3"], 0)
        self.assertEqual(sum(counts.values()), 400)

    def test_aerial_view_keeps_mid_representation_beyond_hero_cap(self) -> None:
        view = self.module.CameraView(
            eye=(0.0, 0.0, 30_000.0),
            forward=(0.0, 0.0, -1.0),
            right=(1.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            horizontal_half_tangent=0.5,
            vertical_half_tangent=0.5,
            ground_z=0.0,
            focus_xy=(0.0, 0.0),
            altitude=30_000.0,
        )
        plan = self.module._plan_working_set(
            tiles=self._tiles(),
            view=view,
            hero_cap=48,
            hero_guard_minimum=16,
            lod0_cap=12,
            lod1_cap=64,
            lod2_cap=196,
        )
        self.assertGreater(len(plan.visible_tile_refs), 48)
        visible_levels = {
            tile_ref: plan.detail_levels[tile_ref]
            for tile_ref in plan.visible_tile_refs
        }
        self.assertEqual(
            sum(level == "HERO" for level in visible_levels.values()),
            48,
        )
        self.assertGreater(
            sum(level == "MID" for level in visible_levels.values()),
            0,
        )
        self.assertNotIn("FAR", set(visible_levels.values()))
        self.assertEqual(len(plan.detail_paths), 400)

    def test_frustum_edge_crossing_tile_is_not_missed(self) -> None:
        thin_frustum_footprint = (
            (-2.0, 0.4),
            (2.0, 0.4),
            (2.0, 0.6),
            (-2.0, 0.6),
        )
        self.assertTrue(
            self.module._rectangle_intersects_polygon(
                (0.0, 0.0, 1.0, 1.0),
                thin_frustum_footprint,
            )
        )
        self.assertFalse(
            self.module._rectangle_intersects_polygon(
                (3.0, 3.0, 4.0, 4.0),
                thin_frustum_footprint,
            )
        )

    def test_relief_aabb_crossing_frustum_is_not_missed(self) -> None:
        tile = self.module.TileHeader(
            tile_ref="RIDGE",
            tile_path="/Tiles/RIDGE",
            terrain_path="/Tiles/RIDGE/Terrain",
            hero_detail_path="/Tiles/RIDGE/Details",
            mid_detail_path="/Tiles/RIDGE/DetailsMid",
            far_detail_path="/Tiles/RIDGE/DetailsFar",
            bounds=(100.0, -10.0, 200.0, 10.0),
            terrain_lods=("LOD1", "LOD2", "LOD3"),
            collision_lods=("NEAR", "FAR"),
            ground_z=60.0,
            minimum_z=0.0,
            maximum_z=120.0,
        )
        horizontal_view = self.module.CameraView(
            eye=(0.0, 0.0, 100.0),
            forward=(1.0, 0.0, 0.0),
            right=(0.0, 1.0, 0.0),
            up=(0.0, 0.0, 1.0),
            horizontal_half_tangent=0.1,
            vertical_half_tangent=0.1,
            ground_z=0.0,
            focus_xy=(1_000.0, 0.0),
            altitude=100.0,
        )
        self.assertTrue(
            self.module._tile_visible(tile, horizontal_view, guard=1.0)
        )

    def test_empty_frustum_falls_back_to_near_guard_and_far_coverage(self) -> None:
        view = self.module.CameraView(
            eye=(0.0, 0.0, 1_000.0),
            forward=(0.0, 1.0, 0.0),
            right=(1.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            horizontal_half_tangent=0.01,
            vertical_half_tangent=0.01,
            ground_z=0.0,
            focus_xy=(100_000.0, 100_000.0),
            altitude=1_000.0,
        )
        plan = self.module._plan_working_set(
            tiles=self._tiles(),
            view=view,
            hero_cap=48,
            hero_guard_minimum=16,
            lod0_cap=0,
            lod1_cap=32,
            lod2_cap=128,
        )
        hero_count = list(plan.detail_levels.values()).count("HERO")
        self.assertGreaterEqual(hero_count, 16)
        self.assertLessEqual(hero_count, 48)
        self.assertGreater(
            list(plan.detail_levels.values()).count("FAR"),
            0,
        )
        self.assertEqual(len(plan.detail_paths), 400)
        self.assertEqual(len(plan.terrain_lods), 400)

    def test_full_contract_rejects_hero_only_detail_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            zone_root = Path(directory)
            build = zone_root / "build"
            build.mkdir()
            root = build / "Z16_root.usdc"
            root.write_bytes(b"root")
            digest = hashlib.sha256(root.read_bytes()).hexdigest()
            coverage = []
            terrain = []
            hero_details = []
            for index in range(4):
                tile_ref = f"T{index}"
                terrain_path = f"build/payloads/{tile_ref}.usdc"
                detail_path = f"build/details/{tile_ref}_details.usdc"
                coverage.append(
                    {
                        "tile_ref": tile_ref,
                        "terrain_payload": terrain_path,
                        "detail_lods": {
                            "HERO": detail_path,
                            "MID": f"build/details-mid/{tile_ref}.usdc",
                            "FAR": f"build/details-far/{tile_ref}.usdc",
                        },
                        "detail_lod_counts": {
                            level: {
                                "buildings": 0,
                                "roads": 1,
                                "hydrology": 0,
                                "vegetation": 0,
                            }
                            for level in ("HERO", "MID", "FAR")
                        },
                        "terrain_lods": ["LOD1", "LOD2", "LOD3"],
                        "collision_lods": ["NEAR", "FAR"],
                    }
                )
                terrain.append({"path": terrain_path})
                hero_details.append({"path": detail_path})
            receipt_path = build / "build-receipt.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "zone_id": "Z16",
                        "source_profile": "full",
                        "root_usd": {
                            "path": "build/Z16_root.usdc",
                            "sha256": digest,
                        },
                        "payloads": terrain,
                        "detail_payloads": hero_details,
                        "tile_coverage": coverage,
                        "layers": {
                            "collisions": {
                                "prim_count": 4,
                                "levels": ["NEAR", "FAR"],
                                "near_spacing_m": 4.0,
                                "far_spacing_m": 32.0,
                            },
                            "detail_streaming": {
                                "prim_count": 4,
                                "levels": ["HERO", "MID", "FAR"],
                                "terrain_is_never_unloaded_for_detail_streaming": True,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "MID/FAR"):
                self.module._read_build_contract(
                    receipt_path=receipt_path,
                    usd_path=root.resolve(),
                    zone_id="Z16",
                    expected_tile_count=4,
                )

    def test_full_contract_accepts_complete_hero_mid_far_catalogs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            zone_root = Path(directory)
            build = zone_root / "build"
            build.mkdir()
            root = build / "Z16_root.usdc"
            root.write_bytes(b"root")
            digest = hashlib.sha256(root.read_bytes()).hexdigest()
            terrain = []
            hero = []
            mid = []
            far = []
            coverage = []
            for index in range(2):
                tile_ref = f"T{index}"
                terrain_path = f"build/payloads/{tile_ref}.usdc"
                hero_path = f"build/details/{tile_ref}.usdc"
                mid_path = f"build/details-mid/{tile_ref}.usdc"
                far_path = f"build/details-far/{tile_ref}.usdc"
                terrain.append({"path": terrain_path})
                hero.append({"path": hero_path})
                mid.append({"path": mid_path})
                far.append({"path": far_path})
                coverage.append(
                    {
                        "tile_ref": tile_ref,
                        "terrain_payload": terrain_path,
                        "terrain_lods": ["LOD1", "LOD2", "LOD3"],
                        "collision_lods": ["NEAR", "FAR"],
                        "detail_lods": {
                            "HERO": hero_path,
                            "MID": mid_path,
                            "FAR": far_path,
                        },
                        "detail_lod_counts": {
                            level: {
                                "buildings": 0,
                                "roads": 1,
                                "hydrology": 0,
                                "vegetation": 0,
                            }
                            for level in ("HERO", "MID", "FAR")
                        },
                    }
                )
            receipt_path = build / "build-receipt.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "zone_id": "Z16",
                        "source_profile": "full",
                        "root_usd": {
                            "path": "build/Z16_root.usdc",
                            "sha256": digest,
                        },
                        "payloads": terrain,
                        "detail_payloads": hero,
                        "detail_mid_payloads": mid,
                        "detail_far_payloads": far,
                        "tile_coverage": coverage,
                        "layers": {
                            "collisions": {
                                "prim_count": 2,
                                "levels": ["NEAR", "FAR"],
                                "near_spacing_m": 4.0,
                                "far_spacing_m": 32.0,
                            },
                            "detail_streaming": {
                                "prim_count": 2,
                                "levels": ["HERO", "MID", "FAR"],
                                "terrain_is_never_unloaded_for_detail_streaming": True,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            _, by_ref = self.module._read_build_contract(
                receipt_path=receipt_path,
                usd_path=root.resolve(),
                zone_id="Z16",
                expected_tile_count=2,
            )
            self.assertEqual(set(by_ref), {"T0", "T1"})

    def test_plan_replaces_far_only_after_mid_or_hero_loads(self) -> None:
        class Prim:
            def IsValid(self) -> bool:
                return True

            def IsLoaded(self) -> bool:
                return True

        class Stage:
            def __init__(self) -> None:
                self.events = []

            def Load(self, path, _policy) -> None:
                self.events.append(("load", path))

            def Unload(self, path) -> None:
                self.events.append(("unload", path))

            def GetPrimAtPath(self, _path):
                return Prim()

        class Context:
            def __init__(self, active_stage) -> None:
                self.active_stage = active_stage

            def get_stage(self):
                return self.active_stage

            def get_stage_loading_status(self):
                return "", 1, 1

            def get_stage_streaming_status(self):
                return False

        class App:
            async def next_update_async(self) -> None:
                return None

        stage = Stage()
        plan = self.module.WorkingSetPlan(
            terrain_lods={
                "/Tiles/T0/Terrain": "LOD3",
                "/Tiles/T1/Terrain": "LOD3",
                "/Tiles/T2/Terrain": "LOD3",
                "/Tiles/T3/Terrain": "LOD3",
            },
            collision_lods={
                "T0": "NEAR",
                "T1": "NEAR",
                "T2": "FAR",
                "T3": "FAR",
            },
            collision_paths={
                f"T{index}": f"/Tiles/T{index}/Terrain"
                for index in range(4)
            },
            detail_levels={
                "T0": "MID",
                "T1": "HERO",
                "T2": "FAR",
                "T3": "FAR",
            },
            detail_paths={
                "T0": "/Tiles/T0/DetailsMid",
                "T1": "/Tiles/T1/Details",
                "T2": "/Tiles/T2/DetailsFar",
                "T3": "/Tiles/T3/DetailsFar",
            },
            visible_tile_refs=("T0", "T1"),
        )
        current_lods = {
            "/Tiles/T0/Terrain": "LOD3",
            "/Tiles/T1/Terrain": "LOD3",
            "/Tiles/T2/Terrain": "LOD3",
            "/Tiles/T3/Terrain": "LOD3",
        }
        current_collision_lods = {
            "T0": "NEAR",
            "T1": "NEAR",
            "T2": "NEAR",
            "T3": "NEAR",
        }
        active_detail_levels = {
            "T0": "FAR",
            "T1": "FAR",
            "T2": "HERO",
            "T3": "HERO",
        }
        active_detail_paths = {
            "T0": "/Tiles/T0/DetailsFar",
            "T1": "/Tiles/T1/DetailsFar",
            "T2": "/Tiles/T2/Details",
            "T3": "/Tiles/T3/Details",
        }
        with (
            patch.object(self.module.Sdf, "Path", side_effect=lambda path: path, create=True),
            patch.object(
                self.module.Usd,
                "LoadWithDescendants",
                object(),
                create=True,
            ),
            patch.object(
                self.module.omni.kit.app,
                "get_app",
                return_value=App(),
                create=True,
            ),
            patch.object(
                self.module,
                "_session_select_collision",
                side_effect=lambda _stage, path, level: stage.events.append(
                    ("collision", path, level)
                ),
            ),
        ):
            asyncio.run(
                self.module._apply_plan(
                    context=Context(stage),
                    stage=stage,
                    plan=plan,
                    current_lods=current_lods,
                    current_collision_lods=current_collision_lods,
                    active_detail_levels=active_detail_levels,
                    active_detail_paths=active_detail_paths,
                    detail_transition_cap=1,
                    detail_settle_maximum_updates=4,
                    lod_transition_cap=32,
                    initial=False,
                )
            )
        for tile_ref, replacement in (
            ("T0", "/Tiles/T0/DetailsMid"),
            ("T1", "/Tiles/T1/Details"),
        ):
            old = f"/Tiles/{tile_ref}/DetailsFar"
            self.assertLess(
                stage.events.index(("load", replacement)),
                stage.events.index(("unload", old)),
            )
        self.assertEqual(active_detail_levels["T0"], "MID")
        self.assertEqual(active_detail_levels["T1"], "HERO")
        self.assertEqual(
            sum(active_detail_levels[ref] == "FAR" for ref in ("T2", "T3")),
            1,
        )
        self.assertEqual(current_collision_lods["T2"], "FAR")
        self.assertEqual(current_collision_lods["T3"], "NEAR")
        self.assertLess(
            stage.events.index(("unload", "/Tiles/T2/Details")),
            stage.events.index(
                ("collision", "/Tiles/T2/Terrain", "FAR")
            ),
        )
        self.assertNotIn(
            ("unload", "/Tiles/T0/Terrain"),
            stage.events,
        )

    def test_failed_detail_demotion_keeps_hero_payload_and_near_collision(self) -> None:
        class Prim:
            def IsValid(self) -> bool:
                return False

            def IsLoaded(self) -> bool:
                return False

        class Stage:
            def __init__(self) -> None:
                self.events = []

            def Load(self, path, _policy) -> None:
                self.events.append(("load", path))

            def Unload(self, path) -> None:
                self.events.append(("unload", path))

            def GetPrimAtPath(self, _path):
                return Prim()

        class Context:
            def __init__(self, stage) -> None:
                self.stage = stage

            def get_stage(self):
                return self.stage

            def get_stage_loading_status(self):
                return "loading", 0, 1

            def get_stage_streaming_status(self):
                return True

        class App:
            async def next_update_async(self) -> None:
                return None

        stage = Stage()
        context = Context(stage)
        plan = self.module.WorkingSetPlan(
            terrain_lods={"/Tiles/T0/Terrain": "LOD3"},
            collision_lods={"T0": "FAR"},
            collision_paths={"T0": "/Tiles/T0/Terrain"},
            detail_levels={"T0": "FAR"},
            detail_paths={"T0": "/Tiles/T0/DetailsFar"},
            visible_tile_refs=(),
        )
        active_detail_levels = {"T0": "HERO"}
        active_detail_paths = {"T0": "/Tiles/T0/Details"}
        current_collision_lods = {"T0": "NEAR"}
        with (
            patch.object(self.module.Sdf, "Path", side_effect=lambda path: path, create=True),
            patch.object(
                self.module.Usd,
                "LoadWithDescendants",
                object(),
                create=True,
            ),
            patch.object(
                self.module.omni.kit.app,
                "get_app",
                return_value=App(),
                create=True,
            ),
            patch.object(
                self.module,
                "_session_select_collision",
            ) as collision_selection,
        ):
            with self.assertRaisesRegex(RuntimeError, "did not settle"):
                asyncio.run(
                    self.module._apply_plan(
                        context=context,
                        stage=stage,
                        plan=plan,
                        current_lods={"/Tiles/T0/Terrain": "LOD3"},
                        current_collision_lods=current_collision_lods,
                        active_detail_levels=active_detail_levels,
                        active_detail_paths=active_detail_paths,
                        detail_transition_cap=1,
                        detail_settle_maximum_updates=2,
                        lod_transition_cap=1,
                        initial=False,
                    )
                )
        self.assertEqual(active_detail_levels, {"T0": "HERO"})
        self.assertEqual(
            active_detail_paths,
            {"T0": "/Tiles/T0/Details"},
        )
        self.assertEqual(current_collision_lods, {"T0": "NEAR"})
        collision_selection.assert_not_called()
        self.assertFalse(
            any(event[0] == "unload" for event in stage.events)
        )

    def test_script_uses_load_none_and_never_saves_source_layers(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("UsdContextInitialLoadSet.LOAD_NONE", source)
        self.assertIn("stage.GetSessionLayer()", source)
        self.assertIn("stage.SetEditTarget(stage.GetSessionLayer())", source)
        self.assertNotIn("GetRootLayer().Save(", source)
        self.assertNotIn("stage.Export(", source)


if __name__ == "__main__":
    unittest.main()

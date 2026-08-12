from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


for dependency in ("pymeshlab", "trimesh"):
    if importlib.util.find_spec(dependency) is None:
        raise unittest.SkipTest(f"optional mesh dependency is unavailable: {dependency}")


MODULE_PATH = Path(__file__).parents[1] / "tools" / "hunyuan3d" / "asset4sim_mesh.py"
SPEC = importlib.util.spec_from_file_location("asset4sim_mesh", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FaceBudgetTests(unittest.TestCase):
    def test_global_average_and_complexity_order_are_preserved(self) -> None:
        targets = MODULE.allocate_face_targets(
            [0.65, 0.9, 1.0, 1.4, 2.0],
            target_average=5_000,
            minimum_faces=2_500,
            maximum_faces=12_000,
        )
        self.assertEqual(sum(targets), 25_000)
        self.assertEqual(targets, sorted(targets))
        self.assertGreater(targets[-1], 5_000)
        self.assertLess(targets[0], 5_000)

    def test_invalid_contract_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.allocate_face_targets([1.0], target_average=2_000, minimum_faces=2_500)

    def test_extreme_reduction_is_split_into_bounded_stages(self) -> None:
        stages = MODULE.progressive_face_targets(995_260, 5_000, maximum_ratio=2.5)
        previous = 995_260
        for target in stages:
            self.assertLessEqual(previous / target, 2.5)
            previous = target
        self.assertEqual(stages[-1], 5_000)


class RepairTests(unittest.TestCase):
    def test_triangular_hole_is_closed(self) -> None:
        box = MODULE.trimesh.creation.box()
        box.update_faces([index != 0 for index in range(len(box.faces))])
        box.remove_unreferenced_vertices()
        self.assertFalse(box.is_watertight)

        repaired = MODULE.repair_mesh(box, fill_holes=True)

        self.assertTrue(repaired.is_watertight)
        self.assertEqual(MODULE.boundary_edge_count(repaired), 0)

    def test_simplification_reduces_a_dense_sphere(self) -> None:
        sphere = MODULE.trimesh.creation.icosphere(subdivisions=4)
        simplified = MODULE.simplify_mesh(sphere, 1_000)
        self.assertLessEqual(len(simplified.faces), 1_200)
        self.assertGreaterEqual(len(simplified.faces), 900)

    def test_quality_gate_accepts_progressive_sphere_reduction(self) -> None:
        sphere = MODULE.trimesh.creation.icosphere(subdivisions=4)
        simplified, attempts = MODULE.simplify_with_quality_gate(
            sphere,
            1_000,
            maximum_faces=2_000,
            quality_samples=2_000,
        )
        self.assertTrue(attempts[-1]["quality"]["passed"])
        self.assertLessEqual(len(simplified.faces), 2_000)


if __name__ == "__main__":
    unittest.main()

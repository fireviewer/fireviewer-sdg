from __future__ import annotations

import unittest

import numpy as np

from fireviewer_sdg.canopy import (
    CanopyCandidate,
    detect_canopy_candidates,
    select_canopy_instances,
)


class CanopyTests(unittest.TestCase):
    def test_candidates_follow_mnh_summits_and_forest_mask(self) -> None:
        values = np.zeros((9, 9), dtype=np.float32)
        values[2, 2] = 14.0
        values[6, 6] = 20.0
        values[2, 7] = 25.0  # Outside the source forest polygon.
        polygon = [[0.0, 0.0], [8.0, 0.0], [8.0, 10.0], [0.0, 10.0]]
        detected = detect_canopy_candidates(
            values=values,
            bounds=(0.0, 0.0, 10.0, 10.0),
            forest_polygons=[polygon],
            minimum_height_metres=3.0,
            nms_radius_metres=1.0,
            analysis_samples=9,
        )
        self.assertEqual(sorted(round(item.height) for item in detected), [14, 20])

    def test_broad_plateau_is_reduced_by_metric_nms(self) -> None:
        values = np.zeros((11, 11), dtype=np.float32)
        values[4:7, 4:7] = 12.0
        polygon = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        detected = detect_canopy_candidates(
            values=values,
            bounds=(0.0, 0.0, 10.0, 10.0),
            forest_polygons=[polygon],
            nms_radius_metres=3.0,
            analysis_samples=11,
        )
        self.assertEqual(len(detected), 1)

    def test_budget_is_repeatable_and_preserves_source_positions(self) -> None:
        candidates = [
            CanopyCandidate(float(index), float(index * 2), 10.0, index)
            for index in range(20)
        ]
        first = select_canopy_instances(candidates, budget=7, deterministic_seed=16)
        second = select_canopy_instances(candidates, budget=7, deterministic_seed=16)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        self.assertTrue(set(first).issubset(candidates))

    def test_invalid_or_empty_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            detect_canopy_candidates(
                values=np.zeros((2, 2), dtype=np.float32),
                bounds=(0.0, 0.0, 1.0, 1.0),
                forest_polygons=[[[0, 0], [1, 0], [0, 1]]],
            )
        self.assertEqual(
            detect_canopy_candidates(
                values=np.zeros((4, 4), dtype=np.float32),
                bounds=(0.0, 0.0, 1.0, 1.0),
                forest_polygons=[],
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()

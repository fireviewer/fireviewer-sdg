"""Deterministic LiDAR-MNH canopy extraction for tiled USD vegetation.

This module deliberately does not scatter trees uniformly inside forest
polygons.  It finds canopy summits in the LiDAR-derived MNH raster, constrains
them to the BD TOPO vegetation mask, and only then applies a stable budget.
That keeps tree positions and heights tied to observable canopy structure while
allowing the USD builder to stream a bounded number of photoreal instances.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CanopyCandidate:
    x: float
    y: float
    height: float
    source_index: int


def _point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (current[1] > y) != (previous[1] > y):
            crossing = (
                (previous[0] - current[0])
                * (y - current[1])
                / (previous[1] - current[1])
                + current[0]
            )
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def _polygon_records(
    polygons: Iterable[Sequence[Sequence[float]]],
) -> list[tuple[tuple[float, float, float, float], Sequence[Sequence[float]]]]:
    records = []
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        records.append(((min(xs), min(ys), max(xs), max(ys)), polygon))
    return records


def detect_canopy_candidates(
    *,
    values: np.ndarray,
    bounds: tuple[float, float, float, float],
    forest_polygons: Iterable[Sequence[Sequence[float]]],
    minimum_height_metres: float = 3.0,
    nms_radius_metres: float = 2.5,
    analysis_samples: int = 667,
) -> list[CanopyCandidate]:
    """Return LiDAR-MNH canopy summits inside source vegetation polygons."""

    if values.ndim != 2 or min(values.shape) < 3:
        raise ValueError("MNH canopy raster must be two-dimensional")
    if minimum_height_metres <= 0 or nms_radius_metres <= 0:
        raise ValueError("canopy height and NMS radius must be positive")
    xmin, ymin, xmax, ymax = (float(value) for value in bounds)
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("canopy bounds are invalid")
    polygons = _polygon_records(forest_polygons)
    if not polygons:
        return []

    rows_count = min(max(3, analysis_samples), values.shape[0])
    columns_count = min(max(3, analysis_samples), values.shape[1])
    source_rows = np.linspace(0, values.shape[0] - 1, rows_count, dtype=np.intp)
    source_columns = np.linspace(
        0, values.shape[1] - 1, columns_count, dtype=np.intp
    )
    sampled = values[np.ix_(source_rows, source_columns)].astype(
        np.float32, copy=True
    )
    sampled[~np.isfinite(sampled)] = -np.inf
    resolution_x = (xmax - xmin) / float(columns_count - 1)
    resolution_y = (ymax - ymin) / float(rows_count - 1)
    pixel_radius = max(
        1,
        int(
            math.ceil(
                nms_radius_metres / max(1e-6, min(resolution_x, resolution_y))
            )
        ),
    )

    padded = np.pad(
        sampled,
        ((pixel_radius, pixel_radius), (pixel_radius, pixel_radius)),
        mode="constant",
        constant_values=-np.inf,
    )
    local_maximum = np.full_like(sampled, -np.inf)
    for row_offset in range(pixel_radius * 2 + 1):
        for column_offset in range(pixel_radius * 2 + 1):
            view = padded[
                row_offset : row_offset + rows_count,
                column_offset : column_offset + columns_count,
            ]
            np.maximum(local_maximum, view, out=local_maximum)
    maxima = np.argwhere(
        (sampled >= float(minimum_height_metres)) & (sampled >= local_maximum)
    )
    if maxima.size == 0:
        return []

    # Tallest candidates win NMS conflicts.  Row/column tie-breaks make broad
    # plateaus deterministic rather than producing every equal-valued pixel.
    ordered = sorted(
        (
            (
                -float(sampled[int(row), int(column)]),
                int(row),
                int(column),
            )
            for row, column in maxima
        )
    )
    occupied: dict[tuple[int, int], list[tuple[float, float]]] = {}
    result: list[CanopyCandidate] = []
    cell_size = nms_radius_metres
    radius_squared = nms_radius_metres * nms_radius_metres
    for _negative_height, row, column in ordered:
        x = xmin + float(column) * resolution_x
        y = ymax - float(row) * resolution_y
        if not any(
            left <= x <= right
            and bottom <= y <= top
            and _point_in_polygon(x, y, polygon)
            for (left, bottom, right, top), polygon in polygons
        ):
            continue
        cell = (
            int(math.floor((x - xmin) / cell_size)),
            int(math.floor((y - ymin) / cell_size)),
        )
        conflict = False
        for cell_y in range(cell[1] - 1, cell[1] + 2):
            for cell_x in range(cell[0] - 1, cell[0] + 2):
                for existing_x, existing_y in occupied.get((cell_x, cell_y), ()):
                    if (existing_x - x) ** 2 + (existing_y - y) ** 2 < radius_squared:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                break
        if conflict:
            continue
        occupied.setdefault(cell, []).append((x, y))
        result.append(
            CanopyCandidate(
                x=x,
                y=y,
                height=float(sampled[row, column]),
                source_index=row * columns_count + column,
            )
        )
    return result


def select_canopy_instances(
    candidates: Sequence[CanopyCandidate],
    *,
    budget: int,
    deterministic_seed: int,
) -> list[CanopyCandidate]:
    """Select a spatially neutral, repeatable subset without moving summits."""

    if budget < 0:
        raise ValueError("canopy budget must not be negative")
    if len(candidates) <= budget:
        return list(candidates)
    scored = sorted(
        candidates,
        key=lambda item: (
            (
                item.source_index * 1_103_515_245
                + deterministic_seed * 2_654_435_761
                + 12_345
            )
            & 0xFFFFFFFF,
            item.source_index,
        ),
    )
    return scored[:budget]

#!/usr/bin/env python3
"""Repair and adaptively simplify Hunyuan3D meshes for Asset4Sim.

The batch allocator treats ``target_average`` as a triangle budget for the
whole set.  Simple assets receive fewer triangles and geometrically complex
assets receive more, while respecting the configured lower and upper bounds.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pymeshlab
import trimesh


LOG = logging.getLogger("asset4sim.hunyuan3d.mesh")


@dataclass(frozen=True)
class MeshMetrics:
    vertices: int
    faces: int
    components: int
    boundary_edges: int
    watertight: bool
    sharp_edge_ratio: float
    adjacency_angle_p90_degrees: float
    complexity_score: float
    complexity_weight: float


@dataclass(frozen=True)
class GeometryQuality:
    normalized_p95_distance: float
    normalized_p99_distance: float
    normalized_max_distance: float
    normal_angle_p95_degrees: float
    normalized_bounds_drift: float
    passed: bool


def as_trimesh(mesh: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    """Return one triangle mesh without silently discarding scene geometry."""

    if isinstance(mesh, trimesh.Trimesh):
        return mesh.copy()
    if isinstance(mesh, trimesh.Scene):
        geometries = [geometry for geometry in mesh.geometry.values() if len(geometry.faces)]
        if not geometries:
            raise ValueError("The scene contains no triangle geometry")
        return trimesh.util.concatenate(geometries)
    raise TypeError(f"Unsupported mesh type: {type(mesh)!r}")


def boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    if len(mesh.faces) == 0:
        return 0
    counts = np.bincount(mesh.edges_unique_inverse)
    return int(np.count_nonzero(counts == 1))


def measure_mesh(mesh: trimesh.Trimesh | trimesh.Scene) -> MeshMetrics:
    current = as_trimesh(mesh)
    finite_angles = np.asarray(current.face_adjacency_angles, dtype=np.float64)
    finite_angles = finite_angles[np.isfinite(finite_angles)]

    if finite_angles.size:
        sharp_ratio = float(np.mean(finite_angles >= math.radians(30.0)))
        angle_p90 = float(np.degrees(np.quantile(finite_angles, 0.90)))
    else:
        sharp_ratio = 0.0
        angle_p90 = 0.0

    try:
        components = max(1, len(current.split(only_watertight=False)))
    except Exception:
        components = 1

    # The score favors actual surface changes over raw tessellation density.
    # It is intentionally bounded so raw meshes with millions of triangles do
    # not automatically consume the whole batch budget.
    angle_term = min(angle_p90 / 75.0, 1.0)
    sharp_term = min(sharp_ratio / 0.20, 1.0)
    component_term = min(math.log2(components) / 4.0, 1.0) if components > 1 else 0.0
    score = float(np.clip(0.50 * angle_term + 0.35 * sharp_term + 0.15 * component_term, 0.0, 1.0))
    weight = 0.65 + 1.35 * score

    return MeshMetrics(
        vertices=int(len(current.vertices)),
        faces=int(len(current.faces)),
        components=components,
        boundary_edges=boundary_edge_count(current),
        watertight=bool(current.is_watertight),
        sharp_edge_ratio=sharp_ratio,
        adjacency_angle_p90_degrees=angle_p90,
        complexity_score=score,
        complexity_weight=weight,
    )


def allocate_face_targets(
    complexity_weights: Sequence[float],
    target_average: int = 5_000,
    minimum_faces: int = 2_500,
    maximum_faces: int = 12_000,
) -> list[int]:
    """Allocate a bounded triangle budget while preserving the requested mean."""

    if not complexity_weights:
        return []
    if minimum_faces < 4 or maximum_faces < minimum_faces:
        raise ValueError("Invalid minimum/maximum face bounds")
    if not minimum_faces <= target_average <= maximum_faces:
        raise ValueError("target_average must be within the configured bounds")

    weights = np.asarray(complexity_weights, dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Complexity weights must be finite and positive")

    desired_total = int(target_average) * len(weights)
    lower = 0.0
    upper = maximum_faces / float(np.min(weights)) * 2.0
    for _ in range(80):
        scale = (lower + upper) / 2.0
        total = float(np.clip(weights * scale, minimum_faces, maximum_faces).sum())
        if total < desired_total:
            lower = scale
        else:
            upper = scale

    raw_targets = np.clip(weights * ((lower + upper) / 2.0), minimum_faces, maximum_faces)
    targets = np.rint(raw_targets).astype(np.int64)

    # Rounding is corrected deterministically, without changing the relative
    # ordering of the complexity allocation.
    delta = desired_total - int(targets.sum())
    if delta:
        if delta > 0:
            candidates = np.argsort(-(raw_targets - np.floor(raw_targets)))
            step = 1
            bound = maximum_faces
        else:
            candidates = np.argsort(raw_targets - np.floor(raw_targets))
            step = -1
            bound = minimum_faces
        remaining = abs(delta)
        while remaining:
            changed = False
            for index in candidates:
                if (step > 0 and targets[index] < bound) or (step < 0 and targets[index] > bound):
                    targets[index] += step
                    remaining -= 1
                    changed = True
                    if remaining == 0:
                        break
            if not changed:
                break

    return [int(value) for value in targets]


def adaptive_single_target(
    mesh: trimesh.Trimesh | trimesh.Scene,
    target_average: int = 5_000,
    minimum_faces: int = 2_500,
    maximum_faces: int = 12_000,
) -> int:
    """Estimate a standalone target; batch processing uses global allocation."""

    weight = measure_mesh(mesh).complexity_weight
    return int(np.clip(round(target_average * weight), minimum_faces, maximum_faces))


def _meshset_from_trimesh(mesh: trimesh.Trimesh) -> pymeshlab.MeshSet:
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        ),
        "asset4sim_source",
    )
    return mesh_set


def _trimesh_from_meshset(mesh_set: pymeshlab.MeshSet) -> trimesh.Trimesh:
    current = mesh_set.current_mesh()
    return trimesh.Trimesh(
        vertices=np.asarray(current.vertex_matrix(), dtype=np.float64),
        faces=np.asarray(current.face_matrix(), dtype=np.int64),
        process=False,
    )


def _apply_filter(mesh_set: pymeshlab.MeshSet, name: str, **kwargs: object) -> None:
    try:
        mesh_set.apply_filter(name, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"MeshLab filter {name!r} failed: {exc}") from exc


def repair_mesh(
    mesh: trimesh.Trimesh | trimesh.Scene,
    *,
    fill_holes: bool = True,
    maximum_hole_edges: int = 100_000,
    floater_face_ratio: float = 0.0005,
) -> trimesh.Trimesh:
    """Clean topology and close boundary loops before UV generation."""

    source = as_trimesh(mesh)
    if len(source.faces) < 4:
        raise ValueError("The mesh has too few faces to repair")

    mesh_set = _meshset_from_trimesh(source)
    for filter_name in (
        "meshing_remove_duplicate_vertices",
        "meshing_remove_duplicate_faces",
        "meshing_remove_null_faces",
        "meshing_remove_unreferenced_vertices",
    ):
        _apply_filter(mesh_set, filter_name)

    # Only microscopic disconnected components are removed; meaningful small
    # parts (handles, aerials, brackets) are retained for the complexity pass.
    _apply_filter(
        mesh_set,
        "compute_selection_by_small_disconnected_components_per_face",
        nbfaceratio=float(floater_face_ratio),
    )
    _apply_filter(mesh_set, "compute_selection_transfer_face_to_vertex", inclusive=False)
    _apply_filter(mesh_set, "meshing_remove_selected_vertices_and_faces")

    _apply_filter(mesh_set, "meshing_repair_non_manifold_vertices", vertdispratio=0.0)
    _apply_filter(mesh_set, "meshing_repair_non_manifold_edges", method="Remove Faces")

    if fill_holes:
        _apply_filter(
            mesh_set,
            "meshing_close_holes",
            maxholesize=int(maximum_hole_edges),
            selected=False,
            newfaceselected=False,
            selfintersection=True,
            refinehole=False,
        )

    _apply_filter(mesh_set, "meshing_remove_unreferenced_vertices")
    _apply_filter(mesh_set, "meshing_re_orient_faces_coherently")
    repaired = _trimesh_from_meshset(mesh_set)
    repaired.remove_unreferenced_vertices()
    return repaired


def progressive_face_targets(start_faces: int, target_faces: int, maximum_ratio: float = 2.5) -> list[int]:
    """Build decreasing QEM stages so no pass collapses an extreme face ratio."""

    if start_faces <= target_faces:
        return []
    if target_faces < 4 or maximum_ratio <= 1.0:
        raise ValueError("Invalid progressive simplification contract")
    stages: list[int] = []
    current = int(start_faces)
    while current > target_faces:
        next_target = max(int(target_faces), int(math.ceil(current / maximum_ratio)))
        if next_target >= current:
            next_target = current - 1
        stages.append(next_target)
        current = next_target
    return stages


def _decimate_once(
    mesh: trimesh.Trimesh,
    target_faces: int,
    *,
    preserve_topology: bool,
) -> trimesh.Trimesh:
    mesh_set = _meshset_from_trimesh(mesh)
    common = dict(
        targetfacenum=int(target_faces),
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3.0,
        preservenormal=True,
        optimalplacement=True,
        planarquadric=False,
        qualityweight=False,
        autoclean=True,
        selected=False,
    )
    _apply_filter(
        mesh_set,
        "meshing_decimation_quadric_edge_collapse",
        preservetopology=bool(preserve_topology),
        **common,
    )
    current_faces = int(mesh_set.current_mesh().face_number())
    if preserve_topology and current_faces > max(target_faces + 500, round(target_faces * 1.20)):
        _apply_filter(
            mesh_set,
            "meshing_decimation_quadric_edge_collapse",
            preservetopology=False,
            **common,
        )
    _apply_filter(mesh_set, "meshing_remove_unreferenced_vertices")
    result = _trimesh_from_meshset(mesh_set)
    result.remove_unreferenced_vertices()
    return result


def simplify_mesh(
    mesh: trimesh.Trimesh | trimesh.Scene,
    target_faces: int,
    *,
    preserve_topology: bool = True,
    maximum_stage_ratio: float = 2.5,
) -> trimesh.Trimesh:
    """Progressive quality-weighted QEM simplification before UV unwrapping."""

    source = as_trimesh(mesh)
    if len(source.faces) <= target_faces:
        return source

    result = source
    stages = progressive_face_targets(len(source.faces), target_faces, maximum_stage_ratio)
    for stage_index, stage_target in enumerate(stages, start=1):
        before_faces = len(result.faces)
        result = _decimate_once(result, stage_target, preserve_topology=preserve_topology)
        LOG.info(
            "Progressive retopo stage %s/%s: %s -> %s faces (target %s)",
            stage_index,
            len(stages),
            before_faces,
            len(result.faces),
            stage_target,
        )
    return result


def _surface_samples(
    mesh: trimesh.Trimesh,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if not np.all(np.isfinite(areas)) or areas.sum() <= 0:
        raise ValueError("Cannot sample a mesh with invalid face areas")
    face_indices = rng.choice(len(mesh.faces), size=count, replace=True, p=areas / areas.sum())
    triangles = np.asarray(mesh.triangles[face_indices], dtype=np.float64)
    first = rng.random(count)
    second = rng.random(count)
    root = np.sqrt(first)
    points = (
        (1.0 - root)[:, None] * triangles[:, 0]
        + (root * (1.0 - second))[:, None] * triangles[:, 1]
        + (root * second)[:, None] * triangles[:, 2]
    )
    normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float64)
    return points, normals


def compare_geometry_quality(
    reference: trimesh.Trimesh | trimesh.Scene,
    candidate: trimesh.Trimesh | trimesh.Scene,
    *,
    sample_count: int = 20_000,
    maximum_normalized_p95_distance: float = 0.012,
    maximum_normalized_p99_distance: float = 0.030,
    maximum_normal_angle_p95_degrees: float = 35.0,
    maximum_normalized_bounds_drift: float = 0.010,
) -> GeometryQuality:
    """Approximate bidirectional surface error and normal preservation."""

    source = as_trimesh(reference)
    result = as_trimesh(candidate)
    diagonal = float(np.linalg.norm(source.bounds[1] - source.bounds[0]))
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Reference mesh has an invalid bounding box")

    source_points, source_normals = _surface_samples(source, sample_count, seed=0xA55E7)
    result_points, _ = _surface_samples(result, sample_count, seed=0xB16B00B5)
    _, source_distances, result_triangles = trimesh.proximity.closest_point(result, source_points)
    _, result_distances, _ = trimesh.proximity.closest_point(source, result_points)
    distances = np.concatenate([source_distances, result_distances]) / diagonal

    nearest_normals = np.asarray(result.face_normals[result_triangles], dtype=np.float64)
    dots = np.abs(np.einsum("ij,ij->i", source_normals, nearest_normals))
    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    bounds_drift = float(np.max(np.abs(source.bounds - result.bounds)) / diagonal)
    p95 = float(np.quantile(distances, 0.95))
    p99 = float(np.quantile(distances, 0.99))
    max_distance = float(np.max(distances))
    normal_p95 = float(np.quantile(angles, 0.95))
    passed = bool(
        p95 <= maximum_normalized_p95_distance
        and p99 <= maximum_normalized_p99_distance
        and normal_p95 <= maximum_normal_angle_p95_degrees
        and bounds_drift <= maximum_normalized_bounds_drift
        and boundary_edge_count(result) == 0
    )
    return GeometryQuality(
        normalized_p95_distance=p95,
        normalized_p99_distance=p99,
        normalized_max_distance=max_distance,
        normal_angle_p95_degrees=normal_p95,
        normalized_bounds_drift=bounds_drift,
        passed=passed,
    )


def simplify_with_quality_gate(
    mesh: trimesh.Trimesh | trimesh.Scene,
    target_faces: int,
    *,
    maximum_faces: int,
    preserve_topology: bool = True,
    quality_samples: int = 20_000,
) -> tuple[trimesh.Trimesh, list[dict[str, object]]]:
    """Promote only assets whose progressive low-poly result fails QA."""

    source = as_trimesh(mesh)
    attempts: list[dict[str, object]] = []
    attempt_target = min(int(target_faces), int(maximum_faces))
    while True:
        candidate = simplify_mesh(source, attempt_target, preserve_topology=preserve_topology)
        quality = compare_geometry_quality(source, candidate, sample_count=quality_samples)
        attempts.append(
            {
                "target_faces": attempt_target,
                "actual_faces": int(len(candidate.faces)),
                "quality": asdict(quality),
            }
        )
        if quality.passed:
            return candidate, attempts
        if attempt_target >= maximum_faces:
            raise RuntimeError(
                "Geometry quality gate failed at the maximum budget "
                f"({maximum_faces} faces): {asdict(quality)}"
            )
        attempt_target = min(maximum_faces, max(attempt_target + 500, int(math.ceil(attempt_target * 1.35))))


def repair_and_retopologize(
    mesh: trimesh.Trimesh | trimesh.Scene,
    *,
    target_faces: int | None = None,
    target_average: int = 5_000,
    minimum_faces: int = 2_500,
    maximum_faces: int = 20_000,
    fill_holes: bool = True,
    preserve_topology: bool = True,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    before = measure_mesh(mesh)
    repaired = repair_mesh(mesh, fill_holes=fill_holes)
    after_repair = measure_mesh(repaired)
    chosen_target = target_faces or adaptive_single_target(
        repaired,
        target_average=target_average,
        minimum_faces=minimum_faces,
        maximum_faces=maximum_faces,
    )
    chosen_target = int(np.clip(chosen_target, minimum_faces, maximum_faces))
    result, quality_attempts = simplify_with_quality_gate(
        repaired,
        chosen_target,
        maximum_faces=maximum_faces,
        preserve_topology=preserve_topology,
    )
    final = measure_mesh(result)
    report: dict[str, object] = {
        "before": asdict(before),
        "after_repair": asdict(after_repair),
        "target_faces": chosen_target,
        "accepted_faces": int(len(result.faces)),
        "quality_attempts": quality_attempts,
        "final": asdict(final),
        "holes_closed": max(0, before.boundary_edges - after_repair.boundary_edges),
    }
    return result, report


def _iter_mesh_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".glb", ".gltf", ".obj", ".ply"}:
            yield path


def process_batch(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    files = list(_iter_mesh_files(input_dir))
    if not files:
        raise ValueError(f"No supported meshes found under {input_dir}")

    repaired_paths: list[Path] = []
    reports: list[dict[str, object]] = []
    repair_cache = output_dir / "_repaired_cache"
    for path in files:
        loaded = trimesh.load(path, force="scene")
        before = measure_mesh(loaded)
        repaired = repair_mesh(loaded, fill_holes=not args.no_fill_holes)
        repaired_metrics = measure_mesh(repaired)
        relative = path.relative_to(input_dir)
        repaired_path = repair_cache / relative.with_suffix(".ply")
        repaired_path.parent.mkdir(parents=True, exist_ok=True)
        repaired.export(repaired_path, file_type="ply")
        repaired_paths.append(repaired_path)
        reports.append(
            {
                "asset_id": relative.parts[0],
                "source": str(path),
                "relative_source": relative.as_posix(),
                "repaired_cache": str(repaired_path),
                "before": asdict(before),
                "after_repair": asdict(repaired_metrics),
                "holes_closed": max(0, before.boundary_edges - repaired_metrics.boundary_edges),
            }
        )

    targets = allocate_face_targets(
        [float(item["after_repair"]["complexity_weight"]) for item in reports],
        target_average=args.target_average,
        minimum_faces=args.minimum_faces,
        maximum_faces=args.maximum_faces,
    )

    for path, repaired_path, target, report in zip(files, repaired_paths, targets, reports, strict=True):
        repaired = as_trimesh(trimesh.load(repaired_path, force="mesh"))
        simplified, quality_attempts = simplify_with_quality_gate(
            repaired,
            target,
            maximum_faces=args.maximum_faces,
            preserve_topology=not args.no_preserve_topology,
            quality_samples=args.quality_samples,
        )
        relative = path.relative_to(input_dir).with_suffix(".glb")
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        simplified.export(destination, file_type="glb")
        report["target_faces"] = target
        report["accepted_faces"] = int(len(simplified.faces))
        report["quality_attempts"] = quality_attempts
        report["final"] = asdict(measure_mesh(simplified))
        report["output"] = str(destination)
        report["output_bytes"] = destination.stat().st_size
        LOG.info("%s -> %s faces (target %s)", relative, len(simplified.faces), target)

    manifest = {
        "contract": {
            "target_average_faces": args.target_average,
            "minimum_faces": args.minimum_faces,
            "maximum_faces": args.maximum_faces,
            "fill_holes": not args.no_fill_holes,
            "preserve_topology": not args.no_preserve_topology,
            "progressive_maximum_stage_ratio": 2.5,
            "quality_samples": args.quality_samples,
        },
        "asset_count": len(reports),
        "allocated_average_faces": sum(targets) / len(targets),
        "final_average_faces": sum(int(item["final"]["faces"]) for item in reports) / len(reports),
        "assets": reports,
    }
    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-average", type=int, default=5_000)
    parser.add_argument("--minimum-faces", type=int, default=2_500)
    parser.add_argument("--maximum-faces", type=int, default=20_000)
    parser.add_argument("--quality-samples", type=int, default=20_000)
    parser.add_argument("--no-fill-holes", action="store_true")
    parser.add_argument("--no-preserve-topology", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    return process_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())

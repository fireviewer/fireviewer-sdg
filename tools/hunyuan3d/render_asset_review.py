#!/usr/bin/env python3
"""Render manifest-indexed GLBs into individual visual QA sheets.

The renderer resolves assets by their immutable manifest ``asset_id`` instead
of relying on directory ordering. Each capture contains two generated-model
views, the source reference image, and basic GLB/USD exploitability metrics.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFont, ImageOps


CAPTURE_WIDTH = 1600
CAPTURE_HEIGHT = 900
VIEW_SIZE = 620
BACKGROUND_RGB = np.array([31, 36, 45], dtype=np.uint8)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\seguisb.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(30, bold=True)
FONT_SUBTITLE = font(21)
FONT_LABEL = font(18, bold=True)
FONT_BODY = font(17)
FONT_SMALL = font(15)


def index_files(paths: Iterable[Path], extension: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for root in paths:
        if not root.is_dir():
            continue
        for path in root.rglob(f"*.{extension}"):
            asset_id = path.stem
            if asset_id in result:
                duplicates.setdefault(asset_id, [result[asset_id]]).append(path)
            else:
                result[asset_id] = path
    if duplicates:
        details = "; ".join(f"{asset_id}: {items}" for asset_id, items in sorted(duplicates.items()))
        raise RuntimeError(f"ambiguous {extension.upper()} assets: {details}")
    return result


def texture_size(material: Any) -> list[int] | None:
    for name in ("baseColorTexture", "image"):
        image = getattr(material, name, None)
        if image is not None and getattr(image, "size", None):
            return [int(image.size[0]), int(image.size[1])]
    return None


def inspect_scene(scene: trimesh.Scene, glb_path: Path, usd_path: Path | None) -> dict[str, Any]:
    meshes = [mesh for mesh in scene.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("GLB contains no triangle mesh")

    vertices = sum(len(mesh.vertices) for mesh in meshes)
    faces = sum(len(mesh.faces) for mesh in meshes)
    uv_meshes = 0
    textured_meshes = 0
    texture_sizes: list[list[int]] = []
    finite_vertices = True
    for mesh in meshes:
        finite_vertices = finite_vertices and bool(np.all(np.isfinite(mesh.vertices)))
        uv = getattr(mesh.visual, "uv", None)
        if uv is not None and len(uv) == len(mesh.vertices) and np.all(np.isfinite(uv)):
            uv_meshes += 1
        material = getattr(mesh.visual, "material", None)
        size = texture_size(material) if material is not None else None
        if size:
            textured_meshes += 1
            texture_sizes.append(size)

    bounds = np.asarray(scene.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0] if bounds.shape == (2, 3) else np.zeros(3)
    bounds_valid = bool(
        bounds.shape == (2, 3)
        and np.all(np.isfinite(bounds))
        and np.max(extents) > 1e-9
        and np.all(extents >= 0)
    )
    nonzero_extents = extents[extents > 1e-9]
    aspect_ratio = float(np.max(nonzero_extents) / np.min(nonzero_extents)) if len(nonzero_extents) else None
    return {
        "glb_path": str(glb_path.resolve()),
        "glb_bytes": glb_path.stat().st_size,
        "usd_path": str(usd_path.resolve()) if usd_path else None,
        "usd_exists": bool(usd_path and usd_path.is_file() and usd_path.stat().st_size > 0),
        "usd_bytes": usd_path.stat().st_size if usd_path and usd_path.is_file() else 0,
        "meshes": len(meshes),
        "vertices": vertices,
        "faces": faces,
        "finite_vertices": finite_vertices,
        "uv_meshes": uv_meshes,
        "textured_meshes": textured_meshes,
        "texture_sizes": texture_sizes,
        "bounds": bounds.tolist() if bounds.shape == (2, 3) else None,
        "extents": extents.tolist() if bounds.shape == (2, 3) else None,
        "bounds_valid": bounds_valid,
        "aspect_ratio": aspect_ratio,
    }


def normalized_scene(scene: trimesh.Scene) -> trimesh.Scene:
    result = scene.copy()
    bounds = np.asarray(result.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError("invalid scene bounds")
    center = (bounds[0] + bounds[1]) * 0.5
    max_extent = float(np.max(bounds[1] - bounds[0]))
    if max_extent <= 1e-9:
        raise ValueError("degenerate scene bounds")
    scale = 2.15 / max_extent
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale
    transform[:3, 3] = -center * scale
    result.apply_transform(transform)
    return result


def camera_pose(eye: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    target = np.zeros(3, dtype=np.float64) if target is None else target
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def build_render_scene(scene: trimesh.Scene) -> tuple[pyrender.Scene, pyrender.Node, list[pyrender.Node]]:
    render_scene = pyrender.Scene.from_trimesh_scene(
        normalized_scene(scene),
        bg_color=np.append(BACKGROUND_RGB.astype(np.float32) / 255.0, 1.0),
        ambient_light=np.array([0.42, 0.42, 0.42, 1.0], dtype=np.float32),
    )
    camera = pyrender.PerspectiveCamera(yfov=math.radians(38.0), znear=0.05, zfar=100.0)
    camera_node = render_scene.add(camera, pose=np.eye(4))
    lights: list[pyrender.Node] = []
    for intensity in (3.3, 1.6, 1.0):
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
        lights.append(render_scene.add(light, pose=np.eye(4)))
    return render_scene, camera_node, lights


def render_view(
    renderer: pyrender.OffscreenRenderer,
    render_scene: pyrender.Scene,
    camera_node: pyrender.Node,
    light_nodes: list[pyrender.Node],
    eye: tuple[float, float, float],
) -> tuple[Image.Image, float, float]:
    eye_array = np.asarray(eye, dtype=np.float64)
    pose = camera_pose(eye_array)
    render_scene.set_pose(camera_node, pose)
    render_scene.set_pose(light_nodes[0], pose)
    render_scene.set_pose(light_nodes[1], camera_pose(-eye_array + np.array([0.0, 1.0, 0.0])))
    render_scene.set_pose(light_nodes[2], camera_pose(np.array([0.0, 4.0, 0.5])))
    color, depth = renderer.render(render_scene, flags=pyrender.RenderFlags.RGBA)
    visible = depth > 0
    coverage = float(np.count_nonzero(visible) / visible.size)
    variance = float(np.std(color[:, :, :3][visible])) if np.any(visible) else 0.0
    return Image.fromarray(color, mode="RGBA").convert("RGB"), coverage, variance


def fit_reference(path: Path | None, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, (25, 29, 36))
    if path is None or not path.is_file():
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 18), "Référence absente", font=FONT_BODY, fill=(240, 120, 120))
        return canvas
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((size[0] - 12, size[1] - 12), Image.Resampling.LANCZOS)
        x = (size[0] - image.width) // 2
        y = (size[1] - image.height) // 2
        canvas.paste(image, (x, y))
    return canvas


def status_for(metrics: dict[str, Any], views: list[dict[str, float]]) -> tuple[str, list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    if metrics["faces"] <= 0 or metrics["vertices"] <= 0:
        failures.append("géométrie vide")
    if not metrics["finite_vertices"] or not metrics["bounds_valid"]:
        failures.append("coordonnées ou limites invalides")
    if not metrics["usd_exists"]:
        failures.append("USD absent ou vide")
    if metrics["uv_meshes"] != metrics["meshes"]:
        failures.append("UV absents sur au moins un mesh")
    if metrics["textured_meshes"] != metrics["meshes"]:
        failures.append("texture absente sur au moins un mesh")
    if any(view["coverage"] < 0.01 for view in views):
        failures.append("objet non visible dans une vue")
    if any(view["coverage"] > 0.92 for view in views):
        failures.append("objet probablement cadré hors champ")
    if any(view["variance"] < 3.0 for view in views):
        failures.append("rendu sans contraste exploitable")
    if metrics["faces"] > 50_000:
        failures.append("plus de 50 000 faces")
    elif metrics["faces"] > 10_000:
        warnings.append("complexité élevée (>10k faces)")
    elif metrics["faces"] > 5_000:
        warnings.append("au-dessus de la cible 5k")
    if metrics["aspect_ratio"] and metrics["aspect_ratio"] > 250:
        warnings.append("proportions extrêmement allongées")
    if failures:
        return "fail", failures, warnings
    if warnings:
        return "warning", failures, warnings
    return "pass", failures, warnings


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    width: int,
    *,
    font_value: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    spacing: int = 5,
) -> int:
    average = max(1, int(width / max(1, draw.textlength("M", font=font_value))))
    lines = textwrap.wrap(text, width=average, break_long_words=False, break_on_hyphens=False) or [""]
    y = xy[1]
    for line in lines:
        draw.text((xy[0], y), line, font=font_value, fill=fill)
        box = draw.textbbox((xy[0], y), line or "M", font=font_value)
        y += box[3] - box[1] + spacing
    return y


def compose_capture(
    asset: dict[str, Any],
    views: list[Image.Image],
    metrics: dict[str, Any] | None,
    view_metrics: list[dict[str, float]],
    status: str,
    failures: list[str],
    warnings: list[str],
    error: str | None,
) -> Image.Image:
    canvas = Image.new("RGB", (CAPTURE_WIDTH, CAPTURE_HEIGHT), (16, 20, 27))
    draw = ImageDraw.Draw(canvas)
    status_color = {"pass": (77, 205, 128), "warning": (244, 183, 64), "fail": (239, 92, 92)}[status]
    draw.rectangle((0, 0, CAPTURE_WIDTH, 76), fill=(23, 28, 36))
    draw.rectangle((0, 0, 12, CAPTURE_HEIGHT), fill=status_color)
    title = Path(asset["source"]).stem if asset.get("source") else asset["asset_id"]
    draw.text((30, 18), f"{asset['index']:03d} — {title}", font=FONT_TITLE, fill=(242, 245, 250))
    draw.text((1225, 24), status.upper(), font=FONT_TITLE, fill=status_color)

    panel_positions = [(24, 104), (664, 104)]
    labels = ("Vue 3/4", "Vue opposée")
    for position, label, image in zip(panel_positions, labels, views):
        fitted = ImageOps.fit(image, (VIEW_SIZE, VIEW_SIZE), method=Image.Resampling.LANCZOS)
        canvas.paste(fitted, position)
        draw.rectangle((position[0], position[1], position[0] + VIEW_SIZE, position[1] + VIEW_SIZE), outline=(79, 89, 104), width=2)
        draw.rectangle((position[0] + 12, position[1] + 12, position[0] + 150, position[1] + 45), fill=(10, 12, 16))
        draw.text((position[0] + 22, position[1] + 17), label, font=FONT_LABEL, fill=(235, 238, 244))

    reference = fit_reference(Path(asset["source"]) if asset.get("source") else None, (270, 270))
    canvas.paste(reference, (1308, 104))
    draw.rectangle((1308, 104, 1578, 374), outline=(79, 89, 104), width=2)
    draw.text((1320, 114), "Référence", font=FONT_LABEL, fill=(235, 238, 244), stroke_width=2, stroke_fill=(10, 12, 16))

    info_x = 1308
    y = 396
    draw.text((info_x, y), "Contrôles", font=FONT_LABEL, fill=(235, 238, 244))
    y += 34
    if metrics is not None:
        info_lines = [
            f"Meshes : {metrics['meshes']}",
            f"Sommets : {metrics['vertices']:,}".replace(",", " "),
            f"Faces : {metrics['faces']:,}".replace(",", " "),
            f"UV : {metrics['uv_meshes']}/{metrics['meshes']}",
            f"Textures : {metrics['textured_meshes']}/{metrics['meshes']}",
            f"GLB : {metrics['glb_bytes'] / 1_048_576:.1f} Mio",
            f"USD : {'oui' if metrics['usd_exists'] else 'NON'}",
            f"Couverture : {view_metrics[0]['coverage'] * 100:.1f}% / {view_metrics[1]['coverage'] * 100:.1f}%",
        ]
        for line in info_lines:
            draw.text((info_x, y), line, font=FONT_BODY, fill=(210, 216, 226))
            y += 27
    if error:
        y = draw_wrapped(draw, error, (info_x, y + 8), 260, font_value=FONT_SMALL, fill=(244, 135, 135))

    messages = failures + warnings
    if messages:
        y += 8
        draw.text((info_x, y), "À contrôler", font=FONT_LABEL, fill=status_color)
        y += 30
        for message in messages:
            y = draw_wrapped(draw, f"• {message}", (info_x, y), 260, font_value=FONT_SMALL, fill=(230, 218, 200))
    else:
        draw.text((info_x, y + 8), "Contrôles automatiques OK", font=FONT_BODY, fill=status_color)

    draw.rectangle((24, 748, 1284, 876), fill=(22, 27, 35), outline=(57, 66, 78), width=1)
    draw.text((40, 765), asset["asset_id"], font=FONT_LABEL, fill=(225, 231, 241))
    draw_wrapped(draw, str(asset.get("glb_path") or "GLB absent"), (40, 803), 1220, font_value=FONT_SMALL, fill=(155, 167, 184))
    draw_wrapped(draw, str(asset.get("usd_path") or "USD absent"), (40, 836), 1220, font_value=FONT_SMALL, fill=(155, 167, 184))
    return canvas


def failure_views(message: str) -> list[Image.Image]:
    result: list[Image.Image] = []
    for _ in range(2):
        image = Image.new("RGB", (VIEW_SIZE, VIEW_SIZE), tuple(BACKGROUND_RGB.tolist()))
        draw = ImageDraw.Draw(image)
        draw.text((30, 30), "RENDU IMPOSSIBLE", font=FONT_TITLE, fill=(239, 92, 92))
        draw_wrapped(draw, message, (30, 90), VIEW_SIZE - 60, font_value=FONT_BODY, fill=(235, 205, 205))
        result.append(image)
    return result


def render_asset(renderer: pyrender.OffscreenRenderer, asset: dict[str, Any], capture_path: Path) -> dict[str, Any]:
    error: str | None = None
    metrics: dict[str, Any] | None = None
    view_metrics: list[dict[str, float]] = []
    failures: list[str] = []
    warnings: list[str] = []
    status = "fail"
    try:
        glb_path = Path(asset["glb_path"])
        scene = trimesh.load(glb_path, force="scene", process=False)
        if not isinstance(scene, trimesh.Scene):
            scene = trimesh.Scene(scene)
        metrics = inspect_scene(scene, glb_path, Path(asset["usd_path"]) if asset.get("usd_path") else None)
        render_scene, camera_node, lights = build_render_scene(scene)
        rendered_views: list[Image.Image] = []
        for eye in ((3.2, 2.25, 3.2), (-3.0, 1.85, -3.25)):
            image, coverage, variance = render_view(renderer, render_scene, camera_node, lights, eye)
            rendered_views.append(image)
            view_metrics.append({"coverage": coverage, "variance": variance})
        status, failures, warnings = status_for(metrics, view_metrics)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        rendered_views = failure_views(error)
        failures = ["chargement ou rendu impossible"]

    capture = compose_capture(asset, rendered_views, metrics, view_metrics, status, failures, warnings, error)
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    capture.save(capture_path, format="PNG", optimize=True)
    return {
        **asset,
        "capture_path": str(capture_path.resolve()),
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "error": error,
        "metrics": metrics,
        "views": view_metrics,
    }


def status_color(status: str) -> tuple[int, int, int]:
    return {"pass": (77, 205, 128), "warning": (244, 183, 64), "fail": (239, 92, 92)}[status]


def make_contact_sheet(
    results: list[dict[str, Any]],
    path: Path,
    *,
    columns: int = 6,
    tile_width: int = 260,
    tile_height: int = 178,
) -> None:
    rows = math.ceil(len(results) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (13, 16, 22))
    draw = ImageDraw.Draw(sheet)
    for position, result in enumerate(results):
        row, column = divmod(position, columns)
        x, y = column * tile_width, row * tile_height
        image_width = tile_width - 10
        image_height = int(round(image_width * 9 / 16))
        with Image.open(result["capture_path"]) as source:
            thumb = ImageOps.fit(source.convert("RGB"), (image_width, image_height), method=Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x + 5, y + 5))
        color = status_color(result["status"])
        label_top = y + image_height + 8
        draw.rectangle((x + 5, label_top, x + tile_width - 5, y + tile_height - 5), fill=(22, 27, 35))
        draw.rectangle((x + 5, label_top, x + 11, y + tile_height - 5), fill=color)
        label_length = 27 if tile_width <= 260 else 45
        label = f"{result['index']:03d} {Path(result['source']).stem[:label_length]}"
        draw.text((x + 17, label_top + 3), label, font=FONT_SMALL, fill=(225, 231, 241))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG", optimize=True)


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index", "asset_id", "status", "faces", "vertices", "meshes", "textured_meshes",
        "uv_meshes", "glb_path", "usd_path", "capture_path", "failures", "warnings", "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in results:
            metrics = item.get("metrics") or {}
            writer.writerow({
                "index": item["index"],
                "asset_id": item["asset_id"],
                "status": item["status"],
                "faces": metrics.get("faces"),
                "vertices": metrics.get("vertices"),
                "meshes": metrics.get("meshes"),
                "textured_meshes": metrics.get("textured_meshes"),
                "uv_meshes": metrics.get("uv_meshes"),
                "glb_path": item.get("glb_path"),
                "usd_path": item.get("usd_path"),
                "capture_path": item.get("capture_path"),
                "failures": " | ".join(item.get("failures", [])),
                "warnings": " | ".join(item.get("warnings", [])),
                "error": item.get("error") or "",
            })


def write_html(results: list[dict[str, Any]], path: Path) -> None:
    first_index = results[0]["index"] if results else 0
    last_index = results[-1]["index"] if results else 0
    cards = []
    html_root = path.parent.resolve()
    for item in results:
        capture = Path(item["capture_path"]).resolve().relative_to(html_root).as_posix()
        title = Path(item["source"]).stem
        cards.append(
            f'<a class="card {html.escape(item["status"])}" href="{html.escape(capture)}">'
            f'<img loading="lazy" src="{html.escape(capture)}" alt="Asset {item["index"]:03d}">'
            f'<span>{item["index"]:03d} — {html.escape(title)}</span></a>'
        )
    document = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Asset4Sim — revue visuelle {first_index:03d}–{last_index:03d}</title>
<style>
body{{margin:0;background:#0d1117;color:#e6edf3;font:16px Segoe UI,Arial,sans-serif}}
header{{position:sticky;top:0;z-index:2;background:#171d26;padding:16px 24px;border-bottom:1px solid #303846}}
h1{{font-size:22px;margin:0 0 5px}} p{{margin:0;color:#9facbd}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;padding:18px}}
.card{{display:block;color:#e6edf3;text-decoration:none;background:#171d26;border:2px solid #303846;border-radius:8px;overflow:hidden}}
.card.pass{{border-color:#4dcd80}} .card.warning{{border-color:#f4b740}} .card.fail{{border-color:#ef5c5c}}
.card img{{display:block;width:100%;aspect-ratio:16/9;object-fit:cover}} .card span{{display:block;padding:10px 12px}}
</style></head><body><header><h1>Asset4Sim — captures {first_index:03d} à {last_index:03d}</h1>
<p>Cliquer sur une fiche pour ouvrir la capture complète. Vert : OK, orange : avertissement, rouge : échec.</p></header>
<main>{''.join(cards)}</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def resolve_assets(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    indexed_items = [
        (int(item.get("index", ordinal)), item)
        for ordinal, item in enumerate(manifest["assets"], start=1)
    ]
    requested = [(index, item) for index, item in indexed_items if args.start <= index <= args.end]
    if len({index for index, _ in requested}) != len(requested):
        raise RuntimeError("manifest contains duplicate asset indices")
    glbs = index_files(args.glb_roots, "glb")
    usds = index_files(args.usd_roots, "usd")
    assets: list[dict[str, Any]] = []
    for index, item in requested:
        asset_id = item["asset_id"]
        glb_path = glbs.get(asset_id)
        if glb_path is None:
            raise FileNotFoundError(f"GLB missing for manifest asset {index}: {asset_id}")
        assets.append({
            "index": index,
            "asset_id": asset_id,
            "source": item.get("source", item.get("source_reference")),
            "glb_path": str(glb_path.resolve()),
            "usd_path": str(usds[asset_id].resolve()) if asset_id in usds else None,
        })
    uses_explicit_indices = any("index" in item for item in manifest["assets"])
    if not uses_explicit_indices and len(assets) != args.end - args.start + 1:
        raise RuntimeError(f"expected {args.end - args.start + 1} assets, resolved {len(assets)}")
    return assets


def run(args: argparse.Namespace) -> int:
    assets = resolve_assets(args)
    captures_root = args.output_dir / "captures"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(viewport_width=VIEW_SIZE, viewport_height=VIEW_SIZE, point_size=1.0)
    results: list[dict[str, Any]] = []
    try:
        for asset in assets:
            capture_path = captures_root / f"{asset['index']:03d}_{asset['asset_id']}.png"
            if capture_path.exists() and not args.overwrite:
                raise FileExistsError(f"capture already exists (use --overwrite): {capture_path}")
            result = render_asset(renderer, asset, capture_path)
            results.append(result)
            print(json.dumps({
                "index": result["index"],
                "asset_id": result["asset_id"],
                "status": result["status"],
                "capture": result["capture_path"],
            }, ensure_ascii=False), flush=True)
    finally:
        renderer.delete()

    summary = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "start": args.start,
        "end": args.end,
        "asset_count": len(results),
        "pass_count": sum(item["status"] == "pass" for item in results),
        "warning_count": sum(item["status"] == "warning" for item in results),
        "fail_count": sum(item["status"] == "fail" for item in results),
        "captures_root": str(captures_root.resolve()),
        "assets": results,
    }
    report = args.output_dir / "asset-review-report.json"
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(results, args.output_dir / "asset-review-report.csv")
    make_contact_sheet(results, args.output_dir / "contact-sheet-all.png")
    for offset in range(0, len(results), 24):
        page = results[offset : offset + 24]
        make_contact_sheet(
            page,
            args.output_dir / f"contact-sheet-{page[0]['index']:03d}-{page[-1]['index']:03d}.png",
            columns=4,
            tile_width=400,
            tile_height=250,
        )
    write_html(results, args.output_dir / "index.html")
    print(json.dumps({key: summary[key] for key in ("asset_count", "pass_count", "warning_count", "fail_count")}), flush=True)
    return 1 if summary["fail_count"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--glb-root", dest="glb_roots", type=Path, action="append", required=True)
    parser.add_argument("--usd-root", dest="usd_roots", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=102)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.start < 1 or args.end < args.start:
        raise ValueError("invalid start/end range")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

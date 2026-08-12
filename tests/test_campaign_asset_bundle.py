from __future__ import annotations

import copy
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg import asset_bundle, campaign_asset_bundle  # noqa: E402
from fireviewer_sdg.simready_assets import (  # noqa: E402
    MANIFEST_PROFILE,
    MANIFEST_SCHEMA_VERSION,
    PHOTOREAL_FAMILY_MINIMUMS,
    PHOTOREAL_LIBRARY_POLICY,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_png_header(
    path: Path,
    *,
    width: int = 4096,
    height: int = 4096,
    identity: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + identity.encode("utf-8")
    )


def _write_official_manifest(
    *,
    volume: Path,
    strategy: str = "native_variant_set",
    level_count: int = 3,
) -> Path:
    input_root = volume / "input"
    wrapper_root = input_root / "nvidia-simready-lock"
    cache_root = input_root / "nvidia-asset-cache"
    environment: dict[str, dict[str, list[dict[str, object]]]] = {
        "vegetation": {},
        "buildings": {},
    }
    serial = 0
    for kind, family_minimums in PHOTOREAL_FAMILY_MINIMUMS.items():
        for family, minimum in family_minimums.items():
            entries: list[dict[str, object]] = []
            for index in range(minimum):
                serial += 1
                stem = f"{kind}-{family}-{index:02d}"
                source = cache_root / stem / f"{stem}.usda"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(
                    "#usda 1.0\n"
                    '(\n    defaultPrim = "World"\n)\n'
                    'def Xform "World" {}\n',
                    encoding="utf-8",
                )
                dependency = _record(source, root=volume)
                wrapper = wrapper_root / f"{stem}.usda"
                wrapper.parent.mkdir(parents=True, exist_ok=True)
                wrapper.write_text(
                    "#usda 1.0\n"
                    '(\n    defaultPrim = "Asset"\n)\n'
                    'def Xform "Asset" (\n'
                    f"    prepend references = "
                    f"@../nvidia-asset-cache/{stem}/{stem}.usda@\n"
                    ")\n{}\n",
                    encoding="utf-8",
                )
                if strategy == "source_default_only":
                    levels = ["/World"]
                elif strategy == "native_prim_hierarchy":
                    levels = [f"/World/LOD{lod}" for lod in range(level_count)]
                else:
                    levels = [
                        f"/World:lodVariant=LOD{lod}"
                        for lod in range(level_count)
                    ]
                asset_id = f"{kind}.{family}:{serial:03d}"
                entries.append(
                    {
                        "asset_id": asset_id,
                        "family": f"{kind}.{family}",
                        "identity": {
                            "source_name": stem,
                            "source_identity": f"nvidia/test/{stem}",
                        },
                        "path": wrapper.relative_to(input_root).as_posix(),
                        "sha256": _sha256(wrapper),
                        "source_cache_path": source.relative_to(input_root).as_posix(),
                        # The live NVIDIA materializer writes volume-relative
                        # dependency paths, not manifest-parent-relative paths.
                        "materialized_files": [dependency],
                        "content_lock_sha256": hashlib.sha256(
                            json.dumps([dependency], sort_keys=True).encode("utf-8")
                        ).hexdigest(),
                        "quality_validation": "native_metadata_passed",
                        "anchor_validation": {"state": "passed"},
                        "materials": {"state": "passed"},
                        "ground_anchor_m": [0.0, 0.0, 0.0],
                        "lod": {
                            "state": "passed",
                            "strategy": strategy,
                            "levels": levels,
                            "level_count": len(levels),
                        },
                    }
                )
            environment[kind][family] = entries
    actor_template = environment["buildings"]["habitat"][0]
    selected_actor_assets: dict[str, dict[str, object]] = {}
    for selection_id in campaign_asset_bundle.SELECTED_ACTOR_GROUP_IDS:
        entry = copy.deepcopy(actor_template)
        entry["asset_id"] = f"selected-actor.{selection_id}:test"
        entry["family"] = f"selected-actors.{selection_id}"
        entry["identity"] = {
            "source_name": selection_id,
            "source_identity": f"sketchfab/test/{selection_id}",
        }
        selected_actor_assets[selection_id] = entry
    actors: dict[str, dict[str, object]] = {}
    for class_id in asset_bundle.REQUIRED_ACTOR_CLASSES:
        entry = copy.deepcopy(actor_template)
        entry["asset_id"] = f"actors.{class_id}:test"
        entry["family"] = f"actors.{class_id}"
        entry["identity"] = {
            "source_name": class_id,
            "source_identity": f"curated/test/actors/{class_id}",
        }
        actors[class_id] = entry
    selected_environment_assets: dict[str, dict[str, object]] = {}
    for selection_id in campaign_asset_bundle.SELECTED_ENVIRONMENT_GROUP_IDS:
        entry = copy.deepcopy(actor_template)
        entry["asset_id"] = f"selected-environment.{selection_id}:test"
        entry["family"] = f"selected-environment.{selection_id}"
        entry["identity"] = {
            "source_name": selection_id,
            "source_identity": f"sketchfab/test/{selection_id}",
        }
        selected_environment_assets[selection_id] = entry
    manifest = input_root / "simready-assets-hd-v3.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "profile": MANIFEST_PROFILE,
                "library_policy": PHOTOREAL_LIBRARY_POLICY,
                "family_minimums": PHOTOREAL_FAMILY_MINIMUMS,
                "discovery": {
                    "mode": "materialized_photoreal_asset_library_v3",
                    "missing_environment": [],
                },
                "environment": environment,
                "selected_actor_group": {
                    "group_id": campaign_asset_bundle.SELECTED_ACTOR_GROUP_ID,
                    "selection_count": len(
                        campaign_asset_bundle.SELECTED_ACTOR_GROUP_IDS
                    ),
                    "selection_order": list(
                        campaign_asset_bundle.SELECTED_ACTOR_GROUP_IDS
                    ),
                    "assets": selected_actor_assets,
                },
                "actors": actors,
                "selected_environment_group": {
                    "group_id": (
                        campaign_asset_bundle.SELECTED_ENVIRONMENT_GROUP_ID
                    ),
                    "selection_count": len(
                        campaign_asset_bundle.SELECTED_ENVIRONMENT_GROUP_IDS
                    ),
                    "selection_order": list(
                        campaign_asset_bundle.SELECTED_ENVIRONMENT_GROUP_IDS
                    ),
                    "assets": selected_environment_assets,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _write_ground_manifest(*, volume: Path) -> Path:
    root = volume / "input" / "ground-pbr-4k"
    materials: dict[str, dict[str, object]] = {}
    for role in asset_bundle.PBR_MATERIAL_ROLES:
        role_root = root / "materials" / role
        material = role_root / f"{role}.usda"
        material.parent.mkdir(parents=True, exist_ok=True)
        material.write_text(
            "#usda 1.0\n"
            f'(\n    customLayerData = {{ string role = "{role}" }}\n)\n'
            'def Material "Material" {\n'
            '    token outputs:surface.connect = </Material/Surface.outputs:surface>\n'
            '    def Shader "Surface" {\n'
            '        uniform token info:id = "UsdPreviewSurface"\n'
            '        token outputs:surface\n'
            "    }\n"
            "}\n",
            encoding="utf-8",
        )
        textures: dict[str, dict[str, object]] = {}
        for semantic in (
            *asset_bundle.PBR_REQUIRED_TEXTURES,
            *asset_bundle.PBR_OPTIONAL_TEXTURES,
        ):
            texture = role_root / f"{role}-{semantic}.png"
            _write_png_header(texture, identity=f"{role}:{semantic}")
            textures[semantic] = {
                **_record(texture, root=root),
                "width_px": 4096,
                "height_px": 4096,
                "color_space": "sRGB" if semantic == "base_color" else "raw",
            }
        materialx = role_root / f"{role}.mtlx"
        materialx.write_text("<materialx version=\"1.38\"/>\n", encoding="utf-8")
        materials[role] = {
            "material_id": f"fireviewer.pbr.test.{role}",
            "material_file": _record(material, root=root),
            "material_prim_path": "/Material",
            "metres_per_uv_tile": 4.0,
            "textures": textures,
            "source": {
                "provider": "test",
                "license": "CC0",
                "materialx": _record(materialx, root=root),
            },
        }
    manifest = root / "manifest-v3.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "state": "GROUND_PBR_MATERIALS_INSTALLED",
                "pbr_materials": materials,
            }
        ),
        encoding="utf-8",
    )
    return manifest


class CampaignAssetBundleTests(unittest.TestCase):
    def test_resolver_accepts_rebased_official_source_inside_volume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            manifest_parent = (
                volume / "input" / "asset-bundles" / ("a" * 64)
            )
            source = volume / "input" / "nvidia-asset-cache" / "asset.usd"
            manifest_parent.mkdir(parents=True)
            source.parent.mkdir(parents=True)
            source.write_bytes(b"locked-nvidia-usd")

            resolved, portable = campaign_asset_bundle._resolve_input_file(
                raw_path="../../nvidia-asset-cache/asset.usd",
                manifest_parent=manifest_parent,
                volume_root=volume,
                label="actors.canadair.source_cache_path",
            )

            self.assertEqual(resolved, source.resolve())
            self.assertEqual(
                portable,
                PurePosixPath("input/nvidia-asset-cache/asset.usd"),
            )

    def test_resolver_refuses_multiple_volume_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            manifest_parent = volume / "input"
            manifest_parent.mkdir(parents=True)
            (manifest_parent / "asset.usd").write_bytes(b"manifest-copy")
            (volume / "asset.usd").write_bytes(b"volume-copy")

            with self.assertRaisesRegex(
                campaign_asset_bundle.CampaignAssetBundleError,
                "resolves ambiguously",
            ):
                campaign_asset_bundle._resolve_input_file(
                    raw_path="asset.usd",
                    manifest_parent=manifest_parent,
                    volume_root=volume,
                    label="actors.canadair.path",
                )

    def test_missing_selected_actor_blocks_campaign_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            volume.mkdir()
            official = _write_official_manifest(volume=volume)
            payload = json.loads(official.read_text(encoding="utf-8"))
            missing_id = campaign_asset_bundle.SELECTED_ACTOR_GROUP_IDS[-1]
            payload["selected_actor_group"]["assets"].pop(missing_id)
            official.write_text(json.dumps(payload), encoding="utf-8")
            destination = volume / "input" / "campaign-assets"

            with self.assertRaisesRegex(
                campaign_asset_bundle.CampaignAssetBundleError,
                f"exact selected Chrome actor group.*{missing_id}",
            ):
                campaign_asset_bundle.assemble_campaign_asset_bundle(
                    volume_root=volume,
                    official_manifest_path=official,
                    ground_manifest_path=_write_ground_manifest(volume=volume),
                    destination_root=destination,
                    receipt_path=volume / "contracts" / "campaign-assets.json",
                )

            self.assertFalse(destination.exists())

    def test_semantic_actor_roles_remain_an_independent_required_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            volume.mkdir()
            official = _write_official_manifest(volume=volume)
            payload = json.loads(official.read_text(encoding="utf-8"))
            payload["actors"].pop("canadair")
            official.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                campaign_asset_bundle.CampaignAssetBundleError,
                "exact seven semantic actor roles.*canadair",
            ):
                campaign_asset_bundle.assemble_campaign_asset_bundle(
                    volume_root=volume,
                    official_manifest_path=official,
                    ground_manifest_path=_write_ground_manifest(volume=volume),
                    destination_root=volume / "input" / "campaign-assets",
                    receipt_path=volume / "contracts" / "campaign-assets.json",
                )

    def test_base_landscape_bundle_ignores_actors_but_campaign_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            volume.mkdir()
            official = _write_official_manifest(volume=volume)
            payload = json.loads(official.read_text(encoding="utf-8"))
            payload.pop("actors")
            payload.pop("selected_actor_group")
            official.write_text(json.dumps(payload), encoding="utf-8")
            ground = _write_ground_manifest(volume=volume)
            destination = volume / "input" / "base-landscape-assets"
            receipt = volume / "contracts" / "base-landscape-assets.json"

            result = (
                campaign_asset_bundle.assemble_base_landscape_environment_bundle(
                    volume_root=volume,
                    official_manifest_path=official,
                    ground_manifest_path=ground,
                    destination_root=destination,
                    receipt_path=receipt,
                )
            )

            self.assertEqual(
                result["state"],
                campaign_asset_bundle.BASE_LANDSCAPE_ASSEMBLY_STATE,
            )
            self.assertEqual(result["selected_actor_count"], 0)
            manifest = json.loads(
                (destination / "manifest-v3.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("actors", manifest)
            self.assertNotIn("selected_actor_group", manifest)
            self.assertEqual(
                set(manifest["selected_environment_group"]["assets"]),
                set(campaign_asset_bundle.SELECTED_ENVIRONMENT_GROUP_IDS),
            )
            for families in manifest["environment"].values():
                for entries in families.values():
                    for entry in entries:
                        self.assertEqual(
                            set(entry["lod_paths"]),
                            set(asset_bundle.LOD_LEVELS),
                        )
            reused = (
                campaign_asset_bundle.assemble_base_landscape_environment_bundle(
                    volume_root=volume,
                    official_manifest_path=official,
                    ground_manifest_path=ground,
                    destination_root=destination,
                    receipt_path=receipt,
                )
            )
            self.assertTrue(reused["reused"])

            with self.assertRaisesRegex(
                campaign_asset_bundle.CampaignAssetBundleError,
                "exact selected Chrome actor group",
            ):
                campaign_asset_bundle.assemble_campaign_asset_bundle(
                    volume_root=volume,
                    official_manifest_path=official,
                    ground_manifest_path=ground,
                    destination_root=volume / "input" / "campaign-assets",
                    receipt_path=volume / "contracts" / "campaign-assets.json",
                )

    def test_base_landscape_selected_habitat_satisfies_combined_minimum_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            volume.mkdir()
            official = _write_official_manifest(volume=volume)
            payload = json.loads(official.read_text(encoding="utf-8"))
            habitat = payload["environment"]["buildings"]["habitat"]
            self.assertEqual(len(habitat), 4)
            removed = habitat.pop(0)
            removed["asset_id"] = (
                "buildings.habitat:985a-defective-not-admitted"
            )
            official.write_text(json.dumps(payload), encoding="utf-8")
            ground = _write_ground_manifest(volume=volume)
            destination = volume / "input" / "base-landscape-assets"

            campaign_asset_bundle.assemble_base_landscape_environment_bundle(
                volume_root=volume,
                official_manifest_path=official,
                ground_manifest_path=ground,
                destination_root=destination,
                receipt_path=volume / "contracts" / "base-landscape.json",
            )

            manifest = json.loads(
                (destination / "manifest-v3.json").read_text(encoding="utf-8")
            )
            output_habitat = manifest["environment"]["buildings"]["habitat"]
            self.assertEqual(len(output_habitat), 5)
            self.assertEqual(
                len({entry["asset_id"] for entry in output_habitat}),
                len(output_habitat),
            )
            habitat_selection_ids = {
                selection_id
                for selection_id, target
                in campaign_asset_bundle.SELECTED_ENVIRONMENT_TARGET_BY_ID.items()
                if target == ("buildings", "habitat")
            }
            selected_references = manifest["selected_environment_group"]["assets"]
            for selection_id in habitat_selection_ids:
                reference = selected_references[selection_id]
                self.assertNotIn("lod_paths", reference)
                candidate = output_habitat[reference["environment_index"]]
                self.assertEqual(candidate["selection_id"], selection_id)
                self.assertEqual(candidate["asset_id"], reference["asset_id"])
                self.assertEqual(
                    sum(
                        entry["asset_id"] == reference["asset_id"]
                        for entry in output_habitat
                    ),
                    1,
                )

            with self.assertRaisesRegex(
                campaign_asset_bundle.CampaignAssetBundleError,
                r"environment\.buildings\.habitat requires at least 4 assets; "
                r"direct=3, selected_environment_group=0, combined=3",
            ):
                campaign_asset_bundle.assemble_campaign_asset_bundle(
                    volume_root=volume,
                    official_manifest_path=official,
                    ground_manifest_path=ground,
                    destination_root=volume / "input" / "campaign-assets",
                    receipt_path=volume / "contracts" / "campaign.json",
                )

    def test_base_landscape_fails_when_combined_family_minimum_is_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            volume.mkdir()
            official = _write_official_manifest(volume=volume)
            payload = json.loads(official.read_text(encoding="utf-8"))
            payload["environment"]["buildings"]["habitat"] = payload[
                "environment"
            ]["buildings"]["habitat"][:1]
            official.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                campaign_asset_bundle.CampaignAssetBundleError,
                r"environment\.buildings\.habitat requires at least 4 assets; "
                r"direct=1, selected_environment_group=2, combined=3",
            ):
                campaign_asset_bundle.assemble_base_landscape_environment_bundle(
                    volume_root=volume,
                    official_manifest_path=official,
                    ground_manifest_path=_write_ground_manifest(volume=volume),
                    destination_root=volume / "input" / "base-landscape-assets",
                    receipt_path=volume / "contracts" / "base-landscape.json",
                )

    def test_scene_optimizer_uses_fixed_strict_lod_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {
                level: root / f"{level.casefold()}.usdc"
                for level in asset_bundle.LOD_LEVELS
            }
            attempts: list[float] = []

            def flatten(**_kwargs: object) -> None:
                outputs["HERO"].write_bytes(b"hero")

            def decimate(*, stage_path: Path, retained_percent: float) -> None:
                attempts.append(retained_percent)
                stage_path.write_bytes(f"retained={retained_percent:g}".encode())

            def validate(**_kwargs: object) -> dict[str, dict[str, object]]:
                return {
                    level: {
                        "geometry_point_count": 100,
                        "geometry_face_count": faces,
                        "world_bounds": {"dimensions": [1.0, 1.0, 1.0]},
                    }
                    for level, faces in (
                        ("HERO", 100),
                        ("MID", 50),
                        ("FAR", 20),
                    )
                }

            with (
                patch.object(
                    campaign_asset_bundle,
                    "_flatten_hero_stage",
                    side_effect=flatten,
                ),
                patch.object(
                    campaign_asset_bundle,
                    "_scene_optimizer_decimate",
                    side_effect=decimate,
                ),
                patch.object(
                    campaign_asset_bundle,
                    "_validate_generated_native_chain",
                    side_effect=validate,
                ),
            ):
                metrics = campaign_asset_bundle._build_scene_optimizer_lods(
                    official_wrapper=root / "source.usdc",
                    output_paths=outputs,
                    bundle_root=root,
                )

        self.assertEqual(attempts, [60.0, 20.0])
        self.assertEqual(
            metrics["FAR"]["scene_optimizer_retained_percent"],
            20.0,
        )

    def test_far_retry_restarts_from_pristine_hero_and_records_actual_target(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {
                level: root / f"{level.casefold()}.usdc"
                for level in asset_bundle.LOD_LEVELS
            }
            attempts: list[float] = []
            far_inputs: list[bytes] = []

            def flatten(**_kwargs: object) -> None:
                outputs["HERO"].write_bytes(b"pristine-hero")

            def decimate(*, stage_path: Path, retained_percent: float) -> None:
                attempts.append(retained_percent)
                if stage_path == outputs["FAR"]:
                    far_inputs.append(stage_path.read_bytes())
                stage_path.write_text(
                    f"{retained_percent:g}",
                    encoding="utf-8",
                )

            def native_metrics(path: Path, **_kwargs: object) -> dict[str, object]:
                if path == outputs["HERO"]:
                    faces, dimensions = 100, [1.0, 1.0, 1.0]
                elif path == outputs["MID"]:
                    faces, dimensions = 60, [1.0, 1.0, 1.0]
                else:
                    retained = float(path.read_text(encoding="utf-8"))
                    faces = int(retained)
                    dimensions = (
                        [0.64, 0.75, 0.68]
                        if retained == 20.0
                        else [0.70, 0.80, 0.75]
                    )
                return {
                    "geometry_point_count": faces,
                    "geometry_face_count": faces,
                    "world_bounds": {"dimensions": dimensions},
                }

            with (
                patch.object(
                    campaign_asset_bundle,
                    "_flatten_hero_stage",
                    side_effect=flatten,
                ),
                patch.object(
                    campaign_asset_bundle,
                    "_scene_optimizer_decimate",
                    side_effect=decimate,
                ),
                patch.object(
                    campaign_asset_bundle._asset_contract,
                    "_native_usd_metrics",
                    side_effect=native_metrics,
                ),
            ):
                metrics = campaign_asset_bundle._build_scene_optimizer_lods(
                    official_wrapper=root / "source.usdc",
                    output_paths=outputs,
                    bundle_root=root,
                )

        self.assertEqual(attempts, [60.0, 20.0, 30.0])
        self.assertEqual(far_inputs, [b"pristine-hero", b"pristine-hero"])
        self.assertEqual(
            metrics["FAR"]["scene_optimizer_retained_percent"],
            30.0,
        )

    def test_far_retry_fails_closed_when_every_identity_attempt_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {
                level: root / f"{level.casefold()}.usdc"
                for level in asset_bundle.LOD_LEVELS
            }
            attempts: list[float] = []
            far_inputs: list[bytes] = []

            def flatten(**_kwargs: object) -> None:
                outputs["HERO"].write_bytes(b"pristine-hero")

            def decimate(*, stage_path: Path, retained_percent: float) -> None:
                attempts.append(retained_percent)
                if stage_path == outputs["FAR"]:
                    far_inputs.append(stage_path.read_bytes())
                stage_path.write_text(
                    f"{retained_percent:g}",
                    encoding="utf-8",
                )

            def native_metrics(path: Path, **_kwargs: object) -> dict[str, object]:
                if path == outputs["HERO"]:
                    faces = 100
                    dimensions = [1.0, 1.0, 1.0]
                elif path == outputs["MID"]:
                    faces = 60
                    dimensions = [1.0, 1.0, 1.0]
                else:
                    faces = int(float(path.read_text(encoding="utf-8")))
                    dimensions = [0.64, 0.75, 0.68]
                return {
                    "geometry_point_count": faces,
                    "geometry_face_count": faces,
                    "world_bounds": {"dimensions": dimensions},
                }

            with (
                patch.object(
                    campaign_asset_bundle,
                    "_flatten_hero_stage",
                    side_effect=flatten,
                ),
                patch.object(
                    campaign_asset_bundle,
                    "_scene_optimizer_decimate",
                    side_effect=decimate,
                ),
                patch.object(
                    campaign_asset_bundle._asset_contract,
                    "_native_usd_metrics",
                    side_effect=native_metrics,
                ),
            ):
                with self.assertRaisesRegex(
                    campaign_asset_bundle.CampaignAssetBundleError,
                    r"attempts \[20%, 30%, 40%, 50%\]",
                ):
                    campaign_asset_bundle._build_scene_optimizer_lods(
                        official_wrapper=root / "source.usdc",
                        output_paths=outputs,
                        bundle_root=root,
                    )

        self.assertEqual(attempts, [60.0, 20.0, 30.0, 40.0, 50.0])
        self.assertEqual(far_inputs, [b"pristine-hero"] * 4)

    def test_scene_optimizer_refuses_mutated_hero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {
                level: root / f"{level.casefold()}.usdc"
                for level in asset_bundle.LOD_LEVELS
            }

            def flatten(**_kwargs: object) -> None:
                outputs["HERO"].write_bytes(b"locked-hero")

            def decimate(*, stage_path: Path, retained_percent: float) -> None:
                stage_path.write_bytes(f"retained={retained_percent:g}".encode())
                if retained_percent == campaign_asset_bundle.MID_RETAINED_PERCENT:
                    outputs["HERO"].write_bytes(b"mutated-hero")

            with (
                patch.object(
                    campaign_asset_bundle,
                    "_flatten_hero_stage",
                    side_effect=flatten,
                ),
                patch.object(
                    campaign_asset_bundle,
                    "_scene_optimizer_decimate",
                    side_effect=decimate,
                ),
            ):
                with self.assertRaisesRegex(
                    campaign_asset_bundle.CampaignAssetBundleError,
                    "modified HERO.*MID",
                ):
                    campaign_asset_bundle._build_scene_optimizer_lods(
                        official_wrapper=root / "source.usdc",
                        output_paths=outputs,
                        bundle_root=root,
                    )

    def test_scene_optimizer_does_not_pin_open_mesh_boundaries(self) -> None:
        arguments = campaign_asset_bundle._scene_optimizer_arguments(12.5)

        self.assertEqual(arguments["reductionFactor"], 12.5)
        self.assertIs(arguments["pinBoundaries"], False)

    def test_cli_starts_and_closes_headless_isaac_runtime(self) -> None:
        runtime = Mock()
        result = {"state": campaign_asset_bundle.ASSEMBLY_STATE}
        argv = [
            "--volume-root",
            "volume",
            "--official-manifest",
            "official.json",
            "--ground-manifest",
            "ground.json",
            "--destination-root",
            "destination",
            "--receipt",
            "receipt.json",
        ]

        with (
            patch.object(
                campaign_asset_bundle,
                "_start_isaac_runtime",
                return_value=runtime,
            ) as start_runtime,
            patch.object(
                campaign_asset_bundle,
                "assemble_campaign_asset_bundle",
                return_value=result,
            ) as assemble,
            patch("builtins.print") as print_result,
        ):
            exit_code = campaign_asset_bundle.main(argv)

        self.assertEqual(exit_code, 0)
        start_runtime.assert_called_once_with()
        assemble.assert_called_once()
        runtime.close.assert_called_once_with()
        print_result.assert_called_once_with(
            json.dumps(result, sort_keys=True),
            flush=True,
        )

    def test_cli_routes_explicit_base_landscape_environment_mode(self) -> None:
        runtime = Mock()
        result = {
            "state": campaign_asset_bundle.BASE_LANDSCAPE_ASSEMBLY_STATE
        }
        argv = [
            "--volume-root",
            "volume",
            "--official-manifest",
            "official.json",
            "--ground-manifest",
            "ground.json",
            "--destination-root",
            "destination",
            "--receipt",
            "receipt.json",
            "--mode",
            "base-landscape-environment",
        ]

        with (
            patch.object(
                campaign_asset_bundle,
                "_start_isaac_runtime",
                return_value=runtime,
            ),
            patch.object(
                campaign_asset_bundle,
                "assemble_base_landscape_environment_bundle",
                return_value=result,
            ) as assemble,
            patch.object(
                campaign_asset_bundle,
                "assemble_campaign_asset_bundle",
            ) as campaign_assemble,
            patch("builtins.print"),
        ):
            exit_code = campaign_asset_bundle.main(argv)

        self.assertEqual(exit_code, 0)
        assemble.assert_called_once()
        campaign_assemble.assert_not_called()
        runtime.close.assert_called_once_with()

    def test_cli_closes_isaac_runtime_when_assembly_fails(self) -> None:
        runtime = Mock()
        argv = [
            "--volume-root",
            "volume",
            "--official-manifest",
            "official.json",
            "--ground-manifest",
            "ground.json",
            "--destination-root",
            "destination",
            "--receipt",
            "receipt.json",
        ]

        with (
            patch.object(
                campaign_asset_bundle,
                "_start_isaac_runtime",
                return_value=runtime,
            ),
            patch.object(
                campaign_asset_bundle,
                "assemble_campaign_asset_bundle",
                side_effect=campaign_asset_bundle.CampaignAssetBundleError(
                    "Scene Optimizer failed"
                ),
            ),
        ):
            with self.assertRaisesRegex(
                campaign_asset_bundle.CampaignAssetBundleError,
                "Scene Optimizer failed",
            ):
                campaign_asset_bundle.main(argv)

        runtime.close.assert_called_once_with()

    def test_runtime_start_fails_closed_without_isaac_sim(self) -> None:
        with patch.dict(sys.modules, {"isaacsim": None}):
            with self.assertRaisesRegex(
                campaign_asset_bundle.CampaignAssetBundleError,
                "requires Isaac Sim's packaged Python runtime",
            ):
                campaign_asset_bundle._start_isaac_runtime()

    def test_assembles_native_provider_lods_and_reuses_locked_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            volume.mkdir()
            official = _write_official_manifest(volume=volume)
            ground = _write_ground_manifest(volume=volume)
            destination = volume / "input" / "campaign-assets"
            receipt = volume / "contracts" / "campaign-assets.json"

            result = campaign_asset_bundle.assemble_campaign_asset_bundle(
                volume_root=volume,
                official_manifest_path=official,
                ground_manifest_path=ground,
                destination_root=destination,
                receipt_path=receipt,
            )

            self.assertEqual(
                result["state"],
                campaign_asset_bundle.ASSEMBLY_STATE,
            )
            self.assertFalse(result["reused"])
            self.assertEqual(
                result["asset_count"],
                sum(
                    sum(families.values())
                    for families in PHOTOREAL_FAMILY_MINIMUMS.values()
                )
                + len(asset_bundle.REQUIRED_ACTOR_CLASSES)
                + len(campaign_asset_bundle.SELECTED_ACTOR_GROUP_IDS)
                + len(
                    campaign_asset_bundle.SELECTED_ENVIRONMENT_GROUP_IDS
                ),
            )
            self.assertEqual(
                result["selected_actor_count"],
                len(campaign_asset_bundle.SELECTED_ACTOR_GROUP_IDS),
            )
            self.assertEqual(
                result["selected_environment_count"],
                len(campaign_asset_bundle.SELECTED_ENVIRONMENT_GROUP_IDS),
            )
            manifest_path = destination / "manifest-v3.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["selected_actor_group"]["selection_order"],
                list(campaign_asset_bundle.SELECTED_ACTOR_GROUP_IDS),
            )
            self.assertEqual(
                set(manifest["selected_actor_group"]["assets"]),
                set(campaign_asset_bundle.SELECTED_ACTOR_GROUP_IDS),
            )
            self.assertEqual(
                manifest["selected_environment_group"]["selection_order"],
                list(campaign_asset_bundle.SELECTED_ENVIRONMENT_GROUP_IDS),
            )
            for selection_id in (
                campaign_asset_bundle.SELECTED_ENVIRONMENT_GROUP_IDS
            ):
                selected = manifest["selected_environment_group"]["assets"][
                    selection_id
                ]
                kind, family = (
                    campaign_asset_bundle.SELECTED_ENVIRONMENT_TARGET_BY_ID[
                        selection_id
                    ]
                )
                self.assertIn(
                    selected["asset_id"],
                    {
                        entry["asset_id"]
                        for entry in manifest["environment"][kind][family]
                    },
                )
            self.assertEqual(
                set(manifest["actors"]),
                set(asset_bundle.REQUIRED_ACTOR_CLASSES),
            )
            self.assertEqual(
                len(
                    {
                        entry["asset_id"]
                        for entry in manifest["actors"].values()
                    }
                ),
                len(asset_bundle.REQUIRED_ACTOR_CLASSES),
            )
            first = manifest["environment"]["vegetation"]["trees"][0]
            self.assertEqual(set(first["lod_paths"]), set(asset_bundle.LOD_LEVELS))
            self.assertEqual(
                len(
                    {
                        record["path"]
                        for record in first["lod_paths"].values()
                    }
                ),
                3,
            )
            wrapper_texts = [
                (destination / record["path"]).read_text(encoding="utf-8")
                for record in first["lod_paths"].values()
            ]
            self.assertTrue(any('lodVariant = "LOD0"' in text for text in wrapper_texts))
            self.assertTrue(any('lodVariant = "LOD1"' in text for text in wrapper_texts))
            self.assertTrue(any('lodVariant = "LOD2"' in text for text in wrapper_texts))
            for role in asset_bundle.PBR_MATERIAL_ROLES:
                self.assertEqual(
                    set(manifest["pbr_materials"][role]["textures"]),
                    set(asset_bundle.PBR_REQUIRED_TEXTURES),
                )

            marker = json.loads(
                (destination / asset_bundle.INSTALL_MARKER).read_text(
                    encoding="utf-8"
                )
            )
            validated = asset_bundle._validate_reuse(
                destination=destination,
                expected_sha256=marker["bundle_sha256"],
                manifest_relative=Path("manifest-v3.json"),
            )
            self.assertEqual(validated["state"], "ASSET_BUNDLE_INSTALLED")

            reused = campaign_asset_bundle.assemble_campaign_asset_bundle(
                volume_root=volume,
                official_manifest_path=official,
                ground_manifest_path=ground,
                destination_root=destination,
                receipt_path=receipt,
            )
            self.assertTrue(reused["reused"])

    def test_source_default_requires_real_scene_optimizer_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            volume.mkdir()
            official = _write_official_manifest(
                volume=volume,
                strategy="source_default_only",
                level_count=1,
            )
            ground = _write_ground_manifest(volume=volume)
            destination = volume / "input" / "campaign-assets"

            with patch.object(
                campaign_asset_bundle,
                "_build_scene_optimizer_lods",
                side_effect=campaign_asset_bundle.CampaignAssetBundleError(
                    "Scene Optimizer operation decimateMeshes is unavailable"
                ),
            ):
                with self.assertRaisesRegex(
                    campaign_asset_bundle.CampaignAssetBundleError,
                    "decimateMeshes is unavailable",
                ):
                    campaign_asset_bundle.assemble_campaign_asset_bundle(
                        volume_root=volume,
                        official_manifest_path=official,
                        ground_manifest_path=ground,
                        destination_root=destination,
                        receipt_path=volume / "contracts" / "receipt.json",
                    )
            self.assertFalse(destination.exists())

    def test_two_provider_levels_fail_with_exact_asset_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            volume.mkdir()
            official = _write_official_manifest(
                volume=volume,
                strategy="native_prim_hierarchy",
                level_count=2,
            )
            ground = _write_ground_manifest(volume=volume)
            destination = volume / "input" / "campaign-assets"

            with self.assertRaises(
                campaign_asset_bundle.CampaignAssetBundleError
            ) as raised:
                campaign_asset_bundle.assemble_campaign_asset_bundle(
                    volume_root=volume,
                    official_manifest_path=official,
                    ground_manifest_path=ground,
                    destination_root=destination,
                    receipt_path=volume / "contracts" / "receipt.json",
                )
            message = str(raised.exception)
            self.assertIn("environment.vegetation.trees[0]", message)
            self.assertIn("strategy='native_prim_hierarchy'", message)
            self.assertIn("level_count=2", message)
            self.assertFalse(destination.exists())

    def test_source_default_records_distinct_generated_lods(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            volume.mkdir()
            official = _write_official_manifest(
                volume=volume,
                strategy="source_default_only",
                level_count=1,
            )
            ground = _write_ground_manifest(volume=volume)
            destination = volume / "input" / "campaign-assets"
            generated = 0

            def fake_native_builder(
                *,
                official_wrapper: Path,
                output_paths: dict[str, Path],
                bundle_root: Path,
            ) -> dict[str, dict[str, object]]:
                nonlocal generated
                self.assertTrue(official_wrapper.is_file())
                self.assertTrue(official_wrapper.is_relative_to(bundle_root))
                generated += 1
                metrics: dict[str, dict[str, object]] = {}
                for index, level in enumerate(asset_bundle.LOD_LEVELS):
                    output = output_paths[level]
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(
                        f"native-{generated}-{level}-{100-index * 40}".encode()
                    )
                    metrics[level] = {
                        "geometry_point_count": 100 - index * 40,
                        "geometry_face_count": 80 - index * 30,
                        "world_bounds": {"dimensions": [1.0, 1.0, 1.0]},
                    }
                return metrics

            with patch.object(
                campaign_asset_bundle,
                "_build_scene_optimizer_lods",
                side_effect=fake_native_builder,
            ):
                campaign_asset_bundle.assemble_campaign_asset_bundle(
                    volume_root=volume,
                    official_manifest_path=official,
                    ground_manifest_path=ground,
                    destination_root=destination,
                    receipt_path=volume / "contracts" / "receipt.json",
                )

            manifest = json.loads(
                (destination / "manifest-v3.json").read_text(encoding="utf-8")
            )
            asset_count = sum(
                sum(families.values())
                for families in PHOTOREAL_FAMILY_MINIMUMS.values()
            ) + len(asset_bundle.REQUIRED_ACTOR_CLASSES) + len(
                campaign_asset_bundle.SELECTED_ACTOR_GROUP_IDS
            ) + len(campaign_asset_bundle.SELECTED_ENVIRONMENT_GROUP_IDS)
            self.assertEqual(generated, asset_count)
            for families in manifest["environment"].values():
                for entries in families.values():
                    for entry in entries:
                        records = entry["lod_paths"]
                        self.assertEqual(
                            len({record["sha256"] for record in records.values()}),
                            3,
                        )
                        self.assertEqual(
                            entry["campaign_lod_generation"]["strategy"],
                            "scene_optimizer_decimateMeshes",
                        )
                        self.assertEqual(
                            entry["lod"],
                            {
                                "state": "passed",
                                "strategy": "scene_optimizer_decimateMeshes",
                                "levels": list(asset_bundle.LOD_LEVELS),
                                "level_count": len(asset_bundle.LOD_LEVELS),
                            },
                        )
            self.assertEqual(
                set(manifest["actors"]),
                set(asset_bundle.REQUIRED_ACTOR_CLASSES),
            )
            for entry in manifest["actors"].values():
                self.assertEqual(
                    set(entry["lod_paths"]),
                    set(asset_bundle.LOD_LEVELS),
                )


if __name__ == "__main__":
    unittest.main()

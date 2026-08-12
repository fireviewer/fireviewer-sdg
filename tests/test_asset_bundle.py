from __future__ import annotations

import hashlib
import json
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg import asset_bundle  # noqa: E402
from fireviewer_sdg.simready_assets import (  # noqa: E402
    MANIFEST_PROFILE,
    MANIFEST_SCHEMA_VERSION,
    PHOTOREAL_FAMILY_MINIMUMS,
    PHOTOREAL_LIBRARY_POLICY,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_portable_bundle(root: Path) -> Path:
    environment: dict[str, dict[str, list[dict[str, object]]]] = {
        "vegetation": {},
        "buildings": {},
    }
    for kind, family_minimums in PHOTOREAL_FAMILY_MINIMUMS.items():
        for family, minimum in family_minimums.items():
            entries: list[dict[str, object]] = []
            for index in range(minimum):
                stem = f"{kind}-{family}-{index}"
                asset_id = f"{kind}.{family}:{stem}"
                source_identity = f"test/{kind}/{family}/{stem}"
                lineage = hashlib.sha256(
                    f"{asset_id}\0{source_identity}".encode("utf-8")
                ).hexdigest()
                source = root / "sources" / f"{stem}.usdc"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(f"source-{stem}".encode("utf-8"))
                lod_paths: dict[str, dict[str, object]] = {}
                for level in asset_bundle.LOD_LEVELS:
                    wrapper = root / "wrappers" / f"{stem}-{level.casefold()}.usda"
                    wrapper.parent.mkdir(parents=True, exist_ok=True)
                    wrapper.write_text(
                        "#usda 1.0\n"
                        f'(\n    customLayerData = {{ string lod = "{level}" }}\n)\n'
                        f'def Xform "Asset" (prepend references = '
                        f"@../sources/{stem}.usdc@) {{}}\n",
                        encoding="utf-8",
                    )
                    lod_paths[level] = {
                        "path": wrapper.relative_to(root).as_posix(),
                        "sha256": _sha256(wrapper),
                        "size_bytes": wrapper.stat().st_size,
                        "lineage_id": lineage,
                    }
                dependency = {
                    "path": source.relative_to(root).as_posix(),
                    "sha256": _sha256(source),
                    "size_bytes": source.stat().st_size,
                }
                entries.append(
                    {
                        "asset_id": asset_id,
                        "identity": {
                            "source_name": stem,
                            "source_identity": source_identity,
                        },
                        "lod_lineage_id": lineage,
                        "path": lod_paths["HERO"]["path"],
                        "sha256": lod_paths["HERO"]["sha256"],
                        "lod_paths": lod_paths,
                        "source_cache_path": source.relative_to(root).as_posix(),
                        "materialized_files": [dependency],
                        "content_lock_sha256": hashlib.sha256(
                            json.dumps([dependency], sort_keys=True).encode("utf-8")
                        ).hexdigest(),
                    }
                )
            environment[kind][family] = entries

    actor_source = root / "sources" / "reviewed-actors.usdc"
    actor_source.write_bytes(b"reviewed-actor-source")
    actor_dependency = {
        "path": actor_source.relative_to(root).as_posix(),
        "sha256": _sha256(actor_source),
        "size_bytes": actor_source.stat().st_size,
    }
    actors: dict[str, dict[str, object]] = {}
    for class_id in asset_bundle.REQUIRED_ACTOR_CLASSES:
        asset_id = f"actors.{class_id}:test"
        source_identity = f"curated/test/actors/{class_id}"
        lineage = hashlib.sha256(
            f"{asset_id}\0{source_identity}".encode("utf-8")
        ).hexdigest()
        lod_paths: dict[str, dict[str, object]] = {}
        for level in asset_bundle.LOD_LEVELS:
            wrapper = root / "wrappers" / (
                f"actor-{class_id}-{level.casefold()}.usda"
            )
            wrapper.write_text(
                "#usda 1.0\n"
                f'(\n    customLayerData = {{ string actor = "{class_id}" '
                f'string lod = "{level}" }}\n)\n'
                'def Xform "Asset" (prepend references = '
                '@../sources/reviewed-actors.usdc@) {}\n',
                encoding="utf-8",
            )
            lod_paths[level] = {
                "path": wrapper.relative_to(root).as_posix(),
                "sha256": _sha256(wrapper),
                "size_bytes": wrapper.stat().st_size,
                "lineage_id": lineage,
            }
        actors[class_id] = {
            "asset_id": asset_id,
            "identity": {
                "source_name": class_id,
                "source_identity": source_identity,
            },
            "lod_lineage_id": lineage,
            "path": lod_paths["HERO"]["path"],
            "sha256": lod_paths["HERO"]["sha256"],
            "lod_paths": lod_paths,
            "source_cache_path": actor_source.relative_to(root).as_posix(),
            "materialized_files": [actor_dependency],
            "content_lock_sha256": hashlib.sha256(
                json.dumps([actor_dependency], sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }

    pbr_materials: dict[str, dict[str, object]] = {}
    for role_index, role in enumerate(asset_bundle.PBR_MATERIAL_ROLES):
        material_file = root / "materials" / role / f"{role}.usda"
        material_file.parent.mkdir(parents=True, exist_ok=True)
        material_file.write_text(
            "#usda 1.0\n"
            'def Material "Material" {\n'
            '    token outputs:surface.connect = </Material/Surface.outputs:surface>\n'
            '    def Shader "Surface" {\n'
            '        uniform token info:id = "UsdPreviewSurface"\n'
            '        token outputs:surface\n'
            '        color3f inputs:diffuseColor.connect = '
            '</Material/BaseColor.outputs:rgb>\n'
            '        normal3f inputs:normal.connect = </Material/Normal.outputs:rgb>\n'
            '        float inputs:roughness.connect = '
            '</Material/Roughness.outputs:r>\n'
            "    }\n"
            '    def Shader "BaseColor" {\n'
            '        uniform token info:id = "UsdUVTexture"\n'
            f'        asset inputs:file = @{role}-base_color.png@\n'
            '        float3 outputs:rgb\n'
            "    }\n"
            '    def Shader "Normal" {\n'
            '        uniform token info:id = "UsdUVTexture"\n'
            f'        asset inputs:file = @{role}-normal.png@\n'
            '        normal3f outputs:rgb\n'
            "    }\n"
            '    def Shader "Roughness" {\n'
            '        uniform token info:id = "UsdUVTexture"\n'
            f'        asset inputs:file = @{role}-roughness.png@\n'
            '        float outputs:r\n'
            "    }\n"
            "}\n",
            encoding="utf-8",
        )
        textures: dict[str, dict[str, object]] = {}
        for texture_index, texture_role in enumerate(
            asset_bundle.PBR_REQUIRED_TEXTURES
        ):
            texture = material_file.parent / f"{role}-{texture_role}.png"
            color = (
                (role_index * 31 + texture_index * 7 + 1) % 256,
                (role_index * 17 + texture_index * 43 + 2) % 256,
                (role_index * 53 + texture_index * 11 + 3) % 256,
            )
            Image.new("RGB", (2048, 2048), color).save(texture, "PNG")
            textures[texture_role] = {
                "path": texture.relative_to(root).as_posix(),
                "sha256": _sha256(texture),
                "size_bytes": texture.stat().st_size,
                "width_px": 2048,
                "height_px": 2048,
                "color_space": "sRGB" if texture_role == "base_color" else "raw",
            }
        pbr_materials[role] = {
            "material_id": f"fireviewer.pbr.{role}",
            "material_file": {
                "path": material_file.relative_to(root).as_posix(),
                "sha256": _sha256(material_file),
                "size_bytes": material_file.stat().st_size,
            },
            "material_prim_path": "/Material",
            "metres_per_uv_tile": 4.0,
            "textures": textures,
        }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "profile": MANIFEST_PROFILE,
        "family_minimums": PHOTOREAL_FAMILY_MINIMUMS,
        "library_policy": PHOTOREAL_LIBRARY_POLICY,
        "discovery": {
            "mode": "materialized_photoreal_asset_library_v3",
            "missing_environment": [],
        },
        "environment": environment,
        "actors": actors,
        "pbr_materials": pbr_materials,
    }
    manifest_path = root / "manifest-v3.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _zip_tree(source: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(source).as_posix())


class AssetBundleTests(unittest.TestCase):
    def test_asset_contract_requires_all_reviewed_actor_classes(self) -> None:
        payload = {
            "actors": {
                class_id: {}
                for class_id in asset_bundle.REQUIRED_ACTOR_CLASSES
                if class_id != "hard_negative_crop_duster"
            }
        }

        with self.assertRaisesRegex(
            ValueError,
            "exact reviewed actor classes.*hard_negative_crop_duster",
        ):
            asset_bundle._actor_entries(payload)

    def test_installs_portable_v3_manifest_and_reuses_exact_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = root / "volume"
            source = root / "source"
            volume.mkdir()
            source.mkdir()
            _write_portable_bundle(source)
            archive = volume / "downloads" / "assets.zip"
            archive.parent.mkdir()
            _zip_tree(source, archive)
            digest = _sha256(archive)
            destination = volume / "input" / "asset-bundles" / digest
            receipt = volume / "contracts" / "asset-bundle-install.json"

            result = asset_bundle.install_asset_bundle(
                archive_path=archive,
                expected_sha256=digest,
                volume_root=volume,
                destination_root=destination,
                manifest_relative="manifest-v3.json",
                receipt_path=receipt,
                minimum_free_after_install_bytes=0,
            )

            self.assertEqual(result["state"], "ASSET_BUNDLE_INSTALLED")
            self.assertEqual(result["bundle_sha256"], digest)
            self.assertTrue((destination / asset_bundle.INSTALL_MARKER).is_file())
            runtime_manifest = json.loads(
                (destination / "manifest-v3.json").read_text(encoding="utf-8")
            )
            first = runtime_manifest["environment"]["vegetation"]["trees"][0]
            dependency = first["materialized_files"][0]
            self.assertEqual(
                dependency["path"],
                (
                    destination
                    / "sources"
                    / "vegetation-trees-0.usdc"
                ).relative_to(volume).as_posix(),
            )
            self.assertEqual(
                first["content_lock_sha256"],
                hashlib.sha256(
                    json.dumps(
                        first["materialized_files"], sort_keys=True
                    ).encode("utf-8")
                ).hexdigest(),
            )
            reused = asset_bundle.install_asset_bundle(
                archive_path=archive,
                expected_sha256=digest,
                volume_root=volume,
                destination_root=destination,
                manifest_relative="manifest-v3.json",
                receipt_path=receipt,
                minimum_free_after_install_bytes=0,
            )
            self.assertEqual(reused, result)

    def test_rejects_hash_mismatch_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = root / "volume"
            volume.mkdir()
            archive = volume / "assets.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("manifest-v3.json", "{}")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                asset_bundle.install_asset_bundle(
                    archive_path=archive,
                    expected_sha256="0" * 64,
                    volume_root=volume,
                    destination_root=volume / "input" / "bundle",
                    manifest_relative="manifest-v3.json",
                    receipt_path=volume / "contracts" / "receipt.json",
                    minimum_free_after_install_bytes=0,
                )

    def test_rejects_archive_traversal_and_does_not_write_outside_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = root / "volume"
            volume.mkdir()
            archive = volume / "assets.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escaped.usdc", "forbidden")
            digest = _sha256(archive)
            with self.assertRaisesRegex(ValueError, "stay below"):
                asset_bundle.install_asset_bundle(
                    archive_path=archive,
                    expected_sha256=digest,
                    volume_root=volume,
                    destination_root=volume / "input" / "bundle",
                    manifest_relative="manifest-v3.json",
                    receipt_path=volume / "contracts" / "receipt.json",
                    minimum_free_after_install_bytes=0,
                )
            self.assertFalse((volume / "input" / "escaped.usdc").exists())
            self.assertFalse((root / "escaped.usdc").exists())

    def test_rejects_zip_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = root / "volume"
            volume.mkdir()
            archive = volume / "assets.zip"
            link = zipfile.ZipInfo("linked.usdc")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(link, "../outside.usdc")
            digest = _sha256(archive)
            with self.assertRaisesRegex(ValueError, "links are forbidden"):
                asset_bundle.install_asset_bundle(
                    archive_path=archive,
                    expected_sha256=digest,
                    volume_root=volume,
                    destination_root=volume / "input" / "bundle",
                    manifest_relative="manifest-v3.json",
                    receipt_path=volume / "contracts" / "receipt.json",
                    minimum_free_after_install_bytes=0,
                )

    def test_rejects_absolute_manifest_dependency_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = root / "volume"
            source = root / "source"
            volume.mkdir()
            source.mkdir()
            manifest_path = _write_portable_bundle(source)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = payload["environment"]["vegetation"]["trees"][0]
            entry["materialized_files"][0]["path"] = "/host/secret.usdc"
            entry["content_lock_sha256"] = hashlib.sha256(
                json.dumps(
                    entry["materialized_files"], sort_keys=True
                ).encode("utf-8")
            ).hexdigest()
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            archive = volume / "assets.zip"
            _zip_tree(source, archive)
            digest = _sha256(archive)
            with self.assertRaisesRegex(ValueError, "stay below"):
                asset_bundle.install_asset_bundle(
                    archive_path=archive,
                    expected_sha256=digest,
                    volume_root=volume,
                    destination_root=volume / "input" / "bundle",
                    manifest_relative="manifest-v3.json",
                    receipt_path=volume / "contracts" / "receipt.json",
                    minimum_free_after_install_bytes=0,
                )

    def test_rejects_bundle_without_complete_ground_pbr_materials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = root / "volume"
            source = root / "source"
            volume.mkdir()
            source.mkdir()
            manifest_path = _write_portable_bundle(source)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            del payload["pbr_materials"]["water"]
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            archive = volume / "assets.zip"
            _zip_tree(source, archive)
            digest = _sha256(archive)
            with self.assertRaisesRegex(ValueError, "pbr_materials must contain"):
                asset_bundle.install_asset_bundle(
                    archive_path=archive,
                    expected_sha256=digest,
                    volume_root=volume,
                    destination_root=volume / "input" / "bundle",
                    manifest_relative="manifest-v3.json",
                    receipt_path=volume / "contracts" / "receipt.json",
                    minimum_free_after_install_bytes=0,
                )

    def test_rejects_declared_pbr_dimensions_that_do_not_match_texture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = root / "volume"
            source = root / "source"
            volume.mkdir()
            source.mkdir()
            manifest_path = _write_portable_bundle(source)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["pbr_materials"]["soil"]["textures"]["normal"]["width_px"] = 4096
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            archive = volume / "assets.zip"
            _zip_tree(source, archive)
            digest = _sha256(archive)
            with self.assertRaisesRegex(ValueError, "matching its declared dimensions"):
                asset_bundle.install_asset_bundle(
                    archive_path=archive,
                    expected_sha256=digest,
                    volume_root=volume,
                    destination_root=volume / "input" / "bundle",
                    manifest_relative="manifest-v3.json",
                    receipt_path=volume / "contracts" / "receipt.json",
                    minimum_free_after_install_bytes=0,
                )

    def test_rejects_asset_without_three_distinct_local_lod_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = root / "volume"
            source = root / "source"
            volume.mkdir()
            source.mkdir()
            manifest_path = _write_portable_bundle(source)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            tree = payload["environment"]["vegetation"]["trees"][0]
            tree["lod_paths"]["FAR"] = dict(tree["lod_paths"]["MID"])
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            archive = volume / "assets.zip"
            _zip_tree(source, archive)
            digest = _sha256(archive)
            with self.assertRaisesRegex(ValueError, "must be distinct"):
                asset_bundle.install_asset_bundle(
                    archive_path=archive,
                    expected_sha256=digest,
                    volume_root=volume,
                    destination_root=volume / "input" / "bundle",
                    manifest_relative="manifest-v3.json",
                    receipt_path=volume / "contracts" / "receipt.json",
                    minimum_free_after_install_bytes=0,
                )

    def test_rejects_extraction_when_declared_size_would_consume_margin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = root / "volume"
            volume.mkdir()
            archive = volume / "assets.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("manifest-v3.json", "{}")
            digest = _sha256(archive)
            with patch.object(
                asset_bundle.shutil,
                "disk_usage",
                return_value=SimpleNamespace(total=10, used=9, free=1),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "insufficient free space"
                ):
                    asset_bundle.install_asset_bundle(
                        archive_path=archive,
                        expected_sha256=digest,
                        volume_root=volume,
                        destination_root=volume / "input" / "bundle",
                        manifest_relative="manifest-v3.json",
                        receipt_path=volume / "contracts" / "receipt.json",
                        minimum_free_after_install_bytes=10,
                    )

    def test_native_lod_gate_requires_decreasing_geometry_for_every_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = root / "volume"
            source = root / "source"
            volume.mkdir()
            source.mkdir()
            _write_portable_bundle(source)
            archive = volume / "assets.zip"
            _zip_tree(source, archive)
            digest = _sha256(archive)
            destination = volume / "input" / "bundle"
            asset_bundle.install_asset_bundle(
                archive_path=archive,
                expected_sha256=digest,
                volume_root=volume,
                destination_root=destination,
                manifest_relative="manifest-v3.json",
                receipt_path=volume / "contracts" / "install.json",
                minimum_free_after_install_bytes=0,
            )

            def metrics(
                path: Path,
                **_kwargs: object,
            ) -> dict[str, object]:
                level = next(
                    candidate
                    for candidate in asset_bundle.LOD_LEVELS
                    if f"-{candidate.casefold()}." in path.name
                )
                complexity = {"HERO": 3_000, "MID": 900, "FAR": 90}[level]
                return {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "used_layer_count": 2,
                    "used_layers": [str(path.resolve())],
                    "geometry_prim_count": 2,
                    "material_bound_prim_count": 2,
                    "geometry_point_count": 3_000,
                    "geometry_face_count": complexity,
                    "world_bounds": {
                        "minimum": [0.0, 0.0, 0.0],
                        "maximum": [10.0, 10.0, 10.0],
                        "dimensions": [10.0, 10.0, 10.0],
                    },
                }

            with patch.object(
                asset_bundle,
                "_native_usd_metrics",
                side_effect=metrics,
            ):
                result = asset_bundle.validate_native_lod_quality(
                    manifest_path=destination / "manifest-v3.json",
                    volume_root=volume,
                    bundle_root=destination,
                    receipt_path=volume / "contracts" / "native-lods.json",
                )
            self.assertEqual(result["state"], "NATIVE_ASSET_LODS_VALIDATED")
            self.assertEqual(
                result["asset_count"],
                26 + len(asset_bundle.REQUIRED_ACTOR_CLASSES),
            )
            manifest = destination / "manifest-v3.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            pbr_structural = asset_bundle._validate_pbr_materials(
                payload=payload,
                manifest_parent=manifest.parent,
            )
            marker = json.loads(
                (destination / asset_bundle.INSTALL_MARKER).read_text(
                    encoding="utf-8"
                )
            )
            pbr_receipt = volume / "contracts" / "native-pbr.json"
            pbr_metrics: dict[str, dict[str, object]] = {}
            for role in asset_bundle.PBR_MATERIAL_ROLES:
                entry = payload["pbr_materials"][role]
                material_file = manifest.parent / entry["material_file"]["path"]
                pbr_metrics[role] = {
                    "material_file": str(material_file.resolve()),
                    "material_file_sha256": _sha256(material_file),
                    "material_prim_path": entry["material_prim_path"],
                    "used_layer_count": 1,
                    "used_layers": [str(material_file.resolve())],
                    "reachable_shader_prim_count": 4,
                    "connected_surface_output_count": 1,
                    "connected_displacement_output_count": 0,
                    "connected_textures": {
                        semantic: [
                            str(
                                (
                                    manifest.parent
                                    / texture_record["path"]
                                ).resolve()
                            )
                        ]
                        for semantic, texture_record in entry["textures"].items()
                    },
                }
            pbr_receipt.write_text(
                json.dumps(
                    {
                        "state": "NATIVE_PBR_MATERIALS_VALIDATED",
                        "manifest_sha256": _sha256(manifest),
                        "bundle_content_inventory_sha256": marker[
                            "content_inventory_sha256"
                        ],
                        "material_roles": list(asset_bundle.PBR_MATERIAL_ROLES),
                        "structural_materials_sha256": (
                            asset_bundle._canonical_sha256(pbr_structural)
                        ),
                        "native_material_metrics_sha256": (
                            asset_bundle._canonical_sha256(pbr_metrics)
                        ),
                        "materials": pbr_metrics,
                    }
                ),
                encoding="utf-8",
            )
            reuse = asset_bundle.verify_native_quality_receipts(
                manifest_path=manifest,
                volume_root=volume,
                bundle_root=destination,
                native_lod_receipt=volume / "contracts" / "native-lods.json",
                native_pbr_receipt=pbr_receipt,
            )
            self.assertEqual(
                reuse["state"],
                "NATIVE_ASSET_BUNDLE_RECEIPTS_CURRENT",
            )

            def flat_metrics(
                path: Path,
                **kwargs: object,
            ) -> dict[str, object]:
                value = metrics(path, **kwargs)
                value["geometry_point_count"] = 100
                value["geometry_face_count"] = 100
                return value

            with patch.object(
                asset_bundle,
                "_native_usd_metrics",
                side_effect=flat_metrics,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "not a real decreasing"
                ):
                    asset_bundle.validate_native_lod_quality(
                        manifest_path=destination / "manifest-v3.json",
                        volume_root=volume,
                        bundle_root=destination,
                        receipt_path=volume / "contracts" / "invalid-lods.json",
                    )

    def test_native_dependency_paths_must_belong_to_locked_bundle_inventory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            material = bundle / "material.usda"
            texture = bundle / "texture.png"
            outside = root / "outside.png"
            material.write_text("#usda 1.0\n", encoding="utf-8")
            texture.write_bytes(b"locked")
            outside.write_bytes(b"not-locked")
            locked = {material.resolve(), texture.resolve()}
            resolved = asset_bundle._resolved_asset_path(
                asset_path=SimpleNamespace(
                    path="texture.png",
                    resolvedPath=str(texture),
                ),
                material_file=material,
                bundle_root=bundle,
                locked_paths=locked,
            )
            self.assertEqual(resolved, texture.resolve())
            with self.assertRaisesRegex(RuntimeError, "outside its locked bundle"):
                asset_bundle._resolved_asset_path(
                    asset_path=SimpleNamespace(
                        path=str(outside),
                        resolvedPath=str(outside),
                    ),
                    material_file=material,
                    bundle_root=bundle,
                    locked_paths=locked,
                )

            external_stage = SimpleNamespace(
                GetUsedLayers=lambda: [
                    SimpleNamespace(
                        identifier=str(outside),
                        realPath=str(outside),
                    )
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "outside its locked bundle"):
                asset_bundle._native_used_layers(
                    stage=external_stage,
                    bundle_root=bundle,
                    locked_paths=locked,
                    label="test stage",
                )

    def test_native_used_layers_ignores_only_the_stage_session_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            root_layer = bundle / "hero.usdc"
            root_layer.write_bytes(b"locked")
            session = SimpleNamespace(
                identifier="anon:0x123:hero-session.usda",
                realPath="",
            )
            stage = SimpleNamespace(
                GetSessionLayer=lambda: session,
                GetUsedLayers=lambda: [
                    SimpleNamespace(
                        identifier=session.identifier,
                        realPath="",
                    ),
                    SimpleNamespace(
                        identifier=str(root_layer),
                        realPath=str(root_layer),
                    ),
                ],
            )

            used = asset_bundle._native_used_layers(
                stage=stage,
                bundle_root=bundle,
                locked_paths={root_layer.resolve()},
                label="test stage",
            )

            self.assertEqual(used, [str(root_layer.resolve())])

            stage.GetUsedLayers = lambda: [
                session,
                SimpleNamespace(
                    identifier="anon:0x456:foreign.usda",
                    realPath="",
                ),
            ]
            with self.assertRaisesRegex(
                RuntimeError,
                "non-local USD layer: anon:0x456:foreign.usda",
            ):
                asset_bundle._native_used_layers(
                    stage=stage,
                    bundle_root=bundle,
                    locked_paths={root_layer.resolve()},
                    label="test stage",
                )

    def test_native_asset_path_extraction_ignores_only_unset_values(self) -> None:
        class FakeAssetPath:
            def __init__(self, path: str = "", resolved_path: str = "") -> None:
                self.path = path
                self.resolvedPath = resolved_path

        authored = FakeAssetPath("textures/albedo.png")
        resolved = FakeAssetPath(resolved_path="/locked/textures/albedo.png")
        pxr = SimpleNamespace(Sdf=SimpleNamespace(AssetPath=FakeAssetPath))

        with patch.dict(sys.modules, {"pxr": pxr}):
            extracted = asset_bundle._asset_paths_from_value(
                [FakeAssetPath(), authored, resolved]
            )

        self.assertEqual(extracted, [authored, resolved])


if __name__ == "__main__":
    unittest.main()

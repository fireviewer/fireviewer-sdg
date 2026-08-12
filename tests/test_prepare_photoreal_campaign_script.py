from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "runpod" / "prepare-photoreal-campaign.sh"


class PreparePhotorealCampaignScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = SCRIPT.read_text(encoding="utf-8")

    def test_script_has_strict_non_destructive_shell_contract(self) -> None:
        self.assertTrue(self.script.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("set -Eeuo pipefail", self.script)
        self.assertIn("umask 027", self.script)
        self.assertNotIn("accept-review", self.script)
        self.assertNotIn("simulation-gate", self.script)
        self.assertNotIn("training_capture_campaign", self.script)
        self.assertNotIn("StopPod", self.script)
        self.assertNotIn("terminate", self.script)

    def test_runtime_bridge_proves_pxr_gdal_and_numpy_together(self) -> None:
        self.assertIn("run_native_python()", self.script)
        self.assertIn("GEOSPATIAL_SITE_PACKAGES", self.script)
        self.assertIn("LD_LIBRARY_PATH", self.script)
        self.assertIn("from osgeo import gdal, osr", self.script)
        self.assertIn("from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade", self.script)
        self.assertIn('"state":"NATIVE_TERRAIN_RUNTIME_READY"', self.script)

    def test_exact_four_base_full_zone_build_precedes_terrain(self) -> None:
        self.assertIn("FW_OMNI_BASE_ZONES must contain exactly four zones", self.script)
        self.assertIn("FW_OMNI_BASE_ZONES contains a duplicate zone", self.script)
        self.assertIn("FW_OMNI_LIDAR_SCOPE=full-zone", self.script)
        self.assertIn('bash "${SETUP_SCRIPT}" bases', self.script)
        runtime = self.script.index("verify_native_runtime\n")
        bases = self.script.index("ensure_four_bases\n", runtime)
        terrain = self.script.index("ensure_terrain_and_composition\n", bases)
        self.assertLess(runtime, bases)
        self.assertLess(bases, terrain)

    def test_all_four_bases_author_exact_400_tile_pbr_and_real_composition(self) -> None:
        self.assertIn('for zone in "${BASE_ZONES[@]}"', self.script)
        self.assertIn("-m fireviewer_sdg.terrain_pbr prepare-native", self.script)
        self.assertIn("-m fireviewer_sdg.terrain_pbr author-native", self.script)
        self.assertIn("COMPOSITE_GROUND_MATERIAL_NATIVE_VALIDATED", self.script)
        self.assertIn("-m fireviewer_sdg.composition_source build", self.script)
        self.assertIn("-m fireviewer_sdg.composition_source verify", self.script)
        self.assertIn("-m fireviewer_sdg.composition_source export", self.script)
        self.assertIn("COMPOSITION_SOURCE_READY", self.script)
        self.assertIn("BASE_COMPOSITION_READY zone=%s tiles=400", self.script)
        for argument in (
            "--asset-lod-validation",
            "--asset-pbr-validation",
            "--ground-authoring-receipt",
            "--prepared-output",
        ):
            self.assertIn(argument, self.script)

    def test_terrain_pbr_uses_only_the_dedicated_ground_bundle(self) -> None:
        self.assertIn(
            'GROUND_BUNDLE_ROOT="${FW_OMNI_GROUND_BUNDLE_ROOT:-'
            '${VOLUME_ROOT}/input/ground-pbr-4k}"',
            self.script,
        )
        self.assertIn(
            'GROUND_MATERIAL_MANIFEST="${FW_OMNI_GROUND_MATERIAL_MANIFEST:-'
            '${GROUND_BUNDLE_ROOT}/manifest-v3.json}"',
            self.script,
        )
        self.assertIn(
            'GROUND_BUNDLE_MARKER="${GROUND_BUNDLE_ROOT}/'
            '.fireviewer-asset-bundle.json"',
            self.script,
        )
        terrain_function = self.script.split(
            "ensure_terrain_and_composition() {", 1
        )[1].split("\n}", 1)[0]
        self.assertIn('--bundle-root "${GROUND_BUNDLE_ROOT}"', terrain_function)
        self.assertIn(
            '--material-manifest "${GROUND_MATERIAL_MANIFEST}"',
            terrain_function,
        )
        self.assertNotIn('if [[ ! -f "${request}" ]]', terrain_function)
        self.assertNotIn('if [[ -f "${receipt}" ]]', terrain_function)
        self.assertNotIn('--bundle-root "${ASSET_BUNDLE_ROOT}"', terrain_function)
        self.assertNotIn('--material-manifest "${ASSET_MANIFEST}"', terrain_function)
        main_function = self.script.split("main() {", 1)[1].split("\n}", 1)[0]
        self.assertIn('require_directory "${GROUND_BUNDLE_ROOT}"', main_function)
        self.assertIn('require_file "${GROUND_MATERIAL_MANIFEST}"', main_function)
        self.assertIn('"${GROUND_BUNDLE_MARKER}"', main_function)
        self.assertIn("ASSET_BUNDLE_INSTALLED", main_function)

    def test_resume_paths_verify_or_reuse_atomic_receipts(self) -> None:
        self.assertIn("bases_are_complete()", self.script)
        self.assertIn("authoring_is_complete()", self.script)
        self.assertIn("VARIANT_AUTHORING_REUSED simulations=20", self.script)
        self.assertIn("variant-plan directory exists without", self.script)
        self.assertIn("variant-scenes exists without", self.script)
        self.assertIn("mktemp --tmpdir=", self.script)
        self.assertIn('mv -- "${temporary}" "${CAMPAIGN_VERIFICATION}"', self.script)

    def test_campaign_is_planned_authored_and_fully_reverified_on_every_run(self) -> None:
        self.assertIn("-m fireviewer_sdg.native_variant_campaign plan", self.script)
        self.assertIn("-m fireviewer_sdg.native_variant_campaign author", self.script)
        self.assertIn("-m fireviewer_sdg.native_variant_campaign verify", self.script)
        self.assertIn("--authoring-receipt", self.script)
        self.assertIn("VARIANT_CAMPAIGN_VERIFIED", self.script)
        campaign_function = self.script.split(
            "ensure_variant_campaign() {", 1
        )[1].split("\n}", 1)[0]
        self.assertNotIn(
            'if [[ -f "${CAMPAIGN_VERIFICATION}" ]]',
            campaign_function,
        )
        self.assertEqual(
            campaign_function.count(
                "-m fireviewer_sdg.native_variant_campaign verify"
            ),
            1,
        )
        self.assertIn(
            'validate_campaign_verification_receipt "${temporary}"',
            campaign_function,
        )
        self.assertIn(
            'mv -- "${temporary}" "${CAMPAIGN_VERIFICATION}"',
            campaign_function,
        )
        self.assertIn(
            'validate_campaign_verification_receipt "${CAMPAIGN_VERIFICATION}"',
            campaign_function,
        )
        validator = self.script.split(
            "validate_campaign_verification_receipt() {", 1
        )[1].split("\nrun_native_python() {", 1)[0]
        for binding in (
            '"plan_sha256":sha256(plan_path)',
            '"authoring_receipt_sha256":sha256(authoring_path)',
            '"root_usd_rehashed":20',
            '"build_receipts_rehashed":20',
            '"identity_contracts_verified":20',
            '"hash_operations"',
            '"bytes_hashed"',
        ):
            self.assertIn(binding, validator)
        self.assertEqual(self.script.count("sha256sum"), 2)

    def test_sim01_gate_internal_qa_then_review_pending(self) -> None:
        self.assertIn("-m fireviewer_sdg.omniverse_scene_gate", self.script)
        self.assertIn(
            "-m fireviewer_sdg.sim01_qa_renderer produce",
            self.script,
        )
        self.assertIn("-m fireviewer_sdg.sim01_quality_gate", self.script)
        self.assertIn("SIM01_INTERNAL_QA_PASSED", self.script)
        qa_function = self.script.split(
            "ensure_sim01_internal_qa() {", 1
        )[1].split("\n}", 1)[0]
        self.assertNotIn('if [[ -f "${QA_RECEIPT}" ]]', qa_function)
        self.assertNotIn('if [[ ! -f "${QA_REVIEW_CAMERA_PLAN}"', qa_function)
        self.assertNotIn("return", qa_function)
        self.assertIn("-m fireviewer_sdg.omniverse_pod review-pending", self.script)
        pending_function = self.script.split(
            "ensure_review_pending() {", 1
        )[1].split("\n}", 1)[0]
        self.assertNotIn(
            'if [[ -f "${REVIEW_PENDING_RECEIPT}" ]]',
            pending_function,
        )
        self.assertNotIn("return", pending_function)
        self.assertIn(
            '--internal-qa-receipt "${QA_RECEIPT}"',
            pending_function,
        )
        scene_gate = self.script.index("ensure_sim01_scene_gate\n")
        internal_qa = self.script.index("ensure_sim01_internal_qa\n", scene_gate)
        review_pending = self.script.index("ensure_review_pending\n", internal_qa)
        self.assertLess(scene_gate, internal_qa)
        self.assertLess(internal_qa, review_pending)
        for evidence in (
            "--review-camera-plan",
            "--proof-pack",
            "--quality-report",
            "--stability-report",
        ):
            self.assertIn(evidence, self.script)
        for expected_output in (
            "review-camera-plan.json",
            "proof-pack.json",
            "quality-report.json",
            "stability-report.json",
        ):
            self.assertIn(expected_output, self.script)

    def test_no_fire_or_human_decision_is_manufactured(self) -> None:
        self.assertIn(
            "No fire simulation, capture campaign, or review acceptance was started.",
            self.script,
        )
        self.assertNotIn("fireviewer_sdg.fire", self.script)
        self.assertNotIn("omni.flow", self.script)
        self.assertNotIn("FIRE_SIMULATION_ALLOWED", self.script)
        self.assertNotIn("--acknowledge", self.script)

    def test_bash_syntax_when_bash_is_available(self) -> None:
        candidates = [
            shutil.which("bash"),
            r"D:\Programs\Git\bin\bash.exe",
            r"C:\Program Files\Git\bin\bash.exe",
        ]
        bash = next(
            (
                candidate
                for candidate in candidates
                if candidate is not None and Path(candidate).is_file()
            ),
            None,
        )
        if bash is None:
            self.skipTest("bash is unavailable on this host")
        if subprocess.run(
            [bash, "--version"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).returncode != 0:
            for candidate in candidates[1:]:
                if candidate is None or not Path(candidate).is_file():
                    continue
                if subprocess.run(
                    [candidate, "--version"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                ).returncode == 0:
                    bash = candidate
                    break
            else:
                self.skipTest(
                    "the discovered bash runtimes cannot start on this host"
                )
        completed = subprocess.run(
            [bash, "-n", str(SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()

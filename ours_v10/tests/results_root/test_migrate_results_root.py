from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.autodl.migrate_results_root import (
    EXPERIMENT_ROOTS,
    MigrationConflict,
    execute_migration,
    plan_migration,
)


EXPECTED_ROOTS = (
    "camera_context",
    "camera_iteration",
    "camera_head_amplification",
    "camera_hidden_state_attribution",
    "camera_hidden_causal_preference",
    "camera_hidden_replacement",
    "camera_hidden_adaptive_alpha",
    "local_global_consistency",
    "camera_refiner_data_construction",
    "vggt_hallucination",
    "camera_refiner_training",
)


class ResultsRootMigrationTest(unittest.TestCase):
    def test_allowlist_is_exact_and_excludes_non_result_roots(self):
        self.assertEqual(EXPERIMENT_ROOTS, EXPECTED_ROOTS)
        self.assertTrue(
            {"datasets", "ckpt", "hf_home", "VGGT_Hallucination"}.isdisjoint(
                EXPERIMENT_ROOTS
            )
        )

    def test_moves_complete_legacy_root_and_second_run_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "camera_context" / "results" / "run_a" / "metric.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("value", encoding="utf-8")

            report = execute_migration(
                plan_migration(root, root / "results"),
                create_links=False,
            )

            destination = root / "results" / "camera_context"
            self.assertEqual(
                (destination / "results" / "run_a" / "metric.json").read_text(
                    encoding="utf-8"
                ),
                "value",
            )
            self.assertFalse((root / "camera_context").exists())
            self.assertIn("camera_context", report["migrated_roots"])

            second = execute_migration(
                plan_migration(root, root / "results"),
                create_links=False,
            )
            self.assertEqual(second["migrated_roots"], [])
            self.assertIn("camera_context", second["canonical_roots"])

    def test_partial_merge_moves_unique_files_and_removes_identical_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "camera_hidden_replacement"
            canonical = root / "results" / "camera_hidden_replacement"
            (legacy / "results" / "run_a").mkdir(parents=True)
            (canonical / "results" / "run_a").mkdir(parents=True)
            (legacy / "results" / "run_a" / "same.json").write_text(
                "same", encoding="utf-8"
            )
            (canonical / "results" / "run_a" / "same.json").write_text(
                "same", encoding="utf-8"
            )
            (legacy / "state").mkdir()
            (legacy / "state" / "calibration_run.txt").write_text(
                "run_a\n", encoding="utf-8"
            )

            report = execute_migration(
                plan_migration(root, root / "results"),
                create_links=False,
            )

            self.assertFalse(legacy.exists())
            self.assertEqual(
                (canonical / "state" / "calibration_run.txt").read_text(
                    encoding="utf-8"
                ),
                "run_a\n",
            )
            self.assertEqual(
                (canonical / "results" / "run_a" / "same.json").read_text(
                    encoding="utf-8"
                ),
                "same",
            )
            self.assertGreaterEqual(report["identical_files_removed"], 1)

    def test_any_conflict_aborts_before_an_unrelated_root_moves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_conflict = root / "camera_context" / "results"
            canonical_conflict = root / "results" / "camera_context" / "results"
            legacy_conflict.mkdir(parents=True)
            canonical_conflict.mkdir(parents=True)
            (legacy_conflict / "summary.json").write_text("old", encoding="utf-8")
            (canonical_conflict / "summary.json").write_text("new", encoding="utf-8")
            unrelated = root / "camera_iteration" / "results" / "run_b"
            unrelated.mkdir(parents=True)
            (unrelated / "summary.json").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(MigrationConflict, "summary.json"):
                plan_migration(root, root / "results")

            self.assertTrue(unrelated.is_dir())
            self.assertFalse((root / "results" / "camera_iteration").exists())

    def test_dry_run_reports_actions_without_changing_filesystem(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "vggt_hallucination" / "results"
            source.mkdir(parents=True)
            (source / "summary.csv").write_text("metric\n", encoding="utf-8")

            report = execute_migration(
                plan_migration(root, root / "results"),
                dry_run=True,
                create_links=False,
            )

            self.assertTrue(source.is_dir())
            self.assertFalse((root / "results" / "vggt_hallucination").exists())
            self.assertTrue(report["dry_run"])
            self.assertIn("vggt_hallucination", report["migrated_roots"])

    def test_wrong_legacy_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong = root / "wrong"
            wrong.mkdir()
            legacy = root / "camera_context"
            try:
                legacy.symlink_to(wrong, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")

            with self.assertRaisesRegex(MigrationConflict, "symlink"):
                plan_migration(root, root / "results")

    def test_results_root_cannot_be_nested_inside_a_migrated_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "camera_context").mkdir()

            with self.assertRaisesRegex(MigrationConflict, "nested"):
                plan_migration(
                    root,
                    root / "camera_context" / "unified_results",
                )


if __name__ == "__main__":
    unittest.main()

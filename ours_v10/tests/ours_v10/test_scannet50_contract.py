"""CPU safety and protocol contracts for the v10 ScanNet-50 adapter."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.ours_v10.scannet50_contract import (
    audit_frame_selection, aggregate_completed, cleanup_regenerable,
    safe_run_paths, validate_metrics,
)


KEYS = ("chamfer_distance", "ate", "are", "rpe_rot", "rpe_trans",
        "inference_time_ms", "scale_factor", "aligned_chamfer_distance",
        "aligned_ate", "aligned_are", "aligned_rpe_rot", "aligned_rpe_trans",
        "aligned_scale_factor")


def metrics(value=.1):
    return {key: float(value) for key in KEYS}


class ScanNet50ContractTest(unittest.TestCase):
    def test_four_budgets_use_baseline_ids_and_short_scene(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared, baselines = root / "prepared", root / "baselines"
            scenes = ("scene0000_00", "scene0013_02")
            for scene, count in zip(scenes, (1200, 80)):
                folder = prepared / scene
                folder.mkdir(parents=True)
                (folder / "manifest.json").write_text(json.dumps(dict(frame_ids=list(range(count)))))
            for budget in (100, 300, 500, 1000):
                folder = baselines / f"vggt_star_f{budget}_scannet50"
                folder.mkdir(parents=True)
                (folder / "summary.json").write_text(json.dumps(dict(complete=True)))
                for scene, count in zip(scenes, (1200, 80)):
                    from scannet_eval.sens import sample_frame_ids
                    ids = sample_frame_ids(list(range(count)), budget)
                    target = folder / scene
                    target.mkdir()
                    (target / "result.json").write_text(json.dumps(dict(frame_ids=ids)))
            rows = audit_frame_selection(scenes, (100, 300, 500, 1000), prepared, baselines)
            self.assertEqual(len(rows), 8)
            self.assertEqual([r["actual_frames"] for r in rows if r["scene_id"] == scenes[1]], [80] * 4)
            self.assertEqual(rows[0]["frame_ids"][:3], [0, 1, 13])
            result = baselines / "vggt_star_f100_scannet50" / scenes[0] / "result.json"
            result.write_text(json.dumps(dict(frame_ids=[0, 1, 2])))
            with self.assertRaisesRegex(ValueError, "VGGT"):
                audit_frame_selection(scenes, (100,), prepared, baselines)

    def test_paths_reject_traversal_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out, scratch = root / "out", root / "scratch"
            out.mkdir(); scratch.mkdir()
            paths = safe_run_paths(out, scratch, "scene0000_00", 100,
                                   allowed_output_base=out, allowed_scratch_base=scratch)
            self.assertTrue(str(paths.output).startswith(str(out)))
            with self.assertRaisesRegex(ValueError, "scene"):
                safe_run_paths(out, scratch, "../escape", 100,
                               allowed_output_base=out, allowed_scratch_base=scratch)
            (scratch / "linked").symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                safe_run_paths(out, scratch / "linked", "scene0000_00", 100,
                               allowed_output_base=out, allowed_scratch_base=scratch)

    def test_incomplete_result_cannot_clean_and_receipt_is_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, scratch = root / "output", root / "scratch"
            output.mkdir(); (scratch / "input_0001").mkdir(parents=True)
            tensor = scratch / "input_0001" / "inputs.pt"
            tensor.write_bytes(b"regenerable input")
            (scratch / "owner.json").write_text(json.dumps(dict(output=str(output.resolve()), run_id="run-1")))
            (output / "summary.json").write_text(json.dumps(dict(complete=False)))
            with self.assertRaisesRegex(ValueError, "complete"):
                cleanup_regenerable(scratch, output, "run-1")
            self.assertTrue(tensor.exists())
            (output / "summary.json").write_text(json.dumps(dict(complete=True)))
            (output / "metrics.json").write_text(json.dumps(metrics()))
            (output / "result.json").write_text(json.dumps(dict(metrics=metrics())))
            for name in ("run_manifest.json", "forward_manifest.json", "stitch_manifest.json",
                         "trajectory.npz", "alignment_edges.json"):
                (output / name).write_bytes(b"retained")
            receipt = cleanup_regenerable(scratch, output, "run-1")
            self.assertEqual(receipt["deleted"][0]["bytes"], len(b"regenerable input"))
            self.assertFalse(tensor.exists())
            self.assertTrue((output / "cleanup_receipt.json").is_file())

    def test_cleanup_refuses_symlink_even_inside_owned_scratch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, scratch = root / "out", root / "scratch"
            output.mkdir(); (scratch / "input_0001").mkdir(parents=True)
            outside = root / "outside"; outside.write_bytes(b"untouchable")
            (scratch / "input_0001" / "inputs.pt").symlink_to(outside)
            (scratch / "owner.json").write_text(json.dumps(dict(output=str(output.resolve()), run_id="id")))
            (output / "summary.json").write_text(json.dumps(dict(complete=True)))
            (output / "metrics.json").write_text(json.dumps(metrics()))
            (output / "result.json").write_text(json.dumps(dict(metrics=metrics())))
            for name in ("run_manifest.json", "forward_manifest.json", "stitch_manifest.json",
                         "trajectory.npz", "alignment_edges.json"):
                (output / name).write_bytes(b"retained")
            with self.assertRaisesRegex(ValueError, "symlink"):
                cleanup_regenerable(scratch, output, "id")
            self.assertEqual(outside.read_bytes(), b"untouchable")

    def test_aggregate_uses_only_complete_scene_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for scene, value, complete in (("scene0000_00", .1, True), ("scene0013_02", 100., False)):
                p = root / "f100" / scene
                p.mkdir(parents=True)
                (p / "summary.json").write_text(json.dumps(dict(complete=complete)))
                (p / "metrics.json").write_text(json.dumps(metrics(value)))
                (p / "result.json").write_text(json.dumps(dict(metrics=metrics(value))))
                for name in ("run_manifest.json", "forward_manifest.json", "stitch_manifest.json",
                             "trajectory.npz", "alignment_edges.json"):
                    (p / name).write_bytes(b"retained")
                if complete:
                    (p / "COMPLETE.json").write_text('{}')
            summary = aggregate_completed(root, 100, ("scene0000_00", "scene0013_02"))
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(summary["average_metrics"]["ate"], .1)
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_metrics({"ate": .1})


if __name__ == "__main__":
    unittest.main()

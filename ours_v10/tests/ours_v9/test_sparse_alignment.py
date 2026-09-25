"""CPU contract tests for v9's opt-in sparse joint Sim(3)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.ours_v3.geometry import Sim3
from vggt.v9.sparse_alignment import (
    SparseAlignmentConfig, V9AlignmentStitcher, align_overlap_sparse,
    select_sparse_correspondences,
)
from vggt.v8.joint_alignment import JointAlignmentConfig, align_overlap_joint


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def predictions(size=32, overlap=("f1",), seed=9):
    rng = np.random.default_rng(seed)
    ids_a = ["f0", *overlap]
    ids_b = [*overlap, "f2"]
    all_ids = list(dict.fromkeys([*ids_a, *ids_b]))
    true = Sim3(1.25, rz(.17), np.array([.3, -.2, .4]))
    maps = {frame: rng.normal(size=(size, size, 3)).astype(np.float32)
            for frame in all_ids}
    centers = {frame: rng.normal(size=3) for frame in all_ids}
    rotations = {frame: rz(.1 * index) for index, frame in enumerate(all_ids)}

    def build(ids, transformed):
        point_maps = np.stack([true.apply(maps[frame]) if transformed else maps[frame]
                               for frame in ids]).astype(np.float32)
        poses = np.repeat(np.eye(4)[None], len(ids), axis=0).astype(np.float32)
        for index, frame in enumerate(ids):
            poses[index, :3, 3] = (true.apply(centers[frame]) if transformed
                                   else centers[frame])
            poses[index, :3, :3] = (true.rotation @ rotations[frame] if transformed
                                    else rotations[frame])
        return dict(frame_ids=list(ids), world_points=point_maps,
                    world_points_conf=np.ones((len(ids), size, size), np.float32),
                    c2w=poses,
                    intrinsics=np.repeat(np.eye(3)[None], len(ids), axis=0),
                    depth=np.ones((len(ids), size, size, 1), np.float32),
                    confidence=np.ones((len(ids), size, size), np.float32))

    return build(ids_a, True), build(ids_b, False), true


class SparseAlignmentTest(unittest.TestCase):
    def test_deterministic_grid_and_frame_pixel_correspondence(self):
        a, b, true = predictions(overlap=("f1", "f1b"))
        a["world_points_conf"][1, 1, 1] = 5
        b["world_points_conf"][0, 1, 1] = 5
        first = select_sparse_correspondences(a, b, SparseAlignmentConfig())
        second = select_sparse_correspondences(a, b, SparseAlignmentConfig())
        self.assertEqual(first.index_sha256, second.index_sha256)
        self.assertEqual(first.frame_ids, ["f1", "f1b"])
        self.assertEqual(first.selected_per_frame, [256, 256])
        self.assertEqual(len(first.source), 512)
        self.assertIn(("f1", 1, 1), first.indices)
        self.assertEqual(len(first.indices), len(set(first.indices)))
        self.assertTrue(np.all(first.confidence_a > first.confidence_threshold))
        self.assertTrue(np.all(first.confidence_b > first.confidence_threshold))
        np.testing.assert_allclose(true.apply(first.source), first.target, atol=3e-6)
        for frame in first.frame_ids:
            positions = [(r // 2, c // 2) for f, r, c in first.indices if f == frame]
            self.assertEqual(len(positions), len(set(positions)))

    def test_confidence_and_nonfinite_candidates_are_filtered(self):
        a, b, _ = predictions()
        a["world_points_conf"][1, 0, 0] = 0
        b["world_points"][0, 0, 1, :] = np.nan
        selected = select_sparse_correspondences(a, b, SparseAlignmentConfig())
        self.assertNotIn(("f1", 0, 0), selected.indices)
        self.assertNotIn(("f1", 0, 1), selected.indices)
        self.assertEqual(len(selected.source), 256)
        self.assertGreater(selected.candidate_per_frame[0], len(selected.source))

    def test_sparse_initializer_never_calls_full_alignment(self):
        a, b, true = predictions()
        with patch("vggt.v9.sparse_alignment.full_align_overlap_joint",
                   side_effect=AssertionError("full initializer called")):
            transform, stats = align_overlap_sparse(a, b, SparseAlignmentConfig())
        self.assertFalse(stats["fallback"])
        self.assertEqual(stats["pairs"], 256)
        self.assertEqual(stats["direction"], "B_local -> A_local")
        self.assertAlmostEqual(transform.scale, true.scale, places=4)
        np.testing.assert_allclose(transform.rotation, true.rotation, atol=2e-4)
        np.testing.assert_allclose(transform.translation, true.translation, atol=2e-4)
        self.assertGreaterEqual(stats["timing_seconds"]["initialization"], 0)
        self.assertGreaterEqual(stats["timing_seconds"]["optimization"], 0)

    def test_coverage_failure_falls_back_and_records_reason(self):
        a, b, _ = predictions()
        a["world_points_conf"][1] = 0
        b["world_points_conf"][0] = 0
        a["world_points_conf"][1, :8, :8] = 1
        b["world_points_conf"][0, :8, :8] = 1
        _, stats = align_overlap_sparse(a, b, SparseAlignmentConfig())
        self.assertTrue(stats["fallback"])
        self.assertIn("coverage", stats["fallback_reason"])
        self.assertEqual(stats["alignment_mode"], "sparse_point_camera_joint")
        self.assertIn("fallback_full_seconds", stats["timing_seconds"])

    def test_optimizer_failure_uses_the_unchanged_full_joint_fit(self):
        a, b, _ = predictions()
        expected, _ = align_overlap_joint(a, b, JointAlignmentConfig(
            mode="point_camera_joint"))
        with patch("vggt.v9.sparse_alignment._optimize",
                   side_effect=RuntimeError("simulated nonconvergence")):
            actual, stats = align_overlap_sparse(a, b, SparseAlignmentConfig())
        self.assertTrue(stats["fallback"])
        self.assertIn("nonconvergence", stats["fallback_reason"])
        self.assertEqual(stats["selected_pairs"], 256)
        self.assertEqual(stats["pairs"], 1024)
        self.assertEqual(actual.scale, expected.scale)
        np.testing.assert_array_equal(actual.rotation, expected.rotation)
        np.testing.assert_array_equal(actual.translation, expected.translation)

    def test_stitch_composition_and_front_ownership(self):
        a, b, true = predictions()
        with tempfile.TemporaryDirectory() as directory:
            stitch = V9AlignmentStitcher(Path(directory) / "alignment",
                                         SparseAlignmentConfig())
            first, _ = stitch.add(a, 0)
            second, _ = stitch.add(b, 1)
            result = stitch.finish(["f0", "f1", "f2"])
            self.assertEqual(first, [0, 1])
            self.assertEqual(second, [1])
            self.assertEqual(result["source_window"].tolist(), [0, 0, 1])
            edge = json.loads((Path(directory) / "alignment" /
                               "edge_0000_0001.json").read_text())
            self.assertEqual(edge["status"], "success")
            self.assertEqual(edge["direction"], "B_local -> A_local")
            self.assertAlmostEqual(edge["adjacent"]["scale"], true.scale, places=4)
            np.testing.assert_allclose(result["c2w"][2, :3, 3],
                                       true.apply(b["c2w"][1, :3, 3]), atol=2e-4)

    def test_three_window_transform_composition(self):
        a, b, true = predictions()
        c = {key: value.copy() if isinstance(value, np.ndarray) else list(value)
             for key, value in b.items()}
        c["frame_ids"] = ["f2", "f3"]
        c["world_points"][0] = b["world_points"][1]
        c["c2w"][0] = b["c2w"][1]
        c["world_points"][1] = b["world_points"][1] + 1
        c["c2w"][1, :3, 3] = b["c2w"][1, :3, 3] + 1
        with tempfile.TemporaryDirectory() as directory:
            stitch = V9AlignmentStitcher(Path(directory) / "alignment",
                                         SparseAlignmentConfig())
            for index, value in enumerate((a, b, c)):
                stitch.add(value, index)
            result = stitch.finish(["f0", "f1", "f2", "f3"])
            self.assertEqual(result["source_window"].tolist(), [0, 0, 1, 2])
            np.testing.assert_allclose(stitch.global_transform.rotation,
                                       true.rotation, atol=2e-4)
            self.assertAlmostEqual(stitch.global_transform.scale, true.scale,
                                   places=4)


if __name__ == "__main__":
    unittest.main()

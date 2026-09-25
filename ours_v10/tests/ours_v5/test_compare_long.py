import tempfile
import unittest
from pathlib import Path
import numpy as np

from experiments.compare_long.evaluate import evaluate_poses
from experiments.compare_long.run_scene0150_100 import compare_raw_windows


class TrajectoryProtocolTest(unittest.TestCase):
    def test_one_whole_sequence_sim3_and_exact_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            pose_dir = Path(temporary) / "pose"
            pose_dir.mkdir()
            ids = [f"{i:06d}" for i in range(100)]
            gt = np.repeat(np.eye(4)[None], 100, axis=0)
            index = np.arange(100, dtype=np.float64)
            gt[:, 0, 3] = index * 0.1
            gt[:, 1, 3] = np.sin(index * 0.3)
            gt[:, 2, 3] = np.cos(index * 0.13)
            for frame, pose in zip(ids, gt):
                np.savetxt(pose_dir / f"{frame}.txt", pose)
            pred = gt.copy()
            pred[:, :3, 3] = (gt[:, :3, 3] - [1.0, -2.0, 0.5]) / 1.7
            metrics, arrays = evaluate_poses(ids, pred, temporary)
            self.assertEqual(metrics["frames"], 100)
            self.assertLess(metrics["ate_rmse_m"], 1e-12)
            self.assertLess(metrics["adjacent_translation_rmse_m"], 1e-12)
            self.assertLess(metrics["adjacent_rotation_rmse_deg"], 1e-12)
            self.assertEqual(
                [(r["before"], r["after"]) for r in metrics["boundary_errors"]],
                [("000059", "000060"), ("000089", "000090")],
            )
            self.assertAlmostEqual(metrics["alignment"]["scale"], 1.7)
            np.testing.assert_array_equal(arrays["pred_raw"], pred)

    def test_raw_window_comparison_keeps_duplicate_overlap_instances(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = root / "vggt_long/native_long/_tmp_results_unaligned"
            native.mkdir(parents=True)
            for mode in ("independent", "camera_exchange"):
                for index in range(3):
                    (root / f"ours_v5_{mode}/windows/{index:04d}").mkdir(parents=True)
            for index, length in enumerate((60, 60, 40)):
                pose = np.repeat(np.eye(4)[None], length, axis=0)
                depth = np.ones((length, 1, 1), dtype=np.float32)
                points = np.ones((length, 1, 1, 3), dtype=np.float32)
                native_data = dict(extrinsic=pose, intrinsic=np.repeat(np.eye(3)[None], length, axis=0),
                                   depth=depth, world_points=points, world_points_conf=depth,
                                   depth_conf=depth)
                np.save(native / f"chunk_{index}.npy", native_data)
                for mode in ("independent", "camera_exchange"):
                    data = dict(c2w=pose, intrinsics=native_data["intrinsic"], depth=depth[..., None],
                                world_points=points, world_points_conf=depth, depth_conf=depth)
                    np.savez_compressed(root / f"ours_v5_{mode}/windows/{index:04d}/local.npz", **data)
            rows = compare_raw_windows(root)
            self.assertEqual(len(rows), 9)
            self.assertEqual([r["start"] for r in rows[::3]], [0, 30, 60])
            self.assertTrue(all(r["fields"]["world_points"]["max_abs"] == 0 for r in rows))

    def test_missing_boundary_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            pose_dir = Path(temporary) / "pose"
            pose_dir.mkdir()
            ids = [f"{i:06d}" for i in range(90)]
            poses = np.repeat(np.eye(4)[None], 90, axis=0)
            index = np.arange(90, dtype=np.float64)
            poses[:, 0, 3] = index
            poses[:, 1, 3] = np.sin(index)
            for frame, pose in zip(ids, poses):
                np.savetxt(pose_dir / f"{frame}.txt", pose)
            with self.assertRaisesRegex(ValueError, "boundaries missing"):
                evaluate_poses(ids, poses, temporary)


if __name__ == "__main__":
    unittest.main()

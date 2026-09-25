"""CPU gates for the opt-in Virtual KITTI 1.3.1 Scene20 entrypoints."""
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from experiments.ours_v9.vkitti_131 import (
    evaluate_trajectory, inspect_condition, scene20_windows,
)
from experiments.ours_v9.vkitti_align import validate_prediction_pair
from experiments.ours_v9.vkitti_predict import prediction_values, worker_paths
from experiments.ours_v9.compare_frozen_sparse import sampled_incremental_rss


def fixture(root, count=4):
    raw = root / "extracted"
    rgb = raw / "vkitti_1.3.1_rgb" / "0020" / "clone"
    rgb.mkdir(parents=True)
    pose_dir = raw / "vkitti_1.3.1_extrinsicsgt"
    pose_dir.mkdir()
    centers = np.array([[0., 0, 0], [1., 0, 0], [1., 1, 0], [2., 1, 0]])[:count]
    lines = ["frame r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0.1 0.2 1"]
    for index, center in enumerate(centers):
        Image.new("RGB", (12, 6), (index + 3, 9, 12)).save(rgb / f"{index:05d}.png")
        pose = np.eye(4)
        pose[:3, 3] = -center
        lines.append(f"{index} " + " ".join(str(x) for x in pose.reshape(-1)))
    (pose_dir / "0020_clone.txt").write_text("\n".join(lines) + "\n")
    return raw, centers


class VirtualKittiEntryTest(unittest.TestCase):
    def test_official_world_to_camera_pose_is_inverted_before_ate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, centers = fixture(root)
            inventory = inspect_condition(raw, "clone", expected_frames=4,
                                          require_full=False)
            self.assertEqual(inventory.frame_ids,
                             ("00000", "00001", "00002", "00003"))
            np.testing.assert_allclose(inventory.gt_c2w[:, :3, 3], centers)
            pred = np.repeat(np.eye(4)[None], 4, axis=0)
            pred[:, :3, 3] = centers
            result = dict(frame_ids=np.asarray(inventory.frame_ids), c2w=pred,
                          source_window=np.array([0, 0, 1, 1]))
            output = root / "evaluation"
            output.mkdir()
            metrics = evaluate_trajectory(result, raw, "clone", output,
                                          expected_frames=4, require_full=False)
            self.assertLess(metrics["ate_rmse_m"], 1e-10)
            self.assertEqual(metrics["protocol_id"],
                             "virtual-kitti-1.3.1-ate-sim3-v1")
            self.assertEqual(metrics["ownership_boundaries"]["count"], 1)

    def test_exact_837_order_tail_window_and_sixteen_edges(self):
        ids = tuple(f"{index:05d}" for index in range(837))
        windows = scene20_windows(ids)
        self.assertEqual(len(windows), 17)
        self.assertEqual(windows[0], (0, 60))
        self.assertEqual(windows[-1], (800, 837))
        self.assertEqual(len(windows) - 1, 16)
        self.assertEqual(scene20_windows(ids[:65]), [(0, 60), (50, 65)])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            scene20_windows((*ids[:10], ids[11], *ids[12:]))

    def test_pair_requires_same_frozen_tensor_weights_and_windows(self):
        base = dict(dataset="Virtual KITTI 1.3.1", scene="Scene20",
                    condition="clone", frame_ids=[f"{i:05d}" for i in range(837)],
                    windows=[[0, 60], [50, 110]],
                    input_sha256="a", image_tensor_sha256="b",
                    checkpoint_sha256="c", precision="bf16",
                    configuration=dict(input="/shared/inputs.pt", frames=837,
                                       window_size=60, overlap=10,
                                       backend_profile="native_vggt",
                                       correspondence_attention_path="native_sdpa",
                                       query_chunk_size=512,
                                       cache_local_kv_dtype=True,
                                       dense_head_frame_chunk=None,
                                       reuse_image_encoding=False))
        # This fixture is intentionally a short malformed window list.
        with self.assertRaisesRegex(ValueError, "window"):
            validate_prediction_pair(dict(base, communication_mode="independent"),
                                     dict(base, communication_mode="overlap_correspondence"))
        from experiments.ours_v6.windows import make_windows
        base["windows"] = make_windows(837, 60, 10)
        validate_prediction_pair(dict(base, communication_mode="independent"),
                                 dict(base, communication_mode="overlap_correspondence"))
        changed = json.loads(json.dumps(base))
        changed["image_tensor_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "image_tensor_sha256"):
            validate_prediction_pair(dict(base, communication_mode="independent"),
                                     dict(changed, communication_mode="overlap_correspondence"))

    def test_local_prediction_export_has_only_alignment_fields(self):
        import torch
        pred = dict(frame_ids=["00000"], c2w=torch.eye(4)[None],
                    intrinsics=torch.eye(3)[None], depth=torch.ones(1, 2, 2),
                    world_points=torch.ones(1, 2, 2, 3),
                    world_points_conf=torch.ones(1, 2, 2),
                    pose_encoding=torch.zeros(1, 9), confidence=torch.ones(1, 2, 2))
        result = prediction_values(pred, ["00000"])
        self.assertEqual(set(result), {"frame_ids", "c2w", "intrinsics", "depth",
                                       "world_points", "world_points_conf"})
        self.assertNotIn("gt_c2w", result)
        pred["world_points"][0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            prediction_values(pred, ["00000"])

    def test_worker_reads_shared_input_and_writes_mode_subdirectory(self):
        root = Path("/tmp/unique_vkitti_output")
        manifest, mode_dir = worker_paths(root, "independent")
        self.assertEqual(manifest, root / "input_manifest.json")
        self.assertEqual(mode_dir, root / "independent")
        with self.assertRaisesRegex(ValueError, "worker mode"):
            worker_paths(root, "wrong")

    def test_edge_rss_is_distinct_from_process_peak(self):
        with sampled_incremental_rss(.001) as stats:
            sample = bytearray(128 * 1024)
            sample[0] = 1
        self.assertGreaterEqual(stats["peak_bytes"], stats["baseline_bytes"])
        self.assertEqual(stats["incremental_peak_bytes"],
                         stats["peak_bytes"] - stats["baseline_bytes"])

    def test_failed_cpu_entry_keeps_failed_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "new_result"
            command = [sys.executable, "-m", "experiments.ours_v9.vkitti_align",
                "--prediction-root", str(root / "absent"), "--condition", "clone",
                "--output", str(output), "--smoke"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((output / "FAILED.json").is_file())
            self.assertFalse((output / "COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()

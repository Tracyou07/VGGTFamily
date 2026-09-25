"""CPU checks for opt-in independent window streaming."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.ours_v6.windows import make_windows
from experiments.ours_v8.stream_independent import (DenseDiskWriter,
    numpy_prediction, run_independent_window)
from experiments.ours_v8.worker import parse_args
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.layers.patch_embed import PatchEmbed
from vggt.models.aggregator import Aggregator
from vggt.models.vggt import VGGT
from vggt.v8.model import WindowReconstructor
import vggt.v8.model as v8_model_module


def tiny_model():
    torch.manual_seed(831)
    model = VGGT.__new__(VGGT)
    nn.Module.__init__(model)
    model.aggregator = Aggregator(
        img_size=8, patch_size=2, embed_dim=8, depth=4, num_heads=2,
        num_register_tokens=1, patch_embed="conv",
        cached_layer_indices=(0, 1, 2, 3))
    model.aggregator.patch_embed = PatchEmbed(img_size=8, patch_size=2, embed_dim=8)
    model.camera_head = CameraHead(dim_in=16, trunk_depth=1, num_heads=2)
    model.camera_head.pose_branch.fc2.weight[7:].zero_()
    model.camera_head.pose_branch.fc2.bias[7:] = .25
    kwargs = dict(dim_in=16, patch_size=2, features=8, out_channels=[8] * 4,
                  intermediate_layer_idx=[0, 1, 2, 3])
    model.depth_head = DPTHead(**kwargs, output_dim=2, activation="exp")
    model.point_head = DPTHead(**kwargs, output_dim=4)
    model.track_head = None
    return model.eval().requires_grad_(False)


class StreamIndependentTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.was_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    def tearDown(self):
        torch.set_grad_enabled(self.was_grad_enabled)

    def test_one_window_at_a_time_matches_original_for_overlap_and_tail(self):
        model = tiny_model()
        wrapper = WindowReconstructor(model)
        keys = ("pose_encoding", "c2w", "intrinsics", "depth", "depth_conf",
                "confidence", "world_points", "world_points_conf")
        for frames, window_size, overlap in ((7, 4, 2), (9, 4, 0), (3, 4, 1)):
            images = torch.rand(frames, 3, 8, 8)
            ids = [f"original-{index}" for index in range(frames)]
            old = wrapper(images, ids, mode="independent",
                          window_size=window_size, overlap=overlap)
            observed_window_counts = []
            real_aggregate = v8_model_module.aggregate_windows

            def spy(*args, **kwargs):
                observed_window_counts.append(len(args[3]))
                return real_aggregate(*args, **kwargs)

            with patch.object(v8_model_module, "aggregate_windows", side_effect=spy):
                for index, (lo, hi) in enumerate(make_windows(frames, window_size, overlap)):
                    new, timing, memory, head_bytes = run_independent_window(
                        wrapper, images, ids, lo, hi, window_size, overlap, 512,
                        cache_local_kv_dtype=True,
                        correspondence_attention_path="native_sdpa")
                    self.assertEqual(new["frame_ids"], ids[lo:hi])
                    self.assertEqual(set(keys).issubset(new), True)
                    self.assertGreater(timing["head_seconds"], 0)
                    self.assertGreater(head_bytes, 0)
                    self.assertEqual(head_bytes, memory["head_cache_bytes"])
                    for key in keys:
                        torch.testing.assert_close(new[key], old["predictions"][index][key],
                                                   atol=2e-5, rtol=2e-5)
            self.assertEqual(observed_window_counts, [1] * len(old["windows"]))

    def test_dense_disk_writer_preserves_front_window_ownership(self):
        with tempfile.TemporaryDirectory() as folder:
            scratch = Path(folder) / "dense"
            def sample(values):
                values = np.asarray(values, dtype=np.float32)
                return {key: values.reshape(-1, 1) for key in
                        ("c2w", "intrinsics", "depth", "confidence",
                         "world_points", "world_points_conf")}
            first = sample([10, 11, 12])
            second = sample([99, 13])
            writer = DenseDiskWriter(scratch, 4, first)
            writer.append(first, [0, 1, 2])
            writer.append(second, [1])
            for array in writer.arrays().values():
                np.testing.assert_array_equal(array[:, 0], [10, 11, 12, 13])
            writer.cleanup_success()
            self.assertFalse(scratch.exists())

    def test_cpu_prediction_conversion_keeps_all_fields_for_export_and_cloud(self):
        prediction = dict(frame_ids=["a"],
            world_points=torch.ones(1, 4, 4, 3),
            world_points_conf=torch.ones(1, 4, 4),
            depth=torch.ones(1, 4, 4, 1),
            confidence=torch.ones(1, 4, 4),
            c2w=torch.eye(4)[None],intrinsics=torch.eye(3)[None])
        converted = numpy_prediction(prediction)
        self.assertEqual(set(converted), set(prediction))
        self.assertTrue(all(not torch.is_tensor(value) for value in converted.values()))
        with tempfile.TemporaryDirectory() as folder:
            from experiments.ours_v6.runtime import write_point_cloud
            output = Path(folder) / "cloud.ply"
            write_point_cloud(output, converted, torch.ones(1, 3, 4, 4), [0], 0)
            self.assertTrue(output.exists())

    def test_opt_in_and_mode_restrictions(self):
        base = ["--input", "fixed.pt", "--output", "new_run", "--gpu", "4",
                "--frames", "100", "--mode", "independent"]
        self.assertFalse(parse_args(base).stream_independent)
        self.assertTrue(parse_args(base + ["--whole-task-measurement",
                                           "--stream-independent"]).stream_independent)
        for invalid in (
            base + ["--stream-independent"],
            base + ["--whole-task-measurement", "--stream-independent",
                    "--reuse-image-encoding"],
            base[:-1] + ["overlap_correspondence", "--whole-task-measurement",
                         "--stream-independent"],
        ):
            with self.assertRaises(SystemExit):
                parse_args(invalid)


if __name__ == "__main__":
    unittest.main()

import inspect
import unittest

import torch
from vggt.models.aggregator import Aggregator
from vggt.layers.patch_embed import PatchEmbed
from vggt.v7.scheduler import aggregate_windows, initialize
from vggt.v7.model import WindowReconstructor
from experiments.ours_v6.windows import make_windows

torch.set_num_threads(1)


def tiny():
    torch.manual_seed(71)
    aggregator = Aggregator(img_size=4, patch_size=2, embed_dim=8, depth=3,
                            num_heads=2, num_register_tokens=2,
                            patch_embed="conv", cached_layer_indices=(0, 2))
    aggregator.patch_embed = PatchEmbed(img_size=4, patch_size=2, embed_dim=8)
    return aggregator.double().eval().requires_grad_(False)


class ScheduleTest(unittest.TestCase):
    def setUp(self):
        self.aggregator = tiny()
        self.images = torch.rand(7, 3, 4, 4, dtype=torch.float64)
        self.windows = make_windows(7, 4, 2)

    def test_windows_and_tail(self):
        self.assertEqual(make_windows(100, 60, 30), [(0, 60), (30, 90), (60, 100)])
        self.assertEqual(make_windows(1000, 60, 30)[-1], (960, 1000))
        for params in ((0, 60, 30), (5, 0, 0), (5, 3, 3)):
            with self.assertRaises(ValueError):
                make_windows(*params)

    def test_reference_and_duplicate_overlap_states(self):
        states, positions, count = initialize(self.aggregator, self.images, self.windows)
        self.assertEqual(len(states), 3)
        self.assertEqual(count, self.aggregator.patch_start_idx + 4)
        for state, (lo, hi) in zip(states, self.windows):
            camera = state.reshape(1, hi - lo, count, 8)[0, :, 0]
            torch.testing.assert_close(camera[0], self.aggregator.camera_token[0, 0, 0])
            torch.testing.assert_close(camera[1:], self.aggregator.camera_token[0, 1, 0].expand_as(camera[1:]))
        before = states[1].clone()
        states[0][0, 2 * count + 1] += 1
        self.assertTrue(torch.equal(before, states[1]))
        self.assertIsNot(positions[0], positions[1])

    def test_independent_matches_original_every_cached_layer(self):
        actual, patch, memory, selection = aggregate_windows(self.aggregator, self.images, self.windows, "independent")
        self.assertEqual(patch, self.aggregator.patch_start_idx)
        for cached, (lo, hi) in zip(actual, self.windows):
            expected, _ = self.aggregator(self.images[lo:hi][None])
            for got, reference in zip(cached, expected):
                if reference is None:
                    self.assertIsNone(got)
                else:
                    torch.testing.assert_close(got, reference, atol=1e-9, rtol=1e-9)
        self.assertGreater(memory["window_state_bytes"], 0)
        self.assertGreater(memory["head_cache_bytes"], 0)
        self.assertEqual(selection["selected_count"], 1)
        self.assertEqual(selection["indices"], [0])

    def test_three_modes_single_window_and_order(self):
        original = self.aggregator(self.images[None])[0]
        for mode in ("independent", "camera_exchange", "camera_patch_exchange"):
            one, _, _, _ = aggregate_windows(self.aggregator, self.images, [(0, 7)], mode)
            for got, reference in zip(one[0], original):
                if reference is not None:
                    torch.testing.assert_close(got, reference, atol=1e-9, rtol=1e-9)
            front, _, _, _ = aggregate_windows(self.aggregator, self.images, self.windows, mode)
            reverse, _, _, _ = aggregate_windows(self.aggregator, self.images, self.windows, mode, reverse=True)
            for first, second in zip(front, reverse):
                for a, b in zip(first, second):
                    if a is not None:
                        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)

    def test_bank_reuses_source_rope_positions(self):
        from vggt.v6.attention import project_qkv
        from vggt.v7.attention import communication_bank, selected_indices
        states, positions, count = initialize(self.aggregator, self.images, self.windows)
        block = self.aggregator.global_blocks[0]
        camera = int(self.aggregator.camera_token.shape[2])
        register = int(self.aggregator.register_token.shape[2])
        idx = selected_indices(states[0].shape[1], count, camera, register,
                               (0,), "camera_patch_exchange", states[0].device)
        full = project_qkv(block.attn, block.norm1(states[0]), positions[0])
        bank = communication_bank(block, states[0], positions[0], count,
                                  camera, register, (0,), "camera_patch_exchange")
        torch.testing.assert_close(bank[0], full[1][:, :, idx], atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(bank[1], full[2][:, :, idx], atol=1e-9, rtol=1e-9)

    def test_ratio_zero_reduces_to_camera_and_selection_reused(self):
        from unittest.mock import patch
        from vggt.v7 import scheduler
        observed = []
        original = scheduler.global_step
        def traced(*args, **kwargs):
            observed.append(tuple(args[7]))
            return original(*args, **kwargs)
        with patch.object(scheduler, "global_step", traced):
            zero, _, _, record = aggregate_windows(self.aggregator, self.images,
                self.windows, "camera_patch_exchange", patch_exchange_ratio=0)
        camera, _, _, _ = aggregate_windows(self.aggregator, self.images,
            self.windows, "camera_exchange", patch_exchange_ratio=0)
        self.assertEqual(record["indices"], [])
        self.assertEqual(observed, [()] * self.aggregator.depth)
        for a, b in zip(zero, camera):
            for x, y in zip(a, b):
                if x is not None:
                    torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_removed_batch_parameter_and_frozen_weights(self):
        for function in (initialize, aggregate_windows, WindowReconstructor.forward):
            self.assertNotIn("batch_size", inspect.signature(function).parameters)
            self.assertNotIn("window_batch_size", inspect.signature(function).parameters)
        snapshot = {name: value.clone() for name, value in self.aggregator.state_dict().items()}
        aggregate_windows(self.aggregator, self.images, self.windows, "camera_patch_exchange")
        self.assertEqual(sum(p.numel() for p in self.aggregator.parameters() if p.requires_grad), 0)
        for name, value in self.aggregator.state_dict().items():
            self.assertTrue(torch.equal(value, snapshot[name]))
        with self.assertRaises(TypeError):
            aggregate_windows(self.aggregator, self.images, self.windows,
                              "camera_patch_exchange", batch_size=2)


if __name__ == "__main__":
    unittest.main()

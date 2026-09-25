import gc
import unittest
import weakref
from unittest.mock import patch

import torch

from tests.ours_v7.test_scheduler import tiny
from vggt.v6.scheduler import initialize
from vggt.v7.scheduler import aggregate_windows
from experiments.ours_v6.windows import make_windows


class ImageEncodingReuseTest(unittest.TestCase):
    def setUp(self):
        self.aggregator = tiny()
        self.images = torch.rand(7, 3, 4, 4, dtype=torch.float64)
        self.windows = make_windows(7, 4, 2)

    def test_one_encoder_call_and_same_window_features(self):
        expected, _, _, _ = aggregate_windows(
            self.aggregator, self.images, self.windows, 'independent')
        with patch.object(self.aggregator.patch_embed, 'forward',
                          wraps=self.aggregator.patch_embed.forward) as encoder:
            actual, _, _, _ = aggregate_windows(
                self.aggregator, self.images, self.windows, 'independent',
                reuse_image_encoding=True)
        self.assertEqual(encoder.call_count, 1)
        self.assertEqual(encoder.call_args.args[0].shape[0], 7)
        for window_actual, window_expected in zip(actual, expected):
            for got, ref in zip(window_actual, window_expected):
                if ref is not None:
                    torch.testing.assert_close(got, ref, atol=1e-9, rtol=1e-9)

    def test_cached_patch_slices_keep_overlap_window_states_independent(self):
        normalized = (self.images[None] - self.aggregator._resnet_mean) / self.aggregator._resnet_std
        patches = self.aggregator.patch_embed(normalized.reshape(7, 3, 4, 4))
        states, _, tokens = initialize(self.aggregator, self.images, self.windows,
                                       precomputed_patch_tokens=patches)
        # Frame 2 is the third frame in window 0 and first frame in window 1.
        before = states[1].clone()
        states[0][0, 2 * tokens + self.aggregator.patch_start_idx] += 1
        torch.testing.assert_close(states[1], before, atol=0, rtol=0)
        self.assertNotEqual(states[0].untyped_storage().data_ptr(),
                            states[1].untyped_storage().data_ptr())

    def test_reuse_does_not_retain_patch_cache_after_aggregation(self):
        original = self.aggregator.patch_embed.forward
        refs = []

        def record(x):
            out = original(x)
            refs.append(weakref.ref(out))
            return out

        with patch.object(self.aggregator.patch_embed, 'forward', side_effect=record):
            aggregate_windows(self.aggregator, self.images, self.windows,
                              'camera_patch_exchange', reuse_image_encoding=True)
        gc.collect()
        self.assertEqual(len(refs), 1)
        self.assertIsNone(refs[0]())

    def test_default_still_encodes_each_window(self):
        with patch.object(self.aggregator.patch_embed, 'forward',
                          wraps=self.aggregator.patch_embed.forward) as encoder:
            aggregate_windows(self.aggregator, self.images, self.windows,
                              'independent')
        self.assertEqual(encoder.call_count, 3)
        self.assertEqual([call.args[0].shape[0] for call in encoder.call_args_list],
                         [4, 4, 3])


if __name__ == '__main__':
    unittest.main()

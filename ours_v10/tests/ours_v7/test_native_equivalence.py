"""CPU tests for the read-only native-equivalence diagnostic path."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from experiments.ours_v7.diagnostic_native_equivalence import (
    native_full_kv_grouped_block, native_manual, record_v7_globals)
from vggt.layers.block import Block
from vggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D
from vggt.models.aggregator import slice_expand_and_flatten
from vggt.v6.scheduler import initialize
from vggt.v7.attention import selected_indices


class FakeAggregator(nn.Module):
    def forward(self, images):
        token = images.mean(dim=(-1, -2)).unsqueeze(2)
        return [torch.cat((token, token), dim=2)], 1


class FakeCamera(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, features):
        self.calls += 1
        return [features[-1][:, :, 0]]


class FakeDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, features, images, patch_start_idx):
        self.calls += 1
        assert patch_start_idx == 1
        assert features[-1].shape[1] == images.shape[1]
        value = images.mean(dim=2, keepdim=False)[..., None]
        return value, value + 1


class NativeEquivalenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def test_full_kv_query_grouping_preserves_native_block(self):
        for rope_enabled in (False, True):
            with self.subTest(rope=rope_enabled):
                rope = RotaryPositionEmbedding2D() if rope_enabled else None
                block = Block(16, 2, qk_norm=True, rope=rope).double().eval()
                x = torch.randn(1, 17, 16, dtype=torch.float64)
                pos = torch.randint(0, 4, (1, 17, 2)) if rope_enabled else None
                native = block(x, pos=pos)
                grouped = native_full_kv_grouped_block(
                    block, x, pos, [(10, 17), (0, 3), (3, 10)])
                torch.testing.assert_close(grouped, native, atol=1e-10, rtol=1e-10)

    def test_grouping_rejects_missing_or_duplicate_query(self):
        block = Block(16, 2).double().eval()
        x = torch.randn(1, 5, 16, dtype=torch.float64)
        with self.assertRaises(ValueError):
            native_full_kv_grouped_block(block, x, None, [(0, 3), (4, 5)])
        with self.assertRaises(ValueError):
            native_full_kv_grouped_block(block, x, None, [(0, 3), (2, 5)])

    def test_manual_native_path_calls_joint_heads_once(self):
        model = SimpleNamespace(aggregator=FakeAggregator(), camera_head=FakeCamera(),
                                depth_head=FakeDense(), point_head=FakeDense())
        images = torch.randn(1, 5, 3, 4, 4)
        outputs = native_manual(model, images)
        self.assertEqual(model.camera_head.calls, 1)
        self.assertEqual(model.depth_head.calls, 1)
        self.assertEqual(model.point_head.calls, 1)
        for field in ("pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"):
            self.assertEqual(outputs[field].shape[1], 5)

    def test_global_diagnostic_wrapper_restores_on_success_and_error(self):
        import vggt.v7.scheduler as scheduler
        original = scheduler.global_step
        calls = []
        capture = SimpleNamespace(on_v7_global=lambda layer, result: calls.append((layer, result)))
        value = [torch.ones(1)]
        with patch.object(scheduler, "global_step", return_value=value) as fake:
            with record_v7_globals(capture):
                self.assertIs(scheduler.global_step(), value)
                self.assertIs(scheduler.global_step(), value)
            self.assertIs(scheduler.global_step, fake)
        self.assertIs(scheduler.global_step, original)
        self.assertEqual([x[0] for x in calls], [0, 1])
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with record_v7_globals(capture):
                raise RuntimeError("deliberate")
        self.assertIs(scheduler.global_step, original)

    def test_full_patch_bank_still_excludes_remote_register(self):
        # 1 camera + 2 register + 4 patch per frame. All four patches selected.
        indices = selected_indices(2 * 7, 7, 1, 2, (0, 1, 2, 3),
                                   "camera_patch_exchange", torch.device("cpu"))
        self.assertEqual(indices.tolist(), [0, 3, 4, 5, 6, 7, 10, 11, 12, 13])
        self.assertNotIn(1, indices.tolist())
        self.assertNotIn(2, indices.tolist())

    def test_overlap_copy_and_window_first_reference(self):
        camera = torch.zeros(1, 2, 1, 8)
        camera[:, 0] = 3
        camera[:, 1] = -2
        register = torch.zeros(1, 2, 2, 8)
        register[:, 0] = 5
        register[:, 1] = -4
        agg = SimpleNamespace(training=False, aa_order=["frame", "global"],
            aa_block_size=1, camera_token=camera, register_token=register,
            patch_start_idx=3, patch_size=2, rope=True,
            position_getter=PositionGetter(),
            patch_embed=lambda x: torch.zeros(x.shape[0], 4, 8),
            _resnet_mean=torch.zeros(1, 1, 3, 1, 1),
            _resnet_std=torch.ones(1, 1, 3, 1, 1))
        images = torch.zeros(4, 3, 4, 4)
        patches = torch.zeros(4, 4, 8)
        states, positions, count = initialize(agg, images, [(0, 3), (2, 4)],
                                               precomputed_patch_tokens=patches)
        native_camera = slice_expand_and_flatten(camera, 1, 4)
        self.assertTrue(torch.equal(states[0].reshape(3, count, 8)[2, 0], native_camera[2, 0]))
        self.assertFalse(torch.equal(states[1].reshape(2, count, 8)[0, 0], native_camera[2, 0]))
        self.assertTrue(torch.equal(states[0].reshape(3, count, 8)[2, 3:],
                                    states[1].reshape(2, count, 8)[0, 3:]))
        self.assertFalse(states[0].untyped_storage().data_ptr() ==
                         states[1].untyped_storage().data_ptr())
        self.assertTrue(torch.equal(positions[0].reshape(3, count, 2)[2],
                                    positions[1].reshape(2, count, 2)[0]))


if __name__ == "__main__":
    unittest.main()

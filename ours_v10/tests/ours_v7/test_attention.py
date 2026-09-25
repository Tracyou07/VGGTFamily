import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from vggt.layers.attention import Attention
from vggt.layers.block import Block
from vggt.v6.attention import project_qkv
from vggt.v7.attention import global_step, selected_indices
from vggt.v7.sampling import select_patch_positions, selection_manifest

torch.set_num_threads(1)


class V7AttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.p = 6  # camera, register, 2x2 patch
        self.states = [torch.randn(1, 12, 8, dtype=torch.float64),
                       torch.randn(1, 18, 8, dtype=torch.float64)]
        self.block = Block(8, 2, qk_norm=True).double().eval()
        self.selected = (0, 3)

    def run_sparse(self, mode, selected=None, states=None, order=None):
        return global_step(self.block, states or self.states, [None, None],
                           self.p, mode, 1, 1,
                           self.selected if selected is None else selected,
                           order=order)

    def dense(self, mode, selected):
        normalized = [self.block.norm1(t) for t in self.states]
        qkv = [project_qkv(self.block.attn, t, None) for t in normalized]
        q, k, v = [torch.cat([a[i] for a in qkv], dim=2) for i in range(3)]
        owner = torch.tensor([j for j, x in enumerate(self.states) for _ in range(x.shape[1])])
        offsets = torch.tensor([i % self.p for x in self.states for i in range(x.shape[1])])
        cross = offsets == 0
        if mode == "camera_patch_exchange":
            for index in selected:
                cross |= offsets == 2 + index
        if mode == "independent":
            cross[:] = False
        mask = (owner[:, None] == owner[None, :]) | (
            cross[:, None] & cross[None, :]
        )
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                               scale=self.block.attn.scale)
        y = self.block.attn.proj_drop(self.block.attn.proj(
            y.transpose(1, 2).reshape(1, -1, 8)
        ))
        pieces = y.split([x.shape[1] for x in self.states], dim=1)
        result = []
        for x, piece in zip(self.states, pieces):
            x = x + self.block.ls1(piece)
            result.append(x + self.block.ls2(self.block.mlp(self.block.norm2(x))))
        return result

    def test_dense_oracle_and_no_all_scene_scores(self):
        original = torch.nn.functional.scaled_dot_product_attention
        calls = []
        def traced(q, k, v, **kwargs):
            self.assertIsNone(kwargs.get("attn_mask"))
            calls.append((q.shape[-2], k.shape[-2]))
            return original(q, k, v, **kwargs)
        for mode in ("independent", "camera_exchange", "camera_patch_exchange"):
            expected = self.dense(mode, self.selected)
            with patch("torch.nn.functional.scaled_dot_product_attention", traced):
                actual = self.run_sparse(mode)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=1e-9, rtol=1e-9)
        self.assertNotIn((30, 30), calls)

    def test_query_tiling_preserves_single_softmax_visibility(self):
        for size in (1, 2, 8, 64, 1024):
            expected = self.dense("camera_patch_exchange", self.selected)
            actual = global_step(self.block, self.states, [None, None], self.p,
                "camera_patch_exchange", 1, 1, self.selected, query_chunk_size=size)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)

    def test_visibility_by_gradient(self):
        block = Block(8, 2).double().eval()
        block.norm1 = nn.Identity()
        block.norm2 = nn.Identity()
        with torch.no_grad():
            block.attn.qkv.weight.zero_()
            block.attn.qkv.bias.zero_()
            block.attn.qkv.weight[16:] = torch.eye(8, dtype=torch.float64)
            block.attn.proj.weight.copy_(torch.eye(8, dtype=torch.float64))
            block.attn.proj.bias.zero_()
            for p in block.mlp.parameters():
                p.zero_()
        x = [t.clone().requires_grad_() for t in self.states]
        y = global_step(block, x, [None, None], self.p,
                        "camera_patch_exchange", 1, 1, self.selected)
        for query in (0, 2, 5):
            grad = torch.autograd.grad(y[0][0, query, 0], x[1], retain_graph=True)[0]
            self.assertGreater(float(grad[0, 0::self.p].abs().sum()), 0)
            self.assertGreater(float(grad[0, 2::self.p].abs().sum()), 0)
            self.assertGreater(float(grad[0, 5::self.p].abs().sum()), 0)
            self.assertEqual(float(grad[0, 1::self.p].abs().max()), 0)
            self.assertEqual(float(grad[0, 3::self.p].abs().max()), 0)
            self.assertEqual(float(grad[0, 4::self.p].abs().max()), 0)
        for query in (1, 3, 4):
            grad = torch.autograd.grad(y[0][0, query, 0], x[1], retain_graph=True, allow_unused=True)[0]
            self.assertTrue(grad is None or float(grad.abs().max()) == 0)

    def test_degenerate_order_and_overlap_copy(self):
        zero = self.run_sparse("camera_patch_exchange", selected=())
        camera = self.run_sparse("camera_exchange")
        for a, b in zip(zero, camera):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        all_patch = tuple(range(4))
        for a, b in zip(self.run_sparse("camera_patch_exchange", selected=all_patch),
                        self.dense("camera_patch_exchange", all_patch)):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
        forward = self.run_sparse("camera_patch_exchange")
        reverse = self.run_sparse("camera_patch_exchange", order=[1, 0])
        for a, b in zip(forward, reverse):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        for mode in ("independent", "camera_exchange", "camera_patch_exchange"):
            single = global_step(self.block, [self.states[0]], [None], self.p, mode,
                                 1, 1, self.selected)[0]
            torch.testing.assert_close(single, self.block(self.states[0]), atol=0, rtol=0)
        copied = [x.clone() for x in self.states]
        copied[0][0, 8, 0] += 5  # same logical frame can have a different window state
        self.assertTrue(torch.equal(copied[1], self.states[1]))

    def test_bf16_cpu_if_available(self):
        block = Block(8, 2).eval().requires_grad_(False)
        x = [t.float() for t in self.states]
        try:
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                result = global_step(block, x, [None, None], self.p,
                                     "camera_patch_exchange", 1, 1, self.selected)
        except RuntimeError as exc:
            self.skipTest(f"CPU BF16 unsupported: {exc}")
        self.assertTrue(all(torch.isfinite(t).all() for t in result))


class SamplingTest(unittest.TestCase):
    def test_exact_unique_deterministic_and_bounds(self):
        for h, w, ratio in ((28, 37, 0.1), (3, 7, 0.1), (2, 4, 0),
                            (2, 4, 1), (1, 1, 0.1)):
            got = select_patch_positions(h, w, ratio)
            self.assertEqual(got, select_patch_positions(h, w, ratio))
            self.assertEqual(len(got), min(h*w, math.ceil(ratio*h*w)))
            self.assertEqual(len(got), len(set(got)))
            self.assertTrue(all(0 <= i < h*w for i in got))
            record = selection_manifest(h, w, ratio)
            self.assertEqual(record["indices"], list(got))
            self.assertEqual(record["actual_ratio"], len(got)/(h*w))
        with self.assertRaises(ValueError):
            select_patch_positions(2, 4, 1.1)

    def test_spatial_coverage_and_frame_expansion(self):
        picked = select_patch_positions(20, 30, 0.1)
        quadrants = {(i // 30 >= 10, i % 30 >= 15) for i in picked}
        self.assertEqual(len(quadrants), 4)
        idx = selected_indices(12, 6, 1, 1, (0, 3), "camera_patch_exchange", torch.device("cpu"))
        self.assertEqual(idx.tolist(), [0, 2, 5, 6, 8, 11])


if __name__ == "__main__":
    unittest.main()

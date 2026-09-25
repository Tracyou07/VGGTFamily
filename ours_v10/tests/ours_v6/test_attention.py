import unittest
from unittest.mock import patch

import torch
from torch import nn
from vggt.layers.attention import Attention
from vggt.layers.block import Block
from vggt.v6.attention import (project_qkv, communication_bank, exchange_attention,
                               global_step, special_indices)

torch.set_num_threads(1)


class TopologyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        self.attn = Attention(8, 2, qk_norm=True).double().eval()
        self.x = [torch.randn(1, 8, 8, dtype=torch.float64),
                  torch.randn(1, 12, 8, dtype=torch.float64)]
        self.p = 4

    def dense_reference(self, mode):
        qs, ks, vs = [], [], []
        owner, kind = [], []
        for window, x in enumerate(self.x):
            q, k, v = project_qkv(self.attn, x, None)
            qs.append(q); ks.append(k); vs.append(v)
            owner.extend([window] * x.shape[1])
            kind.extend([i % self.p for i in range(x.shape[1])])
        q, k, v = [torch.cat(group, dim=2) for group in (qs, ks, vs)]
        owner = torch.tensor(owner)
        kind = torch.tensor(kind)
        if mode == "independent":
            mask = owner[:, None] == owner[None, :]
        elif mode == "camera_exchange":
            mask = (owner[:, None] == owner[None, :]) | ((kind[:, None] == 0) & (kind[None, :] == 0))
        else:
            mask = (owner[:, None] == owner[None, :]) | ((kind[:, None] < 2) & (kind[None, :] < 2))
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = self.attn.proj_drop(self.attn.proj(y.transpose(1, 2).reshape(1, -1, 8)))
        return list(y.split([len(x[0]) for x in self.x], dim=1))

    def sparse(self, mode, x=None):
        x = self.x if x is None else x
        count = 1 if mode == "camera_exchange" else 2
        bank = [communication_bank(nn.Identity(), t, None, self.p, count) for t in x]
        return [exchange_attention(self.attn, t, None, self.p, bank, i, count)
                for i, t in enumerate(x)]

    def test_dense_oracle_and_no_full_score_or_mask(self):
        block = Block(8, 2, qk_norm=True).double().eval()
        block.attn = self.attn
        calls = []
        original = torch.nn.functional.scaled_dot_product_attention
        def traced(q, k, v, **kwargs):
            self.assertIsNone(kwargs.get("attn_mask"))
            calls.append((q.shape[2], k.shape[2]))
            return original(q, k, v, **kwargs)
        # Use the production global_step, including original norm/residual.
        with patch("torch.nn.functional.scaled_dot_product_attention", traced):
            for mode in ("independent", "camera_exchange", "camera_register_exchange"):
                if mode == "independent":
                    expected = [block(t) for t in self.x]
                else:
                    old = [block.norm1(t) for t in self.x]
                    qs, ks, vs = [], [], []
                    owner, kind = [], []
                    for w, t in enumerate(old):
                        q, k, v = project_qkv(self.attn, t, None)
                        qs.append(q); ks.append(k); vs.append(v)
                        owner.extend([w] * len(t[0]))
                        kind.extend([i % self.p for i in range(len(t[0]))])
                    q, k, v = [torch.cat(z, dim=2) for z in (qs, ks, vs)]
                    owner = torch.tensor(owner); kind = torch.tensor(kind)
                    limit = 1 if mode == "camera_exchange" else 2
                    mask = (owner[:, None] == owner[None, :]) | ((kind[:, None] < limit) & (kind[None, :] < limit))
                    y = original(q, k, v, attn_mask=mask)
                    y = self.attn.proj_drop(self.attn.proj(y.transpose(1, 2).reshape(1, -1, 8)))
                    pieces = y.split([len(t[0]) for t in self.x], dim=1)
                    expected = []
                    for state, piece in zip(self.x, pieces):
                        h = state + block.ls1(piece)
                        expected.append(h + block.ls2(block.mlp(block.norm2(h))))
                actual = global_step(block, self.x, [None, None], self.p, mode, 1, 1)
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-9)
        self.assertNotIn((20, 20), calls)

    def controlled(self):
        attention = Attention(8, 2, qkv_bias=False).double().eval()
        with torch.no_grad():
            attention.qkv.weight.zero_()
            attention.qkv.weight[16:] = torch.eye(8, dtype=torch.float64)
            attention.proj.weight.copy_(torch.eye(8, dtype=torch.float64))
            attention.proj.bias.zero_()
        return attention

    def test_direct_dependency_and_patch_isolation(self):
        block = Block(8, 2).double().eval()
        block.norm1 = nn.Identity()
        block.norm2 = nn.Identity()
        block.attn = self.controlled()
        with torch.no_grad():
            for parameter in block.mlp.parameters():
                parameter.zero_()
        x = [t.clone().requires_grad_() for t in self.x]
        y = global_step(block, x, [None, None], self.p, "camera_register_exchange", 1, 1)
        for query in (0, 1):
            gradient = torch.autograd.grad(y[0][0, query, 0], x[1], retain_graph=True)[0]
            self.assertGreater(float(gradient[0, 0::4].abs().sum()), 0)
            self.assertGreater(float(gradient[0, 1::4].abs().sum()), 0)
            self.assertEqual(float(gradient[0, 2::4].abs().max()), 0)
        gradient = torch.autograd.grad(y[0][0, 2, 0], x[1], allow_unused=True)[0]
        self.assertTrue(gradient is None or float(gradient.abs().max()) == 0)
        camera_only = global_step(block, x, [None, None], self.p, "camera_exchange", 1, 1)
        grad = torch.autograd.grad(camera_only[0][0, 0, 0], x[1], retain_graph=True)[0]
        self.assertEqual(float(grad[0, 1::4].abs().max()), 0)

    def test_same_layer_patch_perturbation_and_next_layer_relay(self):
        block = Block(8, 2).double().eval()
        block.norm1 = nn.Identity()
        block.norm2 = nn.Identity()
        block.attn = self.controlled()
        with torch.no_grad():
            for parameter in block.mlp.parameters():
                parameter.zero_()
        baseline = [torch.zeros_like(t) for t in self.x]
        patch_change = [t.clone() for t in baseline]
        patch_change[1][0, 2, 0] = 1
        special_change = [t.clone() for t in baseline]
        special_change[1][0, 1, 0] = 1
        a = global_step(block, baseline, [None, None], self.p, "camera_register_exchange", 1, 1)
        b = global_step(block, patch_change, [None, None], self.p, "camera_register_exchange", 1, 1)
        c = global_step(block, special_change, [None, None], self.p, "camera_register_exchange", 1, 1)
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertGreater(float((a[0][0, 1] - c[0][0, 1]).abs().max()), 0)
        self.assertEqual(float((a[0][0, 2] - c[0][0, 2]).abs().max()), 0)
        frame = self.controlled()
        after_a = frame(a[0].reshape(2, self.p, 8))
        after_c = frame(c[0].reshape(2, self.p, 8))
        self.assertGreater(float((after_a[0, 2] - after_c[0, 2]).abs().max()), 0)

    def test_order_and_single_window_bf16(self):
        block = Block(8, 2).eval().requires_grad_(False)
        x = [torch.randn(1, 8, 8), torch.randn(1, 12, 8)]
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            for mode in ("independent", "camera_exchange", "camera_register_exchange"):
                front = global_step(block, x, [None, None], self.p, mode, 1, 1)
                back = global_step(block, x, [None, None], self.p, mode, 1, 1, order=[1, 0])
                for a, b in zip(front, back):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                one = global_step(block, [x[0]], [None], self.p, mode, 1, 1)[0]
                self.assertTrue(torch.equal(one, block(x[0])))
        with self.assertRaises(TypeError):
            global_step(block, x, [None, None], self.p, "camera_register_exchange",
                        1, 1, batch_size=2)


if __name__ == "__main__":
    unittest.main()

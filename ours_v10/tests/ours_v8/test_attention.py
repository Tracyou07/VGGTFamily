"""Independent dense visibility oracle for v8 overlap correspondence."""
import json
import math
import os
import unittest

import torch
from torch.nn import functional as F

from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D
from vggt.v8.attention import (build_correspondences, global_step,
                               paired_softmax_attention,
                               native_sdpa_attention, prepare_native_kv)

METRICS = []
TOLERANCES = {torch.float64: (1e-10, 1e-10),
              torch.float32: (2e-5, 2e-5)}


def layout(windows, scene_ids, dtype):
    """Manually label each token; never call production index construction."""
    torch.manual_seed(87)
    tokens_per_frame = 7  # camera, two register, 2x2 patch
    states, positions, metadata = [], [], []
    for window, (lo, hi) in enumerate(windows):
        state = torch.randn(1, (hi-lo)*tokens_per_frame, 16, dtype=dtype)
        # Make equal-image overlap copies distinguishable after initialization.
        state[:, ::5] += (window + 1) * 0.4
        states.append(state)
        p = []
        for frame in scene_ids[lo:hi]:
            for offset in range(tokens_per_frame):
                kind = "camera" if offset == 0 else ("register" if offset <= 2 else "patch")
                patch = offset - 3 if kind == "patch" else None
                metadata.append((window, frame, kind, patch))
                p.append((0, 0) if patch is None else (patch // 2 + 1, patch % 2 + 1))
        positions.append(torch.tensor(p, dtype=torch.long).unsqueeze(0))
    return states, positions, metadata, tokens_per_frame


def dense_reference(block, states, positions, metadata, enabled):
    """Full tiny score/mask oracle based solely on explicit token metadata."""
    attn = block.attn
    qkv = []
    for x, pos in zip(states, positions):
        norm = block.norm1(x)
        raw = F.linear(norm, attn.qkv.weight, attn.qkv.bias)
        q, k, v = raw.reshape(1, x.shape[1], 3, attn.num_heads,
                              attn.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)
        q, k = attn.rope(q, pos), attn.rope(k, pos)
        qkv.append((q, k, v))
    q, k, v = [torch.cat([values[i] for values in qkv], dim=2) for i in range(3)]
    mask = torch.tensor([
        [a[0] == b[0] or (enabled and a[2] == "patch" and b[2] == "patch"
                          and a[1] == b[1] and a[3] == b[3] and abs(a[0]-b[0]) == 1)
         for b in metadata] for a in metadata], dtype=torch.bool)
    score = (q @ k.transpose(-2, -1)) * (attn.head_dim ** -0.5)
    pre = score.masked_fill(~mask[None, None], -torch.inf).softmax(-1) @ v
    pre = pre.transpose(1, 2).reshape(1, len(metadata), -1)
    post = F.linear(pre, attn.proj.weight, attn.proj.bias)
    lengths = [state.shape[1] for state in states]
    pre, post = pre.split(lengths, dim=1), post.split(lengths, dim=1)
    full = []
    for x, y in zip(states, post):
        residual = x + block.ls1(y)
        full.append(residual + block.ls2(block.mlp(block.norm2(residual))))
    return pre, post, full, mask


def compare(case, stage, actual, expected, dtype):
    atol, rtol = TOLERANCES[dtype]
    delta = (actual - expected).abs()
    okay = bool(torch.isfinite(actual).all() and
                (delta <= atol + rtol * expected.abs()).all())
    METRICS.append(dict(case=case, stage=stage, dtype=str(dtype),
                        max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                        finite=bool(torch.isfinite(actual).all()), passed=okay))
    if not okay:
        raise AssertionError(f"{case} {stage}: max {float(delta.max())}")


class V8AttentionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self._grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    def tearDown(self):
        torch.set_grad_enabled(self._grad_enabled)

    def check_case(self, windows, dtype, mode, chunk, order=None,
                   attention_path="explicit"):
        scene_ids = tuple(f"real-{i+100}" for i in range(max(hi for _, hi in windows)))
        states, positions, metadata, count = layout(windows, scene_ids, dtype)
        block = Block(16, 2, qk_norm=True, init_values=0.1,
                      rope=RotaryPositionEmbedding2D()).to(dtype).eval()
        mapping = build_correspondences(scene_ids, windows,
            [scene_ids[lo:hi] for lo, hi in windows], [(2,2)]*len(windows),
            count, 1, 2, torch.device("cpu"), mode)
        expected_pre, expected_post, expected_full, mask = dense_reference(
            block, states, positions, metadata, mode == "overlap_correspondence")
        before, after = [], []
        def capture(_module, args, output):
            before.append(args[0].detach().clone())
            after.append(output.detach().clone())
        handle = block.attn.proj.register_forward_hook(capture)
        try:
            actual = global_step(block, states, positions, mapping, mode,
                                 query_chunk_size=chunk, order=order,
                                 attention_path=attention_path)
        finally:
            handle.remove()
        sequence = list(range(len(windows))) if order is None else order
        case = f"windows={windows};dtype={dtype};mode={mode};chunk={chunk};order={sequence}"
        for call, window in enumerate(sequence):
            compare(case, f"before_projection_w{window}", before[call], expected_pre[window], dtype)
            compare(case, f"after_projection_w{window}", after[call], expected_post[window], dtype)
        for window in range(len(windows)):
            compare(case, f"global_block_w{window}", actual[window], expected_full[window], dtype)
        self.assertTrue(bool(mask.diag().all()))
        groups = mapping["groups"]
        for window, (_, _) in enumerate(windows):
            all_cross = [int(i) for q, _ in groups[window].values() for i in q.tolist()]
            all_local = mapping["local_queries"][window].tolist()
            self.assertEqual(sorted(all_cross + all_local), list(range(states[window].shape[1])))
            self.assertEqual(len(all_cross), len(set(all_cross)))
        return block, states, positions, mapping, actual

    def test_native_sdpa_disabled_is_native_local_attention(self):
        for dtype in (torch.float64, torch.float32):
            torch.manual_seed(95)
            q = torch.randn(1, 2, 5, 8, dtype=dtype)
            k = torch.randn(1, 2, 11, 8, dtype=dtype)
            v = torch.randn_like(k)
            rk = torch.randn_like(q); rv = torch.randn_like(q)
            cache = prepare_native_kv(k, v, capacity=5)
            actual, mask_bytes = native_sdpa_attention(
                q, cache, rk, rv, remote_enabled=False, scale=8 ** -0.5)
            expected = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, scale=8 ** -0.5)
            torch.testing.assert_close(actual, expected, atol=1e-10 if dtype==torch.float64 else 2e-5,
                                       rtol=1e-10 if dtype==torch.float64 else 2e-5)
            self.assertEqual(mask_bytes, 5 * 16)

    def test_native_sdpa_joint_matches_explicit_and_dense_block(self):
        for dtype in (torch.float64, torch.float32):
            torch.manual_seed(96)
            q = torch.randn(1, 2, 5, 8, dtype=dtype)
            k = torch.randn(1, 2, 11, 8, dtype=dtype)
            v = torch.randn_like(k)
            rk = torch.randn_like(q); rv = torch.randn_like(q)
            cache = prepare_native_kv(k, v, capacity=5)
            actual, _ = native_sdpa_attention(q, cache, rk, rv, True, 8 ** -0.5)
            expected = paired_softmax_attention(q, k, v, rk, rv, 5)
            torch.testing.assert_close(actual, expected, atol=1e-10 if dtype==torch.float64 else 2e-5,
                                       rtol=1e-10 if dtype==torch.float64 else 2e-5)
            self.check_case([(0,3),(2,5),(4,6)], dtype,
                            "overlap_correspondence", 3,
                            attention_path="native_sdpa")

    def test_two_and_three_windows_both_dtypes(self):
        scenes = ([(0,3),(2,5)], [(0,3),(2,5),(4,6)])
        for windows in scenes:
            for dtype in (torch.float64, torch.float32):
                for chunk in (1, 3, 8):
                    with self.subTest(windows=windows,dtype=dtype,chunk=chunk):
                        self.check_case(windows, dtype, "overlap_correspondence", chunk)

    def test_independent_and_no_overlap_are_original_blocks(self):
        for windows, mode in (([(0,3),(2,5)], "independent"),
                              ([(0,2),(2,5)], "overlap_correspondence"),
                              ([(0,3)], "overlap_correspondence")):
            block, states, positions, mapping, actual = self.check_case(
                windows, torch.float64, mode, 2)
            self.assertEqual(mapping["pair_count"], 0)
            for i, x in enumerate(states):
                self.assertTrue(torch.equal(actual[i], block(x, pos=positions[i])))

    def test_reverse_window_order_and_tail_group(self):
        windows = [(0,3),(2,5),(4,6)]
        block, states, positions, mapping, forward = self.check_case(
            windows, torch.float64, "overlap_correspondence", 3)
        backward = global_step(block, states, positions, mapping,
            "overlap_correspondence", query_chunk_size=3, order=[2,1,0])
        for a, b in zip(forward, backward):
            self.assertTrue(torch.equal(a,b))

    def test_layer_local_conversion_cache_on_off(self):
        block, states, positions, mapping, uncached = self.check_case(
            [(0,3),(2,5)], torch.float32, "overlap_correspondence", 2)
        cached = global_step(block, states, positions, mapping,
            "overlap_correspondence", query_chunk_size=2,
            cache_local_kv_dtype=True)
        for a, b in zip(uncached, cached):
            self.assertTrue(torch.equal(a, b))
        native_uncached = global_step(block, states, positions, mapping,
            "overlap_correspondence", query_chunk_size=2,
            attention_path="native_sdpa")
        native_cached = global_step(block, states, positions, mapping,
            "overlap_correspondence", query_chunk_size=2,
            cache_local_kv_dtype=True, attention_path="native_sdpa")
        for a, b in zip(native_uncached, native_cached):
            self.assertTrue(torch.equal(a, b))
        q = torch.ones((1, 2, 1, 8), dtype=torch.float32)
        k = torch.ones((1, 2, 2, 8), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "conversion cache"):
            paired_softmax_attention(q, k, k, q, q, 1,
                                     local_accumulated=(k.double(), k.double()))

    def test_only_corresponding_remote_v_can_change_target_query(self):
        q = torch.tensor([[[[1., 0.]]]], dtype=torch.float64)
        local_k = torch.tensor([[[[1., 0.], [0., 1.]]]], dtype=torch.float64)
        local_v = torch.tensor([[[[1., 2.], [3., 4.]]]], dtype=torch.float64)
        remote_k = torch.tensor([[[[1., 0.]]]], dtype=torch.float64)
        remote_v = torch.tensor([[[[5., 6.]]]], dtype=torch.float64)
        original = paired_softmax_attention(q, local_k, local_v, remote_k, remote_v, 1)
        changed = paired_softmax_attention(q, local_k, local_v, remote_k, remote_v+10, 1)
        self.assertGreater(float((changed-original).abs().max()), 1.0)
        # A nonmatching remote token is absent from the function's five inputs.
        irrelevant = torch.tensor([[[[999., 999.]]]], dtype=torch.float64)
        irrelevant += 100
        self.assertTrue(torch.equal(original,
            paired_softmax_attention(q, local_k, local_v, remote_k, remote_v, 1)))

    def test_local_query_does_not_read_remote_in_same_block(self):
        windows = [(0,3),(2,5)]
        block, states, positions, mapping, baseline = self.check_case(
            windows, torch.float64, "overlap_correspondence", 2)
        altered = [x.clone() for x in states]
        # Window 1's shared frame 2, patch 0 changes. Window 0's camera,
        # register, other patch and nonoverlap patch must not read it directly.
        altered[1][:, 3, 0] += 7.0
        updated = global_step(block, altered, positions, mapping,
                              "overlap_correspondence", query_chunk_size=2)
        local = mapping["local_queries"][0]
        self.assertTrue(torch.equal(updated[0][:, local], baseline[0][:, local]))
        target = mapping["groups"][0][1][0][0]
        self.assertGreater(float((updated[0][:, target] - baseline[0][:, target]).abs().max()), 1e-5)
        nonmatching = [x.clone() for x in states]
        nonmatching[1][:, 4, 0] += 7.0  # same frame, different patch
        unrelated = global_step(block, nonmatching, positions, mapping,
                                "overlap_correspondence", query_chunk_size=2)
        self.assertTrue(torch.equal(unrelated[0][:, target], baseline[0][:, target]))

    def test_frame_id_grid_and_multiple_match_errors(self):
        ids = tuple(f"f{i}" for i in range(6))
        args = (ids, [(0,3),(2,5)], [ids[0:3], ids[2:5]], [(2,2),(2,2)],
                7, 1, 2, torch.device("cpu"))
        wrong = list(args)
        wrong[2] = [ids[0:3], (ids[3], ids[2], ids[4])]
        with self.assertRaisesRegex(ValueError, "frame IDs"):
            build_correspondences(*wrong)
        wrong = list(args)
        wrong[3] = [(2,2),(1,4)]
        with self.assertRaisesRegex(ValueError, "patch grids differ"):
            build_correspondences(*wrong)
        wrong = list(args)
        wrong[4] = 8
        with self.assertRaisesRegex(ValueError, "patch grid"):
            build_correspondences(*wrong)
        with self.assertRaisesRegex(ValueError, "multiple remote correspondences"):
            build_correspondences(ids, [(0,3),(1,4),(2,5)],
                [ids[0:3],ids[1:4],ids[2:5]], [(2,2)]*3,
                7,1,2,torch.device("cpu"))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(V8AttentionTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    path = os.environ.get("V8_CPU_RESULTS")
    if path:
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(dict(passed=result.wasSuccessful(), tests=result.testsRun,
                           tolerances={str(k):v for k,v in TOLERANCES.items()},
                           metrics=METRICS), stream, indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)

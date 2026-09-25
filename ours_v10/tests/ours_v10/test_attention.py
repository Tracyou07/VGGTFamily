"""Independent dense oracle for v10 global camera and overlap visibility."""
import unittest

import torch
from torch.nn import functional as F

from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D
from vggt.v10.attention import MODES, build_correspondences, global_step

TOL = {torch.float64: (1e-10, 1e-10), torch.float32: (2e-5, 2e-5)}


def fixture(windows, dtype):
    torch.manual_seed(145)
    ids = tuple(f"frame{i}" for i in range(max(hi for _, hi in windows)))
    states, positions, tokens = [], [], []
    for w, (lo, hi) in enumerate(windows):
        x = torch.randn(1, (hi-lo)*4, 16, dtype=dtype)
        x[:, ::4] += w * .7  # Overlap cameras remain distinct instances.
        states.append(x)
        pos = []
        for frame in ids[lo:hi]:
            for offset in range(4):
                kind = "camera" if offset == 0 else "register" if offset == 1 else "patch"
                patch = offset-2 if kind == "patch" else None
                tokens.append((w, frame, kind, patch))
                pos.append((0,0) if patch is None else (1,patch+1))
        positions.append(torch.tensor(pos, dtype=torch.long).unsqueeze(0))
    block = Block(16, 2, qk_norm=True, init_values=.1,
                  rope=RotaryPositionEmbedding2D()).to(dtype).eval()
    return ids, states, positions, tokens, block


def independent_dense(block, states, positions, tokens, mode):
    """Token metadata and full tiny boolean matrix; no production indices."""
    attn = block.attn
    projections = []
    for x, pos in zip(states, positions):
        raw = F.linear(block.norm1(x), attn.qkv.weight, attn.qkv.bias)
        q, k, v = raw.reshape(1, x.shape[1], 3, attn.num_heads,
                              attn.head_dim).permute(2,0,3,1,4).unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)
        q, k = attn.rope(q, pos), attn.rope(k, pos)
        projections.append((q,k,v))
    q,k,v = [torch.cat([p[i] for p in projections], dim=2) for i in range(3)]
    camera = mode in ("camera_only", "camera_global_overlap")
    overlap = mode in ("overlap_correspondence", "camera_global_overlap")
    visibility = torch.tensor([[a[0] == b[0] or
        (camera and a[2] == b[2] == "camera" and a[0] != b[0]) or
        (overlap and a[2] == b[2] == "patch" and a[1] == b[1] and
         a[3] == b[3] and abs(a[0]-b[0]) == 1)
        for b in tokens] for a in tokens], dtype=torch.bool, device=q.device)
    scores = (q @ k.transpose(-2,-1)) * (attn.head_dim ** -.5)
    pre = scores.masked_fill(~visibility[None,None], -torch.inf).softmax(-1) @ v
    pre = pre.transpose(1,2).reshape(1,len(tokens),-1)
    post = F.linear(pre, attn.proj.weight, attn.proj.bias)
    lengths = [x.shape[1] for x in states]
    before, after = pre.split(lengths,1), post.split(lengths,1)
    outputs = []
    for x, y in zip(states, after):
        intermediate = x + block.ls1(y)
        outputs.append(intermediate + block.ls2(block.mlp(block.norm2(intermediate))))
    return before, after, outputs, visibility


class V10AttentionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    def tearDown(self):
        torch.set_grad_enabled(self.grad)

    def run_case(self, windows, mode, dtype, reverse=False):
        ids, states, positions, tokens, block = fixture(windows,dtype)
        mapping = build_correspondences(ids, windows,
            [ids[lo:hi] for lo,hi in windows], [(1,2)]*len(windows),
            4, 1, 1, torch.device("cpu"), mode)
        expected_pre, expected_post, expected, mask = independent_dense(
            block, states, positions, tokens, mode)
        captured = []
        hook = block.attn.proj.register_forward_hook(
            lambda _mod,args,out: captured.append((args[0].clone(),out.clone())))
        order = list(reversed(range(len(windows)))) if reverse else None
        try:
            actual = global_step(block, states, positions, mapping, mode,
                query_chunk_size=3, order=order, attention_path="native_sdpa")
        finally:
            hook.remove()
        ordered = list(range(len(windows))) if order is None else order
        atol,rtol = TOL[dtype]
        for call,w in enumerate(ordered):
            torch.testing.assert_close(captured[call][0],expected_pre[w],atol=atol,rtol=rtol)
            torch.testing.assert_close(captured[call][1],expected_post[w],atol=atol,rtol=rtol)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
        self.assertTrue(bool(mask.diag().all()))
        for w,state in enumerate(states):
            query_groups = [mapping["local_queries"][w],mapping["camera_queries"][w]]
            query_groups.extend(pair[0] for pair in mapping["groups"][w].values())
            all_queries = [int(v) for group in query_groups for v in group]
            self.assertEqual(sorted(all_queries),list(range(state.shape[1])))
        return actual,mapping,states,positions,block

    def test_four_modes_match_independent_dense_one_two_three_unequal_windows(self):
        for windows in ([(0,2)],[(0,3),(2,5)],[(0,3),(2,5),(4,6)]):
            for dtype in (torch.float64,torch.float32):
                for mode in MODES:
                    with self.subTest(windows=windows,dtype=dtype,mode=mode):
                        self.run_case(windows,mode,dtype)

    def test_order_permutation_cannot_change_same_layer_outputs(self):
        windows=[(0,3),(2,5),(4,6)]
        for mode in ("camera_only","camera_global_overlap"):
            forward,*_ = self.run_case(windows,mode,torch.float64)
            backward,*_ = self.run_case(windows,mode,torch.float64,reverse=True)
            for a,b in zip(forward,backward):
                self.assertTrue(torch.equal(a,b))

    def test_camera_bank_keeps_overlap_duplicate_instances(self):
        _,mapping,_,_,_ = self.run_case([(0,3),(2,5),(4,6)],
                                        "camera_global_overlap",torch.float64)
        self.assertEqual(mapping["camera_bank"][0]["remote_tokens"],5)
        self.assertEqual(mapping["camera_bank"][0]["remote_duplicate_frame_instances"],1)
        self.assertEqual(mapping["camera_bank"][0]["remote_also_local_frame_instances"],1)
        self.assertEqual(mapping["duplicate_window_frame_instances"],2)

    def test_camera_and_patch_groups_have_distinct_remote_visibility(self):
        windows=[(0,3),(2,5)]
        ids,states,pos,_,block=fixture(windows,torch.float64)
        mapping=build_correspondences(ids,windows,[ids[0:3],ids[2:5]],
            [(1,2)]*2,4,1,1,torch.device("cpu"),"camera_global_overlap")
        baseline=global_step(block,states,pos,mapping,"camera_global_overlap",
                             query_chunk_size=2)
        changed=[x.clone() for x in states]
        changed[1][:,0,0] += 4.0  # remote camera K/V, not a patch token
        updated=global_step(block,changed,pos,mapping,"camera_global_overlap",
                            query_chunk_size=2)
        camera=int(mapping["camera_queries"][0][0])
        self.assertGreater(float((updated[0][:,camera]-baseline[0][:,camera]).abs().max()),1e-8)
        patch=int(mapping["groups"][0][1][0][0])
        self.assertTrue(torch.equal(updated[0][:,patch],baseline[0][:,patch]))


if __name__ == "__main__":
    unittest.main()

import unittest
import torch
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D
from vggt.v7.attention import global_step
from experiments.ours_v7.diagnostic_key_cache import cached_sdpa_keys
class Float32Norm(torch.nn.Module):
    # CUDA autocast promotes these normalization outputs to FP32; CPU does not.
    def __init__(self,norm):
        super().__init__();self.norm=norm
    def forward(self,x):
        return self.norm(x.float())
class KeyCacheTest(unittest.TestCase):
    def test_cpu_bf16_exact_and_cache_released(self):
        torch.manual_seed(7);torch.set_num_threads(1)
        block=Block(8,2,qk_norm=True,rope=RotaryPositionEmbedding2D(frequency=100)).eval().requires_grad_(False)
        states=[torch.randn(1,12,8),torch.randn(1,18,8)]
        block.attn.q_norm=Float32Norm(block.attn.q_norm)
        block.attn.k_norm=Float32Norm(block.attn.k_norm)
        positions=[torch.zeros(1,x.shape[1],2,dtype=torch.long) for x in states]
        with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
            expected=global_step(block,states,positions,6,'camera_patch_exchange',1,1,(0,3),query_chunk_size=2)
            with cached_sdpa_keys() as stats:
                actual=global_step(block,states,positions,6,'camera_patch_exchange',1,1,(0,3),query_chunk_size=2)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertEqual(stats['active_window_caches'],0)
        self.assertGreater(stats['casts'],0)
        self.assertGreater(stats['hits'],0)
    def test_disabled_autocast_preserves_float32(self):
        block=Block(8,2,qk_norm=True).eval().requires_grad_(False)
        states=[torch.randn(1,12,8),torch.randn(1,18,8)]
        with torch.inference_mode():
            expected=global_step(block,states,[None,None],6,'camera_patch_exchange',1,1,(0,3),query_chunk_size=2)
            with cached_sdpa_keys():
                actual=global_step(block,states,[None,None],6,'camera_patch_exchange',1,1,(0,3),query_chunk_size=2)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=0,rtol=0)
if __name__=='__main__':unittest.main()

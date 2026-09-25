"""v10 scheduler preserves inherited v8 modes and layer barriers."""
import unittest
from unittest.mock import patch

import torch

from vggt.models.aggregator import Aggregator
from vggt.layers.patch_embed import PatchEmbed
from vggt.v8.scheduler import aggregate_windows as v8_aggregate
from vggt.v10 import scheduler


def aggregator():
    torch.manual_seed(218)
    model=Aggregator(img_size=4,patch_size=2,embed_dim=8,depth=3,
        num_heads=2,num_register_tokens=1,patch_embed="conv",
        cached_layer_indices=(0,2))
    model.patch_embed=PatchEmbed(img_size=4,patch_size=2,embed_dim=8)
    return model.double().eval().requires_grad_(False)


class V10IntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.grad=torch.is_grad_enabled();torch.set_grad_enabled(False)

    def tearDown(self):
        torch.set_grad_enabled(self.grad)

    def test_old_modes_match_v8_at_every_cached_layer(self):
        model=aggregator()
        images=torch.rand(6,3,4,4,dtype=torch.float64)
        ids=[f"f{i}" for i in range(6)]
        windows=[(0,3),(2,5),(4,6)]
        for mode in ("independent","overlap_correspondence"):
            old=v8_aggregate(model,images,ids,windows,mode,query_chunk_size=3,
                             correspondence_attention_path="native_sdpa")
            new=scheduler.aggregate_windows(model,images,ids,windows,mode,
                query_chunk_size=3,correspondence_attention_path="native_sdpa")
            self.assertEqual(old[1],new[1])
            self.assertEqual(old[3]["pair_count"],new[3]["pair_count"])
            for old_window,new_window in zip(old[0],new[0]):
                for a,b in zip(old_window,new_window):
                    if a is None:
                        self.assertIsNone(b)
                    else:
                        self.assertTrue(torch.equal(a,b))

    def test_camera_modes_keep_layer_barrier_and_window_order(self):
        model=aggregator()
        images=torch.rand(6,3,4,4,dtype=torch.float64)
        ids=[f"f{i}" for i in range(6)]
        windows=[(0,3),(2,5),(4,6)]
        events=[]
        hooks=[block.register_forward_hook(
            lambda _m,_a,_o,l=layer: events.append(("frame",l)))
            for layer,block in enumerate(model.frame_blocks)]
        original=scheduler.global_step
        def traced(block,*args,**kwargs):
            layer=next(i for i,b in enumerate(model.global_blocks) if b is block)
            events.append(("global",layer))
            return original(block,*args,**kwargs)
        try:
            with patch.object(scheduler,"global_step",traced):
                result=scheduler.aggregate_windows(model,images,ids,windows,
                    "camera_global_overlap",query_chunk_size=3,
                    correspondence_attention_path="native_sdpa")
        finally:
            for handle in hooks:handle.remove()
        for layer in range(model.depth):
            self.assertEqual(events[4*layer:4*layer+4],
                             [("frame",layer)]*3+[("global",layer)])
        reverse=scheduler.aggregate_windows(model,images,ids,windows,
            "camera_global_overlap",query_chunk_size=3,reverse=True,
            correspondence_attention_path="native_sdpa")
        for a,b in zip(result[0],reverse[0]):
            for x,y in zip(a,b):
                if x is not None:self.assertTrue(torch.equal(x,y))
        self.assertEqual(result[3]["camera_bank"]["duplicate_window_frame_instances"],2)
        self.assertEqual(result[3]["camera_bank"]["max_remote_tokens"],6)


if __name__=="__main__":
    unittest.main()

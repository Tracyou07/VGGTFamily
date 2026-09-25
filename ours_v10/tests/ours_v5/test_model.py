import unittest
import torch
from torch import nn
from vggt.models.vggt import VGGT
from vggt.models.aggregator import Aggregator
from vggt.layers.patch_embed import PatchEmbed
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.v5.model import WindowReconstructor

class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(82);torch.set_num_threads(1)
        m=VGGT.__new__(VGGT);nn.Module.__init__(m)
        m.aggregator=Aggregator(img_size=8,patch_size=2,embed_dim=8,depth=4,num_heads=2,num_register_tokens=1,patch_embed='conv',cached_layer_indices=(0,1,2,3))
        m.aggregator.patch_embed=PatchEmbed(img_size=8,patch_size=2,embed_dim=8)
        m.camera_head=CameraHead(dim_in=16,trunk_depth=1,num_heads=2)
        # Random, untrained heads can clamp FoV to zero. Fix only fixture FoV rows.
        with torch.no_grad():
            m.camera_head.pose_branch.fc2.weight[7:].zero_();m.camera_head.pose_branch.fc2.bias[7:]=.25
        kw=dict(dim_in=16,patch_size=2,features=8,out_channels=[8]*4,intermediate_layer_idx=[0,1,2,3])
        m.depth_head=DPTHead(**kw,output_dim=2,activation='exp');m.point_head=DPTHead(**kw,output_dim=4);m.track_head=None
        self.m=m.eval().requires_grad_(False);self.images=torch.rand(5,3,8,8)
    def test_original_heads_with_required_prediction_cache(self):
        wrapper=WindowReconstructor(self.m)
        out=wrapper(self.images,list('abcde'),mode='independent',window_size=3,overlap=1,batch_size=2)
        for pred,(lo,hi) in zip(out['predictions'],out['windows']):
            with torch.inference_mode():ref=self.m(self.images[lo:hi][None])
            for key in ('depth','depth_conf','world_points','world_points_conf'):
                torch.testing.assert_close(pred[key],ref[key][0],atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(pred['pose_encoding'],ref['pose_enc'][0],atol=2e-5,rtol=2e-5)
            self.assertEqual(pred['frame_ids'],list('abcde')[lo:hi])
        self.assertEqual(sum(p.numel() for p in wrapper.parameters() if p.requires_grad),0)
    def test_camera_exchange_runs_and_input_contract(self):
        w=WindowReconstructor(self.m)
        result=w(self.images,list('abcde'),window_size=3,overlap=1)
        self.assertEqual(result['windows'],[(0,3),(2,5)])
        self.assertEqual(len(result['predictions']),2)
        for p in result['predictions']:
            self.assertTrue(torch.isfinite(p['world_points']).all())
        with self.assertRaises(ValueError):w(self.images,['a']*5)
        with self.assertRaises(ValueError):w(self.images[None],list('abcde'))
if __name__=='__main__':unittest.main()

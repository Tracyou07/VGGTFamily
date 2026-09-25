import unittest
import torch
from vggt.models.aggregator import Aggregator
from vggt.layers.patch_embed import PatchEmbed
from vggt.v5.scheduler import initialize, aggregate_windows, make_windows

torch.set_num_threads(1)
def tiny():
    torch.manual_seed(71)
    a=Aggregator(img_size=4,patch_size=2,embed_dim=8,depth=3,num_heads=2,num_register_tokens=1,patch_embed='conv',cached_layer_indices=(0,2))
    a.patch_embed=PatchEmbed(img_size=4,patch_size=2,embed_dim=8)
    return a.double().eval().requires_grad_(False)
class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.a=tiny();self.images=torch.rand(7,3,4,4,dtype=torch.float64)
        self.windows=make_windows(7,4,2)
    def test_long_window_campaign_uses_60_30_and_real_tail(self):
        import json
        from experiments.ours_v5.runtime import ROOT
        config=json.loads((ROOT/'configs/v5_validation.json').read_text())
        self.assertEqual((config['window_size'],config['overlap'],config['diagnostic_frames']),(60,30,100))
        self.assertEqual(make_windows(100,config['window_size'],config['overlap']),[(0,60),(30,90),(60,100)])
        self.assertEqual(make_windows(1000,config['window_size'],config['overlap'])[-1],(960,1000))

    def test_windows(self):
        self.assertEqual(make_windows(100,30,10),[(0,30),(20,50),(40,70),(60,90),(80,100)])
        self.assertEqual(make_windows(1000,30,10)[-1],(980,1000))
        for args in [(0,30,10),(5,0,0),(5,3,3),(5,3,-1)]:
            with self.assertRaises(ValueError):make_windows(*args)
    def test_initialization_reference_and_independent_instances(self):
        states,pos,p=initialize(self.a,self.images,self.windows,2)
        for state,(lo,hi) in zip(states,self.windows):
            ct=state.reshape(1,hi-lo,p,8)[0,:,0]
            torch.testing.assert_close(ct[0],self.a.camera_token[0,0,0]);torch.testing.assert_close(ct[1:],self.a.camera_token[0,1,0].expand_as(ct[1:]))
        before=states[1].clone();states[0][0,2*p+2]+=1
        self.assertTrue(torch.equal(before,states[1]))
    def test_independent_matches_original_all_caches(self):
        actual,p=aggregate_windows(self.a,self.images,self.windows,'independent',2)
        for values,(lo,hi) in zip(actual,self.windows):
            expected,ep=self.a(self.images[lo:hi][None]);self.assertEqual(p,ep)
            for a,b in zip(values,expected):
                if b is None:self.assertIsNone(a)
                else:torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-9)
    def test_single_window(self):
        w=[(0,7)]
        a,_=aggregate_windows(self.a,self.images,w,'camera_exchange',1)
        b,_=self.a(self.images[None])
        for x,y in zip(a[0],b):
            if y is not None:torch.testing.assert_close(x,y,atol=1e-9,rtol=1e-9)
    def test_groups_order_tail_and_frozen(self):
        snapshot={k:v.clone() for k,v in self.a.state_dict().items()}
        a,p=aggregate_windows(self.a,self.images,self.windows,'camera_exchange',1)
        b,_=aggregate_windows(self.a,self.images,self.windows,'camera_exchange',2,reverse=True)
        for w,(x,y) in enumerate(zip(a,b)):
            for xx,yy in zip(x,y):
                if xx is not None:
                    self.assertEqual(xx.shape[1],self.windows[w][1]-self.windows[w][0]);torch.testing.assert_close(xx,yy,atol=1e-9,rtol=1e-9)
        self.assertEqual(sum(p.numel() for p in self.a.parameters() if p.requires_grad),0)
        for k,v in self.a.state_dict().items():self.assertTrue(torch.equal(v,snapshot[k]))
    def test_scene_calls_do_not_share_state(self):
        a,_=aggregate_windows(self.a,self.images,self.windows,'camera_exchange',2)
        aggregate_windows(self.a,1-self.images,self.windows,'camera_exchange',2)
        b,_=aggregate_windows(self.a,self.images,self.windows,'camera_exchange',2)
        torch.testing.assert_close(a[0][-1],b[0][-1],atol=0,rtol=0)
if __name__=='__main__':unittest.main()

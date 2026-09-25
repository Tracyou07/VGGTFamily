import unittest
import torch
from experiments.ours_v4.core import pack_groups,instance_inputs,infer_batches
from vggt.models.aggregator import Aggregator,slice_expand_and_flatten
from vggt.layers.overlap_windows import make_windows


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator=Aggregator(img_size=28,patch_size=14,embed_dim=16,depth=2,num_heads=4,
            patch_embed='conv',cached_layer_indices=(0,1))
    def forward(self,images):
        features,_=self.aggregator(images)
        return dict(layer0=features[0],head_input=features[-1])


class PackingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model=Tiny().eval().requires_grad_(False)
        self.windows=make_windows(7,4,2)
        self.instances=instance_inputs(torch.rand(7,3,28,28),self.windows)

    def run_instances(self,instances,batch):
        with torch.inference_mode(): return infer_batches(self.model,instances,self.windows,batch)

    def test_schedule_and_tail(self):
        self.assertEqual(make_windows(100),((0,30),(20,50),(40,70),(60,90),(80,100)))
        self.assertEqual(pack_groups(self.windows,2),[[0,1],[2],[3]])
        for args in ((100,30,30),(100,30,-1)):
            with self.assertRaises(ValueError): make_windows(*args)

    def test_batched_equals_separate_head_caches(self):
        a=self.run_instances(self.instances,1); b=self.run_instances(self.instances,2)
        for first,second in zip(a,b):
            for key in first: torch.testing.assert_close(first[key],second[key],atol=1e-6,rtol=1e-5)

    def test_reference_reset_per_window(self):
        tokens=torch.tensor([[[[1.]],[[2.]]]])
        actual=slice_expand_and_flatten(tokens,2,4).reshape(2,4)
        self.assertEqual(actual.tolist(),[[1.,2.,2.,2.],[1.,2.,2.,2.]])

    def test_overlap_instance_mutation_does_not_leak(self):
        before=self.run_instances(self.instances,2)
        # frame2@window0 only, not frame2@window1.
        snapshot=self.instances[1].clone()
        self.instances[0][2].add_(.5)
        self.assertTrue(torch.equal(snapshot,self.instances[1]))
        after=self.run_instances(self.instances,2)
        for key in before[1]: torch.testing.assert_close(before[1][key],after[1][key],atol=0,rtol=0)
        self.assertFalse(torch.equal(before[0]['head_input'],after[0]['head_input']))

    def test_batch_order(self):
        with torch.inference_mode():
            a=self.model(torch.stack(self.instances[:2]))
            b=self.model(torch.stack(self.instances[:2][::-1]))
        for key in a: torch.testing.assert_close(a[key],b[key].flip(0),atol=1e-6,rtol=1e-5)

if __name__=='__main__': unittest.main()

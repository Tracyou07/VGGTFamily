import tempfile,unittest
from pathlib import Path
import torch
from experiments.ours_v4.validate import compare
import test_packing

class GateTests(unittest.TestCase):
    def test_capture_compare_detects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); a=root/'a'; b=root/'b'; a.mkdir(); b.mkdir()
            value=dict(c2w=torch.eye(4)[None],depth=torch.ones(1,2,2,1),pose_encoding=torch.zeros(1,9))
            torch.save(value,a/'window_0000.pt'); torch.save(value,b/'window_0000.pt')
            tolerance=dict(atol=1e-5,rtol=1e-5,center_m=1e-4,rotation_deg=.02)
            self.assertTrue(compare(a,b,tolerance)['passed'])
            value['depth'][0,0,0,0]=float('nan'); torch.save(value,b/'window_0000.pt')
            self.assertFalse(compare(a,b,tolerance)['passed'])

    def test_frozen_model_and_exclusive_frame(self):
        fixture=test_packing.PackingTests(); fixture.setUp()
        before={k:v.clone() for k,v in fixture.model.state_dict().items()}
        first=fixture.run_instances(fixture.instances,2)
        fixture.instances[0][0].mul_(0)
        second=fixture.run_instances(fixture.instances,2)
        for k in first[1]: torch.testing.assert_close(first[1][k],second[1][k],atol=0,rtol=0)
        self.assertEqual(sum(p.numel() for p in fixture.model.parameters() if p.requires_grad),0)
        for k,v in fixture.model.state_dict().items(): self.assertTrue(torch.equal(v,before[k]))

if __name__=='__main__': unittest.main()

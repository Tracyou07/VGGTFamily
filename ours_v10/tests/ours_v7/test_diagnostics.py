"""CPU contracts for read-only diagnostics."""
import unittest
import numpy as np
import torch
from experiments.ours_v7 import diagnostics as d
from test_model import ModelTest

class DiagnosticTest(unittest.TestCase):
    def test_statistics_finite_and_fixed_tolerance(self):
        a=np.array([1.,2.,3.]); b=a+np.array([0.,0.,1.])
        x=d.difference(a,b)
        self.assertEqual(x['max_abs'],1.)
        self.assertAlmostEqual(x['mean_abs'],1/3)
        self.assertFalse(x['within_tolerance'])
        self.assertFalse(d.difference(a,np.array([1.,2.,np.nan]))['finite'])
    def test_storage_views_count_once(self):
        x=torch.zeros(10); y=x[2:5]; z=x.clone()
        self.assertEqual(d.storage_bytes([x,y]),40)
        self.assertEqual(d.storage_bytes([x,y,z]),80)
    def test_input_slicing(self):
        x=torch.arange(20).reshape(5,4)
        for lo,hi in [(0,3),(2,5)]:
            self.assertEqual(d.tensor_hash(x[lo:hi]),d.tensor_hash(x.clone()[lo:hi]))
        self.assertNotEqual(d.tensor_hash(x[:3]),d.tensor_hash(x[2:]))
    def test_ownership_and_independent_assembly(self):
        from experiments.ours_v3.geometry import Sim3
        p=[{'frame_ids':['a','b'],'c2w':np.tile(np.eye(4),(2,1,1))},
           {'frame_ids':['b','c'],'c2w':np.tile(np.eye(4),(2,1,1))}]
        t=[Sim3(1.,np.eye(3),np.zeros(3)),Sim3(2.,np.eye(3),np.array([1.,2.,3.]))]
        for owner,expected in [('first',[0,0,1]),('last',[0,1,1])]:
            a,ids=d.assemble(p,t,owner,'ours')
            b,_=d.assemble(p,t,owner,'long')
            np.testing.assert_array_equal(ids,expected)
            np.testing.assert_array_equal(a,b)
            from experiments.ours_v7.diagnostic_alignment import native_assembly
            actual=native_assembly(p,t,owner,[(0,2),(1,3)])
            np.testing.assert_allclose(a,actual,atol=1e-12,rtol=0)
    def test_hooks_do_not_modify_calculation(self):
        fixture=ModelTest();fixture.setUp(); m=fixture.m
        with torch.inference_mode():
            a=m(fixture.images[None])
            with d.Observer(m,enabled=True,cuda=False) as obs:
                b=m(fixture.images[None])
        self.assertTrue(obs.rows)
        for k in ['pose_enc','depth','world_points']:
            torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)
if __name__=='__main__':unittest.main()

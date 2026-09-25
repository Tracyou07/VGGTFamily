import unittest
import numpy as np
import torch
from vggt.layers.overlap_windows import make_windows, visibility_groups, slice_features
from experiments.ours_v3.geometry import Sim3, fit_sim3, robust_sim3, AlignmentConfig, transform_predictions, append_unique


class WindowTests(unittest.TestCase):
    def test_windows_and_tail(self):
        self.assertEqual(make_windows(55), ((0,30),(20,50),(40,55)))
        self.assertEqual(make_windows(1000)[-1], (980,1000))
        self.assertEqual(make_windows(25), ((0,25),(20,25)))
        for args in ((0,30,10),(20,30,30),(20,30,-1)):
            with self.assertRaises(ValueError): make_windows(*args)

    def test_visibility_unique_queries_and_keys(self):
        groups = visibility_groups(55, make_windows(55))
        lookup = {q:k for qs,k in groups for q in qs}
        self.assertEqual(lookup[25], tuple(range(50)))
        self.assertEqual(lookup[45], tuple(range(20,55)))
        self.assertEqual(sorted(q for qs,_ in groups for q in qs), list(range(55)))
        for _, keys in groups: self.assertEqual(len(keys),len(set(keys)))

    def test_slice_caches_by_ids(self):
        x=torch.arange(20).reshape(1,5,2,2)
        result=slice_features([x,None,x+1],['a','b','c','d','e'],['d','b'])
        self.assertTrue(torch.equal(result[0],x[:,[3,1]]))
        self.assertIsNone(result[1])
        with self.assertRaises(ValueError): slice_features([x],['a']*5,['a'])


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(3).normal(size=(200,3))
        self.s=Sim3(2.,np.array([[0.,-1,0],[1,0,0],[0,0,1]]),np.array([1.,2.,3.]))

    def test_fit_and_composition_direction(self):
        recovered=fit_sim3(self.x,self.s.apply(self.x))
        np.testing.assert_allclose(recovered.apply(self.x),self.s.apply(self.x),atol=1e-10)
        b=Sim3(.7,np.eye(3),np.array([3.,0.,0.]))
        np.testing.assert_allclose(self.s.compose(b).apply(self.x),self.s.apply(b.apply(self.x)))

    def test_reflection_never_returns_improper_rotation(self):
        target=self.x.copy(); target[:,0]*=-1
        fitted=fit_sim3(self.x,target)
        self.assertAlmostEqual(np.linalg.det(fitted.rotation),1.)
        self.assertGreater(np.linalg.norm(fitted.apply(self.x)-target),1.)

    def test_outliers_and_failures(self):
        target=self.s.apply(self.x); target[:60]=np.random.default_rng(9).normal(size=(60,3))*10
        fitted,stats=robust_sim3(self.x,target,AlignmentConfig(min_inliers=50,threshold=.01))
        np.testing.assert_allclose(fitted.apply(self.x[60:]),target[60:],atol=1e-9)
        self.assertEqual(stats['inlier_count'],140)
        for source in (np.empty((0,3)),np.zeros((100,3)),np.column_stack([np.arange(100),np.zeros((100,2))])):
            with self.assertRaises(ValueError): robust_sim3(source,source,AlignmentConfig())

    def test_camera_depth_and_dedup(self):
        cameras=np.repeat(np.eye(4)[None],3,axis=0); cameras[:,:3,3]=self.x[:3]
        depth=np.ones((3,2,2,1))
        pose,scaled=transform_predictions(cameras,depth,self.s)
        np.testing.assert_allclose(pose[:,:3,3],self.s.apply(self.x[:3]))
        np.testing.assert_allclose(pose[0,:3,:3].T@pose[0,:3,:3],np.eye(3))
        np.testing.assert_allclose(scaled,2)
        seen=set(); self.assertEqual(append_unique(seen,['a','b']),[0,1])
        self.assertEqual(append_unique(seen,['b','c']),[1])

if __name__=='__main__': unittest.main()

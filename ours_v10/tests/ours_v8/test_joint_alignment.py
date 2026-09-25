import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from experiments.ours_v3.geometry import Sim3
from vggt.v8.joint_alignment import (AlignmentStitcher, JointAlignmentConfig,
                                     align_overlap_joint, loss_components)
from vggt.v5.alignment import LongAlignmentConfig, align_overlap


def rz(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def prediction(points, centers, rotations, ids):
    frames,height,width,_=points.shape
    poses=np.repeat(np.eye(4)[None],frames,axis=0)
    poses[:,:3,:3]=rotations;poses[:,:3,3]=centers
    intr=np.repeat(np.eye(3)[None],frames,axis=0)
    return dict(frame_ids=list(ids),world_points=points.astype(np.float32),
                world_points_conf=np.ones((frames,height,width),np.float32),
                c2w=poses.astype(np.float32),intrinsics=intr.astype(np.float32),
                depth=np.ones((frames,height,width,1),np.float32),
                confidence=np.ones((frames,height,width),np.float32))


def pair(wrong_camera=False,repeated_centers=False,outlier=False):
    rng=np.random.default_rng(17)
    bpts=rng.normal(size=(3,4,5,3))
    true=Sim3(1.25,rz(.23),np.array([.4,-.3,.2]))
    apts=true.apply(bpts)
    if outlier:apts[0,0,0]+=20
    centers=rng.normal(size=(3,3))
    if repeated_centers:centers[:]=centers[0]
    brot=np.stack([rz(x) for x in (.1,-.2,.3)])
    ac=true.apply(centers);ar=np.einsum('ij,fjk->fik',true.rotation,brot)
    if wrong_camera:
        ac=ac+np.array([.5,-.2,.1]);ar=np.einsum('ij,fjk->fik',rz(.3),ar)
    ids=['a','b','c']
    return prediction(apts,ac,ar,ids),prediction(bpts,centers,brot,ids),true


class JointAlignmentTest(unittest.TestCase):
    def test_known_sim3_and_camera_rotation_has_no_scale(self):
        a,b,true=pair()
        for mode in ('point_normalized_control','point_camera_joint'):
            transform,stats=align_overlap_joint(a,b,JointAlignmentConfig(mode=mode,max_iterations=20,chunk_size=11))
            self.assertAlmostEqual(transform.scale,true.scale,places=4)
            np.testing.assert_allclose(transform.rotation,true.rotation,atol=2e-4)
            np.testing.assert_allclose(transform.translation,true.translation,atol=2e-4)
            self.assertTrue(stats['optimization']['success'])
            self.assertGreater(stats['geometry_scale'],0)
            mapped=np.einsum('ij,fjk->fik',transform.rotation,b['c2w'][:,:3,:3])
            np.testing.assert_allclose(mapped,a['c2w'][:,:3,:3],atol=2e-4)
        components=loss_components(true,a,b,JointAlignmentConfig(mode='point_camera_joint'))
        self.assertLess(components['point'],1e-10)
        self.assertLess(components['center'],1e-10)
        self.assertLess(components['rotation'],1e-10)

    def test_camera_toggle_isolates_wrong_camera_constraint(self):
        a,b,true=pair(wrong_camera=True)
        control,_=align_overlap_joint(a,b,JointAlignmentConfig(mode='point_normalized_control',max_iterations=20,chunk_size=13))
        joint,stats=align_overlap_joint(a,b,JointAlignmentConfig(mode='point_camera_joint',max_iterations=20,chunk_size=13))
        self.assertLess(abs(control.scale-true.scale),1e-3)
        self.assertGreater(np.linalg.norm(joint.translation-control.translation),1e-4)
        self.assertEqual(stats['weights'],{'point':1.0,'center':1.0,'rotation':1.0})

    def test_chunked_losses_equal_one_shot_and_outlier_is_finite(self):
        a,b,true=pair(outlier=True)
        cfg=JointAlignmentConfig(mode='point_camera_joint',chunk_size=7)
        small=loss_components(true,a,b,cfg,chunk_size=7)
        full=loss_components(true,a,b,cfg,chunk_size=10_000)
        for key in ('point','center','rotation','total'):
            self.assertAlmostEqual(small[key],full[key],places=11)
            self.assertTrue(np.isfinite(small[key]))

    def test_repeated_centers_allowed_but_collinear_and_nonfinite_fail(self):
        a,b,_=pair(repeated_centers=True)
        align_overlap_joint(a,b,JointAlignmentConfig(mode='point_camera_joint',max_iterations=20,chunk_size=17))
        line=np.zeros((2,2,4,3),np.float32);line[...,0]=np.arange(4)
        bad=prediction(line,np.zeros((2,3)),np.repeat(np.eye(3)[None],2,0),['a','b'])
        with self.assertRaisesRegex(ValueError,'degenerate'):
            align_overlap_joint(bad,bad,JointAlignmentConfig(mode='point_normalized_control'))
        bad2={k:(v.copy() if isinstance(v,np.ndarray) else list(v)) for k,v in a.items()}
        bad2['c2w'][0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            align_overlap_joint(bad2,b,JointAlignmentConfig(mode='point_camera_joint'))

    def test_legacy_mode_and_front_window_ownership(self):
        a,b,_=pair()
        expected,_=align_overlap(a,b,LongAlignmentConfig())
        actual,_=align_overlap_joint(a,b,JointAlignmentConfig(mode='point_legacy'))
        self.assertEqual(actual.scale,expected.scale)
        np.testing.assert_array_equal(actual.rotation,expected.rotation)
        np.testing.assert_array_equal(actual.translation,expected.translation)
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            stitch=AlignmentStitcher(f'{folder}/alignment',JointAlignmentConfig(mode='point_legacy'))
            first,_=stitch.add(a,0);second,_=stitch.add(b,1)
            result=stitch.finish(['a','b','c'])
            self.assertEqual(first,[0,1,2]);self.assertEqual(second,[])
            self.assertEqual(result['source_window'].tolist(),[0,0,0])

    def test_optimizer_failure_is_not_silently_accepted(self):
        a,b,_=pair()
        failed=SimpleNamespace(success=False,status=1,message='iteration limit',nit=1,
                               nfev=1,x=np.zeros(7),jac=np.zeros(7))
        with patch('vggt.v8.joint_alignment.minimize',return_value=failed):
            with self.assertRaisesRegex(RuntimeError,'iteration limit'):
                align_overlap_joint(a,b,JointAlignmentConfig(mode='point_camera_joint'))

    def test_new_mode_stitcher_writes_edge_diagnostics(self):
        a,b,_=pair()
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'alignment'
            stitch=AlignmentStitcher(output,JointAlignmentConfig(
                mode='point_normalized_control',max_iterations=20,chunk_size=11))
            stitch.add(a,0);stitch.add(b,1)
            edge=json.loads((output/'edge_0000_0001.json').read_text())
            self.assertEqual(edge['status'],'success')
            self.assertEqual(edge['direction'],'B_local -> A_local')
            self.assertIn('initial_loss',edge);self.assertIn('final_loss',edge)

if __name__=='__main__':unittest.main()

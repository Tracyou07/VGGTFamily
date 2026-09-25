import tempfile,unittest
from pathlib import Path
import numpy as np
from vggt.v5.alignment import LongAlignmentConfig,align_overlap,load_long,Stitcher
from experiments.ours_v3.geometry import Sim3

class AlignmentTest(unittest.TestCase):
    def pair(self):
        rng=np.random.default_rng(9);a=rng.normal(size=(3,8,8,3)).astype('float32')
        ang=.3;R=np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]])
        S=Sim3(1.7,R,np.array([.2,-.3,.5])); b=((a-S.translation)@S.rotation/S.scale).astype('float32')
        conf=rng.uniform(.5,2,size=(3,8,8)).astype('float32')
        def pred(points,ids):
            return dict(frame_ids=ids,world_points=points,world_points_conf=conf.copy(),c2w=np.tile(np.eye(4),(3,1,1)),intrinsics=np.tile(np.eye(3),(3,1,1)),depth=np.ones((3,8,8,1)),confidence=conf.copy())
        return pred(a,['a','b','c']),pred(b,['a','b','c']),S
    def test_exact_original_oracle_and_known_transform(self):
        a,b,S=self.pair();cfg=LongAlignmentConfig()
        got,stats=align_overlap(a,b,cfg)
        conf=a['world_points_conf'];cfb=b['world_points_conf'];threshold=.1*min(np.median(conf),np.median(cfb))
        raw=load_long().weighted_align_point_maps(a['world_points'],conf,b['world_points'],cfb,None,threshold,cfg.vendor_config())
        np.testing.assert_array_equal(got.rotation,raw[1]);np.testing.assert_array_equal(got.translation,raw[2]);self.assertEqual(got.scale,raw[0])
        np.testing.assert_allclose(got.apply(b['world_points']),a['world_points'],atol=1e-5)
        self.assertEqual(stats['pairs'],192);self.assertEqual(stats['common_frame_ids'],['a','b','c'])
    def test_outliers_and_frame_id_order(self):
        a,b,S=self.pair();b['world_points'][0,0,0]+=2
        b={k:(v[::-1].copy() if isinstance(v,np.ndarray) else v[::-1]) for k,v in b.items()}
        got,stats=align_overlap(a,b,LongAlignmentConfig())
        self.assertLess(abs(got.scale-S.scale),.03);self.assertEqual(stats['common_frame_ids'],['a','b','c'])
    def test_invalid_inputs_stop(self):
        for kind in ['empty','nan','line','zero','duplicate']:
            a,b,S=self.pair()
            if kind=='empty':b['frame_ids']=['d','e','f']
            if kind=='nan':b['world_points'][0,0,0,0]=np.nan
            if kind=='line':a['world_points'][...,1:]=0;b['world_points'][...,1:]=0
            if kind=='zero':a['world_points_conf'][:]=0
            if kind=='duplicate':b['frame_ids']=['a','a','c']
            with self.assertRaises(ValueError,msg=kind):align_overlap(a,b,LongAlignmentConfig())
    def test_stitch_ownership_scale_rotation_composition(self):
        a,b,S=self.pair()
        a['frame_ids']=['a','b','c']; b['frame_ids']=['b','c','d'];b['pose_encoding']=np.zeros((3,9))
        # Shared physical frames b/c correspond to a indices 1/2 and b indices 0/1.
        b['world_points'][:2]=((a['world_points'][1:]-S.translation)@S.rotation/S.scale)
        c={k:(v.copy() if isinstance(v,np.ndarray) else list(v)) for k,v in b.items()}
        T=Sim3(.8,np.eye(3),np.array([1.,0.,0.]));c['frame_ids']=['c','d','e']
        c['world_points'][:2]=(b['world_points'][1:]-T.translation)/T.scale
        with tempfile.TemporaryDirectory() as d:
            st=Stitcher(Path(d)/"alignment",LongAlignmentConfig())
            fresh,pa=st.add(a,0);self.assertEqual(fresh,[0,1,2])
            original_front=pa['world_points'].copy();original_pose=pa['c2w'].copy()
            fresh,pb=st.add(b,1);self.assertNotIn('pose_encoding',pb);self.assertEqual(fresh,[2]);np.testing.assert_allclose(pb['depth'],S.scale,atol=1e-5)
            np.testing.assert_allclose(pb['c2w'][0,:3,:3],S.rotation,atol=1e-5)
            np.testing.assert_array_equal(pa['world_points'],original_front)
            np.testing.assert_array_equal(pa['c2w'],original_pose)
            import json
            edge=json.loads((Path(d)/'alignment/edge_0000_0001.json').read_text())
            self.assertEqual(edge['direction'],'B_local -> A_local')
            fresh,pc=st.add(c,2);self.assertEqual(fresh,[2]);np.testing.assert_allclose(pc['depth'],S.scale*T.scale,atol=1e-5)
            np.testing.assert_allclose(pc['c2w'][0,:3,3],S.compose(T).translation,atol=1e-5)
            self.assertEqual(st.finish(['a','b','c','d','e'])['source_window'].tolist(),[0,0,0,1,2])
    def test_tail_window_composition_and_front_ownership(self):
        from experiments.ours_v6.windows import make_windows
        windows=make_windows(6,3,1)
        self.assertEqual(windows,[(0,3),(2,5),(4,6)])
        rng=np.random.default_rng(19)
        global_points=rng.normal(size=(6,8,8,3)).astype("float32")
        theta=.2
        rotation=np.array([[np.cos(theta),-np.sin(theta),0],
                           [np.sin(theta),np.cos(theta),0],[0,0,1.]])
        first=Sim3(1.2,rotation,np.array([.2,-.1,.3]))
        second=Sim3(.9,np.eye(3),np.array([-.2,.4,.1]))
        w0=global_points[:3]
        w1=((global_points[2:5]-first.translation)@first.rotation/first.scale).astype("float32")
        w2_in_first=((global_points[4:6]-first.translation)@first.rotation/first.scale)
        w2=((w2_in_first-second.translation)@second.rotation/second.scale).astype("float32")
        def prediction(points,ids):
            n=len(ids)
            return dict(frame_ids=ids,world_points=points,
                world_points_conf=np.ones((n,8,8),dtype="float32"),
                c2w=np.tile(np.eye(4),(n,1,1)),
                intrinsics=np.tile(np.eye(3),(n,1,1)),
                depth=np.ones((n,8,8,1)),confidence=np.ones((n,8,8)))
        with tempfile.TemporaryDirectory() as d:
            stitch=Stitcher(Path(d)/"alignment",LongAlignmentConfig())
            inputs=[prediction(w0,['0','1','2']),prediction(w1,['2','3','4']),
                    prediction(w2,['4','5'])]
            sources=[]
            for i,item in enumerate(inputs):
                fresh,_=stitch.add(item,i)
                sources.extend([i]*len(fresh))
            result=stitch.finish([str(i) for i in range(6)])
            self.assertEqual(sources,[0,0,0,1,1,2])
            self.assertEqual(result["source_window"].tolist(),sources)
            np.testing.assert_allclose(result["c2w"][5,:3,3],
                first.compose(second).translation,atol=2e-4)

    def test_failure_artifact_and_depth_point_consistency(self):
        from experiments.ours_v3.geometry import unproject_pixels,transform_predictions
        a,b,S=self.pair()
        with tempfile.TemporaryDirectory() as d:
            st=Stitcher(Path(d)/'alignment');st.add(a,0);b['world_points_conf'][:]=0
            with self.assertRaises(ValueError):st.add(b,1)
            self.assertTrue((Path(d)/'alignment/edge_0000_0001.json').is_file())
        pose,depth=transform_predictions(a['c2w'],a['depth'],S)
        rows=np.array([1,2]);cols=np.array([2,3])
        before=unproject_pixels(a['depth'][0,...,0],a['intrinsics'][0],a['c2w'][0],rows,cols)
        after=unproject_pixels(depth[0,...,0],a['intrinsics'][0],pose[0],rows,cols)
        np.testing.assert_allclose(after,S.apply(before),atol=1e-10)

    def test_no_gt_interface_and_vendor_hash(self):
        import inspect
        self.assertNotIn('gt',inspect.signature(align_overlap).parameters)
        self.assertTrue(load_long().__file__.endswith('vendor/vggtlong/loop_utils/sim3utils.py'))
if __name__=='__main__':unittest.main()

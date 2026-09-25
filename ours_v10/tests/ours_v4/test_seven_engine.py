import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.ours_v3.geometry import AlignmentConfig
from experiments.ours_v3.stitch import Stitcher

class EngineTests(unittest.TestCase):
    def test_deferred_stitch_matches_original_and_writes_edges(self):
        from experiments.sevenscenes.engine import stitch_predictions
        cfg=AlignmentConfig(pixel_stride=1,min_inliers=3,max_points=100,confidence_quantile=0)
        depth=np.ones((3,8,8,1),np.float32)*2
        pose=np.repeat(np.eye(4)[None],3,axis=0); intr=np.repeat(np.eye(3)[None],3,axis=0)
        a=dict(frame_ids=['0','1','2'],depth=depth,confidence=np.ones((3,8,8)),c2w=pose,intrinsics=intr)
        b=dict(frame_ids=['1','2','3'],depth=depth/2,confidence=np.ones((3,8,8)),c2w=pose.copy(),intrinsics=intr)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); original=Stitcher(root/'original',cfg)
            original.add(a,0); original.add(b,1); expected=original.finish(['0','1','2','3'])
            actual,seconds,exports=stitch_predictions([a,b],['0','1','2','3'],root/'new',cfg)
            for key in ('c2w','intrinsics','source_window'): np.testing.assert_array_equal(actual[key],expected[key])
            np.testing.assert_allclose(actual['depth'],2)
            self.assertTrue((root/'new/alignment/edge_0000_0001.json').exists())
            self.assertGreater(seconds,0); self.assertGreater(exports,0)

    def test_gate_rejects_window_and_global_mismatch_without_alignment(self):
        from experiments.sevenscenes.engine import compare_predictions
        pred=dict(frame_ids=['0'],c2w=np.eye(4)[None],intrinsics=np.eye(3)[None],depth=np.ones((1,2,2,1)),confidence=np.ones((1,2,2)))
        other={k:v.copy() if isinstance(v,np.ndarray) else list(v) for k,v in pred.items()}
        tol=dict(atol=.02,rtol=.02,center_m=.01,rotation_deg=.5)
        self.assertTrue(compare_predictions([pred],[other],tol)['passed'])
        other['c2w'][0,0,3]=.1
        self.assertFalse(compare_predictions([pred],[other],tol)['passed'])
        other['frame_ids']=['wrong']
        with self.assertRaises(ValueError): compare_predictions([pred],[other],tol)

if __name__=='__main__': unittest.main()

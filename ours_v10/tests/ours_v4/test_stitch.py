import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.ours_v3.geometry import AlignmentConfig, overlap_correspondences
from experiments.ours_v3.stitch import Stitcher


def prediction(ids, scale=1):
    n=len(ids); depth=np.ones((n,8,8,1))*scale
    depth[:,:,4:]*=1.2
    c2w=np.repeat(np.eye(4)[None],n,axis=0)
    for i,frame in enumerate(ids): c2w[i,0,3]=float(frame)*.1*scale
    return dict(frame_ids=ids,depth=depth,confidence=np.ones((n,8,8)),
                c2w=c2w,intrinsics=np.repeat(np.eye(3)[None],n,axis=0))


class StitchTests(unittest.TestCase):
    def test_correspondence_uses_ids_and_same_pixel(self):
        a=prediction(['0','1','2']); b=prediction(['2','1','3'],2)
        cfg=AlignmentConfig(pixel_stride=1,min_inliers=20)
        source,target,labels=overlap_correspondences(a,b,cfg)
        np.testing.assert_allclose(source/2,target)
        self.assertEqual(set(row[0] for row in labels),{'1','2'})

    def test_chain_dedup_scale_and_failure(self):
        cfg=AlignmentConfig(pixel_stride=1,min_inliers=20,threshold=.001,max_rmse=.001)
        with tempfile.TemporaryDirectory() as folder:
            stitch=Stitcher(folder,cfg)
            stitch.add(prediction(['0','1','2']),0)
            fresh,pose,depth=stitch.add(prediction(['2','3','4'],2),1)
            self.assertEqual(fresh,[1,2])
            np.testing.assert_allclose(depth,prediction(['2','3','4'])['depth'],atol=1e-10)
            stitch.add(prediction(['4','5'],3),2)
            result=stitch.finish(['0','1','2','3','4','5'])
            np.testing.assert_allclose(result['c2w'][:,0,3],np.arange(6)*.1,atol=1e-9)
            bad=prediction(['5','6']); bad['confidence'][:]=np.nan
            with self.assertRaises(ValueError): stitch.add(bad,3)
            self.assertTrue((Path(folder)/'edge_0002_0003.json').exists())
            self.assertFalse((Path(folder)/'COMPLETE.json').exists())

if __name__=='__main__': unittest.main()

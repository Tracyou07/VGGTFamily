import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np


class SevenScenesTests(unittest.TestCase):
    def core(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.sevenscenes'), '7Scenes integration is missing')
        from experiments.sevenscenes import common
        return common

    def test_sampling_and_frame_paths(self):
        c=self.core()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); folder=root/'chess/seq-03'; folder.mkdir(parents=True)
            for i in range(12): (folder/f'frame-{i:06}.color.png').touch()
            ids,paths=c.frames_for_sequence(root,'chess/seq-03',3)
            self.assertEqual(ids,['000000','000003','000006','000009'])
            self.assertEqual(paths[-1],folder/'frame-000009.color.png')
            self.assertEqual(c.frames_for_sequence(root,'chess/seq-03',10)[0],['000000','000010'])
            with self.assertRaises(ValueError): c.frames_for_sequence(root,'../escape',3)
            (folder/'frame-000002.color.png').unlink()
            with self.assertRaises(ValueError): c.frames_for_sequence(root,'chess/seq-03',3)

    def test_sim3_depth_pose_point_consistency(self):
        c=self.core()
        from experiments.ours_v3.geometry import Sim3,transform_predictions
        depth=np.ones((2,3,4,1),np.float32)*2
        intr=np.repeat(np.eye(3)[None],2,axis=0)
        pose=np.repeat(np.eye(4)[None],2,axis=0); pose[1,0,3]=.3
        rot=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        sim=Sim3(2.,rot,np.array([.5,1.,3.]))
        gp,gd=transform_predictions(pose,depth,sim)
        np.testing.assert_allclose(c.point_maps(gd,intr,gp),sim.apply(c.point_maps(depth,intr,pose)),atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(gp[:,:3,:3]),1.)

    def test_summary_requires_all_18_unique_valid_sequences(self):
        c=self.core(); expected=[f'seq{i}' for i in range(18)]
        rows=[dict(scene_id=s,acc=.1,comp=.2,nc1=.8,nc2=.6,mean_nc=.7) for s in expected]
        self.assertFalse(c.summarize(rows[:1],expected)['complete'])
        d=c.summarize(rows,expected)
        self.assertEqual(d['valid_sequences'],18); self.assertTrue(d['complete']); self.assertAlmostEqual(d['mean_nc'],.7)
        with self.assertRaises(ValueError): c.summarize(rows+[rows[0]],expected)
        rows[0]['acc']=float('nan')
        with self.assertRaises(ValueError): c.summarize(rows,expected)

    def test_resume_rejects_missing_corrupt_and_failed_artifacts(self):
        c=self.core()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); row=dict(scene_id='a',acc=.1,comp=.2,nc1=.8,nc2=.6,mean_nc=.7,frame_ids=['0'])
            c.write_json(p/'metrics.json',row); (p/'trajectory.npz').write_bytes(b'trajectory'); c.write_json(p/'diagnostics.json',{})
            self.assertIsNone(c.valid_attempt(p,'fingerprint','a',['0']))
            c.seal_attempt(p,'fingerprint','a',['0'])
            self.assertEqual(c.valid_attempt(p,'fingerprint','a',['0'])['scene_id'],'a')
            self.assertIsNone(c.valid_attempt(p,'wrong','a',['0']))
            self.assertIsNone(c.valid_attempt(p,'fingerprint','a',['1']))
            (p/'trajectory.npz').write_bytes(b'corrupt')
            self.assertIsNone(c.valid_attempt(p,'fingerprint','a',['0']))
            c.seal_attempt(p,'fingerprint','a',['0']); c.write_json(p/'FAILED.json',{'reason':'failed'})
            self.assertIsNone(c.valid_attempt(p,'fingerprint','a',['0']))

    def test_windows_short_tail_and_ownership(self):
        self.core()
        from vggt.layers.overlap_windows import make_windows
        from experiments.ours_v4.core import pack_groups
        from experiments.ours_v3.geometry import append_unique
        windows=make_windows(55,30,10)
        self.assertEqual(windows,((0,30),(20,50),(40,55)))
        self.assertEqual(pack_groups(windows,2),[[0,1],[2]])
        seen=set(); owned=[]
        for lo,hi in windows:
            ids=list(range(lo,hi)); owned.append([ids[i] for i in append_unique(seen,ids)])
        self.assertEqual(owned,[list(range(30)),list(range(30,50)),list(range(50,55))])

if __name__=='__main__': unittest.main()

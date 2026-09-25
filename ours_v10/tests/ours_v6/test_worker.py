"""CPU execution of the real worker/artifact path; CUDA APIs are mocked, not tested."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import test_model as fixtures
from experiments.ours_v6 import worker
from experiments.ours_v6.runtime import ROOT,sha256
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

class WorkerTest(unittest.TestCase):
    def test_cpu_artifact_smoke(self):
        fixture=fixtures.ModelTest();fixture.setUp();model=fixture.m;images=fixture.images
        config=json.loads((ROOT/'configs/v6_validation.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);scene=root/'scene';(scene/'pose').mkdir(parents=True)
            ids=list('abcde')
            with torch.inference_mode():
                raw=model(images[None]);ext,_=pose_encoding_to_extri_intri(raw['pose_enc'],image_size_hw=(8,8))
                bottom=torch.zeros(1,5,1,4);bottom[...,0,3]=1
                poses=torch.linalg.inv(torch.cat([ext,bottom],dim=-2))[0].numpy()
            for i,frame in enumerate(ids):np.savetxt(scene/'pose'/f'{frame}.txt',poses[i])
            inp=root/'input.pt';torch.save(dict(images=images,frame_ids=ids,scene_root=str(scene),preprocessing={'elapsed_seconds':.01}),inp)
            output=root/'run';output.mkdir()
            args=argparse.Namespace(input=inp,output=output,gpu='0',frames=5,window_size=3,overlap=1,mode='camera_register_exchange')
            original_autocast=torch.autocast
            def autocast(device_type,**kw):return original_autocast('cpu',enabled=False)
            def digest(path):return config['checkpoint_sha256'] if str(path)==config['checkpoint'] else sha256(path)
            with ExitStack() as stack:
                stack.enter_context(patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'0'}))
                stack.enter_context(patch.object(worker,'preflight',return_value={'gpu_uuid':'CPU-TEST'}))
                stack.enter_context(patch.object(worker,'source_identity',return_value={'commit':'cpu-test','status':'','sha256':{}}))
                stack.enter_context(patch.object(worker,'sha256',side_effect=digest))
                stack.enter_context(patch('safetensors.torch.load_file',return_value=model.state_dict()))
                stack.enter_context(patch('vggt.models.vggt.VGGT',return_value=model))
                stack.enter_context(patch.object(torch.nn.Module,'cuda',lambda self,*a,**kw:self))
                stack.enter_context(patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self))
                stack.enter_context(patch('torch.autocast',side_effect=autocast))
                for name in ('synchronize','reset_peak_memory_stats','empty_cache'):stack.enter_context(patch.object(torch.cuda,name,return_value=None))
                for name in ('max_memory_allocated','max_memory_reserved'):stack.enter_context(patch.object(torch.cuda,name,return_value=0))
                worker.execute(args)
            self.assertTrue((output/'COMPLETE.json').is_file())
            for name in ('run_manifest.json','global_trajectory.npz','global_predictions.npz','trajectory.png','point_head_preview.png','trajectory_metrics.json'):
                self.assertTrue((output/name).is_file(),name)
            with np.load(output/'windows/0000/local.npz') as local:
                self.assertEqual(local['frame_ids'].tolist(),ids[:3])
            self.assertEqual(list(output.glob('window_*.pt')),[])
            self.assertTrue(np.isfinite(json.loads((output/'trajectory_metrics.json').read_text())['ate_rmse_m']))
            self.assertTrue((output/'alignment/edge_0000_0001_correspondence_mask.npz').is_file())
            self.assertTrue((output/'evaluation_summary.json').is_file())
            with np.load(output/'global_trajectory.npz') as global_data:
                self.assertEqual(global_data['source_window'].tolist(), [0,0,0,1,1])
                global_pose=global_data['c2w'].copy()
            with np.load(output/'global_predictions.npz') as dense:
                self.assertEqual(dense['source_window'].tolist(), [0,0,0,1,1])
                self.assertEqual(len(dense['depth']),5)
                self.assertEqual(len(dense['world_points']),5)
                np.testing.assert_array_equal(dense['c2w'],global_pose)
if __name__=='__main__':unittest.main()

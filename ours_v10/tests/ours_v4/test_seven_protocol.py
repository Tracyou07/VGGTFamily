"""Metric regression: exact existing evaluator, actual criterion/Open3D on CPU."""
import ast
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch

class ProtocolTest(unittest.TestCase):
    def test_identical_metrics_against_existing_long_adapter(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.sevenscenes.protocol'),'shared evaluator missing')
        from experiments.sevenscenes.protocol import bootstrap,evaluate_scene
        root=Path('/home/ubuntu/yjh/feedforwardreconstruct/eval/7scenes')
        _,criterion=bootstrap(root)
        source=(root/'adapters/eval_long_7scenes.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_scene')
        import open3d as o3d
        namespace=dict(np=np,torch=torch,o3d=o3d)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(root/'adapters/eval_long_7scenes.py'),'exec'),namespace)
        y,x=np.mgrid[:224,:224]/223
        points=np.stack((x,y,1+.1*np.sin(8*x)*np.cos(9*y)),axis=-1).astype(np.float32)
        gt=[dict(img=torch.zeros(1,3,224,224),pts3d=torch.from_numpy(points.copy())[None],
            valid_mask=torch.ones(1,224,224,dtype=torch.bool),camera_pose=torch.eye(4)[None])]
        pred=(points*1.13+.02)[None]; conf=np.ones((1,224,224),np.float32)
        old=namespace['evaluate_scene'](gt,pred.copy(),conf.copy(),criterion)
        new=evaluate_scene(gt,pred.copy(),conf.copy(),criterion)
        self.assertEqual(set(old),set(new))
        # Open3D's two independent ICP calls can differ by a few nanounits.
        for metric in old:
            self.assertAlmostEqual(old[metric],new[metric],delta=1e-7)

if __name__=='__main__': unittest.main()

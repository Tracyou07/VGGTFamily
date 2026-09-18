import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from scannet_eval.backends.common import (load_state, checked_load, validate_scene, w2c_to_c2w, finish_prediction, transform_projective, extract_long_chunks, extract_slam_submaps)
from scannet_eval.backends import create_backend, doctor_backend

class BackendTests(unittest.TestCase):
    def test_safe_checkpoint_formats_and_wrappers(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = {'weight': torch.ones(2, 3), 'bias': torch.zeros(2)}
            for filename, wrapper in [('raw.pth', state), ('wrapped.pt', {'model': state}), ('state.pt', {'state_dict': state})]:
                torch.save(wrapper, root / filename)
                self.assertEqual(set(load_state(root / filename)), set(state))
            save_file(state, root / 'weights.safetensors')
            self.assertTrue(torch.equal(load_state(root / 'weights.safetensors')['weight'], state['weight']))
            torch.save({'bad': Path('unsafe')}, root / 'unsafe.pt')
            with self.assertRaises(Exception): load_state(root / 'unsafe.pt')

    def test_weight_validation_rejects_missing_core_shape_and_unknown_keys(self):
        model = torch.nn.Linear(3, 2)
        state = model.state_dict()
        checked_load(model, dict(state, **{'point_head.unused': torch.ones(1)}), ('point_head.',))
        for bad in ({'weight': state['weight']}, dict(state, weight=torch.zeros(4, 3)), dict(state, surprise=torch.ones(1))):
            with self.assertRaises(ValueError): checked_load(model, bad)
        self.assertTrue(torch.equal(model.weight, state['weight']))

    def test_scene_coverage_rejected_before_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'a.jpg'; p.touch()
            for ids, paths in [((), ()), ((1, 1), (p, p)), ((1, 2), (p,)), ((1,), (p.with_name('missing.jpg'),))]:
                with self.assertRaises(ValueError): validate_scene(SimpleNamespace(frame_ids=ids, image_paths=paths))
            validate_scene(SimpleNamespace(frame_ids=(7,), image_paths=(p,)))

    def test_rigid_inverse_and_prediction_count(self):
        w2c = np.eye(4)[None].repeat(2, 0); w2c[1, :3, 3] = [1, 2, 3]
        poses = w2c_to_c2w(w2c[:, :3])
        np.testing.assert_allclose(poses[1, :3, 3], [-1, -2, -3])
        with self.assertRaises(ValueError): finish_prediction(np.ones((2, 3)), poses[:1], (4, 9), 1, 0, 0, {}, 0)
        with self.assertRaises(ValueError): finish_prediction(np.ones((2, 3)), poses * 2, (4, 9), 1, 0, 0, {}, 0)

    def test_finite_filter_and_optional_cap(self):
        pts = np.array([[0,0,0], [100,1,1], [np.nan,2,3], [np.inf,0,0]])
        pred = finish_prediction(pts, np.eye(4)[None], (20,), 1, 0, 0, {}, 0)
        np.testing.assert_equal(pred.points, pts[:2])
        self.assertEqual(pred.metadata['invalid_points_removed'], 2)
        self.assertEqual(len(finish_prediction(pts, np.eye(4)[None], (20,), 1, 0, 0, {}, 1).points), 1)

    def test_projective_camera_local_points_and_invalid_denominator(self):
        H = np.eye(4); H[0,3] = 10; H[3,2] = -1
        result, valid = transform_projective(np.array([[1,2,0], [1,2,1], [1,2,2.]]), H)
        np.testing.assert_equal(valid, [True, False, True])
        np.testing.assert_allclose(result[valid], [[11,2,0], [-11,-2,-2]])

    def test_long_sim3_and_last_overlap_ownership(self):
        poses = np.eye(4)[None].repeat(2, 0)
        chunks = [dict(world_points=np.array([[[1.,0,0]], [[2.,0,0]]]), world_points_conf=np.ones((2,1)), extrinsic=poses), dict(world_points=np.array([[[3.,0,0]], [[4.,0,0]]]), world_points_conf=np.ones((2,1)), extrinsic=poses)]
        points, cameras, meta = extract_long_chunks(chunks, [(0,2), (1,3)], [(2., np.eye(3), np.array([10,0,0]))], 3, .75)
        np.testing.assert_allclose(points[:,0], [1,16,18])
        np.testing.assert_allclose(cameras[:,0,3], [0,10,10])
        self.assertEqual(meta['frame_owner'], [0,1,1])
        single, _, _ = extract_long_chunks(chunks[:1], [(0,2)], [], 2, .75)
        self.assertEqual(len(single), 2)

    def test_slam_first_overlap_ownership_and_native_rigid_pose(self):
        class Submap:
            def __init__(self, start, names): self.start=start; self.img_names=names; self.pointclouds=np.ones((len(names),1,3)); self.conf_masks=np.ones((len(names),1))*2; self.conf_threshold=1
            def get_id(self): return self.start
            def get_lc_status(self): return False
            def get_all_poses_world(self, graph):
                out=np.eye(4)[None].repeat(len(self.img_names),0); out[:,0,3]=self.start; return out
        graph=SimpleNamespace(get_homography=lambda i: np.array([[1,0,0,i],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
        points, poses, meta=extract_slam_submaps([Submap(0,['a','b']),Submap(2,['b','c'])], graph, {'a':7,'b':20,'c':99}, (7,20,99))
        np.testing.assert_equal(points[:,0],[1,2,4]); np.testing.assert_equal(poses[:,0,3],[0,0,2])
        self.assertEqual(meta['frame_owner'], [0,0,2])
        with self.assertRaises(ValueError): extract_slam_submaps([Submap(0,['a'])],graph,{'a':7},(7,20))

    @unittest.skipUnless(os.environ.get('SCANNET_NATIVE_SLAM_TEST') == '1', 'native SLAM fixture requires monst3r')
    def test_native_slam_rq_pose_with_rotation_translation_and_calibration(self):
        from scannet_eval.backends import resolve_config
        from scannet_eval.backends.runtime import install_source
        root=Path(resolve_config('slam',{})['project_root'])
        install_source(root,'vggt_slam')
        from vggt_slam.submap import Submap
        R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        C=np.array([3.,4.,5.]); K=np.array([[500.,2.,320.],[0,510.,240.],[0,0,1.]])
        c2w=np.eye(4); c2w[:3,:3]=R; c2w[:3,3]=C
        proj=np.eye(4); proj[:3,:3]=K
        sub=Submap(0); sub.poses=np.eye(4)[None]; sub.proj_mats=proj[None]
        graph=SimpleNamespace(get_homography=lambda i:c2w)
        np.testing.assert_allclose(sub.get_all_poses_world(graph)[0],c2w,atol=1e-10)

    def test_fast_bf16_forward_decodes_rigid_cameras_in_fp32(self):
        from contextlib import nullcontext
        from scannet_eval.backends import resolve_config
        from scannet_eval.backends.runtime import install_source
        root=Path(resolve_config('fastvggt',{})['project_root'])
        if not root.exists(): self.skipTest('requires the configured FastVGGT source')
        install_source(root,'vggt')
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from scannet_eval.vendor.fastvggt_eval_utils import infer_vggt_and_reconstruct
        # Nontrivial BF16 quaternion: decoding in BF16 violates rigid geometry.
        enc=torch.tensor([[[.1,.2,.3,.23,.41,.17,.81,1.,1.]]],dtype=torch.bfloat16)
        raw,_=pose_encoding_to_extri_intri(enc,(2,2))
        expected,_=pose_encoding_to_extri_intri(enc.float(),(2,2))
        raw_rotation=raw.float().numpy()[0,:,:3,:3]
        self.assertGreater(np.max(np.abs(raw_rotation@raw_rotation.transpose(0,2,1)-np.eye(3))),.002)
        predictions={'pose_enc':enc,'depth':torch.ones(1,1,2,2,1),'depth_conf':torch.ones(1,1,2,2)*2}
        model=lambda *args,**kwargs: predictions
        with patch('torch.cuda.synchronize'), patch.object(torch.Tensor,'cuda',lambda tensor:tensor), patch('torch.cuda.amp.autocast',return_value=nullcontext()):
            ext,_,points,_,_,_=infer_vggt_and_reconstruct(model,torch.zeros(1,3,2,2),torch.bfloat16,1.)
        np.testing.assert_allclose(ext,expected.numpy()[0],atol=1e-7)
        validate=w2c_to_c2w(ext)
        self.assertEqual(validate.shape,(1,4,4))
        self.assertTrue(np.isfinite(np.concatenate(points)).all())

    def test_doctor_no_allocation_and_invalid_config(self):
        with patch('torch.cuda.is_available', side_effect=AssertionError('GPU access')):
            result=doctor_backend('long', {'project_root':'/missing','checkpoint':'/missing'})
        self.assertFalse(result['ready']); self.assertTrue(result['errors'])
        with self.assertRaises(ValueError): create_backend('unknown',{},'cpu')
        with self.assertRaises(ValueError): create_backend('vggt_original',{'image_size':512},'cpu')
        with self.assertRaises(ValueError): create_backend('slam',{'max_loops':2},'cpu')

if __name__ == '__main__': unittest.main()

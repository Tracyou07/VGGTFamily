"""CPU checks for the opt-in native VGGT* backend contract."""
import os
import unittest
from unittest.mock import patch
import torch

from experiments.ours_v7 import backend_profiles as profiles
from experiments.ours_v7 import worker

class BackendProfilesTest(unittest.TestCase):
    def test_early_environment_selection_and_conflicts(self):
        env={}
        self.assertEqual(profiles.prepare_environment("legacy", env, explicit=False),"legacy")
        self.assertEqual(env["CUBLAS_WORKSPACE_CONFIG"],":4096:8")
        with self.assertRaisesRegex(ValueError,"CUBLAS_WORKSPACE_CONFIG"):
            profiles.prepare_environment("native_vggt",dict(env),explicit=True)
        self.assertEqual(profiles.prepare_environment("native_vggt",{},explicit=True),"native_vggt")
        with self.assertRaisesRegex(ValueError,"CUBLAS_WORKSPACE_CONFIG"):
            profiles.prepare_environment("legacy",{"CUBLAS_WORKSPACE_CONFIG":":16:8"},explicit=True)

    def test_profiles_effective_flags_and_default_compatibility(self):
        with patch.object(torch.backends.cudnn,"version",return_value=91900):
            self._check_profile_flags()

    def _check_profile_flags(self):
        before=profiles.snapshot(torch)
        try:
            profiles.apply_backend_profile("native_vggt",torch)
            native=profiles.snapshot(torch)
            self.assertEqual(native["cudnn_allow_tf32"],True)
            self.assertEqual(native["matmul_allow_tf32"],False)
            self.assertEqual(native["deterministic_algorithms"],False)
            self.assertEqual(native["cudnn_deterministic"],False)
            self.assertEqual(native["float32_matmul_precision"],"highest")
            self.assertEqual(native["sdpa"],dict(flash=True,memory_efficient=True,math=True))
            profiles.apply_backend_profile("legacy",torch)
            legacy=profiles.snapshot(torch)
            self.assertTrue(legacy["deterministic_algorithms"])
            self.assertFalse(legacy["cudnn_allow_tf32"])
            self.assertFalse(legacy["cudnn_benchmark"])
            self.assertFalse(legacy["matmul_allow_tf32"])
        finally:
            profiles.restore_snapshot(torch,before)

    def test_worker_defaults_and_dense_heads_unsplit(self):
        args=worker.parse_args(["--input","in.pt","--output","out","--gpu","4",
            "--frames","100","--mode","camera_patch_exchange"])
        self.assertEqual(args.backend_profile,"legacy")
        self.assertIsNone(args.dense_head_frame_chunk)
        native=worker.parse_args(["--input","in.pt","--output","out","--gpu","4",
            "--frames","100","--mode","independent","--backend-profile","native_vggt"])
        self.assertIsNone(native.dense_head_frame_chunk)
        self.assertEqual(native.window_size,args.window_size)
        self.assertEqual(native.overlap,args.overlap)
        self.assertEqual(native.patch_exchange_ratio,args.patch_exchange_ratio)

if __name__=="__main__":unittest.main()

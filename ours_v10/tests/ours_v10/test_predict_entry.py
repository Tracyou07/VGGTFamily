"""Frozen-input v10 prediction entrypoint contracts, without CUDA."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.ours_v6.runtime import sha256
from experiments.ours_v9.vkitti_predict import tensor_sha256
from experiments.ours_v10.predict import parse_args, validate_frozen_input


class V10PredictionEntryTest(unittest.TestCase):
    def test_all_four_modes_and_full_frame_default(self):
        for mode in ("independent","camera_only","overlap_correspondence",
                     "camera_global_overlap"):
            args=parse_args(["--frozen-input-root","/tmp/frozen",
                "--output-root","/tmp/fresh","--mode",mode,"--gpu","4",
                "--backend-profile","native_vggt"])
            self.assertEqual(args.mode,mode)
            self.assertEqual(args.frames,837)
        with self.assertRaises(SystemExit):
            parse_args(["--frozen-input-root","/tmp/frozen","--output-root",
                        "/tmp/fresh","--mode","camera_exchange","--gpu","4",
                        "--backend-profile","native_vggt"])
        with self.assertRaises(SystemExit):
            parse_args(["--frozen-input-root","/tmp/frozen","--output-root",
                        "/tmp/fresh","--mode","camera_only","--gpu","4"])

    def test_same_preprocessed_input_and_checkpoint_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ids=[f"{i:05d}" for i in range(100)]
            images=torch.arange(100*3*2*2,dtype=torch.float32).reshape(100,3,2,2)
            payload=dict(images=images,frame_ids=ids,
                         rgb_paths=[f"/{i:05d}.png" for i in range(100)],
                         preprocessing=dict(loader="crop",shape=list(images.shape)))
            torch.save(payload,root/"inputs.pt")
            checkpoint=root/"model.safetensors"
            checkpoint.write_bytes(b"fixed weights")
            manifest=dict(dataset="Virtual KITTI 1.3.1",scene="Scene20",
                condition="clone",frame_ids=ids,input_path=str(root/"inputs.pt"),
                input_sha256=sha256(root/"inputs.pt"),
                image_tensor_sha256=tensor_sha256(images),
                checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint))
            (root/"input_manifest.json").write_text(json.dumps(manifest))
            saved,source,frame_ids=validate_frozen_input(root,100)
            self.assertEqual(frame_ids,ids)
            self.assertTrue(torch.equal(saved["images"],images))
            self.assertEqual(source["checkpoint_sha256"],sha256(checkpoint))
            manifest["image_tensor_sha256"]="wrong"
            (root/"input_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"image tensor"):
                validate_frozen_input(root,100)


if __name__=="__main__":
    unittest.main()

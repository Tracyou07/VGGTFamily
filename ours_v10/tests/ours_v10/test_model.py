"""Four-mode prediction interface keeps inherited v8 heads and fields."""
import unittest

import torch
from torch import nn

from vggt.models.aggregator import Aggregator
from vggt.models.vggt import VGGT
from vggt.layers.patch_embed import PatchEmbed
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.v8.model import WindowReconstructor as V8WindowReconstructor
from vggt.v10.model import WindowReconstructor as V10WindowReconstructor


def small_model():
    torch.manual_seed(264)
    model=VGGT.__new__(VGGT)
    nn.Module.__init__(model)
    model.aggregator=Aggregator(img_size=8,patch_size=2,embed_dim=8,depth=4,
        num_heads=2,num_register_tokens=1,patch_embed="conv",
        cached_layer_indices=(0,1,2,3))
    model.aggregator.patch_embed=PatchEmbed(img_size=8,patch_size=2,embed_dim=8)
    model.camera_head=CameraHead(dim_in=16,trunk_depth=1,num_heads=2)
    with torch.no_grad():
        model.camera_head.pose_branch.fc2.weight[7:].zero_()
        model.camera_head.pose_branch.fc2.bias[7:]=.25
    kwargs=dict(dim_in=16,patch_size=2,features=8,out_channels=[8]*4,
                intermediate_layer_idx=[0,1,2,3])
    model.depth_head=DPTHead(**kwargs,output_dim=2,activation="exp")
    model.point_head=DPTHead(**kwargs,output_dim=4)
    model.track_head=None
    return model.eval().requires_grad_(False)


class V10ModelTest(unittest.TestCase):
    def test_opt_in_cpu_head_cache_preserves_all_predictions(self):
        torch.set_num_threads(1)
        model = small_model()
        images = torch.rand(5, 3, 8, 8)
        ids = [f"f{i}" for i in range(5)]
        reconstructor = V10WindowReconstructor(model)
        options = dict(mode="camera_global_overlap", window_size=3, overlap=1,
                       query_chunk_size=2, correspondence_attention_path="native_sdpa")
        reference = reconstructor(images, ids, **options)
        optimized = reconstructor(images, ids, offload_head_features=True, **options)
        self.assertTrue(optimized["cpu_offload"])
        self.assertEqual(reference["windows"], optimized["windows"])
        for old_window, new_window in zip(reference["predictions"], optimized["predictions"]):
            self.assertEqual(old_window["frame_ids"], new_window["frame_ids"])
            for key in ("pose_encoding", "c2w", "intrinsics", "depth", "depth_conf",
                        "world_points", "world_points_conf"):
                self.assertTrue(torch.equal(old_window[key], new_window[key]), key)

    def test_opt_in_streamed_qkv_preserves_camera_mode_predictions(self):
        torch.set_num_threads(1)
        model = small_model()
        images = torch.rand(5, 3, 8, 8)
        ids = [f"f{i}" for i in range(5)]
        reconstructor = V10WindowReconstructor(model)
        for mode in ("camera_only", "camera_global_overlap"):
            options = dict(mode=mode, window_size=3, overlap=1,
                           query_chunk_size=2, correspondence_attention_path="native_sdpa")
            reference = reconstructor(images, ids, **options)
            optimized = reconstructor(images, ids, stream_projected_qkv=True,
                                      offload_head_features=True, **options)
            self.assertEqual(reference["windows"], optimized["windows"])
            for old_window, new_window in zip(reference["predictions"],
                                              optimized["predictions"]):
                self.assertEqual(old_window["frame_ids"], new_window["frame_ids"])
                for key in ("pose_encoding", "c2w", "intrinsics", "depth", "depth_conf",
                            "world_points", "world_points_conf"):
                    self.assertTrue(torch.equal(old_window[key], new_window[key]),
                                    f"{mode}: {key}")

    def test_inherited_predictions_exact_and_new_modes_change_output(self):
        torch.set_num_threads(1)
        model=small_model()
        images=torch.rand(5,3,8,8)
        ids=[f"f{i}" for i in range(5)]
        v8=V8WindowReconstructor(model)
        v10=V10WindowReconstructor(model)
        kwargs=dict(window_size=3,overlap=1,query_chunk_size=2,
                    correspondence_attention_path="native_sdpa")
        old={mode:v8(images,ids,mode=mode,**kwargs) for mode in
             ("independent","overlap_correspondence")}
        results={mode:v10(images,ids,mode=mode,**kwargs) for mode in
             ("independent","camera_only","overlap_correspondence",
              "camera_global_overlap")}
        for mode in old:
            for a,b in zip(old[mode]["predictions"],results[mode]["predictions"]):
                self.assertEqual(a["frame_ids"],b["frame_ids"])
                for key in ("pose_encoding","c2w","intrinsics","depth",
                            "depth_conf","world_points","world_points_conf"):
                    self.assertTrue(torch.equal(a[key],b[key]),f"{mode}: {key}")
        for mode in results:
            self.assertEqual(results[mode]["windows"],[(0,3),(2,5)])
            for pred in results[mode]["predictions"]:
                for key in ("pose_encoding","c2w","intrinsics","depth",
                            "depth_conf","world_points","world_points_conf"):
                    self.assertTrue(bool(torch.isfinite(pred[key]).all()),key)
        for base,new in (("independent","camera_only"),
                         ("overlap_correspondence","camera_global_overlap")):
            difference=max(float((a["pose_encoding"]-b["pose_encoding"]).abs().max())
                for a,b in zip(results[base]["predictions"],
                               results[new]["predictions"]))
            self.assertGreater(difference,0.0)


if __name__=="__main__":
    unittest.main()

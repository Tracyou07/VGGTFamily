"""CPU integration checks for the v8 scheduler, heads, and launch interface."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from experiments.ours_v6.windows import make_windows
from experiments.ours_v8.worker import parse_args
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.layers.patch_embed import PatchEmbed
from vggt.models.aggregator import Aggregator
from vggt.models.vggt import VGGT
from vggt.v8 import scheduler
from vggt.v8.model import WindowReconstructor


def small_aggregator():
    torch.manual_seed(81)
    aggregator = Aggregator(img_size=4, patch_size=2, embed_dim=8, depth=3,
                            num_heads=2, num_register_tokens=2,
                            patch_embed="conv", cached_layer_indices=(0, 2))
    aggregator.patch_embed = PatchEmbed(img_size=4, patch_size=2, embed_dim=8)
    return aggregator.double().eval().requires_grad_(False)


class V8IntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self._grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    def tearDown(self):
        torch.set_grad_enabled(self._grad)

    def test_fixed_schedule_and_independent_native_cache(self):
        self.assertEqual(make_windows(100, 60, 30),
                         [(0, 60), (30, 90), (60, 100)])
        agg = small_aggregator()
        images = torch.rand(7, 3, 4, 4, dtype=torch.float64)
        frame_ids = [f"original-{i}" for i in range(7)]
        windows = make_windows(7, 4, 2)
        actual, patch_start, _, summary = scheduler.aggregate_windows(
            agg, images, frame_ids, windows, "independent")
        self.assertEqual(summary["pair_count"], 0)
        self.assertEqual(patch_start, agg.patch_start_idx)
        for cache, (lo, hi) in zip(actual, windows):
            reference, _ = agg(images[lo:hi][None])
            for got, expected in zip(cache, reference):
                if expected is None:
                    self.assertIsNone(got)
                else:
                    torch.testing.assert_close(got, expected, atol=1e-10, rtol=1e-10)

    def test_layer_barrier_and_reverse_window_order(self):
        agg = small_aggregator()
        images = torch.rand(7, 3, 4, 4, dtype=torch.float64)
        ids = [f"frame-{i}" for i in range(7)]
        windows = [(0,4),(2,6),(4,7)]
        events = []
        handles = [block.register_forward_hook(
            lambda _module, _args, _out, layer=layer: events.append(("frame", layer)))
            for layer, block in enumerate(agg.frame_blocks)]
        original = scheduler.global_step
        def traced(block, *args, **kwargs):
            layer = next(i for i, value in enumerate(agg.global_blocks) if value is block)
            events.append(("global", layer))
            return original(block, *args, **kwargs)
        try:
            with patch.object(scheduler, "global_step", traced):
                forward, _, _, summary = scheduler.aggregate_windows(
                    agg, images, ids, windows, "overlap_correspondence",
                    query_chunk_size=3)
        finally:
            for handle in handles:
                handle.remove()
        for layer in range(agg.depth):
            start = layer * 4
            self.assertEqual(events[start:start+4],
                             [("frame",layer)]*3 + [("global",layer)])
        self.assertEqual(summary["pair_count"], 32)
        backward, _, _, _ = scheduler.aggregate_windows(
            agg, images, ids, windows, "overlap_correspondence",
            query_chunk_size=3, reverse=True)
        for first, second in zip(forward, backward):
            for a,b in zip(first,second):
                if a is not None:
                    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)

    def test_image_encoding_reuse_and_kv_cache_preserve_window_states(self):
        agg = small_aggregator()
        images = torch.rand(7, 3, 4, 4, dtype=torch.float64)
        ids = [f"original-{i}" for i in range(7)]
        windows = [(0,4),(2,6),(4,7)]
        plain, _, _, _ = scheduler.aggregate_windows(
            agg, images, ids, windows, "overlap_correspondence", query_chunk_size=3)
        reused, _, _, _ = scheduler.aggregate_windows(
            agg, images, ids, windows, "overlap_correspondence", query_chunk_size=3,
            reuse_image_encoding=True, cache_local_kv_dtype=True)
        for first, second in zip(plain, reused):
            for a, b in zip(first, second):
                if a is not None:
                    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)

    def test_window_heads_keep_all_required_fields(self):
        torch.manual_seed(82)
        model = VGGT.__new__(VGGT)
        nn.Module.__init__(model)
        model.aggregator = Aggregator(img_size=8,patch_size=2,embed_dim=8,
            depth=4,num_heads=2,num_register_tokens=1,patch_embed="conv",
            cached_layer_indices=(0,1,2,3))
        model.aggregator.patch_embed = PatchEmbed(img_size=8,patch_size=2,embed_dim=8)
        model.camera_head = CameraHead(dim_in=16,trunk_depth=1,num_heads=2)
        model.camera_head.pose_branch.fc2.weight[7:].zero_()
        model.camera_head.pose_branch.fc2.bias[7:] = .25
        kwargs = dict(dim_in=16,patch_size=2,features=8,out_channels=[8]*4,
                      intermediate_layer_idx=[0,1,2,3])
        model.depth_head = DPTHead(**kwargs,output_dim=2,activation="exp")
        model.point_head = DPTHead(**kwargs,output_dim=4)
        model.track_head = None
        model.eval().requires_grad_(False)
        images = torch.rand(5,3,8,8)
        ids = list("abcde")
        wrapper = WindowReconstructor(model)
        independent = wrapper(images,ids,mode="independent",window_size=3,overlap=1)
        communicated = wrapper(images,ids,mode="overlap_correspondence",
                               window_size=3,overlap=1,query_chunk_size=2)
        self.assertEqual(communicated["windows"],[(0,3),(2,5)])
        self.assertEqual(communicated["correspondence"]["pair_count"],32)
        required = ("pose_encoding","c2w","intrinsics","depth","depth_conf",
                    "world_points","world_points_conf")
        for result in (independent, communicated):
            for pred,(lo,hi) in zip(result["predictions"],result["windows"]):
                self.assertEqual(pred["frame_ids"],ids[lo:hi])
                for field in required:
                    self.assertTrue(torch.isfinite(pred[field]).all(),field)
        for pred,(lo,hi) in zip(independent["predictions"],independent["windows"]):
            original = model(images[lo:hi][None])
            for field, native in (("pose_encoding","pose_enc"),("depth","depth"),
                                  ("depth_conf","depth_conf"),
                                  ("world_points","world_points"),
                                  ("world_points_conf","world_points_conf")):
                torch.testing.assert_close(pred[field],original[native][0],
                                           atol=2e-5,rtol=2e-5)

    def test_launch_defaults_and_removed_v7_modes(self):
        required = ["--input","fixed.pt","--output","new_run","--gpu","0",
                    "--frames","100","--mode","overlap_correspondence"]
        args = parse_args(required)
        self.assertEqual((args.window_size,args.overlap,args.query_chunk_size),(60,30,16))
        self.assertEqual(args.correspondence_attention_path,"explicit")
        self.assertEqual(args.alignment_mode,"point_legacy")
        self.assertFalse(args.cache_local_kv_dtype)
        self.assertTrue(parse_args(required+["--cache-local-kv-dtype"]).cache_local_kv_dtype)
        self.assertEqual(parse_args(required+["--correspondence-attention-path","native_sdpa"])
                         .correspondence_attention_path,"native_sdpa")
        self.assertEqual(parse_args(required+["--alignment-mode","point_camera_joint"])
                         .alignment_mode,"point_camera_joint")
        with self.assertRaises(SystemExit):
            parse_args(required[:-1]+["camera_patch_exchange"])
        with self.assertRaises(SystemExit):
            parse_args(required+["--query-chunk-size","0"])


if __name__ == "__main__":
    unittest.main()

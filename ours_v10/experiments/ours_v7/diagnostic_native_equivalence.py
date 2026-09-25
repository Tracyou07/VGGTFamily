"""Read-only full-native equivalence and v7 100% divergence diagnostic.

Writes only JSON statistics. It does not export activations, predictions, or
change model code. Run in a fresh process with CUDA_VISIBLE_DEVICES set.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.ours_v7.backend_profiles import apply_backend_profile, snapshot

REPO = Path("/home/ubuntu/yjh/feedforwardreconstruct/ours_v7")
NATIVE_REPO = Path("/home/ubuntu/yjh/feedforwardreconstruct/vggt")
HISTORY = Path("/data/yjh/output/vggt/ours_v7_diagnostics/20260922T104000Z_w60_o10_patch100")
BASELINE = Path("/data/yjh/output/vggt/fixed_input_baselines/20260922T031101Z_fixed_baselines_vggt_star")
FIELDS = ("pose_encoding", "c2w", "intrinsics", "depth", "depth_conf",
          "world_points", "world_points_conf")
SAMPLE_FRAMES = (0, 1, 50, 55, 59, 60, 99)
SAMPLE_TOKENS = (0, 1, 4, 5, 5 + 518, 1040)
ATOL = RTOL = 2e-5  # pre-existing raw-output information gate, not a structural gate


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def difference(reference, actual):
    """Numerical comparison on CPU or GPU, without retaining a copy."""
    a = torch.as_tensor(reference)
    b = torch.as_tensor(actual)
    if a.shape != b.shape:
        return {"shape_equal": False, "reference_shape": list(a.shape),
                "actual_shape": list(b.shape)}
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"shape_equal": True, "shape": list(a.shape), "finite": False}
    delta = (a.float() - b.float()).double()
    base = a.float().double()
    max_abs = float(delta.abs().max())
    mean_abs = float(delta.abs().mean())
    relative_l2 = float(torch.linalg.vector_norm(delta) /
                        torch.linalg.vector_norm(base).clamp_min(1e-300))
    exact = bool(torch.equal(a, b))
    within = bool((delta.abs() <= ATOL + RTOL * base.abs()).all())
    return {"shape_equal": True, "shape": list(a.shape), "finite": finite,
            "max_abs": max_abs, "mean_abs": mean_abs, "relative_l2": relative_l2,
            "exact": exact, "within_2e_5": within}


def native_manual(model, images):
    """One original full-sequence aggregator call and original joint heads."""
    features, patch_start = model.aggregator(images)
    with torch.autocast(device_type=images.device.type, enabled=False):
        pose = model.camera_head(features)[-1]
        depth, depth_conf = model.depth_head(features, images=images,
                                            patch_start_idx=patch_start)
        points, points_conf = model.point_head(features, images=images,
                                               patch_start_idx=patch_start)
    return {"pose_enc": pose, "depth": depth, "depth_conf": depth_conf,
            "world_points": points, "world_points_conf": points_conf}


def native_full_kv_grouped_block(block, x, pos, chunks):
    """Diagnostic scheduling-only variant: all queries still use all K/V."""
    attention = block.attn
    normalized = block.norm1(x)
    raw = attention.qkv(normalized).reshape(
        x.shape[0], x.shape[1], 3, attention.num_heads, attention.head_dim)
    q, k, v = raw.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attention.q_norm(q), attention.k_norm(k)
    if attention.rope is not None:
        q, k = attention.rope(q, pos), attention.rope(k, pos)
    groups = []
    for lo, hi in chunks:
        if not 0 <= lo < hi <= x.shape[1]:
            raise ValueError("invalid query group")
        groups.append((lo, hi))
    if sorted(i for lo, hi in groups for i in range(lo, hi)) != list(range(x.shape[1])):
        raise ValueError("query groups must cover each original index once")
    output = torch.empty_like(q)
    for lo, hi in groups:
        # One softmax across the complete, original K/V for every Query.
        output[:, :, lo:hi] = F.scaled_dot_product_attention(
            q[:, :, lo:hi], k, v, dropout_p=0.0, scale=attention.scale)
    y = attention.proj_drop(attention.proj(output.transpose(1, 2).reshape_as(x)))
    x = x + block.ls1(y)
    return x + block.ls2(block.mlp(block.norm2(x)))


def native_values(pred, image_hw):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    with torch.autocast("cuda", enabled=False):
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pred["pose_enc"].float(), image_size_hw=image_hw)
        bottom = torch.zeros((*extrinsic.shape[:2], 1, 4),
                             device=extrinsic.device, dtype=extrinsic.dtype)
        bottom[..., 0, 3] = 1
        c2w = torch.linalg.inv(torch.cat([extrinsic, bottom], dim=-2))
    values = {"pose_encoding": pred["pose_enc"], "c2w": c2w,
              "intrinsics": intrinsic,
              **{key: pred[key] for key in FIELDS if key not in
                 ("pose_encoding", "c2w", "intrinsics")}}
    return {key: value[0].detach().float().cpu() for key, value in values.items()}


class Capture:
    def __init__(self, aggregator):
        self.phase = "none"
        self.native = {}
        self.patch = None
        self.rows = []
        self.window_counter = {}
        self.v7_global_layer_count = 0
        self.handles = []
        self.windows = ((0, 60), (50, 100))
        self.handles.append(aggregator.patch_embed.register_forward_hook(self.on_patch))
        for stage, blocks in (("frame", aggregator.frame_blocks),
                              ("global", aggregator.global_blocks)):
            for layer, block in enumerate(blocks):
                self.handles.append(block.register_forward_hook(
                    lambda _m, _a, y, stage=stage, layer=layer:
                    self.on_layer(stage, layer, y)))

    def on_patch(self, _module, _args, value):
        x = value["x_norm_patchtokens"] if isinstance(value, dict) else value
        if self.phase == "native":
            self.patch = x.detach().cpu().clone()
        elif self.phase == "v7":
            self.rows.append({"stage": "image_encoder", "layer": None,
                              "window": None, "frame": None,
                              **difference(self.patch, x.detach().cpu())})
            self.patch = None

    def on_layer(self, stage, layer, value):
        x = value.detach().reshape(-1, 1041, value.shape[-1])
        if self.phase == "native":
            self.native[(stage, layer)] = {
                frame: x[frame, SAMPLE_TOKENS].cpu().clone()
                for frame in SAMPLE_FRAMES}
        elif self.phase == "v7":
            key = (stage, layer)
            window = self.window_counter.get(key, 0)
            self.window_counter[key] = window + 1
            lo, hi = self.windows[window]
            for frame in SAMPLE_FRAMES:
                if lo <= frame < hi:
                    self.rows.append({"stage": stage, "layer": layer,
                                      "window": window, "frame": frame,
                                      "sample_token_indices": SAMPLE_TOKENS,
                                      **difference(self.native[key][frame],
                                                   x[frame-lo, SAMPLE_TOKENS].cpu())})

    def on_v7_global(self, layer, values):
        # v7 global_step calls block components directly, so Block.forward hooks
        # cannot observe this result. This wrapper records its returned states.
        self.v7_global_layer_count += 1
        for window, value in enumerate(values):
            lo, hi = self.windows[window]
            x = value.detach().reshape(hi-lo, 1041, value.shape[-1])
            for frame in SAMPLE_FRAMES:
                if lo <= frame < hi:
                    self.rows.append({"stage": "global", "layer": layer,
                                      "window": window, "frame": frame,
                                      "sample_token_indices": SAMPLE_TOKENS,
                                      **difference(self.native[("global", layer)][frame],
                                                   x[frame-lo, SAMPLE_TOKENS].cpu())})

    def close(self):
        for handle in self.handles:
            handle.remove()


@contextmanager
def record_v7_globals(capture):
    import vggt.v7.scheduler as scheduler
    original = scheduler.global_step
    count = 0

    def traced(*args, **kwargs):
        nonlocal count
        result = original(*args, **kwargs)
        capture.on_v7_global(count, result)
        count += 1
        return result

    scheduler.global_step = traced
    try:
        yield
    finally:
        scheduler.global_step = original


def initialization(model, images_cpu):
    from vggt.models.aggregator import slice_expand_and_flatten
    from vggt.v6.scheduler import initialize
    agg = model.aggregator
    images = images_cpu[None].cuda()
    normalized = (images - agg._resnet_mean) / agg._resnet_std
    patch = agg.patch_embed(normalized.reshape(100, 3, *images.shape[-2:]))
    if isinstance(patch, dict):
        patch = patch["x_norm_patchtokens"]
    native = torch.cat([slice_expand_and_flatten(agg.camera_token, 1, 100),
                        slice_expand_and_flatten(agg.register_token, 1, 100), patch], dim=1)
    spatial = agg.position_getter(100, 28, 37, device=images.device) + 1
    native_pos = torch.cat([torch.zeros(100, 5, 2, device=images.device,
                                        dtype=spatial.dtype), spatial], dim=1)
    states, positions, token_count = initialize(agg, images_cpu,
        [(0, 60), (50, 100)], precomputed_patch_tokens=patch)
    if token_count != 1041:
        raise AssertionError("unexpected token count")
    rows = []
    for w, (lo, hi) in enumerate(((0, 60), (50, 100))):
        actual = states[w].reshape(hi-lo, 1041, -1)
        reference = native[lo:hi]
        rows.append({"stage": "initialization", "window": w,
                     "component": "all_tokens", **difference(reference, actual)})
        rows.append({"stage": "initialization", "window": w,
                     "component": "rope_positions", **difference(
                         native_pos[lo:hi], positions[w].reshape(hi-lo, 1041, 2))})
        for frame in (lo, min(lo+1,hi-1)):
            local = frame-lo
            for name, region in (("camera", slice(0,1)), ("register", slice(1,5)),
                                 ("patch", slice(5,None))):
                rows.append({"stage": "initialization", "window": w,
                             "frame": frame, "component": name,
                             **difference(reference[local, region], actual[local, region])})
    del images, normalized, patch, native, states, positions
    return rows


def source_manifest(out, historical, profile):
    historical_source = json.loads((HISTORY / "source_manifest.json").read_text())
    baseline = json.loads((BASELINE / "run_manifest.json").read_text())
    inputs = Path(historical_source["input"])
    checkpoint = Path(historical_source["checkpoint"])
    sources = ["vggt/models/vggt.py", "vggt/models/aggregator.py",
               "vggt/layers/attention.py", "vggt/layers/block.py",
               "vggt/heads/camera_head.py", "vggt/heads/dpt_head.py"]
    manifest = {
        "ours_commit": git(REPO, "rev-parse", "HEAD"),
        "ours_status": git(REPO, "status", "--short"),
        "native_commit": git(NATIVE_REPO, "rev-parse", "HEAD"),
        "native_status": git(NATIVE_REPO, "status", "--short"),
        "historical_v7_commit": historical_source["code_commit"],
        "historical_v7_100_run_manifest": str(historical),
        "historical_baseline_manifest": str(BASELINE / "run_manifest.json"),
        "historical_v7_ate_m": historical["evaluation_summary"]["ate_rmse_m"],
        "inputs": str(inputs), "input_sha256": sha(inputs),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "historical_input_sha256": historical_source["input_sha256"],
        "historical_checkpoint_sha256": historical_source["checkpoint_sha256"],
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "frame_ids": historical["frame_ids"],
        "windows": historical["windows"],
        "historical_configuration": historical["configuration"],
        "historical_backend_effective": historical["backend_profile"]["effective"],
        "diagnostic_backend_effective": profile,
        "source_sha256": {name: {"ours": sha(REPO/name), "native": sha(NATIVE_REPO/name)}
                          for name in sources},
        "patch_selection_count": historical["patch_selection"]["selected_count"],
        "patch_selection_total": historical["patch_selection"]["total_patches"],
    }
    manifest["checks"] = {
        "input_hash_matches_history": manifest["input_sha256"] == manifest["historical_input_sha256"],
        "checkpoint_hash_matches_both": (manifest["checkpoint_sha256"] ==
                                          manifest["historical_checkpoint_sha256"] ==
                                          manifest["baseline_checkpoint_sha256"]),
        "base_model_and_heads_byte_identical": all(v["ours"] == v["native"]
                                                    for v in manifest["source_sha256"].values()),
        "all_patches_selected": manifest["patch_selection_count"] == manifest["patch_selection_total"],
    }
    write(out/"source_manifest.json", manifest)
    if not all(manifest["checks"].values()):
        raise AssertionError(f"source mismatch: {manifest['checks']}")
    return manifest


@torch.inference_mode()
def run(out):
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    from vggt.v7.model import WindowReconstructor
    from experiments.ours_v7.diagnostic_key_cache import cached_sdpa_keys
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise RuntimeError("diagnostic expects checked idle H20 GPU 4")
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.set_num_threads(4)
    apply_backend_profile("native_vggt", torch)
    historical = json.loads((HISTORY/"formal_w60_o10_patch100_1"/"run_manifest.json").read_text())
    source = source_manifest(out, historical, snapshot(torch))
    saved = torch.load(source["inputs"], map_location="cpu", weights_only=True)
    images_cpu = saved["images"][:100]
    ids = saved["frame_ids"][:100]
    if ids != source["frame_ids"] or list(images_cpu.shape) != [100, 3, 392, 518]:
        raise AssertionError("fixed frame IDs or tensor shape changed")
    model = VGGT().eval().requires_grad_(False)
    model.load_state_dict(load_file(source["checkpoint"]), strict=True)
    model.cuda()
    image_gpu = images_cpu[None].cuda()
    capture = Capture(model.aggregator)
    result = {"status": "incomplete", "tolerance_information": {"atol": ATOL, "rtol": RTOL},
              "input_shape": list(images_cpu.shape), "frame_ids_match": True,
              "native_reference": {}, "initialization": [], "layer_samples": [],
              "v7_raw_vs_native": [], "v7_raw_vs_history": []}
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            capture.phase = "native"
            direct_gpu = model(image_gpu)
            direct = native_values(direct_gpu, images_cpu.shape[-2:])
            del direct_gpu
            capture.phase = "none"
            manual_gpu = native_manual(model, image_gpu)
            manual = native_values(manual_gpu, images_cpu.shape[-2:])
            del manual_gpu
            for field in FIELDS:
                result["native_reference"][field] = difference(direct[field], manual[field])
            result["initialization"] = initialization(model, images_cpu)
            capture.phase = "v7"
            with cached_sdpa_keys() as cache_stats, record_v7_globals(capture):
                v7 = WindowReconstructor(model)(
                    images_cpu, ids, mode="camera_patch_exchange",
                    window_size=60, overlap=10, query_chunk_size=64,
                    patch_exchange_ratio=1.0, dense_head_frame_chunk=None,
                    exchange_sdpa_backend="flash", local_query_chunk_size="all",
                    cross_query_chunk_size=64, reuse_image_encoding=True)
            capture.phase = "none"
            result["key_cache_stats"] = cache_stats
            result["v7_global_layers_observed"] = capture.v7_global_layer_count
            if capture.v7_global_layer_count != model.aggregator.depth:
                raise AssertionError("diagnostic missed v7 global layers")
            result["layer_samples"] = capture.rows
            for window, (lo, hi) in enumerate(v7["windows"]):
                pred = v7["predictions"][window]
                path = HISTORY/f"formal_w60_o10_patch100_1"/"windows"/f"{window:04d}"/"local.npz"
                with np.load(path, allow_pickle=False) as old:
                    for field in FIELDS:
                        result["v7_raw_vs_native"].append({"window": window,
                            "field": field, **difference(direct[field][lo:hi], pred[field])})
                        result["v7_raw_vs_history"].append({"window": window,
                            "field": field, **difference(torch.from_numpy(old[field]), pred[field])})
                if pred["frame_ids"] != ids[lo:hi]:
                    raise AssertionError("v7 frame ID mapping mismatch")
        result["native_reference_all_exact"] = all(x.get("exact") for x in
                                                   result["native_reference"].values())
        result["v7_history_all_exact"] = all(x.get("exact") for x in result["v7_raw_vs_history"])
        result["all_finite"] = all(x.get("finite", False) for group in
            (result["native_reference"].values(), result["initialization"],
             result["layer_samples"], result["v7_raw_vs_native"], result["v7_raw_vs_history"])
            for x in group)
        result["status"] = "pass" if result["all_finite"] else "fail"
    finally:
        capture.close()
        write(out/"native_equivalence_results.json", result)
        divergence = {
            "native_reference_all_exact": result.get("native_reference_all_exact"),
            "initialization_first_nonexact": next((r for r in result["initialization"]
                                                    if not r.get("exact")), None),
            "first_sampled_frame_attention_nonexact": next((r for r in result["layer_samples"]
                if r["stage"] == "frame" and not r.get("exact")), None),
            "first_sampled_global_attention_nonexact": next((r for r in result["layer_samples"]
                if r["stage"] == "global" and not r.get("exact")), None),
            "sampling_scope": {"frames": SAMPLE_FRAMES, "tokens": SAMPLE_TOKENS,
                "note": "Layer max/mean/L2 are for selected tokens, not full activations; initialization and head comparisons use complete tensors."},
            "v7_history_all_exact": result.get("v7_history_all_exact"),
            "status": result["status"],
        }
        write(out/"first_divergence.json", divergence)
    print(json.dumps({"status": result["status"],
                      "native_reference_all_exact": result.get("native_reference_all_exact"),
                      "v7_history_all_exact": result.get("v7_history_all_exact"),
                      "initialization_first_nonexact": divergence["initialization_first_nonexact"],
                      "first_sampled_frame": divergence["first_sampled_frame_attention_nonexact"],
                      "first_sampled_global": divergence["first_sampled_global_attention_nonexact"]},
                     indent=2, default=str), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_dir():
        raise SystemExit("output must be an existing unique directory")
    raise SystemExit(run(args.output))

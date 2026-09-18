"""Child-only native adapters. External source files are read, never edited.

Model load and native output conversions were audited from the standalone ScanNet
runtime, then adapted here to separate load from the measured inference call.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

from . import resolve_config
from .common import (BackendPrediction, checked_load, load_state, w2c_to_c2w,
                     extract_long_chunks, extract_slam_submaps, stage_images)


def install_source(root, namespace, package_root=None):
    root = Path(root).resolve()
    expected = Path(package_root or root / namespace).resolve()
    for name, module in list(sys.modules.items()):
        if name == namespace or name.startswith(namespace + "."):
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(expected):
                raise RuntimeError(f"namespace collision: {name} from {filename}; fresh child required")
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()


def install_backend_sources(key, config):
    root = Path(config["project_root"])
    if key in ("vggt", "vggt_star"):
        install_source(root, "vggt")
    elif key == "streamvggt":
        install_source(root / "src", "streamvggt")
    elif key == "vggt_omega":
        install_source(root, "vggt_omega")
    elif key == "vggt_long":
        sys.path.insert(0, config["dependency_path"])
        install_source(root, "base_models")
        install_source(root / "base_models", "vggt")
    elif key == "vggt_slam":
        install_source(root, "vggt_slam")
        install_source(root / "third_party/salad", "salad")
        install_source(root / "third_party/vggt", "vggt")


def as_numpy(value):
    return value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def depth_points(points, depth, confidence, threshold):
    points, depth, confidence = np.asarray(points), np.asarray(depth), np.asarray(confidence)
    if depth.shape == points.shape[:-1] + (1,):
        depth = depth[..., 0]
    if points.shape[:-1] != depth.shape or confidence.shape != depth.shape:
        raise ValueError("depth/points/confidence shapes disagree")
    mask = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence >= threshold)
    return points[mask], {"confidence_rule": "depth_conf >= threshold; finite confidence, positive finite depth",
                         "depth_conf_thresh": float(threshold), "depth_or_confidence_removed": int((~mask).sum())}


@contextmanager
def offline_hub(source, mode, torch_home=None):
    """Permit only audited local DINO loading and refuse accidental downloads."""
    import torch
    old, old_home = torch.hub.load, os.environ.get("TORCH_HOME")
    if torch_home:
        os.environ["TORCH_HOME"] = str(torch_home)
    def load(repo_or_dir, model, *args, **kwargs):
        if mode == "long" and repo_or_dir == "./LoopModels/dinov2":
            repo_or_dir = str(Path(source) / "LoopModels/dinov2")
            kwargs.update(source="local", pretrained=False)
        elif mode == "slam" and repo_or_dir == "facebookresearch/dinov2":
            repo_or_dir = str(Path(torch_home) / "hub/facebookresearch_dinov2_main")
            kwargs.update(source="local", pretrained=False)
        elif kwargs.get("source") != "local":
            raise RuntimeError(f"unexpected online torch.hub request: {repo_or_dir}")
        return old(repo_or_dir, model, *args, **kwargs)
    torch.hub.load = load
    try:
        yield
    finally:
        torch.hub.load = old
        if old_home is None:
            os.environ.pop("TORCH_HOME", None)
        else:
            os.environ["TORCH_HOME"] = old_home


def preload_long_models(native):
    """Replace only this run's load methods after successful complete preloading."""
    native.model.load()
    native.model.load = lambda: None
    detector = native.loop_detector
    detector.load_model()
    detector.load_model = lambda: (detector.model, detector.device)


class _Backend:
    def __init__(self, key, config, device, work_dir):
        self.key, self.config = key, resolve_config(key, config)
        self.device, self.work_dir = str(device), Path(work_dir)
        self.weights = {}
        self.loaded = False
        self.source_revision = None

    def _begin_load(self):
        install_backend_sources(self.key, self.config)
        # Metadata I/O is outside the synchronized inference boundary.
        revision = subprocess.run(["git", "-C", self.config["project_root"], "rev-parse", "HEAD"],
                                  text=True, capture_output=True, timeout=15)
        self.source_revision = revision.stdout.strip() if revision.returncode == 0 else None

    def _prediction(self, ids, poses, points, metadata):
        if points is not None:
            points = np.asarray(points)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("native world points must be (M,3)")
            finite = np.isfinite(points).all(axis=1)
            metadata["invalid_points_removed"] = int((~finite).sum())
            points = points[finite]
            metadata["points_before_cap"] = len(points)
            cap = self.config["max_points"]
            if cap and len(points) > cap:
                points = points[np.random.RandomState(33).choice(len(points), cap, replace=False)]
        metadata.update(pose_convention="c2w", pose_scale="rigid", model_key=self.key,
            source_root=self.config["project_root"], source_revision=self.source_revision,
            checkpoint=self.config["checkpoint"], resolved_config=self.config, python=sys.executable,
            weights=self.weights, parameter_dtype="float32", autocast_dtype="bfloat16",
            timing_scope="preprocessing, forward, native reconstruction, stitching and loop closure; excludes load and metrics",
            peak_memory_scope="reset after all models loaded; resident models plus inference peak")
        return BackendPrediction(tuple(ids), poses, points, metadata)


class VGGTBackend(_Backend):
    def load(self):
        if self.loaded:
            return
        import torch
        self._begin_load()
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        model = VGGT()
        self.weights = checked_load(model, load_state(self.config["checkpoint"]))
        self.model = model.eval().to(self.device)
        self.preprocess, self.decode, self.unproject = load_and_preprocess_images, pose_encoding_to_extri_intri, unproject_depth_map_to_point_map
        self.loaded = True

    def infer(self, frame_ids, image_paths):
        import torch
        images = self.preprocess([str(p) for p in image_paths]).to(self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = self.model(images)
        extrinsic, intrinsic = self.decode(predictions["pose_enc"].float(), images.shape[-2:])
        ext, intr = as_numpy(extrinsic)[0], as_numpy(intrinsic)[0]
        depth, conf = as_numpy(predictions["depth"])[0], as_numpy(predictions["depth_conf"])[0]
        points, metadata = depth_points(self.unproject(depth, ext, intr), depth, conf, self.config["depth_conf_thresh"])
        metadata.update(native_api="VGGT.forward", camera_decode_dtype="float32")
        return self._prediction(frame_ids, w2c_to_c2w(ext), points, metadata)


class VGGTStarBackend(VGGTBackend):
    """Starred VGGT uses its own configured repository in a fresh child."""


class StreamVGGTBackend(_Backend):
    def load(self):
        if self.loaded:
            return
        self._begin_load()
        from streamvggt.models.streamvggt import StreamVGGT
        from streamvggt.utils.load_fn import load_and_preprocess_images
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
        from streamvggt.utils.geometry import unproject_depth_map_to_point_map
        model = StreamVGGT()
        self.weights = checked_load(model, load_state(self.config["checkpoint"]))
        self.model = model.eval().to(self.device)
        self.preprocess, self.decode, self.unproject = load_and_preprocess_images, pose_encoding_to_extri_intri, unproject_depth_map_to_point_map
        self.loaded = True

    def infer(self, frame_ids, image_paths):
        import torch
        images = self.preprocess([str(p) for p in image_paths]).to(self.device)
        frames = [{"img": image.unsqueeze(0)} for image in images]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = self.model.inference(frames)
        if len(result.ress) != len(frame_ids):
            raise ValueError("Stream native inference omitted frames")
        pose = torch.stack([r["camera_pose"] for r in result.ress], dim=1).float()
        depth = as_numpy(torch.stack([r["depth"] for r in result.ress], dim=1))[0]
        conf = as_numpy(torch.stack([r["depth_conf"] for r in result.ress], dim=1))[0]
        extrinsic, intrinsic = self.decode(pose, images.shape[-2:])
        ext, intr = as_numpy(extrinsic)[0], as_numpy(intrinsic)[0]
        points, metadata = depth_points(self.unproject(depth, ext, intr), depth, conf, self.config["depth_conf_thresh"])
        metadata.update(native_api="StreamVGGT.inference sequential KV cache", camera_decode_dtype="float32")
        return self._prediction(frame_ids, w2c_to_c2w(ext), points, metadata)


class VGGTOmegaBackend(_Backend):
    def load(self):
        if self.loaded:
            return
        import torch
        self._begin_load()
        root = Path(self.config["project_root"])
        spec = importlib.util.spec_from_file_location("_virtual_kitti_native_omega_run", root / "run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        checkpoint = self.config["checkpoint"]
        class SafeTorch:
            def __getattr__(self, key):
                return getattr(torch, key)
            def load(self, path, *args, **kwargs):
                if Path(path).resolve() != Path(checkpoint).resolve():
                    raise ValueError("unexpected Omega checkpoint request")
                return load_state(path)
        module.torch = SafeTorch()
        self.model = module.load_model(checkpoint, self.device)
        self.native = module
        self.weights = {"strict": True, "safe_deserializer": True}
        self.loaded = True

    def infer(self, frame_ids, image_paths):
        predictions = self.native.run_inference(self.model, [str(p) for p in image_paths],
            self.config["image_size"], self.config["image_mode"], self.config["patch_size"], self.device)
        points, metadata = depth_points(predictions["world_points_from_depth"], predictions["depth"],
                                       predictions["depth_conf"], self.config["depth_conf_thresh"])
        metadata["native_api"] = "run.py load_model / run_inference"
        return self._prediction(frame_ids, w2c_to_c2w(predictions["extrinsic"]), points, metadata)


def long_native_config(config, model):
    """Apply and expose every explicit Virtual KITTI native control before load."""
    config["Weights"].update(model="VGGT", VGGT=model["checkpoint"],
        SALAD=model["salad_checkpoint"], DNIO=model["dino_checkpoint"])
    config["Model"].update(chunk_size=model["chunk_size"], overlap=model["overlap"],
        loop_chunk_size=model["loop_chunk_size"], loop_enable=model["loop_closure"],
        useDBoW=False, using_sim3=True, reference_frame_mid=False, calib=False, delete_temp_files=False)
    config["Model"]["IRLS"]["tol"] = "1e-9"
    config["Model"]["Pointcloud_Save"].update(use_confidence_filtering=True,
        conf_threshold_coef=model["conf_threshold_coef"])
    config["Loop"]["SIM3_Optimizer"]["lang_version"] = "python"
    config["Loop"]["SALAD"]["batch_size"] = model["salad_batch_size"]
    return config


class VGGTLongBackend(_Backend):
    def load(self):
        if self.loaded:
            return
        import torch
        import yaml
        self._begin_load()
        import vggt_long
        from base_models.base_model import VGGTAdapter
        from base_models.vggt.models.vggt import VGGT
        from base_models.vggt.utils.load_fn import load_and_preprocess_images
        from base_models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
        root = Path(self.config["project_root"])
        config = yaml.safe_load((root / "configs/base_config.yaml").read_text())
        config = long_native_config(config, self.config)
        self.output = self.work_dir / "native_long"
        self.output.mkdir(exist_ok=False)
        weights = self.weights
        class SafeAdapter(VGGTAdapter):
            def load(adapter):
                model = VGGT()
                weights.update(checked_load(model, load_state(adapter.config["Weights"]["VGGT"])))
                adapter.model = model.eval().to(adapter.device)
            def infer_chunk(adapter, paths):
                # Audited standard-reference native path, with explicit FP32 camera decoding.
                images = load_and_preprocess_images(paths).to(adapter.device)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    predictions = adapter.model(images)
                ext, intr = pose_encoding_to_extri_intri(predictions["pose_enc"].float(), images.shape[-2:])
                bottom = torch.tensor([0, 0, 0, 1], dtype=ext.dtype, device=ext.device).view(1, 1, 1, 4)
                bottom = bottom.expand(ext.shape[0], ext.shape[1], 1, 4)
                return {"world_points": predictions["world_points"], "world_points_conf": predictions["world_points_conf"],
                        "extrinsic": torch.linalg.inv(torch.cat([ext, bottom], dim=2)), "intrinsic": intr,
                        "depth": predictions["depth"], "depth_conf": predictions["depth_conf"],
                        "images": predictions["images"], "mask": None}
        with offline_hub(root, "long"):
            self.native = vggt_long.VGGT_Long(str(self.work_dir / "input_frames"), str(self.output), config)
            self.native.model = SafeAdapter(config, device=self.device)
            preload_long_models(self.native)
        self.native_config = config
        self.loaded = True

    def infer(self, frame_ids, image_paths):
        import torch
        stage_images(frame_ids, image_paths, self.work_dir)
        with torch.inference_mode():
            self.native.run()
        # Native writes these dictionaries in this new child-owned output only.
        chunks = [np.load(Path(self.native.result_unaligned_dir) / f"chunk_{k}.npy",
                          allow_pickle=True).item() for k in range(len(self.native.chunk_indices))]
        points, poses, metadata = extract_long_chunks(chunks, self.native.chunk_indices,
            self.native.sim3_list, len(frame_ids), self.config["conf_threshold_coef"])
        exported = np.loadtxt(self.output / "camera_poses.txt").reshape(-1, 4, 4)
        if not np.allclose(poses, exported, atol=1e-4, rtol=0):
            raise ValueError("native Long exported cameras disagree with chunk ownership")
        metadata.update(native_api="VGGT_Long.run with preloaded VGGT and SALAD", resolved_native_config=self.native_config,
            loop_pairs=len(self.native.loop_list), loop_constraints=len(self.native.loop_sim3_list),
            point_source="unaligned native chunks plus final cumulative Sim3; last chunk owns overlap")
        return self._prediction(frame_ids, poses, points, metadata)


class VGGTSLAMBackend(_Backend):
    def load(self):
        if self.loaded:
            return
        import torch
        self._begin_load()
        import vggt_slam.solver as solver_module
        import vggt.utils.geometry as geometry
        from vggt.models.vggt import VGGT
        root = Path(self.config["project_root"])
        if not Path(geometry.__file__).resolve().is_relative_to((root / "third_party/vggt").resolve()):
            raise RuntimeError("SLAM requires native camera-local geometry")
        model = VGGT()
        self.weights = checked_load(model, load_state(self.config["checkpoint"]))
        model = model.eval().to(self.device)
        class InferenceModel:
            calls = 0
            def __call__(wrapper, *args, **kwargs):
                wrapper.calls += 1
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    predictions = model(*args, **kwargs)
                predictions["pose_enc"] = predictions["pose_enc"].float()
                return predictions
        self.model = InferenceModel()
        class NullViewer:
            def __init__(self, *args, **kwargs):
                pass
        old_viewer = solver_module.Viewer
        solver_module.Viewer = NullViewer
        try:
            with offline_hub(root, "slam", self.config["torch_home"]):
                self.solver = solver_module.Solver(init_conf_threshold=self.config["conf_percentile"],
                                                  lc_thres=self.config["lc_thres"])
        finally:
            solver_module.Viewer = old_viewer
        self.loaded = True

    def infer(self, frame_ids, image_paths):
        _, paths = stage_images(frame_ids, image_paths, self.work_dir)
        optimize_calls = 0
        for offset in range(0, len(paths), self.config["submap_size"]):
            window = paths[offset:offset + self.config["submap_size"] + 1]
            if len(window) == 1 and offset > 0:
                continue
            predictions = self.solver.run_predictions(window, self.model, max_loops=self.config["max_loops"],
                                                       clip_model=None, clip_preprocess=None)
            self.solver.add_points(predictions)
            self.solver.graph.optimize()
            optimize_calls += 1
        points, poses, metadata = extract_slam_submaps(self.solver.map.ordered_submaps_by_key(),
            self.solver.graph, dict(zip(paths, frame_ids)), frame_ids)
        metadata.update(native_api="Solver.run_predictions / add_points / graph.optimize",
            forward_calls=self.model.calls, graph_optimize_calls=optimize_calls,
            accepted_loop_closures=self.solver.graph.get_num_loops(), camera_decode_dtype="float32",
            depth_geometry="native camera-local maps transformed by optimized SL4; native RQ camera decomposition")
        return self._prediction(frame_ids, poses, points, metadata)


ADAPTERS = {"vggt": VGGTBackend, "vggt_star": VGGTStarBackend, "streamvggt": StreamVGGTBackend,
            "vggt_slam": VGGTSLAMBackend, "vggt_long": VGGTLongBackend, "vggt_omega": VGGTOmegaBackend}


def create_backend(model_key, config, device="cuda:0", work_dir=None):
    from . import normalize_model_key
    key = normalize_model_key(model_key)
    if work_dir is None or not Path(work_dir).is_absolute():
        raise ValueError("absolute child work directory required")
    return ADAPTERS[key](key, config, device, work_dir)

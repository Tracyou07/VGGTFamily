import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import pytest


@pytest.mark.parametrize("value,expected", [
    ("VGGT", "vggt"), ("VGGT*", "vggt_star"), ("StreamVGGT", "streamvggt"),
    ("VGGT-SLAM", "vggt_slam"), ("VGGT-Long", "vggt_long"), ("VGGT-Omega", "vggt_omega")])
def test_model_keys_normalize_to_six_native_models(value, expected):
    from kitti_eval.backends import normalize_model_key
    assert normalize_model_key(value) == expected


@pytest.mark.parametrize("value", ["fastvggt", "FastVGGT", "fast_vggt", "long", "unknown"])
def test_unsupported_backends_are_rejected(value):
    from kitti_eval.backends import normalize_model_key
    with pytest.raises(ValueError, match="MODEL"):
        normalize_model_key(value)


def prediction(**changes):
    from kitti_eval.backends.common import BackendPrediction
    values = dict(frame_ids=("000003", "000010"), poses_c2w=np.repeat(np.eye(4)[None], 2, axis=0),
                  world_points=np.array([[1., 2., 3.]]),
                  metadata={"pose_convention": "c2w", "pose_scale": "rigid"})
    values.update(changes)
    return BackendPrediction(**values)


def test_prediction_accepts_rigid_original_ids_and_optional_world_points():
    assert prediction().frame_ids == ("000003", "000010")
    assert prediction(world_points=None).world_points is None


@pytest.mark.parametrize("change", [
    {"metadata": {}}, {"metadata": {"pose_convention": "w2c", "pose_scale": "rigid"}},
    {"metadata": {"pose_convention": "c2w", "pose_scale": "unknown"}},
    {"metadata": {"pose_convention": "c2w", "pose_scale": "rigid", "value": float("inf")}},
    {"poses_c2w": np.eye(4)[None]}, {"poses_c2w": np.full((2, 4, 4), np.nan)},
    {"poses_c2w": np.repeat(np.diag([2., 2., 2., 1.])[None], 2, axis=0)},
    {"world_points": np.array([[1., 2., np.inf]])}, {"world_points": np.ones((1, 4))},
    {"frame_ids": ("000010", "000003")}, {"frame_ids": ("3", "10")}])
def test_prediction_rejects_ambiguous_or_invalid_geometry(change):
    with pytest.raises(ValueError):
        prediction(**change)


def test_requested_frame_order_must_match():
    from kitti_eval.backends.common import validate_prediction
    with pytest.raises(ValueError, match="FRAME"):
        validate_prediction(prediction(), ("000003", "000009"))


def test_w2c_conversion_is_explicit_and_does_not_absorb_scale():
    from kitti_eval.backends.common import w2c_to_c2w
    extrinsic = np.array([[[0., -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, 4]]])
    result = w2c_to_c2w(extrinsic)
    np.testing.assert_allclose(result[0, :3, 3], [-3, 2, -4])
    with pytest.raises(ValueError):
        w2c_to_c2w(extrinsic * 2)


def test_importing_parent_api_never_imports_model_or_torch():
    code = """
import builtins
old = builtins.__import__
def guarded(name, *a, **k):
    if name.split('.')[0] in {'torch', 'vggt', 'base_models', 'streamvggt', 'vggt_slam', 'vggt_long', 'vggt_omega'}:
        raise AssertionError(name)
    return old(name, *a, **k)
builtins.__import__ = guarded
import kitti_eval.backends
import kitti_eval.backend_worker
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_config_loads_exact_six_models_and_immutable_nested_controls(kitti_fixture):
    from kitti_eval.config import load_config
    path = Path(__file__).parents[1] / "configs/h20.json"
    config = load_config(path)
    assert set(config.models) == {"vggt", "vggt_star", "streamvggt", "vggt_slam", "vggt_long", "vggt_omega"}
    long = config.models["vggt_long"]
    assert (long["chunk_size"], long["overlap"], long["loop_chunk_size"]) == (75, 30, 20)
    assert long["loop_closure"] and long["using_sim3"] and long["retrieval"] == "salad"
    with pytest.raises(TypeError):
        long["chunk_size"] = 60
    assert config.models["vggt_slam"]["interpreter"].endswith("/monst3r/bin/python")
    assert kitti_fixture.config.models == {}


def test_doctor_rejects_missing_files_without_launching_model(tmp_path):
    from kitti_eval.backends import doctor_backend
    result = doctor_backend("vggt", {"interpreter": sys.executable, "project_root": str(tmp_path),
                                    "checkpoint": str(tmp_path / "missing.safetensors"), "image_size": 518})
    assert not result.ready
    assert any("missing.safetensors" in b.message for b in result.blockers)
    assert any("vggt/models/vggt.py" in b.message for b in result.blockers)


def test_checkpoint_validation_catches_truncation_and_lfs_pointer(tmp_path):
    from kitti_eval.backends.common import inspect_checkpoint
    pointer = tmp_path / "model.pt"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 12345\n")
    with pytest.raises(ValueError, match="CHECKPOINT"):
        inspect_checkpoint(pointer)
    bad = tmp_path / "model.safetensors"
    header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    bad.write_bytes(len(header).to_bytes(8, "little") + header + b"1234")
    with pytest.raises(ValueError, match="CHECKPOINT"):
        inspect_checkpoint(bad)
    bad.write_bytes(len(header).to_bytes(8, "little") + header + b"12345678")
    assert inspect_checkpoint(bad)["tensor_count"] == 1


def test_long_reuses_preloaded_model_and_retrieval_inside_timing(monkeypatch):
    from kitti_eval.backends.native import preload_long_models
    from kitti_eval.resources import measure_inference
    events = []
    class Model:
        def load(self): events.append("model_load")
    class Retrieval:
        model, device = None, None
        def load_model(self):
            events.append("retrieval_load")
            self.model, self.device = object(), "cuda:0"
            return self.model, self.device
        def run(self):
            self.load_model()
            events.append("retrieve")
    class Native:
        model, loop_detector = Model(), Retrieval()
        def run(self):
            self.loop_detector.run()
            self.model.load()
            events.extend(["reconstruct", "stitch", "loop"])
    native = Native()
    class Backend:
        def load(self): preload_long_models(native)
        def infer(self, ids, paths):
            native.run()
            return prediction()
    class Cuda:
        def reset_peak_memory_stats(self): events.append("reset")
        def synchronize(self): pass
        def max_memory_allocated(self): return 0
        def max_memory_reserved(self): return 0
    monkeypatch.setattr("kitti_eval.resources._smi", lambda label: {})
    measure_inference(Backend(), ("000003", "000010"), (Path("a"), Path("b")), Cuda(), lambda: 0.)
    assert events == ["model_load", "retrieval_load", "reset", "retrieve", "reconstruct", "stitch", "loop"]


def test_long_chunk_scale_transforms_translation_but_not_rotation():
    from kitti_eval.backends.common import extract_long_chunks
    chunks = []
    for _ in range(2):
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[:, :3, 3] = [[1, 0, 0], [2, 0, 0]]
        chunks.append({"world_points": np.ones((2, 1, 3)), "world_points_conf": np.ones((2, 1)),
                       "extrinsic": poses})
    points, poses, metadata = extract_long_chunks(chunks, [(0, 2), (1, 3)],
        [(2., np.eye(3), np.array([5., 0, 0]))], 3)
    np.testing.assert_allclose(poses[:, :3, 3], [[1, 0, 0], [7, 0, 0], [9, 0, 0]])
    np.testing.assert_allclose(poses[:, :3, :3], np.repeat(np.eye(3)[None], 3, axis=0))
    assert metadata["frame_owner"] == [0, 1, 1]

def test_doctor_uses_configured_interpreter_and_blocks_import_writes(tmp_path):
    from kitti_eval.backends import doctor_backend
    root = tmp_path / "project"
    model_dir = root / "vggt/models"
    model_dir.mkdir(parents=True)
    marker = tmp_path / "forbidden-model-write"
    (root / "vggt/__init__.py").write_text("")
    (model_dir / "__init__.py").write_text("")
    (model_dir / "vggt.py").write_text(f"open({str(marker)!r}, 'w').write('modified')")
    checkpoint = tmp_path / "weights.safetensors"
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    checkpoint.write_bytes(len(header).to_bytes(8, "little") + header + b"1234")
    status = doctor_backend("vggt", {"interpreter": sys.executable, "project_root": str(root),
                                   "checkpoint": str(checkpoint), "image_size": 518})
    assert not marker.exists(), "doctor allowed an imported module to modify external files"
    assert not list(root.rglob("__pycache__"))
    assert not status.ready
    assert any("read-only" in b.message for b in status.blockers)
    assert status.diagnostics["probe"]["python"] == sys.executable

@pytest.mark.parametrize("key", ["vggt", "vggt_star", "streamvggt"])
def test_native_forward_uses_fp32_camera_decode_inside_boundary(key, tmp_path):
    # The real CPU tensor dependency also belongs in a child, preserving parent isolation.
    code = ("import runpy, pytest; from pathlib import Path; "
            f"ns = runpy.run_path({__file__!r}); "
            f"ns['_exercise_native_forward']({key!r}, Path({str(tmp_path)!r}), pytest.MonkeyPatch())")
    completed = subprocess.run([sys.executable, "-B", "-c", code], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr


def _exercise_native_forward(key, tmp_path, monkeypatch):
    import types
    from contextlib import nullcontext
    import torch
    from kitti_eval.backends.native import create_backend
    from kitti_eval.resources import measure_inference
    events = []
    namespace = "streamvggt" if key == "streamvggt" else "vggt"
    def module(name, **attrs):
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))
    def preprocess(paths):
        assert paths == [str(tmp_path / "000003.png"), str(tmp_path / "000010.png")]
        events.append("preprocess")
        return torch.zeros(2, 3, 2, 2)
    def decode(pose, size):
        assert pose.dtype == torch.float32 and size == (2, 2)
        events.append("decode")
        ext = torch.eye(4)[:3].repeat(1, 2, 1, 1)
        ext[0, :, 0, 3] = torch.tensor([1., 2.])
        return ext, torch.eye(3).repeat(1, 2, 1, 1)
    def unproject(depth, ext, intr):
        events.append("unproject")
        return np.zeros((2, 2, 2, 3))
    class Model:
        def __init__(self): events.append("construct")
        def eval(self): return self
        def to(self, device):
            assert device == "cuda:0"
            return self
        def __call__(self, images):
            events.append("forward")
            return {"pose_enc": torch.zeros(1, 2, 9, dtype=torch.bfloat16),
                    "depth": torch.ones(1, 2, 2, 2, 1), "depth_conf": torch.ones(1, 2, 2, 2) * 2}
        def inference(self, frames):
            assert len(frames) == 2 and frames[0]["img"].shape == (1, 3, 2, 2)
            events.append("stream")
            return types.SimpleNamespace(ress=[{"camera_pose": torch.zeros(1, 9, dtype=torch.bfloat16),
                "depth": torch.ones(1, 2, 2, 1), "depth_conf": torch.ones(1, 2, 2) * 2} for _ in frames])
    module(namespace + ".models." + ("streamvggt" if key == "streamvggt" else "vggt"),
           **({"StreamVGGT": Model} if key == "streamvggt" else {"VGGT": Model}))
    module(namespace + ".utils.load_fn", load_and_preprocess_images=preprocess)
    module(namespace + ".utils.pose_enc", pose_encoding_to_extri_intri=decode)
    module(namespace + ".utils.geometry", unproject_depth_map_to_point_map=unproject)
    monkeypatch.setattr("kitti_eval.backends.native._Backend._begin_load", lambda self: None)
    monkeypatch.setattr("kitti_eval.backends.native.load_state", lambda path: {"weight": 1})
    def checked(model, state):
        events.append("weights")
        return {"loaded_keys": 1}
    monkeypatch.setattr("kitti_eval.backends.native.checked_load", checked)
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)
    monkeypatch.setattr("kitti_eval.resources._smi", lambda label: {})
    class Cuda:
        def reset_peak_memory_stats(self): events.append("reset")
        def synchronize(self): events.append("sync")
        def max_memory_allocated(self): return 0
        def max_memory_reserved(self): return 0
    backend = create_backend(key, {"interpreter": sys.executable, "project_root": str(tmp_path),
                                   "checkpoint": str(tmp_path / "weights")}, work_dir=tmp_path)
    prediction, _ = measure_inference(backend, ("000003", "000010"),
        (tmp_path / "000003.png", tmp_path / "000010.png"), Cuda(), lambda: 0.)
    assert events == ["construct", "weights", "reset", "sync", "preprocess",
                      "stream" if key == "streamvggt" else "forward", "decode", "unproject", "sync"]
    np.testing.assert_allclose(prediction.poses_c2w[:, :3, 3], [[-1, 0, 0], [-2, 0, 0]])
    assert prediction.world_points.shape == (8, 3)
    backend.load()
    assert events.count("construct") == 1


def test_native_namespace_collision_is_rejected_before_path_install(tmp_path, monkeypatch):
    import types
    from kitti_eval.backends.native import install_source
    monkeypatch.setitem(sys.modules, "vggt", types.SimpleNamespace(__file__="/unrelated/vggt/__init__.py"))
    with pytest.raises(RuntimeError, match="namespace collision"):
        install_source(tmp_path, "vggt")

@pytest.mark.parametrize("operation,ready", [("scratch", True), ("cuda", False), ("network", False)])
def test_doctor_allows_disposable_temporary_files_but_never_cuda_or_network(tmp_path, operation, ready):
    from kitti_eval.backends import doctor_backend
    root = tmp_path / "project"
    model_dir = root / "vggt/models"
    utils = root / "vggt/utils"
    model_dir.mkdir(parents=True)
    utils.mkdir()
    for p in (root / "vggt/__init__.py", model_dir / "__init__.py", utils / "__init__.py",
              utils / "load_fn.py", utils / "pose_enc.py", utils / "geometry.py"):
        p.write_text("")
    code = {"scratch": "import tempfile\nwith tempfile.TemporaryFile() as f: f.write(b'ok')\n",
            "cuda": "import torch\ntorch.cuda.init()\n",
            "network": "import socket\nsocket.create_connection(('example.invalid', 443))\n"}[operation]
    (model_dir / "vggt.py").write_text(code)
    checkpoint = tmp_path / "model.safetensors"
    header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    checkpoint.write_bytes(len(header).to_bytes(8, "little") + header + b"1234")
    status = doctor_backend("vggt", {"interpreter": sys.executable, "project_root": str(root),
                                    "checkpoint": str(checkpoint)})
    assert status.ready == ready, status.blockers
    if not ready:
        assert any("forbid" in b.message for b in status.blockers)
    else:
        assert not Path(status.diagnostics["probe"]["scratch_directory"]).exists()


@pytest.mark.parametrize("field", ["poses_c2w", "world_points"])
@pytest.mark.parametrize("imaginary", [0., 2., float("nan"), float("inf")])
def test_prediction_rejects_complex_arrays_before_real_conversion(field, imaginary):
    import warnings
    value = (np.repeat(np.eye(4)[None], 2, axis=0) if field == "poses_c2w"
             else np.array([[1., 2., 3.]])).astype(np.complex128)
    value.imag.flat[0] = imaginary
    with warnings.catch_warnings(record=True) as recorded:
        with pytest.raises(ValueError, match="BACKEND_"):
            prediction(**{field: value})
    assert not recorded, "validation must reject before a lossy ComplexWarning cast"


@pytest.mark.parametrize("field", ["poses_c2w", "world_points"])
@pytest.mark.parametrize("dtype", [str, object, bool])
def test_prediction_rejects_non_real_numeric_array_dtypes(field, dtype):
    value = (np.repeat(np.eye(4)[None], 2, axis=0) if field == "poses_c2w"
             else np.array([[1., 2., 3.]])).astype(dtype)
    with pytest.raises(ValueError, match="BACKEND_"):
        prediction(**{field: value})


def test_w2c_conversion_rejects_complex_input_before_homogeneous_assignment():
    import warnings
    from kitti_eval.backends.common import w2c_to_c2w
    value = np.eye(4, dtype=np.complex128)[None]
    value.imag[0, 0, 0] = float("nan")
    with warnings.catch_warnings(record=True) as recorded:
        with pytest.raises(ValueError, match="BACKEND_"):
            w2c_to_c2w(value)
    assert not recorded

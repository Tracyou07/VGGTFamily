import dataclasses
import json
import os
from pathlib import Path
import signal
import sys
import numpy as np
import pytest


@pytest.fixture
def fake_request(tmp_path):
    from kitti_eval.backends.common import BackendRequest
    from PIL import Image
    image = tmp_path / "000007.png"
    Image.new("RGB", (4, 3)).save(image)
    # A real child interpreter facade: no model dependency and no CUDA calls.
    child = tmp_path / "python"
    child.write_text("#!" + sys.executable + "\n" + r"""
import hashlib, json, os, pathlib, signal, struct, sys, time, zipfile
import numpy as np
p = json.loads(pathlib.Path(sys.argv[-1]).read_text())
out = pathlib.Path(p['output_dir'])
mode = p['model_config'].get('fixture_mode', 'success')
if mode == 'timeout': time.sleep(30)
if mode == 'signal': os.kill(os.getpid(), signal.SIGTERM)
if mode == 'oom':
    print('torch.OutOfMemoryError: CUDA out of memory', file=sys.stderr)
    sys.exit(1)
if mode == 'missing': sys.exit(0)
ids = p['frame_ids']
prediction = out / 'prediction.npz'
with prediction.open('wb') as f:
    saver = np.savez_compressed if mode == 'compressed_corrupt' else np.savez
    saver(f, frame_ids=np.asarray(ids), poses_c2w=np.repeat(np.eye(4)[None], len(ids), axis=0))
if mode == 'compressed_corrupt':
    with zipfile.ZipFile(prediction) as archive:
        member = archive.getinfo('frame_ids.npy')
    payload = bytearray(prediction.read_bytes())
    name_size, extra_size = struct.unpack_from('<HH', payload, member.header_offset + 26)
    offset = member.header_offset + 30 + name_size + extra_size
    payload[offset] = 0x07  # DEFLATE BTYPE=3 is reserved: guaranteed zlib.error.
    prediction.write_bytes(payload)
if mode == 'corrupt': prediction.write_bytes(b'not an npz')
result = {'schema_version': 1, 'status': 'success', 'failure_code': None, 'message': '',
    'request_id': p['request_id'], 'provenance_id': p['provenance_id'], 'model_key': p['model_key'],
    'sequence': p['sequence'], 'prediction_sha256': hashlib.sha256(prediction.read_bytes()).hexdigest(),
    'metadata': {'pose_convention': 'c2w', 'pose_scale': 'rigid'},
    'resources': {'inference_seconds': 2.5, 'peak_allocated_mib': 128., 'peak_reserved_mib': 192.,
                  'nvidia_smi_before': {}, 'nvidia_smi_after': {}}}
if mode == 'wrong_request': result['request_id'] = 'stale'
if mode == 'wrong_provenance': result['provenance_id'] = 'stale'
if mode == 'hash_mismatch': result['prediction_sha256'] = '0' * 64
if mode == 'invalid_resources': result['resources']['inference_seconds'] = -1
if mode == 'failure_envelope':
    result.update(status='error', failure_code='NATIVE_ERROR')
(out / 'worker_result.json').write_text(json.dumps(result))
if mode == 'nonzero': sys.exit(3)
""")
    child.chmod(0o755)
    return BackendRequest(model_key="vggt", model_config={"interpreter": str(child),
        "project_root": str(tmp_path), "checkpoint": str(tmp_path / "weights"), "image_size": 518},
        frame_ids=("000007",), image_paths=(image,), output_dir=tmp_path / "output",
        request_id="req-1", provenance_id="a" * 64, sequence="00", device="cuda:0")


def test_request_keeps_prepared_original_frame_ids_and_decoded_paths(kitti_fixture, tmp_path):
    from kitti_eval.backends.common import BackendRequest
    from kitti_eval.data import prepare_sequence
    prepared = prepare_sequence(kitti_fixture.config, "00")
    request = BackendRequest.from_prepared(prepared, model_key="VGGT", model_config={
        "interpreter": sys.executable, "project_root": str(tmp_path), "checkpoint": str(tmp_path / "weights"),
        "image_size": 518}, output_dir=tmp_path / "work", provenance_id="f" * 64)
    assert request.frame_ids == ("000000", "000001")
    assert request.image_paths == prepared.image_paths
    assert request.sequence == "00"


def test_request_freezes_model_config_and_round_trips(fake_request):
    from kitti_eval.backends.common import BackendRequest
    with pytest.raises(TypeError):
        fake_request.model_config["image_size"] = 224
    cloned = BackendRequest.from_dict(fake_request.to_dict())
    assert cloned.to_dict() == fake_request.to_dict()


def test_parent_launches_configured_interpreter_and_validates_output(fake_request):
    from kitti_eval.backend_worker import launch_worker
    result = launch_worker(fake_request, timeout_s=10)
    assert result.status == "success", result.message
    assert result.prediction.frame_ids == ("000007",)
    assert result.usage.peak_reserved_mib == 192
    assert result.returncode == 0 and result.signal is None
    payload = json.loads((fake_request.output_dir / "request.json").read_text())
    assert payload["request_id"] == "req-1"
    assert payload["provenance_id"] == "a" * 64


@pytest.mark.parametrize("mode,status,code", [
    ("timeout", "timeout", "BACKEND_TIMEOUT"), ("signal", "error", "BACKEND_SIGNAL"),
    ("oom", "oom", "BACKEND_OOM"), ("missing", "error", "BACKEND_OUTPUT_MISSING"),
    ("corrupt", "error", "BACKEND_OUTPUT_INVALID"), ("wrong_request", "error", "BACKEND_ID_MISMATCH"),
    ("wrong_provenance", "error", "BACKEND_ID_MISMATCH"), ("hash_mismatch", "error", "BACKEND_OUTPUT_INVALID"),
    ("nonzero", "error", "BACKEND_EXIT"), ("invalid_resources", "error", "BACKEND_OUTPUT_INVALID"),
    ("failure_envelope", "error", "BACKEND_OUTPUT_INVALID")])
def test_parent_classifies_child_failures(fake_request, mode, status, code):
    from kitti_eval.backend_worker import launch_worker
    request = dataclasses.replace(fake_request, model_config={**fake_request.model_config, "fixture_mode": mode})
    result = launch_worker(request, timeout_s=.01 if mode == "timeout" else 10.)
    assert result.status == status
    assert result.failure_code == code, result.message
    assert result.prediction is None and result.usage is None
    if mode == "signal": assert result.signal == signal.SIGTERM


def test_existing_worker_artifacts_are_not_accepted_as_new_success(fake_request):
    from kitti_eval.backend_worker import launch_worker
    assert launch_worker(fake_request, timeout_s=10).status == "success"
    again = launch_worker(fake_request, timeout_s=10)
    assert again.status == "error" and again.failure_code == "BACKEND_OUTPUT_EXISTS"


def test_worker_main_writes_atomic_prediction_and_resource_envelope(fake_request, monkeypatch):
    import types
    from kitti_eval import backend_worker
    from kitti_eval.backends.common import BackendPrediction
    from kitti_eval.results import atomic_write_json, read_json
    events = []
    class Backend:
        def load(self): events.append("load")
        def infer(self, ids, paths):
            events.append("infer")
            return BackendPrediction(ids, np.eye(4)[None], None,
                                     {"pose_convention": "c2w", "pose_scale": "rigid"})
    class Cuda:
        def set_device(self, device): events.append("set_device")
        def reset_peak_memory_stats(self): events.append("reset")
        def synchronize(self): events.append("sync")
        def max_memory_allocated(self): return 8 * 2**20
        def max_memory_reserved(self): return 16 * 2**20
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=Cuda()))
    monkeypatch.setattr("kitti_eval.backends.native.create_backend", lambda *a, **k: Backend())
    monkeypatch.setattr("kitti_eval.resources._smi", lambda label: {"label": label})
    fake_request.output_dir.mkdir()
    request_file = fake_request.output_dir / "request.json"
    atomic_write_json(request_file, fake_request.to_dict())
    assert backend_worker.main(["--request", str(request_file)]) == 0
    record = read_json(fake_request.output_dir / "worker_result.json")
    assert record["request_id"] == fake_request.request_id
    assert record["resources"]["peak_allocated_mib"] == 8
    assert events == ["set_device", "load", "reset", "sync", "infer", "sync"]
    with np.load(fake_request.output_dir / "prediction.npz", allow_pickle=False) as saved:
        assert saved["frame_ids"].tolist() == ["000007"]
    assert not list(fake_request.output_dir.glob("*.tmp"))


def test_worker_main_serializes_oom_exception(fake_request, monkeypatch):
    import types
    from kitti_eval import backend_worker
    from kitti_eval.results import atomic_write_json, read_json
    class Backend:
        def load(self): raise RuntimeError("CUDA out of memory")
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(set_device=lambda x: None)))
    monkeypatch.setattr("kitti_eval.backends.native.create_backend", lambda *a, **k: Backend())
    fake_request.output_dir.mkdir()
    path = fake_request.output_dir / "request.json"
    atomic_write_json(path, fake_request.to_dict())
    assert backend_worker.main(["--request", str(path)]) == 1
    saved = read_json(fake_request.output_dir / "worker_result.json")
    assert saved["status"] == "oom" and saved["failure_code"] == "BACKEND_OOM"
    assert not (fake_request.output_dir / "prediction.npz").exists()

def test_request_rejects_image_frame_identity_mismatch(fake_request, tmp_path):
    from PIL import Image
    wrong = tmp_path / "000008.png"
    Image.new("RGB", (4, 3)).save(wrong)
    with pytest.raises(ValueError, match="INPUT"):
        dataclasses.replace(fake_request, image_paths=(wrong,))


def test_corrupt_zip_is_a_structured_output_failure(fake_request):
    from kitti_eval.backend_worker import launch_worker, _load_success
    import hashlib
    result = launch_worker(fake_request, timeout_s=10)
    assert result.status == "success"
    path = fake_request.output_dir / "prediction.npz"
    data = path.read_bytes()
    path.write_bytes(data[:-20])
    record_path = fake_request.output_dir / "worker_result.json"
    record = json.loads(record_path.read_text())
    record["prediction_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record_path.write_text(json.dumps(record))
    result = _load_success(fake_request, 0)
    assert result.status == "error" and result.failure_code == "BACKEND_OUTPUT_INVALID"

def test_worker_redirects_native_caches_out_of_external_repositories(fake_request, monkeypatch):
    import types
    from kitti_eval.backend_worker import execute_request
    from kitti_eval.backends.common import BackendPrediction
    observed = {}
    class Backend:
        def load(self):
            observed.update({key: os.environ.get(key) for key in ("NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME")})
        def infer(self, ids, paths):
            return BackendPrediction(ids, np.eye(4)[None], None, {"pose_convention": "c2w", "pose_scale": "rigid"})
    class Cuda:
        def set_device(self, device): pass
        def reset_peak_memory_stats(self): pass
        def synchronize(self): pass
        def max_memory_allocated(self): return 0
        def max_memory_reserved(self): return 0
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=Cuda()))
    monkeypatch.setattr("kitti_eval.backends.native.create_backend", lambda *a, **k: Backend())
    monkeypatch.setattr("kitti_eval.resources._smi", lambda label: {})
    before = {key: os.environ.get(key) for key in ("NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME")}
    execute_request(fake_request)
    assert all(value and Path(value).is_relative_to(fake_request.output_dir) for value in observed.values())
    assert {key: os.environ.get(key) for key in before} == before


def test_parent_normalizes_damaged_compressed_npz(fake_request):
    from kitti_eval.backend_worker import launch_worker
    request = dataclasses.replace(fake_request,
        model_config={**fake_request.model_config, "fixture_mode": "compressed_corrupt"})
    result = launch_worker(request, timeout_s=10)
    assert result.status == "error" and result.failure_code == "BACKEND_OUTPUT_INVALID"
    assert result.prediction is None and result.usage is None
    assert result.returncode == 0

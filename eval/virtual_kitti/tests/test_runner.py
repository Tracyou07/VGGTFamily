import json
import os
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest

@pytest.fixture
def run_fixture(tmp_path, monkeypatch):
    from virtual_kitti_eval.config import load_config
    from virtual_kitti_eval.data import prepare_sequence
    from .conftest import make_fixture, MAIN
    f = make_fixture(tmp_path,MAIN)
    ids = MAIN
    model = tmp_path / "model"
    (model / "vggt/models").mkdir(parents=True)
    (model / "vggt/models/vggt.py").write_text("# fixture\n")
    checkpoint = tmp_path / "weights.safetensors"
    header = json.dumps({"w": {"dtype":"F32", "shape":[1], "data_offsets":[0,4]}}).encode()
    checkpoint.write_bytes(len(header).to_bytes(8,"little")+header+bytes(4))
    child = tmp_path / "python-fixture"
    child.write_text("#!"+sys.executable+"\n"+r"""
import hashlib, json, os, pathlib, sys
import numpy as np
assert 'torch' not in sys.modules
if '--version' in sys.argv:
    print('Python fixture 1')
    sys.exit(0)
if '--doctor' in sys.argv:
    print(json.dumps({'blockers': [], 'diagnostics': {'torch_loaded': False}}))
    sys.exit(0)
p = json.loads(pathlib.Path(sys.argv[-1]).read_text())
out = pathlib.Path(p['output_dir'])
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
if p['sequence'] == p['model_config'].get('fixture_fail'):
    print('fixture worker failed', file=sys.stderr)
    sys.exit(3)
poses = np.repeat(np.eye(4)[None], 4, axis=0)
poses[:, :3, 3] = [[0,0,0], [1,0,0], [0,2,0], [0,0,3]]
with (out / 'prediction.npz').open('wb') as f:
    np.savez(f, frame_ids=np.asarray(p['frame_ids']), poses_c2w=poses)
result = dict(schema_version=1, status='success', failure_code=None, message='',
    request_id=p['request_id'], provenance_id=p['provenance_id'], model_key=p['model_key'],
    sequence=p['sequence'], prediction_sha256=hashlib.sha256((out/'prediction.npz').read_bytes()).hexdigest(),
    metadata=dict(pose_convention='c2w', pose_scale='rigid', torch_loaded=False),
    resources=dict(inference_seconds=2.5, peak_allocated_mib=0., peak_reserved_mib=0.,
                   nvidia_smi_before={}, nvidia_smi_after={}))
(out / 'worker_result.json').write_text(json.dumps(result))
""")
    child.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    smi.write_text("#!/bin/sh\ncase \"$1\" in\n *query-compute-apps*) exit 0;;\n *memory.total*) printf '7, GPU-fixture-seven, 97871, 90000\\n';;\n *) printf '7, GPU-fixture-seven\\n';;\nesac\n")
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir)+os.pathsep+os.environ["PATH"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    values = json.loads(f.config_path.read_text())
    values["models"] = {"vggt": dict(project_root=str(model), checkpoint=str(checkpoint), interpreter=str(child))}
    path = f.config_path; path.write_text(json.dumps(values))
    config = load_config(path)
    for seq in ids: prepare_sequence(config,seq)
    return SimpleNamespace(config_path=path,config=config,values=values,
                           output=tmp_path/"output",model=model,checkpoint=checkpoint)

def make_plan(f, sequences=("Scene01/Clone",)):
    from virtual_kitti_eval.runner import preflight_run
    return preflight_run(f.config_path, "VGGT", sequences, f.output, "cuda:0")

def test_preflight_is_readonly_and_immutable(run_fixture):
    from virtual_kitti_eval.provenance import RunPlan
    p = make_plan(run_fixture)
    assert isinstance(p, RunPlan) and not p.output_dir.exists()
    assert len(p.prepared_sequences[0].frame_ids) == 4
    with pytest.raises(TypeError):
        p.config_payload["execution_environment"]["CUDA_VISIBLE_DEVICES"] = "0"

@pytest.mark.parametrize("blocker", ["config","manifest","dependency","checkpoint","output","device","disk","sequence"])
def test_all_preflight_blockers_precede_worker(run_fixture, monkeypatch, blocker):
    from virtual_kitti_eval import runner
    f = run_fixture
    monkeypatch.setattr(runner, "launch_worker", lambda *a,**k: pytest.fail("worker started"))
    device, ids = "cuda:0", ("Scene01/Clone",)
    if blocker == "config": f.config_path.write_text("{}")
    elif blocker == "manifest": (f.config.prepared_root/"Scene01/Clone/manifest.json").write_text("{}")
    elif blocker == "dependency": (f.model/"vggt/models/vggt.py").unlink()
    elif blocker == "checkpoint": f.checkpoint.write_bytes(b"truncated")
    elif blocker == "output":
        f.output.mkdir()
        (f.output/"unrelated.txt").write_text("preserve")
    elif blocker == "device": device = "cuda:1"
    elif blocker == "disk":
        monkeypatch.setattr(runner.shutil,"disk_usage",lambda p: SimpleNamespace(free=0))
    elif blocker == "sequence": ids = ("11",)
    with pytest.raises(ValueError):
        runner.preflight_run(f.config_path,"vggt",ids,f.output,device)
    assert not (f.output/"run_manifest.json").exists()

@pytest.mark.parametrize("damage", ["partial","invalid","stale","extra_frames","missing_manifest"])
def test_resume_quarantines_and_reruns(run_fixture, damage):
    from virtual_kitti_eval.runner import run_evaluation
    from virtual_kitti_eval.io import content_sha256
    p = make_plan(run_fixture)
    run_evaluation(p)
    d = p.output_dir/"Scene01/Clone"
    original = json.loads((d/"worker/request.json").read_text())["request_id"]
    if damage == "partial": (d/"result.json").unlink()
    elif damage == "invalid": (d/"metrics.json").write_text('{"rmse_m": NaN}')
    elif damage == "missing_manifest": (p.output_dir/"run_manifest.json").unlink()
    else:
        r = json.loads((d/"result.json").read_text())
        if damage == "stale": r["provenance_id"] = "0"*64
        else:
            m = json.loads((d/"metrics.json").read_text())
            r["input_frames"] = m["matched_frames"] = 5
            r["metrics_sha256"] = content_sha256(m)
            (d/"metrics.json").write_text(json.dumps(m))
        (d/"result.json").write_text(json.dumps(r))
    summary = run_evaluation(make_plan(run_fixture),resume=True)
    assert summary.per_sequence["Scene01/Clone"] < 1e-10
    assert json.loads((d/"worker/request.json").read_text())["request_id"] != original
    assert list((p.output_dir/"failures").glob("Clone-*-*"))

def test_exact_resume_and_nonresume_collision(run_fixture, monkeypatch):
    from virtual_kitti_eval import runner
    p = make_plan(run_fixture)
    runner.run_evaluation(p)
    monkeypatch.setattr(runner,"launch_worker",lambda *a,**k: pytest.fail("worker started"))
    assert runner.run_evaluation(make_plan(run_fixture),resume=True).per_sequence["Scene01/Clone"] < 1e-10
    with pytest.raises(ValueError,match="EXISTS"): runner.run_evaluation(p)

def test_environment_change_rejected_before_output(run_fixture, monkeypatch):
    from virtual_kitti_eval.runner import run_evaluation
    p = make_plan(run_fixture)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    with pytest.raises(ValueError,match="ENVIRONMENT"): run_evaluation(p)
    assert not p.output_dir.exists()

def test_all_sequences_rechecked_before_launch(run_fixture, monkeypatch):
    from virtual_kitti_eval import runner
    p = make_plan(run_fixture,("Scene01/Clone","Scene01/Fog"))
    (run_fixture.config.raw_root/"vkitti_1.3.1_extrinsicsgt/0001_fog.txt").write_text("corrupted")
    monkeypatch.setattr(runner,"launch_worker",lambda *a,**k: pytest.fail("worker started"))
    with pytest.raises(ValueError): runner.run_evaluation(p)
    assert not p.output_dir.exists()

def test_metric_failure_keeps_terminal_artifact(run_fixture, monkeypatch):
    from virtual_kitti_eval import runner
    p = make_plan(run_fixture)
    def bad_metric(*args):
        raise ValueError("fixture degenerate prediction")
    monkeypatch.setattr(runner, "ate_rmse_m", bad_metric)
    summary = runner.run_evaluation(p)
    assert summary.failures["Scene01/Clone"]["status"] == "error"
    failure = json.loads((p.output_dir / "Scene01/Clone/failure.json").read_text())
    assert failure["code"] == "METRIC_OR_PREDICTION_INVALID"


def test_unexpected_worker_exception_is_terminal_and_continues(run_fixture, monkeypatch):
    from virtual_kitti_eval import runner
    p = make_plan(run_fixture, ("Scene01/Clone", "Scene01/Fog"))
    original = runner.launch_worker
    def broken(request, timeout):
        if request.sequence == "Scene01/Clone":
            raise OSError("fixture launcher failure")
        return original(request, timeout)
    monkeypatch.setattr(runner, "launch_worker", broken)
    summary = runner.run_evaluation(p)
    assert summary.failures["Scene01/Clone"]["status"] == "error"
    assert "Scene01/Fog" in summary.per_sequence
    assert (p.output_dir / "Scene01/Clone/failure.json").is_file()


def test_preflight_rejects_output_overlapping_source(run_fixture):
    from virtual_kitti_eval.runner import preflight_run
    with pytest.raises(ValueError, match="OUTPUT_COLLISION"):
        preflight_run(run_fixture.config_path, "vggt", ("Scene01/Clone",),
                      run_fixture.model / "result", "cuda:0")


def test_preflight_rejects_duplicate_json_keys(run_fixture):
    from virtual_kitti_eval.runner import preflight_run
    text = run_fixture.config_path.read_text()
    run_fixture.config_path.write_text(text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))
    with pytest.raises(ValueError, match="JSON"):
        make_plan(run_fixture)

def test_run_sequence_accepts_matching_verified_input(run_fixture):
    from virtual_kitti_eval.runner import run_sequence
    from virtual_kitti_eval.data import verify_prepared_sequence
    p = make_plan(run_fixture)
    prepared = verify_prepared_sequence(run_fixture.config, "Scene01/Clone")
    assert run_sequence(p, prepared).status == "success"

@pytest.mark.parametrize("source", ["config", "model", "checkpoint"])
def test_plan_rejects_source_drift_after_preflight(run_fixture, source, monkeypatch):
    from virtual_kitti_eval import runner
    p = make_plan(run_fixture)
    target = {"config": run_fixture.config_path,
              "model": run_fixture.model / "vggt/models/vggt.py",
              "checkpoint": run_fixture.checkpoint}[source]
    with target.open("ab") as stream:
        stream.write(b" ")
    monkeypatch.setattr(runner, "launch_worker", lambda *a, **k: pytest.fail("worker started"))
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        runner.run_evaluation(p)
    assert not p.output_dir.exists()

def test_source_drift_after_success_fails_every_remaining_sequence(run_fixture, monkeypatch):
    from virtual_kitti_eval import runner
    from virtual_kitti_eval.provenance import build_provenance
    from virtual_kitti_eval.results import read_json
    plan = make_plan(run_fixture, ("Scene01/Clone", "Scene01/Fog", "Scene01/Morning"))
    original_manifest = build_provenance(plan)
    original_worker = runner.launch_worker
    launched = []
    def drift_after_first(request, timeout):
        launched.append(request.sequence)
        result = original_worker(request, timeout)
        with (run_fixture.model / "vggt/models/vggt.py").open("a") as stream:
            stream.write("# source changed after worker success\n")
        return result
    monkeypatch.setattr(runner, "launch_worker", drift_after_first)
    summary = runner.run_evaluation(plan)
    assert launched == ["Scene01/Clone"]
    assert not summary.complete
    assert read_json(plan.output_dir / "run_manifest.json") == original_manifest
    first = read_json(plan.output_dir / "Scene01/Clone/result.json")
    assert first["status"] == "success"
    assert first["provenance_id"] == original_manifest["provenance_id"]
    for sequence in ("Scene01/Fog", "Scene01/Morning"):
        directory = plan.output_dir / sequence
        terminal = read_json(directory / "result.json")
        failure = read_json(directory / "failure.json")
        assert terminal["status"] == "error" and terminal["metrics_sha256"] is None
        assert terminal["provenance_id"] == original_manifest["provenance_id"]
        assert terminal["worker_exit_state"] == {"returncode": None, "signal": None}
        assert failure["code"] == "SOURCE_CHANGED"
        assert failure["provenance_id"] == original_manifest["provenance_id"]
        assert not (directory / "worker").exists()
        assert not (directory / "metrics.json").exists()
    assert not (plan.output_dir / ".run.lock").exists()


@pytest.mark.parametrize("relative", ["runs/output", "another/nested/output"])
def test_output_inside_consumed_slam_source_rejected_before_writes(run_fixture, relative):
    from virtual_kitti_eval.runner import preflight_run
    f = run_fixture
    torch_home = f.config_path.parent / "torch"
    consumed = torch_home / "hub/facebookresearch_dinov2_main"
    consumed.mkdir(parents=True)
    (consumed / "hubconf.py").write_text("# source")
    weights = torch_home / "hub/checkpoints/dino_salad.ckpt"
    weights.parent.mkdir()
    import zipfile
    with zipfile.ZipFile(weights, "w") as archive:
        archive.writestr("archive/data.pkl", b"fixture checkpoint")
        archive.writestr("archive/data/0", bytes(4))
        archive.writestr("archive/version", b"3")
    for name in ("vggt_slam/solver.py", "third_party/vggt/vggt/models/vggt.py",
                 "third_party/salad/salad/models_salad/backbones/dinov2.py"):
        path = f.model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture source")
    f.values["models"] = {"vggt_slam": {**f.values["models"]["vggt"], "torch_home": str(torch_home)}}
    f.config_path.write_text(json.dumps(f.values))
    output = consumed / relative
    before = sorted(str(p.relative_to(consumed)) for p in consumed.rglob("*"))
    with pytest.raises(ValueError, match="OUTPUT_COLLISION"):
        preflight_run(f.config_path, "vggt_slam", ("Scene01/Clone",), output, "cuda:0")
    assert not output.exists()
    assert before == sorted(str(p.relative_to(consumed)) for p in consumed.rglob("*"))


@pytest.mark.parametrize("failure", ["timeout", "nonzero"])
def test_cli_interpreter_probe_failures_are_structured_json(run_fixture, monkeypatch, capsys, failure):
    import subprocess
    from virtual_kitti_eval.cli import main
    original_run = subprocess.run
    def failing_version(command, *args, **kwargs):
        if "--version" in command:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, 15)
            raise subprocess.CalledProcessError(3, command, stderr="fixture failed")
        return original_run(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", failing_version)
    code = main(["run", "--config", str(run_fixture.config_path), "--model", "vggt",
                 "--sequence", "Scene01/Clone", "--output", str(run_fixture.output)])
    captured = capsys.readouterr()
    assert code == 1
    payload = json.loads(captured.out, parse_constant=lambda x: pytest.fail("nonfinite JSON"))
    assert payload["code"] == "RUN_FAILED"
    assert not payload["requested_complete"] and not payload["formal_complete"]
    assert "Traceback" not in captured.out + captured.err
    assert not run_fixture.output.exists()


def test_nested_condition_symlink_output_is_rejected_before_worker(run_fixture,tmp_path):
    from virtual_kitti_eval.runner import preflight_run
    f = run_fixture
    scene = f.output/"Scene01";scene.mkdir(parents=True)
    target = tmp_path/"unrelated";target.mkdir()
    (scene/"Clone").symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError,match="OUTPUT_COLLISION"):
        preflight_run(f.config_path,"vggt",("Scene01/Clone",),f.output,"cuda:0")
    assert not list(target.iterdir())


def test_subset_change_quarantines_unrequested_condition_without_poisoning_result(run_fixture):
    from virtual_kitti_eval import runner
    first = make_plan(run_fixture,("Scene01/Clone","Scene01/Fog"))
    runner.run_evaluation(first)
    second = make_plan(run_fixture,("Scene01/Fog",))
    summary = runner.run_evaluation(second,resume=True)
    assert "Scene01/Fog" in summary.per_sequence
    assert not (run_fixture.output/"Scene01/Clone").exists()
    assert not any(v.get("code") == "UNEXPECTED_SEQUENCE" for v in summary.failures.values())


@pytest.mark.parametrize("asset",["salad_checkpoint","dino_checkpoint","dependency_path"])
def test_auxiliary_native_asset_drift_blocks_resume(run_fixture,asset):
    from virtual_kitti_eval import runner
    f = run_fixture
    value = f.config_path.parent/asset
    if asset == "dependency_path":
        value.mkdir(); changed=value/"helper.py";changed.write_text("# helper")
    else:
        changed=value;changed.write_bytes(b"weights")
    f.values["models"]["vggt"][asset]=str(value)
    f.config_path.write_text(json.dumps(f.values))
    plan = make_plan(f)
    with changed.open("ab") as stream:stream.write(b" changed")
    with pytest.raises(ValueError,match="SOURCE_CHANGED"):runner.run_evaluation(plan)
    assert not f.output.exists()

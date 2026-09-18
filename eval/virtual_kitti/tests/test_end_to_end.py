import json
import subprocess
import sys
from .test_runner import run_fixture

def cli(*args):
    return subprocess.run([sys.executable, "-B", "-c",
        "import sys; from virtual_kitti_eval.cli import main; code=main(sys.argv[1:]); "
        "assert 'torch' not in sys.modules; raise SystemExit(code)", *map(str, args)],
                          text=True,capture_output=True,timeout=90)

def test_full_cpu_fake_cli_artifacts(run_fixture):
    f = run_fixture
    c = cli("run","--config",f.config_path,"--model","vggt","--sequence",*f.config.sequence_ids,
            "--output",f.output,"--device","cuda:0")
    assert c.returncode == 0, c.stdout+c.stderr
    summary = json.loads(c.stdout)
    assert summary["requested_complete"] and summary["formal_complete"] and summary["complete"]
    required = ["run_manifest.json","all_sequences_metrics.json","summary.json"]
    for seq in f.config.sequence_ids:
        required += [f"{seq}/"+name for name in ("metrics.json","result.json","worker/request.json",
            "worker/worker_result.json","worker/prediction.npz","worker/stdout.log","worker/stderr.log")]
    assert all((f.output/name).is_file() for name in required)
    from virtual_kitti_eval.results import read_json
    for path in f.output.rglob("*.json"): read_json(path)
    worker = read_json(f.output/"Scene01/Clone/worker/worker_result.json")
    assert worker["metadata"]["torch_loaded"] is False
    assert worker["resources"]["peak_allocated_mib"] == 0
    assert cli("aggregate","--output",f.output).returncode == 0
    table = cli("export-table","--output",f.output)
    assert table.returncode == 0 and "| VGGT |" in table.stdout

def test_subset_success_not_formal_completion(run_fixture):
    f = run_fixture
    c = cli("run","--config",f.config_path,"--model","vggt","--sequence","Scene01/Clone","--output",f.output)
    assert c.returncode == 0, c.stdout+c.stderr
    s = json.loads(c.stdout)
    assert s["requested_complete"] and not s["formal_complete"] and not s["complete"]
    assert cli("aggregate","--output",f.output).returncode == 1

def test_failure_continues_and_exits_nonzero(run_fixture):
    f = run_fixture
    f.values["models"]["vggt"]["fixture_fail"] = "Scene01/Clone"
    f.config_path.write_text(json.dumps(f.values))
    c = cli("run","--config",f.config_path,"--model","vggt","--sequence","Scene01/Clone","Scene01/Fog","--output",f.output)
    assert c.returncode == 1, c.stdout+c.stderr
    s = json.loads(c.stdout)
    assert not s["requested_complete"] and s["failures"]["Scene01/Clone"]["status"] == "error"
    assert "Scene01/Fog" in s["per_sequence"] and (f.output/"Scene01/Clone/failure.json").is_file()

def test_run_requires_explicit_subset(run_fixture):
    f = run_fixture
    c = cli("run","--config",f.config_path,"--model","vggt","--output",f.output)
    assert c.returncode == 2 and "--sequence" in c.stderr and not f.output.exists()

def test_thin_wrappers_resolve_root_and_preserve_physical_gpu(run_fixture):
    import os
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHON": sys.executable}
    preflight = subprocess.run(["bash", str(root / "scripts/preflight_h20.sh"),
        "--config", str(run_fixture.config_path)], cwd="/tmp", env=env,
        text=True, capture_output=True, timeout=30)
    assert preflight.returncode == 0, preflight.stdout + preflight.stderr
    run = subprocess.run(["bash", str(root / "scripts/run_one.sh"), "vggt", "Scene01/Clone",
        str(run_fixture.output), "--config", str(run_fixture.config_path)],
        cwd="/tmp", env=env, text=True, capture_output=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout)["requested_complete"]
    assert "--sequence Scene01/Clone" in run.stderr and "CUDA_VISIBLE_DEVICES=7" in run.stderr
    assert not (run_fixture.output / "Scene01/Fog").exists()


def test_virtual_kitti_runtime_is_independent_and_model_free_in_parent():
    code = """
import builtins
old = builtins.__import__
def guarded(name,*a,**k):
    if name.split('.')[0] in {'kitti_eval','waymo_eval','torch'}: raise AssertionError(name)
    return old(name,*a,**k)
builtins.__import__ = guarded
import virtual_kitti_eval.runner
import virtual_kitti_eval.backend_worker
"""
    completed = subprocess.run([sys.executable,"-B","-c",code],capture_output=True,text=True)
    assert completed.returncode == 0, completed.stderr

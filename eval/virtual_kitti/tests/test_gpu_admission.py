"""GPU admission fixtures never access a real GPU or signal a process."""
import json
import subprocess
from types import SimpleNamespace
import pytest
from virtual_kitti_eval import runner
from .test_runner import run_fixture, make_plan

INVENTORY = "0, GPU-zero, 97871, 95000\n7, GPU-fixture-seven, 97871, 90000\n"
PROCESS = "GPU-fixture-seven, 4321, python train.py, 100\n"

def mock_queries(monkeypatch, *, inventory=INVENTORY, processes=""):
    original = subprocess.run
    state = {"inventory": inventory, "processes": processes}
    def run(command, *args, **kwargs):
        if command[0] == "nvidia-smi":
            if "--query-gpu=index,uuid" in command:  # old API, for behavioral RED evidence
                return SimpleNamespace(stdout="0, GPU-zero\n7, GPU-fixture-seven\n", returncode=0)
            output = state["processes"] if any("query-compute-apps" in v for v in command) else state["inventory"]
            if isinstance(output, Exception):
                raise output
            return SimpleNamespace(stdout=output, returncode=0)
        return original(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    return state

def test_free_gpu_reports_physical_selection(monkeypatch):
    mock_queries(monkeypatch)
    diagnostic = runner._validate_device("cuda:0")
    assert diagnostic["selected_gpu_uuid"] == "GPU-fixture-seven"
    assert diagnostic["observed_free_mib"] == 90000
    assert diagnostic["required_free_mib"] == 81920
    assert diagnostic["compute_processes"] == []

@pytest.mark.parametrize("mask,device,expected", [
    ("7,0","cuda:1","GPU-zero"), ("GPU-fixture","cuda","GPU-fixture-seven"),
    ("GPU-zero,GPU-fixture-seven","cuda:1","GPU-fixture-seven"),
    (None,"cuda:0","GPU-zero"),
])
def test_logical_visibility_maps_to_uuid(monkeypatch,mask,device,expected):
    mock_queries(monkeypatch)
    if mask is None: monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    else: monkeypatch.setenv("CUDA_VISIBLE_DEVICES",mask)
    assert runner._validate_device(device)["selected_gpu_uuid"] == expected

@pytest.mark.parametrize("inventory,processes,code", [
    (INVENTORY.replace("90000","81919"), "", "GPU_MEMORY_INSUFFICIENT"),
    (INVENTORY, PROCESS, "GPU_BUSY"),
    ("garbage", "", "GPU_QUERY_FAILED"),
    (INVENTORY.replace("90000","NaN"), "", "GPU_QUERY_FAILED"),
    (INVENTORY, "GPU-fixture-seven, bad-pid, python, 100", "GPU_QUERY_FAILED"),
    (INVENTORY, subprocess.TimeoutExpired("nvidia-smi",15), "GPU_QUERY_FAILED"),
])
def test_gpu_blockers_fail_closed_with_diagnostics(monkeypatch,inventory,processes,code):
    mock_queries(monkeypatch,inventory=inventory,processes=processes)
    with pytest.raises(ValueError,match=code) as caught:
        runner._validate_device("cuda:0")
    assert caught.value.code == code
    diagnostic = caught.value.diagnostics
    assert {"selected_gpu_uuid","observed_free_mib","required_free_mib","compute_processes"} <= diagnostic.keys()
    if code != "GPU_QUERY_FAILED":
        assert diagnostic["selected_gpu_uuid"] == "GPU-fixture-seven"
    if code == "GPU_BUSY":
        assert diagnostic["compute_processes"] == [
            dict(gpu_uuid="GPU-fixture-seven",pid=4321,process_name="python train.py",used_mib=100)]

@pytest.mark.parametrize("mask",["", "7,7", "GPU-", "99"])
def test_bad_visibility_fails_closed(monkeypatch,mask):
    mock_queries(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES",mask)
    with pytest.raises(ValueError,match="GPU_QUERY_FAILED"):
        runner._validate_device("cuda:0")

def test_explicit_lower_threshold_is_honored(run_fixture,monkeypatch):
    f=run_fixture
    f.values.update(gpu_min_free_mib=1000,gpu_require_no_compute_processes=False)
    f.config_path.write_text(json.dumps(f.values))
    mock_queries(monkeypatch,inventory=INVENTORY.replace("90000","1001"),processes=PROCESS)
    assert make_plan(f).device == "cuda:0"

def test_busy_preflight_reports_json_and_never_launches(run_fixture,monkeypatch,capsys):
    from virtual_kitti_eval.cli import main
    mock_queries(monkeypatch,processes=PROCESS)
    monkeypatch.setattr(runner,"launch_worker",lambda *a: pytest.fail("worker launched"))
    code=main(["run","--config",str(run_fixture.config_path),"--model","vggt",
               "--sequence","Scene01/Clone","--output",str(run_fixture.output)])
    payload=json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["code"] == "GPU_BUSY"
    assert payload["diagnostics"]["selected_gpu_uuid"] == "GPU-fixture-seven"
    assert not run_fixture.output.exists()

def test_gpu_busy_before_first_worker_blocks_launch(run_fixture,monkeypatch):
    state=mock_queries(monkeypatch)
    plan=make_plan(run_fixture)
    state["processes"]=PROCESS
    monkeypatch.setattr(runner,"launch_worker",lambda *a: pytest.fail("worker launched"))
    with pytest.raises(ValueError,match="GPU_BUSY"):
        runner.run_evaluation(plan)
    assert not run_fixture.output.exists()

def test_gpu_becoming_busy_between_serial_inputs_stops_later_workers(run_fixture,monkeypatch):
    state=mock_queries(monkeypatch)
    plan=make_plan(run_fixture,("Scene01/Clone","Scene02/Clone"))
    launched=[]
    original=runner.launch_worker
    def worker(request,timeout):
        launched.append(request.sequence)
        result=original(request,timeout)
        state["processes"]=PROCESS
        return result
    monkeypatch.setattr(runner,"launch_worker",worker)
    runner.run_evaluation(plan)
    assert launched == ["Scene01/Clone"]
    directory=plan.output_dir/"Scene02/Clone"
    failure=json.loads((directory/"failure.json").read_text())
    assert failure["code"] == "GPU_BUSY"
    assert failure["diagnostics"]["observed_free_mib"] == 90000
    assert failure["diagnostics"]["compute_processes"][0]["pid"] == 4321
    assert not (directory/"worker").exists()

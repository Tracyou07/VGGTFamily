import json
import numpy as np
import pytest
from PIL import Image
from .test_runner import run_fixture, make_plan

def mutate(f,kind):
    if kind=="part":
        (f.config.archive_root/"new.zip.part").write_bytes(b"incomplete")
    elif kind=="extra_image":
        Image.new("RGB",(16,12)).save(f.config.raw_root/"sequences/00/image_2/000003.png")
    elif kind=="removed_image":
        (f.config.raw_root/"sequences/00/image_2/000002.png").unlink()
    elif kind=="reordered_image":
        path=f.config.raw_root/"sequences/00/image_2"
        a,b=path/"000000.png",path/"000001.png"
        first,second=a.read_bytes(),b.read_bytes()
        a.write_bytes(second); b.write_bytes(first)
    elif kind=="coherent_growth":
        Image.new("RGB",(16,12)).save(f.config.raw_root/"sequences/00/image_2/000003.png")
        with (f.config.raw_root/"poses/00.txt").open("a") as stream:
            np.savetxt(stream,np.eye(4)[:3].reshape(1,12))
        with (f.config.raw_root/"sequences/00/times.txt").open("a") as stream:
            stream.write("0.3\n")

@pytest.mark.parametrize("kind,code",[("part","INCOMPLETE_ARCHIVE"),("extra_image","COUNT_MISMATCH"),
    ("removed_image","COUNT_MISMATCH"),("reordered_image","SOURCE_CHANGED"),("coherent_growth","SOURCE_CHANGED")])
def test_doctor_and_run_preflight_reject_current_raw_drift(run_fixture,monkeypatch,capsys,kind,code):
    from kitti_eval import runner
    from kitti_eval.cli import main
    mutate(run_fixture,kind)
    monkeypatch.setattr(runner,"launch_worker",lambda *a: pytest.fail("worker launched"))
    assert main(["doctor","--config",str(run_fixture.config_path),"--sequence","00"]) == 1
    payload=json.loads(capsys.readouterr().out)
    assert payload["blockers"][0]["code"] == code
    with pytest.raises(ValueError,match=code):
        make_plan(run_fixture)
    assert not run_fixture.output.exists()

@pytest.mark.parametrize("kind",["part","extra_image"])
def test_raw_rechecked_before_cached_success_can_resume(run_fixture,monkeypatch,kind):
    from kitti_eval import runner
    plan=make_plan(run_fixture)
    runner.run_evaluation(plan)
    before=(plan.output_dir/"00/result.json").read_bytes()
    mutate(run_fixture,kind)
    monkeypatch.setattr(runner,"launch_worker",lambda *a: pytest.fail("worker launched"))
    with pytest.raises(ValueError):
        runner.run_evaluation(plan,resume=True)
    assert (plan.output_dir/"00/result.json").read_bytes() == before

def test_raw_drift_between_workers_fails_remaining_sequence(run_fixture,monkeypatch):
    from kitti_eval import runner
    plan=make_plan(run_fixture,("00","01"))
    launched=[]
    original=runner.launch_worker
    def launch(request,timeout):
        launched.append(request.sequence)
        worker=original(request,timeout)
        mutate(run_fixture,"part")
        return worker
    monkeypatch.setattr(runner,"launch_worker",launch)
    runner.run_evaluation(plan)
    assert launched == ["00"]
    failure=json.loads((plan.output_dir/"01/failure.json").read_text())
    assert "INCOMPLETE_ARCHIVE" in failure["message"]
    assert not (plan.output_dir/"01/worker").exists()

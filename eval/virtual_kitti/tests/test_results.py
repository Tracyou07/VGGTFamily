import json
from dataclasses import replace
import numpy as np
import pytest
from virtual_kitti_eval.results import aggregate, export_table, load_result_pair, read_json, atomic_write_json
from virtual_kitti_eval.provenance import build_provenance
from virtual_kitti_eval.io import content_sha256
from .conftest import MAIN

HEADER = "| Model | Calibration | " + " | ".join(f"Scene {s} {c}" for s in ("01","02")
    for c in ("Clone","Fog","Morning","Overcast","Rain","Sunset")) + " | Status |"
RESOURCE_HEADER = "| Model | Scene / condition | Input frames | Inference time (s) ↓ | Peak VRAM (MiB) ↓ | Status |"

def test_conditions_and_exact_columns_remain_separate(vkitti_result_dir):
    f = vkitti_result_dir
    summary = aggregate(f.output)
    assert summary.complete
    assert summary.per_sequence["Scene01/Clone"] == 1.
    assert summary.per_sequence["Scene01/Fog"] == 2.
    assert len(summary.per_sequence) == 12
    table = export_table(f.output)
    assert table.splitlines()[0] == HEADER
    assert RESOURCE_HEADER in table
    assert "Avg." not in table

def test_missing_and_failed_condition_keep_aggregate_incomplete(vkitti_result_dir):
    f = vkitti_result_dir
    (f.output / "Scene01/Fog/metrics.json").unlink()
    summary = aggregate(f.output)
    assert not summary.complete
    assert "Scene01/Fog" in summary.failures
    assert "Scene01/Fog" not in summary.per_sequence
    assert "incomplete" in export_table(f.output)

@pytest.mark.parametrize("drift", ["checkpoint", "model_source", "raw", "config", "frame_hash", "metric", "provenance"])
def test_resume_rejects_stale_or_mismatched_inputs(vkitti_result_dir, drift):
    f = vkitti_result_dir
    directory = f.output / "Scene01/Clone"
    if drift == "checkpoint": f.checkpoint.write_bytes(b"changed")
    elif drift == "model_source": (f.model_root / "model.py").write_text("# changed")
    elif drift == "config": f.config_path.write_text(f.config_path.read_text() + "\n")
    elif drift == "raw": (f.raw / "vkitti_1.3.1_rgb/0001/clone/00000.png").write_bytes(b"bad")
    elif drift == "provenance": f.provenance["device"] = "changed"
    else:
        metric = read_json(directory / "metrics.json")
        if drift == "frame_hash": metric["frame_ids_sha256"] = "0" * 64
        else: metric["rmse_m"] = 999.
        atomic_write_json(directory / "metrics.json", metric)
    with pytest.raises(ValueError): load_result_pair(directory, f.provenance)

def test_finite_atomic_json_does_not_replace_existing_file(tmp_path):
    file = tmp_path / "value.json"
    atomic_write_json(file, {"good": 1})
    before = file.read_bytes()
    with pytest.raises(ValueError): atomic_write_json(file, {"bad": float("nan")})
    assert file.read_bytes() == before
    assert list(tmp_path.iterdir()) == [file]

@pytest.mark.parametrize("payload", ['{"a": 1, "a": 2}', '{"x": NaN}', '{"x": 1e999}'])
def test_strict_json_rejects_duplicate_and_nonfinite(tmp_path, payload):
    path = tmp_path / "value.json"; path.write_text(payload)
    with pytest.raises(ValueError): read_json(path)

def test_runplan_snapshots_nested_config_and_pose_arrays(vkitti_result_dir):
    plan = vkitti_result_dir.plan
    with pytest.raises(TypeError): plan.model_config["checkpoint"] = "changed"
    with pytest.raises(ValueError): plan.prepared_sequences[0].poses_c2w.setflags(write=True)
    assert build_provenance(plan) == vkitti_result_dir.provenance

def test_oversized_alignment_is_invalid_result(vkitti_result_dir):
    f = vkitti_result_dir
    directory = f.output / "Scene01/Clone"
    metric = read_json(directory / "metrics.json")
    metric["alignment"]["translation"][0] = 10 ** 500
    atomic_write_json(directory / "metrics.json", metric)
    result = read_json(directory / "result.json")
    result["metrics_sha256"] = content_sha256(metric)
    atomic_write_json(directory / "result.json", result)
    summary = aggregate(f.output)
    assert not summary.complete
    assert summary.failures["Scene01/Clone"]["code"] == "INVALID_RESULT"

def test_failed_worker_is_preserved_and_never_resumable(vkitti_result_dir):
    from virtual_kitti_eval.results import write_result_pair
    f = vkitti_result_dir
    directory = f.output / "Scene01/Fog"
    result = read_json(directory / "result.json")
    result.update(status="oom", inference_seconds=None, peak_allocated_mib=None,
        peak_reserved_mib=None, worker_exit_state={"returncode": 1, "signal": None})
    write_result_pair(directory, result, None)
    with pytest.raises(ValueError, match="FAILED_RESULT"):
        load_result_pair(directory, f.provenance)
    summary = aggregate(f.output)
    assert not summary.complete
    assert "Scene01/Fog" not in summary.per_sequence
    assert summary.failures["Scene01/Fog"]["status"] == "oom"
    assert summary.resources["Scene01/Fog"]["input_frames"] == 4

def test_unconfigured_condition_artifact_cannot_enter_formal_results(vkitti_result_dir):
    f = vkitti_result_dir
    rogue = f.output / "Scene06/Clone"
    rogue.mkdir(parents=True)
    (rogue / "result.json").write_text("{}")
    summary = aggregate(f.output)
    assert not summary.complete
    assert summary.failures["Scene06/Clone"]["code"] == "UNEXPECTED_SEQUENCE"
    assert "Scene06/Clone" not in summary.per_sequence

@pytest.mark.parametrize("alias_kind", ["condition", "scene"])
def test_directory_alias_cannot_duplicate_condition_score_or_resource(vkitti_result_dir, alias_kind):
    f = vkitti_result_dir
    if alias_kind == "condition":
        requested = f.output / "Scene01/Fog"
        requested.rename(f.root / "saved_fog")
        requested.symlink_to(f.output / "Scene01/Clone", target_is_directory=True)
        rejected = "Scene01/Fog"
    else:
        requested = f.output / "Scene02"
        requested.rename(f.root / "saved_scene02")
        requested.symlink_to(f.output / "Scene01", target_is_directory=True)
        rejected = "Scene02/Clone"
    summary = aggregate(f.output)
    assert not summary.complete
    assert summary.failures[rejected]["code"] == "INVALID_RESULT"
    assert rejected not in summary.per_sequence
    assert rejected not in summary.resources
    assert summary.per_sequence["Scene01/Clone"] == 1.
    assert summary.resources["Scene01/Clone"]["sequence"] == "Scene01/Clone"
    on_disk = read_json(f.output / "all_sequences_metrics.json")
    assert rejected not in on_disk["sequences"]


def test_resume_rejects_condition_directory_alias(vkitti_result_dir):
    f = vkitti_result_dir
    fog = f.output / "Scene01/Fog"
    fog.rename(f.root / "saved_fog")
    fog.symlink_to(f.output / "Scene01/Clone", target_is_directory=True)
    with pytest.raises(ValueError, match="RESULT_DIRECTORY_ALIAS"):
        load_result_pair(fog, f.provenance)


def test_alias_does_not_attribute_failed_worker_resources_to_another_condition(vkitti_result_dir):
    from virtual_kitti_eval.results import write_result_pair
    f = vkitti_result_dir
    clone = f.output / "Scene01/Clone"
    result = read_json(clone / "result.json")
    result.update(status="oom", inference_seconds=None, peak_allocated_mib=None,
        peak_reserved_mib=None, worker_exit_state={"returncode": 1, "signal": None})
    write_result_pair(clone, result, None)
    fog = f.output / "Scene01/Fog"
    fog.rename(f.root / "saved_fog")
    fog.symlink_to(clone, target_is_directory=True)
    summary = aggregate(f.output)
    assert summary.failures["Scene01/Fog"]["code"] == "INVALID_RESULT"
    assert "Scene01/Fog" not in summary.resources
    assert summary.resources["Scene01/Clone"]["status"] == "oom"

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from .test_provenance import plan


def pair(provenance, sequence, value=1., status="success"):
    from kitti_eval.io import content_sha256
    ids = provenance["sequences"][sequence]["frame_ids"]
    result = dict(schema_version=1, model_key=provenance["model_key"], sequence=sequence, status=status,
        input_frames=len(ids), inference_seconds=2.5, peak_allocated_mib=128., peak_reserved_mib=192.,
        worker_exit_state={"returncode": 0 if status == "success" else 1, "signal": None},
        provenance_id=provenance["provenance_id"])
    metric = dict(schema_version=1, protocol_id=provenance["metric_protocol_id"], matched_frames=len(ids),
        rmse_m=value, alignment={"scale": 1., "rotation": np.eye(3).tolist(), "translation": [0.,0.,0.]},
        frame_ids_sha256=content_sha256(ids), sequence=sequence, model_key=provenance["model_key"],
        provenance_id=provenance["provenance_id"])
    result["metrics_sha256"] = content_sha256(metric) if status == "success" else None
    return result, metric


@pytest.fixture
def result_dir(plan):
    from kitti_eval.provenance import build_provenance
    from kitti_eval.results import atomic_write_json
    # Real Task 2 manifests for eleven three-frame source sequences.
    from kitti_eval.config import KittiConfig
    from kitti_eval.data import prepare_sequence
    from PIL import Image
    import shutil
    root = plan.output_dir.parent
    raw = root / "aggregate_raw"
    shutil.copytree(Path(plan.config_payload["raw_root"]), raw)
    Image.new("RGB", (16, 12)).save(raw / "sequences/00/image_2/000002.png")
    pose_path = raw / "poses/00.txt"
    pose_path.write_text(pose_path.read_text() + "1 0 0 0 0 1 0 1 0 0 1 0\n")
    (raw / "sequences/00/times.txt").write_text("0.0\n0.1\n0.2\n")
    for i in range(1, 11):
        shutil.copytree(raw / "sequences/00", raw / f"sequences/{i:02d}")
        shutil.copyfile(raw / "poses/00.txt", raw / f"poses/{i:02d}.txt")
    ids = tuple(f"{i:02d}" for i in range(11))
    sequences_file = root / "aggregate_sequences.txt"
    sequences_file.write_text("\n".join(ids) + "\n")
    config = KittiConfig(1, raw, Path(plan.config_payload["archive_root"]),
                         root / "aggregate_prepared", sequences_file, ids)
    prepared = tuple(prepare_sequence(config, s) for s in ids)
    payload = {**dict(plan.config_payload), "raw_root": str(raw), "prepared_root": str(config.prepared_root),
               "sequences_file": str(sequences_file), "sequence_ids": list(ids)}
    run = replace(plan, sequence_ids=ids, prepared_sequences=prepared, config_payload=payload)
    provenance = build_provenance(run)
    plan.output_dir.mkdir()
    atomic_write_json(plan.output_dir / "run_manifest.json", provenance)
    for i in range(11):
        sequence = f"{i:02d}"
        directory = plan.output_dir / sequence
        directory.mkdir()
        result, metrics = pair(provenance, sequence, float(i))
        atomic_write_json(directory / "result.json", result)
        atomic_write_json(directory / "metrics.json", metrics)
    return plan.output_dir


def test_strict_json_read_and_atomic_write(tmp_path, monkeypatch):
    from kitti_eval.results import atomic_write_json, read_json
    path = tmp_path / "x.json"
    atomic_write_json(path, {"old": 1})
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            atomic_write_json(path, {"value": bad})
        assert read_json(path) == {"old": 1}
    import os
    def interrupted(*args):
        raise OSError("interrupted replace")
    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(OSError):
        atomic_write_json(path, {"new": 2})
    assert read_json(path) == {"old": 1}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("text", ['{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}', '{"x":1,"x":2}'])
def test_invalid_json_rejected(tmp_path, text):
    from kitti_eval.results import read_json
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_json(path)


def test_complete_aggregate_and_formal_means(result_dir):
    from kitti_eval.results import aggregate, read_json
    summary = aggregate(result_dir)
    assert summary.complete
    assert list(summary.per_sequence) == [f"{i:02d}" for i in range(11)]
    assert summary.avg_ate_rmse_m == pytest.approx(5.0)
    assert summary.avg_star_ate_rmse_m == pytest.approx(5.4)
    for name in ("all_sequences_metrics.json", "average_metrics.json", "summary.json"):
        assert read_json(result_dir / name)
    assert (result_dir / "failures").is_dir()


@pytest.mark.parametrize("status", ["oom", "error", "timeout"])
def test_failures_excluded_and_recorded(result_dir, status):
    from kitti_eval.results import aggregate, atomic_write_json, read_json
    path = result_dir / "01" / "result.json"
    result = read_json(path)
    result["status"] = status
    result["metrics_sha256"] = None
    result["worker_exit_state"]["returncode"] = 1
    atomic_write_json(path, result)
    summary = aggregate(result_dir)
    assert not summary.complete
    assert "01" not in summary.per_sequence
    assert summary.avg_ate_rmse_m is None
    assert summary.avg_star_ate_rmse_m == pytest.approx(5.4)
    assert summary.failures["01"]["status"] == status
    assert read_json(result_dir / "failures/01.json")["status"] == status


@pytest.mark.parametrize("fault", ["partial", "provenance", "frame_count", "frame_hash", "nonfinite", "reflection", "exit"])
def test_invalid_pairs_are_not_resumable_or_aggregated(result_dir, fault):
    from kitti_eval.results import aggregate, atomic_write_json, read_json, load_result_pair
    directory = result_dir / "00"
    metrics = read_json(directory / "metrics.json")
    if fault == "partial":
        (directory / "result.json").unlink()
    elif fault == "exit":
        result = read_json(directory / "result.json")
        result["worker_exit_state"]["returncode"] = 1
        atomic_write_json(directory / "result.json", result)
    else:
        changes = {"provenance": ("provenance_id", "0" * 64), "frame_count": ("matched_frames", 4),
            "frame_hash": ("frame_ids_sha256", "0" * 64), "nonfinite": ("rmse_m", float("nan")),
            "reflection": ("alignment", {"scale": 1., "rotation": np.diag([-1,1,1]).tolist(), "translation": [0,0,0]})}
        key, value = changes[fault]
        metrics[key] = value
        (directory / "metrics.json").write_text(json.dumps(metrics))
    manifest = read_json(result_dir / "run_manifest.json")
    with pytest.raises(ValueError):
        load_result_pair(directory, manifest)
    summary = aggregate(result_dir)
    assert not summary.complete and "00" not in summary.per_sequence
    assert "00" in summary.failures


def test_stale_source_rejects_entire_aggregate(result_dir):
    from kitti_eval.results import aggregate, read_json
    manifest = read_json(result_dir / "run_manifest.json")
    Path(manifest["model_config"]["project_root"], "model.py").write_text("changed")
    summary = aggregate(result_dir)
    assert not summary.complete
    assert summary.per_sequence == {}
    assert summary.failures


def test_cli_incomplete_then_complete_and_table_order(result_dir):
    from kitti_eval.cli import main
    from kitti_eval.results import export_table
    artifact = result_dir / "10/metrics.json"
    saved = artifact.read_bytes()
    artifact.unlink()
    assert main(["aggregate", "--output", str(result_dir)]) == 1
    table = export_table(result_dir)
    assert table.splitlines()[0] == "| Model | LC | Calibration | Recon. | Avg. | Avg.* | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | Status |"
    assert "—" in table
    assert "| Model | Sequence | Frames | Time (s) | Peak VRAM (MiB) | Status |" in table
    artifact.write_bytes(saved)
    assert main(["aggregate", "--output", str(result_dir)]) == 0
    assert main(["export-table", "--output", str(result_dir)]) == 0

def test_metric_serializer_and_pair_writer_match_and_resume(result_dir):
    from kitti_eval.metrics import ate_rmse_m
    from kitti_eval.results import metrics_record, write_result_pair, load_result_pair, read_json
    manifest = read_json(result_dir / "run_manifest.json")
    points = np.array([[0.,0,0],[1,0,0],[0,1,0]])
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, :3, 3] = points
    ids = tuple(manifest["sequences"]["00"]["frame_ids"])
    metric = metrics_record(ate_rmse_m(ids, poses, ids, poses), ids, "00", "vggt_long", manifest["provenance_id"])
    result, _ = pair(manifest, "00")
    write_result_pair(result_dir / "00", result, metric)
    loaded_result, loaded_metric = load_result_pair(result_dir / "00", manifest)
    assert loaded_metric["rmse_m"] == pytest.approx(0, abs=1e-12)
    assert loaded_result["input_frames"] == 3

@pytest.mark.parametrize("value", [True, -1, "nan", 10 ** 1000, None])
def test_invalid_numeric_metric_fails_closed(result_dir, value):
    from kitti_eval.results import aggregate, read_json
    path = result_dir / "00/metrics.json"
    payload = read_json(path)
    payload["rmse_m"] = value
    path.write_text(json.dumps(payload))
    summary = aggregate(result_dir)
    assert not summary.complete and "00" not in summary.per_sequence


def test_unexpected_sequence_makes_aggregate_incomplete(result_dir):
    from kitti_eval.results import aggregate
    extra = result_dir / "11"
    extra.mkdir()
    (extra / "result.json").write_text("{}")
    summary = aggregate(result_dir)
    assert not summary.complete
    assert summary.failures["11"]["code"] == "UNEXPECTED_SEQUENCE"


def test_missing_run_manifest_fails_without_accepting_old_summary(result_dir):
    from kitti_eval.results import aggregate, read_json
    aggregate(result_dir)
    (result_dir / "run_manifest.json").unlink()
    summary = aggregate(result_dir)
    assert not summary.complete and not summary.per_sequence
    assert read_json(result_dir / "average_metrics.json")["avg_ate_rmse_m"] is None


def test_export_preserves_resource_units_and_allocated_peak(result_dir):
    from kitti_eval.results import export_table
    output = export_table(result_dir)
    assert "| VGGT-Long | 00 | 3 | 2.500 | 128.000 | success |" in output
    assert "| VGGT-Long | ✓ | — | Dense | 5.000 | 5.400 |" in output

def test_copied_output_directory_is_not_exact_provenance(result_dir):
    import shutil
    from kitti_eval.results import aggregate, read_json, load_result_pair
    copied = result_dir.parent / "copied"
    shutil.copytree(result_dir, copied)
    with pytest.raises(ValueError, match="OUTPUT"):
        load_result_pair(copied / "00", read_json(copied / "run_manifest.json"))
    assert not aggregate(copied).complete

def test_overflowing_alignment_is_invalid_instead_of_crashing(result_dir):
    from kitti_eval.results import aggregate, read_json, load_result_pair
    path = result_dir / "00/metrics.json"
    record = read_json(path)
    record["alignment"]["rotation"][0][0] = 10 ** 1000
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_result_pair(result_dir / "00", read_json(result_dir / "run_manifest.json"))
    assert not aggregate(result_dir).complete

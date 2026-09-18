from dataclasses import replace
from pathlib import Path
import json
import sys

import pytest


@pytest.fixture
def plan(kitti_fixture, tmp_path):
    from kitti_eval.data import prepare_sequence
    from kitti_eval.provenance import RunPlan
    fixture = kitti_fixture
    prepared = prepare_sequence(fixture.config, "00")
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.py").write_text("MODEL = 1\n")
    package = tmp_path / "package"
    package.mkdir()
    (package / "metric.py").write_text("METRIC = 1\n")
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"fixture checkpoint")
    config = dict(schema_version=1, raw_root=str(fixture.config.raw_root),
                  archive_root=str(fixture.config.archive_root), prepared_root=str(fixture.config.prepared_root),
                  sequences_file=str(fixture.config.sequences_file), sequence_ids=["00"])
    return RunPlan(fixture.config_path, config, "vggt_long",
        {"project_root": str(model), "checkpoint": str(checkpoint), "interpreter": sys.executable,
         "loop_closure": True, "chunk_size": 75},
        ("00",), (prepared,), tmp_path / "output", "cpu", 60.0,
        ("kitti-eval", "run", "--sequence", "00"), package, "kitti-odometry-ate-sim3-v1")


def test_provenance_is_deterministic_and_live_validated(plan):
    from kitti_eval.provenance import build_provenance, validate_provenance
    first = build_provenance(plan)
    assert first == build_provenance(plan)
    assert first["sequences"]["00"]["frame_ids"] == ["000000", "000001"]
    validate_provenance(first)


@pytest.mark.parametrize("field", ["model", "checkpoint", "package", "config", "manifest", "raw", "prepared"])
def test_stale_sources_rejected(plan, field, kitti_fixture):
    from kitti_eval.provenance import build_provenance, validate_provenance
    provenance = build_provenance(plan)
    targets = {
        "model": Path(plan.model_config["project_root"]) / "model.py",
        "checkpoint": Path(plan.model_config["checkpoint"]),
        "package": plan.repository_root / "metric.py", "config": plan.config_path,
        "manifest": plan.prepared_sequences[0].manifest_path,
        "raw": kitti_fixture.pose_file,
        "prepared": plan.prepared_sequences[0].manifest_path.parent / "poses_c2w.npy"}
    with targets[field].open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SOURCE|PROVENANCE"):
        validate_provenance(provenance)


@pytest.mark.parametrize("change", [
    {"device": "cuda:1"}, {"timeout_s": 61}, {"command": ("different",)},
    {"model_config": None}, {"metric_protocol_id": "other"},
])
def test_all_run_controls_affect_provenance(plan, change):
    from kitti_eval.provenance import build_provenance
    if change.get("model_config", 1) is None:
        change = {"model_config": {**plan.model_config, "chunk_size": 76}}
    first = build_provenance(plan)
    assert first["provenance_id"] != build_provenance(replace(plan, **change))["provenance_id"]


def test_forged_manifest_or_nonfinite_config_rejected(plan):
    from kitti_eval.provenance import build_provenance, validate_provenance
    payload = build_provenance(plan)
    payload["device"] = "changed"
    with pytest.raises(ValueError, match="PROVENANCE"):
        validate_provenance(payload)
    with pytest.raises(ValueError):
        build_provenance(replace(plan, timeout_s=float("nan")))
    with pytest.raises(ValueError):
        build_provenance(replace(plan, config_payload={"x": float("inf")}))


def test_wrong_prepared_frame_ids_rejected(plan):
    from kitti_eval.provenance import build_provenance
    wrong = replace(plan.prepared_sequences[0], frame_ids=("other",))
    with pytest.raises(ValueError, match="FRAME|MANIFEST"):
        build_provenance(replace(plan, prepared_sequences=(wrong,)))

def test_cannot_mutate_nested_run_config(plan):
    with pytest.raises(TypeError):
        plan.model_config["chunk_size"] = 99


@pytest.mark.parametrize("value", [[], {}, True, -1, "0"])
def test_rehashed_invalid_provenance_controls_rejected(plan, value):
    from kitti_eval.provenance import build_provenance, validate_provenance
    from kitti_eval.io import content_sha256
    payload = build_provenance(plan)
    payload["timeout_s"] = value
    payload.pop("provenance_id")
    payload["provenance_id"] = content_sha256(payload)
    with pytest.raises(ValueError, match="PROVENANCE"):
        validate_provenance(payload)


def test_outputs_inside_repository_do_not_invalidate_provenance(plan):
    from kitti_eval.provenance import build_provenance, validate_provenance
    plan = replace(plan, output_dir=plan.repository_root / "custom_output")
    payload = build_provenance(plan)
    plan.output_dir.mkdir()
    (plan.output_dir / "summary.json").write_text('{"run": 1}')
    validate_provenance(payload)

def test_changed_sequence_inventory_source_rejected(plan):
    from kitti_eval.provenance import build_provenance, validate_provenance
    payload = build_provenance(plan)
    Path(plan.config_payload["sequences_file"]).write_text("00\n01\n")
    with pytest.raises(ValueError, match="SOURCE"):
        validate_provenance(payload)


def test_rehashed_inconsistent_model_path_rejected(plan):
    from kitti_eval.provenance import build_provenance, validate_provenance
    from kitti_eval.io import content_sha256
    payload = build_provenance(plan)
    payload["model_config"]["project_root"] = str(plan.repository_root)
    payload.pop("provenance_id")
    payload["provenance_id"] = content_sha256(payload)
    with pytest.raises(ValueError, match="PROVENANCE"):
        validate_provenance(payload)


def test_prepared_manifest_requires_strict_json(plan):
    from kitti_eval.provenance import build_provenance
    from kitti_eval.io import sha256_file
    prepared = plan.prepared_sequences[0]
    path = prepared.manifest_path
    # Duplicate same-valued key: ordinary json.loads silently accepts it.
    path.write_text(path.read_text().replace('"schema_version":1', '"schema_version":1,"schema_version":1'))
    prepared = replace(prepared, manifest_sha256=sha256_file(path))
    with pytest.raises(ValueError, match="JSON"):
        build_provenance(replace(plan, prepared_sequences=(prepared,)))


def test_extra_untracked_source_file_invalidates_provenance(plan):
    from kitti_eval.provenance import build_provenance, validate_provenance
    payload = build_provenance(plan)
    Path(plan.model_config["project_root"], "new.py").write_text("new behavior")
    with pytest.raises(ValueError, match="SOURCE"):
        validate_provenance(payload)

@pytest.mark.parametrize("field", ["image_paths", "poses_c2w", "timestamps_s", "intrinsics"])
def test_in_memory_prepared_values_must_match_manifest(plan, field):
    import numpy as np
    from kitti_eval.provenance import build_provenance
    prepared = plan.prepared_sequences[0]
    if field == "image_paths":
        value = prepared.image_paths[::-1]
    else:
        value = np.array(getattr(prepared, field), copy=True)
        value.flat[0] += 1
    bad = replace(prepared, **{field: value})
    with pytest.raises(ValueError, match="PREPARED"):
        build_provenance(replace(plan, prepared_sequences=(bad,)))

@pytest.mark.parametrize("field", ["poses_c2w", "timestamps_s", "intrinsics"])
def test_run_plan_snapshots_caller_owned_arrays(plan, field):
    import numpy as np
    from kitti_eval.provenance import build_provenance, validate_provenance
    expected = np.array(getattr(plan.prepared_sequences[0], field), copy=True)
    caller_array = expected.copy()
    caller_prepared = replace(plan.prepared_sequences[0], **{field: caller_array})
    snapshot = replace(plan, prepared_sequences=(caller_prepared,))
    provenance = build_provenance(snapshot)

    caller_array.flat[0] += 1

    np.testing.assert_array_equal(getattr(snapshot.prepared_sequences[0], field), expected)
    assert build_provenance(snapshot) == provenance
    validate_provenance(provenance)


@pytest.mark.parametrize("field", ["poses_c2w", "timestamps_s", "intrinsics"])
def test_run_plan_arrays_cannot_be_mutated_or_made_writable(plan, field):
    import numpy as np
    from kitti_eval.provenance import build_provenance, validate_provenance
    caller_array = np.array(getattr(plan.prepared_sequences[0], field), copy=True)
    snapshot = replace(plan, prepared_sequences=(
        replace(plan.prepared_sequences[0], **{field: caller_array}),))
    provenance = build_provenance(snapshot)
    array = getattr(snapshot.prepared_sequences[0], field)

    with pytest.raises(ValueError):
        array.flat[0] += 1
    with pytest.raises(ValueError):
        array.setflags(write=True)

    assert build_provenance(snapshot) == provenance
    validate_provenance(provenance)


def test_run_plan_snapshots_prepared_tuple_fields(plan):
    from kitti_eval.provenance import build_provenance, validate_provenance
    prepared = plan.prepared_sequences[0]
    caller_ids, caller_paths = list(prepared.frame_ids), list(prepared.image_paths)
    caller_sequences = [replace(prepared, frame_ids=caller_ids, image_paths=caller_paths)]
    snapshot = replace(plan, prepared_sequences=caller_sequences)
    assert snapshot.prepared_sequences[0].frame_ids == prepared.frame_ids
    assert snapshot.prepared_sequences[0].image_paths == prepared.image_paths
    provenance = build_provenance(snapshot)

    caller_ids[0] = "changed"
    caller_paths.reverse()
    caller_sequences.clear()

    assert snapshot.prepared_sequences[0].frame_ids == prepared.frame_ids
    assert snapshot.prepared_sequences[0].image_paths == prepared.image_paths
    assert build_provenance(snapshot) == provenance
    validate_provenance(provenance)

@pytest.mark.parametrize("field", ["dependency_path", "salad_checkpoint", "dino_checkpoint", "torch_home"])
def test_external_native_assets_are_live_provenance_inputs(plan, tmp_path, field):
    from kitti_eval.provenance import build_provenance, validate_provenance
    path = tmp_path / field
    if field == "torch_home":
        target = path / "hub/facebookresearch_dinov2_main/native.py"
        target.parent.mkdir(parents=True)
    elif field == "dependency_path":
        path.mkdir()
        target = path / "native.py"
    else:
        target = path
    target.write_text("original")
    model = {**plan.model_config, field: str(path)}
    if field == "torch_home":
        (path / "hub/checkpoints").mkdir(parents=True)
        (path / "hub/checkpoints/dino_salad.ckpt").write_bytes(b"salad")
    changed_plan = replace(plan, model_config=model)
    provenance = build_provenance(changed_plan)
    target.write_text("modified")
    with pytest.raises(ValueError, match="SOURCE"):
        validate_provenance(provenance)
    assert build_provenance(changed_plan)["provenance_id"] != provenance["provenance_id"]

def test_slam_provenance_ignores_unrelated_torch_cache_but_checks_consumed_weights(plan, tmp_path):
    from kitti_eval.provenance import build_provenance, validate_provenance
    root = tmp_path / "torch"
    (root / "hub/facebookresearch_dinov2_main").mkdir(parents=True)
    (root / "hub/facebookresearch_dinov2_main/hubconf.py").write_text("source")
    (root / "hub/checkpoints").mkdir()
    weight = root / "hub/checkpoints/dino_salad.ckpt"
    weight.write_bytes(b"salad")
    p = replace(plan, model_config={**plan.model_config, "torch_home": str(root)})
    provenance = build_provenance(p)
    (root / "unrelated.pth").write_bytes(b"unrelated")
    assert build_provenance(p) == provenance
    weight.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SOURCE"):
        validate_provenance(provenance)

import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from kitti_eval.config import DatasetValidationError
from kitti_eval.data import inspect_raw_sequence, prepare_sequence, verify_prepared_sequence

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def test_nested_partial_archive_blocks_before_output(kitti_fixture):
    f = kitti_fixture
    archive = f.config.archive_root / "nested"
    archive.mkdir()
    (archive / "color.zip.part").touch()
    status = inspect_raw_sequence(f.config, "00")
    assert not status.ready
    assert "INCOMPLETE_ARCHIVE" in [b.code for b in status.blockers]
    with pytest.raises(DatasetValidationError, match="INCOMPLETE_ARCHIVE"):
        prepare_sequence(f.config, "00")
    assert not f.config.prepared_root.exists()
    assert "torch" not in sys.modules

@pytest.mark.parametrize("fault,code", [
    ("empty", "EMPTY_IMAGES"), ("pose_missing", "MISSING_POSES"),
    ("calib_missing", "MISSING_CALIBRATION"), ("times_missing", "MISSING_TIMESTAMPS"),
    ("pose_count", "COUNT_MISMATCH"), ("time_count", "COUNT_MISMATCH"),
    ("duplicate", "DUPLICATE_FRAME_ID"), ("nan", "INVALID_POSES"),
    ("nonrigid", "INVALID_POSES"), ("reflection", "INVALID_POSES"),
    ("calib_nan", "INVALID_CALIBRATION"), ("time_nan", "INVALID_TIMESTAMPS"),
    ("time_reverse", "INVALID_TIMESTAMPS"), ("bad_frame", "INVALID_FRAME_ID"),
    ("corrupt_png", "INVALID_IMAGE"), ("truncated_jpeg", "INVALID_IMAGE"),
])
def test_invalid_inputs_fail_before_output(kitti_fixture, fault, code):
    f = kitti_fixture
    if fault == "empty":
        for p in f.images.iterdir(): p.unlink()
    elif fault.endswith("_missing"):
        {"pose_missing": f.pose_file, "calib_missing": f.calib, "times_missing": f.times}[fault].unlink()
    elif fault == "pose_count": np.savetxt(f.pose_file, f.poses[:1, :3].reshape(1, 12))
    elif fault == "time_count": f.times.write_text("0\n")
    elif fault == "duplicate": Image.new("RGB", (16,12)).save(f.images / "000000.jpg")
    elif fault in ("nan", "nonrigid", "reflection"):
        poses = f.poses.copy()
        poses[0, 0, 0] = {"nan": np.nan, "nonrigid": 2, "reflection": -1}[fault]
        np.savetxt(f.pose_file, poses[:, :3].reshape(2, 12))
    elif fault == "calib_nan": f.calib.write_text(f.calib.read_text().replace("100.0", "nan", 1))
    elif fault == "time_nan": f.times.write_text("0\nnan\n")
    elif fault == "time_reverse": f.times.write_text("0.1\n0\n")
    elif fault == "bad_frame": (f.images / "000000.png").rename(f.images / "bad.png")
    elif fault == "corrupt_png":
        p = f.images / "000000.png"
        p.write_bytes(p.read_bytes()[:45])
    elif fault == "truncated_jpeg":
        p = f.images / "000000.jpg"
        Image.new("RGB", (16,12)).save(p)
        p.write_bytes(p.read_bytes()[:-20])
        (f.images / "000000.png").unlink()
    with pytest.raises(DatasetValidationError, match=code):
        prepare_sequence(f.config, "00")
    assert not f.config.prepared_root.exists()

def test_camera_zero_pose_is_converted_to_camera_two(kitti_fixture):
    f = kitti_fixture
    p = prepare_sequence(f.config, "00")
    np.testing.assert_allclose(p.poses_c2w[0, :3, 3], f.expected_camera2_center)
    np.testing.assert_allclose(p.poses_c2w[1, :3, 3], [0, .5, 0], atol=1e-10)
    np.testing.assert_allclose(p.intrinsics, [[100,0,8],[0,100,6],[0,0,1]])
    assert not p.poses_c2w.flags.writeable
    assert not p.timestamps_s.flags.writeable
    assert not p.intrinsics.flags.writeable

def test_manifest_and_immutable_references(kitti_fixture):
    f = kitti_fixture
    before = {str(p): (digest(p), p.stat().st_mtime_ns) for p in f.config.raw_root.rglob("*") if p.is_file()}
    prepared = prepare_sequence(f.config, "00")
    assert before == {str(p): (digest(p), p.stat().st_mtime_ns) for p in f.config.raw_root.rglob("*") if p.is_file()}
    assert sorted(p.name for p in prepared.manifest_path.parent.iterdir()) == ["calibration.json", "manifest.json", "poses_c2w.npy"]
    manifest = json.loads(prepared.manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["protocol_id"] == "kitti-odometry-image2-c2w-v1"
    assert manifest["frame_ids"] == ["000000", "000001"]
    assert manifest["timestamps_s"] == [0., .1]
    assert manifest["image_paths"] == ["sequences/00/image_2/000000.png", "sequences/00/image_2/000001.png"]
    for source in manifest["sources"]:
        assert not Path(source["path"]).is_absolute()
        assert source["size_bytes"] == (f.config.raw_root / source["path"]).stat().st_size
        assert source["sha256"] == digest(f.config.raw_root / source["path"])
    content_hash = manifest.pop("content_sha256")
    assert content_hash == hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert prepared.manifest_sha256 == digest(prepared.manifest_path)
    assert prepare_sequence(f.config, "00").manifest_sha256 == prepared.manifest_sha256

@pytest.mark.parametrize("source", ["image", "pose"])
def test_verify_detects_changed_source(kitti_fixture, source):
    f = kitti_fixture
    prepare_sequence(f.config, "00")
    p = f.images / "000000.png" if source == "image" else f.pose_file
    data = bytearray(p.read_bytes())
    data[-8] = (data[-8] + 1) % 255
    p.write_bytes(data)
    with pytest.raises(DatasetValidationError, match="SOURCE_CHANGED"):
        verify_prepared_sequence(f.config, "00")

@pytest.mark.parametrize("fault", ["hash", "ordering", "schema", "path", "poses", "calibration"])
def test_verify_detects_prepared_corruption(kitti_fixture, fault):
    f = kitti_fixture
    p = prepare_sequence(f.config, "00")
    m = json.loads(p.manifest_path.read_text())
    if fault == "hash": m["timestamps_s"][0] = 99
    elif fault == "ordering": m["frame_ids"].reverse()
    elif fault == "schema": m["schema_version"] = 2
    elif fault == "path": m["image_paths"][0] = "../../outside.png"
    elif fault == "poses":
        poses = p.poses_c2w.copy()
        poses[0, 0, 0] = 3
        np.save(p.manifest_path.parent / "poses_c2w.npy", poses)
    elif fault == "calibration":
        (p.manifest_path.parent / "calibration.json").write_text("{}")
    if fault in ("ordering", "schema", "path"):
        m.pop("content_sha256")
        m["content_sha256"] = hashlib.sha256(json.dumps(m, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    p.manifest_path.write_text(json.dumps(m))
    with pytest.raises(DatasetValidationError):
        verify_prepared_sequence(f.config, "00", verify_hashes=False)

def test_failed_atomic_write_cleans_temporary(kitti_fixture, monkeypatch):
    import kitti_eval.data as data
    def fail(*args, **kwargs): raise OSError("injected disk failure")
    monkeypatch.setattr(data.np, "save", fail)
    with pytest.raises(DatasetValidationError, match="PREPARE_IO"):
        prepare_sequence(kitti_fixture.config, "00")
    root = kitti_fixture.config.prepared_root
    assert not (root / "00").exists()
    assert not list(root.iterdir())

def test_doctor_and_subset_cli(kitti_fixture, capsys):
    from kitti_eval.cli import main
    f = kitti_fixture
    assert main(["doctor", "--config", str(f.config_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ready"] and output["sequences"]["00"]["ready"]
    assert output["paths"]["raw_root"] == str(f.config.raw_root)
    assert not f.config.prepared_root.exists()
    assert main(["prepare", "--config", str(f.config_path), "--sequence", "00"]) == 0
    capsys.readouterr()
    assert main(["verify", "--config", str(f.config_path), "--sequence", "00"]) == 0
    capsys.readouterr()
    assert main(["prepare", "--config", str(f.config_path), "--sequence", "01"]) == 1
    assert "INVALID_SEQUENCE" in capsys.readouterr().out


def test_calibration_overflow_fails_before_output(kitti_fixture):
    f = kitti_fixture
    f.calib.write_text("P0: 1 0 0 1e308 0 1 0 0 0 0 1 0\nP2: 1 0 0 -1e308 0 1 0 0 0 0 1 0\n")
    with pytest.raises(DatasetValidationError, match="INVALID_CALIBRATION"):
        prepare_sequence(f.config, "00")
    assert not f.config.prepared_root.exists()


def test_converted_pose_overflow_fails_before_output(kitti_fixture):
    f = kitti_fixture
    f.calib.write_text("P0: 1 0 0 0 0 1 0 0 0 0 1 0\nP2: 1 0 0 -1e308 0 1 0 0 0 0 1 0\n")
    f.poses[0, 0, 3] = 1e308
    np.savetxt(f.pose_file, f.poses[:, :3].reshape(2, 12))
    with pytest.raises(DatasetValidationError, match="INVALID_POSES"):
        prepare_sequence(f.config, "00")
    assert not f.config.prepared_root.exists()


def test_internal_image_symlink_preserves_logical_frame_path(kitti_fixture):
    f = kitti_fixture
    image = f.images / "000000.png"
    shared = f.config.raw_root / "shared.png"
    image.rename(shared)
    image.symlink_to(shared)
    prepared = prepare_sequence(f.config, "00")
    manifest = json.loads(prepared.manifest_path.read_text())
    assert manifest["sources"][0]["path"] == "sequences/00/image_2/000000.png"
    assert image.is_symlink()
    assert prepared.image_paths[0] == image
    np.testing.assert_allclose(prepared.poses_c2w[0, :3, 3], f.expected_camera2_center)

def test_split_color_and_aux_roots_use_direct_paths(kitti_fixture):
    from kitti_eval.config import load_config

    f = kitti_fixture
    raw = f.config.raw_root
    color_root = raw / "color-dataset"
    aux_root = raw / "aux-dataset"
    shutil.move(raw / "sequences", color_root / "sequences")
    shutil.move(raw / "poses", aux_root / "poses")
    calibration = color_root / "sequences/00/calib.txt"
    (aux_root / "sequences/00").mkdir(parents=True)
    shutil.move(calibration, aux_root / "sequences/00/calib.txt")

    values = json.loads(f.config_path.read_text())
    values["color_root"] = "raw/color-dataset"
    values["aux_root"] = "raw/aux-dataset"
    f.config_path.write_text(json.dumps(values))
    config = load_config(f.config_path)

    status = inspect_raw_sequence(config, "00")
    assert status.ready
    prepared = prepare_sequence(config, "00")
    manifest = json.loads(prepared.manifest_path.read_text())
    assert prepared.image_paths[0] == color_root / "sequences/00/image_2/000000.png"
    assert manifest["image_paths"] == [
        "color-dataset/sequences/00/image_2/000000.png",
        "color-dataset/sequences/00/image_2/000001.png",
    ]
    assert [item["path"] for item in manifest["sources"]][-3:] == [
        "aux-dataset/poses/00.txt",
        "aux-dataset/sequences/00/calib.txt",
        "color-dataset/sequences/00/times.txt",
    ]

from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from .conftest import make_fixture, MAIN
from virtual_kitti_eval.config import load_config, DatasetValidationError
from virtual_kitti_eval.data import inspect_raw_sequence, prepare_sequence, verify_prepared_sequence, make_backend_request
from virtual_kitti_eval.io import content_sha256

def codes(status):
    return {block.code for block in status.blockers}

def test_v203_rejected_before_layout_and_no_outputs(tmp_path):
    fixture = make_fixture(tmp_path, version="2.0.3")
    (fixture.archives / "vkitti_2.0.3_rgb.tar.part").write_bytes(b"partial")
    status = inspect_raw_sequence(fixture.config, "Scene01/Clone")
    assert not status.ready
    assert codes(status) == {"DATASET_VERSION_MISMATCH", "INCOMPLETE_RGB_ARCHIVE"}
    with pytest.raises(DatasetValidationError):
        prepare_sequence(fixture.config, "Scene01/Clone")
    assert not fixture.config.prepared_root.exists()

def test_unverified_layout_and_missing_version_are_blocked(vkitti_131_fixture):
    f = vkitti_131_fixture
    # A marker alone never licenses guessed Scene/Camera_0 layout.
    for path in f.images.glob("*.png"):
        path.unlink()
    f.images.rmdir()
    status = inspect_raw_sequence(f.config, "Scene01/Clone")
    assert "MISSING_RGB" in codes(status)
    unknown = f.root / "unverified"
    unknown.mkdir()
    (unknown / "VERSION").write_text("1.3.1")
    assert codes(inspect_raw_sequence(replace(f.config, raw_root=unknown), "Scene01/Clone")) == {"DATASET_LAYOUT_UNVERIFIED"}
    (unknown / "VERSION").unlink()
    assert codes(inspect_raw_sequence(replace(f.config, raw_root=unknown), "Scene01/Clone")) == {"DATASET_VERSION_MISMATCH"}

def test_prepare_orders_decodes_inverts_w2c_and_exposes_rgb_only(vkitti_131_fixture):
    f = vkitti_131_fixture
    prepared = prepare_sequence(f.config, "Scene01/Clone")
    assert prepared.frame_ids == ("00000", "00001", "00002", "00003")
    np.testing.assert_allclose(prepared.poses_c2w[:, :3, 3], f.centers, atol=1e-12)
    np.testing.assert_allclose(prepared.poses_c2w[0, :3, :3], [[0,1,0],[-1,0,0],[0,0,1]])
    assert prepared.timestamps_s is None
    assert not prepared.poses_c2w.flags.writeable
    with pytest.raises(ValueError):
        prepared.poses_c2w.setflags(write=True)
    request = make_backend_request(prepared)
    assert set(vars(request)) == {"frame_ids", "image_paths"}
    assert set(request.input_fields) == {"frame_ids", "image_paths"}
    assert all("_rgb/" in str(p) for p in request.image_paths)
    manifest = json.loads(prepared.manifest_path.read_text())
    assert not any(any(word in p["path"] for word in ("depth", "scenegt", "instance", "semantic")) for p in manifest["sources"])
    before = prepared.manifest_path.read_bytes()
    assert prepare_sequence(f.config, prepared.sequence).manifest_path.read_bytes() == before

@pytest.mark.parametrize("defect,expected", [
    ("gap", "INVALID_FRAME_ID"), ("corrupt", "INVALID_IMAGE"), ("mode", "INVALID_IMAGE"),
    ("count", "COUNT_MISMATCH"), ("duplicate", "INVALID_FRAME_ID"),
    ("unordered", "INVALID_FRAME_ID"), ("nan", "INVALID_POSES"),
    ("nonrigid", "INVALID_POSES"), ("reflection", "INVALID_POSES"),
    ("camera2", "INVALID_POSES")])
def test_raw_defects_block_before_preparation(vkitti_131_fixture, defect, expected):
    f = vkitti_131_fixture
    lines = f.gt.read_text().splitlines()
    if defect == "gap":
        (f.images / "00001.png").rename(f.images / "00010.png")
    elif defect == "corrupt":
        image = f.images / "00001.png"
        image.write_bytes(image.read_bytes()[:42])
    elif defect == "mode":
        Image.new("L", (12,6)).save(f.images / "00001.png")
    elif defect == "count":
        f.gt.write_text("\n".join(lines[:-1]))
    elif defect in ("duplicate", "unordered"):
        if defect == "duplicate":
            lines[2] = lines[1]
        else:
            lines[1], lines[2] = lines[2], lines[1]
        f.gt.write_text("\n".join(lines))
    else:
        row = lines[1].split()
        if defect == "nan": row[1] = "nan"
        if defect == "nonrigid": row[2] = "-2"
        if defect == "reflection": row[2] = "1"
        if defect == "camera2": row.insert(1, "0")
        lines[1] = " ".join(row)
        f.gt.write_text("\n".join(lines))
    assert expected in codes(inspect_raw_sequence(f.config, "Scene01/Clone"))
    with pytest.raises(DatasetValidationError):
        prepare_sequence(f.config, "Scene01/Clone")
    assert not f.config.prepared_root.exists()

def test_optional_scenes_only_explicit_and_formal_version_fixed(tmp_path):
    f = make_fixture(tmp_path, MAIN)
    assert f.config.sequence_ids == MAIN
    assert codes(inspect_raw_sequence(f.config, "Scene06/Clone")) == {"INVALID_SEQUENCE"}
    text = f.config_path.read_text()
    for key, value in (("dataset_version", "2.0.3"), ("camera", "Camera_1")):
        payload = json.loads(text); payload[key] = value
        f.config_path.write_text(json.dumps(payload))
        with pytest.raises(DatasetValidationError): load_config(f.config_path)
    f.config_path.write_text(text)
    f.config.sequences_file.write_text("Scene06/Clone\nScene18/Fog\nScene20/Rain\n")
    assert load_config(f.config_path).sequence_ids == ("Scene06/Clone", "Scene18/Fog", "Scene20/Rain")

@pytest.mark.parametrize("ids", ["Scene01/clone", "Scene03/Clone", "Scene01/Clone\nScene01/Clone", "../Clone"])
def test_invalid_sequence_lists_are_rejected(vkitti_131_fixture, ids):
    f = vkitti_131_fixture
    f.config.sequences_file.write_text(ids)
    with pytest.raises(DatasetValidationError): load_config(f.config_path)

def test_resume_rejects_raw_changed_and_rehashed_prepared_tampering(vkitti_131_fixture):
    f = vkitti_131_fixture
    p = prepare_sequence(f.config, "Scene01/Clone")
    Image.new("RGB", (12,6), (99,2,3)).save(f.images / "00000.png")
    with pytest.raises(DatasetValidationError): verify_prepared_sequence(f.config, p.sequence)
    with pytest.raises(DatasetValidationError): prepare_sequence(f.config, p.sequence)

def test_verify_rejects_rehashed_wrong_c2w(vkitti_131_fixture):
    from virtual_kitti_eval.io import file_record
    f = vkitti_131_fixture
    p = prepare_sequence(f.config, "Scene01/Clone")
    poses_path = p.manifest_path.parent / "poses_c2w.npy"
    array = np.load(poses_path)
    array[:, :3, 3] += 50
    np.save(poses_path, array)
    manifest = json.loads(p.manifest_path.read_text())
    manifest["prepared_files"] = [file_record(poses_path, poses_path.parent)]
    manifest.pop("content_sha256")
    manifest["content_sha256"] = content_sha256(manifest)
    p.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DatasetValidationError): verify_prepared_sequence(f.config, p.sequence)

def test_full_pixel_decode_rejects_png_with_valid_chunk_crc(vkitti_131_fixture):
    import struct
    import zlib
    f = vkitti_131_fixture
    path = f.images / "00001.png"
    raw = bytearray(path.read_bytes())
    offset = 8
    while offset < len(raw):
        size = struct.unpack(">I", raw[offset:offset + 4])[0]
        if raw[offset + 4:offset + 8] == b"IDAT":
            raw[offset + 8] = 0  # Invalid zlib header, but repair the PNG chunk CRC.
            crc = zlib.crc32(raw[offset + 4:offset + 8 + size])
            raw[offset + 8 + size:offset + 12 + size] = struct.pack(">I", crc)
            break
        offset += size + 12
    path.write_bytes(raw)
    with Image.open(path) as image:
        image.verify()  # Container integrity alone succeeds.
    assert "INVALID_IMAGE" in codes(inspect_raw_sequence(f.config, "Scene01/Clone"))

def test_manifest_strict_types_and_symlink_escape(vkitti_131_fixture):
    f = vkitti_131_fixture
    p = prepare_sequence(f.config, "Scene01/Clone")
    original = p.manifest_path.read_bytes()
    manifest = json.loads(original)
    manifest["schema_version"] = True
    p.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DatasetValidationError):
        verify_prepared_sequence(f.config, p.sequence)
    outside = f.root / "outside.json"
    outside.write_bytes(original)
    p.manifest_path.unlink()
    p.manifest_path.symlink_to(outside)
    with pytest.raises(DatasetValidationError):
        verify_prepared_sequence(f.config, p.sequence)


def test_prepared_complex_arrays_are_rejected_even_with_matching_values_and_hashes(vkitti_131_fixture):
    from virtual_kitti_eval.io import file_record
    f = vkitti_131_fixture
    p = prepare_sequence(f.config, "Scene01/Clone")
    path = p.manifest_path.parent / "poses_c2w.npy"
    np.save(path, np.load(path).astype(complex), allow_pickle=False)
    manifest = json.loads(p.manifest_path.read_text())
    manifest["prepared_files"] = [file_record(path, path.parent)]
    manifest.pop("content_sha256")
    manifest["content_sha256"] = content_sha256(manifest)
    p.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DatasetValidationError):
        verify_prepared_sequence(f.config, p.sequence)


@pytest.mark.parametrize("damaged",[False,True])
def test_prepared_pose_npz_container_is_rejected_as_structured_failure(vkitti_131_fixture,damaged):
    import hashlib
    f = vkitti_131_fixture
    prepared = prepare_sequence(f.config,"Scene01/Clone")
    path = prepared.manifest_path.parent/"poses_c2w.npy"
    with path.open("wb") as stream: np.savez_compressed(stream,poses_c2w=prepared.poses_c2w)
    if damaged: path.write_bytes(path.read_bytes()[:-12])
    manifest = json.loads(prepared.manifest_path.read_text())
    manifest["prepared_files"][0].update(size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest.pop("content_sha256")
    manifest["content_sha256"] = content_sha256(manifest)
    prepared.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DatasetValidationError):
        verify_prepared_sequence(f.config,"Scene01/Clone")

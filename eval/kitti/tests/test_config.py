import json
from pathlib import Path
import pytest
from kitti_eval.config import load_config, read_sequence_ids, DatasetValidationError

@pytest.mark.parametrize("text", ["0\n", "000\n", "AA\n", "00\n00\n", "01\n00\n", "", "００\n"])
def test_bad_sequence_lists(tmp_path, text):
    p = tmp_path / "seq.txt"
    p.write_text(text)
    with pytest.raises(DatasetValidationError, match="INVALID_SEQUENCE"):
        read_sequence_ids(p)

def test_canonical_sequence_list():
    assert read_sequence_ids(Path(__file__).parents[1] / "configs/sequences.txt") == tuple(f"{i:02d}" for i in range(11))

def test_resolved_config(kitti_fixture):
    config = kitti_fixture.config
    assert config.schema_version == 1
    assert config.raw_root == kitti_fixture.images.parents[2]
    assert config.sequence_ids == ("00",)
    assert not config.prepared_root.exists()

@pytest.mark.parametrize("change", [{"schema_version": 2}, {"schema_version": True}, {"raw_root": ""}, {"raw_root": 3}, {"surprise": 1}, {"prepared_root": "raw/out"}, {"prepared_root": "archives/out"}])
def test_invalid_config(kitti_fixture, change):
    p = kitti_fixture.config_path
    data = json.loads(p.read_text())
    data.update(change)
    p.write_text(json.dumps(data))
    with pytest.raises(DatasetValidationError, match="INVALID_CONFIG"):
        load_config(p)

def test_h20_config():
    config = load_config(Path(__file__).parents[1] / "configs/h20.json")
    assert str(config.raw_root) == "/data/yjh/share/datasets/KITTI_Odometry/extracted"
    assert str(config.color_root) == "/data/yjh/share/datasets/KITTI_Odometry/extracted/data_odometry_color/dataset"
    assert str(config.aux_root) == "/data/yjh/share/datasets/KITTI_Odometry/extracted/aux/dataset"
    assert str(config.archive_root) == "/data/yjh/share/datasets/KITTI_Odometry/archives"
    assert str(config.prepared_root) == "/data/yjh/share/datasets/KITTI_Odometry/prepared_vggtlong_v1"

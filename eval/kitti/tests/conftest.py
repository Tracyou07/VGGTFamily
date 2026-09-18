from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from PIL import Image

@pytest.fixture
def kitti_fixture(tmp_path):
    from kitti_eval.config import load_config
    raw = tmp_path / "raw"
    images = raw / "sequences/00/image_2"
    images.mkdir(parents=True)
    (raw / "poses").mkdir()
    (tmp_path / "archives").mkdir()
    for i in range(2):
        Image.new("RGB", (16, 12), (20 + i, 40, 60)).save(images / f"{i:06d}.png")
    k = np.array([[100, 0, 8], [0, 100, 6], [0, 0, 1.]])
    # Camera 0 center is 0.1 in rectified reference; camera 2 center is 0.6.
    projections = [k @ np.column_stack([np.eye(3), [-c, 0, 0]]) for c in (0.1, 0.6)]
    calib = images.parent / "calib.txt"
    calib.write_text("\n".join(f"P{i}: " + " ".join(map(str, p.ravel())) for i, p in zip((0, 2), projections)) + "\n")
    times = images.parent / "times.txt"
    times.write_text("0.0\n0.1\n")
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[0, :3, 3] = [1, 2, 3]
    poses[1, :3, :3] = [[0,-1,0],[1,0,0],[0,0,1]]
    pose_file = raw / "poses/00.txt"
    np.savetxt(pose_file, poses[:, :3].reshape(2, 12))
    (tmp_path / "sequences.txt").write_text("00\n")
    values = dict(schema_version=1, raw_root="raw", archive_root="archives",
                  prepared_root="prepared", sequences_file="sequences.txt")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(values))
    config = load_config(config_path)
    return SimpleNamespace(config=config, config_path=config_path, images=images,
        calib=calib, times=times, pose_file=pose_file, poses=poses,
        expected_camera2_center=np.array([1.5, 2, 3]))

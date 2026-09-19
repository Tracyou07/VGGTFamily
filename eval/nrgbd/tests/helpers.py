from pathlib import Path
import numpy as np
from PIL import Image

SCENES = (
    "breakfast_room",
    "complete_kitchen",
    "green_room",
    "grey_white_room",
    "kitchen",
    "morning_apartment",
    "staircase",
    "thin_geometry",
    "whiteroom",
)


def make_dataset(root: Path, frames=21):
    for scene in SCENES:
        d = root / scene
        (d / "images").mkdir(parents=True)
        (d / "depth").mkdir()
        poses = []
        for i in range(frames):
            Image.fromarray(np.full((6, 8, 3), i % 255, np.uint8)).save(
                d / "images" / f"img{i}.png"
            )
            Image.fromarray(np.full((6, 8), 1000 + i, np.uint16)).save(
                d / "depth" / f"depth{i}.png"
            )
            pose = np.eye(4)
            pose[0, 3] = i
            poses.extend(" ".join(map(str, row)) + "\n" for row in pose)
        (d / "poses.txt").write_text("".join(poses))
    (root / "archives").mkdir()

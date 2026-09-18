from __future__ import annotations

import io
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scannet_eval.sens import SensFormatError, SensReader, select_frame_ids


def _encoded_image(fmt: str, color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (2, 2), color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def write_sens(
    path: Path,
    *,
    color_format: str = "JPEG",
    poses: list[np.ndarray] | None = None,
    imu_count: int = 0,
) -> list[bytes]:
    poses = poses or [np.eye(4, dtype=np.float32)]
    compression = {"PNG": 1, "JPEG": 2}[color_format]
    colors = [_encoded_image(color_format, (20 + i, 40, 60)) for i in range(len(poses))]
    name = b"synthetic"
    intrinsic = np.array(
        [[100.0, 0.0, 1.0, 0.0], [0.0, 110.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype="<f4",
    )
    with path.open("wb") as stream:
        stream.write(struct.pack("<I", 4))
        stream.write(struct.pack("<Q", len(name)))
        stream.write(name)
        for matrix in (intrinsic, np.eye(4), intrinsic, np.eye(4)):
            stream.write(np.asarray(matrix, dtype="<f4").tobytes())
        stream.write(struct.pack("<iiIIII f Q", compression, 0, 2, 2, 2, 2, 1000.0, len(poses)))
        for index, (pose, payload) in enumerate(zip(poses, colors, strict=True)):
            depth = struct.pack("<4H", 100, 101, 102, 103)
            stream.write(np.asarray(pose, dtype="<f4").tobytes())
            stream.write(struct.pack("<QQQQ", index * 2, index * 2 + 1, len(payload), len(depth)))
            stream.write(payload)
            stream.write(depth)
        if imu_count:
            stream.write(struct.pack("<Q", imu_count))
            stream.write(bytes(128 * imu_count))
    return colors


class SensReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_reader_indexes_payloads_and_preserves_encoded_color(self) -> None:
        path = self.root / "scene.sens"
        payloads = write_sens(path, poses=[np.eye(4), np.eye(4)])

        reader = SensReader(path)

        self.assertEqual(reader.header.version, 4)
        self.assertEqual(reader.header.sensor_name, "synthetic")
        self.assertEqual(reader.header.color_size, (2, 2))
        self.assertEqual(reader.header.color_extension, ".jpg")
        self.assertEqual(len(reader.frames), 2)
        self.assertEqual(reader.frames[1].frame_id, 1)
        self.assertEqual(reader.read_color_payload(reader.frames[1]), payloads[1])
        self.assertEqual(list(reader.iter_color_payloads(reader.frames)), payloads)

    def test_reader_supports_png_color_payloads(self) -> None:
        path = self.root / "scene.sens"
        payloads = write_sens(path, color_format="PNG")

        reader = SensReader(path)

        self.assertEqual(reader.header.color_extension, ".png")
        self.assertEqual(reader.read_color_payload(reader.frames[0]), payloads[0])

    def test_reader_rejects_truncated_header_or_payload(self) -> None:
        complete = self.root / "complete.sens"
        write_sens(complete)
        data = complete.read_bytes()
        for cut in (2, 12, -1):
            with self.subTest(cut=cut):
                truncated = self.root / f"truncated-{cut}.sens"
                truncated.write_bytes(data[:cut] if cut >= 0 else data[:-1])
                with self.assertRaisesRegex(SensFormatError, "truncated"):
                    SensReader(truncated)

    def test_reader_rejects_wrong_version(self) -> None:
        path = self.root / "scene.sens"
        path.write_bytes(struct.pack("<I", 3))

        with self.assertRaisesRegex(SensFormatError, "version 4"):
            SensReader(path)

    def test_reader_accepts_exact_scan_net_imu_trailer(self) -> None:
        path = self.root / "scene.sens"
        write_sens(path, imu_count=2)

        reader = SensReader(path)

        self.assertEqual(reader.imu_frame_count, 2)

    def test_select_frame_ids_filters_nonfinite_poses_and_matches_fastvggt(self) -> None:
        poses = []
        for index in range(6):
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = index
            poses.append(pose)
        poses[1][2, 3] = np.nan
        path = self.root / "scene.sens"
        write_sens(path, poses=poses)
        reader = SensReader(path)

        all_valid = select_frame_ids(reader.frames)
        selected = select_frame_ids(reader.frames, max_frames=3)

        self.assertEqual(all_valid, (0, 2, 3, 4, 5))
        self.assertEqual(selected, (0, 2, 4))

    def test_select_frame_ids_validates_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_frames"):
            select_frame_ids((), max_frames=0)


if __name__ == "__main__":
    unittest.main()

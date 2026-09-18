from __future__ import annotations

import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scannet_eval.data import (
    DatasetValidationError,
    _verify_image_bytes,
    load_scene,
    prepare_dataset,
    read_scene_list,
    validate_dataset,
)


def _jpeg(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_raw_scene(raw_root: Path, scene_id: str, *, frame_count: int = 6) -> list[bytes]:
    sens_dir = raw_root / "raw_sens" / "scans" / scene_id
    mesh_dir = raw_root / "raw" / "scans" / scene_id
    sens_dir.mkdir(parents=True)
    mesh_dir.mkdir(parents=True)
    payloads = [_jpeg((20 + i, 40, 60)) for i in range(frame_count)]
    intrinsic = np.array(
        [[100.0, 0.0, 1.0, 0.0], [0.0, 110.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype="<f4",
    )
    name = b"synthetic"
    with (sens_dir / f"{scene_id}.sens").open("wb") as stream:
        stream.write(struct.pack("<IQ", 4, len(name)))
        stream.write(name)
        for matrix in (intrinsic, np.eye(4), intrinsic, np.eye(4)):
            stream.write(np.asarray(matrix, dtype="<f4").tobytes())
        stream.write(struct.pack("<iiIIII f Q", 2, 0, 2, 2, 2, 2, 1000.0, frame_count))
        for index, payload in enumerate(payloads):
            pose = np.eye(4, dtype="<f4")
            pose[0, 3] = index
            if index == 1:
                pose[2, 3] = np.inf
            depth = struct.pack("<4H", 1, 2, 3, 4)
            stream.write(pose.tobytes())
            stream.write(struct.pack("<QQQQ", index, index, len(payload), len(depth)))
            stream.write(payload)
            stream.write(depth)
    ply = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n"
    (mesh_dir / f"{scene_id}_vh_clean_2.ply").write_bytes(ply)
    return payloads


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _tree_snapshot(root: Path) -> dict[str, tuple[str, bytes | str]]:
    snapshot: dict[str, tuple[str, bytes | str]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = ("symlink", str(path.readlink()))
        elif path.is_dir():
            snapshot[relative] = ("directory", "")
        else:
            snapshot[relative] = ("file", path.read_bytes())
    return snapshot


class DataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_read_scene_list_ignores_blank_lines_and_comments(self) -> None:
        scene_list = self.root / "scenes.txt"
        scene_list.write_text("# evaluation subset\n scene0000_00 \n\nscene0001_01 # note\n", encoding="utf-8")

        self.assertEqual(read_scene_list(scene_list), ["scene0000_00", "scene0001_01"])

    def test_read_scene_list_rejects_duplicates_and_unsafe_ids(self) -> None:
        duplicate = self.root / "duplicate.txt"
        duplicate.write_text("scene0000_00\nscene0000_00\n", encoding="utf-8")
        unsafe = self.root / "unsafe.txt"
        unsafe.write_text("../scene0000_00\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_scene_list(duplicate)
        with self.assertRaisesRegex(ValueError, "invalid scene id"):
            read_scene_list(unsafe)

    def test_image_validation_rejects_truncated_jpeg_pixel_payload(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (90, 130, 170)).save(buffer, format="JPEG")
        payload = buffer.getvalue()[:-10]
        with Image.open(io.BytesIO(payload)) as image:
            self.assertEqual(image.size, (32, 32))
            image.verify()

        with self.assertRaisesRegex(DatasetValidationError, "failed to decode"):
            _verify_image_bytes(payload, (32, 32), 7)

    def test_prepare_exports_selected_original_frames_and_complete_hashes(self) -> None:
        raw_root = self.root / "raw-root"
        output_root = self.root / "prepared"
        payloads = _write_raw_scene(raw_root, "scene0000_00")

        result = prepare_dataset(raw_root, output_root, ["scene0000_00"], workers=1)

        self.assertEqual(result["prepared"], 1)
        self.assertEqual(result["reused"], 0)
        scene = load_scene(output_root, "scene0000_00", max_frames=3)
        self.assertEqual(scene.frame_ids, (0, 2, 4))
        self.assertEqual(scene.poses_c2w.shape, (3, 4, 4))
        np.testing.assert_allclose(scene.poses_c2w[:, 0, 3], [0.0, 2.0, 4.0])
        np.testing.assert_allclose(scene.intrinsics_color, [[100.0, 0.0, 1.0], [0.0, 110.0, 1.0], [0.0, 0.0, 1.0]])
        self.assertTrue(scene.gt_ply.read_bytes().startswith(b"ply\n"))
        self.assertEqual([path.read_bytes() for path in scene.image_paths], [payloads[0], payloads[2], payloads[4]])

        manifest_path = output_root / "scene0000_00" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(scene.manifest_sha256, _sha256(manifest_path))
        self.assertEqual(set(manifest["files"]), {
            "calibration/extrinsics_color.txt",
            "calibration/extrinsics_depth.txt",
            "calibration/intrinsics_color.txt",
            "calibration/intrinsics_depth.txt",
            "calibration/sensor.json",
            "gt/scene0000_00_vh_clean_2.ply",
            "color/000000.jpg",
            "color/000002.jpg",
            "color/000003.jpg",
            "color/000004.jpg",
            "color/000005.jpg",
            "pose/000000.txt",
            "pose/000002.txt",
            "pose/000003.txt",
            "pose/000004.txt",
            "pose/000005.txt",
        })
        self.assertEqual(manifest["frame_ids"], [0, 2, 3, 4, 5])
        self.assertEqual(manifest["frame_selection"], {
            "policy": "fastvggt_build_frame_selection",
            "preparation_cap": None,
        })
        for relative, expected_hash in manifest["files"].items():
            self.assertEqual(_sha256(output_root / "scene0000_00" / relative), expected_hash)
        self.assertEqual(validate_dataset(output_root, ["scene0000_00"])["valid_scenes"], 1)

    def test_validation_rejects_missing_corrupt_and_undecodable_outputs(self) -> None:
        raw_root = self.root / "raw-root"
        output_root = self.root / "prepared"
        _write_raw_scene(raw_root, "scene0000_00")
        prepare_dataset(raw_root, output_root, ["scene0000_00"], workers=1)
        scene_dir = output_root / "scene0000_00"

        (scene_dir / "pose" / "000003.txt").unlink()
        with self.assertRaisesRegex(DatasetValidationError, "missing"):
            validate_dataset(output_root, ["scene0000_00"])

        prepare_dataset(raw_root, output_root, ["scene0000_00"], workers=1)
        image = scene_dir / "color" / "000003.jpg"
        image.write_bytes(b"not an image")
        with self.assertRaisesRegex(DatasetValidationError, "hash"):
            validate_dataset(output_root, ["scene0000_00"])
        manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["files"]["color/000003.jpg"] = _sha256(image)
        (scene_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(DatasetValidationError, "decode"):
            validate_dataset(output_root, ["scene0000_00"], verify_hashes=False)

    def test_validation_rejects_invalid_sensor_metadata_without_hash_checks(self) -> None:
        raw_root = self.root / "raw-root"
        output_root = self.root / "prepared"
        _write_raw_scene(raw_root, "scene0000_00")
        prepare_dataset(raw_root, output_root, ["scene0000_00"], workers=1)
        sensor = output_root / "scene0000_00" / "calibration" / "sensor.json"
        sensor.write_text("not-json", encoding="utf-8")

        with self.assertRaisesRegex(DatasetValidationError, "sensor metadata"):
            validate_dataset(output_root, ["scene0000_00"], verify_hashes=False)

    def test_prepare_recovers_incomplete_cache_and_applies_runtime_cap(self) -> None:
        raw_root = self.root / "raw-root"
        output_root = self.root / "prepared"
        _write_raw_scene(raw_root, "scene0000_00")
        prepare_dataset(raw_root, output_root, ["scene0000_00"], max_frames=3, workers=1)
        scene_dir = output_root / "scene0000_00"
        (scene_dir / "pose" / "000002.txt").unlink()
        (scene_dir / "stale.marker").write_text("partial", encoding="utf-8")

        recovered = prepare_dataset(raw_root, output_root, ["scene0000_00"], max_frames=3, workers=1)

        self.assertEqual(recovered["prepared"], 1)
        self.assertFalse((scene_dir / "stale.marker").exists())
        self.assertEqual(validate_dataset(output_root, ["scene0000_00"])["valid_scenes"], 1)
        self.assertEqual(load_scene(output_root, "scene0000_00", max_frames=2).frame_ids, (0, 2))

        smaller = prepare_dataset(raw_root, output_root, ["scene0000_00"], max_frames=2, workers=1)
        self.assertEqual(smaller["prepared"], 1)
        self.assertEqual(load_scene(output_root, "scene0000_00", max_frames=2).frame_ids, (0, 2))

    def test_prepare_reuses_only_a_complete_matching_cache(self) -> None:
        raw_root = self.root / "raw-root"
        output_root = self.root / "prepared"
        _write_raw_scene(raw_root, "scene0000_00")
        prepare_dataset(raw_root, output_root, ["scene0000_00"], workers=1)

        result = prepare_dataset(raw_root, output_root, ["scene0000_00"], workers=1)

        self.assertEqual(result["prepared"], 0)
        self.assertEqual(result["reused"], 1)

    def test_prepare_rejects_direct_raw_scene_output_collisions_without_mutation(self) -> None:
        for output_subpath in (("raw", "scans"), ("raw_sens", "scans")):
            with self.subTest(output_subpath=output_subpath):
                case_root = self.root / "-".join(output_subpath)
                raw_root = case_root / "raw-root"
                _write_raw_scene(raw_root, "scene0000_00")
                before = _tree_snapshot(raw_root)
                error = None

                try:
                    prepare_dataset(
                        raw_root,
                        raw_root.joinpath(*output_subpath),
                        ["scene0000_00"],
                        max_frames=1,
                        workers=1,
                    )
                except Exception as caught:  # noqa: BLE001 - assert the public failure below
                    error = caught

                self.assertEqual(_tree_snapshot(raw_root), before)
                self.assertIsInstance(error, ValueError)
                self.assertRegex(str(error), "overlap")

    def test_prepare_rejects_symlink_alias_to_raw_scenes_without_mutation(self) -> None:
        raw_root = self.root / "raw-root"
        _write_raw_scene(raw_root, "scene0000_00")
        alias = self.root / "raw-scans-alias"
        alias.symlink_to(raw_root / "raw" / "scans", target_is_directory=True)
        before = _tree_snapshot(raw_root)
        error = None

        try:
            prepare_dataset(
                raw_root,
                alias,
                ["scene0000_00"],
                max_frames=1,
                workers=1,
            )
        except Exception as caught:  # noqa: BLE001 - assert the public failure below
            error = caught

        self.assertEqual(_tree_snapshot(raw_root), before)
        self.assertIsInstance(error, ValueError)
        self.assertRegex(str(error), "overlap")

    def test_prepare_preflights_all_scenes_before_later_symlink_collision(self) -> None:
        raw_root = self.root / "raw-root"
        _write_raw_scene(raw_root, "scene0000_00")
        _write_raw_scene(raw_root, "scene0001_00")
        output_root = self.root / "prepared"
        output_root.mkdir()
        (output_root / "scene0001_00").symlink_to(
            raw_root / "raw" / "scans" / "scene0001_00",
            target_is_directory=True,
        )
        raw_before = _tree_snapshot(raw_root)
        output_before = _tree_snapshot(output_root)
        error = None

        try:
            prepare_dataset(
                raw_root,
                output_root,
                ["scene0000_00", "scene0001_00"],
                max_frames=1,
                workers=1,
            )
        except Exception as caught:  # noqa: BLE001 - assert the public failure below
            error = caught

        self.assertEqual(_tree_snapshot(raw_root), raw_before)
        self.assertEqual(_tree_snapshot(output_root), output_before)
        self.assertIsInstance(error, ValueError)
        self.assertRegex(str(error), "overlap")

    def test_prepare_rejects_cross_scene_source_alias_without_mutation(self) -> None:
        raw_root = self.root / "raw-root"
        _write_raw_scene(raw_root, "scene0000_00")
        _write_raw_scene(raw_root, "scene0001_00")
        output_root = self.root / "prepared"
        output_root.mkdir()
        (output_root / "scene0000_00").symlink_to(
            raw_root / "raw_sens" / "scans" / "scene0001_00",
            target_is_directory=True,
        )
        raw_before = _tree_snapshot(raw_root)
        output_before = _tree_snapshot(output_root)
        error = None

        try:
            prepare_dataset(
                raw_root,
                output_root,
                ["scene0000_00", "scene0001_00"],
                max_frames=1,
                workers=1,
            )
        except Exception as caught:  # noqa: BLE001 - assert the public failure below
            error = caught

        self.assertEqual(_tree_snapshot(raw_root), raw_before)
        self.assertEqual(_tree_snapshot(output_root), output_before)
        self.assertIsInstance(error, ValueError)
        self.assertRegex(str(error), "overlap")

    def test_prepare_allows_sibling_cache_inside_raw_root(self) -> None:
        raw_root = self.root / "raw-root"
        _write_raw_scene(raw_root, "scene0000_00")
        output_root = raw_root / "prepared_scannet50_v1"

        result = prepare_dataset(
            raw_root,
            output_root,
            ["scene0000_00"],
            max_frames=1,
            workers=1,
        )

        self.assertEqual(result["prepared"], 1)
        self.assertEqual(
            load_scene(output_root, "scene0000_00", max_frames=1).frame_ids,
            (0,),
        )


if __name__ == "__main__":
    unittest.main()

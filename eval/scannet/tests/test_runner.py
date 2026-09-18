from __future__ import annotations
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from scannet_eval.runner import (
    OutputConflictError,
    RunPreflightError,
    _prepare_output,
    _runtime_fingerprint,
    aggregate_run,
    run_evaluation,
)

METRIC_KEYS = {
    "chamfer_distance",
    "ate",
    "are",
    "rpe_rot",
    "rpe_trans",
    "inference_time_ms",
}
SCENE_METRIC_KEYS = METRIC_KEYS | {
    "scale_factor",
    "aligned_chamfer_distance",
    "aligned_ate",
    "aligned_are",
    "aligned_rpe_rot",
    "aligned_rpe_trans",
    "aligned_scale_factor",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jpeg_bytes(size, color):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_scene(root, scene_id, offset=0.0):
    scene = Path(root) / scene_id
    for name in ("color", "pose", "calibration", "gt"):
        (scene / name).mkdir(parents=True, exist_ok=True)
    ids = (0, 2, 5, 9)
    positions = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1.5, 1.5, 1))
    for index, frame_id in enumerate(ids):
        Image.new("RGB", (2, 2), (index * 30, 20, 10)).save(
            scene / "color" / f"{frame_id:06d}.jpg"
        )
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(positions[index]) + (offset, 0, 0)
        np.savetxt(scene / "pose" / f"{frame_id:06d}.txt", pose)
    for name in (
        "intrinsics_color",
        "extrinsics_color",
        "intrinsics_depth",
        "extrinsics_depth",
    ):
        np.savetxt(scene / "calibration" / f"{name}.txt", np.eye(4))
    (scene / "calibration" / "sensor.json").write_text(
        json.dumps(
            {
                "color_size": [2, 2],
                "depth_size": [2, 2],
                "color_compression": 2,
                "depth_compression": 1,
                "depth_shift": 1000.0,
            }
        )
    )
    pts = [
        [-1 + offset, -0.5, 0],
        [1 + offset, -0.5, 0.2],
        [-0.5 + offset, 1, 0.5],
        [0.8 + offset, 0.9, 1.2],
        [offset, 0.1, 0.7],
    ]
    ply = scene / "gt" / f"{scene_id}_vh_clean_2.ply"
    ply.write_text(
        "ply\nformat ascii 1.0\n"
        + f"element vertex {len(pts)}\n"
        + "property float x\nproperty float y\nproperty float z\nend_header\n"
        + "\n".join(" ".join(map(str, p)) for p in pts)
        + "\n"
    )
    files = {
        p.relative_to(scene).as_posix(): _sha256(p)
        for p in scene.rglob("*")
        if p.is_file()
    }
    (scene / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene_id": scene_id,
                "frame_ids": list(ids),
                "frame_selection": {
                    "policy": "fastvggt_build_frame_selection",
                    "preparation_cap": None,
                },
                "color_extension": ".jpg",
                "color_size": [2, 2],
                "source": {
                    "sens_size": 1,
                    "sens_mtime_ns": 1,
                    "gt_size": ply.stat().st_size,
                    "gt_sha256": _sha256(ply),
                },
                "files": files,
            },
            sort_keys=True,
        )
    )
    return scene


class FakeBackend:
    def __init__(self, fail=(), metadata=None):
        self.fail = set(fail)
        self.metadata = {} if metadata is None else metadata
        self.calls = []

    def predict(self, scene, work_dir):
        self.calls.append(scene.scene_id)
        if scene.scene_id in self.fail:
            raise RuntimeError("deliberate failure " + scene.scene_id)
        points = []
        body = False
        for line in Path(scene.gt_ply).read_text().splitlines():
            if body:
                points.append([float(x) for x in line.split()])
            elif line == "end_header":
                body = True
        return SimpleNamespace(
            points=np.asarray(points),
            poses_c2w=scene.poses_c2w.copy(),
            frame_ids=scene.frame_ids,
            inference_seconds=0.125,
            peak_allocated_bytes=1024,
            peak_reserved_bytes=2048,
            metadata=self.metadata,
        )


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.a, self.b = "scene0000_00", "scene0001_00"
        _write_scene(self.prepared, self.a)
        _write_scene(self.prepared, self.b, 0.2)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "model.py").write_text("VERSION=1\n")
        self.ckpt = self.root / "model.safetensors"
        self.ckpt.write_bytes(b"checkpoint")
        self.list = self.root / "scenes.txt"
        self.list.write_text(f"{self.a}\n{self.b}\n")
        self.config = self.root / "config.json"
        self.config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "prepared_root": str(self.prepared),
                    "raw_root": str(self.root / "raw"),
                    "scene_list": str(self.list),
                    "models": {
                        "test": {
                            "python": sys.executable,
                            "project_root": str(self.source),
                            "checkpoint": str(self.ckpt),
                            "image_size": 518,
                            "max_points": 0,
                        }
                    },
                }
            )
        )

    def tearDown(self):
        self.t.cleanup()

    @staticmethod
    def factory(backend):
        calls = []

        def make(name, config, device):
            calls.append((name, config, device))
            return backend

        make.calls = calls
        return make

    def test_preflights_every_scene_before_factory(self):
        factory = self.factory(FakeBackend())
        with self.assertRaisesRegex(RunPreflightError, "scene9999_00"):
            run_evaluation(
                self.config,
                "test",
                self.root / "out",
                [self.a, "scene9999_00"],
                backend_factory=factory,
            )
        self.assertEqual(factory.calls, [])

    def test_truncated_cached_jpeg_is_rejected_before_factory(self):
        scene = self.prepared / self.a
        color_paths = sorted((scene / "color").glob("*.jpg"))
        for index, image_path in enumerate(color_paths):
            payload = _jpeg_bytes((32, 32), (90 + index, 130, 170))
            if index == 0:
                payload = payload[:-10]
            image_path.write_bytes(payload)

        sensor_path = scene / "calibration" / "sensor.json"
        sensor = json.loads(sensor_path.read_text())
        sensor["color_size"] = [32, 32]
        sensor_path.write_text(json.dumps(sensor, sort_keys=True))

        manifest_path = scene / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["color_size"] = [32, 32]
        for path in (*color_paths, sensor_path):
            relative = path.relative_to(scene).as_posix()
            manifest["files"][relative] = _sha256(path)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))

        def allocation_sentinel_factory(name, config, device):
            raise AssertionError("backend allocation reached")

        with self.assertRaisesRegex(RunPreflightError, "image failed to decode"):
            run_evaluation(
                self.config,
                "test",
                self.root / "truncated",
                [self.a],
                backend_factory=allocation_sentinel_factory,
            )

    def test_success_preserves_upstream_metrics_and_binds_primary_aggregate_provenance(
        self,
    ):
        backend = FakeBackend(
            metadata={"array": np.array([1, 2]), "scalar": np.float32(0.25)}
        )
        out = self.root / "out"
        summary = run_evaluation(
            self.config,
            "test",
            out,
            [self.a],
            device="cpu",
            backend_factory=self.factory(backend),
        )
        self.assertTrue(summary["complete"])
        self.assertEqual(
            (
                summary["expected_count"],
                summary["success_count"],
                summary["failed_count"],
            ),
            (1, 1, 0),
        )
        record = json.loads((out / self.a / "result.json").read_text())
        self.assertTrue(METRIC_KEYS.issubset(record["metrics"]))
        self.assertIn("scale_factor", record["metrics"])
        self.assertEqual(record["prediction"]["metadata"]["array"], [1, 2])
        provenance = json.loads((out / "run_manifest.json").read_text())
        self.assertEqual(provenance["protocol"]["id"], "fastvggt_scannet_evo132")
        self.assertEqual(provenance["scenes"][0]["selected_frame_ids"], [0, 2, 5, 9])
        self.assertEqual(
            provenance["model"]["checkpoint"]["sha256"], _sha256(self.ckpt)
        )
        self.assertIn("source_fingerprint", provenance["model"])
        self.assertIn("interpreter", provenance["environment"])

    def test_failure_then_resume_preserves_completed_record(self):
        out = self.root / "partial"
        first = run_evaluation(
            self.config,
            "test",
            out,
            [self.a, self.b],
            backend_factory=self.factory(FakeBackend([self.b])),
        )
        self.assertFalse(first["complete"])
        self.assertEqual((first["success_count"], first["failed_count"]), (1, 1))
        self.assertTrue((out / self.a / "result.json").is_file())
        self.assertIn(
            "deliberate failure",
            json.loads((out / "failures" / f"{self.b}.json").read_text())["error"],
        )
        backend = FakeBackend()
        resumed = run_evaluation(
            self.config,
            "test",
            out,
            [self.a, self.b],
            resume=True,
            backend_factory=self.factory(backend),
        )
        self.assertTrue(resumed["complete"])
        self.assertEqual(backend.calls, [self.b])
        aggregate = aggregate_run(out)
        self.assertEqual(
            (aggregate["success_count"], aggregate["failed_count"]), (2, 0)
        )

    def test_changed_provenance_and_unknown_output_are_never_overwritten(self):
        out = self.root / "out"
        run_evaluation(
            self.config,
            "test",
            out,
            [self.a],
            backend_factory=self.factory(FakeBackend()),
        )
        self.ckpt.write_bytes(b"changed")
        factory = self.factory(FakeBackend())
        with self.assertRaisesRegex(OutputConflictError, "provenance"):
            run_evaluation(
                self.config, "test", out, [self.a], resume=True, backend_factory=factory
            )
        self.assertEqual(factory.calls, [])
        unknown = self.root / "unknown"
        unknown.mkdir()
        marker = unknown / "keep.txt"
        marker.write_text("keep")
        with self.assertRaisesRegex(OutputConflictError, "unknown"):
            run_evaluation(
                self.config,
                "test",
                unknown,
                [self.a],
                backend_factory=self.factory(FakeBackend()),
            )
        self.assertEqual(marker.read_text(), "keep")

    def test_gt_override_hash_is_preflighted_and_recorded(self):
        gtroot = self.root / "override"
        targetdir = gtroot / self.a
        targetdir.mkdir(parents=True)
        expected = self.prepared / self.a / "gt" / f"{self.a}_vh_clean_2.ply"
        target = targetdir / expected.name
        target.write_bytes(expected.read_bytes())
        summary = run_evaluation(
            self.config,
            "test",
            self.root / "good",
            [self.a],
            gt_ply_dir=gtroot,
            backend_factory=self.factory(FakeBackend()),
        )
        self.assertEqual(
            summary["provenance"]["overrides"]["gt_ply_dir"], str(gtroot.resolve())
        )
        target.write_bytes(b"ply\nwrong")
        factory = self.factory(FakeBackend())
        with self.assertRaisesRegex(RunPreflightError, "GT override hash"):
            run_evaluation(
                self.config,
                "test",
                self.root / "bad",
                [self.a],
                gt_ply_dir=gtroot,
                backend_factory=factory,
            )
        self.assertEqual(factory.calls, [])

    def test_nonfinite_metadata_becomes_strict_failure(self):
        out = self.root / "nonfinite"
        summary = run_evaluation(
            self.config,
            "test",
            out,
            [self.a],
            backend_factory=self.factory(FakeBackend(metadata={"bad": float("nan")})),
        )
        self.assertFalse(summary["complete"])
        text = (out / "failures" / f"{self.a}.json").read_text()
        self.assertNotIn("NaN", text)
        self.assertIn("non-finite", text)
        with self.assertRaisesRegex(RunPreflightError, "zero successful"):
            aggregate_run(out)

    def test_invalid_chamfer_limit_is_rejected_before_factory(self):
        factory = self.factory(FakeBackend())
        with self.assertRaisesRegex(RunPreflightError, "chamfer_max_dist"):
            run_evaluation(
                self.config,
                "test",
                self.root / "invalid",
                [self.a],
                chamfer_max_dist=float("nan"),
                backend_factory=factory,
            )
        self.assertEqual(factory.calls, [])
        self.assertFalse((self.root / "invalid").exists())

    def test_uncommitted_runtime_source_change_invalidates_resume(self):
        runtime = self.root / "runtime"
        (runtime / "scannet_eval" / "backends").mkdir(parents=True)
        source = runtime / "scannet_eval" / "backends" / "runtime.py"
        source.write_text("VERSION = 1\n")
        (runtime / "eval_scannet.py").write_text("ENTRYPOINT = 1\n")
        first = {"runtime": _runtime_fingerprint(runtime)}
        output = self.root / "runtime-output"
        _prepare_output(output, first, False)
        source.write_text("VERSION = 2\n")
        second = {"runtime": _runtime_fingerprint(runtime)}
        self.assertNotEqual(first, second)
        with self.assertRaisesRegex(OutputConflictError, "provenance"):
            _prepare_output(output, second, True)

    def _completed_output(self, name):
        output = self.root / name
        summary = run_evaluation(
            self.config,
            "test",
            output,
            [self.a],
            backend_factory=self.factory(FakeBackend()),
        )
        self.assertTrue(summary["complete"])
        self.assertEqual(
            set(json.loads((output / self.a / "metrics.json").read_text())),
            SCENE_METRIC_KEYS,
        )
        return output

    def _assert_damaged_scene_blocks_aggregate_and_resume(self, output, message):
        with self.assertRaisesRegex(OutputConflictError, message):
            aggregate_run(output)
        factory = self.factory(FakeBackend())
        with self.assertRaisesRegex(OutputConflictError, message):
            run_evaluation(
                self.config,
                "test",
                output,
                [self.a],
                resume=True,
                backend_factory=factory,
            )
        self.assertEqual(factory.calls, [])

    def test_missing_or_corrupt_metrics_json_blocks_aggregate_and_resume(self):
        missing = self._completed_output("missing-metrics")
        (missing / self.a / "metrics.json").unlink()
        self._assert_damaged_scene_blocks_aggregate_and_resume(missing, "missing")

        corrupt = self._completed_output("corrupt-metrics")
        (corrupt / self.a / "metrics.json").write_text("{")
        self._assert_damaged_scene_blocks_aggregate_and_resume(corrupt, "invalid JSON")

    def test_six_key_or_mismatched_metrics_blocks_aggregate_and_resume(self):
        reduced = self._completed_output("reduced-metrics")
        result_path = reduced / self.a / "result.json"
        result = json.loads(result_path.read_text())
        result["metrics"] = {
            key: value for key, value in result["metrics"].items() if key in METRIC_KEYS
        }
        result_path.write_text(json.dumps(result))
        (reduced / self.a / "metrics.json").write_text(json.dumps(result["metrics"]))
        self._assert_damaged_scene_blocks_aggregate_and_resume(reduced, "13")

        mismatched = self._completed_output("mismatched-metrics")
        metrics_path = mismatched / self.a / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["ate"] += 1.0
        metrics_path.write_text(json.dumps(metrics))
        self._assert_damaged_scene_blocks_aggregate_and_resume(
            mismatched, "does not match"
        )


if __name__ == "__main__":
    unittest.main()

"""CPU tests for v10 ScanNet-50 staged execution."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.ours_v10 import scannet50
from experiments.ours_v10.scannet50_contract import RunPaths, sha256


class ScanNet50PipelineTest(unittest.TestCase):
    def test_memory_optimized_flag_is_opt_in(self):
        common = ["run", "--scene", "scene0000_00", "--frames", "1000",
                  "--gpu", "4", "--output-root", "/tmp/output",
                  "--scratch-root", "/tmp/scratch"]
        self.assertFalse(scannet50.parse_args(common).memory_optimized)
        self.assertTrue(scannet50.parse_args(common + ["--memory-optimized"]).memory_optimized)

    def test_memory_optimized_run_manifest_records_execution_flags(self):
        paths = RunPaths(Path("/tmp/out"), Path("/tmp/scratch"))
        selection = dict(actual_frames=1000, frame_ids=list(range(1000)))
        with mock.patch.object(scannet50, "sha256", return_value=scannet50.CHECKPOINT_SHA256), \
             mock.patch.object(scannet50, "_source_identity", return_value=dict(commit="test")):
            run = scannet50._new_run_manifest("scene0000_00", 1000,
                "camera_global_overlap", 4, paths, selection, {}, True)
        config = run["configuration"]
        self.assertTrue(config["memory_optimized"])
        self.assertTrue(config["offload_head_features"])
        self.assertTrue(config["stream_projected_qkv"])

    def test_front_owner_points_and_window_transform(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            align = root / "alignment"
            align.mkdir()
            identity = dict(scale=1.0, rotation=np.eye(3).tolist(),
                            translation=[0.0, 0.0, 0.0])
            shifted = dict(scale=1.0, rotation=np.eye(3).tolist(),
                           translation=[10.0, 0.0, 0.0])
            (align / "window_0000_transform.json").write_text(json.dumps(identity))
            (align / "window_0001_transform.json").write_text(json.dumps(shifted))
            records = []
            for index, (ids, xs) in enumerate(((('000000', '000001'), (0., 1.)),
                                               (('000001', '000002'), (999., 2.)))):
                path = root / f"{index:04d}.npz"
                points = np.zeros((2, 1, 1, 3), dtype=np.float32)
                points[:, 0, 0, 0] = xs
                np.savez(path, frame_ids=ids, world_points=points,
                         world_points_conf=np.ones((2, 1, 1), dtype=np.float32))
                records.append(dict(path=str(path), sha256=sha256(path)))
            run = dict(frame_ids=[0, 1, 2], windows=[[0, 2], [1, 3]])
            forward = dict(prediction_files=records)
            points, meta = scannet50._point_cloud(root, run, forward,
                                                  dict(source_window=np.array([0, 0, 1])))
            np.testing.assert_array_equal(points[:, 0], [0., 1., 12.])
            self.assertEqual(meta["selected_points"], 3)

    def test_verified_input_reuse_rejects_changed_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "inputs.pt"
            input_path.write_bytes(b"fixed")
            (root / "input_manifest.json").write_text(json.dumps(dict(
                input_path=str(input_path), bytes=5, sha256=sha256(input_path))))
            self.assertTrue(scannet50._reusable_input(root))
            input_path.write_bytes(b"other")
            self.assertFalse(scannet50._reusable_input(root))

    def test_resume_after_score_only_finishes_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, scratch = root / "out" / "f100" / "scene0000_00", root / "scratch" / "f100" / "scene0000_00"
            output.mkdir(parents=True); scratch.mkdir(parents=True)
            source = dict(commit="same")
            run = dict(run_id="run-id", scene_id="scene0000_00", frame_budget=100,
                actual_frames=100, frame_ids=list(range(100)), mode="camera_global_overlap",
                gpu_physical_index=0, source=source, checkpoint_sha256="checkpoint")
            (output / "run_manifest.json").write_text(json.dumps(run))
            (scratch / "owner.json").write_text(json.dumps(dict(
                output=str(output.resolve()), run_id="run-id")))
            args = SimpleNamespace(scene="scene0000_00", frames=100,
                mode="camera_global_overlap", gpu=0, output_root=root / "out",
                scratch_root=root / "scratch", resume=True, memory_optimized=False)
            def cleanup(_scratch, _output, _id):
                (output / "cleanup_receipt.json").write_text('{}')
                return dict(deleted_bytes=0)
            with mock.patch.object(scannet50, "read_scene_list", return_value=["scene0000_00"]*50), \
                 mock.patch.object(scannet50, "audit_frame_selection", return_value=[dict(actual_frames=100, frame_ids=list(range(100)))]), \
                 mock.patch.object(scannet50, "safe_run_paths", return_value=RunPaths(output, scratch)), \
                 mock.patch.object(scannet50, "_disk_preflight", return_value={}), \
                 mock.patch.object(scannet50, "_source_identity", return_value=source), \
                 mock.patch.object(scannet50, "sha256", return_value="checkpoint"), \
                 mock.patch.object(scannet50, "_reusable_score", return_value=True), \
                 mock.patch.object(scannet50, "cleanup_regenerable", side_effect=cleanup), \
                 mock.patch.object(scannet50, "_run_stage") as worker:
                scannet50.run_one(args)
                worker.assert_not_called()
                self.assertTrue((output / "COMPLETE.json").is_file())


if __name__ == "__main__":
    unittest.main()

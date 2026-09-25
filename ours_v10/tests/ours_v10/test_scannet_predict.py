"""CPU contracts for the fixed ScanNet adapter; no model weights or CUDA."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import json

import numpy as np
import torch

from experiments.ours_v10.predict_scannet import (
    FRAME_COUNTS, expected_windows, validate_frozen_input, validate_gt_poses,
)
from experiments.ours_v6.runtime import sha256
from experiments.ours_v10 import align_scannet, predict_scannet, report_scannet


def fixture(root):
    scene = root / "scene0000_00"
    poses = scene / "pose"
    poses.mkdir(parents=True)
    ids = [f"{i:06d}" for i in range(1000)]
    for fid in ids[:100]:
        np.savetxt(poses / f"{fid}.txt", np.eye(4))
    path = root / "inputs.pt"
    torch.save(dict(images=torch.zeros(1000, 3, 1, 1), frame_ids=ids,
                    scene_root=str(scene), preprocessing={"shape": [1000, 3, 1, 1]}), path)
    checkpoint = root / "checkpoint.safetensors"
    checkpoint.write_bytes(b"fixed")
    source = dict(configuration=dict(input=str(path)), frame_ids=ids,
                  input_sha256=sha256(path), checkpoint=str(checkpoint),
                  checkpoint_sha256=sha256(checkpoint))
    return path, source, scene


class ScanNetPredictionEntryTest(unittest.TestCase):
    def test_fixed_windows_and_tail(self):
        self.assertEqual(FRAME_COUNTS, (100, 300, 500, 1000))
        self.assertEqual([len(expected_windows(n)) for n in FRAME_COUNTS], [2, 6, 10, 20])
        self.assertEqual([expected_windows(n)[-1] for n in FRAME_COUNTS],
                         [(50, 100), (250, 300), (450, 500), (950, 1000)])

    def test_frozen_prefixes_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source, _ = fixture(Path(temp))
            a = validate_frozen_input(path, source, 100, shape=(1000, 3, 1, 1))
            b = validate_frozen_input(path, source, 300, shape=(1000, 3, 1, 1), check_gt=False)
            self.assertEqual(a[2], source["frame_ids"][:100])
            self.assertNotEqual(a[3], b[3])
            self.assertEqual(a[4], expected_windows(100))
            self.assertEqual(a[0]["images"].shape[0], 1000)
            source["input_sha256"] = "bad"
            with self.assertRaisesRegex(ValueError, "input.*hash"):
                validate_frozen_input(path, source, 100, shape=(1000, 3, 1, 1), check_gt=False)

    def test_gt_and_frame_order_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source, scene = fixture(Path(temp))
            validate_gt_poses(scene, source["frame_ids"][:100])
            np.savetxt(scene / "pose" / "000099.txt", np.full((4, 4), np.nan))
            with self.assertRaisesRegex(ValueError, "GT"):
                validate_gt_poses(scene, source["frame_ids"][:100])
            source["frame_ids"][1] = "000003"
            with self.assertRaisesRegex(ValueError, "frame"):
                validate_frozen_input(path, source, 100, shape=(1000, 3, 1, 1), check_gt=False)

    def test_reject_other_lengths_and_checkpoint_change(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source, _ = fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, "frame count"):
                validate_frozen_input(path, source, 837, shape=(1000, 3, 1, 1), check_gt=False)
            Path(source["checkpoint"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                validate_frozen_input(path, source, 100, shape=(1000, 3, 1, 1), check_gt=False)

    def test_cli_is_fixed_to_lengths_and_two_modes(self):
        common = ["--output-root", "/tmp/output", "--gpu", "4",
                  "--backend-profile", "native_vggt"]
        for frames in FRAME_COUNTS:
            for mode in predict_scannet.MODES:
                args = predict_scannet.parse_args(common + ["--frames", str(frames), "--mode", mode])
                self.assertEqual((args.frames, args.mode), (frames, mode))
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                predict_scannet.parse_args(common + ["--frames", "837", "--mode", "overlap_correspondence"])
            with self.assertRaises(SystemExit):
                predict_scannet.parse_args(common + ["--frames", "100", "--mode", "independent"])

    def test_alignment_entry_checks_prediction_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, old_source, _ = fixture(root)
            old_manifest = root / "old_manifest.json"
            old_manifest.write_text(json.dumps(old_source))
            mode = "overlap_correspondence"
            run = root / "run" / mode
            (run / "windows").mkdir(parents=True)
            records = []
            for index, (lo, hi) in enumerate(expected_windows(100)):
                pred = run / "windows" / f"{index:04d}" / "local.npz"
                pred.parent.mkdir()
                pred.write_bytes(f"window {index}".encode())
                records.append(dict(window=index, lo=lo, hi=hi,
                                    path=str(pred), sha256=sha256(pred)))
            (run / "COMPLETE.json").write_text('{}')
            manifest = dict(dataset="ScanNet", scene="scene0000_00", communication_mode=mode,
                            configuration=dict(frames=100, input=str(path)),
                            input_sha256=old_source["input_sha256"],
                            checkpoint_sha256=old_source["checkpoint_sha256"],
                            frame_ids=old_source["frame_ids"][:100],
                            windows=expected_windows(100), image_tensor_sha256="prefix",
                            prediction_files=records)
            (run / "run_manifest.json").write_text(json.dumps(manifest))
            with mock.patch.object(align_scannet, "FROZEN_INPUT", path), \
                 mock.patch.object(align_scannet, "FROZEN_MANIFEST", old_manifest), \
                 mock.patch.object(align_scannet, "INPUT_SHA256", old_source["input_sha256"]), \
                 mock.patch.object(align_scannet, "CHECKPOINT_SHA256", old_source["checkpoint_sha256"]), \
                 mock.patch.object(align_scannet, "validate_frozen_input", return_value=(None, None, None, "prefix", None)):
                _, prediction_dir = align_scannet.validate_prediction_set(root / "run", mode)
                self.assertEqual(prediction_dir, run / "windows")
                (run / "windows" / "0001" / "local.npz").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "prediction file identity"):
                    align_scannet.validate_prediction_set(root / "run", mode)

    def test_same_length_report_checks_identity_and_edges(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for mode in predict_scannet.MODES:
                pred, align = root / mode, root / "alignment" / mode
                pred.mkdir(parents=True)
                align.mkdir(parents=True)
                (pred / "COMPLETE.json").write_text('{}')
                (align / "COMPLETE.json").write_text('{}')
                manifest = dict(dataset="ScanNet", scene="scene0000_00", communication_mode=mode,
                    frame_ids=[f"{i:06d}" for i in range(100)], windows=expected_windows(100),
                    input_sha256="input", image_tensor_sha256="prefix", checkpoint_sha256="model",
                    precision="bf16", configuration=dict(window_size=60, overlap=10,
                    backend_profile="native_vggt", correspondence_attention_path="native_sdpa",
                    query_chunk_size=512, cache_local_kv_dtype=True,
                    dense_head_frame_chunk=None, reuse_image_encoding=False),
                    timing=dict(forward_seconds=10), peak_allocated_bytes=10*2**30,
                    peak_reserved_bytes=12*2**30, cpu_peak_rss_bytes=2**30)
                (pred / "run_manifest.json").write_text(json.dumps(manifest))
                edge = dict(edge_index=0, scale=1.0, fallback=False, fallback_reason=None,
                    selected_pairs=256, boundary_translation_error_m=.01,
                    boundary_rotation_error_deg=.1, add_seconds=1.0)
                alignment = dict(edges=[edge], alignment_seconds=1.0,
                    process_peak_rss_bytes=2**30, fallback_edges=[])
                (align / "alignment_comparison.json").write_text(json.dumps(alignment))
                metrics = dict(ate_rmse_m=.05, adjacent=dict(count=99, translation_rmse_m=.01,
                    rotation_rmse_deg=.2), ownership_boundaries=dict(count=1,
                    translation_rmse_m=.03, rotation_rmse_deg=.4),
                    gt_alignment="one whole-trajectory proper Sim(3); never per window")
                (align / "evaluation_summary.json").write_text(json.dumps(metrics))
            rows, edges, _, deltas = report_scannet.collect(root, 100)
            self.assertEqual((len(rows), len(edges), deltas["ate_rmse_m"]), (2, 2, 0.0))
            manifest_path = root / predict_scannet.MODES[1] / "run_manifest.json"
            candidate = json.loads(manifest_path.read_text())
            candidate["image_tensor_sha256"] = "wrong"
            manifest_path.write_text(json.dumps(candidate))
            with self.assertRaisesRegex(ValueError, "inputs or noncommunication"):
                report_scannet.collect(root, 100)


if __name__ == "__main__":
    unittest.main()

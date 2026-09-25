import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from experiments.ours_v6.correspondence import save_overlap_mask
from experiments.ours_v6.metrics import summarize
from experiments.ours_v6 import __main__ as launcher
from experiments.ours_v6.check_runs import check


class InfraTest(unittest.TestCase):
    def test_correspondence_mask_matches_long_confidence_rule(self):
        with tempfile.TemporaryDirectory() as temporary:
            a = dict(frame_ids=["a", "b"],
                     world_points=np.ones((2, 2, 2, 3)),
                     world_points_conf=np.array([[[1., 2.], [3., 4.]]] * 2))
            b = dict(frame_ids=["b", "c"],
                     world_points=np.ones((2, 2, 2, 3)),
                     world_points_conf=np.array([[[2., 3.], [4., 5.]]] * 2))
            path = Path(temporary) / "edge.npz"
            count = save_overlap_mask(a, b, path)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["common_frame_ids"].tolist(), ["b"])
                self.assertEqual(data["preceding_local_index"].tolist(), [1])
                self.assertEqual(data["following_local_index"].tolist(), [0])
                mask = np.unpackbits(data["packed_valid_mask"])[:4].reshape(1, 2, 2)
                threshold = .1 * min(np.median(a["world_points_conf"][1:]),
                                     np.median(b["world_points_conf"][:1]))
                expected = (a["world_points_conf"][1:] > threshold) & (b["world_points_conf"][:1] > threshold)
                np.testing.assert_array_equal(mask, expected)
                self.assertEqual(count, int(expected.sum()))

    def test_summary_distinguishes_true_and_common_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                dict(before="000058", after="000059", boundary=False,
                     translation_error=.1, rotation_error_deg=1.),
                dict(before="000059", after="000060", boundary=True,
                     translation_error=.2, rotation_error_deg=2.),
                dict(before="000089", after="000090", boundary=True,
                     translation_error=.4, rotation_error_deg=4.),
            ]
            (root / "adjacent_pose_errors.json").write_text(json.dumps(rows))
            (root / "trajectory_metrics.json").write_text(json.dumps(dict(ate_rmse_m=.03)))
            result = summarize(root)
            self.assertEqual(result["ownership_boundaries"]["count"], 2)
            self.assertEqual(result["within_window"]["count"], 1)
            self.assertEqual(len(result["common_pairs"]), 2)
            self.assertTrue(all(row["is_ownership_boundary"] for row in result["common_pairs"]))

    def test_public_cli_rejects_old_batch_option_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unused"
            argv = ["ours_v6", "run", "--gpu", "0", "--mode", "independent",
                    "--frames", "61", "--output", str(path), "--batch-size", "2"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    launcher.main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(path.exists())

    def test_v5_manifest_input_and_checkpoint_cross_check(self):
        from experiments.ours_v6.runtime import ROOT
        config = json.loads((ROOT / "configs/v6_validation.json").read_text())
        old = json.loads(Path(config["v5_reference_manifest"]).read_text())
        result = launcher.validate_reference(
            config, old["frame_ids"], old["preprocessing"],
            ROOT / "configs/scene0150_00_frames100.json"
        )
        self.assertTrue(result["frame_prefix_verified"])
        self.assertEqual(result["checkpoint_sha256"], config["checkpoint_sha256"])

    def test_read_only_single_window_gpu_gate_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = {}
            for mode in ("independent", "camera_exchange", "camera_register_exchange"):
                root = Path(temporary) / mode
                worker = root / mode / "windows/0000"
                worker.mkdir(parents=True)
                contract = dict(mode=mode, frame_ids=["a", "b"],
                                windows=[[0, 2]], checkpoint_sha256="test",
                                precision="bf16",
                                preprocessing=dict(shape=[2, 3, 392, 518],
                                                   dtype="torch.float32", loader="original",
                                                   minimum=0.0, maximum=1.0, elapsed_seconds=1.0),
                                data=dict(frame_list_sha256="same"))
                (root / "config.json").write_text(json.dumps(dict(contract=contract)))
                (root / "COMPLETE.json").write_text("{}")
                (root / mode / "COMPLETE.json").write_text("{}")
                (root / mode / "evaluation_summary.json").write_text("{}")
                np.savez_compressed(root / mode / "global_trajectory.npz",
                                    frame_ids=np.asarray(["a", "b"]),
                                    source_window=np.asarray([0, 0]),
                                    c2w=np.repeat(np.eye(4)[None], 2, axis=0))
                values = dict(c2w=np.repeat(np.eye(4)[None], 2, axis=0),
                              depth=np.ones((2, 1, 1, 1)), intrinsics=np.repeat(np.eye(3)[None], 2, axis=0),
                              confidence=np.ones((2, 1, 1)),
                              world_points=np.ones((2, 1, 1, 3)),
                              world_points_conf=np.ones((2, 1, 1)),
                              frame_ids=np.asarray(["a", "b"]))
                np.savez_compressed(worker / "local.npz", **values)
                roots[mode] = root
            result = check(roots, "single")
            self.assertTrue(result["single_window_exact"])
            with np.load(roots["camera_register_exchange"] /
                         "camera_register_exchange/windows/0000/local.npz") as data:
                altered = {key: data[key] for key in data.files}
            altered["depth"] = altered["depth"] + .01
            np.savez_compressed(roots["camera_register_exchange"] /
                                "camera_register_exchange/windows/0000/local.npz", **altered)
            with self.assertRaisesRegex(ValueError, "differs in depth"):
                check(roots, "single")

    def test_config_has_no_window_batch_setting(self):
        from experiments.ours_v6.runtime import ROOT
        config = json.loads((ROOT / "configs/v6_validation.json").read_text())
        self.assertNotIn("window_batch_size", config)
        self.assertEqual(config["communication_modes"],
                         ["independent", "camera_exchange", "camera_register_exchange"])
        self.assertEqual(config["checkpoint_sha256"],
                         "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e")


if __name__ == "__main__":
    unittest.main()

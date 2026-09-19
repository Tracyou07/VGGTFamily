import tempfile
import unittest
from pathlib import Path
import numpy as np
from helpers import make_dataset, SCENES


class DataTests(unittest.TestCase):
    def test_preflight_exact_scenes_and_kf10(self):
        from nrgbd_eval.data import preflight_dataset

        with tempfile.TemporaryDirectory() as t:
            make_dataset(Path(t))
            r = preflight_dataset(Path(t))
            self.assertEqual(tuple(r["scenes"]), SCENES)
            self.assertEqual(r["selected_counts"], {s: 3 for s in SCENES})
            self.assertEqual(r["frame_ids"][SCENES[0]], ["0", "10", "20"])

    def test_missing_pair_fails(self):
        from nrgbd_eval.data import preflight_dataset

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            make_dataset(root)
            (root / SCENES[0] / "depth" / "depth10.png").unlink()
            with self.assertRaisesRegex(ValueError, "modality mismatch"):
                preflight_dataset(root)

    def test_pose_and_depth_contract(self):
        from nrgbd_eval.data import load_scene

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            make_dataset(root)
            scene = load_scene(root, SCENES[0])
            self.assertEqual(scene.model.frame_ids, ("0", "10", "20"))
            self.assertFalse(hasattr(scene.model, "depths"))
            self.assertEqual(scene.depth_m(0).shape, (392, 518))
            self.assertAlmostEqual(float(scene.depth_m(0)[0, 0]), 1.0)
            self.assertEqual(scene.intrinsics.shape, (3, 3, 3))
            self.assertNotEqual(float(scene.intrinsics[0, 0, 0]), 554.2562584220408)
            self.assertTrue(
                np.allclose(scene.poses_c2w[0][:3, 1:3], np.diag([1, -1, -1])[:3, 1:3])
            )

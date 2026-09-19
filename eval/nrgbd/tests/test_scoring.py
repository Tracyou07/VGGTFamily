import unittest
import numpy as np


class ScoringTests(unittest.TestCase):
    def test_exact_cloud_metrics(self):
        from nrgbd_eval.scoring import score_clouds

        p = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        m = score_clouds(p, p, seed=2, point_cap=999999, icp_threshold=0.1)
        self.assertAlmostEqual(m["acc"], 0)
        self.assertAlmostEqual(m["comp"], 0)
        self.assertAlmostEqual(m["nc"], 1)

    def test_sampling_is_deterministic(self):
        from nrgbd_eval.scoring import deterministic_sample

        p = np.arange(300).reshape(100, 3)
        self.assertTrue(
            np.array_equal(deterministic_sample(p, 8, 4), deterministic_sample(p, 8, 4))
        )

    def test_bad_cloud_fails(self):
        from nrgbd_eval.scoring import score_clouds

        with self.assertRaises(ValueError):
            score_clouds(np.empty((0, 3)), np.ones((2, 3)))

    def test_lower_median_matches_torch_nanmedian(self):
        from nrgbd_eval.scoring import lower_nanmedian

        values = np.array([0.0, 2.0, np.nan])
        self.assertEqual(float(lower_nanmedian(values)), 0.0)

    def test_world_points_transform_to_first_camera(self):
        from nrgbd_eval.geometry import transform_points

        pose = np.eye(4)
        pose[0, 3] = 5
        point = np.array([[[6.0, 0, 1]]])
        got = transform_points(point, np.linalg.inv(pose))
        self.assertTrue(np.allclose(got, [[[1, 0, 1]]]))

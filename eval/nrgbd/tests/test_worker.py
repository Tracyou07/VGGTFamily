import unittest
import numpy as np


class WorkerTests(unittest.TestCase):
    def test_all_registered_bridges_expose_infer(self):
        from nrgbd_eval.backend_worker import BRIDGES
        import importlib

        for module_name in BRIDGES.values():
            self.assertTrue(callable(importlib.import_module(module_name).infer))

    def test_normalize_dense_output_to_protocol_shape(self):
        from nrgbd_eval.backend_worker import normalize_dense

        points = np.zeros((2, 4, 6, 3), np.float32)
        points[..., 0] = np.arange(6)[None, None, :]
        out, mask = normalize_dense(points, np.ones((2, 4, 6), bool), (8, 10))
        self.assertEqual(out.shape, (2, 8, 10, 3))
        self.assertEqual(mask.shape, (2, 8, 10))
        self.assertTrue(mask.all())

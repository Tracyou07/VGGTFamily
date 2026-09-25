import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from experiments.ours_v7 import worker
from vggt.layers.block import Block
from vggt.v7.attention import global_step


class QueryScheduleTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        self.block = Block(8, 2, qk_norm=True).double().eval()
        self.states = [torch.randn(1, 12, 8, dtype=torch.float64),
                       torch.randn(1, 18, 8, dtype=torch.float64)]

    def run_step(self, **kwargs):
        return global_step(self.block, self.states, [None, None], 6,
                           "camera_patch_exchange", 1, 1, (0, 3), **kwargs)

    def test_independent_group_chunks_preserve_dense_result_and_order(self):
        expected = self.run_step(query_chunk_size=2)
        for local, cross in ((4, 2), ("all", 2), ("all", "all"), (1, 8)):
            actual = self.run_step(local_query_chunk_size=local,
                                   cross_query_chunk_size=cross)
            reverse = self.run_step(local_query_chunk_size=local,
                                    cross_query_chunk_size=cross, order=[1, 0])
            for got, ref, rev in zip(actual, expected, reverse):
                torch.testing.assert_close(got, ref, atol=1e-9, rtol=1e-9)
                torch.testing.assert_close(got, rev, atol=0, rtol=0)

    def test_full_group_and_tail_calls_keep_one_softmax_per_query(self):
        calls = []
        original = F.scaled_dot_product_attention

        def traced(q, k, v, **kwargs):
            self.assertIsNone(kwargs.get("attn_mask"))
            calls.append((q.shape[-2], k.shape[-2]))
            return original(q, k, v, **kwargs)

        with patch("torch.nn.functional.scaled_dot_product_attention", traced):
            self.run_step(local_query_chunk_size="all", cross_query_chunk_size=4)
        # Cross queries see local K/V plus the other window's selected tokens.
        self.assertEqual(calls.count((4, 21)), 1)
        self.assertEqual(calls.count((2, 21)), 1)
        self.assertEqual(calls.count((4, 24)), 2)
        self.assertEqual(calls.count((1, 24)), 1)
        # Local queries see only their own complete window.
        self.assertEqual(calls.count((6, 12)), 1)
        self.assertEqual(calls.count((9, 18)), 1)
        self.assertEqual(len(calls), 7)

    def test_legacy_default_and_conflict(self):
        implicit = self.run_step()
        explicit = self.run_step(query_chunk_size=64)
        for a, b in zip(implicit, explicit):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            self.run_step(query_chunk_size=128, local_query_chunk_size=256)
        for bad in (0, -1, "bad"):
            with self.assertRaises(ValueError):
                self.run_step(local_query_chunk_size=bad)

    def test_worker_parameters_and_conflict(self):
        base = ["--input", "in.pt", "--output", "out", "--gpu", "4",
                "--frames", "100", "--mode", "camera_patch_exchange"]
        args = worker.parse_args(base)
        self.assertIsNone(args.local_query_chunk_size)
        self.assertIsNone(args.cross_query_chunk_size)
        args = worker.parse_args(base + ["--local-query-chunk-size", "all",
                                         "--cross-query-chunk-size", "256"])
        self.assertEqual(args.local_query_chunk_size, "all")
        self.assertEqual(args.cross_query_chunk_size, 256)
        with self.assertRaises(SystemExit):
            worker.parse_args(base + ["--query-chunk-size", "64",
                                      "--local-query-chunk-size", "256"])


if __name__ == "__main__":
    unittest.main()

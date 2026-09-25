"""CPU contracts for the opt-in exchange-only Flash SDPA scope."""
import unittest
from unittest.mock import patch

import torch
from vggt.layers.block import Block
from vggt.layers.patch_embed import PatchEmbed
from vggt.models.aggregator import Aggregator
import vggt.v7.attention as attention
from vggt.v7.scheduler import aggregate_windows
from experiments.ours_v7 import worker


def sdpa_flags():
    return dict(flash=torch.backends.cuda.flash_sdp_enabled(),
                efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                cudnn=torch.backends.cuda.cudnn_sdp_enabled(),
                math=torch.backends.cuda.math_sdp_enabled())


class ExchangeBackendTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(941)
        self.block = Block(8, 2, qk_norm=True).double().eval()
        self.states = [torch.randn(1, 12, 8, dtype=torch.float64),
                       torch.randn(1, 18, 8, dtype=torch.float64)]

    def step(self, backend):
        return attention.global_step(
            self.block, self.states, [None, None], 6,
            "camera_patch_exchange", 1, 1, (0, 3), query_chunk_size=2,
            exchange_sdpa_backend=backend)

    def test_auto_is_exactly_the_existing_path(self):
        implicit = attention.global_step(
            self.block, self.states, [None, None], 6,
            "camera_patch_exchange", 1, 1, (0, 3), query_chunk_size=2)
        with patch.object(attention, "sdpa_kernel",
                          side_effect=AssertionError("auto entered Flash scope")):
            explicit = self.step("auto")
        for old, new in zip(implicit, explicit):
            torch.testing.assert_close(old, new, atol=0, rtol=0)

    def test_flash_covers_local_and_remote_queries_and_restores_flags(self):
        before = sdpa_flags()
        observed = []

        def fake_attend(module, q, k, v):
            observed.append((int(k.shape[2]), sdpa_flags()))
            return torch.zeros_like(q)

        with patch.object(attention, "_attend", side_effect=fake_attend):
            self.step("flash")
        self.assertEqual(sdpa_flags(), before)
        self.assertTrue(any(length == 12 for length, _ in observed))
        self.assertTrue(any(length > 18 for length, _ in observed))
        for _, flags in observed:
            self.assertEqual(flags, dict(flash=True, efficient=False,
                                         cudnn=False, math=False))

    def test_flash_scope_restores_on_exception_and_does_not_fallback(self):
        before = sdpa_flags()
        with patch.object(attention, "_attend", side_effect=RuntimeError("flash unavailable")):
            with self.assertRaisesRegex(RuntimeError, "flash unavailable"):
                self.step("flash")
        self.assertEqual(sdpa_flags(), before)

    def test_independent_does_not_enter_exchange_scope(self):
        with patch.object(attention, "sdpa_kernel", side_effect=AssertionError("exchange scope entered")):
            actual = attention.global_step(
                self.block, self.states, [None, None], 6,
                "independent", 1, 1, (0, 3),
                exchange_sdpa_backend="flash")
        self.assertEqual(len(actual), 2)

    def test_frame_attention_stays_outside_flash_scope(self):
        aggregator = Aggregator(img_size=4, patch_size=2, embed_dim=8, depth=1,
                                num_heads=2, num_register_tokens=1,
                                patch_embed="conv", cached_layer_indices=(0,))
        aggregator.patch_embed = PatchEmbed(img_size=4, patch_size=2, embed_dim=8)
        aggregator = aggregator.double().eval().requires_grad_(False)
        images = torch.rand(4, 3, 4, 4, dtype=torch.float64)
        before = sdpa_flags()
        frame_flags, exchange_flags = [], []
        original = torch.nn.functional.scaled_dot_product_attention

        def frame_sdpa(*args, **kwargs):
            frame_flags.append(sdpa_flags())
            return original(*args, **kwargs)

        def exchange_sdpa(module, q, k, v):
            exchange_flags.append(sdpa_flags())
            return torch.zeros_like(q)

        with patch("torch.nn.functional.scaled_dot_product_attention", frame_sdpa), \
             patch.object(attention, "_attend", side_effect=exchange_sdpa):
            aggregate_windows(aggregator, images, [(0, 3), (1, 4)],
                              "camera_patch_exchange", exchange_sdpa_backend="flash")
        self.assertTrue(frame_flags)
        self.assertTrue(exchange_flags)
        self.assertTrue(all(flags == before for flags in frame_flags))
        self.assertTrue(all(flags == dict(flash=True, efficient=False,
                                          cudnn=False, math=False)
                            for flags in exchange_flags))
        self.assertEqual(sdpa_flags(), before)

    def test_scope_entry_failure_is_not_swallowed(self):
        before = sdpa_flags()
        with patch.object(attention, "sdpa_kernel", side_effect=RuntimeError("Flash unsupported")):
            with self.assertRaisesRegex(RuntimeError, "Flash unsupported"):
                self.step("flash")
        self.assertEqual(sdpa_flags(), before)

    def test_parser_defaults_and_rejects_invalid_backend(self):
        required = ["--input", "in.pt", "--output", "out", "--gpu", "4",
                    "--frames", "100", "--mode", "camera_patch_exchange"]
        self.assertEqual(worker.parse_args(required).exchange_sdpa_backend, "auto")
        self.assertEqual(worker.parse_args(required + ["--exchange-sdpa-backend", "flash"])
                         .exchange_sdpa_backend, "flash")
        with self.assertRaises(SystemExit):
            worker.parse_args(required + ["--exchange-sdpa-backend", "nope"])


if __name__ == "__main__":
    unittest.main()

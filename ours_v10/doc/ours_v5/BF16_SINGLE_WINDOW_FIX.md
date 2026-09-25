# BF16 single-window correction

Failure: run 20260921T042854Z_v5_single at be5f927efc2045dd54259a43e820950a436366cc.
The original failure directory and all captures were left unchanged. No thresholds,
checkpoint, frame list, preprocessing, prediction head or Long alignment was changed.

## Evidence

Original reference repetition, both disabled variants and FP32 single-window outputs
were exact. BF16 encoder was exact, but head_cache_4 already differed and later
cached features and dense predictions exceeded the unchanged tolerance.

A synthetic 30x1041-token QKV probe on idle H20 GPU4 produced no split error. Thus
random-input success was not treated as sufficient evidence.

A second probe loaded only the real reference frame-side cache at layer4 and the
weights of global block4. It did NOT run the full model, image encoder, benchmark,
or the multiwindow/100-frame experiment. Original block output matched the saved
reference global-side cache exactly, validating replay inputs, RoPE and weights.
For exactly the same QKV, splitting camera/noncamera queries changed BF16 SDPA by:

- max absolute attention output difference: 0.001953125 (966 changed elements);
- full original block versus old v5 block: max 0.0017321109771728516;
- projected attention difference: max 0.00390625.

This isolates numerical changes introduced by query decomposition on real inputs.
It supports rounding accumulation as the cause of later-layer differences; no claim
is made that an SDPA backend switch occurred. Synthetic profile used cuDNN for all calls.

## Fix

When the complete scene contains ONE window, the visibility graph is exactly the
original graph. global_step now calls the original block and skips camera banking;
exchange_attention also delegates to original attention when its bank has one entry.
There are no forbidden cross-window connections in this case. This does not add a
full-scene dense mask or change multiwindow execution, normalization or softmax.
The guard is total window count, NOT compute group size.

## Validation and boundaries

- Two regression tests failed before correction, then passed: require original
  forward execution and exact CPU BF16 equality for single-window block/attention.
- All 29 v5 CPU tests passed, including multiwindow dense-mask oracle, dependency,
  relay, ordering, original heads, Long parity and artifact tests.
- All 24 v4 regression tests passed.
- Replaying the same real layer4 after correction gives zero difference against
  both original block and saved cache. The diagnostic still separately computes
  the old explicit split SDPA and reproduces its nonzero difference.
- Independent code review found no actionable issue in the fix.
- Full corrected single-window GPU gate: NOT RUN.
- Multiwindow gates and 100-frame diagnostic: NOT RUN.
- No tolerance increase and no automatic benchmark continuation.

Probe records: bf16_sdpa_split_probe.json, bf16_real_global_block_probe.json,
bf16_real_global_block_fixed.json. In the real-block JSON the legacy field name
block_original_vs_split measures production global_step; after correction that
production path delegates to the original block. sdpa_whole_vs_split remains the
explicit diagnostic split calculation in both files.

Recommended next step: new output directory, rerun BF16 single-window gate only.
Stop and report its outcome before proceeding to multiwindow. Existing gate seals
must not be reused across the changed code identity.

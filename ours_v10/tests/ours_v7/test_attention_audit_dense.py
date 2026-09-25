"""Independent, small-tensor oracle for the v7 global attention audit.

Run in the v7 repository as: python -B tests/ours_v7/test_attention_audit_dense.py OUT.json
The oracle uses explicit token metadata and a Boolean visibility matrix. It
does not call production selection, QKV projection, or SDPA helpers.
"""
import json
import math
import sys
import traceback

import torch
from torch.nn import functional as F

from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D
from vggt.v7.attention import communication_bank, exchange_attention, global_step

ATOL = 1e-10
RTOL = 1e-10
METRICS = []


def case_state(lengths, selected, rope):
    patch_count, register_count = 10, 2
    per_frame = 1 + register_count + patch_count
    states, positions, metadata = [], [], []
    for window, frames in enumerate(lengths):
        states.append(torch.randn(1, frames * per_frame, 16, dtype=torch.float64))
        pos, meta = [], []
        for frame in range(frames):
            # The first frame of an overlapping window has its own state.
            logical_frame = sum(lengths[:window]) - window + frame
            for offset in range(per_frame):
                kind = "camera" if offset == 0 else ("register" if offset <= register_count else "patch")
                patch = offset - 1 - register_count if kind == "patch" else None
                meta.append((window, logical_frame, kind, patch))
                pos.append((1 + patch // 5, 1 + patch % 5) if patch is not None else (0, 0))
        positions.append(torch.tensor(pos, dtype=torch.long).unsqueeze(0) if rope else None)
        metadata.extend(meta)
    return states, positions, metadata, per_frame


def oracle(block, states, positions, metadata, selected):
    attention = block.attn
    q_list, k_list, v_list = [], [], []
    for x, pos in zip(states, positions):
        normalized = block.norm1(x)
        raw = F.linear(normalized, attention.qkv.weight, attention.qkv.bias)
        raw = raw.reshape(1, x.shape[1], 3, attention.num_heads, attention.head_dim)
        q, k, v = raw.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = attention.q_norm(q), attention.k_norm(k)
        if attention.rope is not None:
            q, k = attention.rope(q, pos), attention.rope(k, pos)
        q_list.append(q)
        k_list.append(k)
        v_list.append(v)
    q, k, v = [torch.cat(parts, dim=2) for parts in (q_list, k_list, v_list)]
    selected = set(selected)

    def cross(token):
        return token[2] == "camera" or (token[2] == "patch" and token[3] in selected)

    visible = torch.tensor([
        [query[0] == key[0] or (cross(query) and cross(key))
         for key in metadata] for query in metadata], dtype=torch.bool)
    scores = (q @ k.transpose(-2, -1)) * (attention.head_dim ** -0.5)
    scores = scores.masked_fill(~visible[None, None], -torch.inf)
    probabilities = scores.softmax(dim=-1)
    pre = probabilities @ v
    pre = pre.transpose(1, 2).reshape(1, len(metadata), -1)
    post = F.linear(pre, attention.proj.weight, attention.proj.bias)
    lengths = [x.shape[1] for x in states]
    pre_parts, post_parts = pre.split(lengths, dim=1), post.split(lengths, dim=1)
    full = []
    for x, projected in zip(states, post_parts):
        residual = x + block.ls1(projected)
        full.append(residual + block.ls2(block.mlp(block.norm2(residual))))
    return pre_parts, post_parts, full, visible


def check(label, stage, actual, expected):
    delta = (actual - expected).abs()
    max_error = float(delta.max())
    mean_error = float(delta.mean())
    allowed = ATOL + RTOL * expected.abs()
    passed = bool(torch.isfinite(actual).all() and (delta <= allowed).all())
    METRICS.append(dict(case=label, stage=stage, max_abs=max_error,
                        mean_abs=mean_error, finite=bool(torch.isfinite(actual).all()), passed=passed))
    if not passed:
        raise AssertionError(f"{label} {stage}: max={max_error}, mean={mean_error}")


def run_case(lengths, ratio, chunks, rope, seed, order=None, distinctive=False):
    torch.manual_seed(seed)
    count = math.ceil(ratio * 10)
    selected = (0, 3, 6, 9, 2, 5, 8, 1, 4, 7)[:count]
    states, positions, metadata, per_frame = case_state(lengths, selected, rope)
    if distinctive:
        for window, state in enumerate(states):
            state[:, ::3] += (window + 1) * torch.linspace(
                -2.0, 2.0, state.shape[-1], dtype=state.dtype)
    block = Block(16, 2, qk_norm=True,
                  rope=RotaryPositionEmbedding2D() if rope else None).double().eval()
    expected_pre, expected_post, expected_full, visible = oracle(block, states, positions, metadata, selected)
    before, after = [], []

    def record(_module, args, output):
        before.append(args[0].detach().clone())
        after.append(output.detach().clone())

    hook = block.attn.proj.register_forward_hook(record)
    try:
        actual = global_step(block, states, positions, per_frame,
                             "camera_patch_exchange", 1, 2, selected,
                             query_chunk_size=64, order=order,
                             local_query_chunk_size=chunks[0],
                             cross_query_chunk_size=chunks[1])
    finally:
        hook.remove()
    sequence = list(range(len(states))) if order is None else order
    label = f"lengths={lengths};ratio={ratio};chunks={chunks};rope={rope};order={sequence};distinctive={distinctive}"
    for call, window in enumerate(sequence):
        check(label, f"pre_projection_window_{window}", before[call], expected_pre[window])
        check(label, f"post_projection_window_{window}", after[call], expected_post[window])
    for window in range(len(states)):
        check(label, f"global_block_window_{window}", actual[window], expected_full[window])
    # The metadata mask itself is an explicit assertion of local-once and cross rules.
    assert visible.diag().all()
    if len(states) > 1:
        assert states[0].untyped_storage().data_ptr() != states[1].untyped_storage().data_ptr()
    if ratio == 0:
        assert all(not visible[i, j] for i, q in enumerate(metadata)
                   for j, k in enumerate(metadata) if q[0] != k[0] and q[2] != "camera")
        camera_only = global_step(block, states, positions, per_frame,
                                  "camera_exchange", 1, 2, (),
                                  local_query_chunk_size=chunks[0],
                                  cross_query_chunk_size=chunks[1])
        for i, (a, b) in enumerate(zip(actual, camera_only)):
            check(label, f"ratio_zero_is_camera_only_{i}", a, b)
    if len(states) == 1:
        assert visible.all()
        check(label, "single_window_native_block", actual[0],
              block(states[0], pos=positions[0]))


def test_visibility_perturbation():
    torch.manual_seed(127)
    selected = (0, 3, 6)
    states, positions, _, per_frame = case_state([2, 3], selected, False)
    block = Block(16, 2, qk_norm=True).double().eval()
    bank = [communication_bank(block, x, None, per_frame, 1, 2,
                               selected, "camera_patch_exchange") for x in states]
    args = (block.attn, block.norm1(states[0]), None, per_frame, bank, 0, 1, 2,
            selected, "camera_patch_exchange")
    original = exchange_attention(*args, local_query_chunk_size="all", cross_query_chunk_size=2)
    remote_k, remote_v = bank[1]
    modified = list(bank)
    modified[1] = (remote_k, remote_v + 4.0)
    changed = exchange_attention(block.attn, block.norm1(states[0]), None,
                                 per_frame, modified, 0, 1, 2, selected,
                                 "camera_patch_exchange", local_query_chunk_size="all",
                                 cross_query_chunk_size=2)
    local_offsets = [i for i in range(states[0].shape[1])
                     if i % per_frame not in (0, 3, 6, 9)]
    cross_offsets = [i for i in range(states[0].shape[1]) if i not in local_offsets]
    check("invisible_remote_bank", "local_query_unchanged",
          changed[:, local_offsets], original[:, local_offsets])
    delta = float((changed[:, cross_offsets] - original[:, cross_offsets]).abs().max())
    if delta < 0.1:
        raise AssertionError(f"visible remote V did not change cross output: {delta}")
    METRICS.append(dict(case="visible_remote_bank", stage="cross_query_detects_change",
                        max_abs=delta, mean_abs=None, finite=True, passed=True))
    # A remote unselected token cannot enter the bank in this layer.
    remote_changed = states[1].clone()
    remote_changed[:, 1] += 50.0  # register token, never selected
    bank_again = communication_bank(block, remote_changed, None, per_frame, 1, 2,
                                    selected, "camera_patch_exchange")
    for old, new in zip(bank[1], bank_again):
        check("invisible_remote_token", "bank_unchanged", new, old)


@torch.inference_mode()
def main(output_path):
    torch.set_num_threads(1)
    result = dict(status="pass", tolerance=dict(atol=ATOL, rtol=RTOL),
                  dtype="float64", cases=[], metrics=METRICS)
    try:
        for lengths in ([2, 3], [2, 1, 3], [2]):
            for ratio in (0.0, 0.1, 0.3, 0.7, 1.0):
                chunks = ("all", 2) if ratio in (0.3, 0.7) else (4, 3)
                for rope in (False, True):
                    run_case(lengths, ratio, chunks, rope, 31)
                    result["cases"].append(dict(lengths=lengths, ratio=ratio,
                                                chunks=chunks, rope=rope, order="forward"))
        # Reversing the processing order must use the same old-state banks.
        run_case([2, 1, 3], 0.7, ("all", 2), True, 31, order=[2, 1, 0])
        result["cases"].append(dict(lengths=[2, 1, 3], ratio=0.7,
                                    chunks=["all", 2], rope=True, order="reverse"))
        run_case([2, 3], 0.3, ("all", 2), True, 31, distinctive=True)
        result["cases"].append(dict(lengths=[2, 3], ratio=0.3,
                                    chunks=["all", 2], rope=True, input="distinctive"))
        test_visibility_perturbation()
    except Exception as exc:
        result["status"] = "fail"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(dict(status=result["status"], cases=len(result["cases"]),
                          metrics=len(METRICS), error=result.get("error")), indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))

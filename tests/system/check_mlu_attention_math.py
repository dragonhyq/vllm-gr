# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise the actual MLU attention adapter against independent FP64 attention.

This is a bounded numerical/structural test, not an end-to-end quality gate.
Inputs are first rounded to BF16, then the reference uses CPU FP64 throughout.
The relative RMS allowance is 1% for the two separately rounded partitions;
all cache writes and prefix-cache repeatability checks must be exact.
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_mlu  # noqa: F401

from vllm_gr.v1.attention.backends import beam_attn_mlu as adapter
from vllm_gr.v1.beam.attention_pool import BeamAttentionPool


def reference(q, k, v, causal=False):
    q, k, v = (x.cpu().double() for x in (q, k, v))
    repeat = q.shape[1] // k.shape[1]
    k, v = (x.repeat_interleave(repeat, dim=1) for x in (k, v))
    scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(q.shape[-1])
    if causal:
        mask = torch.arange(k.shape[0])[None, :] > (
            torch.arange(q.shape[0])[:, None] + k.shape[0] - q.shape[0]
        )
        scores.masked_fill_(mask, -torch.inf)
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


def error(actual, expected):
    actual = actual.cpu().double()
    difference = actual - expected
    relative_rms = float(
        difference.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-15)
    )
    assert torch.isfinite(actual).all()
    assert relative_rms < 0.01, relative_rms
    return {"relative_rms": relative_rms, "max_abs": float(difference.abs().max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261127)
    torch.set_num_threads(2)
    device, dtype = torch.device("mlu"), torch.bfloat16
    result = {
        "reference": "CPU FP64 on BF16 inputs",
        "seed": 20261127,
        "relative_rms_limit": 0.01,
        "cases": [],
    }
    for length in [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 257]:
        h, kh, d, block = 16, 8, 128, 16
        pages = (length + block - 1) // block
        physical = torch.randperm(pages + 3)[:pages]
        cache = torch.full((pages + 3, kh, block, 2 * d), -7, device=device, dtype=dtype)
        kc, vc = adapter.kv_cache_views(cache, d)
        k, v = (torch.randn(length, kh, d, dtype=dtype) for _ in range(2))
        slots = physical[torch.arange(length) // block] * block + torch.arange(length) % block
        adapter.reshape_and_cache(
            k.to(device), v.to(device), kc, vc, slots.to(device), "auto", None, None
        )
        # Recover by physical addresses, independently of adapter KV view strides.
        raw = cache.cpu().reshape(pages + 3, kh, 2, block, d)
        for i in range(length):
            p, offset = int(slots[i]) // block, int(slots[i]) % block
            assert torch.equal(raw[p, :, 0, offset], k[i])
            assert torch.equal(raw[p, :, 1, offset], v[i])
        q = torch.randn(length, h, d, dtype=dtype)
        table = physical[None, :].to(device=device, dtype=torch.int32)
        out, _ = adapter._paged_attention(
            q.to(device),
            kc,
            vc,
            torch.tensor([0, length], device=device, dtype=torch.int32),
            torch.tensor([length], device=device, dtype=torch.int32),
            table,
            d**-0.5,
            True,
        )
        case = {
            "prefix_length": length,
            "cache_write_exact": True,
            "prefill": error(out, reference(q, k, v, True)),
        }
        # Omit whole cached pages and verify the remaining query pages are invariant.
        cached = (length // 2 // block) * block
        if cached:
            hit, _ = adapter._paged_attention(
                q[cached:].to(device),
                kc,
                vc,
                torch.tensor([0, length - cached], device=device, dtype=torch.int32),
                torch.tensor([length], device=device, dtype=torch.int32),
                table,
                d**-0.5,
                True,
            )
            assert torch.equal(hit, out[cached:])
            case["cached_prefix_tokens"] = cached
            case["cached_query_outputs_exact"] = True
        case["decode"] = []
        for width, steps in [(1, 1), (4, 2), (32, 3)]:
            query = torch.randn(width, h, d, dtype=dtype)
            sk, sv = (torch.randn(width, steps, kh, d, dtype=dtype) for _ in range(2))
            pool = BeamAttentionPool(1, width, kh, 4, d, device, dtype)
            for step in range(steps):
                adapter.cache_unshared_kv(
                    pool.unshared_key,
                    pool.unshared_value,
                    sk[:, step].to(device),
                    sv[:, step].to(device),
                    0,
                    step,
                )
            session = SimpleNamespace(
                is_active=True, beam_width=width, decode_step=steps - 1, slot=0
            )
            metadata = SimpleNamespace(
                is_beam_decode=True,
                std_indices=None,
                prefix_indices=torch.arange(width, device=device),
                beam_session=session,
                num_prefix_indices=width,
                prefix_block_table=table,
                suffix_max_q_len=1,
                prefix_cu_seqlens_q_device=torch.tensor(
                    [0, width], device=device, dtype=torch.int32
                ),
                prefix_seqlens_kv_device=torch.tensor([length], device=device, dtype=torch.int32),
            )
            actual = torch.empty_like(query, device=device)
            adapter.cascade_attention(
                actual, query.to(device), kc, vc, metadata, (-1, -1), d**-0.5, 0, 0, beam_pool=pool
            )
            expected = torch.cat(
                [
                    reference(query[i : i + 1], torch.cat([k, sk[i]]), torch.cat([v, sv[i]]))
                    for i in range(width)
                ]
            )
            case["decode"].append(
                {"width": width, "suffix_length": steps, **error(actual, expected)}
            )
        result["cases"].append(case)
        print(json.dumps(case), flush=True)
    result["passed"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

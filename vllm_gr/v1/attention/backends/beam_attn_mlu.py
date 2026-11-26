# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Eager MLU Beam attention using existing TMO and Torch operations.

This adapter supports unquantized decoder attention and one Beam session per
dispatch. Sequence metadata is read on the host; graph capture is unsupported.
"""

from typing import Any

import torch
from vllm_mlu import _mlu_ops as ops

from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    require_beam_decode_session,
    resolve_beam_pool,
)
from vllm_gr.v1.beam.attention_pool import write_eager_step


def is_available() -> bool:
    return ops.tmo is not None and torch.mlu.is_available()


def get_flash_attn_version(*, requires_alibi: bool = False, head_size: int | None = None) -> int:
    del head_size
    if requires_alibi:
        raise NotImplementedError("MLU GR attention does not yet support ALiBi")
    # Like the NPU adapter, TMO does not select a CUDA FlashAttention version.
    return 0


def supports_sinks() -> bool:
    return False


def _require_eager(tensor: torch.Tensor) -> None:
    if tensor.device.type != "mlu" or tensor.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("MLU GR attention requires FP16 or BF16 MLU tensors")
    if torch.mlu.is_current_stream_capturing():
        raise NotImplementedError("MLU GR attention currently supports eager execution only")


def _validate_features(window: Any, softcap: float, sinks: Any, alibi: Any) -> None:
    if window is not None and tuple(window) != (-1, -1):
        raise NotImplementedError("MLU GR attention does not yet support sliding windows")
    if softcap or sinks is not None or alibi is not None:
        raise NotImplementedError("MLU GR attention does not yet support softcap, sinks or ALiBi")


def kv_cache_views(cache: torch.Tensor, head_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpret LBHNC storage using the native TMO per-head K/V packing."""
    if cache.ndim != 4 or cache.shape[-1] != 2 * head_size or not cache.is_contiguous():
        raise ValueError("MLU GR KV cache requires contiguous LBHNC storage and equal K/V heads")
    blocks, heads, block_size, _ = cache.shape
    shape = (blocks, heads, block_size, head_size)
    strides = (cache.stride(0), cache.stride(1), head_size, 1)
    offset = cache.storage_offset()
    key = torch.as_strided(cache, shape, strides, offset)
    value = torch.as_strided(cache, shape, strides, offset + block_size * head_size)
    return key, value


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: Any,
    v_scale: Any,
) -> None:
    del k_scale, v_scale
    _require_eager(key)
    if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
        raise NotImplementedError("MLU GR attention does not yet support quantized KV cache")
    ops.reshape_paged_cache(key, value, key_cache, value_cache, slot_mapping.flatten())


def cache_unshared_kv(
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    slot: int,
    decode_step: int,
    beam_session: Any = None,
) -> None:
    _require_eager(curr_key)
    write_eager_step(
        unshared_key, unshared_value, curr_key, curr_value, slot, decode_step, beam_session
    )


def _flash(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_query: torch.Tensor,
    cu_kv: torch.Tensor,
    max_query: int,
    max_kv: int,
    scale: float,
    causal: bool,
    block_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ops.flash_attention(
        q=query.contiguous(),
        k=key,
        v=value,
        out=None,
        cu_seq_lens_q=cu_query.to(dtype=torch.int32).contiguous(),
        cu_seq_lens_kv=cu_kv.to(dtype=torch.int32).contiguous(),
        alibi_slope=None,
        attn_bias=None,
        max_seq_len_q=max_query,
        max_seq_len_kv=max_kv,
        softmax_scale=scale,
        is_causal=causal,
        compute_dtype=torch.float32,
        return_lse=True,
        block_tables=(
            block_table.to(dtype=torch.int32).contiguous() if block_table is not None else None
        ),
    )


def _paged_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_query: torch.Tensor,
    kv_lengths: torch.Tensor,
    block_table: torch.Tensor,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Per-request eager dispatch also handles empty shared prefixes explicitly;
    # do not rely on a device kernel's output for an empty softmax partition.
    offsets = cu_query.tolist()
    lengths = kv_lengths.tolist()
    if len(offsets) != len(lengths) + 1 or len(lengths) != block_table.shape[0]:
        raise ValueError("MLU paged attention sequence metadata does not match its block table")
    if offsets[0] != 0 or offsets[-1] != query.shape[0]:
        raise ValueError("MLU paged attention query offsets do not cover the query rows")
    result = torch.zeros_like(query)
    lse = torch.full(
        (query.shape[1], query.shape[0]), -torch.inf, device=query.device, dtype=torch.float32
    )
    for index, (start, end, length) in enumerate(zip(offsets, offsets[1:], lengths)):
        if end < start or length < 0 or length > block_table.shape[1] * key.shape[2]:
            raise ValueError("Invalid MLU paged attention sequence length")
        if end == start or length == 0:
            continue
        query_length = end - start
        if causal and query_length > length:
            raise ValueError("Causal MLU prefill requires a KV slot for every query")
        # A cached prefill can omit whole KV pages. Give each causal query
        # page the same TMO geometry as that page had during an uncached
        # forward, including the bottom-right alignment of a partial page.
        pos = start
        while pos < end:
            cached = length - (end - pos)
            count = min(end - pos, key.shape[2] - cached % key.shape[2]) if causal else end - pos
            visible = cached + count if causal else length
            q_lens = torch.tensor([0, count], device=query.device, dtype=torch.int32)
            k_lens = torch.tensor([0, visible], device=query.device, dtype=torch.int32)
            out, state = _flash(
                query[pos : pos + count],
                key,
                value,
                q_lens,
                k_lens,
                count,
                visible,
                scale,
                causal,
                block_table[index : index + 1],
            )
            result[pos : pos + count].copy_(out)
            lse[:, pos : pos + count].copy_(state)
            pos += count
    return result, lse


def prefill_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: Any,
    sliding_window_size: Any,
    scale: float,
    logits_soft_cap: float,
    vllm_flash_attn_version: int,
    s_aux: Any = None,
    alibi_slopes: Any = None,
) -> None:
    del vllm_flash_attn_version
    _require_eager(query)
    _validate_features(sliding_window_size, logits_soft_cap, s_aux, alibi_slopes)
    table = getattr(attn_metadata, "block_table_tensor", None)
    if table is None:
        table = attn_metadata.block_table
    rows = attn_metadata.num_actual_tokens
    out, _ = _paged_attention(
        query[:rows],
        key_cache,
        value_cache,
        attn_metadata.query_start_loc,
        attn_metadata.seq_lens,
        table,
        scale,
        attn_metadata.causal,
    )
    output[:rows].copy_(out)


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: Any,
    sliding_window_size: Any,
    scale: float,
    logits_soft_cap: float,
    vllm_flash_attn_version: int,
    s_aux: Any = None,
    alibi_slopes: Any = None,
    beam_pool: Any = None,
) -> None:
    del vllm_flash_attn_version
    _require_eager(query)
    _validate_features(sliding_window_size, logits_soft_cap, s_aux, alibi_slopes)
    metadata = attn_metadata
    if metadata.is_beam_decode and metadata.prefix_indices is None:
        raise ValueError("MLU Beam decode requires explicit Beam row indices")
    if metadata.std_indices is not None:
        out, _ = _paged_attention(
            query[metadata.std_indices],
            key_cache,
            value_cache,
            metadata.std_cu_seqlens_q_device,
            metadata.std_seqlens_kv_device,
            metadata.std_block_table,
            scale,
            metadata.causal,
        )
        output[metadata.std_indices] = out
    if metadata.prefix_indices is None:
        return

    session = require_beam_decode_session(metadata)
    pool = resolve_beam_pool(metadata, beam_pool)
    rows = metadata.num_prefix_indices
    width = getattr(session, "attention_width", session.beam_width)
    if rows != width or metadata.prefix_block_table.shape[0] != 1:
        raise NotImplementedError("MLU eager Beam attention currently supports one session")
    step = session.decode_step
    slot = session.slot
    if not 0 <= step < pool.max_decode_steps or not 0 <= slot < pool.unshared_key.shape[0]:
        raise ValueError("MLU Beam suffix pool index is out of bounds")
    if rows > pool.unshared_key.shape[1]:
        raise ValueError("MLU Beam width exceeds the suffix pool capacity")
    beam_query = query[metadata.prefix_indices]
    prefix_out, prefix_lse = _paged_attention(
        beam_query,
        key_cache,
        value_cache,
        metadata.prefix_cu_seqlens_q_device,
        metadata.prefix_seqlens_kv_device,
        metadata.prefix_block_table,
        scale,
        False,
    )
    # Existing tensor operations pack each beam's valid suffix for TMO varlen.
    suffix_key = pool.unshared_key[slot, :rows, :, : step + 1].permute(0, 2, 1, 3)
    suffix_value = pool.unshared_value[slot, :rows, :, : step + 1].permute(0, 2, 1, 3)
    suffix_key = suffix_key.reshape(-1, pool.num_kv_heads, pool.head_dim).contiguous()
    suffix_value = suffix_value.reshape_as(suffix_key).contiguous()
    cu_query = torch.arange(rows + 1, device=query.device, dtype=torch.int32)
    cu_kv = cu_query * (step + 1)
    suffix_out, suffix_lse = _flash(
        beam_query,
        suffix_key,
        suffix_value,
        cu_query,
        cu_kv,
        1,
        step + 1,
        scale,
        True,
    )
    merged = torch.empty_like(beam_query)
    ops.merge_attn_states(merged, prefix_out, prefix_lse, suffix_out, suffix_lse)
    output[metadata.prefix_indices] = merged

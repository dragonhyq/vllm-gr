# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Single-request eager MLU Beam execution using native forward and Torch ops."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from vllm_gr.config.models import ConstraintMode, configured_constraint_mode
from vllm_gr.ops.npu.unconstrained_topk import select_unconstrained_candidates
from vllm_gr.v1.beam.attention_pool import BeamAttentionPool
from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchFailure,
    BeamBatchOutput,
    BeamBatchResult,
    BeamOutputSequence,
    BeamRequestOutput,
    validate_terminal_beam_batch_result,
)
from vllm_gr.v1.beam.dispatch import (
    GR_RETIRE_OUTPUT_ATTR,
    GR_RETIRE_RESULT_ATTR,
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
    GRRetireMetadata,
    GRRetireResult,
    GRStageKind,
    GRWorkerResult,
    validate_dispatch,
    validate_dispatch_session,
)
from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView


@dataclass
class MLUBeamSession:
    """The eager attention contract; no CUDA/NPU context or graph resources."""

    beam_width: int
    max_decode_steps: int
    pools: list[Any]
    decode_step: int = 0
    slot: int = 0
    is_active: bool = True

    @property
    def context(self) -> MLUBeamSession:
        return self


class MLUBeamState:
    """Device-resident scores/history with the shared stable parent-order rule."""

    def __init__(self) -> None:
        self.tokens: torch.Tensor | None = None
        self.scores: torch.Tensor | None = None
        self.parents: torch.Tensor | None = None

    def advance(self, candidates: Any, width: int, *, final: bool) -> torch.Tensor:
        scores = candidates.logprobs.masked_fill(~candidates.valid_mask, -torch.inf)
        if self.scores is not None:
            scores = scores + self.scores[:, None]
        # Flattened order is (parent, candidate rank); stable sorting preserves
        # the CPU reference's tie rule. Group surviving nonterminal rows by
        # parent, as CUDA/NPU do, before preparing the next physical batch.
        selected = torch.argsort(scores.flatten(), descending=True, stable=True)[:width]
        parents = selected // scores.shape[1]
        if not final:
            order = torch.argsort(parents, stable=True)
            selected, parents = selected[order], parents[order]
        tokens = candidates.token_ids.flatten()[selected].to(torch.int64)
        self.scores = scores.flatten()[selected]
        history = self.tokens
        parent_history = self.parents
        self.tokens = (
            tokens[:, None]
            if history is None
            else torch.cat((history[parents], tokens[:, None]), dim=1)
        )
        self.parents = (
            parents[:, None]
            if parent_history is None
            else torch.cat((parent_history[parents], parents[:, None]), dim=1)
        )
        return parents


def _consume_native_state(worker: Any, output: Any) -> None:
    # CUDA V2's persistent Decode/retirement bookkeeping, using vendor methods.
    worker.finish_requests(output)
    worker.free_states(output)
    if output.scheduled_new_reqs:
        worker.add_requests(output)
    worker.update_requests(output)
    worker.block_tables.apply_staged_writes()


def _build_decode_common_metadata(
    *,
    width: int,
    seq_len: int,
    positions: torch.Tensor,
    prefix_blocks: torch.Tensor,
    slots: torch.Tensor,
) -> Any:
    """Describe physical one-token Beam rows to the vendor forward context."""
    from vllm_mlu.v1.attention.backends.utils import (
        MLUCommonAttentionMetadata,
        MLUInferMode,
    )

    device = positions.device
    query_cpu = torch.arange(width + 1, dtype=torch.int32, device="cpu")
    lengths_cpu = torch.full((width,), seq_len, dtype=torch.int32, device="cpu")
    query = query_cpu.to(device)
    # This is forward-context metadata, not a native paged-attention plan:
    # each row shares the native prefix, while GR's layer metadata owns the
    # separate suffix pools. PAD slots prevent native writes to shared KV.
    return MLUCommonAttentionMetadata(
        query_start_loc=query,
        query_start_loc_cpu=query_cpu,
        seq_lens=lengths_cpu.to(device),
        seq_lens_cpu_upper_bound=lengths_cpu,
        num_reqs=width,
        num_actual_tokens=width,
        max_query_len=1,
        max_seq_len=seq_len,
        block_table_tensor=prefix_blocks.repeat(width, 1),
        slot_mapping=slots,
        causal=True,
        positions=positions,
        is_prefilling=torch.zeros(width, dtype=torch.bool, device=device),
        seq_start_loc=query * seq_len,
        num_prefill_query_tokens=0,
        num_prefill_kv_tokens=0,
        infer_mode=MLUInferMode.DECODE_ONLY,
    )


def _validate_capabilities(worker: Any, metadata: Any) -> None:
    from vllm_gr.config import GRConfig

    config = worker.vllm_config
    model, parallel = worker.model_config, worker.parallel_config
    if worker.device.type != "mlu" or not model.enforce_eager:
        raise NotImplementedError("MLU Beam execution currently requires eager MLU mode")
    if not worker.use_async_scheduling or any(
        getattr(parallel, name, 1) != 1
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        )
    ):
        raise NotImplementedError("MLU eager Beam requires async scheduling and TP/PP/DP/CP=1")
    if (
        worker.speculative_config is not None
        or worker.lora_config is not None
        or model.is_encoder_decoder
        or getattr(model, "is_hybrid", False)
        or getattr(model, "is_multimodal_model", False)
        or config.kv_transfer_config is not None
        or getattr(config, "ec_transfer_config", None) is not None
        or (
            getattr(config, "aux_output_config", None)
            and config.aux_output_config.enable_return_routed_experts
        )
    ):
        raise NotImplementedError("MLU eager Beam currently requires a plain text decoder")
    if model.dtype not in (
        torch.float16,
        torch.bfloat16,
    ) or worker.cache_config.cache_dtype not in ("auto", "float16", "bfloat16"):
        raise NotImplementedError("MLU eager Beam requires unquantized FP16/BF16 KV")
    params = metadata.beam_params
    view = BeamBatchExecutionView.from_stage_metadata(metadata)
    gr = GRConfig.from_vllm_config(config)
    if (
        view.batch_size != 1
        or params.dynamic_beam_widths is not None
        or params.per_parent_beam_widths is not None
        or view.candidate_top_k < view.beam_width
        or not params.ignore_stop_tokens
        or configured_constraint_mode(gr) is not ConstraintMode.NONE
        or gr.beam_sid_stage_vocab
    ):
        raise NotImplementedError(
            "MLU eager Beam currently supports one prompt, fixed widths, "
            "full-vocabulary unconstrained candidates and ignore_stop_tokens=True"
        )
    groups = worker.kv_cache_config.kv_cache_groups
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    if len(groups) != 1 or not isinstance(groups[0].kv_cache_spec, FullAttentionSpec):
        raise NotImplementedError("MLU eager Beam requires one full-attention KV group")
    spec = groups[0].kv_cache_spec
    if getattr(spec, "head_size_v", spec.head_size) != spec.head_size:
        raise NotImplementedError("MLU Beam requires equal key/value head dimensions")


class MLUEagerBeamStageRunner:
    def __init__(self, worker: Any) -> None:
        self.worker = worker
        self.previous: Any = None
        self.dispatch_session: Any = None
        self.request: Any = None
        self.pending: Any = None
        self.logits: torch.Tensor | None = None
        self.state = MLUBeamState()
        self.attention_session: MLUBeamSession | None = None
        self.execution_error: Exception | None = None
        self.closed = False

    def execute(self, scheduled: Any, original: Any, *args: Any, **kwargs: Any) -> None:
        if self.execution_error is not None:
            raise RuntimeError(
                "MLU Beam owner failed; retirement is required"
            ) from self.execution_error
        if self.closed or self.pending is not None:
            raise RuntimeError("MLU Beam sampling must consume the previous dispatch")
        session = self.dispatch_session or validate_dispatch_session(scheduled)
        (metadata,) = validate_dispatch(scheduled, previous=self.previous, session=session)
        _validate_capabilities(self.worker, metadata)
        if self.previous is None and metadata.session_init is None:
            raise ValueError("MLU Beam Prefill requires session_init")
        # Native v0.33 retains mutable request containers. This eager adapter
        # owns a complete host copy; it does not alter EngineCore's output.
        local = deepcopy(scheduled)
        self.dispatch_session = session
        if self.previous is None:
            self.request = metadata.session_init
        self.pending = metadata
        try:
            if metadata.kind is GRStageKind.PREFILL:
                result = original(self.worker, local, *args, **kwargs)
                native = self.worker.execute_model_state
                if result is not None or native is None or native.hidden_states is None:
                    raise RuntimeError("MLU Prefill did not publish native hidden states")
                batch = native.input_batch
                if tuple(batch.req_ids) != metadata.request_ids:
                    raise RuntimeError("MLU Prefill request order differs from Beam metadata")
                self.logits = self.worker.model.compute_logits(
                    native.hidden_states[batch.logits_indices]
                )
                self.worker.postprocess_num_computed_tokens(batch)
                self.worker.execute_model_state = None
            else:
                _consume_native_state(self.worker, local)
                self.logits = self._decode(metadata)
        except Exception as error:
            self.execution_error = error
            self.worker.execute_model_state = None
            raise

    def _decode(self, metadata: Any) -> torch.Tensor:
        from vllm.forward_context import BatchDescriptor, set_forward_context
        from vllm.model_executor.models.utils import extract_layer_index
        from vllm_mlu.model_executor.models.sp_utils import set_sp_forward_context
        from vllm_mlu.v1.attention.backends.utils import COMMON_METADATA_STR

        from vllm_gr.v1.attention.backends.beam_attn_metadata import (
            BeamAttentionMetadata,
        )

        worker = self.worker
        group = worker.kv_cache_config.kv_cache_groups[0]
        spec = group.kv_cache_spec
        width = metadata.beam_params.beam_width
        if self.state.tokens is None or metadata.kv_blocks is None or len(metadata.kv_blocks) != 1:
            raise RuntimeError("MLU Decode requires previous winners and published KV blocks")
        if self.attention_session is None:
            indices = [extract_layer_index(name) for name in group.layer_names]
            if len(set(indices)) != len(indices):
                raise NotImplementedError("MLU Beam requires one attention layer per model layer")
            pools: list[Any] = [None] * (max(indices) + 1)
            steps = metadata.beam_params.max_tokens - 1
            for index in indices:
                pools[index] = BeamAttentionPool(
                    1,
                    width,
                    spec.num_kv_heads,
                    steps,
                    spec.head_size,
                    worker.device,
                    worker.dtype,
                )
            self.attention_session = MLUBeamSession(width, steps, pools)
        session = self.attention_session
        session.decode_step = metadata.decode_step
        prompt = metadata.num_prompt_tokens
        position = prompt + metadata.decode_step
        device = worker.device
        ids = self.state.tokens[:, -1].contiguous()
        # Native MLU RoPE consumes int32 positions; preserve its buffer contract.
        positions = worker.input_buffers.positions.new_full((width,), position)
        blocks = torch.tensor([metadata.kv_blocks[0]], dtype=torch.int32, device=device)
        query = torch.tensor([0, width], dtype=torch.int32, device=device)
        lengths = torch.tensor([position + 1], dtype=torch.int32, device=device)
        slots = torch.full((width,), -1, dtype=torch.int64, device=device)
        attention = BeamAttentionMetadata(
            num_actual_tokens=width,
            max_query_len=width,
            query_start_loc=query,
            max_seq_len=position + 1,
            seq_lens=lengths,
            block_table=blocks,
            slot_mapping=slots,
            is_beam_decode=True,
            beam_session=session,
            prefix_indices=torch.arange(width, dtype=torch.int64, device=device),
            num_prefix_indices=width,
            prefix_cu_seqlens_q_device=query,
            prefix_seqlens_kv_device=torch.tensor([prompt], dtype=torch.int32, device=device),
            prefix_block_table=blocks,
            prefix_max_q_len=width,
            prefix_max_seq_len=prompt,
            suffix_max_q_len=1,
            block_size=spec.block_size,
            incremental_suffix_kv_len=metadata.decode_step + 1,
        )
        by_layer: dict[str, Any] = dict.fromkeys(group.layer_names, attention)
        by_layer[COMMON_METADATA_STR] = _build_decode_common_metadata(
            width=width,
            seq_len=position + 1,
            positions=positions,
            prefix_blocks=blocks,
            slots=slots,
        )
        with (
            set_forward_context(
                by_layer,
                worker.vllm_config,
                num_tokens=width,
                batch_descriptor=BatchDescriptor(num_tokens=width),
                slot_mapping=dict.fromkeys(group.layer_names, slots),
                skip_compiled=True,
            ),
            set_sp_forward_context(
                by_layer,
                worker.vllm_config,
                width,
                positions=positions,
                model=worker.model,
            ),
        ):
            hidden = worker.model(input_ids=ids, positions=positions)
        return worker.model.compute_logits(hidden)

    def sample(self, grammar_output: Any = None) -> Any:
        if self.execution_error is not None:
            raise RuntimeError("MLU Beam execution failed") from self.execution_error
        if grammar_output is not None or self.pending is None or self.logits is None:
            raise RuntimeError("MLU Beam sampling requires one pending stage and no grammar")
        metadata = self.pending
        try:
            view = BeamBatchExecutionView.from_stage_metadata(metadata)
            if self.logits.ndim != 2 or self.logits.shape[0] != view.num_execution_rows:
                raise RuntimeError("MLU logits do not match the Beam execution rows")
            candidates = select_unconstrained_candidates(
                self.logits,
                torch.tensor([metadata.beam_params.temperature], device=self.logits.device),
                view.candidate_top_k,
            )
            parents = self.state.advance(candidates, view.beam_width, final=metadata.is_last_stage)
            if self.attention_session is not None and not metadata.is_last_stage:
                # Read each parent's complete valid suffix before overwriting.
                # Device index_select materializes a separate tensor, so repeated
                # parents and cycles cannot overwrite data needed by another row.
                extent = metadata.decode_step + 1
                for pool in self.attention_session.pools:
                    if pool is not None:
                        for cache in (pool.unshared_key, pool.unshared_value):
                            valid = cache[0, :, :, :extent]
                            valid.copy_(valid.index_select(0, parents))
            terminal = self._terminal_result() if metadata.is_last_stage else None
            if metadata.is_last_stage:
                self.close()

            from vllm.v1.outputs import ModelRunnerOutput

            req_id = metadata.request_ids[0]
            output = ModelRunnerOutput(
                req_ids=[req_id],
                req_id_to_index={req_id: 0},
                sampled_token_ids=[[0]],
            )
            setattr(
                output,
                GR_WORKER_RESULT_ATTR,
                {
                    metadata.session_id: GRWorkerResult(
                        metadata.session_id,
                        metadata.dispatch_id,
                        metadata.stage_index,
                        1,
                        metadata.output_device_state,
                        metadata.is_last_stage,
                        terminal_result=terminal,
                        resources_released=metadata.is_last_stage,
                        owner_generation=metadata.owner_generation,
                    )
                },
            )
            self.previous = metadata
            self.pending = self.logits = None
            self.worker.execute_model_state = None
            return output
        except Exception as error:
            self.execution_error = error
            raise

    def _terminal_result(self) -> BeamBatchResult:
        request = self.request
        options = request.output_options
        tokens = self.state.tokens.cpu().tolist()
        scores = self.state.scores.cpu().tolist()
        parents = self.state.parents.cpu().tolist()
        sequences = tuple(
            BeamOutputSequence(
                tuple(tokens[index]),
                score if options.include_cumulative_logprob else None,
                tuple(parents[index]) if options.include_parent_beam_indices else (),
                "length",
                None,
            )
            for index, score in enumerate(scores)
            if math.isfinite(score)
        )[: options.num_return_sequences]
        # All admitted paths have equal length and ignore stop tokens, so length
        # normalization cannot change the final score ordering.
        result = BeamBatchResult(
            request.protocol_version,
            request.batch_id,
            BeamBatchOutput((BeamRequestOutput(0, request.items[0].request.request_id, sequences),))
            if sequences
            else None,
            None
            if sequences
            else BeamBatchFailure("no_valid_beams", "No finite Beam candidates", 0),
        )
        validate_terminal_beam_batch_result(request, result)
        return result

    def close(self) -> None:
        # The eager adapter uses the current stream. Finish device reads/writes
        # before acknowledging release of the owner or dropping its KV buffers.
        torch.mlu.current_stream(self.worker.device).synchronize()
        if self.attention_session is not None:
            self.attention_session.is_active = False
            self.attention_session.pools.clear()
        self.attention_session = None
        self.state = MLUBeamState()
        self.logits = None
        self.closed = True


def execute_retirement(worker: Any, scheduled: Any) -> Any:
    metadata = getattr(scheduled, GR_RETIRE_OUTPUT_ATTR)
    if (
        not isinstance(metadata, GRRetireMetadata)
        or scheduled.total_num_scheduled_tokens
        or scheduled.num_scheduled_tokens
        or getattr(scheduled, GR_STAGE_OUTPUT_ATTR, None)
        or metadata.session_id not in scheduled.finished_req_ids
    ):
        raise ValueError("Invalid MLU Beam retirement frame")
    previous = getattr(worker, "_gr_mlu_retirement", None)
    if previous is not None and previous[0] == metadata:
        return previous[1]
    owner = getattr(worker, "_gr_mlu_stage_runner", None)
    if (
        owner is None
        or owner.closed
        or owner.pending is not None
        or owner.previous is None
        or owner.previous.session_id != metadata.session_id
        or owner.previous.owner_generation != metadata.owner_generation
    ):
        raise RuntimeError("MLU retirement has no matching drained Beam owner")
    _consume_native_state(worker, deepcopy(scheduled))
    owner.close()
    from vllm.v1.outputs import ModelRunnerOutput

    output = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
    setattr(
        output,
        GR_RETIRE_RESULT_ATTR,
        GRRetireResult(
            metadata.session_id,
            metadata.owner_generation,
            metadata.retire_id,
            True,
        ),
    )
    worker._gr_mlu_retirement = (metadata, output)
    return output


def close_stage_runner(worker: Any) -> None:
    owner = getattr(worker, "_gr_mlu_stage_runner", None)
    if owner is not None and not owner.closed:
        owner.close()
    worker._gr_mlu_stage_runner = None
    worker._gr_mlu_retirement = None
    worker._gr_mlu_execution_error = None

# Runtime patches and upstream dependencies

Three files describe the same patch set:

| File | Role |
| --- | --- |
| [patch_inventory.yaml](../patch_inventory.yaml) | Authoritative data: installer, target, process roles, risk, tests and removal condition. |
| [Runtime patch inventory](../runtime_patch_inventory.md) | Checked index: active IDs in contract-test order, scope, backend ownership, owner, installer and risk. |
| This page | Rationale: the behavior each hook protects and the upstream assumption it depends on. |

Backend ownership identifies which provider declares the patch. Installation
can still happen later in a worker or during native configuration.

## GR-PATCH-000 — Runtime Plugin Activation

Scope: `core`; backend ID: `null`

The general plugin sets ModelRunner defaults, activates the current provider,
and validates its environment before passing that provider into patch
installation. The frontend and EngineCore use their own ordered plans. Each
patch module verifies its bindings so repeated initialization can reuse an
existing wrapper.

## GR-PATCH-001 — CUSTOM Attention Selector Compatibility

Scope: `core`; backend ID: `null`

An explicit CUSTOM choice, including one restored from serialized GR
configuration, must reach the registered Beam backend. Changes to the private
attention selector or its cache can silently select a different backend.
On MLU, the wrapper accepts the vLLM 0.33 `_run_kv_cache_dtype` keyword and
forwards it unchanged to the cached native fallback selector. CUDA/NPU keep
their existing signature. CUSTOM selection bypasses the native selector as
before, retaining any GR backend import or implementation error.

## GR-PATCH-002 — NPUWorker Initialization Hook

Scope: `backend`; backend ID: `ascend`

The [NPU startup hooks](../../../vllm_gr/v1/worker/npu/startup_patch.py) install
worker utilities before native device initialization constructs the runner.
An early import can precede worker availability, so later initialization
retries the installation. The platform configuration filter can run separately.

## GR-PATCH-003 — FlatLogprobs Fast Append Override

Scope: `core`; backend ID: `null`

The frontend appends flattened log probabilities through the GR implementation
of the native fast-append method in
[logprobs_patch.py](../../../vllm_gr/v1/engine/logprobs_patch.py).
Its arguments and the upstream flattened representation must stay compatible.
The patch test checks offsets, ranks, and decoded tokens for empty and nonempty
entries and omitted optional columns.

## GR-PATCH-004 — OpenAI Beam Serving Override

Scope: `core`; backend ID: `null`

The serving frontend groups prompts into one Beam submission and formats their
ranked candidates for the OpenAI-compatible API. The wrapper depends on the
native serving and protocol methods it replaces. The patch validates the Chat
and Completion sampling methods and the Completion `vllm_xargs` schema for
integer lists. A missing binding or a mismatched schema triggers reinstallation.

The MLU vLLM 0.33 serving caller passes `priority`, including the default zero.
The dispatcher accepts an explicit priority only on MLU and forwards it through
the online request to native input processing. The grouped completion adapter
uses the same native header/body priority resolver as single-prompt serving.
An omitted priority retains GR's arrival-based value, preserving existing
CUDA/NPU calls. Argument forwarding does not establish priority scheduling:
GR's Beam batch ownership policy remains arrival ordered.

## GR-PATCH-005 — EngineArgs Configuration Override

Scope: `core`; backend ID: `null`

GR serve and GRLLM pass a validated snapshot through native
`EngineArgs.additional_config`. The GR hook on `create_engine_config` resolves
device-dependent settings and admits the backend plan before worker startup.

## GR-PATCH-007 — EngineCore Client and AsyncLLM Extensions

Scope: `core`; backend ID: `null`

Client extensions submit a whole Beam session and route its wait, abort and
release operations. A rejected submission must preserve the identity that the
frontend uses to complete the request.

## GR-PATCH-008 — EngineCoreRequestType Extension

Scope: `core`; backend ID: `null`

GR adds a wire discriminator for grouped session submission to the native
request type. An upstream enum value collision could send a valid message to
the wrong handler.

## GR-PATCH-009 — EngineCore Lifecycle and Socket Processing

Scope: `core`; backend ID: `null`

EngineCore admits grouped submissions and routes their terminal results and
retirement acknowledgements. Native socket dispatch and output collection must
keep ordinary requests progressing alongside GR work.

## GR-PATCH-010 — Scheduler Beam Override

Scope: `core`; backend ID: `null`

Scheduling attaches Beam stage and retirement metadata to native scheduler
outputs. Logical token accounting and outstanding worker receipts must agree
before the next owner can be admitted.

The installer and validator accept an optional explicit `scheduler_cls` for
provider-owned overrides. Calls without it retain the upstream module target
used by CUDA/NPU. Both paths use the same admission, prompt-cache guard, stage
metadata, completion and retirement logic. Nested wrappers must execute these
operations once while retaining the vendor `super().schedule()` chain.

Admission defaults to the native `waiting` and `skipped_waiting` queues. A
provider can install `_gr_admission_queue_context(owner_ids)` on its scheduler
class to filter a different queue layout. The context must preserve native
ordering and mutations and restore the original collections on exceptions.
Owner selection, KV reservation and stage backpressure remain shared.

## GR-PATCH-022 — Prefix-cache hash memoization

Scope: `core`; backend ID: `null`

Bounded per-process memoization avoids hashing identical prefix blocks and
prompts repeatedly. Reinstallation checks the wrappers and the EngineCore
binding that constructs the hasher.

On MLU, the factory accepts and forwards vLLM 0.33's optional
`num_hash_lookahead_tokens`. Nonzero lookahead returns the native request hasher
because GR's whole-chain memo omits the following tokens and would emit hashes
before those tokens exist. Zero lookahead retains the shared chain memo;
CUDA/NPU retain their two-argument factory and native calls. The patch markers,
memo diagnostics and EngineCore binding still identify the installed factory.

The native request hasher continues to call the patched per-block hash function.
Its lookahead extra keys participate in that function's digest and memo key,
but GR's SHA-256 encoding is not stock vLLM serialization. This adaptation
does not establish external-cache interoperability or speculative decoding
support. See [MLU compatibility tracking](../mlu_compatibility_tracking.md).

## GR-PATCH-011 — GPU ModelRunner V1/V2 Overrides

Scope: `backend`; backend ID: `cuda`

CUDA ModelRunner V1 and V2 adapt native input preparation, stage execution and
output completion. Request ordering, graph-visible storage and native lifecycle
boundaries have to agree. See the [runner contract](model_runner_v2.md).

## GR-PATCH-012 — NPU ModelRunner Override

Scope: `backend`; backend ID: `ascend`

The Ascend worker adapts native rows and attention metadata for grouped Beam
execution. Cached prefixes, real row lengths and parent selection must stay
aligned with native block tables. See
[Ascend Beam Decode execution](../feature/npu_decode_execution.md).

The [worker stage projection](../../../vllm_gr/v1/worker/npu/stage_projection.py)
wraps the instance logits processor, runner reload and native worker weight-update
methods. It invalidates the projection before native writes and refreshes it after
finalization and device synchronization. The
[operator helper](../../../vllm_gr/ops/npu/stage_projection.py) owns the fixed-address
weights, biases and proxies.

## GR-PATCH-013 — CUSTOM Attention Backend Registration

Scope: `core`; backend ID: `null`

GR registers Beam attention in the upstream CUSTOM slot. Registration and the
selector have separate responsibilities. Another extension claiming that slot
can invalidate the selection.

On MLU, the provider supplies the eager `beam_attn_mlu` execution module. Its
cache views match native TMO's per-head K/V packing in contiguous LBHNC storage;
using the facade's CUDA/NPU token-interleaved views would read different cache
locations. The facade restricts MLU to that layout and disables quantized query
input. Existing TMO cache-write and flash-attention operations, the vendor
attention-state merge, and shared eager suffix writes implement attention
without new kernels. This covers one Beam session per dispatch, not the missing
MLU runner stage execution. Graph capture and unsupported attention features
raise explicitly; see [MLU compatibility tracking](../mlu_compatibility_tracking.md).

## GR-PATCH-015 — Sliced Vocabulary Projection

Scope: `core`; backend ID: `null`

The worker binds the loaded table's token window to each model's logits processor.
Processors without a window use native full-vocabulary projection. Sliced
projection also adapts log-probability collection. Head-derived caches must stay valid across
native weight reload.

## GR-PATCH-016 — NPU-aware Spawn Start Method

Scope: `backend`; backend ID: `ascend`

Ascend subprocesses must use a compatible start method when the parent has
initialized the device. The Ascend provider installs spawn protection before
the native worker initialization hook.

## GR-PATCH-018 — Ascend Penalties Warmup OOM Guard

Scope: `backend`; backend ID: `ascend`

Ascend warmup can allocate penalty tensors that exceed available memory. The
[NPU startup hook](../../../vllm_gr/v1/worker/npu/startup_patch.py) handles the
warmup allocation failure while preserving other execution errors.
The inventory lists no behavioural test for this guard.

## GR-PATCH-019 — Ascend additional_config Sanitizer

Scope: `backend`; backend ID: `ascend`

The [Ascend configuration patch](../../../vllm_gr/platforms/ascend/config_patch.py)
retains GR settings while filtering keys passed to the native configuration
reader. Fresh EngineCore processes install the common plan before native
construction. Worker startup also refreshes the filter's native bindings.

## GR-PATCH-021 — Contiguous Beam Prefix KV

Scope: `backend`; backend ID: `ascend`

The [NPU cache layout patch](../../../vllm_gr/v1/worker/npu/kv_layout_patch.py)
lets eligible Ascend xllm Beam layers request contiguous prefix K/V at native
allocation. Attention row selection is a separate reader (`select_beam_rows`)
that checks layout, strides and format before using the paged block table.
An unsuitable layout keeps the gather path.

## GR-PATCH-023 - MLU V2 Runner Lifecycle and Eager Beam

Scope: `backend`; backend ID: `mlu`

The deferred MLU Worker hook installs lifecycle routing on the vendor V2
runner. Its execute/sample wrappers route nonempty GR stage metadata to the
[eager MLU adapter](../../../vllm_gr/v1/worker/mlu/beam_stage_runner.py), route
retirement separately, and preserve native execution for ordinary/empty
frames. Legacy frames remain rejected. An execute failure must also reach
sampling; native logits from an earlier request must never mask it.

Validate shared dispatch identity, ordering, counts and the admitted execution
scope before model work. Prefill uses native forward, consumes its hidden-state
handoff, and advances the native computed-token counter exactly once. Decode
retains native request/KV bookkeeping but runs physical Beam rows through the
existing MLU attention adapter. Shared pool objects hold suffix KV. Candidate
selection uses the existing Torch-only unconstrained implementation; stable
score order determines survivors, then stable parent grouping defines the next
row order. Reorder suffix KV using separate gathered tensors before overwriting.

Decode also supplies vendor `MLUCommonAttentionMetadata` alongside the per-layer
Beam metadata. Common query offsets describe one token per physical Beam row;
sequence lengths include the prefix and current suffix, with Decode-only mode.
The repeated block table describes the shared native prefix and slots remain
PAD slots. This common object serves vendor forward-context consumers; it is
not a native paged-attention plan for the suffix, which stays in GR pools.
Native attention fallback and parallel consumers require separate adaptation.
Decode position tensors inherit dtype and device from the native MLU position
buffer, keeping the vendor rotary-embedding input contract.

Return shared worker receipts and validated terminal result schemas. Finish
current-stream device work before acknowledging resource release. Explicit
retirement requires the matching drained owner/generation and caches duplicate
receipts. Reload/shutdown close the owned buffers. The initial scope is one
prompt, fixed widths, FP16/BF16 unquantized KV, a plain decoder, async V2,
TP/PP/DP/CP=1, unconstrained candidates, ignored stop tokens and eager mode.
Graphs, constraints, dynamic widths and multiple prompts remain rejected.
Host dispatch copying, synchronous terminal copies and on-demand suffix pool
allocation are deliberate initial costs; performance and memory budgeting
require later work. Runtime integration remains subject to the tracking record.

## GR-PATCH-024 - MLU additional_config Filter

Scope: `backend`; backend ID: `mlu`

The [MLU configuration patch](../../../vllm_gr/platforms/mlu/config_patch.py)
follows the Ascend scoped-filter approach. Native `init_mlu_config` receives
the same engine config with GR transport keys temporarily excluded from
`additional_config`. A `finally` block restores the original mapping, including
the serialized backend plan, even if vendor validation fails. Only reserved GR
keys are filtered; other unknown fields remain subject to native validation.
The MLU provider installs this wrapper during runtime initialization. Native
worker plugin loading installs it before worker construction in a fresh process.
Current MLU platform, EngineCore and Worker callers import the function locally;
future module-global aliases or earlier config initialization need reassessment.

## GR-PATCH-025 - MLU EngineCore Startup Order

Scope: `backend`; backend ID: `mlu`

The [MLU process entry](../../../vllm_gr/platforms/mlu/engine_core_patch.py)
is installed by the provider's configuration service after native configuration,
following the Ascend entrypoint pattern. It must remain module-level so spawn
can import it independently of the parent's patched class state.

In the child, install the vendor config filter, import `vllm_mlu.mlu_hijack`,
complete general-plugin loading, then install and validate the shared GR child
plan and the selected MLU scheduler (GR-PATCH-027). Before delegating directly to
`EngineCoreProc_MluHijack.run_engine_core`,
verify that unwrapping `EngineCore.__init__` reaches the vendor initializer.
The vendor entrypoint's subsequent plugin-loading call is already satisfied.
The initializer is executed once by normal engine construction, preserving
vendor state and GR post-load readiness work.

Importing the vendor patches before general plugins avoids dependence on their
enumeration order. Installing GR after plugin loading repairs any overwritten
child methods. Repeated configuration finalization reasserts the dedicated
entrypoint. CUDA and NPU do not select this service. This adapter depends on
private vendor names and their saved native entrypoint; changes to vendor
startup or additional plugins that replace initialization need reassessment.

## GR-PATCH-026 - MLU NumPy Token Staging

Scope: `backend`; backend ID: `mlu`

The [MLU staging adapter](../../../vllm_gr/v1/worker/mlu/numpy_staging.py)
reuses CUDA's NumPy contents and transfer wrappers after native vendor imports.
Installation occurs inside the deferred MLU runner installer before native
ModelRunner construction. Other platform installers do not select this adapter.

NumPy arrays use an explicit empty-size check and retain bulk collection.
Successful nonempty array staging must update `_staged_write_max_len`: native
vLLM 0.33 uses it to launch every 1024-element write chunk. Without this update,
the older CUDA adapter silently truncates long writes on the multi-row path.
Native list and scalar writes already maintain the field; native clearing
resets it. Invalid arrays must not queue writes. Read-only inputs and existing
transfer output buffers retain the shared adapter's semantics.

The patch leaves vendor `apply_write`, fused `apply`, buffer-pool implementations
and device kernels in place. Repeated installation is idempotent; validation
checks both the stage wrapper and its captured transfer binding. An existing
NumPy wrapper with an unexpected or replaced binding raises instead of nesting
another owner. This does not validate concurrent streams, graphs, model-level
token ownership or Beam execution. Native buffer-layout and write-contract
changes require reassessment when the vendor runtime changes.

## GR-PATCH-027 - MLU Selected Scheduler Beam Dispatch

Scope: `backend`; backend ID: `mlu`

MLU selects a separate vendor `AsyncScheduler` whose inherited
`SchedulerWithProfiler.schedule()` does not delegate to the upstream method.
The [MLU installer](../../../vllm_gr/platforms/mlu/scheduler_patch.py) resolves
the class through `config.scheduler_config.get_scheduler_cls()` and supplies
that class to the shared GR installer and validator. The shared wrapper retains
the original vendor algorithm and forwards `throttle_prefills`; existing
inherited GR hooks are not wrapped twice.

Install during MLU configuration finalization for the parent/in-process path,
and reassert in the spawned child after vendor plugins and the common GR plan.
The validator checks the configured class, not just the upstream scheduler.
Only the standard vendor `AsyncScheduler` is admitted for Beam; an unrelated
or custom scheduler is rejected. Non-Beam configurations are unchanged.

The installer also binds and validates `mlu_admission_queues` as the class's
`_gr_admission_queue_context`. During an owned scheduling pass, both `waiting`
and `kv_holding_waiting` expose only that owner's requests through the shared
queue view. Native scheduling retains its KV-holder-first scan, FCFS/priority
ordering and per-step skipped-request requeue operations. A filtered view of
`deferred_waiting` keeps counts consistent with the visible queues while
writing additions/removals to the original set. It is bookkeeping, not a third
selection queue. Restore all three collection identities after success or
failure, retaining native changes. Do not create a synthetic `skipped_waiting`.

The queue hook is installed only on the selected MLU class; CUDA/NPU retain the
default queue context. Repeated installation repairs a missing/replaced hook
without nesting the schedule wrapper. No-session calls bypass queue filtering.
Future native operations on the deferred set or queue types require review.

This installs Beam admission, prompt-cache protection, stage metadata and
retirement dispatch. It does not supply the missing MLU worker Beam stages.
Validation must distinguish metadata reaching the worker from successful Beam
execution. Native scheduler API changes, alternative launch paths, parallel
execution, completion and retirement still require runtime validation.

## Retired patches

Retired IDs stay in the YAML inventory for traceability. They have no active
installer.

| ID | Historical target | Why it was removed |
| --- | --- | --- |
| `GR-PATCH-006` | `OpenAIServingModels.__init__` | Legacy Catalog serving patch and its tests were removed; canonical V1 constraints load in the worker. |
| `GR-PATCH-014` | LMCache request tracking | Experimental LMCache integration removed; no supported version pin or vLLM 0.30 validation. |
| `GR-PATCH-017` | `NPUWorker.profile_memory` | vllm-ascend 0.30.0rc2 replaced the method with `log_memory_stats`; no native method remained to wrap. |
| `GR-PATCH-020` | `vllm.config.ModelConfig.maybe_untie_word_embeddings` | Model-specific tied-embedding compatibility was retired with its deployment recipe. |

## Maintenance rules

1. Update the YAML entry, the checked index and the rationale here together.
2. Keep the patch ID, scope, backend ID and ordering stable; the
   [upstream contract test](../../../tests/contract/test_upstream_patch_contracts.py)
   verifies them.
3. Run the tests or validators listed for the changed patch. Do not cite a
   test that does not exercise the hook.
4. When upgrading vLLM, recheck private methods, descriptor types, wire tags and
   process-specific bindings.

Retired IDs stay in the YAML inventory for traceability. Benchmark-only shims
belong to their benchmark and do not participate in production startup. See
the [installation guide](../../getting_started/installation/README.md) for
supported runtime requirements.

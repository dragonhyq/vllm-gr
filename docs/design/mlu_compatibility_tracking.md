# MLU compatibility tracking

This document tracks unchanged GR integration code that startup has reached on
the MLU vLLM 0.33 runtime, but whose runtime behavior still needs validation.
It also records known deferred incompatibilities and the current startup
and request boundary. Successful patch installation is not evidence of working inference.

## Evidence baseline

- Recorded: 2026-11-26.
- GR checkout base: `e6cafbeef5ddb3a21e1c68d8608dd528b3144617`, with local MLU
  provider, packaging, CLI import, and native V2 runner lifecycle changes.
- Installed GR distribution: `0.30.1.dev3+ge6cafbeef5dd.dirty.mlu` (editable).
- Native vLLM: `0.33.0+mlu0.17.0.pt214`.
- Vendor distribution: `vllm-mlu==0.17.0+vllm0.33.0.pt214`.
- Python: 3.12.3; Torch: `2.14.0+cpu`; torch-mlu: `1.35.1+torch2.14.0`.
- Device: MLU590-M9DK; driver 6.5.53; CNRT 7.8.0; CNNL 2.3.3;
  CNNLExtra 2.5.1; CNCL 1.30.12; MLU-OPS 1.14.0.
- Requested model: OneRec-1.7B; requested runner: ModelRunner V2.

Reproduce from the GR checkout in the prepared vendor environment, with
`ONEREC_MODEL` set to the downloaded model directory and the native CLI dispatch
patch applied:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve "$ONEREC_MODEL" --gr --enforce-eager \
  --gpu-memory-utilization 0.1
```

The latest run passed Beam/CUSTOM plan admission, vendor config parsing,
worker/device initialization, attention-layer construction, native model
loading, post-load backend-service binding, KV-cache specification collection,
LBHNC layout selection, native profiling, KV allocation and native warmup.
The block-size signature mismatch is resolved:
GR accepts an optional cache specification and forwards it to native TMO on MLU
(MLU-C23). Adding MLU to the worker device allowlist resolved MLU-C22. The
selector, helper import and missing attention implementation failures are also
resolved for this startup path (MLU-C05, MLU-C20, MLU-C21). Worker plan
agreement, service creation, binding, EngineCore initialization and HTTP startup
passed. The MLU hasher factory now accepts and forwards
`num_hash_lookahead_tokens`, resolving MLU-C14. `/health` and `/v1/models`
returned HTTP 200; the model listing identified the requested OneRec checkpoint.
The probe server was then stopped with SIGTERM and exited with code 0. Its API
and engine processes exited, and port 8000 had no listener. There was no new
startup failure. Earlier requests passed priority handling, frontend admission,
vendor EngineCore initialization and NumPy token staging (MLU-C27), but returned
no valid Beam result (MLU-C28). The selected MLU scheduler now uses its native
admission queues (MLU-C29), schedules Prefill and attaches GR stage metadata.
The eager MLU stage adapter now completes the recorded single-prompt request
(MLU-C30): HTTP 200 with 128 distinct, nonempty SID sequences. Vendor common
metadata (MLU-C31) and native int32 Decode positions (MLU-C33) pass the real
model path on card 2. An earlier card-6 retry failed because another process
released memory during profiling (MLU-C32). This is one successful eager,
unconstrained request, not validation of all MLU Beam configurations.
Configuration was BF16, TP=PP=DP=1, ModelRunner V2, max model length 8192,
and native torch.compile/graphs disabled by `--enforce-eager`. Native profiling
ran a dummy forward with attention skipped, then completed sampler warmup.
Native KV initialization and warmup also returned. Prefill, Decode, terminal
result extraction and HTTP response delivery completed for this request.
Graph replay, model accuracy, repeated-request stability and performance are
unverified. Eager mode does not prevent first-use Triton kernel compilation.

The first attempt stopped earlier because another job occupied card 0, leaving
6.11 GiB free against the default 72.55 GiB requested. The retry used an idle
card via `CN_VISIBLE_DEVICES=2`; no other processes were stopped. Logs are
`/tmp/vllm-gr-mlu-attention-startup.log` and
`/tmp/vllm-gr-mlu-attention-startup-card2.log` respectively.

For the latest device-admission run, card 2 had become occupied as well. The
initial default-budget attempt was deliberately stopped during initialization
(`/tmp/vllm-gr-mlu-device-admission-startup.log`). The retry retained
`CN_VISIBLE_DEVICES=2` and used `--gpu-memory-utilization 0.1` to fit the free
memory. It reached the new API failure without an out-of-memory error; its log
is `/tmp/vllm-gr-mlu-device-admission-lowmem-startup.log`. This reduced budget
is a probe setting, not a validated serving capacity.

The block-size fix was retried on card 6 with the same reduced budget. The
180-second probe was terminated by its external timeout (exit 137 after the
termination grace period), without a new Python exception. A stack snapshot
located the worker inside MLU Triton compilation for the native top-k/top-p
sampler during memory profiling. Logs are
`/tmp/vllm-gr-mlu-kernel-block-sizes-startup.log` and
`/tmp/vllm-gr-mlu-kernel-block-sizes-profile-stack.log`. The retry allowed
600 seconds and completed profiling, KV allocation and native warmup in
263.46 seconds, with 22,448 KV-cache tokens. It then reached the hasher
signature error. Its log is
`/tmp/vllm-gr-mlu-kernel-block-sizes-long-startup.log`. The earlier timeout
does not establish a compiler defect, and these timings on a shared device
are not performance results.

The hasher-fix run used the user-selected shared OneRec checkpoint on card 6,
with the same V2/eager/10-percent-memory settings. It loaded 3.32 GiB of model
weights and allocated an LBHNC KV cache for 24,496 tokens. Native profiling,
cache creation and warmup returned in 11.67 seconds with the previously compiled
kernels available. Startup and health-check evidence are in
`/tmp/vllm-gr-mlu-lookahead-startup.log` and
`/tmp/vllm-gr-mlu-lookahead-health.json`. This does not measure inference
performance or validate the configured Beam workload.

### HTTP priority adaptation and resolved frontend rejection

The first real `POST /v1/completions` request returned HTTP 400 because native
vLLM 0.33 passed `priority` to `beam_search_dispatch`, which did not accept it.
The dispatcher now accepts and forwards an explicit priority only on MLU,
including zero. The online adapter uses this value instead of generating an
arrival-based priority. Calls without the argument retain the existing behavior.
Grouped completions on MLU resolve priority through the native header/body
helper as well. The installed method binding and patch ownership are unchanged.

The rerun used OneRec-1.7B on card 6 with the same BF16/V2/eager configuration.
Health and model-list checks passed, then the same single-prompt request used
width 128, `max_tokens=5`, `ignore_eos=true`, and the two SID boundary markers
for three generated positions. It passed dispatch and native input processing
and returned HTTP 400 at `validate_frontend_constraint_mode`:

```text
unconstrained beam_search requires CUDA or NPU
```

At that revision the frontend allowlist excluded MLU. No Beam session was
submitted and no generation stage executed. Work stopped at this failure. The probe server
was terminated and both processes exited; port 8000 was free. Shutdown logged
the native zero-timeout forced EngineCore cleanup and a resource-tracker warning
about one leaked semaphore; graceful in-process cleanup remains unverified.
Evidence is in `/tmp/vllm-gr-mlu-priority-startup.log` and
`/tmp/vllm-gr-mlu-priority-request.json`.

Temporary host probes preserved zero, negative and positive priorities through
`OnlineBeamSearchRequest`; simulated CUDA/NPU calls retained arrival-based
defaults and rejected explicit priority. The 20 existing dispatch and online
entry tests passed after disabling pytest's strict-config default because this
environment lacks the timeout plugin. No new test files were added. These
checks do not establish priority scheduling: GR still grants Beam ownership in
arrival order. Real nonzero-priority, header, chat, grouped-completion, and
cancellation behavior remain unverified beyond host probes and source inspection.

### Unconstrained frontend admission and resolved EngineCore failure

`validate_frontend_constraint_mode` now accepts MLU when the resolved constraint
mode is explicitly `none`, matching provisional worker admission. Dynamic-width
and constraint-table restrictions are unchanged. Temporary host checks passed
for CUDA, NPU, `mlu` and `mlu:0`; CPU/XPU remained rejected and MLU dynamic-width
admission remained rejected. Ruff checks passed. The first host probe used an
omitted constraint configuration instead of explicit `none` and was rejected;
the corrected probe matches the serving configuration. No test files were added.

The same OneRec single-prompt HTTP request, on card 6 with the recorded
BF16/V2/eager configuration, passed frontend validation and session submission.
EngineCore reached the vendor asynchronous step and failed before scheduling
or model execution:

```text
AttributeError: 'EngineCoreProc' object has no attribute 'is_waiting_req'
```

The failing read is in native `EngineCore_MluHijack.step_with_batch_queue`.
The vendor initializer normally sets this field alongside its latency state;
the observed engine instance has the vendor step method without the required
initialization state. A fresh-process inspection reproduced the ordering
problem: GR child initialization captures upstream `EngineCore.__init__` while
general plugins are still unloaded. Loading those plugins later replaces the
live initializer and step method with the vendor methods, but the already
captured GR closure still calls upstream initialization. Upstream initialization
itself loads general plugins, so replacement during that active call cannot
run the vendor initializer retroactively. The inspection is recorded in
`/tmp/vllm-gr-mlu-engine-init-order.log`; it did not construct a model or server.

The ordering fix below loads vendor hooks before GR captures the initializer.
It preserves the complete vendor initialization rather than adding only
`is_waiting_req` or running a second initializer on a constructed engine.
The API returned HTTP 500; the output handler and session cleanup also reported
`EngineDeadError`. No Beam stage result or successful cleanup acknowledgement
was produced. The probe was stopped immediately, both processes exited and
port 8000 was free. Evidence is in
`/tmp/vllm-gr-mlu-frontend-admission-startup.log` and
`/tmp/vllm-gr-mlu-frontend-admission-request.json`.

### MLU child startup ordering and initial token-staging failure

The MLU provider now selects `MLUConfigService`. Its configuration finalizer
installs the module-level `run_mlu_engine_core` process entry (GR-PATCH-025),
following the NPU configuration-finalization pattern. In the spawned child,
the entry installs the config filter, imports `vllm_mlu.mlu_hijack` before
general plugins can run GR, completes plugin loading, and installs/revalidates
the shared GR child patches. It checks that unwrapping `EngineCore.__init__`
reaches `EngineCore_MluHijack.__init__`, then calls the vendor process entry.
Normal construction invokes the vendor initializer once with GR's readiness
wrapper around it. CUDA and NPU keep their existing services and entries.

A temporary host probe verified vendor initializer capture, GR patch validation,
argument/rank forwarding, repeated entry, pickling, entrypoint repair after
replacement, and no installation on simulated CUDA/NPU platforms. An unexpected
initializer was rejected before native entry. The probe constructed no engine;
its log is `/tmp/vllm-gr-mlu-engine-entry-probe.log`. No test files were added.

The real rerun used the same OneRec-1.7B checkpoint, card 6, BF16, V2, eager
execution and 10-percent memory budget. Health and model-list checks returned
HTTP 200. The same width-128, five-token completion passed the previous vendor
state failure, scheduled work and reached `MLUModelRunner.execute_model`.
Native `GPUModelRunner.add_requests()` forwarded NumPy `prefill_token_ids`
through `RequestState.add_request()` to `StagedWriteTensor.stage_write()`, whose
`if not x` check raised:

```text
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

GR's token-buffer/scheduler integration deliberately retains NumPy token arrays.
The MLU lifecycle adapter currently delegates this prefill to the native V2
runner, which still assumes list-like truth testing. At that point this was
the next boundary; no token-staging adaptation had been made. No model forward or Beam stage completed
for this request. The API returned HTTP 500, EngineCore died, and cleanup
reported `EngineDeadError`. The server was stopped immediately; its API and
EngineCore processes exited and port 8000 was free. Shutdown again logged a
forced EngineCore cleanup and one leaked-semaphore warning. Evidence is in
`/tmp/vllm-gr-mlu-engine-entry-startup.log` and
`/tmp/vllm-gr-mlu-engine-entry-request.json`.

The startup wrapper depends on private vendor entrypoint and initializer names.
Other launch paths, process backends, parallel ranks, plugin combinations and
repeated real engine construction remain unverified. The earlier vendor-module
import also changes which hashing-factory alias is captured; see MLU-C14.

### CUDA NumPy staging reuse assessment

CUDA installs `install_numpy_staging()` from
[numpy_staging.py](../../vllm_gr/v1/worker/cuda/numpy_staging.py), which accepts
NumPy token arrays, gathers their contents in bulk and adapts the native
host-to-device transfer. The approach is usable on MLU, but the existing
installer cannot be reused unchanged against this vendor vLLM 0.33 runtime.
Native staged writes now maintain `_staged_write_max_len`, which determines
the number of 1024-element kernel chunks. The CUDA adapter does not update
that field. A temporary device probe confirmed that its unmodified staging
copied only the first 1024 elements of a 2049-element array on the native
multi-row path; the remaining 1025 destination elements were untouched.

In the same temporary process, adding the maximum-write-length update to
the CUDA NumPy staging path fixed the truncation. On the recorded MLU590-M9DK
and vendor vLLM 0.33 environment, eager buffer checks passed lengths
0/1/1024/1025/2049, int32/int64/float32 storage, repeated mixed array/list/scalar
writes, read-only arrays, both ordinary and vendor UVA-emulated contents,
single-row vendor writes, single-row fused writes and multi-row fused fallback.
Transfer into an existing output tensor also preserved its identity and dtype.
The vendor `apply_write` and fused `apply` methods were preserved; their existing
MLU kernels performed the device writes. No new kernels were introduced.

The script and output are `/tmp/vllm-gr-mlu-numpy-staging-probe.py` and
`/tmp/vllm-gr-mlu-numpy-staging-probe.log`. These patches existed only in that
process, which exited successfully. No production staging adapter or test file
was added during that assessment. It established that the MLU installer must
run after vendor patch loading, preserve v0.33 bookkeeping and vendor write
methods, and leave CUDA/NPU activation unchanged. Model inference, concurrent
streams, graph replay and performance were not validated by that probe.

### Installed MLU NumPy staging adapter

GR-PATCH-026 now installs the [MLU adapter](../../vllm_gr/v1/worker/mlu/numpy_staging.py)
inside deferred runner installation, after importing the native MLU runner and
before its construction. The adapter calls the existing CUDA bulk-staging
installer, then wraps successful nonempty array writes to update the v0.33
maximum-write-length field. Native list/scalar paths and vendor device-write
methods remain in place. Validation checks the stage wrapper and its captured
transfer binding together; repeated installation is idempotent, and an
unexpected/replaced NumPy patch binding raises. The platform guard limits
installation to MLU, and CUDA/NPU installers were not changed.

A temporary device probe exercised the production MLU runner installer and
repeated the earlier staging checks on the recorded MLU590/vendor 0.33 runtime.
All lengths, storage dtypes, mixed inputs, repeated writes, read-only inputs,
ordinary/UVA-emulated paths and vendor/fallback fused writes passed. Additional
checks passed for invalid-input rejection without queued writes, a single zero
token, repeated installer identity, simulated non-MLU platforms, and rejection
of an overwritten transfer binding. Evidence is in
`/tmp/vllm-gr-mlu-numpy-staging-installed-probe.log`; its temporary script is
`/tmp/vllm-gr-mlu-numpy-staging-installed-probe.py`. No repository test files were
added. Concurrent streams, graphs, model-level ownership and performance remain
unverified by these buffer probes.

### Prior missing Beam-result failure

The serving rerun used the same OneRec-1.7B checkpoint on card 6, BF16, native
vLLM `0.33.0+mlu0.17.0.pt214`, ModelRunner V2, eager execution and the 10-percent
memory budget. `/health` and `/v1/models` returned HTTP 200. The same width-128,
five-token completion passed token staging and returned HTTP 500 with:

```text
A4/A5 did not attach gr_batch_result to the terminal output
```

`wait_gr_session_async()` received an output with `finished=True`, then
`_terminal_result()` could not find a valid `BeamBatchResult` in
`output.kv_transfer_params["gr_batch_result"]`. It produced the
`worker_result_unavailable` failure, which the frontend raised as
`BeamSearchEngineError`. There was no EngineCore fatal exception or NumPy
truth-value error in this run. A terminal native output is not evidence of
working Beam candidate selection or a valid recommendation result.

Source inspection identified an uncovered scheduler override: the selected
vendor `AsyncScheduler` inherits `SchedulerWithProfiler.schedule()`, which
implements scheduling without delegating to the base `Scheduler.schedule()`.
At that revision, GR's shared scheduler installer wrapped that base method,
including attachment of `gr_stage_metadata`, and its validator checked only
the base class. Consequently the vendor override bypassed this wrapper. The MLU
runner guard rejects marked Beam stages, but it cannot detect missing stage
metadata and delegates unmarked work to native execution. This explains a path
to a native terminal output without a GR result; scheduler/runner/result
integration needs adaptation before claiming Beam support. No scheduler or
result-handling fix was made in this step.

The probe stopped at this failure and terminated the server, which exited with
code 0. API and EngineCore processes were gone and port 8000 was free. Native
shutdown logged forced EngineCore cleanup and a leaked-semaphore warning;
successful Beam retirement and graceful in-process cleanup remain unverified.
Evidence is in `/tmp/vllm-gr-mlu-numpy-staging-startup.log` and
`/tmp/vllm-gr-mlu-numpy-staging-request.json`. No repository test files were added.

### Selected MLU scheduler integration

GR-PATCH-027 now resolves the scheduler from native configuration and applies
the shared installer directly to the vendor `AsyncScheduler`. The shared
installer and validator accept an optional `scheduler_cls`; default CUDA/NPU
calls retain their existing module target. No scheduling algorithm was copied
or replaced: the wrapper invokes the inherited vendor
`SchedulerWithProfiler.schedule()` and forwards `throttle_prefills`, with the
existing GR admission, prompt-cache guard, metadata, completion and retirement
logic around it.

MLU configuration finalization installs and validates this concrete target.
The spawned child repeats installation after vendor plugins and the shared
child plan, before EngineCore construction. Only the standard vendor
`AsyncScheduler` is currently admitted for Beam; other scheduler classes fail
explicitly, and non-Beam configurations pass through.

The temporary host probe in `/tmp/vllm-gr-mlu-scheduler-probe.py` verified the
vendor scheduling method chain, unchanged base/profiler methods, repeated
installation, all `throttle_prefills` call forms, one native scheduling pass,
stage/retirement annotation, prompt-cache limiting, restoration after native
exceptions, rejected alternate schedulers, non-MLU isolation, and child-entry
repair of an unwrapped selected scheduler. Its log is
`/tmp/vllm-gr-mlu-scheduler-probe.log`; it constructed no real scheduler resources,
model or server. The 10 existing nested-scheduler and patch-repair regression
cases passed; strict pytest configuration was disabled because the environment
lacks the timeout plugin. See `/tmp/vllm-gr-mlu-scheduler-regression.log`.
No repository test files were added.

### Prior scheduler admission queue failure

The real serving rerun used the recorded vendor vLLM 0.33 environment,
OneRec-1.7B, card 6, BF16, V2, eager execution and the 10-percent memory budget.
Startup, health and model-list checks passed. The same width-128, five-token
completion entered the shared `patched_schedule()` through the selected MLU
scheduler, confirming the previous override bypass is resolved. It then failed
inside `gr_admission()` before invoking the vendor scheduling body:

```text
AttributeError: 'AsyncScheduler' object has no attribute 'skipped_waiting'
```

The admission context expects the older `waiting` and `skipped_waiting` queue
layout. The installed vLLM 0.33 scheduler instead initializes `waiting`,
`kv_holding_waiting` and a `deferred_waiting` set. The vendor scheduling method
also uses a local `step_skipped_waiting` deque, which is not an instance queue
that GR can filter through the old field. This is a scheduler-state API
mismatch; adding a disconnected empty attribute would not establish correct
Beam ownership or admission over the native queues.

No queue-layout adaptation was present in that run. The request did not attach GR
stage metadata or reach the MLU worker stage guard; the host annotation probe
used a controlled admission context and does not cover native queue behavior.
The earlier missing Beam result is not an end-to-end resolved result path.
EngineCore died, HTTP returned 500, and session cleanup reported
`EngineDeadError`. The probe stopped at this failure, the server exited with
code 0 after SIGTERM, both API and EngineCore processes were gone, and port
8000 was free. Native shutdown again logged forced cleanup and a
leaked-semaphore warning. Evidence is in
`/tmp/vllm-gr-mlu-scheduler-startup.log` and
`/tmp/vllm-gr-mlu-scheduler-request.json`.

### MLU native admission queues

GR-PATCH-027 now installs and validates an MLU-specific
`_gr_admission_queue_context` on the selected vendor `AsyncScheduler`. Shared
`gr_admission()` still selects the owner, reserves the full logical KV extent,
rejects non-owner allocations and applies the outstanding-stage limit. Its
default CUDA/NPU queue context still uses `waiting` and `skipped_waiting`.

The MLU context filters the two actual selection queues, `waiting` and
`kv_holding_waiting`, through the shared `_AdmissionQueue`. The vendor loop
retains its KV-holder-first selection and native queue policy. Its per-step
skipped lists requeue through the same native collections. A view of
`deferred_waiting` limits membership, iteration and counts to the owner while
writing native additions/removals through to the original set. It is not a
replacement selection queue. The adapter restores all three collection
identities on success and exceptions, retaining mutations to the originals;
it never creates an unused `skipped_waiting` attribute.

The temporary `/tmp/vllm-gr-mlu-admission-probe.py` exercised real shared GR
admission with installed native FCFS and priority queue implementations and
controlled scheduler/session state. Default and MLU layouts passed owner
filtering, native requeue, deferred-state changes, full-extent reservation,
rejection of non-owner allocation and restoration after native errors. Separate
checks passed no-session passthrough, stage backpressure, gating exceptions,
lost-reservation errors, hook identity validation and missing-hook repair.
The allocator was controlled; this probe did not construct a full native
scheduler or execute a model. Evidence is
`/tmp/vllm-gr-mlu-admission-probe.log`.

The 10 existing nested-scheduler and patch-repair cases passed again, with the
same missing-timeout-plugin configuration warning; see
`/tmp/vllm-gr-mlu-admission-regression.log`. No repository tests were added.
Multi-session device execution, preemption, blocked/KV-transfer requests,
pause/resume, cancellation and retirement still require integration coverage.
Future native collection operations need review against the filtered views.

### Prior MLU Beam runner guard failure

The serving rerun used the same OneRec-1.7B checkpoint, MLU590 card 6, vendor
vLLM 0.33, BF16, V2, eager mode, TP=PP=DP=1, maximum length 8192 and 10-percent
memory budget. Health and model-list endpoints returned HTTP 200. The same
width-128 request with a five-token public budget and SID boundaries passed GR
admission and vendor scheduling. The dumped `SchedulerOutput` contains a
16-token Prefill, two KV block IDs and nonempty `gr_stage_metadata` for stage 0,
owner generation 1. The frontend normalized the SID-wrapped request to three
generated Beam tokens; the request body was not changed for this probe.

The executor delivered this output to the MLU runner through the installed
lifecycle adapter. Its existing guard raised:

```text
NotImplementedError: MLU Beam stage execution and retirement are not implemented
```

This resolves the observed `skipped_waiting` failure and establishes actual
Prefill dispatch to the runner. It does not execute Prefill attention, select
Beam candidates, decode, return a valid Beam result or retire a session. The
guard remained in place in that run; no worker execution had been added. EngineCore
died, the request returned HTTP 500, and session cleanup reported
`EngineDeadError`. The probe stopped at this failure and shut down the server;
exit code was 0, both API and EngineCore processes were gone, and port 8000 was
free. Native shutdown logged forced cleanup and a leaked-semaphore warning.
Evidence is `/tmp/vllm-gr-mlu-admission-startup.log` and
`/tmp/vllm-gr-mlu-admission-request.json`.

Ruff and the available text/YAML checks passed. Markdownlint could not run:
the existing nodeenv setup fails parsing Node's `0-alpha` version component.
See `/tmp/vllm-gr-mlu-admission-markdownlint.log`. CUDA/NPU device execution,
graphs, accuracy and performance were not tested.

### Eager MLU stage adapter

GR-PATCH-023 now routes stages to
`vllm_gr/v1/worker/mlu/beam_stage_runner.py`. It uses shared dispatch and batch
geometry validation, the existing `BeamAttentionPool`, and the ordinary Torch
unconstrained candidate selector from `ops/npu/unconstrained_topk.py`. Importing
that selector does not import NPU kernels. No attention kernel was added.

Prefill follows CUDA V2's native hidden-state handoff: execute the native
forward on an owned host copy of the dispatch, compute logits from its sampled
rows, update the native computed-token counter and clear the native pending
state. Sampling owns Beam scores/history and returns shared worker receipts.
Stable score ordering chooses survivors, followed by stable parent grouping
for the next physical batch. Decode retains native request/KV bookkeeping,
uses the existing eager MLU attention implementation and reorders each layer's
valid suffix with a separate gathered tensor before overwrite. Terminal output
uses the shared result schema. Current-stream synchronization precedes resource
release, including explicit retirement and reload/shutdown cleanup.

Scope is deliberately checked at stage entry: one prompt, fixed widths,
full-vocabulary unconstrained selection, ignored stop tokens, FP16/BF16
unquantized KV, one full-attention group, a plain decoder, async V2, eager
execution and TP/PP/DP/CP=1. Multiple prompts, dynamic/per-parent widths,
SID-stage vocabulary filtering, tables, speculative decoding, LoRA,
multimodal/encoder/hybrid models, connectors and graphs remain unsupported.
Plan admission is still provisional and broader than this initial runner scope.

The temporary `/tmp/vllm-gr-mlu-stage-probe.py` ran real MLU tensor operations
on card 6. Its three-stage scores, histories and parents matched the CPU Beam
reference for ordinary logits and exact ties. A controlled native Prefill
handoff and mocked Decode model path verified execute/sample routing, native
Prefill counter advancement, device parent-KV reordering, receipts, terminal
schema, release, retirement identity/idempotence, unsupported-stop rejection
and failure propagation. It did not execute the OneRec model or the adapter's
actual Decode forward. Initial probe setup errors (repository import path and
an omitted explicit unconstrained setting) were corrected before these checks.
Evidence is `/tmp/vllm-gr-mlu-stage-probe.log`.

The 18 existing CPU Beam reference/parity tests passed with the known missing
timeout-plugin warning; see `/tmp/vllm-gr-mlu-stage-regression.log`. No repository
test files were added. Native model integration remains a separate check below.

Deferred risks introduced by this adapter include full host dispatch copies,
full-vocabulary sorting, per-stage temporary allocations and synchronous final
copies. Suffix KV is allocated on first Decode after native memory profiling;
its capacity must eventually enter the serving memory budget. Native V2 state,
attention metadata/forward context, cancellation, repeated sessions, partial
failures and cleanup require real integration validation. Device arithmetic
parity with a small reference does not establish model accuracy or performance.

### Prior Decode common-metadata failure

The actual serving run kept the same OneRec-1.7B checkpoint, MLU590 card 6,
vendor vLLM 0.33, BF16, V2, eager mode, TP=PP=DP=1, maximum length 8192 and
10-percent memory budget. Health and model-list endpoints returned HTTP 200.
The unchanged width-128 request passed native Prefill, logits extraction and
the first GR candidate selection. The next dumped scheduler output is Decode,
dispatch/stage 1, decode step 0, owner generation 1, with a 16-token prefix.
This establishes real Prefill execution and progress into Decode, but not
model-output accuracy or complete inference.

The new adapter allocated its suffix pools and built per-layer Beam attention
metadata. While entering the vendor `set_sp_forward_context`,
`get_sp_metadata()` called `get_common_metadata_from_attn_metadata()`. That
helper requires a `common_metadata` entry in the dictionary, in addition to
layer metadata. The adapter then supplied only layer entries, producing:

```text
AssertionError: MLU-V1 only support type(attn_metadata)=dict, and common_metadata in attn_metadata.
```

The failure is before the Decode model forward and suffix attention/cache
writes. It occurs even in this single-device setup because the vendor helper
extracts common metadata before its later parallel-mode decisions. No
common-metadata adaptation or bypass was added in this step. The HTTP request
returned 500, EngineCore died, and cleanup reported `EngineDeadError`. The
probe stopped at the failure and terminated the server with exit code 0; both
API and EngineCore processes were gone and port 8000 was free. Native shutdown
again logged forced cleanup and a leaked-semaphore warning. Evidence is
`/tmp/vllm-gr-mlu-stage-startup.log` and `/tmp/vllm-gr-mlu-stage-request.json`.

Actual Decode, terminal-result delivery, retirement, repeated sessions, graphs,
accuracy and performance remain unverified. The controlled device probe above
does not close these integration gaps.

CUDA/NPU comparison for this failure:

- CUDA V2 builds the per-layer Beam metadata dictionary in
  `GPUBeamStageRunner._bind_attention()` and passes it to standard
  `set_forward_context()` in
  `vllm_gr/v1/worker/cuda/v2/gpu_beam_stage_runner_v2.py:forward_hidden`.
  It does not enter the vendor MLU helper or require its `common_metadata` key.
- NPU's `NPUEagerBeamStageRunner.execute()` prepares the Beam session and
  physical input plan, then calls the original native runner for Decode too.
  Native execution retains ownership of forward-context setup. The checked
  GR integration is `vllm_gr/v1/worker/npu/npu_beam_stage_runner.py`; no NPU
  device run was performed for this comparison.
- Native MLU already supplies the missing contract in
  `vllm_mlu/v1/worker/gpu/attn_utils.py`: it constructs
  `MLUCommonAttentionMetadata` and stores it under `COMMON_METADATA_STR`
  before the runner enters its forward contexts. This is the direct reference
  for the next adaptation. The GR Decode adapter must supply valid vendor
  common metadata for its physical Beam rows while retaining per-layer Beam
  attention metadata. Copying CUDA's layer-only dictionary is insufficient.

Ruff and the text/YAML checks passed for this adapter and its documentation.
Markdownlint remains blocked by the existing nodeenv `0-alpha` parsing error;
see `/tmp/vllm-gr-mlu-stage-markdownlint.log`. CUDA/NPU device execution was not
tested.

### Decode common-metadata integration

The eager MLU adapter now adds a real `MLUCommonAttentionMetadata` object under
the vendor's `COMMON_METADATA_STR`, following native MLU's attention builder.
The CUDA V2 execution structure and per-layer Beam metadata stay in place.
For the first Decode of the recorded request, the common object describes 128
physical rows with query offsets `[0, 1, ..., 128]`, 17-token sequences,
positions 16 and `DECODE_ONLY` mode. CPU and device sequence lengths agree;
KV sequence offsets are cumulative and prefill token counts are zero.

Each common block-table row repeats the shared native prefix mapping. PAD
slots prevent native writes; the generated suffix remains owned by the GR
attention pools. This object is for vendor forward-context consumers, not a
complete native paged-attention mapping of the split prefix/suffix cache.
Native attention fallback, other model consumers, parallel execution and
graphs require separate validation. Host tensor creation, device copies and
block-table repetition add eager per-step allocations.

The temporary `/tmp/vllm-gr-mlu-common-metadata-probe.py` checked widths 1, 4
and 128 on MLU590 card 6, including physical geometry, computed-token counts,
host/device sequence lengths, prefix rows, PAD slots and the real vendor
common-metadata extraction and single-device SP forward context. These passed.
The first probe fixture omitted `parallel_config`; that fixture was corrected
before the passing run. No repository tests were added. Evidence is
`/tmp/vllm-gr-mlu-common-metadata-probe.log`.

The first serving probe expired its 240-second readiness deadline before
sending a request. A read-only stack snapshot showed native sampler warmup in
`MLUBackend.make_optimize_mluir`, through `apply_top_k_top_p_triton` and
`MLUModelRunner._dummy_sampler_run`. There was no application exception or GR
Decode execution. The probe terminated its server (exit -9 after the graceful
deadline), verified both processes were gone, and retried with a 900-second
probe startup allowance. Serving arguments and request content stayed the
same. The initial artifacts are
`/tmp/vllm-gr-mlu-common-metadata-startup-timeout-startup.log` and
`/tmp/vllm-gr-mlu-common-metadata-startup-timeout-request.json`.

The longer card-6 retry completed kernel compilation but failed native memory
profiling: initial free memory was 8.71 GiB and final free memory was 74.57 GiB.
An unrelated worker had occupied roughly 71 GiB at startup and was later
defunct; it was not stopped by this work. The vendor
`MLUWorker.determine_available_memory()` rejected the changed measurement.
The server exited 1 before HTTP readiness or request submission; both owned
processes exited and port 8000 was free. Evidence is
`/tmp/vllm-gr-mlu-common-metadata-startup.log`,
`/tmp/vllm-gr-mlu-common-metadata-request.json` and the earlier compilation stack
at `/tmp/vllm-gr-mlu-common-metadata-warmup-stack.txt`.

The installed CUDA worker has the same free-memory consistency assertion in
`vllm/v1/worker/gpu_worker.py:determine_available_memory`; its normal profiling
path also requires stable device usage. GR's NPU startup workaround only
catches an OOM during penalties warmup, which is unrelated to another process
releasing memory. No workaround for this profiling condition was found in the
GR NPU integration; the native Ascend runtime is not installed here. No
profiling guard or sampler kernel was changed. After the user requested another
card, `cnmon` showed all eight cards idle with no running processes; the next
attempt selected card 2 while preserving the serving options and request.

### Prior Decode position-dtype failure

The card-2 retry used OneRec-1.7B, the same vendor vLLM 0.33 environment,
BF16, V2, eager mode, TP=PP=DP=1, maximum length 8192 and memory utilization
0.1. Startup and both health/model-list endpoints passed. The unchanged
width-128 request again completed Prefill and the first Beam selection, then
passed the vendor common-metadata and SP context setup on Decode stage 1.
The model forward entered Qwen3's first attention layer and failed in its
existing MLU rotary-embedding operation:

```text
MLUEagerBeamStageRunner._decode()
  -> worker.model(input_ids, positions)
    -> Qwen3 model / decoder layer / attention forward
      -> rotary_emb() -> torch.ops.vllm.rope_forward()
        -> mlu_ops.rotary_embedding() -> tmo.apply_rotary()
          RuntimeError: position_ids type need be int32
```

Our Decode positions were explicitly allocated as int64. Native MLU's
`MLUInputBuffers` in `vllm_mlu/v1/worker/gpu/input_batch.py` converts its positions
to int32. The rotary wrapper preserves the passed dtype when offsets are
absent, so our custom Decode path bypasses the native input-buffer contract.
The failure precedes Decode attention/cache writes; it does not establish a
completed Decode stage, terminal delivery, model accuracy or performance.

CUDA/NPU comparison:

- CUDA V2's `V2InputBuffers` explicitly allocates int64 positions for its CUDA
  execution path. `CudaStageHooksV2.initialize_inputs()` binds that storage and
  `bind_device_inputs()` copies the staged values into it. This CUDA dtype
  choice cannot be carried over unchanged to the MLU rotary operation.
- NPU's eager runner obtains its native positions through
  `_get_positions_buffers()`. `NPUEagerStagePlan.prepare_decode_inputs()` fills
  that buffer in place, or constructs values with `dtype=positions.dtype`.
- NPU's dtype-preserving approach is the useful reference for this mismatch;
  MLU should use its native int32 position contract. No rotary kernel change
  is needed for the reported mismatch, and no dtype fix was made in this step.
  The comparison is source-based; CUDA/NPU device execution was not run.

The request returned HTTP 500 in 0.179 seconds and EngineCore died. The probe
stopped at this new failure; the server exited 0 after termination, both owned
processes were gone, and port 8000 was free. Native shutdown again logged
forced cleanup and a leaked-semaphore warning. Evidence is
`/tmp/vllm-gr-mlu-common-metadata-card2-startup.log` and
`/tmp/vllm-gr-mlu-common-metadata-card2-request.json`.

Ruff and the applicable Python import/API checks passed. Text checks passed;
Markdownlint remains blocked by nodeenv's existing `0-alpha` parsing failure
(`/tmp/vllm-gr-mlu-common-metadata-markdownlint.log`). No new repository tests
or attention kernels were added.

### Native MLU Decode position dtype

The eager adapter now creates Decode positions with
`worker.input_buffers.positions.new_full(...)`. This inherits the native MLU
buffer's int32 dtype and device while retaining a separate stage allocation.
The same tensor is passed to the model, vendor SP context and common metadata.
Position values remain the prompt length plus the Decode step. No rotary
kernel or CUDA/NPU execution code changed. This follows NPU's practice of
preserving native buffer types instead of copying CUDA V2's explicit int64
position-storage choice.

The actual serving probe used MLU590 card 2, OneRec-1.7B, vendor vLLM
`0.33.0+mlu0.17.0.pt214`, BF16, V2, eager mode, TP=PP=DP=1, maximum length
8192 and memory utilization 0.1. All cards were occupied again; card 2 had
enough free memory for this existing budget, and its profiling completed.
Neither other processes nor the serving options/request were changed.

Health and model-list endpoints returned 200. The unchanged Beam request
returned HTTP 200 in 0.345 seconds with 128 choices, consecutive indices,
128 nonempty and distinct texts, and `finish_reason="length"` throughout.
All 128 texts matched the expected SID begin/a/b/c/end framing.
The first result was:

```text
<|sid_begin|><s_a_4016><s_b_1350><s_c_6436><|sid_end|>
```

This exercises native Prefill, the two Decode stages for the normalized
three-token Beam budget, candidate selection, suffix-pool use/reordering,
terminal result extraction and public SID framing. No new execution exception
was observed. The timings are one shared-device probe, not a performance
benchmark. Distinct SID-shaped strings do not establish catalog membership,
ranking correctness, model parity or recommendation accuracy.

The probe stopped the server after this one request. It exited 0, both API and
EngineCore processes were gone, and port 8000 was free. Native shutdown still
reported forced EngineCore cleanup and one leaked semaphore. Explicit
retirement/cancellation, graceful in-process cleanup, repeated sessions,
multi-prompt execution, graphs and unsupported configurations remain open.
Evidence is `/tmp/vllm-gr-mlu-position-dtype-card2-startup.log` and
`/tmp/vllm-gr-mlu-position-dtype-card2-request.json`.

Evidence below comes from startup logs, worker stacks and the installer source,
plus separate attention, selector and runner/planning probes. It is not a
complete runtime audit. In particular, reaching an installer does not mean the
function it installed has been called.

## Status meanings

| Status | Meaning |
| --- | --- |
| Reached | Startup executed this path and continued beyond it. |
| Installation check passed | The required patch plan accepted its installer/validator; functional behavior remains unverified. |
| Installer returned | The installer returned before the later failure; its runtime callbacks and any enclosing final validation remain unverified. |
| Known mismatch | Source or signature inspection identifies an incompatibility; the serving run may not have exercised it yet. |
| Not reached | Startup failed before this step. |

## Unchanged code reached during startup

The following code was not adapted for MLU in this work. The final column is
follow-up work, not a claim that every listed concern is a confirmed defect.

| ID | Code / responsibility | Observed evidence | What may fail later / what to verify |
| --- | --- | --- | --- |
| MLU-C01 | [Startup defaults](../../vllm_gr/engine/startup.py): `configure_model_runner_default()` | Reached with explicit `VLLM_USE_V2_MODEL_RUNNER=1`. | GR defaults to V1 without an override, while this native MLU worker requires V2. Resolve the MLU default before claiming the minimal command works without the environment setting. |
| MLU-C02 | [CLI banner](../../vllm_gr/entrypoints/cli/serve_banner.py): `patch_banner()` | Installer returned. | Known target mismatch: it assigns `api_utils.log_version_and_model`, while the installed native launcher imports the helper from `vllm.entrypoints.common.startup`. Verify the actual call-site binding and displayed banner. Assignment can succeed even when the launcher never uses that attribute. |
| MLU-C03 | [CLI config parser](../../vllm_gr/entrypoints/cli/serve_config.py) and [serve command](../../vllm_gr/entrypoints/cli/serve.py): parser construction | Parser construction, default argument normalization, and native API-server dispatch completed. Only the moved native imports were adapted. | Other YAML/CLI combinations and actual serving requests remain unverified against 0.33. |
| MLU-C04 | [Plugin](../../vllm_gr/plugin.py), [loader](../../vllm_gr/platforms/loader.py), and [registry](../../vllm_gr/platforms/registry.py): registration and activation | Reached; the MLU provider was discovered, selected, and passed to patch installation. Its environment check returned. | Recheck the same identity and provider in spawned processes and repeated initialization. Version declarations and platform identity checks do not validate the runner, device kernels, or upstream API behavior. |
| MLU-C05 | [Attention selector](../../vllm_gr/v1/attention/selector_patch.py): `apply_custom_attention_backend_patch()` | Previously failed on `_run_kv_cache_dtype`. The MLU-specific wrapper now accepts it; the serving run reaches CUSTOM backend import. | This row now records an adapted path. Native fallback dtype forwarding/cache behavior passed host probes, but native fallback attention execution remains unverified. CUDA/NPU retain the older signature. |
| MLU-C06 | [Flat log probabilities](../../vllm_gr/v1/engine/logprobs_patch.py): `patch_flat_logprobs()` | Installation check passed before the `batch_and_fork` failure. | Its validator checks the installed function binding. Verify 0.33 candidate field layouts, offsets, ranks, optional decoded tokens, and actual output assembly. |
| MLU-C07 | [OpenAI Beam integration](../../vllm_gr/entrypoints/openai/beam_search_patch.py): `patch_beam_search()` | Installation passed. The first completion failed on the native `priority` keyword; the MLU adapter now forwards it and the rerun reached frontend constraint-mode validation. | This path is now adapted. Verify nonzero priorities, headers, chat, grouped completions, response construction and cancellation with real requests; argument forwarding does not validate scheduling order. |
| MLU-C08 | [Chat sampling conversion](../../vllm_gr/entrypoints/openai/beam_search_patch.py): `patch_sampling()` | Installation passed; the real completion converted width 128 and the five-token budget with SID boundaries and reached frontend device admission. | Other 0.33 request fields, chat conversion, generation budgets and response assembly remain unverified. |
| MLU-C09 | [Engine arguments](../../vllm_gr/engine/arg_utils.py): `patch_create_engine_config()` | The wrapper created the native config, normalized Beam/CUSTOM settings, and attached the admitted MLU plan. Worker re-resolution matched its fingerprint, binding returned and engine startup completed. | Other configurations, constraints and execution remain unverified. |
| MLU-C10 | [Wire request types](../../vllm_gr/v1/engine/wire.py): `register_engine_core_request_types()` | Installer returned, including enum structure/value collision checks. | Verify `SUBMIT_GR_SESSION` serialization and decoding with 0.33 in every process. A collision-free enum extension does not establish socket protocol compatibility. |
| MLU-C11 | [Engine clients](../../vllm_gr/v1/engine/core_client_patch.py): `apply_engine_client_patches()` | Installer and enclosing `batch_and_fork` validation completed. Async client construction launched EngineCore and completed startup; HTTP health returned 200. | Verify async, sync, and in-process submission, output polling, abort, failure propagation, and shutdown against native client state and signatures. |
| MLU-C12 | [Frontend/output adapters](../../vllm_gr/v1/engine/core_client_patch.py): `apply_async_llm_patches()` | Installer returned. | Verify `LLMEngine.add_request`, `AsyncLLM._add_request`, `OutputProcessor.process_outputs`, and session submit/wait/abort/release with native 0.33 request/output schemas. |
| MLU-C13 | [EngineCore integration](../../vllm_gr/v1/engine/engine_core_patch.py): `apply_engine_core_request_patches()` | The recorded request completes Prefill, Decode and terminal output processing, returning HTTP 200 with 128 sequences. | Validate additional IPC/session paths, cancellation, explicit retirement, repeated requests and startup/shutdown. One successful completion does not establish lifecycle correctness. |
| MLU-C14 | [Prefix-cache hashing](../../vllm_gr/v1/core/kv_cache_utils.py): `apply_kv_hash_memo_patch()` | The MLU factory accepts optional lookahead; prior startup and temporary hashing probes passed. GR-PATCH-025 imports the vendor EngineCore module before GR hashing patches, leaving its module-local factory alias outside the existing rebinding. | This is an adapted path with an uncovered alias. Native request hashing still uses GR's patched per-block encoding, but the vendor alias can bypass the whole-chain memo even at zero lookahead. Verify the real binding, cache reuse and external-cache compatibility; see hashing risks below. |
| MLU-C15 | [Shared scheduler hooks](../../vllm_gr/v1/core/sched/scheduler_patch.py) and [MLU target selection](../../vllm_gr/platforms/mlu/scheduler_patch.py) | GR-PATCH-027 wraps the vendor `AsyncScheduler` and supplies its native queue context. The recorded request completes Prefill/Decode scheduling and terminal delivery. | Override bypass and the observed queue-field failure are resolved. Validate multi-session scheduling, blocked/preempted requests, KV reservations/release, logical versus physical rows, abort and retirement. |
| MLU-C16 | [Async output guard](../../vllm_gr/v1/worker/gr_async_output.py): `install_async_output_guard()` | Installer returned; runner-hook installation subsequently completed. | Verify that the MLU executor uses the patched `WorkerProc.enqueue_output` and compatible `AsyncModelRunnerOutput`; check D2H completion and exception propagation with real worker outputs. |
| MLU-C17 | Sliced-logits installation in [patch orchestration](../../vllm_gr/patch.py) | Installer completed after the required Beam patch plan. | Validate model projection, gathering, vocabulary slicing, and weight-update invalidation on MLU. Installation does not establish kernel compatibility. |
| MLU-C18 | Native `MLUWorker.__init__` and `MluConfig.from_vllm_config` | Previously failed on GR transport keys. GR-PATCH-024 now filters them during vendor initialization; the latest serving run passed this boundary. | MLU adaptation is now required here, so this row records a resolved concern rather than unchanged-code evidence. Verify multiple engine configurations and any new vendor call sites; preserve GR data and normal vendor error handling. |
| MLU-C19 | Native worker device setup and Qwen3 model construction | Startup, native KV allocation and warmup passed. One actual OneRec request now completes Prefill, Decode and HTTP result delivery. | Validate model accuracy, additional shapes/configurations, capture and resource cleanup. |
| MLU-C20 | [Beam attention facade](../../vllm_gr/v1/attention/backends/beam_attn.py): CUSTOM backend import | Previously failed importing `is_quantized_kv_cache`. The MLU import now uses `vllm.v1.kv_cache_interface`; the actual serving rerun passed backend import. | CUDA/NPU retain `vllm.utils.torch_utils`. The two helper call sites are unchanged; quantized KV-cache support remains unverified. This row records an adapted path. |
| MLU-C21 | [Beam attention construction](../../vllm_gr/v1/attention/backends/beam_attn.py): `BeamAttentionImpl.execution` | Eager MLU attention construction and synthetic device probes passed. One actual OneRec request completes Prefill and Decode after the position-dtype fix. | Validate prefix/suffix cache correctness against a model baseline, more shapes and repeated use; successful execution alone does not establish numerical parity. |
| MLU-C22 | [Post-load binding](../../vllm_gr/v1/worker/runner_binding.py), [worker validation](../../vllm_gr/platforms/services.py), [device admission](../../vllm_gr/platforms/beam_requests.py) | Worker admission, service binding and startup passed. The eager stage adapter accepts the observed single-prompt request and reaches Decode. | Plan/frontend admission remains broader than the initial runner scope. Dynamic widths, tables, multiple prompts and other unsupported features still require explicit implementation and integration coverage. |
| MLU-C23 | Native KV-cache specification setup and [Beam backend](../../vllm_gr/v1/attention/backends/beam_attn.py): `get_supported_kernel_block_sizes()` | Previously failed on the new `spec` argument. GR now accepts an optional specification and forwards it to native TMO on MLU. Startup passed specification collection and selected LBHNC. | This is an adapted path. CUDA/NPU retain `MultipleOf(16)` and no-argument calls. The MLU declaration follows vendor block-size restrictions; kernel execution across all advertised sizes remains unverified. |
| MLU-C24 | Native `MLUWorker.determine_available_memory()`, V2 `profile_run()` and `MLUModelRunner._dummy_sampler_run()` | The 180-second probe timed out during MLU Triton compilation of native top-k/top-p for OneRec's 176,384-token vocabulary. With a longer limit, compilation and profiling completed without code changes. | Dummy profiling uses `skip_attn=True`; this is not an attention correctness check. The native sampler path remains unchanged and is separate from GR Beam candidate selection. First-use compilation can exceed the earlier probe budget. |
| MLU-C25 | Native KV allocation and warmup | After profiling, startup sized an LBHNC cache for 22,448 tokens, allocated it and returned from native warmup. The combined profile/cache/warmup phase took 263.46 seconds on the shared MLU590. | Validate actual attention metadata, cache writes/reads, Beam suffix allocation/reordering and real requests. Native warmup completion is not Beam execution or accuracy evidence; the reduced memory budget is not a serving capacity recommendation. |
| MLU-C26 | [MLU child process entry](../../vllm_gr/platforms/mlu/engine_core_patch.py) and vendor EngineCore initialization | GR-PATCH-025 loads vendor replacements before GR captures initialization and revalidates the child plan. A host probe confirmed the initializer chain; the real request passed vendor stepping and reached runner token staging. | This is now adapted. Verify other launch paths, process backends, parallelism, vendor upgrades and repeated construction. Successful output processing and retirement remain unverified. |
| MLU-C27 | Native V2 `GPUModelRunner.add_requests`, `RequestState.add_request` and [MLU NumPy staging](../../vllm_gr/v1/worker/mlu/numpy_staging.py) | Resolved for the observed request: GR-PATCH-026 reuses CUDA bulk staging with v0.33 write-length bookkeeping. Production-installer device probes passed, including long writes; the real request progressed to terminal-output handling. | Validate model-level token ownership, concurrent streams and graphs. Vendor device-write methods remain active. Correct buffer writes do not establish Beam inference. |
| MLU-C28 | [Beam terminal-result extraction](../../vllm_gr/v1/engine/beam_session_client.py): `_terminal_result()` | Resolved for the recorded request: terminal Beam extraction and frontend SID framing return HTTP 200 with 128 nonempty, distinct choices and consecutive indices. | Validate additional output options, scores/parent histories, failure responses, cancellation and repeated requests. Output shape is not an accuracy check. |
| MLU-C29 | [GR scheduler admission](../../vllm_gr/v1/core/sched/beam_async_scheduler.py) and [MLU queue context](../../vllm_gr/platforms/mlu/scheduler_patch.py) | GR-PATCH-027 filters native `waiting` and `kv_holding_waiting` queues plus deferred bookkeeping. The actual request passes admission and dispatches Prefill. Native-queue host probes cover ownership, ordering, reservation and exception restoration. | Resolved for the observed request. Validate multi-session execution, preemption, blocked/KV-transfer requests, pause/resume and cancellation. Future collection APIs must preserve the context's assumptions; host probes do not establish those integration paths. |
| MLU-C30 | [MLU runner dispatch](../../vllm_gr/v1/worker/mlu/model_runner_patch.py) and [eager stage adapter](../../vllm_gr/v1/worker/mlu/beam_stage_runner.py) | Scoped eager execution now completes the recorded OneRec request, including Decode and terminal delivery. Earlier device selection/state probes and controlled receipts/retirement passed. | Validate model parity, repeated sessions, suffix KV ownership/reorder correctness, cancellation and explicit retirement. On-demand pools, host copies, full sorting and synchronization require budgeting/performance review. Multi-prompt, graph and constraint paths remain unsupported. |
| MLU-C31 | Vendor `set_sp_forward_context()` -> `get_sp_metadata()` -> `get_common_metadata_from_attn_metadata()` | Resolved for the recorded request: vendor common metadata describes physical Decode rows alongside per-layer Beam metadata. The device probe and complete request pass. | Validate other model consumers. Common block tables describe the shared prefix, not the GR suffix pools; native attention fallback, parallelism and graphs need separate adaptation. |
| MLU-C32 | Vendor `MLUWorker.determine_available_memory()` | A shared-card run failed when free memory rose from 8.71 to 74.57 GiB during profiling. Retrying on idle card 2 passed startup with unchanged serving options. | Maintain stable device usage during profiling. The installed CUDA worker has the same consistency guard; the GR NPU penalties-OOM workaround does not address this condition. No profiling bypass was added. |
| MLU-C33 | [MLU Decode positions](../../vllm_gr/v1/worker/mlu/beam_stage_runner.py) -> native rotary embedding | Resolved for the recorded request: positions inherit the native buffer's int32 dtype/device through `new_full`. Rotary embedding and the complete eager Beam request pass. | Validate other position schemes and model configurations before extending scope. CUDA V2's int64 choice is not portable to MLU; NPU's preservation of native buffer types is the reference pattern. |

### Resolved selector signature and helper import failures

The installed vLLM 0.33 selector calls its cached selector with:

```python
_cached_get_attn_backend(
    backend=backend,
    attn_selector_config=attn_selector_config,
    num_heads=num_heads,
    _run_kv_cache_dtype=cache_dtype,
)
```

The previous GR replacement accepted only:

```python
def _patched_cached_get_attn_backend(backend, attn_selector_config, num_heads=None):
    ...
```

That signature caused the previous `TypeError`. Installation now chooses an
MLU wrapper accepting `_run_kv_cache_dtype=None` as a keyword-only argument and
forwarding it to the original native selector on the fallback path. CUDA/NPU
retain the old signature and argument set. Both wrappers preserve explicit and
configuration-derived CUSTOM intent and propagate GR backend errors. Original
callable tracking, cache clearing, and repeated installation remain intact.

Temporary host probes passed dtype-sensitive native caching, argument forwarding,
CUSTOM routing/error propagation, repeated installation, cache clearing, and
replacement-target handling for simulated MLU/CUDA/NPU callables. The actual
OneRec serving rerun passed this signature boundary and reached backend import.
No test files were added; the fallback device execution path remains unverified.

Returning the CUSTOM `BeamAttentionBackend` class would still not establish
MLU attention support. The previous run failed because its module imported
`is_quantized_kv_cache` from `vllm.utils.torch_utils`, unavailable in 0.33.
The facade now selects `vllm.v1.kv_cache_interface` on MLU and retains the old
import for CUDA/NPU. The installed helper still accepts a dtype string and
returns a boolean, so its two GR call sites require no changes. This does not
establish quantized KV-cache support on MLU.
The provider now returns the eager MLU attention module described below.
The shared metadata builder and runner still contain CUDA/NPU-specific
assumptions; implementing attention primitives does not establish Beam serving.

### Eager attention adapter evidence and limits

The [MLU adapter](../../vllm_gr/v1/attention/backends/beam_attn_mlu.py) reuses
`vllm_mlu._mlu_ops.reshape_paged_cache`, `flash_attention(return_lse=True)`,
and `merge_attn_states`, together with GR's `write_eager_step`. No attention
kernels were added. The provider exposes the same module for layer construction
and the planned attention service. MLU uses native TMO per-head K/V packing in
contiguous LBHNC storage, with cache views preserving the layer storage offset.
CUDA/NPU views and execution remain unchanged.

Temporary device probes passed on MLU590-M9DK with the runtime versions above
and OneRec's attention geometry (16 query heads, 8 KV heads, head dimension 128).
The probes used synthetic inputs rather than model inference:

- BF16 and FP16 paged prefill, including noncontiguous block IDs, multiple
  requests, a nonzero layer storage offset, and cached prefill.
- One eager Beam session with four rows over three decode steps, mixed with
  an ordinary decode row; prefix/suffix LSE merge matched a CPU FP32 reference.
- Empty shared prefixes, untouched padded rows and other pool slots, and
  suffix histories after manually simulated repeated-parent selection.
- BF16 maximum absolute error was below 0.0094; FP16 below 0.0012, with
  tolerances of 0.025 and 0.004 respectively (relative tolerance 0.02).

Primitive and adapter results are in `/tmp/vllm-gr-mlu-tmo-probe.log` and
`/tmp/vllm-gr-mlu-attention-probe.log`. No repository test files were added.
These checks validate synthetic attention arithmetic and cache routing; the
probe's manual parent reorder does not implement runner KV reordering.

The adapter is limited to eager, unquantized decoder attention. It rejects
graph capture, sliding windows, ALiBi, sinks and softcap, and permits only one
Beam session per dispatch. It reads sequence metadata on the host and allocates
temporary packed suffix and attention buffers. Graph support, batched Beam
sessions, active/padded Beam masks, all advertised head/block sizes, metadata
builder integration, model accuracy and performance remain unverified.
Native `--enforce-eager` does not itself implement or validate GR graph policy.

### Hasher adaptation and remaining risks

MLU installs a factory accepting `num_hash_lookahead_tokens=0` and forwards the
value to native vLLM. For nonzero lookahead it returns the native request hasher
before constructing or reading the whole-chain memo. That preserves delayed
hash emission and the extra keys containing following tokens. Zero lookahead
retains the chain memo. CUDA/NPU install the original two-argument signature and
make two-argument native factory calls. Marker attributes, memo diagnostics,
idempotence and the EngineCore factory binding are retained.

Temporary host probes passed lookahead 0/1/2, complete-block boundaries,
incremental hashing, repeated prompts, returned-list isolation, cache-salt
fallbacks, changed lookahead tokens, and absence of chain-memo entries for
nonzero lookahead. Results matched native request traversal with GR per-block
hashing installed. Simulated CUDA/NPU platform checks preserved the legacy
factory signature; these were not CUDA/NPU hardware runs. Evidence is in
`/tmp/vllm-gr-mlu-lookahead-probe.log`; no repository test files were added.

Risks still requiring validation:

- With GR-PATCH-025, the vendor EngineCore module is imported before GR hashing
  installation and captures its own `get_request_block_hasher` alias. The GR
  installer rebinds the native vLLM EngineCore alias, not the vendor alias.
  The vendor initializer can therefore bypass whole-chain memoization even
  at zero lookahead. This does not bypass the patched per-block hash function.
  The earlier factory probes do not validate this newly selected call site.
  A fresh-process probe of the new entry confirmed both bindings without
  constructing an engine; see `/tmp/vllm-gr-mlu-engine-entry-hash-alias.log`.
- The native request hasher calls the patched per-block hash function. GR's
  SHA-256 encoding differs from stock vLLM serialization; these probes do not
  establish stock hash-byte parity or external KV-cache interoperability.
- Nonzero lookahead bypasses only whole-chain memoization. Per-block memo keys
  and digests still include extra keys, and new native extra-key formats need
  review. Multimodal, LoRA and prompt-embedding combinations were not tested.
- The serving probe used zero lookahead with speculative decoding disabled.
  Nonzero host probes do not establish speculative execution support; existing
  GR admission restrictions still apply.
- Actual requests must validate prefix-cache hits/misses and isolation across
  differing lookahead contexts. Startup merely constructs the hasher.

## Current startup and request boundaries

The latest observed startup path is:

```text
EngineArgs.create_engine_config()
  -> MLUConfigService.finalize_native_config()
    -> install_mlu_engine_core_entry()  [reassert process target]
    -> install_mlu_scheduler_patch(config)  [wrap/validate configured class]
  -> admit_backend_plan()
    -> MLUBackendProvider.resolve()  [returned, Beam + CUSTOM preserved]
AsyncLLM -> EngineCoreClient -> launch_core_engines()
  -> run_mlu_engine_core()  [spawned child]
    -> config filter / vendor mlu_hijack / load_general_plugins()
    -> initialize_engine_core_beam_patches()
    -> install_mlu_scheduler_patch(config) / verify vendor initializer
    -> vendor run_engine_core() -> native process entry
  -> EngineCoreProc -> GR init wrapper -> vendor EngineCore init -> UniProcExecutor
    -> init_worker() -> MLUWorker.__init__()  [config filter returned]
    -> init_device() -> deferred MLU lifecycle and NumPy staging installation
      -> MLUModelRunner construction  [completed]
    -> load_model() -> GR load wrapper
      -> native model loader
        -> Qwen3 / BeamAttentionImpl construction  [eager MLU module returned]
        -> weights loaded / native load_model returned
      -> bind_backend_services()
        -> worker plan re-resolution / fingerprint agreement  [passed]
        -> BuiltinPlanSupport.validate_worker()  [MLU accepted]
        -> create_model_runner_adapter() / create_attention_executor()
        -> validate attention layers / publish binding  [returned]
  -> EngineCore._initialize_kv_caches()
    -> executor.get_kv_cache_specs() -> collective_rpc("get_kv_cache_spec")
      -> native Worker.get_kv_cache_spec() -> MLUModelRunner.get_kv_cache_spec()
        -> attn_utils.get_kv_cache_spec() -> customize_attention_spec()
          -> BeamAttentionBackend.get_supported_kernel_block_sizes(spec)
            -> TMOFlashAttentionBackend.get_supported_kernel_block_sizes(spec)
              [returned; LBHNC layout selected afterward]
    -> executor.determine_available_memory()
      -> MLUWorker.determine_available_memory() -> GPUModelRunner.profile_run()
        -> _dummy_run(skip_attn=True)  [returned]
        -> MLUModelRunner._dummy_sampler_run()
          -> Sampler -> apply_top_k_top_p() -> apply_top_k_top_p_triton()
            -> Triton compile -> MLU compiler  [completed with longer timeout]
    -> allocate KV cache / native warmup  [returned]
  -> construct vendor AsyncScheduler  [returned]
  -> get_request_block_hasher(hash_block_size, caching_hash_fn,
                              num_hash_lookahead_tokens)
    -> vendor module-local factory alias  [returned; see MLU-C14]
  -> vendor latency state and is_waiting_req initialized
  -> GR post-init readiness validation  [returned]
  -> EngineCore startup / API-server initialization  [completed]
    -> GET /health and GET /v1/models  [HTTP 200]
    -> probe shutdown  [exit 0; no remaining listener]
```

After installing MLU's native admission queue context, Decode common metadata
and native position dtype, the latest card-2 request completed:

```text
POST /v1/completions -> native create_completion()
  -> beam_search_dispatch(priority=0) -> GR beam_search()
    -> OnlineBeamSearchRequest(priority=0) -> run_beam_search_async()
      -> AsyncBeamSearchClient.prepare_request()
        -> BeamSearchInputProcessor.process_online_inputs()
          -> native process_inputs(priority=0) / assign_request_id()
          -> _create_batch_request() -> _validate_request()
            -> validate_frontend_constraint_mode()  [MLU accepted]
      -> submit_once() -> EngineCore session processing
        -> vendor run_busy_loop() -> GR patched_process_engine_step()
          -> vendor step_with_batch_queue()
            -> AsyncScheduler.schedule(throttle_prefills)
              -> shared GR patched_schedule() -> gr_admission()
                -> mlu_admission_queues() / shared KV reservation
                -> vendor schedule()  [16-token Prefill; returned]
              -> attach_gr_stage_metadata()  [Prefill, stage 0]
            -> model_executor.execute_model() -> Worker.execute_model()
              -> MLUModelRunner.execute_model() -> runner_lifecycle.routed()
                -> bound runner adapter -> MLUEagerBeamStageRunner.execute()
                  -> native Prefill -> hidden states -> logits  [returned]
            -> sample_tokens() -> Beam candidate selection / worker receipt
          -> next vendor step / Decode dispatch 1
            -> MLUEagerBeamStageRunner.execute() -> _decode()
              -> suffix pool allocation / per-layer and common metadata
              -> set_sp_forward_context() -> get_sp_metadata()
                -> get_common_metadata_from_attn_metadata()  [returned]
              -> worker.model() -> Qwen3 decoder / attention forward
                -> rotary_emb() -> rope_forward() -> tmo.apply_rotary()
                  [returned with native int32 positions]
                -> Beam attention / suffix cache -> hidden states -> logits
            -> sample_tokens() -> Beam selection / suffix KV reorder
          -> final Decode stage / selection -> terminal BeamBatchResult
      -> client terminal extraction -> public SID framing
        -> HTTP 200 with 128 sequences
```

Current boundaries:

| Boundary | Status / required follow-up |
| --- | --- |
| MLU runner discovery, installation, and verification | Lifecycle installation, model loading and service binding passed. The eager adapter completed one request through native Prefill, Decode and terminal delivery. |
| `batch_and_fork` as a complete patch group | Base and selected MLU scheduler validation passed. One actual request completes scheduling and terminal delivery; cancellation and explicit retirement remain unverified. |
| EngineCore process entry | MLU configuration selects its dedicated process target. Vendor initialization, GR readiness and selected-scheduler installation passed. The current single-request probe has no execution failure. |
| Sliced-logits patch | Installer completed; actual model operations remain unverified. |
| Engine configuration and backend planning | Beam/CUSTOM admission, native worker config parsing, post-load plan agreement, device validation, local service binding and engine startup passed. |
| Attention, constraints, and Beam stage execution | CUSTOM attention construction, synthetic eager attention and one complete OneRec Beam request pass. Accuracy and cache/model parity are unverified; constraints and graphs remain unsupported. |
| Model loading, requests, graphs, and lifecycle | Startup and one request pass, including terminal HTTP delivery. Explicit retirement, graphs, repeated lifecycle correctness and graceful cleanup remain unverified; shutdown still forces EngineCore cleanup. |

## Provisional Beam plan admission and deferred risks

`MLUBackendProvider.resolve()` now accepts Beam configurations and their CUSTOM
attention requirement so startup can proceed during the MLU port. The normal
CLI defaults configure three Beam widths of 128; engine normalization selects
CUSTOM attention for Beam. Neither setting is disabled to bypass startup.
Admission is a development step, not evidence that Beam inference works.

| Gap | Current behavior and risk | Required follow-up |
| --- | --- | --- |
| Plan admission | The full Beam request remains in the serialized plan and fingerprint. Choices select the MLU V2 module and preserve `requires_attention`; its eager stage adapter checks a narrower runtime scope. | Align supported planning/frontend scopes with verified execution before claiming MLU support. Graphs remain unsupported; a runner identity alone cannot attest Beam capability. |
| CUSTOM attention | Both provider methods return the eager MLU execution module. Constructor use, service creation, layer validation, worker binding, KV specifications, LBHNC selection and native allocation passed. Only unquantized eager single-session attention is implemented. | Validate metadata and cache behavior with real requests and Beam stages. Extend configurations explicitly; retain the documented eager adapter limits. |
| Worker validation | Inherited `BuiltinPlanSupport.validate_worker()` runs shared execution and device admission after native model load. Worker and unconstrained frontend allowlists now provisionally accept MLU. Existing async, dtype, parallelism and other runtime checks remain. | Complete MLU execution coverage before treating admission as support. Dynamic-width admission still excludes MLU; table constraints remain unsupported. |
| Runner execution | The eager adapter supplies stage validation, native Prefill, Torch Beam state/selection, suffix pools, vendor common metadata, receipts and retirement. Controlled probes and one real complete request pass. | Validate model/output parity, repeated requests, KV ownership, cancellation, resource budgeting and explicit retirement before accepting inference traffic. |
| Graphs | Beam graph settings remain in the request, but no MLU GR capture/replay implementation is supplied. Native graph configuration is separate. | Establish eager Beam correctness before enabling and validating GR graph execution. |
| Constraints and other configurations | Table constraints, native runner V1, unknown backend options, and NPU xllm selection still fail admission. Encoder-only and overlap runners remain rejected by the deferred worker hook. | Implement each path explicitly; this provisional admission covers the standard MLU V2 runner only. |
| Resource lifecycle | The complete-request probe exited 0 after SIGTERM, with no remaining API/engine process or port listener. Native shutdown still force-killed EngineCore and reported one leaked semaphore. | Verify partial-load failures, repeated loads, shutdown with live Beam work and per-service resource release. Process exit alone does not validate in-process cleanup. |

Direct host probes verify deterministic Beam plan serialization, preservation of
the request, and import isolation. They do not exercise device kernels or prove
that any admitted Beam stage can execute. No new test files were added.

For the resolved vendor-config conflict, the existing NPU integration provides
a reference: [Ascend config filtering](../../vllm_gr/platforms/ascend/config_patch.py)
temporarily excludes `GR_CONFIG_KEY` and `BACKEND_TRANSPORT_KEYS` during vendor
initialization, then restores the original mapping in `finally`. CUDA's native
worker does not construct an equivalent vendor config from the entire mapping.
The [MLU config filter](../../vllm_gr/platforms/mlu/config_patch.py) now applies
the same scoped filtering to `init_mlu_config`, installed by the provider's
`apply_runtime_patches()` before worker construction. Fresh workers load general
plugins before constructing the native worker. The existing `init_device` hook
would be too late for this fix.

The filter removes only known GR keys, preserves vendor settings and config
identity, restores the original mapping on exceptions, and is idempotent.
Direct host probes passed these behaviors, including propagation of an unknown
vendor-setting error and passthrough when no GR keys are present. These probes
do not validate multi-engine concurrency: as with the Ascend filter, the mapping
is temporarily replaced during a synchronous native call. Concurrent access to
the same config object, direct `MluConfig.from_vllm_config` calls that bypass
`init_mlu_config`, and future callers holding an early function alias require
reassessment. Current inspected native call sites import the function locally.

The existing patch-ownership test permits only `cuda` and `ascend` backend IDs.
It needs MLU coverage when test changes are enabled; adding the MLU inventory
entries does not make that existing restriction pass.

## Deliberate local changes, not compatibility evidence

- CLI imports in `entrypoints/cli/main.py` and `entrypoints/cli/serve.py` select
  the new native CLI modules for MLU. The serving run progressed beyond the
  previous import failures; this is not complete CLI or serving validation.
- The MLU provider's `check_environment()` checks native MLU identity and the
  public vLLM version `0.33.0`; the internal vendor version is only reported.
- `MLUBackendProvider.apply_runtime_patches()` installs the scoped vendor-config
  filter (GR-PATCH-024), returning `1` when newly installed and `0` when already
  present. Its config service reasserts the dedicated MLU EngineCore process
  entry during finalization (GR-PATCH-025). Other MLU multiprocessing paths
  remain unimplemented or unverified.
- Deferred MLU runner installation now adds the NumPy staging adapter
  (GR-PATCH-026), preserving vendor write kernels and maintaining v0.33 chunk
  lengths. The earlier request passed staging but lacked a terminal Beam result.
- The MLU config service and child entry install GR-PATCH-027 on the configured
  vendor scheduler using the shared wrapper and MLU admission queue context.
  The real request passes admission and dispatches Prefill and Decode metadata.
- GR-PATCH-023 now routes to the eager MLU stage adapter. Native Prefill and
  Decode, vendor common metadata, int32 positions and terminal delivery passed
  one recorded request. Additional requests, model accuracy and broader MLU
  support remain unverified.
- Packaging accepts the MLU target and records the MLU version suffix. Package
  installation and provider discovery do not demonstrate execution support.

## Eager prefix-cache repeatability

A prefix cache hit changes the number of prompt rows in the native forward.
The MLU eager Beam runner enables TMO's batch-invariant matrix multiplication
during prefill. Causal paged attention processes query pages with their visible
KV lengths so a cached suffix uses the same launch geometry as the
corresponding pages of a full prompt. The latter change adds attention
launches for longer uncached prompts.

On OneRec-1.7B BF16, MLU590-M9DK, runner V2, width 128 and eager execution,
the three comparison prompts returned identical ordered token sequences across
three requests each. Six additional histories of 15 through 279 prompt tokens
also repeated exactly. Two prompts matched when a different request populated
their shared prefix. Run the regression against a server with prefix caching enabled:

```bash
python tests/system/check_mlu_beam_repeatability.py --base-url "$BASE_URL" --model "$MODEL"
```

Exact CUDA ranking is still open. On the original three prompts the H20-3e
reference and MLU agreed on 8/128, 24/128 and 13/128 ranked positions,
respectively. All three top recommendations matched. Replaying the H20 logits
through the MLU candidate selector and Beam state reproduced all 384 ordered
recommendations. The native model forward already differs in the first QKV
projection on identical BF16 operands. These measurements use vLLM 0.30.0
on CUDA and 0.33.0+mlu0.17.0.pt214 on MLU.

## Updating this record

After each startup fix, record the new failure boundary and update only the
rows whose evidence changed. Keep installer completion separate from functional
validation. For a resolved concern, record the implementation change, exact
runtime and model configuration, verification performed, and remaining limits.
Add new untouched callbacks as startup reaches them; do not assume this snapshot
covers code that has not yet been imported or executed.

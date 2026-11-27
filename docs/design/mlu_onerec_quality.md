# MLU OneRec quality validation

This records the 2026-11-27 validation of runtime revision
`b7012b4f340e98afef2e36bf2356a8216a45056b`. Exact CUDA/MLU recommendation
identity is a diagnostic, not an acceptance requirement. Finite tests support
only the recorded configuration; they do not prove arbitrary program behavior.

The full video task passes the proposed 99% quality-retention policy. Full ad
meets all six observed-score targets; five confidence assessments pass and
mapped-item Pass@32 is inconclusive. Interactive top-1 remains inconclusive
and misses its observed-score target. A fixed ad sample shows degradation.
The strict all-metric confidence policy is therefore not fully satisfied.
There is no blanket accuracy certification for the MLU backend.

The accompanying [aggregate results](mlu_onerec_quality_results.json) retain
per-metric confidence bounds and the unsuccessful sample assessments.

## Protocol and evidence identity

- Model: OpenOneRec/OneRec-1.7B, BF16, TP1, ModelRunner V2, eager execution.
- Requests: one at a time, beam/candidate width 32, three generated SID tokens,
  temperature 0, prefix caching enabled, `ignore_eos=true`, thinking disabled.
- CUDA: H20-3e, vLLM 0.30.0, Torch 2.13.0+cu130; GR revision
  `292362f4693a37a6ce05c88a0e81f54dd53bb2e6`.
- MLU: MLU590-M9DK, vLLM 0.33.0+mlu0.17.0.pt214,
  Torch 2.14.0+cpu, torch-mlu 1.35.1+torch2.14.0. Vendor source revision:
  `263b59048ae0c59922b119a689fff876ba0555f7`.
- Identical checkpoint SHA256 on both backends:
  `cabc217fad6129ab15643753b7f7a2d8791f245f8c25a51c87f92499c574628d`.
  Tokenizer/config hashes and every prepared prompt also match.
- Official templates and scoring helpers: OpenOneRec revision
  `a969edcadd579a06c1966ae1db5984e02f48beff`.
- Data: RecIF snapshot `8f7cf2ee0b949e955a87a708d02024687be232c8`, retrieved
  from a public mirror and checked against original file hashes.

These are controlled comparisons, not an exact reproduction of the upstream
generator: upstream EOS defaults differ. SID-to-item mapping chooses the most
popular mapped item; tied choices are deterministic and shared across backends.
Product uses IID; other recommendation tasks use PID. Invalid or unmapped
outputs count as misses at their original ranks. No rows are filtered or truncated.

## Metrics and acceptance policy

The [OpenOneRec report](https://arxiv.org/html/2512.24762v1#S3.SS3) evaluates
recommendation using Pass@1, Pass@32 and Recall@32. Pass@K asks whether any
relevant item is retrieved; Recall@K measures the fraction of relevant items
retrieved. We report both SID and mapped item-ID metrics. NDCG@32 is an
additional ranking guardrail, extending the Recall/NDCG evaluation used by
[TIGER](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf).

The proposed engineering target is at least 99% of CUDA's score on every
primary metric. The [MLPerf accuracy policy](https://docs.mlcommons.org/inference/#high-accuracy-variants)
provides precedent for 99% and 99.9% reference-relative targets; it does not
define an official OneRec acceptance threshold. The 99% policy was selected
after initial interactive and partial video results had been seen, and is
exploratory. A 99.9% sensitivity analysis is reported separately.

We use 5,000 paired bootstrap replicates, resampling whole users, seed 20261127.
The statistic is `MLU - 0.99 * CUDA`. A nonnegative one-sided 95% lower bound
passes; a negative one-sided 95% upper bound is evidence of degradation;
otherwise the result is inconclusive. All primary metrics must pass. Intervals
are per metric. We also report observed-score target misses separately.
Non-significance does not establish equivalence.

## Completed video and interactive results

Values are percentages. Video has 38,781 rows and distinct users. Interactive
has 1,000 rows but 912 distinct users.

| Task | Metric | CUDA | MLU | 99% assessment |
| --- | --- | ---: | ---: | --- |
| Video | SID Pass@1 | 5.023078 | 5.015343 | Pass |
| Video | SID Pass@32 | 17.389959 | 17.410588 | Pass |
| Video | SID Recall@32 | 2.784968 | 2.787513 | Pass |
| Video | PID Pass@1 | 4.930249 | 4.932828 | Pass |
| Video | PID Pass@32 | 16.938707 | 16.972229 | Pass |
| Video | PID Recall@32 | 2.701457 | 2.705191 | Pass |
| Interactive | SID Pass@1 | 7.000000 | 6.800000 | Inconclusive; observed target missed |
| Interactive | SID Pass@32 | 32.900000 | 33.200000 | Pass |
| Interactive | SID Recall@32 | 20.312897 | 20.693056 | Pass |
| Interactive | PID Pass@1 | 6.700000 | 6.500000 | Inconclusive; observed target missed |
| Interactive | PID Pass@32 | 31.500000 | 31.800000 | Pass |
| Interactive | PID Recall@32 | 19.184008 | 19.603452 | Pass |

Video SID NDCG@32 is 2.972643% / 2.974680% (CUDA / MLU); PID NDCG@32 is
2.897027% / 2.900364%. Both guardrails pass the 99% assessment. All video
99.9% assessments remain inconclusive. CUDA/MLU produced 189/188 invalid SID
candidates and 347,095/347,407 unmapped PID candidates, out of 1,240,992 outputs
each. Neither had duplicate SID or valid PID candidates within a request.

The MLU video execution preserved 23,628 completed serial rows and distributed
the remaining rows over four independent servers. Each server replayed the
same 64 completed inputs before its tail; all 256 ordered output comparisons
matched exactly. The merged result has all 38,781 input IDs and matching prompt
hashes. This execution change is not a latency benchmark.

## Full ad follow-up

The full ad task has 27,677 rows and distinct users. All six observed MLU
primary scores retain at least 99% of CUDA; the lowest retention is 99.6259%
for PID Pass@32. Five confidence assessments pass. PID Pass@32 is inconclusive:
its lower contrast bound is -0.01554 percentage points, rather than evidence
that the quality loss exceeds the 1% relative allowance.

| Metric | CUDA (%) | MLU (%) | 99% assessment |
| --- | ---: | ---: | --- |
| SID Pass@1 | 1.976370 | 2.019728 | Pass |
| SID Pass@32 | 21.675037 | 21.606388 | Pass |
| SID Recall@32 | 7.497444 | 7.531434 | Pass |
| PID Pass@1 | 1.690935 | 1.737905 | Pass |
| PID Pass@32 | 20.280377 | 20.204502 | Inconclusive |
| PID Recall@32 | 6.957859 | 6.994189 | Pass |

SID NDCG@32 is 3.877428% / 3.887265% (CUDA / MLU); PID NDCG@32 is
3.556403% / 3.569666%. Both guardrails pass. All 99.9% assessments remain
inconclusive. Both backends produce zero malformed SID candidates and zero
within-request duplicate SID or valid PID candidates; unmapped PID counts
are 3,906 / 4,006 out of 885,664 generated candidates each.

Both backends use the same eight input ranges and unchanged model settings.
One unstarted CUDA shard moved to a released GPU because its originally
assigned card was occupied. All input IDs, prompt hashes and output counts
were checked. All 1,000 earlier MLU sample outputs reproduce exactly in the
full task; CUDA differs in ordered outputs on one sample row, with unchanged
top-1. The initial sample failure remains reported below.

## Additional tasks and numerical investigation

Before observing their outcomes, we selected 1,000 rows each from ad, product
and label-conditioned recommendation, using seed 20261127. These are samples,
not full-task results. The ad sample shows a real quality gap under the proposed
policy: SID Pass@32 is 20.4% CUDA versus 18.9% MLU, and SID Recall@32 is
6.943333% versus 6.414921%. Product and label-conditioned assessments are
inconclusive. Their results are retained when reporting the full ad follow-up.

Independent FP32 Transformers inference on the same ad sample gives Pass@32
19.3% and Recall@32 6.513095%, also below the CUDA BF16 reference. TF32 was
disabled and attention used SDPA MATH. The explicit shared-prefill FP32 search
matched all 32 outputs of `Transformers.generate` on 67 checked inputs. This
diagnostic does not replace the operational CUDA reference or waive a quality
target miss.

The independent FP32 follow-up on all 27,677 ad inputs gives SID Pass@32
21.776204% versus 21.675037% for CUDA BF16, and Recall@32 7.573326% versus
7.497444%. SID/PID top-32 metrics and NDCG guardrails pass the 99% assessment;
top-1 remains inconclusive. All 1,000 earlier FP32 sample inputs reproduce
exactly in this full run. The sample loss therefore does not generalize to the
full task for this independent reference. It does not erase the initial sample
result or turn an inconclusive MLU confidence assessment into a pass.

Three isolated MLU experiments did not remove the ad sample gap: BF16 rounding
of residual addition before RMSNorm, that change plus CUDA-style SiLU rounding,
and an FP32 final vocabulary projection. None was promoted to production.

We captured native model scores for all 28 ad rows with differing relevant
candidate sets. An independent reconstruction of selection matched both
backends' intermediate token rows and final ordered outputs exactly. MLU
pruned 26 CUDA-only relevant candidates: four at the first stage, 18 at the
second and four at the third. In one case, a candidate ranked fifth in CUDA's
final output vanished because its first code ranked 33rd on MLU. Its score
was tied at the first-stage boundary; the documented token-ID tie rule applied.

CUDA changed the ordered outputs of all 28 selected inputs when replayed with
a warm prefix cache; MLU repeats were exact. These selected cases do not estimate
population repeatability. PyTorch's [numerical accuracy documentation](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html)
explains why devices and batch shapes can differ numerically. Actual quality
preservation still requires measurement.

## Functional checks and limits

The pushed fixes stabilize MLU prefill across cache hits (`699db51`) and release
native request slots after terminal Beam output (`b7012b4`). Three lifecycle
regressions pass, including two that failed before the cleanup fix. The long
benchmark runs exercise request reuse far beyond the 32-slot capacity.

Additional checks passed:

- 50 selected prompts, 150 MLU requests across repeated and reversed orders;
  all ordered recommendations repeat, including prompts up to 19,302 tokens.
- Earlier replay of identical CUDA logits produces all 384 ordered
  recommendations for three fixtures on the MLU selector.
- Independent CPU FP64 attention on BF16 inputs: 12 prefix lengths, 12 causal
  prefill cases and 36 prefix-plus-suffix Beam cases with widths 1, 4 and 32.
  The largest relative RMS errors are 0.1943% for prefill and 0.2805% for decode,
  below the diagnostic's 1% arithmetic-error allowance. That allowance is
  separate from the recommendation quality policy.
- Physical packed KV writes and seven cached-prefix comparisons are exact;
  tests include noncontiguous block tables and page boundaries.

Reproduce the standalone attention check in the prepared MLU environment:

```bash
CN_VISIBLE_DEVICES="$MLU_CARD" python tests/system/check_mlu_attention_math.py \
  --output attention-math.json
```

This evidence does not establish graph replay, TP greater than one, concurrent
request batching, other models, all eight RecIF tasks, or online CTR. A quality
failure must remain visible even when arithmetic and state checks pass.

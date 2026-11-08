# Frozen inference efficiency comparison

Frozen before implementation measurements, 2026-11-08, task #189. This pilot
tests implementation savings and certified reuse. It does not retrain an
architecture or establish Forward–Forward training efficiency (task #190).

## Immutable controls

Use the exact trainer and Kernite revisions pinned by `../single_update/build.rs`
and its original PTX identity guard. Preserve four arms, seeds 7/11, checkpoints
0/5000/10000, data split, FP32 execution, epsilon and parameter budget. Restore
each state and take the next scheduled ordinary source AdamW update. Match its
complete state hash against task #186. The step-10000 update is a post-budget
probe, not additional evidence for the original architecture ranking.

Device: RTX 3050 Laptop GPU, 4 GiB. All compared inference runs use the same
GPU and original dense/activation kernels. CPU FP64 OpenBLAS (four threads)
is permitted for bound preparation and independent numerical audits; candidate
preparation is timed. Compilation and CUDA module loading are excluded and
identified separately from model/cache setup. No Python.

## Methods and serving contract

Each request receives only operand pairs and returns host top-1 predictions.
There is no label access in serving, cache lookup, winner selection or fallback.
Class ties use the first class. Transfers, launches, synchronization, host
argmax/certification, compaction and exact fallback are inside request timing.

1. **Source forward:** the source prefix through logits, with external input
   tokens and without training-only loss/gradient work. All three source weight
   transposes are performed on every batch. Audit against ordinary source
   `forward` logits to establish equivalence.
2. **Cached weights:** exact transposes of the three weight matrices, built
   once per immutable parameter version. Same FP32 GEMM and nonlinear kernels.
   Count cache construction latency and persistent GPU bytes. Reject caches
   from another model version. This method works for fresh operand pairs.
3. **Certified old features:** repeated queries across one parameter update.
   Cache old executed final features and FP64 node statistics for all 9409
   operand pairs. After the update, prepare scalar displacement radii and
   readout contrast norms without evaluating updated hidden features. Gather
   cached features on GPU, apply the new cached readout, and certify its own
   predicted winner against all competitors. Compact uncertified queries into
   an exact cached-weight forward, then scatter those predictions back.
   No answer is cached across timed requests. Charge old-cache construction,
   update-bound preparation and both CPU/GPU cache storage. This workload is
   separate from fresh-input inference. For unchanged models, exact answer
   memoization would be a simpler repeated-query method; it is not this test.

## Fixed inexpensive displacement bound

All norms below are Euclidean or Frobenius; block maxima use scalar absolute
values for real SwiGLU and two-coordinate norms for complex gates. Cache old
node norms and RMS denominators `s(x)=sqrt(mean(x²)+f32::EPSILON)` in FP64.
Transport a displacement radius through RMS as
`min(2*sqrt(128), 2*r/s(old x))`. The anchored normalization bound is the
one used in task #188; clipping uses the two output-norm bounds.

The embedding displacement is computed exactly in FP64 from the rounded
parameter change. For a dense layer `A -> A+D`, use
`norm(A+D)*r + norm(D)*norm(old input)`. The concatenated gate/value radius
bounds each half. A sigmoid gate contributes
`r_gated <= (1 + max_block_norm(old gate)/4)*r_gate`; the plain product uses
`r_gated=r_gate`. For the final bilinear product use
`C*(old_gated_max*r_value + old_value_max*r_gated + r_gated*r_value)`,
with `C=1/sqrt(2)` for complex multiplication and 1 for real/diagonal products.
Sum radii at the residual, and transport through the final RMS. This cheap
bound deliberately omits the extra tangent dense products from #188.

Numerical allowances remain fixed at 1e-3 L2 per executed feature vector and
1e-3 per executed logit relative to the FP64 graph. Centering on old executed
features thus adds 2e-3 to the movement radius. Comparing approximate and
exact executed readouts adds 4e-3 to each class contrast. A cached winner is
accepted only if every observed gap exceeds
`norm(E_winner-E_competitor)*(r+2e-3)+4e-3`, with an additional 1e-8 scaled
strictness tolerance. Check both fixed allowances, the feature balls and all
class-contrast intervals on all queries, retaining every violation. These are
numerically audited allowances, not a verified CUDA rounding-error theorem.
Do not widen them or tune the bound after observing outcomes.

## Quality and timing

For all 24 cases, audit all 9409 queries at batch size 512 (including the final
partial batch). Require bitwise source/cached-weight logits and 100% final
prediction agreement after certified fallback. Report actual train/validation
accuracy separately, pre-fallback coverage and wrong approximate predictions.
Also check the first 512 validation queries at each timing batch size 1/32/512
against the all-query reference. Reject nonfinite outputs or stale caches.

For each case and batch size, use these same first 512 validation inputs in
three warmup and nine measured rounds. Each round processes all three methods;
rotate their order by round number. Retain every raw duration, median and p90
(nearest rank); report throughput and paired speed ratios. A round includes
every request and its host result. Report ratios both against source forward
and against the exact cached-weight control. No best-round selection or
post-hoc subset selection. Enforce a 900-second cohort cap, excluding build,
preserving error and capped cases instead of dropping them.

Time each one-time cache allocation/construction with GPU synchronization.
Report cold cost (setup plus one 512-query measured round), and break-even
number of rounds only when median per-round savings are positive. Common
resident-model creation/upload is reported separately and excluded from warm
request timing. Training/update and independent correctness audits are excluded
from serving latency. Candidate preparation is never charged to an offline
audit. Bound and feature caches are frozen before updated-feature evaluation.

Report explicit allocated GPU buffer bytes and candidate deltas, plus CPU
cache payload bytes. The source engine retains training buffers and sampler
storage; disclose that this is not a minimal inference allocator or total
driver memory. Persist provenance, all outcomes and commands as ignored
artifacts; commit only source, lockfile, protocol and authored findings.

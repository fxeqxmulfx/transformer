# Frozen standalone hidden-gradient training comparison

Task #195, 2026-11-08. Freeze before new candidate training or outcomes.
Task #194 found favorable one-step current-gradient directions at failed
local endpoints, using an AdamW-derived radius. Test a standalone rule
from scratch; do not treat that diagnostic as a successful training result.

## Candidate and controls

Keep the original four arms and 143360 independent real parameters. Reuse
exactly task #190's local objectives L0/L1/L2, fixed random prototypes,
coefficient-one weights, detachment boundaries, gradient accumulation and
classifier. Complex multiplication and the tied embedding are unchanged.

Let a be A's independent real parameters and g the signed pullback of its
current local L1 derivative, both at the old parameter version. Define

`alpha(t) = 0.01 * min(t/10, 1)` for one-indexed update t;
`a_new = FP32(a - alpha(t) * norm(a) * g / norm(g))`.

Norms and signed pullback arithmetic use FP64. Round each new owner once to
FP32 and rebuild the signed complex aliases. If either norm is exactly zero,
leave A unchanged and count the occurrence. Do not divide by zero, clip or
search over steps. Nonfinite inputs/outputs fail the run and remain recorded.
The coefficient is fixed at 0.01: a full step has length 1% of A's current
independent norm before rounding. This is an explicit design choice, not an
optimality claim or an estimate from new validation outcomes. No loss
evaluation, backtracking, terminal gradient or AdamW radius selects it.

A uses no momentum, coordinate preconditioner or weight decay. Its retained
source moment buffers are not read or advanced. E and B receive exactly
their original local-pair AdamW updates, with their original decay, moments,
epsilon and warmup. All local derivatives are collected before any parameter
changes. Source AdamW is launched only for E and B; no A AdamW step is
computed or overwritten. This intervention changes A's optimizer as a whole;
it does not isolate momentum, adaptivity and decay individually.

| Rule | Training update |
| --- | --- |
| ce-bp | Original source CE, full backward, original AdamW on E/A/B. |
| local-pair | Original task #190 local derivatives and AdamW on E/A/B. |
| local-normalized-a | Same local derivatives; normalized current-gradient A rule above, original AdamW on E/B. |

No derivative-free claim: the candidate still computes the local nonlinear
Jacobians and outer-product products. Auxiliary heads are unused at inference.

## Fresh cohort and fixed budget

Use **seeds 17 and 23**, the original initialization/data split/sampler
algorithms and zero initial moments. These seeds were not used to select the
rule in tasks #190/#193/#194. Four arms times two seeds times three rules is
**24 runs**, each **4000 updates**. Evaluate full training and validation CE,
correct counts and accuracy at step 0 and every 100 updates. Continue through
thresholds; no seed search, selected restart, radius grid or extension.

Keep all three engines per arm/seed resident and rotate method order every
100-update chunk. Cap a method at 120 seconds of active setup/training/
evaluation, and the cohort at 900 seconds of wall time, excluding compilation
and correctness preflight. Check caps at chunk boundaries. Preserve all 24
slots, errors, capped histories and endpoint states. Training always starts
from fresh initialization; saved states used in correctness checks never
initialize a run. Previous tasks remain in review, not independently approved.

## Correctness before training

Pin the original trainer/build/PTX and task #190's compiled-source digest
`0f63edd42b7aa9968345bdd7b3815c9d3c9b098d378944449a18f174c2f37aa8`.
Reuse its checked local derivatives. Revalidate its original numerical audit
and CE/source next-step identity before this comparison. Record hashes and
verify original checkpoint inputs unchanged after the study.

For all four arms and both fresh seeds at initialization, compare the new
GPU A update against an independent FP64 CPU calculation in independent
coordinates on the next scheduled batch. Require per-coordinate error at
most `2e-8 + 2e-6*abs(expected)` and relative norm-reduction error at most
`1e-12*(1+abs(expected))`. Require bitwise E/B parameters and both moments
to equal the original local update, and bitwise unchanged A moments. Repeat
these update checks at seed 7's original step-5000 states to cover nonzero
moments; those states are audit inputs only. Rebuild all parameter ties.

Check two deterministic normalized A directions for each fresh initial
state against independent FP64 local-L1 central differences on the first
eight training examples: step 1e-4, tolerance
`2e-5 + 2e-3*max(abs(finite),abs(analytic))`, **16 scalar checks**.
Also test zero gradients, zero A, and nonuniform signed gradients in all
four arms, including the final coordinate and complex aliases. Record the
new kernel PTX hash. Do not loosen tolerances after observing failures.

Fresh-seed CE controls call the unchanged source training step directly;
they have no pre-existing saved trajectory to replay. Validate source
initialization, data and sampler hashes and initial metrics across methods.
Check parameter/moment finiteness and ties at every scheduled evaluation,
and retain these checks and zero-norm counts. All correctness gates must pass
before interpreting training outcomes.

## Quality, cost and interpretation

Report every endpoint and history, best observed accuracy/CE, first observed
99% and 99.5%, and first attainment of the paired CE control's endpoint
accuracy **and** no-worse CE. Compare the candidate with original local-pair
on both endpoint metrics, and with CE at the joint quality criterion.
Use CE's first attainment of its own endpoint criterion as its time reference.
A lower CE alone, a single threshold crossing, or faster steps with worse
quality are not a repair or efficiency win. This pilot is not a statistical
generalization guarantee or a final untouched test evaluation.

Synchronize around measured chunks. Charge initialization/prototypes, all
forward/local derivative work, norms, updates, host launches and scheduled
evaluation/state checks. Report setup, training, evaluation and active time
separately, plus raw chunks and operation counts. The new update uses GPU
norm reduction, scalar normalization, A owner writes and two source E/B
AdamW launches; no CPU parameter transfer implements training updates.

Report actual allocated GPU payload, including auxiliary heads, norm scratch,
the retained A moment buffers and unused global-backward buffers. Do not claim
memory savings from theoretical omitted storage. Exclude compilation/JIT,
preflight, other methods' turns, ownership polling, report serialization and
checkpoint I/O from active timings, and disclose exclusions. Poll all GPU
ownership before/after each group and every 400 updates; foreign or unknown
ownership invalidates clean-device timing claims but does not drop results.

Use Rust/Kernite only. Preserve generated reports, raw histories, checkpoints,
logs and executable outside the temporary worktree, ignored and uncommitted.
Commit source, this protocol, generators and authored findings. Interpret
failure within this fixed budget without silently revising the rule or runs.

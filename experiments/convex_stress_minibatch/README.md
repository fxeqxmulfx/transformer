# Convex stress learners with persistent minibatches

Question: what changes when both convex learners consume the original GPT
stress suite in its original minibatches? This is a separate 210-cell
cohort: 35 tasks, seeds 0/1/2 and two learners. The completed full-batch
cohort in `../convex_stress/` remains frozen. No Python is executed.

| Arm | Update | Endpoint |
| --- | --- | --- |
| Frank-Wolfe | Exact current-minibatch CE oracle, one joint-tensor update per visited query | First observed 99% primary exact accuracy or original update budget |
| Forward-only | Persistent records for every candidate key; eliminate conflicts permanently | One complete epoch, exact comparison with complete-data fitting |

`protocol.json` fixes the comparison before training. The original batch
size, epoch tail, maximum update count, primary split and observation
cadence come from `../convex_stress/suite.json`. Source `Random(seed+10000)`
shuffles examples each epoch. Both arms use the same fixed causal features,
query token, class-zero defaults and full-vocabulary unit-scale CE as v1.
All frozen dataset fingerprints are checked. Evaluation retains source
own-output rollout, EOS, field alias pooling and all endpoint splits.
As in v1, generative training splits additionally report own-output rollout;
the original transformer runner uses teacher forcing alone there. The
`teacher_forced` subreport supplies that training comparison. Primary
held-out sequence metrics retain the original generation convention.

For Frank-Wolfe, query visit `t` uses `eta=2/(t+1)`. An absent query keeps
its tensor. Weighted integer counts of vertices represent the actual
joint tensor: no separately trained routing/decoder product is introduced.
For a candidate value absent from the current minibatch, its gradient is
zero; retaining that candidate's previous oracle class is an exact tie
choice. Histories begin at class zero. Observed rows use the smallest
minimizing class; key ties choose the smallest key. Training sees only
labels in the current batch, never held-out labels. Fixed feature caching
does not initialize trainable state from the full set of training labels.

`ConvexChannel/Batch.lean` proves that query-dependent steps preserve
nonnegativity, value-independent routing marginals and normalization,
and that unvisited queries retain their coefficients. CC2 gives convex
CE; CC8 gives the exact linear oracle. These facts do not prove convergence
of this stochastic algorithm. Its final full-training mean CE and FW gap
are measured separately; a nonzero gap is retained as optimization error.
No full-data optimum stopping rule is assumed for the minibatch learner.

The forward implementation ports CF6 and `convex_channel_minibatch` at
`c1b5f814`, allocating candidate records only for observed queries and
discarding eliminated records. Every surviving candidate persists across
boundaries, not just the currently selected key. Snapshots do not mutate
state. A failed query yields an explicit failure, never a fallback model.
One full epoch is intentional: repeated observations cannot change the
completed consistency result. The first observed 99% crossing is recorded,
but does not stop this equivalence check early. Final comparison covers
every query, key, value and output coefficient, including unseen defaults;
all explicit failures are compared too.

Validation includes an independent dense CE gradient and finite-difference
check, exhaustive tiny linear-oracle enumeration, coordinate-wise update
comparison, absent-query persistence, live/saved forward equality, and all
81 partial Boolean datasets at multiple batch boundaries and prefixes.
Metrics are compared with the unchanged v1 evaluator, including an incorrect
generated answer. Every saved model is replayed on every original split;
explicit forward failures are recomputed. The source fingerprint includes
the reused generator/model sources, new update and evaluation sources,
manifest, protocol, lockfile and launch script.

Reproduction from the repository root:

```sh
cargo test --offline --manifest-path experiments/convex_stress_minibatch/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_minibatch/Cargo.toml -- --check
cargo clippy --offline --all-targets --manifest-path experiments/convex_stress_minibatch/Cargo.toml -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_minibatch/Cargo.toml
bash experiments/convex_stress_minibatch/run_suite.sh experiments/convex_stress_minibatch/runs/minibatch-v1 8
```

The runner refuses to overwrite a cohort or cell. It retains every process
exit, trace, model and explicit failure, then checks all 210 cells and
replays them. Generated evidence is ignored under the root's `runs/` and
`.kanban/artifacts/153/`. CPU f64 execution is appropriate for these sparse
categorical updates. This is not a matched transformer timing comparison.

## Separate branch-amplitude representation diagnostic

The earlier suggestion that branch amplitudes require richer features was
too strong. The existing lag-64 and lag-32 slots suffice for a classifier:
route half the mass to each, and give equal class weight to outputs
consistent with that slot's sign. The true output then has logit `0.5`,
versus `0.25` for the two alternatives. The delimiter selects EOS similarly.
`branch_diagnostic` constructs this feasible witness using the known source
task. It is a posthoc capacity check, not learned weights or a benchmark arm.

On the frozen train/validation/test splits, the witness is 100% correct and
has zero measured feasibility error. Its training CE at scale one is
`3.751587`, worse than the v1 seed-zero learned model's `3.736430`, although
that learned model has 0% exact sequences. Thus this CE comparison does not
support a feature-impossibility explanation. At fixed scale `ln(69)`, the
witness CE is `2.265859` versus `2.332030`; at scale 10 it is `0.475535`
versus `1.998767`. Scaling these existing models leaves their argmax answers
unchanged. These comparisons do not establish what training at another
scale would find. A fixed positive multiplier preserves linear logits and
convex CE in the joint tensor; a jointly learned multiplier would need a
separate convexity argument. The minibatch cohort retains scale one.

```sh
cargo run --offline --release --manifest-path experiments/convex_stress_minibatch/Cargo.toml \
  --bin branch_diagnostic -- \
  experiments/convex_stress/runs/stress-v1/frank_wolfe/gpt-probe-branch-amplitudes/seed0/channel.json
```

## Cohort findings

Pending completion of the frozen 210-cell minibatch cohort.

## Why exact minibatch oracles need another convergence argument

`tests/oracle_bias.rs` gives a concrete obstruction for the implemented
stochastic rule. The two prefixes `[0,2]` and `[1,2]` have respective labels
0 and 1. Lag one identifies both labels, and a single full-batch update
fits them with zero FW gap. In a singleton batch, every candidate key can
assign the same best class to its one observed value. All key costs tie,
so the frozen tie rule chooses lag zero, whose value is always 2. Every
chosen vertex therefore uses a constant feature. Their convex mixture
also predicts the same answer for both inputs, so accuracy is exactly 50%.

The native check retains this obstruction for 1000 updates, with full-data
FW gap above 0.49 and CE more than 0.4 above the feasible optimum. This is
an algorithmic limitation on fully representable observed data, separate
from missing-label identification. Exact batch gradients do not imply
that minimizing each noisy linear objective solves the full CE problem.
Gradient averaging, variance reduction or growing batches need a separate
implementation and convergence check. The frozen stress cohort continues
to measure the original minibatch rule; this diagnostic does not tune it.

## Repairs indicated by the diagnostics

1. Retain gradient information across minibatches, with a convergence argument
   for the chosen estimator. The singleton counterexample rules out assuming
   that the current exact-batch-oracle rule solves the full convex objective.
2. Share joint coefficients across equivalent recall queries. With a fixed
   alignment `k_q(j)` of input slots, constraints of the form
   `W(q,k_q(j),v,c) = W(q',k_q'(j),v,c)` are linear. They can transfer observed
   value decoding to a query that has not seen that value. Separately trained
   routing and decoder factors would introduce products of learned parameters;
   convex sharing must be specified on the joint tensor. The equivalence of
   queries is an additional structural assumption, not a consequence of a
   flat spectrum.
3. Test a fixed positive logit multiplier with ordinary CE. The branch
   witness separates representability from the unit-scale loss preference.
   Multiplication by a fixed constant preserves affine logits and convex CE;
   the diagnostic does not establish the outcome of training at that scale.

The parity and run-count tasks still need separate representation checks.
Their names alone do not establish that a feature extension is necessary.
Flat spectrum remains a rank/error bound. Identification of unseen labels
requires shared structure or a valid observation certificate, and successful
optimization requires an appropriate algorithm even on a convex domain.

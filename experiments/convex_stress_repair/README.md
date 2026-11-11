# Convex stress repair: gradient memory and shared aligned queries

Question: can we remove the proved minibatch oracle failure and the independent
query-decoder failure while retaining a convex joint model and ordinary CE?
Flat spectrum remains a capacity bound, not an identification assumption.

The Lean statements were checked first: `ConvexChannel/Minibatch/` proves the
singleton failure and the two-gradient repair; `ConvexChannel/Shared.lean`
proves feasibility and convexity of the shared model and fixed-scale CE.
The accompanying note is `review/flat_spectrum.md`, MF1–MF3.

| Arm | Parameters and inputs | Update |
| --- | --- | --- |
| Full batch | One joint tensor on shared, aligned causal slots | Refresh every training prefix's gradient, then take one FW step |
| Minibatch | Identical tensor and feature map | Replace only the current batch's stored gradients, then take one FW step |
| Forward only | Identical tensor family and feature map | Persistent consistency records, one complete epoch; explicit failure if no key fits |

All original relative-lag values are retained. Each token-indexed successor or
presence block is permuted relative to the visible final query. Thus the slot
for that query's most recent completed successor has the same index across
queries. One new constant slot handles constant functions without losing the
original lag-zero token. No answer, task name or arithmetic operation enters
feature extraction. The value-to-class map is learned, including transformations
that do not copy the input value. Sharing is an explicit structural restriction;
it can be inappropriate for other tasks and is not implied by flatness.

The optimizer keeps one gradient per consumed training prefix, indexed by a
stable training ID. A revisit replaces its old gradient; repeated visits do
not increase that example's weight. This is an incremental aggregate of possibly
stale gradients, not an unbiased-gradient convergence claim. The linear oracle
uses the entire retained aggregate. Per-key choices on unseen value rows persist.
Integer weights of feasible vertices implement `eta=2/(updates+1)` and represent
the actual joint coefficients; there are no separately learned routing/decoder
factors. Every update is feasible by CC8/CM1.

Class-independent softmax backgrounds are subtracted for sparse storage. MF2
proves that this changes every feasible linear objective by the same constant.
Ordinary full-vocabulary CE is still evaluated, and the endpoint FW gap is
recomputed from fresh gradients on all training prefixes. Fixed scales 1 and 10
are separate prespecified arms; their convexity is proved. The scale is not
trained jointly with the tensor.

## Validation and interpretation

Seven native checks cover the frozen singleton failure, its repair, a full-batch
query-transfer failure, finite-difference gradients, exhaustive tiny linear
oracles, gradient replacement, causal alignment and exact live/saved predictions.
The old singleton learner remains at 50% after 1000 updates, with FW gap above
0.49. The repaired learner fits both examples after two singleton updates and
has gap below `1e-5` at 1000 updates.

The full-batch transfer witness uses three queries: one observes both values;
the other two each miss one value. All four training labels fit with zero gap
under the old independent-query model, but both missing query/value answers
are wrong. The shared model fits all six pairs after one full-batch update,
learning the inverted decoder. An additional negative control retains the
ambiguous two-observation case: without variation identifying a slot, sharing
alone does not guarantee unseen answers. It is deliberately not reported as
a solved case.

## Stress protocol

`protocol.json` is fixed before the repair pilot: eight tasks spanning short
and long data, three arms, two fixed scales and seed zero, for 48 cells. These
are diagnostic caps of 64 full-batch or 512 minibatch updates, not the earlier
full-budget transformer benchmark. The original minibatch sizes, fresh-range
epoch shuffle, source data fingerprints, primary split, exact sequences,
own-output rollout, EOS and residue pooling are retained. Full batch also
stops at a fresh empirical gap of at most `1e-8`; a low held-out score at that
point is retained. Forward-only fitting is compared with a single complete-data
batch. Every saved model or explicit failure is replayed on original splits.

As in the frozen cohorts, generative training splits additionally report rollout;
their `teacher_forced` subreport supplies the original runner's training metric.
The earlier full-batch results and user-stopped minibatch cohort remain separate.
No new transformer run or matched timing comparison is claimed.

## Reproduction

```sh
cargo test --offline --manifest-path experiments/convex_stress_repair/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_repair/Cargo.toml -- --check
cargo clippy --offline --all-targets --manifest-path experiments/convex_stress_repair/Cargo.toml -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_repair/Cargo.toml
bash experiments/convex_stress_repair/run_pilot.sh experiments/convex_stress_repair/runs/pilot-v1 4
```

The runner refuses to overwrite any cell. All 35 frozen tasks can also be run
with `target/release/convex-stress-repair run ARM TASK SEED SCALE OUTPUT`.
Outputs, logs and binaries remain ignored under root `runs/` and
`.kanban/artifacts/160/`. This native sparse CPU implementation executes no Python.

## Findings

The prespecified repair pilot has not yet completed. A full stress-suite pass
has not been established.

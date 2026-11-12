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

All 48 prespecified cells complete and replay at source `12e18759`, fingerprint
`e526fccffc2087e55488e0919f0442b630026ff765e046a5e8f08ef2432cb320`.
There are 36 saved models and 12 explicit forward-only failures; no execution
or replay failures. Primary exact accuracy, as percentages:

| Task | Full, scale 1 / 10 | Minibatch, scale 1 / 10 | Forward, either scale |
| --- | --- | --- | --- |
| Boolean AND shift | 0 / 100 | 0 / 100 | No consistent key |
| Short key/value recall | 100 / 100 | 100 / 100 | 100 |
| Long recall | 100 / 100 | 100 / 100 | 100 |
| Branch generation | 0 / 100 | 0 / 100 | No consistent key |
| Division | 0.95 / 1.76 | 0.95 / 1.35 | No consistent key |
| Parity 16 | 55.41 / 54.14 | 50.96 / 53.50 | No consistent key |
| Depth easy | 18.55 / 20.70 | 18.55 / 19.34 | No consistent key |
| Fibonacci recurrence | 0 / 0 | 0 / 0 | No consistent key |

The full-batch recall repair takes one update on both tasks, compared with
86.13% short and 50% long exact accuracy in the original full-batch cohort.
Minibatches reach 100% at the first observation, update 64 for short and 50 for
long recall; only consumed prefixes contribute to training. Both forward fits
also reach 100% after a full epoch. All three modes have 100% training accuracy
on these recall tasks. At scale 10, branch generation reaches 100% training and
held-out exact sequences in six full-batch or 32 minibatch updates. Its fresh
FW gaps are still 1.2614 / 1.0874: this is a prediction pass, not convergence.

The Boolean AND primary split contains seven examples with one answer class.
Its scale-10 early endpoint passes that split while training exact accuracy is
only 50.78%. This is **not** a complete Boolean-rule repair. At scale 1, full
batch has a fresh gap of about `2.6e-15` and still scores zero on that primary
split. Mandatory sharing has removed useful query-specific capacity; merely
running its full-batch optimizer longer does not fix this endpoint. MF4 proves
the broader sharing obstruction and the unrestricted oracle alternative, tested
separately in [`convex_stress_adaptive`](../convex_stress_adaptive).

There are 16 primary-threshold passes out of 48 cells, including the two narrow
Boolean primary passes just described. The complete stress-suite pass remains
unachieved. Division, parity, counting and recurrence require further structure
or optimization work; a small empirical gap alone does not identify their
held-out answers. Caps and one seed limit all comparisons.

Regenerate the full inventory, preserving failures, with:

```sh
bash experiments/convex_stress_repair/report.sh experiments/convex_stress_repair/runs/pilot-v1
```

The reporter checks the protocol's exact Cartesian product, source fingerprint,
endpoint identity and replay status. A missing planned cell is rejected; a
pending endpoint is retained explicitly and cannot count as a pass.

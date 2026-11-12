# Calibrated convex causal channels on the short and long suite

Does replacing scaled probability logits with a calibrated probability readout
improve the existing convex channel on the original 35 stress tasks?

Lean comes first at `281fb852`: CCa1–CCa2 in
[`review/convex_causal.md`](../../review/convex_causal.md) and
`ConvexCausal/{Basic,Mixture,Compatibility}.lean`. For a probability row Q,
the new logits are `log((1-epsilon) Q + epsilon/V)`. Their ordinary softmax CE
is exactly the negative log of the smoothed target probability and is convex
in the joint channel coefficients. Epsilon is fixed, not learned. The same
proof covers any fixed convex bank; this experiment uses the existing compact
count channel, not the exponential universal prefix table or a spectral table.

| Arm | Readout | Training |
| --- | --- | --- |
| `scaled` | Original `10 Q` | Existing integer-weighted Frank–Wolfe steps |
| `calibrated` | `log(0.99 Q + 0.01/V)` | Same steps with the derivative of the calibrated loss |

Both arms retain counts-v1's query-aligned causal slots, successor/presence
features, raw token counts, shared and query-specific joint coefficients,
linear oracle, missing-row completion and initial uniform predictions. No task
name or arithmetic/Boolean answer rule enters the predictor. The Rust optimizer
is a port of `convex_stress_counts/src/lib.rs` at `281fb852`; the features,
forward consistency learner and original scoring code are imported. Integer
mixtures preserve the original probability constraints. The retained gradient
machinery is unchanged, but this cohort refreshes every observation each update.

`protocol.json` freezes 70 cells: both arms, seed zero, all 25 original short
and ten original long tasks. It uses the original datasets, splits, vocabulary,
teacher supervision, own-output generation, EOS, numeric metrics and recurrence
residue pooling. Every original data hash is checked before training. No rows
are deduplicated. Context capacity reads input and rollout lengths, not answers.

Stopping uses fresh full-training CE excess, the unrestricted full-training FW
gap, 64 updates, or 120 seconds including setup. The time check happens after
a complete update and its fresh diagnostic; the recorded time can exceed 120
seconds. Evaluation and replay are separate. Each process also has an 8 GiB
address-space limit and a 600 second wall limit. Resource failures remain in
the inventory. The launcher permits one or two experiment processes. This is
a controlled loss ablation, not a matched-time transformer comparison. The
short/long tiers retain their original historical membership definitions.

The endpoint is saved before held-out evaluation. No held-out accuracy selects
a candidate, controls stopping or changes budgets. Both arms are reported
independently, with primary 99% exact-sequence passes and passes on every
nonempty split. The larger model class and conditional identification theorem
in CCa4–CCa7 do not imply that this compact class identifies unseen answers.
Tight empirical loss or gap is not an unseen-answer certificate.

## Reproduction and checks

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.ConvexCausal
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexCausalAxioms.lean
cargo test --offline --locked --release --manifest-path experiments/convex_stress_calibrated/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_calibrated/Cargo.toml --check
cargo clippy --offline --locked --manifest-path experiments/convex_stress_calibrated/Cargo.toml --all-targets -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/convex_stress_calibrated/Cargo.toml
experiments/convex_stress_calibrated/target/release/convex-stress-calibrated prepare
# Copy the committed binary to ignored evidence storage; retain its SHA-256,
# source commit and fingerprint before launching the frozen binary.
bash experiments/convex_stress_calibrated/run_suite.sh FROZEN_BINARY FRESH_COHORT 2
FROZEN_BINARY summary COHORT
```

Five native tests check calibrated normalization and ordinary CE, feasible
finite-difference derivatives, convex combinations, an independent exhaustive
small-model gradient/oracle calculation, exact scaled-control equivalence,
learning a count decoder on nine observations and predicting all 256 eight-bit
inputs, saved model equivalence, and rejection of corrupt objectives,
configuration and incomplete inventories. The standalone replay regenerates
the original data, checks saved model hashes and exact integer feasibility,
recomputes CE and the fresh oracle gap without training history, checks the
stop reason and training trace, and reproduces every original split metric.
It also compares the optimizer's loss with the original softmax CE evaluator.
The full endpoint gap replay shares the optimizer's oracle; the independent
exhaustive comparison is a small fixture, not an interval proof on the suite.

## Findings

The protocol and implementation are frozen before the cohort. Full-suite
measurements will be recorded here once all 70 cells have terminal records.
Generated models, traces and summaries remain ignored under `runs/` and board
artifact #199; none is a committed result.

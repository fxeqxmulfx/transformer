# Convex channels with causal token counts

Question: can a generic count feature repair the full-batch representation
failures while retaining the previous convex model and ordinary cross-entropy?

Lean comes first at `4b5ec978`: `ConvexChannel/Counts.lean`, CFt1–CFt3 in
`review/flat_spectrum.md`, embeds every old channel by zeroing the extra keys,
preserves its logits and CE exactly, and proves convexity of the enlarged
objective. A fixed-query two-slot XOR obstruction and a feasible count-based
witness show a strict capacity gain. This witness is not a theorem that the
original complete stress adapter cannot represent parity. Flat spectrum stays
a bound on the enlarged categorical feature width, not a source of labels.

The native model is a frozen port of completion-v1 at `8619a891`. It appends
one occurrence count per vocabulary token to the existing causal slots, aligned
relative to the visible query. Zero count uses PAD; positive counts occupy a
new value range. Every old key and value remains intact. Feature values and
output classes have separate dimensions, so the expanded counts do not add
artificial classes to CE. All counts use only the supplied prefix. There are
no task-name, parity-answer or arithmetic features, and no trained factors
multiplied together. The decoder is learned from ordinary labels.

| Arm | Training on the enlarged domain |
| --- | --- |
| Full batch | Refresh every sample gradient and use the unrestricted oracle with optional sharing and missing-row completion |
| Minibatch | Replace consumed samples' retained gradients using the original task batch sizes |
| Forward only | Retain shared and per-query consistency candidates across one epoch; check exact complete-data equivalence |

The initial prospective protocol has eight original tasks, three arms, two fixed
scales and seed zero: 48 cells. Three parity cases test the new representation;
two depth cases test order-sensitive limitations; Boolean AND, short recall
and ordinary division provide regression and identification controls. These
tasks were selected after observing earlier failures. Data fingerprints,
splits, own-output generation, EOS, primary exact-sequence metrics, update
caps (64 full / 512 minibatch) and observation cadence remain unchanged.
That first cohort is a targeted follow-up; the extension below covers all 35
tasks. Neither is a multi-seed, matched-time or fresh transformer comparison.
Counts absent from training need not have the
right labels, particularly on longer inputs.

Ten native checks cover the eight earlier repair invariants, causal feature
values and unchanged old slots/output vocabulary, and count-rule learning
from nine labeled examples to all 256 eight-bit permutations in all three
modes. The latter also compares fresh CE/gaps against an independent dense
full-vocabulary gradient/oracle. Forward batch equivalence, contradictory-data
failure, exact integer feasibility and saved predictions remain checked.

```sh
cargo test --offline --manifest-path experiments/convex_stress_counts/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_counts/Cargo.toml -- --check
cargo clippy --offline --all-targets --manifest-path experiments/convex_stress_counts/Cargo.toml -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_counts/Cargo.toml
bash experiments/convex_stress_counts/run_pilot.sh experiments/convex_stress_counts/runs/pilot-v1 4
bash experiments/convex_stress_repair/report.sh experiments/convex_stress_counts/runs/pilot-v1 experiments/convex_stress_counts/protocol.json
jq --slurpfile adaptive experiments/convex_stress_adaptive/runs/pilot-v1/summary.json \
  --slurpfile completion experiments/convex_stress_completion/runs/pilot-v1/summary.json \
  --slurpfile association experiments/convex_stress_completion/runs/associative-v1/summary.json \
  -f experiments/convex_stress_counts/compare.jq \
  experiments/convex_stress_counts/runs/pilot-v1/summary.json
```

Every planned cell, including explicit failures, is retained and replayed.
The comparator uses completion-v1 for its overlapping tasks and adaptive-v1
otherwise, validating frozen source and data identities. Generated evidence
and frozen binaries stay ignored under root `runs/` and `.kanban/artifacts/161/`.
No Python is executed. The small native witness and the original-data stress
outcomes are reported separately.

## Complete-suite follow-up

After all 18 parity cells replayed at 100% primary accuracy, while the pilot's
depth controls were still running, `followup.json` fixed **all 27 remaining
tasks**: 162 further cells with the unchanged binary, source fingerprint
`b06514ec0c08f583d74fe431b3e3a97ea898366c849c2c71f0f3742ae03de210`.
The two disjoint cohorts cover every original task. This extension was chosen
after the parity results, not prespecified with the first 48 cells. It neither
replaces failed cells nor changes training, scales, caps, data or metrics.
The combined report checks the exact original 210-cell inventory. The paired
baseline includes the separate six-cell associative completion follow-up, so
its known improvement is not incorrectly attributed to counts.

```sh
bash experiments/convex_stress_counts/run_followup.sh experiments/convex_stress_counts/runs/remaining-v1 8
bash experiments/convex_stress_repair/report.sh experiments/convex_stress_counts/runs/remaining-v1 \
  experiments/convex_stress_counts/protocol.json experiments/convex_stress_counts/followup.json
jq -s --slurpfile suite experiments/convex_stress/suite.json \
  -f experiments/convex_stress_counts/combine.jq \
  experiments/convex_stress_counts/runs/pilot-v1/summary.json \
  experiments/convex_stress_counts/runs/remaining-v1/summary.json
# Save that generated combined JSON locally, then run compare.jq as above,
# adding --argjson expected 210 and supplying the combined report as its input.
```

## Completed findings

All **210 count-model cells** complete and replay at runner source `981a7eca`:
204 saved models, six explicit forward-only fit failures, no execution or
replay errors. The failures are branch generation and both depth tasks at
each scale. The six additional recurrence controls below are separate.
Primary passes mean at least 99% original primary exact accuracy; all-split
passes additionally require that threshold on every reported nonempty split.

| Mode | Primary passes, scale 1 / 10 | All-split passes, scale 1 / 10 |
| --- | --- | --- |
| Full batch | 9/35 / 10/35 | 8/35 / 9/35 |
| Minibatch | 9/35 / 10/35 | 7/35 / 8/35 |
| Forward only | 9/35 / 9/35 | 8/35 / 8/35 |

The best result rises from **7/35 to 10/35**. All three parity tasks are new
passes in every mode and scale; no previous primary pass is lost. Each of
the 18 parity cells has 100% primary accuracy. Full batch reaches 100% training
and primary accuracy in one update with numerical zero gap. Forward-only
also fits training exactly. Minibatches cross at the first scheduled observation,
update 250: training accuracy is 100%, 99.97% and 99.95% for lengths 16, 20 and
24. Their gaps remain nonzero. Every existing nonempty parity split reaches
at least 99%; this does not test new longer count ranges. The saved parity-16 full-batch
model selects the count of token 22 at query 18 and constant EOS routes afterward.

Both recall tasks and associative retrieval remain at 100% training and primary
accuracy. Full batch takes one update; long recall takes 50 minibatch updates
at both scales and associative retrieval 19. Scale-10 branch generation remains
100% in full batch and minibatches, taking 11 and 64 updates respectively.
Those branch gaps are 0.8883 and 0.6077, so a prediction pass is not a convergence
claim. Boolean position-shift OOD failures remain despite primary passes.

Ordinary division remains at **0.9450%** held-out accuracy in all six cells,
with 100% training accuracy and numerical zero full-batch gap. The depth tasks
still fail. Full-batch depth-easy scores 24.80% / 27.54%, and depth-hard
17.97% / 5.08%, at the 64-update caps. Their fresh gaps remain positive; the
scale-10 hard case has gap 0.1508. These caps do not establish their best
attainable classification accuracy. Larger capacity need not improve every
finite-run or held-out score: that hard-depth result is below the prior 17.19%,
and cube-plus CE falls from 1.08% to 0%. All such regressions remain in the
comparison. Where only adaptive-v1 was tested before, both inherited completion
and the new features differ; those deltas are not isolated count ablations.

The full-suite target remains unmet. Counts repair the demonstrated parity
failures; order-sensitive representation, rule selection on unseen combinations
and missing-label identification still need work. Flat-spectrum capacity bounds
and the new equal-factor or separate-cell convergence results do not identify
these unseen CE labels by themselves.

## Independent CE check on saved models

The saved-model auditor is ported from `f3e27323` to the larger value alphabet.
It computes uncentered full-vocabulary gradients and exhaustively minimizes
the linear cost over each query/key/value/class, using no retained gradients
or optimizer oracle. On parity-16 full batch at scale 1 and parity-24 minibatches
at scale 10, independent gaps agree within `3.2e-13` and CE within `6.4e-17`.
The first gap is numerically zero; the second remains `0.0044565` despite
100% primary accuracy. This is an f64 cross-check, not an interval certificate.
A tampered saved gap is rejected. The cohort combiner also rejects a missing
planned cell. Adding this auditor leaves the frozen training binary identical.
The additional division and unit-scale depth-hard controls agree in CE exactly
and in gap within `1.8e-11`; the depth audit sums 943,082 training prefixes and
confirms its small positive gap, not exact convergence.

```sh
cargo run --offline --release --bin audit_saved --manifest-path experiments/convex_stress_counts/Cargo.toml -- \
  experiments/convex_stress_counts/runs/pilot-v1/scale1/full_batch/basis-parity16/seed0 \
  experiments/convex_stress_counts/runs/pilot-v1/scale10/minibatch/basis-parity24/seed0
# Additional controls use the same command with these two directories:
# runs/pilot-v1/scale1/full_batch/arithmetic-div/seed0
# runs/pilot-v1/scale1/full_batch/basis-depth-hard/seed0
```

## Recurrence attribution control

Counts-v1 scores 0% recurrence validation accuracy in both CE modes, against
adaptive-v1's 1.1267%; forward-only remains at 1.1267%. Completion-v1 had not
previously been tested on recurrence. After seeing this result,
`completion_control.json` fixed six diagnostic runs using its unchanged binary,
to check the inherited completion rule before attributing the drop to counts.

All six control cells complete and replay. Every split metric, update count,
reported CE and gap is exactly equal to its counts-v1 counterpart. The 0% CE
result is already present with missing-row completion and no count features.
This control is additional evidence, not a replacement for any count-model
result or a correction to the frozen earlier comparison. It demonstrates that
the useful recall completion can worsen a different task's unseen predictions
with an unchanged full-batch training fit and numerical zero gap.

```sh
cargo build --offline --release --manifest-path experiments/convex_stress_completion/Cargo.toml
bash experiments/convex_stress_counts/run_control.sh experiments/convex_stress_counts/runs/completion-control-v1
bash experiments/convex_stress_repair/report.sh experiments/convex_stress_counts/runs/completion-control-v1 \
  experiments/convex_stress_completion/protocol.json experiments/convex_stress_counts/completion_control.json
jq --slurpfile counts experiments/convex_stress_counts/runs/combined.json \
  -f experiments/convex_stress_counts/compare_control.jq \
  experiments/convex_stress_counts/runs/completion-control-v1/summary.json
```

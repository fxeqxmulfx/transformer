# Convex channels with causal token counts

Question: can a generic count feature repair the full-batch representation
failures while retaining the previous convex model and ordinary cross-entropy?

Lean comes first: `ConvexChannel/Counts.lean`, CFt1–CFt3 in
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

The prospective protocol has eight original tasks, three arms, two fixed
scales and seed zero: 48 cells. Three parity cases test the new representation;
two depth cases test order-sensitive limitations; Boolean AND, short recall
and ordinary division provide regression and identification controls. These
tasks were selected after observing earlier failures. Data fingerprints,
splits, own-output generation, EOS, primary exact-sequence metrics, update
caps (64 full / 512 minibatch) and observation cadence remain unchanged.
This is a targeted follow-up, not a full-suite, multi-seed, matched-time or
fresh transformer comparison. Counts absent from training need not have the
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
  -f experiments/convex_stress_counts/compare.jq \
  experiments/convex_stress_counts/runs/pilot-v1/summary.json
```

Every planned cell, including explicit failures, is retained and replayed.
The comparator uses completion-v1 for its overlapping tasks and adaptive-v1
otherwise, validating frozen source and data identities. Generated evidence
and frozen binaries stay ignored under root `runs/` and `.kanban/artifacts/161/`.
No Python is executed. Stress outcomes will be recorded after this fixed cohort
closes; the small native witness is not a stress result.

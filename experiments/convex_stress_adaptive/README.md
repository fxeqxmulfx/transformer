# Convex stress repair with optional sharing

Question: can gradient memory and shared recall transfer coexist with the full
query-specific capacity of the original convex model?

The earlier repair solved short and long recall, but mandatory sharing removed
some query-specific rules. Lean was extended first at `69709c89`:
`ConvexChannel/Adaptive.lean` proves a shared XOR obstruction, a feasible
query-specific witness, and an exact unrestricted oracle with a sharing
preference. The source is `review/flat_spectrum.md`, MF4. Flat spectrum remains
a capacity bound; it supplies neither an algorithm nor arbitrary unseen labels.

| Arm | Domain | Learning |
| --- | --- | --- |
| Full batch | Original query-indexed joint tensor on aligned causal slots | Refresh all gradients; compute the unrestricted oracle and a shared candidate; prefer sharing only when its linear cost is no larger |
| Minibatch | Identical full tensor domain | Replace gradients only for the current original-size minibatch, retaining the other consumed prefixes' gradients |
| Forward only | Same deterministic feasible vertices | Keep shared and query-wise consistency records; use a shared fit when possible, otherwise the query-wise fit, or report explicit failure |

No separately trained factors are multiplied. Integer counts sum feasible
vertices with step `2/(updates+1)`. Shared and local counts are added before
conversion into logits. Unseen queries use the shared contribution plus the
class-zero vertices assigned by previous unrestricted steps. This completion
is explicit and need not be correct. For each consumed prefix, its stored
gradient is replaced on a revisit. The endpoint CE and unrestricted FW gap are
always recomputed from fresh gradients on all training prefixes.

All original causal inputs are retained; the alignment is inherited unchanged
from `convex_stress_repair`. There are no arithmetic features, task-name inputs,
answer-copy decoder or labels in inference. The optimizer's sparse class
background subtraction uses MF2. Ordinary vocabulary-wide CE is evaluated at
fixed scales 1 and 10; those are separate arms, not a jointly trained scale.

The exact-real oracle theorem is distinct from the native floating computation.
Both candidate costs use f64 and the comparison has no tolerance. Integer
parameter counts enforce exact combinatorial feasibility; reported logits use
f64. A dense independent gradient/oracle check covers the fresh numerical gap.
No general convergence theorem for stale gradients is asserted.

## Protocol and checks

The frozen protocol covers **all 35 tasks**, three arms, two fixed scales and
seed zero: 210 cells. Diagnostic caps are 64 full-batch and 512 minibatch updates,
with the original source batch sizes and observation cadence capped at an epoch.
Original splits, source data fingerprints, primary exact sequence accuracy,
own-output generation, EOS and field pooling remain unchanged. Training
generation includes a separate `teacher_forced` report. A first primary score of
at least 99% stops the run; full batch also stops at fresh empirical gap `1e-8`.
These caps are not the original full-budget benchmark. No fresh transformer or
matched timing comparison is claimed.

Six checks cover the query-specific capacity repair, full-batch missing-pair
transfer, singleton memory, an independent dense unrestricted CE gradient and
oracle, integer joint feasibility and saved-model predictions, forward batch
equivalence, and explicit failure on contradictory labels. All six pass.

```sh
cargo test --offline --manifest-path experiments/convex_stress_adaptive/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_adaptive/Cargo.toml -- --check
cargo clippy --offline --all-targets --manifest-path experiments/convex_stress_adaptive/Cargo.toml -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_adaptive/Cargo.toml
bash experiments/convex_stress_adaptive/run_pilot.sh experiments/convex_stress_adaptive/runs/pilot-v1 4
bash experiments/convex_stress_repair/report.sh experiments/convex_stress_adaptive/runs/pilot-v1 experiments/convex_stress_adaptive/protocol.json
```

The runner refuses to overwrite cells and replays every model or explicit
failure. The report accounts for every planned identity, including execution
errors and pending cells, and rejects a missing or duplicated planned cell.
Generated evidence and frozen binaries remain ignored in root `runs/` and
`.kanban/artifacts/160/`. No Python is executed.

## Findings

All 210 planned cells complete and replay at source `f7108742`, fingerprint
`5a35847751311093af3259b7070362d60304cafedfb0241e9fbf2b82c311c3c2`.
There are 198 saved models, 12 explicit forward-only fit failures, and no
execution or replay errors. Primary passes mean at least 99% exact accuracy on
the original primary split; all-split passes additionally require that threshold
on every reported nonempty split, including training and OOD splits.

| Mode | Primary passes, scale 1 / 10 | All-split passes, scale 1 / 10 |
| --- | --- | --- |
| Full batch | 6/35 / 7/35 | 5/35 / 6/35 |
| Minibatch | 5/35 / 7/35 | 3/35 / 5/35 |
| Forward only | 6/35 / 6/35 | 5/35 / 5/35 |

The old full-budget convex baselines passed 3/35 tasks per arm and seed. Both
scales of full batch and forward-only now pass the two AND tasks, both recall
tasks, linear transport and associative-divisor retrieval. Scale-10 full batch
and minibatches additionally pass branch generation. Scale-1 minibatches miss
associative retrieval, scoring 53.866% despite 100% training accuracy; the
separate [completion follow-up](../convex_stress_completion) repairs this case.
No old primary pass is lost. The budgets and fixed scales differ, so these are
paired accuracy diagnostics, not matched training-cost comparisons.

Full-batch short recall improves from 86.13% to 100%; long recall from 50% to
100%; associative retrieval from 53.87% to 100%. Each reaches the primary
threshold after one update at either scale. At scale 10, branch generation
reaches 100% training and primary exact sequences at full-batch update 11 or
minibatch update 32, with fresh gaps still 0.8883 / 0.9335. These prediction
passes are not claims of empirical convergence. All recorded endpoints and
gaps, including failures, are retained in `runs/pilot-v1/summary.json`.

Ordinary division remains at 0.9450% held-out accuracy, and Fibonacci recurrence
at 1.1267%, despite 100% training accuracy and near-zero full-batch gaps at both
scales. Missing labels remain unidentified. Parity-16 at scale 1 has 55.995%
training and 54.14% primary accuracy with a near-zero empirical gap: the bounded
CE model/objective itself needs revision. Scale-10 parity and depth cases retain
nonzero gaps at the cap, so these runs do not establish their best attainable
accuracy. The final scale-10 depth-hard run scores 17.19% with gap 0.1530.

Restoring capacity fixes the shared-only Boolean training regression: training
and the primary split are now both 100%. Position-shift OOD failures remain.
For example, full-batch AND-shift scores 52/64 (81.25%) on `test/position_shift`.
The saved query-18 vertex selects aligned key 71, the completed successor of
the zero token. Early-zero training data make that slot consistent; a late zero
can instead be followed by the delimiter, an unobserved decoder value. This is
a remaining ambiguity between rules that agree on training, not a full Boolean
rule guarantee. The ten earlier forward controls in the completion cohorts are
kept separate from these 70 forward cells.

The best tested all-task configuration therefore passes **7/35 primary tasks**,
not the full suite. Further representation and identification work is required.
Flat spectrum and balancedness remain capacity statements. The latest WD16
classification concerns squared loss on a fully observed identity; it does not
identify the missing CE labels in these stress tasks.

## Independent saved-model CE audit

`src/bin/audit_saved.rs` reconstructs ordinary uncentered full-vocabulary
gradients and an exhaustive query/key/value/class oracle directly from saved
predictions and frozen training data. It does not reuse the optimizer's sparse
background subtraction, stored gradients or oracle. On the scale-1 full-batch
division, parity-16 and recurrence endpoints, and the completed associative
minibatch endpoint, its gap differs from the reported value by less than
`3.5e-13`, and its CE by less than `3.6e-15`. The three failed full-batch cases
have numerical gaps within `4.5e-13` of zero. This is independent f64 validation,
not an interval-arithmetic proof about the executable.

```sh
jq --slurpfile full experiments/convex_stress/runs/stress-v1/summary.json \
  --slurpfile mini experiments/convex_stress_minibatch/runs/minibatch-v1/partial_comparison.json \
  -f experiments/convex_stress_adaptive/compare.jq \
  experiments/convex_stress_adaptive/runs/pilot-v1/summary.json
cargo run --offline --release --bin audit_saved --manifest-path experiments/convex_stress_adaptive/Cargo.toml -- \
  experiments/convex_stress_adaptive/runs/pilot-v1/scale1/full_batch/arithmetic-div/seed0 \
  experiments/convex_stress_adaptive/runs/pilot-v1/scale1/full_batch/basis-parity16/seed0 \
  experiments/convex_stress_adaptive/runs/pilot-v1/scale1/full_batch/recurrence-fibonacci-k1-c2-lgt/seed0 \
  experiments/convex_stress_completion/runs/associative-v1/scale1/minibatch/associative-divisor-n2-l1-w0/seed0
```

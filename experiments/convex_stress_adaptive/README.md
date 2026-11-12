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

The adaptive stress cohort has not yet completed. A full stress-suite pass is
not established. The earlier shared-only pilot remains separate and unchanged.

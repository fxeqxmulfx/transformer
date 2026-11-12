# Complete unobserved query rows in the convex oracle

Question: can the unrestricted fallback transfer a learned value to a missing
query/value row without changing its training objective or losing capacity?

Lean comes first: `ConvexChannel/Completion.lean` proves MF5 in
`review/flat_spectrum.md`. A retained sample-gradient sum is zero on a row absent
from all consumed inputs. Replacing decoder choices only on those rows leaves
the linear cost unchanged and preserves exact-oracle optimality and feasibility.
This says nothing by itself about the correctness of arbitrary unseen labels.

This is a frozen port of `convex_stress_adaptive` at `f7108742`, with one CE
change: before adding a query-specific vertex, fill unobserved values at its
selected aligned key from the pooled oracle's decoder for that same key.
Observed rows, including observed PAD, are never overwritten, even if their
current gradient is zero. Only training prefixes already consumed contribute.
The original oracle costs and routing choices are retained. Decoder completion
is performed before adding the joint vertex counts, not by multiplying two
separately trained factors. Forward-only fitting is unchanged.

Eight native checks pass. The new full-batch witness includes one query with an
exceptional label, so global sharing cannot fit. The original unrestricted
oracle fits every training label with zero gap but mislabels a missing pair.
Completion answers that pair correctly while retaining exactly the same CE and
gap on training. The dense-gradient check, singleton regression, XOR capacity,
snapshot/domain checks and explicit contradictory-label failure are retained.

| Arm | Change from adaptive-v1 |
| --- | --- |
| Full batch | Complete zero-gradient selected-key rows before each unrestricted vertex |
| Minibatch | Same completion from the retained consumed-prefix gradients |
| Forward only | No change; retained as an exact reproduction control |

The prospective protocol fixes four tasks (Boolean AND shift, short recall,
long recall and division), three arms, two scales and seed zero: 24 cells.
Diagnostic caps, original data fingerprints, primary metrics, generation,
EOS, splits and batch sizes are inherited unchanged. This is a targeted
follow-up, not a new complete stress-suite or transformer comparison. The
previous 210-cell adaptive cohort continues with its own immutable binary.

```sh
cargo test --offline --manifest-path experiments/convex_stress_completion/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_completion/Cargo.toml -- --check
cargo clippy --offline --all-targets --manifest-path experiments/convex_stress_completion/Cargo.toml -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_completion/Cargo.toml
bash experiments/convex_stress_completion/run_pilot.sh experiments/convex_stress_completion/runs/pilot-v1 2
bash experiments/convex_stress_repair/report.sh experiments/convex_stress_completion/runs/pilot-v1 experiments/convex_stress_completion/protocol.json
```

All outputs and frozen binaries remain ignored in root `runs/` and
`.kanban/artifacts/160/`. No Python is used. The stress follow-up has not yet
completed; no full-suite success is claimed.

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
`.kanban/artifacts/160/`. No Python is used.

## Completed results and the associative follow-up

All 24 originally planned cells complete and replay at source `8619a891`,
fingerprint `c896fdbd0a2a3026942e49100f2c315dfc78defb452ad31ebc3aac2fd27b0c54`.
All modes and both scales have 100% training and primary accuracy on Boolean
AND shift, short recall and long recall. Ordinary division remains at
0.9450% held-out accuracy despite 100% training accuracy. There are 18 primary
passes out of 24; the Boolean position-shift OOD splits are not all solved.
The eight forward-only saved models, complete split metrics and update counts
are exactly identical to adaptive-v1.

The CE completion reaches the first 100% primary observation sooner in two
paired runs: short recall at scale 10 takes 64 instead of 128 minibatch updates,
and long recall at scale 1 takes 50 instead of 100. Long recall at scale 10
still takes 50. These are update counts under fixed observation cadence, not a
matched timing claim. The final fresh gaps on long recall are 0.1257 / 0.0808,
so a prediction pass is not full empirical convergence. Full-batch recall
still takes one update at either scale.

After the adaptive results, `followup.json` fixed six further cells on the
associative-divisor task, using the **unchanged binary and training protocol**.
This selection was made after observing adaptive-v1's remaining scale-1
minibatch failure; it is separate from the prespecified four-task cohort.
All six cells complete and replay with 100% training and primary accuracy.
Scale-1 minibatches improve from 53.866% at the old 512-update cap to 100% at
update 19; scale 10 takes 19 rather than 38 updates. Full batch still takes
one update. Both forward controls remain exactly identical. This associative
retrieval task is distinct from the unsolved ordinary modular-division task.

Thirty cells therefore complete across the two stated cohorts, with 24 primary
passes, six division failures and no execution or replay errors. The completed
tests support the missing-row repair; they do not establish a complete 35-task
pass, seed robustness or arbitrary-label identification.

```sh
# Original four-task comparison, including all eight unchanged forward controls.
jq --slurpfile adaptive experiments/convex_stress_adaptive/runs/pilot-v1/summary.json \
  -f experiments/convex_stress_completion/compare.jq \
  experiments/convex_stress_completion/runs/pilot-v1/summary.json

# Separately planned associative follow-up; requires a fresh output directory.
bash experiments/convex_stress_completion/run_followup.sh experiments/convex_stress_completion/runs/associative-v1
bash experiments/convex_stress_repair/report.sh experiments/convex_stress_completion/runs/associative-v1 \
  experiments/convex_stress_completion/protocol.json experiments/convex_stress_completion/followup.json
jq --argjson expected 6 --slurpfile adaptive experiments/convex_stress_adaptive/runs/pilot-v1/summary.json \
  -f experiments/convex_stress_completion/compare.jq \
  experiments/convex_stress_completion/runs/associative-v1/summary.json
```

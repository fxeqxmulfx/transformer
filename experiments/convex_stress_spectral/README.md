# Convex complex matrices with spectral upper bounds

Does coupling the entries through an operator-norm bound repair held-out pair
predictions where the earlier channel reached a numerical empirical optimum?
This pilot tests three unchanged failures from the 35-task GPT stress suite.
It is a standalone matrix model, not the compact transformer drop-in described
in the older `EXPERIMENT_PLAN.md`, and not a completed all-task replacement.

Lean came first: `SpectralBall/Basic.lean` proves convexity of the constraints
and ordinary probe cross-entropy; `SpectralBall/Recovery.lean` proves every
constrained binary CE minimum recovers XE15's unseen sign. See
`review/flat_spectrum.md`, SB1–SB2. This repairs that counterexample by changing
the constraint, and leaves XE15's claim about nuclear-norm training intact.

The learned coordinates are a complex matrix indexed by the last two tokens.
The class code is `exp(2*pi*i*class/vocab)`; logits are its real pairing with
the selected matrix entry. The model receives no arithmetic operator, inverse,
task name or answer rule. The entire matrix has entries of modulus at most one
and operator norm at most R. Flatness is used only to motivate a spectral
upper bound; no flatness or orthogonality equality is imposed. The original
class IDs already carry an ordering, which the fixed circular code uses.

The pair-only adapter discards earlier context, so this pilot does not test
MQAR or long-context capacity. It does not replace the count channel that
currently passes 10/35 tasks. The proof is for a binary 2x2 example, not a
claim of modular identifiability or finite-step convergence. The subsequent
SB3 theorem in `SpectralBall/Ambiguity.lean` gives two flat 4x4 matrices that
agree on 12 distinct observed entries, cover every row and column, and are
both global empirical CE minima with opposite unseen signs. A tight bound
therefore cannot provide a general missing-label guarantee.

## Frozen prospective pilot

`protocol.json` records all choices before the stress results. Every cell uses
seed 0 and the original train/validation inputs, labels, hashes and metrics.
All cells run to their fixed update cap; first 99% crossings are also retained.

| Factor | Choices |
| --- | --- |
| Tasks | Division, multiplication, addition (non-flat control) |
| Radius R | sqrt(vocab), 2 sqrt(vocab), vocab |
| Full batch | Sum CE gradient, step 1, 128 updates |
| Minibatch | Original batches and short tails; sum batch gradient, step 1/sqrt(1+update/updates_per_epoch), 512 updates |
| Forward only | Four scalar CE evaluations per sample, central differences with h=1e-5, step 1, 128 updates |
| Projection | Dykstra: unit disks and clipped complex SVD; tolerance 1e-9, cap 256 |

The forward-only arm uses no analytic derivative or backpropagation during
training. It is a finite-difference optimizer, different from the earlier
exact consistency learner, and uses four loss evaluations per sample.
The final diagnostic computes an analytic projected-gradient residual for
all arms without modifying their saved models. That residual is not a dual
gap or a certified bound on excess CE.

Each Dykstra endpoint is radially contracted if necessary to enforce both
bounds to floating-point accuracy. Caps, final iterate changes and shrink
factors are retained; no approximate projection is presented as exact. The
small dense SVDs run on CPU LAPACK/OpenBLAS, one BLAS thread per process, with
three independent experiment processes. The implementation uses no Python.

## Reproduce

From the repository root, with Rust and a system OpenBLAS containing LAPACK:

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.SpectralBall
LAKE_ARTIFACT_CACHE=false lake env lean scripts/SpectralBallAxioms.lean
OPENBLAS_NUM_THREADS=1 cargo test --offline --release --manifest-path experiments/convex_stress_spectral/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_spectral/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/convex_stress_spectral/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_spectral/Cargo.toml
experiments/convex_stress_spectral/run_pilot.sh experiments/convex_stress_spectral/runs/pilot-v1
```

The runner saves and replays each model, then verifies the exact 27-cell
inventory. Individual cells use `convex-stress-spectral run ARM TASK RADIUS OUT`;
`replay OUT` validates the source/protocol and model hashes, rebuilt data hashes,
constraints and every split metric. Outputs remain ignored under `runs/`.
The native checks cover complex SVD reconstruction and known clipping, CE
derivatives against forward differences, projection feasibility/idempotence
and variational inequalities, the unseen-sign repair in all three modes,
a loose-radius negative control, and malformed model rejection.

## Findings

All 27 planned cells completed and their saved models replayed exactly.
No cell reached 99% held-out accuracy at any scheduled observation. The
producer was frozen at `0fb3cd04`, after the recovery proof `d6820069`, with
source fingerprint
`93ed3546451248f27d8cf0084781b22b3ef71f5bd8c72ea31f028be4b81980d0`.
The new ambiguity theorem was proved after seeing the pilot failures and
did not change the frozen training program.

Held-out exact accuracy, in percent:

| Task | Radius | Full batch | Minibatch | Forward only |
| --- | --- | ---: | ---: | ---: |
| Division | sqrt(vocab) | 1.8471 | 1.8900 | 1.8471 |
| Division | 2 sqrt(vocab) | 0.9450 | 0.9450 | 0.9450 |
| Division | vocab | 0.9450 | 0.9450 | 0.9450 |
| Multiplication | sqrt(vocab) | 1.9129 | 1.8704 | 1.9129 |
| Multiplication | 2 sqrt(vocab) | 2.0191 | 2.0191 | 2.0191 |
| Multiplication | vocab | 2.0191 | 2.0191 | 2.0191 |
| Addition | sqrt(vocab) | 8.8204 | 6.1637 | 8.8204 |
| Addition | 2 sqrt(vocab) | 0 | 0 | 0 |
| Addition | vocab | 0.9564 | 0.9564 | 0.9564 |

Every division and multiplication cell fits 100% of training examples. With
the tight bound, full-batch projected-gradient residuals are `3.11e-13` and
`3.13e-13`; forward-only residuals are about `1e-9`. Independent saved-model
audits find maximum observed phase errors of `1.50e-14` and `1.60e-14` for
full batch, while mean unseen entry moduli are only 0.128 and 0.140. The
model has fitted the observed phases without recovering the unseen ones.
The finite-difference optimizer reaches the same held-out accuracies as
full batch in all nine task/radius cells. Tight-bound minibatch residuals
remain 0.113 and 0.131, so its endpoints are not claimed optimal.

The addition control exposes another limitation: a radius based on flatness
can exclude a non-flat target. A post-run label audit gives target operator
norm 97 for addition, versus sqrt(97) for division and multiplication. This
audit used train plus validation labels only after training; the radii were
not chosen from those labels. Tight-bound addition fits 4702/4704 training
examples with full batch/forward only, and 4703/4704 with minibatches. Its
residuals remain `1.29e-4` and 2.77 respectively. The intermediate radius fits
all training examples but gives 0% held-out accuracy. These are failures,
including regressions against the loose-radius control.

No projection hit its iteration cap. The largest radial shrink factor was
`1.000000000997`. Five native checks pass, as do all 27 independent CE and
prediction audits. Corrupting a reported training CE is rejected by both
replay and the independent auditor; omitting one result is rejected by the
cohort summarizer. These floating-point checks are not interval proofs.

The operator-bound repair is thus valid for SB2's small case and insufficient
for these stress failures. More iterations of the same full-batch division
objective are not supported by its tiny residual and exact observed phase
fit. A further model needs assumptions or shared structure that distinguish
unseen labels; flatness and optimization accuracy alone do not do that.
The best completed all-35 model remains the earlier count channel at 10/35;
this three-task pilot adds no full-suite pass and is not a new transformer
comparison.

Generated evidence is in `runs/pilot-v1/summary.json` and the per-cell traces,
models, results and replay records. Additional audit evidence and frozen
binaries are retained locally in `.kanban/artifacts/162/`. To regenerate the
independent audits after the pilot:

```sh
producer_fp=$(cat experiments/convex_stress_spectral/runs/pilot-v1/source_fingerprint.txt)
for result in experiments/convex_stress_spectral/runs/pilot-v1/*/result.json; do
  OPENBLAS_NUM_THREADS=1 experiments/convex_stress_spectral/target/release/audit_saved "${result%/result.json}" "$producer_fp"
done > experiments/convex_stress_spectral/runs/pilot-v1/independent-audits.jsonl
```

This auditor recomputes scalar CE and predictions directly with sine/cosine
and exponentials, without the training adapter, logits or loss functions.
It also measures observed phase errors and the post-run target norm; its
SVD-based norms use the same native library as the producer.

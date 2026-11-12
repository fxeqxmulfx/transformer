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
claim of modular identifiability or finite-step convergence.

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

No stress outcomes have yet been collected for this protocol.

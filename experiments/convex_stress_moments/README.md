# Shared probability coordinates and complex spectral bounds

Can coupling every Fourier class moment improve unseen pair predictions where
the single-phase spectral model fitted all observed labels but failed to
generalize? This is a standalone convex table model. It does not replace the
compact GPT architecture or the best completed 35-task count-channel baseline
(10/35), and its pair adapter ignores earlier context.

Lean came first at `14fd2932`. The CM1–CM4 derivation in
`review/flat_spectrum.md` and `CoupledMoments/` proves convexity of the joint
constraints and ordinary CE, full class information in the Fourier coordinates,
inconsistency of some independent moment assignments, the ideal observed CE
bound and exact-fit minimum certificate, and uniform-mixture feasibility.
There is no theorem here that all missing modular labels are determined.

Each pair has one learned probability vector P. Its nonzero Fourier readings
`M_f[i,j] = sum_c P[i,j,c] exp(2*pi*i*f*c/V)` all obey `||M_f||op <= R`.
Logits are P at scale one. The loss is usual log-sum-exp cross-entropy, not
`-log(P_y)`. The real probability parametrization is equivalent to complete,
coupled complex Fourier coordinates. No product of learned factors or
orthogonality equality appears. The codes depend only on vocabulary IDs;
the model does not receive an arithmetic operator, inverse or task name.

## Frozen prospective pilot

`protocol.json` fixes 18 cells before stress training. All use the unchanged
original data and split hashes, seed zero, and the original complete-split
metrics and cadence. Radii are chosen without viewing missing answers.

| Factor | Choices |
| --- | --- |
| Task | Division, multiplication, addition control |
| Radius | sqrt(V), V |
| Full batch | Sum CE gradient, step 1, cap 512 |
| Minibatch | Original shuffled batches and tails, retained per-observation gradient, step 1, cap 2048 |
| Forward only | Observed one-hot feasibility constraints, cap 512 |
| Stopping | Cap or feasible mean CE within 1e-8 of the ideal observed CE; never held-out accuracy |

Let A be the product of simplexes and B the moment norm constraints. Full
batch uses `x=proj_A(z)`, `y=proj_B(2x-z-grad CE(x))`, `z += y-x`. The
minibatch arm replaces only the selected observations' gradients in a retained
aggregate. It is incremental aggregated gradient, not unbiased SGD. Unselected
labels cannot affect its first updates. Forward-only ingests the observed labels
as retained one-hot constraints in A and uses `y=proj_B(2x-z)` without any
derivative or backpropagation. The projections repeatedly reuse these constraints.
An exact feasible fit would attain the same global CE minimum by CM2.

The simplex shadow may not yet satisfy the spectral constraints. Every reported
and saved model therefore uses a common mixture with uniform mass as in CM4,
with a 2e-12 norm safety factor. Positive mixing preserves class comparisons in
exact arithmetic. Reported accuracy is measured on the actual floating-point
mixture. We retain alpha, every frequency norm, observed fitting defect, actual
CE excess and the full-gradient splitting residual (or feasibility residual
for forward-only). The residual is not a dual gap. CM2 bounds excess CE above
the constrained optimum by the measured ideal-loss excess for a feasible output;
the numerical estimates are not interval certificates. No finite-step
convergence theorem is claimed for these implementations.

The saved splitting point reconstructs the reported model and endpoint residual.
It is diagnostic state, not a resumable minibatch checkpoint: stale gradient
memory is deliberately not included. Every first 99% observation is retained,
even though it is not a stopping criterion. Exact CE fitting need not uniquely
select a completion, so stopping at the ideal-loss tolerance can leave different
unseen predictions between optimizers.

## Reproduce

Use Rust, system FFTW3 (`libfftw3.so.3`) and OpenBLAS with LAPACK. The small dense
complex SVDs run on CPU, with eight frequency workers and one BLAS thread per
process; the launcher runs three independent cells at once. A synthetic n=97
timing, without stress labels, took 2.27 seconds for ten moment projections.
No Python is used.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.CoupledMoments
LAKE_ARTIFACT_CACHE=false lake env lean scripts/CoupledMomentsAxioms.lean
OPENBLAS_NUM_THREADS=1 cargo test --offline --release --manifest-path experiments/convex_stress_moments/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_moments/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/convex_stress_moments/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_moments/Cargo.toml
bash experiments/convex_stress_moments/run_pilot.sh experiments/convex_stress_moments/runs/pilot-v1
```

Seven native tests cover ordinary CE derivatives by scalar differences, direct
DFT agreement and Parseval, simplex and spectral projection variational checks,
uniform-mixture feasibility, all three modes recovering SB2's binary unseen sign,
unselected-label independence, loose-bound negative controls and malformed
models. `replay OUT` checks producer/protocol and data/model/state hashes,
reconstructs the saved output, recomputes every split metric, all frequency norms,
CE excess and the endpoint full-operator residual. The summarizer requires the
exact planned inventory. Generated models, traces, summaries and binaries stay
ignored and are preserved outside the temporary worktree.

FFTW layout and serialization follow the primary
[advanced transform API](https://www.fftw.org/fftw3_doc/Advanced-Complex-DFTs.html)
and [thread-safety rules](https://www.fftw.org/fftw3_doc/Thread-safety.html).
The DFT uses a positive exponent; only nonzero frequencies are constrained.
The DC mass remains fixed by the simplex. Singular-value clipping is borrowed
from the earlier native spectral pilot and operates on column-major matrices.

## Findings

Stress results are pending. Small-case correctness and the Lean certificates
alone are not evidence of passing the three stress controls or all 35 tasks.

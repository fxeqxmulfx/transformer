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

All 18 planned cells completed and replayed; all 18 independent saved-table
audits pass. The producer was frozen at `2de90819`, after the Lean proof
`14fd2932`, with source fingerprint
`086218be9319a99a37b2e6549eff30f74fd7f1641ce72e1c803f222dbd71ca68`.
Six cells pass the original 99% primary threshold: division and multiplication
with the tight radius in each of the three modes.

Held-out exact accuracy, in percent:

| Task | Radius | Full batch | Minibatch | Forward only |
| --- | --- | ---: | ---: | ---: |
| Division | sqrt(V) | 100 | 99.9785 | 100 |
| Multiplication | sqrt(V) | 100 | 99.9575 | 100 |
| Addition | sqrt(V) | 0 | 0 | 0 |
| Division | V | 0.9450 | 0.9450 | 0.9450 |
| Multiplication | V | 2.0191 | 2.0191 | 2.0191 |
| Addition | V | 0 | 0.9564 | 0 |

Full batch and forward-only reach 100% on division/multiplication at the first
scheduled observation, update 19, and stop there on the training certificate.
The minibatch arms first cross 99% at update 38 and attain the training
certificate at update 57 (three epochs). They retain one wrong answer among
4656 division examples and two among 4705 multiplication examples. These are
not rounded to perfect accuracy. Full batch took approximately 6.9/6.8 seconds,
minibatches 18.2/17.9 seconds and forward-only 6.6/6.4 seconds in the concurrent
pilot. Update counts and exposure differ between modes; this is not a matched
transformer timing comparison.

The successful tight-bound cells have exactly one-hot observed coordinates,
computed moment norms strictly below sqrt(97), and mixture alpha one. Their
mean train CE is 3.592270155512, the one-hot minimum for these bounded logits;
it is not expected to be near zero. Reported ideal-loss excess near -3.5e-13
is summation roundoff. The independent audit uses plain exponentials and
direct Fourier sums instead of the training loss, transform or adapter.
It confirms minimum held-out logit margins of 0.17987/0.16574 for full batch
and 0.18528/0.17181 for forward-only. These correct predictions are well clear
of numerical ties. The audit's class probabilities are the latent P coordinates,
not softmax probabilities of the logits.

The observed one-hot fit and feasible output provide the numerical version of
CM2's loss certificate even while the internal splitting point is still moving:
full-batch and forward-only splitting residuals are about 1.2 at their stops.
Minibatch residuals are about 3.2e-14, yet its three unseen errors remain.
Thus the result does not establish unique completion or universal unseen-label
recovery. The scalar CE excess is a bound on objective suboptimality conditional
on actual feasibility, not a bound on classification error.

The addition failure remains. With the tight bound its true phase matrices
have operator norm 97 rather than sqrt(97), as established by the previous
pilot's post-run audit. Full batch and minibatches reach 100% training accuracy
but retain ideal-CE excess 0.771389 and 0% held-out accuracy; their endpoint
residuals are 3.39e-5 and 4.69e-14. Forward-only cannot attain its requested
one-hot fit, has residual 38.47, and needs alpha 0.20297 for a feasible output.
All three reach their caps. The loose-radius exact fits also fail on addition.
This is a regression from the earlier single-phase tight-bound control's
8.82% full-batch accuracy, though neither passes the task.

Coupling repairs the earlier full-batch division/multiplication failures
(1.85%/1.91% to 100%/100%) on these frozen splits. Flatness-inspired bounds
still need the appropriate geometry and radius. No fresh all-35 run or matched
transformer comparison was made; the best completed all-task model remains
the count channel at 10/35. Earlier short/long results remain separate.

Generated evidence lives in `runs/pilot-v1/summary.json`,
`runs/pilot-v1/independent-audits.jsonl` and the per-cell files. Frozen binaries,
native checks, Lean/index audits and corruption controls are preserved locally
in `.kanban/artifacts/163/`. Reproduce the independent audits with:

```sh
producer_fp=$(cat experiments/convex_stress_moments/runs/pilot-v1/source_fingerprint.txt)
for result in experiments/convex_stress_moments/runs/pilot-v1/*/result.json; do
  OPENBLAS_NUM_THREADS=1 experiments/convex_stress_moments/target/release/audit_saved "${result%/result.json}" "$producer_fp"
done > experiments/convex_stress_moments/runs/pilot-v1/independent-audits.jsonl
```

The auditor recomputes every arithmetic split metric directly from raw saved
coordinates and constructs every nonzero Fourier moment with sine/cosine sums.
Its SVD routine is shared with the producer, so this is independent transform
and prediction verification, not an interval proof of spectral norms. Increasing
reported train CE by 0.01 is rejected by both replay and the independent auditor.
Changing the saved splitting point after updating its hash is rejected by
state replay. Removing one result is rejected by the cohort summarizer.

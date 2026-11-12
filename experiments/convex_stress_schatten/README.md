# Two convex spectral budgets for shared complex moments

Can a bound on the sum of singular values repair addition while preserving the
coupled-moment model's division and multiplication successes? The previous
operator bound excluded the one-hot addition table at R=sqrt(V); increasing R
alone also failed. This experiment keeps the shared Fourier probability model
and ordinary cross-entropy, and adds a nuclear-norm upper bound. It uses fixed
vocabulary codes and the last two input tokens, without arithmetic answer rules.

Lean CM5–CM7 at `9d98a603`, in `review/flat_spectrum.md` and
`CoupledMoments/Schatten/` precedes stress training. It proves convexity, uniform
feasibility, a common-mixture feasibility repair, and the observed CE minimum
certificate. A concrete 2x2 theorem derives the unseen rank-one answer from
three observations, a nuclear budget, and attainment of the actual CE lower
bound. It does not assume correct observed answers or a rank equality. The
full-rank control's additional nuclear budget is proved redundant.

## Frozen prospective protocol

All cells use the original frozen data and split hashes, seed zero, complete
split metrics and observation cadence. Logits are the shared probability
coordinates P at scale one; the loss is log-sum-exp CE, not `-log(P_y)`.
For every nonzero Fourier frequency, `M_f[i,j] = sum_c P[i,j,c] exp(2*pi*i*f*c/V)`.
The two generic budget choices depend only on vocabulary size:

| Factor | Choices |
| --- | --- |
| Tasks | Original addition, division and multiplication, V=97 |
| `rank_one` | `||M_f||op <= V`, `||M_f||nuc <= V` |
| `full_rank` | `||M_f||op <= sqrt(V)`, `||M_f||nuc <= V sqrt(V)` |
| Full batch | Ordinary sum-CE gradient, step 1, cap 128 |
| Minibatch | Original shuffled batches/tails, retained gradient memory, step 1, cap 512 |
| Forward only | Retained observed one-hot constraints, no gradients, cap 128 |
| Stop | Scheduled positive update with feasible mean CE excess <=1e-8, or cap |
| Selection | First candidate attaining the training-only tolerance in fixed order `rank_one, full_rank`; otherwise lowest training CE |

These caps are diagnostic and shorter than the earlier coupled-moment pilot.
Successful full-rank controls should still stop at the same observed-CE
certificate, permitting direct comparison of all saved coordinates. There are
18 planned cells and nine selections. No held-out accuracy is used to stop or
choose a budget; all failed budgets and their computation remain in the cohort.

The common model class is the convex hull of the two fixed feasible sets in
direct probability coordinates. CE remains convex on it. Lean proves that an
exact feasible observed fit in either constituent is a global minimum over
this hull. Training both candidates searches for that certificate. When neither
attains the lower bound, selecting the lower loss is not a general solver of
the hull problem. The numerical tolerance and SVD estimates are not interval
proofs, and objective optimality alone does not imply unseen-label recovery.

The splitting solvers are ported from `convex_stress_moments` at `2de90819`.
Full batch uses `x=proj_A(z)`, `y=proj_B(2x-z-grad CE(x))`, `z+=y-x`.
Minibatches refresh only selected observations in the retained gradient sum;
this is incremental aggregated gradient, not unbiased SGD. Forward-only puts
observed labels into A and omits gradients; it repeatedly reuses the cached
constraints after their first ingestion. No finite-step convergence theorem
is claimed. Endpoint splitting state supports diagnostics, not minibatch resume.

The moment projection now clips singular values to R and, if necessary,
subtracts a common nonnegative shift until their clipped sum meets B. The
shift uses 64 bisections. The simplex shadow is mixed with uniform mass to
enforce both budgets using one positive coefficient and a 2e-12 safety factor.
Every saved model reports actual norms, mixture coefficient, observed fitting
defect, CE excess and full-operator residual. The residual is not a dual gap.

## Reproduce and verify

Use Rust with system FFTW3 and OpenBLAS/LAPACK. Eight frequency workers and one
BLAS thread handle the small complex SVDs on CPU; three independent experiment
processes run at once. No Python is used. Generated outputs stay ignored.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
LAKE_ARTIFACT_CACHE=false lake env lean scripts/SchattenMomentsAxioms.lean
OPENBLAS_NUM_THREADS=1 cargo test --offline --release --manifest-path experiments/convex_stress_schatten/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_schatten/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/convex_stress_schatten/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_schatten/Cargo.toml
bash experiments/convex_stress_schatten/run_pilot.sh experiments/convex_stress_schatten/runs/pilot-v1
```

Eight native checks cover scalar and matrix projection variational properties,
both budget violations, the unchanged operator projection under a redundant
sum budget, rank-one and full-rank binary recovery in every training mode,
ordinary CE derivatives, unselected-label independence, and training-only
selection. The unrestricted control contains reflected solver points as well
as all probability tables, so an unseen cell stays uniform. A bound containing
only probability tables need not keep intermediate projections inactive; the
original control incorrectly assumed that it would.

`replay OUT` recomputes saved metrics, both budgets, hashes, exposure and the
model reconstructed from diagnostic state. The summarizer requires all planned
cells. `select COHORT` writes `hull.json`; `replay-hull COHORT` recomputes the
training decision and subsequent evaluation. `audit_saved OUT FINGERPRINT`
independently uses raw coordinates, plain exponentials and direct Fourier sums;
it shares only the SVD routine for the numerical norm check. The `compare` mode
requires exact equality of saved coordinates with the pinned prior full-rank
controls, rather than just agreement of rounded accuracy.

```sh
schatten_fp=$(cat experiments/convex_stress_schatten/runs/pilot-v1/source_fingerprint.txt)
for result in experiments/convex_stress_schatten/runs/pilot-v1/*/result.json; do
  OPENBLAS_NUM_THREADS=1 experiments/convex_stress_schatten/target/release/audit_saved "${result%/result.json}" "$schatten_fp"
done > experiments/convex_stress_schatten/runs/pilot-v1/independent-audits.jsonl
```

## Findings

Pending the frozen pilot. This pair adapter does not yet provide a fresh
35-task run or matched transformer comparison. The best completed all-task
model remains the count channel at 10/35; those results are separate.

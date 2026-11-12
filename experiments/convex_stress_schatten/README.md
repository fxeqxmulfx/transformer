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

The producer was frozen at `deef98cb`, after the Lean proof `9d98a603`, with
source fingerprint
`913357a9c11a6f700148dd73bb60d23e8e3f00e601bfb6b24aea5825c62ce742`.
All 18 planned cells completed, replayed and passed independent raw-coordinate
audits. All nine training-only hull choices select the successful budget below
and pass selection replay. All six full-rank division/multiplication outputs
match every saved probability coordinate of the earlier pilot exactly.
The rank-one budget repairs addition in every mode, while the original
full-rank budget preserves the division/multiplication successes.

Held-out exact accuracy, in percent, for the successful budgets:

| Task | Budget | Full batch | Minibatch | Forward only |
| --- | --- | ---: | ---: | ---: |
| Addition | rank_one | 100 | 100 | 100 |
| Division | full_rank | 100 | 99.9785 | 100 |
| Multiplication | full_rank | 100 | 99.9575 | 100 |

All nine successful runs meet the training-only ideal-CE tolerance and exceed
the original 99% primary threshold. The minibatch division/multiplication
errors remain one out of 4656 and two out of 4705; these are not perfect
completions. All additions first reach 100% at update 19. Their training-loss
stops occur at updates 114 / 57 / 19 for full batch / minibatch / forward-only,
with ideal-CE excesses `8.87e-9 / 8.03e-10 / 2.03e-10`. The numerical norm and
loss tests are tolerance-based, not exact formal certificates. Addition's
full-operator residuals are `1.29e-6 / 7.00e-7 / 2.43e-7`.

All nine opposite-budget controls score **0%** held-out accuracy and hit their
caps: full-rank addition and rank-one division/multiplication. Their endpoint
ideal-CE excesses are between 0.771389 and 0.819061. These failures are retained,
not replaced by selected results. The training selector separates the budgets
without reading their held-out scores. Selection runs both candidates: total
updates are 242 for full-batch addition, 147 for full-batch division or
multiplication, 569 for each minibatch choice, and 147 for each forward-only
choice. The successful candidate's runtime alone is not the selection cost.

Independent direct Fourier sums confirm both spectral bounds, and plain CE
recomputations confirm all reported metrics. Addition's minimum held-out
logit margin is above 0.9999992 in all modes, well clear of numerical ties.
Its latent target-coordinate mean is above 0.99999998; these coordinates are
the logits P, not their softmax probabilities. The existing full-batch and
forward-only division/multiplication margins stay above 0.1657. Minibatch
errors have negative margins down to -0.02372 and -0.04294, so they are not
rounding ties that can be discarded.

The prior addition controls scored zero with the operator-only bounds. The
repair changes the convex spectral geometry, not the output labels, Fourier
codes or cross-entropy. Division/multiplication still use the same feasible
set as before, by the redundant-budget theorem. Full batch and forward-only
stop at update 19; minibatches stop at 57, as in the previous pilot. An exact
observed fit can still leave multiple feasible unseen answers; the retained
minibatch errors prevent a claim of universal or unique completion.

This pair adapter does not yet provide a fresh 35-task run or matched
transformer comparison. The best completed all-task model remains the count
channel at 10/35; those results are separate. The sources of the new gain are
coupled class moments and spectral constraints suited to different shapes.
There is no demonstrated repair here for arbitrary unseen labels or the
remaining sequence-task failures.

Generated summaries, choices, audits, models and traces live in
`runs/pilot-v1/`. Frozen binaries, hashes, Lean/native logs and corruption
controls are preserved outside the temporary worktree in
`.kanban/artifacts/164/`. The auditor rejects modified reported CE and nuclear
norms; replay rejects a changed splitting state even after rehashing it, a
changed hull selection, and a missing planned result. Reproduce the complete
saved-output verification without training:

```sh
bash experiments/convex_stress_schatten/audit_cohort.sh \
  experiments/convex_stress_schatten/runs/pilot-v1 \
  experiments/convex_stress_moments/runs/pilot-v1 \
  experiments/convex_stress_schatten/runs/pilot-v1-controls
```

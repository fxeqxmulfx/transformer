# Intermediate spectral budgets

Can fixed intermediate operator and nuclear bounds repair the endpoint failures
without changing ordinary cross-entropy, the model's inputs or its solver?
Lean CM9 precedes this pilot at `efaf0d93`. It proves a four-by-four rank-two
binary moment is excluded by both old endpoint budgets and by their convex
hull, and admitted by an intermediate budget. A general extreme-point theorem
explains why mixing old probability tables cannot create a missing one-hot
table. This capacity result does not prove unseen recovery on a stress task.

The native predictor imports the same shared probability table, all nonzero
Fourier class moments, loss, projection and solver from `deef98cb`. Only the
fixed budgets below change. No arithmetic operation or target-specific feature
enters the predictor. The nuclear bound `V sqrt(ceil(V/k))` is motivated by the
rank/Frobenius inequality for unit-modulus square matrices; the model does not
enforce or learn a rank equality, or assume every target has that rank.

| Candidate | Operator bound R | Nuclear bound B |
| --- | --- | --- |
| `rank_one` | V | V |
| `full_rank` | sqrt(V) | V sqrt(V) |
| `shape_2` | min(V,sqrt(2 V)) | V sqrt(ceil(V/2)) |
| `shape_3` | min(V,sqrt(3 V)) | V sqrt(ceil(V/3)) |

The protocol fixes six original tasks, four budgets and three modes: 72
candidate records and 18 training-only choices. The first three tasks diagnose
observed failures: power-division mod 97 with exponents two and three, and
square-times. Addition, division and multiplication are retained controls.
This follow-up was designed after seeing failures in the separately frozen
all-35 cohort; it is not a claim of prospective performance on unseen tasks.
The original full-suite run and its selection remain unchanged.

Full-batch uses the ordinary sum-CE gradient. Minibatches refresh retained
per-observation gradients, with the original shuffle, batches and tails;
this is not unbiased SGD. Forward-only retains observed one-hot constraints
and takes no training derivatives. The caps remain 128 / 512 / 128 updates.
Stopping uses only scheduled observed ideal-CE excess at most `1e-8`, or the
cap. All failed budgets, endpoint metrics and costs remain in the results.

After every candidate finishes, the selector recomputes training CE and takes
the first ideal certificate in the table's order, otherwise the minimum CE
without a hull-optimality certificate. Only then are selected split metrics
evaluated. The earlier CM5/CM7 convexity and exact-fit CE certificate apply to
any fixed budgets. Numerical tolerance is not an interval proof or unique
completion theorem. All-split metrics retain the original benchmark definition.

## Reproduce

Use Rust with FFTW3 and OpenBLAS/LAPACK. The default two independent experiment
processes each use eight frequency workers and one BLAS thread. No Python,
cloud job or extra agent is needed. Generated outputs remain ignored.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
LAKE_ARTIFACT_CACHE=false lake env lean scripts/SchattenMomentsAxioms.lean
OPENBLAS_NUM_THREADS=1 cargo test --offline --release --manifest-path experiments/convex_stress_budgets/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_budgets/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/convex_stress_budgets/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_budgets/Cargo.toml --bins
experiments/convex_stress_budgets/target/release/convex-stress-budgets prepare
bash experiments/convex_stress_budgets/run_pilot.sh experiments/convex_stress_budgets/runs/pilot-v1 2
OPENBLAS_NUM_THREADS=1 experiments/convex_stress_budgets/target/release/audit_saved CANDIDATE_DIRECTORY PRODUCER_FINGERPRINT
experiments/convex_stress_budgets/target/release/audit_saved compare OLD_SCHATTEN_DIRECTORY NEW_DIRECTORY PRODUCER_FINGERPRINT
```

Native checks compare the real SVD with the Lean witness, check vocabulary
boundaries and old budget controls, and fit a repeated-column fixture in all
three modes with exact saved-state replay. Preparation checks all frozen data
hashes and unique pair layouts. The saved-table auditor independently reads raw
coordinates, predicts labels, computes plain-exponential CE and direct Fourier
sums; only LAPACK SVD and data construction are shared. The comparison command
requires every coordinate to match the old Schatten control, for both endpoint
budgets. All 18 original control endpoints must be compared, including failures.

## Findings

All 72 candidate records completed, replayed and passed the independent raw-table
audit. All 18 original endpoint-budget controls match every probability
coordinate exactly. All three native checks, source/protocol inventory checks
and the replay of all 18 training-only selections pass. The frozen producer
is `2df45bb9`, fingerprint
`452ab55b80249eef8e88730e3b013ac38679a9743f7e582c0e71a0a5d516222c`.
Generated evidence is local in `runs/pilot-v1` and root `.kanban/artifacts/168`.

Neither intermediate budget passes any of its 18 task/mode tests at the
original 99% threshold: zero passes across all 36 new-budget candidates.
For the three failure diagnostics, the training-only choice is always
`shape_2`, whose training CE reaches the ideal bound despite poor held-out
accuracy. Results below are percentages on the original validation split.

| Diagnostic | Full batch | Minibatch | Forward only |
| --- | ---: | ---: | ---: |
| Power division, p=97, k=2 | 13.4450 | 8.6770 | 13.5309 |
| Power division, p=97, k=3 | 0 | 0.0430 | 0 |
| Square-times | 39.7450 | 6.1637 | 39.7450 |

The old controls remain selected before the new candidates: addition chooses
`rank_one` and scores 100% in all three modes. Division and multiplication
choose `full_rank`, with 100% full/forward and 99.9785% / 99.9575% minibatch
accuracy respectively. Thus each mode still passes only three of six tasks.
No candidate is selected by validation accuracy; the better-looking `shape_3`
endpoints are retained failures, not substituted into the selection results.

The Lean capacity obstruction and its admission by an intermediate budget
remain valid for the stated four-by-four witness. They do not imply recovery
on these datasets. In the k=2 full-batch `shape_2` run, the independent audit
finds exact observed one-hot rows, valid operator/nuclear bounds and a fresh
splitting residual about `3.1e-14`, while unseen accuracy is 13.4450%.
More updates alone do not address that observed ambiguity. Lean CM10 then
formalizes a feasible uncertain global CE optimum, motivating the separate
[fixed-sharing follow-up](../convex_stress_sharing) without altering this cohort.
Neither this pilot nor its training certificate establishes an all-35 pass
or a fresh matched transformer comparison.

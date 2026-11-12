# Common convex predictions on the complete short/long suite

Does combining the coupled complex-moment class with the existing causal count
class extend the three-task success to the original 35 stress tasks? The
objective remains ordinary cross-entropy, with full-batch, minibatch and
forward-only training. No task answer rule is supplied to a learned predictor.

Lean comes first at `d7e1e6aa`. CM8 in `review/flat_spectrum.md` and
`CoupledMoments/{ScaledLoss,PredictionHull}.lean` proves that a convex hull of
probability-valued prediction families preserves probability rows and convex
CE at a shared fixed scale. Fixed causal dictionaries and pair lookups map
the previous models into these common coordinates. Every old expert remains
a feasible point. Any exact observed one-hot fit is a global CE minimum over
the hull, and the observed ideal-loss excess bounds suboptimality against
every hull competitor. All new declarations have examples where required;
the namespace audit finds no sorry or extra axioms.

This common class is defined directly in prediction coordinates. It does not
multiply freely trained mixture weights and expert parameters. Candidate
search can find a certified feasible point; when no candidate reaches the
bound, choosing the smallest training loss is not a general optimizer of
the hull interior. A small objective excess does not identify unseen labels.

## Frozen full-suite protocol

`protocol.json` fixes all 35 tasks from the original `convex_stress/suite.json`,
all three modes, and the candidate order below: 315 records and 105 selections.
Every unsupported layout and failed fit is retained. Dataset construction
checks the original frozen hashes before training. The new selection never
uses held-out scores, and count-model stopping no longer uses primary accuracy.

| Candidate | Model and eligibility | Full / mini / forward budget |
| --- | --- | --- |
| `rank_one` | Shared P and all nonzero Fourier moments; operator and nuclear budgets V,V; unique two-token training pairs with one final target | 128 / 512 / 128 updates |
| `full_rank` | Same coupled P; operator sqrt(V), redundant nuclear V sqrt(V); same input eligibility | 128 / 512 / 128 updates |
| `counts` | Existing query-aligned causal slots, successor/presence banks and token counts; every task | 64 / 512 updates / one epoch |

The spectral solvers and count learner implementations are imported unchanged
from `deef98cb` and `981a7eca`.
The wrapper changes count stopping to observed CE tolerance, full-batch fresh
FW gap, or cap. Forward counts either returns its exact consistent fit or
records `fit_failure`. Minibatches refresh per-observation gradient memory;
they are incremental aggregated gradients, not unbiased SGD. Forward-only
reuses retained observed constraints and takes no training derivatives.

Pair eligibility reads visible training tokens and supervision positions,
requires distinct pairs, and checks vocabulary size 2..256. It ignores task
names and the values of valid labels. It does not reject special evaluation
conventions such as residue pooling. No examples are deduplicated. All
candidates on an eligible pair dataset use scale one; otherwise the count
model uses scale ten. This is a fixed decision before training. The common
lower bound is `log(exp(scale)+V-1)-scale` per observed prefix.

Training CE always means the ordinary teacher-forced objective over all
supervised prefixes. Complete sequences, own-output rollout, EOS, residue
pooling and numeric metrics retain the original evaluation definitions.
Scale, models and candidate ordering are not selected from these scores.
The existing adapter's context capacity uses declared input/rollout lengths,
including longer split lengths, without reading their labels.

After all candidates finish, the selector recomputes their teacher-forced CE
from saved models. It selects the first candidate within `1e-8` of the shared
ideal CE. Otherwise it chooses the smallest training CE and records absence
of that numerical certificate. With no returned model it records `no_model`.
Every candidate's cost is retained; selected-candidate time alone is not the
cost of the procedure. The certificates are numerical and tolerance-based,
not interval proofs or guarantees of unique completion.

## Reproduce and inspect

Use Rust, system FFTW3 and OpenBLAS/LAPACK. After the memory incident below,
the launcher defaults to one experiment process and accepts at most two,
with eight spectral workers and one BLAS thread each. Every native run has
a hard 16 GiB address-space limit (reduce with `HULL_MEMORY_GIB`) and no core
dump. Generated evidence stays local under ignored `runs/`; the worktree
links that directory to the root checkout. No Python or additional agents run.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.CoupledMoments.PredictionHull
LAKE_ARTIFACT_CACHE=false lake env lean scripts/PredictionHullAxioms.lean
OPENBLAS_NUM_THREADS=1 cargo test --offline --release --manifest-path experiments/convex_stress_hull/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_hull/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/convex_stress_hull/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_hull/Cargo.toml
OPENBLAS_NUM_THREADS=1 experiments/convex_stress_hull/target/release/convex-stress-hull prepare
bash experiments/convex_stress_hull/run_suite.sh experiments/convex_stress_hull/runs/FRESH_COHORT 1
```

Six native checks cover label-independent eligibility, the scaled CE bound,
both integrated learners in all three modes, unseen binary recovery, saved
prediction round trips, training-only selection, and preservation of every
supervised causal prefix. Preparation checks all original data hashes without
training. Replay verifies identities, models, feasibility, all split metrics,
observed CE, spectral splitting-state reconstruction and full-data forward
equivalence. The reported count FW gap is computed by the imported optimizer;
the basic replay checks its recorded stopping criterion and recomputes CE,
but does not independently recompute every count-model gradient/oracle gap.

Two additional saved-model auditors perform those numerical cross-checks:

```sh
OPENBLAS_NUM_THREADS=1 bash experiments/convex_stress_hull/limited.sh 8 experiments/convex_stress_hull/target/release/audit_moments CANDIDATE_DIRECTORY PRODUCER_FINGERPRINT
OPENBLAS_NUM_THREADS=1 bash experiments/convex_stress_hull/limited.sh 8 experiments/convex_stress_hull/target/release/audit_counts CANDIDATE_DIRECTORY PRODUCER_FINGERPRINT
bash experiments/convex_stress_hull/audit_suite.sh COHORT FRESH_AUDIT_DIRECTORY
```

The moment auditor reads raw probability coordinates and independently computes
predictions, ordinary CE and direct Fourier sums, without the producer's FFT,
loss, projection or prediction functions. It checks both spectral budgets and
all split metrics, including native-label and residue-pooled scores separately.
The LAPACK SVD is shared, so its norm checks are not interval certificates.
The count auditor shares the fixed causal adapter and forward map, but computes
uncentered full-vocabulary derivatives and an exhaustive linear oracle without
the optimizer's centered gradients, gradient memory or oracle. It compares
fresh full-data FW gaps for both full-batch and minibatch endpoints, and checks
CE and feasibility for forward-only endpoints. An additional native test checks
residue-pooling alias multiplicities and base shifts. Corrupted recorded CE and
FW-gap controls must be rejected by the corresponding auditors.

The count auditor keeps one query's residuals and only sixteen keys' dense
gradient rows at once. It still exhausts all output classes, values and keys,
including padding, in the same row addition order. Two native checks compare
different block sizes with the former dense reference and reject invalid
records. This changes audit storage, not model capacity, training or the
frozen producer fingerprint. `audit_suite.sh` serializes all completed-cell
audits, imposes an 8 GiB address-space cap, and records peak RSS and exit status.

`summary.json` requires the exact 315-record inventory; `selected.json`
reports all 105 choices, primary passes and passes on every nonempty split.
`replay-selection` recomputes the training decisions and evaluations.
`compare-original OLD NEW` requires equality of every probability coordinate
with a pinned original Schatten control. Historical transformer seed-one
measurements accompany choices under their own label. They are not a fresh
matched-time, seed-matched or parameter-matched comparison: spectral tables
contain V cubed coordinates, and count models use sparse joint tensors.

## Findings

Preparation verifies all 35 original datasets. The input-only check admits
23 tasks to the moment models: 18 with vocabulary 97, four with vocabulary
193, and one with vocabulary 256. This yields 138 planned spectral fits,
105 count fits and 72 explicit ineligible records. These are input-layout
counts, not performance results.

The first suite produced 311 of 315 records before a host OOM on 2026-11-12
at 07:30:05 UTC. The kernel recorded two dense count auditors using about
94 GiB combined; it killed one at 50,920,032 KiB anonymous RSS. Their unbounded
all-query gradient storage caused the incident. Four training jobs subsequently
ended with SIGTERM without endpoint models: minibatch long MQAR counts,
minibatch recurrence with both budgets, and forward-only recurrence full-rank.
Their original traces and a separate interruption manifest are preserved locally
under board artifact #177. No resumable solver state exists for those four jobs.
The 311 completed records are unchanged; no complete 105-choice result or whole
suite pass is claimed from this interrupted cohort.

The replacement blocked auditor completed the saved full-batch long-MQAR
checkpoint (160,000 observed prefixes) under its 8 GiB limit in 78.86 seconds,
with peak RSS 245,352 KiB (240 MiB). Its mean CE matches the producer exactly;
the independent FW gap differs by 8.14e-14. All nine native tests, formatting
and clippy pass, and rebuilding the producer gives the exact frozen binary.
Remaining audit/recovery evidence must retain every incomplete or failed cell.
The previous three-task success and count-only result of 10/35 remain separate;
combining their best reported scores does not establish a whole-suite pass.

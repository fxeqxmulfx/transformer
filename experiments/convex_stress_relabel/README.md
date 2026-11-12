# Class coding before convex spectral completion

Does dependence on the numerical order of class names explain some of the
remaining failures? Lean CM12 comes first at `0c89afcc`: ordinary cross-entropy
is invariant under jointly renaming class coordinates and targets, whereas a
fixed complex-code spectral ball need not be. It proves an exact four-class,
one-code counterexample and a convex hull containing every renamed original
family. A mixture in that hull need not obey the old bounds in one fixed code.

The Rust library implements a categorical proposal for a new fixed class coding.
It reads only training input-pair indices and observed labels. For prime
vocabulary V, each observed rectangle contributes a homogeneous relation among
four class phases in the V-element field. The first shared column anchors each
pair of input rows. Streaming elimination fixes phase zero for class zero and
requires exactly one remaining direction. The proposed code must be a bijection
of the entire vocabulary and admit consistent phase potentials on every
observed edge, including longer cycles. Missing classes, composite vocabularies,
ambiguous directions and inconsistent relations produce explicit rejections.

The finite field describes the **class coding**, not an arithmetic feature of
the input tokens. Phase potentials are discarded after verification. Prediction
reads the existing trained probability table and returns its coordinates in the
original class order. The original full vocabulary remains in softmax and CE.
The proposal has no general correctness guarantee for missing labels.

Once the coding is fixed, the existing CM5/CM9 solver learns one coupled class
distribution per input pair, subject to all nonconstant Fourier operator and
nuclear upper bounds in that coding. The direct hull of these families is convex;
the discrete proposal search is not a certified oracle over that hull. An ideal
observed CE endpoint still certifies the probability-domain lower bound, without
certifying unseen answers. Learning the coding from training data costs every
mode the same proposal pass.

| Frozen factor | Candidates |
| --- | --- |
| Output coding | Training rectangle proposal; original identity control |
| Spectral bounds | Original `rank_one`; original `full_rank` |
| Training | Full batch; incremental aggregated minibatches; forward only |
| Tasks | Permuted addition; addition; subtraction; division; multiplication; sum-cube; sum-fourth-power; absolute difference |

`protocol.json` freezes 96 cells before stress scores. Rejected proposals remain
in the inventory. Selection uses actual original training CE: first an ideal
candidate in coding-then-budget order, otherwise the minimum CE. Existing step
caps, evaluation cadence, original minibatch shuffles and tails are retained.
Held-out labels do not enter proposal, bounds, stopping or candidate selection.
This eight-task follow-up was designed after earlier failures; it is not a new
unseen-family assessment, an all-35 result or a fresh matched transformer test.

## Resources and reproduction

Use Rust with FFTW3 and OpenBLAS/LAPACK. Generated outputs stay ignored and local.
The experiment protocol requires serial numerical jobs and an 8 GiB address-space
limit. A vocabulary cap of 193 and a conservative explicit-buffer estimate of
`32 * 8 * V^3 + 64 MiB` stay below 2 GiB. The runner checks this before dense
allocation. The launcher verifies complete, sealed summaries from both frozen
depth binaries before creating a new cohort. It stops on an incomplete parent,
failed replay or exhausted 16 GiB disk reserve. It neither restarts failed jobs
nor overwrites existing outputs.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.CoupledMoments
LAKE_ARTIFACT_CACHE=false lake env lean scripts/CoupledMomentsAxioms.lean
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo test --offline --release --manifest-path experiments/convex_stress_relabel/Cargo.toml -- --test-threads=1
cargo fmt --manifest-path experiments/convex_stress_relabel/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/convex_stress_relabel/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_relabel/Cargo.toml
bash -n experiments/convex_stress_relabel/run_cohort.sh
```

Freeze the committed release binary and launcher outside the worktree, recording
their SHA-256 hashes, the producer commit and `fingerprint` output. Set
`RELABEL_PRODUCER_COMMIT` and `RELABEL_BINARY` to these saved inputs, and set
`DEPTH_REFERENCE_BINARY` and `DEPTH_RETAINED_BINARY` to the frozen binaries from
`032762e5` and `9da1c677`. Then run:

```sh
bash experiments/convex_stress_relabel/run_cohort.sh experiments/convex_stress_relabel/runs/cohort-v1
```

The default parent directories are the depth experiment's `runs/cohort-v1` and
`runs/retained-v1`. Override them with `DEPTH_REFERENCE_ROOT` and
`DEPTH_RETAINED_ROOT` when relocating preserved evidence. Every cell has a fresh
directory, original-data fingerprints, recomputable coding, saved probability
table, splitting state and SHA-256 digests. Replays regenerate the proposal,
check the saved endpoint against its state, verify exposure counts and stopping,
and compare every original split. The independent raw auditor recomputes CE and
argmax in original class order and forms each Fourier matrix by direct sums,
one frequency at a time. It shares the data generator and LAPACK SVD with the
producer; numerical norm checks are not interval certificates. The summary
requires all 96 results and matching replay seals before making its 24 choices.

## Findings and current status

The complete runner and prediction library pass eight native checks. They cover a
withheld five-class categorical fixture in all three modes, exact saved-state
replay, output-renaming covariance of the proposed code, full-vocabulary CE and
gradient covariance, forward batch independence, malformed maps/observations,
proposal rejection and memory preflight. End-to-end fixture runs also exercise
the independent raw auditor, refusal to overwrite, detection of a renamed
saved map with an updated file digest, rejected proposals without models, and
refusal to summarize incomplete evidence. All three modes predict the withheld
fixture correctly. This is a small correctness test, not a stress-suite result.

The launch gate was exercised while the original depth cohort was incomplete.
Its frozen summarizer returned a missing-outcome error, and the launcher exited
before creating the new cohort directory. The test did not start training.

The runner and auditor are ready; the 96-cell stress execution remains pending
until the original and retained depth cohorts complete. No original-task score
or complete-hull optimization claim is reported for this extension yet. The old
failed models and frozen depth cohorts remain the reference evidence.

## Conditional recovery audit

Lean CM13 at `dade2361` proves a sufficient condition for exact completion. The
observed bipartite graph must be connected and contain a perfect matching; every
coded target moment on the complete table must factor into unit row and column
phases. Under a complete code bank, nuclear budgets at most the input dimension,
valid probability rows and exactly ideal observed CE then force the entire target
table. This strengthens identification under explicit structure; it does not
infer the structure of unseen labels from training consistency.

The independent `graph_audit` Rust crate interprets each saved CM12 endpoint
after the frozen 96-cell cohort finishes. It checks connectivity, constructs and
verifies an observed perfect matching, and checks all observed phase relations
with exact integer arithmetic. For a connected consistent graph it constructs
the unique compatible phase completion using training labels only. It then
compares that completion with saved predictions and original split labels.
Full-table agreement with available ground truth is reported only when every
cell is covered, all labels agree and no duplicate cell has conflicting labels.

The graph conditions, declared budget, numerical CE excess, numerical norm checks
and empirical target-structure check remain separate fields. Floating-point
endpoints are not exact-real Lean certificates. The auditor changes no training,
coding, stopping, prediction or selection. Its five native checks cover missing
cells, output renaming, matching augmentation, disconnected and inconsistent
graphs, and a completion that fits training and the saved model while disagreeing
with an unseen true label. They also check rejection of incomplete or changed
producer summaries, missing cells and output overwrites. No stress-audit result
is available before its parent cohort completes.

Build and run serially with the same 8 GiB process cap. The separate manifest and
source fingerprint preserve the frozen CM12 producer fingerprint. The auditor
requires all 96 parent outcomes and validates every saved result/model digest and
raw-audit replay seal. It bounds model JSON size before loading it, caps vocabulary
at 193, processes one model at a time, and refuses an existing output directory.
An interrupted audit leaves partial records without a completion summary.

```sh
cargo fmt --manifest-path experiments/convex_stress_relabel/graph_audit/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/convex_stress_relabel/graph_audit/Cargo.toml --all-targets -- -D warnings
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo test --offline --release --manifest-path experiments/convex_stress_relabel/graph_audit/Cargo.toml -- --test-threads=1
cargo build --offline --release --manifest-path experiments/convex_stress_relabel/graph_audit/Cargo.toml
bash experiments/convex_stress_hull/limited.sh 8 experiments/convex_stress_relabel/graph_audit/target/release/convex-relabel-graph-audit audit-all experiments/convex_stress_relabel/runs/cohort-v1 experiments/convex_stress_relabel/runs/graph-audit-v1
```

Freeze the committed auditor binary with its SHA-256, producer commit and
`fingerprint` output before running it. Keep generated per-cell records, summary,
resource measurements and process exit status in the preserved ignored runs and
board-artifact directories.

## Constructing the compatible optimum

Lean CM14 at `ae9a35ed` constructs a feasible ordinary-CE minimizer from a full
categorical class bijection and training-compatible phase potentials. It proves
the renamed complete Fourier Gram identity, unit-phase moment factorization,
radius-N/nuclear-budget-N feasibility and exact observed CE optimality. With
connected matching coverage, the nonconstant nuclear bounds identify this
constructed optimum. Unseen true labels can still violate its phase structure.
The tighter `full_rank` operator radius is outside this construction's guarantee.

The separate `phase_start` library implements this candidate. Its input is only
the vocabulary size, original training pair observations and a fixed class
coding. It checks graph connectivity, an observed perfect matching and all phase
equations before allocating a probability table. A failed condition produces an
explicit rejection with no replacement model. The admitted table uses the
existing `rank_one` budgets and unchanged inference in original class order.
It constructs no CE gradient and performs no optimizer updates. Coding discovery
and its additional training-observation pass must both be counted by a runner.

Four native fixture checks verify the withheld categorical answer and minimum
CE, all nonconstant spectra using direct Fourier sums and LAPACK SVD, saved-model
roundtrip, equivariance to independent input/output renamings, explicit graph and
memory rejections, and unchanged predictions when only an unseen true answer is
changed. The phase equations provide a model assumption, not an identification
theorem for arbitrary targets. These checks do not evaluate original stress
tasks or subsequent full-batch/minibatch refinement. The frozen 96-cell producer
and queued audit remain unchanged; a prospective original-data comparison and
independent endpoint audits are still required before reporting a warm-start
stress result.

```sh
cargo fmt --manifest-path experiments/convex_stress_relabel/phase_start/Cargo.toml --check
CARGO_BUILD_JOBS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo test --offline --release --manifest-path experiments/convex_stress_relabel/phase_start/Cargo.toml -- --test-threads=1
CARGO_BUILD_JOBS=1 cargo clippy --offline --manifest-path experiments/convex_stress_relabel/phase_start/Cargo.toml --all-targets -- -D warnings
```

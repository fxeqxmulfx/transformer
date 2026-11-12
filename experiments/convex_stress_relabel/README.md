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

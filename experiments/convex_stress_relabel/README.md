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
`32 * 8 * V^3 + 64 MiB` stay below 2 GiB. The complete launcher must check this
before dense allocation. No stress job may start alongside the original or
retained depth cohort.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.CoupledMoments
LAKE_ARTIFACT_CACHE=false lake env lean scripts/CoupledMomentsAxioms.lean
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo test --offline --release --manifest-path experiments/convex_stress_relabel/Cargo.toml -- --test-threads=1
cargo fmt --manifest-path experiments/convex_stress_relabel/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/convex_stress_relabel/Cargo.toml --all-targets -- -D warnings
```

## Findings and current status

The proposal and prediction library passes six native checks. They cover a
withheld five-class categorical fixture in all three modes, exact saved-state
replay, output-renaming covariance of the proposed code, full-vocabulary CE and
gradient covariance, forward batch independence, malformed maps/observations,
proposal rejection and memory preflight. All three modes predict the withheld
fixture correctly. This is a small correctness test, not a stress-suite result.

The stress runner, independent saved-model audit and 96-cell execution are still
pending. No original-task score or complete-hull optimization claim is reported
for this extension yet. The old failed models and frozen depth cohorts remain
the reference evidence.

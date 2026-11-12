# Convex density extraction from categorical subspaces

Can the convex lift proved in CM21 separate Fourier modes when a supplied
singular subspace is an unknown rotation of the true mode basis?

Lean source: `DensityExtraction` at `1885957b`; manuscript
[`review/flat_spectrum.md`, CM21](../../review/flat_spectrum.md#cm21-convex-density-extraction-in-a-supplied-fourier-subspace).
The numerical input is a centered complex feature matrix `B` with `B*B=n I`
and two categorical row indices. No latent permutation, frequency, mode label
or held-out answer is passed to the extractor.

The variable is a Hermitian positive-semidefinite matrix `M`, with trace one
and affine readings `diag(B M B*)=1`. First maximize the real kernel entry at
the two rows; then maximize its imaginary part on the first optimum face.
There is no rank constraint. A bounded ADMM iteration alternates projection
onto PSD trace-one matrices and the affine readings. PSD projection uses a
Hermitian eigendecomposition and simplex projection of the eigenvalues;
the affine projector uses a normalized row Gram pseudoinverse with a declared
cutoff. The second stage uses the measured first-stage objective as its extra
affine target. This is a finite-precision implementation, not the exact
`LexMaximum` predicate or a proved convergence algorithm.

Every stage stores its matrix, affine state, dual state, row multipliers and
complete scalar trace, including capped/refused attempts. Acceptance checks
affine and splitting residuals, the dual step and the absolute primal-dual
gap. The numerical upper bound is
`lambda_max(C - A* y) + dot(y,b)` for PSD trace-one matrices satisfying `A M=b`.
An independent audit reconstructs these operators by polarization of actual
kernel readings, checks PSD/trace, recomputes the objective and bound, and
checks the returned phase column directly. Full replay repeats both stages.
These floating-point diagnostics are **not** interval or Lean certificates.

| Fixture | Purpose | Finding |
| --- | --- | --- |
| Three modes `1,3,9` modulo 13, complex rotated columns and permuted rows | Recover a character without giving its index to the solver | Both stages accept; the recovered column matches a true character within `1e-6`. |
| Two modes `1,12` modulo 13 | Real scores tie; the imaginary stage must select a mode | The first stage retains a half/half mixture; the second selects the positive-imaginary score. |
| Modulo-two collision | Check the boundary of the separation hypothesis | The PSD trace-one matrix with imaginary off-diagonal entries has constant readings. It must not be assumed diagonal. |
| Centered, re-orthonormalized noisy subspace | Distinguish normalization from the exact-subspace hypothesis | The affine equalities are inconsistent; the extractor explicitly refuses. |
| Capped/invalid input and modified artifacts | Check budgets, replay and refusal paths | Failures remain explicit; changing the matrix, dual, input, trace or audit is rejected. |

Eight native tests cover these cases and the complex eigensolver/coordinate
conventions. The standalone `fixtures-v1` run accepts the rotated case after
46 and 41 iterations; its final diagonal error is `4.11e-9`. The real-tie
case takes 3 and 3 iterations with diagonal error `1.11e-15`. Both artifacts
replay and pass the raw audit. Peak RSS for this fixture run is 7,116 KiB;
the CLI also rejects an attempt to overwrite an existing artifact.
This is an extraction prototype, not an original-task result.
The exact equalities are brittle under subspace estimation error; normalization
alone does not fix that gap. A tolerance relaxation needs its own explicit
convex-domain argument and quantitative limits before being called a repair.

Dimensions are bounded by `n<=193`, rank `<=8`, and 16,384 iterations per
stage. The explicit extraction-buffer forecast is below 80 MiB; launch the
entire native process with the user's 64 GiB limit. Single-thread BLAS keeps
the run deterministic. Generated artifacts stay in the ignored root-linked
`runs/` directory. The frozen earlier producers and all35 short/long,
full/minibatch/forward-only and conventional-transformer comparisons remain
unchanged. Training-only sketch proposals and original-data protocols are
separate follow-up work.

From this directory, using native tools only:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CARGO_BUILD_JOBS=2
prlimit --as=68719476736:68719476736 -- cargo test --locked --offline
cargo fmt -- --check
prlimit --as=68719476736:68719476736 -- cargo clippy --locked --offline --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --locked --offline --bins --examples
prlimit --as=68719476736:68719476736 -- target/release/examples/fixtures runs/fixtures-v1
prlimit --as=68719476736:68719476736 -- target/release/convex-stress-density replay runs/fixtures-v1/rotated.input.json runs/fixtures-v1/rotated.artifact.json
```

`extract INPUT.json NEW_ARTIFACT.json` also accepts externally supplied feature
inputs and refuses an existing destination. Replay compares the artifact with
the separately supplied original input. `examples/fixtures.rs` documents the
input schema and generates reproducible exact-geometry inputs only.

# Approximate spectral recovery on frozen cubic checkpoints

Question: can a quantitative nuclear-norm recovery bound explain the saved
complex-product Rotate90 classifiers for `(a+b)^3 mod 97`?

Protocol fixed before computing diagnostic results, 2026-11-08. No training.
Read all 24 original states from `rotate90_complex/runs/cubic-adamw`: all four
arms, seeds 7 and 11, steps 0, 5000 and 10000. Check immutable state/history and
split hashes. Use the exact source equations at b70811e, including every RMSNorm
and the tied readout. A batched FP64 Rust/OpenBLAS evaluation is checked against
the archived scalar FP64 oracle and the stored GPU correctness counts. It is
not a certified floating-point execution bound. Use at most four BLAS threads
and a five-minute diagnostic budget, separate from compilation. No GPU learning
or optimizer changes are authorized by this diagnostic.

The fixed candidate bank contains all 48 positive-frequency additive characters
`exp(2*pi*i*k*(a+b)/97)` plus the constant. Each complex target table is rank one
and has nuclear norm 97. Train-only ridge regression probes these characters
from the raw residual state with an intercept; relative ridge is fixed at 1e-8
times the mean design Gram diagonal. These analytical probes are never fed into
the network. All frequencies remain in the report, including poor fits.

Fit ideal raw class scores using training-only means grouped by `a+b mod 97`.
All 97 classes compete, including labels absent from the cubic image. The full
real Fourier transform of those means fixes the decoder. Neither probe nor
decoder uses validation values for fitting or selection. Validation labels are
used only to evaluate gaps and accuracy. This is a retrospective diagnostic on
an already examined validation split, not a new unseen test set.

For each split, average eight deterministic perfect matchings contained in the
observation mask to obtain a doubly stochastic matrix `B`. Estimate
`sigma = ||B - J/97||_op` by SVD and use slack `(1-sigma)/2`. Diagonal unitary
phase changes supply the corresponding rank-one character certificates. Check
that every missing cell has a three-edge observed path: tangent matrices have
the row-plus-column form after removing phases, so the four-entry rectangle
identity gives hidden-cell conditioning at most 4 in nuclear norm. The data
correction is the masked feature error. Its nuclear norm is `rho`; the signed
objective excess is `epsilon = ||X||_* - 97`. QR2–QR3 then suggest the bound

```
feature entry error <= 4*(epsilon + rho)/slack + rho.
```

Multiply each feature bound by the modulus of its class-contrast decoder
coefficient and sum. Add the measured source-alignment contrast error. Compare
this with the ideal raw margin, initially granting zero execution error. Also
record the direct measured ideal/source contrast error, the probe errors, and
residual-only accuracy to distinguish failure of the surrogate bound from an
inactive branch. This scalar sum is the componentwise use of QR3; the code does
not claim to instantiate the single-matrix Lean theorem for a different norm.

Every SVD, regression and forward result here is numerical. Positive budgets
would still need certified arithmetic and the matching/conditioning premises
connected to Lean. A nonpositive budget even with zero execution error rejects
this candidate bound for that query; it does not disprove learning, every
possible feature bank, or the exact spectral theorem. Checkpoints are too sparse
to locate the first generalization transition or every later dip.

Run from this directory:

```sh
cargo test --offline --release
cargo run --offline --release -- /path/to/rotate90_complex/runs/cubic-adamw report.json
```

The implementation links system OpenBLAS for FP64 GEMM, Cholesky and complex SVD.
The source oracle in `fixtures/reference.rs` is copied from the native experiment
at b70811e. Model/data formulas are attributed to that same source; no existing
training source or checkpoint is modified.

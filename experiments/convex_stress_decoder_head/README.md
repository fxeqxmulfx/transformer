# Complete complex heads for declared decoder confidence

Can the already learned categorical phase decoders obtain lower ordinary
raw cross-entropy through an explicitly charged complex output head? CM44
at `a8f0020acf63bcd6e48449356818e1ab4f913f8e` proves the actual conjugate
initializer, coefficient budget, fixed-feature CE convexity, and the exact
CE/error identity for arbitrary original record batches.

| Variant | Difference |
| --- | --- |
| Saved categorical source | Frozen learned phase maps and many-to-one raw decoder; original predictor uses margin 1. |
| Confidence head | Materializes the full normalized Fourier code and every real/imaginary conjugate coefficient. Selects a margin from `[1, 2, 4, 8, 16, 24]` using original training CE. |
| Full, mini, forward metadata | Retains the inherited source mode and training information. Each confidence initializer adds zero optimizer updates and zero gradient batches. |

The source maps and original records must match external SHA-256 bindings.
Original record IDs are distinct; repeated inputs and even conflicting raw
targets retain their full multiplicity. The training-only selection uses the
proved identity `log1p((vocabulary-1)*exp(-margin)) + margin*error_rate`.
Ties within `1e-10` keep the earlier bank entry. Every selected head is then
checked using the actual complete complex products and original-record CE.
The full-vocabulary denominator is unchanged.

All coefficients count toward the per-class absolute-sum radius `64*q`.
The inherited latent-moment budget remains a separate source property.
Predictions read a cache computed from the actual complex tensor; loading
rebuilds the code, tensor and cache, and rejects changes to them, the records,
source, selection, endpoint metadata, costs or external bindings. No arithmetic
answer formula or held-out target participates in construction.

The public batch derivative covers every real and imaginary coefficient.
Full batches, repeated inputs at separate record positions and short final
minibatches use their actual record counts. This component exposes the
gradient but does not run a new optimizer. The Fourier convention matches
`convex_stress_kernel/src/model.rs`; the ordinary complete-gradient convention
matches `convex_stress_context/src/gradient.rs`, both read at `a8f0020a`.

Native limits retain `n <= 257`, `q <= 771`, vocabulary at most 1024 and at
most 66,049 original records. A conservative forecast charges source buffers,
the complete code, coefficients, cached logits, gradient and serialization
space. Configurations above 56 GiB of forecast buffers or 32 billion scalar
operations are refused before head allocation. Run all commands under the
64 GiB address-space limit; the largest dimension combination is intentionally
refused by the work cap.

Six Rust tests pass: actual complex products for every declared margin,
many-to-one and repeated phase aliases, correct and wrong-label selection,
complete finite-difference gradients with repeated records and tails,
fixed-feature CE convexity, mutation rejection, endpoint accounting and
bounded-domain refusals. The wrong-label fixture selects margin 2 whereas
its exactly fitted counterpart selects margin 24. These are component tests;
no original stress-suite improvement is claimed here.

Reproduce from this directory with native tooling:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked
cargo fmt --check
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --all-targets -- -D warnings
```

Greater confidence preserves the declared raw winner in exact arithmetic; it
does not identify unseen raw answers and can worsen held-out CE on errors.
Pooled task metrics must be evaluated afresh. Source discovery remains an
upstream discrete procedure, outside the proved fixed-feature head convexity.
An original-data comparison must retain all 35 cases, both initializer
policies, all three modes, unchanged choices, regressions and every baseline.

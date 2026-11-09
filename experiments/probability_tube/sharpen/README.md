# Pairwise encoder bounds

This continuation reduces two sources of conservative amplification in the
full Rotate90 motion bound. It changes the mathematical bound and its native
evaluation; the frozen reproduction and `../audit` sources stay unchanged.

For old gate/value pair amplitudes bounded by G and U, and moving-input
radii rg and ru, the calibrated complex product obeys

```
product_motion <= (U*rg + G*ru + rg*ru)/sqrt(2).
```

The largest old pair replaces the full old vector norm in the linear terms.
The mixed term remains. Matrix amplification uses the smaller of the
Frobenius cap and the existing fourth-root Gram-energy cap. Both bounds are
proved for the source's real matrix maps; no estimated singular value is
assumed exact. Intersect the resulting complete-encoder radius with the
original radius, allowing the two branches to use different certified
operator caps. The same tied-readout and execution-contrast terms then give
an all-rival probability ceiling.

Fourteen Lean theorems in `FeatureMotion/Sharp*.lean` cover pair energies,
finite product motion, the full encoder, executed contrasts, probabilities
and source matrix caps. Each theorem with mathematical hypotheses has a
nonzero example. The [derivation](../../../review/flat_spectrum.md#pairwise-encoder-amplification)
records the assumptions and limits.

Four native tests include the reused independent Gram fixtures, an attained
product bound, original dimensions, source parameter ties, every intermediate
layer, all logit contrasts and probability comparisons. A 32-pair witness
removes a factor sqrt(32) from one linear product term. The original-shape
synthetic perturbation gives a smaller final radius than the previous bound,
below the normalization diameter. These are synthetic checks, not training
findings. Only the Gram calculation expands its FP64 operations outward;
the complete propagation is not a floating-point theorem.

```sh
cargo test --offline --locked --release --manifest-path experiments/probability_tube/sharpen/Cargo.toml
cargo clippy --offline --locked --release --all-targets --manifest-path experiments/probability_tube/sharpen/Cargo.toml -- -D warnings
cargo fmt --manifest-path experiments/probability_tube/sharpen/Cargo.toml --check
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/SharpEncoderAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
LAKE_ARTIFACT_CACHE=false cargo run --offline --locked --release --manifest-path scripts/native_index/Cargo.toml
rg -n 'native_decide|^axiom |linter\..* false|autoImplicit' src
```

The forbidden scan should find nothing (rg exit 1). Logs remain ignored at
the root in `.kanban/artifacts/82/`. No trained checkpoint or future state is
opened by these tests. Measuring useful horizons and bounding full optimizer
potential changes remain separate tasks.

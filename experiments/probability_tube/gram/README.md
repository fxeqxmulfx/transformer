# Query-weighted CE covariance

This library evaluates QG1–QG3 in
[`review/flat_spectrum.md`](../../../review/flat_spectrum.md#query-weighted-feature-covariance).
The Lean modules `GradientEvidence/CEGram` prove the actual unused-class partial
CE coefficient, its finite-motion transport and the varying-batch momentum
condition. This changes the coefficient used for analysis; it does not change
the executed training process.

For each known batch, form `C=T^T diag(q) T` from old executed features and
the prospective per-query ceilings. The full metric-weighted Gram energy is
`sum_kl r_k*r_l*C_kl^2`. Its square root divided by batch size bounds the
weighted CE operator. Intersect with the old weighted feature trace. Transport
this cap by the weighted mean of squared executed-feature radii, then intersect
with `R*M*mean(q)`. The comparison baseline is `R*M*max(q)`.

OpenBLAS forms C with one FP64 matrix product, reused by both metrics. A
`gamma_(2n+2)` allowance covers each preweighted dot product under ordinary
FP64 multiply/add rounding, using `sqrt(d_k*d_l)` as its absolute-sum cap.
Each d is an outward-expanded positive sum `sum_i q_i*T_ik^2`; an additional
`MIN_POSITIVE*(2n+2)*(1+max|T|)` floor covers subnormal arithmetic under the
checked input ranges. Positive reductions, square roots and divisions expand
outward. The full BLAS implementation and the exponential conversion from
GPU probabilities are not formally verified; the numerical upper-bound and
execution premises remain explicit in Lean.

Five independent native tests cover exact dyadic covariance, both actual
source batch sizes, flat covariance versus trace, a finite-difference check
of the true CE derivative with nonzero feature motion, complete source-order
slicing, terminal/inclusive horizon semantics, and invalid inputs. They do
not open trained checkpoints. No training improvement is claimed here.

```sh
cargo test --offline --locked --release --manifest-path experiments/probability_tube/gram/Cargo.toml
cargo clippy --offline --locked --release --all-targets --manifest-path experiments/probability_tube/gram/Cargo.toml -- -D warnings
cargo fmt --manifest-path experiments/probability_tube/gram/Cargo.toml --check
cargo run --offline --locked --release --manifest-path experiments/probability_tube/gram/Cargo.toml --example synthetic_cost
```

The synthetic timing program uses deterministic dyadic 512-by-128 features
and both metrics. Its runtime may inform the new comparison's budget before
that protocol is frozen; it measures no real-training coefficient.

# Complex-product FFNs against the bias-free SwiGLU FFN

Request, 2026-11-20: prove that the FFN of DiffGPT's selected model approximates
better than SiLU, or refute it, and cover the three complex-product variants.
Lean: `Transformer.ComplexFFN` (`src/Transformer/ComplexFFN/`).

## Sources

DiffGPT, `~/python_projects/gpt` at `5a37c51`:

- `research/main_model/README.txt`: the branch `f = R_D(W_F (W_g z ⊙_ℂ W_u z)/√2)`
  with complex-linear projections stored as `[[A,-B],[B,A]]`; its input is the RMS
  output `z'`, and its output is normalized again.
- `research/complex_transformer/README.md`: the radial gate
  `s(g) = sigmoid(sqrt(|g|^2+ε))` and the diagonal and radial products, both with
  complex projections; `research/complex_ffn_suite` uses the radial product with
  real projections.
- `src/native_transformer/src/plain_product.rs`, `complex.rs` (`product::<false>`,
  `product::<true>`, `ε = 2^-23`), `base_ops.rs` (`swiglu`).
- `src/diff_gpt/domain/model/feed_forward.py`: `c_proj(SiLU(gate) * value)`, no biases.

| Variant | Hidden unit | Kernel | Used by |
| --- | --- | --- | --- |
| diagonal | `s(g) (g_R u_R + i g_I u_I)` | `complex.rs`, `product::<false>` | control of `complex_transformer` |
| radial | `s(g) g u / √2` | `complex.rs`, `product::<true>` | `complex_transformer`, `complex_ffn_suite`, `complex_geometry_suite` |
| plain | `g u / √2` | `plain_product.rs` | the selected model |

The Lean blocks (`Defs`) are written unit by unit and take real-linear projections
`V →ₗ[ℝ] ℂ`. Complex linearity is the hypothesis `g (J z) = i g z`, where `J` is
multiplication by `i` on `ℂ^n`. The output `W_F` is any real matrix, given by two
columns per complex unit. The branch RMS is outside the maps; it is odd
(`rmsNormEps_neg`), so every symmetry below survives it.

## Results

FFN1, symmetries (`Basic`). The three products square the sign of their input, so
the three FFNs are even for any projections (`plainFFN_neg`, `radialFFN_neg`,
`diagonalFFN_neg`). With complex-linear projections, plain and radial FFNs satisfy
`f(i z) = -f(z)` (`plainFFN_anti`, `radialFFN_anti`); the radial gate sees only `|g|`.
SwiGLU has neither symmetry: its even part is the product network
`Σ (a_j z)(b_j z) c_j`, its odd part is not zero (`swigluFFN_add_neg`).

FFN2, gaps (`Basic`). If `f ∘ σ = f`, a uniform error `δ` on a set holding `z` and
`σ z` gives `‖T z - T(σ z)‖ ≤ 2δ` (`gap_of_invariant`); if `f ∘ σ = -f`, it gives
`‖T z + T(σ z)‖ ≤ 2δ` (`gap_of_anti`).

FFN3, embedding (`Embedding`). `SiLU(t) - SiLU(-t) = t`, so one real product is two
SwiGLU units exactly (`productFFN_eq_swigluFFN`). Gauss's identity writes a complex
product as three real products (`plainFFN_eq_productFFN`). A plain FFN with `h`
complex units is therefore a bias-free SwiGLU FFN with `6h` units
(`plainFFN_eq_swigluFFN`). With gates `η a` and values `(2/η) b`, one SwiGLU unit is
within `|η| (a z)² |b z| / 2` of the product (`silu_scaled_sub`), so `3h` units
approximate the plain FFN on every bounded set (`swigluFFN_gauss_sub_plainFFN`).

FFN4, separation (`Separation`). One SwiGLU unit `SiLU(ℓ z)(ℓ z) w` changes by
`(ℓ z)² (2σ(ℓ z) - 1) w` under `z ↦ -z`. No plain, radial or diagonal FFN of any
width approximates it with error below `(ℓ z)² |2σ(ℓ z) - 1| ‖w‖ / 2` on a set holding
`±z` (`plainFFN_gap_swigluUnit`, `radialFFN_gap_swigluUnit`,
`diagonalFFN_gap_swigluUnit`); at `ℓ z = 1` that is `tanh(1/2)/2 ≈ 0.231 ‖w‖`. The
four-unit SwiGLU block `|φ z|² w` (`normSq_eq_swigluFFN`) is invariant under `J`;
plain and radial FFNs with complex-linear projections err on it by at least
`|φ z|² ‖w‖` (`plainFFN_gap_normSq`, `radialFFN_gap_normSq`). On the unit circle
the zero map attains this bound.

## Verdict

Refuted for the selected plain product: as a class of functions on the RMS sphere it
lies strictly inside the bias-free SwiGLU class. It equals SwiGLU blocks of six
times its width and misses SwiGLU blocks of width one. Its expressive content is the
holomorphic quadratic maps `f(λ z) = λ² f(z)`.

Parameters: a complex unit has `3D` real parameters (complex-linear rows of `W_g`
and `W_u`, a column of `W_F`, each `D/2` complex entries), as many as one SwiGLU
unit. The exact embedding costs six times the parameters; the limit, three times.
The selected model's FFN at `D = 128` has 171 complex units, `65664` independent
parameters per layer; the SwiGLU control with the same 342 hidden units (`layout.rs`,
`hidden`) has `131328`.

Not claimed: anything about training, depth, initialization or the cubic task.
The results concern one block whose input set is closed under `z ↦ -z` or `z ↦ i z`,
as the RMS sphere is. The plain product's empirical advantage is therefore not one
of approximation power; it is consistent with an inductive bias toward phase-adding
products.

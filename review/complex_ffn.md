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
columns per complex unit. The branch RMS is outside the maps of FFN1–FFN6. It is odd
(`rmsNormEps_neg`), so every symmetry below survives it; the evenness and parallelogram
witnesses do not, and FFN7 replaces them with two-unit witnesses.

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

FFN5, the even part of SwiGLU (`EvenPart`). Without biases the even part of a SwiGLU
FFN is the quadratic form `P/2` of FFN1, and product networks obey the parallelogram
law `P(t(x+y)) + P(t(x-y)) = P x + P y` for `2t² = 1` (`productFFN_parallelogram`).
On an even target `T` and a set closed under negation, a SwiGLU FFN of any width
therefore errs by at least `‖T x + T y - T(t(x+y)) - T(t(x-y))‖ / 4`
(`swigluFFN_gap_even`).

FFN6, incomparability (`Incomparable`). The gate of the radial and diagonal products
breaks that law. With `W_g = W_u = φ`, `W_F = 1`, `φ x = 1` and `φ y = 0`, the defect
of one radial unit is `(s(1) - s(1/√2))/√2` and that of one diagonal unit is
`s(1) - s(1/√2)`, positive for `ε ≥ 0` (`radialGate_inv_sqrt_two_lt`). So no SwiGLU
FFN of any width approximates them with error below `(s(1) - s(1/√2))/(4√2)` and
`(s(1) - s(1/√2))/4` (`swigluFFN_gap_radial`, `swigluFFN_gap_diagonal`); numerically,
about `0.0108` and `0.0153` for `ε = 2^-23`, against unit values near `0.52` and
`0.73`. The four points lie on one sphere when `x ⊥ y` and `‖x‖ = ‖y‖`
(`four_points_mem_sphere`). The complex-linear `φ = fst/r` with `x = r e₁`, `y = r e₂`
in `ℂ²` is such a case, so the bounds hold on the RMS sphere. One plain unit is exactly
`φ₁ z · φ₂ z` (`plainFFN_unit`); one SwiGLU unit errs on it by at least `1` wherever
the product takes every value of modulus one (`swigluUnit_gap_product`), as on the
torus of `ℂ²`, where both units have ten real parameters and the zero map errs by `1`
(`swigluUnit_gap_plainUnit_torus`).

FFN7, after the branch RMS (`RMSGap`, `RMSTwoGate`, `RMSIncomparable`, `RMSSurviving`,
`RMSErased`). With RMS `eps = 0`, `R_D` forgets positive scalars (`rmsNormEps_zero_smul`)
and so erases the evenness witness of FFN4 and the parallelogram witnesses of FFN6. At
`-z` one SwiGLU unit changes by the positive factor `σ(-a z)/σ(a z)`, so after `R_D` it
is even (`rmsNormEps_zero_swigluUnit_neg`). A positive gate is a positive scalar, so one
radial or diagonal unit after `R_D` is its ungated product, a SwiGLU block by FFN3
(`rmsNormEps_zero_radialFFN_unit_swiglu`, `rmsNormEps_zero_diagonalFFN_unit_swiglu`).
With `eps > 0`, as in the model (`2^-23`), the RMS does not forget scale
(`norm_rmsNormEps_lt_smul`). The other witnesses survive. Plain and
radial FFNs with complex-linear projections stay `J`-anti after `R_D`, so at any width
they miss the `J`-invariant four-unit block `R_D(|φ z|² w)` by its norm, as the zero
map does (`plainFFN_rms_gap_normSq`, `radialFFN_rms_gap_normSq`); with `eps = 0` that
norm is `√d` (`norm_rmsNormEps_zero`). One SwiGLU unit after `R_D` stays on the line
through its column, so it misses one plain unit `z₁ z₂` after `R_D` on the torus as much
as the zero map does (`swigluUnit_rms_gap_plainUnit_torus`). For the erased witnesses,
signs suffice: `R_D` multiplies by a nonnegative scalar, so it keeps the sign of every
linear functional (`pos_inner_of_pos_inner_rmsNormEps`), and two units on each side
separate again.

- Signs constrain the even part. If `R_D ∘ S` is within `δ` of an even target `w` with
  `δ ‖ℓ‖ < ⟪ℓ, w z⟫`, the product network `P z = S z + S(-z)` has `⟪ℓ, P z⟫ > 0`
  (`pos_inner_of_rmsNormEps_near`). Product networks satisfy
  `P(αx+βy) - P(αx-βy) = P(βx+αy) - P(βx-αy)` (`productFFN_cross`), so signs
  `+, -, -, +` of `⟪ℓ, w⟫` at these four points with margin `γ` force `δ ≥ γ/‖ℓ‖` at
  any width (`swigluFFN_rms_gap`).
- Two gated units with gates `φ, ψ`, values `ψ, φ`, orthonormal real columns `e, e'`
  and any imaginary columns, where `φ x = ψ y = 1` and `φ y = ψ x = 0`, output
  `κ p q (s(p) e + s(q) e')` at `p x + q y`: `κ = 1/√2` for radial units
  (`radialFFN_twoGate`), `κ = 1` for diagonal ones (`diagonalFFN_twoGate`). Along
  `e - e'` the normalized output has signs `+, -, -, +` for `0 < β < α`. No SwiGLU FFN
  of any width approximates them after `R_D` with error below
  `√d κ α β (s(α) - s(β)) / (√2 √((κ α β)² (s(α)² + s(β)²) + d eps))`
  (`swigluFFN_rms_gap_radial`, `swigluFFN_rms_gap_diagonal`), positive for
  `0 < β < α`, `ε ≥ 0` and `eps ≥ 0` (`radial_rms_bound_pos`, `diagonal_rms_bound_pos`).
- Conversely, the SwiGLU units `SiLU(ρ z)(ρ z) e` and `SiLU(-ρ z)(-ρ z) e'` add up to
  `t² (σ(t) e + σ(-t) e')` with `t = ρ z` (`swigluFFN_pair`); `z ↦ -z` swaps the
  coefficients. Every even map, so every plain, radial or diagonal FFN after `R_D`,
  misses the pair after `R_D` on a set holding `±z` by at least
  `√d t² (σ(t) - σ(-t)) / (√2 √(t⁴ (σ(t)² + σ(-t)²) + d eps))`
  (`even_rms_gap_swigluPair`), positive for `t > 0` (`swigluPair_rms_bound_pos`).

Numerically (awk, not formalized), at `D = 128` with `ε = eps = 2^-23`: for orthogonal
unit vectors `x, y` and `α² + β² = D` the four points lie on the RMS sphere
(`four_points_mem_sphere`).
The zero map errs by about `√D ≈ 11.3` on every target here, since `R_D(0) = 0`. The
two-gate bound is `3.36`, `2.56`, `1.74`, `0.72` for `β = 0.1, 0.5, 1, 2`, that is 30%,
23%, 15% and 6% of the zero map's error, the same for radial and diagonal units: at
`eps = 0` it is `√d (s(α) - s(β)) / (√2 √(s(α)² + s(β)²))`, free of `κ`. The pair bound
is `2.69`, `4.75`, `6.85`, `7.85` for `t = 1/2, 1, 2, 4`, from 24% to 69%.

## Verdict

Refuted for the selected plain product: as a class of functions on the RMS sphere it
lies strictly inside the bias-free SwiGLU class. It equals SwiGLU blocks of six
times its width and misses SwiGLU blocks of width one. After the branch RMS the
inclusion persists and stays strict (FFN7): every plain FFN misses two SwiGLU units,
and with complex-linear projections it does no better than the zero map on the
four-unit block `R_D(|φ z|² w)`. With complex-linear `W_F` as well, a plain FFN is a
holomorphic quadratic map, `f(λ z) = λ² f(z)` for complex `λ`
(not formalized; the theorems use only `λ = -1` and `λ = i`).

Radial and diagonal products: incomparable with SwiGLU. Each misses one unit of the
other at any width: their evenness costs them SwiGLU's odd part, and their gate puts
them outside SwiGLU's quadratic even part. After the branch RMS, with `eps = 0`, these
two arguments give nothing at one unit; with two units of each class they separate the
classes again at any width (FFN7).

Parameters: a complex unit has `3D` real parameters (complex-linear rows of `W_g`
and `W_u`, a column of `W_F`, each `D/2` complex entries), as many as one SwiGLU
unit. The exact embedding costs six times the parameters; the limit, three times.
Before the branch RMS, at one unit each, the plain and SwiGLU classes are incomparable
(FFN4, FFN6); wider
equal-parameter comparisons are not formalized. The selected model's FFN at
`D = 128` has 171 complex units, `65664` independent parameters per layer; the SwiGLU
control with the same 342 hidden units (`layout.rs`, `hidden`) has `131328`.

Not claimed: anything about training, depth, initialization or the cubic task.
The results concern one block whose input set is closed under `z ↦ -z` or `z ↦ i z`,
as the RMS sphere is, and they rely on the blocks having no biases: with biases the
even part of SwiGLU is no longer a quadratic form. The plain product's empirical
advantage is therefore not one of approximation power; it is consistent with an
inductive bias toward phase-adding products.

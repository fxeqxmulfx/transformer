module

public import Transformer.Landscape.Neural.Products
public import Mathlib.Analysis.SpecialFunctions.Sigmoid
public import Mathlib.LinearAlgebra.Complex.Module

/-!
# Complex-product feed-forward blocks and the bias-free SwiGLU block

Source: DiffGPT (`~/python_projects/gpt`, `5a37c51`). The selected model of
`research/main_model/README.txt` has the branch `f = R_D(W_F (W_g z ⊙_ℂ W_u z)/√2)`,
with complex-linear projections stored as `[[A,-B],[B,A]]`. The three products of
`research/complex_transformer/README.md` and the native kernels are

* diagonal (`complex.rs`, `product::<false>`): `s(g) * (g_R u_R, g_I u_I)`;
* radial (`complex.rs`, `product::<true>`): `s(g) * g * u / √2`;
* plain (`plain_product.rs`, the selected model): `g * u / √2`;

with `s(g) = sigmoid(sqrt(|g|^2 + ε))` and `ε = 2^-23` in the kernels. The control
is `src/diff_gpt/domain/model/feed_forward.py` and `base_ops.rs` `swiglu`:
`c_proj(SiLU(gate) * value)`, without biases.

A block is written unit by unit. A SwiGLU unit `j` has gate row `a j`, value row
`b j` and output column `c j`. A complex unit `k` has gate and value rows
`g k, u k : V →ₗ[ℝ] ℂ`; the projection is complex-linear when they commute with
`Complex.I`, which theorems state as a hypothesis. Its real and imaginary outputs
leave through columns `c k` and `c' k` of `W_F`. The final branch RMS is not part
of these maps; `Basic` shows which symmetries survive it.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.ComplexFFN

open Transformer.Landscape.Neural (silu)

variable {V W ι κ : Type*} [Fintype ι] [Fintype κ]
variable [AddCommGroup V] [Module ℝ V] [AddCommGroup W] [Module ℝ W]

/-- The bias-free SwiGLU block of `feed_forward.py`: `Σ_j SiLU(a_j z) (b_j z) c_j`. -/
noncomputable def swigluFFN (a b : ι → V →ₗ[ℝ] ℝ) (c : ι → W) (z : V) : W :=
  ∑ j, (silu (a j z) * b j z) • c j

/-- A sum of products of two real linear forms, `Σ_j (a_j z)(b_j z) c_j`. It is not
an architecture; it is the common quadratic part of the blocks below. -/
noncomputable def productFFN (a b : ι → V →ₗ[ℝ] ℝ) (c : ι → W) (z : V) : W :=
  ∑ j, (a j z * b j z) • c j

/-- The real matrix `W_F` applied to a complex hidden vector: the real part of unit
`k` leaves through column `c k`, its imaginary part through `c' k`. -/
def complexReadout (c c' : κ → W) (x : κ → ℂ) : W :=
  ∑ k, ((x k).re • c k + (x k).im • c' k)

/-- The radial gate of `complex.rs`: `sigmoid(sqrt(g_R^2 + g_I^2 + ε))`. -/
noncomputable def radialGate (ε : ℝ) (w : ℂ) : ℝ :=
  Real.sigmoid (Real.sqrt (Complex.normSq w + ε))

/-- Hidden vector of the plain product (`plain_product.rs`): `g * u / √2`. -/
noncomputable def plainHidden (g u : κ → V →ₗ[ℝ] ℂ) (z : V) (k : κ) : ℂ :=
  g k z * u k z / (Real.sqrt 2 : ℂ)

/-- Hidden vector of the radial product (`complex.rs`, `product::<true>`). -/
noncomputable def radialHidden (ε : ℝ) (g u : κ → V →ₗ[ℝ] ℂ) (z : V) (k : κ) : ℂ :=
  (radialGate ε (g k z) : ℂ) * (g k z * u k z) / (Real.sqrt 2 : ℂ)

/-- Hidden vector of the coordinate product (`complex.rs`, `product::<false>`). -/
noncomputable def diagonalHidden (ε : ℝ) (g u : κ → V →ₗ[ℝ] ℂ) (z : V) (k : κ) : ℂ :=
  (radialGate ε (g k z) : ℂ) * ⟨(g k z).re * (u k z).re, (g k z).im * (u k z).im⟩

/-- The selected model's FFN before its branch RMS: `W_F (W_g z ⊙_ℂ W_u z) / √2`. -/
noncomputable def plainFFN (g u : κ → V →ₗ[ℝ] ℂ) (c c' : κ → W) (z : V) : W :=
  complexReadout c c' (plainHidden g u z)

/-- The radial complex-product FFN before its branch RMS. -/
noncomputable def radialFFN (ε : ℝ) (g u : κ → V →ₗ[ℝ] ℂ) (c c' : κ → W) (z : V) : W :=
  complexReadout c c' (radialHidden ε g u z)

/-- The diagonal (coordinate-product) control FFN before its branch RMS. -/
noncomputable def diagonalFFN (ε : ℝ) (g u : κ → V →ₗ[ℝ] ℂ) (c c' : κ → W) (z : V) : W :=
  complexReadout c c' (diagonalHidden ε g u z)

/-- `Landscape.Neural.silu` is the kernel's `gate * sigmoid(gate)` (`base_ops.rs`). -/
theorem silu_eq_mul_sigmoid (t : ℝ) : silu t = t * Real.sigmoid t := by
  rw [silu, Real.sigmoid_def, div_eq_mul_inv]

/-- The even part of SiLU: `SiLU(t) + SiLU(-t) = t (2σ(t) - 1)`. -/
theorem silu_add_neg (t : ℝ) : silu t + silu (-t) = t * (2 * Real.sigmoid t - 1) := by
  rw [silu_eq_mul_sigmoid, silu_eq_mul_sigmoid, Real.sigmoid_neg]
  ring

/-- The readout is additive in the hidden vector; it negates with it. -/
theorem complexReadout_neg (c c' : κ → W) (x : κ → ℂ) :
    complexReadout c c' (-x) = -complexReadout c c' x := by
  simp only [complexReadout, Pi.neg_apply, Complex.neg_re, Complex.neg_im, neg_smul,
    ← neg_add, Finset.sum_neg_distrib]

/-- With columns `1` and `i` into `ℂ` the readout is the complex-linear `W_F = 1`:
it sums the hidden coordinates. -/
theorem complexReadout_one_I (x : κ → ℂ) :
    complexReadout (fun _ => (1 : ℂ)) (fun _ => Complex.I) x = ∑ k, x k := by
  simp only [complexReadout, Complex.real_smul, mul_one]
  exact Finset.sum_congr rfl fun k _ => Complex.re_add_im (x k)

/-- One plain unit with output columns `√2` and `√2 i` is the product of its two
projections. -/
theorem plainFFN_unit (φ ψ : V →ₗ[ℝ] ℂ) (z : V) :
    plainFFN (fun _ : Unit => φ) (fun _ => ψ) (fun _ => (Real.sqrt 2 : ℂ))
      (fun _ => (Real.sqrt 2 : ℂ) * Complex.I) z = φ z * ψ z := by
  have h2 : (Real.sqrt 2 : ℂ) ≠ 0 := by exact_mod_cast (Real.sqrt_pos.2 two_pos).ne'
  simp only [plainFFN, complexReadout, Fintype.sum_unique, Complex.real_smul]
  calc ((plainHidden (fun _ : Unit => φ) (fun _ => ψ) z default).re : ℂ) * (Real.sqrt 2 : ℂ) +
        ((plainHidden (fun _ : Unit => φ) (fun _ => ψ) z default).im : ℂ) *
          ((Real.sqrt 2 : ℂ) * Complex.I)
      = (Real.sqrt 2 : ℂ) * plainHidden (fun _ : Unit => φ) (fun _ => ψ) z default := by
        rw [← Complex.re_add_im (plainHidden (fun _ : Unit => φ) (fun _ => ψ) z default)]
        simp only [Complex.ofReal_re, Complex.ofReal_im, Complex.add_re, Complex.add_im,
          Complex.mul_re, Complex.mul_im, Complex.I_re, Complex.I_im]
        ring
    _ = φ z * ψ z := by
        simp only [plainHidden]
        field_simp

/-- The radial gate sees only `|g|`; in particular it ignores phase and sign. -/
theorem radialGate_of_normSq_eq (ε : ℝ) {w w' : ℂ}
    (h : Complex.normSq w' = Complex.normSq w) : radialGate ε w' = radialGate ε w := by
  rw [radialGate, radialGate, h]

/-- The radial gate is a sigmoid value, so it is nonnegative. -/
theorem radialGate_nonneg (ε : ℝ) (w : ℂ) : 0 ≤ radialGate ε w :=
  Real.sigmoid_nonneg _

/-- The radial gate is a sigmoid value, so it is at most `1`. -/
theorem radialGate_le_one (ε : ℝ) (w : ℂ) : radialGate ε w ≤ 1 :=
  Real.sigmoid_le_one _

/-- A larger modulus gives a strictly larger radial gate when `ε ≥ 0`. -/
theorem radialGate_lt (ε : ℝ) (hε : 0 ≤ ε) {w w' : ℂ}
    (h : Complex.normSq w < Complex.normSq w') : radialGate ε w < radialGate ε w' := by
  unfold radialGate
  apply Real.sigmoid_lt
  apply Real.sqrt_lt_sqrt (add_nonneg (Complex.normSq_nonneg w) hε)
  linarith

example : radialGate 0 0 < radialGate 0 1 :=
  radialGate_lt 0 le_rfl (by simp)

example : radialGate (2 ^ (-23 : ℤ)) Complex.I = radialGate (2 ^ (-23 : ℤ)) 1 :=
  radialGate_of_normSq_eq _ (by simp)

end Transformer.ComplexFFN

module

public import Transformer.ComplexFFN.Basic
public import Transformer.Grokking.GradientEvidence.ForecastTransport

/-!
# Every plain-product FFN is a SwiGLU FFN

Source: the blocks of `Defs` (DiffGPT `5a37c51`). The paired-gate identity is the
repository's `Landscape.Neural.swiglu_sub_neg_gate` (M9.4b), here for whole blocks.

Opposed gates cancel SiLU's even part, so one real product costs two SwiGLU units
exactly (`productFFN_eq_swigluFFN`). Gauss's identity writes one complex product as
three real products (`plainFFN_eq_productFFN`). Hence a plain FFN with `h` complex
units equals a SwiGLU FFN with `6h` units on the same input and output
(`plainFFN_eq_swigluFFN`). With small gates and large values one SwiGLU unit tends to
one product (`swigluFFN_scaled_sub_productFFN`), so `3h` units approximate a plain FFN
with `h` units, with an explicit error (`swigluFFN_gauss_sub_plainFFN`).

A complex unit of the model has `3D` real parameters (complex-linear rows of `W_g`,
`W_u` and a column of `W_F`, each `D/2` complex entries), as many as one SwiGLU unit.
So the exact embedding costs six times the parameters, and the limit three times.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.ComplexFFN

open Transformer.Landscape.Neural (silu silu_sub_neg)

variable {V W ι κ : Type*} [Fintype ι] [Fintype κ]
variable [AddCommGroup V] [Module ℝ V] [AddCommGroup W] [Module ℝ W]

/-- Renumbering the hidden units does not change a SwiGLU block. -/
theorem swigluFFN_comp_equiv {ι' : Type*} [Fintype ι'] (e : ι' ≃ ι)
    (a b : ι → V →ₗ[ℝ] ℝ) (c : ι → W) :
    swigluFFN (a ∘ e) (b ∘ e) (c ∘ e) = swigluFFN a b c := by
  funext z
  exact Equiv.sum_comp e (fun j => (silu (a j z) * b j z) • c j)

/-- Two SwiGLU units per product, exactly: opposed gates, shared values, opposed
outputs. -/
theorem productFFN_eq_swigluFFN (a b : ι → V →ₗ[ℝ] ℝ) (c : ι → W) :
    productFFN a b c = swigluFFN (Sum.elim a (-a)) (Sum.elim b b) (Sum.elim c (-c)) := by
  funext z
  simp only [swigluFFN, productFFN, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr,
    Pi.neg_apply, LinearMap.neg_apply, smul_neg, Finset.sum_neg_distrib, ← sub_eq_add_neg]
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← sub_smul, ← sub_mul, silu_sub_neg]

/-- Gate rows of Gauss's three products for complex unit `k`: `u_R`, `g_R`, `g_I`. -/
def gaussGate (g u : κ → V →ₗ[ℝ] ℂ) (p : κ × Fin 3) : V →ₗ[ℝ] ℝ :=
  ![Complex.reLm ∘ₗ u p.1, Complex.reLm ∘ₗ g p.1, Complex.imLm ∘ₗ g p.1] p.2

/-- Value rows of Gauss's three products: `g_R + g_I`, `u_I - u_R`, `u_R + u_I`. -/
def gaussValue (g u : κ → V →ₗ[ℝ] ℂ) (p : κ × Fin 3) : V →ₗ[ℝ] ℝ :=
  ![Complex.reLm ∘ₗ g p.1 + Complex.imLm ∘ₗ g p.1,
    Complex.imLm ∘ₗ u p.1 - Complex.reLm ∘ₗ u p.1,
    Complex.reLm ∘ₗ u p.1 + Complex.imLm ∘ₗ u p.1] p.2

/-- Output columns of Gauss's products: `Re(g u) = P₀ - P₂`, `Im(g u) = P₀ + P₁`,
each divided by `√2` as in `plain_product.rs`. -/
noncomputable def gaussOutput (c c' : κ → W) (p : κ × Fin 3) : W :=
  ![(Real.sqrt 2)⁻¹ • (c p.1 + c' p.1), (Real.sqrt 2)⁻¹ • c' p.1,
    -((Real.sqrt 2)⁻¹ • c p.1)] p.2

/-- Gauss: a complex product is three real products, so a plain FFN with `h`
complex units is a product network with `3h` terms. -/
theorem plainFFN_eq_productFFN (g u : κ → V →ₗ[ℝ] ℂ) (c c' : κ → W) :
    plainFFN g u c c' = productFFN (gaussGate g u) (gaussValue g u) (gaussOutput c c') := by
  funext z
  simp only [plainFFN, complexReadout, productFFN, Fintype.sum_prod_type, Fin.sum_univ_three]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp only [plainHidden, gaussGate, gaussValue, gaussOutput, Matrix.cons_val_zero,
    Matrix.cons_val_one, Matrix.cons_val_two, Matrix.head_cons, Matrix.tail_cons,
    LinearMap.comp_apply, LinearMap.add_apply, LinearMap.sub_apply, Complex.reLm_coe,
    Complex.imLm_coe, Complex.div_ofReal_re, Complex.div_ofReal_im, Complex.mul_re,
    Complex.mul_im]
  module

/-- A plain FFN with `h` complex units is exactly a bias-free SwiGLU FFN with `6h`
units, numbered like a hidden layer. -/
theorem plainFFN_eq_swigluFFN (g u : κ → V →ₗ[ℝ] ℂ) (c c' : κ → W) :
    ∃ (a b : Fin (6 * Fintype.card κ) → V →ₗ[ℝ] ℝ) (e : Fin (6 * Fintype.card κ) → W),
      plainFFN g u c c' = swigluFFN a b e := by
  have hcard : Fintype.card ((κ × Fin 3) ⊕ (κ × Fin 3)) = 6 * Fintype.card κ := by
    simp only [Fintype.card_sum, Fintype.card_prod, Fintype.card_fin]
    ring
  let e := (Fintype.equivFinOfCardEq hcard).symm
  refine ⟨Sum.elim (gaussGate g u) (-gaussGate g u) ∘ e,
    Sum.elim (gaussValue g u) (gaussValue g u) ∘ e,
    Sum.elim (gaussOutput c c') (-gaussOutput c c') ∘ e, ?_⟩
  rw [swigluFFN_comp_equiv, plainFFN_eq_productFFN, productFFN_eq_swigluFFN]

/-- One SwiGLU unit with gate `η a` and value `(2/η) b` is within
`|η| (a z)² |b z| / 2` of the product `(a z)(b z)`: `SiLU(t) = t σ(t)` and `σ` is
`1/4`-Lipschitz with `σ(0) = 1/2`. -/
theorem silu_scaled_sub (t s : ℝ) {η : ℝ} (hη : η ≠ 0) :
    |silu (η * t) * (2 / η * s) - t * s| ≤ |η| / 2 * (t ^ 2 * |s|) := by
  have hl := Grokking.GradientEvidence.sigmoid_quarter_lipschitz.dist_le_mul (η * t) 0
  simp only [Real.dist_eq, Real.sigmoid_zero, sub_zero, NNReal.coe_div, NNReal.coe_one,
    NNReal.coe_ofNat] at hl
  have he : silu (η * t) * (2 / η * s) - t * s = 2 * t * s * (Real.sigmoid (η * t) - 2⁻¹) := by
    rw [silu_eq_mul_sigmoid]
    field_simp
  rw [he, abs_mul, abs_mul, abs_mul, abs_two]
  calc 2 * |t| * |s| * |Real.sigmoid (η * t) - 2⁻¹|
      ≤ 2 * |t| * |s| * (1 / 4 * |η * t|) := by gcongr
    _ = |η| / 2 * (t ^ 2 * |s|) := by
      rw [abs_mul, ← sq_abs t]
      ring

example : |silu (1 * 1) * (2 / 1 * 1) - 1 * 1| ≤ |(1 : ℝ)| / 2 * (1 ^ 2 * |(1 : ℝ)|) :=
  silu_scaled_sub 1 1 one_ne_zero

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Small gates and large values: SwiGLU FFNs with `m` units tend to every product
network with `m` terms, uniformly where `Σ_j (a_j z)² |b_j z| ‖c_j‖` is bounded. -/
theorem swigluFFN_scaled_sub_productFFN (a b : ι → V →ₗ[ℝ] ℝ) (c : ι → E) {η : ℝ}
    (hη : η ≠ 0) (z : V) :
    ‖swigluFFN (fun j => η • a j) (fun j => (2 / η) • b j) c z - productFFN a b c z‖ ≤
      |η| / 2 * ∑ j, a j z ^ 2 * |b j z| * ‖c j‖ := by
  simp only [swigluFFN, productFFN, ← Finset.sum_sub_distrib, ← sub_smul, Finset.mul_sum,
    LinearMap.smul_apply, smul_eq_mul]
  refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun j _ => ?_)
  rw [norm_smul, Real.norm_eq_abs]
  calc |silu (η * a j z) * (2 / η * b j z) - a j z * b j z| * ‖c j‖
      ≤ |η| / 2 * (a j z ^ 2 * |b j z|) * ‖c j‖ := by
        gcongr
        exact silu_scaled_sub _ _ hη
    _ = |η| / 2 * (a j z ^ 2 * |b j z| * ‖c j‖) := by ring

example (z : ℝ) := swigluFFN_scaled_sub_productFFN (fun _ : Unit => (LinearMap.id : ℝ →ₗ[ℝ] ℝ))
  (fun _ => LinearMap.id) (fun _ => (1 : ℝ)) one_ne_zero z

/-- `3h` SwiGLU units approximate a plain FFN with `h` complex units: Gauss's products
with gates scaled by `η` and values by `2/η`, at error `O(|η|)` on bounded sets. -/
theorem swigluFFN_gauss_sub_plainFFN (g u : κ → V →ₗ[ℝ] ℂ) (c c' : κ → E) {η : ℝ}
    (hη : η ≠ 0) (z : V) :
    ‖swigluFFN (fun p => η • gaussGate g u p) (fun p => (2 / η) • gaussValue g u p)
        (gaussOutput c c') z - plainFFN g u c c' z‖ ≤
      |η| / 2 * ∑ p, gaussGate g u p z ^ 2 * |gaussValue g u p z| * ‖gaussOutput c c' p‖ := by
  rw [plainFFN_eq_productFFN]
  exact swigluFFN_scaled_sub_productFFN _ _ _ hη z

example (z : ℂ) := swigluFFN_gauss_sub_plainFFN (fun _ : Unit => (LinearMap.id : ℂ →ₗ[ℝ] ℂ))
  (fun _ => LinearMap.id) (fun _ => (1 : ℂ)) (fun _ => Complex.I) one_ne_zero z

end Transformer.ComplexFFN

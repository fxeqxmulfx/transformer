/-
# Homogenized Transformers — the softmax barycenter

The object `lemma:Laplace_method`, `lem:delta_method`, `lem:lemma_app` and
`lem:Ito_formula` of arXiv:2604.01978v1, *Homogenized Transformers*, are all
written in:

  `m_{β,A}[μ](x) = 𝒵_A[μ](x)⁻¹ ∫ e^{β⟨A x, y⟩} y μ(dy)`,

the attention field `EQ:VELOCITY_FIELD_SELF_ATTENTION` with the value matrix
removed, together with the statistic `s_μ(x,x') = E_A⟨m(x), m(x')⟩` of
`eq:s_mu_defs_clean` built from it.

`Laplace.lean` is what estimates `s_μ` at low temperature.

Source: arXiv:2604.01978v1, `eq: mbetaA`, `eq:s_mu_defs_clean`.
-/

import Transformer.Homogenized.MeanField

open scoped BigOperators ENNReal NNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The softmax barycenter -/

/-- The attention weight of `eq: mbetaA`, `e^{β⟨A x, y⟩}`, as a function of the
query-key matrix alone. -/
noncomputable def softWeight {d : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (x y : EucSpace d) : ℝ :=
  Real.exp (β * inner (𝕜 := ℝ) (Matrix.toEuclideanLin A x) y)

theorem softWeight_pos {d : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (x y : EucSpace d) : 0 < softWeight β A x y :=
  Real.exp_pos _

/-- The weight of `eq: mbetaA` is the attention weight of the head: the value
matrix never enters it. -/
theorem softWeight_eq_attnWeight {d : ℕ} (β : ℝ) (θ : HeadParam d) (x y : EucSpace d) :
    softWeight β θ.2 x y = attnWeight β θ x y := rfl

/-- The **softmax barycenter** `m_{β,A}[μ](x)` of `eq: mbetaA`,

  `m_{β,A}[μ](x) = 𝒵_A[μ](x)⁻¹ ∫ e^{β⟨A x, y⟩} y μ(dy)`.

Source: arXiv:2604.01978v1, `eq: mbetaA`. -/
noncomputable def softBary {d : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (μ : Measure (EucSpace d)) (x : EucSpace d) : EucSpace d :=
  (∫ y, softWeight β A x y ∂μ)⁻¹ • ∫ y, softWeight β A x y • y ∂μ

/-- At a Dirac mass the barycenter is the atom, at every temperature: the
normalizer cancels the weight. -/
@[simp] theorem softBary_dirac {d : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (z x : EucSpace d) : softBary β A (Measure.dirac z) x = z := by
  rw [softBary, integral_dirac, integral_dirac, smul_smul,
    inv_mul_cancel₀ (softWeight_pos β A x z).ne', one_smul]

/-- **`m_{β,A}[μ](x)` is a convex combination of unit vectors**, so it lies in
the closed unit ball.  This is the source's parenthetical remark in the proof
of `lem:Ito_formula`, and it is what makes `s_μ(x,y) = E⟨m(x),m(y)⟩` an
integral of a bounded function rather than a junk value.

Source: arXiv:2604.01978v1, proof of `lem:Ito_formula`. -/
theorem norm_softBary_le_one {d : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (μ : Measure (EucSpace d)) [IsProbabilityMeasure μ] (x : EucSpace d)
    (hsph : ∀ᵐ y ∂μ, ‖y‖ = 1)
    (hint : Integrable (fun y => softWeight β A x y) μ) :
    ‖softBary β A μ x‖ ≤ 1 := by
  have hZpos : 0 < ∫ y, softWeight β A x y ∂μ := by
    have hsupp : Function.support (fun y => softWeight β A x y) = Set.univ := by
      ext y
      simp [Function.mem_support, (softWeight_pos β A x y).ne']
    rw [integral_pos_iff_support_of_nonneg (fun y => (softWeight_pos β A x y).le) hint,
      hsupp, measure_univ]
    exact zero_lt_one
  have hbound : ∀ᵐ y ∂μ, ‖softWeight β A x y • y‖ ≤ softWeight β A x y := by
    filter_upwards [hsph] with y hy
    rw [norm_smul, hy, mul_one, Real.norm_eq_abs, abs_of_pos (softWeight_pos β A x y)]
  have hnum : ‖∫ y, softWeight β A x y • y ∂μ‖ ≤ ∫ y, softWeight β A x y ∂μ := by
    refine le_trans (norm_integral_le_integral_norm _) ?_
    exact integral_mono_of_nonneg (Filter.Eventually.of_forall fun y => norm_nonneg _)
      hint hbound
  rw [softBary, norm_smul, norm_inv, Real.norm_eq_abs, abs_of_pos hZpos, inv_mul_eq_div,
    div_le_one hZpos]
  exact hnum

/-- **`s_μ(x,x')` of `eq:s_mu_defs_clean`**, the correlation of two softmax
barycenters averaged over the head law:

  `s_μ(x,x') = E_A⟨m_{β,A}[μ](x), m_{β,A}[μ](x')⟩`.

The source's other statistic `s_μ(x) = E‖m_{β,A}[μ](x)‖²` is this one at
`x' = x`, by `baryCorr_self`; there is one object, not two.

Source: arXiv:2604.01978v1, `eq:s_mu_defs_clean`. -/
noncomputable def baryCorr {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (μ : Measure (EucSpace d)) (x y : EucSpace d) : ℝ :=
  ∫ θ, inner (𝕜 := ℝ) (softBary β θ.2 μ x) (softBary β θ.2 μ y) ∂ρ

/-- `s_μ(x) = s_μ(x,x)`. -/
theorem baryCorr_self {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (μ : Measure (EucSpace d)) (x : EucSpace d) :
    baryCorr β ρ μ x x = ∫ θ, ‖softBary β θ.2 μ x‖ ^ 2 ∂ρ := by
  simp only [baryCorr, real_inner_self_eq_norm_sq]

/-- `s_μ` is symmetric. -/
theorem baryCorr_comm {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (μ : Measure (EucSpace d)) (x y : EucSpace d) :
    baryCorr β ρ μ x y = baryCorr β ρ μ y x := by
  simp only [baryCorr]
  exact integral_congr_ae (Filter.Eventually.of_forall fun θ => real_inner_comm _ _)

/-- The normalization `v ↦ v/‖v‖` is invariant under a positive rescaling.
This is why `lemma:Laplace_method` and `lem:delta_method` compose although the
source writes the first at `A = W W'ᵀ` and the second at
`Â = (dσ_A²)⁻¹ Wᵀ W'`: the quantity `E⟨Ax/‖Ax‖, Ay/‖Ay‖⟩` they share does not
see the constant. -/
theorem normalizeLayer_smul {d : ℕ} {c : ℝ} (hc : 0 < c) (v : EucSpace d) :
    normalizeLayer (c • v) = normalizeLayer v := by
  rw [normalizeLayer, normalizeLayer, norm_smul, Real.norm_eq_abs, abs_of_pos hc,
    mul_inv, smul_smul]
  congr 1
  field_simp

end Homogenized
end Transformer

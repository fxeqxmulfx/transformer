/-
# Kinetic theory for Transformers — the original Volterra kernel

In arXiv:2605.09213v1, §6, proof of `prop:Bessel`, the directed kernel
is the smooth expression `r_λ(σ)e^{-λ(σ-σ')}` when `σ'<σ`, with
`r_λ=λ/(1-e^{-λσ})` and `r_0=1/σ`. Its two-step product factors into
the original kernel times `r_λ(σ')`, the actual derivative of `Y`.
Here are that factorization, continuity of the smooth expression and
a uniform bound on the original positive-position rectangle.
-/

import Transformer.Kinetic.Section6_HardyVariable
import Mathlib.Analysis.Normed.Group.Bounded

open Real Set MeasureTheory

namespace Transformer
namespace Kinetic

/-- The original Hardy kernel expression before its causal indicator.
Source: arXiv:2605.09213v1, §1, `eq:kernel-Klambda`, and §6, proof of `prop:Bessel`. -/
noncomputable def hardyKernel (lam s r : ℝ) : ℝ :=
  hardyRate lam s * Real.exp (-(lam * (s - r)))

/-- At strictly ordered positions the original graphon is its smooth kernel expression.
Source: arXiv:2605.09213v1, `eq:kernel-Klambda`. -/
theorem graphon_eq_hardyKernel {s r : ℝ} (horder : r < s) (lam : ℝ) :
    graphon lam s r = hardyKernel lam s r := by
  rw [graphon, ite_eq_left horder]
  by_cases hl : lam = 0
  · simp only [hl, hardyKernel, hardyRate, ite_true, zero_mul, neg_zero, Real.exp_zero, mul_one]
  · rw [ite_eq_right hl, hardyKernel, hardyRate, ite_eq_right hl]
    ring

example := graphon_eq_hardyKernel one_half_lt_one (-1)

/-- The original Hardy rate is continuous at every positive position.
Source: arXiv:2605.09213v1, §6, the smooth change of variable in `prop:Bessel`. -/
theorem continuousAt_hardyRate (lam : ℝ) {s : ℝ} (hs : 0 < s) :
    ContinuousAt (hardyRate lam) s := by
  by_cases hl : lam = 0
  · subst lam
    have heq : hardyRate 0 = fun z : ℝ => z⁻¹ := by
      funext z
      simp only [hardyRate, ite_true]
    rw [heq]
    exact continuousAt_id.inv₀ hs.ne'
  · have hden := exp_mul_sub_one_ne_zero hl hs
    have hc : ContinuousAt (fun z => lam * Real.exp (lam * z) / (Real.exp (lam * z) - 1)) s := by
      apply ContinuousAt.div
      · fun_prop
      · fun_prop
      · exact hden
    apply hc.congr
    filter_upwards [Ioi_mem_nhds hs] with z hz
    exact (hardyRate_eq hl hz).symm

example := continuousAt_hardyRate (-1) one_half_pos
example := continuousAt_hardyRate 0 one_half_pos

/-- The smooth kernel is continuous in its second position, with the
first position fixed. Source: arXiv:2605.09213v1, §6, the spatial integral in `prop:Bessel`. -/
theorem continuous_hardyKernel_right (lam s : ℝ) : Continuous (hardyKernel lam s) := by
  unfold hardyKernel
  fun_prop

/-- Joint continuity of the actual smooth kernel at a positive first position.
Source: arXiv:2605.09213v1, §6, the spatial integral in `prop:Bessel`. -/
theorem continuousAt_hardyKernel (lam : ℝ) {p : ℝ × ℝ} (hp : 0 < p.1) :
    ContinuousAt (fun p : ℝ × ℝ => hardyKernel lam p.1 p.2) p := by
  unfold hardyKernel
  apply ContinuousAt.mul
  · exact (continuousAt_hardyRate lam hp).comp continuousAt_fst
  · fun_prop

example := continuousAt_hardyKernel (-1) (p := (1, 1 / 2)) one_pos

/-- Every real bias gives a positive smooth kernel at positive positions.
Source: arXiv:2605.09213v1, `eq:kernel-Klambda` and §6, proof of `prop:Bessel`. -/
theorem hardyKernel_pos (lam r : ℝ) {s : ℝ} (hs : 0 < s) : 0 < hardyKernel lam s r :=
  mul_pos (hardyRate_pos lam hs) (Real.exp_pos _)

example := hardyKernel_pos (-1) (1 / 2) one_pos

/-- The two-step smooth kernel factors into the original kernel times
the actual spatial derivative rate.
Source: arXiv:2605.09213v1, §6, the reduction of `eq:Hardy-hom` to the Goursat equation. -/
theorem hardyKernel_mul (lam s r s₀ : ℝ) :
    hardyKernel lam s r * hardyKernel lam r s₀ = hardyKernel lam s s₀ * hardyRate lam r := by
  have he : Real.exp (-(lam * (s - r))) * Real.exp (-(lam * (r - s₀))) =
      Real.exp (-(lam * (s - s₀))) := by
    rw [← Real.exp_add]
    congr 1
    ring
  unfold hardyKernel
  calc hardyRate lam s * Real.exp (-(lam * (s - r))) *
        (hardyRate lam r * Real.exp (-(lam * (r - s₀)))) =
        (hardyRate lam s * hardyRate lam r) *
          (Real.exp (-(lam * (s - r))) * Real.exp (-(lam * (r - s₀)))) := by ring
    _ = _ := by rw [he]; ring

/-- The factorization of the actual directed graphon at ordered positions.
Source: arXiv:2605.09213v1, §6, proof of `prop:Bessel`. -/
theorem graphon_mul_of_order (lam : ℝ) {s r s₀ : ℝ} (hr : s₀ < r) (hs : r < s) :
    graphon lam s r * graphon lam r s₀ = graphon lam s s₀ * hardyRate lam r := by
  rw [graphon_eq_hardyKernel hs, graphon_eq_hardyKernel hr,
    graphon_eq_hardyKernel (hr.trans hs), hardyKernel_mul]

example := graphon_mul_of_order (-1) (s := 1) (r := 1 / 2) (s₀ := 1 / 4)
  (by norm_num) one_half_lt_one

/-- The original kernel is uniformly bounded on each actual compact
positive-position rectangle, for every real bias.
Source: arXiv:2605.09213v1, §6, the Volterra operator in `prop:Bessel`. -/
theorem exists_bound_hardyKernel (lam : ℝ) {s₀ : ℝ} (hs₀ : 0 < s₀) :
    ∃ K : ℝ, 0 ≤ K ∧ ∀ s ∈ Icc s₀ 1, ∀ r ∈ Icc s₀ 1, |hardyKernel lam s r| ≤ K := by
  have hc : ContinuousOn (fun p : ℝ × ℝ => hardyKernel lam p.1 p.2) (Icc s₀ 1 ×ˢ Icc s₀ 1) := by
    intro p hp
    exact (continuousAt_hardyKernel lam (hs₀.trans_le hp.1.1)).continuousWithinAt
  obtain ⟨K, hK⟩ := (isCompact_Icc.prod isCompact_Icc).exists_bound_of_continuousOn hc
  refine ⟨max K 0, le_max_right _ _, fun s hs r hr => ?_⟩
  exact (hK (s, r) ⟨hs, hr⟩).trans (le_max_left _ _)

example := exists_bound_hardyKernel (-1) one_half_pos

/-- The actual directed graphon inherits the same compact rectangle bound.
Source: arXiv:2605.09213v1, §6, the Volterra operator in `prop:Bessel`. -/
theorem exists_bound_graphon (lam : ℝ) {s₀ : ℝ} (hs₀ : 0 < s₀) :
    ∃ K : ℝ, 0 ≤ K ∧ ∀ s ∈ Icc s₀ 1, ∀ r ∈ Icc s₀ 1, |graphon lam s r| ≤ K := by
  obtain ⟨K, hK0, hK⟩ := exists_bound_hardyKernel lam hs₀
  refine ⟨K, hK0, fun s hs r hr => ?_⟩
  by_cases horder : r < s
  · rw [graphon_eq_hardyKernel horder]
    exact hK s hs r hr
  · rw [graphon_of_le (le_of_not_gt horder)]
    simpa using hK0

example := exists_bound_graphon (-1) one_half_pos

/-- The causal graphon's endpoint value does not affect its original
spatial integral: the smooth kernel agrees on the open interval.
Source: arXiv:2605.09213v1, §6, the spatial integral in `eq:Hardy-hom`. -/
theorem integral_graphon_eq_hardyKernel (lam : ℝ) (f : ℝ → ℝ) {s s₀ : ℝ} (horder : s₀ ≤ s) :
    (∫ r in s₀..s, graphon lam s r * f r) = ∫ r in s₀..s, hardyKernel lam s r * f r := by
  apply intervalIntegral.integral_congr_Ioo_of_le horder
  intro r hr
  change graphon lam s r * f r = hardyKernel lam s r * f r
  rw [graphon_eq_hardyKernel hr.2]

example := integral_graphon_eq_hardyKernel (-1) Real.sin one_half_lt_one.le

end Kinetic
end Transformer

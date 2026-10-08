import Transformer.Modes.Section5_SecondDerivativeTest

/-!
# Phase estimates when the first two derivatives do not vanish together

The determinant argument in arXiv:2412.09080v3, §5.5, ensures that the
first and second phase derivatives cannot vanish simultaneously. This
module gives a quantitative estimate from a lower bound
`max (|φ'|, |φ''|) ≥ κ > 0` on an interval of length at most one.

On an interval where `φ''` has one sign, the estimate is
`10 / sqrt (|ρ| κ)`. If `|ρ| κ ≥ 4`, the threshold
`δ = sqrt (κ / |ρ|)` is smaller than `κ`; thus `|φ'| ≤ δ` forces
`|φ''| ≥ κ`. The small-derivative estimate then applies. If `|ρ| κ < 4`,
the interval's length and the unit norm of the kernel give the bound.

The amplitude version follows from bounds on every partial integral.
The Gaussian phase will be split at the zeros of its second derivative
before applying this test. The lower bound remains a condition on the
actual derivatives, with both derivative identities stated explicitly.
Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`.
-/

open Real MeasureTheory
open scoped Interval

namespace Transformer.Modes

/-- The local phase estimate for a nonnegative second derivative.
Source: arXiv:2412.09080v3, §5.5, the determinant and stationary-phase bounds. -/
theorem unit_interval_phase_test_nonneg {a b ρ κ : ℝ} {φ p q : ℝ → ℝ}
    (hab : a ≤ b) (hlen : b - a ≤ 1) (hρ : ρ ≠ 0) (hκ : 0 < κ)
    (hφ : ∀ t ∈ Set.uIcc a b, HasDerivAt φ (p t) t)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hcq : ContinuousOn q (Set.uIcc a b))
    (hq : ∀ t ∈ Set.uIcc a b, 0 ≤ q t)
    (hmin : ∀ t ∈ Set.uIcc a b, κ ≤ max |p t| |q t|) :
    ‖∫ t in a..b, oscillatoryKernel ρ φ t‖ ≤ 10 / Real.sqrt (|ρ| * κ) := by
  have hρpos : 0 < |ρ| := abs_pos.mpr hρ
  by_cases hlarge : 4 ≤ |ρ| * κ
  · let δ := Real.sqrt (κ / |ρ|)
    have hδ : 0 < δ := Real.sqrt_pos.mpr (div_pos hκ hρpos)
    have hsq : δ ^ 2 = κ / |ρ| := Real.sq_sqrt (div_nonneg hκ.le hρpos.le)
    have hmul : |ρ| * δ ^ 2 = κ := by rw [hsq]; field_simp
    have hδle : δ ≤ κ / 2 := by
      apply Real.sqrt_le_iff.mpr
      refine ⟨(half_pos hκ).le, ?_⟩
      apply (div_le_iff₀ hρpos).mpr
      nlinarith [mul_nonneg hκ.le (sub_nonneg.mpr hlarge)]
    have hlow (t : ℝ) (ht : t ∈ Set.uIcc a b) (hpt : |p t| ≤ δ) : κ ≤ q t := by
      have hmax := hmin t ht
      rw [le_max_iff] at hmax
      rcases hmax with hmax | hmax
      · have hlt := hpt.trans_lt (hδle.trans_lt (half_lt_self hκ))
        exact False.elim (hlt.not_ge hmax)
      · simpa only [abs_of_nonneg (hq t ht)] using hmax
    have h := oscillatory_integral_le_of_sublevel_derivative hab hρ hδ hκ hφ hp hcq hq hlow
    exact h.trans_eq (oscillatory_threshold_balance hρ hδ hκ hmul)
  · have hspos : 0 < Real.sqrt (|ρ| * κ) := Real.sqrt_pos.mpr (mul_pos hρpos hκ)
    have hsle : Real.sqrt (|ρ| * κ) ≤ 2 :=
      Real.sqrt_le_iff.mpr ⟨by norm_num, by nlinarith [not_le.mp hlarge]⟩
    have hi := intervalIntegral.norm_integral_le_of_norm_le_const (a := a) (b := b)
      (C := 1) (fun t _ => (norm_oscillatoryKernel ρ φ t).le)
    rw [one_mul, abs_of_nonneg (sub_nonneg.mpr hab)] at hi
    refine (hi.trans hlen).trans ?_
    apply (le_div_iff₀ hspos).mpr
    nlinarith

example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) - 0 ≤ 1 ∧ (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => s) 1 t) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun _ : ℝ => (1 : ℝ)) 0 t) ∧
    ContinuousOn (fun _ : ℝ => (0 : ℝ)) (Set.uIcc (0 : ℝ) 1) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, (0 : ℝ) ≤ 0) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, (1 : ℝ) ≤ max |1| |0|) := by
  exact ⟨zero_le_one, by norm_num, one_ne_zero, one_pos, fun t _ => hasDerivAt_id t,
    fun t _ => hasDerivAt_const t 1, continuousOn_const, fun _ _ => le_rfl, by simp⟩

/-- The local phase estimate for either sign of the second derivative.
Source: arXiv:2412.09080v3, §5.5, the stationary and nonstationary intervals. -/
theorem unit_interval_phase_test {a b ρ κ : ℝ} {φ p q : ℝ → ℝ}
    (hab : a ≤ b) (hlen : b - a ≤ 1) (hρ : ρ ≠ 0) (hκ : 0 < κ)
    (hφ : ∀ t ∈ Set.uIcc a b, HasDerivAt φ (p t) t)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hcq : ContinuousOn q (Set.uIcc a b))
    (hq : (∀ t ∈ Set.uIcc a b, 0 ≤ q t) ∨ (∀ t ∈ Set.uIcc a b, q t ≤ 0))
    (hmin : ∀ t ∈ Set.uIcc a b, κ ≤ max |p t| |q t|) :
    ‖∫ t in a..b, oscillatoryKernel ρ φ t‖ ≤ 10 / Real.sqrt (|ρ| * κ) := by
  rcases hq with hq | hq
  · exact unit_interval_phase_test_nonneg hab hlen hρ hκ hφ hp hcq hq hmin
  · have h := unit_interval_phase_test_nonneg (ρ := -ρ) (φ := fun t => -φ t)
      (p := fun t => -p t) (q := fun t => -q t) hab hlen (neg_ne_zero.mpr hρ) hκ
      (fun t ht => (hφ t ht).neg) (fun t ht => (hp t ht).neg) hcq.neg
      (fun t ht => neg_nonneg.mpr (hq t ht))
      (fun t ht => by simpa only [abs_neg] using hmin t ht)
    have he : oscillatoryKernel (-ρ) (fun t => -φ t) = oscillatoryKernel ρ φ := by
      funext t
      simp only [oscillatoryKernel, neg_mul_neg]
    rw [he, abs_neg] at h
    exact h


example : (-1 / 2 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) - (-1 / 2) ≤ 1 ∧
    (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧
    (∀ t ∈ Set.uIcc (-1 / 2 : ℝ) (1 / 2), HasDerivAt (fun s : ℝ => s ^ 2) (2 * t) t) ∧
    (∀ t ∈ Set.uIcc (-1 / 2 : ℝ) (1 / 2), HasDerivAt (fun s : ℝ => 2 * s) 2 t) ∧
    ContinuousOn (fun _ : ℝ => (2 : ℝ)) (Set.uIcc (-1 / 2 : ℝ) (1 / 2)) ∧
    ((∀ t ∈ Set.uIcc (-1 / 2 : ℝ) (1 / 2), (0 : ℝ) ≤ 2) ∨
      (∀ t ∈ Set.uIcc (-1 / 2 : ℝ) (1 / 2), (2 : ℝ) ≤ 0)) ∧
    (∀ t ∈ Set.uIcc (-1 / 2 : ℝ) (1 / 2), (1 : ℝ) ≤ max |2 * t| |2|) := by
  refine ⟨by norm_num, by norm_num, one_ne_zero, one_pos, ?_, ?_, continuousOn_const,
    Or.inl (fun _ _ => by norm_num), ?_⟩
  · intro t _
    simpa using (hasDerivAt_id t).fun_pow 2
  · intro t _
    simpa using (hasDerivAt_id t).const_mul 2
  · intro t _
    exact (by norm_num : (1 : ℝ) ≤ |2|).trans (le_max_right _ _)

/-- The local phase estimate with a differentiable amplitude, using its
endpoint and total variation.
Source: arXiv:2412.09080v3, §5.5, the Gaussian amplitude estimates. -/
theorem weighted_unit_interval_phase_test {a b ρ κ : ℝ} {φ p q A B : ℝ → ℝ}
    (hab : a ≤ b) (hlen : b - a ≤ 1) (hρ : ρ ≠ 0) (hκ : 0 < κ) (hcφ : Continuous φ)
    (hφ : ∀ t ∈ Set.uIcc a b, HasDerivAt φ (p t) t)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hcq : ContinuousOn q (Set.uIcc a b))
    (hq : (∀ t ∈ Set.uIcc a b, 0 ≤ q t) ∨ (∀ t ∈ Set.uIcc a b, q t ≤ 0))
    (hmin : ∀ t ∈ Set.uIcc a b, κ ≤ max |p t| |q t|)
    (hA : ∀ t ∈ Set.uIcc a b, HasDerivAt A (B t) t)
    (hB : ContinuousOn B (Set.uIcc a b)) :
    ‖∫ t in a..b, oscillatoryKernel ρ φ t * (A t : ℂ)‖ ≤
      (10 / Real.sqrt (|ρ| * κ)) * (|A b| + ∫ t in a..b, |B t|) := by
  have hK : Continuous (oscillatoryKernel ρ φ) := by unfold oscillatoryKernel; fun_prop
  apply weighted_integral_le_of_primitive_bound hab hK hA hB
  intro t ht
  have htab : a ≤ t ∧ t ≤ b := by simpa only [Set.uIcc_of_le hab, Set.mem_Icc] using ht
  have hsub : Set.uIcc a t ⊆ Set.uIcc a b := by
    intro r hr
    rw [Set.uIcc_of_le htab.1] at hr
    rw [Set.uIcc_of_le hab]
    exact ⟨hr.1, hr.2.trans htab.2⟩
  apply unit_interval_phase_test htab.1 (by linarith) hρ hκ
    (fun r hr => hφ r (hsub hr)) (fun r hr => hp r (hsub hr)) (hcq.mono hsub)
    (hq.elim (fun h => Or.inl (fun r hr => h r (hsub hr)))
      (fun h => Or.inr (fun r hr => h r (hsub hr)))) (fun r hr => hmin r (hsub hr))

example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) - 0 ≤ 1 ∧ (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧
    Continuous (fun t : ℝ => t) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => s) 1 t) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun _ : ℝ => (1 : ℝ)) 0 t) ∧
    ContinuousOn (fun _ : ℝ => (0 : ℝ)) (Set.uIcc (0 : ℝ) 1) ∧
    ((∀ t ∈ Set.uIcc (0 : ℝ) 1, (0 : ℝ) ≤ 0) ∨
      (∀ t ∈ Set.uIcc (0 : ℝ) 1, (0 : ℝ) ≤ 0)) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, (1 : ℝ) ≤ max |1| |0|) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => s) 1 t) ∧
    ContinuousOn (fun _ : ℝ => (1 : ℝ)) (Set.uIcc (0 : ℝ) 1) := by
  exact ⟨zero_le_one, by norm_num, one_ne_zero, one_pos, continuous_id,
    fun t _ => hasDerivAt_id t, fun t _ => hasDerivAt_const t 1, continuousOn_const,
    Or.inl (fun _ _ => le_rfl), by simp, fun t _ => hasDerivAt_id t, continuousOn_const⟩

end Transformer.Modes

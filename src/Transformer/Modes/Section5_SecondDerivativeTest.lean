import Transformer.Modes.Section5_StationaryPhase

/-!
# The second derivative test for oscillatory integrals

In arXiv:2412.09080v3, §5.5, the stationary-phase estimate uses a lower
bound for `|φ''|` on each stationary interval. Here that estimate is proved
with the explicit constant `10`: if `|φ''| ≥ κ > 0`, then the oscillatory
integral has norm at most `10 / sqrt (|ρ| κ)`.

For a positive second derivative, choose the first-derivative threshold
`δ = sqrt (κ / |ρ|)` in `Section5_StationaryPhase`. The two outer intervals
and the middle interval then have the same square-root scale. Negating
the phase and frequency handles a negative second derivative. Finally,
continuity and the intermediate value theorem show that a second
derivative bounded away from zero has one sign throughout the interval.

The estimate is uniform in the interval's length and in the location
of a stationary point; the interval may also have no stationary point.
The derivatives are linked to the phase by `HasDerivAt` on the entire
closed interval. The continuity of the second derivative is stated
explicitly, as required to infer its sign. The analytic phase in the
source satisfies all these regularity hypotheses.

Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`.
-/

open Real MeasureTheory
open scoped Interval

namespace Transformer.Modes

/-- The second derivative test for a second derivative at least `κ`.
Source: arXiv:2412.09080v3, §5.5, the stationary-phase bound. -/
theorem second_derivative_test_nonneg {a b ρ κ : ℝ} {φ p q : ℝ → ℝ}
    (hab : a ≤ b) (hρ : ρ ≠ 0) (hκ : 0 < κ)
    (hφ : ∀ t ∈ Set.uIcc a b, HasDerivAt φ (p t) t)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hcq : ContinuousOn q (Set.uIcc a b))
    (hmin : ∀ t ∈ Set.uIcc a b, κ ≤ q t) :
    ‖∫ t in a..b, oscillatoryKernel ρ φ t‖ ≤ 10 / Real.sqrt (|ρ| * κ) := by
  have hρpos : 0 < |ρ| := abs_pos.mpr hρ
  let δ := Real.sqrt (κ / |ρ|)
  have hδ : 0 < δ := Real.sqrt_pos.mpr (div_pos hκ hρpos)
  have hsq : δ ^ 2 = κ / |ρ| := Real.sq_sqrt (div_nonneg hκ.le hρpos.le)
  have hmul : |ρ| * δ ^ 2 = κ := by rw [hsq]; field_simp
  have h := oscillatory_integral_le_of_sublevel_derivative hab hρ hδ hκ hφ hp hcq
    (fun t ht => hκ.le.trans (hmin t ht)) (fun t ht _ => hmin t ht)
  exact h.trans_eq (oscillatory_threshold_balance hρ hδ hκ hmul)

example : (-1 : ℝ) ≤ 1 ∧ (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, HasDerivAt (fun s : ℝ => s ^ 2) (2 * t) t) ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, HasDerivAt (fun s : ℝ => 2 * s) 2 t) ∧
    ContinuousOn (fun _ : ℝ => (2 : ℝ)) (Set.uIcc (-1 : ℝ) 1) ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, (1 : ℝ) ≤ 2) := by
  refine ⟨by norm_num, one_ne_zero, one_pos, ?_, ?_, continuousOn_const,
    fun _ _ => by norm_num⟩
  · intro t _
    simpa using (hasDerivAt_id t).fun_pow 2
  · intro t _
    simpa using (hasDerivAt_id t).const_mul 2

/-- The second derivative test for a second derivative at most `-κ`.
Source: arXiv:2412.09080v3, §5.5, the stationary-phase bound. -/
theorem second_derivative_test_nonpos {a b ρ κ : ℝ} {φ p q : ℝ → ℝ}
    (hab : a ≤ b) (hρ : ρ ≠ 0) (hκ : 0 < κ)
    (hφ : ∀ t ∈ Set.uIcc a b, HasDerivAt φ (p t) t)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hcq : ContinuousOn q (Set.uIcc a b))
    (hmin : ∀ t ∈ Set.uIcc a b, q t ≤ -κ) :
    ‖∫ t in a..b, oscillatoryKernel ρ φ t‖ ≤ 10 / Real.sqrt (|ρ| * κ) := by
  have h := second_derivative_test_nonneg (ρ := -ρ) (φ := fun t => -φ t)
    (p := fun t => -p t) (q := fun t => -q t) hab (neg_ne_zero.mpr hρ) hκ
    (fun t ht => (hφ t ht).neg) (fun t ht => (hp t ht).neg) hcq.neg
    (fun t ht => by have := hmin t ht; linarith)
  have he : oscillatoryKernel (-ρ) (fun t => -φ t) = oscillatoryKernel ρ φ := by
    funext t
    simp only [oscillatoryKernel, neg_mul_neg]
  rw [he, abs_neg] at h
  exact h

example : (-1 : ℝ) ≤ 1 ∧ (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, HasDerivAt (fun s : ℝ => -(s ^ 2)) (-(2 * t)) t) ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, HasDerivAt (fun s : ℝ => -(2 * s)) (-2) t) ∧
    ContinuousOn (fun _ : ℝ => (-2 : ℝ)) (Set.uIcc (-1 : ℝ) 1) ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, (-2 : ℝ) ≤ -1) := by
  refine ⟨by norm_num, one_ne_zero, one_pos, ?_, ?_, continuousOn_const,
    fun _ _ => by norm_num⟩
  · intro t _
    simpa using ((hasDerivAt_id t).fun_pow 2).fun_neg
  · intro t _
    simpa using ((hasDerivAt_id t).const_mul 2).fun_neg

/-- The second derivative test under the source's absolute-value lower
bound; continuity supplies the constant sign of the second derivative.
Source: arXiv:2412.09080v3, §5.5, the intervals `J_j` and `c_*`. -/
theorem second_derivative_test {a b ρ κ : ℝ} {φ p q : ℝ → ℝ}
    (hab : a ≤ b) (hρ : ρ ≠ 0) (hκ : 0 < κ)
    (hφ : ∀ t ∈ Set.uIcc a b, HasDerivAt φ (p t) t)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hcq : ContinuousOn q (Set.uIcc a b))
    (hmin : ∀ t ∈ Set.uIcc a b, κ ≤ |q t|) :
    ‖∫ t in a..b, oscillatoryKernel ρ φ t‖ ≤ 10 / Real.sqrt (|ρ| * κ) := by
  have hne : ∀ t ∈ Set.uIcc a b, q t ≠ 0 := by
    intro t ht he
    have h := hmin t ht
    simp only [he, abs_zero] at h
    exact hκ.not_ge h
  have hsub (t : ℝ) (ht : t ∈ Set.uIcc a b) : Set.Icc a t ⊆ Set.uIcc a b := by
    rw [Set.uIcc_of_le hab] at ht
    intro r hr
    rw [Set.uIcc_of_le hab]
    exact ⟨hr.1, hr.2.trans ht.2⟩
  by_cases hqa : 0 ≤ q a
  · apply second_derivative_test_nonneg hab hρ hκ hφ hp hcq
    intro t ht
    by_cases hqt : 0 ≤ q t
    · simpa only [abs_of_nonneg hqt] using hmin t ht
    have hta : a ≤ t := by
      rw [Set.uIcc_of_le hab] at ht
      exact ht.1
    obtain ⟨c, hc, he⟩ := intermediate_value_Icc' hta (hcq.mono (hsub t ht))
      (show (0 : ℝ) ∈ Set.Icc (q t) (q a) from ⟨(not_le.mp hqt).le, hqa⟩)
    exact False.elim (hne c (hsub t ht hc) he)
  · apply second_derivative_test_nonpos hab hρ hκ hφ hp hcq
    intro t ht
    by_cases hqt : q t ≤ 0
    · have h := hmin t ht
      rw [abs_of_nonpos hqt] at h
      linarith
    have hta : a ≤ t := by
      rw [Set.uIcc_of_le hab] at ht
      exact ht.1
    obtain ⟨c, hc, he⟩ := intermediate_value_Icc hta (hcq.mono (hsub t ht))
      (show (0 : ℝ) ∈ Set.Icc (q a) (q t) from ⟨(not_le.mp hqa).le, (not_le.mp hqt).le⟩)
    exact False.elim (hne c (hsub t ht hc) he)

example : (-1 : ℝ) ≤ 1 ∧ (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, HasDerivAt (fun s : ℝ => s ^ 2) (2 * t) t) ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, HasDerivAt (fun s : ℝ => 2 * s) 2 t) ∧
    ContinuousOn (fun _ : ℝ => (2 : ℝ)) (Set.uIcc (-1 : ℝ) 1) ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, (1 : ℝ) ≤ |2|) := by
  refine ⟨by norm_num, one_ne_zero, one_pos, ?_, ?_, continuousOn_const,
    fun _ _ => by norm_num⟩
  · intro t _
    simpa using (hasDerivAt_id t).fun_pow 2
  · intro t _
    simpa using (hasDerivAt_id t).const_mul 2

end Transformer.Modes

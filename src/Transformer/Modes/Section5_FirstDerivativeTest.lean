import Transformer.Modes.Section5_NonstationaryPhase

/-!
# The first derivative test for a nonstationary phase

The proof of `eq:uniform-decay` in arXiv:2412.09080v3, §5.5, uses
nonstationary integration by parts after removing the stationary intervals.
This module gives the quantitative first derivative estimate needed for
that argument on any interval where `|φ'| ≥ δ > 0` and `φ''` has one sign.

The derivative of `1/φ'` integrates to its endpoint difference. Thus its
total variation is at most `2/δ`; adding the two endpoint contributions
from `Section5_NonstationaryPhase` gives the bound `4/(|ρ| δ)`. This
constant is independent of the interval's length. Both signs of `φ''` are
allowed, by negating the phase and frequency together.

The sign condition is imposed on the actual second derivative through its
`HasDerivAt` hypothesis, rather than on an unrelated function. The source's
smooth phase satisfies these differentiability and continuity conditions
on each interval between zeros of its second derivative.
Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`.
-/

open Real MeasureTheory
open scoped Interval

namespace Transformer.Modes

/-- The variation of the reciprocal of a nonzero monotone phase derivative
is controlled by its two endpoint values.
Source: arXiv:2412.09080v3, §5.5, nonstationary integration by parts. -/
theorem reciprocal_derivative_integral_le {a b δ : ℝ} {p q : ℝ → ℝ}
    (hδ : 0 < δ) (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hcq : ContinuousOn q (Set.uIcc a b))
    (hq : ∀ t ∈ Set.uIcc a b, 0 ≤ q t)
    (hmin : ∀ t ∈ Set.uIcc a b, δ ≤ |p t|) :
    (∫ t in a..b, |(-q t) / (p t) ^ 2|) ≤ 2 / δ := by
  have hne : ∀ t ∈ Set.uIcc a b, p t ≠ 0 := by
    intro t ht hz
    have h := hmin t ht
    simp only [hz, abs_zero] at h
    exact hδ.not_ge h
  have hcp : ContinuousOn p (Set.uIcc a b) :=
    continuousOn_of_forall_continuousAt fun t ht => (hp t ht).continuousAt
  have hc : ContinuousOn (fun t => -q t / (p t) ^ 2) (Set.uIcc a b) :=
    hcq.neg.div (hcp.pow 2) (fun t ht => pow_ne_zero 2 (hne t ht))
  have hi := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (fun t ht => (hp t ht).inv (hne t ht)) hc.intervalIntegrable
  have heq : (∫ t in a..b, |(-q t) / (p t) ^ 2|) =
      -(∫ t in a..b, -q t / (p t) ^ 2) := by
    rw [← intervalIntegral.integral_neg]
    apply intervalIntegral.integral_congr
    intro t ht
    change |(-q t) / (p t) ^ 2| = -((-q t) / (p t) ^ 2)
    rw [neg_div, abs_neg, abs_of_nonneg (div_nonneg (hq t ht) (sq_nonneg _))]
    ring
  have he (t : ℝ) (ht : t ∈ Set.uIcc a b) : |(p t)⁻¹| ≤ 1 / δ := by
    rw [abs_inv, ← one_div]
    exact one_div_le_one_div_of_le hδ (hmin t ht)
  have ha := (abs_le.1 (he a (Set.left_mem_uIcc))).2
  have hb := (abs_le.1 (he b (Set.right_mem_uIcc))).1
  rw [heq, hi]
  change -((p b)⁻¹ - (p a)⁻¹) ≤ 2 / δ
  calc _ ≤ (1 / δ) + (1 / δ) := by linarith
    _ = _ := by ring

example : (0 : ℝ) < 1 ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun _ : ℝ => (1 : ℝ)) 0 t) ∧
    ContinuousOn (fun _ : ℝ => (0 : ℝ)) (Set.uIcc (0 : ℝ) 1) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, (0 : ℝ) ≤ 0) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, (1 : ℝ) ≤ |1|) := by
  exact ⟨one_pos, fun t _ => hasDerivAt_const t 1, continuousOn_const,
    fun _ _ => le_rfl, by simp⟩

/-- The first derivative estimate when the second derivative is
nonnegative, retaining the boundary terms from integration by parts.
Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`. -/
theorem first_derivative_test_nonneg {a b ρ δ : ℝ} {φ p q : ℝ → ℝ}
    (hab : a ≤ b) (hρ : ρ ≠ 0) (hδ : 0 < δ)
    (hφ : ∀ t ∈ Set.uIcc a b, HasDerivAt φ (p t) t)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hcq : ContinuousOn q (Set.uIcc a b))
    (hq : ∀ t ∈ Set.uIcc a b, 0 ≤ q t)
    (hmin : ∀ t ∈ Set.uIcc a b, δ ≤ |p t|) :
    ‖∫ t in a..b, oscillatoryKernel ρ φ t‖ ≤ 4 / (|ρ| * δ) := by
  have hne : ∀ t ∈ Set.uIcc a b, p t ≠ 0 := by
    intro t ht hz
    have h := hmin t ht
    simp only [hz, abs_zero] at h
    exact hδ.not_ge h
  have h := norm_oscillatory_integral_le hab hρ hφ hp
    (A := fun _ => 1) (B := fun _ => 0) (fun t _ => hasDerivAt_const t 1)
    hcq continuousOn_const hne
  simp only [Complex.ofReal_one, mul_one, zero_mul, one_mul, zero_sub] at h
  have he (t : ℝ) (ht : t ∈ Set.uIcc a b) : |1 / p t| ≤ 1 / δ := by
    rw [abs_div, abs_one]
    exact one_div_le_one_div_of_le hδ (hmin t ht)
  have ha := he a Set.left_mem_uIcc
  have hb := he b Set.right_mem_uIcc
  have hi := reciprocal_derivative_integral_le hδ hp hcq hq hmin
  calc _ ≤ _ := h
    _ ≤ ((1 / δ) + (1 / δ) + 2 / δ) / |ρ| := by gcongr
    _ = _ := by ring

example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => s) 1 t) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun _ : ℝ => (1 : ℝ)) 0 t) ∧
    ContinuousOn (fun _ : ℝ => (0 : ℝ)) (Set.uIcc (0 : ℝ) 1) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, (0 : ℝ) ≤ 0) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, (1 : ℝ) ≤ |1|) := by
  exact ⟨zero_le_one, one_ne_zero, one_pos, fun t _ => hasDerivAt_id t,
    fun t _ => hasDerivAt_const t 1, continuousOn_const, fun _ _ => le_rfl, by simp⟩

/-- The same first derivative estimate for a nonpositive second
derivative. Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`. -/
theorem first_derivative_test_nonpos {a b ρ δ : ℝ} {φ p q : ℝ → ℝ}
    (hab : a ≤ b) (hρ : ρ ≠ 0) (hδ : 0 < δ)
    (hφ : ∀ t ∈ Set.uIcc a b, HasDerivAt φ (p t) t)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hcq : ContinuousOn q (Set.uIcc a b))
    (hq : ∀ t ∈ Set.uIcc a b, q t ≤ 0)
    (hmin : ∀ t ∈ Set.uIcc a b, δ ≤ |p t|) :
    ‖∫ t in a..b, oscillatoryKernel ρ φ t‖ ≤ 4 / (|ρ| * δ) := by
  have h := first_derivative_test_nonneg (ρ := -ρ) (φ := fun t => -φ t)
    (p := fun t => -p t) (q := fun t => -q t) hab (neg_ne_zero.mpr hρ) hδ
    (fun t ht => (hφ t ht).neg) (fun t ht => (hp t ht).neg) hcq.neg
    (fun t ht => neg_nonneg.mpr (hq t ht)) (fun t ht => by simpa only [abs_neg] using hmin t ht)
  have he : oscillatoryKernel (-ρ) (fun t => -φ t) = oscillatoryKernel ρ φ := by
    funext t
    simp only [oscillatoryKernel, neg_mul_neg]
  rw [he, abs_neg] at h
  exact h

example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => s) 1 t) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun _ : ℝ => (1 : ℝ)) 0 t) ∧
    ContinuousOn (fun _ : ℝ => (0 : ℝ)) (Set.uIcc (0 : ℝ) 1) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, (0 : ℝ) ≤ 0) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, (1 : ℝ) ≤ |1|) := by
  exact ⟨zero_le_one, one_ne_zero, one_pos, fun t _ => hasDerivAt_id t,
    fun t _ => hasDerivAt_const t 1, continuousOn_const, fun _ _ => le_rfl, by simp⟩

/-- The first derivative test for either monotonicity direction of the
phase derivative. Source: arXiv:2412.09080v3, §5.5, `eq:uniform-decay`. -/
theorem first_derivative_test {a b ρ δ : ℝ} {φ p q : ℝ → ℝ}
    (hab : a ≤ b) (hρ : ρ ≠ 0) (hδ : 0 < δ)
    (hφ : ∀ t ∈ Set.uIcc a b, HasDerivAt φ (p t) t)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hcq : ContinuousOn q (Set.uIcc a b))
    (hq : (∀ t ∈ Set.uIcc a b, 0 ≤ q t) ∨ (∀ t ∈ Set.uIcc a b, q t ≤ 0))
    (hmin : ∀ t ∈ Set.uIcc a b, δ ≤ |p t|) :
    ‖∫ t in a..b, oscillatoryKernel ρ φ t‖ ≤ 4 / (|ρ| * δ) := by
  rcases hq with hq | hq
  · exact first_derivative_test_nonneg hab hρ hδ hφ hp hcq hq hmin
  · exact first_derivative_test_nonpos hab hρ hδ hφ hp hcq hq hmin

example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => s) 1 t) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun _ : ℝ => (1 : ℝ)) 0 t) ∧
    ContinuousOn (fun _ : ℝ => (0 : ℝ)) (Set.uIcc (0 : ℝ) 1) ∧
    ((∀ t ∈ Set.uIcc (0 : ℝ) 1, (0 : ℝ) ≤ 0) ∨
      (∀ t ∈ Set.uIcc (0 : ℝ) 1, (0 : ℝ) ≤ 0)) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, (1 : ℝ) ≤ |1|) := by
  exact ⟨zero_le_one, one_ne_zero, one_pos, fun t _ => hasDerivAt_id t,
    fun t _ => hasDerivAt_const t 1, continuousOn_const,
    Or.inl (fun _ _ => le_rfl), by simp⟩

end Transformer.Modes

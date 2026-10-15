/-
# The original interaction curvature and its decreasing central branch

ArXiv:2411.04990v2, Appendix B, `lemma:interaction`, and §5,
proof of `thm:fixed_centers`. The original curvature is `g=h'`.
Its actual derivative is the original exponential times the sine
times `beta²*sin² - 3*beta*cos - 1`.

When the original curvature is nonnegative, its defining formula
forces `beta*sin² <= cos` and hence a strictly negative derivative
factor. On the original central branch between zero and the actual
interaction peak, the sine is positive. Thus the original curvature
strictly decreases from one to zero there. In particular, every
original distance inside a prescribed subpeak neighborhood has
curvature at least the original curvature at that neighborhood's
edge. This is the actual lower bound used in the frozen-center
stability calculation, for every positive original temperature.
-/

import Transformer.Causal.SectionB_PeakTemperature

open Real Set

namespace Transformer.Causal

/-- The derivative of the actual original interaction curvature has its exact original trigonometric factor.
Source: arXiv:2411.04990v2, Appendix B, the interaction calculation used in §5's proof of `thm:fixed_centers`. -/
theorem hasDerivAt_g_pot (β x : ℝ) : HasDerivAt (g_pot β)
    (Real.exp (β * (Real.cos x - 1)) * Real.sin x *
      (β ^ 2 * Real.sin x ^ 2 - 3 * β * Real.cos x - 1)) x := by
  have he := (((Real.hasDerivAt_cos x).sub_const 1).const_mul β).exp
  have hp := (Real.hasDerivAt_cos x).sub (((Real.hasDerivAt_sin x).pow 2).const_mul β)
  have h := he.mul hp
  apply h.congr_deriv
  simp only [Pi.sub_apply, Pi.pow_apply, pow_one, Nat.cast_ofNat, Nat.reduceSub]
  ring

/-- The actual scalar derivative is the proved original curvature-derivative expression.
Source: arXiv:2411.04990v2, Appendix B, `g=h'`, as used in §5, `thm:fixed_centers`. -/
theorem deriv_g_pot (β x : ℝ) : deriv (g_pot β) x =
    Real.exp (β * (Real.cos x - 1)) * Real.sin x *
      (β ^ 2 * Real.sin x ^ 2 - 3 * β * Real.cos x - 1) :=
  (hasDerivAt_g_pot β x).deriv

/-- The actual original curvature equals one at its original center, for every original temperature.
Source: arXiv:2411.04990v2, Appendix B, the definition of `g` in `lemma:interaction`. -/
theorem g_pot_zero (β : ℝ) : g_pot β 0 = 1 := by simp [g_pot]

/-- The actual original curvature derivative vanishes at the original center.
Source: arXiv:2411.04990v2, Appendix B, evenness of `g` in `lemma:interaction`. -/
theorem deriv_g_pot_zero (β : ℝ) : deriv (g_pot β) 0 = 0 := by rw [deriv_g_pot]; simp

/-- The actual original curvature vanishes exactly at the prescribed original peak point.
Source: arXiv:2411.04990v2, Appendix B, the defining peak equation of `lemma:interaction` (3). -/
theorem g_pot_peak_eq_zero (β : ℝ) (hβ : 0 < β) : g_pot β (interactionPeak β) = 0 := by
  rw [g_pot_peak_factor β hβ, (interactionPeak_properties β hβ).2.2, sub_self, zero_mul, mul_zero]

/-- The actual original central curvature is nonnegative up to and including the actual original peak.
Source: arXiv:2411.04990v2, Appendix B, the sign change of `g` in `lemma:interaction` (3). -/
theorem g_pot_nonneg_of_le_peak (β : ℝ) (hβ : 0 < β) (x : ℝ) (hx : 0 ≤ x)
    (hp : x ≤ interactionPeak β) : 0 ≤ g_pot β x := by
  rcases lt_or_eq_of_le hp with h | h
  · have ht := (interactionPeak_properties β hβ).2.1
    exact ((g_pot_pos_iff_lt_peak β hβ x ⟨hx, by linarith [Real.pi_pos]⟩).mpr h).le
  · rw [h, g_pot_peak_eq_zero β hβ]

/-- Whenever the actual original curvature is nonnegative, its actual derivative factor is at most minus one.
Source: arXiv:2411.04990v2, Appendix B, the central interaction branch used in §5, `thm:fixed_centers`. -/
theorem g_pot_derivative_factor_le_neg_one (β : ℝ) (hβ : 0 < β) (x : ℝ) (hg : 0 ≤ g_pot β x) :
    β ^ 2 * Real.sin x ^ 2 - 3 * β * Real.cos x - 1 ≤ -1 := by
  have hp : 0 ≤ Real.cos x - β * Real.sin x ^ 2 :=
    (mul_nonneg_iff_of_pos_left (Real.exp_pos _)).mp hg
  have hc : 0 ≤ Real.cos x := by nlinarith [mul_nonneg hβ.le (sq_nonneg (Real.sin x))]
  have hm := mul_le_mul_of_nonneg_left (show β * Real.sin x ^ 2 ≤ Real.cos x by linarith) hβ.le
  nlinarith [mul_nonneg hβ.le hc]

/-- The actual original curvature has strictly negative derivative throughout the positive central interaction branch.
Source: arXiv:2411.04990v2, Appendix B, the central monotonicity used in §5, `thm:fixed_centers`. -/
theorem deriv_g_pot_neg_on_core (β : ℝ) (hβ : 0 < β) (x : ℝ) (hx : 0 < x)
    (hp : x ≤ interactionPeak β) : deriv (g_pot β) x < 0 := by
  rw [deriv_g_pot]
  have ht := (interactionPeak_properties β hβ).2.1
  have hs : 0 < Real.sin x := Real.sin_pos_of_pos_of_lt_pi hx (by linarith [Real.pi_pos])
  apply mul_neg_of_pos_of_neg (mul_pos (Real.exp_pos _) hs)
  have h := g_pot_derivative_factor_le_neg_one β hβ x (g_pot_nonneg_of_le_peak β hβ x hx.le hp)
  linarith

/-- The actual original curvature strictly decreases from its center to its actual original interaction peak.
Source: arXiv:2411.04990v2, Appendix B, the central curvature bound used in §5, `thm:fixed_centers`. -/
theorem g_pot_strictAntiOn_core (β : ℝ) (hβ : 0 < β) :
    StrictAntiOn (g_pot β) (Icc 0 (interactionPeak β)) := by
  have hc : Continuous (g_pot β) := by unfold g_pot; fun_prop
  apply strictAntiOn_of_deriv_neg (convex_Icc _ _) hc.continuousOn
  intro x hx
  rw [interior_Icc] at hx
  exact deriv_g_pot_neg_on_core β hβ x hx.1 hx.2.le

/-- Every nonzero original distance in the central branch has curvature strictly below its original center value one.
Source: arXiv:2411.04990v2, Appendix B, the central interaction branch used in §5, `thm:fixed_centers`. -/
theorem g_pot_lt_one_on_core (β : ℝ) (hβ : 0 < β) (x : ℝ) (hx : 0 < x)
    (hp : x ≤ interactionPeak β) : g_pot β x < 1 := by
  have h := g_pot_strictAntiOn_core β hβ (show (0 : ℝ) ∈ Icc 0 (interactionPeak β) from
    ⟨le_rfl, (interactionPeak_properties β hβ).1.le⟩) (show x ∈ Icc 0 (interactionPeak β) from ⟨hx.le, hp⟩) hx
  simpa only [g_pot_zero] using h

/-- Every actual original subpeak distance has curvature at least the actual original curvature at the neighborhood edge.
Source: arXiv:2411.04990v2, §5, `g(phi_k-theta_1) >= g(epsilon/sqrt(beta))` in `thm:fixed_centers`. -/
theorem g_pot_core_lower (β : ℝ) (hβ : 0 < β) (x ε : ℝ) (hx : 0 ≤ x) (hxe : x ≤ ε)
    (he : ε ≤ interactionPeak β) : g_pot β ε ≤ g_pot β x :=
  (g_pot_strictAntiOn_core β hβ).antitoneOn ⟨hx, hxe.trans he⟩ ⟨hx.trans hxe, he⟩ hxe

/-- At every original nonnegative temperature, the actual original curvature is at most one globally.
Source: arXiv:2411.04990v2, Appendix B, the original explicit curvature formula in `lemma:interaction`. -/
theorem g_pot_le_one (β : ℝ) (hβ : 0 ≤ β) (x : ℝ) : g_pot β x ≤ 1 := by
  have hp : Real.cos x - β * Real.sin x ^ 2 ≤ 1 := by
    nlinarith [Real.cos_le_one x, mul_nonneg hβ (sq_nonneg (Real.sin x))]
  have he : Real.exp (β * (Real.cos x - 1)) ≤ 1 :=
    Real.exp_le_one_iff.mpr (mul_nonpos_of_nonneg_of_nonpos hβ (sub_nonpos.mpr (Real.cos_le_one x)))
  calc
    _ ≤ Real.exp (β * (Real.cos x - 1)) * 1 :=
      mul_le_mul_of_nonneg_left hp (Real.exp_pos _).le
    _ ≤ 1 := by simpa only [mul_one] using he

/-- The actual derivative has a quantitative negative bound wherever the original semicircle curvature is nonnegative.
Source: arXiv:2411.04990v2, Appendix B, the central curvature estimate used in §5, `thm:fixed_centers`. -/
theorem deriv_g_pot_le_neg_exp_sin (β : ℝ) (hβ : 0 < β) (x : ℝ) (hx : x ∈ Icc 0 Real.pi)
    (hg : 0 ≤ g_pot β x) :
    deriv (g_pot β) x ≤ -(Real.exp (β * (Real.cos x - 1)) * Real.sin x) := by
  rw [deriv_g_pot]
  have hs : 0 ≤ Real.sin x := Real.sin_nonneg_of_mem_Icc hx
  have h := mul_le_mul_of_nonneg_left (g_pot_derivative_factor_le_neg_one β hβ x hg)
    (mul_nonneg (Real.exp_pos (β * (Real.cos x - 1))).le hs)
  simpa only [mul_neg, mul_one] using h

/-- A genuine original subpeak neighborhood has a strictly positive uniform lower curvature throughout its central interval.
Source: arXiv:2411.04990v2, §5, the positive central contribution in the stability calculation for `thm:fixed_centers`. -/
theorem g_pot_core_lower_pos (β : ℝ) (hβ : 0 < β) (x ε : ℝ) (hx : 0 ≤ x) (hxe : x ≤ ε)
    (he : ε < interactionPeak β) : 0 < g_pot β ε ∧ g_pot β ε ≤ g_pot β x := by
  have ht := (interactionPeak_properties β hβ).2.1
  have hε : ε ∈ Icc 0 Real.pi := ⟨hx.trans hxe, by linarith [Real.pi_pos]⟩
  exact ⟨(g_pot_pos_iff_lt_peak β hβ ε hε).mpr he, g_pot_core_lower β hβ x ε hx hxe he.le⟩

/-- The positive-temperature and positive-curvature premises are jointly realized at the actual original center.
Source: arXiv:2411.04990v2, Appendix B, the original central interaction in `lemma:interaction`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ∈ Icc 0 Real.pi ∧ 0 ≤ g_pot 1 0 :=
  ⟨one_pos, ⟨le_rfl, Real.pi_pos.le⟩, by rw [g_pot_zero]; exact zero_le_one⟩

/-- Positive interior original distances jointly realize the derivative and neighborhood-comparison hypotheses.
Source: arXiv:2411.04990v2, §5, the central interaction neighborhood in `thm:fixed_centers`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < interactionPeak 1 / 4 ∧
    interactionPeak 1 / 4 ≤ interactionPeak 1 / 2 ∧ interactionPeak 1 / 2 < interactionPeak 1 := by
  have hp := (interactionPeak_properties 1 one_pos).1
  exact ⟨one_pos, by linarith, by linarith, by linarith⟩

end Transformer.Causal

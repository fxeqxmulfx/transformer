module

public import Transformer.Landscape.PhaseModel.Basic
public import Transformer.Landscape.SaddleAvoidance.Necessary

/-!
# Phase model — critical points near a zero

At a zero `z` of the phase energy every relation lies in `2πℤ`, so a point
`ψ` sees the relations `x_r = ⟪m_r, ψ - z⟫` through their sines and cosines
(`sin_cos_inner_sub`). The Hessian has the quadratic form
`⟪H h, h⟫ = Σ_r cos ⟪m_r, ψ⟫ ⟪m_r, h⟫²` (`inner_fderiv_gradient_phaseEnergy`),
and at a critical point `Σ_r x_r sin x_r = ⟪∇V(ψ), ψ - z⟫ = 0`
(`sum_inner_sub_mul_sin`). As `x sin x > 0` for `0 < |x| < π`, a critical
point with every `|x_r| < π` has every `x_r = 0`
(`inner_sub_eq_zero_of_gradient_eq_zero`). As `x² cos x < x sin x` for
`0 < |x| ≤ π` (`sq_mul_cos_lt_mul_sin`), at a local minimizer `|x_r| ≤ π`
suffices (`inner_sub_eq_zero_of_isLocalMin`): the Hessian along `ψ - z`
would otherwise be negative.
-/

@[expose] public section

open Real

namespace Transformer.Landscape.PhaseModel

variable {E R : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [Fintype R] (m : R → E)

local notation "⟪" x ", " y "⟫" => inner ℝ x y

/-- At a zero `z`, the relations lie in `2πℤ`, so the sine and cosine of a
relation of `ψ` are those of `⟪m_r, ψ - z⟫`. -/
theorem sin_cos_inner_sub {z : E} (hz : phaseEnergy m z = 0) (ψ : E) (r : R) :
    sin ⟪m r, ψ⟫ = sin ⟪m r, ψ - z⟫ ∧ cos ⟪m r, ψ⟫ = cos ⟪m r, ψ - z⟫ := by
  obtain ⟨k, hk⟩ := (phaseEnergy_eq_zero_iff m z).1 hz r
  rw [inner_sub_right, hk, sin_sub_int_mul_two_pi, cos_sub_int_mul_two_pi]
  exact ⟨rfl, rfl⟩

variable [CompleteSpace E]

/-- **The quadratic form of the Hessian of the phase energy**:
`⟪H h, h⟫ = Σ_r cos ⟪m_r, ψ⟫ ⟪m_r, h⟫²`. -/
theorem inner_fderiv_gradient_phaseEnergy (ψ h : E) :
    ⟪fderiv ℝ (gradient (phaseEnergy m)) ψ h, h⟫ = ∑ r, cos ⟪m r, ψ⟫ * ⟪m r, h⟫ ^ 2 := by
  rw [(hasFDerivAt_gradient_phaseEnergy m ψ).fderiv]
  simp only [FunLike.coe_sum, Finset.sum_apply, ContinuousLinearMap.smulRight_apply,
    smul_apply, innerSL_apply_apply, smul_eq_mul, sum_inner,
    real_inner_smul_left]
  exact Finset.sum_congr rfl fun r _ => by ring

/-- At a critical point `ψ` and a zero `z`,
`Σ_r ⟪m_r, ψ - z⟫ sin ⟪m_r, ψ - z⟫ = ⟪∇V(ψ), ψ - z⟫ = 0`. -/
theorem sum_inner_sub_mul_sin {ψ z : E} (hψ : gradient (phaseEnergy m) ψ = 0)
    (hz : phaseEnergy m z = 0) : ∑ r, ⟪m r, ψ - z⟫ * sin ⟪m r, ψ - z⟫ = 0 := by
  have h0 := congrArg (fun v => ⟪v, ψ - z⟫) hψ
  simp only [gradient_phaseEnergy, sum_inner, real_inner_smul_left, inner_zero_left] at h0
  rw [← h0]
  exact Finset.sum_congr rfl fun r _ => by rw [(sin_cos_inner_sub m hz ψ r).1, mul_comm]

/-- `x sin x ≥ 0` for `|x| ≤ π`. -/
theorem mul_sin_nonneg_of_abs_le {x : ℝ} (h : |x| ≤ π) : 0 ≤ x * sin x := by
  rcases le_total 0 x with hx | hx
  · exact mul_nonneg hx (sin_nonneg_of_nonneg_of_le_pi hx (le_of_abs_le h))
  · exact mul_nonneg_of_nonpos_of_nonpos hx (sin_nonpos_of_nonpos_of_neg_pi_le hx (neg_le_of_abs_le h))

/-- `x² cos x < x sin x` for `0 < |x| ≤ π`: below `π/2` this is `x < tan x`,
above it `cos x ≤ 0 ≤ sin x`, not both zero. -/
theorem sq_mul_cos_lt_mul_sin {x : ℝ} (hx : x ≠ 0) (h : |x| ≤ π) : x ^ 2 * cos x < x * sin x := by
  wlog hpos : 0 < x generalizing x
  · have h1 := this (neg_ne_zero.2 hx) (by rwa [abs_neg])
      (neg_pos.2 (lt_of_le_of_ne (not_lt.1 hpos) hx))
    simpa only [neg_sq, cos_neg, sin_neg, neg_mul_neg] using h1
  have hxpi : x ≤ π := le_of_abs_le h
  rcases lt_or_ge x (π / 2) with hx2 | hx2
  · have hcos : 0 < cos x := cos_pos_of_mem_Ioo ⟨by linarith [pi_pos], hx2⟩
    have htan := lt_tan hpos hx2
    rw [tan_eq_sin_div_cos, lt_div_iff₀ hcos] at htan
    nlinarith
  · have hcos : cos x ≤ 0 := cos_nonpos_of_pi_div_two_le_of_le hx2 (by linarith [pi_pos])
    have hsin : 0 ≤ sin x := sin_nonneg_of_nonneg_of_le_pi hpos.le hxpi
    rcases hcos.lt_or_eq with hc | hc
    · nlinarith [mul_neg_of_pos_of_neg (pow_pos hpos 2) hc, mul_nonneg hpos.le hsin]
    · have hs : sin x ≠ 0 := fun hs => by
        have h1 := sin_sq_add_cos_sq x
        rw [hs, hc] at h1
        norm_num at h1
      rw [hc, mul_zero]
      exact mul_pos hpos (lt_of_le_of_ne hsin (Ne.symm hs))

/-- **A critical point whose relations are within `π` of those of a zero keeps
the relations of the zero**: if `∇V(ψ) = 0`, `V(z) = 0` and
`|⟪m_r, ψ - z⟫| < π` for every row, then `⟪m_r, ψ - z⟫ = 0` for every row. -/
theorem inner_sub_eq_zero_of_gradient_eq_zero {ψ z : E} (hψ : gradient (phaseEnergy m) ψ = 0)
    (hz : phaseEnergy m z = 0) (h : ∀ r, |⟪m r, ψ - z⟫| < π) (r : R) : ⟪m r, ψ - z⟫ = 0 := by
  have hr := (Finset.sum_eq_zero_iff_of_nonneg fun r _ => mul_sin_nonneg_of_abs_le (h r).le).1
    (sum_inner_sub_mul_sin m hψ hz) r (Finset.mem_univ r)
  rcases mul_eq_zero.1 hr with h0 | h0
  · exact h0
  · exact (sin_eq_zero_iff_of_lt_of_lt (neg_lt_of_abs_lt (h r)) (lt_of_abs_lt (h r))).1 h0

/-- **A local minimizer whose relations are within `π` of those of a zero
keeps the relations of the zero**: if `ψ` is a local minimizer of `V`,
`V(z) = 0` and `|⟪m_r, ψ - z⟫| ≤ π` for every row, then `⟪m_r, ψ - z⟫ = 0`
for every row. With `x_r = ⟪m_r, ψ - z⟫`, the Hessian along `ψ - z` gives
`Σ_r x_r² cos x_r ≥ 0` and the gradient `Σ_r x_r sin x_r = 0`, while
`x² cos x < x sin x` for `0 < |x| ≤ π`. -/
theorem inner_sub_eq_zero_of_isLocalMin {ψ z : E} (hmin : IsLocalMin (phaseEnergy m) ψ)
    (hz : phaseEnergy m z = 0) (h : ∀ r, |⟪m r, ψ - z⟫| ≤ π) (r : R) : ⟪m r, ψ - z⟫ = 0 := by
  have hψ := Landscape.SaddleAvoidance.gradient_eq_zero_of_isLocalMin hmin
  have hH := Landscape.SaddleAvoidance.inner_nonneg_of_isLocalMin
    ((contDiff_phaseEnergy m (k := 1)).differentiable one_ne_zero)
    (hasFDerivAt_gradient_phaseEnergy m ψ) hmin (ψ - z)
  rw [← (hasFDerivAt_gradient_phaseEnergy m ψ).fderiv, inner_fderiv_gradient_phaseEnergy] at hH
  have hle : ∀ s, ⟪m s, ψ - z⟫ ^ 2 * cos ⟪m s, ψ - z⟫ - ⟪m s, ψ - z⟫ * sin ⟪m s, ψ - z⟫ ≤ 0 :=
    fun s => by
      rcases eq_or_ne ⟪m s, ψ - z⟫ 0 with h0 | h0
      · rw [h0]
        simp
      · exact (sub_neg.2 (sq_mul_cos_lt_mul_sin h0 (h s))).le
  have hsum : 0 ≤ ∑ s, (⟪m s, ψ - z⟫ ^ 2 * cos ⟪m s, ψ - z⟫ -
      ⟪m s, ψ - z⟫ * sin ⟪m s, ψ - z⟫) := by
    rw [Finset.sum_sub_distrib, sum_inner_sub_mul_sin m hψ hz, sub_zero]
    refine hH.trans_eq (Finset.sum_congr rfl fun s _ => ?_)
    rw [(sin_cos_inner_sub m hz ψ s).2, mul_comm]
  by_contra h0
  have hlt := Finset.sum_lt_sum (fun s _ => hle s)
    ⟨r, Finset.mem_univ r, sub_neg.2 (sq_mul_cos_lt_mul_sin h0 (h r))⟩
  rw [Finset.sum_const_zero] at hlt
  linarith

/-- The hypotheses of the theorems above hold for the single row `1` on the
line, at the zero `ψ = z = 0`, a global minimizer, and for `x = 1`. -/
example : let m := fun _ : Unit => (1 : ℝ)
    phaseEnergy m 0 = 0 ∧ gradient (phaseEnergy m) 0 = 0 ∧ IsLocalMin (phaseEnergy m) 0 ∧
      (∀ r, |⟪m r, (0 : ℝ) - 0⟫| < π) ∧ (1 : ℝ) ≠ 0 ∧ |(1 : ℝ)| ≤ π := by
  intro m
  have h0 : phaseEnergy m 0 = 0 := by simp [m, phaseEnergy]
  refine ⟨h0, gradient_eq_zero_of_phaseEnergy_eq_zero m h0,
    Filter.Eventually.of_forall fun y => by rw [h0]; exact phaseEnergy_nonneg m y,
    fun _ => by simp [pi_pos], one_ne_zero, by rw [abs_one]; linarith [one_le_pi_div_two, pi_pos]⟩

end Transformer.Landscape.PhaseModel

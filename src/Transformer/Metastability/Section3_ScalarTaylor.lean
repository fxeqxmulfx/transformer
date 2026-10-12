/-
# Quadratic bounds for the actual angular interaction energy

ArXiv:2410.06833v1, §3.2, `lem: PL.borjan` and the interaction kernel
`g` preceding `eq: Ht.second.lb`. The derivative of the normalized pair
potential is `-h`, and `|h'|≤1+β` at every angle for `β≥0`.
A scalar mean-value argument converts this into a quadratic energy bound.
These bounds apply to the actual exponential interaction, globally on ℝ.
-/

import Transformer.Causal.Interaction
import Mathlib.Analysis.Calculus.MeanValue

open scoped NNReal
open Set Real

namespace Transformer.Metastability

/-- **The interaction derivative has global absolute bound `1+β`.**
Source: arXiv:2410.06833v1, §3.2, the kernel `g` and the cross-interaction
Hessian bound preceding `eq: Ht.first.lb`. -/
theorem interaction_derivative_abs_le (β : ℝ) (hβ : 0 ≤ β) (x : ℝ) :
    |Causal.g_pot β x| ≤ 1 + β := by
  have hfactor : |Real.cos x - β * Real.sin x ^ 2| ≤ 1 + β := by
    apply abs_le.mpr
    constructor <;> nlinarith [Real.neg_one_le_cos x, Real.cos_le_one x,
      mul_le_mul_of_nonneg_left (Real.sin_sq_le_one x) hβ,
      mul_nonneg hβ (sq_nonneg (Real.sin x))]
  have hexp : Real.exp (β * (Real.cos x - 1)) ≤ 1 :=
    Real.exp_le_one_iff.mpr (mul_nonpos_of_nonneg_of_nonpos hβ (by linarith [Real.cos_le_one x]))
  rw [Causal.g_pot, abs_mul, abs_of_pos (Real.exp_pos _)]
  exact (mul_le_mul hexp hfactor (abs_nonneg _) zero_le_one).trans (by rw [one_mul])

/-- **The actual interaction potential is globally `(1+β)`-Lipschitz.**
Source: arXiv:2410.06833v1, §3.2, `g=h'` in the proof of `lem: PL.borjan`. -/
theorem interaction_sub_abs_le (β : ℝ) (hβ : 0 ≤ β) (x y : ℝ) :
    |Causal.h_pot β x - Causal.h_pot β y| ≤ (1 + β) * |x - y| := by
  let L : ℝ≥0 := ⟨1 + β, by linarith⟩
  have hlip : LipschitzWith L (Causal.h_pot β) := by
    apply lipschitzWith_of_nnnorm_deriv_le (fun u => (Causal.hasDerivAt_h_pot β u).differentiableAt)
    intro u
    rw [(Causal.hasDerivAt_h_pot β u).deriv]
    exact_mod_cast interaction_derivative_abs_le β hβ u
  have h := hlip.norm_sub_le x y
  change ‖Causal.h_pot β x - Causal.h_pot β y‖ ≤ (1 + β) * ‖x - y‖ at h
  simpa only [Real.norm_eq_abs] using h

/-- The global derivative and Lipschitz hypotheses are met at `β=1`. -/
example : (0 : ℝ) ≤ 1 := zero_le_one

/-- The normalized pair potential whose double sum is the angular energy.
Source: arXiv:2410.06833v1, §3.2, `𝖤_β` with the factor `exp β` cancelled. -/
noncomputable def angularPairPotential (β x : ℝ) : ℝ :=
  Real.exp (β * (Real.cos x - 1)) / β

/-- Its actual derivative is minus the interaction potential `h`.
Source: arXiv:2410.06833v1, §3.2, the angular gradient of `𝖤_β`. -/
theorem hasDerivAt_angularPairPotential (β : ℝ) (hβ : β ≠ 0) (x : ℝ) :
    HasDerivAt (angularPairPotential β) (-Causal.h_pot β x) x := by
  have h := ((((Real.hasDerivAt_cos x).sub_const 1).const_mul β).exp).div_const β
  apply h.congr_deriv
  unfold Causal.h_pot
  field_simp

/-- The pair-derivative hypothesis is satisfiable at `β=1`. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-- **A scalar derivative difference bound gives a quadratic upper bound.**
The mean value theorem on the segment from `x` to `y` is enough; no
second derivative is assumed. Source: arXiv:2410.06833v1, §3.2,
the energy-discrepancy estimate of `lem: PL.borjan`. The factor `L`,
rather than the sharper `L/2`, is sufficient for this application. -/
theorem quadratic_upper_of_derivative_bound (f df : ℝ → ℝ) (L : ℝ) (hL : 0 ≤ L)
    (hf : ∀ u : ℝ, HasDerivAt f (df u) u)
    (hlip : ∀ u v : ℝ, |df u - df v| ≤ L * |u - v|) (x y : ℝ) :
    f y - f x ≤ df x * (y - x) + L * (y - x) ^ 2 := by
  let F (t : ℝ) := f (x + t * (y - x))
  have hd (t : ℝ) : HasDerivAt F (df (x + t * (y - x)) * (y - x)) t := by
    simpa only [Function.comp_def, id_eq, one_mul] using
      (hf (x + t * (y - x))).comp t
        (((hasDerivAt_id t).mul_const (y - x)).const_add x)
  have hb (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) :
      df (x + t * (y - x)) * (y - x) ≤ df x * (y - x) + L * (y - x) ^ 2 := by
    have h := hlip (x + t * (y - x)) x
    have hab : |x + t * (y - x) - x| = t * |y - x| := by
      rw [show x + t * (y - x) - x = t * (y - x) by ring,
        abs_mul, abs_of_nonneg ht.1]
    rw [hab] at h
    have hm := mul_le_mul_of_nonneg_right h (abs_nonneg (y - x))
    have hrem : |(df (x + t * (y - x)) - df x) * (y - x)| ≤ L * (y - x) ^ 2 := by
      rw [abs_mul]
      have heq : L * (t * |y - x|) * |y - x| = L * t * (y - x) ^ 2 := by
        rw [show L * (t * |y - x|) * |y - x| = L * t * |y - x| ^ 2 by ring, sq_abs]
      rw [heq] at hm
      have htL := mul_le_mul_of_nonneg_right ht.2 (mul_nonneg hL (sq_nonneg (y - x)))
      nlinarith
    have hsign := le_abs_self ((df (x + t * (y - x)) - df x) * (y - x))
    nlinarith
  have h := (convex_Icc (0 : ℝ) 1).image_sub_le_mul_sub_of_deriv_le
    (continuous_iff_continuousAt.mpr fun t => (hd t).continuousAt).continuousOn
    (fun t _ => (hd t).differentiableAt.differentiableWithinAt)
    (fun t ht => by
      rw [(hd t).deriv]
      rw [interior_Icc] at ht
      exact hb t ⟨ht.1.le, ht.2.le⟩)
    0 ⟨le_rfl, zero_le_one⟩ 1 ⟨zero_le_one, le_rfl⟩ zero_le_one
  have hzero : x + 0 * (y - x) = x := by ring
  have hone : x + 1 * (y - x) = y := by ring
  simpa only [F, hzero, hone, sub_zero, mul_one] using h

/-- The full scalar Taylor remainder is bounded in absolute value.
Apply the upper bound to `f` and `-f`. Source: arXiv:2410.06833v1,
§3.2, the quadratic energy-discrepancy estimate of `lem: PL.borjan`. -/
theorem quadratic_remainder_abs_le (f df : ℝ → ℝ) (L : ℝ) (hL : 0 ≤ L)
    (hf : ∀ u : ℝ, HasDerivAt f (df u) u)
    (hlip : ∀ u v : ℝ, |df u - df v| ≤ L * |u - v|) (x y : ℝ) :
    |f y - f x - df x * (y - x)| ≤ L * (y - x) ^ 2 := by
  have hup := quadratic_upper_of_derivative_bound f df L hL hf hlip x y
  have hlow := quadratic_upper_of_derivative_bound (fun u => -f u) (fun u => -df u)
    L hL (fun u => (hf u).neg) (fun u v => by
      rw [show -df u - -df v = -(df u - df v) by ring, abs_neg]
      exact hlip u v) x y
  apply abs_le.mpr
  constructor <;> linarith

/-- All inputs of both scalar Taylor estimates hold for
`f(u)=u²`, `df(u)=2u`, `L=2`. -/
example : (0 : ℝ) ≤ 2 ∧
    (∀ u : ℝ, HasDerivAt (fun v : ℝ => v ^ 2) (2 * u) u) ∧
    (∀ u v : ℝ, |2 * u - 2 * v| ≤ 2 * |u - v|) := by
  refine ⟨by norm_num, fun u => ?_, fun u v => ?_⟩
  · simpa using hasDerivAt_pow 2 u
  · rw [← mul_sub, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 2)]

/-- **The true pair potential obeys the global quadratic bound.**
Source: arXiv:2410.06833v1, §3.2, the energy discrepancy in `lem: PL.borjan`. -/
theorem angularPairPotential_upper (β : ℝ) (hβ : 0 < β) (x y : ℝ) :
    angularPairPotential β y - angularPairPotential β x ≤
      -Causal.h_pot β x * (y - x) + (1 + β) * (y - x) ^ 2 := by
  apply quadratic_upper_of_derivative_bound (angularPairPotential β) (fun u => -Causal.h_pot β u)
    (1 + β) (by linarith) (hasDerivAt_angularPairPotential β (ne_of_gt hβ))
  intro u v
  rw [show -Causal.h_pot β u - -Causal.h_pot β v =
    -(Causal.h_pot β u - Causal.h_pot β v) by ring, abs_neg]
  exact interaction_sub_abs_le β hβ.le u v

/-- The actual-potential estimate's hypothesis holds at `β=1`. -/
example : (0 : ℝ) < 1 := one_pos

end Transformer.Metastability

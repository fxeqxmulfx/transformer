/-
# The number of modes of a Gaussian KDE — the Gaussian integrals

`lem:gaussian-int` of arXiv:2412.09080v3, §5.1: the one-sided Gaussian moment
in closed form, and the two continuity statements in the shift that the
moment asymptotics of `lem:moments-p` and `lem:phi-t` are read off from.

**What the source says and what is carried here.**

* The first display, `∫_0^∞ uⁿ e^{-αu²} du = ½ Γ((n+1)/2) α^{-(n+1)/2}`, is
  proved: it is Mathlib's `integral_rpow_mul_exp_neg_mul_rpow` at `p = 2`,
  `q = n`, once the `rpow`s are turned into `npow`s.

* The two limits as `ε → 0` are proved by dominated convergence, as in the
  source.  The dominating function is `e·|v|ⁿe^{-v²/2}` for `|ε| ≤ 1`, from
  `(v - ε)² ≥ v²/2 - ε²`, one function in place of the source's piecewise
  `e·2ⁿ` on `[-2, 2]` and `|v|ⁿe^{-v²/4}` outside.

Source: arXiv:2412.09080v3, `lem:gaussian-int`, §5.1.
-/

import Transformer.Modes.Section2_Gt
import Mathlib.MeasureTheory.Integral.Gamma

open scoped BigOperators
open Real MeasureTheory Filter Topology

namespace Transformer
namespace Modes

/-- **Lemma (lem:gaussian-int), first display.**  For `α > 0` and an integer
`m ≥ 0`, `∫_0^∞ uᵐ e^{-αu²} du = ½ Γ((m+1)/2) α^{-(m+1)/2}`.

Source: arXiv:2412.09080v3, `lem:gaussian-int`. -/
theorem integral_pow_mul_exp_neg_mul_sq {α : ℝ} (hα : 0 < α) (m : ℕ) :
    (∫ u in Set.Ioi (0 : ℝ), u ^ m * Real.exp (-α * u ^ 2))
      = 1 / 2 * Real.Gamma (((m : ℝ) + 1) / 2) * α ^ (-((m : ℝ) + 1) / 2) := by
  have hm : (-1 : ℝ) < (m : ℝ) := by
    have h0 : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
    linarith
  have h := integral_rpow_mul_exp_neg_mul_rpow (p := 2) (q := (m : ℝ)) (b := α) two_pos hm hα
  have heq : (∫ u in Set.Ioi (0 : ℝ), u ^ m * Real.exp (-α * u ^ 2))
      = ∫ u in Set.Ioi (0 : ℝ), u ^ (m : ℝ) * Real.exp (-α * u ^ (2 : ℝ)) := by
    refine setIntegral_congr_fun measurableSet_Ioi fun u hu => ?_
    rw [Real.rpow_natCast, Real.rpow_two]
  rw [heq, h]
  ring

/-- The hypothesis of `integral_pow_mul_exp_neg_mul_sq` is satisfiable. -/
example : (0 : ℝ) < 1 := one_pos

/-- `|v|ⁿe^{-v²/2}` is integrable on the line. -/
theorem integrable_pow_mul_exp_neg_half_sq (m : ℕ) :
    Integrable (fun v : ℝ => v ^ m * Real.exp (-(1 / 2) * v ^ 2)) := by
  have hm : (-1 : ℝ) < (m : ℝ) := by
    have := Nat.cast_nonneg (α := ℝ) m
    linarith
  convert integrable_rpow_mul_exp_neg_mul_sq (b := 1 / 2) (by norm_num) hm using 2 with v
  rw [Real.rpow_natCast]

/-- The domination behind both limits: for `|ε| ≤ 1`,
`|v|ⁿ e^{-(v-ε)²} ≤ e · |v|ⁿ e^{-v²/2}`, because
`(v - ε)² - v²/2 + ε² = (v - 2ε)²/2 ≥ 0`. -/
theorem norm_pow_mul_exp_shift_le (m : ℕ) {ε : ℝ} (hε : |ε| ≤ 1) (v : ℝ) :
    ‖v ^ m * Real.exp (-(v - ε) ^ 2)‖
      ≤ Real.exp 1 * ‖v ^ m * Real.exp (-(1 / 2) * v ^ 2)‖ := by
  rw [norm_mul, norm_mul, Real.norm_of_nonneg (Real.exp_pos _).le,
    Real.norm_of_nonneg (Real.exp_pos _).le, mul_left_comm, ← Real.exp_add]
  gcongr
  have := abs_le.mp hε
  nlinarith [sq_nonneg (v - 2 * ε)]

/-- The hypothesis of `norm_pow_mul_exp_shift_le` is satisfiable: `ε = 0`. -/
example : |(0 : ℝ)| ≤ 1 := by norm_num

/-- **Lemma (lem:gaussian-int), second display, first half.**  For fixed `m`,
`∫_0^∞ vᵐ e^{-(v-ε)²} dv → ∫_0^∞ vᵐ e^{-v²} dv` as `ε → 0`.

Source: arXiv:2412.09080v3, `lem:gaussian-int`. -/
theorem tendsto_integral_Ioi_shift (m : ℕ) :
    Tendsto (fun ε : ℝ => ∫ v in Set.Ioi (0 : ℝ), v ^ m * Real.exp (-(v - ε) ^ 2))
      (nhds 0) (nhds (∫ v in Set.Ioi (0 : ℝ), v ^ m * Real.exp (-v ^ 2))) := by
  have hc : ContinuousAt
      (fun ε : ℝ => ∫ v in Set.Ioi (0 : ℝ), v ^ m * Real.exp (-(v - ε) ^ 2)) 0 := by
    refine continuousAt_of_dominated
      (bound := fun v => Real.exp 1 * ‖v ^ m * Real.exp (-(1 / 2) * v ^ 2)‖) ?_ ?_ ?_ ?_
    · exact Eventually.of_forall fun ε =>
        (by fun_prop : Continuous fun v : ℝ => v ^ m * Real.exp (-(v - ε) ^ 2)).aestronglyMeasurable
    · filter_upwards [Metric.closedBall_mem_nhds (0 : ℝ) one_pos] with ε hε
      refine Eventually.of_forall fun v => norm_pow_mul_exp_shift_le m ?_ v
      simpa [Real.dist_eq] using hε
    · exact ((integrable_pow_mul_exp_neg_half_sq m).norm.const_mul _).integrableOn
    · exact Eventually.of_forall fun v => by fun_prop
  simpa using hc.tendsto

/-- **Lemma (lem:gaussian-int), second display, second half.**  For fixed `m`,
`∫_{-∞}^∞ |v|ᵐ e^{-(v-ε)²} dv → 2∫_0^∞ vᵐ e^{-v²} dv` as `ε → 0`.

Source: arXiv:2412.09080v3, `lem:gaussian-int`. -/
theorem tendsto_integral_abs_shift (m : ℕ) :
    Tendsto (fun ε : ℝ => ∫ v : ℝ, |v| ^ m * Real.exp (-(v - ε) ^ 2))
      (nhds 0) (nhds (2 * ∫ v in Set.Ioi (0 : ℝ), v ^ m * Real.exp (-v ^ 2))) := by
  have hc : ContinuousAt (fun ε : ℝ => ∫ v : ℝ, |v| ^ m * Real.exp (-(v - ε) ^ 2)) 0 := by
    refine continuousAt_of_dominated
      (bound := fun v => Real.exp 1 * ‖v ^ m * Real.exp (-(1 / 2) * v ^ 2)‖) ?_ ?_ ?_ ?_
    · exact Eventually.of_forall fun ε =>
        (by fun_prop : Continuous fun v : ℝ => |v| ^ m * Real.exp (-(v - ε) ^ 2)).aestronglyMeasurable
    · filter_upwards [Metric.closedBall_mem_nhds (0 : ℝ) one_pos] with ε hε
      refine Eventually.of_forall fun v => ?_
      have h := norm_pow_mul_exp_shift_le m (ε := ε) (by simpa [Real.dist_eq] using hε) v
      calc _ = ‖v ^ m * Real.exp (-(v - ε) ^ 2)‖ := by
            simp only [norm_mul, norm_pow, Real.norm_eq_abs, abs_abs]
        _ ≤ _ := h
    · exact (integrable_pow_mul_exp_neg_half_sq m).norm.const_mul _
    · exact Eventually.of_forall fun v => by fun_prop
  have h0 : ∫ v : ℝ, |v| ^ m * Real.exp (-(v - 0) ^ 2)
      = 2 * ∫ v in Set.Ioi (0 : ℝ), v ^ m * Real.exp (-v ^ 2) := by
    have h := integral_comp_abs (f := fun x : ℝ => x ^ m * Real.exp (-x ^ 2))
    simp only [sq_abs] at h
    simpa only [sub_zero] using h
  exact h0 ▸ hc.tendsto

end Modes
end Transformer

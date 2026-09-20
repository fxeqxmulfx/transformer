/-
# The number of modes of a Gaussian KDE — the Gaussian integrals

`lem:gaussian-int` of arXiv:2412.09080v3, §5.1: the one-sided Gaussian moment
in closed form, and the two continuity statements in the shift that the
moment asymptotics of `lem:moments-p` and `lem:phi-t` are read off from.

**What the source says and what is carried here.**

* The first display, `∫_0^∞ uⁿ e^{-αu²} du = ½ Γ((n+1)/2) α^{-(n+1)/2}`, is
  proved: it is Mathlib's `integral_rpow_mul_exp_neg_mul_rpow` at `p = 2`,
  `q = n`, once the `rpow`s are turned into `npow`s.

* The two limits as `ε → 0` are stated and unproved.  The source proves them by
  dominated convergence, the dominating function being `e·vⁿe^{-(v-1)²}` for
  `|ε| ≤ 1`.

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

/-- **Lemma (lem:gaussian-int), second display, first half.**  For fixed `m`,
`∫_0^∞ vᵐ e^{-(v-ε)²} dv → ∫_0^∞ vᵐ e^{-v²} dv` as `ε → 0`.

Not proved here; the source's argument is dominated convergence with the
dominating function `e·vᵐe^{-(v-1)²}`, valid for `|ε| ≤ 1`.

Source: arXiv:2412.09080v3, `lem:gaussian-int`. -/
theorem tendsto_integral_Ioi_shift (m : ℕ) :
    Tendsto (fun ε : ℝ => ∫ v in Set.Ioi (0 : ℝ), v ^ m * Real.exp (-(v - ε) ^ 2))
      (nhds 0) (nhds (∫ v in Set.Ioi (0 : ℝ), v ^ m * Real.exp (-v ^ 2))) := by
  sorry

/-- **Lemma (lem:gaussian-int), second display, second half.**  For fixed `m`,
`∫_{-∞}^∞ |v|ᵐ e^{-(v-ε)²} dv → 2∫_0^∞ vᵐ e^{-v²} dv` as `ε → 0`.

Not proved here; same argument.

Source: arXiv:2412.09080v3, `lem:gaussian-int`. -/
theorem tendsto_integral_abs_shift (m : ℕ) :
    Tendsto (fun ε : ℝ => ∫ v : ℝ, |v| ^ m * Real.exp (-(v - ε) ^ 2))
      (nhds 0) (nhds (2 * ∫ v in Set.Ioi (0 : ℝ), v ^ m * Real.exp (-v ^ 2))) := by
  sorry

end Modes
end Transformer

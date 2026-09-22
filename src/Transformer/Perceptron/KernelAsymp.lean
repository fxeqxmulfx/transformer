/-
# Perceptrons and attention's mean-field landscape — `θ_c(β)` for large `β`

The large-`β` expansion of the concavity window of `lem: concavity` of
arXiv:2601.21366v2, `θ_c(β) = β^{-1/2} + O(β^{-3/2})`.

The proof reads everything off the quadratic `β c² + c - β = 0` that
`c = cos θ_c` solves (`quadratic_cos_thetaC`): it says `β sin²θ_c = cos θ_c`,
so `sin²θ_c = 1/β + O(β^{-2})`, and `θ - θ³/6 < sin θ ≤ θ` carries this over
to `θ_c² = 1/β + O(β^{-2})`.  Then
`|θ_c - β^{-1/2}| = |θ_c² - 1/β| / (θ_c + β^{-1/2}) ≤ 3β^{-3/2}`.

Source: arXiv:2601.21366v2, `lem: concavity`.
-/

import Transformer.Perceptron.Kernel
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.Real.Pi.Bounds

open Real

namespace Transformer
namespace Perceptron

/-- **`β²θ_c(β)²` is `β + O(1)`.**  `β - 1 ≤ β²θ_c² ≤ β + 3` for every
`β > 0`: the quantitative form of `θ_c² = 1/β + O(β^{-2})` that both the
expansion of `θ_c` and the Hessian bound `eq:kernel-hess-bound-lambda` rest on.

Source: arXiv:2601.21366v2, proof of `lem: concavity`. -/
theorem sq_mul_sq_thetaC_mem (β : ℝ) (hβ : 0 < β) :
    β - 1 ≤ β ^ 2 * thetaC β ^ 2 ∧ β ^ 2 * thetaC β ^ 2 ≤ β + 3 := by
  have hθ0 := thetaC_pos β hβ
  have hθ1 := thetaC_lt_pi_div_two β hβ
  have hq := quadratic_cos_thetaC β hβ
  set θ := thetaC β
  have hc1 : Real.cos θ ≤ 1 := Real.cos_le_one θ
  have hc0 : 0 < Real.cos θ := Real.cos_pos_of_mem_Ioo ⟨by linarith, hθ1⟩
  -- `β sin²θ = cos θ`
  have hS : β * Real.sin θ ^ 2 = Real.cos θ := by
    linear_combination β * Real.sin_sq_add_cos_sq θ - hq
  have hSle : Real.sin θ ≤ θ := Real.sin_le hθ0.le
  have hS0 : 0 ≤ Real.sin θ := Real.sin_nonneg_of_nonneg_of_le_pi hθ0.le (by linarith [pi_pos])
  have hSge := Real.sin_gt_sub_cube hθ0
  have hθ2 : θ ^ 2 ≤ 5 / 2 := by nlinarith [pi_lt_d2]
  constructor
  · -- the lower bound, from `β(1 - cos θ) ≤ 1` and `sin θ ≤ θ`
    have h1 : β * (1 - Real.cos θ) ≤ 1 := by nlinarith [Real.sin_sq_add_cos_sq θ]
    have h2 : Real.sin θ ^ 2 ≤ θ ^ 2 := pow_le_pow_left₀ hS0 hSle 2
    nlinarith
  · -- the upper bound, from `θ - θ³/6 < sin θ`
    have hf : 0 < 1 - θ ^ 2 / 6 := by linarith
    have h1 : θ * (1 - θ ^ 2 / 6) ≤ Real.sin θ := by linarith
    have h2 : (θ * (1 - θ ^ 2 / 6)) ^ 2 ≤ Real.sin θ ^ 2 :=
      pow_le_pow_left₀ (by positivity) h1 2
    have h3 : β * θ ^ 2 * (1 - θ ^ 2 / 3) ≤ 1 := by nlinarith [sq_nonneg (θ ^ 2)]
    have h4 : β * θ ^ 2 ≤ 3 := by nlinarith
    nlinarith

/-- `sq_mul_sq_thetaC_mem` needs `β > 0`, satisfied by `β = 1`. -/
example : 1 - 1 ≤ (1 : ℝ) ^ 2 * thetaC 1 ^ 2 ∧ (1 : ℝ) ^ 2 * thetaC 1 ^ 2 ≤ 1 + 3 :=
  sq_mul_sq_thetaC_mem 1 one_pos

/-- **Lemma (lem: concavity), the large-`β` expansion.**
`θ_c(β) = β^{-1/2} + O(β^{-3/2})` as `β → ∞`, written with `β^{-1/2} = 1/√β`
and `β^{-3/2} = 1/(β√β)` so that no real power is needed.  The constant is
`3`, from `β = 1` on.

Source: arXiv:2601.21366v2, `lem: concavity`. -/
theorem thetaC_asymptotics :
    ∃ C β₀ : ℝ, 0 < C ∧ 0 < β₀ ∧ ∀ β : ℝ, β₀ ≤ β →
      |thetaC β - 1 / Real.sqrt β| ≤ C / (β * Real.sqrt β) := by
  refine ⟨3, 1, by norm_num, one_pos, fun β hβ1 => ?_⟩
  have hβ : 0 < β := by linarith
  have hθ0 := thetaC_pos β hβ
  obtain ⟨hlow, hup⟩ := sq_mul_sq_thetaC_mem β hβ
  set θ := thetaC β
  -- from `θ²` to `θ`
  have hs0 : 0 < Real.sqrt β := Real.sqrt_pos.mpr hβ
  have hs : Real.sqrt β ^ 2 = β := Real.sq_sqrt hβ.le
  generalize Real.sqrt β = s at hs0 hs ⊢
  subst hs
  have hden : 0 < s * θ + 1 := by positivity
  have heq : θ - 1 / s = (s ^ 2 * θ ^ 2 - 1) / (s * (s * θ + 1)) := by
    field_simp; ring
  rw [heq, abs_div, abs_of_pos (by positivity : 0 < s * (s * θ + 1)),
    div_le_div_iff₀ (by positivity) (by positivity)]
  have hx : |s ^ 2 * θ ^ 2 - 1| * s ^ 2 ≤ 3 := by
    rcases abs_cases (s ^ 2 * θ ^ 2 - 1) with ⟨h, -⟩ | ⟨h, -⟩ <;> rw [h] <;> nlinarith
  nlinarith [mul_pos hs0 hθ0]

end Perceptron
end Transformer

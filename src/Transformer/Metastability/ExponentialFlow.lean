/-
# The flow of `𝖤(x) = x²/2` on `ℝ`, and the constant of `lem: quantitative inequality`

arXiv:2410.06833v1, "Dynamic metastability in the self-attention model", §3.2,
`lem: quantitative inequality`.

`X(t) = e^{-t}` is the gradient flow of the one-dimensional energy
`𝖤(x) = x²/2`, whose gradient is the identity.  It is the smallest example of
the hypotheses of `Transformer.Metastability.quantitative_inequality`, so it
witnesses that they are satisfiable; and, run from `u = X(0) = 1` to
`v = X(1) = e⁻¹` with `c = 1`, it is also what refutes the constant `2c`
printed in the paper:

  `2c‖u - v‖² = 2(1 - e⁻¹)² ≈ 0.799 > 0.432 ≈ (1 - e⁻²)/2 = 𝖤(u) - 𝖤(v)`,

since `4(1 - s)² > 1 - s²` for `s = e⁻¹ < 1/2`.  The corrected constant `c/2`
of `quantitative_inequality` holds here, as it must: `(1 - e⁻¹)²/2 ≈ 0.200`.
-/

import Mathlib.Analysis.Complex.ExponentialBounds
import Transformer.Metastability.QuantitativeInequality

namespace Transformer
namespace Metastability

/-- The four analytic facts about `X(t) = e^{-t}` under `𝖤(x) = x²/2`: it
solves `Ẋ = -∇𝖤(X)`, its gradient is continuous along it, `𝖤 ∘ X` falls at the
rate `‖∇𝖤(X)‖²`, and the Polyak-Łojasiewicz bound holds with `c = 1` on
`[0, 1]`. -/
theorem expFlow_spec :
    (∀ t : ℝ, HasDerivAt (fun r : ℝ => Real.exp (-r)) (-Real.exp (-t)) t) ∧
      Continuous (fun t : ℝ => Real.exp (-t)) ∧
      (∀ t : ℝ, HasDerivAt (fun r : ℝ => Real.exp (-r) ^ 2 / 2) (-‖Real.exp (-t)‖ ^ 2) t) ∧
      ∀ t ∈ Set.Icc (0 : ℝ) 1,
        Real.exp (-t) ^ 2 / 2 - Real.exp (-1) ^ 2 / 2 ≤ 1 / (2 * 1) * ‖Real.exp (-t)‖ ^ 2 := by
  have h1 : ∀ t : ℝ, HasDerivAt (fun r : ℝ => Real.exp (-r)) (-Real.exp (-t)) t := by
    intro t
    simpa using HasDerivAt.exp (hasDerivAt_neg' t)
  refine ⟨h1, Real.continuous_exp.comp continuous_neg, fun t => ?_, fun t _ => ?_⟩
  · have hd := ((h1 t).pow 2).div_const 2
    simp only [Pi.pow_apply] at hd
    rw [Real.norm_eq_abs, sq_abs]
    convert hd using 1
    push_cast
    ring
  · rw [Real.norm_eq_abs, sq_abs]
    have hpos : 0 < Real.exp (-1 : ℝ) := Real.exp_pos _
    nlinarith [sq_nonneg (Real.exp (-1 : ℝ))]

/-- The hypotheses of `quantitative_inequality` are satisfiable, and its
conclusion is not vacuous: the flow above gives `(1/2)‖1 - e⁻¹‖² ≤ (1 - e⁻²)/2`. -/
example : 1 / 2 * ‖(1 : ℝ) - Real.exp (-1)‖ ^ 2 ≤ 1 ^ 2 / 2 - Real.exp (-1) ^ 2 / 2 := by
  obtain ⟨h1, h2, h3, h4⟩ := expFlow_spec
  exact quantitative_inequality (fun x : ℝ => x ^ 2 / 2) id (fun t : ℝ => Real.exp (-t)) 1
    (Real.exp (-1)) 1 1 one_pos zero_le_one h1 h2 h3 (by norm_num) rfl h4

/-- **The constant `2c` of the printed lemma is false.**  The hypotheses of
`quantitative_inequality` do not imply `2c‖u - v‖² ≤ 𝖤(u) - 𝖤(v)`: the flow of
`𝖤(x) = x²/2` from `1` to `e⁻¹` satisfies all of them with `c = 1` and has
`2(1 - e⁻¹)² > (1 - e⁻²)/2`. -/
theorem not_quantitative_inequality_two_mul :
    ¬ ∀ (E gradE X : ℝ → ℝ) (u v c T : ℝ), 0 < c → 0 ≤ T →
        (∀ t : ℝ, HasDerivAt X (-gradE (X t)) t) →
        Continuous (fun t => gradE (X t)) →
        (∀ t : ℝ, HasDerivAt (fun r => E (X r)) (-‖gradE (X t)‖ ^ 2) t) →
        X 0 = u → X T = v →
        (∀ t ∈ Set.Icc (0 : ℝ) T, E (X t) - E v ≤ 1 / (2 * c) * ‖gradE (X t)‖ ^ 2) →
        2 * c * ‖u - v‖ ^ 2 ≤ E u - E v := by
  intro h
  obtain ⟨h1, h2, h3, h4⟩ := expFlow_spec
  have hbad := h (fun x : ℝ => x ^ 2 / 2) id (fun t : ℝ => Real.exp (-t)) 1 (Real.exp (-1)) 1 1
    one_pos zero_le_one h1 h2 h3 (by norm_num) rfl h4
  -- `e⁻¹ < 1/2`, which is what makes `4(1 - s)² > 1 - s²`
  have hpos : 0 < Real.exp (-1 : ℝ) := Real.exp_pos _
  have hhalf : Real.exp (-1 : ℝ) < 1 / 2 := by
    rw [Real.exp_neg, inv_lt_comm₀ (Real.exp_pos 1) (by norm_num)]
    have := Real.exp_one_gt_two
    linarith
  rw [Real.norm_eq_abs, abs_of_pos (by linarith)] at hbad
  nlinarith [hbad, hpos, hhalf]

end Metastability
end Transformer

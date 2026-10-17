/-
# The field's mixed second differences

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 3, `eq:est-Glaub-2nd`:
the field's mixed second difference in two configurations' worth of moves is
controlled by the mixed second differences of the angles and the product of the
first differences (`abs_gptField_second_diff_le`), from the four-point
inequality `abs_four_point_le` for `w_β'`.
-/

import Transformer.Kinetic.Section3_SecondDifferences

namespace Transformer
namespace Kinetic

/-- The field's mixed second difference: with `y = θ¹¹ - θ¹⁰ - θ⁰¹ + θ⁰⁰`,
`u = θ¹¹ - θ¹⁰` and `v = θ¹¹ - θ⁰¹`, it is at most
`L₁ |y_a| + Σ_{b<a} L₁ ω_{a,b} |y_b| + L₂ Σ_{b<a} ω_{a,b} (|u_a| + |u_b|)(|v_a| + |v_b|)`. -/
theorem abs_gptField_second_diff_le (lam β : ℝ) (N : ℕ) {L₁ L₂ : ℝ}
    (hd : ∀ x, HasDerivAt (wBetaDeriv β) (deriv (wBetaDeriv β) x) x)
    (hL : ∀ x y, |wBetaDeriv β x - wBetaDeriv β y| ≤ L₁ * |x - y|) (hL₁ : 0 ≤ L₁)
    (hL₂ : ∀ x y, |deriv (wBetaDeriv β) x - deriv (wBetaDeriv β) y| ≤ L₂ * |x - y|)
    (θ₁₁ θ₁₀ θ₀₁ θ₀₀ : Idx N → ℝ) (a : Idx N) :
    |gptField lam β N θ₁₁ a - gptField lam β N θ₁₀ a - gptField lam β N θ₀₁ a +
        gptField lam β N θ₀₀ a| ≤
      L₁ * |θ₁₁ a - θ₁₀ a - θ₀₁ a + θ₀₀ a| +
      ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a),
        L₁ * alibiWeight lam N a b * |θ₁₁ b - θ₁₀ b - θ₀₁ b + θ₀₀ b| +
      L₂ * ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a), alibiWeight lam N a b *
        ((|θ₁₁ a - θ₁₀ a| + |θ₁₁ b - θ₁₀ b|) * (|θ₁₁ a - θ₀₁ a| + |θ₁₁ b - θ₀₁ b|)) := by
  set S := Finset.univ.filter (fun b : Idx N => b < a)
  rw [gptField_eq_sum, gptField_eq_sum, gptField_eq_sum, gptField_eq_sum,
    ← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  have hterm : ∀ b ∈ S, |alibiWeight lam N a b * wBetaDeriv β (θ₁₁ a - θ₁₁ b) -
      alibiWeight lam N a b * wBetaDeriv β (θ₁₀ a - θ₁₀ b) -
      alibiWeight lam N a b * wBetaDeriv β (θ₀₁ a - θ₀₁ b) +
      alibiWeight lam N a b * wBetaDeriv β (θ₀₀ a - θ₀₀ b)| ≤
      alibiWeight lam N a b * (L₁ * |θ₁₁ a - θ₁₀ a - θ₀₁ a + θ₀₀ a|) +
      L₁ * alibiWeight lam N a b * |θ₁₁ b - θ₁₀ b - θ₀₁ b + θ₀₀ b| +
      L₂ * (alibiWeight lam N a b * ((|θ₁₁ a - θ₁₀ a| + |θ₁₁ b - θ₁₀ b|) *
        (|θ₁₁ a - θ₀₁ a| + |θ₁₁ b - θ₀₁ b|))) := fun b _ => by
    have hω := alibiWeight_nonneg lam N a b
    have h4 := abs_four_point_le hd hL hL₂ (θ₁₁ a - θ₁₁ b) (θ₁₀ a - θ₁₀ b) (θ₀₁ a - θ₀₁ b)
      (θ₀₀ a - θ₀₀ b)
    have hy : |θ₁₁ a - θ₁₁ b - (θ₁₀ a - θ₁₀ b) - (θ₀₁ a - θ₀₁ b) + (θ₀₀ a - θ₀₀ b)| ≤
        |θ₁₁ a - θ₁₀ a - θ₀₁ a + θ₀₀ a| + |θ₁₁ b - θ₁₀ b - θ₀₁ b + θ₀₀ b| := by
      rw [show θ₁₁ a - θ₁₁ b - (θ₁₀ a - θ₁₀ b) - (θ₀₁ a - θ₀₁ b) + (θ₀₀ a - θ₀₀ b) =
        (θ₁₁ a - θ₁₀ a - θ₀₁ a + θ₀₀ a) - (θ₁₁ b - θ₁₀ b - θ₀₁ b + θ₀₀ b) by ring]
      exact abs_sub _ _
    have hu : |θ₁₁ a - θ₁₁ b - (θ₁₀ a - θ₁₀ b)| ≤ |θ₁₁ a - θ₁₀ a| + |θ₁₁ b - θ₁₀ b| := by
      rw [show θ₁₁ a - θ₁₁ b - (θ₁₀ a - θ₁₀ b) = (θ₁₁ a - θ₁₀ a) - (θ₁₁ b - θ₁₀ b) by ring]
      exact abs_sub _ _
    have hv : |θ₁₁ a - θ₁₁ b - (θ₀₁ a - θ₀₁ b)| ≤ |θ₁₁ a - θ₀₁ a| + |θ₁₁ b - θ₀₁ b| := by
      rw [show θ₁₁ a - θ₁₁ b - (θ₀₁ a - θ₀₁ b) = (θ₁₁ a - θ₀₁ a) - (θ₁₁ b - θ₀₁ b) by ring]
      exact abs_sub _ _
    have hL₂0 : 0 ≤ L₂ := by
      have := (abs_nonneg _).trans (hL₂ 0 1)
      simpa using this
    have hprod := mul_le_mul hu hv (abs_nonneg _) (by positivity)
    have hw := mul_le_mul_of_nonneg_left
      (h4.trans (add_le_add (mul_le_mul_of_nonneg_left hy hL₁)
        (mul_le_mul_of_nonneg_left hprod hL₂0))) hω
    rw [show alibiWeight lam N a b * wBetaDeriv β (θ₁₁ a - θ₁₁ b) -
        alibiWeight lam N a b * wBetaDeriv β (θ₁₀ a - θ₁₀ b) -
        alibiWeight lam N a b * wBetaDeriv β (θ₀₁ a - θ₀₁ b) +
        alibiWeight lam N a b * wBetaDeriv β (θ₀₀ a - θ₀₀ b) =
        alibiWeight lam N a b * (wBetaDeriv β (θ₁₁ a - θ₁₁ b) - wBetaDeriv β (θ₁₀ a - θ₁₀ b) -
          wBetaDeriv β (θ₀₁ a - θ₀₁ b) + wBetaDeriv β (θ₀₀ a - θ₀₀ b)) by ring,
      abs_mul, abs_of_nonneg hω]
    nlinarith
  refine (Finset.abs_sum_le_sum_abs _ _).trans ((Finset.sum_le_sum hterm).trans ?_)
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.sum_mul]
  have hw := sum_alibiWeight_le_one lam N a
  have := mul_le_mul_of_nonneg_right hw
    (mul_nonneg hL₁ (abs_nonneg (θ₁₁ a - θ₁₀ a - θ₀₁ a + θ₀₀ a)))
  linarith

/-- The hypotheses of `abs_gptField_second_diff_le` hold for the Lipschitz constants of
`exists_lipschitz_wBetaDeriv`. -/
example (lam β : ℝ) (θ : Idx 2 → ℝ) :=
  have hw := (exists_lipschitz_wBetaDeriv β).choose_spec.choose_spec
  abs_gptField_second_diff_le lam β 2 hw.2.2.1 hw.2.2.2.1 hw.1 hw.2.2.2.2 θ θ θ θ 1

end Kinetic
end Transformer

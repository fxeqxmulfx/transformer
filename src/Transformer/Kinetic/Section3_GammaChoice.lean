/-
# The choice of the exponent

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 5: the envelopes hold for
every `γ ∈ (0, γ₀]`, with a factor `e^{aT/γ}` from the profiles and a factor
`q_γ(i, k)^b = e^{bγℓ}`, `ℓ = log((i+1)/(k+1))`, from the tokens
(`ratioPow_eq_exp`). Taking `γ = √(T/ℓ)` when it is below `γ₀` gives
`aT/γ + bγℓ ≤ aT/γ₀ + (a + b)√(Tℓ)` (`exists_gamma_le`), hence the source's factor
`e^{√(CTℓ)} e^{CT}` (`exists_gamma_exp_le`).
-/

import Transformer.Kinetic.Section3_ChainSums

open Real

namespace Transformer
namespace Kinetic

theorem ratioPow_eq_exp (γ : ℝ) (a b : ℕ) :
    ratioPow γ a b = Real.exp (γ * Real.log (((a : ℝ) + 1) / ((b : ℝ) + 1))) := by
  unfold ratioPow
  rw [Real.rpow_def_of_pos (by positivity), mul_comm]

/-- **The choice of `γ`.** For `T > 0` and `ℓ ≥ 0` some `γ ∈ (0, γ₀]` has
`aT/γ + bγℓ ≤ aT/γ₀ + (a + b)√(Tℓ)`. -/
theorem exists_gamma_le {a b γ₀ T ℓ : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hγ₀ : 0 < γ₀)
    (hT : 0 < T) (hℓ : 0 ≤ ℓ) :
    ∃ γ, 0 < γ ∧ γ ≤ γ₀ ∧ a / γ * T + b * γ * ℓ ≤ a / γ₀ * T + (a + b) * √(T * ℓ) := by
  have hs : 0 ≤ √(T * ℓ) := Real.sqrt_nonneg _
  by_cases h : γ₀ ^ 2 * ℓ ≤ T
  · refine ⟨γ₀, hγ₀, le_rfl, ?_⟩
    have h1 : γ₀ * ℓ ≤ √(T * ℓ) :=
      calc γ₀ * ℓ = √((γ₀ * ℓ) ^ 2) := (Real.sqrt_sq (by positivity)).symm
        _ ≤ √(T * ℓ) := Real.sqrt_le_sqrt (by nlinarith [mul_le_mul_of_nonneg_right h hℓ])
    nlinarith [mul_le_mul_of_nonneg_left h1 hb]
  · rw [not_le] at h
    have hℓ0 : 0 < ℓ := by
      rcases hℓ.eq_or_lt with h0 | h0
      · rw [← h0, mul_zero] at h
        linarith
      · exact h0
    obtain ⟨s, hs_def⟩ : ∃ s, s = √(T * ℓ) := ⟨_, rfl⟩
    have hs0 : 0 < s := by
      rw [hs_def]
      exact Real.sqrt_pos.2 (by positivity)
    have hss : s * s = T * ℓ := by
      rw [hs_def]
      exact Real.mul_self_sqrt (by positivity)
    rw [← hs_def]
    refine ⟨s / ℓ, by positivity, ?_, ?_⟩
    · rw [div_le_iff₀ hℓ0]
      calc s = √(T * ℓ) := hs_def
        _ ≤ √((γ₀ * ℓ) ^ 2) :=
            Real.sqrt_le_sqrt (by nlinarith [mul_lt_mul_of_pos_right h hℓ0])
        _ = γ₀ * ℓ := Real.sqrt_sq (by positivity)
    · have e1 : a / (s / ℓ) * T = a * s := by
        rw [div_div_eq_mul_div, div_mul_eq_mul_div, div_eq_iff hs0.ne', mul_assoc a s s, hss]
        ring
      have e2 : b * (s / ℓ) * ℓ = b * s := by
        rw [mul_assoc, div_mul_cancel₀ s hℓ0.ne']
      have : 0 ≤ a / γ₀ * T := by positivity
      rw [e1, e2]
      linarith

/-- **The exponential factor.** For `T > 0`, `ℓ ≥ 0` and `C` above `nα/γ₀` and
`(nα + m)²`, some `γ ∈ (0, γ₀]` has `e^{nαT/γ} e^{mγℓ} ≤ e^{√(CTℓ)} e^{CT}`. -/
theorem exists_gamma_exp_le {α : ℝ} (hα : 0 ≤ α) (n m : ℕ) {γ₀ T ℓ C : ℝ} (hγ₀ : 0 < γ₀)
    (hT : 0 < T) (hℓ : 0 ≤ ℓ) (hC₁ : n * α / γ₀ ≤ C) (hC₂ : (n * α + m) ^ 2 ≤ C) :
    ∃ γ, 0 < γ ∧ γ ≤ γ₀ ∧ Real.exp (α / γ * T) ^ n * Real.exp (γ * ℓ) ^ m ≤
      Real.exp (√(C * T * ℓ)) * Real.exp (C * T) := by
  obtain ⟨γ, hγ0, hγ, h⟩ := exists_gamma_le (a := n * α) (b := m) (by positivity)
    (by positivity) hγ₀ hT hℓ
  refine ⟨γ, hγ0, hγ, ?_⟩
  rw [← Real.exp_nat_mul, ← Real.exp_nat_mul, ← Real.exp_add, ← Real.exp_add, Real.exp_le_exp]
  have h1 : (n * α + m) * √(T * ℓ) ≤ √(C * T * ℓ) :=
    calc (n * α + m) * √(T * ℓ) = √((n * α + m) ^ 2 * (T * ℓ)) := by
          rw [Real.sqrt_mul (x := (n * α + m) ^ 2) (by positivity), Real.sqrt_sq (by positivity)]
      _ ≤ √(C * T * ℓ) := Real.sqrt_le_sqrt (by
          rw [mul_assoc C]
          exact mul_le_mul_of_nonneg_right hC₂ (by positivity))
  have h2 : n * α / γ₀ * T ≤ C * T := mul_le_mul_of_nonneg_right hC₁ hT.le
  have e : (n : ℝ) * (α / γ * T) + m * (γ * ℓ) = n * α / γ * T + m * γ * ℓ := by ring
  linarith

/-- The hypotheses of `exists_gamma_exp_le` are satisfiable: `α = 0`, `n = m = 1`,
`γ₀ = 1`, `T = ℓ = 1`, `C = 1`. -/
example : ∃ γ : ℝ, 0 < γ ∧ γ ≤ 1 ∧ Real.exp (0 / γ * 1) ^ 1 * Real.exp (γ * 1) ^ 1 ≤
    Real.exp (√(1 * 1 * 1)) * Real.exp (1 * 1) :=
  exists_gamma_exp_le le_rfl 1 1 one_pos one_pos zero_le_one (by norm_num) (by norm_num)

end Kinetic
end Transformer

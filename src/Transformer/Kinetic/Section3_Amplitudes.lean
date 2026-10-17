/-
# The amplitudes of the envelopes

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 5: for `α ≥ max(1, L₁)` and
`γ ≤ 1` every amplitude of the envelopes is a power of `E = e^{αT/γ}`:
`τ, τ/γ ≤ E²`, `T ≤ E` and `e^{L₁T} ≤ E` (`rate_le`). The covariance bound
`abs_covariance_flow_le` is then at most `(3π²/2) E⁶ q K²/(i+1)`
(`covariance_amp_le`), and the third cumulant bound `abs_thirdCumulant_flow_le` at most
`c E¹⁷ q⁶ K²K'/((i+1)(j+1))` (`thirdCumulant_amp_le`): the source's `e^{CT}` and,
through `q`, its `e^{√(CT log(i/k))}`.
-/

import Transformer.Kinetic.Section3_PairSums

open Real

namespace Transformer
namespace Kinetic

/-- The amplitudes as powers of `E = e^{αT/γ}`. -/
theorem rate_le {α γ T L₁ : ℝ} (hα1 : 1 ≤ α) (hL₁α : L₁ ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1)
    (hT : 0 ≤ T) :
    1 ≤ Real.exp (α / γ * T) ∧ profileRate α γ T ≤ Real.exp (α / γ * T) ^ 2 ∧
      profileRate α γ T / γ ≤ Real.exp (α / γ * T) ^ 2 ∧ T ≤ Real.exp (α / γ * T) ∧
      Real.exp (L₁ * T) ≤ Real.exp (α / γ * T) := by
  have hα0 : 0 ≤ α := by linarith
  have hx : α * T ≤ α / γ * T := by
    rw [div_mul_eq_mul_div]
    exact le_div_self (by positivity) hγ0 hγ1
  have hTα : T ≤ α * T := le_mul_of_one_le_left hT hα1
  have hE := Real.add_one_le_exp (α / γ * T)
  have hx' : α * T / γ ≤ Real.exp (α / γ * T) := by
    rw [← div_mul_eq_mul_div]
    linarith
  have hE0 := Real.exp_pos (α / γ * T)
  refine ⟨Real.one_le_exp (by positivity), ?_, ?_, by linarith,
    Real.exp_le_exp.2 ((mul_le_mul_of_nonneg_right hL₁α hT).trans hx)⟩
  · unfold profileRate
    rw [sq]
    exact mul_le_mul_of_nonneg_right (by linarith) hE0.le
  · unfold profileRate
    rw [sq, mul_div_right_comm]
    exact mul_le_mul_of_nonneg_right hx' hE0.le

/-- **The covariance amplitude.** -/
theorem covariance_amp_le {α γ T L₁ K q v : ℝ} (hα1 : 1 ≤ α) (hL₁α : L₁ ≤ α) (hγ0 : 0 < γ)
    (hγ1 : γ ≤ 1) (hT : 0 ≤ T) (hq : 0 ≤ q) (hv : 0 < v) :
    1 / 2 * (K * π * Real.exp (L₁ * T)) ^ 2 *
        (profileRate α γ T * (1 + 2 * profileRate α γ T) * (q / v)) ≤
      3 / 2 * π ^ 2 * v⁻¹ * K ^ 2 * (Real.exp (α / γ * T) ^ 6 * q ^ 1) := by
  obtain ⟨hE1, hτ, -, -, hL⟩ := rate_le hα1 hL₁α hγ0 hγ1 hT
  have hτ0 := profileRate_nonneg (α := α) (γ := γ) (by linarith) hT
  have hE2 : 1 ≤ Real.exp (α / γ * T) ^ 2 := one_le_pow₀ hE1
  have h1 : 1 + 2 * profileRate α γ T ≤ 3 * Real.exp (α / γ * T) ^ 2 := by linarith
  calc 1 / 2 * (K * π * Real.exp (L₁ * T)) ^ 2 *
        (profileRate α γ T * (1 + 2 * profileRate α γ T) * (q / v))
      = 1 / 2 * π ^ 2 * v⁻¹ * K ^ 2 * q *
          (Real.exp (L₁ * T) ^ 2 * (profileRate α γ T * (1 + 2 * profileRate α γ T))) := by
        ring
    _ ≤ 1 / 2 * π ^ 2 * v⁻¹ * K ^ 2 * q * (Real.exp (α / γ * T) ^ 2 *
          (Real.exp (α / γ * T) ^ 2 * (3 * Real.exp (α / γ * T) ^ 2))) := by
        gcongr
    _ = _ := by ring

/-- **The amplitude of the third cumulant.** -/
theorem thirdCumulant_amp_le {α γ T L₁ ω κ K K' q₁ q₂ v : ℝ} (hα1 : 1 ≤ α) (hL₁α : L₁ ≤ α)
    (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) (hT : 0 ≤ T) (hω : 0 ≤ ω) (hκ0 : 0 ≤ κ) (hκK : κ ≤ K)
    (hκK' : κ ≤ K') (hq₁ : 1 ≤ q₁) (hq₂ : 0 ≤ q₂) (hq₂₁ : q₂ ≤ q₁) (hv : 0 < v) :
    3 / 2 * (κ * π * Real.exp (L₁ * T)) ^ 3 * (profileRate α γ T ^ 2 *
          (1 + 2 * profileRate α γ T) * (q₁ * q₂ / v)) +
        2 * (κ * π * Real.exp (L₁ * T)) ^ 2 * (K' * π ^ 2 * Real.exp (2 * L₁ * T) *
          (ω * T * (1 + 2 * (profileRate α γ T / γ)) ^ 3 + profileRate α γ T +
            profileRate α γ T ^ 2)) * pairConst γ (profileRate α γ T) * (q₁ ^ 6 / v) ≤
      (9 / 2 * π ^ 3 + 96 * (27 * ω + 2) * π ^ 4) * v⁻¹ * (K ^ 2 * K') *
        (Real.exp (α / γ * T) ^ 17 * q₁ ^ 6) := by
  obtain ⟨hE1, hτ, hτγ, hTE, hL⟩ := rate_le hα1 hL₁α hγ0 hγ1 hT
  have hτ0 := profileRate_nonneg (α := α) (γ := γ) (by linarith) hT
  obtain ⟨E, hE⟩ : ∃ E, E = Real.exp (α / γ * T) := ⟨_, rfl⟩
  obtain ⟨τ, hτd⟩ : ∃ τ, τ = profileRate α γ T := ⟨_, rfl⟩
  rw [← hE] at hE1 hτ hτγ hTE hL ⊢
  rw [← hτd] at hτ hτγ hτ0 ⊢
  have hκγ : 0 ≤ τ / γ := div_nonneg hτ0 hγ0.le
  have hE0 : 0 ≤ E := by linarith
  have hK'0 : 0 ≤ K' := hκ0.trans hκK'
  have hE2 : 1 ≤ E ^ 2 := one_le_pow₀ hE1
  have hE7 : E ^ 2 ≤ E ^ 7 := pow_le_pow_right₀ hE1 (by norm_num)
  have hE47 : E ^ 4 ≤ E ^ 7 := pow_le_pow_right₀ hE1 (by norm_num)
  have hk3 : κ ^ 3 ≤ K ^ 2 * K' :=
    calc κ ^ 3 = κ ^ 2 * κ := by ring
      _ ≤ K ^ 2 * K' := mul_le_mul (pow_le_pow_left₀ hκ0 hκK 2) hκK' hκ0 (by positivity)
  have he₂ : Real.exp (2 * L₁ * T) ≤ E ^ 2 :=
    calc Real.exp (2 * L₁ * T) = Real.exp (L₁ * T) ^ 2 := by
          rw [← Real.exp_nat_mul]
          congr 1
          push_cast
          ring
      _ ≤ E ^ 2 := pow_le_pow_left₀ (Real.exp_pos _).le hL 2
  have hq : q₁ * q₂ ≤ q₁ ^ 6 :=
    calc q₁ * q₂ ≤ q₁ * q₁ := mul_le_mul_of_nonneg_left hq₂₁ (by linarith)
      _ = q₁ ^ 2 := by ring
      _ ≤ q₁ ^ 6 := pow_le_pow_right₀ hq₁ (by norm_num)
  have hA : τ ^ 2 * (1 + 2 * τ) ≤ 3 * E ^ 6 :=
    calc τ ^ 2 * (1 + 2 * τ) ≤ (E ^ 2) ^ 2 * (3 * E ^ 2) :=
          mul_le_mul (pow_le_pow_left₀ hτ0 hτ 2) (by linarith) (by positivity) (by positivity)
      _ = 3 * E ^ 6 := by ring
  have hW : ω * T * (1 + 2 * (τ / γ)) ^ 3 + τ + τ ^ 2 ≤ (27 * ω + 2) * E ^ 7 := by
    have h3 : (1 + 2 * (τ / γ)) ^ 3 ≤ (3 * E ^ 2) ^ 3 :=
      pow_le_pow_left₀ (by positivity) (by linarith) 3
    have h4 : ω * T * (1 + 2 * (τ / γ)) ^ 3 ≤ ω * E * (3 * E ^ 2) ^ 3 :=
      mul_le_mul (mul_le_mul_of_nonneg_left hTE hω) h3 (by positivity) (by positivity)
    have h5 : τ ^ 2 ≤ (E ^ 2) ^ 2 := pow_le_pow_left₀ hτ0 hτ 2
    nlinarith
  have hΛ0 : 0 ≤ pairConst γ τ := by unfold pairConst; positivity
  have hΛ : pairConst γ τ ≤ 48 * E ^ 6 := by
    unfold pairConst
    have h1 : (1 + τ / γ) ^ 2 ≤ (2 * E ^ 2) ^ 2 := pow_le_pow_left₀ (by positivity) (by linarith) 2
    calc 4 * (1 + 2 * τ) * (1 + τ / γ) ^ 2 ≤ 4 * (3 * E ^ 2) * (2 * E ^ 2) ^ 2 := by
          gcongr
          linarith
      _ = 48 * E ^ 6 := by ring
  calc 3 / 2 * (κ * π * Real.exp (L₁ * T)) ^ 3 * (τ ^ 2 * (1 + 2 * τ) * (q₁ * q₂ / v)) +
        2 * (κ * π * Real.exp (L₁ * T)) ^ 2 * (K' * π ^ 2 * Real.exp (2 * L₁ * T) *
          (ω * T * (1 + 2 * (τ / γ)) ^ 3 + τ + τ ^ 2)) * pairConst γ τ * (q₁ ^ 6 / v)
      = 3 / 2 * π ^ 3 * (κ ^ 3 * Real.exp (L₁ * T) ^ 3) * (τ ^ 2 * (1 + 2 * τ)) * (q₁ * q₂) *
            v⁻¹ +
          2 * π ^ 4 * (κ ^ 2 * Real.exp (L₁ * T) ^ 2) * K' * Real.exp (2 * L₁ * T) *
            (ω * T * (1 + 2 * (τ / γ)) ^ 3 + τ + τ ^ 2) * pairConst γ τ * q₁ ^ 6 * v⁻¹ := by
        ring
    _ ≤ 3 / 2 * π ^ 3 * (K ^ 2 * K' * E ^ 3) * (3 * E ^ 6) * q₁ ^ 6 * v⁻¹ +
          2 * π ^ 4 * (K ^ 2 * E ^ 2) * K' * E ^ 2 * ((27 * ω + 2) * E ^ 7) * (48 * E ^ 6) *
            q₁ ^ 6 * v⁻¹ := by
        gcongr
    _ = 9 / 2 * π ^ 3 * (v⁻¹ * (K ^ 2 * K') * q₁ ^ 6) * E ^ 9 +
          96 * (27 * ω + 2) * π ^ 4 * (v⁻¹ * (K ^ 2 * K') * q₁ ^ 6) * E ^ 17 := by ring
    _ ≤ 9 / 2 * π ^ 3 * (v⁻¹ * (K ^ 2 * K') * q₁ ^ 6) * E ^ 17 +
          96 * (27 * ω + 2) * π ^ 4 * (v⁻¹ * (K ^ 2 * K') * q₁ ^ 6) * E ^ 17 :=
        add_le_add (mul_le_mul_of_nonneg_left (pow_le_pow_right₀ hE1 (by norm_num))
          (by positivity)) le_rfl
    _ = _ := by ring

/-- The hypotheses of `rate_le`, `covariance_amp_le` and `thirdCumulant_amp_le` hold at
`α = γ = 1`, `L₁ = T = 0` and unit constants. -/
example := And.intro (rate_le (α := 1) (L₁ := 0) (γ := 1) (T := 0) le_rfl zero_le_one one_pos
    le_rfl le_rfl) <|
  And.intro (covariance_amp_le (α := 1) (L₁ := 0) (γ := 1) (T := 0) (K := 1) le_rfl
    zero_le_one one_pos le_rfl le_rfl zero_le_one one_pos)
  (thirdCumulant_amp_le (α := 1) (L₁ := 0) (γ := 1) (T := 0) (ω := 0) (κ := 0) (K := 1)
    (K' := 1) (q₁ := 1) (q₂ := 1) (v := 1) le_rfl zero_le_one one_pos le_rfl le_rfl le_rfl
    le_rfl zero_le_one zero_le_one le_rfl zero_le_one le_rfl one_pos)

end Kinetic
end Transformer

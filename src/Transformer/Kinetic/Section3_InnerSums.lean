module

public import Transformer.Kinetic.Section3_PairEnvelope

/-!
# Sums over the later token of a pair

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 4, `eq:est-kap3-th`: the
inner sums `Σ_{M>n} Ĝ^M_z(T) G(x, M)` over the later token `M` of a pair `n < M`.
For `x < z` only `M ≤ x` contribute, where `Ĝ^M_z = τ q_γ(z, x) q_γ(x, M)/(z+1)`
(`sum_profileEnv_pairDecay_up`); for `z < x` the decay of `x` is `1/((x+1)(M+1))`
(`sum_profileEnv_pairDecay_down`). Both sums vanish unless `n` precedes `min(x, z)`.
-/

@[expose] public section

namespace Transformer
namespace Kinetic

/-- `Σ_{M>n} q_γ(x, M) G(x, M) ≤ (1 + q_γ(x, n)²/γ)/(x+1)`. -/
theorem sum_ratioPow_pairDecay_le {γ : ℝ} (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) (N n x : ℕ) :
    ∑ M ∈ Finset.range N, (if n < M then ratioPow γ x M * pairDecay x M else 0) ≤
      (1 + ratioPow γ x n * ratioPow γ x n / γ) / ((x : ℝ) + 1) := by
  refine sum_decay_le hγ0 hγ1 (ratioPow_pos γ x n).le (fun M hM => ?_) N
  rcases lt_trichotomy M x with h | h | h
  · rw [pairDecay_of_lt h, ite_eq_right h.ne, ite_eq_left h, zero_add, mul_one_div]
    exact div_le_div_of_nonneg_right (ratioPow_le_ratioPow_right hγ0.le x hM.le)
      (by positivity)
  · subst h
    rw [pairDecay_self, ratioPow_self, one_mul, ite_eq_left rfl, ite_eq_right (lt_irrefl _),
      add_zero]
  · rw [pairDecay_of_gt h, mul_zero, ite_eq_right h.ne', ite_eq_right (by omega), add_zero]

/-- For `x < z`, `Ĝ^M_z G(x, M) ≤ τ q_γ(z, x)/(z+1) · q_γ(x, M) G(x, M)`. -/
theorem profileEnv_mul_pairDecay_le_up {γ τ : ℝ} {x z : ℕ} (hxz : x < z) (M : ℕ) :
    profileEnv γ τ z M * pairDecay x M ≤
      τ * ratioPow γ z x / ((z : ℝ) + 1) * (ratioPow γ x M * pairDecay x M) := by
  rcases le_or_gt M x with hM | hM
  · rw [profileEnv_of_lt (hM.trans_lt hxz), ← ratioPow_mul γ z x M]
    exact le_of_eq (by ring)
  · simp only [pairDecay_of_gt hM, mul_zero, le_refl]

/-- For `z < x`, `Ĝ^M_z G(x, M) ≤ (x+1)⁻¹ Ĝ^M_z/(M+1)`. -/
theorem profileEnv_mul_pairDecay_le_down {γ τ : ℝ} {x z : ℕ} (hzx : z < x) (M : ℕ) :
    profileEnv γ τ z M * pairDecay x M ≤
      1 / ((x : ℝ) + 1) * (profileEnv γ τ z M / ((M : ℝ) + 1)) := by
  rcases le_or_gt M z with hM | hM
  · rw [pairDecay_of_lt (hM.trans_lt hzx)]
    exact le_of_eq (by simp only [div_eq_mul_inv, mul_inv]; ring)
  · simp only [profileEnv_of_gt hM, zero_mul, zero_div, mul_zero, le_refl]

/-- **The later token after the observed one.** For `x < z`,
`Σ_{M>n} Ĝ^M_z G(x, M) ≤ [n < x] τ q_γ(z, x)/(z+1) · (1 + q_γ(x, n)²/γ)/(x+1)`. -/
theorem sum_profileEnv_pairDecay_up {γ τ : ℝ} (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) (hτ : 0 ≤ τ)
    {x z : ℕ} (hxz : x < z) (N n : ℕ) :
    ∑ M ∈ Finset.range N, (if n < M then profileEnv γ τ z M * pairDecay x M else 0) ≤
      if n < x then τ * ratioPow γ z x / ((z : ℝ) + 1) *
        ((1 + ratioPow γ x n * ratioPow γ x n / γ) / ((x : ℝ) + 1)) else 0 := by
  split_ifs with hn
  · have hpt : ∀ M ∈ Finset.range N,
        (if n < M then profileEnv γ τ z M * pairDecay x M else 0) ≤
          τ * ratioPow γ z x / ((z : ℝ) + 1) *
            (if n < M then ratioPow γ x M * pairDecay x M else 0) := by
      intro M _
      rw [mul_ite, mul_zero]
      split_ifs
      · exact profileEnv_mul_pairDecay_le_up hxz M
      · exact le_rfl
    refine (Finset.sum_le_sum hpt).trans ?_
    rw [← Finset.mul_sum]
    have := ratioPow_pos γ z x
    exact mul_le_mul_of_nonneg_left (sum_ratioPow_pairDecay_le hγ0 hγ1 N n x) (by positivity)
  · refine le_of_eq (Finset.sum_eq_zero fun M _ => ?_)
    split_ifs with hM
    · rw [pairDecay_of_gt (by omega), mul_zero]
    · rfl

/-- **The later token before the observed one.** For `z < x`,
`Σ_{M>n} Ĝ^M_z G(x, M) ≤ [n < z] (1 + τ q_γ(z, n)²/γ)/(z+1) · 1/(x+1)`. -/
theorem sum_profileEnv_pairDecay_down {γ τ : ℝ} (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) (hτ : 0 ≤ τ)
    {x z : ℕ} (hzx : z < x) (N n : ℕ) :
    ∑ M ∈ Finset.range N, (if n < M then profileEnv γ τ z M * pairDecay x M else 0) ≤
      if n < z then (1 + τ * ratioPow γ z n * ratioPow γ z n / γ) / ((z : ℝ) + 1) *
        (1 / ((x : ℝ) + 1)) else 0 := by
  split_ifs with hn
  · have hpt : ∀ M ∈ Finset.range N,
        (if n < M then profileEnv γ τ z M * pairDecay x M else 0) ≤
          1 / ((x : ℝ) + 1) * (if n < M then profileEnv γ τ z M / ((M : ℝ) + 1) else 0) := by
      intro M _
      rw [mul_ite, mul_zero]
      split_ifs
      · exact profileEnv_mul_pairDecay_le_down hzx M
      · exact le_rfl
    refine (Finset.sum_le_sum hpt).trans ?_
    rw [← Finset.mul_sum, mul_comm]
    refine mul_le_mul_of_nonneg_right (sum_decay_le hγ0 hγ1
      (mul_nonneg hτ (ratioPow_pos γ z n).le) (fun M hM => ?_) N) (by positivity)
    rcases lt_trichotomy M z with h | h | h
    · rw [profileEnv_of_lt h, ite_eq_right h.ne, ite_eq_left h, zero_add]
      have := ratioPow_le_ratioPow_right hγ0.le z hM.le
      have := ratioPow_pos γ z M
      calc τ * (ratioPow γ z M / ((z : ℝ) + 1)) / ((M : ℝ) + 1)
          = τ * ratioPow γ z M / (((z : ℝ) + 1) * ((M : ℝ) + 1)) := by
            simp only [div_eq_mul_inv, mul_inv]; ring
        _ ≤ τ * ratioPow γ z n / (((z : ℝ) + 1) * ((M : ℝ) + 1)) := by gcongr
    · subst h
      rw [profileEnv_self, ite_eq_left rfl, ite_eq_right (lt_irrefl _), add_zero]
    · rw [profileEnv_of_gt h, zero_div, ite_eq_right h.ne', ite_eq_right (by omega), add_zero]
  · refine le_of_eq (Finset.sum_eq_zero fun M _ => ?_)
    split_ifs with hM
    · rw [profileEnv_of_gt (by omega), zero_mul]
    · rfl

/-- The hypotheses of `sum_ratioPow_pairDecay_le`, `profileEnv_mul_pairDecay_le_up` and
`profileEnv_mul_pairDecay_le_down` hold at `γ = τ = 1` and the tokens `0 < 1`. -/
example := And.intro (sum_ratioPow_pairDecay_le (γ := 1) one_pos le_rfl 1 0 0) <|
  And.intro (profileEnv_mul_pairDecay_le_up (γ := 1) (τ := 1) (x := 0) (z := 1) Nat.one_pos 0)
    (profileEnv_mul_pairDecay_le_down (γ := 1) (τ := 1) (x := 1) (z := 0) Nat.one_pos 0)

/-- The hypotheses of `sum_profileEnv_pairDecay_up` and `sum_profileEnv_pairDecay_down`
hold at `γ = 1/2`, `τ = 1` and the tokens `0 < 1`. -/
example := And.intro (sum_profileEnv_pairDecay_up (γ := 1 / 2) (τ := 1) (by norm_num)
    (by norm_num) zero_le_one (x := 0) (z := 1) Nat.one_pos 1 0)
  (sum_profileEnv_pairDecay_down (γ := 1 / 2) (τ := 1) (by norm_num) (by norm_num)
    zero_le_one (x := 1) (z := 0) Nat.one_pos 1 0)

end Kinetic
end Transformer

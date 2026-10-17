/-
# Duhamel's supersolution of a pair

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 3, `eq:est-Glaub-2nd`:
for the pair `n < M`, Duhamel's supersolution `V_x = T Σ_c σ_c Ĝ^c_x(T)` is at most
`T (1 + 2κ)³ q_γ(x, n)³ G(x, M)` (`crossProfile_le`), with `κ = τ/γ`: the source's
`|∂²_{θ_n θ_M} θ_x| ≲ (xM)⁻¹ e^{√(CT log(x/n))}`. The chain `n ≤ c ≤ x` loses one
power of the ratio (`ratioPow_sq_mul_le`), and the sum over `M < c < x` one `1/γ`.
-/

import Transformer.Kinetic.Section3_ForcingBound

open Real

namespace Transformer
namespace Kinetic

/-- `q_γ(c, n)² q_γ(x, c) ≤ q_γ(x, n)²` for `c ≤ x`. -/
theorem ratioPow_sq_mul_le {γ : ℝ} (hγ : 0 ≤ γ) {n c x : ℕ} (hcx : c ≤ x) :
    ratioPow γ c n ^ 2 * ratioPow γ x c ≤ ratioPow γ x n ^ 2 := by
  have h := ratioPow_mul γ x c n
  have h1 := ratioPow_le_ratioPow_left hγ hcx n
  have h2 := ratioPow_pos γ x n
  calc ratioPow γ c n ^ 2 * ratioPow γ x c = ratioPow γ c n * ratioPow γ x n := by
        rw [← h]; ring
    _ ≤ ratioPow γ x n * ratioPow γ x n := mul_le_mul_of_nonneg_right h1 h2.le
    _ = ratioPow γ x n ^ 2 := by ring

/-- A term `σ_c Ĝ^c_x(T)` of the supersolution for `n < M < x`. -/
theorem crossForcing_mul_profile_le {α γ T : ℝ} (hα : 0 ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1)
    (hT : 0 ≤ T) {n M x : ℕ} (hnM : n < M) (hMx : M < x) (c : ℕ) :
    crossForcing α γ T n M c * profile α γ c x T ≤
      (1 + 2 * (profileRate α γ T / γ)) ^ 2 * (ratioPow γ x n ^ 2 / ((M : ℝ) + 1)) *
        ((if c = M then profileRate α γ T / ((x : ℝ) + 1) else 0) +
          (if M < c then (if c = x then 1 / ((x : ℝ) + 1) else 0) +
            (if c < x then profileRate α γ T / (((x : ℝ) + 1) * ((c : ℝ) + 1)) else 0)
            else 0)) := by
  have hσ := crossForcing_le hα hγ0 hγ1 hT hnM c
  have hP0 := profile_nonneg hα hγ0 c x hT
  have hτ : 0 ≤ profileRate α γ T := profileRate_nonneg hα hT
  have hx1 : (0 : ℝ) < (x : ℝ) + 1 := by positivity
  set τ := profileRate α γ T
  rcases lt_trichotomy c M with hc | hcM | hc
  · rw [crossForcing_eq_zero_of_lt hc, zero_mul, ite_eq_right hc.ne, ite_eq_right (by omega),
      add_zero, mul_zero]
  · subst hcM
    rw [ite_eq_left rfl, ite_eq_right (lt_irrefl c), add_zero]
    have hG : pairDecay c c = 1 / ((c : ℝ) + 1) := by simp [pairDecay]
    rw [hG] at hσ
    have hP := profile_le_rate (α := α) (T := T) hγ0 hMx
    have hqq := ratioPow_sq_mul_le hγ0.le hMx.le (n := n)
    calc crossForcing α γ T n c c * profile α γ c x T ≤
          (1 + 2 * (τ / γ)) ^ 2 * (ratioPow γ c n ^ 2 * (1 / ((c : ℝ) + 1))) *
            (τ * ratioPow γ x c / ((x : ℝ) + 1)) := mul_le_mul hσ hP hP0 (by positivity)
      _ = (1 + 2 * (τ / γ)) ^ 2 * τ / (((c : ℝ) + 1) * ((x : ℝ) + 1)) *
            (ratioPow γ c n ^ 2 * ratioPow γ x c) := by
          simp only [div_eq_mul_inv, mul_inv]; ring
      _ ≤ (1 + 2 * (τ / γ)) ^ 2 * τ / (((c : ℝ) + 1) * ((x : ℝ) + 1)) * ratioPow γ x n ^ 2 :=
          mul_le_mul_of_nonneg_left hqq (by positivity)
      _ = (1 + 2 * (τ / γ)) ^ 2 * (ratioPow γ x n ^ 2 / ((c : ℝ) + 1)) * (τ / ((x : ℝ) + 1)) := by
          simp only [div_eq_mul_inv, mul_inv]; ring
  · rw [ite_eq_right hc.ne', ite_eq_left hc, zero_add]
    have hG : pairDecay c M = 1 / (((c : ℝ) + 1) * ((M : ℝ) + 1)) := by
      rw [pairDecay, ite_eq_right hc.ne, ite_eq_left hc, zero_add]
    rw [hG] at hσ
    rcases lt_trichotomy c x with hcx | hcx | hcx
    · rw [ite_eq_right hcx.ne, ite_eq_left hcx, zero_add]
      have hP := profile_le_rate (α := α) (T := T) hγ0 hcx
      have hqq := ratioPow_sq_mul_le hγ0.le hcx.le (n := n)
      calc crossForcing α γ T n M c * profile α γ c x T ≤
            (1 + 2 * (τ / γ)) ^ 2 * (ratioPow γ c n ^ 2 * (1 / (((c : ℝ) + 1) * ((M : ℝ) + 1)))) *
              (τ * ratioPow γ x c / ((x : ℝ) + 1)) := mul_le_mul hσ hP hP0 (by positivity)
        _ = (1 + 2 * (τ / γ)) ^ 2 * τ / (((c : ℝ) + 1) * ((M : ℝ) + 1) * ((x : ℝ) + 1)) *
              (ratioPow γ c n ^ 2 * ratioPow γ x c) := by
          simp only [div_eq_mul_inv, mul_inv]; ring
        _ ≤ (1 + 2 * (τ / γ)) ^ 2 * τ / (((c : ℝ) + 1) * ((M : ℝ) + 1) * ((x : ℝ) + 1)) *
              ratioPow γ x n ^ 2 := mul_le_mul_of_nonneg_left hqq (by positivity)
        _ = (1 + 2 * (τ / γ)) ^ 2 * (ratioPow γ x n ^ 2 / ((M : ℝ) + 1)) *
              (τ / (((x : ℝ) + 1) * ((c : ℝ) + 1))) := by
          simp only [div_eq_mul_inv, mul_inv]; ring
    · subst hcx
      rw [ite_eq_left rfl, ite_eq_right (lt_irrefl c), add_zero, profile_self, mul_one]
      refine hσ.trans (le_of_eq ?_)
      simp only [div_eq_mul_inv, mul_inv]; ring
    · rw [ite_eq_right hcx.ne', ite_eq_right (by omega), add_zero, profile_eq_zero_of_lt hcx,
        mul_zero, mul_zero]

/-- **Duhamel's supersolution of a pair.** `V_x ≤ T (1 + 2κ)³ q_γ(x, n)³ G(x, M)`. -/
theorem crossProfile_le {α γ T : ℝ} (hα : 0 ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) (hT : 0 ≤ T)
    (N : ℕ) {n M : ℕ} (hnM : n < M) (x : ℕ) :
    crossProfile α γ T N n M x T ≤
      T * ((1 + 2 * (profileRate α γ T / γ)) ^ 3 * (ratioPow γ x n ^ 3 * pairDecay x M)) := by
  unfold crossProfile
  refine mul_le_mul_of_nonneg_left ?_ hT
  have hτ : 0 ≤ profileRate α γ T := profileRate_nonneg hα hT
  have hκ : 0 ≤ profileRate α γ T / γ := div_nonneg hτ hγ0.le
  rcases lt_trichotomy x M with hxM | hxM | hMx
  · have h0 : ∑ c ∈ Finset.range N, crossForcing α γ T n M c * profile α γ c x T = 0 :=
      Finset.sum_eq_zero fun c _ => by
        rcases le_or_gt c x with hc | hc
        · rw [crossForcing_eq_zero_of_lt (hc.trans_lt hxM), zero_mul]
        · rw [profile_eq_zero_of_lt hc, mul_zero]
    have hG : pairDecay x M = 0 := by
      rw [pairDecay, ite_eq_right hxM.ne', ite_eq_right (by omega), add_zero]
    rw [h0, hG]
    simp
  · subst hxM
    have hpt : ∀ c ∈ Finset.range N, crossForcing α γ T n x c * profile α γ c x T ≤
        if c = x then (1 + 2 * (profileRate α γ T / γ)) ^ 2 *
          (ratioPow γ x n ^ 2 * pairDecay x x) else 0 := by
      intro c _
      rcases lt_trichotomy c x with hc | hc | hc
      · rw [crossForcing_eq_zero_of_lt hc, zero_mul, ite_eq_right hc.ne]
      · subst hc
        rw [ite_eq_left rfl, profile_self, mul_one]
        exact crossForcing_le hα hγ0 hγ1 hT hnM c
      · rw [profile_eq_zero_of_lt hc, mul_zero, ite_eq_right hc.ne']
    refine (Finset.sum_le_sum hpt).trans ?_
    rw [Finset.sum_ite_eq']
    have hq : 1 ≤ ratioPow γ x n := one_le_ratioPow hγ0.le hnM.le
    have hG := pairDecay_nonneg x x
    have h1 : (1 + 2 * (profileRate α γ T / γ)) ^ 2 ≤ (1 + 2 * (profileRate α γ T / γ)) ^ 3 :=
      pow_le_pow_right₀ (by linarith) (by norm_num)
    have h2 : ratioPow γ x n ^ 2 ≤ ratioPow γ x n ^ 3 := pow_le_pow_right₀ hq (by norm_num)
    have h3 := mul_le_mul h1 (mul_le_mul_of_nonneg_right h2 hG) (by positivity) (by positivity)
    split_ifs
    · exact h3
    · exact le_trans (by positivity) h3
  · refine (Finset.sum_le_sum fun c _ =>
      crossForcing_mul_profile_le hα hγ0 hγ1 hT hnM hMx c).trans ?_
    have hx1 : (0 : ℝ) < (x : ℝ) + 1 := by positivity
    have hq : 1 ≤ ratioPow γ x n := one_le_ratioPow hγ0.le (hnM.trans hMx).le
    have hqM : ratioPow γ x M ≤ ratioPow γ x n := ratioPow_le_ratioPow_right hγ0.le x hnM.le
    have hτκ : profileRate α γ T ≤ profileRate α γ T / γ := by
      rw [le_div_iff₀ hγ0]; nlinarith
    have hG : pairDecay x M = 1 / (((x : ℝ) + 1) * ((M : ℝ) + 1)) := by
      rw [pairDecay, ite_eq_right hMx.ne, ite_eq_left hMx, zero_add]
    rw [← Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_ite_eq', hG]
    set τ := profileRate α γ T
    set q := ratioPow γ x n
    have h1 : (if M ∈ Finset.range N then τ / ((x : ℝ) + 1) else 0) ≤ τ / ((x : ℝ) + 1) := by
      split_ifs
      · exact le_rfl
      · positivity
    have h2 := sum_decay_le hγ0 hγ1 (n := M) (x := x) hτ
      (h := fun c => (if c = x then 1 / ((x : ℝ) + 1) else 0) +
        (if c < x then τ / (((x : ℝ) + 1) * ((c : ℝ) + 1)) else 0)) (fun _ _ => le_rfl) N
    obtain ⟨D, hD⟩ : ∃ D, D = (1 + 2 * (τ / γ)) ^ 2 * q ^ 2 / (((x : ℝ) + 1) * ((M : ℝ) + 1)) :=
      ⟨_, rfl⟩
    have hD0 : 0 ≤ D := by rw [hD]; positivity
    have hkey : τ + 1 + τ / γ * ratioPow γ x M ≤ (1 + 2 * (τ / γ)) * q := by
      nlinarith [mul_le_mul_of_nonneg_left hqM hκ, mul_le_mul_of_nonneg_left hq hκ]
    calc _ ≤ (1 + 2 * (τ / γ)) ^ 2 * (q ^ 2 / ((M : ℝ) + 1)) *
          (τ / ((x : ℝ) + 1) + (1 + τ * ratioPow γ x M / γ) / ((x : ℝ) + 1)) :=
          mul_le_mul_of_nonneg_left (add_le_add h1 h2) (by positivity)
      _ = D * (τ + 1 + τ / γ * ratioPow γ x M) := by
          rw [hD]; simp only [div_eq_mul_inv, mul_inv]; ring
      _ ≤ D * ((1 + 2 * (τ / γ)) * q) := mul_le_mul_of_nonneg_left hkey hD0
      _ = (1 + 2 * (τ / γ)) ^ 3 * (q ^ 3 * (1 / (((x : ℝ) + 1) * ((M : ℝ) + 1)))) := by
          rw [hD]; simp only [div_eq_mul_inv, mul_inv]; ring

/-- The hypotheses of `crossProfile_le` are satisfiable: `α = 1`, `γ = 1/2`, `T = 1`,
`n = 0 < M = 1`. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℕ) < 1 := by
  norm_num

end Kinetic
end Transformer

/-
# The envelope of a second difference

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 3: the second difference of
an observable of token `x` in the tokens `l ≠ m` is at most
`K' π² e^{2L₁T} (ω V_x + Ĝ^l_x Ĝ^m_x)` (`abs_obs_second_diff_le`). For `n < M`
both terms are multiples of `E(x; n, M) = q_γ(x, n)³ G(x, M)` (`pairEnv`), so the
bracket is at most `(ωT(1 + 2κ)³ + τ + τ²)(E(x; l, m) + E(x; m, l))`
(`pair_le_pairEnv`): the source's `|∂²_{θ_l θ_m} θ_x| ≲ (x max(l, m))⁻¹ e^{…}`.
-/

import Transformer.Kinetic.Section3_SupersolutionBound

namespace Transformer
namespace Kinetic

theorem pairDecay_self (x : ℕ) : pairDecay x x = 1 / ((x : ℝ) + 1) := by
  simp [pairDecay]

theorem pairDecay_of_lt {x M : ℕ} (h : M < x) :
    pairDecay x M = 1 / (((x : ℝ) + 1) * ((M : ℝ) + 1)) := by
  rw [pairDecay, ite_eq_right h.ne, ite_eq_left h, zero_add]

theorem pairDecay_of_gt {x M : ℕ} (h : x < M) : pairDecay x M = 0 := by
  rw [pairDecay, ite_eq_right h.ne', ite_eq_right (by omega), add_zero]

theorem profileEnv_self (γ τ : ℝ) (x : ℕ) : profileEnv γ τ x x = 1 := by
  simp [profileEnv, chainEnv]

theorem profileEnv_of_lt {γ τ : ℝ} {x c : ℕ} (h : c < x) :
    profileEnv γ τ x c = τ * (ratioPow γ x c / ((x : ℝ) + 1)) := by
  rw [profileEnv, chainEnv, ite_eq_right h.ne, ite_eq_left h, zero_add]

theorem profileEnv_of_gt {γ τ : ℝ} {x c : ℕ} (h : x < c) : profileEnv γ τ x c = 0 := by
  rw [profileEnv, chainEnv, ite_eq_right h.ne', ite_eq_right (by omega), mul_zero, add_zero]

/-- `E(x; n, M) = [n < M] q_γ(x, n)³ G(x, M)`. -/
noncomputable def pairEnv (γ : ℝ) (x n M : ℕ) : ℝ :=
  if n < M then ratioPow γ x n ^ 3 * pairDecay x M else 0

theorem pairEnv_nonneg (γ : ℝ) (x n M : ℕ) : 0 ≤ pairEnv γ x n M := by
  unfold pairEnv
  have := ratioPow_pos γ x n
  have := pairDecay_nonneg x M
  split_ifs <;> positivity

theorem pairEnv_self (γ : ℝ) (x n : ℕ) : pairEnv γ x n n = 0 := by
  simp [pairEnv]

theorem crossProfile_comm (α γ T : ℝ) (N l m a : ℕ) (t : ℝ) :
    crossProfile α γ T N l m a t = crossProfile α γ T N m l a t := by
  unfold crossProfile
  simp_rw [crossForcing_comm α γ T l m]

/-- `Ĝ^n_x Ĝ^M_x ≤ (τ + τ²) E(x; n, M)` for `n < M`. -/
theorem profile_mul_profile_le_pairEnv {α γ T : ℝ} (hα : 0 ≤ α) (hγ0 : 0 < γ) (hT : 0 ≤ T)
    {n M : ℕ} (hnM : n < M) (x : ℕ) :
    profile α γ n x T * profile α γ M x T ≤
      (profileRate α γ T + profileRate α γ T ^ 2) * pairEnv γ x n M := by
  have hτ : 0 ≤ profileRate α γ T := profileRate_nonneg hα hT
  have hx1 : (0 : ℝ) < (x : ℝ) + 1 := by positivity
  have hq0 := ratioPow_pos γ x n
  have hττ : profileRate α γ T ≤ profileRate α γ T + profileRate α γ T ^ 2 :=
    le_add_of_nonneg_right (sq_nonneg _)
  rw [pairEnv, ite_eq_left hnM]
  rcases lt_trichotomy x M with hxM | hxM | hMx
  · rw [profile_eq_zero_of_lt hxM, mul_zero]
    have := pairDecay_nonneg x M
    positivity
  · subst hxM
    have hq : 1 ≤ ratioPow γ x n := one_le_ratioPow hγ0.le hnM.le
    have h3 : ratioPow γ x n ≤ ratioPow γ x n ^ 3 := by
      simpa using pow_le_pow_right₀ hq (by norm_num : 1 ≤ 3)
    rw [profile_self, mul_one, pairDecay_self]
    calc profile α γ n x T ≤ profileRate α γ T * ratioPow γ x n * (1 / ((x : ℝ) + 1)) := by
          rw [mul_one_div]; exact profile_le_rate hγ0 hnM
      _ ≤ (profileRate α γ T + profileRate α γ T ^ 2) * ratioPow γ x n ^ 3 *
            (1 / ((x : ℝ) + 1)) :=
          mul_le_mul_of_nonneg_right (mul_le_mul hττ h3 hq0.le (by positivity)) (by positivity)
      _ = _ := by ring
  · have hq : 1 ≤ ratioPow γ x n := one_le_ratioPow hγ0.le (hnM.trans hMx).le
    have hqM : ratioPow γ x M ≤ ratioPow γ x n := ratioPow_le_ratioPow_right hγ0.le x hnM.le
    have hM1 : (0 : ℝ) < (M : ℝ) + 1 := by positivity
    have hcM : (M : ℝ) + 1 ≤ (x : ℝ) + 1 := by exact_mod_cast Nat.add_le_add_right hMx.le 1
    rw [pairDecay_of_lt hMx]
    have h1 := profile_le_rate (α := α) (T := T) hγ0 (hnM.trans hMx)
    have h2 : profile α γ M x T ≤ profileRate α γ T * ratioPow γ x n / ((M : ℝ) + 1) :=
      (profile_le_rate hγ0 hMx).trans
        (div_le_div₀ (by positivity) (mul_le_mul_of_nonneg_left hqM hτ) hM1 hcM)
    have h3 : ratioPow γ x n ^ 2 ≤ ratioPow γ x n ^ 3 := pow_le_pow_right₀ hq (by norm_num)
    have hτ2 : profileRate α γ T ^ 2 ≤ profileRate α γ T + profileRate α γ T ^ 2 :=
      le_add_of_nonneg_left hτ
    calc profile α γ n x T * profile α γ M x T ≤
          (profileRate α γ T * ratioPow γ x n / ((x : ℝ) + 1)) *
            (profileRate α γ T * ratioPow γ x n / ((M : ℝ) + 1)) :=
          mul_le_mul h1 h2 (profile_nonneg hα hγ0 M x hT) (by positivity)
      _ = profileRate α γ T ^ 2 * ratioPow γ x n ^ 2 *
            (1 / (((x : ℝ) + 1) * ((M : ℝ) + 1))) := by
          simp only [div_eq_mul_inv, mul_inv]; ring
      _ ≤ (profileRate α γ T + profileRate α γ T ^ 2) * ratioPow γ x n ^ 3 *
            (1 / (((x : ℝ) + 1) * ((M : ℝ) + 1))) :=
          mul_le_mul_of_nonneg_right (mul_le_mul hτ2 h3 (by positivity) (by positivity))
            (by positivity)
      _ = _ := by ring

/-- **The envelope of a second difference.** For `l ≠ m` and `ω ≥ 0`,
`ω V_x + Ĝ^l_x Ĝ^m_x ≤ (ωT(1 + 2κ)³ + τ + τ²)(E(x; l, m) + E(x; m, l))`. -/
theorem pair_le_pairEnv {α γ T ω : ℝ} (hα : 0 ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) (hT : 0 ≤ T)
    (hω : 0 ≤ ω) (N : ℕ) {l m : ℕ} (hlm : l ≠ m) (x : ℕ) :
    ω * crossProfile α γ T N l m x T + profile α γ l x T * profile α γ m x T ≤
      (ω * T * (1 + 2 * (profileRate α γ T / γ)) ^ 3 + profileRate α γ T +
        profileRate α γ T ^ 2) * (pairEnv γ x l m + pairEnv γ x m l) := by
  have key : ∀ {n M : ℕ}, n < M → ω * crossProfile α γ T N n M x T +
      profile α γ n x T * profile α γ M x T ≤
      (ω * T * (1 + 2 * (profileRate α γ T / γ)) ^ 3 + profileRate α γ T +
        profileRate α γ T ^ 2) * pairEnv γ x n M := by
    intro n M hnM
    have h1 := mul_le_mul_of_nonneg_left (crossProfile_le hα hγ0 hγ1 hT N hnM x) hω
    have h2 := profile_mul_profile_le_pairEnv hα hγ0 hT hnM x
    rw [pairEnv, ite_eq_left hnM] at h2 ⊢
    linarith
  rcases Nat.lt_or_gt_of_ne hlm with h | h
  · rw [show pairEnv γ x m l = 0 by rw [pairEnv, ite_eq_right (by omega)], add_zero]
    exact key h
  · rw [show pairEnv γ x l m = 0 by rw [pairEnv, ite_eq_right (by omega)], zero_add,
      crossProfile_comm, mul_comm (profile α γ l x T)]
    exact key h

/-- The hypotheses of `pair_le_pairEnv` are satisfiable: `α = ω = 1`, `γ = 1/2`,
`T = 1`, `l = 0 ≠ m = 1`. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℕ) ≠ 1 := by norm_num

end Kinetic
end Transformer

/-
# The forcing of a pair and Duhamel's supersolution

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 3, `eq:est-Glaub-2nd`:
for the pair `n < M`, with `τ = αT e^{αT/γ}` and `κ = τ/γ`, the forcing is at
most `σ_c ≤ (1 + 2κ)² q_γ(c, n)² G(c, M)` (`crossForcing_le`), and Duhamel's
supersolution at time `T` is at most `V_x ≤ T (1 + 2κ)³ q_γ(x, n)³ G(x, M)`
(`crossProfile_le`). This is the source's
`|∂²_{θ_n θ_M} θ_x| ≲ (xM)⁻¹ e^{√(CT log(x/n))}`.
-/

import Transformer.Kinetic.Section3_ForcingEnvelope

open Real

namespace Transformer
namespace Kinetic

theorem crossForcing_comm (α γ T : ℝ) (l m c : ℕ) :
    crossForcing α γ T l m c = crossForcing α γ T m l c := by
  unfold crossForcing
  congr 1
  exact Finset.sum_congr rfl fun b _ => mul_comm _ _

/-- No forcing before the later token: `σ_c = 0` for `c < M`. -/
theorem crossForcing_eq_zero_of_lt {α γ T : ℝ} {l M c : ℕ} (hc : c < M) :
    crossForcing α γ T l M c = 0 := by
  unfold crossForcing
  rw [Finset.sum_eq_zero fun b hb => by
    rw [profile_eq_zero_of_lt hc, profile_eq_zero_of_lt ((Finset.mem_range.1 hb).trans hc)]
    ring, zero_div]

/-- **The forcing of a pair.** -/
theorem crossForcing_le {α γ T : ℝ} (hα : 0 ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) (hT : 0 ≤ T)
    {n M : ℕ} (hnM : n < M) (c : ℕ) :
    crossForcing α γ T n M c ≤
      (1 + 2 * (profileRate α γ T / γ)) ^ 2 * (ratioPow γ c n ^ 2 * pairDecay c M) := by
  set τ := profileRate α γ T
  set κ := τ / γ with hκdef
  have hτ : 0 ≤ τ := profileRate_nonneg hα hT
  have hκ : 0 ≤ κ := div_nonneg hτ hγ0.le
  have hτκ : τ ≤ κ := by rw [hκdef, le_div_iff₀ hγ0]; nlinarith
  have hc1 : (0 : ℝ) < (c : ℝ) + 1 := by positivity
  have hD := pairDecay_nonneg c M
  have hq0 := ratioPow_pos γ c n
  rcases lt_trichotomy c M with hc | hcM | hc
  · rw [crossForcing_eq_zero_of_lt hc]
    positivity
  all_goals unfold crossForcing
  · subst hcM
    have hsum : ∑ b ∈ Finset.range c, (profile α γ c c T + profile α γ c b T) *
        (profile α γ n c T + profile α γ n b T) =
        c * profile α γ n c T + ∑ b ∈ Finset.range c, profile α γ n b T := by
      rw [Finset.sum_congr rfl fun b hb => by
        rw [profile_self, profile_eq_zero_of_lt (Finset.mem_range.1 hb), add_zero, one_mul],
        Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    rw [hsum, div_le_iff₀ hc1]
    have hq := one_le_ratioPow hγ0.le hnM.le
    have h1 : (c : ℝ) * profile α γ n c T ≤ τ * ratioPow γ c n := by
      have := profile_le_rate (α := α) (T := T) hγ0 hnM
      calc (c : ℝ) * profile α γ n c T ≤ c * (τ * ratioPow γ c n / ((c : ℝ) + 1)) :=
            mul_le_mul_of_nonneg_left this (by positivity)
        _ = τ * ratioPow γ c n * ((c : ℝ) / ((c : ℝ) + 1)) := by ring
        _ ≤ τ * ratioPow γ c n * 1 := mul_le_mul_of_nonneg_left
            ((div_le_one hc1).2 (by linarith)) (by positivity)
        _ = τ * ratioPow γ c n := mul_one _
    have h2 : ∑ b ∈ Finset.range c, profile α γ n b T ≤ (1 + κ) * ratioPow γ c n :=
      sum_profile_col_le hα hγ0 hγ1 hT c n
    have hG : pairDecay c c = 1 / ((c : ℝ) + 1) := by simp [pairDecay]
    rw [hG, show (1 + 2 * κ) ^ 2 * (ratioPow γ c n ^ 2 * (1 / ((c : ℝ) + 1))) * ((c : ℝ) + 1) =
      (1 + 2 * κ) ^ 2 * ratioPow γ c n ^ 2 by field_simp]
    have hqq : ratioPow γ c n ≤ ratioPow γ c n ^ 2 := by nlinarith
    have hk : 1 + 2 * κ ≤ (1 + 2 * κ) ^ 2 := by nlinarith
    have h3 := mul_le_mul_of_nonneg_left hqq (by linarith : (0 : ℝ) ≤ 1 + 2 * κ)
    have h4 := mul_le_mul_of_nonneg_right hk (sq_nonneg (ratioPow γ c n))
    nlinarith
  · set a := profile α γ M c T
    set d := profile α γ n c T
    set q := ratioPow γ c n
    set X := q ^ 2 / ((M : ℝ) + 1)
    have hM1 : (0 : ℝ) < (M : ℝ) + 1 := by positivity
    have hMc : (M : ℝ) + 1 ≤ c := by exact_mod_cast hc
    have hX : 0 ≤ X := by positivity
    have hexp : ∑ b ∈ Finset.range c, (a + profile α γ M b T) * (d + profile α γ n b T) =
        c * a * d + a * ∑ b ∈ Finset.range c, profile α γ n b T +
          d * ∑ b ∈ Finset.range c, profile α γ M b T +
          ∑ b ∈ Finset.range c, profile α γ M b T * profile α γ n b T := by
      simp only [add_mul, mul_add, Finset.sum_add_distrib, Finset.sum_const, Finset.card_range,
        nsmul_eq_mul, Finset.mul_sum]
      ring_nf
    rw [hexp, div_le_iff₀ hc1]
    have ha0 : 0 ≤ a := profile_nonneg hα hγ0 M c hT
    have hd0 : 0 ≤ d := profile_nonneg hα hγ0 n c hT
    have hqM : ratioPow γ c M ≤ q := ratioPow_le_ratioPow_right hγ0.le c hnM.le
    have hq := one_le_ratioPow hγ0.le (hnM.trans hc).le
    have hdiv : 1 / ((c : ℝ) + 1) ≤ 1 / ((M : ℝ) + 1) :=
      one_div_le_one_div_of_le hM1 (by linarith)
    have hcM : (M : ℝ) + 1 ≤ (c : ℝ) + 1 := by linarith
    have ha : a ≤ τ * q / ((M : ℝ) + 1) := by
      have h := profile_le_rate (α := α) (T := T) hγ0 hc
      calc a ≤ τ * ratioPow γ c M / ((c : ℝ) + 1) := h
        _ ≤ τ * q / ((M : ℝ) + 1) :=
          div_le_div₀ (mul_nonneg hτ hq0.le) (mul_le_mul_of_nonneg_left hqM hτ) hM1 hcM
    have hd : d ≤ τ * q / ((M : ℝ) + 1) := by
      have h := profile_le_rate (α := α) (T := T) hγ0 (hnM.trans hc)
      calc d ≤ τ * q / ((c : ℝ) + 1) := h
        _ ≤ τ * q / ((M : ℝ) + 1) := div_le_div_of_nonneg_left (mul_nonneg hτ hq0.le) hM1 hcM
    have hca : (c : ℝ) * a ≤ τ * q := by
      have h := profile_le_rate (α := α) (T := T) hγ0 hc
      calc (c : ℝ) * a ≤ c * (τ * ratioPow γ c M / ((c : ℝ) + 1)) :=
            mul_le_mul_of_nonneg_left h (by positivity)
        _ = τ * ratioPow γ c M * ((c : ℝ) / ((c : ℝ) + 1)) := by ring
        _ ≤ τ * q * 1 := mul_le_mul (mul_le_mul_of_nonneg_left hqM hτ)
            ((div_le_one hc1).2 (by linarith)) (by positivity) (by positivity)
        _ = τ * q := mul_one _
    have hV := sum_profile_col_le hα hγ0 hγ1 hT c n
    have hU := (sum_profile_col_le hα hγ0 hγ1 hT c M).trans
      (mul_le_mul_of_nonneg_left hqM (by positivity))
    have hUV := sum_profile_mul_profile_le hα hγ0 hγ1 hT hnM hc
    have h1 : (c : ℝ) * a * d ≤ τ ^ 2 * X := by
      calc (c : ℝ) * a * d ≤ τ * q * (τ * q / ((M : ℝ) + 1)) :=
            mul_le_mul hca hd hd0 (by positivity)
        _ = τ ^ 2 * X := by unfold X; ring
    have h2 : a * ∑ b ∈ Finset.range c, profile α γ n b T ≤ τ * (1 + κ) * X := by
      calc a * ∑ b ∈ Finset.range c, profile α γ n b T ≤
            τ * q / ((M : ℝ) + 1) * ((1 + κ) * q) :=
            mul_le_mul ha hV (Finset.sum_nonneg fun b _ => profile_nonneg hα hγ0 n b hT)
              (by positivity)
        _ = τ * (1 + κ) * X := by unfold X; ring
    have h3 : d * ∑ b ∈ Finset.range c, profile α γ M b T ≤ τ * (1 + κ) * X := by
      calc d * ∑ b ∈ Finset.range c, profile α γ M b T ≤
            τ * q / ((M : ℝ) + 1) * ((1 + κ) * q) :=
            mul_le_mul hd hU (Finset.sum_nonneg fun b _ => profile_nonneg hα hγ0 M b hT)
              (by positivity)
        _ = τ * (1 + κ) * X := by unfold X; ring
    have hG : pairDecay c M = 1 / (((c : ℝ) + 1) * ((M : ℝ) + 1)) := by
      rw [pairDecay, ite_eq_right hc.ne, ite_eq_left hc, zero_add]
    rw [hG, show (1 + 2 * κ) ^ 2 * (q ^ 2 * (1 / (((c : ℝ) + 1) * ((M : ℝ) + 1)))) *
      ((c : ℝ) + 1) = (1 + 2 * κ) ^ 2 * X by unfold X; field_simp]
    have hcoef : τ ^ 2 + τ * (1 + κ) + τ * (1 + κ) + (τ + τ * κ) ≤ (1 + 2 * κ) ^ 2 := by
      nlinarith
    nlinarith [mul_le_mul_of_nonneg_right hcoef hX]

/-- The hypotheses of `crossForcing_le` are satisfiable: `α = 1`, `γ = 1/2`, `T = 1`,
`n = 0 < M = 1`. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℕ) < 1 := by
  norm_num

end Kinetic
end Transformer

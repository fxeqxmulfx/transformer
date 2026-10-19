module

public import Transformer.Kinetic.Section3_Envelopes
public import Transformer.Kinetic.Section3_SecondDifferences

/-!
# The envelope of the forcing of a pair

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 3: the forcing
`σ_c = (c+1)⁻¹ Σ_{b<c} (Ĝ^M_c + Ĝ^M_b)(Ĝ^n_c + Ĝ^n_b)` of Duhamel's
supersolution for the pair `n < M`. With `τ = αT e^{αT/γ}` and `κ = τ/γ` it is at
most `(1 + 2κ)² q_γ(c, n)² G(c, M)` (`crossForcing_le`), where
`G(c, M) = [M = c]/(c+1) + [M < c]/((c+1)(M+1))` is the decay of a pair whose
later token is `M`. Every `1/γ` comes with a `τ`, hence with a factor `T`.
-/

@[expose] public section

open Real

namespace Transformer
namespace Kinetic

/-- `G(x, M) = [M = x]/(x+1) + [M < x]/((x+1)(M+1))`. -/
noncomputable def pairDecay (x M : ℕ) : ℝ :=
  (if M = x then 1 / ((x : ℝ) + 1) else 0) +
    (if M < x then 1 / (((x : ℝ) + 1) * ((M : ℝ) + 1)) else 0)

theorem pairDecay_nonneg (x M : ℕ) : 0 ≤ pairDecay x M := by
  unfold pairDecay
  exact add_nonneg (by split_ifs <;> positivity) (by split_ifs <;> positivity)

theorem profile_eq_zero_of_lt {α γ t : ℝ} {x c : ℕ} (h : x < c) : profile α γ c x t = 0 := by
  rw [profile, ite_eq_left h]

theorem profile_self (α γ t : ℝ) (x : ℕ) : profile α γ x x t = 1 := by
  simp [profile]

theorem profile_le_rate {α γ T : ℝ} (hγ : 0 < γ) {x c : ℕ} (h : c < x) :
    profile α γ c x T ≤ profileRate α γ T * ratioPow γ x c / ((x : ℝ) + 1) :=
  profile_le α hγ h T

/-- **The column sum.** `Σ_{b<c} Ĝ^n_b(T) ≤ (1 + τ/γ) q_γ(c, n)`. -/
theorem sum_profile_col_le {α γ T : ℝ} (hα : 0 ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) (hT : 0 ≤ T)
    (c n : ℕ) : ∑ b ∈ Finset.range c, profile α γ n b T ≤
      (1 + profileRate α γ T / γ) * ratioPow γ c n := by
  set τ := profileRate α γ T
  have hτ : 0 ≤ τ := profileRate_nonneg hα hT
  have hpt : ∀ b ∈ Finset.range c, profile α γ n b T ≤ (if n = b then 1 else 0) +
      (if b ∈ Finset.Ico (n + 1) c then τ * (ratioPow γ b n / ((b : ℝ) + 1)) else 0) := by
    intro b hb
    refine (profile_le_env hγ0 b n).trans (le_of_eq ?_)
    rw [profileEnv, chainEnv, mul_ite, mul_zero]
    congr 1
    by_cases h : n < b
    · rw [ite_eq_left h, ite_eq_left (Finset.mem_Ico.2 ⟨h, Finset.mem_range.1 hb⟩)]
    · rw [ite_eq_right h, ite_eq_right fun h' => h (Finset.mem_Ico.1 h').1]
  refine (Finset.sum_le_sum hpt).trans ?_
  rw [Finset.sum_add_distrib, Finset.sum_ite_eq]
  have h2 := (sum_range_ite_le (f := fun b => τ * (ratioPow γ b n / ((b : ℝ) + 1)))
    (fun b => by have := ratioPow_pos γ b n; positivity) c (n + 1) c).trans_eq
    (Finset.mul_sum _ _ _).symm
  have h3 := mul_le_mul_of_nonneg_left (sum_Ico_ratioPow_div_le hγ0 hγ1 n c) hτ
  have hq0 := ratioPow_pos γ c n
  have h1 : (if n ∈ Finset.range c then (1 : ℝ) else 0) ≤ ratioPow γ c n := by
    split_ifs with h
    · exact one_le_ratioPow hγ0.le (Finset.mem_range.1 h).le
    · exact hq0.le
  have : τ * (ratioPow γ c n / γ) = τ / γ * ratioPow γ c n := by ring
  linarith

/-- The diagonal part `Σ_{b<c} Ĝ^M_b Ĝ^n_b ≤ (τ + τκ) q_γ(c, n)²/(M+1)` for `n < M < c`. -/
theorem sum_profile_mul_profile_le {α γ T : ℝ} (hα : 0 ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1)
    (hT : 0 ≤ T) {n M c : ℕ} (hnM : n < M) (hMc : M < c) :
    ∑ b ∈ Finset.range c, profile α γ M b T * profile α γ n b T ≤
      (profileRate α γ T + profileRate α γ T * (profileRate α γ T / γ)) *
        (ratioPow γ c n ^ 2 / ((M : ℝ) + 1)) := by
  set τ := profileRate α γ T
  have hτ : 0 ≤ τ := profileRate_nonneg hα hT
  have hM1 : (0 : ℝ) < (M : ℝ) + 1 := by positivity
  set e := τ * ratioPow γ c n / ((M : ℝ) + 1)
  have he : 0 ≤ e := by have := ratioPow_pos γ c n; positivity
  have hnn : ∀ b : ℕ, 0 ≤ e * (τ * (ratioPow γ b M / ((b : ℝ) + 1))) := fun b =>
    mul_nonneg he (mul_nonneg hτ (by have := ratioPow_pos γ b M; positivity))
  have hpt : ∀ b ∈ Finset.range c, profile α γ M b T * profile α γ n b T ≤
      (if M = b then e else 0) +
        (if b ∈ Finset.Ico (M + 1) c then e * (τ * (ratioPow γ b M / ((b : ℝ) + 1))) else 0) := by
    intro b hb
    have hbc := Finset.mem_range.1 hb
    rcases lt_trichotomy b M with h | rfl | h
    · rw [profile_eq_zero_of_lt h, zero_mul]
      exact add_nonneg (by split_ifs <;> first | exact he | exact le_rfl)
        (by split_ifs <;> first | exact hnn b | exact le_rfl)
    · rw [profile_self, one_mul, ite_eq_left rfl,
        ite_eq_right fun h' => by have := (Finset.mem_Ico.1 h').1; omega, add_zero]
      refine (profile_le_rate hγ0 hnM).trans ?_
      have := ratioPow_le_ratioPow_left hγ0.le hbc.le n
      unfold e
      gcongr
    · rw [ite_eq_right h.ne, zero_add, ite_eq_left (Finset.mem_Ico.2 ⟨h, hbc⟩)]
      have h1 := profile_le_rate (T := T) (α := α) hγ0 h
      have h2 : profile α γ n b T ≤ e := by
        refine (profile_le_rate hγ0 (hnM.trans h)).trans ?_
        unfold e
        have hqb := ratioPow_le_ratioPow_left hγ0.le hbc.le n
        have hMb : ((M : ℝ) + 1) ≤ (b : ℝ) + 1 := by exact_mod_cast Nat.add_le_add_right h.le 1
        have hq0 := ratioPow_pos γ c n
        rw [div_le_div_iff₀ (by positivity) hM1]
        exact mul_le_mul (mul_le_mul_of_nonneg_left hqb hτ) hMb hM1.le (by positivity)
      calc profile α γ M b T * profile α γ n b T ≤ (τ * ratioPow γ b M / ((b : ℝ) + 1)) * e :=
            mul_le_mul h1 h2 (profile_nonneg hα hγ0 n b hT) (by
              have := ratioPow_pos γ b M; positivity)
        _ = e * (τ * (ratioPow γ b M / ((b : ℝ) + 1))) := by ring
  refine (Finset.sum_le_sum hpt).trans ?_
  rw [Finset.sum_add_distrib, Finset.sum_ite_eq]
  have h2 := (sum_range_ite_le (f := fun b => e * (τ * (ratioPow γ b M / ((b : ℝ) + 1))))
    hnn c (M + 1) c).trans_eq
    (by rw [← Finset.mul_sum, ← Finset.mul_sum])
  have h3 := mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
    (sum_Ico_ratioPow_div_le hγ0 hγ1 M c) hτ) he
  have h1 : (if M ∈ Finset.range c then e else 0) ≤ e := by split_ifs <;> linarith
  have hq := one_le_ratioPow hγ0.le (hnM.trans hMc).le
  have hqM := ratioPow_le_ratioPow_right hγ0.le c hnM.le
  have hfin : e + e * (τ * (ratioPow γ c M / γ)) ≤
      (τ + τ * (τ / γ)) * (ratioPow γ c n ^ 2 / ((M : ℝ) + 1)) := by
    have hr : (τ + τ * (τ / γ)) * (ratioPow γ c n ^ 2 / ((M : ℝ) + 1)) =
        e * (ratioPow γ c n + τ * (ratioPow γ c n / γ)) := by
      unfold e
      ring
    rw [hr]
    have hk : τ * (ratioPow γ c M / γ) ≤ τ * (ratioPow γ c n / γ) :=
      mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right hqM hγ0.le) hτ
    nlinarith [mul_le_mul_of_nonneg_left hk he, mul_le_mul_of_nonneg_left hq he]
  linarith

/-- The hypotheses of `profile_eq_zero_of_lt`, `profile_le_rate`, `sum_profile_col_le` and
`sum_profile_mul_profile_le` hold at `α = T = 1`, `γ = 1/2` (`γ = 1` for the first two) and
the tokens `0 < 1 < 2`. -/
example := And.intro (profile_eq_zero_of_lt (α := 1) (γ := 1) (t := 0) (x := 0) (c := 1)
    Nat.one_pos) <|
  And.intro (profile_le_rate (α := 1) (γ := 1) (T := 1) one_pos (x := 1) (c := 0)
    Nat.one_pos) <|
  And.intro (sum_profile_col_le (α := 1) (γ := 1 / 2) (T := 1) zero_le_one (by norm_num)
    (by norm_num) zero_le_one 1 0)
  (sum_profile_mul_profile_le (α := 1) (γ := 1 / 2) (T := 1) zero_le_one (by norm_num)
    (by norm_num) zero_le_one (n := 0) (M := 1) (c := 2) Nat.one_pos (by norm_num))

end Kinetic
end Transformer

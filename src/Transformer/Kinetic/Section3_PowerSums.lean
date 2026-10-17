/-
# Power sums of the chain ratio

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Steps 2 and 4: the sums
over tokens in `eq:est-cov-phi` and `eq:est-kap3-th`. For `s ≤ 1/2` the ratio
`q_s(x, c) = ((x+1)/(c+1))^s` sums to `Σ_{c<x} q_s(x, c) ≤ 2(x+1)`
(`sum_ratioPow_le`), by `sum_rpow_le` at the exponent `1 - s`, and a chain
`n < M < x` has `Σ 1/(M+1) ≤ q_γ(x, n)/γ` (`sum_Ico_inv_le`).
-/

import Transformer.Kinetic.Section3_ChainSums

open Real

namespace Transformer
namespace Kinetic

theorem ratioPow_pow (γ : ℝ) (r a b : ℕ) : ratioPow γ a b ^ r = ratioPow (r * γ) a b := by
  unfold ratioPow
  rw [← Real.rpow_natCast, ← Real.rpow_mul (by positivity), mul_comm]

theorem ratioPow_le_ratioPow_left {γ : ℝ} (hγ : 0 ≤ γ) {a a' : ℕ} (h : a ≤ a') (b : ℕ) :
    ratioPow γ a b ≤ ratioPow γ a' b := by
  unfold ratioPow
  refine Real.rpow_le_rpow (by positivity) (div_le_div_of_nonneg_right ?_ (by positivity)) hγ
  exact_mod_cast Nat.add_le_add_right h 1

theorem ratioPow_le_ratioPow_right {γ : ℝ} (hγ : 0 ≤ γ) (a : ℕ) {b b' : ℕ} (h : b ≤ b') :
    ratioPow γ a b' ≤ ratioPow γ a b := by
  unfold ratioPow
  refine Real.rpow_le_rpow (by positivity) (div_le_div_of_nonneg_left (by positivity)
    (by positivity) ?_) hγ
  exact_mod_cast Nat.add_le_add_right h 1

/-- **The power sum.** `Σ_{c<x} q_s(x, c) ≤ 2(x+1)` for `0 ≤ s ≤ 1/2`. -/
theorem sum_ratioPow_le {s : ℝ} (hs0 : 0 ≤ s) (hs : s ≤ 1 / 2) (x : ℕ) :
    ∑ c ∈ Finset.range x, ratioPow s x c ≤ 2 * ((x : ℝ) + 1) := by
  rcases Nat.eq_zero_or_pos x with rfl | hx
  · simp
  have hx1 : (0 : ℝ) < (x : ℝ) + 1 := by positivity
  have hterm : ∀ c : ℕ, ratioPow s x c = ((x : ℝ) + 1) ^ s * ((c : ℝ) + 1) ^ (-s) := by
    intro c
    rw [ratioPow, Real.div_rpow hx1.le (by positivity), Real.rpow_neg (by positivity),
      div_eq_mul_inv]
  simp_rw [hterm, ← Finset.mul_sum]
  have h1 := sum_rpow_le (γ := 1 - s) (by linarith) (by linarith) 0 x (by omega)
  rw [show (1 - s) - 1 = -s by ring] at h1
  simp only [Nat.cast_zero, zero_add, Real.one_rpow] at h1
  rw [Finset.range_eq_Ico, Finset.sum_eq_sum_Ico_succ_bot hx]
  simp only [Nat.cast_zero, zero_add, Real.one_rpow]
  set S := ∑ k ∈ Finset.Ico 1 x, ((k : ℝ) + 1) ^ (-s)
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun k _ => by positivity
  have hxs : (x : ℝ) ^ (1 - s) ≤ ((x : ℝ) + 1) ^ (1 - s) :=
    Real.rpow_le_rpow (by positivity) (by linarith) (by linarith)
  have hS : 1 + S ≤ 2 * ((x : ℝ) + 1) ^ (1 - s) := by nlinarith
  calc ((x : ℝ) + 1) ^ s * (1 + S) ≤ ((x : ℝ) + 1) ^ s * (2 * ((x : ℝ) + 1) ^ (1 - s)) :=
        mul_le_mul_of_nonneg_left hS (by positivity)
    _ = 2 * ((x : ℝ) + 1) := by
        rw [mul_left_comm, ← Real.rpow_add hx1, show s + (1 - s) = 1 by ring, Real.rpow_one]

/-- `Σ_{c<x} q_γ(x, c)^r ≤ 2(x+1)` for `rγ ≤ 1/2`. -/
theorem sum_ratioPow_pow_le {γ : ℝ} (hγ : 0 ≤ γ) {r : ℕ} (hr : (r : ℝ) * γ ≤ 1 / 2) (x : ℕ) :
    ∑ c ∈ Finset.range x, ratioPow γ x c ^ r ≤ 2 * ((x : ℝ) + 1) := by
  simp_rw [ratioPow_pow]
  exact sum_ratioPow_le (by positivity) hr x

/-- `Σ_{n<M<x} q_γ(M, n)/(M+1) ≤ q_γ(x, n)/γ`, also when the chain is empty. -/
theorem sum_Ico_ratioPow_div_le {γ : ℝ} (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) (n x : ℕ) :
    ∑ M ∈ Finset.Ico (n + 1) x, ratioPow γ M n / ((M : ℝ) + 1) ≤ ratioPow γ x n / γ := by
  rcases lt_or_ge n x with h | h
  · exact (sum_ratioPow_div_le hγ0 hγ1 h).trans
      (div_le_div_of_nonneg_right (by linarith) hγ0.le)
  · rw [Finset.Ico_eq_empty_of_le (by omega), Finset.sum_empty]
    exact div_nonneg (ratioPow_pos γ x n).le hγ0.le

/-- `Σ_{n<M<x} 1/(M+1) ≤ q_γ(x, n)/γ`. -/
theorem sum_Ico_inv_le {γ : ℝ} (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) (n x : ℕ) :
    ∑ M ∈ Finset.Ico (n + 1) x, 1 / ((M : ℝ) + 1) ≤ ratioPow γ x n / γ :=
  (Finset.sum_le_sum fun M hM => div_le_div_of_nonneg_right
    (one_le_ratioPow hγ0.le (by have := (Finset.mem_Ico.1 hM).1; omega))
    (by positivity)).trans (sum_Ico_ratioPow_div_le hγ0 hγ1 n x)

/-- A sum over `range N` of terms that vanish off `[a, b)` is at most the sum over `[a, b)`. -/
theorem sum_range_ite_le {f : ℕ → ℝ} (hf : ∀ M, 0 ≤ f M) (N a b : ℕ) :
    ∑ M ∈ Finset.range N, (if M ∈ Finset.Ico a b then f M else 0) ≤
      ∑ M ∈ Finset.Ico a b, f M := by
  rw [← Finset.sum_filter]
  exact Finset.sum_le_sum_of_subset_of_nonneg (fun M hM => (Finset.mem_filter.1 hM).2)
    fun M _ _ => hf M

/-- The hypotheses of `ratioPow_le_ratioPow_left`, `ratioPow_le_ratioPow_right`,
`sum_Ico_ratioPow_div_le`, `sum_Ico_inv_le` and `sum_range_ite_le` hold at `γ = 1`. -/
example := And.intro (ratioPow_le_ratioPow_left (γ := 1) zero_le_one (a := 0) (a' := 1)
    zero_le_one 0) <|
  And.intro (ratioPow_le_ratioPow_right (γ := 1) zero_le_one 1 (b := 0) (b' := 1)
    zero_le_one) <|
  And.intro (sum_Ico_ratioPow_div_le (γ := 1) one_pos le_rfl 0 2) <|
  And.intro (sum_Ico_inv_le (γ := 1) one_pos le_rfl 0 2)
    (sum_range_ite_le (f := fun _ => 1) (fun _ => zero_le_one) 2 0 1)

/-- The hypotheses of `sum_ratioPow_le` and `sum_ratioPow_pow_le` hold at `s = 1/4`, and at
`γ = 1/16` with `r = 6`. -/
example := And.intro (sum_ratioPow_le (s := 1 / 4) (by norm_num) (by norm_num) 1)
  (sum_ratioPow_pow_le (γ := 1 / 16) (r := 6) (by norm_num) (by norm_num) 1)

end Kinetic
end Transformer

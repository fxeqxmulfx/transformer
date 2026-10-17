/-
# Envelopes of the profiles

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Steps 2 and 4. At time `T`
the profile is at most `[c = x] + τ F(x, c)` (`profile_le_env`), with
`τ = αT e^{αT/γ}` and `F(x, c) = q_γ(x, c)/(x+1)` for `c < x`: the source's
`Ĝ^c_x ≲ δ_{cx} + x⁻¹ e^{√(CT log(x/c))}`. Weighted by `q_γ(x, c)^r` with
`(r+1)γ ≤ 1/2` its sum is at most `1 + 2τ` (`sum_profileEnv_mul_le`), and a sum
over `M > n` of a term with the decay `G(x, M)` of a pair is at most
`(1 + c q_γ(x, n)/γ)/(x+1)` (`sum_decay_le`).
-/

import Transformer.Kinetic.Section3_PowerSums
import Transformer.Kinetic.Section3_Supersolution

open Real

namespace Transformer
namespace Kinetic

/-- `τ = αT e^{αT/γ}`, the time factor of the profile (`profile_le`). -/
noncomputable def profileRate (α γ T : ℝ) : ℝ := α * T * Real.exp (α / γ * T)

/-- `F(x, c) = q_γ(x, c)/(x+1)` for `c < x`, else `0`. -/
noncomputable def chainEnv (γ : ℝ) (x c : ℕ) : ℝ :=
  if c < x then ratioPow γ x c / ((x : ℝ) + 1) else 0

/-- The envelope `[c = x] + τ F(x, c)` of the profile `Ĝ^c_x(T)`. -/
noncomputable def profileEnv (γ τ : ℝ) (x c : ℕ) : ℝ :=
  (if c = x then 1 else 0) + τ * chainEnv γ x c

theorem profileRate_nonneg {α γ T : ℝ} (hα : 0 ≤ α) (hT : 0 ≤ T) : 0 ≤ profileRate α γ T := by
  unfold profileRate
  positivity

theorem chainEnv_nonneg (γ : ℝ) (x c : ℕ) : 0 ≤ chainEnv γ x c := by
  unfold chainEnv
  have := ratioPow_pos γ x c
  split_ifs <;> positivity

theorem profileEnv_nonneg {γ τ : ℝ} (hτ : 0 ≤ τ) (x c : ℕ) : 0 ≤ profileEnv γ τ x c :=
  add_nonneg (by split_ifs <;> norm_num) (mul_nonneg hτ (chainEnv_nonneg γ x c))

/-- **The profile under its envelope.** -/
theorem profile_le_env {α γ T : ℝ} (hγ : 0 < γ) (x c : ℕ) :
    profile α γ c x T ≤ profileEnv γ (profileRate α γ T) x c := by
  unfold profileEnv chainEnv
  rcases lt_trichotomy c x with h | rfl | h
  · rw [ite_eq_right h.ne, ite_eq_left h, zero_add]
    calc profile α γ c x T ≤ α * T * Real.exp (α / γ * T) * ratioPow γ x c / ((x : ℝ) + 1) :=
          profile_le α hγ h T
      _ = profileRate α γ T * (ratioPow γ x c / ((x : ℝ) + 1)) := by unfold profileRate; ring
  · simp [profile]
  · rw [ite_eq_right h.ne', ite_eq_right (by omega), profile, ite_eq_left h]
    simp

/-- `Σ_c q_γ(x, c)^r F(x, c) ≤ 2` for `(r+1)γ ≤ 1/2`. -/
theorem sum_chainEnv_mul_le {γ : ℝ} (hγ : 0 ≤ γ) {r : ℕ} (hr : ((r : ℝ) + 1) * γ ≤ 1 / 2)
    (N x : ℕ) : ∑ c ∈ Finset.range N, ratioPow γ x c ^ r * chainEnv γ x c ≤ 2 := by
  have hx : (0 : ℝ) < (x : ℝ) + 1 := by positivity
  unfold chainEnv
  simp_rw [mul_ite, mul_zero]
  rw [← Finset.sum_filter]
  calc ∑ c ∈ (Finset.range N).filter (· < x), ratioPow γ x c ^ r * (ratioPow γ x c / ((x : ℝ) + 1))
      ≤ ∑ c ∈ Finset.range x, ratioPow γ x c ^ r * (ratioPow γ x c / ((x : ℝ) + 1)) :=
        Finset.sum_le_sum_of_subset_of_nonneg
          (fun c hc => Finset.mem_range.2 (Finset.mem_filter.1 hc).2)
          fun c _ _ => by have := ratioPow_pos γ x c; positivity
    _ = (∑ c ∈ Finset.range x, ratioPow γ x c ^ (r + 1)) / ((x : ℝ) + 1) := by
        rw [Finset.sum_div]
        exact Finset.sum_congr rfl fun c _ => by ring
    _ ≤ 2 * ((x : ℝ) + 1) / ((x : ℝ) + 1) :=
        div_le_div_of_nonneg_right (sum_ratioPow_pow_le hγ (by push_cast; linarith) x) hx.le
    _ = 2 := by field_simp

/-- **The weighted sum of the envelope.** `Σ_c q_γ(x, c)^r ([c = x] + τ F(x, c)) ≤ 1 + 2τ`. -/
theorem sum_profileEnv_mul_le {γ τ : ℝ} (hγ : 0 ≤ γ) (hτ : 0 ≤ τ) {r : ℕ}
    (hr : ((r : ℝ) + 1) * γ ≤ 1 / 2) (N x : ℕ) :
    ∑ c ∈ Finset.range N, ratioPow γ x c ^ r * profileEnv γ τ x c ≤ 1 + 2 * τ := by
  have hsplit : ∑ c ∈ Finset.range N, ratioPow γ x c ^ r * profileEnv γ τ x c =
      (∑ c ∈ Finset.range N, if c = x then ratioPow γ x c ^ r else 0) +
        τ * ∑ c ∈ Finset.range N, ratioPow γ x c ^ r * chainEnv γ x c := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun c _ => ?_
    unfold profileEnv
    split_ifs <;> ring
  rw [hsplit, Finset.sum_ite_eq']
  have h1 : (if x ∈ Finset.range N then ratioPow γ x x ^ r else 0) ≤ 1 := by
    split_ifs <;> simp [ratioPow_self]
  nlinarith [sum_chainEnv_mul_le hγ hr N x]

/-- **Sums against the decay of a pair.** If `h(M) ≤ [M = x]/(x+1) + [M < x] c/((x+1)(M+1))`
for `M > n`, then `Σ_{M>n} h(M) ≤ (1 + c q_γ(x, n)/γ)/(x+1)`. -/
theorem sum_decay_le {γ : ℝ} (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) {n x : ℕ} {c : ℝ} (hc : 0 ≤ c)
    {h : ℕ → ℝ} (hh : ∀ M, n < M → h M ≤ (if M = x then 1 / ((x : ℝ) + 1) else 0) +
      (if M < x then c / (((x : ℝ) + 1) * ((M : ℝ) + 1)) else 0)) (N : ℕ) :
    ∑ M ∈ Finset.range N, (if n < M then h M else 0) ≤
      (1 + c * ratioPow γ x n / γ) / ((x : ℝ) + 1) := by
  have hx : (0 : ℝ) < (x : ℝ) + 1 := by positivity
  have hpt : ∀ M ∈ Finset.range N, (if n < M then h M else 0) ≤
      (if M = x then 1 / ((x : ℝ) + 1) else 0) +
        (if M ∈ Finset.Ico (n + 1) x then c / ((x : ℝ) + 1) * (1 / ((M : ℝ) + 1)) else 0) := by
    intro M _
    by_cases h1 : n < M
    · rw [ite_eq_left h1]
      refine (hh M h1).trans (le_of_eq ?_)
      congr 1
      by_cases h2 : M < x
      · rw [ite_eq_left h2, ite_eq_left (Finset.mem_Ico.2 ⟨h1, h2⟩), div_mul_div_comm, mul_one]
      · rw [ite_eq_right h2, ite_eq_right fun h => h2 (Finset.mem_Ico.1 h).2]
    · rw [ite_eq_right h1]
      exact add_nonneg (by split_ifs <;> positivity) (by split_ifs <;> positivity)
  refine (Finset.sum_le_sum hpt).trans ?_
  rw [Finset.sum_add_distrib, Finset.sum_ite_eq']
  have h1 : (if x ∈ Finset.range N then 1 / ((x : ℝ) + 1) else 0) ≤ 1 / ((x : ℝ) + 1) := by
    split_ifs
    · exact le_rfl
    · positivity
  have h2 := (sum_range_ite_le (fun M => by positivity) N (n + 1) x).trans_eq
    (Finset.mul_sum (Finset.Ico (n + 1) x) (fun M => 1 / ((M : ℝ) + 1)) (c / ((x : ℝ) + 1))).symm
  have h3 := mul_le_mul_of_nonneg_left (sum_Ico_inv_le hγ0 hγ1 n x)
    (by positivity : 0 ≤ c / ((x : ℝ) + 1))
  have h4 : (1 + c * ratioPow γ x n / γ) / ((x : ℝ) + 1) =
      1 / ((x : ℝ) + 1) + c / ((x : ℝ) + 1) * (ratioPow γ x n / γ) := by ring
  linarith

/-- The hypotheses of `sum_profileEnv_mul_le` are satisfiable: `γ = 1/16`, `r = 6`. -/
example : (0 : ℝ) ≤ 1 / 16 ∧ (((6 : ℕ) : ℝ) + 1) * (1 / 16) ≤ 1 / 2 := by norm_num

end Kinetic
end Transformer

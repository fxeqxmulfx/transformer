/-
# The chain sums

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 1. A perturbation of
token `b` reaches token `a > b` through chains `b < j_r < … < j_1 < a`, each
link weighing `ω ≲ 1/(j+1)`. The ratio `q_γ(a, b) = ((a+1)/(b+1))^γ` measures
them: `γ Σ_{b<c<a} (c+1)^{γ-1} ≤ a^γ - (b+1)^γ` (`sum_rpow_le`), so
`Σ_{b<c<a} q_γ(c, b)/(c+1) ≤ (q_γ(a, b) - 1)/γ` (`sum_ratioPow_div_le`).
-/

import Transformer.Basic
import Mathlib.Analysis.Convex.SpecificFunctions.Basic

open Real

namespace Transformer
namespace Kinetic

/-- `q_γ(a, b) = ((a+1)/(b+1))^γ`. -/
noncomputable def ratioPow (γ : ℝ) (a b : ℕ) : ℝ := (((a : ℝ) + 1) / ((b : ℝ) + 1)) ^ γ

theorem ratioPow_pos (γ : ℝ) (a b : ℕ) : 0 < ratioPow γ a b := by
  unfold ratioPow
  positivity

theorem one_le_ratioPow {γ : ℝ} (hγ : 0 ≤ γ) {a b : ℕ} (hba : b ≤ a) : 1 ≤ ratioPow γ a b := by
  unfold ratioPow
  refine Real.one_le_rpow ?_ hγ
  rw [one_le_div (by positivity)]
  exact_mod_cast Nat.add_le_add_right hba 1

theorem ratioPow_self (γ : ℝ) (a : ℕ) : ratioPow γ a a = 1 := by
  simp [ratioPow, div_self (by positivity : ((a : ℝ) + 1) ≠ 0)]

theorem ratioPow_mul (γ : ℝ) (a b c : ℕ) : ratioPow γ a b * ratioPow γ b c = ratioPow γ a c := by
  unfold ratioPow
  rw [← Real.mul_rpow (by positivity) (by positivity)]
  congr 1
  field_simp

/-- `γ n^{γ-1} ≤ n^γ - (n-1)^γ` for `γ ∈ [0, 1]`, `n ≥ 1`: Bernoulli's
inequality for the concave power. -/
theorem mul_rpow_sub_one_le {γ : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1) {n : ℝ} (hn : 1 ≤ n) :
    γ * n ^ (γ - 1) ≤ n ^ γ - (n - 1) ^ γ := by
  have hn0 : 0 < n := by linarith
  have hb := rpow_one_add_le_one_add_mul_self (s := -1 / n)
    (by rw [neg_div, neg_le_neg_iff, div_le_one hn0]; exact hn) hγ0 hγ1
  have hsplit : (n - 1) ^ γ = (1 + -1 / n) ^ γ * n ^ γ := by
    rw [← Real.mul_rpow (by rw [neg_div, ← sub_eq_add_neg, sub_nonneg, div_le_one hn0]; exact hn)
      hn0.le]
    congr 1
    field_simp
    ring
  rw [hsplit, Real.rpow_sub_one hn0.ne']
  have hpow := Real.rpow_pos_of_pos hn0 γ
  have : (1 + -1 / n) ^ γ * n ^ γ ≤ (1 + γ * (-1 / n)) * n ^ γ :=
    mul_le_mul_of_nonneg_right hb hpow.le
  have heq : (1 + γ * (-1 / n)) * n ^ γ = n ^ γ - γ * (n ^ γ / n) := by
    field_simp
    ring
  linarith

/-- **The chain sum.** `γ Σ_{b<c<a} (c+1)^{γ-1} ≤ a^γ - (b+1)^γ` for
`γ ∈ [0, 1]` and `b < a`. -/
theorem sum_rpow_le {γ : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1) (b : ℕ) :
    ∀ a, b + 1 ≤ a → γ * ∑ c ∈ Finset.Ico (b + 1) a, ((c : ℝ) + 1) ^ (γ - 1) ≤
      (a : ℝ) ^ γ - ((b : ℝ) + 1) ^ γ := by
  intro a ha
  induction a, ha using Nat.le_induction with
  | base => simp
  | succ a ha ih =>
    rw [Finset.sum_Ico_succ_top ha, mul_add]
    have h := mul_rpow_sub_one_le hγ0 hγ1 (n := (a : ℝ) + 1)
      (by linarith [(Nat.cast_nonneg a : (0 : ℝ) ≤ a)])
    push_cast
    rw [add_sub_cancel_right] at h
    linarith

/-- `Σ_{b<c<a} q_γ(c, b) / (c+1) ≤ (q_γ(a, b) - 1) / γ` for `γ ∈ (0, 1]`. -/
theorem sum_ratioPow_div_le {γ : ℝ} (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) {a b : ℕ} (hba : b < a) :
    ∑ c ∈ Finset.Ico (b + 1) a, ratioPow γ c b / ((c : ℝ) + 1) ≤ (ratioPow γ a b - 1) / γ := by
  have hb : (0 : ℝ) < (b : ℝ) + 1 := by positivity
  have hterm : ∀ c : ℕ, ratioPow γ c b / ((c : ℝ) + 1) =
      ((c : ℝ) + 1) ^ (γ - 1) / ((b : ℝ) + 1) ^ γ := by
    intro c
    rw [ratioPow, Real.div_rpow (by positivity) hb.le, Real.rpow_sub_one (by positivity)]
    ring
  simp_rw [hterm, ← Finset.sum_div]
  have hs := sum_rpow_le hγ0.le hγ1 b a hba
  have hpow := Real.rpow_pos_of_pos hb γ
  have ha : (a : ℝ) ^ γ ≤ ((a : ℝ) + 1) ^ γ :=
    Real.rpow_le_rpow (Nat.cast_nonneg a) (by linarith) hγ0.le
  rw [div_le_div_iff₀ hpow hγ0, ratioPow, Real.div_rpow (by positivity) hb.le, sub_mul,
    div_mul_cancel₀ _ hpow.ne', one_mul]
  nlinarith

/-- The hypotheses of `sum_ratioPow_div_le` are satisfiable. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℕ) < 1 := by norm_num

end Kinetic
end Transformer

import Transformer.AdamBeyond.Section3_GenRun

/-
# Adam and beyond — §3: inside one block of Theorem 2

A block starts after a step `t` with `t mod C = 0` past the warm-up: the gradient
is `C` at step `t + 1` and `-1` at the steps `t + 2, …, t + C`.  At its `j`-th step
`m_{t+j} ≤ β₁^{j-1}(C+1) - 1`, `(1-β₂)β₂^{j-1}C² ≤ v_{t+j} ≤ β₂^{j-1}C² + 1`
(`gen_inblock`), and `m` decreases along the block (`genM_block_anti`).  So the
steps `δ_j = -α_{t+j} m_{t+j}/√v_{t+j}` are first non-positive, then positive,
and each is at least `α_{t+j}/√v_{t+j} - 2α_{t+j}γ^{j-1}/√(1-β₂)`, `γ = β₁/√β₂`
(`genD_ge`).

Source: arXiv:1904.09237, Appendix, proof of Theorem 2.
-/

namespace Transformer
namespace AdamBeyond

variable {C : ℕ} {β₁ β₂ α : ℝ}

/-- The step `δ_n = -α_n m_n/√v_n`. -/
noncomputable def genD (C : ℕ) (β₁ β₂ α : ℝ) (n : ℕ) : ℝ :=
  -(α / Real.sqrt (n : ℕ) * (genM C β₁ β₂ α n / Real.sqrt (genV C β₁ β₂ α n)))

theorem genX_succ' (n : ℕ) :
    genX C β₁ β₂ α (n + 1) = max (-1) (min 1 (genX C β₁ β₂ α n + genD C β₁ β₂ α (n + 1))) := by
  rw [genX_succ, genD, sub_eq_add_neg]

/-- The slopes of a block. -/
theorem genSlope_block {k : ℕ} (hC : 2 ≤ C) :
    genSlope C (C + k * C + 1) = C ∧
      ∀ j, 2 ≤ j → j ≤ C → genSlope C (C + k * C + j) = -1 := by
  refine ⟨?_, fun j hj hj' => ?_⟩
  · have : (C + k * C + 1) % C = 1 := by
      rw [show C + k * C + 1 = 1 + (k + 1) * C by ring, Nat.add_mul_mod_self_right]
      exact Nat.mod_eq_of_lt (by omega)
    simp [genSlope, this]
  · have : (C + k * C + j) % C ≠ 1 := by
      rw [show C + k * C + j = j + (k + 1) * C by ring, Nat.add_mul_mod_self_right]
      rcases hj'.lt_or_eq with h | h
      · rw [Nat.mod_eq_of_lt h]; omega
      · subst h; simp
    have h2 : ¬ C + k * C + j ≤ C := by omega
    simp [genSlope, this, h2]

variable (hC : 1 ≤ C) (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ ≤ 1) (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ ≤ 1)
  {t : ℕ} (h₁ : genSlope C (t + 1) = C) (h₂ : ∀ j, 2 ≤ j → j ≤ C → genSlope C (t + j) = -1)
include hC hβ₁ hβ₁' hβ₂ hβ₂' h₁ h₂

/-- The bounds at the `j`-th step of a block. -/
theorem gen_inblock {j : ℕ} (hj : 1 ≤ j) (hjC : j ≤ C) :
    genM C β₁ β₂ α (t + j) ≤ β₁ ^ (j - 1) * (C + 1) - 1 ∧
      β₂ ^ (j - 1) * (1 - β₂) * (C : ℝ) ^ 2 ≤ genV C β₁ β₂ α (t + j) ∧
      genV C β₁ β₂ α (t + j) ≤ β₂ ^ (j - 1) * (C : ℝ) ^ 2 + 1 := by
  induction j, hj using Nat.le_induction with
  | base =>
    obtain ⟨-, m2, v1, v2⟩ := gen_bounds hC hβ₁ hβ₁' hβ₂ hβ₂' α t
    rw [genM_succ, genV_succ, h₁]
    simp only [Nat.sub_self, pow_zero, one_mul]
    refine ⟨by nlinarith, by nlinarith, by nlinarith⟩
  | succ j hj ih =>
    obtain ⟨m1, v1, v2⟩ := ih (by omega)
    obtain ⟨i, rfl⟩ : ∃ i, j = i + 1 := ⟨j - 1, by omega⟩
    rw [← add_assoc, genM_succ, genV_succ, add_assoc, h₂ _ (by omega) hjC]
    simp only [Nat.add_sub_cancel] at m1 v1 v2 ⊢
    have hp : 0 ≤ β₁ ^ i := pow_nonneg hβ₁ i
    have hq : 0 ≤ β₂ ^ i := pow_nonneg hβ₂ i
    refine ⟨?_, ?_, ?_⟩
    · rw [pow_succ β₁ i]; nlinarith
    · rw [pow_succ β₂ i]; nlinarith [mul_le_mul_of_nonneg_left v1 hβ₂]
    · rw [pow_succ β₂ i]; nlinarith [mul_le_mul_of_nonneg_left v2 hβ₂]

omit h₁ in
/-- `m` decreases along a block. -/
theorem genM_block_anti {j k : ℕ} (hj : 1 ≤ j) (hjk : j ≤ k) (hkC : k ≤ C) :
    genM C β₁ β₂ α (t + k) ≤ genM C β₁ β₂ α (t + j) := by
  induction k, hjk using Nat.le_induction with
  | base => exact le_rfl
  | succ k hk ih =>
    have m1 := (gen_bounds hC hβ₁ hβ₁' hβ₂ hβ₂' α (t + k)).1
    rw [← add_assoc, genM_succ, add_assoc, h₂ _ (by omega) hkC]
    nlinarith [ih (by omega)]

/-- The lower bound on the `j`-th step of a block:
`δ_{t+j} ≥ α_{t+j}/√v_{t+j} - 2α_{t+j}γ^{j-1}/√(1-β₂)`, `γ = β₁/√β₂`. -/
theorem genD_ge (hβ₂0 : 0 < β₂) (hβ₂1 : β₂ < 1) (hα : 0 ≤ α) {j : ℕ} (hj : 1 ≤ j) (hjC : j ≤ C) :
    α / Real.sqrt (t + j : ℕ) / Real.sqrt (genV C β₁ β₂ α (t + j)) -
      2 * (α / Real.sqrt (t + j : ℕ)) * (β₁ / Real.sqrt β₂) ^ (j - 1) / Real.sqrt (1 - β₂) ≤
      genD C β₁ β₂ α (t + j) := by
  obtain ⟨m1, v1, -⟩ := gen_inblock hC hβ₁ hβ₁' hβ₂ hβ₂' h₁ h₂ (α := α) hj hjC
  have hC1 : (1 : ℝ) ≤ C := by exact_mod_cast hC
  set a := α / Real.sqrt (t + j : ℕ)
  set M := genM C β₁ β₂ α (t + j)
  set V := genV C β₁ β₂ α (t + j)
  set b := β₁ ^ (j - 1)
  set r := Real.sqrt β₂ ^ (j - 1)
  set q := Real.sqrt (1 - β₂)
  have ha : 0 ≤ a := by positivity
  have hb : 0 ≤ b := by positivity
  have hr : 0 < r := pow_pos (Real.sqrt_pos.2 hβ₂0) _
  have hq : 0 < q := Real.sqrt_pos.2 (by linarith)
  have hs : r * q * C ≤ Real.sqrt V := by
    refine Real.le_sqrt_of_sq_le ?_
    rw [mul_pow, mul_pow, ← pow_mul, mul_comm (j - 1) 2, pow_mul, Real.sq_sqrt hβ₂,
      Real.sq_sqrt (by linarith)]
    exact v1
  have hs0 : 0 < Real.sqrt V := lt_of_lt_of_le (by positivity) hs
  have k1 : a / Real.sqrt V - a * b * (C + 1) / Real.sqrt V ≤ genD C β₁ β₂ α (t + j) := by
    have : a * (1 - b * (C + 1)) / Real.sqrt V ≤ a * -M / Real.sqrt V :=
      div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left (by linarith) ha) hs0.le
    unfold genD
    calc _ = a * (1 - b * (C + 1)) / Real.sqrt V := by ring
      _ ≤ a * -M / Real.sqrt V := this
      _ = _ := by ring
  have k2 : a * b * (C + 1) / Real.sqrt V ≤ 2 * a * (β₁ / Real.sqrt β₂) ^ (j - 1) / q := by
    rw [div_pow, show 2 * a * (β₁ ^ (j - 1) / r) / q = 2 * a * b / (r * q) by
      field_simp; rfl, div_le_div_iff₀ hs0 (by positivity)]
    have : (C + 1) * (r * q) ≤ 2 * Real.sqrt V := by nlinarith [mul_pos hr hq]
    have hab : 0 ≤ a * b := mul_nonneg ha hb
    nlinarith
  linarith

/-- The hypotheses are satisfiable: the first block of `C = 2`, `β₁ = 0`, `β₂ = 1/2`. -/
example : (1 : ℕ) ≤ 2 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    genSlope 2 (2 + 0 * 2 + 1) = 2 ∧ ∀ j, 2 ≤ j → j ≤ 2 → genSlope 2 (2 + 0 * 2 + j) = -1 := by
  obtain ⟨h₁, h₂⟩ := genSlope_block (C := 2) (k := 0) le_rfl
  exact ⟨by norm_num, le_rfl, by norm_num, by norm_num, by norm_num, by exact_mod_cast h₁, h₂⟩

end AdamBeyond
end Transformer

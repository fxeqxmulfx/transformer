import Transformer.AdamBeyond.Section3_GenStep

/-
# Adam and beyond — §3: the regret of Theorem 2, and its constants

Every block starts at `x = 1` (`gen_start`, from `gen_block`), so against
`x* = -1` a block costs `2C` at its first step and gains back at most `2` at each
of the `C - 1` others: at least `2` per block (`gen_block_regret`), and
`R_T ≥ T/C` once `T ≥ 4C` (`gen_regret`).  The constants exist for every
`0 ≤ β₁ < √β₂`, `β₂ < 1` (`gen_constants`).

Source: arXiv:1904.09237, Appendix, proof of Theorem 2.
-/

open Finset Filter

namespace Transformer
namespace AdamBeyond

open AMSGrad

variable {C : ℕ} {β₁ β₂ α : ℝ}
  (hC : 2 ≤ C) (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ ≤ 1) (hβ₂0 : 0 < β₂) (hβ₂1 : β₂ < 1) (hα : 0 < α)
  {N : ℕ} (hCN : C = 2 * N) (hN : β₂ ^ N * (C : ℝ) ^ 2 ≤ 1)
  (hγ1 : β₁ / Real.sqrt β₂ < 1) (hγ : 4 ≤ N * Real.sqrt (1 - β₂) * (1 - β₁ / Real.sqrt β₂))
include hC hβ₁ hβ₁' hβ₂0 hβ₂1 hα hCN hN hγ1 hγ

/-- Every block starts at `x = 1`: `x_{C + kC + 1} = 1`. -/
theorem gen_start (k : ℕ) : genX C β₁ β₂ α (C + k * C) = 1 := by
  induction k with
  | zero => simpa using (gen_warmup C β₁ β₂ α le_rfl).2.2
  | succ k ih =>
    obtain ⟨h₁, h₂⟩ := genSlope_block (k := k) hC
    rw [show C + (k + 1) * C = C + k * C + C by ring]
    exact gen_block (by omega) hβ₁ hβ₁' hβ₂0 hβ₂1 hα h₁ h₂ (by nlinarith) hCN hN hγ1 hγ ih

/-- The first `i` steps of a block, `1 ≤ i ≤ C`, cost at least `2C - 2(i - 1)`
against `x* = -1`. -/
theorem gen_block_regret (k : ℕ) {i : ℕ} (hi : 1 ≤ i) (hiC : i ≤ C) :
    2 * C - 2 * (i - 1 : ℕ) ≤ ∑ n ∈ Ioc (C + k * C) (C + k * C + i),
      genSlope C n * (genX C β₁ β₂ α (n - 1) + 1) := by
  obtain ⟨h₁, h₂⟩ := genSlope_block (k := k) hC
  induction i, hi using Nat.le_induction with
  | base =>
    rw [Nat.Ioc_succ_singleton, sum_singleton, h₁, Nat.add_sub_cancel,
      gen_start hC hβ₁ hβ₁' hβ₂0 hβ₂1 hα hCN hN hγ1 hγ]
    simp; ring_nf; rfl
  | succ i hi ih =>
    have h2' := h₂ (i + 1) (by omega) hiC
    rw [← add_assoc] at h2'
    rw [← add_assoc, sum_Ioc_succ_top (by omega), h2']
    simp only [Nat.add_sub_cancel]
    have := genX_le_one C β₁ β₂ α (C + k * C + i)
    have hc : ((i - 1 : ℕ) : ℝ) = i - 1 := by rw [Nat.cast_sub hi]; simp
    rw [hc] at ih
    linarith [ih (by omega)]

/-- `R_T ≥ 2k` after the warm-up and `k` blocks, whatever part of the next block
follows. -/
theorem gen_regret_blocks (k : ℕ) {i : ℕ} (hiC : i ≤ C) :
    2 * (k : ℝ) ≤ (genSetup C β₁ β₂ α).regret adamRule (fun _ => -1) (C + k * C + i) := by
  have e : ∀ T, (genSetup C β₁ β₂ α).regret adamRule (fun _ => -1) T =
      ∑ n ∈ Ioc 0 T, genSlope C n * (genX C β₁ β₂ α (n - 1) + 1) := by
    intro T
    rw [Setup.regret, show Icc 1 T = Ioc 0 T by ext; simp; omega]
    refine sum_congr rfl fun n _ => ?_
    simp only [genSetup, Setup.x, genX]
    ring
  have hw : ∑ n ∈ Ioc 0 C, genSlope C n * (genX C β₁ β₂ α (n - 1) + 1) = 0 :=
    sum_eq_zero fun n hn => by rw [mem_Ioc] at hn; simp [genSlope, hn.2]
  have hk : ∀ k : ℕ, 2 * (k : ℝ) ≤ ∑ n ∈ Ioc 0 (C + k * C),
      genSlope C n * (genX C β₁ β₂ α (n - 1) + 1) := by
    intro k
    induction k with
    | zero => simp [hw]
    | succ k ih =>
      rw [show C + (k + 1) * C = C + k * C + C by ring,
        ← sum_Ioc_consecutive _ (Nat.zero_le _) (Nat.le_add_right _ C)]
      have := gen_block_regret hC hβ₁ hβ₁' hβ₂0 hβ₂1 hα hCN hN hγ1 hγ k (by omega) le_rfl
      have hC' : ((C - 1 : ℕ) : ℝ) = C - 1 := by rw [Nat.cast_sub (by omega)]; simp
      rw [hC'] at this
      push_cast
      linarith
  rw [e, ← sum_Ioc_consecutive _ (Nat.zero_le _) (Nat.le_add_right _ i)]
  rcases Nat.eq_zero_or_pos i with h | h
  · subst h; simpa using hk k
  · have := gen_block_regret hC hβ₁ hβ₁' hβ₂0 hβ₂1 hα hCN hN hγ1 hγ k h hiC
    have hi : ((i - 1 : ℕ) : ℝ) ≤ C := by exact_mod_cast (by omega : i - 1 ≤ C)
    linarith [hk k]

/-- `R_T ≥ T/C` for `T ≥ 4C`: the average regret does not vanish. -/
theorem gen_regret {T : ℕ} (hT : 4 * C ≤ T) :
    1 / (C : ℝ) * T ≤ (genSetup C β₁ β₂ α).regret adamRule (fun _ => -1) T := by
  obtain ⟨k, i, hk, hi, hTk⟩ : ∃ k i, 3 ≤ k ∧ i < C ∧ T = C + k * C + i :=
    ⟨(T - C) / C, (T - C) % C, (Nat.le_div_iff_mul_le (by omega)).2 (by omega),
      Nat.mod_lt _ (by omega), by have := Nat.div_add_mod' (T - C) C; omega⟩
  have h := gen_regret_blocks hC hβ₁ hβ₁' hβ₂0 hβ₂1 hα hCN hN hγ1 hγ k hi.le
  rw [← hTk] at h
  have hCpos : (0 : ℝ) < C := by exact_mod_cast (by omega : 0 < C)
  have hTle : (T : ℝ) ≤ 2 * k * C := by
    have := Nat.mul_le_mul_right C hk
    have : T ≤ 2 * (k * C) := by omega
    rw [← mul_assoc] at this
    exact_mod_cast this
  rw [one_div_mul_eq_div, div_le_iff₀ hCpos]
  nlinarith

omit hC hβ₁ hβ₁' hβ₂0 hβ₂1 hα hCN hN hγ1 hγ in
/-- The constants of the proof of Theorem 2: `N ≥ 1` with `β₂^N (2N)² ≤ 1` and
`N √(1-β₂)(1-γ) ≥ 4`, `γ = β₁/√β₂`.  arXiv:1904.09237, Appendix, proof of
Theorem 2, eq:p-condition. -/
theorem gen_constants {β₁ β₂ : ℝ} (hβ₁ : 0 ≤ β₁) (hβ₂ : β₂ < 1) (hγ : β₁ < Real.sqrt β₂) :
    ∃ N : ℕ, 1 ≤ N ∧ β₂ ^ N * ((2 * N : ℕ) : ℝ) ^ 2 ≤ 1 ∧
      4 ≤ N * Real.sqrt (1 - β₂) * (1 - β₁ / Real.sqrt β₂) := by
  have hs : 0 < Real.sqrt β₂ := lt_of_le_of_lt hβ₁ hγ
  have hβ₂0 : 0 < β₂ := Real.sqrt_pos.1 hs
  have hq : 0 < Real.sqrt (1 - β₂) * (1 - β₁ / Real.sqrt β₂) :=
    mul_pos (Real.sqrt_pos.2 (by linarith)) (by rw [sub_pos, div_lt_one hs]; exact hγ)
  have t1 := tendsto_pow_const_mul_const_pow_of_abs_lt_one 2
    (show |β₂| < 1 by rw [abs_of_pos hβ₂0]; exact hβ₂)
  have ev1 : ∀ᶠ N : ℕ in atTop, (N : ℝ) ^ 2 * β₂ ^ N ≤ 1 / 4 :=
    t1.eventually (Iic_mem_nhds (by norm_num))
  have ev2 : ∀ᶠ N : ℕ in atTop, 4 / (Real.sqrt (1 - β₂) * (1 - β₁ / Real.sqrt β₂)) ≤ N :=
    tendsto_natCast_atTop_atTop.eventually_ge_atTop _
  obtain ⟨N, h1, h2, h3⟩ := (ev1.and (ev2.and (eventually_ge_atTop 1))).exists
  refine ⟨N, h3, ?_, ?_⟩
  · push_cast; nlinarith
  · rw [div_le_iff₀ hq] at h2; linarith

/-- The hypotheses are satisfiable: `β₁ = 0`, `β₂ = 1/2`, and the constants of
`gen_constants`. -/
example : ∃ N : ℕ, 1 ≤ N ∧ (1 / 2 : ℝ) ^ N * ((2 * N : ℕ) : ℝ) ^ 2 ≤ 1 ∧
    4 ≤ N * Real.sqrt (1 - 1 / 2) * (1 - 0 / Real.sqrt (1 / 2)) :=
  gen_constants le_rfl (by norm_num) (Real.sqrt_pos.2 (by norm_num))

end AdamBeyond
end Transformer

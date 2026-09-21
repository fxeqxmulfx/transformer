import Transformer.AMSGrad.Section4_CounterRun
import Mathlib.Algebra.Field.GeomSum

/-
# AMSGrad — Corollary 4.5 is false as stated

§4 of arXiv:1904.03590v4, Corollary 4.5: on `signSetup`, summing the block
bounds of `sign_term` up to `T = 2^{2j+1} - 1`,
`Σ_{t≤T} s_t x_t ≤ -T + 2 Σ_{k≤2j} 2^{⌊k/2⌋+2} ≤ -T + 24·2^j`, while
`Σ_{t≤T} s_t = (2^{2j+1} + 1)/3`; so `R(T) ≤ -T/2` once `j ≥ 8`.

Source: arXiv:1904.03590v4, §4, Corollary 4.5.
-/

open Finset Filter

namespace Transformer
namespace AMSGrad

/-- `[1, 2^{K+1})` is the union of the blocks `[2^k, 2^{k+1})`, `k ≤ K`.
arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem sum_blocks (f : ℕ → ℝ) (K : ℕ) :
    ∑ t ∈ Ico 1 (2 ^ (K + 1)), f t =
      ∑ k ∈ range (K + 1), ∑ t ∈ Ico (2 ^ k) (2 ^ (k + 1)), f t := by
  induction K with
  | zero => simp
  | succ K ih =>
    rw [sum_range_succ, ← ih, sum_Ico_consecutive _ Nat.one_le_two_pow
      (Nat.pow_le_pow_right two_pos (by omega))]

/-- `Σ_{t ∈ [2^k, 2^{k+1})} s_t x_t ≤ -2^k + 2·2^{⌊k/2⌋+2}`.
arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem sign_block_sum (k : ℕ) :
    ∑ t ∈ Ico (2 ^ k) (2 ^ (k + 1)), blockSign t * signSetup.x amsgradRule t 0 ≤
      -(2 : ℝ) ^ k + 2 * 2 ^ (k / 2 + 2) := by
  refine (sum_le_sum fun t ht => sign_term (mem_Ico.1 ht).1 (mem_Ico.1 ht).2).trans ?_
  rw [sum_add_distrib, ← mul_sum, sum_const, sum_boole, Nat.card_Ico]
  have hc : ((Ico (2 ^ k) (2 ^ (k + 1))).filter (fun t => t < 2 ^ k + 2 ^ (k / 2 + 2))).card ≤
      2 ^ (k / 2 + 2) := by
    refine (card_le_card (t := Ico (2 ^ k) (2 ^ k + 2 ^ (k / 2 + 2))) fun t ht => ?_).trans ?_
    · simp only [mem_filter, mem_Ico] at ht ⊢
      omega
    · rw [Nat.card_Ico, Nat.add_sub_cancel_left]
  have hc' : (((Ico (2 ^ k) (2 ^ (k + 1))).filter
      (fun t => t < 2 ^ k + 2 ^ (k / 2 + 2))).card : ℝ) ≤ 2 ^ (k / 2 + 2) := by
    exact_mod_cast hc
  rw [show 2 ^ (k + 1) - 2 ^ k = 2 ^ k by rw [pow_succ]; omega]
  simp only [nsmul_eq_mul, mul_neg, mul_one]
  push_cast
  linarith

/-- `Σ_{t ∈ [2^k, 2^{k+1})} s_t = (-2)^k`.
arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem sign_block_sign (k : ℕ) :
    ∑ t ∈ Ico (2 ^ k) (2 ^ (k + 1)), blockSign t = (-2 : ℝ) ^ k := by
  rw [sum_congr rfl fun t ht => blockSign_block (mem_Ico.1 ht).1 (mem_Ico.1 ht).2, sum_const,
    Nat.card_Ico, show 2 ^ (k + 1) - 2 ^ k = 2 ^ k by rw [pow_succ]; omega, nsmul_eq_mul]
  push_cast
  rw [← mul_pow]; norm_num

/-- `Σ_{k ≤ 2j} 2^{⌊k/2⌋} ≤ 3·2^j`.
arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem sum_half_pow_le (j : ℕ) : ∑ k ∈ range (2 * j + 1), 2 ^ (k / 2) ≤ 3 * 2 ^ j := by
  induction j with
  | zero => simp
  | succ j ih =>
    rw [show 2 * (j + 1) + 1 = 2 * j + 1 + 1 + 1 by ring, sum_range_succ, sum_range_succ,
      show (2 * j + 1) / 2 = j by omega, show (2 * j + 1 + 1) / 2 = j + 1 by omega, pow_succ]
    omega

/-- **Corollary 4.5 is false as stated.**  On `signSetup`, which satisfies its
hypotheses in both settings (`isOnlineConvex_sign`, `sign_hyp`), `R(T) ≤ -T/2`
for infinitely many `T` and every `x* ∈ F`, the minimizer of `Σ_{t≤T} f_t`
included; so `R(T)/T` does not tend to `0`.  Only the upper half, `cor_lambda`
and `cor_inv`, is kept.

Source: arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem not_cor_lower :
    ∃ᶠ T : ℕ in atTop, ∀ xstar ∈ Set.Icc (fun _ : Fin 1 => (-1 : ℝ)) (fun _ => 1),
      signSetup.regret amsgradRule xstar T ≤ -((T : ℝ) / 2) := by
  rw [frequently_atTop]
  intro a
  have hP1 : 1 ≤ 2 ^ (2 * (a + 8) + 1) := Nat.one_le_two_pow
  refine ⟨2 ^ (2 * (a + 8) + 1) - 1, ?_, fun xstar hx => ?_⟩
  · have := Nat.lt_two_pow_self (n := 2 * (a + 8) + 1); omega
  obtain ⟨u, hu⟩ : ∃ u : ℝ, u = 2 ^ (a + 8) := ⟨_, rfl⟩
  have hu8 : 256 ≤ u := by
    rw [hu, show (256 : ℝ) = 2 ^ 8 by norm_num]
    exact pow_le_pow_right₀ one_le_two (by omega)
  have hP : (2 : ℝ) ^ (2 * (a + 8) + 1) = 2 * u ^ 2 := by
    rw [hu, ← pow_mul, pow_succ, mul_comm (a + 8) 2]; ring
  have hT : ((2 ^ (2 * (a + 8) + 1) - 1 : ℕ) : ℝ) = 2 * u ^ 2 - 1 := by
    rw [Nat.cast_sub hP1]; push_cast; rw [hP]
  have e : signSetup.regret amsgradRule xstar (2 ^ (2 * (a + 8) + 1) - 1) =
      ∑ t ∈ Ico 1 (2 ^ (2 * (a + 8) + 1)), blockSign t * signSetup.x amsgradRule t 0 -
        xstar 0 * ∑ t ∈ Ico 1 (2 ^ (2 * (a + 8) + 1)), blockSign t := by
    simp only [Setup.regret]
    rw [← Finset.Ico_add_one_right_eq_Icc,
      show 2 ^ (2 * (a + 8) + 1) - 1 + 1 = 2 ^ (2 * (a + 8) + 1) by omega]
    change ∑ t ∈ _, (blockSign t * signSetup.x amsgradRule t 0 - blockSign t * xstar 0) = _
    rw [sum_sub_distrib, ← sum_mul]
    ring
  have h1 : ∑ t ∈ Ico 1 (2 ^ (2 * (a + 8) + 1)), blockSign t * signSetup.x amsgradRule t 0 ≤
      -(2 * u ^ 2 - 1) + 24 * u := by
    have hs : ∑ k ∈ range (2 * (a + 8) + 1), (2 : ℝ) ^ (k / 2) ≤ 3 * u := by
      rw [hu]; exact_mod_cast sum_half_pow_le (a + 8)
    have eq : ∑ k ∈ range (2 * (a + 8) + 1), (-(2 : ℝ) ^ k + 2 * 2 ^ (k / 2 + 2)) =
        -(2 * u ^ 2 - 1) + 8 * ∑ k ∈ range (2 * (a + 8) + 1), (2 : ℝ) ^ (k / 2) := by
      rw [sum_add_distrib, sum_neg_distrib, ← mul_sum, geom_sum_eq (by norm_num), hP]
      simp only [pow_add, ← sum_mul]
      ring
    rw [sum_blocks]
    refine (sum_le_sum fun k _ => sign_block_sum k).trans ?_
    rw [eq]
    linarith
  have h2 : ∑ t ∈ Ico 1 (2 ^ (2 * (a + 8) + 1)), blockSign t = (2 * u ^ 2 + 1) / 3 := by
    rw [sum_blocks, sum_congr rfl fun k _ => sign_block_sign k, geom_sum_eq (by norm_num),
      Odd.neg_pow ⟨a + 8, rfl⟩, hP]
    ring
  rw [e, h2, hT]
  have hx0 := mul_nonneg (show 0 ≤ xstar 0 + 1 by linarith [hx.1 0])
    (show 0 ≤ (2 * u ^ 2 + 1) / 3 by positivity)
  have hq := mul_le_mul_of_nonneg_right hu8 (show (0 : ℝ) ≤ u by linarith)
  rw [← pow_two] at hq
  linarith

end AMSGrad
end Transformer

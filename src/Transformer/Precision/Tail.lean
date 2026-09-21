/-
# Unnormalized weights in FP8: the tail is cut

FP8 flash attention (Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao,
"FlashAttention-3", arXiv:2407.08608, §3.3) stores not the softmax weights but
`p_j = exp(s_j - m) ∈ (0, 1]`, `m` the running maximum, rounds them to E4M3
for the product with the values, and divides by `S = Σ_j p_j` at the end.
The dead zone then no longer depends on `n` (`qAttn_eq_zero`): every key more
than `ln 2^{10} ≈ 6.9` below the maximum is stored as `0`, at any length.
What depends on `n` is how much is lost.

If every key but the top one `i` is in the dead zone, the numerator keeps
`v_i` alone while the denominator, summed before rounding, keeps all of them:
the head outputs `v_i / S` (`tail_eq`).  With the keys within `D` of the top,
`S ≥ 1 + (n - 1) e^{-D}`, so the output is shrunk by that factor, which grows
with the context: for `v ≡ 1`, where the exact output is `1`, the computed one
is at most `1 / (1 + (n - 1) e^{-D})`.  At `n = 2^{17}`, `D = 8` that is below
`0.023`.

The denominator is taken exact: in the kernel it is summed in binary32 before
the conversion to FP8.  The statement is this repository's own.
-/

import Transformer.Precision.Float
import Mathlib.Analysis.Complex.ExponentialBounds

open scoped BigOperators

namespace Transformer
namespace Precision

variable {n : ℕ} {E' : Type*} [AddCommGroup E'] [Module ℝ E']

/-- **The tail is cut.**  If `Q` sends `[0, θ)` to `0` and keeps `1`, and every
key but `i` is below the top by an amount that puts `exp(s_j - s_i)` under `θ`
but by at most `D`, then the head with rounded unnormalized weights outputs
`v_i / S`, and `1 / S ≤ 1 / (1 + (n - 1) e^{-D})`. -/
theorem tail_eq {Q : ℝ → ℝ} {θ D : ℝ} (hQ0 : ∀ x, 0 ≤ x → x < θ → Q x = 0) (hQ1 : Q 1 = 1)
    {s : Idx n → ℝ} (i : Idx n)
    (hs : ∀ j, j ≠ i → s i - D ≤ s j ∧ Real.exp (s j - s i) < θ) (v : Idx n → E') :
    (∑ j, Real.exp (s j - s i))⁻¹ • ∑ j, Q (Real.exp (s j - s i)) • v j =
        (∑ j, Real.exp (s j - s i))⁻¹ • v i ∧
      (∑ j, Real.exp (s j - s i))⁻¹ ≤ 1 / (1 + (n - 1) * Real.exp (-D)) := by
  refine ⟨?_, ?_⟩
  · congr 1
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ i), sub_self, Real.exp_zero, hQ1, one_smul,
      add_eq_left]
    refine Finset.sum_eq_zero fun j hj => ?_
    rw [hQ0 _ (Real.exp_pos _).le (hs j (Finset.ne_of_mem_erase hj)).2, zero_smul]
  · have hrest : ((n : ℝ) - 1) * Real.exp (-D) ≤
        ∑ j ∈ Finset.univ.erase i, Real.exp (s j - s i) := by
      have := Finset.card_nsmul_le_sum (Finset.univ.erase i) (fun j => Real.exp (s j - s i))
        (Real.exp (-D)) fun j hj => Real.exp_le_exp.2 (by
          linarith [(hs j (Finset.ne_of_mem_erase hj)).1])
      rw [Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul, Nat.cast_sub (Fin.pos i)] at this
      simpa using this
    have hS : 1 + ((n : ℝ) - 1) * Real.exp (-D) ≤ ∑ j, Real.exp (s j - s i) := by
      rw [← Finset.add_sum_erase _ _ (Finset.mem_univ i), sub_self, Real.exp_zero]
      linarith
    have hpos : 0 < 1 + ((n : ℝ) - 1) * Real.exp (-D) := by
      have : (1 : ℝ) ≤ n := by exact_mod_cast Fin.pos i
      nlinarith [Real.exp_pos (-D)]
    rw [← one_div]
    exact one_div_le_one_div_of_le hpos hS

/-- `e^{-8} < 2^{-10}`: eight nats below the top is in the E4M3 dead zone. -/
theorem exp_neg_eight_lt : Real.exp (-8) < 1 / 1024 := by
  rw [Real.exp_neg, ← one_div, one_div_lt_one_div (Real.exp_pos _) (by norm_num),
    show (8 : ℝ) = ((8 : ℕ) : ℝ) by norm_num, ← Real.exp_one_pow]
  have := Real.exp_one_gt_d9
  calc (1024 : ℝ) < 2.7182818283 ^ 8 := by norm_num
    _ ≤ Real.exp 1 ^ 8 := by gcongr

/-- The hypotheses of `tail_eq` are satisfiable: two keys `0, -8` with the
dead zone `[0, 1/1024)` of E4M3. -/
example : (∃ Q, IsNearest (grid 4 3) Q) ∧
    ∀ j : Idx 2, j ≠ 0 → (![0, -8] : Idx 2 → ℝ) 0 - 8 ≤ ![0, -8] j ∧
      Real.exp ((![0, -8] : Idx 2 → ℝ) j - ![0, -8] 0) < 1 / 1024 := by
  refine ⟨exists_isNearest_grid (by norm_num) 3, fun j hj => ?_⟩
  obtain rfl : j = 1 := by revert j; decide
  simpa using exp_neg_eight_lt

/-- **FP8 E4M3 unnormalized weights.**  Rounding `exp(s_j - s_i)` to E4M3
zeroes every key more than `ln 2^{10}` below the top; if that is every key but
the top one, and all of them are within `D`, the head outputs `v_i / S` with
`1 / S ≤ 1 / (1 + (n - 1) e^{-D})`. -/
theorem tail_e4m3 {Q : ℝ → ℝ} (hQ : IsNearest (grid 4 3) Q) {D : ℝ} {s : Idx n → ℝ} (i : Idx n)
    (hs : ∀ j, j ≠ i → s i - D ≤ s j ∧ Real.exp (s j - s i) < 1 / 1024) (v : Idx n → E') :
    (∑ j, Real.exp (s j - s i))⁻¹ • ∑ j, Q (Real.exp (s j - s i)) • v j =
        (∑ j, Real.exp (s j - s i))⁻¹ • v i ∧
      (∑ j, Real.exp (s j - s i))⁻¹ ≤ 1 / (1 + (n - 1) * Real.exp (-D)) :=
  tail_eq (fun _ h0 h => ieee_eq_zero (by norm_num) hQ h0 (by
      rw [minSub_e4m3]; norm_num at h ⊢; linarith))
    (hQ.eq_self ⟨7 * 2 ^ 3, by norm_num, by norm_num [ieee, bias]⟩) i hs v

/-- **The loss at the context of gpt-oss.**  With `2^{17}` keys eight nats
below the top, the output of `tail_e4m3` is shrunk below `1/40`. -/
theorem tail_e4m3_131072 : 1 / (1 + ((2 ^ 17 : ℕ) - 1 : ℝ) * Real.exp (-8)) < 1 / 40 := by
  have hup : Real.exp 8 < 3000 := by
    rw [show (8 : ℝ) = ((8 : ℕ) : ℝ) by norm_num, ← Real.exp_one_pow]
    have := Real.exp_one_lt_d9
    calc Real.exp 1 ^ 8 < 2.7182818286 ^ 8 := by gcongr
      _ < 3000 := by norm_num
  have hpos := Real.exp_pos 8
  have h39 : (39 : ℝ) < ((2 ^ 17 : ℕ) - 1 : ℝ) * Real.exp (-8) := by
    rw [Real.exp_neg, ← div_eq_mul_inv, lt_div_iff₀ hpos]
    push_cast
    nlinarith
  rw [one_div_lt_one_div (by linarith) (by norm_num)]
  linarith

end Precision
end Transformer

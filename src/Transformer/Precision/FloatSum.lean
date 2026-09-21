/-
# The online-softmax sum in floating point

Flash attention keeps the running maximum `m` of the scores and sums
`p_k = exp(s_k - m) ∈ (0, 1]`, weighted by the values (Dao, Fu, Ermon, Rudra,
Ré, "FlashAttention", arXiv:2205.14135, §3.1).  With scores in a window of
width `D`, every `p_k ≥ e^{-D}`, so the exact sum of `n` of them is at least
`n e^{-D}`.  Summed one at a time in a format of `M + 1` significant bits,
it never exceeds `2^{M+2}` (`accum_le`), so the computed sum is at most
`2^{M+2} e^D / n` times the exact one (`accum_softmax_le`).

With `v ≡ 1` that ratio is the computed output itself: past `2^{M+2} e^D`
keys it is below `1`, and it goes to `0` like `1/n`.  Per format:

| format | significant bits | the sum is wrong past |
| --- | ---: | ---: |
| binary32 | 24 | `2^{25} e^D` |
| binary16 | 11 | `2^{12} e^D` |
| bfloat16 | 8 | `2^9 e^D` |
| FP8 E4M3 | 4 | `2^5 e^D` |
| FP8 E5M2 | 3 | `2^4 e^D` |

This is the sequential order; `pairwise_exact` is the way out.
-/

import Transformer.Precision.Float

open scoped BigOperators

namespace Transformer
namespace Precision

variable {M : ℕ} {G : Set ℝ} {Q : ℝ → ℝ}

/-- **The sequential online-softmax sum is capped.**  Increments in
`[e^{-D}, 1]`, summed from `0` with rounding to nearest in `M + 1` significant
bits: after `n` of them, the computed sum is at most `2^{M+2} e^D / n` times the
exact one. -/
theorem accum_softmax_le (hG : HasSignificand (M + 1) G) (hQ : IsNearest G Q) {a t : ℕ → ℝ}
    (ha : ∀ k, a (k + 1) = Q (a k + t k)) (hT : (2 : ℝ) ^ (M + 2) ∈ G) (h0 : a 0 = 0) {D : ℝ}
    (ht : ∀ k, Real.exp (-D) ≤ t k ∧ t k ≤ 1) {n : ℕ} (hn : 0 < n) :
    a n ≤ 2 ^ (M + 2) * Real.exp D / n * ∑ k ∈ Finset.range n, t k := by
  have hle := accum_le hG hQ ha (s := 0) (by simpa using hT) (by rw [h0]; positivity)
    (fun k => by simpa using (ht k).2) n
  rw [zpow_zero, mul_one] at hle
  have hS : (n : ℝ) * Real.exp (-D) ≤ ∑ k ∈ Finset.range n, t k := by
    have := Finset.card_nsmul_le_sum (Finset.range n) t (Real.exp (-D)) (fun k _ => (ht k).1)
    simpa [nsmul_eq_mul] using this
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  calc a n ≤ 2 ^ (M + 2) := hle
    _ = 2 ^ (M + 2) * Real.exp D / n * (n * Real.exp (-D)) := by
        rw [Real.exp_neg]; field_simp
    _ ≤ 2 ^ (M + 2) * Real.exp D / n * ∑ k ∈ Finset.range n, t k := by gcongr

/-- The hypotheses of `accum_softmax_le` are satisfiable: in `{0, 1, 2, 4}`,
increments `1 = e^{-0}`, one term. -/
example : ∃ (Q : ℝ → ℝ) (a t : ℕ → ℝ), HasSignificand (0 + 1) ({0, 1, 2, 4} : Set ℝ) ∧
    IsNearest {0, 1, 2, 4} Q ∧ (∀ k, a (k + 1) = Q (a k + t k)) ∧
    (2 : ℝ) ^ (0 + 2) ∈ ({0, 1, 2, 4} : Set ℝ) ∧ a 0 = 0 ∧
    (∀ k, Real.exp (-0) ≤ t k ∧ t k ≤ 1) ∧ 0 < 1 := by
  obtain ⟨Q, hQ⟩ := exists_isNearest (G := ({0, 1, 2, 4} : Set ℝ)) (Set.toFinite _) ⟨0, by simp⟩
  exact ⟨Q, fun k => Nat.rec 0 (fun _ ak => Q (ak + 1)) k, fun _ => 1, hasSignificand_one_pow_two,
    hQ, fun _ => rfl, by norm_num, rfl, fun _ => by norm_num, one_pos⟩

variable {a t : ℕ → ℝ} {D : ℝ} {n : ℕ}

/-- **binary32**: the sum is wrong past `2^{25} e^D` keys. -/
theorem accum_softmax_f32_le (hQ : IsNearest (grid 8 23) Q) (ha : ∀ k, a (k + 1) = Q (a k + t k))
    (h0 : a 0 = 0) (ht : ∀ k, Real.exp (-D) ≤ t k ∧ t k ≤ 1) (hn : 0 < n) :
    a n ≤ 2 ^ 25 * Real.exp D / n * ∑ k ∈ Finset.range n, t k :=
  accum_softmax_le (grid_hasSignificand 8 23) hQ ha
    ⟨152 * 2 ^ 23, by norm_num, by norm_num [ieee, bias]⟩ h0 ht hn

/-- **binary16**: the sum is wrong past `2^{12} e^D` keys. -/
theorem accum_softmax_f16_le (hQ : IsNearest (grid 5 10) Q) (ha : ∀ k, a (k + 1) = Q (a k + t k))
    (h0 : a 0 = 0) (ht : ∀ k, Real.exp (-D) ≤ t k ∧ t k ≤ 1) (hn : 0 < n) :
    a n ≤ 2 ^ 12 * Real.exp D / n * ∑ k ∈ Finset.range n, t k :=
  accum_softmax_le (grid_hasSignificand 5 10) hQ ha
    ⟨27 * 2 ^ 10, by norm_num, by norm_num [ieee, bias]⟩ h0 ht hn

/-- **bfloat16**: the sum is wrong past `2^9 e^D` keys. -/
theorem accum_softmax_bf16_le (hQ : IsNearest (grid 8 7) Q) (ha : ∀ k, a (k + 1) = Q (a k + t k))
    (h0 : a 0 = 0) (ht : ∀ k, Real.exp (-D) ≤ t k ∧ t k ≤ 1) (hn : 0 < n) :
    a n ≤ 2 ^ 9 * Real.exp D / n * ∑ k ∈ Finset.range n, t k :=
  accum_softmax_le (grid_hasSignificand 8 7) hQ ha
    ⟨136 * 2 ^ 7, by norm_num, by norm_num [ieee, bias]⟩ h0 ht hn

/-- **FP8 E4M3**: the sum is wrong past `2^5 e^D` keys. -/
theorem accum_softmax_e4m3_le (hQ : IsNearest (grid 4 3) Q) (ha : ∀ k, a (k + 1) = Q (a k + t k))
    (h0 : a 0 = 0) (ht : ∀ k, Real.exp (-D) ≤ t k ∧ t k ≤ 1) (hn : 0 < n) :
    a n ≤ 2 ^ 5 * Real.exp D / n * ∑ k ∈ Finset.range n, t k :=
  accum_softmax_le (grid_hasSignificand 4 3) hQ ha
    ⟨12 * 2 ^ 3, by norm_num, by norm_num [ieee, bias]⟩ h0 ht hn

/-- **FP8 E5M2**: the sum is wrong past `2^4 e^D` keys. -/
theorem accum_softmax_e5m2_le (hQ : IsNearest (grid 5 2) Q) (ha : ∀ k, a (k + 1) = Q (a k + t k))
    (h0 : a 0 = 0) (ht : ∀ k, Real.exp (-D) ≤ t k ∧ t k ≤ 1) (hn : 0 < n) :
    a n ≤ 2 ^ 4 * Real.exp D / n * ∑ k ∈ Finset.range n, t k :=
  accum_softmax_le (grid_hasSignificand 5 2) hQ ha
    ⟨19 * 2 ^ 2, by norm_num, by norm_num [ieee, bias]⟩ h0 ht hn

/-- The hypotheses of the per-format bounds are satisfiable: every format has a
nearest rounding, and the increments `1 = e^{-0}` from `0`. -/
example : (∃ Q, IsNearest (grid 8 23) Q) ∧ (∃ Q, IsNearest (grid 5 10) Q) ∧
    (∃ Q, IsNearest (grid 8 7) Q) ∧ (∃ Q, IsNearest (grid 4 3) Q) ∧
    (∃ Q, IsNearest (grid 5 2) Q) ∧ ∀ Q : ℝ → ℝ, ∃ a t : ℕ → ℝ,
      (∀ k, a (k + 1) = Q (a k + t k)) ∧ a 0 = 0 ∧ ∀ k, Real.exp (-0) ≤ t k ∧ t k ≤ 1 :=
  ⟨exists_isNearest_grid (by norm_num) 23, exists_isNearest_grid (by norm_num) 10,
    exists_isNearest_grid (by norm_num) 7, exists_isNearest_grid (by norm_num) 3,
    exists_isNearest_grid (by norm_num) 2, fun Q => ⟨fun k => Nat.rec 0 (fun _ ak => Q (ak + 1)) k,
      fun _ => 1, fun _ => rfl, rfl, fun _ => by norm_num⟩⟩

end Precision
end Transformer

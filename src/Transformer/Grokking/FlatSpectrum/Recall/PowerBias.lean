module

public import Transformer.Grokking.FlatSpectrum.Recall.LogBias

/-!
# The bias `−α log(1 + r)`: the power prior and its recall bound

Source: `review/flat_spectrum.md`, *A positional bias at the scale `s log n`* (added), item PB6,
for the ALiBi bias of arXiv:2505.22842v4, §2.2, Lemma 2 (`Transformer.BAM.alibi`), and the scale
`s log n` of SSMax, arXiv:2501.19399v1, §2.

The bias `−α log(1 + r)` is the power prior `(1 + r)^{−α}`. By Bernoulli's inequality and a
telescoping sum, its weights over the `n` distances sum to at most `n^{1−α} / (1 − α)` for
`0 ≤ α < 1` (`sum_rpow_neg_le_of_lt_one`) and to at most `α / (α − 1)` for `α > 1`
(`sum_rpow_neg_le_of_one_lt`), against `1 + log n` at `α = 1` (`LogBias`). The rivals of a key
weigh at least `1` if it is not the nearest (`one_le_sum_erase_rpow_neg`) and at least
`n^{1−α} / 2` (`rpow_one_sub_le_two_mul_sum_erase`). A key at the distance `R` whose raw score
leads by `g` is answered within `2 C (R + 1)^α n^{−sg} ∑_r (1 + r)^{−α}` (`abs_sub_powerBias_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset

/-- **Bernoulli's step below `α = 1`** (`review/flat_spectrum.md`, PB6): for `0 ≤ α < 1` and
`x ≥ 0`, `(1 − α) (x + 1)^{−α} ≤ (x + 1)^{1−α} − x^{1−α}`. -/
lemma mul_rpow_neg_le_sub_of_lt_one {α x : ℝ} (h0 : 0 ≤ α) (h1 : α < 1) (hx : 0 ≤ x) :
    (1 - α) * (x + 1) ^ (-α) ≤ (x + 1) ^ (1 - α) - x ^ (1 - α) := by
  have hx1 : 0 < x + 1 := by linarith
  have hB := rpow_one_add_le_one_add_mul_self (s := -1 / (x + 1)) (p := 1 - α)
    (by rw [neg_div, neg_le_neg_iff, div_le_one hx1]; linarith) (by linarith) (by linarith)
  have e1 : 1 + -1 / (x + 1) = x / (x + 1) := by
    field_simp
    ring
  rw [e1, Real.div_rpow hx hx1.le, div_le_iff₀ (Real.rpow_pos_of_pos hx1 _)] at hB
  have e2 : (x + 1) ^ (-α) = (x + 1) ^ (1 - α) / (x + 1) := by
    rw [← Real.rpow_sub_one hx1.ne']
    ring_nf
  have e3 : (1 + (1 - α) * (-1 / (x + 1))) * (x + 1) ^ (1 - α) =
      (x + 1) ^ (1 - α) - (1 - α) * ((x + 1) ^ (1 - α) / (x + 1)) := by ring
  rw [e2]
  linarith

/-- At `α = 1/2` and `x = 3`. -/
example := mul_rpow_neg_le_sub_of_lt_one (α := 1 / 2) (x := 3) (by norm_num) (by norm_num)
  (by norm_num)

/-- **Bernoulli's step above `α = 1`** (`review/flat_spectrum.md`, PB6): for `α > 1` and
`x ≥ 1`, `(α − 1) (x + 1)^{−α} ≤ x^{1−α} − (x + 1)^{1−α}`. -/
lemma mul_rpow_neg_le_sub_of_one_lt {α x : ℝ} (hα : 1 < α) (hx : 1 ≤ x) :
    (α - 1) * (x + 1) ^ (-α) ≤ x ^ (1 - α) - (x + 1) ^ (1 - α) := by
  have hx0 : 0 < x := by linarith
  have hx1 : 0 < x + 1 := by linarith
  have hlog : 1 / (x + 1) ≤ Real.log (x + 1) - Real.log x := by
    rw [← Real.log_div hx1.ne' hx0.ne']
    refine le_trans (le_of_eq ?_) (Real.one_sub_inv_le_log_of_pos (by positivity))
    field_simp
    ring
  have hexp := Real.add_one_le_exp ((Real.log (x + 1) - Real.log x) * (α - 1))
  have e1 : x ^ (1 - α) =
      (x + 1) ^ (1 - α) * Real.exp ((Real.log (x + 1) - Real.log x) * (α - 1)) := by
    rw [Real.rpow_def_of_pos hx0, Real.rpow_def_of_pos hx1, ← Real.exp_add]
    congr 1
    ring
  have e2 : (x + 1) ^ (-α) = (x + 1) ^ (1 - α) / (x + 1) := by
    rw [← Real.rpow_sub_one hx1.ne']
    ring_nf
  have hP := Real.rpow_pos_of_pos hx1 (1 - α)
  have h3 : (α - 1) / (x + 1) ≤ Real.exp ((Real.log (x + 1) - Real.log x) * (α - 1)) - 1 := by
    have h4 := mul_le_mul_of_nonneg_right hlog (by linarith : (0 : ℝ) ≤ α - 1)
    have h5 : (α - 1) / (x + 1) = 1 / (x + 1) * (α - 1) := by ring
    linarith
  rw [e1, e2]
  calc (α - 1) * ((x + 1) ^ (1 - α) / (x + 1)) = (x + 1) ^ (1 - α) * ((α - 1) / (x + 1)) := by
        ring
    _ ≤ (x + 1) ^ (1 - α) * (Real.exp ((Real.log (x + 1) - Real.log x) * (α - 1)) - 1) :=
        mul_le_mul_of_nonneg_left h3 hP.le
    _ = _ := by ring

/-- At `α = 2` and `x = 1`. -/
example := mul_rpow_neg_le_sub_of_one_lt (α := 2) (x := 1) (by norm_num) le_rfl

/-- **The power prior below `α = 1`** (`review/flat_spectrum.md`, PB6): for `0 ≤ α < 1` the
weights `(1 + r)^{−α}` of the distances `r < n` sum to at most `n^{1−α} / (1 − α)`. -/
lemma sum_rpow_neg_le_of_lt_one {α : ℝ} (h0 : 0 ≤ α) (h1 : α < 1) (n : ℕ) :
    ∑ r : Fin n, ((r : ℝ) + 1) ^ (-α) ≤ (n : ℝ) ^ (1 - α) / (1 - α) := by
  have key (k : ℕ) : (1 - α) * ∑ r ∈ range k, ((r : ℝ) + 1) ^ (-α) ≤ (k : ℝ) ^ (1 - α) := by
    induction k with
    | zero => simp [Real.zero_rpow (by linarith : (1 - α) ≠ 0)]
    | succ k ih =>
      rw [sum_range_succ, mul_add]
      have := mul_rpow_neg_le_sub_of_lt_one h0 h1 (Nat.cast_nonneg k)
      push_cast
      linarith
  rw [Fin.sum_univ_eq_sum_range (fun r : ℕ => ((r : ℝ) + 1) ^ (-α)) n, le_div_iff₀ (by linarith),
    mul_comm]
  exact key n

/-- At `α = 1/2` and four distances. -/
example := sum_rpow_neg_le_of_lt_one (α := 1 / 2) (by norm_num) (by norm_num) 4

/-- **The power prior above `α = 1`** (`review/flat_spectrum.md`, PB6): for `α > 1` the weights
`(1 + r)^{−α}` of the distances `r < n` sum to at most `α / (α − 1)` at every length `n`. -/
lemma sum_rpow_neg_le_of_one_lt {α : ℝ} (hα : 1 < α) (n : ℕ) :
    ∑ r : Fin n, ((r : ℝ) + 1) ^ (-α) ≤ α / (α - 1) := by
  have key (k : ℕ) : (α - 1) * ∑ r ∈ range k, ((r : ℝ) + 1 + 1) ^ (-α) ≤
      1 - ((k : ℝ) + 1) ^ (1 - α) := by
    induction k with
    | zero => simp
    | succ k ih =>
      rw [sum_range_succ, mul_add]
      have := mul_rpow_neg_le_sub_of_one_lt hα
        (by linarith [(Nat.cast_nonneg k : (0 : ℝ) ≤ k)] : (1 : ℝ) ≤ (k : ℝ) + 1)
      push_cast
      linarith
  rw [Fin.sum_univ_eq_sum_range (fun r : ℕ => ((r : ℝ) + 1) ^ (-α)) n, le_div_iff₀ (by linarith)]
  rcases n with _ | k
  · simp only [range_zero, sum_empty, zero_mul]
    linarith
  · rw [sum_range_succ']
    push_cast
    simp only [zero_add, Real.one_rpow]
    nlinarith [key k, Real.rpow_nonneg (by positivity : (0 : ℝ) ≤ (k : ℝ) + 1) (1 - α)]

/-- At `α = 2` and four distances. -/
example := sum_rpow_neg_le_of_one_lt (α := 2) (by norm_num) 4

/-- **The rivals of a far key weigh at least the nearest token** (`review/flat_spectrum.md`,
PB6): in the prior `(1 + r)^{−α}` the distances other than `R ≥ 1` weigh at least `1`, the
weight of the distance `0`. -/
lemma one_le_sum_erase_rpow_neg (α : ℝ) {n : ℕ} {R : Fin n} (hR : (R : ℕ) ≠ 0) :
    1 ≤ ∑ r ∈ univ.erase R, ((r : ℝ) + 1) ^ (-α) := by
  have h0 : (⟨0, by have := R.isLt; omega⟩ : Fin n) ∈ univ.erase R :=
    mem_erase.2 ⟨fun h => hR (by rw [← h]), mem_univ _⟩
  simpa using single_le_sum (f := fun r : Fin n => ((r : ℝ) + 1) ^ (-α))
    (fun r _ => by positivity) h0

/-- At `α = 2`, the key at the distance `1` of three. -/
example := one_le_sum_erase_rpow_neg 2 (n := 3) (R := 1) (by decide)

/-- **The rivals of every key weigh `n^{1−α} / 2`** (`review/flat_spectrum.md`, PB6): for
`α ≥ 0` and `n ≥ 2`, in the prior `(1 + r)^{−α}` the `n − 1` distances other than `R` weigh at
least `n^{1−α} / 2`. -/
lemma rpow_one_sub_le_two_mul_sum_erase {α : ℝ} (hα : 0 ≤ α) {n : ℕ} (hn : 2 ≤ n) (R : Fin n) :
    (n : ℝ) ^ (1 - α) ≤ 2 * ∑ r ∈ univ.erase R, ((r : ℝ) + 1) ^ (-α) := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have hP := Real.rpow_pos_of_pos hn0 (-α)
  have hsum := sum_le_sum fun r (_ : r ∈ univ.erase R) =>
    (Real.rpow_le_rpow_of_nonpos (by positivity) (by
      have := r.isLt
      exact_mod_cast (by omega : (r : ℕ) + 1 ≤ n)) (by linarith) :
      (n : ℝ) ^ (-α) ≤ ((r : ℝ) + 1) ^ (-α))
  rw [sum_const, card_erase_of_mem (mem_univ R), card_univ, Fintype.card_fin, nsmul_eq_mul,
    Nat.cast_sub (by omega : 1 ≤ n), Nat.cast_one] at hsum
  have hn2 : (2 : ℝ) ≤ n := by exact_mod_cast hn
  rw [sub_eq_add_neg, Real.rpow_add hn0, Real.rpow_one]
  nlinarith

/-- At `α = 1/2`, the key at the distance `0` of four. -/
example := rpow_one_sub_le_two_mul_sum_erase (α := 1 / 2) (by norm_num) (n := 4) (by norm_num) 0

/-- **The bias `−α log(1 + r)` at the scale `s log n`** (`review/flat_spectrum.md`, PB6): over `n`
tokens at the distances `r` with the bias `−α log(1 + r)`, if the raw score of the token at the
distance `R` leads every other by `g` and the values are bounded by `C`, attention at the scale
`s log n`, `s ≥ 0`, answers its value within `2 C (R + 1)^α n^{−sg} ∑_r (1 + r)^{−α}`. -/
theorem abs_sub_powerBias_le {s g C α : ℝ} (hs : 0 ≤ s) {n : ℕ} {z v : Fin n → ℝ} {R : Fin n}
    (hz : ∀ r, r ≠ R → z r + g ≤ z R) (hv : ∀ r, |v r| ≤ C) :
    |v R - biasedLogScaleAttention s (fun r : Fin n => -(α * Real.log ((r : ℝ) + 1))) z v| ≤
      2 * C * (((R : ℝ) + 1) ^ α * (n : ℝ) ^ (-(s * g)) *
        ∑ r : Fin n, ((r : ℝ) + 1) ^ (-α)) := by
  have h := abs_sub_biasedLogScaleAttention_le hs
    (fun r : Fin n => -(α * Real.log ((r : ℝ) + 1))) hz hv
  rw [Fintype.card_fin] at h
  refine h.trans (mul_le_mul_of_nonneg_left ?_ (by linarith [(abs_nonneg _).trans (hv R)]))
  have hw (r : Fin n) :
      Real.exp (-(α * Real.log ((r : ℝ) + 1)) - -(α * Real.log ((R : ℝ) + 1))) =
        ((R : ℝ) + 1) ^ α * ((r : ℝ) + 1) ^ (-α) := by
    rw [Real.rpow_def_of_pos (by positivity), Real.rpow_def_of_pos (by positivity),
      ← Real.exp_add]
    congr 1
    ring
  have hsum : ∑ r ∈ univ.erase R,
      Real.exp (-(α * Real.log ((r : ℝ) + 1)) - -(α * Real.log ((R : ℝ) + 1))) ≤
        ((R : ℝ) + 1) ^ α * ∑ r : Fin n, ((r : ℝ) + 1) ^ (-α) := by
    rw [mul_sum]
    refine le_trans (sum_le_sum_of_subset_of_nonneg (erase_subset _ _)
      fun _ _ _ => (Real.exp_pos _).le) (le_of_eq (sum_congr rfl fun r _ => hw r))
  exact (mul_le_mul_of_nonneg_left hsum (by positivity)).trans_eq (by ring)

/-- Three tokens at `α = 2`, the query's own token leading by `1`. -/
example := abs_sub_powerBias_le (s := 1) (g := 1) (C := 1) (α := 2) zero_le_one (n := 3)
  (z := ![1, 0, 0]) (v := ![1, -1, -1]) (R := 0) (fun r hr => by fin_cases r <;> simp_all)
  fun r => by fin_cases r <;> simp

end Transformer.Grokking.FlatSpectrum.Recall

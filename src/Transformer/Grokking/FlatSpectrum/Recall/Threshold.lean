module

public import Transformer.Grokking.FlatSpectrum.Recall.LogScale

/-!
# One-hot raw scores at the scale `s log n`: the worst error, and the threshold itself

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), item SA5, for
the attention scale `β_n = γ log n` of arXiv:2512.01868v4, §6 (*Long context Transformers*), on
the task of arXiv:2312.04927v1, §3.

Softmax attention misses the value of a token by at most `2 C` times the weight of the other
tokens, for values bounded by `C` (`abs_sub_softmaxAttention_le`), and the indicator dictionary,
`+1` at the token and `−1` at the others, misses it by exactly twice that weight
(`one_sub_softmaxAttention_indicator`). One-hot raw scores, `B` at the queried key and `−B` at
the other `m` keys, leave the other keys the weight `m / (n^{2sB} + m)` at the scale `s log n`
over `n = m + 1` keys (`oneHot_erase_div_sum`). So the worst error over the dictionaries of
values `±1` is exactly

    2m / (n^{2sB} + m)                (`isGreatest_oneHot_logScaleAttention_error`),

at every `s`, `B` and `m`. It tends to `2`, `1` or `0` as `2 s B` is below, at or above `1`;
`oneHot_logScaleAttention_threshold` states the first and the last. At `2 s B = 1` it is
`2m / (2m + 1)` (`oneHot_logScaleAttention_at_threshold`). That is below `1`, so every
dictionary is answered with the right sign at every number of keys; and from some number of
keys on it exceeds every `ε < 1`, so no accuracy better than `1` holds at every number of keys.
The queried key keeps the weight `n / (2n − 1)`, which tends to `1/2`, as the diagonal weight of
the equiangular model of that section does at its critical point `γ (1 − ρ) = 1`
(`Transformer.MeanField.tendsto_equiDiag_of_eq`): softmax sees only the lead of the queried
key, `2 s B log n` here and `γ (1 − ρ) log n` there.

The diff-gpt model takes `s = 1` with raw scores in `[−1, 1]`
(`Transformer.MHC.PeriRMS.score_bound`): there one-hot raw scores `±B` meet the threshold at
`B = 1/2`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset

/-- **The other tokens bound the miss** (`review/flat_spectrum.md`, SA5): softmax attention
misses the value of the token `j₀` by at most `2 C` times the weight of the other tokens, for
values bounded by `C`. -/
theorem abs_sub_softmaxAttention_le {ι : Type*} [Fintype ι] [DecidableEq ι] {s v : ι → ℝ}
    {C : ℝ} (hv : ∀ j, |v j| ≤ C) (j₀ : ι) :
    |v j₀ - softmaxAttention s v| ≤
      2 * C * ((∑ j ∈ univ.erase j₀, Real.exp (s j)) / ∑ j, Real.exp (s j)) := by
  have hZ : 0 < ∑ j, Real.exp (s j) := sum_pos (fun j _ => Real.exp_pos _) ⟨j₀, mem_univ _⟩
  have hnum : v j₀ - softmaxAttention s v =
      (∑ j ∈ univ.erase j₀, Real.exp (s j) * (v j₀ - v j)) / ∑ j, Real.exp (s j) := by
    rw [sum_erase _ (by rw [sub_self, mul_zero]), softmaxAttention, eq_div_iff hZ.ne', sub_mul,
      div_mul_cancel₀ _ hZ.ne', mul_sum, ← sum_sub_distrib]
    exact sum_congr rfl fun j _ => by ring
  rw [hnum, abs_div, abs_of_pos hZ, ← mul_div_assoc, div_le_div_iff_of_pos_right hZ, mul_sum]
  refine (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun j _ => ?_)
  rw [abs_mul, abs_of_pos (Real.exp_pos _), mul_comm]
  exact mul_le_mul_of_nonneg_right (by linarith [abs_sub (v j₀) (v j), hv j, hv j₀])
    (Real.exp_pos _).le

/-- Two tokens with scores `1` and `0` and values `±1`. -/
example := abs_sub_softmaxAttention_le (s := (![1, 0] : Fin 2 → ℝ)) (v := ![1, -1]) (C := 1)
  (fun j => by fin_cases j <;> simp) 0

/-- One-hot raw scores `±B` at the scale `s log n` over `n = m + 1` keys leave the keys other
than the queried one the weight `m / (n^{2sB} + m)`. -/
lemma oneHot_erase_div_sum (s B : ℝ) {m : ℕ} (q : Fin (m + 1)) :
    (∑ k ∈ univ.erase q, Real.exp (s * Real.log (Fintype.card (Fin (m + 1))) *
        if k = q then B else -B)) /
      ∑ k, Real.exp (s * Real.log (Fintype.card (Fin (m + 1))) * if k = q then B else -B) =
      m / (((m : ℝ) + 1) ^ (2 * s * B) + m) := by
  have hx : (0 : ℝ) < (m : ℝ) + 1 := by positivity
  set c := s * Real.log (Fintype.card (Fin (m + 1))) with hc
  have hR' : ∑ k ∈ univ.erase q, Real.exp (c * if k = q then B else -B) =
      ∑ _k ∈ univ.erase q, Real.exp (c * -B) :=
    sum_congr rfl fun k hk => by rw [ite_eq_right (ne_of_mem_erase hk)]
  have hR : ∑ k ∈ univ.erase q, Real.exp (c * if k = q then B else -B) =
      m * Real.exp (c * -B) := by
    rw [hR', sum_const, card_erase_of_mem (mem_univ q), card_univ, Fintype.card_fin,
      Nat.add_sub_cancel, nsmul_eq_mul]
  have hZ : ∑ k, Real.exp (c * if k = q then B else -B) =
      Real.exp (c * B) + m * Real.exp (c * -B) := by
    rw [← add_sum_erase _ _ (mem_univ q), ite_eq_left rfl, hR]
  have he : Real.exp (c * B) * Real.exp (c * B) = ((m : ℝ) + 1) ^ (2 * s * B) := by
    rw [← Real.exp_add, hc, Real.rpow_def_of_pos hx, Fintype.card_fin, Nat.cast_add,
      Nat.cast_one]
    ring_nf
  have hneg : Real.exp (c * -B) = (Real.exp (c * B))⁻¹ := by rw [mul_neg, Real.exp_neg]
  have hE := Real.exp_pos (c * B)
  rw [hR, hZ, hneg, ← he]
  field_simp

/-- **The worst error of one-hot raw scores** (`review/flat_spectrum.md`, SA5): with the raw
score `B` at the queried key and `−B` at the other `m` keys, attention at the scale
`s log (m + 1)` answers the dictionaries of values `±1` with errors whose greatest is
`2m / ((m + 1)^{2sB} + m)`, which the indicator dictionary attains. -/
theorem isGreatest_oneHot_logScaleAttention_error (s B : ℝ) {m : ℕ} (q : Fin (m + 1)) :
    IsGreatest (Set.range fun π : Fin (m + 1) → Bool => |(if π q then (1 : ℝ) else -1) -
        logScaleAttention s (fun k => if k = q then B else -B)
          fun k => if π k then (1 : ℝ) else -1|)
      (2 * m / (((m : ℝ) + 1) ^ (2 * s * B) + m)) := by
  have hw := oneHot_erase_div_sum s B q
  refine ⟨⟨fun k => decide (k = q), ?_⟩, ?_⟩
  · dsimp only
    rw [logScaleAttention, abs_recall_error_decide, one_sub_softmaxAttention_indicator]
    rw [mul_div_assoc, hw, ← mul_div_assoc, abs_of_nonneg (by positivity)]
  · rintro _ ⟨π, rfl⟩
    have h := abs_sub_softmaxAttention_le
      (s := fun k => s * Real.log (Fintype.card (Fin (m + 1))) * if k = q then B else -B)
      (v := fun k => if π k then (1 : ℝ) else -1) (C := 1) (fun k => by split_ifs <;> simp) q
    beta_reduce at h
    rw [hw, mul_one, ← mul_div_assoc] at h
    exact h

/-- At `s = 1`, `B = 1` and four keys the worst error is `6 / (4² + 3)`. -/
example := isGreatest_oneHot_logScaleAttention_error 1 1 (3 : Fin 4)

/-- **At the threshold `2 s B = 1`** (`review/flat_spectrum.md`, SA5): one-hot raw scores `±B`
at the scale `s log (m + 1)` answer every dictionary of values `±1` within `2m / (2m + 1) < 1`,
so with the right sign at every number of keys; and for every `ε < 1`, from some number of keys
on, every query has a dictionary answered with an error above `ε`. -/
theorem oneHot_logScaleAttention_at_threshold {s B : ℝ} (h : 2 * s * B = 1) :
    (∀ (m : ℕ) (q : Fin (m + 1)) (π : Fin (m + 1) → Bool), |(if π q then (1 : ℝ) else -1) -
        logScaleAttention s (fun k => if k = q then B else -B)
          fun k => if π k then (1 : ℝ) else -1| < 1) ∧
    ∀ ε : ℝ, ε < 1 → ∃ M : ℕ, ∀ m ≥ M, ∀ q : Fin (m + 1), ∃ π : Fin (m + 1) → Bool,
      ε < |(if π q then (1 : ℝ) else -1) -
        logScaleAttention s (fun k => if k = q then B else -B)
          fun k => if π k then (1 : ℝ) else -1| := by
  have hW (m : ℕ) : 2 * (m : ℝ) / (((m : ℝ) + 1) ^ (2 * s * B) + m) = 2 * m / (2 * m + 1) := by
    rw [h, Real.rpow_one]
    ring
  refine ⟨fun m q π => ?_, fun ε hε => ?_⟩
  · have h1 := (isGreatest_oneHot_logScaleAttention_error s B q).2 (Set.mem_range_self π)
    rw [hW] at h1
    refine h1.trans_lt ?_
    rw [div_lt_one (by positivity)]
    linarith
  · obtain ⟨M, hM⟩ := exists_nat_gt (ε / (2 * (1 - ε)))
    rw [div_lt_iff₀ (by linarith)] at hM
    refine ⟨M, fun m hm q => ?_⟩
    obtain ⟨π, hπ⟩ := (isGreatest_oneHot_logScaleAttention_error s B q).1
    refine ⟨π, lt_of_lt_of_eq ?_ hπ.symm⟩
    have hm' : (M : ℝ) ≤ m := by exact_mod_cast hm
    have h2 := mul_le_mul_of_nonneg_right hm' (by linarith : (0 : ℝ) ≤ 2 * (1 - ε))
    rw [hW, lt_div_iff₀ (by positivity)]
    linarith

/-- Raw scores `±1/2` at `s = 1`, the threshold of the diff-gpt model's raw scores. -/
example := oneHot_logScaleAttention_at_threshold (s := 1) (B := 1 / 2) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Recall

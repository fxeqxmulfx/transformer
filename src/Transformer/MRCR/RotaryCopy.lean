module

public import Transformer.MRCR.NearMissThreshold
public import Transformer.MRCR.Returns

/-!
# A RoPE head loses every position at long range

In no paper. In MRCR "the model must reproduce the output from the conversation (the needle)",
a chunk of text up to 512 tokens after a random string (arXiv:2409.12640v2, `evals.tex`, line
127, and `eval_framework.tex`, line 52); MRCR-V2 is arXiv:2507.06261v6, Appendix, line 75 of
`8_appendix/1_evaluation_additional_details.tex`. Written token by token, the copy can follow
the answer by its position: the token written at the step `ℓ` is the token `ℓ` of the answer, at
a distance from the writing token that stays fixed along the copy. A head that copies so scores
the visible tokens by their distance only, and to return the token at the distance `q` at every
length its score needs a gap there at every length, the hypothesis of SA5
(`Transformer.Grokking.FlatSpectrum.Recall.abs_logScaleAttention_sub_le`).

A RoPE score of the distance (`Transformer.MRCR.ropeScore`, arXiv:2309.00071v3, §2.1, Eq. (5))
has no such gap. Among the `n` positions `0, …, n − 1` at the distances from the writing token,
every `q` has at least `n / (2 (N + 1)^K) − 1` others whose score trails its own by at most
`2π ∑_i (|a_i| + |b_i|) / (N + 1)` (`le_card_nearReturns`): the returns of
`Transformer.MRCR.exists_returns`, taken forward from a `q` in the first half and backward from a
`q` in the second. With `N` so large that `s` times that margin is below `1`, linearly many near
misses trail by less than `1/s`, and `Transformer.MRCR.exists_lt_logScaleAttention_nearMiss_error`
applies with the exponent `1`. For all frequencies and coefficients and every scale `s ≥ 0`, from
some length on, attention at the scale `s log n` answers the indicator dictionary of every
position, `1` there and `−1` elsewhere, with an error above every `ε < 2`
(`exists_lt_logScaleAttention_ropeScore_error`).

So a positional RoPE head copies by position at no fixed scale at every length; a previous-token
head is the case `q = 1`. The proof needs at least `4 (N + 1)^K` tokens, astronomically many for
`K = 64` frequencies: the theorem is about every length, not about 1M tokens. A head that copies
by content instead, the induction head of `Transformer.Grokking.FlatSpectrum.Recall`, matches the
token before; where that token recurs, the matches tie and attention averages them
(`Transformer.MRCR.abs_logScaleAttention_sub_average_le`).
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Filter Transformer.Grokking.FlatSpectrum.Recall

variable {K : ℕ}

/-- **Every position has a linear number of near returns**: among the positions `0, …, n − 1` at
the distances from the writing token, every `q` has at least `n / (2 (N + 1)^K) − 1` others whose
RoPE score trails its own by at most `2π ∑_i (|a_i| + |b_i|) / (N + 1)`. -/
theorem le_card_nearReturns (θ a b : Fin K → ℝ) (N : ℕ) {n : ℕ} (q : Fin n) :
    (n : ℝ) / (2 * ((N : ℝ) + 1) ^ K) - 1 ≤ ((univ.filter fun k : Fin n =>
      ropeScore θ a b (q : ℕ) ≤ ropeScore θ a b (k : ℕ) +
        2 * Real.pi / ((N : ℝ) + 1) * ropeNorm a b).erase q).card := by
  have hq := q.isLt
  obtain ⟨R, hcard, hR⟩ := exists_returns θ N (H := (n + 1) / 2) (by omega)
  have hRT : R.card ≤ ((univ.filter fun k : Fin n => ropeScore θ a b (q : ℕ) ≤
      ropeScore θ a b (k : ℕ) + 2 * Real.pi / ((N : ℝ) + 1) * ropeNorm a b).erase q).card := by
    rw [← card_map Fin.valEmbedding]
    refine card_le_card_of_injOn (fun d => if 2 * (q : ℕ) < n then (q : ℕ) + d else (q : ℕ) - d)
      (fun d hd => ?_) fun d₁ hd₁ d₂ hd₂ h => ?_
    · obtain ⟨h1, h2, h3⟩ := hR d hd
      rw [Finset.mem_coe, Finset.mem_map]
      by_cases hq2 : 2 * (q : ℕ) < n
      · refine ⟨⟨q + d, by omega⟩, mem_erase.2 ⟨fun h => ?_, mem_filter.2 ⟨mem_univ _, ?_⟩⟩,
          by simp [hq2]⟩
        · have := congrArg Fin.val h
          simp only at this
          omega
        · have := (abs_le.1 (abs_ropeScore_add_sub_le (a := a) (b := b) h3 (q : ℕ))).1
          push_cast
          linarith
      · refine ⟨⟨q - d, by omega⟩, mem_erase.2 ⟨fun h => ?_, mem_filter.2 ⟨mem_univ _, ?_⟩⟩,
          by simp [hq2]⟩
        · have := congrArg Fin.val h
          simp only at this
          omega
        · have := (abs_le.1 (abs_ropeScore_sub_sub_le (a := a) (b := b) h3 (q : ℕ))).1
          simp only
          rw [Nat.cast_sub (by omega)]
          linarith
    · have h₁ := (hR d₁ hd₁).2.1
      have h₂ := (hR d₂ hd₂).2.1
      by_cases hq2 : 2 * (q : ℕ) < n
      · simp only [hq2, ite_true] at h
        omega
      · simp only [hq2, ite_false] at h
        omega
  have hH : (n : ℝ) ≤ 2 * (((n + 1) / 2 : ℕ) : ℝ) := by
    exact_mod_cast (by omega : n ≤ 2 * ((n + 1) / 2))
  have hP : (0 : ℝ) < ((N : ℝ) + 1) ^ K := by positivity
  have hdiv : (n : ℝ) / (2 * ((N : ℝ) + 1) ^ K) ≤ (((n + 1) / 2 : ℕ) : ℝ) / ((N : ℝ) + 1) ^ K := by
    rw [div_le_div_iff₀ (by positivity) hP]
    nlinarith
  calc (n : ℝ) / (2 * ((N : ℝ) + 1) ^ K) - 1
      ≤ (((n + 1) / 2 : ℕ) : ℝ) / ((N : ℝ) + 1) ^ K - 1 := by linarith
    _ ≤ R.card := hcard
    _ ≤ _ := Nat.cast_le.2 hRT

/-- One frequency, three positions, the middle one. -/
example := le_card_nearReturns (fun _ : Fin 1 => 1) (fun _ => 1) (fun _ => 0) 4 (n := 3) 1

/-- **A RoPE head loses every position at long range** (MRCR, arXiv:2409.12640v2,
`eval_framework.tex`; RoPE as in arXiv:2309.00071v3, §2.1, Eq. (5)): for every RoPE score of the
distance and every scale `s ≥ 0`, from some length on, attention at the scale `s log n` answers
the indicator dictionary of every position, `1` there and `−1` elsewhere, with an error above
every `ε < 2`. -/
theorem exists_lt_logScaleAttention_ropeScore_error (θ a b : Fin K → ℝ) {s ε : ℝ} (hs : 0 ≤ s)
    (hε : ε < 2) : ∃ M : ℕ, ∀ m ≥ M, ∀ q : Fin (m + 1),
      ε < 1 - logScaleAttention s (fun k : Fin (m + 1) => ropeScore θ a b (k : ℕ))
        fun k => if k = q then 1 else -1 := by
  obtain ⟨N, hN⟩ : ∃ N : ℕ, s * (2 * Real.pi / ((N : ℝ) + 1) * ropeNorm a b) < 1 := by
    refine ⟨⌈2 * Real.pi * s * ropeNorm a b⌉₊, ?_⟩
    have h1 := Nat.le_ceil (2 * Real.pi * s * ropeNorm a b)
    have hpos : (0 : ℝ) < (⌈2 * Real.pi * s * ropeNorm a b⌉₊ : ℝ) + 1 := by positivity
    rw [show s * (2 * Real.pi / ((⌈2 * Real.pi * s * ropeNorm a b⌉₊ : ℝ) + 1) * ropeNorm a b) =
      2 * Real.pi * s * ropeNorm a b / ((⌈2 * Real.pi * s * ropeNorm a b⌉₊ : ℝ) + 1) by ring,
      div_lt_one hpos]
    linarith
  have hP : (0 : ℝ) < ((N : ℝ) + 1) ^ K := by positivity
  refine exists_lt_logScaleAttention_nearMiss_error hs (c := 1 / (4 * ((N : ℝ) + 1) ^ K))
    (α := 1) (by positivity) hN hε (fun _ _ k => ropeScore θ a b (k : ℕ))
    (fun _ q => univ.filter fun k => ropeScore θ a b (q : ℕ) ≤
      ropeScore θ a b (k : ℕ) + 2 * Real.pi / ((N : ℝ) + 1) * ropeNorm a b) ?_
    fun _ _ _ hk => (mem_filter.1 hk).2
  filter_upwards [eventually_ge_atTop (4 * (N + 1) ^ K)] with m hm q
  have h := le_card_nearReturns θ a b N q
  have hm' : 4 * ((N : ℝ) + 1) ^ K ≤ (m : ℝ) + 1 := by
    exact_mod_cast (by omega : 4 * (N + 1) ^ K ≤ m + 1)
  have h4 : 1 ≤ ((m : ℝ) + 1) / (4 * ((N : ℝ) + 1) ^ K) := by
    rw [le_div_iff₀ (by positivity)]
    linarith
  have h2 : ((m : ℝ) + 1) / (2 * ((N : ℝ) + 1) ^ K) =
      2 * (((m : ℝ) + 1) / (4 * ((N : ℝ) + 1) ^ K)) := by
    field_simp
    ring
  push_cast at h
  rw [Real.rpow_one, one_div_mul_eq_div]
  linarith

/-- One frequency at the scale `2 log n`. -/
example := exists_lt_logScaleAttention_ropeScore_error (fun _ : Fin 1 => 1) (fun _ => 1)
  (fun _ => 0) (s := 2) (ε := 1) (by norm_num) (by norm_num)

end Transformer.MRCR

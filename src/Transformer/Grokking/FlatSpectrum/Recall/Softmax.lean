module

public import Mathlib.Analysis.SpecialFunctions.Exp
public import Transformer.Grokking.FlatSpectrum.Recall.Defs

/-!
# Softmax attention: a leading score and bounded scores

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), items SA1 and
SA3, for the attention of Yang, Cadilhac and Chiang, arXiv:2506.16055v3, §2.1,
`att(q, K, v) = sumdiv((exp q Kᵀ) ∘ v, exp q Kᵀ)`.

Softmax attention returns the average of the values `v_j` with the weights `exp s_j`
(`softmaxAttention`).

SA1. If one token leads every other score by `g`, the average is within `2 B (n − 1) e^{−g}` of
its value, for values bounded by `B` (`abs_softmaxAttention_sub_le`).

SA3. If the scores of the `n` tokens are bounded by `B`, every token has at most the weight
`e^{2B} / (e^{2B} + n − 1)`: the others keep at least `ρ = (n − 1) / (e^{2B} + n − 1)` of the
sum of the weights (`mul_sum_exp_le_sum_erase`). The attention is the value `v₀` of a token and
the average `a` of the others, mixed with the weight of the token, so

    ρ |v₀ − a| ≤ |v₀ − att|                  (`mul_abs_sub_le_abs_sub_softmaxAttention`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset

/-- **Softmax attention at the query** (arXiv:2506.16055v3, §2.1, `att`): the average of the
values `v j` with the weights `exp (s j)` of the scores. -/
noncomputable def softmaxAttention {ι : Type*} [Fintype ι] (s v : ι → ℝ) : ℝ :=
  (∑ j, Real.exp (s j) * v j) / ∑ j, Real.exp (s j)

/-- **A leading score concentrates the attention** (`review/flat_spectrum.md`, SA1): if the
token `j₀` leads every other score by `g` and the values are bounded by `B`, the attention is
within `2 B (n − 1) e^{−g}` of the value of `j₀`. -/
theorem abs_softmaxAttention_sub_le {ι : Type*} [Fintype ι] [DecidableEq ι] {s v : ι → ℝ}
    {j₀ : ι} {g B : ℝ} (hs : ∀ j, j ≠ j₀ → s j + g ≤ s j₀) (hv : ∀ j, |v j| ≤ B) :
    |softmaxAttention s v - v j₀| ≤ 2 * B * (Fintype.card ι - 1) * Real.exp (-g) := by
  have hB : 0 ≤ B := (abs_nonneg _).trans (hv j₀)
  have hZ0 : Real.exp (s j₀) ≤ ∑ j, Real.exp (s j) :=
    single_le_sum (f := fun j => Real.exp (s j)) (fun j _ => (Real.exp_pos _).le) (mem_univ j₀)
  have hZ : 0 < ∑ j, Real.exp (s j) := (Real.exp_pos _).trans_le hZ0
  have hnum : softmaxAttention s v - v j₀ =
      (∑ j ∈ univ.erase j₀, Real.exp (s j) * (v j - v j₀)) / ∑ j, Real.exp (s j) := by
    rw [sum_erase _ (by rw [sub_self, mul_zero]), softmaxAttention, eq_div_iff hZ.ne', sub_mul,
      div_mul_cancel₀ _ hZ.ne', mul_comm (v j₀), sum_mul, ← sum_sub_distrib]
    exact sum_congr rfl fun j _ => by ring
  have hterm (j : ι) (hj : j ∈ univ.erase j₀) :
      |Real.exp (s j) * (v j - v j₀)| ≤ Real.exp (s j₀) * Real.exp (-g) * (2 * B) := by
    rw [abs_mul, abs_of_pos (Real.exp_pos _), ← Real.exp_add]
    refine mul_le_mul (Real.exp_le_exp.2 (by linarith [hs j (ne_of_mem_erase hj)]))
      ?_ (abs_nonneg _) (Real.exp_pos _).le
    calc |v j - v j₀| ≤ |v j| + |v j₀| := abs_sub _ _
      _ ≤ 2 * B := by linarith [hv j, hv j₀]
  rw [hnum, abs_div, abs_of_pos hZ, div_le_iff₀ hZ]
  calc |∑ j ∈ univ.erase j₀, Real.exp (s j) * (v j - v j₀)|
      ≤ ∑ j ∈ univ.erase j₀, |Real.exp (s j) * (v j - v j₀)| := abs_sum_le_sum_abs _ _
    _ ≤ ∑ _j ∈ univ.erase j₀, Real.exp (s j₀) * Real.exp (-g) * (2 * B) := sum_le_sum hterm
    _ = 2 * B * (Fintype.card ι - 1) * Real.exp (-g) * Real.exp (s j₀) := by
        rw [sum_const, card_erase_of_mem (mem_univ j₀), card_univ, nsmul_eq_mul,
          Nat.cast_sub (Fintype.card_pos_iff.2 ⟨j₀⟩), Nat.cast_one]
        ring
    _ ≤ 2 * B * (Fintype.card ι - 1) * Real.exp (-g) * ∑ j, Real.exp (s j) :=
        mul_le_mul_of_nonneg_left hZ0 (mul_nonneg (mul_nonneg (mul_nonneg zero_le_two hB)
          (sub_nonneg.2 (by exact_mod_cast Fintype.card_pos_iff.2 ⟨j₀⟩)))
          (Real.exp_pos _).le)

/-- Two tokens with scores `1` and `0` and values `±1`: the attention is within `2 e^{−1}` of
the value of the first. -/
example : |softmaxAttention (![1, 0] : Fin 2 → ℝ) ![1, -1] - 1| ≤
    2 * 1 * ((Fintype.card (Fin 2) : ℝ) - 1) * Real.exp (-1) :=
  abs_softmaxAttention_sub_le (j₀ := 0) (fun j hj => by fin_cases j <;> simp_all)
    fun j => by fin_cases j <;> simp

/-- **Bounded scores keep weight off every token** (`review/flat_spectrum.md`, SA3): with `n`
scores bounded by `B`, the tokens other than `j₀` hold at least `(n − 1) / (e^{2B} + n − 1)` of
the sum of the weights. -/
lemma mul_sum_exp_le_sum_erase {ι : Type*} [Fintype ι] [DecidableEq ι] {s : ι → ℝ} {B : ℝ}
    (hs : ∀ j, |s j| ≤ B) (j₀ : ι) :
    ((Fintype.card ι : ℝ) - 1) / (Real.exp (2 * B) + Fintype.card ι - 1) *
      ∑ j, Real.exp (s j) ≤ ∑ j ∈ univ.erase j₀, Real.exp (s j) := by
  have hm : (1 : ℝ) ≤ Fintype.card ι := by exact_mod_cast Fintype.card_pos_iff.2 ⟨j₀⟩
  have hden : 0 < Real.exp (2 * B) + Fintype.card ι - 1 := by
    linarith [Real.exp_pos (2 * B)]
  have hE : Real.exp (s j₀) ≤ Real.exp B := Real.exp_le_exp.2 (le_of_abs_le (hs j₀))
  have hR : ((Fintype.card ι : ℝ) - 1) * Real.exp (-B) ≤
      ∑ j ∈ univ.erase j₀, Real.exp (s j) := by
    have h := card_nsmul_le_sum (univ.erase j₀) (fun j => Real.exp (s j)) (Real.exp (-B))
      fun j _ => Real.exp_le_exp.2 (neg_le_of_abs_le (hs j))
    rwa [card_erase_of_mem (mem_univ j₀), card_univ, nsmul_eq_mul,
      Nat.cast_sub (Fintype.card_pos_iff.2 ⟨j₀⟩), Nat.cast_one] at h
  have h2B : Real.exp (2 * B) = Real.exp B * Real.exp B := by
    rw [← Real.exp_add, two_mul]
  have hBB : Real.exp (-B) * Real.exp B = 1 := by
    rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]
  have h1 : ((Fintype.card ι : ℝ) - 1) * Real.exp B ≤
      (∑ j ∈ univ.erase j₀, Real.exp (s j)) * (Real.exp B * Real.exp B) :=
    calc ((Fintype.card ι : ℝ) - 1) * Real.exp B
        = (Fintype.card ι - 1) * Real.exp (-B) * (Real.exp B * Real.exp B) := by
          rw [mul_assoc, ← mul_assoc (Real.exp (-B)), hBB, one_mul]
      _ ≤ _ := mul_le_mul_of_nonneg_right hR (by positivity)
  have h2 := mul_le_mul_of_nonneg_left hE (sub_nonneg.2 hm)
  rw [div_mul_eq_mul_div, div_le_iff₀ hden, ← add_sum_erase _ _ (mem_univ j₀), h2B]
  linarith

/-- Two tokens with scores `1` and `0`, bounded by `1`. -/
example := mul_sum_exp_le_sum_erase (s := (![1, 0] : Fin 2 → ℝ)) (B := 1)
  (fun j => by fin_cases j <;> simp) 0

/-- **Bounded scores leave the attention between a token and the others**
(`review/flat_spectrum.md`, SA3): the attention misses the value of the token `j₀` by at least
`(n − 1) / (e^{2B} + n − 1)` of its distance to the average of the other tokens. -/
theorem mul_abs_sub_le_abs_sub_softmaxAttention {ι : Type*} [Fintype ι] [DecidableEq ι]
    {s v : ι → ℝ} {B : ℝ} (hs : ∀ j, |s j| ≤ B) (j₀ : ι) :
    ((Fintype.card ι : ℝ) - 1) / (Real.exp (2 * B) + Fintype.card ι - 1) *
      |v j₀ - (∑ j ∈ univ.erase j₀, Real.exp (s j) * v j) /
        ∑ j ∈ univ.erase j₀, Real.exp (s j)| ≤ |v j₀ - softmaxAttention s v| := by
  have hρ := mul_sum_exp_le_sum_erase hs j₀
  set R := ∑ j ∈ univ.erase j₀, Real.exp (s j)
  set N := ∑ j ∈ univ.erase j₀, Real.exp (s j) * v j
  have hZ : ∑ j, Real.exp (s j) = Real.exp (s j₀) + R := (add_sum_erase _ _ (mem_univ j₀)).symm
  have hNum : ∑ j, Real.exp (s j) * v j = Real.exp (s j₀) * v j₀ + N :=
    (add_sum_erase _ (fun j => Real.exp (s j) * v j) (mem_univ j₀)).symm
  have hR0 : 0 ≤ R := sum_nonneg fun _ _ => (Real.exp_pos _).le
  have hZ0 : 0 < Real.exp (s j₀) + R := add_pos_of_pos_of_nonneg (Real.exp_pos _) hR0
  have hA : v j₀ - softmaxAttention s v = (R * v j₀ - N) / (Real.exp (s j₀) + R) := by
    rw [softmaxAttention, hZ, hNum, eq_div_iff hZ0.ne', sub_mul, div_mul_cancel₀ _ hZ0.ne']
    ring
  rw [hZ] at hρ
  rcases hR0.eq_or_lt with hR | hR
  · have h0 : ((Fintype.card ι : ℝ) - 1) / (Real.exp (2 * B) + Fintype.card ι - 1) ≤ 0 := by
      exact nonpos_of_mul_nonpos_left (hρ.trans hR.symm.le) hZ0
    exact (mul_nonpos_of_nonpos_of_nonneg h0 (abs_nonneg _)).trans (abs_nonneg _)
  · have hx : v j₀ - N / R = (R * v j₀ - N) / R := by field_simp
    rw [hA, hx, abs_div, abs_div, abs_of_pos hZ0, abs_of_pos hR, le_div_iff₀ hZ0]
    calc (Fintype.card ι - 1) / (Real.exp (2 * B) + Fintype.card ι - 1) *
          (|R * v j₀ - N| / R) * (Real.exp (s j₀) + R)
        = (Fintype.card ι - 1) / (Real.exp (2 * B) + Fintype.card ι - 1) *
          (Real.exp (s j₀) + R) * (|R * v j₀ - N| / R) := by ring
      _ ≤ R * (|R * v j₀ - N| / R) :=
          mul_le_mul_of_nonneg_right hρ (div_nonneg (abs_nonneg _) hR.le)
      _ = |R * v j₀ - N| := mul_div_cancel₀ _ hR.ne'

/-- Two tokens with scores `1` and `0` and values `±1`. -/
example := mul_abs_sub_le_abs_sub_softmaxAttention (s := (![1, 0] : Fin 2 → ℝ))
  (v := ![1, -1]) (B := 1) (fun j => by fin_cases j <;> simp) 0

end Transformer.Grokking.FlatSpectrum.Recall

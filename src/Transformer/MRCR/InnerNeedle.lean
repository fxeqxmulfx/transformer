module

public import Transformer.Grokking.FlatSpectrum.Recall.Fading
public import Transformer.Grokking.FlatSpectrum.Recall.LogScale

/-!
# Distance picks no inner needle

Source: MRCR of arXiv:2409.12640v2, `eval_framework.tex`, lines 35–37: for a topic and a format
the context holds "a list of model outputs, in the order that they were presented in the
context", and the query asks for "the entry with the desired index", which "requires the model
to use information about the ordering of multiple needles"; MRCR-V2 of arXiv:2507.06261v6,
Appendix, line 75 of `8_appendix/1_evaluation_additional_details.tex`, has eight needles. The
attention is that of SSMax, arXiv:2501.19399v1, §2.2, at the scale `s log n`.

The asks of one key are the same text. Let a needle at the distance `d` from the query have the
raw score `c + h d`, any `h : ℕ → ℝ`: content shared by the needles plus position, as for RoPE,
ALiBi or no position at all. Place `k` needles equally spaced by `G`, the last at the distance
`D` (`spaced`). For a needle `i` with a needle on each side, the placements with the last needle
at `D` and at `D + G` do not both give `i` the largest raw score among the needles: one of them
gives a neighbour of `i` a raw score at least that of `i` (`exists_spaced_le_of_inner`), whatever
`h`, `G` and `D`. There, whatever the other tokens, attention at the scale `s log n` answers the
indicator dictionary of `i` with a value at most `0`, an error of at least `1`
(`exists_spaced_logScaleAttention_indicator_nonpos`). At a fixed query position `P` a raw score
`h' p` of the absolute position `p` is the raw score `h' (P − p)` of the distance, so the same
holds for it.

So one layer that scores the needles by their shared content and their position does not return
the `i`-th of eight identical asks, `1 < i < 8`, at every placement; the count of earlier asks
has to come from an earlier layer. The argument needs a needle on each side: a bias that falls
with the distance returns the last needle (`Transformer.MRCR.abs_logScaleAttention_sub_last_le`).
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Transformer.Grokking.FlatSpectrum.Recall

/-- `k` needles equally spaced by `G`, the last at the distance `D` from the query: the needle `j`
sits at the distance `D + (k − 1 − j) G`. -/
def spaced (k D G : ℕ) (j : Fin k) : ℕ := D + (k - 1 - j.val) * G

/-- Equally spaced needles come in order: the first is the farthest from the query. -/
lemma spaced_strictAnti {k D G : ℕ} (hG : 0 < G) : StrictAnti (spaced k D G) := fun a b hab => by
  have := Fin.lt_def.1 hab
  have := b.isLt
  exact Nat.add_lt_add_left (Nat.mul_lt_mul_of_pos_right (by omega) hG) D

/-- Three needles one token apart, the last next to the query. -/
example : StrictAnti (spaced 3 1 1) := spaced_strictAnti one_pos

/-- **Distance picks no inner needle** (MRCR, arXiv:2409.12640v2, `eval_framework.tex`): for every
raw score `h` of the distance, every spacing `G`, every distance `D` and every needle `i` of `k`
with a needle on each side, the needles spaced by `G` with the last at the distance `D` or at
`D + G` give a neighbour of `i` a raw score at least that of `i`. -/
theorem exists_spaced_le_of_inner {k : ℕ} (h : ℕ → ℝ) (D G : ℕ) (i : Fin k) (h0 : 0 < i.val)
    (hk : i.val + 1 < k) :
    ∃ D' ∈ ({D, D + G} : Finset ℕ), ∃ j, j ≠ i ∧ h (spaced k D' G i) ≤ h (spaced k D' G j) := by
  have e1 : spaced k D G ⟨i.val - 1, by omega⟩ = spaced k (D + G) G i := by
    simp only [spaced]
    rw [show k - 1 - (i.val - 1) = k - 1 - i.val + 1 by omega]
    generalize k - 1 - i.val = a
    ring
  have e2 : spaced k D G i = spaced k (D + G) G ⟨i.val + 1, hk⟩ := by
    simp only [spaced]
    rw [show k - 1 - i.val = k - 1 - (i.val + 1) + 1 by omega]
    generalize k - 1 - (i.val + 1) = a
    ring
  by_cases hA : h (spaced k (D + G) G i) ≤ h (spaced k (D + G) G ⟨i.val + 1, hk⟩)
  · exact ⟨D + G, by simp, ⟨i.val + 1, hk⟩, fun he => by simp [Fin.ext_iff] at he, hA⟩
  · refine ⟨D, by simp, ⟨i.val - 1, by omega⟩, fun he => ?_, ?_⟩
    · rw [Fin.ext_iff] at he
      dsimp at he
      omega
    · rw [e1, e2]
      exact (not_le.1 hA).le

/-- Three needles, the score the distance itself: the middle one is not picked. -/
example := exists_spaced_le_of_inner (k := 3) (fun d => (d : ℝ)) 1 1 1 (by decide) (by decide)

/-- **One layer returns no inner needle by position** (MRCR, arXiv:2409.12640v2,
`eval_framework.tex`): for every raw score `h` of the distance, scale `s ≥ 0`, content score `c`
shared by the needles, spacing `G`, distance `D` and inner needle `i`, one of the placements
`spaced k D G` and `spaced k (D + G) G` makes attention at the scale `s log n` answer the indicator
dictionary of `i` with a value at most `0`, whatever the other tokens: over every set of tokens
`ι` that holds the needles at `e`, with any raw scores there. -/
theorem exists_spaced_logScaleAttention_indicator_nonpos {k : ℕ} (h : ℕ → ℝ) {s : ℝ}
    (hs : 0 ≤ s) (c : ℝ) (D G : ℕ) (i : Fin k) (h0 : 0 < i.val) (hk : i.val + 1 < k) :
    ∃ D' ∈ ({D, D + G} : Finset ℕ),
      ∀ {ι : Type} [Fintype ι] [DecidableEq ι] (e : Fin k → ι), Function.Injective e →
        ∀ z : ι → ℝ, (∀ j, z (e j) = c + h (spaced k D' G j)) →
          logScaleAttention s z (fun t => if t = e i then 1 else -1) ≤ 0 := by
  obtain ⟨D', hD', j, hji, hle⟩ := exists_spaced_le_of_inner h D G i h0 hk
  refine ⟨D', hD', fun {ι} _ _ e he z hz => ?_⟩
  have hn : (1 : ℝ) ≤ Fintype.card ι := by exact_mod_cast Fintype.card_pos_iff.2 ⟨e i⟩
  have hc : 0 ≤ s * Real.log (Fintype.card ι) := mul_nonneg hs (Real.log_nonneg hn)
  set x : ι → ℝ := fun t => s * Real.log (Fintype.card ι) * z t with hxdef
  have hx : x (e i) ≤ x (e j) := by
    simp only [hxdef, hz]
    exact mul_le_mul_of_nonneg_left (by linarith) hc
  have h1 := one_sub_softmaxAttention_indicator x (e i)
  have hej : e j ∈ univ.erase (e i) := mem_erase.2 ⟨fun h' => hji (he h'), mem_univ _⟩
  have hS : Real.exp (x (e i)) ≤ ∑ t ∈ univ.erase (e i), Real.exp (x t) :=
    (Real.exp_le_exp.2 hx).trans
      (single_le_sum (f := fun t => Real.exp (x t)) (fun t _ => (Real.exp_pos _).le) hej)
  have hZ : ∑ t, Real.exp (x t) = Real.exp (x (e i)) + ∑ t ∈ univ.erase (e i), Real.exp (x t) :=
    (add_sum_erase _ _ (mem_univ _)).symm
  have hpos : 0 < ∑ t, Real.exp (x t) := sum_pos (fun t _ => Real.exp_pos _) ⟨e i, mem_univ _⟩
  have h2 : 1 ≤ 2 * (∑ t ∈ univ.erase (e i), Real.exp (x t)) / ∑ t, Real.exp (x t) := by
    rw [le_div_iff₀ hpos]
    linarith
  change softmaxAttention x _ ≤ 0
  linarith

/-- Three needles one token apart, the score the distance itself, at `s = 1`. -/
example := exists_spaced_logScaleAttention_indicator_nonpos (k := 3) (fun d => (d : ℝ))
  zero_le_one 0 1 1 1 (by decide) (by decide)

end Transformer.MRCR

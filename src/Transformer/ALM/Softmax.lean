/-
# Softmax approximates hardmax exponentially well

`Transformer.ALM.Basic` shows that the paraboloid head is an exact argmax.
A real head uses softmax at inverse temperature `β`, and this file bounds the
resulting error.  Nothing here mentions `score`: the statements are about an
arbitrary score vector with a gap, so they sit at the shallowest possible
import level.

Two bounds, in increasing sharpness:

* `softmax_winner_ge` — the published form, `w i₀ ≥ 1 - (n-1)·exp(-β δ)`.
* `softmax_winner_sharp` — the reciprocal form `w i₀ ≥ 1/(1+c)` that the
  proof of the former actually establishes before weakening it.

The weakening `1/(1+c) ≥ 1-c` is not harmless: `1 - (n-1)e^{-βδ}` is
*vacuous* (negative) as soon as `β δ ≤ log (n-1)`, whereas `1/(1+c)` always
lies in `(0,1)`.  Removing the remaining factor `n` is the business of
`Transformer.ALM.Lattice`.
-/

import Transformer.Basic
import Mathlib.Analysis.SpecialFunctions.Exp

open scoped BigOperators
open Real

namespace Transformer
namespace ALM

/-! ### The published bound -/

/-- **Exponentially small approximation error.**

If the score of `i₀` exceeds every other score by at least `δ > 0`, then the
softmax weight at inverse temperature `β` satisfies

  `w i₀ ≥ 1 - (n - 1) · exp(-β δ)`.

Together with `score_gap` and `one_le_dist_sq_of_int` (which give `δ ≥ 1`
for integer keys), this is the precise form of the claim that a standard
softmax head realizes exact lookup up to error `O(n · e^{-β})`. -/
theorem softmax_winner_ge {n : ℕ} (β : ℝ) (hβ : 0 ≤ β)
    (s : Fin n → ℝ) (i₀ : Fin n) (δ : ℝ)
    (hgap : ∀ j, j ≠ i₀ → s j + δ ≤ s i₀) :
    1 - ((n : ℝ) - 1) * Real.exp (-(β * δ))
      ≤ Real.exp (β * s i₀) / ∑ j, Real.exp (β * s j) := by
  set A := Real.exp (β * s i₀) with hA
  have hApos : 0 < A := Real.exp_pos _
  set c := ((n : ℝ) - 1) * Real.exp (-(β * δ)) with hc
  have hcnn : 0 ≤ c := by
    have hn : (0 : ℝ) ≤ (n : ℝ) - 1 := by
      rcases Nat.eq_zero_or_pos n with h | h
      · exfalso; exact absurd i₀.isLt (by simp [h])
      · have : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast h
        linarith
    exact mul_nonneg hn (le_of_lt (Real.exp_pos _))
  -- every non-winning term is at most `A * exp(-(β δ))`
  have hterm : ∀ j ∈ Finset.univ.erase i₀,
      Real.exp (β * s j) ≤ A * Real.exp (-(β * δ)) := by
    intro j hj
    have hjne : j ≠ i₀ := Finset.ne_of_mem_erase hj
    have h1 : β * s j ≤ β * s i₀ - β * δ := by
      have := hgap j hjne
      nlinarith [hβ]
    calc Real.exp (β * s j) ≤ Real.exp (β * s i₀ - β * δ) := Real.exp_le_exp.mpr h1
      _ = A * Real.exp (-(β * δ)) := by rw [hA, ← Real.exp_add]; ring_nf
  -- hence the denominator is at most `A (1 + c)`
  have hcard : (Finset.univ.erase i₀).card = n - 1 := by
    rw [Finset.card_erase_of_mem (Finset.mem_univ i₀), Finset.card_univ,
      Fintype.card_fin]
  have hsum : ∑ j, Real.exp (β * s j) ≤ A * (1 + c) := by
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ i₀)]
    have := Finset.sum_le_card_nsmul _ _ (A * Real.exp (-(β * δ))) hterm
    rw [hcard, nsmul_eq_mul] at this
    have hn1 : ((n - 1 : ℕ) : ℝ) = (n : ℝ) - 1 := by
      have : 1 ≤ n := Nat.one_le_iff_ne_zero.mpr (by
        rintro rfl; exact absurd i₀.isLt (by simp))
      push_cast [Nat.cast_sub this]
      ring
    rw [hn1] at this
    calc A + ∑ j ∈ Finset.univ.erase i₀, Real.exp (β * s j)
        ≤ A + ((n : ℝ) - 1) * (A * Real.exp (-(β * δ))) := by linarith
      _ = A * (1 + c) := by rw [hc]; ring
  have hden_pos : 0 < ∑ j, Real.exp (β * s j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨i₀, Finset.mem_univ i₀⟩
  -- `A / denom ≥ 1/(1+c) ≥ 1 - c`
  have hstep : A / (A * (1 + c)) ≤ A / ∑ j, Real.exp (β * s j) :=
    div_le_div_of_nonneg_left (le_of_lt hApos) hden_pos hsum
  have h1c : (0 : ℝ) < 1 + c := by linarith
  have hAne : A ≠ 0 := ne_of_gt hApos
  have h1cne : (1 : ℝ) + c ≠ 0 := ne_of_gt h1c
  have hAc : A / (A * (1 + c)) = 1 / (1 + c) := by
    field_simp
  have hfinal : 1 - c ≤ 1 / (1 + c) := by
    rw [le_div_iff₀ (by linarith)]
    nlinarith [sq_nonneg c]
  calc 1 - c ≤ 1 / (1 + c) := hfinal
    _ = A / (A * (1 + c)) := hAc.symm
    _ ≤ A / ∑ j, Real.exp (β * s j) := hstep

/-! ### The sharp reciprocal bound -/

/-- **Sharp softmax bound.**  If the competitors contribute at most `c` to the
normalized partition function, the winner's softmax weight is at least
`1/(1+c)`.

This is what the proof of `softmax_winner_ge` actually shows; that theorem
then discards the difference by `1/(1+c) ≥ 1-c`. -/
theorem softmax_winner_sharp {n : ℕ} (β : ℝ) (s : Fin n → ℝ) (i₀ : Fin n) (c : ℝ)
    (hc : ∑ j ∈ Finset.univ.erase i₀, Real.exp (β * (s j - s i₀)) ≤ c) :
    1 / (1 + c) ≤ Real.exp (β * s i₀) / ∑ j, Real.exp (β * s j) := by
  set A := Real.exp (β * s i₀) with hA
  have hApos : 0 < A := Real.exp_pos _
  set c' := ∑ j ∈ Finset.univ.erase i₀, Real.exp (β * (s j - s i₀)) with hc'
  have hc'nn : 0 ≤ c' := Finset.sum_nonneg fun j _ => le_of_lt (Real.exp_pos _)
  have hden : ∑ j, Real.exp (β * s j) = A * (1 + c') := by
    rw [← Finset.add_sum_erase _ (fun j => Real.exp (β * s j)) (Finset.mem_univ i₀),
      mul_add, mul_one, hc', Finset.mul_sum]
    congr 1
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [hA, ← Real.exp_add]
    congr 1
    ring
  have h1c' : (0 : ℝ) < 1 + c' := by linarith
  have hAc : A / (A * (1 + c')) = 1 / (1 + c') := by
    field_simp
  rw [hden, hAc]
  exact one_div_le_one_div_of_le h1c' (by linarith)

/-- The published estimate is a corollary: `1 - c ≤ 1/(1+c)`. -/
theorem softmax_winner_ge' {n : ℕ} (β : ℝ) (s : Fin n → ℝ) (i₀ : Fin n) (c : ℝ)
    (hcnn : 0 ≤ c)
    (hc : ∑ j ∈ Finset.univ.erase i₀, Real.exp (β * (s j - s i₀)) ≤ c) :
    1 - c ≤ Real.exp (β * s i₀) / ∑ j, Real.exp (β * s j) := by
  refine le_trans ?_ (softmax_winner_sharp β s i₀ c hc)
  rw [le_div_iff₀ (by linarith)]
  nlinarith [sq_nonneg c]

end ALM
end Transformer

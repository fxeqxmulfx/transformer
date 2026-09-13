/-
# What the head returns, not just where its mass sits

Every softmax bound in this development — `softmax_winner_ge`,
`softmax_winner_lengthfree`, `softmax_winner_int_sharp_one`,
`softmax_at_index_ge` — bounds one *weight* from below.  A head does not
return a weight.  It returns

    out = ∑ⱼ wⱼ · Vⱼ,

the convex combination of the stored values, and the claim the machine rests
on is that this vector is the value of the key the index names.  That step was
never taken: concentration of the weights was proved and concentration of the
output asserted.

`dist_weighted_sum_le` takes it, and takes it for an arbitrary convex
combination in an arbitrary normed space — nothing about softmax, scores or
the lattice enters.  If the weights are nonnegative, sum to one, and put mass
`w i₀` on one index, the output sits within `(1 - w i₀) · C` of `V i₀`, where
`C` bounds the spread of the values around `V i₀`.  That is the exact
statement: the weight bound *is* the output bound, with the diameter as the
only extra constant.

The rest is instantiation.  `softmax_output_close` supplies the softmax
weights, and `head_output_at_index` composes it with
`Transformer.ALM.SoftmaxIndex`: on the lookup path the running head's output
differs from the value of the index's own answer by at most
`(n-1)·e^{-β}·C`, and by `head_output_at_index_untied` the same holds at any
integer query the index does not tie at.

Source of the head: `transformer_vm/attention/hull2d_cht.h`, lines 203-215
(the lookup) and 70-83 (`resolve`, which aggregates the values).
-/

import Transformer.ALM.SoftmaxIndex

open scoped BigOperators

namespace Transformer
namespace ALM

variable {m n : ℕ}

/-! ### A convex combination concentrated on one index -/

/-- **Concentration of the weights is concentration of the output.**  For any
convex combination in a normed space, the distance from the output to `V i₀`
is at most the mass off `i₀` times the spread of the values around `V i₀`.
No hypothesis relates the weights to any score. -/
theorem dist_weighted_sum_le {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (w : Fin n → ℝ) (hw0 : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1)
    (V : Fin n → E) (i₀ : Fin n) (C : ℝ) (hC : ∀ j, ‖V j - V i₀‖ ≤ C) :
    ‖(∑ j, w j • V j) - V i₀‖ ≤ (1 - w i₀) * C := by
  have hsplit := Finset.sum_erase_add Finset.univ w (Finset.mem_univ i₀)
  have hrw : (∑ j, w j • V j) - V i₀ = ∑ j, w j • (V j - V i₀) := by
    simp only [smul_sub]
    rw [Finset.sum_sub_distrib, ← Finset.sum_smul, hsum, one_smul]
  rw [hrw]
  calc ‖∑ j, w j • (V j - V i₀)‖
      ≤ ∑ j, ‖w j • (V j - V i₀)‖ := norm_sum_le _ _
    _ = ∑ j, w j * ‖V j - V i₀‖ := by
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (hw0 j)]
    _ = ∑ j ∈ Finset.univ.erase i₀, w j * ‖V j - V i₀‖ := by
        rw [← Finset.sum_erase_add _ _ (Finset.mem_univ i₀)]
        simp
    _ ≤ ∑ j ∈ Finset.univ.erase i₀, w j * C :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hC j) (hw0 j)
    _ = (∑ j ∈ Finset.univ.erase i₀, w j) * C := by rw [Finset.sum_mul]
    _ = (1 - w i₀) * C := by rw [show ∑ j ∈ Finset.univ.erase i₀, w j = 1 - w i₀ by linarith]

/-- The hypotheses are satisfiable: the point mass on `0` is a convex
combination, and any single value has zero spread around itself. -/
example : (∀ j : Fin 2, 0 ≤ (if j = 0 then (1 : ℝ) else 0)) ∧
    (∑ j : Fin 2, (if j = 0 then (1 : ℝ) else 0)) = 1 ∧
    ∀ j : Fin 2, ‖(fun _ : Fin 2 => (0 : ℝ)) j - (fun _ : Fin 2 => (0 : ℝ)) 0‖ ≤ 0 :=
  ⟨fun j => by positivity, by simp, fun j => by simp⟩

/-! ### The softmax weights are a convex combination -/

/-- Softmax weights are nonnegative. -/
theorem softmax_weight_nonneg (β : ℝ) (s : Fin n → ℝ) (j : Fin n) :
    0 ≤ Real.exp (β * s j) / ∑ k, Real.exp (β * s k) :=
  div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun _ _ => (Real.exp_pos _).le)

/-- And they sum to one, the denominator being a nonempty sum of exponentials. -/
theorem softmax_weight_sum [Nonempty (Fin n)] (β : ℝ) (s : Fin n → ℝ) :
    ∑ j, Real.exp (β * s j) / ∑ k, Real.exp (β * s k) = 1 := by
  have hpos : 0 < ∑ k, Real.exp (β * s k) :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
  rw [← Finset.sum_div, div_self (ne_of_gt hpos)]

/-- **The head's output, bounded by its weight bound.**  Whatever gives a lower
bound `1 - ε` on the softmax weight at `i₀` gives the bound `ε · C` on the
distance from the head's output to `V i₀`. -/
theorem softmax_output_close {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [Nonempty (Fin n)] (β : ℝ) (s : Fin n → ℝ) (V : Fin n → E) (i₀ : Fin n) (C ε : ℝ)
    (hC : ∀ j, ‖V j - V i₀‖ ≤ C)
    (hw : 1 - ε ≤ Real.exp (β * s i₀) / ∑ k, Real.exp (β * s k)) :
    ‖(∑ j, (Real.exp (β * s j) / ∑ k, Real.exp (β * s k)) • V j) - V i₀‖ ≤ ε * C := by
  have hC0 : 0 ≤ C := le_trans (by simp) (hC i₀)
  refine le_trans (dist_weighted_sum_le _ (softmax_weight_nonneg β s)
    (softmax_weight_sum β s) V i₀ C hC) ?_
  exact mul_le_mul_of_nonneg_right (by linarith) hC0

/-- The hypotheses are satisfiable: at `ε = 1` the weight bound `1 - ε ≤ w` is
free, since weights are nonnegative. -/
example (β : ℝ) (s : Fin 2 → ℝ) :
    1 - 1 ≤ Real.exp (β * s 0) / ∑ k, Real.exp (β * s k) := by
  simpa using softmax_weight_nonneg β s 0

/-! ### And so the head returns the index's value -/

/-- **The running head returns the value of the key the index names.**  On the
lookup path — the query is one of the stored keys — the head's output is within
`(n-1)·e^{-β}·C` of the value stored at the index's own answer, `C` bounding
how far the values spread from it.  This is `softmax_at_index_ge` carried from
the weight to the vector the head actually emits. -/
theorem head_output_at_index {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (I : NNIndex) [Nonempty (Fin n)] (β : ℝ) (hβ : 0 ≤ β)
    (K : Fin n → (Fin m → ℤ)) (hinj : Function.Injective K) (i₀ : Fin n)
    (V : Fin n → E) (C : ℝ) (hC : ∀ j, ‖V j - V i₀‖ ≤ C) :
    ‖(∑ j, (Real.exp (β * score (embInt (K i₀)) (embInt (K j)))
          / ∑ k, Real.exp (β * score (embInt (K i₀)) (embInt (K k)))) • V j)
        - V (I.ans (fun j => embInt (K j)) (embInt (K i₀)))‖
      ≤ ((n : ℝ) - 1) * Real.exp (-(β * 1)) * C := by
  have hinj' : Function.Injective (fun j => embInt (K j)) := by
    intro a b h
    refine hinj ?_
    funext i
    have := congrFun (congrArg (fun x : EucSpace m => (x : Fin m → ℝ)) h) i
    simp only [embInt_apply] at this
    exact_mod_cast this
  have hans := I.ans_eq_of_query_mem (fun j => embInt (K j)) hinj' i₀
  rw [hans]
  refine softmax_output_close β (fun j => score (embInt (K i₀)) (embInt (K j))) V i₀ C
    (((n : ℝ) - 1) * Real.exp (-(β * 1))) hC ?_
  have h := softmax_at_index_ge I β hβ K hinj i₀
  rwa [hans] at h

/-- **And off it, at any integer query the index does not tie at.**  The same
bound with no assumption on where the query sits, only that the index's answer
is the unique maximizer there — the complement of the tie hyperplanes of
`Transformer.ALM.TieHyperplane`. -/
theorem head_output_at_index_untied {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (I : NNIndex) [Nonempty (Fin n)] (β : ℝ) (hβ : 0 ≤ β)
    (K : Fin n → (Fin m → ℤ)) (q : Fin m → ℤ)
    (hno : ∀ j, j ≠ I.ans (fun i => embInt (K i)) (embInt q) →
      score (embInt q) (embInt (K j)) ≠
        score (embInt q) (embInt (K (I.ans (fun i => embInt (K i)) (embInt q)))))
    (V : Fin n → E) (C : ℝ)
    (hC : ∀ j, ‖V j - V (I.ans (fun i => embInt (K i)) (embInt q))‖ ≤ C) :
    ‖(∑ j, (Real.exp (β * score (embInt q) (embInt (K j)))
          / ∑ k, Real.exp (β * score (embInt q) (embInt (K k)))) • V j)
        - V (I.ans (fun i => embInt (K i)) (embInt q))‖
      ≤ ((n : ℝ) - 1) * Real.exp (-(β * 1)) * C :=
  softmax_output_close β (fun j => score (embInt q) (embInt (K j))) V _ C _ hC
    (softmax_at_index_ge_of_untied I β hβ K q hno)

/-- The hypotheses are satisfiable: two distinct scalar keys carrying the
scalar values `0` and `1`, whose spread around the first is at most `1`. -/
example : Function.Injective (fun j : Fin 2 => fun _ : Fin 1 => (j : ℤ)) ∧
    ∀ j : Fin 2, ‖(![0, 1] : Fin 2 → ℝ) j - (![0, 1] : Fin 2 → ℝ) 0‖ ≤ 1 := by
  constructor
  · intro a b h
    refine Fin.ext ?_
    have h0 : ((a : ℕ) : ℤ) = ((b : ℕ) : ℤ) := congrFun h 0
    exact_mod_cast h0
  · intro j
    fin_cases j <;> norm_num

end ALM
end Transformer

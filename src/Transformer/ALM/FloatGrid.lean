/-
# The exactness the walk needs is a property of the grid

`Transformer.ALM.FloatTie` bounds the merge test from both sides, but the two
bounds are not equally grounded.  Soundness (`fp_tie_no_false_positive`) asks
only `2δ < 1` and holds for any rounding.  Completeness
(`fp_tie_no_false_negative`) asks that the score routine return the *exact*
dot product, and the only routine in the development that does is `exactScore`,
which is the dot product.  So the completeness half was, in effect, a statement
about real arithmetic again, with the justification — "binary64 is exact on
integers below `2^53`" — left in prose.

The justification is a theorem, and it is not about binary64.  What matters is
that the routine returns a value *on the integer grid* and is accurate to
better than a unit: two integers within a unit of each other are equal, so such
a routine cannot be inexact on lattice data.  `fp_exact_of_grid` says this, and
it discharges the `hexact` hypothesis of the whole completeness chain from a
condition an implementation can be checked against.

That the condition is weaker than exactness is not a technicality:
`roundScore`, which rounds every score to the nearest integer and is wrong on
almost every real input, satisfies it — and is therefore exact where the
machine is used.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 280-283.
-/

import Transformer.ALM.FloatWalk

namespace Transformer
namespace ALM

open Classical

variable {n : ℕ}

/-! ### A grid-valued routine is exact on the grid -/

/-- **Two integers a unit apart are the same integer.**  A score routine whose
result lands on the integer grid and whose error is below one unit returns the
exact score whenever the exact score is itself an integer — no relative-error
model, no exponent bookkeeping, just the spacing of `ℤ`. -/
theorem fp_eval_exact_of_grid (S : FPScore) (hδ : S.δ < 1) (q k : ℝ × ℝ) {ze zd : ℤ}
    (hgrid : S.eval q k = (ze : ℝ)) (hdot : dot q k = (zd : ℝ)) :
    S.eval q k = dot q k := by
  have h := S.eval_err q k
  rw [hgrid, hdot] at h ⊢
  have h1 : |((ze - zd : ℤ) : ℝ)| < 1 := by
    push_cast
    exact lt_of_le_of_lt h hδ
  have hz : (ze - zd : ℤ) = 0 := Int.abs_lt_one_iff.mp (by exact_mod_cast h1)
  have : ze = zd := by omega
  rw [this]

/-- **So on lattice data exactness is free.**  Integer keys at an integer query
score integers, so a grid-valued routine accurate to better than a unit agrees
with the dot product on every key in range.  This is the hypothesis
`fp_tie_no_false_negative` and `fp_walk_collects` ask for, now supplied by a
property of the implementation rather than assumed of it. -/
theorem fp_exact_of_grid (S : FPScore) (K : ℕ → ℝ) (hK : ∀ j, ∃ z : ℤ, K j = (z : ℝ))
    (q : ℤ) (N : ℕ) (hδ : S.δ < 1)
    (hgrid : ∀ i ≤ N, ∃ z : ℤ, S.eval (liftQuery (q : ℝ)) (liftKey (K i)) = (z : ℝ)) :
    ∀ i ≤ N, S.eval (liftQuery (q : ℝ)) (liftKey (K i))
      = dot (liftQuery (q : ℝ)) (liftKey (K i)) := by
  intro i hi
  obtain ⟨ze, hze⟩ := hgrid i hi
  obtain ⟨z, hz⟩ := hK i
  refine fp_eval_exact_of_grid S hδ _ _ (zd := 2 * z * q - z ^ 2) hze ?_
  rw [dot_liftQuery, lineEval_liftKey_int hz]

/-! ### A routine that is exact nowhere else -/

/-- **Rounding to the nearest integer.**  A score routine that is wrong on
almost every real input — its error reaches half a unit — and still meets the
grid condition.  It witnesses that `fp_exact_of_grid` is about the grid and not
about exact arithmetic in disguise. -/
noncomputable def roundScore : FPScore where
  eval := fun q k => ((round (dot q k) : ℤ) : ℝ)
  δ := 1 / 2
  δ_nonneg := by norm_num
  eval_err := fun q k => by
    rw [abs_sub_comm]
    exact abs_sub_round (dot q k)

/-- It is grid-valued everywhere, and below a unit of error. -/
theorem roundScore_grid (q k : ℝ × ℝ) : ∃ z : ℤ, roundScore.eval q k = (z : ℝ) :=
  ⟨round (dot q k), rfl⟩

/-- And it is not `exactScore`: at the query `(1/2, 1)` against the key
`(1, 0)` it returns `1`, not `1/2`. -/
example : roundScore.eval (1 / 2, 1) (1, 0) ≠ exactScore.eval (1 / 2, 1) (1, 0) := by
  show ((round (dot (1 / 2, 1) ((1 : ℝ), (0 : ℝ))) : ℤ) : ℝ) ≠ dot (1 / 2, 1) (1, 0)
  rw [show dot ((1 : ℝ) / 2, (1 : ℝ)) ((1 : ℝ), (0 : ℝ)) = 1 / 2 by simp [dot]]
  norm_num [round_eq]

/-! ### The walk, completed from that condition -/

/-- **The loops collect exactly the winners, for a rounded score routine.**
`fp_walk_collects` with its exactness hypothesis replaced by the grid
condition: the merge test of `HullHalf::query` accepts every maximizer and
nothing else, assuming of the arithmetic only that breakpoints are accurate to
better than half a unit and scores land on the grid within a unit. -/
theorem fp_walk_collects_of_grid (S : FPScore) (F : FPArith) [Nonempty (Fin n)]
    (K : Fin n → ℝ) (hK : ∀ i, ∃ z : ℤ, K i = (z : ℝ)) (q : ℤ) (M : ℝ)
    (hbd : ∀ j ≤ keyCard K - 1, |sortedKey K j| ≤ M) (hu : F.u * M < 1 / 2)
    (hδ : S.δ < 1)
    (hgrid : ∀ i ≤ keyCard K - 1,
      ∃ z : ℤ, S.eval (liftQuery (q : ℝ)) (liftKey (sortedKey K i)) = (z : ℝ)) :
    fpTieSet S F K (q : ℝ) = argmaxSet (sortedKey K) (q : ℝ) (keyCard K - 1) :=
  fp_walk_collects S F K hK q M hbd hu
    (fp_exact_of_grid S (sortedKey K) (sortedKey_int K hK) q (keyCard K - 1) hδ hgrid)

/-- **And so each loop still runs at most once.**  The trichotomy of
`Transformer.ALM.HullCost` at the index the floating-point search returns, with
no exactness assumed of the score routine. -/
theorem fp_walk_trichotomy_of_grid (S : FPScore) (F : FPArith) [Nonempty (Fin n)]
    (K : Fin n → ℝ) (hK : ∀ i, ∃ z : ℤ, K i = (z : ℝ)) (q : ℤ) (M : ℝ)
    (hbd : ∀ j ≤ keyCard K - 1, |sortedKey K j| ≤ M) (hu : F.u * M < 1 / 2)
    (hδ : S.δ < 1)
    (hgrid : ∀ i ≤ keyCard K - 1,
      ∃ z : ℤ, S.eval (liftQuery (q : ℝ)) (liftKey (sortedKey K i)) = (z : ℝ)) :
    fpTieSet S F K (q : ℝ) = {fpProbe F K (q : ℝ)} ∨
      fpTieSet S F K (q : ℝ) = {fpProbe F K (q : ℝ), fpProbe F K (q : ℝ) + 1} ∨
      fpTieSet S F K (q : ℝ) = {fpProbe F K (q : ℝ) - 1, fpProbe F K (q : ℝ)} :=
  fp_walk_trichotomy S F K hK q M hbd hu
    (fp_exact_of_grid S (sortedKey K) (sortedKey_int K hK) q (keyCard K - 1) hδ hgrid)

/-- The hypotheses hold for the rounding routine, which is *not* covered by
`fp_walk_collects` directly: `roundScore` is exact on no line of the plane in
general, and its `δ = 1/2` does not even satisfy the `2δ < 1` of the soundness
half. -/
example [Nonempty (Fin n)] (K : Fin n → ℤ) (q : ℤ) :
    (∀ i, ∃ z : ℤ, ((K i : ℝ)) = (z : ℝ)) ∧ roundScore.δ < 1 ∧ ¬ (2 * roundScore.δ < 1) ∧
      (∀ i ≤ keyCard (fun i => (K i : ℝ)) - 1, ∃ z : ℤ,
        roundScore.eval (liftQuery (q : ℝ))
          (liftKey (sortedKey (fun i => (K i : ℝ)) i)) = (z : ℝ)) ∧
      ∃ M : ℝ, (∀ j ≤ keyCard (fun i => (K i : ℝ)) - 1,
          |sortedKey (fun i => (K i : ℝ)) j| ≤ M) ∧ exactArith.u * M < 1 / 2 := by
  refine ⟨fun i => ⟨K i, rfl⟩, by norm_num [roundScore], by norm_num [roundScore],
    fun i _ => roundScore_grid _ _, ?_⟩
  obtain ⟨M, hM⟩ := exists_bound_sortedKey (fun i => (K i : ℝ))
  exact ⟨M, hM, by simp [exactArith]⟩

end ALM
end Transformer

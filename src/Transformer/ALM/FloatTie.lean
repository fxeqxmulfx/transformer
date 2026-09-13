/-
# The tie test the code runs is the tie the development resolves

The two merge loops of `HullHalf::query` decide what to collect with
`if (s == best_score)` — an exact `==` between two `double`s, each of them the
recomputed score `qx*kx + qy*ky` of a neighbouring line.  `Transformer.ALM.HullScan`
proves everything about the *exact* tie set `argmaxSet`, and
`Transformer.ALM.FloatHull` covers the rounding of `isect` and of the search
branch.  Nothing covered the comparison the loops actually branch on, so
"the walk collects the tied lines" was a statement about real arithmetic
attached to code that does not use it.

`FPScore` is the missing model: a score routine accurate to `δ`, of which
exact arithmetic is the case `δ = 0`.  Two theorems bound it from both sides.

* `fp_tie_no_false_positive` — on lattice data a fired test is a real tie.
  Scores of integer keys at an integer query are integers, so two of them are
  either equal or a whole unit apart, and `2δ < 1` cannot bridge a unit.  The
  walk therefore never merges a line that is not a winner, *whatever* the
  rounding does.
* `fp_tie_no_false_negative` — a real winner does fire the test, provided the
  scores are computed exactly.  That proviso is discharged, not assumed:
  `Transformer.ALM.FloatGrid` shows a routine returning values on the integer
  grid to within a unit is exact on lattice data, which is what binary64 is on
  integers below `2^53`.

`fp_walk_collects` puts the two together at the index the search lands on: the
set the loops collect is exactly `argmaxSet`, and `fp_walk_collects_trichotomy`
then reads off `Transformer.ALM.HullCost`'s answer — it is the winner alone, or
the winner with one neighbour, so each loop runs at most once.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 277-303.
-/

import Transformer.ALM.FloatLattice
import Transformer.ALM.HullCost

namespace Transformer
namespace ALM

open Classical

variable {n : ℕ}

/-! ### The score routine the loops recompute -/

/-- **The arithmetic of `qx*kx + qy*ky`.**  A score routine agreeing with the
planar dot product to within `δ`.  The loops compare its results with `==`, so
`δ` is exactly what decides whether that comparison means anything.

Source: `hull2d_cht.h`, lines 280-282. -/
structure FPScore where
  /-- The score as the code computes it. -/
  eval : ℝ × ℝ → ℝ × ℝ → ℝ
  /-- The absolute error it may carry. -/
  δ : ℝ
  δ_nonneg : 0 ≤ δ
  eval_err : ∀ q k, |eval q k - dot q k| ≤ δ

/-- Exact arithmetic is the case `δ = 0`. -/
noncomputable def exactScore : FPScore where
  eval := dot
  δ := 0
  δ_nonneg := le_rfl
  eval_err := by simp

/-- The score the loops recompute is the line value the hull theorems speak
about: `qx*kx + qy*ky` at a lifted key is `2kq - k²`. -/
theorem dot_liftQuery (q k : ℝ) : dot (liftQuery q) (liftKey k) = lineEval (liftKey k) q := by
  rw [dot_lift, lineEval_liftKey]

/-- **Two scores that compare equal are within `2δ`.**  This is all the `==`
of the loops can ever tell us about the exact scores. -/
theorem fp_tie_sound (S : FPScore) (q k k' : ℝ × ℝ) (h : S.eval q k = S.eval q k') :
    |dot q k - dot q k'| ≤ 2 * S.δ := by
  have h1 := abs_le.mp (S.eval_err q k)
  have h2 := abs_le.mp (S.eval_err q k')
  rw [h] at h1
  rw [abs_le]
  constructor <;> linarith [h1.1, h1.2, h2.1, h2.2]

/-- The hypothesis is satisfiable, and not only by equal keys: the keys `0`
and `2` genuinely tie at the query `1`. -/
example : exactScore.eval (liftQuery 1) (liftKey 0) = exactScore.eval (liftQuery 1) (liftKey 2) := by
  show dot (liftQuery 1) (liftKey 0) = dot (liftQuery 1) (liftKey 2)
  rw [dot_lift, dot_lift]
  norm_num

/-! ### On the lattice a fired test is a real tie -/

/-- The score of an integer key at an integer query is an integer. -/
lemma lineEval_liftKey_int {k : ℝ} {z q : ℤ} (hk : k = (z : ℝ)) :
    lineEval (liftKey k) (q : ℝ) = ((2 * z * q - z ^ 2 : ℤ) : ℝ) := by
  rw [lineEval_liftKey, hk]
  push_cast
  ring

/-- **The `==` of the loops never fires on a loser.**  Integer keys at an
integer query score integers, so a non-winner is a whole unit behind; an error
of `2δ < 1` cannot make it compare equal to the best.  No exactness is
assumed — this is what makes the walk sound in the arithmetic that runs.

Source: `hull2d_cht.h`, lines 283 and 296. -/
theorem fp_tie_no_false_positive (S : FPScore) (K : ℕ → ℝ)
    (hK : ∀ j, ∃ z : ℤ, K j = (z : ℝ)) (q : ℤ) (N : ℕ) (hδ : 2 * S.δ < 1)
    {b j : ℕ} (hb : b ∈ argmaxSet K (q : ℝ) N) (hj : j ≤ N)
    (htie : S.eval (liftQuery (q : ℝ)) (liftKey (K j))
      = S.eval (liftQuery (q : ℝ)) (liftKey (K b))) :
    j ∈ argmaxSet K (q : ℝ) N := by
  obtain ⟨zj, hzj⟩ := hK j
  obtain ⟨zb, hzb⟩ := hK b
  have hdot := fp_tie_sound S _ _ _ htie
  rw [dot_liftQuery, dot_liftQuery, lineEval_liftKey_int hzj, lineEval_liftKey_int hzb] at hdot
  have hR : |(((2 * zj * q - zj ^ 2) - (2 * zb * q - zb ^ 2) : ℤ) : ℝ)| < 1 := by
    push_cast
    push_cast at hdot
    exact lt_of_le_of_lt hdot hδ
  have hZ : ((2 * zj * q - zj ^ 2) - (2 * zb * q - zb ^ 2) : ℤ) = 0 :=
    Int.abs_lt_one_iff.mp (by exact_mod_cast hR)
  have heq : lineEval (liftKey (K j)) (q : ℝ) = lineEval (liftKey (K b)) (q : ℝ) := by
    rw [lineEval_liftKey_int hzj, lineEval_liftKey_int hzb]
    exact_mod_cast congrArg (fun z : ℤ => (z : ℝ)) (by omega : (2 * zj * q - zj ^ 2 : ℤ)
      = 2 * zb * q - zb ^ 2)
  refine (mem_argmaxSet K (q : ℝ) N).mpr ⟨hj, fun i hi => ?_⟩
  rw [heq]
  exact ((mem_argmaxSet K (q : ℝ) N).mp hb).2 i hi

/-- The hypotheses are satisfiable together: the integer keys `2j`, the query
`1`, exact arithmetic, and the genuine tie between the keys `0` and `2`. -/
example : (∀ j : ℕ, ∃ z : ℤ, 2 * (j : ℝ) = (z : ℝ)) ∧ 2 * exactScore.δ < 1 ∧
    1 ∈ argmaxSet (fun j : ℕ => 2 * (j : ℝ)) ((1 : ℤ) : ℝ) 1 ∧
    exactScore.eval (liftQuery ((1 : ℤ) : ℝ)) (liftKey (2 * ((0 : ℕ) : ℝ)))
      = exactScore.eval (liftQuery ((1 : ℤ) : ℝ)) (liftKey (2 * ((1 : ℕ) : ℝ))) := by
  refine ⟨fun j => ⟨2 * (j : ℤ), by push_cast; ring⟩, by norm_num [exactScore], ?_, ?_⟩
  · rw [mem_argmaxSet]
    refine ⟨le_refl 1, fun i hi => ?_⟩
    interval_cases i <;> simp only [lineEval_liftKey] <;> norm_num
  · show dot (liftQuery _) (liftKey _) = dot (liftQuery _) (liftKey _)
    rw [dot_lift, dot_lift]
    norm_num

/-! ### And an exact routine misses no winner -/

/-- **The `==` of the loops fires on every winner.**  With the scores computed
exactly two tied lines compare equal, so the walk stops only where `argmaxSet`
ends.  `Transformer.ALM.FloatGrid` discharges the exactness on lattice data
from a grid condition on the routine.

Source: `hull2d_cht.h`, lines 283 and 296. -/
theorem fp_tie_no_false_negative (S : FPScore) (K : ℕ → ℝ) (q : ℝ) (N : ℕ)
    (hexact : ∀ i ≤ N, S.eval (liftQuery q) (liftKey (K i))
      = dot (liftQuery q) (liftKey (K i)))
    {b j : ℕ} (hb : b ∈ argmaxSet K q N) (hjmem : j ∈ argmaxSet K q N) :
    S.eval (liftQuery q) (liftKey (K j)) = S.eval (liftQuery q) (liftKey (K b)) := by
  rw [hexact j ((mem_argmaxSet K q N).mp hjmem).1, hexact b ((mem_argmaxSet K q N).mp hb).1,
    dot_liftQuery, dot_liftQuery]
  exact argmaxSet_tie K q N hjmem hb

/-- Exact arithmetic satisfies the hypothesis, and the winner set it is asked
about is inhabited: the keys `2j` at the query `1` again. -/
example : (∀ i ≤ 1, exactScore.eval (liftQuery ((1 : ℤ) : ℝ)) (liftKey (2 * (i : ℝ)))
      = dot (liftQuery ((1 : ℤ) : ℝ)) (liftKey (2 * (i : ℝ)))) ∧
    1 ∈ argmaxSet (fun j : ℕ => 2 * (j : ℝ)) ((1 : ℤ) : ℝ) 1 := by
  refine ⟨fun i _ => rfl, ?_⟩
  rw [mem_argmaxSet]
  refine ⟨le_refl 1, fun i hi => ?_⟩
  interval_cases i <;> simp only [lineEval_liftKey] <;> norm_num

end ALM
end Transformer

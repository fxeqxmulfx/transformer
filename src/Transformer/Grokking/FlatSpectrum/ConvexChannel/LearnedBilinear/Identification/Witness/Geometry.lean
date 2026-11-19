module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Duality
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Restriction

/-!
# A genuinely held-out face of the actual sequence family

Source: `review/convex/moments.md`, CM87. Observe three Boolean prefixes with
nonconstant labels and withhold the fourth corner. Its two varying context
positions are nonadjacent. The exact coefficient relation includes every bias,
position, adjacent pair, output class and complex coordinate.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Witness

open scoped BigOperators Matrix
open LearnedHinge (Prefix)
open Restriction (cube cube_query)

/-- Three indexed training sequences, CM87. The fourth face corner is absent. -/
def tokens : Fin 3 → Prefix Bool (3 + 1) :=
  ![cube false false false, cube true false false, cube false false true]

/-- Original-object training labels, CM87. They are not constant. -/
def labels : Fin 3 → Bool := ![false, true, true]

/-- A held-out input, without its answer, CM87. -/
def query : Prefix Bool (3 + 1) := cube true false true

/-- Every observed sequence differs from the query, CM87. -/
theorem query_unseen (i : Fin 3) : query ≠ tokens i := by
  fin_cases i <;> decide

/-- The observations include both original output objects, CM87. -/
theorem labels_nonconstant : labels 0 ≠ labels 1 := by decide

/-- The complete context row of the held-out corner has this affine
face relation, CM87. Nonadjacent variation is essential to the identity. -/
theorem context_face :
    contextRow (past query) = contextRow (past (tokens 1)) +
      contextRow (past (tokens 2)) - contextRow (past (tokens 0)) := by
  funext d
  have hs (f : Fin (3 - 1) → ℂ) :
      (∑ k, f k) = f ⟨0, by norm_num⟩ + f ⟨1, by norm_num⟩ := by
    change (∑ k : Fin 2, f k) = _
    exact Fin.sum_univ_two f
  fin_cases d <;> simp only [Pi.sub_apply, Pi.add_apply, contextRow, hs]
  all_goals norm_num [past, query, tokens, cube, pairAt, pairFlag]

/-- Swapping two original objects negates the margin coefficient, CM87. -/
theorem coefficient_swap (x : Prefix Bool (3 + 1)) (a b : Bool) (z : RealCoordinate Bool 3) :
    coefficient x a b z = -coefficient x b a z := by
  unfold coefficient
  ring

/-- The held-out true-versus-false coefficient is the sum of the three
observed correct-versus-incorrect coefficients, CM87. All weights are positive. -/
theorem coefficient_face (z : RealCoordinate Bool 3) :
    coefficient query true false z = coefficient (tokens 0) false true z +
      coefficient (tokens 1) true false z + coefficient (tokens 2) true false z := by
  rw [coefficient_swap (tokens 0) false true]
  have h := congrFun context_face z.2.2.2
  simp only [Pi.sub_apply, Pi.add_apply] at h
  unfold coefficient
  rw [h]
  simp [query, tokens, cube_query, mul_sub, mul_add, Complex.sub_re, Complex.add_re]
  ring

/-- The complete weight table, including zero self-comparisons, CM87. -/
def weights (i : Fin 3) (c : Bool) : ℝ := if c = labels i then 0 else 1

/-- Explicit nonnegative mass-three certificate checked on every coordinate,
CM87. The query answer is a consequence, not a checker input. -/
theorem checked_certificate : Certificate tokens labels query true false 3 weights := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i c
    unfold weights
    split <;> norm_num
  · intro i
    simp [weights]
  · norm_num [weights, labels, Fin.sum_univ_three, Fintype.sum_bool]
  · intro z
    simpa [weights, labels, Fin.sum_univ_three, Fintype.sum_bool] using coefficient_face z

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Witness

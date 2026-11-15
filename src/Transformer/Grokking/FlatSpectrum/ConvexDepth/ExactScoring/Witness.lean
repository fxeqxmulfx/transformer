module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ExactScoring.Basic

/-!
# A feasible complete head can have an exact unobserved tie

Source: review/flat_spectrum.md, CM63. Two active input programs agree after
decoding on two nonconstant observations and disagree at a third input.
Improved arithmetic does not identify which one supplies its unknown answer.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ExactScoring

open scoped BigOperators
open CoordinatePrograms

/-- Two fixed input-only quantities, with different scales at the anchors.
They cease to be proportional at the unobserved third input, CM63. -/
def tieFeatures (p : Fin 2) (x : Fin 3) : Fin 5 → ℝ :=
  ![if x = 2 then (if p = 0 then 0 else 2)
    else (if p = 0 then (x.val : ℝ) else 2 * (x.val : ℝ)), 1, 0, 0, 0]

/-- Only inputs zero and one are observed, CM63. -/
def tieRows (p : Fin 2) (a : Fin 2) : Fin 5 → ℝ := tieFeatures p a.castSucc

/-- Distinct positive-mass decoders both satisfy the copy constraint, CM63. -/
noncomputable def tieParameters (p : Fin 2 × Fin 6) : ℝ :=
  if p.1 = 0 then ![1, 0, 0, 0, 1, 1] p.2 else ![1 / 2, 0, 0, 0, 1, 1] p.2

/-- Class coordinates are a declared injective two-label geometry, CM63. -/
def tieCoordinates (c : Fin 2) : ℝ := c.val

/-- The joint domain is nonempty with sufficient weighted charge18, two distinct observed
labels and two active programs; no unseen equation is imposed, CM63. -/
theorem tie_feasible :
    tieParameters ∈ domain tieRows (fun _ a => (a.val : ℝ)) (fun _ => 16) 32 1 := by
  norm_num [domain, ParameterSharing.weightedDomain, Fintype.sum_prod_type,
    Fin.sum_univ_succ, Fin.forall_fin_succ, costs, charge, center,
    tieRows, tieFeatures, tieParameters]

/-- Actual complete products choose zero at the first anchor and one at the
second, so the witness does not fit a constant target, CM63. -/
theorem tie_observed_nonconstant :
    headLogits (features tieFeatures) (head tieCoordinates tieParameters) 0 1 <
        headLogits (features tieFeatures) (head tieCoordinates tieParameters) 0 0 ∧
    headLogits (features tieFeatures) (head tieCoordinates tieParameters) 1 0 <
        headLogits (features tieFeatures) (head tieCoordinates tieParameters) 1 1 := by
  simp only [head_logits]
  norm_num [score, center, tieCoordinates, tieParameters, tieFeatures,
    SignedContext.normalizer, SignedContext.mass, bankRow, row,
    Fintype.sum_prod_type, Fin.sum_univ_succ]

/-- At the third input the two complete raw logits are exactly tied despite
nonzero masses and distinct observed labels. Rounding cannot certify a target,
CM63. Both source Fourier blocks are retained in this statement. -/
theorem tie_unobserved :
    headLogits (features tieFeatures) (head tieCoordinates tieParameters) 2 0 =
      headLogits (features tieFeatures) (head tieCoordinates tieParameters) 2 1 := by
  simp only [head_logits]
  norm_num [score, center, tieCoordinates, tieParameters, tieFeatures, Fin.sum_univ_succ]

/-- The tied input actually separates the two scaled decoders, CM63. The
normalizer is positive and input-only, not a disappearing coefficient. -/
theorem tie_distinct_programs :
    center (fun j => tieParameters (0, j)) (tieFeatures 0 2) = 0 ∧
      center (fun j => tieParameters (1, j)) (tieFeatures 1 2) = 1 ∧
      tieParameters (0, 5) = 1 ∧ tieParameters (1, 5) = 1 ∧
      SignedContext.normalizer (bankRow tieFeatures 2) = 6 := by
  norm_num [center, tieFeatures, tieParameters, SignedContext.normalizer, SignedContext.mass,
    bankRow, row, Fintype.sum_prod_type, Fin.sum_univ_succ]

/-- Each of the two programs supplies a different possible binary target,
CM63. This is a witness family, not an input to a learned predictor. -/
def tieTarget (p : Fin 2) (x : Fin 3) : Fin 2 :=
  if x = 2 then p else if x = 1 then 1 else 0

/-- Both target completions agree with every nonconstant observed label,
CM63. The third input remains outside the observations. -/
theorem tie_target_anchors (p a : Fin 2) : tieTarget p a.castSucc = a := by
  fin_cases p <;> fin_cases a <;> norm_num [tieTarget]

/-- Each possible target is actually realized by its program's scaled
decoder on every input, CM63; the ambiguity is not a fabricated label table. -/
theorem tie_target_realizable (p : Fin 2) (x : Fin 3) :
    center (fun j => tieParameters (p, j)) (tieFeatures p x) =
      tieCoordinates (tieTarget p x) := by
  fin_cases p <;> fin_cases x <;>
    norm_num [center, tieParameters, tieFeatures, tieCoordinates, tieTarget,
      Fin.sum_univ_succ]

/-- Realizability by some retained program plus the exact same observed
answers admits opposite unobserved targets, CM63. Numerical consistency
alone cannot select which completion is correct. -/
theorem tie_two_targets :
    (∀ a : Fin 2, tieTarget 0 a.castSucc = tieTarget 1 a.castSucc) ∧
      tieTarget 0 2 ≠ tieTarget 1 2 ∧
      (∀ x, center (fun j => tieParameters (0, j)) (tieFeatures 0 x) =
        tieCoordinates (tieTarget 0 x)) ∧
      (∀ x, center (fun j => tieParameters (1, j)) (tieFeatures 1 x) =
        tieCoordinates (tieTarget 1 x)) := by
  refine ⟨?_, ?_, tie_target_realizable 0, tie_target_realizable 1⟩
  · intro a
    rw [tie_target_anchors, tie_target_anchors]
  · norm_num [tieTarget]

/-- Every expanded real/imaginary column is covered by the original budget,
CM63. The example is feasible for the actual complete tensor. -/
theorem tie_complete_budget : head tieCoordinates tieParameters ∈ headBudget 32 := by
  apply head_budget tieCoordinates _ tie_feasible.1
  intro c
  fin_cases c <;> norm_num [tieCoordinates]

/-- A complete coefficient tensor with nonzero imaginary entries in two
distinct raw columns, for checking the independent contraction identity, CM63. -/
def complexWitness (k : Fin 4) (c : Fin 2) : ℂ :=
  ⟨(k.val : ℝ) + c.val, (k.val : ℝ) - c.val⟩

/-- The signed input row and full four-point code genuinely use imaginary
products; the two unnormalized scores differ, CM63. -/
theorem complex_witness_numerators :
    numerator (![2, -1, 0, 1] : Fin 4 → ℝ) (realSource fourier4)
        (complexCoefficients complexWitness) 0 = 4 ∧
      numerator (![2, -1, 0, 1] : Fin 4 → ℝ) (realSource fourier4)
        (complexCoefficients complexWitness) 1 = 8 := by
  norm_num [numerator, realSource, fourier4, complexCoefficients, complexWitness,
    Fintype.sum_prod_type, Fintype.sum_bool, Fin.sum_univ_succ]

/-- An arbitrarily small absolute product residual permits opposite strict
winners, CM63. Consequently a residual threshold alone cannot replace exact
ordering or a valid separating-interval certificate. -/
theorem small_residual_opposite_winners (ε : ℝ) (hε : 0 < ε) :
    (∀ c : Fin 2, |(![ε, 0] : Fin 2 → ℝ) c - (![0, ε] : Fin 2 → ℝ) c| ≤ ε) ∧
      (![ε, 0] : Fin 2 → ℝ) 1 < (![ε, 0] : Fin 2 → ℝ) 0 ∧
      (![0, ε] : Fin 2 → ℝ) 0 < (![0, ε] : Fin 2 → ℝ) 1 := by
  constructor
  · intro c
    fin_cases c <;> simp [abs_of_pos hε]
  · constructor <;> simpa using hε

/-- The residual obstruction has a positive, finite satisfying instance,
CM63; neither vector is constant and neither comparison is a tie. -/
example := small_residual_opposite_winners (1 / 1024) (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ExactScoring

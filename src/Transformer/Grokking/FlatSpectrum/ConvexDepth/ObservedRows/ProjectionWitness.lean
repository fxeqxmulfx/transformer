module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows.Projection
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows.Witness

/-!
# An off-domain proposal with different anchor counts and unknown rows

Source: `review/convex/moments.md`, CM66. The proposal violates nonnegativity,
contains nonzero off-label entries, and has unequal unknown class coordinates.
The exact reduced target retains all these contributions to the objective.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows

open scoped BigOperators
open LatentPrograms (Parameters witnessProgram witnessSamples)

/-- A full, deliberately infeasible projection target, CM66. It is a
numerical proposal, not a supplied true decoder or additional training label. -/
def witnessProposal : Parameters Bool (Fin 4) Bool :=
  (fun j => if j then 2 else -1,
    fun j s c => if j then (if c then (s.val : ℝ) + 1 else 2 * s.val - 1)
      else if c then 3 else -2)

/-- The projection identity genuinely permits proposals outside the
original domain, CM66. This proposal has a negative first program mass. -/
theorem witness_proposal_outside :
    witnessProposal ∉ BoundedLatent.domain witnessProgram [false, true] witnessSamples 1 8 := by
  intro h
  have hm := h.1.1 false
  norm_num [witnessProposal] at hm

/-- The original two programs have different induced mass weights,
CM66, because their observed dictionaries cover one and three states. -/
theorem witness_projection_weights :
    massWeight witnessLabels false = 2 ∧ massWeight witnessLabels true = 4 := by
  have hf : Fintype.card (Known witnessLabels false) = 1 := by decide
  have ht : Fintype.card (Known witnessLabels true) = 3 := by decide
  constructor <;> unfold massWeight
  · rw [hf]; norm_num
  · rw [ht]; norm_num

/-- Elimination uses the full proposal's selected coordinates, CM66:
its reduced centers are -3/2 and 5/4, not the original masses -1 and 2. -/
theorem witness_projection_centers :
    (projectionTarget witnessLabels witnessProposal).1 false = -(3 / 2 : ℝ) ∧
    (projectionTarget witnessLabels witnessProposal).1 true = (5 / 4 : ℝ) := by
  have hs (j : Bool) :
      (∑ s : Known witnessLabels j, anchorProposal witnessLabels witnessProposal j s.1) =
        if j then (3 : ℝ) else -2 := by
    rw [← Finset.sum_subtype (Finset.univ.filter (fun s => witnessLabels j s ≠ none))
      (by intro s; simp)]
    rw [Finset.sum_filter]
    cases j <;> norm_num [Fin.sum_univ_four, witnessLabels, dictionary,
      LatentPrograms.lookup, witnessProgram, witnessSamples, anchorProposal, witnessProposal]
  dsimp only [projectionTarget]
  simp only [massLinear]
  rw [hs false, hs true, witness_projection_weights.1, witness_projection_weights.2]
  norm_num [witnessProposal]

/-- Unknown proposal coordinates are copied without averaging or choosing
a raw completion, CM66. This retained row is nonuniform and outside its simplex. -/
theorem witness_projection_unknown :
    (projectionTarget witnessLabels witnessProposal).2 true ⟨3, by decide⟩ false = 5 ∧
    (projectionTarget witnessLabels witnessProposal).2 true ⟨3, by decide⟩ true = 4 := by
  norm_num [projectionTarget, witnessProposal]

/-- The complete squared objective at a feasible nonconstant decoder is
123 for this off-domain proposal, CM66. It includes all off-label coordinates. -/
theorem witness_projection_distance :
    fullSquaredDistance (expand witnessLabels (witnessPoint 4 false)) witnessProposal = 123 := by
  rw [witness_expand 4 false (by norm_num) (by norm_num)]
  norm_num [fullSquaredDistance, BoundedLatent.witnessPoint,
    LatentPrograms.witnessPoint, LatentPrograms.vertex, LatentPrograms.ofDecoders,
    LatentPrograms.completion, LatentPrograms.lookup, witnessProgram, witnessSamples,
    witnessProposal, Fintype.sum_bool, Fin.sum_univ_four]

example := massWeight_pos witnessLabels true
example := known_proposal_distance witnessLabels (witnessPoint 4 false) witnessProposal
  true ⟨2, by decide⟩
example := known_proposal_sum witnessLabels (witnessPoint 4 false) witnessProposal true
example := projection_distance witnessLabels witnessProposal (witnessPoint 4 false)
example := projection_comparison witnessLabels witnessProposal (witnessPoint 2 true)
  (witnessPoint 4 false)
example := projection_minimizer_iff witnessProgram [false, true] witnessSamples 1 8
  witnessProposal (witnessPoint 4 false)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows

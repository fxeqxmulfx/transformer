module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.FamilyExtension.Complex
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent.Witness

/-!
# A nonconstant complex model survives a strict program-family extension

Source: `review/convex/moments.md`, CM67. Two additional nonconstant input
programs are present. The old covered logits, imaginary coordinates and
opposite feasible unobserved answers all survive the same-budget embedding.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.FamilyExtension

open LatentPrograms (features head witnessProgram witnessSamples)

/-- Two fixed additional input transformations, CM67, without raw labels. -/
def witnessAdded (j : Bool) (x : Fin 4 × Fin 4) : Fin 4 :=
  if j then x.1 + x.2 else x.1

/-- Both added programs change with their inputs, CM67. -/
theorem witness_added_nonconstant :
    witnessAdded true (0, 0) ≠ witnessAdded true (0, 1) ∧
      witnessAdded false (0, 1) ≠ witnessAdded false (1, 1) := by decide

/-- Both opposite unknown completions remain feasible for the expanded
family at every admitted confidence, CM67. New variables remain in the domain. -/
theorem witness_mem (ρ : ℝ) (default : Bool) (hρ : 1 ≤ ρ) (hR : ρ ≤ 8) :
    embed (BoundedLatent.witnessPoint ρ default) ∈
      BoundedLatent.domain (programs witnessProgram witnessAdded)
        (order [false, true] [false, true]) witnessSamples 1 8 := by
  exact embed_mem witnessProgram witnessAdded [false, true] [false, true]
    witnessSamples (BoundedLatent.witnessPoint_mem ρ default hρ hR)

/-- Family widening alone still permits opposite strict raw answers at
an unobserved state, CM67. All four scores are actual complete complex products. -/
theorem witness_unseen_opposite :
    headLogits (features fourier4 (programs witnessProgram witnessAdded))
      (head fourier4 1 (embed (BoundedLatent.witnessPoint 4 false))) (0, 3) false = 4 ∧
    headLogits (features fourier4 (programs witnessProgram witnessAdded))
      (head fourier4 1 (embed (BoundedLatent.witnessPoint 4 false))) (0, 3) true = 0 ∧
    headLogits (features fourier4 (programs witnessProgram witnessAdded))
      (head fourier4 1 (embed (BoundedLatent.witnessPoint 4 true))) (0, 3) false = 0 ∧
    headLogits (features fourier4 (programs witnessProgram witnessAdded))
      (head fourier4 1 (embed (BoundedLatent.witnessPoint 4 true))) (0, 3) true = 4 := by
  simpa only [logits_embed fourier4 fourier4_unitary] using BoundedLatent.witness_unseen_opposite

/-- A nonzero imaginary feature and head entry survive in the old branch,
CM67. This embedding retains a genuinely complex nonconstant readout. -/
theorem witness_imaginary_coordinates :
    features fourier4 (programs witnessProgram witnessAdded)
      (Sum.inl true, (1, true)) (3, 1) = -(1 / 2 : ℝ) ∧
    head fourier4 1 (embed (K := Bool) (BoundedLatent.witnessPoint 4 true))
      (Sum.inl true, (1, true)) false = -2 := by
  change features fourier4 witnessProgram (true, (1, true)) (3, 1) = -(1 / 2 : ℝ) ∧
    head fourier4 1 (BoundedLatent.witnessPoint 4 true) (true, (1, true)) false = -2
  exact BoundedLatent.witness_imaginary_coordinates

example := witness_mem 4 true (by norm_num) (by norm_num)
example := embed_mem witnessProgram witnessAdded [false, true] [false, true]
  witnessSamples (BoundedLatent.witnessPoint_mem 4 true (by norm_num) (by norm_num))
example := retained_inl_iff witnessProgram witnessAdded [false, true] [false, true]
  witnessSamples true
example := retained_inr_iff witnessProgram witnessAdded [false, true] [false, true]
  witnessSamples false
example := logits_embed fourier4 fourier4_unitary witnessProgram witnessAdded
  (BoundedLatent.witnessPoint 4 true) (3, 2)
example := head_budget_embed fourier4 fourier4_unitary witnessProgram witnessAdded
  [false, true] [false, true] witnessSamples
  (BoundedLatent.witnessPoint_mem 4 true (by norm_num) (by norm_num))

-- Duplicate record values keep their own IDs, and the singleton is an actual tail.
example := batchLoss_embed fourier4 fourier4_unitary witnessProgram witnessAdded
  (fun i : Fin 4 => if i < 2 then (0, 1) else (0, 2)) (fun i => decide (2 ≤ i.val))
  Finset.univ (BoundedLatent.witnessPoint 4 true)
example := batchLoss_embed fourier4 fourier4_unitary witnessProgram witnessAdded
  (fun i : Fin 4 => if i < 2 then (0, 1) else (0, 2)) (fun i => decide (2 ≤ i.val))
  {3} (BoundedLatent.witnessPoint 4 true)
example := attainable_loss fourier4 fourier4_unitary witnessProgram witnessAdded
  [false, true] [false, true] witnessSamples
  (BoundedLatent.witnessPoint_mem 4 true (by norm_num) (by norm_num))
  (fun i : Fin 4 => if i < 2 then (0, 1) else (0, 2)) (fun i => decide (2 ≤ i.val))
  Finset.univ

end Transformer.Grokking.FlatSpectrum.ConvexDepth.FamilyExtension

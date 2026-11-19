module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows.Head

/-!
# Nonconstant observed rows with opposite unobserved completions

Source: `review/convex/moments.md`, CM65. The CM53/64 witness learns a many-to-one
dictionary. Its unknown row remains a variable after the exact reduction, and
the complete and unweighted reduced Euclidean metrics demonstrably differ.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows

open scoped BigOperators
open LatentPrograms (witnessProgram witnessSamples features head)

/-- The actual nonconstant training dictionary used in CM65's examples. -/
abbrev witnessLabels := dictionary witnessProgram witnessSamples

/-- Reduced coordinates of either feasible unknown completion, CM65. -/
def witnessPoint (ρ : ℝ) (default : Bool) : Reduced witnessLabels :=
  restrict witnessLabels (BoundedLatent.witnessPoint ρ default)

/-- Both unknown completions inhabit the reduced domain over the complete
confidence interval, CM65. The decoder has two states sharing a recorded label. -/
theorem witnessPoint_mem (ρ : ℝ) (default : Bool) (hρ : 1 ≤ ρ) (hR : ρ ≤ 8) :
    witnessPoint ρ default ∈ domain witnessLabels
      (LatentPrograms.retained witnessProgram [false, true] witnessSamples) 1 8 := by
  exact restrict_mem witnessProgram [false, true] witnessSamples
    (BoundedLatent.witnessPoint_mem ρ default hρ hR)

/-- Every coefficient of the old feasible tensor is recovered, CM65. -/
theorem witness_expand (ρ : ℝ) (default : Bool) (hρ : 1 ≤ ρ) (hR : ρ ≤ 8) :
    expand witnessLabels (witnessPoint ρ default) = BoundedLatent.witnessPoint ρ default := by
  exact expand_restrict witnessProgram [false, true] witnessSamples
    (BoundedLatent.witnessPoint_mem ρ default hρ hR)

/-- The unknown raw row is still free in the reduced representation,
CM65, rather than silently replaced by a learned or prescribed answer. -/
theorem witness_free_row :
    (witnessPoint 4 false).2 true ⟨3, by decide⟩ false = 4 ∧
    (witnessPoint 4 false).2 true ⟨3, by decide⟩ true = 0 ∧
    (witnessPoint 4 true).2 true ⟨3, by decide⟩ false = 0 ∧
    (witnessPoint 4 true).2 true ⟨3, by decide⟩ true = 4 := by
  norm_num [witnessPoint, restrict, BoundedLatent.witnessPoint,
    LatentPrograms.witnessPoint, LatentPrograms.vertex, LatentPrograms.ofDecoders,
    LatentPrograms.completion, LatentPrograms.lookup, witnessSamples, witnessProgram]

/-- The two reduced feasible models keep opposite actual complete complex
margins at an unseen state, CM65. Memory reduction adds no identifying premise. -/
theorem witness_unseen_opposite :
    headLogits (features fourier4 witnessProgram)
      (head fourier4 1 (expand witnessLabels (witnessPoint 4 false))) (0, 3) false = 4 ∧
    headLogits (features fourier4 witnessProgram)
      (head fourier4 1 (expand witnessLabels (witnessPoint 4 false))) (0, 3) true = 0 ∧
    headLogits (features fourier4 witnessProgram)
      (head fourier4 1 (expand witnessLabels (witnessPoint 4 true))) (0, 3) false = 0 ∧
    headLogits (features fourier4 witnessProgram)
      (head fourier4 1 (expand witnessLabels (witnessPoint 4 true))) (0, 3) true = 4 := by
  rw [witness_expand 4 false (by norm_num) (by norm_num),
    witness_expand 4 true (by norm_num) (by norm_num)]
  exact BoundedLatent.witness_unseen_opposite

/-- Removing three observed rows changes unweighted Euclidean distances:
the original squared distance is20, while the stored-coordinate distance is8,
CM65. An optimizer claiming the old geometry must retain the missing mass weights. -/
theorem witness_metric_difference :
    fullSquaredDistance (expand witnessLabels (witnessPoint 2 false))
      (expand witnessLabels (witnessPoint 4 false)) = 20 ∧
    reducedSquaredDistance witnessLabels (witnessPoint 2 false) (witnessPoint 4 false) = 8 := by
  have hf : fullSquaredDistance (expand witnessLabels (witnessPoint 2 false))
      (expand witnessLabels (witnessPoint 4 false)) = 20 := by
    rw [witness_expand 2 false (by norm_num) (by norm_num),
      witness_expand 4 false (by norm_num) (by norm_num)]
    norm_num [fullSquaredDistance, BoundedLatent.witnessPoint,
      LatentPrograms.witnessPoint, LatentPrograms.vertex, LatentPrograms.ofDecoders,
      LatentPrograms.completion, LatentPrograms.lookup, witnessSamples, witnessProgram,
      Fintype.sum_bool, Fin.sum_univ_four]
  refine ⟨hf, ?_⟩
  have hk : Fintype.card (Known witnessLabels true) = 3 := by decide
  have hd := distance_difference witnessLabels (witnessPoint 2 false) (witnessPoint 4 false)
  have hm : (∑ j, (Fintype.card (Known witnessLabels j) : ℝ) *
      ((witnessPoint 2 false).1 j - (witnessPoint 4 false).1 j) ^ 2) = 12 := by
    rw [Fintype.sum_bool, hk]
    norm_num [witnessPoint, restrict, BoundedLatent.witnessPoint,
      LatentPrograms.witnessPoint, LatentPrograms.vertex, LatentPrograms.ofDecoders]
  rw [hf, hm] at hd
  linarith

example := witnessPoint_mem 4 true (by norm_num) (by norm_num)
example := witness_expand 4 true (by norm_num) (by norm_num)
example := expand_unknown witnessLabels (witnessPoint 4 true) true ⟨3, by decide⟩ false
example := expand_known witnessLabels (witnessPoint 4 true) true 2 true (by decide) false
example := restrict_expand witnessLabels (witnessPoint 4 false)
example := expand_combination witnessLabels (witnessPoint 2 true) (witnessPoint 4 false) 2 3
example := domain_convex witnessLabels [true] 1 8
example := restrict_mem witnessProgram [false, true] witnessSamples
  (BoundedLatent.witnessPoint_mem 4 true (by norm_num) (by norm_num))
example := expand_restrict witnessProgram [false, true] witnessSamples
  (BoundedLatent.witnessPoint_mem 4 true (by norm_num) (by norm_num))
example := expand_mem witnessProgram [false, true] witnessSamples
  (witnessPoint_mem 4 true (by norm_num) (by norm_num))
example := expand_mem_iff witnessProgram [false, true] witnessSamples 1 8 (witnessPoint 4 true)
example := expanded_domain witnessProgram [false, true] witnessSamples 1 8
example := known_row_distance witnessLabels (witnessPoint 2 false) (witnessPoint 4 false)
  true ⟨2, by decide⟩
example := decoder_distance witnessLabels (witnessPoint 2 false) (witnessPoint 4 false) true
example := full_distance_reduced witnessLabels (witnessPoint 2 false) (witnessPoint 4 false)
example := distance_difference witnessLabels (witnessPoint 2 false) (witnessPoint 4 false)
example := complete_head_eq fourier4 witnessProgram [false, true] witnessSamples
  (BoundedLatent.witnessPoint_mem 4 true (by norm_num) (by norm_num))
example := complete_budget fourier4 fourier4_unitary witnessProgram [false, true] witnessSamples
  (witnessPoint_mem 4 true (by norm_num) (by norm_num))
example := ce_convex fourier4 fourier4_unitary witnessProgram [false, true] witnessSamples 1 8
  (3, 2) true

-- Distinct record IDs keep a duplicate and an actual one-record tail.
example := batchLoss_eq fourier4 witnessProgram [false, true] witnessSamples
  (BoundedLatent.witnessPoint_mem 4 true (by norm_num) (by norm_num))
  (fun i : Fin 4 => if i < 2 then (0, 1) else (0, 2)) (fun i => decide (2 ≤ i.val)) Finset.univ
example := batchLoss_convex fourier4 fourier4_unitary witnessProgram [false, true] witnessSamples
  1 8 (fun i : Fin 4 => if i < 2 then (0, 1) else (0, 2)) (fun i => decide (2 ≤ i.val)) Finset.univ
example := batchLoss_convex fourier4 fourier4_unitary witnessProgram [false, true] witnessSamples
  1 8 (fun i : Fin 4 => if i < 2 then (0, 1) else (0, 2)) (fun i => decide (2 ≤ i.val)) {3}

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows

module

public import Transformer.Grokking.Convex.ConvexDepth.ContextInterpolation.Batch
public import Transformer.Grokking.Convex.ConvexDepth.QuotientHead

/-!
# Distinguishing raw aliases requires context information

Source: `review/convex/moments.md`, CM42. Two observed source states carry
different raw labels with the same declared readout. Fixed context weights
distinguish two unseen queries in the actual unitary complex head. A second
target agrees on both observations and all readouts but reverses the unseen
aliases, so locality is additional information and not a consequence of fit.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.ContextInterpolation

open Landscape.Neural (crossEntropy)

/-- A many-to-one declared readout with two nontrivial fibers, CM42. -/
def aliasReadout (c : Fin 4) : Fin 2 := ⟨c.val % 2, by omega⟩

/-- Only the two observed sources supply raw labels, CM42. -/
def sourceLabels (i : Fin 2) : Fin 4 := if i = 0 then 0 else 2

/-- Fixed context neighborhoods, independent of both output coefficients
and unseen targets, CM42. Observations lie at contexts 0 and 3. -/
def nearestSource (x : Fin 4) : Fin 2 := if x.val < 2 then 0 else 1

/-- First possible target: the observed label persists in each neighborhood. -/
def localTarget (x : Fin 4) : Fin 4 := if x.val < 2 then 0 else 2

/-- Second possible target: the two unobserved raw aliases are exchanged. -/
def otherTarget (x : Fin 4) : Fin 4 :=
  if x = 1 then 2 else if x = 2 then 0 else localTarget x

/-- The two observed aliases agree in the declared answer space, CM42. -/
theorem source_aliases : sourceLabels 0 ≠ sourceLabels 1 ∧
    aliasReadout (sourceLabels 0) = aliasReadout (sourceLabels 1) := by
  norm_num [sourceLabels, aliasReadout]

/-- The actual unitary head has the locally identified raw margin at every
context, CM42. It stores only the two observed source labels. -/
theorem witness_logits (x : Fin 4) (c : Fin 4) :
    headLogits (features (pointWeights nearestSource)
      (complexFeatures (categoricalFourierOn (Fin 2))))
      (complexCoefficients (complexInitializer (categoricalFourierOn (Fin 2)) sourceLabels 4))
      x c = if c = localTarget x then 4 else 0 := by
  rw [initializer_logits _ (categoricalFourierOn_unitary (Fin 2))]
  fin_cases x <;> fin_cases c <;>
    norm_num [labelMass, pointWeights, nearestSource, sourceLabels, localTarget, Fin.sum_univ_two]

/-- Both genuinely unobserved queries get different raw predictions from
the same latent readout, CM42. A single fixed raw representative cannot do so. -/
theorem distinguishes_unseen_aliases :
    headLogits (features (pointWeights nearestSource)
      (complexFeatures (categoricalFourierOn (Fin 2))))
      (complexCoefficients (complexInitializer (categoricalFourierOn (Fin 2)) sourceLabels 4))
      1 0 = 4 ∧
    headLogits (features (pointWeights nearestSource)
      (complexFeatures (categoricalFourierOn (Fin 2))))
      (complexCoefficients (complexInitializer (categoricalFourierOn (Fin 2)) sourceLabels 4))
      2 2 = 4 ∧ aliasReadout (localTarget 1) = aliasReadout (localTarget 2) := by
  rw [witness_logits, witness_logits]
  norm_num [localTarget, aliasReadout]

/-- Exact raw CE at each context is the four-class margin loss, CM42,
under the stated local target. No decoded/pool loss replaces ordinary CE. -/
theorem witness_ce (x : Fin 4) :
    crossEntropy (headLogits (features (pointWeights nearestSource)
      (complexFeatures (categoricalFourierOn (Fin 2))))
      (complexCoefficients (complexInitializer (categoricalFourierOn (Fin 2)) sourceLabels 4)) x)
      (localTarget x) = Real.log (1 + 3 * Real.exp (-4)) := by
  have h : headLogits (features (pointWeights nearestSource)
      (complexFeatures (categoricalFourierOn (Fin 2))))
      (complexCoefficients (complexInitializer (categoricalFourierOn (Fin 2)) sourceLabels 4)) x =
      headLogits (complexFeatures (categoricalFourierOn (Fin 2)))
        (complexCoefficients (complexInitializer (categoricalFourierOn (Fin 2)) sourceLabels 4))
        (nearestSource x) := by
    funext c
    rw [witness_logits, complexInitializer_logits _ (categoricalFourierOn_unitary (Fin 2))]
    fin_cases x <;> simp [nearestSource, sourceLabels, localTarget]
  rw [h]
  have hl : localTarget x = sourceLabels (nearestSource x) := by
    fin_cases x <;> simp [localTarget, sourceLabels, nearestSource]
  rw [hl, complexInitializer_ce _ (categoricalFourierOn_unitary (Fin 2))]
  norm_num

/-- Without locality, the same data and declared readout do not determine
the unseen raw answers, CM42. This retains CM40's identifiability obstruction. -/
theorem same_data_different_raw_answers :
    localTarget 0 = otherTarget 0 ∧ localTarget 3 = otherTarget 3 ∧
    (∀ x, aliasReadout (localTarget x) = aliasReadout (otherTarget x)) ∧
    localTarget 1 ≠ otherTarget 1 ∧ localTarget 2 ≠ otherTarget 2 := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · norm_num [localTarget, otherTarget]
  · norm_num [localTarget, otherTarget]
  · intro x
    fin_cases x <;> norm_num [localTarget, otherTarget, aliasReadout]
  · norm_num [localTarget, otherTarget]
  · norm_num [localTarget, otherTarget]

end Transformer.Grokking.Convex.ConvexDepth.ContextInterpolation

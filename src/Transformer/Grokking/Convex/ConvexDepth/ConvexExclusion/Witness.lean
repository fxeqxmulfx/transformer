module

public import Transformer.Grokking.Convex.ConvexDepth.ConvexExclusion.Head
public import Transformer.Grokking.Convex.ConvexDepth.RawRegions.Witness

/-!
# Nonzero complete Fourier head for exterior raw-label identification

Source: `review/convex/moments.md`, CM49. The four observed rows stay exact;
two queries outside every observed hull use the positive source mixtures of
CM48 after convex exclusion identifies their classes. The feature weights
intentionally do not interpolate the query coordinates. The actual Fourier
source, complete head, full-vocabulary CE, repeated records and short tail
are retained. No training rule is inferred from the geometric hypotheses.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.ConvexExclusion

open RawRegions
open scoped BigOperators
open Landscape.Neural (crossEntropy)

/-- Original observations followed by two exterior queries, CM49. -/
def exclusionQueries : Fin 6 → ℝ := ![-3, -1, 1, 3, -4, 4]

/-- The fixed positive weights use observed sources bearing the identified
label. This explicit fixture never supplies an unknown label to the head. -/
theorem exclusionWeights_label (x : Fin 6) (i : Fin 4) (hi : regionWeights x i ≠ 0) :
    regionLabels i = cutTarget 0 (exclusionQueries x) := by
  fin_cases x <;> fin_cases i <;>
    norm_num [regionWeights, leftWeights, regionLabels, cutTarget, exclusionQueries] at *

/-- Prediction-feature barycenters deliberately differ from input queries.
CM49 uses separate positive certificates for identification and prediction. -/
theorem prediction_barycenters_differ :
    (∑ i, regionWeights 4 i • regionPoints i) ≠ exclusionQueries 4 ∧
      (∑ i, regionWeights 5 i • regionPoints i) ≠ exclusionQueries 5 := by
  norm_num [regionWeights, leftWeights, regionPoints, exclusionQueries, Fin.sum_univ_four]

/-- Both query answers are identified for every compatible convex target
with the declared exterior pool, despite being outside all observed hulls. -/
theorem exterior_fourier_logits (target : ℝ → Fin 4) (hconv : ConvexClasses target)
    (hobs : ∀ i, target (regionPoints i) = regionLabels i)
    (hl : target (-4) ∈ rawPool) (hr : target 4 ∈ rawPool) :
    headLogits (ContextInterpolation.features regionWeights (complexFeatures fourier4))
        (complexCoefficients (complexInitializer fourier4 regionLabels 4)) 4 =
        (fun c => if c = target (exclusionQueries 4) then 4 else 0) ∧
      headLogits (ContextInterpolation.features regionWeights (complexFeatures fourier4))
        (complexCoefficients (complexInitializer fourier4 regionLabels 4)) 5 =
        (fun c => if c = target (exclusionQueries 5) then 4 else 0) := by
  constructor
  · exact certified_initializer_logits regionPoints regionLabels target hconv hobs rawPool
      exclusionQueries regionWeights fourier4 fourier4_unitary 4 0 4 hl left_exclusions
      (regionWeights_sum 4) (by
        simpa [exclusionQueries, cutTarget] using exclusionWeights_label 4)
  · exact certified_initializer_logits regionPoints regionLabels target hconv hobs rawPool
      exclusionQueries regionWeights fourier4 fourier4_unitary 5 2 4 hr right_exclusions
      (regionWeights_sum 5) (by
        have h := exclusionWeights_label 5
        norm_num [exclusionQueries, cutTarget] at h
        exact h)

/-- Every row's actual complete Fourier logits are nonzero margin-four
logits for a concrete compatible target, including the exterior queries. -/
theorem exclusion_logits (x : Fin 6) :
    headLogits (ContextInterpolation.features regionWeights (complexFeatures fourier4))
      (complexCoefficients (complexInitializer fourier4 regionLabels 4)) x =
      fun c => if c = cutTarget 0 (exclusionQueries x) then 4 else 0 := by
  exact same_label_logits fourier4 fourier4_unitary regionLabels regionWeights x
    (cutTarget 0 (exclusionQueries x)) 4 (regionWeights_sum x) (exclusionWeights_label x)

/-- The ordinary four-class raw CE remains exactly the margin loss, CM49. -/
theorem exclusion_ce (x : Fin 6) :
    crossEntropy
      (headLogits (ContextInterpolation.features regionWeights (complexFeatures fourier4))
        (complexCoefficients (complexInitializer fourier4 regionLabels 4)) x)
      (cutTarget 0 (exclusionQueries x)) = Real.log (1 + 3 * Real.exp (-4)) := by
  rw [exclusion_logits, fixed_margin_ce]
  norm_num

/-- The identified exterior answers are distinct aliases of one decoded
value. The whole many-to-one fiber is the explicit candidate pool. -/
theorem exterior_aliases :
    cutTarget 0 (exclusionQueries 4) ≠ cutTarget 0 (exclusionQueries 5) ∧
      ContextInterpolation.aliasReadout (cutTarget 0 (exclusionQueries 4)) =
        ContextInterpolation.aliasReadout (cutTarget 0 (exclusionQueries 5)) := by
  norm_num [exclusionQueries, cutTarget, ContextInterpolation.aliasReadout]

/-- In this witness, the even decoder fiber is exactly the declared pool. -/
theorem decoder_pool (c : Fin 4) : ContextInterpolation.aliasReadout c = 0 ↔ c ∈ rawPool := by
  fin_cases c <;> norm_num [ContextInterpolation.aliasReadout, rawPool]

-- Each new conditional head theorem has a concrete nonzero Fourier instance.
example := certified_margin regionPoints regionLabels (cutTarget 0)
  (cutTarget_convex 0) (fun i => (cutTargets_observed i).1) rawPool
  exclusionQueries regionWeights (complexFeatures fourier4)
  (complexCoefficients (complexInitializer fourier4 regionLabels 4)) 4 0 2 4
  (cut_in_pool 0 (-4)) left_exclusions (regionWeights_nonneg 4) (regionWeights_sum 4)
  (by simpa [exclusionQueries, cutTarget] using exclusionWeights_label 4) (by
    intro i hi
    have hlabel : regionLabels i = 0 := by
      simpa [exclusionQueries, cutTarget] using exclusionWeights_label 4 i hi
    rw [complexInitializer_logits fourier4 fourier4_unitary]
    norm_num [hlabel])
example := certified_ce regionPoints regionLabels (cutTarget 0)
  (cutTarget_convex 0) (fun i => (cutTargets_observed i).1) rawPool
  exclusionQueries regionWeights (complexFeatures fourier4)
  (complexCoefficients (complexInitializer fourier4 regionLabels 4)) 4 0
  (Real.log (1 + 3 * Real.exp (-4))) (cut_in_pool 0 (-4)) left_exclusions
  (regionWeights_nonneg 4) (regionWeights_sum 4)
  (by simpa [exclusionQueries, cutTarget] using exclusionWeights_label 4) (by
    intro i hi
    convert (complexInitializer_ce fourier4 fourier4_unitary regionLabels 4 i).le using 1
    norm_num)
example := certified_initializer_logits regionPoints regionLabels (cutTarget 0)
  (cutTarget_convex 0) (fun i => (cutTargets_observed i).1) rawPool exclusionQueries
  regionWeights fourier4 fourier4_unitary 4 0 4 (cut_in_pool 0 (-4)) left_exclusions
  (regionWeights_sum 4) (by simpa [exclusionQueries, cutTarget] using exclusionWeights_label 4)
example := certified_initializer_ce regionPoints regionLabels (cutTarget 0)
  (cutTarget_convex 0) (fun i => (cutTargets_observed i).1) rawPool exclusionQueries
  regionWeights fourier4 fourier4_unitary 4 0 4 (cut_in_pool 0 (-4)) left_exclusions
  (regionWeights_sum 4) (by simpa [exclusionQueries, cutTarget] using exclusionWeights_label 4)
example := certified_initializer regionPoints regionLabels (cutTarget 0)
  (cutTarget_convex 0) (fun i => (cutTargets_observed i).1) rawPool exclusionQueries
  regionWeights fourier4 fourier4_unitary 4 0 4 (cut_in_pool 0 (-4)) left_exclusions
  (regionWeights_nonneg 4) (regionWeights_sum 4)
  (by simpa [exclusionQueries, cutTarget] using exclusionWeights_label 4)
example := positive_complex_bounds fourier4 fourier4_unitary regionWeights 4
  (regionWeights_nonneg 4) (regionWeights_sum 4)
  (complexCoefficients (complexInitializer fourier4 regionLabels 4)) 32 region_budget
example := target_eq_of_decoder_exclusions regionPoints regionLabels (cutTarget 0)
  (cutTarget_convex 0) (fun i => (cutTargets_observed i).1)
  ContextInterpolation.aliasReadout 0 (x := (-4 : ℝ)) (a := (0 : Fin 4))
  (by norm_num [cutTarget, ContextInterpolation.aliasReadout])
  (fun c hc hne => left_exclusions c ((decoder_pool c).mp hc) hne)
example := exterior_fourier_logits (cutTarget 0) (cutTarget_convex 0)
  (fun i => (cutTargets_observed i).1) (cut_in_pool 0 (-4)) (cut_in_pool 0 4)
example := exclusionWeights_label 4 0 (by norm_num [regionWeights, leftWeights])

-- Original Dirac rows preserve repeated records and the one-record tail.
-- All coefficients are still trained with ordinary CE in the convex head domain.
example := ContextInterpolation.batchLoss_convex regionWeights (complexFeatures fourier4)
  (fun s => observedQuery (regionRecordStates s)) (fun s => regionLabels (regionRecordStates s))
  (Finset.univ : Finset (Fin 5)) 32
example := ContextInterpolation.batch_loss_eq regionWeights (complexFeatures fourier4) regionState
  (fun s => observedQuery (regionRecordStates s)) (fun s => regionLabels (regionRecordStates s))
  (Finset.univ : Finset (Fin 5)) (fun s _ i => regionRecordRows s i)
  (complexCoefficients (complexInitializer fourier4 regionLabels 4))
example := ContextInterpolation.batch_fderiv_eq regionWeights (complexFeatures fourier4) regionState
  (fun s => observedQuery (regionRecordStates s)) (fun s => regionLabels (regionRecordStates s))
  ({4} : Finset (Fin 5)) (fun s _ i => regionRecordRows s i)
  (complexCoefficients (complexInitializer fourier4 regionLabels 4))
example := fresh_observed_ce_indistinguishable (complexFeatures fourier4)
  (complexCoefficients (complexInitializer fourier4 regionLabels 4))

end Transformer.Grokking.Convex.ConvexDepth.ConvexExclusion

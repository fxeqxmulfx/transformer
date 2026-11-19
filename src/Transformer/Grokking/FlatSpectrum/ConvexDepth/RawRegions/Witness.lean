module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.RawRegions.Head
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextInterpolation.Witness

/-!
# Nonzero Fourier realization of hull-certified raw labels

Source: `review/convex/moments.md`, CM48. Four observed points supply two
different raw labels with the same declared readout. Two genuinely unseen
points have same-label barycentric certificates. The complete four-state
Fourier head realizes their raw margins without using either unseen label.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.RawRegions

open scoped BigOperators
open Landscape.Neural (crossEntropy)

/-- Four observations followed by two held-out input points, CM48. -/
def regionQueries : Fin 6 → ℝ := ![-3, -1, 1, 3, -2, 2]

/-- Observed inputs retain their Dirac rows; unseen inputs use two same-label
source points. This is an explicit certificate fixture, not a discovery algorithm. -/
noncomputable def regionWeights (x : Fin 6) (i : Fin 4) : ℝ :=
  if x.val < 4 then (if i.val = x.val then 1 else 0)
  else if x = 4 then leftWeights i else if 2 ≤ i.val then 1 / 2 else 0

theorem regionWeights_nonneg (x : Fin 6) (i : Fin 4) : 0 ≤ regionWeights x i := by
  fin_cases x <;> fin_cases i <;> norm_num [regionWeights, leftWeights]

theorem regionWeights_sum (x : Fin 6) : ∑ i, regionWeights x i = 1 := by
  simp only [Fin.sum_univ_four]
  fin_cases x <;> norm_num [regionWeights, leftWeights]

/-- These weights certify the input coordinates, not merely feature-space membership. -/
theorem regionWeights_barycenter (x : Fin 6) :
    ∑ i, regionWeights x i • regionPoints i = regionQueries x := by
  fin_cases x <;>
    norm_num [regionWeights, leftWeights, regionPoints, regionQueries, Fin.sum_univ_four]

/-- Every nonzero weight uses the raw label attached to its observed source. -/
theorem regionWeights_label (x : Fin 6) (i : Fin 4) (hi : regionWeights x i ≠ 0) :
    regionLabels i = cutTarget 0 (regionQueries x) := by
  fin_cases x <;> fin_cases i <;>
    norm_num [regionWeights, leftWeights, regionLabels, cutTarget, regionQueries] at *

def observedQuery (i : Fin 4) : Fin 6 := ⟨i.val, by omega⟩

/-- Every original observed feature row is exact for every head tensor.
CM42 therefore preserves each original batch CE and its full-head derivatives. -/
theorem regionWeights_observed (i j : Fin 4) :
    regionWeights (observedQuery i) j = if j = i then 1 else 0 := by
  fin_cases i <;> fin_cases j <;> norm_num [regionWeights, observedQuery]

/-- Actual complete real/imaginary Fourier products give raw margin four,
including both unseen points, CM48. The head contains only observed labels. -/
theorem region_logits (x : Fin 6) :
    headLogits (ContextInterpolation.features regionWeights (complexFeatures fourier4))
      (complexCoefficients (complexInitializer fourier4 regionLabels 4)) x =
      fun c => if c = cutTarget 0 (regionQueries x) then 4 else 0 := by
  exact same_label_logits fourier4 fourier4_unitary regionLabels regionWeights x
    (cutTarget 0 (regionQueries x)) 4 (regionWeights_sum x) (regionWeights_label x)

/-- The ordinary full-vocabulary raw CE is the four-class margin loss,
CM48. No probability pooling or alternative loss enters this identity. -/
theorem region_ce (x : Fin 6) :
    crossEntropy
      (headLogits (ContextInterpolation.features regionWeights (complexFeatures fourier4))
        (complexCoefficients (complexInitializer fourier4 regionLabels 4)) x)
      (cutTarget 0 (regionQueries x)) = Real.log (1 + 3 * Real.exp (-4)) := by
  rw [region_logits, fixed_margin_ce]
  norm_num

/-- The two held-out queries are not observed and have distinct raw answers
despite equal declared readouts, CM48. Their answers use nonzero complex logits. -/
theorem region_unseen_aliases :
    regionQueries 4 ∉ Set.range regionPoints ∧ regionQueries 5 ∉ Set.range regionPoints ∧
      cutTarget 0 (regionQueries 4) ≠ cutTarget 0 (regionQueries 5) ∧
      ContextInterpolation.aliasReadout (cutTarget 0 (regionQueries 4)) =
        ContextInterpolation.aliasReadout (cutTarget 0 (regionQueries 5)) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rintro ⟨i, hi⟩
    fin_cases i <;> norm_num [regionPoints, regionQueries] at hi
  · rintro ⟨i, hi⟩
    fin_cases i <;> norm_num [regionPoints, regionQueries] at hi
  · norm_num [cutTarget, regionQueries]
  · norm_num [cutTarget, regionQueries, ContextInterpolation.aliasReadout]

-- Conditional head results use actual nonzero Fourier coefficients.
example := certified_margin regionPoints regionLabels (cutTarget 0)
  (cutTarget_convex 0) (fun i => (cutTargets_observed i).1)
  regionQueries regionWeights (complexFeatures fourier4)
  (complexCoefficients (complexInitializer fourier4 regionLabels 4)) 4 0 2 4
  (regionWeights_nonneg 4) (regionWeights_sum 4) (by
    simpa [regionQueries, cutTarget] using regionWeights_label 4)
  (regionWeights_barycenter 4) (by
    intro i hi
    have hlabel : regionLabels i = 0 := by
      simpa [regionQueries, cutTarget] using regionWeights_label 4 i hi
    rw [complexInitializer_logits fourier4 fourier4_unitary]
    norm_num [hlabel])

example := certified_ce regionPoints regionLabels (cutTarget 0)
  (cutTarget_convex 0) (fun i => (cutTargets_observed i).1)
  regionQueries regionWeights (complexFeatures fourier4)
  (complexCoefficients (complexInitializer fourier4 regionLabels 4)) 4 0
  (Real.log (1 + 3 * Real.exp (-4)))
  (regionWeights_nonneg 4) (regionWeights_sum 4) (by
    simpa [regionQueries, cutTarget] using regionWeights_label 4)
  (regionWeights_barycenter 4) (by
    intro i hi
    convert (complexInitializer_ce fourier4 fourier4_unitary regionLabels 4 i).le using 1
    norm_num)

example := same_label_logits fourier4 fourier4_unitary regionLabels regionWeights 4 0 4
  (regionWeights_sum 4) (by simpa [regionQueries, cutTarget] using regionWeights_label 4)
example := regionWeights_label 4 0 (by norm_num [regionWeights, leftWeights])
example := ContextInterpolation.complex_features_bound fourier4 fourier4_unitary
  regionWeights 4 (regionWeights_nonneg 4) (regionWeights_sum 4) (0, true)
example := ContextInterpolation.features_at regionWeights (complexFeatures fourier4)
  (observedQuery 0) 0 (regionWeights_observed 0) (1, true)

/-- The same complete coefficient budget covers all six input rows. -/
theorem region_budget :
    complexCoefficients (complexInitializer fourier4 regionLabels 4) ∈ headBudget 32 := by
  convert complexInitializer_budget fourier4 fourier4_unitary regionLabels 4 using 1
  norm_num

example := ContextInterpolation.logits_bound regionWeights (complexFeatures fourier4)
  4 (regionWeights_nonneg 4) (regionWeights_sum 4) (by norm_num : (0 : ℝ) ≤ 1)
  (fun k i => complexFeatures_bound fourier4 fourier4_unitary k i) region_budget 0

/-- Five original record IDs repeat the first observation and end in a
one-record tail. Their input rows stay exact before any optimizer update. -/
def regionRecordStates : Fin 5 → Fin 4 := ![0, 0, 1, 2, 3]
def regionState (x : Fin 6) : Fin 4 := if h : x.val < 4 then ⟨x.val, h⟩ else 0

theorem regionRecordRows (s : Fin 5) (i : Fin 4) :
    regionWeights (observedQuery (regionRecordStates s)) i =
      if i = regionState (observedQuery (regionRecordStates s)) then 1 else 0 := by
  fin_cases s <;> fin_cases i <;>
    norm_num [regionWeights, observedQuery, regionRecordStates, regionState]

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

end Transformer.Grokking.FlatSpectrum.ConvexDepth.RawRegions

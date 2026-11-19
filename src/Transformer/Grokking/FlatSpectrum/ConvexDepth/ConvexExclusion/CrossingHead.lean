module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion.Crossing
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion.Witness

/-!
# Actual complete Fourier head after planar crossing exclusions

Source: `review/convex/moments.md`, CM50. Convex target regions and a declared
candidate pool identify exterior queries missed by the observed-point test.
Fixed positive source mixtures then use the unchanged complete Fourier head;
ordinary raw CE, coefficient/feature bounds and observed derivatives remain.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion

open RawRegions
open scoped BigOperators
open Landscape.Neural (crossEntropy)

/-- The four observations followed by the two exterior planar queries. -/
def crossingQueries : Fin 6 → ℝ × ℝ :=
  ![(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, 0), (2, 0)]

/-- A nonconstant compatible target, used only to inhabit the hypotheses. -/
noncomputable def crossingTarget (x : ℝ × ℝ) : Fin 4 := cutTarget 0 x.1

/-- All four raw-class regions of the concrete target are convex, CM50. -/
theorem crossingTarget_convex : ConvexClasses crossingTarget := by
  intro c p hp q hq a b ha hb hab
  change cutTarget 0 (a * p.1 + b * q.1) = c
  exact (cutTarget_convex 0 c) hp hq ha hb hab

/-- The nonconstant target fits every original observation. -/
theorem crossingTarget_observed (i : Fin 4) :
    crossingTarget (crossingPoints i) = regionLabels i := by
  fin_cases i <;> norm_num [crossingTarget, cutTarget, crossingPoints, regionLabels]

/-- This target's raw labels belong to the declared pool. -/
theorem crossingTarget_pool (x : ℝ × ℝ) : crossingTarget x ∈ rawPool :=
  cut_in_pool 0 x.1

/-- Every other candidate in the declared pool conflicts at the left query. -/
theorem crossing_left_exclusions (c : Fin 4) (hc : c ∈ rawPool) (hne : c ≠ 0) :
    Excluded crossingPoints regionLabels (-2, 0) c := by
  have h : c = 0 ∨ c = 2 := by simpa [rawPool] using hc
  rcases h with rfl | rfl
  · exact (hne rfl).elim
  · exact left_crossing_excluded

/-- Every other candidate conflicts at the symmetric right query. -/
theorem crossing_right_exclusions (c : Fin 4) (hc : c ∈ rawPool) (hne : c ≠ 2) :
    Excluded crossingPoints regionLabels (2, 0) c := by
  have h : c = 0 ∨ c = 2 := by simpa [rawPool] using hc
  rcases h with rfl | rfl
  · exact right_crossing_excluded
  · exact (hne rfl).elim

/-- Both exterior answers follow for every compatible convex target with
this pool, CM50. Neither missing answer is an input hypothesis. -/
theorem crossing_identifies (target : ℝ × ℝ → Fin 4)
    (hconv : ConvexClasses target)
    (hobs : ∀ i, target (crossingPoints i) = regionLabels i)
    (hl : target (-2, 0) ∈ rawPool) (hr : target (2, 0) ∈ rawPool) :
    target (-2, 0) = 0 ∧ target (2, 0) = 2 := by
  constructor
  · exact target_eq_of_exclusions crossingPoints regionLabels target hconv hobs
      rawPool hl crossing_left_exclusions
  · exact target_eq_of_exclusions crossingPoints regionLabels target hconv hobs
      rawPool hr crossing_right_exclusions

/-- The positive prediction mixtures use the observed sources of the
identified class; they are separate from the intersection-coordinate weights. -/
theorem crossingWeights_label (x : Fin 6) (i : Fin 4) (hi : regionWeights x i ≠ 0) :
    regionLabels i = crossingTarget (crossingQueries x) := by
  fin_cases x <;> fin_cases i <;>
    norm_num [regionWeights, leftWeights, regionLabels, crossingTarget, cutTarget,
      crossingQueries] at *

/-- The observed query embeddings are the original planar inputs. -/
theorem crossing_observed_query (i : Fin 4) :
    crossingQueries (observedQuery i) = crossingPoints i := by
  fin_cases i <;> rfl

/-- The actual complete nonzero head gives margin four for both answers
identified by crossing certificates, CM50, for every compatible target. -/
theorem crossing_certified_logits (target : ℝ × ℝ → Fin 4)
    (hconv : ConvexClasses target)
    (hobs : ∀ i, target (crossingPoints i) = regionLabels i)
    (hl : target (-2, 0) ∈ rawPool) (hr : target (2, 0) ∈ rawPool) :
    headLogits (ContextInterpolation.features regionWeights (complexFeatures fourier4))
        (complexCoefficients (complexInitializer fourier4 regionLabels 4)) 4 =
        (fun c => if c = target (crossingQueries 4) then 4 else 0) ∧
      headLogits (ContextInterpolation.features regionWeights (complexFeatures fourier4))
        (complexCoefficients (complexInitializer fourier4 regionLabels 4)) 5 =
        (fun c => if c = target (crossingQueries 5) then 4 else 0) := by
  constructor
  · exact certified_initializer_logits crossingPoints regionLabels target hconv hobs rawPool
      crossingQueries regionWeights fourier4 fourier4_unitary 4 0 4 hl crossing_left_exclusions
      (regionWeights_sum 4) (by
        simpa [crossingQueries, crossingTarget, cutTarget] using crossingWeights_label 4)
  · exact certified_initializer_logits crossingPoints regionLabels target hconv hobs rawPool
      crossingQueries regionWeights fourier4 fourier4_unitary 5 2 4 hr crossing_right_exclusions
      (regionWeights_sum 5) (by
        have h := crossingWeights_label 5
        norm_num [crossingQueries, crossingTarget, cutTarget] at h
        exact h)

/-- Every original and exterior row has actual complete margin-four logits
for the nonconstant compatible target, CM50. -/
theorem crossing_logits (x : Fin 6) :
    headLogits (ContextInterpolation.features regionWeights (complexFeatures fourier4))
      (complexCoefficients (complexInitializer fourier4 regionLabels 4)) x =
      fun c => if c = crossingTarget (crossingQueries x) then 4 else 0 := by
  exact same_label_logits fourier4 fourier4_unitary regionLabels regionWeights x
    (crossingTarget (crossingQueries x)) 4 (regionWeights_sum x) (crossingWeights_label x)

/-- Ordinary full-vocabulary CE is the actual four-class margin loss, CM50. -/
theorem crossing_ce (x : Fin 6) :
    crossEntropy
      (headLogits (ContextInterpolation.features regionWeights (complexFeatures fourier4))
        (complexCoefficients (complexInitializer fourier4 regionLabels 4)) x)
      (crossingTarget (crossingQueries x)) = Real.log (1 + 3 * Real.exp (-4)) := by
  rw [crossing_logits, fixed_margin_ce]
  norm_num

/-- Prediction weights do not interpolate either exterior input coordinate.
The separate crossing certificate supplies identification, CM50. -/
theorem crossing_prediction_barycenters_differ :
    (∑ i, regionWeights 4 i • crossingPoints i) ≠ crossingQueries 4 ∧
      (∑ i, regionWeights 5 i • crossingPoints i) ≠ crossingQueries 5 := by
  constructor <;>
    norm_num [regionWeights, leftWeights, crossingPoints, crossingQueries, Fin.sum_univ_four]

/-- The two identified raw answers share a many-to-one decoded value. -/
theorem crossing_aliases :
    crossingTarget (crossingQueries 4) ≠ crossingTarget (crossingQueries 5) ∧
      ContextInterpolation.aliasReadout (crossingTarget (crossingQueries 4)) =
        ContextInterpolation.aliasReadout (crossingTarget (crossingQueries 5)) := by
  norm_num [crossingTarget, cutTarget, crossingQueries, ContextInterpolation.aliasReadout]

-- Every identification hypothesis has a nonconstant concrete witness.
example := crossing_left_exclusions 2 (by simp [rawPool]) (by decide)
example := crossing_right_exclusions 0 (by simp [rawPool]) (by decide)
example := crossing_identifies crossingTarget crossingTarget_convex crossingTarget_observed
  (crossingTarget_pool (-2, 0)) (crossingTarget_pool (2, 0))
example := crossing_certified_logits crossingTarget crossingTarget_convex crossingTarget_observed
  (crossingTarget_pool (-2, 0)) (crossingTarget_pool (2, 0))
example := crossingWeights_label 4 0 (by norm_num [regionWeights, leftWeights])

-- Bounds and original repeated-record/tail derivatives use the complete head.
example (x : Fin 6) := positive_complex_bounds fourier4 fourier4_unitary regionWeights x
  (regionWeights_nonneg x) (regionWeights_sum x)
  (complexCoefficients (complexInitializer fourier4 regionLabels 4)) 32 region_budget
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

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion

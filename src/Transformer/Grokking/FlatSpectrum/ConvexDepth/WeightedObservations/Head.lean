module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.WeightedObservations.Basic

/-!
# Grouping the actual full-vocabulary complex-head CE

Source: `review/flat_spectrum.md`, CM60. Equal complete features and raw labels
permit multiplicity-weighted evaluation of the actual CM57 objective and its
derivatives. Every coefficient remains learned jointly. This is an exact
algebraic identity, not a floating-point error bound or a clean-label oracle.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.WeightedObservations

open scoped BigOperators
open Transformer.Landscape.Neural (crossEntropy)

variable {I K F X C : Type*} [DecidableEq K] [Fintype K] [Fintype F] [Fintype C]

/-- Ordinary full-vocabulary CE on representatives, with original batch
multiplicity and normalization (CM60). Every raw class remains in each CE. -/
noncomputable def groupedHeadLoss (φ : F → X → ℝ) (representatives : K → X)
    (labels : K → C) (batch : Finset I) (key : I → K) (W : F → C → ℝ) : ℝ :=
  groupedMean batch key
    (fun k => crossEntropy (headLogits φ W (representatives k)) (labels k))

/-- Equal complete feature vectors and raw labels preserve the actual head's
original mean CE at every coefficient tensor (CM60). -/
theorem batchHeadLoss_eq (φ : F → X → ℝ) (inputs : I → X) (labels : I → C)
    (representatives : K → X) (groupLabels : K → C) (batch : Finset I) (key : I → K)
    (hfeatures : ∀ i ∈ batch, ∀ f, φ f (inputs i) = φ f (representatives (key i)))
    (hlabels : ∀ i ∈ batch, labels i = groupLabels (key i)) (W : F → C → ℝ) :
    Fallback.batchLoss φ inputs labels batch W =
      groupedHeadLoss φ representatives groupLabels batch key W := by
  unfold groupedHeadLoss
  rw [groupedMean_eq]
  unfold Fallback.batchLoss headLoss recordMean
  congr 1
  rw [Finset.sum_coe_sort batch (fun i => crossEntropy (headLogits φ W (inputs i)) (labels i))]
  apply Finset.sum_congr rfl
  intro i hi
  rw [hlabels i hi]
  congr 1
  funext c
  unfold headLogits
  apply Finset.sum_congr rfl
  intro f hf
  rw [hfeatures i hi f]

/-- Actual finite-vocabulary CE is differentiable in the complete tensor.
Differentiable affine logits and the positive softmax sum justify the log (CM60). -/
theorem batch_differentiable (φ : F → X → ℝ) (inputs : I → X) (labels : I → C)
    (batch : Finset I) (W : F → C → ℝ) :
    DifferentiableAt ℝ (Fallback.batchLoss φ inputs labels batch) W := by
  have hrow (i : ↥batch) : DifferentiableAt ℝ
      (fun V : F → C → ℝ => crossEntropy (headLogits φ V (inputs i.val)) (labels i.val)) W := by
    unfold crossEntropy
    apply DifferentiableAt.fun_sub
    · apply DifferentiableAt.log
      · unfold headLogits
        fun_prop
      · exact ne_of_gt (Transformer.Landscape.Neural.sum_exp_pos _ (labels i.val))
    · unfold headLogits
      fun_prop
  exact (DifferentiableAt.fun_sum (fun i _ => hrow i)).const_mul (batch.card : ℝ)⁻¹

/-- Equality holds as a function of the complete learned tensor, so it also
preserves its Fréchet derivative. At differentiability points this is the
actual derivative; elsewhere the library convention is retained (CM60). -/
theorem batch_fderiv_eq (φ : F → X → ℝ) (inputs : I → X) (labels : I → C)
    (representatives : K → X) (groupLabels : K → C) (batch : Finset I) (key : I → K)
    (hfeatures : ∀ i ∈ batch, ∀ f, φ f (inputs i) = φ f (representatives (key i)))
    (hlabels : ∀ i ∈ batch, labels i = groupLabels (key i)) (W : F → C → ℝ) :
    fderiv ℝ (Fallback.batchLoss φ inputs labels batch) W =
      fderiv ℝ (groupedHeadLoss φ representatives groupLabels batch key) W := by
  have hfun : Fallback.batchLoss φ inputs labels batch =
      groupedHeadLoss φ representatives groupLabels batch key := by
    funext V
    exact batchHeadLoss_eq φ inputs labels representatives groupLabels batch key
      hfeatures hlabels V
  rw [hfun]

/-- A genuine derivative witness transfers, including derivative existence;
no numerical finite-difference premise is substituted (CM60). -/
theorem batch_hasFDerivAt (φ : F → X → ℝ) (inputs : I → X) (labels : I → C)
    (representatives : K → X) (groupLabels : K → C) (batch : Finset I) (key : I → K)
    (hfeatures : ∀ i ∈ batch, ∀ f, φ f (inputs i) = φ f (representatives (key i)))
    (hlabels : ∀ i ∈ batch, labels i = groupLabels (key i))
    (W : F → C → ℝ) (D : (F → C → ℝ) →L[ℝ] ℝ)
    (hd : HasFDerivAt (Fallback.batchLoss φ inputs labels batch) D W) :
    HasFDerivAt (groupedHeadLoss φ representatives groupLabels batch key) D W := by
  have hfun : Fallback.batchLoss φ inputs labels batch =
      groupedHeadLoss φ representatives groupLabels batch key := by
    funext V
    exact batchHeadLoss_eq φ inputs labels representatives groupLabels batch key
      hfeatures hlabels V
  rw [← hfun]
  exact hd

variable {P A : Type*} [Fintype P] [DecidableEq P]

/-- The actual CM57 joint domain retains convex ordinary CE after exact
grouping. This changes neither its priors nor its feasibility requirements,
and in particular does not repair contradictory noisy anchors (CM60). -/
theorem coordinate_batch_convex (e : C → ℝ) (he : ∀ c, |e c| ≤ 1)
    (h : P → X → Fin 5 → ℝ) (representatives : K → X) (labels : K → C)
    (batch : Finset I) (key : I → K) (rows : P → A → Fin 5 → ℝ)
    (values : P → A → ℝ) (B : P → ℝ) (R δ : ℝ) :
    ConvexOn ℝ (CoordinatePrograms.domain rows values B R δ)
      (fun θ => groupedHeadLoss (CoordinatePrograms.features h) representatives labels
        batch key (CoordinatePrograms.head e θ)) := by
  have hfun : (fun θ => Fallback.batchLoss (CoordinatePrograms.features h)
      (fun i => representatives (key i)) (fun i => labels (key i)) batch
      (CoordinatePrograms.head e θ)) =
      (fun θ => groupedHeadLoss (CoordinatePrograms.features h) representatives labels
        batch key (CoordinatePrograms.head e θ)) := by
    funext θ
    exact batchHeadLoss_eq (CoordinatePrograms.features h)
      (fun i => representatives (key i)) (fun i => labels (key i))
      representatives labels batch key (fun _ _ _ => rfl) (fun _ _ => rfl)
      (CoordinatePrograms.head e θ)
  rw [← hfun]
  exact CoordinatePrograms.domain_batchLoss_convex e he h _ _ batch rows values B R δ

/-- Distinct raw inputs zero and two share their complete parity features;
the third original record belongs to a different group (CM60). -/
def toyInputs : Fin 3 → Fin 4 := ![0, 2, 1]
def toyRepresentatives (k : Bool) : Fin 4 := if k then 1 else 0
def toyFeatures (f : Fin 2) (x : Fin 4) : ℝ := if f = 0 then 1 else (x.val % 2 : ℕ)

/-- The concrete grouping has equal complete features without requiring
identical raw inputs; both feature directions are used (CM60). -/
theorem toy_features (i : Fin 3) (f : Fin 2) :
    toyFeatures f (toyInputs i) = toyFeatures f (toyRepresentatives (toyKey i)) := by
  fin_cases i <;> fin_cases f <;>
    norm_num [toyFeatures, toyInputs, toyRepresentatives, toyKey]

example (W : Fin 2 → Bool → ℝ) :
    Fallback.batchLoss toyFeatures toyInputs toyKey Finset.univ W =
      groupedHeadLoss toyFeatures toyRepresentatives id Finset.univ toyKey W :=
  batchHeadLoss_eq toyFeatures toyInputs toyKey toyRepresentatives id Finset.univ toyKey
    (fun i _ f => toy_features i f) (fun _ _ => rfl) W

example (W : Fin 2 → Bool → ℝ) :
    fderiv ℝ (Fallback.batchLoss toyFeatures toyInputs toyKey Finset.univ) W =
      fderiv ℝ (groupedHeadLoss toyFeatures toyRepresentatives id Finset.univ toyKey) W :=
  batch_fderiv_eq toyFeatures toyInputs toyKey toyRepresentatives id Finset.univ toyKey
    (fun i _ f => toy_features i f) (fun _ _ => rfl) W

example (W : Fin 2 → Bool → ℝ) :
    HasFDerivAt
      (groupedHeadLoss toyFeatures toyRepresentatives id Finset.univ toyKey)
      (fderiv ℝ (Fallback.batchLoss toyFeatures toyInputs toyKey Finset.univ) W) W :=
  batch_hasFDerivAt toyFeatures toyInputs toyKey toyRepresentatives id Finset.univ toyKey
    (fun i _ f => toy_features i f) (fun _ _ => rfl) W _
    (batch_differentiable toyFeatures toyInputs toyKey Finset.univ W).hasFDerivAt

example := coordinate_batch_convex CoordinatePrograms.exampleCoordinates
  ClassGeometry.exampleCoordinates_bound CoordinatePrograms.exampleFeatures
  id CoordinatePrograms.exampleTarget Finset.univ CoordinatePrograms.exampleInputs
  CoordinatePrograms.exampleRows (fun _ => CoordinatePrograms.exampleValues) (fun _ => 16) 128 1

example := CoordinatePrograms.example_feasible

end Transformer.Grokking.FlatSpectrum.ConvexDepth.WeightedObservations

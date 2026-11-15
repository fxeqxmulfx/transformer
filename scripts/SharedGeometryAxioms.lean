import Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback.Batch
import Transformer.Grokking.FlatSpectrum.ConvexDepth.ComplexHead
import Transformer.Grokking.FlatSpectrum.ConvexDepth.ParameterSharing

open Transformer.Grokking.FlatSpectrum.ConvexDepth
open scoped BigOperators

#print axioms ConvexOn.comp_affineMap
#print axioms ConvexOn.smul
#print axioms headLoss_convex
#print axioms headBudget_convex
#print axioms headLogits_bound
#print axioms complexFeatures_bound
#print axioms Fallback.batch_loss_eq
#print axioms ParameterSharing.weightedDomain_convex
#print axioms ParameterSharing.parameterMap_budget
#print axioms ParameterSharing.logits_bound
#print axioms ParameterSharing.loss_convex
#print axioms ParameterSharing.batchLoss_convex

namespace SharedGeometryAudit

abbrev Coordinate := Fin 2 × (Fin 4 × Bool)
abbrev Input := Fin 2 × Fin 4

-- Two complete four-state complex blocks. Each global parameter changes
-- coefficients in both blocks and both raw classes, including imaginary slots.
noncomputable def shared : (Fin 2 → ℝ) →ₗ[ℝ] (Coordinate → Bool → ℝ) where
  toFun θ k c := (if c then 1 else -1) * (k.1.val + 1) * (k.2.1.val + 1) *
    (if k.2.2 then θ 1 else θ 0) / 128
  map_add' θ η := by
    funext k c
    cases h : k.2.2 <;> simp [h, mul_add, add_div]
  map_smul' a θ := by
    funext k c
    cases c <;> cases h : k.2.2 <;> simp [h, smul_eq_mul] <;> ring

-- A fixed nonzero offset is an affine part of the actual complete tensor.
noncomputable def offset (k : Coordinate) (c : Bool) : ℝ :=
  (if c then 1 else -1) * (k.1.val + 1) * (k.2.1.val + 1) /
    (if k.2.2 then 256 else 128)
noncomputable def sharing : (Fin 2 → ℝ) →ᵃ[ℝ] (Coordinate → Bool → ℝ) :=
  shared.toAffineMap + AffineMap.const ℝ (Fin 2 → ℝ) offset

noncomputable def features (k : Coordinate) (x : Input) : ℝ :=
  if k.1 = x.1 then complexFeatures fourier4 k.2 x.2 else 0
def inputs (s : Fin 3) : Input := if s = 2 then (1, 1) else (0, 0)
def labels (s : Fin 3) : Bool := s == 2

-- The budget is imposed on every expanded raw-class coefficient, not only
-- on the compact parameter vector. This nonzero point is actually feasible.
example : sharing ![1, 2] ∈ headBudget 2 := by
  intro c
  cases c <;>
    norm_num [headBudget, sharing, shared, offset, Fintype.sum_prod_type,
      Fin.sum_univ_two, Fin.sum_univ_four, Fintype.sum_bool]

-- The affine-map preimage is the exact feasible domain for tied parameters.
-- Each batch retains original record identities, repetitions and its real size.
example (batch : Finset (Fin 3)) :
    ConvexOn ℝ (sharing ⁻¹' headBudget 2)
      (fun θ => Fallback.batchLoss features inputs labels batch (sharing θ)) := by
  have h := (headLoss_convex features (fun s : ↥batch => inputs s.val)
    (fun s : ↥batch => labels s.val) 2).comp_affineMap sharing
  have hc : 0 ≤ (batch.card : ℝ)⁻¹ := inv_nonneg.mpr (Nat.cast_nonneg _)
  simpa [Fallback.batchLoss, Function.comp_def, smul_eq_mul] using ConvexOn.smul hc h

example (θ : Fin 2 → ℝ) (hθ : sharing θ ∈ headBudget 2) (x : Input) (c : Bool) :
    |headLogits features (sharing θ) x c| ≤ 2 := by
  have h := headLogits_bound features hθ (by norm_num : (0 : ℝ) ≤ 1) x
    (fun k => by
      unfold features
      split_ifs
      · exact complexFeatures_bound fourier4 fourier4_unitary k.2 x.2
      · norm_num) c
  simpa using h

-- Distinct observed records0 and1 repeat the same input. The final batch
-- contains only record2. The preceding example applies to both actual batches.
example : (Finset.univ : Finset (Fin 3)).card = 3 := by decide
example : ({2} : Finset (Fin 3)).card = 1 := by decide

end SharedGeometryAudit

#print axioms SharedGeometryAudit.sharing

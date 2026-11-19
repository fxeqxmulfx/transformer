import Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback.Batch
import Transformer.Grokking.FlatSpectrum.ConvexDepth.SignedContext.Basic
import Transformer.Grokking.FlatSpectrum.ConvexDepth.ParameterSharing

/-!
Task336 before original heldout evaluation. Source: `review/convex/moments.md`
CM38/CM55, and the fixed partition expansion audit at47ba037f. This instantiates
the fallback composition with52 shared coordinates, the maximum in the native
partitioned library. Coverage and the learned descriptor tree are fixed during
CE; unsupported inputs retain the complete old head. Repeated records and the
short batch keep their actual losses and derivatives. The map remains arbitrary:
its coefficient costs are the separate CM55 premise, audited at47ba037f and
checked on every expanded native state. No convexity of discovery, floating-point
refinement, or unseen-label correctness is inferred.
-/

open Transformer.Grokking.FlatSpectrum.ConvexDepth

#print axioms Fallback.logits
#print axioms Fallback.coefficients_mem
#print axioms Fallback.parent_logits
#print axioms Fallback.batchLoss_convex
#print axioms Fallback.batch_loss_eq
#print axioms Fallback.batch_fderiv_eq
#print axioms SignedContext.normalized_features_bound
#print axioms ParameterSharing.parameterMap_budget
#print axioms ParameterSharing.batchLoss_convex
#print axioms ConvexOn.comp_affineMap

namespace PartitionedBoundaryFallbackAudit

-- CM38 applies to the actual shared observed rows; there is no Dirac premise.
noncomputable def row (x : Fin 3) : Fin 4 → ℝ :=
  ![1, 1, (x.val : ℝ), -(x.val : ℝ)]
noncomputable def candidate : Fin 4 × Bool → Fin 3 → ℝ :=
  ContextInterpolation.features (fun x => SignedContext.normalized (row x))
    (complexFeatures fourier4)
def known (x : Fin 3) : Bool := x != 2
def inputs (s : Fin 3) : Fin 3 := if s = 2 then 1 else 0
def labels (s : Fin 3) : Fin 4 := if s = 2 then 1 else 0
noncomputable def previous (k : Fin 1) (x : Fin 3) : ℝ := 2 + k.val + x.val
noncomputable def parent (k : Fin 1) (c : Fin 4) : ℝ :=
  if c = 0 then 3 + k.val else -3 - k.val

-- Three original record IDs include a repeated input and a shorter tail.
example (W : Fin 4 × Bool → Fin 4 → ℝ) :
    Fallback.batchLoss (Fallback.features known candidate previous) inputs labels
      Finset.univ (Fallback.coefficients W parent) =
    Fallback.batchLoss candidate inputs labels Finset.univ W := by
  apply Fallback.batch_loss_eq
  intro s _
  fin_cases s <;> norm_num [inputs, known]

example (W : Fin 4 × Bool → Fin 4 → ℝ) :
    fderiv ℝ (fun V => Fallback.batchLoss (Fallback.features known candidate previous)
      inputs labels ({2} : Finset (Fin 3)) (Fallback.coefficients V parent)) W =
    fderiv ℝ (Fallback.batchLoss candidate inputs labels ({2} : Finset (Fin 3))) W := by
  apply Fallback.batch_fderiv_eq
  intro s hs
  have : s = 2 := by simpa using hs
  subst s
  norm_num [inputs, known]

example := Fallback.batchLoss_convex known candidate previous inputs labels
  (Finset.univ : Finset (Fin 3)) parent 65539

-- CM38's actual gated loss pulls back through the fixed CM55 sharing map.
-- Its exact domain charges the complete head, with no independent-factor claim.
example (F : (Fin 52 → ℝ) →ᵃ[ℝ] ((Fin 4 × Bool) ⊕ Fin 1 → Fin 4 → ℝ)) :
    ConvexOn ℝ (F ⁻¹' Fallback.parentDomain parent 65539)
      (fun θ => Fallback.batchLoss (Fallback.features known candidate previous)
        inputs labels Finset.univ (F θ)) := by
  exact (Fallback.batchLoss_convex known candidate previous inputs labels
    (Finset.univ : Finset (Fin 3)) parent 65539).comp_affineMap F

-- The original covered-batch equality holds as a function of shared parameters,
-- so the actual ordinary CE derivatives agree there, including the short tail.
example (F : (Fin 52 → ℝ) →ᵃ[ℝ] (Fin 4 × Bool → Fin 4 → ℝ)) (θ : Fin 52 → ℝ) :
    fderiv ℝ (fun η => Fallback.batchLoss (Fallback.features known candidate previous)
      inputs labels ({2} : Finset (Fin 3)) (Fallback.coefficients (F η) parent)) θ =
    fderiv ℝ (fun η => Fallback.batchLoss candidate inputs labels
      ({2} : Finset (Fin 3)) (F η)) θ := by
  have h : (fun η => Fallback.batchLoss (Fallback.features known candidate previous)
      inputs labels ({2} : Finset (Fin 3)) (Fallback.coefficients (F η) parent)) =
      (fun η => Fallback.batchLoss candidate inputs labels ({2} : Finset (Fin 3)) (F η)) := by
    funext η
    apply Fallback.batch_loss_eq
    intro s hs
    have : s = 2 := by simpa using hs
    subst s
    norm_num [inputs, known]
  rw [h]

-- The unsupported query keeps nonzero old logits for every new coefficient.
example (W : Fin 4 × Bool → Fin 4 → ℝ) :
    headLogits (Fallback.features known candidate previous)
      (Fallback.coefficients W parent) 2 0 = 12 := by
  rw [Fallback.logits]
  norm_num [known, headLogits, previous, parent, Fin.sum_univ_one]

end PartitionedBoundaryFallbackAudit

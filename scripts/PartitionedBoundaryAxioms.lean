import Transformer.Grokking.FlatSpectrum.ConvexDepth.ParameterSharing
import Transformer.Grokking.FlatSpectrum.ConvexDepth.SignedContext

/-!
Prospective task335 revision audit, before fitting a partitioned constructor.
CM55 and CM51 in `review/flat_spectrum.md`: a fixed input-descriptor partition
gates shared boundary parameters. Learning a partition precedes convex CE;
the partition is fixed during CE. Each leaf still shares parameters across
multiple blocks. All expanded coefficients and categorical offsets are charged.
No unseen-label identification or floating-point refinement is inferred.
-/
open Transformer.Grokking.FlatSpectrum.ConvexDepth
open scoped BigOperators

namespace PartitionedBoundaryAudit

abbrev P := (Fin 4 × Bool) ⊕ (Fin 2 × Fin 2)
abbrev K := Fin 4 × Option (Fin 4 × Bool)
abbrev X := Fin 4 × (Fin 4 → ℝ)

-- A constant descriptor and a signed block descriptor; no raw-class number.
def descriptor (b : Fin 4) (j : Fin 2) : ℝ :=
  if j = 0 then 1 else if b.val < 2 then -1 else 1
noncomputable def expansion (p : P) (k : K) (c : Bool) : ℝ :=
  match k.2 with
  | none => 0
  | some q => (if c then 1 else -1) *
      match p with
      | .inl r => if q = r then 1 else 0
      | .inr j => if q.2 || k.1.val % 2 != j.1.val then 0 else descriptor k.1 j.2 / 2
def offset (k : K) (c : Bool) : ℝ :=
  if k.2 = none then (if c then 3 else 2) else 0
def weights (p : P) : ℝ := match p with | .inl _ => 4 | .inr _ => 4
noncomputable def features (k : K) (x : X) : ℝ :=
  if k.1 = x.1 then
    match k.2 with
    | none => 1
    | some q => ContextInterpolation.features (fun y : X => SignedContext.normalized y.2)
        (complexFeatures fourier4) q x
  else 0

example (p : P) (c : Bool) : (∑ k, |expansion p k c|) ≤ weights p := by
  rcases p with ⟨i, b⟩ | ⟨j, h⟩
  · fin_cases i <;> cases c <;> cases b <;>
      norm_num [expansion, descriptor, weights, Fintype.sum_prod_type,
        Fintype.sum_option, Fin.sum_univ_two, Fin.sum_univ_four, Fintype.sum_bool]
  · fin_cases j <;> fin_cases h <;> cases c <;> norm_num [expansion, descriptor, weights, Fintype.sum_prod_type,
      Fintype.sum_option, Fin.sum_univ_two, Fin.sum_univ_four, Fintype.sum_bool]

example : offset ∈ headBudget 12 := by
  intro c
  cases c <;> norm_num [headBudget, offset, Fintype.sum_prod_type,
    Fintype.sum_option, Fin.sum_univ_two, Fin.sum_univ_four, Fintype.sum_bool]

example (x : X) (k : K) : |features k x| ≤ 1 := by
  unfold features
  split_ifs
  · cases k.2 with
    | none => norm_num
    | some q =>
      exact SignedContext.complex_features_bound fourier4 fourier4_unitary
        (fun y : X => y.2) x q
  · norm_num

-- The exact affine preimage charges the full tensor even with signed sharing.
example (inputs : Fin 3 → X) (labels : Fin 3 → Bool) (batch : Finset (Fin 3)) :
    ConvexOn ℝ (ParameterSharing.parameterMap expansion offset ⁻¹' headBudget 16)
      (fun θ => Fallback.batchLoss features inputs labels batch
        (ParameterSharing.parameterMap expansion offset θ)) := by
  have h := (headLoss_convex features (fun s : ↥batch => inputs s.val)
    (fun s : ↥batch => labels s.val) 16).comp_affineMap
      (ParameterSharing.parameterMap expansion offset)
  have hc : 0 ≤ (batch.card : ℝ)⁻¹ := inv_nonneg.mpr (Nat.cast_nonneg _)
  simpa [Fallback.batchLoss, Function.comp_def, smul_eq_mul] using ConvexOn.smul hc h

-- A nonzero shared boundary parameter changes both blocks, with budget slack.
noncomputable def point (p : P) : ℝ := if p = Sum.inr (0, 1) then 1 / 4 else 0
example : point ∈ ParameterSharing.weightedDomain weights 1 := by
  norm_num [ParameterSharing.weightedDomain, weights, point, Fintype.sum_sum_type,
    Fintype.sum_prod_type, Fin.sum_univ_two, Fin.sum_univ_four, Fintype.sum_bool]

end PartitionedBoundaryAudit

#print axioms ParameterSharing.parameterMap_budget
#print axioms ParameterSharing.logits_bound
#print axioms ParameterSharing.batchLoss_convex
#print axioms SignedContext.complex_features_bound
#print axioms SignedContext.normalized_logits

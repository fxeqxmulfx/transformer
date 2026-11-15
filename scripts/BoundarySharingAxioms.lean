import Transformer.Grokking.FlatSpectrum.ConvexDepth.ParameterSharing
import Transformer.Grokking.FlatSpectrum.ConvexDepth.SignedContext

/-!
Prospective task335 audit, before fitting. CM55 and CM51 in
`review/flat_spectrum.md`: replace independent block offsets by a fixed linear
map of input-only block descriptors. All real and imaginary coefficients and
fixed categorical offsets are charged. Feature normalization is a bound;
neither input descriptors nor convexity identify missing raw labels.
-/
open Transformer.Grokking.FlatSpectrum.ConvexDepth
open scoped BigOperators

namespace BoundarySharingAudit

abbrev P := (Fin 4 × Bool) ⊕ Fin 2
abbrev K := Fin 2 × Option (Fin 4 × Bool)
abbrev X := Fin 2 × (Fin 4 → ℝ)

-- A constant descriptor and a signed block descriptor; no raw-class number.
def descriptor (b j : Fin 2) : ℝ := if j = 0 then 1 else if b = 0 then -1 else 1
noncomputable def expansion (p : P) (k : K) (c : Bool) : ℝ :=
  match k.2 with
  | none => 0
  | some q => (if c then 1 else -1) *
      match p with
      | .inl r => if q = r then 1 else 0
      | .inr j => if q.2 then 0 else descriptor k.1 j / 2
def offset (k : K) (c : Bool) : ℝ :=
  if k.2 = none then (if c then 3 else 2) else 0
def weights (p : P) : ℝ := match p with | .inl _ => 2 | .inr _ => 4
noncomputable def features (k : K) (x : X) : ℝ :=
  if k.1 = x.1 then
    match k.2 with
    | none => 1
    | some q => ContextInterpolation.features (fun y : X => SignedContext.normalized y.2)
        (complexFeatures fourier4) q x
  else 0

example (p : P) (c : Bool) : (∑ k, |expansion p k c|) ≤ weights p := by
  rcases p with ⟨i, b⟩ | j
  · fin_cases i <;> cases c <;> cases b <;>
      norm_num [expansion, descriptor, weights, Fintype.sum_prod_type,
        Fintype.sum_option, Fin.sum_univ_two, Fin.sum_univ_four, Fintype.sum_bool]
  · fin_cases j <;> cases c <;> norm_num [expansion, descriptor, weights, Fintype.sum_prod_type,
      Fintype.sum_option, Fin.sum_univ_two, Fin.sum_univ_four, Fintype.sum_bool]

example : offset ∈ headBudget 6 := by
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
noncomputable def point (p : P) : ℝ := if p = Sum.inr 1 then 1 / 4 else 0
example : point ∈ ParameterSharing.weightedDomain weights 1 := by
  norm_num [ParameterSharing.weightedDomain, weights, point, Fintype.sum_sum_type,
    Fintype.sum_prod_type, Fin.sum_univ_two, Fin.sum_univ_four, Fintype.sum_bool]

end BoundarySharingAudit

#print axioms ParameterSharing.parameterMap_budget
#print axioms ParameterSharing.logits_bound
#print axioms ParameterSharing.batchLoss_convex
#print axioms SignedContext.complex_features_bound
#print axioms SignedContext.normalized_logits

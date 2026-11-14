module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.SignedContext.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ComplexHeadBudget

/-!
# A signed unseen-query witness with the complete Fourier head

Source: `review/flat_spectrum.md`, CM51. Four original points omit (1,1).
The fixed row (-1,1,1,0) reconstructs its affine descriptors. A declared affine
reference supplies a missing raw margin; an alternative target fits every
observation and disagrees there. The reference assumption is therefore material.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.SignedContext

open scoped BigOperators
open Landscape.Neural (crossEntropy)

def points : Fin 5 → Fin 2 → ℝ := ![![0, 0], ![0, 1], ![1, 0], ![2, 0], ![1, 1]]
def observed (i : Fin 4) : Fin 5 := ⟨i.val, by omega⟩
def labels : Fin 4 → Fin 4 := ![0, 2, 0, 0]

/-- The signed missing-query row is fixed from input geometry; original rows
are Dirac. This fixture is not a kernel-discovery algorithm. -/
noncomputable def weights (x : Fin 5) (i : Fin 4) : ℝ :=
  if x.val < 4 then (if i.val = x.val then 1 else 0) else exampleRow i

noncomputable def context (x : Fin 5) : Fin 4 → ℝ := normalized (weights x)

/-- An explicit affine reference score model, separate from the target labels. -/
def reference (p : Fin 2 → ℝ) (c : Fin 4) : ℝ :=
  if c = 0 then 4 * (1 - p 1) else if c = 2 then 4 * p 1 else 0

/-- The missing input has an observable signed affine reconstruction, CM51. -/
theorem input_reconstruction :
    (∑ i, weights 4 i = 1) ∧
      (∀ k, ∑ i, weights 4 i * points (observed i) k = points 4 k) ∧
      points 4 ∉ Set.range (fun i => points (observed i)) := by
  refine ⟨?_, ?_, ?_⟩
  · norm_num [weights, exampleRow, Fin.sum_univ_four]
  · intro k
    fin_cases k <;> norm_num [weights, exampleRow, points, observed, Fin.sum_univ_four]
  · rintro ⟨i, hi⟩
    have h0 := congrFun hi 0
    have h1 := congrFun hi 1
    fin_cases i <;> norm_num [points, observed] at *

theorem signed_normalization : weights 4 0 = -1 ∧ normalizer (weights 4) = 3 := by
  norm_num [weights, exampleRow, normalizer, mass, Fin.sum_univ_four]

/-- Actual complete source logits agree with the declared reference at every
observed point. No assumption about the fifth point enters this identity. -/
theorem source_reference (i : Fin 4) :
    headLogits (complexFeatures fourier4)
      (complexCoefficients (complexInitializer fourier4 labels 4)) i =
      reference (points (observed i)) := by
  rw [complexInitializer_logits fourier4 fourier4_unitary]
  funext c
  fin_cases i <;> fin_cases c <;> norm_num [labels, reference, points, observed]

/-- The reference's query residual is zero because its supplied scores are
affine in the reconstructed descriptors, CM51. Other targets need not satisfy it. -/
theorem reference_reconstruction (c : Fin 4) :
    (∑ i, weights 4 i * reference (points (observed i)) c) = reference (points 4) c := by
  fin_cases c <;> norm_num [weights, exampleRow, reference, points, observed, Fin.sum_univ_four]

/-- The complete complex product yields a nonzero raw margin after signed
normalization, CM51. The tensor contains only the four observed labels. -/
theorem query_logits :
    headLogits (ContextInterpolation.features context (complexFeatures fourier4))
      (complexCoefficients (complexInitializer fourier4 labels 4)) 4 =
      fun c => if c = 2 then 4 / 3 else 0 := by
  funext c
  rw [ContextInterpolation.initializer_logits fourier4 fourier4_unitary]
  fin_cases c <;> norm_num [context, normalized, normalizer, mass, weights, exampleRow,
    labels, ContextInterpolation.labelMass, Fin.sum_univ_four]

/-- Ordinary four-class CE, with every class in its denominator, CM51. -/
theorem query_ce :
    crossEntropy
      (headLogits (ContextInterpolation.features context (complexFeatures fourier4))
        (complexCoefficients (complexInitializer fourier4 labels 4)) 4) 2 =
      Real.log (1 + 3 * Real.exp (-(4 / 3))) := by
  rw [query_logits, fixed_margin_ce]
  norm_num

-- The abstract residual and strict-margin premises have nonzero Fourier instances.
example := query_error weights (complexFeatures fourier4)
  (complexCoefficients (complexInitializer fourier4 labels 4))
  (fun i => reference (points (observed i))) (reference (points 4)) 4 2
  (ε := 0) (δ := 0) (by intro i; rw [source_reference]; simp)
  (by rw [reference_reconstruction]; simp)
example := strict_margin weights (complexFeatures fourier4)
  (complexCoefficients (complexInitializer fourier4 labels 4))
  (fun i => reference (points (observed i))) (reference (points 4)) 4 2 0
  (ε := 0) (δ := 0) (by intro i c; rw [source_reference]; simp)
  (by intro c; rw [reference_reconstruction]; simp)
  (by norm_num [reference, points])

def compatibleTarget (x : Fin 5) : Fin 4 := if x = 1 ∨ x = 4 then 2 else 0
def alternativeTarget (x : Fin 5) : Fin 4 := if x = 1 then 2 else 0

/-- Observations and a many-to-one readout do not establish the reference
model at the unseen query. These two targets agree on both but disagree raw. -/
theorem target_ambiguity :
    (∀ i, compatibleTarget (observed i) = labels i ∧
      alternativeTarget (observed i) = labels i) ∧
      compatibleTarget 4 = 2 ∧ alternativeTarget 4 = 0 ∧
      (compatibleTarget 4).val % 2 = (alternativeTarget 4).val % 2 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i
    fin_cases i <;> norm_num [compatibleTarget, alternativeTarget, observed, labels]
  all_goals norm_num [compatibleTarget, alternativeTarget]

/-- The unchanged complete initializer fits within the original charged budget. -/
theorem head_budget :
    complexCoefficients (complexInitializer fourier4 labels 4) ∈ headBudget 32 := by
  convert complexInitializer_budget fourier4 fourier4_unitary labels 4 using 1
  norm_num

example := logits_bound weights (complexFeatures fourier4) 4 (by norm_num : (0 : ℝ) ≤ 1)
  (fun k i => ConvexDepth.complexFeatures_bound fourier4 fourier4_unitary k i) head_budget 2

/-- Original records repeat the first state and have a one-record tail. -/
def records : Fin 5 → Fin 4 := ![0, 0, 1, 2, 3]
def state (x : Fin 5) : Fin 4 := if h : x.val < 4 then ⟨x.val, h⟩ else 0

theorem observed_rows (s : Fin 5) (i : Fin 4) :
    context (observed (records s)) i = if i = state (observed (records s)) then 1 else 0 := by
  apply normalized_observed
  intro j
  fin_cases s <;> fin_cases j <;> norm_num [weights, observed, records, state]

example := ContextInterpolation.batchLoss_convex context (complexFeatures fourier4)
  (fun s => observed (records s)) (fun s => labels (records s)) (Finset.univ : Finset (Fin 5)) 32
example := ContextInterpolation.batch_loss_eq context (complexFeatures fourier4) state
  (fun s => observed (records s)) (fun s => labels (records s)) (Finset.univ : Finset (Fin 5))
  (fun s _ i => observed_rows s i) (complexCoefficients (complexInitializer fourier4 labels 4))
example := ContextInterpolation.batch_fderiv_eq context (complexFeatures fourier4) state
  (fun s => observed (records s)) (fun s => labels (records s)) (Finset.univ : Finset (Fin 5))
  (fun s _ i => observed_rows s i) (complexCoefficients (complexInitializer fourier4 labels 4))
example := ContextInterpolation.batch_fderiv_eq context (complexFeatures fourier4) state
  (fun s => observed (records s)) (fun s => labels (records s)) ({4} : Finset (Fin 5))
  (fun s _ i => observed_rows s i) (complexCoefficients (complexInitializer fourier4 labels 4))

end Transformer.Grokking.FlatSpectrum.ConvexDepth.SignedContext

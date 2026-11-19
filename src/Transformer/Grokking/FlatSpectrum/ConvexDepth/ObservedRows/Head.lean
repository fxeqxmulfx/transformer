module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows.Geometry

/-!
# The unchanged complete complex head and ordinary cross-entropy

Source: `review/convex/moments.md`, CM65. Exact reconstruction preserves the
full coefficient tensor and therefore every raw logit and CE value, including
unseen queries. The unitary capacity bound applies to the reconstructed head.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows

open scoped BigOperators
open LatentPrograms (Parameters features head retained)
open Landscape.Neural (crossEntropy)

variable {J X S C : Type*} [Fintype J] [Fintype S] [Fintype C]
  [DecidableEq S] [DecidableEq C]

/-- Reconstructing a feasible model preserves its entire complex head,
CM65, including all imaginary coefficients and classes at unobserved states. -/
theorem complete_head_eq (U : Matrix S S ℂ) (program : J → X → S) (order : List J)
    (samples : List (X × C)) {δ R : ℝ} {q : Parameters J S C}
    (hq : q ∈ BoundedLatent.domain program order samples δ R) :
    head U 1 (expand (dictionary program samples) (restrict (dictionary program samples) q)) =
      head U 1 q := by
  rw [expand_restrict program order samples hq]

/-- Every original record batch has exactly the same ordinary CE after
coordinate reduction, CM65. Record IDs, duplicates, and a short tail are unchanged. -/
theorem batchLoss_eq {I : Type*} (U : Matrix S S ℂ) (program : J → X → S)
    (order : List J) (samples : List (X × C)) {δ R : ℝ} {q : Parameters J S C}
    (hq : q ∈ BoundedLatent.domain program order samples δ R)
    (inputs : I → X) (labels : I → C) (batch : Finset I) :
    BoundedLatent.batchLoss U program inputs labels batch
        (expand (dictionary program samples) (restrict (dictionary program samples) q)) =
      BoundedLatent.batchLoss U program inputs labels batch q := by
  rw [expand_restrict program order samples hq]

/-- The full flat-source coefficient bound survives exact row elimination,
CM65; compression does not excuse an uncharged complex coefficient. -/
theorem complete_budget (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) {δ R : ℝ}
    {r : Reduced (dictionary program samples)}
    (hr : r ∈ domain (dictionary program samples) (retained program order samples) δ R) :
    head U 1 (expand (dictionary program samples) r) ∈ headBudget (2 * R * Fintype.card S) := by
  exact BoundedLatent.head_budget U hU program order samples (expand_mem program order samples hr)

/-- Ordinary full-vocabulary CE remains convex in all reduced variables,
CM65. The source and dictionary are fixed; unknown rows are still learned jointly. -/
theorem ce_convex (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) (δ R : ℝ)
    (x : X) (y : C) :
    ConvexOn ℝ (domain (dictionary program samples) (retained program order samples) δ R)
      (fun r => crossEntropy (headLogits (features U program)
        (head U 1 (expand (dictionary program samples) r)) x) y) := by
  refine ⟨domain_convex _ _ _ _, ?_⟩
  intro a ha b hb t u ht hu htu
  have h := (BoundedLatent.ce_convex U hU program order samples δ R x y).2
    (expand_mem program order samples ha) (expand_mem program order samples hb) ht hu htu
  simpa only [expand_combination] using h

/-- The actual full or minibatch objective is convex in the reduced
coordinates, CM65, with its original record multiplicities and denominator. -/
theorem batchLoss_convex {I : Type*} (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) (δ R : ℝ)
    (inputs : I → X) (labels : I → C) (batch : Finset I) :
    ConvexOn ℝ (domain (dictionary program samples) (retained program order samples) δ R)
      (fun r => BoundedLatent.batchLoss U program inputs labels batch
        (expand (dictionary program samples) r)) := by
  refine ⟨domain_convex _ _ _ _, ?_⟩
  intro a ha b hb t u ht hu htu
  have h := (BoundedLatent.batchLoss_convex U hU program order samples δ R
    inputs labels batch).2 (expand_mem program order samples ha)
    (expand_mem program order samples hb) ht hu htu
  simpa only [expand_combination] using h

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows

module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent.Initialization

/-!
# Complete complex bounds and convex ordinary CE with learned confidence

Source: `review/convex/moments.md`, CM64. Reuse CM53's actual complex head with
fixed margin one; confidence is already in the joint coefficients. The total
mass cap bounds every complete column. Both full and minibatch CE are convex.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent

open scoped BigOperators
open Landscape.Neural (crossEntropy)
open LatentPrograms (Parameters features head)

variable {J X S C : Type*} [Fintype J] [Fintype S] [Fintype C]
  [DecidableEq S] [DecidableEq C]

/-- A bounded total mass controls every complete realified complex column,
CM64. The bound is independent of program count; storage and work are not. -/
theorem head_budget (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) {δ R : ℝ}
    {q : Parameters J S C} (hq : q ∈ domain program order samples δ R) :
    head U 1 q ∈ headBudget (2 * R * Fintype.card S) := by
  intro c
  rw [Fintype.sum_prod_type]
  calc
    _ ≤ ∑ j, 2 * q.1 j * (Fintype.card S : ℝ) := by
      apply Finset.sum_le_sum
      intro j hj
      apply tableHead_budget U hU (fun s d => 1 * q.2 j s d) (hq.1.1 j) _ c
      intro s d
      rw [one_mul, abs_of_nonneg (hq.1.2.1 j s d)]
      exact coefficient_le_mass program order samples hq.1 j s d
    _ = (2 * Fintype.card S) * mass q := by
      unfold mass
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro j hj
      ring
    _ ≤ (2 * Fintype.card S) * R :=
      mul_le_mul_of_nonneg_left hq.2.2 (by positivity)
    _ = _ := by ring

/-- Ordinary full-vocabulary CE is convex in all joint variables, CM64.
The program family and source code are fixed, while total confidence is learned. -/
theorem ce_convex (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) (δ R : ℝ)
    (x : X) (y : C) : ConvexOn ℝ (domain program order samples δ R)
      (fun q => crossEntropy (headLogits (features U program) (head U 1 q) x) y) := by
  refine ⟨domain_convex program order samples δ R, ?_⟩
  intro a ha b hb t u ht hu htu
  change crossEntropy (headLogits (features U program) (head U 1 (t • a + u • b)) x) y ≤ _
  rw [LatentPrograms.logits_linear U hU]
  exact (ConvexRecall.convex_crossEntropy y).2
    (Set.mem_univ _) (Set.mem_univ _) ht hu htu

/-- CE averaged over original record IDs, CM64. Equal inputs with distinct
IDs remain distinct observations; a short final batch uses its actual size. -/
noncomputable def batchLoss {I : Type*} (U : Matrix S S ℂ) (program : J → X → S)
    (inputs : I → X) (labels : I → C) (batch : Finset I) (q : Parameters J S C) : ℝ :=
  (∑ i ∈ batch, crossEntropy (headLogits (features U program) (head U 1 q) (inputs i))
    (labels i)) / (batch.card : ℝ)

/-- Every actual record batch retains convex ordinary CE, CM64. Taking all
record IDs gives the full objective, without deduplicating repeated examples. -/
theorem batchLoss_convex {I : Type*} (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) (δ R : ℝ)
    (inputs : I → X) (labels : I → C) (batch : Finset I) :
    ConvexOn ℝ (domain program order samples δ R)
      (batchLoss U program inputs labels batch) := by
  refine ⟨domain_convex program order samples δ R, ?_⟩
  intro a ha b hb t u ht hu htu
  let f := fun q i => crossEntropy (headLogits (features U program) (head U 1 q) (inputs i))
    (labels i)
  have h (i : I) : f (t • a + u • b) i ≤ t * f a i + u * f b i :=
    (ce_convex U hU program order samples δ R (inputs i) (labels i)).2 ha hb ht hu htu
  change (∑ i ∈ batch, f (t • a + u • b) i) / (batch.card : ℝ) ≤
    t * ((∑ i ∈ batch, f a i) / (batch.card : ℝ)) +
      u * ((∑ i ∈ batch, f b i) / (batch.card : ℝ))
  calc
    _ ≤ (∑ i ∈ batch, (t * f a i + u * f b i)) / (batch.card : ℝ) :=
      div_le_div_of_nonneg_right (Finset.sum_le_sum (fun i hi => h i)) (Nat.cast_nonneg _)
    _ = _ := by
      rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
      ring

example := head_budget (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  LatentPrograms.exampleProgram [false, true] LatentPrograms.exampleSamples examplePoint_mem

example := ce_convex (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  LatentPrograms.exampleProgram [false, true] LatentPrograms.exampleSamples 1 8 (true, true) true

example := batchLoss_convex (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  LatentPrograms.exampleProgram [false, true] LatentPrograms.exampleSamples 1 8
  (fun i : Bool => (false, i)) id Finset.univ

example := batchLoss_convex (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  LatentPrograms.exampleProgram [false, true] LatentPrograms.exampleSamples 1 8
  (fun i : Bool => (false, i)) id {true}

end Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent

module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.BoxHead

/-!
# Feasible table updates with fixed latent-default rows

Source: `review/flat_spectrum.md`, CM32. Real table coordinates provide a
convex parameterization of the actual complex readout. Clip observed rows
into a fixed box and assign each default its declared row. Every proposal,
including full-batch and minibatch directions, then yields a feasible head.
No target labels enter the feature extractor or the default preservation proof.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open Matrix
open Landscape.Neural (crossEntropy)

variable {I C O Z : Type*}

/-- An entrywise real logit bound, CM32. This is a convex sufficient domain
for the actual complex coefficient budget, rather than a spectral equality. -/
def stateTableBox (A : ℝ) : Set (Matrix I C ℝ) := {T | ∀ s c, |T s c| ≤ A}

/-- All table entries jointly form a convex domain, CM32. -/
theorem stateTableBox_convex (A : ℝ) : Convex ℝ (stateTableBox (I := I) (C := C) A) := by
  intro T hT V hV a b ha hb hab s c
  change |a * T s c + b * V s c| ≤ A
  calc
    _ ≤ |a * T s c| + |b * V s c| := abs_add_le _ _
    _ = a * |T s c| + b * |V s c| := by
      rw [abs_mul, abs_mul, abs_of_nonneg ha, abs_of_nonneg hb]
    _ ≤ a * A + b * A := add_le_add
      (mul_le_mul_of_nonneg_left (hT s c) ha) (mul_le_mul_of_nonneg_left (hV s c) hb)
    _ = A := by rw [← add_mul, hab, one_mul]

/-- Ordinary full-vocabulary CE of the actual complex head remains convex
in the bounded table coordinates, CM32. The unitary code is fixed. -/
theorem tableHead_loss_convex [Fintype I] [DecidableEq I] [Fintype C]
    {S : Type*} [Fintype S] (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (inputs : S → I) (labels : S → C) (A : ℝ) :
    ConvexOn ℝ (stateTableBox A)
      (fun T => headLoss (complexFeatures U) inputs labels
        (complexCoefficients (tableHead U T))) := by
  refine ⟨stateTableBox_convex A, ?_⟩
  intro T _ V _ a b ha hb hab
  simp only [headLoss, tableHead_logits U hU, smul_eq_mul,
    Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro s hs
  have hrow : (a • T + b • V) (inputs s) = a • T (inputs s) + b • V (inputs s) := by
    funext c
    rfl
  rw [hrow]
  exact (ConvexRecall.convex_crossEntropy (labels s)).2
    (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- Explicit scalar clipping used only on proposed observed-state logits,
CM32. Default rows are supplied separately below. -/
def boxClip (A x : ℝ) : ℝ := max (-A) (min A x)

/-- Every clipped scalar lies in the requested nonnegative-radius box. -/
theorem boxClip_bound {A : ℝ} (hA : 0 ≤ A) (x : ℝ) : |boxClip A x| ≤ A := by
  rw [abs_le]
  exact ⟨le_max_left _ _, max_le (by linarith) (min_le_left _ _)⟩

/-- Feasible entries are unchanged by clipping, CM32. -/
theorem boxClip_eq {A x : ℝ} (hx : |x| ≤ A) : boxClip A x = x := by
  rw [boxClip, min_eq_right (abs_le.mp hx).2, max_eq_right (abs_le.mp hx).1]

/-- Project observed proposals coordinatewise and restore fixed default
rows. The defaults are parameters fixed before training, CM32. -/
def boundedTableProposal (A : ℝ) (defaults : Z → C → ℝ)
    (proposal : Matrix (O ⊕ Z) C ℝ) : Matrix (O ⊕ Z) C ℝ :=
  fun s c => match s with
    | Sum.inl o => boxClip A (proposal (Sum.inl o) c)
    | Sum.inr z => defaults z c

/-- Every proposal yields a bounded table when the fixed default rows are
bounded. There is no step-size or gradient-convergence premise, CM32. -/
theorem boundedTableProposal_mem {A : ℝ} (hA : 0 ≤ A) (defaults : Z → C → ℝ)
    (hd : ∀ z c, |defaults z c| ≤ A) (proposal : Matrix (O ⊕ Z) C ℝ) :
    boundedTableProposal A defaults proposal ∈ stateTableBox A := by
  intro s c
  cases s with
  | inl o => exact boxClip_bound hA _
  | inr z => exact hd z c

/-- Feasible tables with the declared default rows are fixed points of
the update projection; this is a retraction, CM32. -/
theorem boundedTableProposal_eq {A : ℝ} (defaults : Z → C → ℝ)
    (T : Matrix (O ⊕ Z) C ℝ) (hT : T ∈ stateTableBox A)
    (hd : ∀ z, T (Sum.inr z) = defaults z) : boundedTableProposal A defaults T = T := by
  ext s c
  cases s with
  | inl o => exact boxClip_eq (hT (Sum.inl o) c)
  | inr z => exact congrFun (hd z).symm c

/-- The box together with affine default equalities, CM32. This is the
native table-coordinate feasible set, transported linearly to complex heads. -/
def defaultTableDomain (A : ℝ) (defaults : Z → C → ℝ) : Set (Matrix (O ⊕ Z) C ℝ) :=
  {T | T ∈ stateTableBox A ∧ ∀ z, T (Sum.inr z) = defaults z}

/-- Fixed default rows preserve joint convexity of the feasible table
domain, CM32. -/
theorem defaultTableDomain_convex (A : ℝ) (defaults : Z → C → ℝ) :
    Convex ℝ (defaultTableDomain (O := O) A defaults) := by
  intro T hT V hV a b ha hb hab
  refine ⟨stateTableBox_convex A hT.1 hV.1 ha hb hab, ?_⟩
  intro z
  funext c
  change a * T (Sum.inr z) c + b * V (Sum.inr z) c = defaults z c
  rw [hT.2 z, hV.2 z, ← add_mul, hab, one_mul]

/-- Every clipped proposal satisfies both bounds and affine defaults.
Feasibility is obtained by the algorithm, rather than assumed of its output. -/
theorem boundedTableProposal_feasible {A : ℝ} (hA : 0 ≤ A) (defaults : Z → C → ℝ)
    (hd : ∀ z c, |defaults z c| ≤ A) (proposal : Matrix (O ⊕ Z) C ℝ) :
    boundedTableProposal A defaults proposal ∈ defaultTableDomain A defaults := by
  refine ⟨boundedTableProposal_mem hA defaults hd proposal, ?_⟩
  intro z
  funext c
  rfl

/-- The actual complex forward pass retains default logits after every
projected proposal, CM32. This also holds for an unbounded proposed step. -/
theorem boundedTableProposal_default [Fintype O] [Fintype Z] [DecidableEq O] [DecidableEq Z]
    (U : Matrix (O ⊕ Z) (O ⊕ Z) ℂ) (hU : U ∈ Matrix.unitaryGroup (O ⊕ Z) ℂ)
    (A : ℝ) (defaults : Z → C → ℝ) (proposal : Matrix (O ⊕ Z) C ℝ) (z : Z) :
    headLogits (complexFeatures U)
      (complexCoefficients (tableHead U (boundedTableProposal A defaults proposal))) (Sum.inr z) =
      defaults z := by
  rw [tableHead_logits U hU]
  rfl

/-- Any projected native proposal fits the actual64N complex-coordinate
budget while keeping every default. No optimizer feasibility is assumed. -/
theorem boundedTableProposal_budget [Fintype O] [Fintype Z] [DecidableEq O] [DecidableEq Z]
    (U : Matrix (O ⊕ Z) (O ⊕ Z) ℂ) (hU : U ∈ Matrix.unitaryGroup (O ⊕ Z) ℂ)
    (defaults : Z → C → ℝ) (hd : ∀ z c, |defaults z c| ≤ 32)
    (proposal : Matrix (O ⊕ Z) C ℝ) :
    complexCoefficients (tableHead U (boundedTableProposal 32 defaults proposal)) ∈
      headBudget (64 * ((Fintype.card O : ℝ) + (Fintype.card Z : ℝ))) := by
  have h := tableHead_native_budget U hU _
    (boundedTableProposal_mem (by norm_num) defaults hd proposal)
  simpa only [Fintype.card_sum, Nat.cast_add] using h

example := tableHead_loss_convex fourier4 fourier4_unitary
  (fun s : Fin 4 => s) (fun s => decide (s.val < 2)) 32

example := boxClip_bound (by norm_num : (0 : ℝ) ≤ 32) 1000

example := boxClip_eq (by norm_num : |(24 : ℝ)| ≤ 32)

example := boundedTableProposal_mem (O := Bool) (A := 32) (by norm_num)
  (fun (_ : Bool) (c : Bool) => if c then (24 : ℝ) else 0)
  (by intro z c; cases c <;> norm_num) (fun _ _ => 1000)

example := boundedTableProposal_eq (O := Bool) (A := 32)
  (fun (_ : Bool) (_ : Bool) => (24 : ℝ)) (fun _ _ => 24)
  (by intro s c; norm_num) (by intro z; rfl)

example := boundedTableProposal_feasible (O := Bool) (A := 32) (by norm_num)
  (fun (_ : Bool) (c : Bool) => if c then 24 else 0)
  (by intro z c; cases c <;> norm_num) (fun _ _ => 1000)

example := boundedTableProposal_default (categoricalFourierOn (Bool ⊕ Bool))
  (categoricalFourierOn_unitary _) 32 (fun _ (c : Bool) => if c then 24 else 0)
  (fun _ _ => 1000) true

example := boundedTableProposal_budget (categoricalFourierOn (Bool ⊕ Bool))
  (categoricalFourierOn_unitary _) (fun _ (c : Bool) => if c then 24 else 0)
  (by intro z c; cases c <;> norm_num) (fun _ _ => 1000)

example : boxClip 32 1000 = 32 ∧ boxClip 32 (-1000) = -32 := by
  norm_num [boxClip]

end Transformer.Grokking.FlatSpectrum.ConvexDepth

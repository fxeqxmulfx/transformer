module

public import Transformer.Grokking.Convex.ConvexDepth.GatedContext.Batch

/-!+# Local mixtures add actual complex-head capacity

Source: `review/convex/moments.md`, CM46. CM45's bank gives identical logits
at two unseen queries under every shared mixture. Fixed input gates and
anchored local mixtures give opposite strict raw margins while preserving
the same observed fit. This is capacity, not unseen-target identification.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.GatedContext

open scoped BigOperators
open Landscape.Neural (crossEntropy)

/-- The fixed anchor weights both context maps equally, CM46. -/
noncomputable def witnessBase : Bool → ℝ :=
  (1 / 2 : ℝ) • ContextMixture.witnessMix false + (1 / 2 : ℝ) • ContextMixture.witnessMix true

/-- The gate uses only the input, with the two unseen queries in different groups. -/
def witnessRoute (x : Fin 4) : Bool := decide (x = 3)

/-- One-hot fixed gates are a special case of overlapping stochastic gates. -/
noncomputable def witnessGate : Fin 4 → Bool → ℝ :=
  ContextInterpolation.pointWeights witnessRoute

theorem witnessBase_mem : witnessBase ∈ ContextMixture.simplex := by
  constructor
  · intro j; cases j <;> norm_num [witnessBase, ContextMixture.witnessMix]
  · norm_num [witnessBase, ContextMixture.witnessMix, Fintype.sum_bool]

theorem witnessGate_mem (x : Fin 4) : witnessGate x ∈ ContextMixture.simplex := by
  exact ⟨ContextInterpolation.pointWeights_nonneg witnessRoute x,
    ContextInterpolation.pointWeights_sum witnessRoute x⟩

theorem witnessLocal_mem : ContextMixture.witnessMix ∈ rows := by
  intro g
  exact ContextMixture.witnessMix_mem g

/-- Every shared mixture gives identical actual logits at the two unseen
queries, CM46, regardless of whether its coefficients are feasible. -/
theorem witness_global_same (a : Bool → ℝ) :
    headLogits (ContextMixture.features ContextMixture.witnessBank a)
      ContextMixture.witnessHead 2 =
    headLogits (ContextMixture.features ContextMixture.witnessBank a)
      ContextMixture.witnessHead 3 := by
  rw [ContextMixture.logits, ContextMixture.logits]
  simp only [Fintype.sum_bool, ContextMixture.witness_bank_logits]
  norm_num [ContextMixture.witnessState]

/-- No shared mixture gives the two requested opposite strict margins,
CM46. This obstruction concerns actual Fourier products. -/
theorem witness_global_obstruction (a : Bool → ℝ) :
    ¬ (headLogits (ContextMixture.features ContextMixture.witnessBank a)
          ContextMixture.witnessHead 2 2 <
        headLogits (ContextMixture.features ContextMixture.witnessBank a)
          ContextMixture.witnessHead 2 0 ∧
      headLogits (ContextMixture.features ContextMixture.witnessBank a)
          ContextMixture.witnessHead 3 0 <
        headLogits (ContextMixture.features ContextMixture.witnessBank a)
          ContextMixture.witnessHead 3 2) := by
  have h0 := congrFun (witness_global_same a) 0
  have h2 := congrFun (witness_global_same a) 2
  intro ⟨ha, hb⟩
  linarith

/-- Actual anchored logits at unseen inputs 2 and 3 are respectively
(15,9) and (9,15) on raw labels (0,2), CM46. The radius is one half. -/
theorem witness_local_logits (reverse : Bool) (c : Fin 4) :
    headLogits (features witnessGate ContextMixture.witnessBank witnessBase (1 / 2)
      ContextMixture.witnessMix) ContextMixture.witnessHead (if reverse then 3 else 2) c =
      if c = 0 then (if reverse then 9 else 15)
      else if c = 2 then (if reverse then 15 else 9) else 0 := by
  rw [logits]
  simp only [ContextMixture.logits, Fintype.sum_bool,
    Pi.add_apply, Pi.smul_apply, smul_eq_mul, ContextMixture.witness_bank_logits]
  cases reverse <;> fin_cases c <;>
    norm_num [witnessBase, witnessGate, ContextInterpolation.pointWeights,
      witnessRoute, ContextMixture.witnessMix, ContextMixture.witnessState,
      ContextMixture.witnessLabels]

/-- Feasible anchored local mixtures achieve both opposite raw margins,
CM46, which the preceding shared-mixture obstruction excludes. -/
theorem witness_local_margins :
    headLogits (features witnessGate ContextMixture.witnessBank witnessBase (1 / 2)
        ContextMixture.witnessMix) ContextMixture.witnessHead 2 2 + 6 =
      headLogits (features witnessGate ContextMixture.witnessBank witnessBase (1 / 2)
        ContextMixture.witnessMix) ContextMixture.witnessHead 2 0 ∧
    headLogits (features witnessGate ContextMixture.witnessBank witnessBase (1 / 2)
        ContextMixture.witnessMix) ContextMixture.witnessHead 3 0 + 6 =
      headLogits (features witnessGate ContextMixture.witnessBank witnessBase (1 / 2)
        ContextMixture.witnessMix) ContextMixture.witnessHead 3 2 := by
  have h := witness_local_logits
  norm_num at h
  norm_num [h]

/-- On either unseen query, the wrong raw-target CE is exactly six larger,
CM46. All four raw classes remain in the denominator. -/
theorem witness_wrong_target_ce (reverse : Bool) :
    crossEntropy (headLogits (features witnessGate ContextMixture.witnessBank witnessBase (1 / 2)
      ContextMixture.witnessMix) ContextMixture.witnessHead (if reverse then 3 else 2))
        (if reverse then 0 else 2) =
    crossEntropy (headLogits (features witnessGate ContextMixture.witnessBank witnessBase (1 / 2)
      ContextMixture.witnessMix) ContextMixture.witnessHead (if reverse then 3 else 2))
        (if reverse then 2 else 0) + 6 := by
  simp only [crossEntropy, witness_local_logits]
  cases reverse <;> norm_num <;> ring

/-- The local model keeps CM45's original batches exactly, for every label
assignment and tail, CM46. Original record IDs are not collapsed. -/
theorem witness_batch_eq (labels : Fin 3 → Fin 4) (batch : Finset (Fin 3)) :
    Fallback.batchLoss (features witnessGate ContextMixture.witnessBank witnessBase (1 / 2)
      ContextMixture.witnessMix) ContextInterpolation.batchInputs labels batch
      ContextMixture.witnessHead =
    Fallback.batchLoss (ContextMixture.features ContextMixture.witnessBank
      (ContextMixture.witnessMix false)) ContextInterpolation.batchInputs labels batch
      ContextMixture.witnessHead := by
  rw [batch_loss_eq witnessGate ContextMixture.witnessBank (complexFeatures fourier4)
    witnessBase (1 / 2) ContextMixture.witnessMix witnessBase_mem.2
    (fun g => (ContextMixture.witnessMix_mem g).2) ContextInterpolation.batchInputs labels batch
    (fun s _ => (witnessGate_mem _).2) (fun s _ => ContextMixture.witness_rows s)]
  rw [ContextMixture.batch_loss_eq ContextMixture.witnessBank (complexFeatures fourier4)
    (ContextMixture.witnessMix false) (ContextMixture.witnessMix_mem false).2
    ContextInterpolation.batchInputs labels batch (fun s _ => ContextMixture.witness_rows s)]

/-- Fitted original targets keep the same small ordinary CE, CM46. -/
theorem witness_fitted_ce :
    Fallback.batchLoss (features witnessGate ContextMixture.witnessBank witnessBase (1 / 2)
      ContextMixture.witnessMix) ContextInterpolation.batchInputs
      (fun s => ContextMixture.witnessLabels (ContextInterpolation.batchInputs s))
      (Finset.univ : Finset (Fin 3)) ContextMixture.witnessHead =
        Real.log (1 + 3 * Real.exp (-24)) := by
  rw [witness_batch_eq]
  exact ContextMixture.witness_fitted_ce false

/-- One wrong target among the three original records adds eight to CE,
CM46; the single-record tail adds the full wrong margin of 24. -/
theorem witness_batch_and_tail_ce :
    Fallback.batchLoss (features witnessGate ContextMixture.witnessBank witnessBase (1 / 2)
      ContextMixture.witnessMix) ContextInterpolation.batchInputs (fun _ => (0 : Fin 4))
      (Finset.univ : Finset (Fin 3)) ContextMixture.witnessHead =
        Real.log (1 + 3 * Real.exp (-24)) + 8 ∧
    Fallback.batchLoss (features witnessGate ContextMixture.witnessBank witnessBase (1 / 2)
      ContextMixture.witnessMix) ContextInterpolation.batchInputs (fun _ => (0 : Fin 4))
      ({2} : Finset (Fin 3)) ContextMixture.witnessHead =
        Real.log (1 + 3 * Real.exp (-24)) + 24 := by
  simp only [witness_batch_eq]
  exact ⟨ContextMixture.witness_batch_ce false, ContextMixture.witness_tail_ce false⟩

end Transformer.Grokking.Convex.ConvexDepth.GatedContext

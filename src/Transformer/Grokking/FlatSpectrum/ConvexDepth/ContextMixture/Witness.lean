module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextMixture.Batch

/-!
# Competing contexts with the same observed complex head

Source: `review/convex/moments.md`, CM45. Both banks preserve two observations,
but choose different states elsewhere. Raw labels 0 and 2 share a readout.
The actual Fourier head has opposite unseen margins under two feasible
mixtures. Original repeated-record and tail losses stay unchanged.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextMixture

open scoped BigOperators
open Landscape.Neural (crossEntropy)

/-- Both maps keep observed inputs 0 and 1; the other inputs use opposite states. -/
def witnessState (j : Bool) (x : Fin 4) : Fin 4 :=
  if x.val < 2 then x else if j then 1 else 0

/-- The bank uses actual complex source coordinates, including imaginary entries. -/
noncomputable def witnessBank (j : Bool) : Fin 4 × Bool → Fin 4 → ℝ :=
  DecoderConfidence.features fourier4 (witnessState j)

/-- A many-to-one raw decoder whose two used labels share reduction modulo two. -/
def witnessLabels (i : Fin 4) : Fin 4 := if i.val % 2 = 0 then 0 else 2

/-- The complete conjugate tensor is fixed while contexts are mixed. -/
noncomputable def witnessHead : Fin 4 × Bool → Fin 4 → ℝ :=
  complexCoefficients (complexInitializer fourier4 witnessLabels 24)

/-- Reversing the preference exchanges the two positive mixture weights. -/
noncomputable def witnessMix (reverse j : Bool) : ℝ := if j = reverse then 3 / 4 else 1 / 4

theorem witnessMix_mem (reverse : Bool) : witnessMix reverse ∈ simplex := by
  constructor
  · intro j
    unfold witnessMix
    split_ifs <;> norm_num
  · cases reverse <;> norm_num [witnessMix, Fintype.sum_bool]

theorem witnessBank_bound (j : Bool) (x : Fin 4) (k : Fin 4 × Bool) :
    |witnessBank j k x| ≤ 1 := by
  exact DecoderConfidence.features_bound fourier4 fourier4_unitary (witnessState j) x k

theorem witnessHead_mem : witnessHead ∈ headBudget 256 := by
  have h := complexInitializer_native_budget fourier4 fourier4_unitary witnessLabels
  norm_num at h
  exact h

theorem witness_rows (s : Fin 3) (j : Bool) (k : Fin 4 × Bool) :
    witnessBank j k (ContextInterpolation.batchInputs s) =
      complexFeatures fourier4 k (ContextInterpolation.batchInputs s) := by
  fin_cases s <;>
    simp [witnessBank, DecoderConfidence.features, witnessState, ContextInterpolation.batchInputs]

/-- Each bank's actual logits come from its fixed declared source state. -/
theorem witness_bank_logits (j : Bool) (x : Fin 4) :
    headLogits (witnessBank j) witnessHead x =
      fun c => if c = witnessLabels (witnessState j x) then 24 else 0 := by
  exact DecoderConfidence.logits fourier4 fourier4_unitary (witnessState j) witnessLabels 24 x

/-- The actual mixed raw logits are 18 and 6, exchanged by reversing the mixture. -/
theorem witness_mixed (reverse : Bool) (c : Fin 4) :
    headLogits (features witnessBank (witnessMix reverse)) witnessHead 2 c =
      if c = 0 then (if reverse then 6 else 18)
      else if c = 2 then (if reverse then 18 else 6) else 0 := by
  rw [logits]
  simp only [Finset.sum_apply, Fintype.sum_bool, Pi.smul_apply, smul_eq_mul, witness_bank_logits]
  cases reverse <;> fin_cases c <;> norm_num [witnessMix, witnessState, witnessLabels]

/-- The two feasible mixtures have opposite strict raw margins at an unseen input. -/
theorem witness_opposite_margins :
    headLogits (features witnessBank (witnessMix false)) witnessHead 2 2 + 12 =
      headLogits (features witnessBank (witnessMix false)) witnessHead 2 0 ∧
    headLogits (features witnessBank (witnessMix true)) witnessHead 2 0 + 12 =
      headLogits (features witnessBank (witnessMix true)) witnessHead 2 2 := by
  norm_num [witness_mixed]

/-- The full-vocabulary wrong-target loss is exactly 12 larger, CM45. -/
theorem witness_wrong_target_ce :
    crossEntropy (headLogits (features witnessBank (witnessMix false)) witnessHead 2) 2 =
      crossEntropy (headLogits (features witnessBank (witnessMix false)) witnessHead 2) 0 + 12 := by
  simp only [crossEntropy, witness_mixed]
  norm_num
  ring

/-- Original IDs 0 and 1 repeat a correct raw target; ID 2 has a wrong one.
The full original batch charges one third of the wrong margin, CM45. -/
theorem witness_batch_ce (reverse : Bool) :
    Fallback.batchLoss (features witnessBank (witnessMix reverse))
      ContextInterpolation.batchInputs (fun _ => (0 : Fin 4)) (Finset.univ : Finset (Fin 3))
      witnessHead = Real.log (1 + 3 * Real.exp (-24)) + 8 := by
  rw [batch_loss_eq witnessBank (complexFeatures fourier4) _ (witnessMix_mem reverse).2
    ContextInterpolation.batchInputs _ Finset.univ (fun s _ => witness_rows s)]
  change Fallback.batchLoss (DecoderConfidence.features fourier4 id)
    ContextInterpolation.batchInputs _ (Finset.univ : Finset (Fin 3)) _ = _
  unfold witnessHead
  rw [DecoderConfidence.batch_ce fourier4 fourier4_unitary id witnessLabels
    ContextInterpolation.batchInputs _ Finset.univ (by simp)]
  norm_num [DecoderConfidence.marginCE, DecoderConfidence.errorRate,
    Fin.sum_univ_three, ContextInterpolation.batchInputs, witnessLabels]

/-- The one-record tail charges the full wrong margin, with the same result
under either mixture, CM45. It does not reuse the full batch's denominator. -/
theorem witness_tail_ce (reverse : Bool) :
    Fallback.batchLoss (features witnessBank (witnessMix reverse))
      ContextInterpolation.batchInputs (fun _ => (0 : Fin 4)) ({2} : Finset (Fin 3))
      witnessHead = Real.log (1 + 3 * Real.exp (-24)) + 24 := by
  rw [batch_loss_eq witnessBank (complexFeatures fourier4) _ (witnessMix_mem reverse).2
    ContextInterpolation.batchInputs _ {2} (fun s _ => witness_rows s)]
  change Fallback.batchLoss (DecoderConfidence.features fourier4 id)
    ContextInterpolation.batchInputs _ ({2} : Finset (Fin 3)) _ = _
  unfold witnessHead
  rw [DecoderConfidence.batch_ce fourier4 fourier4_unitary id witnessLabels
    ContextInterpolation.batchInputs _ {2} (by simp)]
  norm_num [DecoderConfidence.marginCE, DecoderConfidence.errorRate,
    ContextInterpolation.batchInputs, witnessLabels]

/-- The matching original targets are genuinely fitted by both mixtures,
CM45, even though their unseen raw predictions differ. -/
theorem witness_fitted_ce (reverse : Bool) :
    Fallback.batchLoss (features witnessBank (witnessMix reverse))
      ContextInterpolation.batchInputs (fun s => witnessLabels (ContextInterpolation.batchInputs s))
      (Finset.univ : Finset (Fin 3)) witnessHead = Real.log (1 + 3 * Real.exp (-24)) := by
  rw [batch_loss_eq witnessBank (complexFeatures fourier4) _ (witnessMix_mem reverse).2
    ContextInterpolation.batchInputs _ Finset.univ (fun s _ => witness_rows s)]
  change Fallback.batchLoss (DecoderConfidence.features fourier4 id)
    ContextInterpolation.batchInputs _ (Finset.univ : Finset (Fin 3)) _ = _
  unfold witnessHead
  have h := DecoderConfidence.fitted_batch_ce fourier4 fourier4_unitary id witnessLabels
    ContextInterpolation.batchInputs
    (fun s => witnessLabels (ContextInterpolation.batchInputs s))
    Finset.univ (by simp) (fun _ _ => rfl) 24
  norm_num [DecoderConfidence.marginCE] at h
  exact h

-- Every conditional generic result has a nonzero actual complex instance.
example : (0 : Fin 4).val % 2 = (2 : Fin 4).val % 2 := by norm_num
example := features_bound witnessBank (witnessMix_mem false) 2
  (fun j k => witnessBank_bound j 2 k) (0, true)
example := logits_bound witnessBank (witnessMix_mem false) 2 (by norm_num : (0 : ℝ) ≤ 1)
  (fun j k => witnessBank_bound j 2 k) witnessHead_mem 0
example := complex_features_bound fourier4 fourier4_unitary
  (fun j => ContextInterpolation.pointWeights (witnessState j)) (fun _ _ => 1 / 2)
  (witnessMix_mem false) 2
  (fun j => ContextInterpolation.pointWeights_nonneg (witnessState j) 2)
  (fun j => ContextInterpolation.pointWeights_sum (witnessState j) 2)
  (fun _ => by norm_num) (fun _ => by norm_num) (0, true)
example := features_at witnessBank (complexFeatures fourier4) (witnessMix false)
  (witnessMix_mem false).2 0 (by intro j k; exact witness_rows 0 j k) (0, false)
example := ce_le_mix witnessBank (witnessMix_mem false) witnessHead 2 0
example := logits_change_bound witnessBank (witnessMix false) (witnessMix true) 2
  (by norm_num : (0 : ℝ) ≤ 1) (fun j k => witnessBank_bound j 2 k) witnessHead_mem 0
example := ce_quadratic witnessBank (witnessMix false) (witnessMix true) 2 0
  (by norm_num : (0 : ℝ) ≤ 1) (fun j k => witnessBank_bound j 2 k) witnessHead_mem
example := batch_loss_eq witnessBank (complexFeatures fourier4) (witnessMix false)
  (witnessMix_mem false).2 ContextInterpolation.batchInputs (fun _ => (0 : Fin 4))
  (Finset.univ : Finset (Fin 3)) (fun s _ => witness_rows s) witnessHead
example := observed_loss_constant witnessBank (complexFeatures fourier4)
  (witnessMix_mem false) (witnessMix_mem true) ContextInterpolation.batchInputs
  (fun _ => (0 : Fin 4)) ({2} : Finset (Fin 3)) (fun s _ => witness_rows s) witnessHead
example := batch_fderiv_eq witnessBank (complexFeatures fourier4) (witnessMix false)
  (witnessMix_mem false).2 ContextInterpolation.batchInputs (fun _ => (0 : Fin 4))
  (Finset.univ : Finset (Fin 3)) (fun s _ => witness_rows s) witnessHead
example := batch_first_order witnessBank witnessHead ContextInterpolation.batchInputs
  (fun _ => (0 : Fin 4)) ({2} : Finset (Fin 3)) (witnessMix false) (witnessMix true)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextMixture

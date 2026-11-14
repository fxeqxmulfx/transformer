module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.DecoderConfidence.Basic

/-!
# Exact raw CE and declared-label errors on original record batches

Source: `review/flat_spectrum.md`, CM44. The actual complex initializer's
normalized CE is a correct-margin term plus margin times empirical error.
Original record IDs retain multiplicity and short-tail normalization. This
is an identity for arbitrary targets, not an unseen-correctness certificate.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.DecoderConfidence

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {I X C S : Type*} [Fintype I] [DecidableEq I] [Fintype C] [DecidableEq C]

omit [Fintype I] [DecidableEq I] [Fintype C] in
/-- The empirical fraction of original record IDs whose declared raw label
differs from their target, CM44. Repeated inputs still have distinct IDs. -/
noncomputable def errorRate (declared targets : S → C) (batch : Finset S) : ℝ :=
  (batch.card : ℝ)⁻¹ * ∑ s ∈ batch, if declared s = targets s then 0 else 1

omit [DecidableEq I] [DecidableEq C] in
/-- The entire output tensor has convex ordinary batch CE with the state
map and code fixed, CM44. No training of two multiplied factors is asserted. -/
theorem batchLoss_convex (U : Matrix I I ℂ) (state : X → I)
    (inputs : S → X) (targets : S → C) (batch : Finset S) (R : ℝ) :
    ConvexOn ℝ (headBudget R) (Fallback.batchLoss (features U state) inputs targets batch) := by
  have h := headLoss_convex (features U state)
    (fun s : ↥batch => inputs s.val) (fun s : ↥batch => targets s.val) R
  have hc : 0 ≤ (batch.card : ℝ)⁻¹ := inv_nonneg.mpr (Nat.cast_nonneg _)
  refine ⟨headBudget_convex R, ?_⟩
  intro W hW V hV a b ha hb hab
  have hm := mul_le_mul_of_nonneg_left (h.2 hW hV ha hb hab) hc
  change (batch.card : ℝ)⁻¹ * _ ≤ a * ((batch.card : ℝ)⁻¹ * _) +
    b * ((batch.card : ℝ)⁻¹ * _)
  calc
    _ ≤ (batch.card : ℝ)⁻¹ * (a * headLoss _ _ _ W + b * headLoss _ _ _ V) := hm
    _ = _ := by ring

/-- Exact normalized raw CE of the complete complex head on any nonempty
batch, CM44. The error term uses the actual target of every original record. -/
theorem batch_ce (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (state : X → I) (decoder : I → C) (inputs : S → X) (targets : S → C)
    (batch : Finset S) (hne : batch.Nonempty) (a : ℝ) :
    Fallback.batchLoss (features U state) inputs targets batch
      (complexCoefficients (complexInitializer U decoder a)) =
      marginCE (Fintype.card C) a +
        a * errorRate (fun s => decoder (state (inputs s))) targets batch := by
  have hcard : (batch.card : ℝ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr hne)
  unfold Fallback.batchLoss headLoss
  rw [Finset.sum_coe_sort batch (fun s : S => crossEntropy
    (headLogits (features U state) (complexCoefficients (complexInitializer U decoder a))
      (inputs s)) (targets s))]
  simp only [initializer_ce U hU, Finset.sum_add_distrib,
    Finset.sum_const, nsmul_eq_mul]
  have hs : (∑ s ∈ batch, if decoder (state (inputs s)) = targets s then (0 : ℝ) else a) =
      a * ∑ s ∈ batch, if decoder (state (inputs s)) = targets s then (0 : ℝ) else 1 := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro s _
    split_ifs <;> ring
  rw [hs, mul_add, ← mul_assoc, inv_mul_cancel₀ hcard, one_mul]
  unfold errorRate
  ring

omit [Fintype I] [DecidableEq I] [Fintype C] in
/-- Exact observed raw fit gives zero empirical error, CM44. This premise
concerns the specified batch only, and supplies no held-out labels. -/
theorem errorRate_zero (declared targets : S → C) (batch : Finset S)
    (hfit : ∀ s ∈ batch, declared s = targets s) : errorRate declared targets batch = 0 := by
  unfold errorRate
  have hsum : (∑ s ∈ batch, if declared s = targets s then (0 : ℝ) else 1) = 0 := by
    apply Finset.sum_eq_zero
    intro s hs
    rw [ite_eq_left (hfit s hs)]
  rw [hsum, mul_zero]

/-- An exactly fitted nonempty original batch has the correct-margin CE,
CM44. A different target batch must satisfy its own fit premise. -/
theorem fitted_batch_ce (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (state : X → I) (decoder : I → C) (inputs : S → X) (targets : S → C)
    (batch : Finset S) (hne : batch.Nonempty)
    (hfit : ∀ s ∈ batch, decoder (state (inputs s)) = targets s) (a : ℝ) :
    Fallback.batchLoss (features U state) inputs targets batch
      (complexCoefficients (complexInitializer U decoder a)) = marginCE (Fintype.card C) a := by
  rw [batch_ce U hU state decoder inputs targets batch hne,
    errorRate_zero _ _ _ hfit, mul_zero, add_zero]

/-- Original IDs0 and1 repeat input0; ID2 is the distinct input4, CM44. -/
def batchInputs (s : Fin 3) : Fin 6 := if s = 2 then 4 else 0

/-- The full repeated-input batch has one wrong declared raw label, CM44. -/
theorem witness_error :
    errorRate (fun s => witnessDecoder (witnessState (batchInputs s)))
      (fun _ => false) (Finset.univ : Finset (Fin 3)) = 1 / 3 := by
  norm_num [errorRate, Fin.sum_univ_three, batchInputs, witnessState, witnessDecoder]

/-- The short final minibatch has error fraction one, not one third, CM44. -/
theorem witness_tail_error :
    errorRate (fun s => witnessDecoder (witnessState (batchInputs s)))
      (fun _ => false) ({2} : Finset (Fin 3)) = 1 := by
  norm_num [errorRate, batchInputs, witnessState, witnessDecoder]

/-- The actual nonzero Fourier head and repeated targets instantiate the
full CE/error identity, CM44: one wrong answer costs a third of margin24. -/
theorem witness_batch_ce :
    Fallback.batchLoss (features fourier4 witnessState) batchInputs (fun _ => false)
      (Finset.univ : Finset (Fin 3))
      (complexCoefficients (complexInitializer fourier4 witnessDecoder 24)) =
      Real.log (1 + Real.exp (-24)) + 8 := by
  rw [batch_ce fourier4 fourier4_unitary witnessState witnessDecoder batchInputs
    (fun _ => false) Finset.univ (by simp), witness_error]
  norm_num [marginCE]

/-- The actual one-record final minibatch charges the whole wrong margin,
CM44. It does not reuse the full batch's denominator. -/
theorem witness_tail_ce :
    Fallback.batchLoss (features fourier4 witnessState) batchInputs (fun _ => false)
      ({2} : Finset (Fin 3))
      (complexCoefficients (complexInitializer fourier4 witnessDecoder 24)) =
      Real.log (1 + Real.exp (-24)) + 24 := by
  rw [batch_ce fourier4 fourier4_unitary witnessState witnessDecoder batchInputs
    (fun _ => false) {2} (by simp), witness_tail_error]
  norm_num [marginCE]

/-- A genuinely fitted repeated batch has the correct confidence loss at
margin24, CM44. Its targets are the declared Boolean decoder values. -/
theorem witness_fitted_ce :
    Fallback.batchLoss (features fourier4 witnessState) batchInputs
      (fun s => witnessDecoder (witnessState (batchInputs s))) (Finset.univ : Finset (Fin 3))
      (complexCoefficients (complexInitializer fourier4 witnessDecoder 24)) =
      Real.log (1 + Real.exp (-24)) := by
  rw [fitted_batch_ce fourier4 fourier4_unitary witnessState witnessDecoder batchInputs
    _ Finset.univ (by simp) (fun _ _ => rfl)]
  norm_num [marginCE]

example := batchLoss_convex fourier4 witnessState batchInputs (fun _ => false)
  (Finset.univ : Finset (Fin 3)) 256
example := batch_ce fourier4 fourier4_unitary witnessState witnessDecoder batchInputs
  (fun _ => false) (Finset.univ : Finset (Fin 3)) (by simp) 24
example := batch_ce fourier4 fourier4_unitary witnessState witnessDecoder batchInputs
  (fun _ => false) ({2} : Finset (Fin 3)) (by simp) 1
example := errorRate_zero (fun s => witnessDecoder (witnessState (batchInputs s)))
  (fun s => witnessDecoder (witnessState (batchInputs s)))
  (Finset.univ : Finset (Fin 3)) (fun _ _ => rfl)
example := fitted_batch_ce fourier4 fourier4_unitary witnessState witnessDecoder batchInputs
  (fun s => witnessDecoder (witnessState (batchInputs s)))
  (Finset.univ : Finset (Fin 3)) (by simp) (fun _ _ => rfl) 24

end Transformer.Grokking.FlatSpectrum.ConvexDepth.DecoderConfidence

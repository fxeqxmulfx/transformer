import Transformer.GPTMini.Sparsemax.QKIndependentDirections

/-!
# Incorrect outputs are not matrix minima on independent inputs

Derived from arXiv:1602.02068v2, §2.2 and §2.5, and the actual
shared linear Q/K matrices and normalized dot product at `73f8a0b`.
With independent inputs and the explicit anchored unit-key restriction,
a wrong ordinary output has no zero joint matrix derivative and is not
a local minimum of its squared output error. The affine decoder lift
starts at the given key matrix and realizes every nearby key row.

Local minimality is transported through that continuous lift, so sparsemax
differentiability at an inactive support boundary is not assumed. The
output target is ordinary task data, not an attention route. The readout
is one unrotated row before XSA and output projection; this does not
exclude gradient cancellation between multiple rows or give a uniform
gradient lower bound. Arbitrary dependent embeddings are not covered;
the input-width obstruction is proved in `InputDecoder`.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Wrong ordinary outputs exclude zero joint Q/K matrix derivatives for
any independent input row. Source: the derived input accessibility and
anchor restrictions for §2.5 of arXiv:1602.02068v2 at `73f8a0b`. -/
theorem qkIndependentSquaredError_no_zero_projection_derivative {F h d N : ℕ}
    (inputs : Fin (d + 1 + N) → (Fin F → ℝ))
    (queries columns : Fin F → EucSpace h) (transverse : EucSpace h)
    (frame : Module.Basis (Fin d) ℝ E) (base : E) (scales : Fin d → ℝ)
    (ordinaryValues : Fin N → E) (eps cap : ℝ) (parameters : Fin (d + 1) → ℝ)
    (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N)) (target : E)
    (hi : LinearIndependent ℝ inputs) (hq : ‖projectionEvaluation inputs queries i‖ = 1)
    (ht : ‖transverse‖ = 1) (ho : inner (𝕜 := ℝ) (projectionEvaluation inputs queries i) transverse = 0)
    (he : eps ≤ 1) (hc : 0 < cap) (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i)
    (hkeys : projectionEvaluation inputs columns = qkAnchoredKeys
      (projectionEvaluation inputs queries i) transverse cap parameters ordinary)
    (hbad : frozenValueReadout (anchoredValues frame base scales ordinaryValues)
      (sparseWeights (projectedQKScores (Real.log (qkAnchorGain cap ordinary))
        eps queries columns inputs i) i) ≠ target) :
    ¬ HasFDerivAt (𝕜 := ℝ) (projectedSquaredLoss (anchoredValues frame base scales ordinaryValues)
      (Real.log (qkAnchorGain cap ordinary)) eps inputs i target) 0 (queries, columns) := by
  let output := frozenValueReadout (anchoredValues frame base scales ordinaryValues)
    (sparseWeights (projectedQKScores (Real.log (qkAnchorGain cap ordinary))
      eps queries columns inputs i) i)
  unfold projectedSquaredLoss
  exact qkIndependentTaskLoss_no_zero_projection_derivative inputs queries columns transverse
    frame base scales ordinaryValues eps cap parameters ordinary i (squaredReadoutLoss target)
    (2 • innerSL ℝ (output - target)) hi hq ht ho he hc hcap hvisible hkeys
    (squaredReadoutLoss_gradient_ne_zero target output hbad)
    (squaredReadoutLoss_hasFDerivAt target output)

/-- Nonstandard sparse inputs inhabit every wrong-output matrix premise.
Source context: arXiv:1602.02068v2, §2.5, derived matrix restriction. -/
example : ¬ HasFDerivAt (𝕜 := ℝ) (projectedSquaredLoss
    (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
    (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0))) (1 / 1000000) mixedRowInputs 2 (0 : ℝ)) 0
    (mixedQKQueries, mixedQKKeys (fun _ => 0)) := by
  apply qkIndependentSquaredError_no_zero_projection_derivative mixedRowInputs mixedQKQueries
    (mixedQKKeys (fun _ => 0)) qkTransverseExample (Module.Basis.singleton (Fin 1) ℝ)
    0 (fun _ => 0) (fun _ : Fin 1 => 7) (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0)
    (fun _ : Fin 1 => 0) 2 0 mixedRowInputs_independent
    (by rw [mixedQKQueries_row]; exact qkFrame_example.1) qkFrame_example.2.1
    (by rw [mixedQKQueries_row]; exact qkFrame_example.2.2) (by norm_num) (by norm_num)
    (by norm_num) (fun a => Fin.le_last (Fin.castAdd 1 a))
    (by rw [mixedQKQueries_row]; exact mixedQKKeys_row _)
  rw [mixedQKReadout_initial]
  norm_num

/-- Incorrect outputs are not local minima in the actual joint Q/K
matrices on independent inputs. Source: the derived accessibility lift
for §2.5 of arXiv:1602.02068v2, with shared projections at `73f8a0b`;
support-boundary differentiability is not assumed. -/
theorem qkIndependentSquaredError_not_isLocalMin_projections {F h d N : ℕ}
    (inputs : Fin (d + 1 + N) → (Fin F → ℝ))
    (queries columns : Fin F → EucSpace h) (transverse : EucSpace h)
    (frame : Module.Basis (Fin d) ℝ E) (base : E) (scales : Fin d → ℝ)
    (ordinaryValues : Fin N → E) (eps cap : ℝ) (parameters : Fin (d + 1) → ℝ)
    (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N)) (target : E)
    (hi : LinearIndependent ℝ inputs) (hq : ‖projectionEvaluation inputs queries i‖ = 1)
    (ht : ‖transverse‖ = 1) (ho : inner (𝕜 := ℝ) (projectionEvaluation inputs queries i) transverse = 0)
    (he : eps ≤ 1) (hc : 0 < cap) (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i)
    (hkeys : projectionEvaluation inputs columns = qkAnchoredKeys
      (projectionEvaluation inputs queries i) transverse cap parameters ordinary)
    (hbad : frozenValueReadout (anchoredValues frame base scales ordinaryValues)
      (sparseWeights (projectedQKScores (Real.log (qkAnchorGain cap ordinary))
        eps queries columns inputs i) i) ≠ target) :
    ¬ IsLocalMin (projectedSquaredLoss (anchoredValues frame base scales ordinaryValues)
      (Real.log (qkAnchorGain cap ordinary)) eps inputs i target) (queries, columns) := by
  intro hm
  obtain ⟨decoder, hd⟩ := inputDecoder_exists inputs hi
  have hm0 : IsLocalMin (projectedSquaredLoss (anchoredValues frame base scales ordinaryValues)
      (Real.log (qkAnchorGain cap ordinary)) eps inputs i target)
      (queries, inputProjectionUpdate inputs decoder columns (projectionEvaluation inputs columns)) := by
    simpa only [inputProjectionUpdate_center] using hm
  have hm' := hm0.comp_continuous
    (g := fun keys => (queries, inputProjectionUpdate inputs decoder columns keys))
    (b := projectionEvaluation inputs columns)
    (continuousAt_const.prodMk (inputProjectionUpdate_continuousAt inputs decoder columns _))
  have hf : (projectedSquaredLoss (anchoredValues frame base scales ordinaryValues)
      (Real.log (qkAnchorGain cap ordinary)) eps inputs i target ∘
      fun keys => (queries, inputProjectionUpdate inputs decoder columns keys)) =
      (fun keys => squaredReadoutLoss target (frozenValueReadout
        (anchoredValues frame base scales ordinaryValues) (sparseWeights (fun n => score
          (Real.log (qkAnchorGain cap ordinary)) eps (projectionEvaluation inputs queries i) (keys n)) i))) := by
    funext keys
    simp only [Function.comp_def, projectedSquaredLoss]
    rw [projectedQKScores_inputProjectionUpdate _ _ _ _ _ _ hd]
  rw [hf, hkeys] at hm'
  rw [projectedQKScores_evaluation, hkeys] at hbad
  exact qkAnchoredSquaredError_not_isLocalMin (projectionEvaluation inputs queries i)
    transverse frame base scales ordinaryValues eps cap parameters ordinary i target
    hq ht ho he hc hcap hvisible hbad hm'

/-- A wrong sparse output on mixed inputs satisfies all local-minimum premises.
Source context: arXiv:1602.02068v2, §2.5, derived joint matrix result. -/
example : ¬ IsLocalMin (projectedSquaredLoss
    (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
    (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0))) (1 / 1000000) mixedRowInputs 2 (0 : ℝ))
    (mixedQKQueries, mixedQKKeys (fun _ => 0)) := by
  apply qkIndependentSquaredError_not_isLocalMin_projections mixedRowInputs mixedQKQueries
    (mixedQKKeys (fun _ => 0)) qkTransverseExample (Module.Basis.singleton (Fin 1) ℝ)
    0 (fun _ => 0) (fun _ : Fin 1 => 7) (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0)
    (fun _ : Fin 1 => 0) 2 0 mixedRowInputs_independent
    (by rw [mixedQKQueries_row]; exact qkFrame_example.1) qkFrame_example.2.1
    (by rw [mixedQKQueries_row]; exact qkFrame_example.2.2) (by norm_num) (by norm_num)
    (by norm_num) (fun a => Fin.le_last (Fin.castAdd 1 a))
    (by rw [mixedQKQueries_row]; exact mixedQKKeys_row _)
  rw [mixedQKReadout_initial]
  norm_num

/-- The finite zero-loss correction changes an actual shared key matrix.
Source: the derived squared-error correction for §2.2 and §2.5 of
arXiv:1602.02068v2, with matrix evaluation and QKNorm at `73f8a0b`. -/
theorem mixedQKCorrection_changes_key_matrix :
    mixedQKKeys correctionAnchorParameters ≠ mixedQKKeys (fun _ => 0) := by
  intro heq
  have hz := mixedQKCorrection_zero_loss
  rw [heq, mixedQKCorrection_initial_loss] at hz
  norm_num at hz

end Transformer.GPTMini.Sparsemax

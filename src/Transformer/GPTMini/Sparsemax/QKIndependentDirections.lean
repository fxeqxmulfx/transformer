import Transformer.GPTMini.Sparsemax.MixedInputQK

/-!
# Ordinary task directions in shared matrices on independent inputs

Derived from the active-support Jacobian in arXiv:1602.02068v2,
§2.5, and shared Q/K projections and QKNorm at `73f8a0b`.
The inputs may be any linearly independent vectors. Their continuous
linear decoder exists by `inputDecoder_exists`, and the affine lift
starts at the actual current key matrix. Thus a nonzero ordinary output
derivative cannot become a zero derivative in every shared key-matrix
direction, or in every joint Q/K direction, at the restricted key family.

The key-family, unit-frame, cap and visible value-anchor conditions are
explicit architectural restrictions. The decoder is proved from input
independence and is not an additional assumed solution. Values and inputs
are fixed along the matrix perturbation. This is an unrotated single-row
readout before XSA and output projection; shared-row losses may cancel.
Independence requires the row size to fit the input width, as proved by
`inputIndependent_width_bound`. No global convergence claim is made.
The current matrix may have an arbitrary component in directions unseen
by the decoder. The lift retains that component and still passes through
the current point, so the theorem does not assume a special matrix inverse.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Independent inputs lift ordinary nonzero task directions to the actual
current shared key matrix. Source: the derived accessibility condition
for §2.5 of arXiv:1602.02068v2 and the projections at `73f8a0b`. -/
theorem qkIndependentTaskLoss_no_zero_key_projection_derivative {F h d N : ℕ}
    (inputs : Fin (d + 1 + N) → (Fin F → ℝ))
    (queries columns : Fin F → EucSpace h) (transverse : EucSpace h)
    (frame : Module.Basis (Fin d) ℝ E) (base : E) (scales : Fin d → ℝ)
    (ordinaryValues : Fin N → E) (eps cap : ℝ) (parameters : Fin (d + 1) → ℝ)
    (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N)) (loss : E → ℝ) (gradient : E →L[ℝ] ℝ)
    (hi : LinearIndependent ℝ inputs) (hq : ‖projectionEvaluation inputs queries i‖ = 1)
    (ht : ‖transverse‖ = 1) (ho : inner (𝕜 := ℝ) (projectionEvaluation inputs queries i) transverse = 0)
    (he : eps ≤ 1) (hc : 0 < cap) (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i)
    (hkeys : projectionEvaluation inputs columns = qkAnchoredKeys
      (projectionEvaluation inputs queries i) transverse cap parameters ordinary) (hg : gradient ≠ 0)
    (hl : HasFDerivAt (𝕜 := ℝ) loss gradient (frozenValueReadout
      (anchoredValues frame base scales ordinaryValues) (sparseWeights (projectedQKScores
        (Real.log (qkAnchorGain cap ordinary)) eps queries columns inputs i) i))) :
    ¬ HasFDerivAt (𝕜 := ℝ) (fun k => loss (frozenValueReadout
      (anchoredValues frame base scales ordinaryValues) (sparseWeights (projectedQKScores
        (Real.log (qkAnchorGain cap ordinary)) eps queries k inputs i) i))) 0 columns := by
  intro hz
  obtain ⟨decoder, hd⟩ := inputDecoder_exists inputs hi
  have hz' : HasFDerivAt (𝕜 := ℝ) (fun k => loss (frozenValueReadout
      (anchoredValues frame base scales ordinaryValues) (sparseWeights (projectedQKScores
        (Real.log (qkAnchorGain cap ordinary)) eps queries k inputs i) i))) 0
      (inputProjectionUpdate inputs decoder columns (projectionEvaluation inputs columns)) := by
    simpa only [inputProjectionUpdate_center] using hz
  have hc' := hz'.comp (projectionEvaluation inputs columns)
    (inputProjectionUpdate_differentiableAt inputs decoder columns
      (projectionEvaluation inputs columns)).hasFDerivAt
  have hf : ((fun k => loss (frozenValueReadout
      (anchoredValues frame base scales ordinaryValues) (sparseWeights (projectedQKScores
        (Real.log (qkAnchorGain cap ordinary)) eps queries k inputs i) i))) ∘
      inputProjectionUpdate inputs decoder columns) = (fun keys => loss (frozenValueReadout
        (anchoredValues frame base scales ordinaryValues) (sparseWeights (fun n => score
          (Real.log (qkAnchorGain cap ordinary)) eps (projectionEvaluation inputs queries i) (keys n)) i))) := by
    funext keys
    simp only [Function.comp_def]
    rw [projectedQKScores_inputProjectionUpdate _ _ _ _ _ _ hd]
  rw [hf, ContinuousLinearMap.zero_comp, hkeys] at hc'
  rw [projectedQKScores_evaluation, hkeys] at hl
  exact qkAnchoredTaskLoss_no_zero_key_derivative (projectionEvaluation inputs queries i)
    transverse frame base scales ordinaryValues eps cap parameters ordinary i loss gradient
    hq ht ho he hc hcap hvisible hg hl hc'

/-- Mixed, nonorthogonal inputs satisfy every shared-key task premise.
Source context: arXiv:1602.02068v2, §2.5, derived matrix accessibility. -/
example : ¬ HasFDerivAt (𝕜 := ℝ) (fun k : Fin 3 → EucSpace 2 => frozenValueReadout
    (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
    (sparseWeights (projectedQKScores (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0)))
      (1 / 1000000) mixedQKQueries k mixedRowInputs 2) 2)) 0 (mixedQKKeys (fun _ => 0)) := by
  apply qkIndependentTaskLoss_no_zero_key_projection_derivative mixedRowInputs mixedQKQueries
    (mixedQKKeys (fun _ => 0)) qkTransverseExample (Module.Basis.singleton (Fin 1) ℝ)
    0 (fun _ => 0) (fun _ : Fin 1 => 7) (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0)
    (fun _ : Fin 1 => 0) 2 (fun x : ℝ => x) (ContinuousLinearMap.id ℝ ℝ)
    mixedRowInputs_independent (by rw [mixedQKQueries_row]; exact qkFrame_example.1)
    qkFrame_example.2.1 (by rw [mixedQKQueries_row]; exact qkFrame_example.2.2)
    (by norm_num) (by norm_num) (by norm_num) (fun a => Fin.le_last (Fin.castAdd 1 a))
    (by rw [mixedQKQueries_row]; exact mixedQKKeys_row _)
  · intro hz
    have hn := congrArg (fun f : ℝ →L[ℝ] ℝ => f 1) hz
    norm_num at hn
  · exact (ContinuousLinearMap.id ℝ ℝ).hasFDerivAt

/-- The same ordinary derivative cannot vanish in all joint Q/K columns.
Source: the derived input-accessibility extension of §2.5 of
arXiv:1602.02068v2 through shared matrices and QKNorm at `73f8a0b`. -/
theorem qkIndependentTaskLoss_no_zero_projection_derivative {F h d N : ℕ}
    (inputs : Fin (d + 1 + N) → (Fin F → ℝ))
    (queries columns : Fin F → EucSpace h) (transverse : EucSpace h)
    (frame : Module.Basis (Fin d) ℝ E) (base : E) (scales : Fin d → ℝ)
    (ordinaryValues : Fin N → E) (eps cap : ℝ) (parameters : Fin (d + 1) → ℝ)
    (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N)) (loss : E → ℝ) (gradient : E →L[ℝ] ℝ)
    (hi : LinearIndependent ℝ inputs) (hq : ‖projectionEvaluation inputs queries i‖ = 1)
    (ht : ‖transverse‖ = 1) (ho : inner (𝕜 := ℝ) (projectionEvaluation inputs queries i) transverse = 0)
    (he : eps ≤ 1) (hc : 0 < cap) (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i)
    (hkeys : projectionEvaluation inputs columns = qkAnchoredKeys
      (projectionEvaluation inputs queries i) transverse cap parameters ordinary) (hg : gradient ≠ 0)
    (hl : HasFDerivAt (𝕜 := ℝ) loss gradient (frozenValueReadout
      (anchoredValues frame base scales ordinaryValues) (sparseWeights (projectedQKScores
        (Real.log (qkAnchorGain cap ordinary)) eps queries columns inputs i) i))) :
    ¬ HasFDerivAt (𝕜 := ℝ) (fun p => loss (frozenValueReadout
      (anchoredValues frame base scales ordinaryValues) (sparseWeights (projectedQKScores
        (Real.log (qkAnchorGain cap ordinary)) eps p.1 p.2 inputs i) i))) 0 (queries, columns) := by
  intro hz
  have hd := (hasFDerivAt_const (𝕜 := ℝ) queries columns).prodMk (hasFDerivAt_id columns)
  have hk : HasFDerivAt (𝕜 := ℝ) (fun k => loss (frozenValueReadout
      (anchoredValues frame base scales ordinaryValues) (sparseWeights (projectedQKScores
        (Real.log (qkAnchorGain cap ordinary)) eps queries k inputs i) i))) 0 columns := by
    simpa only [Function.comp_def, ContinuousLinearMap.zero_comp] using hz.comp columns hd
  exact qkIndependentTaskLoss_no_zero_key_projection_derivative inputs queries columns transverse
    frame base scales ordinaryValues eps cap parameters ordinary i loss gradient hi hq ht ho
    he hc hcap hvisible hkeys hg hl hk

/-- All joint-matrix premises hold on the mixed sparse scalar example.
Source context: arXiv:1602.02068v2, §2.5, derived shared-matrix lift. -/
example : ¬ HasFDerivAt (𝕜 := ℝ) (fun p : (Fin 3 → EucSpace 2) × (Fin 3 → EucSpace 2) =>
    frozenValueReadout (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0
      (fun _ => 0) (fun _ : Fin 1 => 7)) (sparseWeights (projectedQKScores
        (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0))) (1 / 1000000)
        p.1 p.2 mixedRowInputs 2) 2)) 0 (mixedQKQueries, mixedQKKeys (fun _ => 0)) := by
  apply qkIndependentTaskLoss_no_zero_projection_derivative mixedRowInputs mixedQKQueries
    (mixedQKKeys (fun _ => 0)) qkTransverseExample (Module.Basis.singleton (Fin 1) ℝ)
    0 (fun _ => 0) (fun _ : Fin 1 => 7) (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0)
    (fun _ : Fin 1 => 0) 2 (fun x : ℝ => x) (ContinuousLinearMap.id ℝ ℝ)
    mixedRowInputs_independent (by rw [mixedQKQueries_row]; exact qkFrame_example.1)
    qkFrame_example.2.1 (by rw [mixedQKQueries_row]; exact qkFrame_example.2.2)
    (by norm_num) (by norm_num) (by norm_num) (fun a => Fin.le_last (Fin.castAdd 1 a))
    (by rw [mixedQKQueries_row]; exact mixedQKKeys_row _)
  · intro hz
    have hn := congrArg (fun f : ℝ →L[ℝ] ℝ => f 1) hz
    norm_num at hn
  · exact (ContinuousLinearMap.id ℝ ℝ).hasFDerivAt

end Transformer.GPTMini.Sparsemax

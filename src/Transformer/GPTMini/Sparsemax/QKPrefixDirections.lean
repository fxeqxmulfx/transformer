import Transformer.GPTMini.Sparsemax.QKPrefixMatrix

/-!
# Nonzero ordinary matrix directions with only anchor accessibility

Derived from arXiv:1602.02068v2, §2.5, and shared projections and
QKNorm at `73f8a0b`. A decoder isolates the `d + 1` anchors and kills
ordinary inputs. A differentiable matrix path then realizes every finite
anchor-score change through the actual current key matrix. A nonzero
ordinary output derivative cannot vanish in all key columns or joint
Q/K columns. Full input independence and a context-width bound are absent.

The partial decoder has a proved implementation using dedicated channels
in `PrefixInputs`. Its hypotheses are inhabited by `LongContextQK`, where
one hundred ordinary inputs share one embedding channel. The unit frame,
key-family restriction, active value anchors and single unrotated row
before XSA and output projection remain explicit. The theorem concerns
the ordinary task derivative and introduces no attention-route target.
It does not assert that different row losses cannot cancel these directions.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- A nonzero ordinary task derivative reaches the current shared key
matrix through a partial anchor decoder. Source: the derived weaker
accessibility restriction for §2.5 of arXiv:1602.02068v2 at `73f8a0b`. -/
theorem qkPrefixTaskLoss_no_zero_key_projection_derivative {F h d N : ℕ}
    (inputs : Fin (d + 1 + N) → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin (d + 1) → ℝ))
    (queries columns : Fin F → EucSpace h) (transverse : EucSpace h)
    (frame : Module.Basis (Fin d) ℝ E) (base : E) (scales : Fin d → ℝ)
    (ordinaryValues : Fin N → E) (eps cap : ℝ) (parameters : Fin (d + 1) → ℝ)
    (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N)) (loss : E → ℝ) (gradient : E →L[ℝ] ℝ)
    (hd : ∀ a, decoder (inputs (Fin.castAdd N a)) = basis a)
    (hzero : ∀ n, decoder (inputs (Fin.natAdd (d + 1) n)) = 0)
    (hq : ‖projectionEvaluation inputs queries i‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) (projectionEvaluation inputs queries i) transverse = 0)
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
  have hz' : HasFDerivAt (𝕜 := ℝ) (fun k => loss (frozenValueReadout
      (anchoredValues frame base scales ordinaryValues) (sparseWeights (projectedQKScores
        (Real.log (qkAnchorGain cap ordinary)) eps queries k inputs i) i))) 0
      (qkPrefixMatrix inputs decoder columns (projectionEvaluation inputs queries i)
        transverse cap parameters ordinary) := by
    simpa only [qkPrefixMatrix_center inputs decoder columns _ transverse cap parameters ordinary hkeys] using hz
  have hp := hz'.comp parameters (qkPrefixMatrix_differentiableAt inputs decoder columns
    (projectionEvaluation inputs queries i) transverse cap parameters ordinary hc).hasFDerivAt
  have hf : ((fun k => loss (frozenValueReadout
      (anchoredValues frame base scales ordinaryValues) (sparseWeights (projectedQKScores
        (Real.log (qkAnchorGain cap ordinary)) eps queries k inputs i) i))) ∘
      fun p => qkPrefixMatrix inputs decoder columns (projectionEvaluation inputs queries i)
        transverse cap p ordinary) = (fun p => loss (frozenValueReadout
          (anchoredValues frame base scales ordinaryValues) (sparseWeights
            (qkAnchoredScores (projectionEvaluation inputs queries i) transverse eps cap p ordinary) i))) := by
    funext p
    simp only [Function.comp_def]
    rw [projectedQKScores_qkPrefixMatrix inputs decoder queries columns transverse eps cap
      parameters p ordinary i hd hzero hkeys]
  rw [hf, ContinuousLinearMap.zero_comp] at hp
  rw [projectedQKScores_evaluation, hkeys] at hl
  exact qkAnchoredTaskLoss_no_zero_anchor_derivative (projectionEvaluation inputs queries i)
    transverse frame base scales ordinaryValues eps cap parameters ordinary i loss gradient
    hq ht ho he hc hcap hvisible hg hl hp

/-- Every shared-key task premise is inhabited on a dependent long context.
Source context: arXiv:1602.02068v2, §2.5, partial matrix accessibility. -/
example : ¬ HasFDerivAt (𝕜 := ℝ) (fun k : Fin 3 → EucSpace 2 => frozenValueReadout
    (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 100 => 7))
    (sparseWeights (projectedQKScores (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 100 => 0)))
      (1 / 1000000) longQKQueries k (longQKInputs 100) (Fin.natAdd 2 0)) (Fin.natAdd 2 0))) 0
    (longQKKeys 100 (fun _ => 0)) := by
  apply qkPrefixTaskLoss_no_zero_key_projection_derivative (longQKInputs 100) (prefixInputDecoder 2 1)
    longQKQueries (longQKKeys 100 (fun _ => 0)) qkTransverseExample (Module.Basis.singleton (Fin 1) ℝ)
    0 (fun _ => 0) (fun _ : Fin 100 => 7) (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0)
    (fun _ : Fin 100 => 0) (Fin.natAdd 2 0) (fun x : ℝ => x) (ContinuousLinearMap.id ℝ ℝ)
    (longQKInput_anchor 100) (longQKInput_ordinary 100)
    (by rw [longQKQueries_row]; exact qkFrame_example.1) qkFrame_example.2.1
    (by rw [longQKQueries_row]; exact qkFrame_example.2.2) (by norm_num) (by norm_num)
    (by norm_num) (longQK_anchor_visible 100 0)
    (by rw [longQKQueries_row]; exact longQKKeys_row 100 _)
  · intro hz
    have hn := congrArg (fun f : ℝ →L[ℝ] ℝ => f 1) hz
    norm_num at hn
  · exact (ContinuousLinearMap.id ℝ ℝ).hasFDerivAt

/-- Nonzero ordinary task derivatives cannot vanish in all joint Q/K
matrix columns under only partial anchor accessibility. Source: the
derived §2.5 restriction and actual shared projections at `73f8a0b`. -/
theorem qkPrefixTaskLoss_no_zero_projection_derivative {F h d N : ℕ}
    (inputs : Fin (d + 1 + N) → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin (d + 1) → ℝ))
    (queries columns : Fin F → EucSpace h) (transverse : EucSpace h)
    (frame : Module.Basis (Fin d) ℝ E) (base : E) (scales : Fin d → ℝ)
    (ordinaryValues : Fin N → E) (eps cap : ℝ) (parameters : Fin (d + 1) → ℝ)
    (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N)) (loss : E → ℝ) (gradient : E →L[ℝ] ℝ)
    (hd : ∀ a, decoder (inputs (Fin.castAdd N a)) = basis a)
    (hzero : ∀ n, decoder (inputs (Fin.natAdd (d + 1) n)) = 0)
    (hq : ‖projectionEvaluation inputs queries i‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) (projectionEvaluation inputs queries i) transverse = 0)
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
  have hd' := (hasFDerivAt_const (𝕜 := ℝ) queries columns).prodMk (hasFDerivAt_id columns)
  have hk : HasFDerivAt (𝕜 := ℝ) (fun k => loss (frozenValueReadout
      (anchoredValues frame base scales ordinaryValues) (sparseWeights (projectedQKScores
        (Real.log (qkAnchorGain cap ordinary)) eps queries k inputs i) i))) 0 columns := by
    simpa only [Function.comp_def, ContinuousLinearMap.zero_comp] using hz.comp columns hd'
  exact qkPrefixTaskLoss_no_zero_key_projection_derivative inputs decoder queries columns transverse
    frame base scales ordinaryValues eps cap parameters ordinary i loss gradient hd hzero hq ht ho
    he hc hcap hvisible hkeys hg hl hk

/-- All joint matrix premises have the same long, dependent sparse instance.
Source context: arXiv:1602.02068v2, §2.5, partial decoder task direction. -/
example : ¬ HasFDerivAt (𝕜 := ℝ) (fun p : (Fin 3 → EucSpace 2) × (Fin 3 → EucSpace 2) =>
    frozenValueReadout (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0
      (fun _ => 0) (fun _ : Fin 100 => 7)) (sparseWeights (projectedQKScores
        (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 100 => 0))) (1 / 1000000)
        p.1 p.2 (longQKInputs 100) (Fin.natAdd 2 0)) (Fin.natAdd 2 0))) 0
    (longQKQueries, longQKKeys 100 (fun _ => 0)) := by
  apply qkPrefixTaskLoss_no_zero_projection_derivative (longQKInputs 100) (prefixInputDecoder 2 1)
    longQKQueries (longQKKeys 100 (fun _ => 0)) qkTransverseExample (Module.Basis.singleton (Fin 1) ℝ)
    0 (fun _ => 0) (fun _ : Fin 100 => 7) (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0)
    (fun _ : Fin 100 => 0) (Fin.natAdd 2 0) (fun x : ℝ => x) (ContinuousLinearMap.id ℝ ℝ)
    (longQKInput_anchor 100) (longQKInput_ordinary 100)
    (by rw [longQKQueries_row]; exact qkFrame_example.1) qkFrame_example.2.1
    (by rw [longQKQueries_row]; exact qkFrame_example.2.2) (by norm_num) (by norm_num)
    (by norm_num) (longQK_anchor_visible 100 0)
    (by rw [longQKQueries_row]; exact longQKKeys_row 100 _)
  · intro hz
    have hn := congrArg (fun f : ℝ →L[ℝ] ℝ => f 1) hz
    norm_num at hn
  · exact (ContinuousLinearMap.id ℝ ℝ).hasFDerivAt

end Transformer.GPTMini.Sparsemax

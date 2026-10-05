import Transformer.GPTMini.Sparsemax.QKAnchors
import Transformer.GPTMini.Sparsemax.TrainableAnchors

/-!
# Nonzero ordinary task directions in actual key vectors

Derived from arXiv:1602.02068v2, §2.5, and the normalized dot product
and value aggregation in `Attention.forward` at `73f8a0b`.
This row is unrotated and is read before XSA and the output projection.
The unit-key realization enforces the existing anchored score and value
restrictions. A nonzero ordinary output derivative cannot vanish in all
anchor parameters, nor in all actual key-vector directions at the
constructed keys. The latter conclusion follows by the ordinary chain
rule through a differentiable key chart, not a surrogate backward rule.

The frame, positive cap, prefix visibility and epsilon restrictions are
explicit. The nonzero-gradient hypothesis concerns the ordinary output
loss. No attention labels, teacher, auxiliary softmax or zero-projection
assumption is introduced. Keys can subsequently leave the restricted
family under unconstrained updates; the all-parameter guarantee concerns
the chart parameterization. Shared-row losses may still cancel directions.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Actual QKNorm scores keep the value differences spanning the output.
Source: the derived prefix and unit-key restrictions for §2.2 and §2.5
of arXiv:1602.02068v2, with `Attention.forward` at `73f8a0b`. -/
theorem qkAnchoredValues_active_span {h d N : ℕ}
    (query transverse : EucSpace h) (frame : Module.Basis (Fin d) ℝ E)
    (base : E) (scales : Fin d → ℝ) (ordinaryValues : Fin N → E) (eps cap : ℝ)
    (parameters : Fin (d + 1) → ℝ) (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N))
    (hq : ‖query‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) query transverse = 0) (he : eps ≤ 1) (hc : 0 < cap)
    (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i) :
    Submodule.span ℝ (activeValueDifferences
      (qkAnchoredScores query transverse eps cap parameters ordinary) i
      (anchoredValues frame base scales ordinaryValues)) = ⊤ := by
  rw [qkAnchoredScores_eq query transverse eps cap parameters ordinary hq ht ho he hc]
  exact anchoredValues_active_span frame base scales ordinaryValues cap parameters ordinary i
    hc hcap hvisible

/-- The full-span hypotheses hold on an actual normalized sparse row.
Source context: arXiv:1602.02068v2, §2.5, derived Q/K restriction. -/
example : Submodule.span ℝ (activeValueDifferences
    (qkAnchoredScores qkQueryExample qkTransverseExample (1 / 1000000) (1 / 8)
      (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) 2
    (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7))) = ⊤ :=
  qkAnchoredValues_active_span qkQueryExample qkTransverseExample
    (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7)
    (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) 2
    qkFrame_example.1 qkFrame_example.2.1 qkFrame_example.2.2 (by norm_num) (by norm_num)
    (by norm_num) (fun a => Fin.le_last (Fin.castAdd 1 a))

/-- The actual QKNorm row retains nonzero task derivatives in its learned
anchor coordinates for every finite assignment. Source: the derived
prefix and unit-key restrictions for §2.2 and §2.5 of arXiv:1602.02068v2,
evaluated by the normalized dot product and readout at `73f8a0b`. -/
theorem qkAnchoredTaskLoss_no_zero_anchor_derivative {h d N : ℕ}
    (query transverse : EucSpace h) (frame : Module.Basis (Fin d) ℝ E)
    (base : E) (scales : Fin d → ℝ) (ordinaryValues : Fin N → E) (eps cap : ℝ)
    (parameters : Fin (d + 1) → ℝ) (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N))
    (loss : E → ℝ) (gradient : E →L[ℝ] ℝ)
    (hq : ‖query‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) query transverse = 0) (he : eps ≤ 1) (hc : 0 < cap)
    (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i) (hg : gradient ≠ 0)
    (hl : HasFDerivAt (𝕜 := ℝ) loss gradient
      (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
        (sparseWeights (qkAnchoredScores query transverse eps cap parameters ordinary) i))) :
    ¬ HasFDerivAt (𝕜 := ℝ)
      (fun p => loss (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
        (sparseWeights (qkAnchoredScores query transverse eps cap p ordinary) i))) 0 parameters := by
  have hrow (p : Fin (d + 1) → ℝ) :=
    qkAnchoredScores_eq query transverse eps cap p ordinary hq ht ho he hc
  have hf : (fun p => loss
      (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
        (sparseWeights (qkAnchoredScores query transverse eps cap p ordinary) i))) =
      (fun p => loss (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
        (sparseWeights (anchoredScores cap p ordinary) i))) := by
    funext p
    rw [hrow p]
  rw [hf]
  rw [hrow parameters] at hl
  exact anchoredTaskLoss_no_zero_parameter_derivative frame base scales ordinaryValues cap
    parameters ordinary i loss gradient hc hcap hvisible hg hl

/-- Every QKNorm, span and task premise is satisfied by an ordinary scalar
loss on a sparse row. Source context: arXiv:1602.02068v2, §2.5,
the derived Q/K realization and readout at `73f8a0b`. -/
example : ¬ HasFDerivAt (𝕜 := ℝ)
    (fun p : Fin 2 → ℝ => frozenValueReadout
      (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
      (sparseWeights (qkAnchoredScores qkQueryExample qkTransverseExample (1 / 1000000)
        (1 / 8) p (fun _ : Fin 1 => 0)) 2)) 0 (fun _ => 0) := by
  apply qkAnchoredTaskLoss_no_zero_anchor_derivative qkQueryExample qkTransverseExample
    (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7)
    (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) 2
    (fun x : ℝ => x) (ContinuousLinearMap.id ℝ ℝ) qkFrame_example.1 qkFrame_example.2.1
    qkFrame_example.2.2 (by norm_num) (by norm_num) (by norm_num)
    (fun a => Fin.le_last (Fin.castAdd 1 a))
  · intro hz
    have hn := congrArg (fun f : ℝ →L[ℝ] ℝ => f 1) hz
    norm_num at hn
  · exact (ContinuousLinearMap.id ℝ ℝ).hasFDerivAt

/-- The ordinary task derivative cannot be zero in all actual K-vector
directions at any constructed key assignment. Source: the derived unit
frame and active prefix for §2.5 of arXiv:1602.02068v2, with the actual
normalized dot product at `73f8a0b`. Full sparsemax differentiability
at support boundaries is not assumed. -/
theorem qkAnchoredTaskLoss_no_zero_key_derivative {h d N : ℕ}
    (query transverse : EucSpace h) (frame : Module.Basis (Fin d) ℝ E)
    (base : E) (scales : Fin d → ℝ) (ordinaryValues : Fin N → E) (eps cap : ℝ)
    (parameters : Fin (d + 1) → ℝ) (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N))
    (loss : E → ℝ) (gradient : E →L[ℝ] ℝ)
    (hq : ‖query‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) query transverse = 0) (he : eps ≤ 1) (hc : 0 < cap)
    (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i) (hg : gradient ≠ 0)
    (hl : HasFDerivAt (𝕜 := ℝ) loss gradient
      (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
        (sparseWeights (qkAnchoredScores query transverse eps cap parameters ordinary) i))) :
    ¬ HasFDerivAt (𝕜 := ℝ)
      (fun keys : Fin (d + 1 + N) → EucSpace h => loss
        (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
          (sparseWeights (fun n => score (Real.log (qkAnchorGain cap ordinary)) eps query
            (keys n)) i))) 0 (qkAnchoredKeys query transverse cap parameters ordinary) := by
  intro hz
  have hd := qkAnchoredKeys_differentiableAt query transverse cap parameters ordinary hc
  have hzero : HasFDerivAt (𝕜 := ℝ)
      (fun p => loss (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
        (sparseWeights (qkAnchoredScores query transverse eps cap p ordinary) i))) 0 parameters := by
    unfold qkAnchoredScores
    simpa only [Function.comp_def, ContinuousLinearMap.zero_comp] using
      hz.comp parameters hd.hasFDerivAt
  exact qkAnchoredTaskLoss_no_zero_anchor_derivative query transverse frame base scales
    ordinaryValues eps cap parameters ordinary i loss gradient hq ht ho he hc hcap
    hvisible hg hl hzero

/-- Nonzero actual-key derivatives have an inhabited sparse scalar task.
Source context: arXiv:1602.02068v2, §2.5, normalized scores at `73f8a0b`. -/
example : ¬ HasFDerivAt (𝕜 := ℝ)
    (fun keys : Fin 3 → EucSpace 2 => frozenValueReadout
      (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
      (sparseWeights (fun n => score (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0)))
        (1 / 1000000) qkQueryExample (keys n)) 2)) 0
    (qkAnchoredKeys qkQueryExample qkTransverseExample (1 / 8)
      (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) := by
  apply qkAnchoredTaskLoss_no_zero_key_derivative qkQueryExample qkTransverseExample
    (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7)
    (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) 2
    (fun x : ℝ => x) (ContinuousLinearMap.id ℝ ℝ) qkFrame_example.1 qkFrame_example.2.1
    qkFrame_example.2.2 (by norm_num) (by norm_num) (by norm_num)
    (fun a => Fin.le_last (Fin.castAdd 1 a))
  · intro hz
    have hn := congrArg (fun f : ℝ →L[ℝ] ℝ => f 1) hz
    norm_num at hn
  · exact (ContinuousLinearMap.id ℝ ℝ).hasFDerivAt

end Transformer.GPTMini.Sparsemax

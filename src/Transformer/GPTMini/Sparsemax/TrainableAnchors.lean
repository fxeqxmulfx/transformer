import Transformer.GPTMini.Sparsemax.AnchorTransfer
import Transformer.GPTMini.Sparsemax.AnchoredValues

/-!
# Ordinary task derivatives reach trainable anchor parameters

Derived from arXiv:1602.02068v2, §2.5, `sparsemax_gradient`, and the
value sum in `Attention.forward` at `73f8a0b`. The architecture changes
are the visible scaled-basis prefix and independent bounded anchor scores.
Their local inverse turns an actual active score transfer into a
differentiable curve of trainable parameters with the same task derivative.

Consequently a nonzero output derivative cannot vanish in all anchor
parameter directions, for any finite assignment of the parameters.
No derivative is assigned by definition, no full projection derivative
is assumed at support boundaries, and no target attention route is used.
The guarantee is for one row's ordinary task loss; cancellations between
rows sharing parameters and whole-model convergence are not asserted.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The lifted parameter curve has the actual task derivative of the
active score pair. Source: the derived independent coordinates for
§2.5 of arXiv:1602.02068v2, composed with the frozen value sum at `73f8a0b`. -/
theorem anchoredTaskLoss_active_pair_hasDerivAt {A N : ℕ} (cap : ℝ)
    (parameters : Fin A → ℝ) (ordinary : Fin N → ℝ) (i : Fin (A + N))
    (j k : Fin A) (values : Fin (A + N) → E) (loss : E → ℝ)
    (gradient : E →L[ℝ] ℝ) (hc : 0 < cap) (hne : j ≠ k)
    (hj : 0 < sparseWeights (anchoredScores cap parameters ordinary) i (Fin.castAdd N j))
    (hk : 0 < sparseWeights (anchoredScores cap parameters ordinary) i (Fin.castAdd N k))
    (hl : HasFDerivAt (𝕜 := ℝ) loss gradient
      (frozenValueReadout values (sparseWeights (anchoredScores cap parameters ordinary) i))) :
    HasDerivAt (fun t => loss (frozenValueReadout values (sparseWeights
      (anchoredScores cap (boundedTransferParameters cap parameters j k t) ordinary) i)))
      (gradient (values (Fin.castAdd N j) - values (Fin.castAdd N k))) 0 := by
  have hpair : Fin.castAdd N j ≠ Fin.castAdd N k :=
    fun he => hne (Fin.castAdd_injective A N he)
  have hraw := outerLoss_active_pair_hasDerivAt (anchoredScores cap parameters ordinary) i
    (Fin.castAdd N j) (Fin.castAdd N k) (fun p => loss (frozenValueReadout values p))
    (pullbackValueGradient gradient values) hpair hj hk
    (valueReadoutLoss_hasFDerivAt values loss gradient _ hl)
  rw [pullbackValueGradient_active_pair] at hraw
  apply hraw.congr_of_eventuallyEq
  exact (boundedTransferParameters_realizes_transfer cap parameters ordinary j k hc).mono
    fun t ht => congrArg (fun z => loss (frozenValueReadout values (sparseWeights z i))) ht

/-- Every active-pair derivative premise is realized by two anchors and
an inactive ordinary slot. Source context: arXiv:1602.02068v2, §2.5. -/
example : HasDerivAt
    (fun t => frozenValueReadout (basis 0 : Fin 3 → ℝ) (sparseWeights
      (anchoredScores (1 / 8) (boundedTransferParameters (1 / 8) (fun _ : Fin 2 => 0) 0 1 t)
        (fun _ : Fin 1 => 0)) 2)) 1 0 := by
  simpa [basis] using
    anchoredTaskLoss_active_pair_hasDerivAt (1 / 8) (fun _ : Fin 2 => 0)
      (fun _ : Fin 1 => 0) 2 0 1 (basis 0 : Fin 3 → ℝ) (fun x : ℝ => x)
      (ContinuousLinearMap.id ℝ ℝ) (by norm_num) (by decide)
      (by norm_num [anchoredScores_sparse_example])
      (by norm_num [anchoredScores_sparse_example])
      (ContinuousLinearMap.id ℝ ℝ).hasFDerivAt

/-- A separating active pair excludes a zero derivative with respect
to the learned anchor parameters themselves. Source: the derived local
inverse for §2.5 of arXiv:1602.02068v2, with the ordinary value readout
at `73f8a0b`; the conclusion is no longer just about raw scores. -/
theorem anchoredTaskLoss_active_pair_no_zero_parameter_derivative {A N : ℕ} (cap : ℝ)
    (parameters : Fin A → ℝ) (ordinary : Fin N → ℝ) (i : Fin (A + N))
    (j k : Fin A) (values : Fin (A + N) → E) (loss : E → ℝ)
    (gradient : E →L[ℝ] ℝ) (hc : 0 < cap) (hne : j ≠ k)
    (hj : 0 < sparseWeights (anchoredScores cap parameters ordinary) i (Fin.castAdd N j))
    (hk : 0 < sparseWeights (anchoredScores cap parameters ordinary) i (Fin.castAdd N k))
    (hl : HasFDerivAt (𝕜 := ℝ) loss gradient
      (frozenValueReadout values (sparseWeights (anchoredScores cap parameters ordinary) i)))
    (hsep : gradient (values (Fin.castAdd N j) - values (Fin.castAdd N k)) ≠ 0) :
    ¬ HasFDerivAt (𝕜 := ℝ)
      (fun p => loss (frozenValueReadout values (sparseWeights (anchoredScores cap p ordinary) i)))
      0 parameters := by
  intro hz
  have hz' : HasFDerivAt (𝕜 := ℝ)
      (fun p => loss (frozenValueReadout values (sparseWeights (anchoredScores cap p ordinary) i)))
      0 (boundedTransferParameters cap parameters j k 0) := by
    rw [boundedTransferParameters_zero cap parameters j k hc]
    exact hz
  have hzero : HasDerivAt
      (fun t => loss (frozenValueReadout values (sparseWeights
        (anchoredScores cap (boundedTransferParameters cap parameters j k t) ordinary) i))) 0 0 := by
    simpa only [Function.comp_def, zero_apply] using hz'.comp_hasDerivAt (0 : ℝ)
      (boundedTransferParameters_differentiableAt cap parameters j k hc).hasDerivAt
  exact hsep ((anchoredTaskLoss_active_pair_hasDerivAt cap parameters ordinary i j k
    values loss gradient hc hne hj hk hl).unique hzero)

/-- A sparse ordinary scalar readout inhabits all the no-zero-parameter
premises. Source context: arXiv:1602.02068v2, §2.5, derived anchor scores. -/
example : ¬ HasFDerivAt (𝕜 := ℝ)
    (fun p : Fin 2 → ℝ => frozenValueReadout (basis 0 : Fin 3 → ℝ)
      (sparseWeights (anchoredScores (1 / 8) p (fun _ : Fin 1 => 0)) 2)) 0 (fun _ => 0) := by
  exact anchoredTaskLoss_active_pair_no_zero_parameter_derivative (1 / 8)
    (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) 2 0 1 (basis 0 : Fin 3 → ℝ) (fun x : ℝ => x)
    (ContinuousLinearMap.id ℝ ℝ) (by norm_num) (by decide)
    (by norm_num [anchoredScores_sparse_example])
    (by norm_num [anchoredScores_sparse_example])
    (ContinuousLinearMap.id ℝ ℝ).hasFDerivAt (by norm_num [basis])

/-- Enforced support and value span give a nonzero ordinary task
direction in trainable anchor parameters at every finite assignment.
Source: the derived prefix architecture for §2.2 and §2.5 of
arXiv:1602.02068v2 and the value sum at `73f8a0b`.
No span premise or query/key accessibility premise remains. -/
theorem anchoredTaskLoss_no_zero_parameter_derivative {d N : ℕ}
    (frame : Module.Basis (Fin d) ℝ E) (base : E) (scales : Fin d → ℝ)
    (ordinaryValues : Fin N → E) (cap : ℝ) (parameters : Fin (d + 1) → ℝ)
    (ordinaryScores : Fin N → ℝ) (i : Fin (d + 1 + N))
    (loss : E → ℝ) (gradient : E →L[ℝ] ℝ) (hc : 0 < cap)
    (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i) (hg : gradient ≠ 0)
    (hl : HasFDerivAt (𝕜 := ℝ) loss gradient
      (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
        (sparseWeights (anchoredScores cap parameters ordinaryScores) i))) :
    ¬ HasFDerivAt (𝕜 := ℝ)
      (fun p => loss (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
        (sparseWeights (anchoredScores cap p ordinaryScores) i))) 0 parameters := by
  obtain ⟨k, hsep⟩ := anchoredValues_separates_gradient frame base scales ordinaryValues gradient hg
  have hp (a : Fin (d + 1)) :=
    anchoredScores_anchor_positive cap parameters ordinaryScores i hc hcap hvisible a
  exact anchoredTaskLoss_active_pair_no_zero_parameter_derivative cap parameters ordinaryScores i
    k.succ 0 (anchoredValues frame base scales ordinaryValues) loss gradient hc
    (Fin.succ_ne_zero k) (hp k.succ) (hp 0) hl hsep

/-- The full architecture theorem has inhabited ordinary task premises
and a visible inactive slot. Source context: arXiv:1602.02068v2, §2.5. -/
example : ¬ HasFDerivAt (𝕜 := ℝ)
    (fun p : Fin 2 → ℝ => (ContinuousLinearMap.proj 0 : (Fin 1 → ℝ) →L[ℝ] ℝ)
      (frozenValueReadout
        (anchoredValues (Pi.basisFun ℝ (Fin 1)) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
        (sparseWeights (anchoredScores (1 / 8) p (fun _ : Fin 1 => 0)) 2))) 0 (fun _ => 0) := by
  apply anchoredTaskLoss_no_zero_parameter_derivative (Pi.basisFun ℝ (Fin 1)) 0 (fun _ => 0)
    (fun _ : Fin 1 => 7) (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) 2
    (fun x : Fin 1 → ℝ => x 0) (ContinuousLinearMap.proj 0) (by norm_num) (by norm_num)
    (fun a => Fin.le_last (Fin.castAdd 1 a))
  · intro h
    have he := congrArg (fun f : (Fin 1 → ℝ) →L[ℝ] ℝ => f (fun _ => 1)) h
    norm_num at he
  · exact (ContinuousLinearMap.proj 0 : (Fin 1 → ℝ) →L[ℝ] ℝ).hasFDerivAt

end Transformer.GPTMini.Sparsemax

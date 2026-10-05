import Transformer.GPTMini.Sparsemax.TrainableAnchors
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Calculus.LocalExtr.Basic

/-!
# Wrong anchored outputs have a useful trainable direction

Derived from arXiv:1602.02068v2, §2.2 and §2.5, with the ordinary
value sum at `73f8a0b`. For squared distance to an ordinary output target,
the output derivative is nonzero exactly when the output is wrong.
The anchored construction transmits it to trainable score parameters.
Every positive-error finite assignment therefore fails to be a local
minimum of this row loss, including at inactive support boundaries.

The result applies in any finite-dimensional real inner-product output
space. The target is an output vector, not an attention position. It does
not supply a uniform gradient lower bound, prove the target attainable by
fixed values and bounded scores, or imply whole-model convergence.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Ordinary squared output error. Source context: the value readout at
`73f8a0b`, followed by squared distance; §2.5 of arXiv:1602.02068v2
supplies the sparsemax score path. No routing target is present. -/
def squaredReadoutLoss (target output : E) : ℝ := ‖output - target‖ ^ 2

/-- The ordinary squared loss has its actual output derivative.
Source context: `Attention.forward` at `73f8a0b` and the chain rule
used with §2.5 of arXiv:1602.02068v2; no score supervision is added. -/
theorem squaredReadoutLoss_hasFDerivAt (target output : E) :
    HasFDerivAt (𝕜 := ℝ) (squaredReadoutLoss target)
      (2 • innerSL ℝ (output - target)) output := by
  unfold squaredReadoutLoss
  simpa only [ContinuousLinearMap.comp_id, id_eq] using
    ((hasFDerivAt_id output).sub_const target).norm_sq

/-- A wrong output has a nonzero squared-loss derivative.
Source context: the ordinary task loss after the value sum at `73f8a0b`,
composed with the derived restrictions for arXiv:1602.02068v2, §2.5. -/
theorem squaredReadoutLoss_gradient_ne_zero (target output : E) (hne : output ≠ target) :
    (2 • innerSL ℝ (output - target) : E →L[ℝ] ℝ) ≠ 0 := by
  intro h
  have he := congrArg (fun f : E →L[ℝ] ℝ => f (output - target)) h
  have hp : 0 < ‖output - target‖ ^ 2 :=
    sq_pos_of_ne_zero (ne_of_gt (norm_pos_iff.mpr (sub_ne_zero.mpr hne)))
  simp only [two_smul, add_apply, innerSL_apply_apply, zero_apply,
    real_inner_self_eq_norm_sq] at he
  nlinarith

/-- A wrong scalar output inhabits the nonzero-gradient premise.
Source context: the ordinary output target, with the §2.5 derived score path. -/
example : (2 • innerSL ℝ ((1 / 2 : ℝ) - 0) : ℝ →L[ℝ] ℝ) ≠ 0 :=
  squaredReadoutLoss_gradient_ne_zero _ _ (by norm_num)

/-- The ordinary squared-loss derivative vanishes exactly at its target.
Source context: the output loss after the value sum at `73f8a0b`,
used with the derived score direction of arXiv:1602.02068v2, §2.5. -/
theorem squaredReadoutLoss_gradient_eq_zero_iff (target output : E) :
    (2 • innerSL ℝ (output - target) : E →L[ℝ] ℝ) = 0 ↔ output = target := by
  constructor
  · intro h
    by_contra hne
    exact squaredReadoutLoss_gradient_ne_zero target output hne h
  · intro h
    rw [h, sub_self]
    simp

/-- Translating the anchor frame translates its initial scalar output.
Source context: the derived construction for arXiv:1602.02068v2,
§2.2 and §2.5; the ordinary value has exactly zero weight. -/
theorem anchoredScalarReadout_zero_parameters (base : ℝ) (ordinary : Fin 1 → ℝ) :
    frozenValueReadout
      (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) base (fun _ => 0) ordinary)
      (sparseWeights
        (anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) 2) =
      base + 1 / 2 := by
  let values : Fin 3 → ℝ :=
    anchoredValues (Module.Basis.singleton (Fin 1) ℝ) base (fun _ => 0) ordinary
  have h0 : values 0 = base := by
    change values (Fin.castAdd 1 (0 : Fin 2)) = base
    simp only [values, anchoredValues, Fin.addCases_left, Fin.cases_zero]
  have h1 : values 1 = base + 1 := by
    change values (Fin.castAdd 1 (0 : Fin 1).succ) = base + 1
    simp only [values, anchoredValues, Fin.addCases_left, Fin.cases_succ, Real.exp_zero,
      one_smul, Module.Basis.singleton_apply]
  have h2 : values 2 = ordinary 0 := by
    change values (Fin.natAdd 2 (0 : Fin 1)) = ordinary 0
    simp only [values, anchoredValues, Fin.addCases_right]
  change frozenValueReadout values _ = base + 1 / 2
  rw [frozenValueReadout_apply, Fin.sum_univ_three, h0, h1, h2, anchoredScores_sparse_example]
  norm_num
  ring

/-- The sparse anchor example has a concrete scalar output.
Source context: the derived construction for arXiv:1602.02068v2,
§2.2 and §2.5, with arbitrary ordinary value `7` at an inactive slot. -/
theorem anchoredScalarReadout_sparse_example :
    frozenValueReadout
      (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
      (sparseWeights
        (anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) 2) =
      (1 / 2 : ℝ) := by
  simpa only [zero_add] using anchoredScalarReadout_zero_parameters 0 (fun _ => 7)

/-- Every wrong output supplies a nonzero direction in learned anchor
parameters, for every finite assignment of all value and score parameters.
Source: the derived enforced span and local inverse for §2.2 and §2.5
of arXiv:1602.02068v2, followed by ordinary squared output error. -/
theorem anchoredSquaredError_no_zero_parameter_derivative {d N : ℕ}
    (frame : Module.Basis (Fin d) ℝ E) (base : E) (scales : Fin d → ℝ)
    (ordinaryValues : Fin N → E) (cap : ℝ) (parameters : Fin (d + 1) → ℝ)
    (ordinaryScores : Fin N → ℝ) (i : Fin (d + 1 + N)) (target : E)
    (hc : 0 < cap) (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i)
    (hbad : frozenValueReadout (anchoredValues frame base scales ordinaryValues)
      (sparseWeights (anchoredScores cap parameters ordinaryScores) i) ≠ target) :
    ¬ HasFDerivAt (𝕜 := ℝ)
      (fun p => squaredReadoutLoss target
        (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
          (sparseWeights (anchoredScores cap p ordinaryScores) i))) 0 parameters := by
  let output := frozenValueReadout (anchoredValues frame base scales ordinaryValues)
    (sparseWeights (anchoredScores cap parameters ordinaryScores) i)
  exact anchoredTaskLoss_no_zero_parameter_derivative frame base scales ordinaryValues cap
    parameters ordinaryScores i (squaredReadoutLoss target) (2 • innerSL ℝ (output - target))
    hc hcap hvisible (squaredReadoutLoss_gradient_ne_zero target output hbad)
    (squaredReadoutLoss_hasFDerivAt target output)

/-- The complete wrong-output hypotheses hold on a row with an exact
zero weight. Source context: §2.2 and §2.5's derived anchored architecture. -/
example : ¬ HasFDerivAt (𝕜 := ℝ)
    (fun p : Fin 2 → ℝ => squaredReadoutLoss (0 : ℝ)
      (frozenValueReadout
        (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
        (sparseWeights (anchoredScores (1 / 8) p (fun _ : Fin 1 => 0)) 2))) 0 (fun _ => 0) := by
  apply anchoredSquaredError_no_zero_parameter_derivative (Module.Basis.singleton (Fin 1) ℝ)
    0 (fun _ => 0) (fun _ : Fin 1 => 7) (1 / 8) (fun _ : Fin 2 => 0)
    (fun _ : Fin 1 => 0) 2 0 (by norm_num) (by norm_num)
    (fun a => Fin.le_last (Fin.castAdd 1 a))
  rw [anchoredScalarReadout_sparse_example]
  norm_num

/-- A wrong output is not a local minimum of the learned anchor loss.
Source: the actual active-pair direction from §2.5 of
arXiv:1602.02068v2, lifted through the derived architecture.
Fermat's theorem is applied to a differentiable parameter curve, so
no full differentiability premise is imposed on support boundaries. -/
theorem anchoredSquaredError_not_isLocalMin {d N : ℕ}
    (frame : Module.Basis (Fin d) ℝ E) (base : E) (scales : Fin d → ℝ)
    (ordinaryValues : Fin N → E) (cap : ℝ) (parameters : Fin (d + 1) → ℝ)
    (ordinaryScores : Fin N → ℝ) (i : Fin (d + 1 + N)) (target : E)
    (hc : 0 < cap) (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i)
    (hbad : frozenValueReadout (anchoredValues frame base scales ordinaryValues)
      (sparseWeights (anchoredScores cap parameters ordinaryScores) i) ≠ target) :
    ¬ IsLocalMin
      (fun p => squaredReadoutLoss target
        (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
          (sparseWeights (anchoredScores cap p ordinaryScores) i))) parameters := by
  let values := anchoredValues frame base scales ordinaryValues
  let output := frozenValueReadout values (sparseWeights (anchoredScores cap parameters ordinaryScores) i)
  let gradient : E →L[ℝ] ℝ := 2 • innerSL ℝ (output - target)
  have hg : gradient ≠ 0 := squaredReadoutLoss_gradient_ne_zero target output hbad
  obtain ⟨k, hsep⟩ := anchoredValues_separates_gradient frame base scales ordinaryValues gradient hg
  have hp (a : Fin (d + 1)) :=
    anchoredScores_anchor_positive cap parameters ordinaryScores i hc hcap hvisible a
  have hd := anchoredTaskLoss_active_pair_hasDerivAt cap parameters ordinaryScores i k.succ 0
    values (squaredReadoutLoss target) gradient hc (Fin.succ_ne_zero k) (hp k.succ) (hp 0)
    (squaredReadoutLoss_hasFDerivAt target output)
  intro hm
  rw [← boundedTransferParameters_zero cap parameters k.succ 0 hc] at hm
  have hcurve := hm.comp_continuous
    (boundedTransferParameters_continuousAt cap parameters k.succ 0 hc)
  exact hsep (hcurve.hasDerivAt_eq_zero (by simpa only [Function.comp_def] using hd))

/-- All local-minimum exclusion premises have a concrete sparse instance.
Source context: arXiv:1602.02068v2, §2.5, ordinary squared output error. -/
example : ¬ IsLocalMin
    (fun p : Fin 2 → ℝ => squaredReadoutLoss (0 : ℝ)
      (frozenValueReadout
        (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
        (sparseWeights (anchoredScores (1 / 8) p (fun _ : Fin 1 => 0)) 2))) (fun _ => 0) := by
  apply anchoredSquaredError_not_isLocalMin (Module.Basis.singleton (Fin 1) ℝ)
    0 (fun _ => 0) (fun _ : Fin 1 => 7) (1 / 8) (fun _ : Fin 2 => 0)
    (fun _ : Fin 1 => 0) 2 0 (by norm_num) (by norm_num)
    (fun a => Fin.le_last (Fin.castAdd 1 a))
  rw [anchoredScalarReadout_sparse_example]
  norm_num

end Transformer.GPTMini.Sparsemax

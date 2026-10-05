import Transformer.GPTMini.Sparsemax.AnchoredSquaredError

/-!
# A finite trainable correction with a visible exact zero

Derived example for arXiv:1602.02068v2, §2.2 and §2.5, using the
value sum at `73f8a0b` and the new anchored parameterization. Two
anchor values are `(-1/16, 15/16)` and an ordinary value is `7`.
The ordinary scalar output target is `1/2` throughout.

Starting from zero anchor parameters, the readout is `7/16` and its
squared error is `1/256`. The lifted active-pair derivative is `-1/8`.
A score transfer of `1/16` is realized by finite parameters
`(-log 3, log 3)`, gives zero error, and retains the third weight at zero.
This is a constructed row correction, not a trained-transformer result
or a guarantee that an arbitrary fixed value hull contains its target.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex

/-- One scalar instance of the translated scaled-basis value rule.
Source context: the derived anchor construction for §2.5 of
arXiv:1602.02068v2 and the value readout at `73f8a0b`. -/
def correctionAnchorValues : Fin 3 → ℝ :=
  anchoredValues (Module.Basis.singleton (Fin 1) ℝ) (-(1 / 16))
    (fun _ => 0) (fun _ : Fin 1 => 7)

/-- Finite learned parameters for the corrective score transfer.
Source context: arXiv:1602.02068v2, §2.5, through the derived inverse chart. -/
def correctionAnchorParameters : Fin 2 → ℝ :=
  fun a => if a = 0 then -Real.log 3 else Real.log 3

/-- The unchanged ordinary squared output loss in the example.
Source context: the value sum at `73f8a0b` followed by the scalar
target `1/2`; no target for an attention position is introduced. -/
def correctionAnchorLoss (parameters : Fin 2 → ℝ) : ℝ :=
  squaredReadoutLoss (1 / 2 : ℝ) (frozenValueReadout correctionAnchorValues
    (sparseWeights (anchoredScores (1 / 8) parameters (fun _ : Fin 1 => 0)) 2))

/-- The constructed frame has its stated scalar values at all slots.
Source context: the derived anchor example for §2.5 of
arXiv:1602.02068v2 and the ordinary value sum at `73f8a0b`. -/
theorem correctionAnchorValues_apply :
    correctionAnchorValues =
      (fun n : Fin 3 => if n = 2 then 7 else if n = 0 then -(1 / 16) else 15 / 16) := by
  funext n
  fin_cases n
  · change anchoredValues (Module.Basis.singleton (Fin 1) ℝ) (-(1 / 16))
      (fun _ => 0) (fun _ : Fin 1 => 7) (Fin.castAdd 1 (0 : Fin 2)) = -(1 / 16)
    simp only [anchoredValues, Fin.addCases_left, Fin.cases_zero]
  · change anchoredValues (Module.Basis.singleton (Fin 1) ℝ) (-(1 / 16))
      (fun _ => 0) (fun _ : Fin 1 => 7) (Fin.castAdd 1 (0 : Fin 1).succ) = 15 / 16
    simp only [anchoredValues, Fin.addCases_left, Fin.cases_succ, Real.exp_zero,
      one_smul, Module.Basis.singleton_apply]
    norm_num
  · change anchoredValues (Module.Basis.singleton (Fin 1) ℝ) (-(1 / 16))
      (fun _ => 0) (fun _ : Fin 1 => 7) (Fin.natAdd 2 (0 : Fin 1)) = 7
    simp only [anchoredValues, Fin.addCases_right]

/-- The initial sparse readout has the stated nonzero output error.
Source context: the derived scalar example for §2.2 and §2.5 of
arXiv:1602.02068v2, with an inactive ordinary value. -/
theorem correctionAnchorReadout_initial :
    frozenValueReadout correctionAnchorValues (sparseWeights
      (anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) 2) =
      (7 / 16 : ℝ) := by
  have h := anchoredScalarReadout_zero_parameters (-(1 / 16)) (fun _ => 7)
  norm_num at h
  exact h

/-- The initial ordinary loss is positive despite an exact inactive slot.
Source context: arXiv:1602.02068v2, §2.5, the derived scalar correction. -/
theorem correctionAnchorLoss_initial :
    correctionAnchorLoss (fun _ => 0) = (1 / 256 : ℝ) := by
  rw [correctionAnchorLoss, correctionAnchorReadout_initial]
  norm_num [squaredReadoutLoss]

/-- The corrective parameters are exactly the lifted finite score step.
Source context: §2.5 of arXiv:1602.02068v2, active-pair transfer,
realized by the new independent anchor coordinate inverse. -/
theorem correctionAnchorParameters_eq_lift :
    correctionAnchorParameters =
      boundedTransferParameters (1 / 8) (fun _ : Fin 2 => 0) 1 0 (1 / 16) := by
  have hlog : Real.log (1 / 3 : ℝ) = -Real.log 3 := by
    simpa only [one_div] using Real.log_inv (3 : ℝ)
  funext a
  fin_cases a <;> norm_num [correctionAnchorParameters, boundedTransferParameters,
    inverseCoordinate, transferScores, boundedCoordinate, basis, hlog]

/-- The finite corrective parameters give the stated actual raw scores.
Source context: the new chart for §2.2 of arXiv:1602.02068v2;
both anchor scores remain strictly inside their cap `1/8`. -/
theorem correctionAnchorScores :
    anchoredScores (1 / 8) correctionAnchorParameters (fun _ : Fin 1 => 0) =
      (fun n : Fin 3 => if n = 2 then -(9 / 8) else if n = 0 then -(1 / 16) else 1 / 16) := by
  have he : Real.exp (Real.log 3) = 3 := Real.exp_log (by norm_num)
  have hn : Real.exp (-Real.log 3) = (1 / 3 : ℝ) := by rw [Real.exp_neg, he]; norm_num
  funext n
  fin_cases n <;> norm_num [anchoredScores, Fin.addCases, correctionAnchorParameters,
    boundedCoordinate, he, hn]

/-- The actual variational projection retains its visible third zero
after the finite parameter step. Source: arXiv:1602.02068v2, §2.2,
`sparsemax_closedform`, applied to the new scalar anchor example. -/
theorem correctionAnchorProjection :
    sparseWeights
      (anchoredScores (1 / 8) correctionAnchorParameters (fun _ : Fin 1 => 0)) 2 =
      (fun n : Fin 3 => if n = 2 then 0 else if n = 0 then 7 / 16 else 9 / 16) := by
  let scores := anchoredScores (1 / 8) correctionAnchorParameters (fun _ : Fin 1 => 0)
  have hclosed : thresholdWeights scores 2 (-(1 / 2)) =
      (fun n : Fin 3 => if n = 2 then 0 else if n = 0 then 7 / 16 else 9 / 16) := by
    funext n
    fin_cases n <;> norm_num [thresholdWeights, scores, correctionAnchorScores]
  have hsum : ∑ n : Fin 3, thresholdWeights scores 2 (-(1 / 2)) n = 1 := by
    rw [hclosed]
    norm_num [Fin.sum_univ_three]
  rw [← thresholdWeights_eq_sparseWeights scores 2 (-(1 / 2)) hsum]
  exact hclosed

/-- The finite learned parameter step reaches zero ordinary output error.
Source context: §2.2 and §2.5 of arXiv:1602.02068v2, the derived
anchored scalar example with target `1/2` and unchanged values. -/
theorem correctionAnchorLoss_final : correctionAnchorLoss correctionAnchorParameters = 0 := by
  norm_num [correctionAnchorLoss, squaredReadoutLoss, frozenValueReadout, sum_apply,
    Fin.sum_univ_three, correctionAnchorValues_apply,
    correctionAnchorProjection]

/-- The lifted learned-parameter curve has a negative ordinary loss
derivative at the initial positive-error point. Source: the actual
active-pair direction of arXiv:1602.02068v2, §2.5, through the derived
score inverse; this is neither a routing loss nor a custom gradient. -/
theorem correctionAnchorLoss_hasDerivAt :
    HasDerivAt
      (fun t => correctionAnchorLoss
        (boundedTransferParameters (1 / 8) (fun _ : Fin 2 => 0) 1 0 t)) (-(1 / 8)) 0 := by
  have hl : HasFDerivAt (𝕜 := ℝ) (squaredReadoutLoss (1 / 2 : ℝ))
      (2 • innerSL ℝ (-(1 / 16 : ℝ))) (frozenValueReadout correctionAnchorValues
        (sparseWeights
          (anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) 2)) := by
    rw [correctionAnchorReadout_initial]
    convert squaredReadoutLoss_hasFDerivAt (1 / 2 : ℝ) (7 / 16) using 1
    norm_num
  have h := anchoredTaskLoss_active_pair_hasDerivAt (1 / 8) (fun _ : Fin 2 => 0)
    (fun _ : Fin 1 => 0) 2 1 0 correctionAnchorValues (squaredReadoutLoss (1 / 2 : ℝ))
    (2 • innerSL ℝ (-(1 / 16 : ℝ))) (by norm_num) (by decide)
    (by norm_num [anchoredScores_sparse_example])
    (by norm_num [anchoredScores_sparse_example]) hl
  norm_num [correctionAnchorValues_apply,
    smul_apply, innerSL_apply_apply, Real.inner_apply] at h
  simpa only [correctionAnchorLoss, correctionAnchorValues_apply] using h

end Transformer.GPTMini.Sparsemax

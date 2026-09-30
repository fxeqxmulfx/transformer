/-
# Muon — algebra of the reported scaling-law fits

arXiv:2502.16982, §3.2, `tab:fit`, and Appendix B,
(`tab:dense_scaling_param`). The decimal coefficients define the reported
empirical fits. Algebraic identities about these functions do not prove
their accuracy or the optimality of the hyperparameters for actual training.
-/

import Transformer.Muon.Section2_Models
import Mathlib.Analysis.SpecialFunctions.Pow.Real

noncomputable section

namespace Transformer.Muon

/-- Muon's reported loss fit, arXiv:2502.16982, §3.2, `tab:fit`. -/
def muonLossFit (C : ℝ) : ℝ := (2506 / 1000) * C ^ (-(52 / 1000 : ℝ))

/-- AdamW's reported loss fit, arXiv:2502.16982, §3.2, `tab:fit`. -/
def adamLossFit (C : ℝ) : ℝ := (2608 / 1000) * C ^ (-(54 / 1000 : ℝ))

/-- Compute giving equal loss in the two literal reported fit functions,
arXiv:2502.16982, §3.2, `tab:fit`. -/
def equalLossMuonCompute (C : ℝ) : ℝ := (2506 / 2608 : ℝ) ^ (250 / 13 : ℝ) * C ^ (27 / 26 : ℝ)

/-- The reported fits are positive at positive compute,
arXiv:2502.16982, §3.2, `tab:fit`. -/
theorem lossFits_pos (C : ℝ) (hC : 0 < C) : 0 < muonLossFit C ∧ 0 < adamLossFit C := by
  unfold muonLossFit adamLossFit
  constructor <;> positivity

/-- Positive compute budgets exist, arXiv:2502.16982, §3.2. -/
example : (0 : ℝ) < 1 := by norm_num

/-- The fitted loss ratio varies as `C^0.002`, rather than being a constant.
Source: arXiv:2502.16982, §3.2, `tab:fit`. -/
theorem lossFit_ratio (C : ℝ) (hC : 0 < C) :
    muonLossFit C / adamLossFit C = (2506 / 2608 : ℝ) * C ^ (1 / 500 : ℝ) := by
  unfold muonLossFit adamLossFit
  rw [mul_div_mul_comm, ← Real.rpow_sub hC]
  norm_num

/-- Positive compute budgets exist, arXiv:2502.16982, §3.2. -/
example : (0 : ℝ) < 1 := by norm_num

/-- Exact equal-loss compute for the reported fits,
arXiv:2502.16982, §3.2, `tab:fit`. This is a conditional fit calculation,
not verification of the experimental “about 52%” observation. -/
theorem equalLossMuonCompute_spec (C : ℝ) (hC : 0 < C) :
    muonLossFit (equalLossMuonCompute C) = adamLossFit C := by
  have hr : 0 < (2506 / 2608 : ℝ) := by norm_num
  have hpow : 0 < C ^ (27 / 26 : ℝ) := Real.rpow_pos_of_pos hC _
  unfold muonLossFit equalLossMuonCompute adamLossFit
  rw [Real.mul_rpow (Real.rpow_pos_of_pos hr _).le hpow.le,
    ← Real.rpow_mul hr.le, ← Real.rpow_mul hC.le]
  norm_num [Real.rpow_neg_one]
  ring

/-- Positive compute budgets exist, arXiv:2502.16982, §3.2. -/
example : (0 : ℝ) < 1 := by norm_num

/-- The equal-loss FLOPs fraction depends on the budget as `C^(1/26)`.
Consequently the paper's observed 52% saving is not an exact universal
constant of the printed fits. Source: arXiv:2502.16982, §3.2, `tab:fit`. -/
theorem equalLoss_compute_ratio (C : ℝ) (hC : 0 < C) :
    equalLossMuonCompute C / C = (2506 / 2608 : ℝ) ^ (250 / 13 : ℝ) * C ^ (1 / 26 : ℝ) := by
  unfold equalLossMuonCompute
  rw [mul_div_assoc, ← Real.rpow_sub_one hC.ne']
  norm_num

/-- Positive compute budgets exist, arXiv:2502.16982, §3.2. -/
example : (0 : ℝ) < 1 := by norm_num

/-- The baseline's fitted model size, arXiv:2502.16982, Appendix B,
`tab:dense_scaling_param`. -/
def modelSizeFit (C : ℝ) : ℝ := (483359 / 10000000) * C ^ (5112684 / 10000000 : ℝ)

/-- The baseline's fitted token count, arXiv:2502.16982, Appendix B,
`tab:dense_scaling_param`. -/
def tokensFit (C : ℝ) : ℝ := (34480927 / 10000000) * C ^ (4887316 / 10000000 : ℝ)

/-- The baseline's fitted learning rate, arXiv:2502.16982, Appendix B,
`tab:dense_scaling_param`. -/
def learningRateFit (C : ℝ) : ℝ := (127339 / 10000000) * C ^ (-(574752 / 10000000 : ℝ))

/-- The baseline's fitted batch size, arXiv:2502.16982, Appendix B,
`tab:dense_scaling_param`. -/
def batchSizeFit (C : ℝ) : ℝ := (65202 / 10000000) * C ^ (4137915 / 10000000 : ℝ)

/-- All four fitted hyperparameters are positive for positive compute,
arXiv:2502.16982, Appendix B, `tab:dense_scaling_param`. -/
theorem baselineFits_pos (C : ℝ) (hC : 0 < C) :
    0 < modelSizeFit C ∧ 0 < tokensFit C ∧ 0 < learningRateFit C ∧ 0 < batchSizeFit C := by
  unfold modelSizeFit tokensFit learningRateFit batchSizeFit
  exact ⟨by positivity, by positivity, by positivity, by positivity⟩

/-- Positive compute budgets exist, arXiv:2502.16982, Appendix B. -/
example : (0 : ℝ) < 1 := by norm_num

/-- The printed model-size and token-count exponents sum to one, so their
product is proportional to compute.
Source: arXiv:2502.16982, Appendix B, `tab:dense_scaling_param`. -/
theorem baseline_compute_product (C : ℝ) (hC : 0 < C) :
    6 * modelSizeFit C * tokensFit C =
      (6 * (483359 / 10000000 : ℝ) * (34480927 / 10000000)) * C := by
  unfold modelSizeFit tokensFit
  calc
    _ = (6 * (483359 / 10000000 : ℝ) * (34480927 / 10000000)) *
        (C ^ (5112684 / 10000000 : ℝ) * C ^ (4887316 / 10000000 : ℝ)) := by ring
    _ = _ := by rw [← Real.rpow_add hC]; norm_num

/-- Positive compute budgets exist, arXiv:2502.16982, Appendix B. -/
example : (0 : ℝ) < 1 := by norm_num

/-- Rounded fit coefficients do not satisfy `C=6ND` exactly, although the
experiment's search uses that equality. This records rounding rather than
equating the empirical fit with the exact compute constraint.
Source: arXiv:2502.16982, Appendix B, `tab:dense_scaling_param`. -/
theorem baseline_rounded_compute (C : ℝ) (hC : 0 < C) :
    6 * modelSizeFit C * tokensFit C ≠ C := by
  rw [baseline_compute_product C hC]
  norm_num
  nlinarith

/-- Positive compute budgets exist, arXiv:2502.16982, Appendix B. -/
example : (0 : ℝ) < 1 := by norm_num

end Transformer.Muon

module

public import Transformer.RoPE.YaRN.Calibration
public import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
# Why a linear log rule can approximate a conditional square-root law

Original analysis of the empirical fit in arXiv:2309.00071v3, §3.3,
Eq. (15). The independent variable here is log growth in the number of
competitors, not in the number of tokens. These ratios differ at finite
context lengths. No claim is made that the assumptions of this retention
objective describe a trained LLaMA's logits or its perplexity optimum.
-/

@[expose] public section

namespace Transformer.RoPE.YaRN

/-- Q/K multiplier as a function of log growth in competitor count. -/
noncomputable def gainAtLogGrowth (gap growth : ℝ) : ℝ := Real.sqrt (1 + growth / gap)

/-- The local linear coefficient is `1/(2*gap)`, not a universal constant.
This is an actual derivative at zero growth, rather than a fitted line. -/
theorem gain_at_log_growth_derivative (gap : ℝ) :
    HasDerivAt (gainAtLogGrowth gap) (1 / (2 * gap)) 0 := by
  have hi := ((hasDerivAt_id (0 : ℝ)).div_const gap).const_add 1
  have hs := hi.sqrt (by simp)
  change HasDerivAt (fun growth : ℝ => Real.sqrt (1 + growth / gap)) (1 / (2 * gap)) 0
  simpa only [id_eq, zero_div, add_zero, Real.sqrt_one, div_div, mul_comm, one_mul] using hs

/-- A gap of five produces YaRN's coefficient `0.1` only as a local slope
in log competitor growth, under this explicitly chosen retention model. -/
theorem coefficient_one_tenth_at_gap_five :
    HasDerivAt (gainAtLogGrowth 5) (1 / 10) 0 := by
  have h := gain_at_log_growth_derivative 5
  norm_num at h
  exact h

/-- The tangent approximation overestimates the required Q/K gain: its
squared logit gain has this exact extra quadratic term. No such term is
required by the retention model. -/
theorem linear_approximation_excess {gap growth : ℝ} (hd : 0 < gap) (hu : 0 ≤ growth) :
    (1 + growth / (2 * gap)) ^ 2 - gainAtLogGrowth gap growth ^ 2 =
      growth ^ 2 / (4 * gap ^ 2) := by
  have hi : 0 ≤ 1 + growth / gap := by positivity
  unfold gainAtLogGrowth
  rw [Real.sq_sqrt hi]
  field_simp
  ring

example : (1 + (2 : ℝ) / (2 * 5)) ^ 2 - gainAtLogGrowth 5 2 ^ 2 = 2 ^ 2 / (4 * 5 ^ 2) :=
  linear_approximation_excess (by norm_num) (by norm_num)

/-- At every strictly positive growth, the linear tangent already gives a
strictly larger logit gain than the exact conditional retention law. -/
theorem linear_approximation_strict_excess {gap growth : ℝ}
    (hd : 0 < gap) (hu : 0 < growth) :
    gainAtLogGrowth gap growth ^ 2 < (1 + growth / (2 * gap)) ^ 2 := by
  have he := linear_approximation_excess hd hu.le
  have hp : 0 < growth ^ 2 / (4 * gap ^ 2) := by positivity
  linarith

example : gainAtLogGrowth 5 2 ^ 2 < (1 + (2 : ℝ) / (2 * 5)) ^ 2 :=
  linear_approximation_strict_excess (by norm_num) (by norm_num)

end Transformer.RoPE.YaRN

import Transformer.Grokking.AdamW.ScalarDenominator

/-!
# Strict adaptive dissipation from an inserted native input

Source: PyTorch 2.14.1 AdamW, adam.py lines 445--475 second
moment insertion and lines 529--547 completed-clock corrections,
ported at lab commit 79f4fb0. This module retains the variance and
completed clock; it does not replace the denominator by epsilon.

At legal first beta zero, an input whose magnitude exceeds a
positive floor inserts its square into the actual second moment.
Even with arbitrary nonnegative retained variance, the corrected
denominator exceeds epsilon by an explicit square-root amount.
Consequently its adaptive increment loses a positive numerical gap
relative to the linear epsilon ceiling.

These scalar estimates take a current input floor as a hypothesis.
The CE application must derive such inputs on its own initialized
trajectory before concluding critical collapse; no future gradients,
parameter convergence, or successful classification are encoded here.
The second beta is arbitrary in [0,1), including retained beta2=0.98.
A nonnegative rate is allowed in the one-step estimate; strict mass
loss in a feedback application will require positive rate.

First beta zero is an explicit legal mathematical control, not a
reset or alteration of the frozen beta1=0.9 transformer experiments.
The formulas use exact reals and non-AMSGrad native AdamW. No claim
about learned Q/K, floating-point kernels or stochastic data follows.
-/

namespace Transformer.Grokking.AdamW

/-- The newly inserted input square gives an actual denominator
floor at legal first beta zero. Source: native AdamW at 79f4fb0,
second insertion and both corrections; retained variance is arbitrary
nonnegative, and the input floor need only be nonnegative. -/
theorem next_buffer_zero_first_denominator_input_floor
    (b2 eps variance gradient lower : ℝ) (clock : ℕ)
    (hb : 0 ≤ b2) (h2 : b2 < 1) (hv : 0 ≤ variance)
    (hl : 0 ≤ lower) (hg : lower ≤ |gradient|) :
    eps + Real.sqrt ((1 - b2) * lower ^ 2) ≤
      nextBufferDenominator 0 b2 eps variance gradient clock := by
  have hproduct := mul_nonneg (sub_nonneg.mpr hg) (add_nonneg (abs_nonneg gradient) hl)
  have hsq : lower ^ 2 ≤ gradient ^ 2 := by
    nlinarith only [hproduct, sq_abs gradient]
  have hvariance := mul_nonneg hb hv
  have hinsert := mul_le_mul_of_nonneg_left hsq (show 0 ≤ 1 - b2 by linarith only [h2])
  have hnext : (1 - b2) * lower ^ 2 ≤ b2 * variance + (1 - b2) * gradient ^ 2 := by
    linarith only [hvariance, hinsert]
  have hcorrection := bias_correction_positive b2 (clock + 1) hb h2 (by omega)
  have hpow := pow_nonneg hb (clock + 1)
  have hx : 0 ≤ (1 - b2) * lower ^ 2 := mul_nonneg (by linarith only [h2]) (sq_nonneg lower)
  have hscaled := mul_nonneg hx hpow
  have hdiv : (1 - b2) * lower ^ 2 ≤
      (b2 * variance + (1 - b2) * gradient ^ 2) / (1 - b2 ^ (clock + 1)) := by
    apply (le_div_iff₀ hcorrection).mpr
    nlinarith only [hscaled, hnext]
  have hroot := Real.sqrt_le_sqrt hdiv
  simpa only [nextBufferDenominator, zero_pow (by omega : clock + 1 ≠ 0),
    sub_zero, one_mul, add_comm] using add_le_add_left hroot eps

example : (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧
    (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ |(-1 : ℝ)| := by norm_num

/-- A positive native input floor makes the actual corrected
denominator strictly exceed epsilon. Source: native AdamW at
79f4fb0; the squared insertion remains positive for every beta2<1,
without a future variance limit or a supplied denominator. -/
theorem next_buffer_zero_first_denominator_strict_input_floor
    (b2 eps variance gradient lower : ℝ) (clock : ℕ)
    (hb : 0 ≤ b2) (h2 : b2 < 1) (hv : 0 ≤ variance)
    (hl : 0 < lower) (hg : lower ≤ |gradient|) :
    eps < nextBufferDenominator 0 b2 eps variance gradient clock := by
  have hsq : 0 < lower ^ 2 := by
    simpa only [pow_two] using mul_pos hl hl
  have hroot := Real.sqrt_pos.mpr (mul_pos (show 0 < 1 - b2 by linarith only [h2]) hsq)
  have hfloor := next_buffer_zero_first_denominator_input_floor b2 eps variance gradient lower clock
    hb h2 hv (le_of_lt hl) hg
  linarith only [hfloor, hroot]

example : (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧
    (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ |(-1 : ℝ)| := by norm_num

/-- A negative input bounded away from zero loses an explicit
positive amount from the linear epsilon adaptive ceiling. Source:
native AdamW at 79f4fb0; the actual second buffer, completed clock,
decay and insertion all remain in the step. No parameter sign or
current first-moment sign is needed when first beta is zero. -/
theorem scalar_zero_first_beta_adaptive_gap
    (b2 eps decay rate gradient lower : ℝ) (state : ScalarState)
    (hb : 0 ≤ b2) (h2 : b2 < 1) (hv : 0 ≤ state.variance)
    (he : 0 < eps) (heta : 0 ≤ rate) (hl : 0 < lower) (hg : gradient ≤ -lower) :
    let gap := lower / eps - lower / (eps + Real.sqrt ((1 - b2) * lower ^ 2))
    0 < gap ∧
      (scalarNativeStep 0 b2 eps decay rate state gradient).parameter ≤
        (1 - rate * decay) * state.parameter + rate * ((-gradient) / eps - gap) := by
  dsimp only
  let ceiling := eps + Real.sqrt ((1 - b2) * lower ^ 2)
  have hsq : 0 < lower ^ 2 := by simpa only [pow_two] using mul_pos hl hl
  have hroot := Real.sqrt_pos.mpr (mul_pos (show 0 < 1 - b2 by linarith only [h2]) hsq)
  have hceiling : eps < ceiling := by dsimp only [ceiling]; linarith only [hroot]
  have hceilingPos : 0 < ceiling := lt_trans he hceiling
  have hnegative : gradient ≤ 0 := by linarith only [hg, hl]
  have hr : lower ≤ -gradient := by linarith only [hg]
  have hfloor : ceiling ≤ nextBufferDenominator 0 b2 eps state.variance gradient state.clock := by
    exact next_buffer_zero_first_denominator_input_floor b2 eps state.variance gradient lower state.clock
      hb h2 hv (le_of_lt hl) (by simpa only [abs_of_nonpos hnegative] using hr)
  have hgap : 0 < lower / eps - lower / ceiling := by
    have h := div_lt_div_of_pos_left hl he hceiling
    linarith only [h]
  have hreciprocal : (1 : ℝ) / ceiling ≤ 1 / eps :=
    div_le_div_of_nonneg_left (by norm_num) he (le_of_lt hceiling)
  have hproduct := mul_nonneg (sub_nonneg.mpr hr) (sub_nonneg.mpr hreciprocal)
  have hidentity : (-gradient) / eps - (lower / eps - lower / ceiling) -
      (-gradient) / ceiling = ((-gradient) - lower) * (1 / eps - 1 / ceiling) := by ring
  have hlinear : (-gradient) / ceiling ≤ (-gradient) / eps - (lower / eps - lower / ceiling) := by
    linarith only [hproduct, hidentity]
  have hactual : (-gradient) / nextBufferDenominator 0 b2 eps state.variance gradient state.clock ≤
      (-gradient) / ceiling :=
    div_le_div_of_nonneg_left (neg_nonneg.mpr hnegative) hceilingPos hfloor
  refine ⟨hgap, ?_⟩
  calc
    (scalarNativeStep 0 b2 eps decay rate state gradient).parameter =
        (1 - rate * decay) * state.parameter +
          rate * ((-gradient) / nextBufferDenominator 0 b2 eps state.variance gradient state.clock) := by
      rw [scalar_parameter_denominator]
      simp only [scalarNativeStep, zero_mul, sub_zero, one_mul, zero_add]
      ring
    _ ≤ (1 - rate * decay) * state.parameter +
        rate * ((-gradient) / eps - (lower / eps - lower / ceiling)) :=
      add_le_add le_rfl (mul_le_mul_of_nonneg_left (le_trans hactual hlinear) heta)

example : (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) ≤ (seededScalarState 1).variance ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) < 1 / 2 ∧ (-1 : ℝ) ≤ -(1 / 2 : ℝ) := by
  norm_num [seededScalarState]

/-- Positive rate makes the actual adaptive ceiling strict for an
input bounded away from zero. Source: native AdamW at 79f4fb0;
this is a current numerical decrement, not an assumed asymptotic
loss decrease or an assertion about classification. -/
theorem scalar_zero_first_beta_strict_adaptive_ceiling
    (b2 eps decay rate gradient lower : ℝ) (state : ScalarState)
    (hb : 0 ≤ b2) (h2 : b2 < 1) (hv : 0 ≤ state.variance)
    (he : 0 < eps) (heta : 0 < rate) (hl : 0 < lower) (hg : gradient ≤ -lower) :
    (scalarNativeStep 0 b2 eps decay rate state gradient).parameter <
      (1 - rate * decay) * state.parameter + rate * ((-gradient) / eps) := by
  obtain ⟨hgap, hstep⟩ := scalar_zero_first_beta_adaptive_gap b2 eps decay rate gradient lower state
    hb h2 hv he (le_of_lt heta) hl hg
  have hpositive := mul_pos heta hgap
  nlinarith only [hstep, hpositive]

example : (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) ≤ (seededScalarState 1).variance ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) < 1 / 2 ∧ (-1 : ℝ) ≤ -(1 / 2 : ℝ) := by
  norm_num [seededScalarState]

end Transformer.Grokking.AdamW

import Transformer.Grokking.AdamW.RetainedInputDissipation

/-!
# A generated geometric error from the native first-clock correction

Source: PyTorch 2.14.1 AdamW, adam.py lines 529--547 completed-clock
first correction and full denominator, ported at 79f4fb0; retained
current parameter estimates at 315f601 and 9d87822. All numerical
buffers and their completed clock remain in the original step.

Separate the reciprocal first correction from the epsilon baseline.
Its excess is at most a geometric first-beta power times the newly
retained numerator divided by the first-clock epsilon floor. A current
upper bound on that numerator then supplies a fixed geometric budget
for the true adaptive parameter increment.

The scalar application still has to derive that newly retained
moment bound from actual initialized feedback. No future parameter,
gradient, variance or denominator limit is supplied here. The second
beta is unrestricted in the scalar epsilon-floor inequality; native
sign/variance induction in the CE application uses its legal range.

The estimate does not set the denominator to epsilon. It bounds the
complete square-root term and explicitly retains the finite-clock
correction error. These current bounds alone imply neither critical
parameter collapse nor circuit selection. A weighted native observer
must combine them with the true moment-feedback law and input-induced
dissipation. Exact reals and non-AMSGrad native AdamW are explicit;
frozen transformer optimizers, buffers and checkpoints are unchanged.
-/

namespace Transformer.Grokking.AdamW

/-- The corrected epsilon reciprocal has a geometric excess above
its uncorrected baseline. Source: native completed-clock correction
at 79f4fb0; the power uses clock+1, and a legal first-clock floor
bounds the numerical remainder without a future input limit. -/
theorem corrected_epsilon_ratio_ceiling (beta eps numerator : ℝ) (clock : ℕ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (he : 0 < eps) (hn : 0 ≤ numerator) :
    numerator / ((1 - beta ^ (clock + 1)) * eps) ≤ numerator / eps +
      (numerator / ((1 - beta) * eps)) * beta ^ (clock + 1) := by
  have hc := bias_correction_positive beta (clock + 1) hb h1 (by omega)
  have hf : 0 < (1 - beta) * eps := mul_pos (by linarith only [h1]) he
  have hp := pow_le_of_le_one hb (le_of_lt h1) (show clock + 1 ≠ 0 by omega)
  have hden : (1 - beta) * eps ≤ (1 - beta ^ (clock + 1)) * eps := by
    nlinarith only [mul_le_mul_of_nonneg_right hp (le_of_lt he)]
  have hupper := div_le_div_of_nonneg_left hn hf hden
  have hscaled := mul_le_mul_of_nonneg_right hupper (pow_nonneg hb (clock + 1))
  have hidentity : numerator / ((1 - beta ^ (clock + 1)) * eps) =
      numerator / eps + (numerator / ((1 - beta ^ (clock + 1)) * eps)) * beta ^ (clock + 1) := by
    field_simp [ne_of_gt he, ne_of_gt hc]
    ring
  nlinarith only [hscaled, hidentity]

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- A current bound on the newly retained negative moment generates
a complete native parameter ceiling with explicit geometric error.
Source: native AdamW at 79f4fb0; current variance is retained and no
instantaneous variance or corrected-denominator match is assumed. -/
theorem scalar_parameter_bias_error_ceiling
    (b1 b2 eps decay rate gradient bound : ℝ) (state : ScalarState)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (hm : state.moment ≤ 0) (hg : gradient ≤ 0)
    (hbound : -(scalarNativeStep b1 b2 eps decay rate state gradient).moment ≤ bound) :
    (scalarNativeStep b1 b2 eps decay rate state gradient).parameter ≤
      (1 - rate * decay) * state.parameter + rate *
        ((-(scalarNativeStep b1 b2 eps decay rate state gradient).moment) / eps) +
        rate * bound / ((1 - b1) * eps) * b1 ^ (state.clock + 1) := by
  let magnitude := -(scalarNativeStep b1 b2 eps decay rate state gradient).moment
  have hn : 0 ≤ magnitude := neg_nonneg.mpr
    (scalar_native_moment_nonpos b1 b2 eps decay rate gradient state hb1 (le_of_lt h1) hm hg)
  have hc := bias_correction_positive b1 (state.clock + 1) hb1 h1 (by omega)
  have hden : (1 - b1 ^ (state.clock + 1)) * eps ≤
      nextBufferDenominator b1 b2 eps state.variance gradient state.clock := by
    have hroot := mul_nonneg (le_of_lt hc) (Real.sqrt_nonneg
      ((b2 * state.variance + (1 - b2) * gradient ^ 2) / (1 - b2 ^ (state.clock + 1))))
    unfold nextBufferDenominator
    nlinarith only [hroot]
  have hactual := div_le_div_of_nonneg_left hn (mul_pos hc he) hden
  have hcorrected := corrected_epsilon_ratio_ceiling b1 eps magnitude state.clock hb1 h1 he hn
  have hfirst : 0 < (1 - b1) * eps := mul_pos (by linarith only [h1]) he
  have hcap := mul_le_mul_of_nonneg_right
    (div_le_div_of_nonneg_right hbound (le_of_lt hfirst)) (pow_nonneg hb1 (state.clock + 1))
  have hdivision : magnitude / nextBufferDenominator b1 b2 eps state.variance gradient state.clock ≤
      magnitude / eps + (bound / ((1 - b1) * eps)) * b1 ^ (state.clock + 1) := by
    nlinarith only [hactual, hcorrected, hcap]
  calc
    (scalarNativeStep b1 b2 eps decay rate state gradient).parameter =
        (1 - rate * decay) * state.parameter + rate *
          (magnitude / nextBufferDenominator b1 b2 eps state.variance gradient state.clock) := by
      rw [scalar_parameter_denominator]
      ring
    _ ≤ (1 - rate * decay) * state.parameter + rate *
        (magnitude / eps + (bound / ((1 - b1) * eps)) * b1 ^ (state.clock + 1)) :=
      add_le_add le_rfl (mul_le_mul_of_nonneg_left hdivision heta)
    _ = (1 - rate * decay) * state.parameter + rate * (magnitude / eps) +
        rate * bound / ((1 - b1) * eps) * b1 ^ (state.clock + 1) := by ring

example :
    let state : ScalarState := ⟨1, -10, 200, 1⟩
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
      state.moment ≤ 0 ∧ (-1 : ℝ) ≤ 0 ∧
      -(scalarNativeStep (9 / 10) (49 / 50) 1 (3 / 2) (1 / 1000) state (-1)).moment ≤ 10 := by
  norm_num [scalarNativeStep]

/-- The true current completed clock supplies a geometric error
budget indexed by its old count. Source: native AdamW at 79f4fb0;
the newly retained moment bound is current numerical data, and the
CE path must generate it together with its initialized clock law. -/
theorem scalar_parameter_clock_error_ceiling
    (b1 b2 eps decay rate gradient bound : ℝ) (state : ScalarState)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (hm : state.moment ≤ 0) (hg : gradient ≤ 0)
    (hbound : -(scalarNativeStep b1 b2 eps decay rate state gradient).moment ≤ bound) :
    (scalarNativeStep b1 b2 eps decay rate state gradient).parameter ≤
      (1 - rate * decay) * state.parameter + rate *
        ((-(scalarNativeStep b1 b2 eps decay rate state gradient).moment) / eps) +
        rate * bound / ((1 - b1) * eps) * b1 ^ state.clock := by
  have hn := scalar_native_moment_nonpos b1 b2 eps decay rate gradient state hb1 (le_of_lt h1) hm hg
  have hboundNonnegative := le_trans (neg_nonneg.mpr hn) hbound
  have hcoeff : 0 ≤ rate * bound / ((1 - b1) * eps) :=
    div_nonneg (mul_nonneg heta hboundNonnegative) (le_of_lt (mul_pos (by linarith only [h1]) he))
  have hpower : b1 ^ (state.clock + 1) ≤ b1 ^ state.clock := by
    rw [pow_succ]
    simpa only [mul_one] using mul_le_mul_of_nonneg_left (le_of_lt h1) (pow_nonneg hb1 state.clock)
  have hscaled := mul_le_mul_of_nonneg_left hpower hcoeff
  exact le_trans (scalar_parameter_bias_error_ceiling b1 b2 eps decay rate gradient bound state hb1 h1 he heta hm hg hbound)
    (add_le_add le_rfl hscaled)

example :
    let state : ScalarState := ⟨1, -10, 200, 1⟩
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
      state.moment ≤ 0 ∧ (-1 : ℝ) ≤ 0 ∧
      -(scalarNativeStep (9 / 10) (49 / 50) 1 (3 / 2) (1 / 1000) state (-1)).moment ≤ 10 := by
  norm_num [scalarNativeStep]

/-- The native parameter/first-moment weight cancels the epsilon
baseline and leaves a geometric completed-clock error. Source:
native AdamW at 79f4fb0 and the complete estimate above; the CE
application separately supplies its true retained feedback law. -/
theorem scalar_parameter_moment_clock_ceiling
    (b1 b2 eps decay rate gradient bound : ℝ) (state : ScalarState)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (hm : state.moment ≤ 0) (hg : gradient ≤ 0)
    (hbound : -(scalarNativeStep b1 b2 eps decay rate state gradient).moment ≤ bound) :
    let next := scalarNativeStep b1 b2 eps decay rate state gradient
    ((1 - b1) * eps) * next.parameter + rate * b1 * (-next.moment) ≤
      ((1 - b1) * eps) * (1 - rate * decay) * state.parameter +
        rate * (-next.moment) + rate * bound * b1 ^ state.clock := by
  dsimp only
  have hstep := scalar_parameter_clock_error_ceiling b1 b2 eps decay rate gradient bound state
    hb1 h1 he heta hm hg hbound
  have hf : 0 < (1 - b1) * eps := mul_pos (by linarith only [h1]) he
  have hscaled := mul_le_mul_of_nonneg_left hstep (le_of_lt hf)
  calc
    _ ≤ ((1 - b1) * eps) *
        ((1 - rate * decay) * state.parameter + rate *
          ((-(scalarNativeStep b1 b2 eps decay rate state gradient).moment) / eps) +
            rate * bound / ((1 - b1) * eps) * b1 ^ state.clock) +
        rate * b1 * (-(scalarNativeStep b1 b2 eps decay rate state gradient).moment) :=
      add_le_add hscaled le_rfl
    _ = _ := by
      have hfirst : 1 - b1 ≠ 0 := ne_of_gt (show 0 < 1 - b1 by linarith only [h1])
      field_simp [ne_of_gt he, hfirst]
      ring

example :
    let state : ScalarState := ⟨1, -10, 200, 1⟩
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
      state.moment ≤ 0 ∧ (-1 : ℝ) ≤ 0 ∧
      -(scalarNativeStep (9 / 10) (49 / 50) 1 (3 / 2) (1 / 1000) state (-1)).moment ≤ 10 := by
  norm_num [scalarNativeStep]

end Transformer.Grokking.AdamW

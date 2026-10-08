import Transformer.Grokking.AdamW.ScalarLimits
import Mathlib.Topology.Order.OrderClosed

/-!
# Actual retained AdamW denominator and vanishing-input limit

Source: PyTorch 2.14.1 AdamW, adam.py's completed-clock bias
corrections and addcdiv update, ported at lab commit 79f4fb0.
Move the first correction into the actual denominator without
resetting either moment. Derive its limit from the generated input
and native state recurrence, including an arbitrary retained clock.

The denominator below is a numerical expression, not a stability
predicate. A finite supplied input limit forces its limit to
abs(input) + epsilon. In particular, vanishing inputs give an
eventual epsilon-neighborhood bound. Parameter convergence and
vanishing CE inputs must be established by the application; they
are not consequences of clipping, sign data or these identities.
The formula uses exact reals and non-AMSGrad native updates. No
floating-point error, stochastic input convergence or GPTMini
generalization is asserted.
-/

namespace Transformer.Grokking.AdamW

open Filter

/-- Numerical denominator for the newly inserted first moment.
Source: native AdamW at 79f4fb0, both completed-clock corrections;
the square-root term uses the actual retained second insertion. -/
noncomputable def nextBufferDenominator (b1 b2 eps variance gradient : ℝ) (clock : ℕ) : ℝ :=
  (1 - b1 ^ (clock + 1)) *
    (Real.sqrt ((b2 * variance + (1 - b2) * gradient ^ 2) / (1 - b2 ^ (clock + 1))) + eps)

/-- The actual next direction divides the new retained numerator by
its complete denominator. Source: native AdamW at 79f4fb0; identity
retains clock indexing and does not replace variance by gradient squared. -/
theorem next_buffer_direction_denominator (b1 b2 eps moment variance gradient : ℝ) (clock : ℕ) :
    nextBufferDirection b1 b2 eps moment variance gradient clock =
      (b1 * moment + (1 - b1) * gradient) /
        nextBufferDenominator b1 b2 eps variance gradient clock := by
  unfold nextBufferDirection nextBufferDenominator
  rw [div_div]

/-- Positive epsilon and a valid first beta make the actual numerical
denominator positive. Source: native AdamW at 79f4fb0; nonnegativity
of the real square root needs no supplied instantaneous variance match. -/
theorem next_buffer_denominator_pos (b1 b2 eps variance gradient : ℝ) (clock : ℕ)
    (hb : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) :
    0 < nextBufferDenominator b1 b2 eps variance gradient clock := by
  have hc := bias_correction_positive b1 (clock + 1) hb h1 (by omega)
  have hs := Real.sqrt_nonneg ((b2 * variance + (1 - b2) * gradient ^ 2) / (1 - b2 ^ (clock + 1)))
  exact mul_pos hc (by linarith)

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 := by norm_num

/-- Exact parameter movement uses the newly retained moment and
the actual corrected denominator. Source: native AdamW at 79f4fb0;
both decay and adaptive insertion belong to the same native step. -/
theorem scalar_parameter_denominator (b1 b2 eps decay rate gradient : ℝ) (state : ScalarState) :
    (scalarNativeStep b1 b2 eps decay rate state gradient).parameter =
      (1 - rate * decay) * state.parameter + rate *
        (-(scalarNativeStep b1 b2 eps decay rate state gradient).moment) /
          nextBufferDenominator b1 b2 eps state.variance gradient state.clock := by
  change (1 - rate * decay) * state.parameter - rate *
    nextBufferDirection b1 b2 eps state.moment state.variance gradient state.clock = _
  rw [next_buffer_direction_denominator]
  dsimp only [scalarNativeStep]
  ring

/-- An actual upper denominator bound yields a lower parameter
increment while negative retained inputs persist. Source: native
AdamW at 79f4fb0; the application must derive the bound from its path. -/
theorem scalar_parameter_denominator_floor (b1 b2 eps decay rate gradient ceiling : ℝ)
    (state : ScalarState) (hb : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (hm : state.moment ≤ 0) (hg : gradient ≤ 0)
    (hd : nextBufferDenominator b1 b2 eps state.variance gradient state.clock ≤ ceiling) :
    (1 - rate * decay) * state.parameter + rate *
      (-(scalarNativeStep b1 b2 eps decay rate state gradient).moment) / ceiling ≤
        (scalarNativeStep b1 b2 eps decay rate state gradient).parameter := by
  have hn := scalar_native_moment_nonpos b1 b2 eps decay rate gradient state hb (le_of_lt h1) hm hg
  have hpos := next_buffer_denominator_pos b1 b2 eps state.variance gradient state.clock hb h1 he
  have hdiv := div_le_div_of_nonneg_left
    (show 0 ≤ rate * (-(scalarNativeStep b1 b2 eps decay rate state gradient).moment) from
      mul_nonneg heta (neg_nonneg.mpr hn)) hpos hd
  rw [scalar_parameter_denominator]
  linarith

example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (seededScalarState 1).moment ≤ 0 ∧ (-1 : ℝ) ≤ 0 ∧
    nextBufferDenominator 0 0 1 (seededScalarState 1).variance (-1) (seededScalarState 1).clock ≤ 2 := by
  norm_num [nextBufferDenominator, seededScalarState]

/-- The true corrected denominator has its native input-limit value.
Source: native AdamW at 79f4fb0; derive variance and correction limits
from the retained recurrence rather than postulating new buffer values. -/
theorem scalar_next_denominator_tendsto (b1 b2 eps decay rate value : ℝ)
    (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ n, state (n + 1) = scalarNativeStep b1 b2 eps decay rate (state n) (gradient n))
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (hg : Tendsto gradient atTop (nhds value)) :
    Tendsto (fun n => nextBufferDenominator b1 b2 eps (state n).variance (gradient n) (state n).clock)
      atTop (nhds (|value| + eps)) := by
  have hv : Tendsto (fun n => (state n).variance) atTop (nhds (value ^ 2)) := by
    exact Tendsto.congr (fun n => (scalar_path_fields b1 b2 eps decay rate state gradient n hstep).2.1.symm)
      (moment_tendsto_of_input_tendsto b2 (state 0).variance (value ^ 2) (fun n => gradient n ^ 2)
        hb2 h2 (hg.pow 2))
  have hc := scalar_completed_clock_tendsto b1 b2 eps decay rate state gradient hstep
  have hvn : Tendsto (fun n => b2 * (state n).variance + (1 - b2) * gradient n ^ 2)
      atTop (nhds (value ^ 2)) := by
    have ht := (hv.const_mul b2).add ((hg.pow 2).const_mul (1 - b2))
    have heq : b2 * value ^ 2 + (1 - b2) * value ^ 2 = value ^ 2 := by ring
    simpa only [heq] using ht
  have hvc := corrected_buffer_clock_tendsto b2 (value ^ 2) _ _ hb2 h2 hvn hc
  have hp := (tendsto_pow_atTop_nhds_zero_of_lt_one hb1 h1).comp hc
  have hbc : Tendsto (fun n => 1 - b1 ^ ((state n).clock + 1)) atTop (nhds 1) := by
    simpa only [Function.comp_apply, sub_zero] using tendsto_const_nhds.sub hp
  simpa only [nextBufferDenominator, Real.sqrt_sq_eq_abs, one_mul] using
    hbc.mul (hvc.sqrt.add_const eps)

example : (∀ n : ℕ, zeroScalarStateAt (n + 1) =
      scalarNativeStep 0 0 1 (1 / 10) (1 / 1000) (zeroScalarStateAt n) 0) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧
    Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (nhds 0) := by
  exact ⟨fun n => (scalar_native_zero_step 0 0 1 (1 / 10) (1 / 1000) n).symm,
    by norm_num, by norm_num, by norm_num, by norm_num, tendsto_const_nhds⟩

/-- Vanishing actual inputs eventually have denominator below any
strict epsilon ceiling. Source: native AdamW at 79f4fb0; completed
clocks and retained variances are derived, not reset for the estimate. -/
theorem scalar_zero_next_denominator_tail (b1 b2 eps decay rate ceiling : ℝ)
    (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ n, state (n + 1) = scalarNativeStep b1 b2 eps decay rate (state n) (gradient n))
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (hg : Tendsto gradient atTop (nhds 0)) (hd : eps < ceiling) :
    ∀ᶠ n in atTop,
      nextBufferDenominator b1 b2 eps (state n).variance (gradient n) (state n).clock < ceiling := by
  have ht := scalar_next_denominator_tendsto b1 b2 eps decay rate 0 state gradient hstep hb1 h1 hb2 h2 hg
  have hl : Tendsto (fun n => nextBufferDenominator b1 b2 eps (state n).variance (gradient n) (state n).clock)
      atTop (nhds eps) := by simpa only [abs_zero, zero_add] using ht
  exact hl.eventually_lt_const hd

example : (∀ n : ℕ, zeroScalarStateAt (n + 1) =
      scalarNativeStep 0 0 1 (1 / 10) (1 / 1000) (zeroScalarStateAt n) 0) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧
    Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (nhds 0) ∧ (1 : ℝ) < 2 := by
  exact ⟨fun n => (scalar_native_zero_step 0 0 1 (1 / 10) (1 / 1000) n).symm,
    by norm_num, by norm_num, by norm_num, by norm_num, tendsto_const_nhds, by norm_num⟩

end Transformer.Grokking.AdamW

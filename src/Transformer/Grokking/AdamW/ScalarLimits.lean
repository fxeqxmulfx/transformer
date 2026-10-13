import Transformer.Grokking.AdamW.ScalarRecurrence
import Transformer.Grokking.AdamW.MomentLimits

/-!
# Necessary finite parameter limits of the retained native recurrence

Source: PyTorch 2.14.1 AdamW at lab commit 9ff30f2, simultaneous
parameter decay, moment insertions and completed-clock corrections.
The supplied state sequence must satisfy the actual native update.
Derive its buffers and growing clock, then their limits from input
convergence. No convergence or instantaneous matching of moments is
assumed. Arbitrary retained finite initial buffers remain explicit.

At a positive constant rate, any finite parameter limit with a finite
gradient limit must balance parameter decay against the derived native
normalized direction. This is a necessary condition, not a proof of
parameter convergence, gradient descent, or rule selection. The CE
application must derive input convergence from its actual feedback;
stochastic minibatch gradients need not converge at stable parameters.
-/

namespace Transformer.Grokking.AdamW

open Filter

/-- Retained native states carry their actual causal moment histories
and completed clocks. Source: native AdamW at 9ff30f2; no moment is
reset and every supplied gradient, including zero, increments the clock. -/
theorem scalar_path_fields (b1 b2 eps decay rate : ℝ) (state : ℕ → ScalarState)
    (gradient : ℕ → ℝ) (n : ℕ)
    (hstep : ∀ k, state (k + 1) = scalarNativeStep b1 b2 eps decay rate (state k) (gradient k)) :
    (state n).moment = momentAt b1 gradient (state 0).moment n ∧
      (state n).variance = momentAt b2 (fun k => gradient k ^ 2) (state 0).variance n ∧
      (state n).clock = (state 0).clock + n := by
  induction n with
  | zero => exact ⟨rfl, rfl, rfl⟩
  | succ n ih =>
    rw [hstep n]
    change b1 * (state n).moment + (1 - b1) * gradient n = momentAt b1 _ _ (n + 1) ∧
      b2 * (state n).variance + (1 - b2) * gradient n ^ 2 = momentAt b2 _ _ (n + 1) ∧
      (state n).clock + 1 = (state 0).clock + (n + 1)
    rw [ih.1, ih.2.1, ih.2.2]
    refine ⟨rfl, rfl, ?_⟩
    omega

example : ∀ n : ℕ,
    ({ parameter := 1, moment := -1, variance := 1, clock := n + 2 } : ScalarState) =
      scalarNativeStep 0 0 1 (1 / 2) (1 / 1000)
        { parameter := 1, moment := -1, variance := 1, clock := n + 1 } (-1) := by
  intro n
  norm_num [scalarNativeStep, nextBufferDirection, zero_pow (by omega : n + 1 + 1 ≠ 0)]

/-- The actual completed clock diverges rather than converging to a
finite state coordinate. Source: native AdamW at 9ff30f2, one supplied
gradient per step; arbitrary retained starting clocks are permitted. -/
theorem scalar_completed_clock_tendsto (b1 b2 eps decay rate : ℝ) (state : ℕ → ScalarState)
    (gradient : ℕ → ℝ)
    (hstep : ∀ k, state (k + 1) = scalarNativeStep b1 b2 eps decay rate (state k) (gradient k)) :
    Tendsto (fun n => (state n).clock + 1) atTop atTop := by
  apply Tendsto.congr (f₁ := fun n => n + ((state 0).clock + 1))
    (fun n => by rw [(scalar_path_fields b1 b2 eps decay rate state gradient n hstep).2.2]; omega)
  exact tendsto_add_atTop_nat _

example : ∀ n : ℕ,
    ({ parameter := 1, moment := -1, variance := 1, clock := n + 2 } : ScalarState) =
      scalarNativeStep 0 0 1 (1 / 2) (1 / 1000)
        { parameter := 1, moment := -1, variance := 1, clock := n + 1 } (-1) := by
  intro n
  norm_num [scalarNativeStep, nextBufferDirection, zero_pow (by omega : n + 1 + 1 ≠ 0)]

/-- Actual retained directions converge to the normalized input limit
under the native recurrence. Source: native AdamW at 9ff30f2; derive
both moment limits and the unbounded clock rather than hypothesizing
them. Epsilon and valid beta ranges control the limiting denominator. -/
theorem scalar_direction_tendsto (b1 b2 eps decay rate value : ℝ)
    (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ k, state (k + 1) = scalarNativeStep b1 b2 eps decay rate (state k) (gradient k))
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (hg : Tendsto gradient atTop (nhds value)) :
    Tendsto (fun n => nextBufferDirection b1 b2 eps (state n).moment (state n).variance
      (gradient n) (state n).clock) atTop (nhds (value / (|value| + eps))) := by
  have hm : Tendsto (fun n => (state n).moment) atTop (nhds value) := by
    exact Tendsto.congr (fun n => (scalar_path_fields b1 b2 eps decay rate state gradient n hstep).1.symm)
      (moment_tendsto_of_input_tendsto b1 (state 0).moment value gradient hb1 h1 hg)
  have hv : Tendsto (fun n => (state n).variance) atTop (nhds (value ^ 2)) := by
    exact Tendsto.congr (fun n => (scalar_path_fields b1 b2 eps decay rate state gradient n hstep).2.1.symm)
      (moment_tendsto_of_input_tendsto b2 (state 0).variance (value ^ 2) (fun k => gradient k ^ 2)
        hb2 h2 (hg.pow 2))
  have hc := scalar_completed_clock_tendsto b1 b2 eps decay rate state gradient hstep
  have hmn : Tendsto (fun n => b1 * (state n).moment + (1 - b1) * gradient n)
      atTop (nhds value) := by
    have ht := (hm.const_mul b1).add (hg.const_mul (1 - b1))
    have heq : b1 * value + (1 - b1) * value = value := by ring
    simpa only [heq] using ht
  have hvn : Tendsto (fun n => b2 * (state n).variance + (1 - b2) * gradient n ^ 2)
      atTop (nhds (value ^ 2)) := by
    have ht := (hv.const_mul b2).add ((hg.pow 2).const_mul (1 - b2))
    have heq : b2 * value ^ 2 + (1 - b2) * value ^ 2 = value ^ 2 := by ring
    simpa only [heq] using ht
  have hmc := corrected_buffer_clock_tendsto b1 value _ _ hb1 h1 hmn hc
  have hvc := corrected_buffer_clock_tendsto b2 (value ^ 2) _ _ hb2 h2 hvn hc
  have hd : Real.sqrt (value ^ 2) + eps ≠ 0 := by
    rw [Real.sqrt_sq_eq_abs]
    have ha := abs_nonneg value
    linarith
  simpa only [Pi.div_def, nextBufferDirection, Real.sqrt_sq_eq_abs] using
    hmc.div (hvc.sqrt.add_const eps) hd

example : (∀ n : ℕ,
      ({ parameter := 1, moment := -1, variance := 1, clock := n + 2 } : ScalarState) =
        scalarNativeStep 0 0 1 (1 / 2) (1 / 1000)
          { parameter := 1, moment := -1, variance := 1, clock := n + 1 } (-1)) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    Tendsto (fun _ : ℕ => (-1 : ℝ)) atTop (nhds (-1)) := by
  refine ⟨?_, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, tendsto_const_nhds⟩
  intro n
  norm_num [scalarNativeStep, nextBufferDirection, zero_pow (by omega : n + 1 + 1 ≠ 0)]

/-- Any finite convergent native parameter/input pair balances decay
against the actual normalized direction. Source: native AdamW at
9ff30f2, constant positive rate; finite convergence of parameters is
explicit, and convergence of moments is a derived consequence. -/
theorem scalar_limit_balance (b1 b2 eps decay rate parameter value : ℝ)
    (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ k, state (k + 1) = scalarNativeStep b1 b2 eps decay rate (state k) (gradient k))
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate)
    (hp : Tendsto (fun n => (state n).parameter) atTop (nhds parameter))
    (hg : Tendsto gradient atTop (nhds value)) :
    decay * parameter + value / (|value| + eps) = 0 := by
  have hd := scalar_direction_tendsto b1 b2 eps decay rate value state gradient hstep hb1 h1 hb2 h2 he hg
  have hn : Tendsto (fun n => (state (n + 1)).parameter) atTop (nhds parameter) := by
    simpa only [Function.comp_def] using hp.comp (tendsto_add_atTop_nat 1)
  have hu := (hp.const_mul (1 - rate * decay)).sub (hd.const_mul rate)
  have hl : Tendsto (fun n => (state (n + 1)).parameter) atTop
      (nhds ((1 - rate * decay) * parameter - rate * (value / (|value| + eps)))) := by
    apply Tendsto.congr _ hu
    intro n
    rw [hstep n]
    rfl
  have heq := tendsto_nhds_unique hn hl
  have hz : rate * (decay * parameter + value / (|value| + eps)) = 0 := by nlinarith [heq]
  exact (mul_eq_zero.mp hz).resolve_left (ne_of_gt heta)

example : (∀ n : ℕ,
      ({ parameter := 1, moment := -1, variance := 1, clock := n + 2 } : ScalarState) =
        scalarNativeStep 0 0 1 (1 / 2) (1 / 1000)
          { parameter := 1, moment := -1, variance := 1, clock := n + 1 } (-1)) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1) ∧
    Tendsto (fun _ : ℕ => (-1 : ℝ)) atTop (nhds (-1)) := by
  refine ⟨?_, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    tendsto_const_nhds, tendsto_const_nhds⟩
  intro n
  norm_num [scalarNativeStep, nextBufferDirection, zero_pow (by omega : n + 1 + 1 ≠ 0)]

end Transformer.Grokking.AdamW

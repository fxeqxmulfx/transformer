import Transformer.Grokking.CircuitEfficiency.SectionC_NativeGradientLimits
import Transformer.Grokking.AdamW.ScalarLimits

/-!
# Finite limits of the closed clipped native product-CE trajectory

Sources: Varma et al., arXiv:2309.02390v1, appendix C's four product
factors and train CE; retained PyTorch 2.14.1 AdamW/clipping at lab
commit 367b3b2. Derive every gradient and buffer limit from current
parameter convergence and the actual closed recurrence. The growing
clock is never required to have a finite limit.

Any finite parameter limit must balance uniform parameter decay
against the actual normalized clipped derivative. Without decay,
positive clipping bound forces every finite limit to be the zero
parameter point. This needs no positivity premise on the parameters.
The actual cold trace satisfies all convergence hypotheses, including
retained zero buffers and increasing clocks; nonzero seeds need a
separate dynamics analysis. Plain CE/native decay is distinct from
appendix C's coupled penalty/GD. No stochastic GPTMini convergence
or floating-point kernel bridge is asserted.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- The actual CE callback vanishes at the fully cold parameter point.
Source: appendix C's product chain rule; the native shared multiplier
cannot manufacture a nonzero derivative from four zero factors. -/
theorem native_cold_applied_gradient (remaining : ℕ) (bound : ℝ) (clock : ℕ) (i : Fin 4) :
    appliedNativeSubweightGradient remaining bound (fun _ => zeroScalarStateAt clock) i = 0 := by
  change coordinateClipFactor bound _ * rawNativeSubweightGradient remaining _ i = 0
  rw [native_raw_gradient_partner]
  change _ * ((0 : ℝ) * _) = 0
  rw [zero_mul, mul_zero]

/-- A cold native step retains its actual zero buffers and advances
all four clocks. Sources: appendix C product CE and native AdamW at
367b3b2; no beta, rate or epsilon premise is needed for this identity. -/
theorem native_cold_step (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ) (clock : ℕ) :
    nativeSubweightStep remaining bound b1 b2 eps decay rate (fun _ => zeroScalarStateAt clock) =
      (fun _ => zeroScalarStateAt (clock + 1)) := by
  funext i
  change scalarNativeStep b1 b2 eps decay rate (zeroScalarStateAt clock) _ = _
  rw [native_cold_applied_gradient]
  exact scalar_native_zero_step b1 b2 eps decay rate clock

/-- The actual closed cold path is stationary in parameters and
buffers, with an unbounded completed clock. Sources: appendix C's
product CE and retained native AdamW at 367b3b2. The initial point
uses native zero buffers; a zero-parameter state with stale momentum
is not identified with this cold trace. All four updates are retained
and evaluated through the actual current CE callback. -/
theorem native_cold_path (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ) (n : ℕ) :
    nativeSubweightPath remaining bound b1 b2 eps decay rate (fun _ => zeroScalarStateAt 0) n =
      (fun _ => zeroScalarStateAt n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change nativeSubweightStep remaining bound b1 b2 eps decay rate _ = _
    rw [ih]
    exact native_cold_step remaining bound b1 b2 eps decay rate n

/-- Every coordinate of a finite parameter limit satisfies the native
balance with its actual CE callback. Sources: appendix C product CE
and native AdamW at 367b3b2; input and moment convergence are derived,
not supplied, and reference buffers/clock are not limiting data. -/
theorem native_closed_limit_balance (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState) (i : Fin 4)
    (hstep : ∀ n, state (n + 1) = nativeSubweightStep remaining bound b1 b2 eps decay rate (state n))
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate)
    (hp : ∀ j, Tendsto (fun n => (state n j).parameter) atTop (nhds (reference j).parameter)) :
    decay * (reference i).parameter + appliedNativeSubweightGradient remaining bound reference i /
      (|appliedNativeSubweightGradient remaining bound reference i| + eps) = 0 := by
  apply scalar_limit_balance b1 b2 eps decay rate (reference i).parameter
    (appliedNativeSubweightGradient remaining bound reference i) (fun n => state n i)
    (fun n => appliedNativeSubweightGradient remaining bound (state n) i)
    _ hb1 h1 hb2 h2 he heta (hp i)
    (native_applied_gradient_tendsto remaining bound state reference i hp)
  intro n
  exact congrFun (hstep n) i

example : (∀ n : ℕ, (fun _ : Fin 4 => zeroScalarStateAt (n + 1)) =
      nativeSubweightStep 111 1 (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
        (fun _ => zeroScalarStateAt n)) ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 1000 ∧
    (∀ i : Fin 4, Tendsto (fun n : ℕ => (zeroScalarStateAt n).parameter)
      atTop (nhds ((fun _ : Fin 4 => zeroScalarStateAt 0) i).parameter)) := by
  refine ⟨fun n => (native_cold_step 111 1 _ _ _ _ _ n).symm, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  exact tendsto_const_nhds

/-- At positive clip bound, all four zero actual applied gradients
force all four parameters to vanish. Sources: appendix C's actual
partner derivatives and native clipping at 367b3b2; the common CE
slope is strictly negative at every finite parameter point. -/
theorem native_zero_gradient_parameters (remaining : ℕ) (bound : ℝ) (state : NativeSubweightState)
    (hclip : 0 < bound) (hg : ∀ i, appliedNativeSubweightGradient remaining bound state i = 0) :
    ∀ i, (state i).parameter = 0 := by
  intro i
  have hc := coordinate_clip_factor_pos bound (rawNativeSubweightGradient remaining state) hclip
  have hz := hg (nativeFactorPartner i)
  change coordinateClipFactor bound _ * rawNativeSubweightGradient remaining _ _ = 0 at hz
  have hr := (mul_eq_zero.mp hz).resolve_left (ne_of_gt hc)
  rw [native_raw_gradient_partner] at hr
  have hs := table_train_ce_slope_negative remaining (nativeTotalScore state)
  have hp := (mul_eq_zero.mp hr).resolve_right (ne_of_lt hs)
  have hi : nativeFactorPartner (nativeFactorPartner i) = i := by fin_cases i <;> rfl
  simpa only [hi] using hp

example : (0 : ℝ) < 1 ∧
    (∀ i, appliedNativeSubweightGradient 111 1 (fun _ => zeroScalarStateAt 0) i = 0) := by
  exact ⟨by norm_num, native_cold_applied_gradient 111 1 0⟩

/-- Without decay, the only possible finite limit of actual native
CE parameters is zero. Sources: appendix C product CE and native
AdamW at 367b3b2; this is conditional on parameter convergence,
without gradient, moment-matching or parameter-sign hypotheses. -/
theorem native_no_decay_finite_limit_zero (remaining : ℕ) (bound b1 b2 eps rate : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState)
    (hstep : ∀ n, state (n + 1) = nativeSubweightStep remaining bound b1 b2 eps 0 rate (state n))
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate)
    (hp : ∀ j, Tendsto (fun n => (state n j).parameter) atTop (nhds (reference j).parameter)) :
    ∀ i, (reference i).parameter = 0 := by
  apply native_zero_gradient_parameters remaining bound reference hclip
  intro i
  have hb := native_closed_limit_balance remaining bound b1 b2 eps 0 rate state reference i
    hstep hb1 h1 hb2 h2 he heta hp
  have hz : appliedNativeSubweightGradient remaining bound reference i /
      (|appliedNativeSubweightGradient remaining bound reference i| + eps) = 0 := by
    simpa only [zero_mul, zero_add] using hb
  have hd : |appliedNativeSubweightGradient remaining bound reference i| + eps ≠ 0 := by positivity
  exact (div_eq_zero_iff.mp hz).resolve_right hd

example : (∀ n : ℕ, (fun _ : Fin 4 => zeroScalarStateAt (n + 1)) =
      nativeSubweightStep 111 1 (9 / 10) (49 / 50) (1 / 100000000) 0 (1 / 1000)
        (fun _ => zeroScalarStateAt n)) ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 1000 ∧
    (∀ i : Fin 4, Tendsto (fun n : ℕ => (zeroScalarStateAt n).parameter)
      atTop (nhds ((fun _ : Fin 4 => zeroScalarStateAt 0) i).parameter)) := by
  refine ⟨fun n => (native_cold_step 111 1 _ _ _ _ _ n).symm, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  exact tendsto_const_nhds

end Transformer.Grokking.CircuitEfficiency

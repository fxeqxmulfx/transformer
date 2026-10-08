import Transformer.Grokking.CircuitEfficiency.SectionC_GainGradientLimits

/-!
# Continuous actual gained-CE parameter families

Sources: Varma et al., arXiv:2309.02390v1, appendix C multiclass
product CE; actual parameter-only callback and native shared clipping
at c268c1f/9970d92. Along any continuous physical parameter family,
both gained products, every true raw partial, the full norm-two
clip factor, the actual common CE scale and every applied input
are continuous. The family may contain zero parameters or gradients.

Every finite competitor remains in the positive CE denominator.
The clipping norm uses all four squared partials and its added
1e-6 strip, so continuity does not require a nonzero gradient norm.
No independent clipped-input family or pointwise derivative signs
are supplied. Optimizer buffers and clocks do not enter the CE
callback; their retained numerical dynamics remain a separate task.

These mathematical continuity laws permit actual balance existence
arguments at fixed numerical decay and epsilon. They do not prove
smoothness at clipping/norm boundaries, convergence or attraction.
Fixed physical gains and complete table CE differ from learned
GPTMini Q/K, FFN and stochastic minibatches. The source's coupled
norm-cost GD and floating-point implementation are not transferred.

Proofs extend the actual parameter-limit calculations in
GainGradientLimits at 9970d92 from sequences to continuous families.
The parameter family has no accuracy or optimizer-success field.
Neither positive physical factors nor finite optimizer clock limits
are required by these complete CE and shared clipping calculations.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW
open scoped BigOperators

/-- Both actual gained product logits vary continuously with their
physical factors. Source: appendix C sim-overall-logits and explicit
physical readouts at 4e10267; all four factors remain in the score. -/
theorem gain_native_total_score_continuous (genGain memGain : ℝ) (state : ℝ → NativeSubweightState)
    (hp : ∀ i, Continuous (fun x => (state x i).parameter)) :
    Continuous (fun x => gainNativeTotalScore genGain memGain (state x)) := by
  have hg := ((hp 0).mul (hp 1)).const_mul genGain
  have hm := ((hp 2).mul (hp 3)).const_mul memGain
  simpa only [gainNativeTotalScore, physicalCircuitScore, Pi.mul_def, Pi.add_def] using hg.add hm

example : ∀ i : Fin 4, Continuous (fun x : ℝ =>
    ({ parameter := x + (i.val : ℝ) + 1, moment := 0, variance := 0, clock := 0 } : ScalarState).parameter) := by
  intro i
  exact (continuous_id.add_const (i.val : ℝ)).add_const 1

/-- Every actual true CE partial is continuous along the physical
family. Source: appendix C's product chain rule and true derivatives
at 9970d92; positive finite-class CE denominators never vanish. -/
theorem gain_native_raw_gradient_continuous (remaining : ℕ) (genGain memGain : ℝ)
    (state : ℝ → NativeSubweightState) (i : Fin 4)
    (hp : ∀ j, Continuous (fun x => (state x j).parameter)) :
    Continuous (fun x => rawGainNativeGradient remaining genGain memGain (state x) i) := by
  have hs := gain_native_total_score_continuous genGain memGain state hp
  have hd := (hs.rexp.add_const (remaining : ℝ)).add_const 1
  have hn : ∀ x, Real.exp (gainNativeTotalScore genGain memGain (state x)) + (remaining : ℝ) + 1 ≠ 0 := by
    intro x
    positivity
  have hc : Continuous (fun _ : ℝ => -((remaining : ℝ) + 1)) := continuous_const
  have ht := ((hp (nativeFactorPartner i)).const_mul (nativeFactorGain genGain memGain i)).mul (hc.div hd hn)
  simpa only [Pi.div_def, Pi.mul_def, rawGainNativeGradient] using ht

example : ∀ i : Fin 4, Continuous (fun x : ℝ =>
    ({ parameter := x + (i.val : ℝ) + 1, moment := 0, variance := 0, clock := 0 } : ScalarState).parameter) := by
  intro i
  exact (continuous_id.add_const (i.val : ℝ)).add_const 1

/-- The actual full raw-gradient norm is continuous, including at
zero gradient. Source: native norm_type=2 at 9970d92, keeping all
four squared true partials and the real square root. -/
theorem gain_native_gradient_norm_continuous (remaining : ℕ) (genGain memGain : ℝ)
    (state : ℝ → NativeSubweightState)
    (hp : ∀ i, Continuous (fun x => (state x i).parameter)) :
    Continuous (fun x => coordinateGradientNorm (rawGainNativeGradient remaining genGain memGain (state x))) := by
  have hsum := continuous_finsetSum Finset.univ
    (fun i _ => (gain_native_raw_gradient_continuous remaining genGain memGain state i hp).pow 2)
  simpa only [coordinateGradientNorm, Function.comp_def, Pi.pow_apply] using Real.continuous_sqrt.comp hsum

example : ∀ i : Fin 4, Continuous (fun x : ℝ =>
    ({ parameter := x + (i.val : ℝ) + 1, moment := 0, variance := 0, clock := 0 } : ScalarState).parameter) := by
  intro i
  exact (continuous_id.add_const (i.val : ℝ)).add_const 1

/-- Actual shared clipping stays continuous across its saturation
boundary and zero gradient. Source: native clip_grad_norm_ at
9970d92; its norm-plus-1e-6 denominator remains strictly positive. -/
theorem gain_native_clip_factor_continuous (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : ℝ → NativeSubweightState)
    (hp : ∀ i, Continuous (fun x => (state x i).parameter)) :
    Continuous (fun x => coordinateClipFactor bound (rawGainNativeGradient remaining genGain memGain (state x))) := by
  have hn := gain_native_gradient_norm_continuous remaining genGain memGain state hp
  have hd : ∀ x, coordinateGradientNorm (rawGainNativeGradient remaining genGain memGain (state x)) + 1 / 1000000 ≠ 0 := by
    intro x
    have hs := Real.sqrt_nonneg (∑ i, rawGainNativeGradient remaining genGain memGain (state x) i ^ 2)
    unfold coordinateGradientNorm
    linarith
  have hb : Continuous (fun _ : ℝ => bound) := continuous_const
  have hunit : Continuous (fun _ : ℝ => (1 : ℝ)) := continuous_const
  simpa only [coordinateClipFactor, Pi.div_def] using hunit.min (hb.div (hn.add_const (1 / 1000000)) hd)

example : ∀ i : Fin 4, Continuous (fun x : ℝ =>
    ({ parameter := x + (i.val : ℝ) + 1, moment := 0, variance := 0, clock := 0 } : ScalarState).parameter) := by
  intro i
  exact (continuous_id.add_const (i.val : ℝ)).add_const 1

/-- The common CE feedback scale is continuous as generated by
current physical parameters. Sources: appendix C full CE and shared
clipping at c268c1f; no independent scale function is postulated. -/
theorem gain_ce_gradient_scale_continuous (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : ℝ → NativeSubweightState)
    (hp : ∀ i, Continuous (fun x => (state x i).parameter)) :
    Continuous (fun x => gainCEGradientScale remaining genGain memGain bound (state x)) := by
  have hc := gain_native_clip_factor_continuous remaining genGain memGain bound state hp
  have hs := gain_native_total_score_continuous genGain memGain state hp
  have hd := (hs.rexp.add_const (remaining : ℝ)).add_const 1
  have hn : ∀ x, Real.exp (gainNativeTotalScore genGain memGain (state x)) + (remaining : ℝ) + 1 ≠ 0 := by
    intro x
    positivity
  have hcount : Continuous (fun _ : ℝ => (remaining : ℝ) + 1) := continuous_const
  simpa only [gainCEGradientScale, Pi.div_def, Pi.mul_def] using hc.mul (hcount.div hd hn)

example : ∀ i : Fin 4, Continuous (fun x : ℝ =>
    ({ parameter := x + (i.val : ℝ) + 1, moment := 0, variance := 0, clock := 0 } : ScalarState).parameter) := by
  intro i
  exact (continuous_id.add_const (i.val : ℝ)).add_const 1

/-- Every actually inserted clipped CE partial varies continuously
with the numerical parameter family. Sources: appendix C true
partials and native shared clipping at 9970d92; buffers are retained
by the optimizer rather than substituted into this callback. -/
theorem gain_native_applied_gradient_continuous (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : ℝ → NativeSubweightState) (i : Fin 4)
    (hp : ∀ j, Continuous (fun x => (state x j).parameter)) :
    Continuous (fun x => appliedGainNativeGradient remaining genGain memGain bound (state x) i) := by
  have hc := gain_native_clip_factor_continuous remaining genGain memGain bound state hp
  have hr := gain_native_raw_gradient_continuous remaining genGain memGain state i hp
  simpa only [appliedGainNativeGradient, coordinateClippedGradient, Pi.mul_def] using hc.mul hr

example : ∀ i : Fin 4, Continuous (fun x : ℝ =>
    ({ parameter := x + (i.val : ℝ) + 1, moment := 0, variance := 0, clock := 0 } : ScalarState).parameter) := by
  intro i
  exact (continuous_id.add_const (i.val : ℝ)).add_const 1

end Transformer.Grokking.CircuitEfficiency

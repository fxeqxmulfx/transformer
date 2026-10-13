import Transformer.Grokking.CircuitEfficiency.SectionC_GainRecurrence
import Mathlib.Topology.Order.OrderClosed

/-!
# Actual gained-forward CE input limits from physical parameter limits

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C;
native AdamW/clipping at lab commit 9970d92. Derive the actual score,
coordinate derivative, shared gradient norm, clipping and applied
input limits from convergence of current physical parameters. Actual
train and test CE retain both gained products and all finite classes.
Reference buffers and clocks are not optimizer-state limiting data.

Gains change the actual forward and chain-rule factors; decay remains
uniform and no circuit cost is added to CE. Raw zero gradients cause
no singularity in the native norm-plus-1e-6 clipping denominator.
The statements supply neither gradient/moment convergence nor a
successful margin as independent premises. Parameter convergence is
explicit, not deduced from clipping. Fixed-table deterministic CE is
not identified with changing learned features, stochastic minibatches
or floating-point GPTMini kernels.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW
open scoped BigOperators

/-- Both physical gained products have their actual score limit.
Source: appendix C, sim-overall-logits with section 3 efficiency in
fixed physical readouts rather than the source's assigned norm cost. -/
theorem gain_native_total_score_tendsto (genGain memGain : ℝ) (state : ℕ → NativeSubweightState)
    (reference : NativeSubweightState)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) :
    Tendsto (fun n => gainNativeTotalScore genGain memGain (state n)) atTop
      (nhds (gainNativeTotalScore genGain memGain reference)) := by
  have hg := ((hp 0).mul (hp 1)).const_mul genGain
  have hm := ((hp 2).mul (hp 3)).const_mul memGain
  simpa only [gainNativeTotalScore, physicalCircuitScore] using hg.add hm

example : ∀ i : Fin 4, Tendsto (fun n : ℕ =>
    ({ parameter := (i.val : ℝ) + 1, moment := 0, variance := 0, clock := n } : ScalarState).parameter)
    atTop (nhds ((i.val : ℝ) + 1)) := by
  intro i
  exact tendsto_const_nhds

/-- Every actual coordinate CE derivative converges to its actual
finite-point value. Source: appendix C's true multiclass CE and
product chain rule, with each physical gain retained in both places. -/
theorem gain_native_raw_gradient_tendsto (remaining : ℕ) (genGain memGain : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState) (i : Fin 4)
    (hp : ∀ j, Tendsto (fun n => (state n j).parameter) atTop (nhds (reference j).parameter)) :
    Tendsto (fun n => rawGainNativeGradient remaining genGain memGain (state n) i) atTop
      (nhds (rawGainNativeGradient remaining genGain memGain reference i)) := by
  have hs := gain_native_total_score_tendsto genGain memGain state reference hp
  have hd := (hs.rexp.add_const (remaining : ℝ)).add_const 1
  have hn : Real.exp (gainNativeTotalScore genGain memGain reference) + (remaining : ℝ) + 1 ≠ 0 := by positivity
  have hc : Tendsto (fun _ : ℕ => -((remaining : ℝ) + 1)) atTop
      (nhds (-((remaining : ℝ) + 1))) := tendsto_const_nhds
  have ht := ((hp (nativeFactorPartner i)).const_mul (nativeFactorGain genGain memGain i)).mul (hc.div hd hn)
  simpa only [Pi.div_def, rawGainNativeGradient] using ht

example : ∀ i : Fin 4, Tendsto (fun n : ℕ =>
    ({ parameter := (i.val : ℝ) + 1, moment := 0, variance := 0, clock := n } : ScalarState).parameter)
    atTop (nhds ((i.val : ℝ) + 1)) := by
  intro i
  exact tendsto_const_nhds

/-- The shared norm of all actual gained partials has its finite
reference value as limit. Source: native norm_type=2 at 9970d92;
different gains do not replace the norm by separate circuit clips. -/
theorem gain_native_gradient_norm_tendsto (remaining : ℕ) (genGain memGain : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) :
    Tendsto (fun n => coordinateGradientNorm (rawGainNativeGradient remaining genGain memGain (state n)))
      atTop (nhds (coordinateGradientNorm (rawGainNativeGradient remaining genGain memGain reference))) := by
  have ht := tendsto_finsetSum Finset.univ
    (fun i _ => (gain_native_raw_gradient_tendsto remaining genGain memGain state reference i hp).pow 2)
  simpa only [coordinateGradientNorm] using ht.sqrt

example : ∀ i : Fin 4, Tendsto (fun n : ℕ =>
    ({ parameter := (i.val : ℝ) + 1, moment := 0, variance := 0, clock := n } : ScalarState).parameter)
    atTop (nhds ((i.val : ℝ) + 1)) := by
  intro i
  exact tendsto_const_nhds

/-- Shared clipping converges without a nonzero-gradient premise.
Source: clip_grad_norm_ at 9970d92; its actual positive epsilon strip
keeps the denominator nonzero at every finite reference point. -/
theorem gain_native_clip_factor_tendsto (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) :
    Tendsto (fun n => coordinateClipFactor bound (rawGainNativeGradient remaining genGain memGain (state n)))
      atTop (nhds (coordinateClipFactor bound (rawGainNativeGradient remaining genGain memGain reference))) := by
  have hn := gain_native_gradient_norm_tendsto remaining genGain memGain state reference hp
  have hd : coordinateGradientNorm (rawGainNativeGradient remaining genGain memGain reference) + 1 / 1000000 ≠ 0 := by
    have hs := Real.sqrt_nonneg (∑ i, rawGainNativeGradient remaining genGain memGain reference i ^ 2)
    unfold coordinateGradientNorm
    linarith
  have hb : Tendsto (fun _ : ℕ => bound) atTop (nhds bound) := tendsto_const_nhds
  have hunit : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1) := tendsto_const_nhds
  simpa only [coordinateClipFactor, Pi.div_def] using hunit.min (hb.div (hn.add_const (1 / 1000000)) hd)

example : ∀ i : Fin 4, Tendsto (fun n : ℕ =>
    ({ parameter := (i.val : ℝ) + 1, moment := 0, variance := 0, clock := n } : ScalarState).parameter)
    atTop (nhds ((i.val : ℝ) + 1)) := by
  intro i
  exact tendsto_const_nhds

/-- The actual native input limit is derived from parameter feedback.
Sources: appendix C true CE and clipping at 9970d92; neither future
input convergence nor instantaneous moment/input equality is assumed. -/
theorem gain_native_applied_gradient_tendsto (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState) (i : Fin 4)
    (hp : ∀ j, Tendsto (fun n => (state n j).parameter) atTop (nhds (reference j).parameter)) :
    Tendsto (fun n => appliedGainNativeGradient remaining genGain memGain bound (state n) i) atTop
      (nhds (appliedGainNativeGradient remaining genGain memGain bound reference i)) := by
  have hc := gain_native_clip_factor_tendsto remaining genGain memGain bound state reference hp
  have hg := gain_native_raw_gradient_tendsto remaining genGain memGain state reference i hp
  simpa only [appliedGainNativeGradient, coordinateClippedGradient] using hc.mul hg

example : ∀ i : Fin 4, Tendsto (fun n : ℕ =>
    ({ parameter := (i.val : ℝ) + 1, moment := 0, variance := 0, clock := n } : ScalarState).parameter)
    atTop (nhds ((i.val : ℝ) + 1)) := by
  intro i
  exact tendsto_const_nhds

/-- Actual multiclass train CE converges to the actual finite-point
CE, which is not replaced by zero confidence loss. Source: appendix C
train-loss formula with both physical circuit outputs. -/
theorem gain_native_train_ce_tendsto (remaining : ℕ) (genGain memGain : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) :
    Tendsto (fun n => tableTrainCE remaining (gainNativeTotalScore genGain memGain (state n))) atTop
      (nhds (tableTrainCE remaining (gainNativeTotalScore genGain memGain reference))) := by
  have hs := gain_native_total_score_tendsto genGain memGain state reference hp
  have hd := (hs.rexp.add_const (remaining : ℝ)).add_const 1
  have hn : Real.exp (gainNativeTotalScore genGain memGain reference) + (remaining : ℝ) + 1 ≠ 0 := by positivity
  simpa only [table_train_ce_formula] using (hd.log hn).sub hs

example : ∀ i : Fin 4, Tendsto (fun n : ℕ =>
    ({ parameter := (i.val : ℝ) + 1, moment := 0, variance := 0, clock := n } : ScalarState).parameter)
    atTop (nhds ((i.val : ℝ) + 1)) := by
  intro i
  exact tendsto_const_nhds

/-- Actual held-out CE retains both competing gained outputs and
every remaining class at its finite-point limit. Source: appendix C
test-loss formula; a successful margin is not a convergence premise. -/
theorem gain_native_heldout_ce_tendsto (remaining : ℕ) (genGain memGain : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) :
    Tendsto (fun n => Transformer.Grokking.NaiveLoss.crossEntropy (heldoutTableLogits remaining
      (physicalCircuitScore genGain (state n 0).parameter (state n 1).parameter)
      (physicalCircuitScore memGain (state n 2).parameter (state n 3).parameter)) 0) atTop
      (nhds (Transformer.Grokking.NaiveLoss.crossEntropy (heldoutTableLogits remaining
        (physicalCircuitScore genGain (reference 0).parameter (reference 1).parameter)
        (physicalCircuitScore memGain (reference 2).parameter (reference 3).parameter)) 0)) := by
  have hg := ((hp 0).mul (hp 1)).const_mul genGain
  have hm := ((hp 2).mul (hp 3)).const_mul memGain
  have hd := (hg.rexp.add hm.rexp).add_const (remaining : ℝ)
  have hn : Real.exp (physicalCircuitScore genGain (reference 0).parameter (reference 1).parameter) +
      Real.exp (physicalCircuitScore memGain (reference 2).parameter (reference 3).parameter) + (remaining : ℝ) ≠ 0 := by positivity
  simpa only [table_heldout_ce_formula, physicalCircuitScore] using (hd.log hn).sub hg

example : ∀ i : Fin 4, Tendsto (fun n : ℕ =>
    ({ parameter := (i.val : ℝ) + 1, moment := 0, variance := 0, clock := n } : ScalarState).parameter)
    atTop (nhds ((i.val : ℝ) + 1)) := by
  intro i
  exact tendsto_const_nhds

end Transformer.Grokking.CircuitEfficiency

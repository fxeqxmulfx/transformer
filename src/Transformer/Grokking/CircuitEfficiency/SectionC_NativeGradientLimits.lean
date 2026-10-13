import Transformer.Grokking.CircuitEfficiency.SectionC_NativeSigns
import Mathlib.Topology.Order.OrderClosed

/-!
# Actual fixed-table CE callback limits from parameter limits

Sources: Varma et al., arXiv:2309.02390v1, appendix C, product logits,
actual train/test CE; native norm-two clipping at lab commit 306cabf.
Derive limits of actual derivatives, their shared norm, clipping factor
and applied coordinates from convergence of the four current parameters.
The parameter reference encodes only a finite parameter point; its
buffers and clock are not declared limits of the retained optimizer.

This closes the CE feedback premise for the native finite-limit balance.
No successful future gradient, moment/input equality or finite clock
limit is assumed. Plain CE and parameter-wise native decay remain
distinct from appendix C's GD/coupled circuit-norm penalty. Fixed tables
do not model learned GPTMini features or stochastic minibatch variation;
floating-point forward/autograd/clipping bridges are still absent.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW
open scoped BigOperators

/-- Current products converge to the actual training-score reference.
Source: appendix C, sim-overall-logits; all four parameters contribute. -/
theorem native_total_score_tendsto (state : ℕ → NativeSubweightState) (reference : NativeSubweightState)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) :
    Tendsto (fun n => nativeTotalScore (state n)) atTop (nhds (nativeTotalScore reference)) := by
  have hg := (hp 0).mul (hp 1)
  have hm := (hp 2).mul (hp 3)
  simpa only [nativeTotalScore] using hg.add hm

example : ∀ i : Fin 4, Tendsto (fun n : ℕ =>
    ({ parameter := (i.val : ℝ) + 1, moment := 0, variance := 0, clock := n } : ScalarState).parameter)
    atTop (nhds ((i.val : ℝ) + 1)) := by
  intro i
  exact tendsto_const_nhds

/-- Each actual current CE partial converges to the actual partial at
the finite parameter reference. Source: appendix C's product chain
rule and true multiclass CE; no gradient convergence is supplied. -/
theorem native_raw_gradient_tendsto (remaining : ℕ) (state : ℕ → NativeSubweightState)
    (reference : NativeSubweightState) (i : Fin 4)
    (hp : ∀ j, Tendsto (fun n => (state n j).parameter) atTop (nhds (reference j).parameter)) :
    Tendsto (fun n => rawNativeSubweightGradient remaining (state n) i) atTop
      (nhds (rawNativeSubweightGradient remaining reference i)) := by
  have hs := native_total_score_tendsto state reference hp
  have hd := (hs.rexp.add_const (remaining : ℝ)).add_const 1
  have hn : Real.exp (nativeTotalScore reference) + (remaining : ℝ) + 1 ≠ 0 := by positivity
  have hc : Tendsto (fun _ : ℕ => -((remaining : ℝ) + 1)) atTop
      (nhds (-((remaining : ℝ) + 1))) := tendsto_const_nhds
  have hm := hc.div hd hn
  have ht := (hp (nativeFactorPartner i)).mul hm
  simpa only [Pi.div_def, native_raw_gradient_partner] using ht

example : ∀ i : Fin 4, Tendsto (fun n : ℕ =>
    ({ parameter := (i.val : ℝ) + 1, moment := 0, variance := 0, clock := n } : ScalarState).parameter)
    atTop (nhds ((i.val : ℝ) + 1)) := by
  intro i
  exact tendsto_const_nhds

/-- The shared norm of all four actual partials has the corresponding
finite-point limit. Source: native norm_type=2 at 306cabf, including
the four-coordinate sum rather than independent coordinate clipping. -/
theorem native_gradient_norm_tendsto (remaining : ℕ) (state : ℕ → NativeSubweightState)
    (reference : NativeSubweightState)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) :
    Tendsto (fun n => coordinateGradientNorm (rawNativeSubweightGradient remaining (state n)))
      atTop (nhds (coordinateGradientNorm (rawNativeSubweightGradient remaining reference))) := by
  have ht := tendsto_finsetSum Finset.univ
    (fun i _ => (native_raw_gradient_tendsto remaining state reference i hp).pow 2)
  simpa only [coordinateGradientNorm] using ht.sqrt

example : ∀ i : Fin 4, Tendsto (fun n : ℕ =>
    ({ parameter := (i.val : ℝ) + 1, moment := 0, variance := 0, clock := n } : ScalarState).parameter)
    atTop (nhds ((i.val : ℝ) + 1)) := by
  intro i
  exact tendsto_const_nhds

/-- The actual shared clipping coefficient converges, including its
source epsilon strip. Source: clip_grad_norm_ at 306cabf; the positive
norm-plus-1e-6 denominator does not require a nonzero raw gradient. -/
theorem native_clip_factor_tendsto (remaining : ℕ) (bound : ℝ) (state : ℕ → NativeSubweightState)
    (reference : NativeSubweightState)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) :
    Tendsto (fun n => coordinateClipFactor bound (rawNativeSubweightGradient remaining (state n)))
      atTop (nhds (coordinateClipFactor bound (rawNativeSubweightGradient remaining reference))) := by
  have hn := native_gradient_norm_tendsto remaining state reference hp
  have hd : coordinateGradientNorm (rawNativeSubweightGradient remaining reference) + 1 / 1000000 ≠ 0 := by
    have hs := Real.sqrt_nonneg (∑ i, rawNativeSubweightGradient remaining reference i ^ 2)
    unfold coordinateGradientNorm
    linarith
  have hb : Tendsto (fun _ : ℕ => bound) atTop (nhds bound) := tendsto_const_nhds
  have hunit : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1) := tendsto_const_nhds
  have ht := hb.div (hn.add_const (1 / 1000000)) hd
  simpa only [coordinateClipFactor, Pi.div_def] using hunit.min ht

example : ∀ i : Fin 4, Tendsto (fun n : ℕ =>
    ({ parameter := (i.val : ℝ) + 1, moment := 0, variance := 0, clock := n } : ScalarState).parameter)
    atTop (nhds ((i.val : ℝ) + 1)) := by
  intro i
  exact tendsto_const_nhds

/-- Each actual applied clipped derivative converges to the actual
clipped derivative at the finite parameter point. Sources: appendix C
CE and clip_grad_norm_ at 306cabf; this derives the native input limit. -/
theorem native_applied_gradient_tendsto (remaining : ℕ) (bound : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState) (i : Fin 4)
    (hp : ∀ j, Tendsto (fun n => (state n j).parameter) atTop (nhds (reference j).parameter)) :
    Tendsto (fun n => appliedNativeSubweightGradient remaining bound (state n) i) atTop
      (nhds (appliedNativeSubweightGradient remaining bound reference i)) := by
  have hc := native_clip_factor_tendsto remaining bound state reference hp
  have hg := native_raw_gradient_tendsto remaining state reference i hp
  simpa only [appliedNativeSubweightGradient, coordinateClippedGradient] using hc.mul hg

example : ∀ i : Fin 4, Tendsto (fun n : ℕ =>
    ({ parameter := (i.val : ℝ) + 1, moment := 0, variance := 0, clock := n } : ScalarState).parameter)
    atTop (nhds ((i.val : ℝ) + 1)) := by
  intro i
  exact tendsto_const_nhds

/-- Actual multiclass train CE has its actual finite-parameter limit.
Source: appendix C train-loss formula; a stable finite score is not
silently replaced by perfect confidence or a zero training loss. -/
theorem native_train_ce_tendsto (remaining : ℕ) (state : ℕ → NativeSubweightState)
    (reference : NativeSubweightState)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) :
    Tendsto (fun n => tableTrainCE remaining (nativeTotalScore (state n))) atTop
      (nhds (tableTrainCE remaining (nativeTotalScore reference))) := by
  have hs := native_total_score_tendsto state reference hp
  have hd := (hs.rexp.add_const (remaining : ℝ)).add_const 1
  have hn : Real.exp (nativeTotalScore reference) + (remaining : ℝ) + 1 ≠ 0 := by positivity
  simpa only [table_train_ce_formula] using (hd.log hn).sub hs

example : ∀ i : Fin 4, Tendsto (fun n : ℕ =>
    ({ parameter := (i.val : ℝ) + 1, moment := 0, variance := 0, clock := n } : ScalarState).parameter)
    atTop (nhds ((i.val : ℝ) + 1)) := by
  intro i
  exact tendsto_const_nhds

/-- Actual multiclass held-out CE converges with the current products.
Source: appendix C test-loss formula; both competing weights and the
remaining classes stay in the denominator, without a margin premise. -/
theorem native_heldout_ce_tendsto (remaining : ℕ) (state : ℕ → NativeSubweightState)
    (reference : NativeSubweightState)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) :
    Tendsto (fun n => Transformer.Grokking.NaiveLoss.crossEntropy (heldoutTableLogits remaining
      ((state n 0).parameter * (state n 1).parameter) ((state n 2).parameter * (state n 3).parameter)) 0)
      atTop (nhds (Transformer.Grokking.NaiveLoss.crossEntropy (heldoutTableLogits remaining
        ((reference 0).parameter * (reference 1).parameter) ((reference 2).parameter * (reference 3).parameter)) 0)) := by
  have hg := (hp 0).mul (hp 1)
  have hm := (hp 2).mul (hp 3)
  have hd := (hg.rexp.add hm.rexp).add_const (remaining : ℝ)
  have hn : Real.exp ((reference 0).parameter * (reference 1).parameter) +
      Real.exp ((reference 2).parameter * (reference 3).parameter) + (remaining : ℝ) ≠ 0 := by positivity
  simpa only [table_heldout_ce_formula] using (hd.log hn).sub hg

example : ∀ i : Fin 4, Tendsto (fun n : ℕ =>
    ({ parameter := (i.val : ℝ) + 1, moment := 0, variance := 0, clock := n } : ScalarState).parameter)
    atTop (nhds ((i.val : ℝ) + 1)) := by
  intro i
  exact tendsto_const_nhds

end Transformer.Grokking.CircuitEfficiency

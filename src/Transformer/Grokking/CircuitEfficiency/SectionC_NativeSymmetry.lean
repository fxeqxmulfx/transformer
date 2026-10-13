import Transformer.Grokking.CircuitEfficiency.SectionC_NativePositive

/-!
# A retained native symmetry obstruction to automatic rule selection

Source: Varma et al., arXiv:2309.02390v1, appendix C, equal-speed
initialization and Gen/Mem train/test tables; native AdamW at 793b191.
The two tables are indistinguishable on training data. Under plain CE,
equal complete native states receive equal raw derivatives, equal shared
clipping and equal retained-buffer updates, at every finite clock.

This is a counterexample to transferring the source's cost-asymmetric
coupled-penalty selection mechanism to uniform parameter-wise native
decay. It does not refute the paper's original GD/coupled-penalty model.
Both circuits can be formed and training decisions correct, while their
held-out CE is at least log two forever. No tie-breaking error or learned
GPTMini feature model is inferred from the absence of strict test margin.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Equality of parameters, both retained buffers and clock for each
Gen/Mem factor pair. Source: equal-speed initialization in appendix C;
unlike equality of current logits, this controls future native updates. -/
def NativeCircuitSymmetry (state : NativeSubweightState) : Prop :=
  state 0 = state 2 ∧ state 1 = state 3

/-- Equal factor seeds initialize equal actual zero-buffer states.
Source: appendix C equal-speed initialization and native AdamW. -/
theorem native_equal_seed_symmetry (first second : ℝ) :
    NativeCircuitSymmetry (seededNativeSubweights ((first, second), (first, second))) := by
  exact ⟨rfl, rfl⟩

/-- Actual CE derivatives of corresponding factors agree in the
symmetry region. Source: appendix C training tables; the common slope
comes from the actual multiclass CE, not an imposed optimizer target. -/
theorem native_symmetric_raw_gradient (remaining : ℕ) (state : NativeSubweightState)
    (hs : NativeCircuitSymmetry state) :
    rawNativeSubweightGradient remaining state 0 = rawNativeSubweightGradient remaining state 2 ∧
      rawNativeSubweightGradient remaining state 1 = rawNativeSubweightGradient remaining state 3 := by
  constructor
  · rw [native_raw_gradient_partner, native_raw_gradient_partner]
    change (state 1).parameter * _ = (state 3).parameter * _
    rw [hs.2]
  · rw [native_raw_gradient_partner, native_raw_gradient_partner]
    change (state 0).parameter * _ = (state 2).parameter * _
    rw [hs.1]

example : NativeCircuitSymmetry (seededNativeSubweights ((0, 1), (0, 1))) := by
  exact native_equal_seed_symmetry 0 1

/-- Shared norm-two clipping preserves corresponding derivative
equalities. Source: clip_grad_norm_ at 793b191; no independently chosen
per-coordinate clip or optimizer is substituted. -/
theorem native_symmetric_applied_gradient (remaining : ℕ) (bound : ℝ)
    (state : NativeSubweightState) (hs : NativeCircuitSymmetry state) :
    appliedNativeSubweightGradient remaining bound state 0 =
        appliedNativeSubweightGradient remaining bound state 2 ∧
      appliedNativeSubweightGradient remaining bound state 1 =
        appliedNativeSubweightGradient remaining bound state 3 := by
  have hg := native_symmetric_raw_gradient remaining state hs
  constructor
  · change coordinateClipFactor bound _ * _ = coordinateClipFactor bound _ * _
    rw [hg.1]
  · change coordinateClipFactor bound _ * _ = coordinateClipFactor bound _ * _
    rw [hg.2]

example : NativeCircuitSymmetry (seededNativeSubweights ((0, 1), (0, 1))) := by
  exact native_equal_seed_symmetry 0 1

/-- Retained native parameter/moment/clock updates preserve the full
symmetry. Source: native AdamW at 793b191; equality does not rely on
clearing buffers, negligible epsilon, or a continuous-time limit. -/
theorem native_symmetric_step (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) (hs : NativeCircuitSymmetry state) :
    NativeCircuitSymmetry (nativeSubweightStep remaining bound b1 b2 eps decay rate state) := by
  have hg := native_symmetric_applied_gradient remaining bound state hs
  constructor
  · change scalarNativeStep b1 b2 eps decay rate (state 0) _ =
      scalarNativeStep b1 b2 eps decay rate (state 2) _
    rw [hs.1, hg.1]
  · change scalarNativeStep b1 b2 eps decay rate (state 1) _ =
      scalarNativeStep b1 b2 eps decay rate (state 3) _
    rw [hs.2, hg.2]

example : NativeCircuitSymmetry (seededNativeSubweights ((0, 1), (0, 1))) := by
  exact native_equal_seed_symmetry 0 1

/-- Equal initialization stays equal at every finite native step.
Sources: appendix C equal-speed seeds and the actual CE/native loop at
793b191. The output-loss symmetry is not broken by retained moments. -/
theorem native_equal_seed_path_symmetry (remaining : ℕ) (bound b1 b2 eps decay rate first second : ℝ)
    (n : ℕ) : NativeCircuitSymmetry (nativeSubweightPath remaining bound b1 b2 eps decay rate
      (seededNativeSubweights ((first, second), (first, second))) n) := by
  induction n with
  | zero => exact native_equal_seed_symmetry first second
  | succ n ih => exact native_symmetric_step remaining bound b1 b2 eps decay rate _ ih

/-- Equal complete factor states imply equal actual product logits.
Source: appendix C, sim-overall-logits; this direction is insufficient
to replace full-state equality in the retained update invariant. -/
theorem native_symmetric_products (state : NativeSubweightState) (hs : NativeCircuitSymmetry state) :
    (state 0).parameter * (state 1).parameter = (state 2).parameter * (state 3).parameter := by
  rw [hs.1, hs.2]

example : NativeCircuitSymmetry (seededNativeSubweights ((0, 1), (0, 1))) := by
  exact native_equal_seed_symmetry 0 1

/-- Equal competing held-out weights have true multiclass CE at
least log two. Source: appendix C test-loss formula; the remaining
classes can only increase the loss, regardless of product sign. -/
theorem table_equal_weights_heldout_ce_floor (remaining : ℕ) (weight : ℝ) :
    Real.log 2 ≤ Transformer.Grokking.NaiveLoss.crossEntropy
      (heldoutTableLogits remaining weight weight) 0 := by
  have hm : 2 * Real.exp weight ≤ Real.exp weight + Real.exp weight + (remaining : ℝ) := by
    have hn : (0 : ℝ) ≤ (remaining : ℝ) := Nat.cast_nonneg remaining
    linarith
  have hl := Real.log_le_log (mul_pos (by norm_num : (0 : ℝ) < 2) (Real.exp_pos weight)) hm
  rw [Real.log_mul (by norm_num) (Real.exp_ne_zero weight), Real.log_exp] at hl
  rw [table_heldout_ce_formula]
  linarith

/-- Full native symmetry prevents the true held-out CE from falling
below log two. Sources: appendix C's held-out tables and the checked
native invariant; this is stronger than an ambiguous argmax tie claim. -/
theorem native_symmetric_heldout_ce_floor (remaining : ℕ) (state : NativeSubweightState)
    (hs : NativeCircuitSymmetry state) :
    Real.log 2 ≤ Transformer.Grokking.NaiveLoss.crossEntropy (heldoutTableLogits remaining
      ((state 0).parameter * (state 1).parameter)
      ((state 2).parameter * (state 3).parameter)) 0 := by
  rw [native_symmetric_products state hs]
  exact table_equal_weights_heldout_ce_floor remaining _

example : NativeCircuitSymmetry (seededNativeSubweights ((0, 1), (0, 1))) := by
  exact native_equal_seed_symmetry 0 1

/-- The closed native algorithm never achieves held-out CE below
log two from equal seeds, for any finite number of updates. Sources:
appendix C tables and native AdamW at 793b191 with the stated plain-CE
deviation. Positivity/formation alone cannot imply eventual grokking. -/
theorem native_equal_seed_path_heldout_ce_floor (remaining : ℕ)
    (bound b1 b2 eps decay rate first second : ℝ) (n : ℕ) :
    let state := nativeSubweightPath remaining bound b1 b2 eps decay rate
      (seededNativeSubweights ((first, second), (first, second))) n
    Real.log 2 ≤ Transformer.Grokking.NaiveLoss.crossEntropy (heldoutTableLogits remaining
      ((state 0).parameter * (state 1).parameter)
      ((state 2).parameter * (state 3).parameter)) 0 := by
  exact native_symmetric_heldout_ce_floor remaining _
    (native_equal_seed_path_symmetry remaining bound b1 b2 eps decay rate first second n)

end Transformer.Grokking.CircuitEfficiency

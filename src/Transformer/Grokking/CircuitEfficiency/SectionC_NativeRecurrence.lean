import Transformer.Grokking.CircuitEfficiency.SectionC_AdaptiveFirstStep
import Transformer.Grokking.AdamW.ScalarStability
import Transformer.Grokking.AdamW.GradientClipping
import Mathlib.Data.Fin.VecNotation

/-!
# Closed native AdamW iteration of actual fixed-table product CE

Sources: Varma et al., arXiv:2309.02390v1, appendix C, sim-overall-
logits and train CE; PyTorch 2.14.1 native AdamW and norm-two clipping
at lab commit 793b191. Compute all four actual CE partials at the old
parameters, clip their shared vector, then update all coordinates
simultaneously with retained moments and completed-update clocks.

Deviation from appendix C: its coupled circuit-norm penalty and GD
are replaced by plain CE, parameter-wise decoupled decay and native
AdamW. The source tables remain fixed; this is not learned GPTMini
feature discovery. Exact-real clipping preserves the source's 1e-6;
floating-point kernels, minibatch implementation and rounding bridges
are not supplied. No successful margin is encoded in the iteration.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- States of the Gen first/second and Mem first/second coordinates,
in that order. Source: appendix C's four subweights and native AdamW. -/
abbrev NativeSubweightState := Fin 4 → ScalarState

/-- Extract the current four product factors, retaining their source
pair order. Source: arXiv:2309.02390v1, appendix C's logits. -/
def nativeSubweightParameters (state : NativeSubweightState) : (ℝ × ℝ) × (ℝ × ℝ) :=
  (((state 0).parameter, (state 1).parameter), ((state 2).parameter, (state 3).parameter))

/-- Initialize each source factor with the actual native zero buffers
and clocks. Source: PyTorch 2.14.1 AdamW at 793b191. -/
def seededNativeSubweights (parameters : (ℝ × ℝ) × (ℝ × ℝ)) : NativeSubweightState :=
  ![seededScalarState parameters.1.1, seededScalarState parameters.1.2,
    seededScalarState parameters.2.1, seededScalarState parameters.2.2]

/-- Raw CE gradient candidate, proved equal to the actual derivatives
below. Source: appendix C's CE and the checked product chain rule;
zero coupled multiplier leaves plain CE, independently of exponent two. -/
noncomputable def rawNativeSubweightGradient (remaining : ℕ) (state : NativeSubweightState) : Fin 4 → ℝ :=
  let p := nativeSubweightParameters state
  let g := subweightGradient remaining 0 0 0 2 p.1.1 p.1.2 p.2.1 p.2.2
  ![g.1.1, g.1.2, g.2.1, g.2.2]

/-- Gradient actually inserted into each retained native buffer.
Source: norm-two clip_grad_norm_ at 793b191, one shared coefficient. -/
noncomputable def appliedNativeSubweightGradient (remaining : ℕ) (bound : ℝ)
    (state : NativeSubweightState) : Fin 4 → ℝ :=
  coordinateClippedGradient bound (rawNativeSubweightGradient remaining state)

/-- The parameter/gradient/moment loop is closed at every step.
Source: actual appendix C product CE, native AdamW and clipping at
793b191; the four updates read the same old parameter state. -/
noncomputable def nativeSubweightStep (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) : NativeSubweightState :=
  fun i => scalarNativeStep b1 b2 eps decay rate (state i)
    (appliedNativeSubweightGradient remaining bound state i)

/-- Actual repeated native updates without resetting moments or
supplying a prescribed gradient stream. Source: appendix C product
CE with the stated native-algorithm deviation at 793b191. -/
noncomputable def nativeSubweightPath (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) : ℕ → NativeSubweightState
  | 0 => initial
  | n + 1 => nativeSubweightStep remaining bound b1 b2 eps decay rate
      (nativeSubweightPath remaining bound b1 b2 eps decay rate initial n)

/-- The zero-coupled-penalty objective is actual finite-class CE on
the source's product logits. Source: appendix C, train-loss formula;
the artificial coupled penalty is not applied as native weight decay. -/
theorem plain_subweight_loss_is_ce (remaining : ℕ) (a b c d : ℝ) :
    subweightLoss remaining 0 0 0 2 a b c d = tableTrainCE remaining (a * b + c * d) := by
  simp [subweightLoss, tableBudgetLoss, powerBudgetLoss]

/-- Every raw coordinate is the corresponding actual CE partial,
not a successful-learning assumption. Source: appendix C, product
logits, derived by the checked four-variable chain rule. -/
theorem native_raw_gradient_derivatives (remaining : ℕ) (state : NativeSubweightState) :
    let p := nativeSubweightParameters state
    rawNativeSubweightGradient remaining state =
      ![deriv (fun t => subweightLoss remaining 0 0 0 2 t p.1.2 p.2.1 p.2.2) p.1.1,
        deriv (fun t => subweightLoss remaining 0 0 0 2 p.1.1 t p.2.1 p.2.2) p.1.2,
        deriv (fun t => subweightLoss remaining 0 0 0 2 p.1.1 p.1.2 t p.2.2) p.2.1,
        deriv (fun t => subweightLoss remaining 0 0 0 2 p.1.1 p.1.2 p.2.1 t) p.2.2] := by
  dsimp only
  have hg := subweight_gradient_eq_derivatives remaining 0 0 0 2
    (nativeSubweightParameters state).1.1 (nativeSubweightParameters state).1.2
    (nativeSubweightParameters state).2.1 (nativeSubweightParameters state).2.2 (by norm_num)
  exact congrArg (fun g : (ℝ × ℝ) × (ℝ × ℝ) => ![g.1.1, g.1.2, g.2.1, g.2.2]) hg

/-- Applied gradients satisfy the bound generated by the actual
clipping wrapper at every visited parameter state. Source: native
clip_grad_norm_ at 793b191; no raw-gradient bound is assumed. -/
theorem native_applied_gradient_abs_bound (remaining : ℕ) (bound : ℝ)
    (state : NativeSubweightState) (i : Fin 4) (hb : 0 < bound) :
    |appliedNativeSubweightGradient remaining bound state i| ≤ bound := by
  exact coordinate_clipped_abs_bound bound (rawNativeSubweightGradient remaining state) i hb

example : (0 : ℝ) < 1 := by norm_num

/-- Each coordinate retains its actual completed-update count along
the closed CE trajectory. Source: native AdamW at 793b191, all four
parameters have a supplied derivative on every iteration. -/
theorem native_path_clocks (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (n : ℕ) (i : Fin 4) :
    (nativeSubweightPath remaining bound b1 b2 eps decay rate initial n i).clock =
      (initial i).clock + n := by
  induction n with
  | zero => simp [nativeSubweightPath]
  | succ n ih =>
    change (scalarNativeStep b1 b2 eps decay rate _ _).clock = (initial i).clock + (n + 1)
    rw [scalar_native_clock, ih]
    omega

/-- Source q=113 and seeds 0.005/1 have raw norm below 0.999, so
the native unit bound does not clip their first CE gradient. Sources:
appendix C's simulation table and clip_grad_norm_ at 793b191. -/
theorem source_initial_native_clipping_inactive :
    coordinateClipFactor 1 (rawNativeSubweightGradient 111
      (seededNativeSubweights ((0, 1 / 200), (0, 1)))) = 1 := by
  have hn : coordinateGradientNorm (rawNativeSubweightGradient 111
      (seededNativeSubweights ((0, 1 / 200), (0, 1)))) ≤ 999 / 1000 := by
    unfold coordinateGradientNorm
    apply Real.sqrt_le_iff.mpr
    constructor
    · norm_num
    · norm_num [rawNativeSubweightGradient, nativeSubweightParameters, seededNativeSubweights,
        seededScalarState, subweightGradient, circuitMarginal, Fin.sum_univ_succ]
  apply coordinate_clip_exact_threshold _ _ |>.mpr
  linarith

/-- At an unclipped initialization, the closed native iteration
agrees with the earlier actual first-step factor result. Sources:
appendix C's product CE and native AdamW/clipping at 793b191;
the nontrivial source seeds satisfy the clipping premise above. -/
theorem native_seeded_first_parameters (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ)
    (parameters : (ℝ × ℝ) × (ℝ × ℝ))
    (hc : coordinateClipFactor bound (rawNativeSubweightGradient remaining
      (seededNativeSubweights parameters)) = 1) :
    nativeSubweightParameters (nativeSubweightStep remaining bound b1 b2 eps decay rate
      (seededNativeSubweights parameters)) =
      subweightAdamWFirstStep remaining b1 b2 eps decay rate parameters := by
  have hg : appliedNativeSubweightGradient remaining bound (seededNativeSubweights parameters) =
      rawNativeSubweightGradient remaining (seededNativeSubweights parameters) := by
    funext i
    unfold appliedNativeSubweightGradient coordinateClippedGradient
    rw [hc, one_mul]
  unfold nativeSubweightParameters nativeSubweightStep
  simp only [hg]
  simp [rawNativeSubweightGradient, nativeSubweightParameters, seededNativeSubweights,
    subweightAdamWFirstStep, scalar_native_first_parameter]
  simp [seededScalarState]

example : coordinateClipFactor 1 (rawNativeSubweightGradient 111
    (seededNativeSubweights ((0, 1 / 200), (0, 1)))) = 1 := by
  exact source_initial_native_clipping_inactive

end Transformer.Grokking.CircuitEfficiency

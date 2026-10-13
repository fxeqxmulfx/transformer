import Transformer.Grokking.CircuitEfficiency.SectionC_GainGradient
import Transformer.Grokking.AdamW.ScalarLimits

/-!+# Closed native feedback for physical forward efficiency

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C;
PyTorch 2.14.1 AdamW/clipping as ported at lab commit c268c1f. Clip
all actual gained-forward CE derivatives with one shared norm-two
coefficient, then advance every retained scalar state simultaneously.
All four coordinates have the same decay, learning rate and epsilon.
Both moment histories and completed clocks are retained; the raw
callback is the true CE derivative proved in GainGradient.

Derive its shared positive CE magnitude, finite-point signs and bounds,
parameter-only callback dependence, actual clock law and a fully cold
trajectory. Unit gains recover the previous native update. These are
exact-real fixed-table laws, not a supplied gradient stream, unequal
decay groups or the source's coupled circuit-norm objective. A cold
pair with stale momentum is not declared invariant. Positive interior
allocation, attraction, delay and learned stochastic GPTMini transfer
remain separate obligations.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Actual native input after clipping the complete true CE vector.
Sources: appendix C product CE and clip_grad_norm_ at c268c1f. -/
noncomputable def appliedGainNativeGradient (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : NativeSubweightState) : Fin 4 → ℝ :=
  coordinateClippedGradient bound (rawGainNativeGradient remaining genGain memGain state)

/-- Shared magnitude of the actual clipped multiclass CE slope.
Sources: appendix C train CE and native clipping at c268c1f; it uses
the current gained score and all four raw coordinate derivatives. -/
noncomputable def gainCEGradientScale (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : NativeSubweightState) : ℝ :=
  coordinateClipFactor bound (rawGainNativeGradient remaining genGain memGain state) *
    (((remaining : ℝ) + 1) / (Real.exp (gainNativeTotalScore genGain memGain state) + (remaining : ℝ) + 1))

/-- Simultaneous actual native update with one uniform decay.
Sources: true appendix C CE and retained AdamW at c268c1f; fixed
gains change the forward, not the scalar optimizer configuration. -/
noncomputable def gainNativeStep (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) : NativeSubweightState :=
  fun i => scalarNativeStep b1 b2 eps decay rate (state i)
    (appliedGainNativeGradient remaining genGain memGain bound state i)

/-- Actual repeated CE feedback with retained buffers and clocks.
Sources: appendix C product logits and native AdamW at c268c1f. -/
noncomputable def gainNativePath (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) : ℕ → NativeSubweightState
  | 0 => initial
  | n + 1 => gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n)

/-- The common applied CE magnitude never vanishes at a finite
point with positive cap. Sources: appendix C true CE and shared native
clipping at c268c1f; no parameter sign or nonzero raw norm is needed. -/
theorem gain_ce_gradient_scale_pos (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : NativeSubweightState) (hclip : 0 < bound) :
    0 < gainCEGradientScale remaining genGain memGain bound state := by
  unfold gainCEGradientScale
  exact mul_pos (coordinate_clip_factor_pos bound _ hclip) (div_pos (by positivity) (by positivity))

example : (0 : ℝ) < 1 := by norm_num

/-- All applied partials use the same actual magnitude, their own
fixed gain and current partner. Sources: appendix C chain rule and
native shared clipping at c268c1f; this is not assigned feedback. -/
theorem gain_native_applied_gradient_scale (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : NativeSubweightState) (i : Fin 4) :
    appliedGainNativeGradient remaining genGain memGain bound state i =
      -(gainCEGradientScale remaining genGain memGain bound state *
        nativeFactorGain genGain memGain i * (state (nativeFactorPartner i)).parameter) := by
  change coordinateClipFactor bound (rawGainNativeGradient remaining genGain memGain state) *
    (nativeFactorGain genGain memGain i * (state (nativeFactorPartner i)).parameter *
      (-((remaining : ℝ) + 1) / (Real.exp (gainNativeTotalScore genGain memGain state) + (remaining : ℝ) + 1))) = _
  unfold gainCEGradientScale
  ring

/-- Positive physical gain and partner give a strictly negative
actual applied partial. Sources: appendix C true CE and native clip at
c268c1f; positive clipping cannot manufacture a flat finite CE slope. -/
theorem gain_native_applied_gradient_negative (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : NativeSubweightState) (i : Fin 4) (hclip : 0 < bound)
    (hg : 0 < nativeFactorGain genGain memGain i)
    (hp : 0 < (state (nativeFactorPartner i)).parameter) :
    appliedGainNativeGradient remaining genGain memGain bound state i < 0 := by
  rw [gain_native_applied_gradient_scale]
  exact neg_neg_of_pos (mul_pos (mul_pos (gain_ce_gradient_scale_pos remaining _ _ _ state hclip) hg) hp)

example : (0 : ℝ) < 1 ∧ 0 < nativeFactorGain 3 2 0 ∧
    0 < (seededNativeSubweights ((1, 1), (1, 1)) (nativeFactorPartner 0)).parameter := by
  norm_num [nativeFactorGain, nativeFactorPartner, seededNativeSubweights, seededScalarState]

/-- The same wrapper enforces its coordinate bound on every actual
feedback input. Source: native norm-two clipping at c268c1f. -/
theorem gain_native_applied_gradient_abs_bound (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : NativeSubweightState) (i : Fin 4) (hclip : 0 < bound) :
    |appliedGainNativeGradient remaining genGain memGain bound state i| ≤ bound := by
  exact coordinate_clipped_abs_bound bound (rawGainNativeGradient remaining genGain memGain state) i hclip

example : (0 : ℝ) < 1 := by norm_num

/-- Current equal parameters give equal actual CE inputs, regardless
of stale buffers or clocks. Sources: appendix C forward and native
clipping at c268c1f; this does not equate the resulting optimizer steps. -/
theorem gain_native_equal_parameters_applied_gradient (remaining : ℕ) (genGain memGain bound : ℝ)
    (state reference : NativeSubweightState) (hp : ∀ i, (state i).parameter = (reference i).parameter) :
    appliedGainNativeGradient remaining genGain memGain bound state =
      appliedGainNativeGradient remaining genGain memGain bound reference := by
  have hs : gainNativeTotalScore genGain memGain state = gainNativeTotalScore genGain memGain reference := by
    unfold gainNativeTotalScore
    rw [hp 0, hp 1, hp 2, hp 3]
  have hr : rawGainNativeGradient remaining genGain memGain state = rawGainNativeGradient remaining genGain memGain reference := by
    funext i
    unfold rawGainNativeGradient
    rw [hp, hs]
  unfold appliedGainNativeGradient
  rw [hr]

example : ∀ i : Fin 4,
    ({ parameter := 1, moment := -1, variance := 1, clock := 37 } : ScalarState).parameter =
      (seededNativeSubweights ((1, 1), (1, 1)) i).parameter := by
  intro i
  fin_cases i <;> rfl

/-- Unit gains reproduce the earlier retained native CE step.
Sources: appendix C's unweighted physical readouts and native AdamW
at c268c1f; equality includes both buffers and the completed clock. -/
theorem gain_native_unit_step (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ) (state : NativeSubweightState) :
    gainNativeStep remaining 1 1 bound b1 b2 eps decay rate state =
      nativeSubweightStep remaining bound b1 b2 eps decay rate state := by
  unfold gainNativeStep nativeSubweightStep appliedGainNativeGradient appliedNativeSubweightGradient
  rw [gain_native_unit_raw_gradient]

/-- All actual coordinate clocks retain their starting count and
advance at every CE update. Source: native AdamW at c268c1f. -/
theorem gain_native_path_clocks (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (n : ℕ) (i : Fin 4) :
    (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).clock =
      (initial i).clock + n := by
  induction n with
  | zero => simp [gainNativePath]
  | succ n ih =>
    change (scalarNativeStep b1 b2 eps decay rate _ _).clock = _
    rw [scalar_native_clock, ih]
    omega

/-- Fully cold actual gained-CE inputs vanish. Sources: appendix C
product derivatives and native clipping at c268c1f; both partners
are zero and no retained momentum is assumed to be present. -/
theorem gain_native_cold_applied_gradient (remaining : ℕ) (genGain memGain bound : ℝ) (clock : ℕ) (i : Fin 4) :
    appliedGainNativeGradient remaining genGain memGain bound (fun _ => zeroScalarStateAt clock) i = 0 := by
  rw [gain_native_applied_gradient_scale]
  change -(_ * _ * (0 : ℝ)) = 0
  rw [mul_zero, neg_zero]

/-- Actual cold feedback advances zero buffers and growing clocks.
Sources: appendix C product CE and retained native AdamW at c268c1f. -/
theorem gain_native_cold_step (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ) (clock : ℕ) :
    gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate (fun _ => zeroScalarStateAt clock) =
      (fun _ => zeroScalarStateAt (clock + 1)) := by
  funext i
  change scalarNativeStep b1 b2 eps decay rate (zeroScalarStateAt clock) _ = _
  rw [gain_native_cold_applied_gradient]
  exact scalar_native_zero_step b1 b2 eps decay rate clock

/-- The closed gained-feedback path has a genuinely cold trace.
Sources: appendix C and native AdamW at c268c1f; it witnesses finite
parameter limits without assuming finite limiting optimizer clocks. -/
theorem gain_native_cold_path (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ) (n : ℕ) :
    gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (fun _ => zeroScalarStateAt 0) n =
      (fun _ => zeroScalarStateAt n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate _ = _
    rw [ih]
    exact gain_native_cold_step remaining genGain memGain bound b1 b2 eps decay rate n

end Transformer.Grokking.CircuitEfficiency

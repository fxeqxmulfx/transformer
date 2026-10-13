import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryCollapse

/-!
# Actual retained buffers, denominator and cold feedback under strong decay

Sources: Varma et al., arXiv:2309.02390v1, appendix C actual product
CE and section 3 competition; retained native AdamW/clipping at lab
commit 1880c50, including initialized legal-beta mass contraction.

The generated applied inputs tend to zero. Their actual causal retained
moment histories therefore make both buffers tend to zero, from any
nonnegative retained initialization. The complete corrected denominator
approaches the original positive epsilon. Completed clocks diverge;
no finite optimizer-state reference or clock reset is substituted.

The generated physical parameter limits also give the actual shared
CE/clipping coefficient's cold limit, including every finite training
class and the norm-plus-1e-6 clipping strip. Reference parameters are
zero only for evaluating this continuous numerical feedback; reference
buffers/clocks are not claimed to be the optimizer-state limit.

The shared coefficient's cold limit is strictly positive. Generated
coordinate inputs vanish through the physical partner factors even
though this finite-class CE score multiplier does not vanish.

All limits follow from static strong decay and legal native betas;
no input or parameter convergence is independently supplied. These
results prepare an actual asymptotic recurrence with retained memory.
They do not imply relative Gen selection, a positive margin, learned
circuit formation or stochastic/numerical GPTMini convergence. Fixed
physical tables and uniform decoupled native decay differ from appendix
C's coupled assigned norm/GD; no floating-point transfer is asserted.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Both original retained buffers tend to zero at every coordinate
under initialized legal-beta strong decay. Sources: appendix C true
inputs and native histories at 1880c50; input convergence is generated
by the actual path rather than assumed or replaced by a reset. -/
theorem gain_native_memory_buffers_tendsto_zero (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial)
    (hstrong : genGain * (2 - b1) / ((1 - b1) * eps) < decay) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    ∀ i, Tendsto (fun n => (path n i).moment) atTop (nhds 0) ∧
      Tendsto (fun n => (path n i).variance) atTop (nhds 0) := by
  have hg := gain_native_memory_inputs_tendsto_zero remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hd hs hstrong
  let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
  dsimp only
  intro i
  let state := fun n => path n i
  let gradient := fun n => appliedGainNativeGradient remaining genGain memGain bound (path n) i
  have hstep : ∀ n, state (n + 1) = scalarNativeStep b1 b2 eps decay rate (state n) (gradient n) := fun _ => rfl
  have hm := moment_tendsto_of_input_tendsto b1 (state 0).moment 0 gradient hb1 h1 (hg i)
  have hvariance := moment_tendsto_of_input_tendsto b2 (state 0).variance (0 ^ 2) (fun n => gradient n ^ 2)
    hb2 h2 ((hg i).pow 2)
  constructor
  · exact Tendsto.congr (fun n => (scalar_path_fields b1 b2 eps decay rate state gradient n hstep).1.symm) hm
  · have hv := Tendsto.congr (fun n => (scalar_path_fields b1 b2 eps decay rate state gradient n hstep).2.1.symm) hvariance
    simpa only [zero_pow (by norm_num : (2 : ℕ) ≠ 0)] using hv

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧
    (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 100 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    (3 : ℝ) * (2 - 9 / 10) / ((1 - 9 / 10) * 1) < 100 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num⟩

/-- The original complete next-input denominator tends to epsilon,
while completed native clocks diverge. Sources: native AdamW corrections
at 1880c50 and appendix C generated inputs; no finite-state reference,
instantaneous variance match or future convergence input is supplied. -/
theorem gain_native_memory_denominator_and_clock_limits (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hstrong : genGain * (2 - b1) / ((1 - b1) * eps) < decay) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    ∀ i, Tendsto (fun n => nextBufferDenominator b1 b2 eps (path n i).variance
      (appliedGainNativeGradient remaining genGain memGain bound (path n) i) (path n i).clock) atTop (nhds eps) ∧
      Tendsto (fun n => (path n i).clock + 1) atTop atTop := by
  have hg := gain_native_memory_inputs_tendsto_zero remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hd hs hstrong
  let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
  dsimp only
  intro i
  let state := fun n => path n i
  let gradient := fun n => appliedGainNativeGradient remaining genGain memGain bound (path n) i
  have hstep : ∀ n, state (n + 1) = scalarNativeStep b1 b2 eps decay rate (state n) (gradient n) := fun _ => rfl
  have hden := scalar_next_denominator_tendsto b1 b2 eps decay rate 0 state gradient hstep hb1 h1 hb2 h2 (hg i)
  refine ⟨?_, scalar_completed_clock_tendsto b1 b2 eps decay rate state gradient hstep⟩
  simpa only [abs_zero, zero_add] using hden

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧
    (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 100 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    (3 : ℝ) * (2 - 9 / 10) / ((1 - 9 / 10) * 1) < 100 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num⟩

/-- The original shared CE/clipping scale has its computed cold limit
on the initialized legal-beta strong-decay path. Sources: appendix C
full class count and native clipping strip at 1880c50; the zero numerical
reference follows from generated physical limits, not a future premise. -/
theorem gain_native_memory_ce_scale_tendsto_cold (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hstrong : genGain * (2 - b1) / ((1 - b1) * eps) < decay) :
    0 < coldGainCEGradientScale remaining bound ∧
    Tendsto (fun n => gainCEGradientScale remaining genGain memGain bound
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n))
      atTop (nhds (coldGainCEGradientScale remaining bound)) := by
  have hp := gain_native_memory_parameters_tendsto_zero remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hd hs hstrong
  let reference := seededNativeSubweights ((0, 0), (0, 0))
  have hr : ∀ i, (reference i).parameter = 0 := by
    intro i
    fin_cases i <;> rfl
  have hpr : ∀ i, Tendsto (fun n =>
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).parameter)
      atTop (nhds (reference i).parameter) := by
    intro i
    rw [hr i]
    exact hp i
  have hscale := gain_ce_gradient_scale_tendsto remaining genGain memGain bound _ reference hpr
  refine ⟨cold_gain_ce_gradient_scale_pos remaining bound hclip, ?_⟩
  simpa only [gain_ce_gradient_scale_zero_parameters remaining genGain memGain bound reference hr] using hscale

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧
    (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 100 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    (3 : ℝ) * (2 - 9 / 10) / ((1 - 9 / 10) * 1) < 100 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num⟩

end Transformer.Grokking.CircuitEfficiency

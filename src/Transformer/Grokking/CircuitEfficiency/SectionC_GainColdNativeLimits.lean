import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdCollapse

/-!
# Complete retained native limits at the sharp strict cold threshold

Sources: Varma et al., arXiv:2309.02390v1, section 3 competition and
appendix C product CE; retained native limit port at 1880c50 and sharp
initialized parameter/input collapse at 1692df8. Replace that port's
conservative first-clock decay hypothesis by the proved strict cold gap.

Every actual clipped input now tends to zero by initialized feedback.
The native causal first and second buffers therefore tend to zero from
arbitrary nonnegative retained initial data. Their actual complete
corrected denominators tend to positive epsilon, and completed clocks
diverge. No finite whole-optimizer-state reference or reset is used.

Generated parameter limits give the true shared CE/clipping coefficient
its attained cold limit, including every finite class and the native
norm-plus-1e-6 strip. That scalar limit is strictly positive even though
every applied coordinate input vanishes through its partner parameter.

All limits follow from legal initial native data and one strict static
cold threshold. Convergence of parameters, inputs, buffers or denominators
is not an independent premise. The all-zero physical reference only
evaluates the numerical continuous callback; it is not a limiting clock.

These limits prepare absolute output/confidence laws. Relative Gen/Mem
selection, critical equality, stable weak-decay behavior and learned
stochastic/numerical GPTMini transfer remain separate. Fixed gained
tables/plain CE/uniform native decay differ from assigned coupled norm
cost and GD in appendix C. No thermodynamic system-size scaling follows,
and no optimizer, buffer or experiment checkpoint is changed.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Both actual retained native buffers vanish above the sharp strict
cold threshold. Sources: appendix C applied inputs, native histories
at 1880c50 and their initialized input limit at 1692df8; no supplied
buffer convergence, instantaneous matching or reset is used. -/
theorem gain_native_strict_cold_buffers_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hkeep : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial)
    (hgap : coldGainCEGradientScale remaining bound * genGain < decay * eps) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    ∀ i, Tendsto (fun n => (path n i).moment) atTop (nhds 0) ∧
      Tendsto (fun n => (path n i).variance) atTop (nhds 0) := by
  have hg := gain_native_strict_cold_inputs_tendsto remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep hs hgap
  let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
  dsimp only
  intro i
  let state := fun n => path n i
  let gradient := fun n => appliedGainNativeGradient remaining genGain memGain bound (path n) i
  have hstep : ∀ n, state (n + 1) = scalarNativeStep b1 b2 eps decay rate (state n) (gradient n) := fun _ => rfl
  have hm := moment_tendsto_of_input_tendsto b1 (state 0).moment 0 gradient hb1 h1 (hg i)
  have hv := moment_tendsto_of_input_tendsto b2 (state 0).variance (0 ^ 2) (fun n => gradient n ^ 2)
    hb2 h2 ((hg i).pow 2)
  constructor
  · exact Tendsto.congr (fun n => (scalar_path_fields b1 b2 eps decay rate state gradient n hstep).1.symm) hm
  · have ht := Tendsto.congr (fun n => (scalar_path_fields b1 b2 eps decay rate state gradient n hstep).2.1.symm) hv
    simpa only [zero_pow (by norm_num : (2 : ℕ) ≠ 0)] using ht

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * 2 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 < (2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

/-- Every complete actual denominator approaches epsilon while its
completed native clock diverges. Sources: native corrections at
1880c50 and initialized input limits at 1692df8; both retained buffer
insertions remain in the denominator and no clock reset is performed. -/
theorem gain_native_strict_cold_denominator_clock_limits (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hkeep : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial)
    (hgap : coldGainCEGradientScale remaining bound * genGain < decay * eps) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    ∀ i, Tendsto (fun n => nextBufferDenominator b1 b2 eps (path n i).variance
      (appliedGainNativeGradient remaining genGain memGain bound (path n) i) (path n i).clock) atTop (nhds eps) ∧
      Tendsto (fun n => (path n i).clock + 1) atTop atTop := by
  have hg := gain_native_strict_cold_inputs_tendsto remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep hs hgap
  let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
  dsimp only
  intro i
  let state := fun n => path n i
  let gradient := fun n => appliedGainNativeGradient remaining genGain memGain bound (path n) i
  have hstep : ∀ n, state (n + 1) = scalarNativeStep b1 b2 eps decay rate (state n) (gradient n) := fun _ => rfl
  have hden := scalar_next_denominator_tendsto b1 b2 eps decay rate 0 state gradient hstep hb1 h1 hb2 h2 (hg i)
  refine ⟨?_, scalar_completed_clock_tendsto b1 b2 eps decay rate state gradient hstep⟩
  simpa only [abs_zero, zero_add] using hden

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * 2 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 < (2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

/-- The original shared CE/clipping coefficient approaches its positive
attained cold value above the strict threshold. Sources: appendix C
finite-class slope, native strip at ca253e4 and parameter limits at
1692df8; applied inputs vanish through partners rather than this coefficient. -/
theorem gain_native_strict_cold_ce_scale_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hkeep : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial)
    (hgap : coldGainCEGradientScale remaining bound * genGain < decay * eps) :
    0 < coldGainCEGradientScale remaining bound ∧
      Tendsto (fun n => gainCEGradientScale remaining genGain memGain bound
        (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n))
        atTop (nhds (coldGainCEGradientScale remaining bound)) := by
  have hp := gain_native_strict_cold_parameters_tendsto remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep hs hgap
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

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * 2 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 < (2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

end Transformer.Grokking.CircuitEfficiency


import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdSeededCollapse

/-!
# Complete initialized native limits including critical equality

Sources: Varma et al., arXiv:2309.02390v1, section 3 CE/decay
competition and appendix C product partials; actual initialized
critical collapse at cb3a0d6 and native buffer port at 1880c50.

The actual physical mass collapse at or above the cold threshold
forces every actual clipped CE input to vanish. Both retained
causal buffers then vanish, their completed-clock denominator tends
to epsilon, their adaptive directions vanish, and completed clocks
diverge. Arbitrary legal betas,
including the original 0.9/0.98 pair, keep their native histories.

All these limits follow from standard nonnegative zero-buffer
initialization, legal hyperparameters and the static threshold.
No future physical parameter, input, first/second moment or
denominator limit is assumed, and no correction clock is reset.
The actual feedback generates the input stream used by the scalar
native port. An unbounded clock is not a finite optimizer attractor.

Positive rate and epsilon, positive physical gains, a positive clip
cap and nonnegative remaining decay are explicit. The shared CE
coefficient need not vanish when every partner input vanishes.
Output/confidence limits and relative Gen/Mem decisions are separate.

Exact-real fixed gained tables/plain CE/uniform decoupled AdamW
differ from appendix C's assigned coupled norm-cost GD. Learned
stochastic/numerical GPTMini and thermodynamic size scaling remain
unproved. Frozen model, optimizer, runs and checkpoints are unchanged.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Every actual clipped CE input vanishes at or above the cold
threshold, including equality with retained first beta. Sources:
appendix C partner partials, true input ceiling at 696630b and
initialized mass collapse at cb3a0d6; no input limit is a premise. -/
theorem gain_native_seeded_cold_inputs_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    ∀ i, Tendsto (fun n => appliedGainNativeGradient remaining genGain memGain bound
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
        (seededNativeSubweights ((a, b), (c, d))) n) i) atTop (nhds 0) := by
  have hm := gain_native_seeded_cold_mass_tendsto remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep ha hb hc hd hthreshold
  let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights ((a, b), (c, d)))
  have hsign := fun n => gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate
    (seededNativeSubweights ((a, b), (c, d))) n hgen hmem hclip hb1 h1 hb2 h2 he (le_of_lt heta) hkeep
      (native_seeded_nonnegative a b c d ha hb hc hd)
  have hupper : Tendsto (fun n => (coldGainCEGradientScale remaining bound * genGain) * gainNativeParameterMass (path n))
      atTop (nhds 0) := by simpa only [mul_zero] using hm.const_mul (coldGainCEGradientScale remaining bound * genGain)
  intro i
  have hcap := fun n => gain_native_cold_applied_gradient_mass_cap remaining genGain memGain bound (path n) i
    hgen hmem hgain hclip (hsign n)
  have ht := squeeze_zero (fun n => (hcap n).1) (fun n => (hcap n).2) hupper
  simpa only [neg_neg, neg_zero] using ht.neg

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  norm_num [coldGainCEGradientScale]

/-- Both actual retained buffers vanish, including critical equality.
Sources: appendix C's generated applied inputs, critical collapse at
cb3a0d6 and native causal limits at 1880c50; neither instantaneous
buffer matching nor any future moment convergence is supplied. -/
theorem gain_native_seeded_cold_buffers_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights ((a, b), (c, d)))
    ∀ i, Tendsto (fun n => (path n i).moment) atTop (nhds 0) ∧
      Tendsto (fun n => (path n i).variance) atTop (nhds 0) := by
  have hg := gain_native_seeded_cold_inputs_tendsto remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep ha hb hc hd hthreshold
  let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights ((a, b), (c, d)))
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
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  norm_num [coldGainCEGradientScale]

/-- Actual completed-clock denominators tend to epsilon, while their
completed clocks diverge, including critical equality. Sources:
native correction port at 1880c50 and initialized collapse at
cb3a0d6; no finite whole-state reference or clock reset is used. -/
theorem gain_native_seeded_cold_denominator_clock_limits (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights ((a, b), (c, d)))
    ∀ i, Tendsto (fun n => nextBufferDenominator b1 b2 eps (path n i).variance
      (appliedGainNativeGradient remaining genGain memGain bound (path n) i) (path n i).clock) atTop (nhds eps) ∧
      Tendsto (fun n => (path n i).clock + 1) atTop atTop := by
  have hg := gain_native_seeded_cold_inputs_tendsto remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep ha hb hc hd hthreshold
  let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights ((a, b), (c, d)))
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
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  norm_num [coldGainCEGradientScale]

/-- The actual corrected adaptive directions vanish without resetting
either buffer, also at critical equality. Sources: native direction
port at 1880c50, appendix C applied CE and initialized collapse at
cb3a0d6; the input limit is generated by the same feedback path. -/
theorem gain_native_seeded_cold_directions_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights ((a, b), (c, d)))
    ∀ i, Tendsto (fun n => nextBufferDirection b1 b2 eps (path n i).moment (path n i).variance
      (appliedGainNativeGradient remaining genGain memGain bound (path n) i) (path n i).clock) atTop (nhds 0) := by
  have hg := gain_native_seeded_cold_inputs_tendsto remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep ha hb hc hd hthreshold
  let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights ((a, b), (c, d)))
  dsimp only
  intro i
  have hstep : ∀ n, path (n + 1) i = scalarNativeStep b1 b2 eps decay rate (path n i)
      (appliedGainNativeGradient remaining genGain memGain bound (path n) i) := fun _ => rfl
  simpa only [zero_div] using scalar_direction_tendsto b1 b2 eps decay rate 0 (fun n => path n i)
    (fun n => appliedGainNativeGradient remaining genGain memGain bound (path n) i) hstep hb1 h1 hb2 h2 he (hg i)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  norm_num [coldGainCEGradientScale]

end Transformer.Grokking.CircuitEfficiency

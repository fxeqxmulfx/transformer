import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdSeededNativeLimits

/-!
# Actual critical weighted mass, CE coefficient and physical scores

Sources: Varma et al., arXiv:2309.02390v1, section 3 CE/decay
competition and appendix C product logits; actual critical physical
collapse at cb3a0d6 and retained native limits at 970ea1f.

At or above the greater-gain cold threshold, initialized total
physical mass and retained negative first-moment mass vanish, so
their weighted native observer has zero limit. Its earlier finite
limit is no longer an undetermined positive value on these paths.

The actual shared clipped CE coefficient approaches its attained
strictly positive cold value, even though every applied input
vanishes through its partner parameter. Complete true CE and the
native norm-plus-1e-6 clipping denominator stay in this coefficient.
Both physical Gen/Mem scores and their held-out margin tend to zero.

All limits include exact critical equality and arbitrary legal
retained betas, with standard nonnegative zero-buffer seeds.
Positive rate/epsilon/gains/cap and nonnegative remaining decay
are static data. No future observer, coefficient, score or parameter
limit is independently assumed. The zero physical reference only
evaluates the continuous callback; it is not a finite clock attractor.

Absolute vanishing scores do not decide eventual relative ordering
or exact classification. Their complete confidence and CE limits
must still be derived from the full multiclass readout. Exact-real
fixed gained tables/plain CE/uniform decoupled AdamW differ from
appendix C's assigned coupled norm-cost GD and learned GPTMini.
Stochastic/numerical and thermodynamic system-size transfer remain
separate. Frozen measurements, optimizers and checkpoints are unchanged.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- The actual CE coefficient tends to its positive cold value,
also at critical equality. Sources: appendix C complete CE, native
clip strip at ca253e4 and initialized collapse at cb3a0d6; applied
inputs vanish through partners rather than the shared coefficient. -/
theorem gain_native_seeded_cold_ce_scale_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    0 < coldGainCEGradientScale remaining bound ∧
      Tendsto (fun n => gainCEGradientScale remaining genGain memGain bound
        (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
          (seededNativeSubweights ((a, b), (c, d))) n)) atTop (nhds (coldGainCEGradientScale remaining bound)) := by
  have hp := gain_native_seeded_cold_parameters_tendsto remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep ha hb hc hd hthreshold
  let reference := seededNativeSubweights ((0, 0), (0, 0))
  have hr : ∀ i, (reference i).parameter = 0 := by
    intro i
    fin_cases i <;> rfl
  have hpr : ∀ i, Tendsto (fun n => (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
      (seededNativeSubweights ((a, b), (c, d))) n i).parameter) atTop (nhds (reference i).parameter) := by
    intro i
    rw [hr i]
    exact hp i
  have hscale := gain_ce_gradient_scale_tendsto remaining genGain memGain bound _ reference hpr
  refine ⟨cold_gain_ce_gradient_scale_pos remaining bound hclip, ?_⟩
  simpa only [gain_ce_gradient_scale_zero_parameters remaining genGain memGain bound reference hr] using hscale

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  norm_num [coldGainCEGradientScale]

/-- Total retained negative first-moment mass vanishes, including
critical equality. Sources: appendix C's four coordinates and native
buffer limits at 970ea1f; the current applied inputs do not replace
the retained moments in this numerical observable. -/
theorem gain_native_seeded_cold_moment_mass_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    Tendsto (fun n => gainNativeNegativeMomentMass (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
      (seededNativeSubweights ((a, b), (c, d))) n)) atTop (nhds 0) := by
  have hbuf := gain_native_seeded_cold_buffers_tendsto remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep ha hb hc hd hthreshold
  simpa only [gainNativeNegativeMomentMass, zero_add, neg_zero] using
    (((hbuf 0).1.add (hbuf 1).1).add ((hbuf 2).1.add (hbuf 3).1)).neg

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  norm_num [coldGainCEGradientScale]

/-- The generated retained weighted observer itself tends to zero,
including critical equality. Sources: appendix C factors, observer
at 4748aba and initialized native buffers at 970ea1f; the finite
observer limit is determined from the actual physical trajectory. -/
theorem gain_native_seeded_cold_weighted_mass_tendsto_zero (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    Tendsto (fun n => coldGainNativeWeightedMass b1 eps rate (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
      (seededNativeSubweights ((a, b), (c, d))) n)) atTop (nhds 0) := by
  have hp := gain_native_seeded_cold_mass_tendsto remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep ha hb hc hd hthreshold
  have hu := gain_native_seeded_cold_moment_mass_tendsto remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep ha hb hc hd hthreshold
  simpa only [coldGainNativeWeightedMass, mul_zero, zero_add] using
    (hp.const_mul ((1 - b1) * eps)).add (hu.const_mul (rate * b1))

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  norm_num [coldGainCEGradientScale]

/-- Both actual circuit scores and their held-out margin vanish,
also at exact critical equality. Sources: appendix C product logits
and generated physical collapse at cb3a0d6; no future score limit,
held-out decision or relative Gen dominance is a premise. -/
theorem gain_native_seeded_cold_scores_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights ((a, b), (c, d)))
    Tendsto (fun n => physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter) atTop (nhds 0) ∧
      Tendsto (fun n => physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter) atTop (nhds 0) ∧
      Tendsto (fun n => physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter -
        physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter) atTop (nhds 0) := by
  have hp := gain_native_seeded_cold_parameters_tendsto remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep ha hb hc hd hthreshold
  let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights ((a, b), (c, d)))
  have hg : Tendsto (fun n => physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter) atTop (nhds 0) := by
    simpa only [physicalCircuitScore, mul_zero] using ((hp 0).mul (hp 1)).const_mul genGain
  have hm : Tendsto (fun n => physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter) atTop (nhds 0) := by
    simpa only [physicalCircuitScore, mul_zero] using ((hp 2).mul (hp 3)).const_mul memGain
  exact ⟨hg, hm, by simpa only [sub_zero] using hg.sub hm⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  norm_num [coldGainCEGradientScale]

end Transformer.Grokking.CircuitEfficiency

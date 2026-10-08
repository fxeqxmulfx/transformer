import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdSeededCollapse
import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdInstability

/-!
# The initialized native cold threshold includes equality

Sources: Varma et al., arXiv:2309.02390v1, section 3 CE/decay
competition and appendix C product feedback; original weak-decay
noncollapse at 6a99a48 and initialized critical collapse at cb3a0d6.

Standard nonnegative zero-buffer seeds with one positive Gen partner
give an exact classification for the original retained native path:
total physical mass tends to zero if and only if cold coefficient
times greater Gen gain is at most decay times epsilon. Critical
equality now belongs to the collapse side. Every physical coordinate
has zero limit under exactly the same threshold.

Both betas retain their native histories. Positive decay/rate/epsilon,
strictly positive remaining decay, greater Gen gain and a positive
Gen partner are explicit. No future parameter/input/buffer limit or
successful classifier is supplied. Without the positive Gen seed,
the absent-Gen invariant prevents this necessary threshold argument.

For the binary gains 3/2, cap=1, epsilon=1, rate=0.001 and original
betas 0.9/0.98, the exact boundary is decay=1.5. Epsilon=1 is a
mathematical control and differs from the frozen transformer runs.
The finite class count and native clipping strip remain in the
general threshold; no fitted spectral exponent replaces either.

If rate times the cold Gen coefficient is at least epsilon, no
positive remaining-decay choice can reach the collapse side. This
static obstruction follows from the exact boundary and keeps the
small-epsilon native regime distinct from the epsilon=1 control.

This is an absolute-mass classification, not stable relative Gen/Mem
selection or a transition-time law. Exact-real fixed gained tables,
plain CE and uniform decoupled AdamW differ from appendix C's
assigned coupled norm-cost GD and learned stochastic/numerical GPTMini.
No thermodynamic system-size conclusion follows. Frozen model,
optimizer, measurements, runs and checkpoints are unchanged.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Initialized total physical mass collapses exactly at or above
the cold threshold, including equality. Sources: appendix C's true
CE, original retained noncollapse at 6a99a48 and generated critical
collapse at cb3a0d6; all premises are static initialization data. -/
theorem gain_native_seeded_cold_threshold (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate) (hkeep : 0 < 1 - rate * decay)
    (ha : 0 ≤ a) (hb : 0 < b) (hc : 0 ≤ c) (hd : 0 ≤ d) :
    Tendsto (fun n => gainNativeParameterMass (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
      (seededNativeSubweights ((a, b), (c, d))) n)) atTop (nhds 0) ↔
        coldGainCEGradientScale remaining bound * genGain ≤ decay * eps := by
  have hs := native_seeded_nonnegative a b c d ha (le_of_lt hb) hc hd
  have hi : 0 < (seededNativeSubweights ((a, b), (c, d)) 1).parameter := hb
  constructor
  · intro hm
    by_contra h
    exact gain_native_cold_regime_not_mass_collapse remaining genGain memGain bound b1 b2 eps decay rate
      (seededNativeSubweights ((a, b), (c, d))) hmem hgain hclip hb1 h1 hb2 h2 he hdecay heta hkeep hs hi
        (lt_of_not_ge h) hm
  · intro hthreshold
    exact gain_native_seeded_cold_mass_tendsto remaining genGain memGain bound b1 b2 eps decay rate a b c d
      (lt_trans hmem hgain) hmem (le_of_lt hgain) hclip hb1 h1 hb2 h2 he heta (le_of_lt hkeep)
        ha (le_of_lt hb) hc hd hthreshold

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 3 / 2 ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : ℝ) < 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by
  norm_num

/-- All four initialized physical coordinates have zero limits
exactly at or above the threshold. Sources: appendix C coordinates,
generated critical collapse at cb3a0d6 and original noncollapse at
6a99a48; no parameter convergence premise remains in the statement. -/
theorem gain_native_seeded_cold_parameter_threshold (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate) (hkeep : 0 < 1 - rate * decay)
    (ha : 0 ≤ a) (hb : 0 < b) (hc : 0 ≤ c) (hd : 0 ≤ d) :
    (∀ i, Tendsto (fun n => (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
      (seededNativeSubweights ((a, b), (c, d))) n i).parameter) atTop (nhds 0)) ↔
        coldGainCEGradientScale remaining bound * genGain ≤ decay * eps := by
  constructor
  · intro hp
    apply (gain_native_seeded_cold_threshold remaining genGain memGain bound b1 b2 eps decay rate a b c d
      hmem hgain hclip hb1 h1 hb2 h2 he hdecay heta hkeep ha hb hc hd).mp
    have hm := ((hp 0).add (hp 1)).add ((hp 2).add (hp 3))
    simpa only [gainNativeParameterMass, zero_add] using hm
  · intro hthreshold
    exact gain_native_seeded_cold_parameters_tendsto remaining genGain memGain bound b1 b2 eps decay rate a b c d
      (lt_trans hmem hgain) hmem (le_of_lt hgain) hclip hb1 h1 hb2 h2 he heta (le_of_lt hkeep)
        ha (le_of_lt hb) hc hd hthreshold

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 3 / 2 ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : ℝ) < 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by
  norm_num

/-- Binary source-style factors with original retained betas and
epsilon=1 have the exact decay boundary 1.5. Sources: section 3
initialization, appendix C CE and the complete generated threshold
above; epsilon differs from the frozen transformer experiments. -/
theorem gain_native_binary_original_betas_cold_threshold (seed decay : ℝ)
    (hseed : 0 < seed) (hdecay : 0 < decay) (hkeep : decay < 1000) :
    Tendsto (fun n => gainNativeParameterMass (gainNativePath 0 3 2 1 (9 / 10) (49 / 50) 1 decay (1 / 1000)
      (seededNativeSubweights ((0, seed), (0, 1))) n)) atTop (nhds 0) ↔ (3 / 2 : ℝ) ≤ decay := by
  have h := gain_native_seeded_cold_threshold 0 3 2 1 (9 / 10) (49 / 50) 1 decay (1 / 1000) 0 seed 0 1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) hdecay (by norm_num) (by nlinarith only [hkeep]) (by norm_num) hseed (by norm_num) (by norm_num)
  norm_num [coldGainCEGradientScale] at h
  exact h

example : (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 3 / 2 ∧ (3 / 2 : ℝ) < 1000 := by
  norm_num

/-- A rate/epsilon obstruction excludes total collapse for every
positive remaining-decay choice. Sources: appendix C feedback,
initialized native threshold above and actual decoupled decay;
no future convergence or fixed numerical experiment is a premise. -/
theorem gain_native_seeded_cold_rate_obstruction (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate) (hkeep : 0 < 1 - rate * decay)
    (ha : 0 ≤ a) (hb : 0 < b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hrate : eps ≤ rate * (coldGainCEGradientScale remaining bound * genGain)) :
    ¬Tendsto (fun n => gainNativeParameterMass (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
      (seededNativeSubweights ((a, b), (c, d))) n)) atTop (nhds 0) := by
  intro hm
  have hthreshold := (gain_native_seeded_cold_threshold remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hmem hgain hclip hb1 h1 hb2 h2 he hdecay heta hkeep ha hb hc hd).mp hm
  have hscaled := mul_le_mul_of_nonneg_left hthreshold (le_of_lt heta)
  have hremaining := mul_pos hkeep he
  nlinarith only [hrate, hscaled, hremaining]

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : ℝ) < 1 - (1 / 1000) * (1 / 10) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    (1 / 100000000 : ℝ) ≤ (1 / 1000) * (coldGainCEGradientScale 0 1 * 3) := by
  norm_num [coldGainCEGradientScale]

end Transformer.Grokking.CircuitEfficiency

import Transformer.Grokking.CircuitEfficiency.SectionC_GainMassRates

/-!
# Current unequal-pair efficiency advantage after a quantified error budget

Sources: Varma et al., arXiv:2309.02390v1, section 3's efficiency
competition and appendix C's partner-product dynamics; actual native
same-current-CE proxy law and initialized relative limits at d655756.

For positive Gen/Mem pair masses with Gen no greater, their balanced
proxies have the previously proved relative-rate advantage. The
unequal Gen correction can consume part of that advantage. Once its
explicit error per unit current mass is at most half of the gap,
the actual-form Gen relative mass multiplier still beats Mem's by
at least rate times half the gap. Mem's own unequal correction is
nonnegative and can only help this comparison.

The gap is generated from a positive actual CE floor and finite
physical box, rather than being asserted as a future successful
margin. This file is the current algebraic comparison; the later
initialized path proof must derive its small-error tail from the
proved actual relative-error limit, and iterate the actual update.

The candidate mass-ratio increment is strictly greater than one.
It concerns exact-real fixed gained tables and zero betas with
uniform native decay, differing from appendix C's coupled-cost GD
and the preserved nonzero-beta learned GPTMini. It alone proves
neither a path crossing nor permanent held-out correctness.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- Half-gap ratio growth available after the explicit current Gen
relative error budget. Sources: appendix C gained partials and the
native proxy estimates at d655756; this numerical expression contains
no future trajectory, success event or parameter limit. -/
noncomputable def gainUnequalRatioGrowth (lower eps genGain memGain ceiling decay rate : ℝ) : ℝ :=
  1 + rate * gainBalancedRelativeGap lower eps genGain memGain ceiling /
    (2 * (1 - rate * decay + rate * memGain / eps))

/-- The unequal-pair candidate ratio multiplier is strictly greater
than one under the fixed positive floor/epsilon/efficiency conditions.
Sources: section 3 efficiency and actual native estimates at d655756;
all hypotheses concern constants, without a future balance premise. -/
theorem gain_unequal_ratio_growth_one_lt (lower eps genGain memGain ceiling decay rate : ℝ)
    (hlower : 0 < lower) (he : 0 < eps) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hc : 0 ≤ ceiling) (heta : 0 < rate) (hd : 0 < 1 - rate * decay) :
    1 < gainUnequalRatioGrowth lower eps genGain memGain ceiling decay rate := by
  have hg := lt_trans hmem hgain
  have hgap : 0 < gainBalancedRelativeGap lower eps genGain memGain ceiling := by
    unfold gainBalancedRelativeGap
    positivity
  have hden : 0 < 2 * (1 - rate * decay + rate * memGain / eps) := by positivity
  have hincrement := div_pos (mul_pos heta hgap) hden
  unfold gainUnequalRatioGrowth
  linarith only [hincrement]

example : (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧
    (0 : ℝ) ≤ 2 ∧ (0 : ℝ) < 1 / 1000 ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) := by norm_num

/-- The actual-form unequal Gen relative mass multiplier exceeds
Mem by a quantitative positive half-gap while Gen is no greater.
Sources: appendix C partner inputs and native balancing law at
d655756; the small-error premise is an explicit current numerical
inequality, to be generated from initialized limits on the path. -/
theorem gain_pair_mass_relative_gap (lower scale eps genGain memGain ceiling decay rate a b c d : ℝ)
    (hlower : 0 < lower) (hscale : lower ≤ scale) (hunit : scale ≤ 1) (he : 0 < eps)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (heta : 0 < rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d) (hweak : a + b ≤ c + d)
    (hceiling : c + d ≤ 2 * ceiling)
    (herror : gainPairBalancingError genGain scale eps a b / (a + b) ≤
      gainBalancedRelativeGap lower eps genGain memGain ceiling / 2) :
    0 < gainBalancedRelativeGap lower eps genGain memGain ceiling ∧
      rate * gainBalancedRelativeGap lower eps genGain memGain ceiling / 2 ≤
        ((gainPairStep genGain scale eps decay rate a b).1 + (gainPairStep genGain scale eps decay rate a b).2) / (a + b) -
          ((gainPairStep memGain scale eps decay rate c d).1 + (gainPairStep memGain scale eps decay rate c d).2) / (c + d) := by
  have hs := lt_of_lt_of_le hlower hscale
  have hg := lt_trans hmem hgain
  have hmeanGen : 0 ≤ (a + b) / 2 := by positivity
  have hmeanOrder : (a + b) / 2 ≤ (c + d) / 2 := by linarith only [hweak]
  have hmeanCeiling : (c + d) / 2 ≤ ceiling := by linarith only [hceiling]
  have hgap := gain_balanced_relative_gap_floor lower scale eps genGain memGain ceiling
    ((a + b) / 2) ((c + d) / 2) hlower hscale hunit he hmem hgain hmeanGen hmeanOrder hmeanCeiling
  have hgen := gain_pair_step_relative_mass_multiplier genGain scale eps decay rate a b
    (le_of_lt hg) (le_of_lt hs) he ha hb hgMass
  have hmemEq := gain_pair_step_relative_mass_multiplier memGain scale eps decay rate c d
    (le_of_lt hmem) (le_of_lt hs) he hc hdseed hmMass
  have hmemError := div_nonneg (gain_pair_balancing_error_nonneg memGain scale eps c d
    (le_of_lt hmem) (le_of_lt hs) he hc hdseed) (le_of_lt hmMass)
  have hweightedGap := mul_le_mul_of_nonneg_left hgap.2 (le_of_lt heta)
  have hweightedError := mul_le_mul_of_nonneg_left herror (le_of_lt heta)
  have hweightedMem := mul_nonneg (le_of_lt heta) hmemError
  refine ⟨hgap.1, ?_⟩
  nlinarith only [hgen, hmemEq, hweightedGap, hweightedError, hweightedMem]

example : (0 : ℝ) < 1 / 10 ∧ (1 / 10 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) < 1 / 4 + 1 / 4 ∧ (0 : ℝ) < 1 + 1 ∧ (1 / 4 : ℝ) + 1 / 4 ≤ 1 + 1 ∧
    (1 : ℝ) + 1 ≤ 2 * 2 ∧
    gainPairBalancingError 3 (1 / 2) 1 (1 / 4) (1 / 4) / (1 / 4 + 1 / 4) ≤
      gainBalancedRelativeGap (1 / 10) 1 3 2 2 / 2 := by
  norm_num [gainPairBalancingError, gainBalancedRelativeGap]

/-- A quantitative advantage between current relative mass multipliers
gives multiplicative growth of the mass ratio. Sources: appendix C's
competing products and native half-gap comparison at d655756; this
algebraic transfer takes only current/next numbers. Its actual-path
application must derive both the ceiling and the advantage from CE.
No update recurrence, loss decrease, eventual crossing, parameter
convergence or future successful reference is encoded in a definition.
The upper multiplier is derived positive from the next Mem mass. -/
theorem gain_mass_ratio_from_relative_advantage (genMass memMass nextGenMass nextMemMass upper increment : ℝ)
    (hgen : 0 < genMass) (hmem : 0 < memMass) (hnext : 0 < nextMemMass)
    (hinc : 0 ≤ increment) (hceiling : nextMemMass / memMass ≤ upper)
    (hgap : increment ≤ nextGenMass / genMass - nextMemMass / memMass) :
    (1 + increment / upper) * (genMass / memMass) ≤ nextGenMass / nextMemMass := by
  have hupper := lt_of_lt_of_le (div_pos hnext hmem) hceiling
  have hpart : 0 ≤ increment / upper := div_nonneg hinc (le_of_lt hupper)
  have hweighted : increment / upper * (nextMemMass / memMass) ≤ increment := by
    calc
      _ ≤ increment / upper * upper := mul_le_mul_of_nonneg_left hceiling hpart
      _ = increment := div_mul_cancel₀ increment (ne_of_gt hupper)
  have hmultiplier : (1 + increment / upper) * (nextMemMass / memMass) ≤
      nextGenMass / genMass := by
    nlinarith only [hweighted, hgap]
  have hmassWeighted := mul_le_mul_of_nonneg_right hmultiplier (le_of_lt hgen)
  rw [div_mul_cancel₀ nextGenMass (ne_of_gt hgen)] at hmassWeighted
  apply (le_div_iff₀ hnext).mpr
  calc
    _ = (1 + increment / upper) * (nextMemMass / memMass) * genMass := by ring
    _ ≤ nextGenMass := hmassWeighted

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 3 ∧ (0 : ℝ) ≤ 1 ∧
    (3 : ℝ) / 2 ≤ 2 ∧ (1 : ℝ) ≤ 3 / 1 - 3 / 2 := by norm_num

/-- A positive current relative multiplier advantage strictly increases
the positive Gen/Mem mass ratio. Sources: the native half-gap comparison
at d655756 and appendix C's competing products; positivity is applied
to the present masses, without assuming any positive limiting mass. -/
theorem gain_mass_ratio_strict_from_relative_advantage (genMass memMass nextGenMass nextMemMass upper increment : ℝ)
    (hgen : 0 < genMass) (hmem : 0 < memMass) (hnext : 0 < nextMemMass)
    (hinc : 0 < increment) (hceiling : nextMemMass / memMass ≤ upper)
    (hgap : increment ≤ nextGenMass / genMass - nextMemMass / memMass) :
    genMass / memMass < nextGenMass / nextMemMass := by
  have hupper := lt_of_lt_of_le (div_pos hnext hmem) hceiling
  have hbound := gain_mass_ratio_from_relative_advantage genMass memMass nextGenMass nextMemMass upper increment
    hgen hmem hnext (le_of_lt hinc) hceiling hgap
  have hpositive := mul_pos (div_pos hinc hupper) (div_pos hgen hmem)
  nlinarith only [hbound, hpositive]

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧
    (3 : ℝ) / 2 ≤ 2 ∧ (1 : ℝ) ≤ 3 / 1 - 3 / 2 := by norm_num

end Transformer.Grokking.CircuitEfficiency

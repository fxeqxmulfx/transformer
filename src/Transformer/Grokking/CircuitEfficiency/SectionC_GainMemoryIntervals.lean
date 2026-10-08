import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryWeights

/-!
# Generated numerical intervals for actual retained pair coefficients

Sources: Varma et al., arXiv:2309.02390v1, section 3 efficiency and
appendix C gained factors; original native pair measurements and
initialized coefficient limits at lab commit fa00206.

The current pair coefficients have explicit numerical cold limits,
including the actual finite-class CE and clipping strip. Static strong
native decay and legal betas generate those limits from current initial
sign data. Strict bounds around these computed numbers therefore give
a finite tail of bounds on the original current coefficients.

No future parameter, buffer, denominator or coefficient convergence is
supplied. The interval margins are numerical inequalities between fixed
constants, not a future rule-success or gradient condition. A rational
example witnesses separate Gen bounds with beta1=0.9 and beta2=0.98.
The second moment and completed clocks remain in every actual update.

These laws prepare the full retained Gen/Mem weight comparison. They do
not prove physical pair balance, held-out correctness or confidence.
Fixed tables and uniform native decay differ from appendix C coupled-cost
GD. Learned stochastic/floating-point GPTMini transfer remains open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Computed cold coefficient of negative retained first moment in
a pair measurement. Source: native coefficient limit at fa00206. -/
noncomputable def coldGainWeightedMomentCoefficient (b1 eps rate weight : ℝ) : ℝ :=
  rate * b1 / eps + weight * b1

/-- Computed cold coefficient of a pair parameter, including decay,
current CE partner insertion and the measurement's moment insertion.
Sources: appendix C actual partials and native limits at fa00206. -/
noncomputable def coldGainWeightedParameterCoefficient (remaining : ℕ)
    (gain bound b1 eps decay rate weight : ℝ) : ℝ :=
  (1 - rate * decay) + rate * (1 - b1) * coldGainCEGradientScale remaining bound * gain / eps +
    weight * (1 - b1) * coldGainCEGradientScale remaining bound * gain

/-- Both actual weighted coefficients converge to their computed
cold numbers on the initialized legal-beta strong-decay path. Sources:
appendix C gained feedback and native limits at fa00206; future
parameter/coefficient convergence is generated, not assumed. -/
theorem gain_native_weighted_coefficients_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate weight : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hstrong : genGain * (2 - b1) / ((1 - b1) * eps) < decay) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    ∀ i, Tendsto (fun n => gainNativeWeightedMomentCoefficient remaining genGain memGain bound b1 b2 eps rate weight (path n) i)
      atTop (nhds (coldGainWeightedMomentCoefficient b1 eps rate weight)) ∧
      Tendsto (fun n => gainNativeWeightedParameterCoefficient remaining genGain memGain bound b1 b2 eps decay rate weight (path n) i)
        atTop (nhds (coldGainWeightedParameterCoefficient remaining (nativeFactorGain genGain memGain i) bound b1 eps decay rate weight)) := by
  have hcoeff := gain_native_memory_coefficients_tendsto remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hd hs hstrong
  have hc := (gain_native_memory_ce_scale_tendsto_cold remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hd hs hstrong).2
  dsimp only
  intro i
  have hm := (hcoeff i).1.add_const (weight * b1)
  have hi := (hc.const_mul (weight * (1 - b1))).mul_const (nativeFactorGain genGain memGain i)
  have hp := ((hcoeff i).2.const_add (1 - rate * decay)).add hi
  constructor
  · simpa only [gainNativeWeightedMomentCoefficient, coldGainWeightedMomentCoefficient] using hm
  · simpa only [gainNativeWeightedParameterCoefficient, coldGainWeightedParameterCoefficient] using hp

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧
    (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 100 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    (3 : ℝ) * (2 - 9 / 10) / ((1 - 9 / 10) * 1) < 100 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num⟩

/-- Strict intervals around computed cold coefficients generate a
finite tail of intervals on the original actual coordinate. Sources:
appendix C gained factors and initialized native limits at fa00206;
all four interval hypotheses concern explicit static numerical limits. -/
theorem gain_native_weighted_coefficient_interval_tail (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate weight lower upper : ℝ)
    (initial : NativeSubweightState) (i : Fin 4)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hstrong : genGain * (2 - b1) / ((1 - b1) * eps) < decay)
    (hpl : lower < coldGainWeightedParameterCoefficient remaining (nativeFactorGain genGain memGain i) bound b1 eps decay rate weight)
    (hpu : coldGainWeightedParameterCoefficient remaining (nativeFactorGain genGain memGain i) bound b1 eps decay rate weight < upper)
    (hml : lower * weight < coldGainWeightedMomentCoefficient b1 eps rate weight)
    (hmu : coldGainWeightedMomentCoefficient b1 eps rate weight < upper * weight) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    ∃ start : ℕ, ∀ n, start ≤ n →
      (lower ≤ gainNativeWeightedParameterCoefficient remaining genGain memGain bound b1 b2 eps decay rate weight (path n) i ∧
        gainNativeWeightedParameterCoefficient remaining genGain memGain bound b1 b2 eps decay rate weight (path n) i ≤ upper) ∧
      (lower * weight ≤ gainNativeWeightedMomentCoefficient remaining genGain memGain bound b1 b2 eps rate weight (path n) i ∧
        gainNativeWeightedMomentCoefficient remaining genGain memGain bound b1 b2 eps rate weight (path n) i ≤ upper * weight) := by
  have hl := gain_native_weighted_coefficients_tendsto remaining genGain memGain bound b1 b2 eps decay rate weight initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hd hs hstrong
  have htail := ((hl i).2.eventually_const_lt hpl).and ((hl i).2.eventually_lt_const hpu)
  have hmoment := ((hl i).1.eventually_const_lt hml).and ((hl i).1.eventually_lt_const hmu)
  obtain ⟨start, hstart⟩ := eventually_atTop.mp (htail.and hmoment)
  dsimp only
  refine ⟨start, ?_⟩
  intro n hn
  have ht := hstart n hn
  exact ⟨⟨le_of_lt ht.1.1, le_of_lt ht.1.2⟩, ⟨le_of_lt ht.2.1, le_of_lt ht.2.2⟩⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧
    (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 100 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    (3 : ℝ) * (2 - 9 / 10) / ((1 - 9 / 10) * 1) < 100 ∧
    (911 / 1000 : ℝ) < coldGainWeightedParameterCoefficient 0 (nativeFactorGain 3 2 0) 1 (9 / 10) 1 100 (1 / 1000) (2 / 25) ∧
    coldGainWeightedParameterCoefficient 0 (nativeFactorGain 3 2 0) 1 (9 / 10) 1 100 (1 / 1000) (2 / 25) < 1 ∧
    (911 / 1000 : ℝ) * (2 / 25) < coldGainWeightedMomentCoefficient (9 / 10) 1 (1 / 1000) (2 / 25) ∧
    coldGainWeightedMomentCoefficient (9 / 10) 1 (1 / 1000) (2 / 25) < 1 * (2 / 25) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [coldGainWeightedParameterCoefficient, coldGainWeightedMomentCoefficient, coldGainCEGradientScale, nativeFactorGain]

/-- Explicit distinct positive pair comparison rates with retained
nonzero beta1. Sources: appendix C gained product competition and
native cold limits at fa00206. The Gen lower rate is 0.911 and the
Mem upper rate is 0.910. These numerical gaps certify interval premises;
they do not substitute a constant matrix for the actual update. -/
theorem fixed_gain_memory_separating_rates :
    let genWeight := (2 / 25 : ℝ)
    let memWeight := (19 / 200 : ℝ)
    let genRate := (911 / 1000 : ℝ)
    let memRate := (91 / 100 : ℝ)
    0 < memRate ∧ memRate < genRate ∧ genRate < 1 ∧
    genRate < coldGainWeightedParameterCoefficient 0 3 1 (9 / 10) 1 100 (1 / 1000) genWeight ∧
    coldGainWeightedParameterCoefficient 0 3 1 (9 / 10) 1 100 (1 / 1000) genWeight < 1 ∧
    genRate * genWeight < coldGainWeightedMomentCoefficient (9 / 10) 1 (1 / 1000) genWeight ∧
    coldGainWeightedMomentCoefficient (9 / 10) 1 (1 / 1000) genWeight < 1 * genWeight ∧
    0 < coldGainWeightedParameterCoefficient 0 2 1 (9 / 10) 1 100 (1 / 1000) memWeight ∧
    coldGainWeightedParameterCoefficient 0 2 1 (9 / 10) 1 100 (1 / 1000) memWeight < memRate ∧
    0 * memWeight < coldGainWeightedMomentCoefficient (9 / 10) 1 (1 / 1000) memWeight ∧
    coldGainWeightedMomentCoefficient (9 / 10) 1 (1 / 1000) memWeight < memRate * memWeight := by
  dsimp only
  norm_num [coldGainWeightedParameterCoefficient, coldGainWeightedMomentCoefficient, coldGainCEGradientScale]

end Transformer.Grokking.CircuitEfficiency

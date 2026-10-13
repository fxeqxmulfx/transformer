import Transformer.Grokking.CircuitEfficiency.SectionC_GainPairDifference

/-!
# Positive pair-mass growth relative to unequal-factor contraction

Sources: Varma et al., arXiv:2309.02390v1, section 3's initially
unbalanced product factors and appendix C's true CE; normalized
zero-beta partner inputs at lab commit 620ad1a.

On a finite nonnegative physical box, a positive shared CE scale
floor bounds each normalized partner input below. The pair sum has
an explicit multiplier strictly greater than pure decay. Its gain
monotonicity supplies a common lower multiplier for both mechanisms.
Combine mass growth with checked difference contraction to obtain a
strictly less-than-one multiplier for relative pair asymmetry.

This comparison permits mass itself to shrink: the lower multiplier
need not exceed one. It proves a current actual-form estimate, not
positive limiting mass or successful held-out allocation. The actual
CE floor/physical box must be generated from initialization on the
same native path before iteration. Fixed tables, exact reals, legal
zero betas and uniform native decay differ from coupled-cost GD and
preserved learned GPTMini. No future balance/limit is a hypothesis.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- Uniform pair-mass multiplier from a positive current CE scale
floor and finite physical box. Sources: normalized native partner
inputs at 620ad1a; this numerical constant contains no path assertion. -/
noncomputable def gainPairMassMultiplier (lower gain eps decay rate ceiling : ℝ) : ℝ :=
  1 - rate * decay + rate * lower * (gain / (gain * ceiling + eps))

/-- Shared scale/box bounds force a normalized partner-input floor.
Sources: appendix C partner partials and native normalization at
620ad1a; all hypotheses describe a current numerical point. -/
theorem gain_pair_input_floor (lower scale gain eps ceiling factor : ℝ)
    (hl : 0 < lower) (hs : lower ≤ scale) (hu : scale ≤ 1) (hg : 0 < gain)
    (he : 0 < eps) (hf : 0 ≤ factor) (hc : factor ≤ ceiling) :
    lower * gain * factor / (gain * ceiling + eps) ≤
      scale * gain * factor / (scale * gain * factor + eps) := by
  have hscale : 0 ≤ scale := le_of_lt (lt_of_lt_of_le hl hs)
  have hnum := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hs (le_of_lt hg)) hf
  have hscaled := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hu (le_of_lt hg)) hf
  have hfactor := mul_le_mul_of_nonneg_left hc (le_of_lt hg)
  have hden : scale * gain * factor + eps ≤ gain * ceiling + eps := by
    linarith only [hscaled, hfactor]
  have hd : 0 < scale * gain * factor + eps := by positivity
  exact le_trans (div_le_div_of_nonneg_right hnum (le_of_lt (lt_of_lt_of_le hd hden)))
    (div_le_div_of_nonneg_left (by positivity) hd hden)

example : (0 : ℝ) < 1 / 10 ∧ (1 / 10 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 2 := by
  norm_num

/-- Positive rate/gain/scale give mass amplification beyond pure
decay and a relative-difference ratio below one. Source: the exact
native input floor at 620ad1a; no mass limit is presumed. -/
theorem gain_pair_mass_multiplier_interval (lower gain eps decay rate ceiling : ℝ)
    (hl : 0 < lower) (hg : 0 < gain) (he : 0 < eps) (heta : 0 < rate)
    (hc : 0 ≤ ceiling) (hd : 0 ≤ 1 - rate * decay) :
    0 < gainPairMassMultiplier lower gain eps decay rate ceiling ∧
      1 - rate * decay < gainPairMassMultiplier lower gain eps decay rate ceiling ∧
      0 ≤ (1 - rate * decay) / gainPairMassMultiplier lower gain eps decay rate ceiling ∧
      (1 - rate * decay) / gainPairMassMultiplier lower gain eps decay rate ceiling < 1 := by
  have hpart : 0 < rate * lower * (gain / (gain * ceiling + eps)) := by positivity
  have hpos : 0 < gainPairMassMultiplier lower gain eps decay rate ceiling := by
    unfold gainPairMassMultiplier
    linarith only [hd, hpart]
  have hgap : 1 - rate * decay < gainPairMassMultiplier lower gain eps decay rate ceiling := by
    unfold gainPairMassMultiplier
    linarith only [hpart]
  exact ⟨hpos, hgap, div_nonneg hd (le_of_lt hpos), (div_lt_iff₀ hpos).mpr (by simpa using hgap)⟩

example : (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 2 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) := by
  norm_num

/-- A greater physical gain gives at least the same mass multiplier
on a common physical box. Source: normalized native inputs at
620ad1a; this permits one lower multiplier for unequal Gen/Mem pairs. -/
theorem gain_pair_mass_multiplier_gain_mono (lower genGain memGain eps decay rate ceiling : ℝ)
    (hl : 0 ≤ lower) (hm : 0 ≤ memGain) (hg : memGain ≤ genGain) (he : 0 < eps)
    (heta : 0 ≤ rate) (hc : 0 ≤ ceiling) :
    gainPairMassMultiplier lower memGain eps decay rate ceiling ≤
      gainPairMassMultiplier lower genGain eps decay rate ceiling := by
  have hgen : 0 ≤ genGain := le_trans hm hg
  have hmd : 0 < memGain * ceiling + eps := by positivity
  have hgd : 0 < genGain * ceiling + eps := by positivity
  have hnum := mul_le_mul_of_nonneg_right hg (le_of_lt he)
  have hquot : memGain / (memGain * ceiling + eps) ≤ genGain / (genGain * ceiling + eps) := by
    apply (div_le_div_iff₀ hmd hgd).mpr
    nlinarith only [hnum]
  have hweighted := mul_le_mul_of_nonneg_left hquot (mul_nonneg heta hl)
  unfold gainPairMassMultiplier
  linarith only [hweighted]

example : (0 : ℝ) ≤ 1 / 10 ∧ (0 : ℝ) ≤ 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) ≤ 2 := by
  norm_num

/-- The actual-form unequal pair sum dominates its current mass
times the explicit multiplier. Sources: appendix C factor sums
and native input floors at 620ad1a; no factor equality is assumed. -/
theorem gain_pair_step_mass_floor (lower scale gain eps decay rate ceiling a b : ℝ)
    (hl : 0 < lower) (hs : lower ≤ scale) (hu : scale ≤ 1) (hg : 0 < gain)
    (he : 0 < eps) (heta : 0 ≤ rate) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hac : a ≤ ceiling) (hbc : b ≤ ceiling) :
    gainPairMassMultiplier lower gain eps decay rate ceiling * (a + b) ≤
      (gainPairStep gain scale eps decay rate a b).1 + (gainPairStep gain scale eps decay rate a b).2 := by
  have hleft := mul_le_mul_of_nonneg_left (gain_pair_input_floor lower scale gain eps ceiling b hl hs hu hg he hb hbc) heta
  have hright := mul_le_mul_of_nonneg_left (gain_pair_input_floor lower scale gain eps ceiling a hl hs hu hg he ha hac) heta
  calc
    _ = (1 - rate * decay) * a + rate * (lower * gain * b / (gain * ceiling + eps)) +
        ((1 - rate * decay) * b + rate * (lower * gain * a / (gain * ceiling + eps))) := by
      unfold gainPairMassMultiplier
      ring
    _ ≤ (1 - rate * decay) * a + rate * (scale * gain * b / (scale * gain * b + eps)) +
        ((1 - rate * decay) * b + rate * (scale * gain * a / (scale * gain * a + eps))) := by
      linarith only [hleft, hright]
    _ = _ := by
      unfold gainPairStep
      dsimp only
      ring

example : (0 : ℝ) < 1 / 10 ∧ (1 / 10 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 2 ∧ (1 : ℝ) ≤ 2 := by
  norm_num

/-- Positive pair mass and the checked floor turn absolute contraction
into a strict relative-asymmetry contraction. Sources: appendix C's
unbalanced factors and native estimates at 620ad1a; neither mass
convergence nor later successful allocation appears in the premises. -/
theorem gain_pair_step_relative_difference (lower scale gain eps decay rate ceiling a b : ℝ)
    (hl : 0 < lower) (hs : lower ≤ scale) (hu : scale ≤ 1) (hg : 0 < gain) (he : 0 < eps)
    (heta : 0 < rate) (hd : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hac : a ≤ ceiling) (hbc : b ≤ ceiling) (hmass : 0 < a + b)
    (hsmall : rate * (decay + gain / eps) ≤ 1) :
    let next := gainPairStep gain scale eps decay rate a b
    |next.1 - next.2| / (next.1 + next.2) ≤
      ((1 - rate * decay) / gainPairMassMultiplier lower gain eps decay rate ceiling) * (|a - b| / (a + b)) := by
  have hc : 0 ≤ ceiling := le_trans ha hac
  have hmult := gain_pair_mass_multiplier_interval lower gain eps decay rate ceiling hl hg he heta hc hd
  have hf := gain_pair_step_mass_floor lower scale gain eps decay rate ceiling a b hl hs hu hg he (le_of_lt heta) ha hb hac hbc
  have hdifference := gain_pair_step_difference_contraction gain scale eps decay rate a b (le_of_lt hg)
    (le_of_lt (lt_of_lt_of_le hl hs)) hu he (le_of_lt heta) ha hb hsmall
  have hfloor : 0 < gainPairMassMultiplier lower gain eps decay rate ceiling * (a + b) := mul_pos hmult.1 hmass
  have hnext : 0 < (gainPairStep gain scale eps decay rate a b).1 + (gainPairStep gain scale eps decay rate a b).2 :=
    lt_of_lt_of_le hfloor hf
  calc
    _ ≤ (1 - rate * decay) * |a - b| / _ := div_le_div_of_nonneg_right hdifference (le_of_lt hnext)
    _ ≤ (1 - rate * decay) * |a - b| / (gainPairMassMultiplier lower gain eps decay rate ceiling * (a + b)) :=
      div_le_div_of_nonneg_left (mul_nonneg hd (abs_nonneg _)) hfloor hf
    _ = _ := by field_simp [ne_of_gt hmult.1, ne_of_gt hmass]

example : (0 : ℝ) < 1 / 10 ∧ (1 / 10 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 2 ∧ (1 : ℝ) ≤ 2 ∧ (0 : ℝ) < 0 + 1 ∧ (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by
  norm_num

end Transformer.Grokking.CircuitEfficiency

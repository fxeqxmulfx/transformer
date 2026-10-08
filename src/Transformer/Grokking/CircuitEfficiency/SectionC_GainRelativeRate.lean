import Transformer.Grokking.CircuitEfficiency.SectionC_GainBalancedStep

/-!
# A positive relative-growth advantage before balanced Gen catches Mem

Sources: Varma et al., arXiv:2309.02390v1, section 3's circuit
efficiency and appendix C's product CE; actual native reduction at
lab commit 23b05c3. Factor the proved zero-beta amplitude step into
its current amplitude and a multiplicative rate. Compare the Gen
and Mem rates at the same actual shared CE multiplier.

When Gen is no stronger, greater physical gain gives a strictly
positive relative-rate gap. Derive an explicit uniform gap from a
positive lower CE scale, its unit ceiling and a finite factor box.
Those quantities are already derived on initialized native paths;
here the comparison is algebraic at one current point, not a future
gradient, convergence or successful-margin assumption.

Zero betas and equal factors within each pair specialize native
AdamW and the source's unbalanced first-zero-factor setup. Physical
gains and uniform decoupled decay differ from the source's assigned
coupled circuit norm/GD. Later finite-time escape still needs an
iteration on the actual recurrence, not just a strict single-step gap.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- Actual balanced zero-beta input divided by its current amplitude.
Source: native amplitude formula at 23b05c3; the factor dependence
is retained in the native epsilon normalization. -/
noncomputable def gainBalancedRelativeRate (gain scale eps factor : ℝ) : ℝ :=
  scale * gain / (scale * gain * factor + eps)

/-- Explicit quantitative gap from the actual scale floor and factor
box. Sources: appendix C competing gained products and the native
normalization at 23b05c3; no trajectory is encoded in this number. -/
noncomputable def gainBalancedRelativeGap (lower eps genGain memGain ceiling : ℝ) : ℝ :=
  lower * eps * (genGain - memGain) / ((genGain * ceiling + eps) * (memGain * ceiling + eps))

/-- Algebraic factorization of the already-derived actual balanced
step. Source: native zero-beta formula at 23b05c3; this identity is
bookkeeping for the later relative-growth iteration. -/
theorem gain_balanced_factor_step_multiplier (gain scale eps decay rate factor : ℝ) :
    gainBalancedFactorStep gain scale eps decay rate factor =
      factor * (1 - rate * decay + rate * gainBalancedRelativeRate gain scale eps factor) := by
  unfold gainBalancedFactorStep gainBalancedRelativeRate
  ring

/-- At the same current amplitude, the actual normalized relative
rates differ by this exact gained-CE expression. Sources: appendix C
chain factors and native normalization at 23b05c3; denominator
nonvanishing is explicit rather than hidden in a rate definition. -/
theorem gain_balanced_relative_gap_identity (scale eps genGain memGain factor : ℝ)
    (hg : scale * genGain * factor + eps ≠ 0) (hm : scale * memGain * factor + eps ≠ 0) :
    gainBalancedRelativeRate genGain scale eps factor - gainBalancedRelativeRate memGain scale eps factor =
      scale * eps * (genGain - memGain) /
        ((scale * genGain * factor + eps) * (scale * memGain * factor + eps)) := by
  unfold gainBalancedRelativeRate
  generalize hgd : scale * genGain * factor + eps = genDenominator at hg ⊢
  generalize hmd : scale * memGain * factor + eps = memDenominator at hm ⊢
  field_simp [hg, hm]
  rw [← hgd, ← hmd]
  ring

example : (1 / 2 : ℝ) * 3 * 1 + 1 ≠ 0 ∧ (1 / 2 : ℝ) * 2 * 1 + 1 ≠ 0 := by
  norm_num

/-- Before balanced Gen catches Mem, its current relative rate exceeds
Mem by a fixed positive box/floor gap. Sources: appendix C efficiency
and native factor normalization at 23b05c3; hypotheses are numerical
current bounds, later supplied by the actual seeded-path estimates. -/
theorem gain_balanced_relative_gap_floor (lower scale eps genGain memGain ceiling a b : ℝ)
    (hlower : 0 < lower) (hscale : lower ≤ scale) (hunit : scale ≤ 1) (he : 0 < eps)
    (hmem : 0 < memGain) (hgain : memGain < genGain)
    (ha : 0 ≤ a) (hab : a ≤ b) (hb : b ≤ ceiling) :
    0 < gainBalancedRelativeGap lower eps genGain memGain ceiling ∧
      gainBalancedRelativeGap lower eps genGain memGain ceiling ≤
        gainBalancedRelativeRate genGain scale eps a - gainBalancedRelativeRate memGain scale eps b := by
  have hs : 0 < scale := lt_of_lt_of_le hlower hscale
  have hg : 0 < genGain := lt_trans hmem hgain
  have hbn : 0 ≤ b := le_trans ha hab
  have hc : 0 ≤ ceiling := le_trans hbn hb
  have hda : 0 < scale * genGain * a + eps := by positivity
  have hdg : 0 < scale * genGain * b + eps := by positivity
  have hdm : 0 < scale * memGain * b + eps := by positivity
  have hbig : 0 < (genGain * ceiling + eps) * (memGain * ceiling + eps) := by positivity
  have hgap : 0 < gainBalancedRelativeGap lower eps genGain memGain ceiling := by
    unfold gainBalancedRelativeGap
    positivity
  have hgscale := mul_le_mul_of_nonneg_right hunit (mul_nonneg (le_of_lt hg) hbn)
  have hmscale := mul_le_mul_of_nonneg_right hunit (mul_nonneg (le_of_lt hmem) hbn)
  have hgbox := mul_le_mul_of_nonneg_left hb (le_of_lt hg)
  have hmbox := mul_le_mul_of_nonneg_left hb (le_of_lt hmem)
  have hgden : scale * genGain * b + eps ≤ genGain * ceiling + eps := by
    nlinarith only [hgscale, hgbox]
  have hmden : scale * memGain * b + eps ≤ memGain * ceiling + eps := by
    nlinarith only [hmscale, hmbox]
  have hproduct := mul_le_mul hgden hmden (le_of_lt hdm)
    (show 0 ≤ genGain * ceiling + eps by positivity)
  have hnum : lower * eps * (genGain - memGain) ≤ scale * eps * (genGain - memGain) := by
    have hm := mul_le_mul_of_nonneg_right hscale (show 0 ≤ eps * (genGain - memGain) by positivity)
    nlinarith only [hm]
  have hfirst := div_le_div_of_nonneg_right hnum (le_of_lt hbig)
  have hsecond := div_le_div_of_nonneg_left
    (show 0 ≤ scale * eps * (genGain - memGain) by positivity) (mul_pos hdg hdm) hproduct
  have hidentity := gain_balanced_relative_gap_identity scale eps genGain memGain b (ne_of_gt hdg) (ne_of_gt hdm)
  have hmon : gainBalancedRelativeRate genGain scale eps b ≤ gainBalancedRelativeRate genGain scale eps a := by
    unfold gainBalancedRelativeRate
    apply div_le_div_of_nonneg_left (by positivity) hda
    have hm := mul_le_mul_of_nonneg_left hab (show 0 ≤ scale * genGain by positivity)
    linarith only [hm]
  refine ⟨hgap, ?_⟩
  unfold gainBalancedRelativeGap
  linarith only [hfirst, hsecond, hidentity, hmon]

example : (0 : ℝ) < 1 / 10 ∧ (1 / 10 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) ≤ 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 2 := by
  norm_num

/-- The current Mem relative rate has a fixed native epsilon ceiling.
Source: actual zero-beta normalization at 23b05c3; no future direction
or buffer reset is used in this estimate. -/
theorem gain_balanced_relative_rate_ceiling (gain scale eps factor : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (hunit : scale ≤ 1) (he : 0 < eps) (hf : 0 ≤ factor) :
    gainBalancedRelativeRate gain scale eps factor ≤ gain / eps := by
  have hnum : scale * gain ≤ gain := by
    have hm := mul_le_mul_of_nonneg_right hunit hg
    simpa only [one_mul] using hm
  have hden : eps ≤ scale * gain * factor + eps := by
    have hm := mul_nonneg (mul_nonneg hs hg) hf
    linarith only [hm]
  have hfirst := div_le_div_of_nonneg_left (mul_nonneg hs hg) he hden
  have hsecond := div_le_div_of_nonneg_right hnum (le_of_lt he)
  unfold gainBalancedRelativeRate
  exact le_trans hfirst hsecond

example : (0 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := by
  norm_num

/-- Nonnegative factors/gain/scale have a nonnegative actual normalized
relative input. Source: native zero-beta factor formula at 23b05c3;
the positive epsilon ensures the displayed denominator is positive. -/
theorem gain_balanced_relative_rate_nonneg (gain scale eps factor : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (he : 0 < eps) (hf : 0 ≤ factor) :
    0 ≤ gainBalancedRelativeRate gain scale eps factor := by
  have hn : 0 ≤ scale * gain := mul_nonneg hs hg
  have hd : 0 < scale * gain * factor + eps := by
    have hm := mul_nonneg hn hf
    linarith only [hm, he]
  unfold gainBalancedRelativeRate
  exact div_nonneg hn (le_of_lt hd)

example : (0 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := by
  norm_num

end Transformer.Grokking.CircuitEfficiency

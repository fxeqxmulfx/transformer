import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdBounds

/-!
# Actual retained native collapse above the strict cold threshold

Sources: Varma et al., arXiv:2309.02390v1, section 3's CE/decay
competition and appendix C's gained product feedback; native masses
at ada36f3, cold ceiling at 696630b and generated tails at 8420a09.

If cold coefficient times the greater physical gain is strictly below
decay times epsilon, choose a floor midway between epsilon and that
coefficient divided by decay. It is positive, strictly below epsilon,
and leaves a strict feedback gap. Completed clocks generate the actual
common denominator tail; retained feedback generates positive weights
and a strict contraction factor for parameter and negative moment mass.

The original initialized path then has a geometric weighted ceiling
after the generated finite start. Nonnegative masses are bounded by
the weighted sum divided by its positive weights, so both tend to zero.
No future mass, applied-input, variance or denominator limit is assumed.

This closes the strict sufficient-collapse threshold opposite the
initialized weak-decay cold obstruction. Equality is still separate.
Neither shrinking mass nor moment convergence asserts relative Gen/Mem
selection, held-out confidence or learned GPTMini grokking. The result
is exact-real fixed tables/plain CE/uniform native decoupled decay,
rather than the source's assigned coupled norm-cost GD. No optimizer,
buffer, checkpoint or thermodynamic system-size limit is changed.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- A strict cold feedback gap generates positive retained weights
and a geometric ceiling on the original native trajectory. Sources:
section 3 competition, appendix C partials, actual envelopes at
8420a09 and native retained contraction at 4e81567; no future limits. -/
theorem gain_native_strict_cold_tail_contraction (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hkeep : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial)
    (hgap : coldGainCEGradientScale remaining bound * genGain < decay * eps) :
    let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    ∃ a b q : ℝ, ∃ start : ℕ, 0 < a ∧ 0 < b ∧ 0 ≤ q ∧ q < 1 ∧
      ∀ k, a * gainNativeParameterMass (state (start + k)) +
        b * gainNativeNegativeMomentMass (state (start + k)) ≤ q ^ k *
          (a * gainNativeParameterMass (state start) + b * gainNativeNegativeMomentMass (state start)) := by
  let coefficient := coldGainCEGradientScale remaining bound * genGain
  let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
  have hc : 0 < coefficient := mul_pos (cold_gain_ce_gradient_scale_pos remaining bound hclip) hgen
  have hdecay : 0 < decay := by
    by_contra hd
    have hh := mul_nonpos_of_nonpos_of_nonneg (le_of_not_gt hd) (le_of_lt he)
    linarith only [hc, hgap, hh]
  have hdivide : coefficient / decay < eps :=
    (div_lt_iff₀ hdecay).mpr (by nlinarith only [hgap])
  let floor := (eps + coefficient / decay) / 2
  have hfloor : 0 < floor := by dsimp only [floor]; positivity
  have hfloorLt : floor < eps := by dsimp only [floor]; linarith only [hdivide]
  have hidentity : 2 * decay * floor = decay * eps + coefficient := by
    dsimp only [floor]
    field_simp [ne_of_gt hdecay]
  have hfloorGap : coefficient < decay * floor := by nlinarith only [hidentity, hgap]
  obtain ⟨start, htail⟩ := gain_native_cold_mass_tail_envelopes remaining genGain memGain bound b1 b2 eps decay rate floor initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he (le_of_lt heta) hkeep hs hfloor hfloorLt
  obtain ⟨a, b, q, ha, hb, hq, hq1, hpRow, hmRow⟩ :=
    retained_feedback_contraction_weights b1 floor rate decay coefficient hb1 h1 hfloor heta hc hkeep hfloorGap
  have hsign : ∀ n, NonnegativeNativeState (state n) := fun n =>
    gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
      hgen hmem hclip hb1 h1 hb2 h2 he (le_of_lt heta) hkeep hs
  refine ⟨a, b, q, start, ha, hb, hq, hq1, ?_⟩
  exact retained_feedback_weight_tail_power b1 (1 - rate * decay) rate floor coefficient a b q
    (fun n => gainNativeParameterMass (state n)) (fun n => gainNativeNegativeMomentMass (state n)) start
    (le_of_lt heta) hfloor (le_of_lt ha) (le_of_lt hb) hq
    (fun n => (gain_native_masses_nonnegative _ (hsign n)).1)
    (fun n => (gain_native_masses_nonnegative _ (hsign n)).2.1)
    hpRow hmRow (fun n hn => (htail n hn).1) (fun n hn => (htail n hn).2)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * 2 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 < (2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

/-- Above the strict cold threshold both actual parameter and retained
negative moment masses tend to zero. Sources: section 3 decay/CE,
appendix C actual inputs and native tail bounds at 8420a09; the path
retains both buffers and no convergence is supplied as a hypothesis. -/
theorem gain_native_strict_cold_masses_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hkeep : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial)
    (hgap : coldGainCEGradientScale remaining bound * genGain < decay * eps) :
    let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    Tendsto (fun n => gainNativeParameterMass (state n)) atTop (nhds 0) ∧
      Tendsto (fun n => gainNativeNegativeMomentMass (state n)) atTop (nhds 0) := by
  let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
  let p := fun n => gainNativeParameterMass (state n)
  let m := fun n => gainNativeNegativeMomentMass (state n)
  obtain ⟨a, b, q, start, ha, hb, hq, hq1, hpower⟩ :=
    gain_native_strict_cold_tail_contraction remaining genGain memGain bound b1 b2 eps decay rate initial
      hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep hs hgap
  have hsign : ∀ n, NonnegativeNativeState (state n) := fun n =>
    gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
      hgen hmem hclip hb1 h1 hb2 h2 he (le_of_lt heta) hkeep hs
  have hp : ∀ n, 0 ≤ p n := fun n => (gain_native_masses_nonnegative _ (hsign n)).1
  have hm : ∀ n, 0 ≤ m n := fun n => (gain_native_masses_nonnegative _ (hsign n)).2.1
  let w := fun n => a * p n + b * m n
  have hw : ∀ n, 0 ≤ w n := fun n => add_nonneg
    (mul_nonneg (le_of_lt ha) (hp n)) (mul_nonneg (le_of_lt hb) (hm n))
  have hceiling : Tendsto (fun k => q ^ k * w start) atTop (nhds 0) := by
    simpa only [zero_mul] using (tendsto_pow_atTop_nhds_zero_of_lt_one hq hq1).mul_const (w start)
  have hshift : Tendsto (fun k => w (start + k)) atTop (nhds 0) :=
    squeeze_zero (fun k => hw _) hpower hceiling
  have ht : Tendsto w atTop (nhds 0) :=
    (tendsto_add_atTop_iff_nat start).mp (by simpa only [Nat.add_comm] using hshift)
  have hpBound : ∀ n, p n ≤ w n / a := by
    intro n
    apply (le_div_iff₀ ha).mpr
    have hh := mul_nonneg (le_of_lt hb) (hm n)
    dsimp only [w]
    nlinarith only [hh]
  have hmBound : ∀ n, m n ≤ w n / b := by
    intro n
    apply (le_div_iff₀ hb).mpr
    have hh := mul_nonneg (le_of_lt ha) (hp n)
    dsimp only [w]
    nlinarith only [hh]
  exact ⟨squeeze_zero hp hpBound (by simpa only [zero_div] using ht.div_const a),
    squeeze_zero hm hmBound (by simpa only [zero_div] using ht.div_const b)⟩

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

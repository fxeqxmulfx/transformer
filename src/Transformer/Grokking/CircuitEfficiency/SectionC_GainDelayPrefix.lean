import Transformer.Grokking.CircuitEfficiency.SectionC_GainGrowthBounds

/-!
# A train-correct, uniquely wrong-test native prefix

Sources: Varma et al., arXiv:2309.02390v1, section 3 Slow vs fast
learning and appendix C's competing fixed tables; native AdamW at
lab commit 29715f4. Compare the derived actual Gen upper envelope
and Mem pure-decay floor uniformly through a finite update budget.
The only smallness condition concerns numerical initial data and
the previously derived constants, not future path observations.

Mem starts with two positive factors, so this comparison isolates
an already train-fitted stage. A later small-seed construction must
show the initial inequality is attainable for every fixed budget.
This prefix alone does not prove a later transition or convergence.
All retained moments, variances and clocks evolve by ordinary native
AdamW; no optimizer resets, teacher or altered training objective
are introduced. Uniform decoupled decay and physical gains differ
from the source's coupled assigned norm cost and GD. The fixed-table
exact-real result is not a learned or floating-point GPTMini claim.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- A positive Mem score above Gen uniquely selects the source's
wrong held-out class against all competitors. Source: appendix C's
fixed test tables; uniqueness excludes a mere argmax-tie artifact. -/
theorem heldout_table_mem_strict_correct (remaining : ℕ) (x y : ℝ) (hy : 0 < y) (hxy : x < y) :
    Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits remaining x y)
      (0 : Fin (remaining + 1)).succ := by
  intro k hk
  have hw : heldoutTableLogits remaining x y (0 : Fin (remaining + 1)).succ = y := by
    simp [heldoutTableLogits]
  rw [hw]
  by_cases hz : k = 0
  · rw [hz]
    simpa only [heldoutTableLogits, ite_true] using hxy
  · simpa only [heldoutTableLogits, hz, hk, ite_false] using hy

example : (0 : ℝ) < 2 ∧ (1 : ℝ) < 2 := by norm_num

/-- Initial envelope smallness forces actual Mem dominance at every
clock through the stated budget. Sources: section 3's slow formation,
appendix C product logits and retained AdamW at 29715f4; neither a
future weak rule nor later successful allocation is assumed. -/
theorem gain_native_mem_dominant_prefix (remaining budget : ℕ)
    (genGain memGain bound b1 b2 eps decay rate amplitude : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hdecay : 0 ≤ decay) (hd : 0 < 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (ha : 0 < amplitude)
    (ha2 : amplitude ≤ (initial 2).parameter) (ha3 : amplitude ≤ (initial 3).parameter)
    (hmargin : genGain * (gainGenGrowthFactor b1 eps rate genGain ^ budget *
      gainGenGrowthWeight b1 eps rate initial) ^ 2 < memGain * ((1 - rate * decay) ^ budget * amplitude) ^ 2) :
    ∀ n, n ≤ budget →
      let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n
      0 < physicalCircuitScore memGain (state 2).parameter (state 3).parameter ∧
        physicalCircuitScore genGain (state 0).parameter (state 1).parameter <
          physicalCircuitScore memGain (state 2).parameter (state 3).parameter := by
  intro n hn
  dsimp only
  let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n
  have hsnow := gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
    hgen hmem hclip hb1 h1 hb2 h2 he heta (le_of_lt hd) hs
  have hr := gain_gen_growth_factor_two_le b1 eps rate genGain h1 he heta (le_of_lt hgen)
  have hi := gain_gen_growth_weight_covers_mass b1 eps rate initial h1 he heta hs
  have hweight : 0 ≤ gainGenGrowthWeight b1 eps rate initial := le_trans hi.1 hi.2
  have hscale := mul_le_mul_of_nonneg_right
    (pow_le_pow_right₀ (show 1 ≤ gainGenGrowthFactor b1 eps rate genGain by linarith only [hr]) hn) hweight
  have hnn := mul_nonneg (pow_nonneg (show 0 ≤ gainGenGrowthFactor b1 eps rate genGain by linarith only [hr]) n) hweight
  have hsq : (gainGenGrowthFactor b1 eps rate genGain ^ n * gainGenGrowthWeight b1 eps rate initial) ^ 2 ≤
      (gainGenGrowthFactor b1 eps rate genGain ^ budget * gainGenGrowthWeight b1 eps rate initial) ^ 2 := by
    nlinarith only [hscale, hnn]
  have hgenbound := le_trans
    (gain_gen_native_path_score_ceiling remaining genGain memGain bound b1 b2 eps decay rate initial n
      hgen hmem hclip hb1 h1 hb2 h2 he heta hdecay (le_of_lt hd) hs)
    (mul_le_mul_of_nonneg_left hsq (le_of_lt hgen))
  have hk : 1 - rate * decay ≤ 1 := by nlinarith only [mul_nonneg heta hdecay]
  have hfloor := mul_le_mul_of_nonneg_right (pow_le_pow_of_le_one (le_of_lt hd) hk hn) (le_of_lt ha)
  have h2floor := gain_native_parameter_path_decay_floor remaining genGain memGain bound b1 b2 eps decay rate initial n 2
    hgen hmem hclip hb1 h1 hb2 h2 he heta (le_of_lt hd) hs
  have h3floor := gain_native_parameter_path_decay_floor remaining genGain memGain bound b1 b2 eps decay rate initial n 3
    hgen hmem hclip hb1 h1 hb2 h2 he heta (le_of_lt hd) hs
  have h2bound := le_trans hfloor (le_trans (mul_le_mul_of_nonneg_left ha2 (pow_nonneg (le_of_lt hd) n)) h2floor)
  have h3bound := le_trans hfloor (le_trans (mul_le_mul_of_nonneg_left ha3 (pow_nonneg (le_of_lt hd) n)) h3floor)
  have hpositive : 0 < (1 - rate * decay) ^ budget * amplitude := by positivity
  have hproduct := mul_le_mul h2bound h3bound (le_of_lt hpositive) (hsnow 2).1
  have hmembound : memGain * ((1 - rate * decay) ^ budget * amplitude) ^ 2 ≤
      physicalCircuitScore memGain (state 2).parameter (state 3).parameter := by
    simpa only [physicalCircuitScore, pow_two] using mul_le_mul_of_nonneg_left hproduct (le_of_lt hmem)
  have hmpositive : 0 < memGain * ((1 - rate * decay) ^ budget * amplitude) ^ 2 := by positivity
  exact ⟨lt_of_lt_of_le hmpositive hmembound,
    lt_of_le_of_lt hgenbound (lt_of_lt_of_le hmargin hmembound)⟩

example :
    let initial := seededNativeSubweights ((0, 1 / 200), (1, 1))
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 0 ≤ (1 / 1000 : ℝ) ∧
    0 ≤ (1 / 10 : ℝ) ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ NonnegativeNativeState initial ∧
    (0 : ℝ) < 1 ∧ 1 ≤ (initial 2).parameter ∧ 1 ≤ (initial 3).parameter ∧
    3 * (gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 ^ 3 *
      gainGenGrowthWeight (9 / 10) 1 (1 / 1000) initial) ^ 2 < 2 * ((1 - (1 / 1000 : ℝ) * (1 / 10)) ^ 3 * 1) ^ 2 := by
  dsimp only
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num, by norm_num [seededNativeSubweights, seededScalarState],
    by norm_num [seededNativeSubweights, seededScalarState], ?_⟩
  norm_num [gainGenGrowthFactor, gainGenGrowthWeight, gainGenParameterMass,
    gainGenNegativeMomentMass, seededNativeSubweights, seededScalarState]

/-- The actual prefix fits the source's train class and uniquely
selects Mem's wrong test class. Sources: section 3's plateau description,
appendix C tables and native AdamW at 29715f4; this is a full-class
decision statement with an initial-data condition, not an assumed plateau. -/
theorem gain_native_train_correct_wrong_test_prefix (remaining budget : ℕ)
    (genGain memGain bound b1 b2 eps decay rate amplitude : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hdecay : 0 ≤ decay) (hd : 0 < 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (ha : 0 < amplitude)
    (ha2 : amplitude ≤ (initial 2).parameter) (ha3 : amplitude ≤ (initial 3).parameter)
    (hmargin : genGain * (gainGenGrowthFactor b1 eps rate genGain ^ budget *
      gainGenGrowthWeight b1 eps rate initial) ^ 2 < memGain * ((1 - rate * decay) ^ budget * amplitude) ^ 2) :
    ∀ n, n ≤ budget →
      let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n
      let x := physicalCircuitScore genGain (state 0).parameter (state 1).parameter
      let y := physicalCircuitScore memGain (state 2).parameter (state 3).parameter
      Transformer.Grokking.NaiveLoss.StrictCorrect (trainTableLogits remaining x y) 0 ∧
        Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits remaining x y) (0 : Fin (remaining + 1)).succ := by
  intro n hn
  dsimp only
  have hm := gain_native_mem_dominant_prefix remaining budget genGain memGain bound b1 b2 eps decay rate amplitude initial
    hgen hmem hclip hb1 h1 hb2 h2 he heta hdecay hd hs ha ha2 ha3 hmargin n hn
  have hsnow := gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
    hgen hmem hclip hb1 h1 hb2 h2 he heta (le_of_lt hd) hs
  have hx : 0 ≤ physicalCircuitScore genGain
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n 0).parameter
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n 1).parameter :=
    mul_nonneg (le_of_lt hgen) (mul_nonneg (hsnow 0).1 (hsnow 1).1)
  exact ⟨(train_table_strict_correct_iff remaining _ _).mpr (by linarith only [hx, hm.1]),
    heldout_table_mem_strict_correct remaining _ _ hm.1 hm.2⟩

example :
    let initial := seededNativeSubweights ((0, 1 / 200), (1, 1))
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 0 ≤ (1 / 1000 : ℝ) ∧
    0 ≤ (1 / 10 : ℝ) ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ NonnegativeNativeState initial ∧
    (0 : ℝ) < 1 ∧ 1 ≤ (initial 2).parameter ∧ 1 ≤ (initial 3).parameter ∧
    3 * (gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 ^ 3 *
      gainGenGrowthWeight (9 / 10) 1 (1 / 1000) initial) ^ 2 < 2 * ((1 - (1 / 1000 : ℝ) * (1 / 10)) ^ 3 * 1) ^ 2 := by
  dsimp only
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num, by norm_num [seededNativeSubweights, seededScalarState],
    by norm_num [seededNativeSubweights, seededScalarState], ?_⟩
  norm_num [gainGenGrowthFactor, gainGenGrowthWeight, gainGenParameterMass,
    gainGenNegativeMomentMass, seededNativeSubweights, seededScalarState]

end Transformer.Grokking.CircuitEfficiency

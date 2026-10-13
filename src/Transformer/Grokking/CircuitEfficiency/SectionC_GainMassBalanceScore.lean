import Transformer.Grokking.CircuitEfficiency.SectionC_GainUnequalOrderTail

/-!
# Relative balance turns unequal mass dominance into actual task decisions

Sources: Varma et al., arXiv:2309.02390v1, appendix C's product
train/test logits and section 3's physical efficiency; original
retained native balance and mass-selection laws at commit 483b2b1.

Pair product equals one quarter of squared pair mass minus squared
factor difference. An explicit squared-relative-asymmetry budget
therefore gives a product lower bound without a positive mass limit.
For Gen gain greater than Mem gain, choose the budget as half their
relative gain difference. If Gen mass is at least Mem mass, the true
physical Gen-minus-Mem score is at least (gainGen-gainMem)/8 times
current Mem mass squared. This is positive at every positive-mass
point satisfying the current budget, even for unequal factors.

Apply this scalar comparison to the actual current native state
and its original train/held-out logits. Derive strict correctness
against every competing class, not just an abstract mass inequality.
The actual-path application must generate the balance budget and
mass dominance from initial data; no future margin is assumed here.

The lower margin may approach zero with current Mem mass. It is
not a uniform positive limiting margin or a CE-loss bound. Exact
reals, fixed gained tables, zero-beta native selection and uniform
decay differ from appendix C's assigned coupled norm/GD and learned
GPTMini. No floating-point or learned-feature transfer is asserted.
Factor signs are not hidden in the scalar product bound; a balance
budget below one derives a positive product from positive pair mass.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss

/-- A squared relative-balance budget bounds the current product
below by its fixed-sum maximum times one minus the budget. Sources:
appendix C's product forward and checked balance identity at 483b2b1;
current positive mass is explicit, without a limiting-mass hypothesis. -/
theorem gain_pair_balance_product_floor (a b allowance : ℝ)
    (hmass : 0 < a + b) (hbalance : gainPairRelativeDifference a b ^ 2 ≤ allowance) :
    (1 - allowance) * (a + b) ^ 2 ≤ 4 * a * b := by
  have hidentity := gain_pair_relative_product a b (ne_of_gt hmass)
  have hcancel := div_mul_cancel₀ (4 * a * b) (ne_of_gt (pow_pos hmass 2))
  rw [hidentity] at hcancel
  have hweighted := mul_le_mul_of_nonneg_right hbalance (sq_nonneg (a + b))
  nlinarith only [hcancel, hweighted]

example : (0 : ℝ) < 1 + 1 ∧ gainPairRelativeDifference 1 1 ^ 2 ≤ 1 / 10 := by
  norm_num [gainPairRelativeDifference]

/-- A squared relative-asymmetry budget below one forces positive
current product even without separate factor-sign hypotheses.
Sources: appendix C product logits and the exact relative identity
at 483b2b1; current positive mass is enough for the normalization.
This is a current product certificate, not a positive limiting
product or a permanent trajectory claim. -/
theorem gain_pair_balance_product_positive (a b allowance : ℝ)
    (hmass : 0 < a + b) (hbalance : gainPairRelativeDifference a b ^ 2 ≤ allowance)
    (hallowance : allowance < 1) : 0 < a * b := by
  have hfloor := gain_pair_balance_product_floor a b allowance hmass hbalance
  have hleft : 0 < 1 - allowance := by linarith only [hallowance]
  have hpositive := mul_pos hleft (pow_pos hmass 2)
  nlinarith only [hfloor, hpositive]

example : (0 : ℝ) < 1 + 2 ∧ gainPairRelativeDifference 1 2 ^ 2 ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 := by
  norm_num [gainPairRelativeDifference]

/-- Efficient Gen mass dominance and a quantitative balance budget
give a positive true physical score margin for unequal factors.
Sources: appendix C product logits and physical efficiency at 483b2b1;
factor signs and individual parameter limits are unnecessary for this
scalar inequality, since the sum/balance conditions control Gen. -/
theorem gain_pair_mass_balance_score_margin (genGain memGain a b c d : ℝ)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hmMass : 0 < c + d)
    (horder : c + d ≤ a + b)
    (hbalance : gainPairRelativeDifference a b ^ 2 ≤ (genGain - memGain) / (2 * genGain)) :
    0 < (genGain - memGain) * (c + d) ^ 2 / 8 ∧
      (genGain - memGain) * (c + d) ^ 2 / 8 ≤
        physicalCircuitScore genGain a b - physicalCircuitScore memGain c d := by
  have hg := lt_trans hmem hgain
  have hgMass := lt_of_lt_of_le hmMass horder
  have hproduct := gain_pair_balance_product_floor a b ((genGain - memGain) / (2 * genGain)) hgMass hbalance
  have hweighted := mul_le_mul_of_nonneg_left hproduct (le_of_lt hg)
  have hcoefficient : genGain * (1 - (genGain - memGain) / (2 * genGain)) = (genGain + memGain) / 2 := by
    field_simp [ne_of_gt hg]
    ring
  have hGenFloor : (genGain + memGain) / 2 * (a + b) ^ 2 ≤ 4 * physicalCircuitScore genGain a b := by
    calc
      _ = genGain * ((1 - (genGain - memGain) / (2 * genGain)) * (a + b) ^ 2) := by rw [← hcoefficient]; ring
      _ ≤ genGain * (4 * a * b) := hweighted
      _ = _ := by unfold physicalCircuitScore; ring
  have hsumSq := mul_le_mul horder horder (le_of_lt hmMass) (le_of_lt hgMass)
  have hsqWeighted := mul_le_mul_of_nonneg_left hsumSq
    (show 0 ≤ (genGain + memGain) / 2 by positivity)
  have hMemNonneg := mul_nonneg (le_of_lt hmem) (sq_nonneg (c - d))
  have hMemCeiling : 4 * physicalCircuitScore memGain c d ≤ memGain * (c + d) ^ 2 := by
    unfold physicalCircuitScore
    nlinarith only [hMemNonneg]
  have hgap : 0 < genGain - memGain := by linarith only [hgain]
  constructor
  · exact div_pos (mul_pos hgap (pow_pos hmMass 2)) (by norm_num)
  · nlinarith only [hGenFloor, hsqWeighted, hMemCeiling]

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 + 1 ∧ (1 : ℝ) + 1 ≤ 1 + 2 ∧
    gainPairRelativeDifference 1 2 ^ 2 ≤ (3 - 2 : ℝ) / (2 * 3) := by
  norm_num [gainPairRelativeDifference]

/-- Current unequal mass order and a derived relative-balance budget
give strictly correct actual train and held-out decisions against all
classes. Sources: appendix C actual logits and physical efficiency at
483b2b1; the pointwise hypotheses are not a future success predicate. -/
theorem gain_native_mass_balance_correct (remaining : ℕ) (genGain memGain : ℝ) (state : NativeSubweightState)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hp : ∀ i, 0 ≤ (state i).parameter)
    (hmMass : 0 < (state 2).parameter + (state 3).parameter)
    (horder : (state 2).parameter + (state 3).parameter ≤ (state 0).parameter + (state 1).parameter)
    (hbalance : gainPairRelativeDifference (state 0).parameter (state 1).parameter ^ 2 ≤
      (genGain - memGain) / (2 * genGain)) :
    StrictCorrect (trainTableLogits remaining
      (physicalCircuitScore genGain (state 0).parameter (state 1).parameter)
      (physicalCircuitScore memGain (state 2).parameter (state 3).parameter)) 0 ∧
      StrictCorrect (heldoutTableLogits remaining
        (physicalCircuitScore genGain (state 0).parameter (state 1).parameter)
        (physicalCircuitScore memGain (state 2).parameter (state 3).parameter)) 0 := by
  have hm := gain_pair_mass_balance_score_margin genGain memGain
    (state 0).parameter (state 1).parameter (state 2).parameter (state 3).parameter hmem hgain hmMass horder hbalance
  have hy : 0 ≤ physicalCircuitScore memGain (state 2).parameter (state 3).parameter :=
    mul_nonneg (le_of_lt hmem) (mul_nonneg (hp 2) (hp 3))
  have hscore : physicalCircuitScore memGain (state 2).parameter (state 3).parameter <
      physicalCircuitScore genGain (state 0).parameter (state 1).parameter := by linarith only [hm.1, hm.2]
  have hx : 0 < physicalCircuitScore genGain (state 0).parameter (state 1).parameter := lt_of_le_of_lt hy hscore
  exact ⟨(train_table_strict_correct_iff remaining _ _).mpr (by linarith only [hx, hy]),
    heldout_table_strict_correct remaining _ _ hx hscore⟩

example :
    let state := seededNativeSubweights ((1, 2), (1, 1))
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (∀ i, 0 ≤ (state i).parameter) ∧
    0 < (state 2).parameter + (state 3).parameter ∧
    (state 2).parameter + (state 3).parameter ≤ (state 0).parameter + (state 1).parameter ∧
    gainPairRelativeDifference (state 0).parameter (state 1).parameter ^ 2 ≤ (3 - 2 : ℝ) / (2 * 3) := by
  dsimp only
  refine ⟨by norm_num, by norm_num, ?_, by norm_num [seededNativeSubweights, seededScalarState],
    by norm_num [seededNativeSubweights, seededScalarState],
    by norm_num [seededNativeSubweights, seededScalarState, gainPairRelativeDifference]⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

end Transformer.Grokking.CircuitEfficiency

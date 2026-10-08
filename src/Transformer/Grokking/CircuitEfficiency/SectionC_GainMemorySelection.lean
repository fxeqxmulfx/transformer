import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryMixing

/-!
# Generated true product and decision selection with native retained memory

Sources: Varma et al., arXiv:2309.02390v1, section 3 competing
circuits and appendix C product logits; original native physical
coverage and positive partner mixing at lab commit eb049c7.

For the pinned binary gained-CE path, initial full numerical signs
and positive present Gen parameter/moment mass suffice for permanent
true Gen product dominance and correct train/held-out decisions after
a generated finite clock. Beta1=0.9 and beta2=0.98, both retained
buffers, full denominator corrections and shared clipping remain.

The proof combines the actual uniform Gen factor cone with arbitrary
small Mem/Gen physical sum ratios. A present-state product argument
makes the actual gained score positive and greater than Mem. Complete
signed factor-balance convergence is not an independent premise.
No future convergence, coefficient interval, factor ratio, margin or
successful reference is supplied to the initialized path theorem.

This proves eventual decision selection, not arbitrary delay, positive
limiting confidence or robustness. The same static decay can make all
physical parameters and both retained buffers tend to zero. The task
has two classes with fixed gained lookup tables, not learned heads.
Uniform native decoupled decay differs from appendix C coupled-cost
GD. Learned stochastic/floating-point GPTMini transfer remains open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- A positive physical Gen sum, its uniform factor cone and a small
Mem sum give a positive gained Gen score beating Mem. Sources:
appendix C actual products and native cone constant at eb049c7;
this is a present numerical implication, not a future margin premise. -/
theorem fixed_gain_pair_cone_score_margin (a b c d : ℝ)
    (hc : 0 ≤ c) (hd : 0 ≤ d) (hmass : 0 < a + b)
    (ha : (1 / 3220000 : ℝ) * (a + b) ≤ a)
    (hb : (1 / 3220000 : ℝ) * (a + b) ≤ b)
    (hrelative : c + d < (1 / 3220000 : ℝ) * (a + b)) :
    0 < physicalCircuitScore 3 a b ∧ physicalCircuitScore 2 c d < physicalCircuitScore 3 a b := by
  let lower := (1 / 3220000 : ℝ) * (a + b)
  have hl : 0 < lower := mul_pos (by norm_num) hmass
  have hap : 0 < a := lt_of_lt_of_le hl ha
  have hbp : 0 < b := lt_of_lt_of_le hl hb
  have hgen := le_trans (mul_le_mul_of_nonneg_right ha (le_of_lt hl))
    (mul_le_mul_of_nonneg_left hb (le_of_lt hap))
  have hsum : 0 ≤ c + d := add_nonneg hc hd
  have hmem := le_trans (mul_le_mul_of_nonneg_right (show c ≤ c + d by linarith only [hd]) hd)
    (mul_le_mul_of_nonneg_left (show d ≤ c + d by linarith only [hc]) hsum)
  change c + d < lower at hrelative
  have hsep := mul_pos (show 0 < lower - (c + d) by linarith only [hrelative])
    (show 0 < lower + (c + d) by linarith only [hl, hsum])
  have hsquare := mul_pos hl hl
  unfold physicalCircuitScore
  refine ⟨mul_pos (by norm_num) (mul_pos hap hbp), ?_⟩
  change lower * lower ≤ a * b at hgen
  nlinarith only [hgen, hmem, hsep, hsquare]

example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 + 1 ∧
    (1 / 3220000 : ℝ) * (1 + 1) ≤ 1 ∧ (1 / 3220000 : ℝ) * (1 + 1) ≤ 1 ∧
    (0 : ℝ) + 0 < (1 / 3220000 : ℝ) * (1 + 1) := by norm_num

/-- True gained physical Gen products are positive and permanently
beat Mem after a generated finite native clock. Sources: section 3
competition, appendix C products and native cone/relative laws at
eb049c7; all future numerical conditions are generated from initial
signs and positive current Gen measurement, not supplied independently. -/
theorem fixed_gain_memory_score_margin_tail (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) (hgen : 0 < fixedGainGenMemoryMass initial) :
    ∃ start : ℕ, ∀ n, start ≤ n →
      0 < physicalCircuitScore 3 (fixedGainMemoryPath initial n 0).parameter (fixedGainMemoryPath initial n 1).parameter ∧
      physicalCircuitScore 2 (fixedGainMemoryPath initial n 2).parameter (fixedGainMemoryPath initial n 3).parameter <
        physicalCircuitScore 3 (fixedGainMemoryPath initial n 0).parameter (fixedGainMemoryPath initial n 1).parameter := by
  obtain ⟨massStart, hmass⟩ := fixed_gain_memory_physical_mass_dominance initial hs hgen (1 / 3220000) (by norm_num)
  obtain ⟨coneStart, hcone⟩ := fixed_gain_memory_gen_pair_cone_tail initial hs
  refine ⟨max massStart (coneStart + 1), ?_⟩
  intro n hn
  have hmn := le_trans (le_max_left _ _) hn
  have hcn := le_trans (le_max_right _ _) hn
  cases n with
  | zero => omega
  | succ k =>
    have hr := hmass (k + 1) hmn
    have hh := hcone k (by omega)
    have hsnow := fixed_gain_memory_signs initial hs (k + 1)
    unfold gainGenParameterMass at hr hh
    have hg : 0 < (fixedGainMemoryPath initial (k + 1) 0).parameter +
        (fixedGainMemoryPath initial (k + 1) 1).parameter := by
      nlinarith only [hr, (hsnow 2).1, (hsnow 3).1]
    exact fixed_gain_pair_cone_score_margin _ _ _ _ (hsnow 2).1 (hsnow 3).1 hg hh.1 hh.2 hr

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    0 < fixedGainGenMemoryMass (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [fixedGainGenMemoryMass, gainNativePairMemoryMass, seededNativeSubweights, seededScalarState, nativeFactorPartner]⟩

/-- The original legal-beta nonlinear native path eventually makes
permanent strict correct train and held-out decisions. Sources:
appendix C actual logits, section 3 competition and original native
mixing/coverage at eb049c7; no parameter convergence, future margin,
coefficient tail or external gradient/success reference is supplied. -/
theorem fixed_gain_memory_eventual_correct (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) (hgen : 0 < fixedGainGenMemoryMass initial) :
    ∃ start : ℕ, ∀ n, start ≤ n →
      let state := fixedGainMemoryPath initial n
      let x := physicalCircuitScore 3 (state 0).parameter (state 1).parameter
      let y := physicalCircuitScore 2 (state 2).parameter (state 3).parameter
      Transformer.Grokking.NaiveLoss.StrictCorrect (trainTableLogits 0 x y) 0 ∧
        Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits 0 x y) 0 := by
  obtain ⟨start, htail⟩ := fixed_gain_memory_score_margin_tail initial hs hgen
  refine ⟨start, ?_⟩
  intro n hn
  dsimp only
  have hscore := htail n hn
  have hsnow := fixed_gain_memory_signs initial hs n
  have hmem : 0 ≤ physicalCircuitScore 2 (fixedGainMemoryPath initial n 2).parameter
      (fixedGainMemoryPath initial n 3).parameter := mul_nonneg (by norm_num) (mul_nonneg (hsnow 2).1 (hsnow 3).1)
  exact ⟨(train_table_strict_correct_iff 0 _ _).2 (by linarith only [hscore.1, hmem]),
    heldout_table_strict_correct 0 _ _ hscore.1 hscore.2⟩

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    0 < fixedGainGenMemoryMass (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [fixedGainGenMemoryMass, gainNativePairMemoryMass, seededNativeSubweights, seededScalarState, nativeFactorPartner]⟩

/-- Permanent correct original native decisions coexist with a
held-out gained score margin tending to zero. Sources: appendix C
actual logits and native initialized selection/collapse at eb049c7;
strict finite-clock accuracy does not supply positive limiting margin. -/
theorem fixed_gain_memory_correct_with_margin_collapse (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) (hgen : 0 < fixedGainGenMemoryMass initial) :
    (∃ start : ℕ, ∀ n, start ≤ n →
      let state := fixedGainMemoryPath initial n
      let x := physicalCircuitScore 3 (state 0).parameter (state 1).parameter
      let y := physicalCircuitScore 2 (state 2).parameter (state 3).parameter
      Transformer.Grokking.NaiveLoss.StrictCorrect (trainTableLogits 0 x y) 0 ∧
        Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits 0 x y) 0) ∧
    Filter.Tendsto (fun n =>
      physicalCircuitScore 3 (fixedGainMemoryPath initial n 0).parameter (fixedGainMemoryPath initial n 1).parameter -
        physicalCircuitScore 2 (fixedGainMemoryPath initial n 2).parameter (fixedGainMemoryPath initial n 3).parameter)
      Filter.atTop (nhds 0) := by
  have hp := gain_native_memory_parameters_tendsto_zero 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) initial
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hs (by norm_num)
  have hg := ((hp 0).mul (hp 1)).const_mul (3 : ℝ)
  have hm := ((hp 2).mul (hp 3)).const_mul (2 : ℝ)
  refine ⟨fixed_gain_memory_eventual_correct initial hs hgen, ?_⟩
  simpa only [physicalCircuitScore, fixedGainMemoryPath, mul_zero, sub_zero] using hg.sub hm

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    0 < fixedGainGenMemoryMass (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [fixedGainGenMemoryMass, gainNativePairMemoryMass, seededNativeSubweights, seededScalarState, nativeFactorPartner]⟩

end Transformer.Grokking.CircuitEfficiency

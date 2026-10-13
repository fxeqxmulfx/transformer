import Transformer.Grokking.CircuitEfficiency.SectionC_GainDelayPrefix
import Transformer.Grokking.CircuitEfficiency.SectionC_GainBounds

/-!
# Positive efficient Gen seeds allow arbitrarily long wrong-test prefixes

Sources: Varma et al., arXiv:2309.02390v1, section 3 Slow vs fast
learning and appendix C's two-factor tables; native AdamW/clipping
at lab commit f31f874. For any finite budget, explicitly choose a
positive Gen partner seed below the derived initial-data ceiling.
The task, gains, betas, epsilon, decay, rate and Mem seed stay fixed.

The initially zero first Gen factor activates immediately under the
actual CE input and stays positive at every later finite clock. Thus
the wrong-test prefix does not depend on an exactly absent Gen pair
or a supplied future plateau. Both buffers and clocks remain native.
Mem starts train-fitted; its formation from the source's untrained
initialization is not proved here. Later successful allocation and
parameter convergence remain separate obligations. The loose bound
may require seeds below machine range, so this exact-real fixed-table
possibility is not a measured or numerical GPTMini delay prediction.
Uniform decoupled AdamW and physical gains differ from coupled-cost GD.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Every finite budget admits a strictly positive numerical seed
meeting the native envelope comparison. Sources: section 3's small
partner seeds and the actual native bounds at f31f874; the constructed
seed changes initial data alone, not the task or optimizer constants. -/
theorem exists_positive_gain_delay_seed (budget : ℕ) (genGain memGain b1 eps decay rate amplitude : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (h1 : b1 < 1) (he : 0 < eps)
    (heta : 0 ≤ rate) (hd : 0 < 1 - rate * decay) (ha : 0 < amplitude) :
    ∃ seed : ℝ, 0 < seed ∧ genGain * (gainGenGrowthFactor b1 eps rate genGain ^ budget * seed) ^ 2 <
      memGain * ((1 - rate * decay) ^ budget * amplitude) ^ 2 := by
  let R := gainGenGrowthFactor b1 eps rate genGain
  let F := (1 - rate * decay) ^ budget * amplitude
  let Q := memGain * F ^ 2 / genGain
  let seed := Real.sqrt Q / (2 * R ^ budget)
  have hr := gain_gen_growth_factor_two_le b1 eps rate genGain h1 he heta (le_of_lt hgen)
  have hrpos : 0 < R := by dsimp only [R]; linarith only [hr]
  have hrpow : 0 < R ^ budget := by positivity
  have hf : 0 < F := by dsimp only [F]; positivity
  have hq : 0 < Q := by dsimp only [Q]; positivity
  have hseed : 0 < seed := div_pos (Real.sqrt_pos.mpr hq) (by positivity)
  have hcancel : R ^ budget * seed = Real.sqrt Q / 2 := by
    dsimp only [seed]
    field_simp [ne_of_gt hrpow]
  have hscore : genGain * (R ^ budget * seed) ^ 2 = memGain * F ^ 2 / 4 := by
    rw [hcancel]
    calc
      genGain * (Real.sqrt Q / 2) ^ 2 = genGain * (Real.sqrt Q ^ 2) / 4 := by ring
      _ = genGain * Q / 4 := by rw [Real.sq_sqrt (le_of_lt hq)]
      _ = memGain * F ^ 2 / 4 := by dsimp only [Q]; field_simp
  refine ⟨seed, hseed, ?_⟩
  change genGain * (R ^ budget * seed) ^ 2 < memGain * F ^ 2
  rw [hscore]
  have hp : 0 < memGain * F ^ 2 := by positivity
  linarith only [hp]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ (0 : ℝ) < 1 := by norm_num

/-- A positive initial Gen partner makes the other factor positive
after every actual native step. Sources: section 3 partner formation,
appendix C's true CE and native AdamW at f31f874; the partner's
finite-clock positivity is derived while the initial first factor may be zero. -/
theorem gain_native_positive_gen_partner_activates (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (n : ℕ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter) :
    PositiveScalarState (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial (n + 1) 0) := by
  have hn := gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
    hgen hmem hclip hb1 h1 hb2 h2 he (le_of_lt heta) (le_of_lt hd) hs
  have hpartner := gain_native_positive_coordinate_path remaining genGain memGain bound b1 b2 eps decay rate initial n 1
    hgen hmem hclip hb1 h1 hb2 h2 he (le_of_lt heta) hd hs hi
  exact scalar_negative_gradient_activates b1 b2 eps decay rate _ _ hb1 h1 hb2 (le_of_lt h2) he heta
    (le_of_lt hd) (hn 0) (gain_native_applied_gradient_negative remaining genGain memGain bound _ 0 hclip
      (show 0 < nativeFactorGain genGain memGain 0 from hgen) hpartner.1)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 < (1 / 1000 : ℝ) ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    0 < (seededNativeSubweights ((0, 1 / 200), (1, 1)) 1).parameter := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [seededNativeSubweights, seededScalarState]⟩

/-- With the same task and native constants, an efficient positive
Gen pair can remain weaker than Mem for any prescribed finite budget.
Sources: section 3's efficiency/slow-learning ingredients and appendix
C tables, native AdamW at f31f874. Gen forms at step one and is
positive thereafter; only its initial partner seed varies with the budget. -/
theorem gain_native_arbitrary_wrong_test_prefix (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate amplitude : ℝ)
    (hmem : 0 < memGain) (hbetter : memGain < genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hdecay : 0 ≤ decay) (hd : 0 < 1 - rate * decay) (ha : 0 < amplitude) :
    ∀ budget : ℕ, ∃ seed : ℝ, 0 < seed ∧
      let initial := seededNativeSubweights ((0, seed), (amplitude, amplitude))
      let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
      (∀ n, 0 < physicalCircuitScore genGain (path (n + 1) 0).parameter (path (n + 1) 1).parameter) ∧
      ∀ n, n ≤ budget →
        let x := physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter
        let y := physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter
        Transformer.Grokking.NaiveLoss.StrictCorrect (trainTableLogits remaining x y) 0 ∧
          Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits remaining x y) (0 : Fin (remaining + 1)).succ := by
  intro budget
  have hgen := lt_trans hmem hbetter
  obtain ⟨seed, hseed, hbound⟩ := exists_positive_gain_delay_seed budget genGain memGain b1 eps decay rate amplitude
    hgen hmem h1 he (le_of_lt heta) hd ha
  let initial := seededNativeSubweights ((0, seed), (amplitude, amplitude))
  have hs : NonnegativeNativeState initial :=
    native_seeded_nonnegative _ _ _ _ le_rfl (le_of_lt hseed) (le_of_lt ha) (le_of_lt ha)
  have hi : 0 < (initial 1).parameter := hseed
  have hweight : gainGenGrowthWeight b1 eps rate initial = seed := by
    simp [gainGenGrowthWeight, gainGenParameterMass, gainGenNegativeMomentMass,
      initial, seededNativeSubweights, seededScalarState]
  refine ⟨seed, hseed, ?_, ?_⟩
  · intro n
    have hf := gain_native_positive_gen_partner_activates remaining genGain memGain bound b1 b2 eps decay rate initial n
      hgen hmem hclip hb1 h1 hb2 h2 he heta hd hs hi
    have hp := gain_native_positive_coordinate_path remaining genGain memGain bound b1 b2 eps decay rate initial (n + 1) 1
      hgen hmem hclip hb1 h1 hb2 h2 he (le_of_lt heta) hd hs hi
    exact mul_pos hgen (mul_pos hf.1 hp.1)
  · apply gain_native_train_correct_wrong_test_prefix remaining budget genGain memGain bound b1 b2 eps decay rate amplitude initial
      hgen hmem hclip hb1 h1 hb2 h2 he (le_of_lt heta) hdecay hd hs ha le_rfl le_rfl
    simpa only [hweight] using hbound

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 < (1 / 1000 : ℝ) ∧ 0 ≤ (1 / 10 : ℝ) ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ (0 : ℝ) < 1 := by norm_num

/-- One fixed nonzero native configuration has arbitrarily long
train-correct uniquely wrong-test prefixes with positive efficient Gen.
Sources: section 3's small seeds and the exact native model at f31f874;
the epsilon matches the lab scale, but the lookup-table gains are toy
constants and the real seed may lie below floating-point range. -/
theorem fixed_native_arbitrary_wrong_test_prefix (remaining : ℕ) :
    ∀ budget : ℕ, ∃ seed : ℝ, 0 < seed ∧
      let initial := seededNativeSubweights ((0, seed), (1, 1))
      let path := gainNativePath remaining 3 2 1 (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000) initial
      (∀ n, 0 < physicalCircuitScore 3 (path (n + 1) 0).parameter (path (n + 1) 1).parameter) ∧
      ∀ n, n ≤ budget →
        let x := physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter
        let y := physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter
        Transformer.Grokking.NaiveLoss.StrictCorrect (trainTableLogits remaining x y) 0 ∧
          Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits remaining x y) (0 : Fin (remaining + 1)).succ := by
  exact gain_native_arbitrary_wrong_test_prefix remaining 3 2 1 (9 / 10) (49 / 50)
    (1 / 100000000) (1 / 10) (1 / 1000) 1 (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

/-- The same fixed-native delayed prefixes have bounded physical
parameters and both retained buffers for all time. Sources: section
3's small seeds, appendix C tables and native AdamW at 3a44336;
boundedness and positive Gen formation are proved on the very same
trajectories, without inferring parameter convergence or later success. -/
theorem fixed_native_bounded_wrong_test_prefix (remaining : ℕ) :
    ∀ budget : ℕ, ∃ seed ceiling : ℝ, 0 < seed ∧ 0 < ceiling ∧
      let initial := seededNativeSubweights ((0, seed), (1, 1))
      let path := gainNativePath remaining 3 2 1 (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000) initial
      (∀ n, NonnegativeNativeState (path n) ∧ ∀ i,
        (path n i).parameter ≤ ceiling ∧ -(path n i).moment ≤ 1 ∧ (path n i).variance ≤ 1 ^ 2) ∧
      (∀ n, 0 < physicalCircuitScore 3 (path (n + 1) 0).parameter (path (n + 1) 1).parameter) ∧
      ∀ n, n ≤ budget →
        let x := physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter
        let y := physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter
        Transformer.Grokking.NaiveLoss.StrictCorrect (trainTableLogits remaining x y) 0 ∧
          Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits remaining x y) (0 : Fin (remaining + 1)).succ := by
  intro budget
  obtain ⟨seed, hseed, hpositive, hprefix⟩ := fixed_native_arbitrary_wrong_test_prefix remaining budget
  obtain ⟨ceiling, hceiling, hbounded⟩ := gain_native_seeded_bounded remaining 3 2 1 (9 / 10) (49 / 50)
    (1 / 100000000) (1 / 10) (1 / 1000) 0 seed 1 1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (le_of_lt hseed) (by norm_num) (by norm_num)
  exact ⟨seed, ceiling, hseed, hceiling, hbounded, hpositive, hprefix⟩

end Transformer.Grokking.CircuitEfficiency

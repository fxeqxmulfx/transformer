import Transformer.Grokking.CircuitEfficiency.SectionC_GainBalancedDelay

/-!
# A vanishing positive Gen initialization family with increasing native delay

Sources: Varma et al., arXiv:2309.02390v1, section 3's small
Gen seeds and appendix C's product logits; actual native growth
envelope and delayed-selection results at lab commit 4f52406.

Choose an explicit geometric family of positive balanced Gen seeds.
The task, gains, cap, epsilon, decay, rate and Mem amplitude stay
fixed; the family index changes only the initial Gen amplitude.
The seeds tend to zero and their true initial-envelope comparison
guarantees uniquely wrong held-out decisions through that index.
Every positive member can separately have eventual permanent success
under the already-checked greater-gain/positive-decay/rate conditions.

This family supports a later test of whether small-initialization and
long-time decision limits commute. It supplies neither a system-size
limit nor a distribution, temperature or thermodynamic singularity.
Legal zero betas, positive balanced factors, exact reals and fixed
gained tables differ from coupled-cost GD, source-style untrained
first factors and preserved learned GPTMini. Very small real seeds
need not be representable by the numerical training implementation.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss Filter

/-- Explicit geometric Gen amplitude from the actual native envelope.
Sources: section 3's small initialization and 4f52406. Its index
changes initial amplitude alone, rather than an input callback or task. -/
noncomputable def gainDelaySeedSequence (genGain memGain eps decay rate amplitude : ℝ) (index : ℕ) : ℝ :=
  Real.sqrt (memGain / genGain) * amplitude / 4 *
    ((1 - rate * decay) / gainGenGrowthFactor 0 eps rate genGain) ^ index

/-- The seed-family geometric ratio lies strictly between zero and
one in the retained native region. Sources: the actual epsilon-floor
envelope at 4f52406; all restrictions concern fixed constants. -/
theorem gain_delay_seed_ratio_interval (genGain eps decay rate : ℝ)
    (hg : 0 < genGain) (he : 0 < eps) (heta : 0 ≤ rate) (hdecay : 0 ≤ decay)
    (hd : 0 < 1 - rate * decay) :
    0 < (1 - rate * decay) / gainGenGrowthFactor 0 eps rate genGain ∧
      (1 - rate * decay) / gainGenGrowthFactor 0 eps rate genGain < 1 := by
  have hr := gain_gen_growth_factor_two_le 0 eps rate genGain (by norm_num) he heta (le_of_lt hg)
  have hrpos : 0 < gainGenGrowthFactor 0 eps rate genGain := by linarith only [hr]
  have hk : 1 - rate * decay ≤ 1 := by nlinarith only [mul_nonneg heta hdecay]
  have hlt : 1 - rate * decay < gainGenGrowthFactor 0 eps rate genGain := by linarith only [hk, hr]
  exact ⟨div_pos hd hrpos, (div_lt_iff₀ hrpos).mpr (by simpa only [one_mul] using hlt)⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) ≤ 1 / 10 ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) := by norm_num

/-- Every member is a strictly positive physical Gen initialization.
Sources: section 3's positive seed ingredient and native envelope at
4f52406; zero Gen is approached but never used by a finite family member. -/
theorem gain_delay_seed_sequence_pos (genGain memGain eps decay rate amplitude : ℝ) (index : ℕ)
    (hg : 0 < genGain) (hm : 0 < memGain) (he : 0 < eps) (heta : 0 ≤ rate)
    (hd : 0 < 1 - rate * decay) (ha : 0 < amplitude) :
    0 < gainDelaySeedSequence genGain memGain eps decay rate amplitude index := by
  have hr := gain_gen_growth_factor_two_le 0 eps rate genGain (by norm_num) he heta (le_of_lt hg)
  have hrpos : 0 < gainGenGrowthFactor 0 eps rate genGain := by linarith only [hr]
  have hs : 0 < Real.sqrt (memGain / genGain) := Real.sqrt_pos.mpr (div_pos hm hg)
  unfold gainDelaySeedSequence
  positivity

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ (0 : ℝ) < 1 := by norm_num

/-- The explicit seed family tends to zero at the fixed constants.
Source: section 3's small Gen initialization and native envelope at
4f52406; the other gain/amplitude need no sign condition for this
numerical limit alone, unlike positivity and task-delay results. -/
theorem gain_delay_seed_sequence_tendsto_zero (genGain memGain eps decay rate amplitude : ℝ)
    (hg : 0 < genGain) (he : 0 < eps) (heta : 0 ≤ rate) (hdecay : 0 ≤ decay)
    (hd : 0 < 1 - rate * decay) :
    Tendsto (gainDelaySeedSequence genGain memGain eps decay rate amplitude) atTop (nhds 0) := by
  have hratio := gain_delay_seed_ratio_interval genGain eps decay rate hg he heta hdecay hd
  have hpow := tendsto_pow_atTop_nhds_zero_of_lt_one (le_of_lt hratio.1) hratio.2
  change Tendsto (fun n => Real.sqrt (memGain / genGain) * amplitude / 4 *
    ((1 - rate * decay) / gainGenGrowthFactor 0 eps rate genGain) ^ n) atTop (nhds 0)
  simpa only [mul_zero] using
    hpow.const_mul (Real.sqrt (memGain / genGain) * amplitude / 4)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) ≤ 1 / 10 ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) := by norm_num

/-- Each member's initial Gen envelope is exactly one quarter of
the Mem pure-decay score at that family's budget. Sources: actual
native envelope and section 3 seed construction at 4f52406;
this identity alone does not assume later allocation or convergence. -/
theorem gain_delay_seed_sequence_envelope (genGain memGain eps decay rate amplitude : ℝ) (index : ℕ)
    (hg : 0 < genGain) (hm : 0 < memGain) (he : 0 < eps) (heta : 0 ≤ rate) :
    genGain * (gainGenGrowthFactor 0 eps rate genGain ^ index *
      (2 * gainDelaySeedSequence genGain memGain eps decay rate amplitude index)) ^ 2 =
        memGain * ((1 - rate * decay) ^ index * amplitude) ^ 2 / 4 := by
  have hr := gain_gen_growth_factor_two_le 0 eps rate genGain (by norm_num) he heta (le_of_lt hg)
  have hrpos : 0 < gainGenGrowthFactor 0 eps rate genGain := by linarith only [hr]
  have hrpow : 0 < gainGenGrowthFactor 0 eps rate genGain ^ index := pow_pos hrpos index
  have hscaled : gainGenGrowthFactor 0 eps rate genGain ^ index *
      (2 * gainDelaySeedSequence genGain memGain eps decay rate amplitude index) =
        Real.sqrt (memGain / genGain) * amplitude / 2 * (1 - rate * decay) ^ index := by
    unfold gainDelaySeedSequence
    rw [div_pow]
    field_simp [ne_of_gt hrpow]
    ring
  rw [hscaled]
  calc
    _ = genGain * (Real.sqrt (memGain / genGain) ^ 2) *
        ((1 - rate * decay) ^ index * amplitude) ^ 2 / 4 := by ring
    _ = _ := by
      rw [Real.sq_sqrt (div_nonneg (le_of_lt hm) (le_of_lt hg))]
      field_simp

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 := by norm_num

/-- The same fixed native task has a uniquely wrong-test prefix
through each positive family's own index. Sources: section 3's
slow-learning ingredient, appendix C tables and native envelope at
4f52406; the entire prefix is generated from initial data. -/
theorem gain_native_delay_seed_sequence_prefix (remaining index : ℕ)
    (genGain memGain bound eps decay rate amplitude : ℝ)
    (hg : 0 < genGain) (hm : 0 < memGain) (hclip : 0 < bound) (he : 0 < eps)
    (heta : 0 ≤ rate) (hdecay : 0 ≤ decay) (hd : 0 < 1 - rate * decay) (ha : 0 < amplitude) :
    let seed := gainDelaySeedSequence genGain memGain eps decay rate amplitude index
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((seed, seed), (amplitude, amplitude)))
    ∀ n, n ≤ index → StrictCorrect (trainTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0 ∧
      StrictCorrect (heldoutTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) (0 : Fin (remaining + 1)).succ := by
  let seed := gainDelaySeedSequence genGain memGain eps decay rate amplitude index
  let initial := seededNativeSubweights ((seed, seed), (amplitude, amplitude))
  have hseed := gain_delay_seed_sequence_pos genGain memGain eps decay rate amplitude index hg hm he heta hd ha
  have hs := native_seeded_nonnegative seed seed amplitude amplitude
    (le_of_lt hseed) (le_of_lt hseed) (le_of_lt ha) (le_of_lt ha)
  have hweight : gainGenGrowthWeight 0 eps rate initial = 2 * seed := by
    change seed + seed + rate / ((1 - 0) * eps) * -(0 + 0) = 2 * seed
    ring
  have hscore := gain_delay_seed_sequence_envelope genGain memGain eps decay rate amplitude index hg hm he heta
  have hmemScore : 0 < memGain * ((1 - rate * decay) ^ index * amplitude) ^ 2 := by positivity
  apply gain_native_train_correct_wrong_test_prefix remaining index genGain memGain bound 0 0 eps decay rate amplitude initial
    hg hm hclip (by norm_num) (by norm_num) (by norm_num) (by norm_num) he heta hdecay hd hs ha le_rfl le_rfl
  rw [hweight]
  change genGain * (gainGenGrowthFactor 0 eps rate genGain ^ index *
    (2 * gainDelaySeedSequence genGain memGain eps decay rate amplitude index)) ^ 2 < _
  rw [hscore]
  linarith only [hmemScore]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) ≤ 1 / 10 ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ (0 : ℝ) < 1 := by norm_num

end Transformer.Grokking.CircuitEfficiency

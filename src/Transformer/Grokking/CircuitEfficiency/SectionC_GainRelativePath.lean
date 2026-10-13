import Transformer.Grokking.CircuitEfficiency.SectionC_GainRelativeStep

/-!
# Initialized actual native relative pair balance without parameter limits

Sources: Varma et al., arXiv:2309.02390v1, section 3's unbalanced
zero-first-factor seeds and appendix C's true product CE; native
feedback and current relative estimates at lab commit d9b30bf.

Start with nonnegative zero-buffer seeds and positive sums in both
pairs. Positive native decay derives one finite physical/buffer box.
Its true clipped CE floor derives a constant relative ratio in [0,1).
Actual pair-mass floors keep both sums positive at every finite clock,
and induction bounds both relative differences by q^n times their
initial values. Both relative differences therefore tend to zero,
and both normalized products approach their fixed-sum maximum,
even if the individual physical parameters have no finite limits.

No future box, callback, balance or successful reference is supplied.
The nonempty sufficient small-rate condition is explicit. Relative
balance is a step toward efficient-circuit competition, not itself
held-out success, a positive limiting margin or CE-loss convergence.
Exact-real fixed tables, zero betas and uniform native decay differ
from appendix C's coupled-cost GD and preserved learned GPTMini.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW Filter

/-- Nonnegative native seeds with positive pair sums generate a finite
box, positive masses, and both full-clock relative-asymmetry ceilings.
Sources: section 3 unbalanced factors and actual native CE at d9b30bf;
all constants and trajectory premises are derived from initial data. -/
theorem gain_native_seeded_relative_path_bound (remaining : ℕ)
    (genGain memGain bound eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain)
    (hclip : 0 < bound) (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    ∃ ceiling : ℝ, 0 < ceiling ∧
      let q := gainNativeRelativeRatio remaining genGain memGain bound ceiling eps decay rate
      let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
        (seededNativeSubweights ((a, b), (c, d)))
      0 ≤ q ∧ q < 1 ∧
      (∀ n, NonnegativeNativeState (path n) ∧ ∀ i,
        (path n i).parameter ≤ ceiling ∧ -(path n i).moment ≤ bound ∧ (path n i).variance ≤ bound ^ 2) ∧
      (∀ n, 0 < (path n 0).parameter + (path n 1).parameter ∧
        0 < (path n 2).parameter + (path n 3).parameter) ∧
      ∀ n, gainPairRelativeDifference (path n 0).parameter (path n 1).parameter ≤
          q ^ n * gainPairRelativeDifference a b ∧
        gainPairRelativeDifference (path n 2).parameter (path n 3).parameter ≤
          q ^ n * gainPairRelativeDifference c d := by
  have hpart := mul_nonneg (le_of_lt heta) (div_nonneg (le_of_lt hgen) (le_of_lt he))
  have hd : 0 ≤ 1 - rate * decay := by nlinarith only [hsmall, hpart]
  obtain ⟨ceiling, hceiling, hbox⟩ := gain_native_seeded_bounded remaining genGain memGain bound 0 0 eps decay rate
    a b c d hgen hmem hclip (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    he hdecay (le_of_lt heta) hd ha hb hc hdseed
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
    (seededNativeSubweights ((a, b), (c, d)))
  let q := gainNativeRelativeRatio remaining genGain memGain bound ceiling eps decay rate
  have hq := gain_native_relative_ratio_interval remaining genGain memGain bound ceiling eps decay rate
    hgen hmem hclip (le_of_lt hceiling) he heta hd
  have hparam : ∀ n i, 0 ≤ (path n i).parameter ∧ (path n i).parameter ≤ ceiling := by
    intro n i
    exact ⟨((hbox n).1 i).1, ((hbox n).2 i).1⟩
  have hl := gain_ce_feedback_floor_pos remaining genGain memGain bound ceiling
    (le_of_lt hgen) (le_of_lt hceiling) hclip
  have hmult := gain_pair_mass_multiplier_interval _ memGain eps decay rate ceiling
    hl hmem he heta (le_of_lt hceiling) hd
  have hmass : ∀ n, 0 < (path n 0).parameter + (path n 1).parameter ∧
      0 < (path n 2).parameter + (path n 3).parameter := by
    intro n
    induction n with
    | zero => exact ⟨hgMass, hmMass⟩
    | succ n ih =>
      have hf := gain_native_pair_mass_floor remaining genGain memGain bound ceiling eps decay rate (path n)
        hgen hmem hgain hclip he (le_of_lt heta) (hparam n)
      exact ⟨lt_of_lt_of_le (mul_pos hmult.1 ih.1) hf.1,
        lt_of_lt_of_le (mul_pos hmult.1 ih.2) hf.2⟩
  have hpowers : ∀ n, gainPairRelativeDifference (path n 0).parameter (path n 1).parameter ≤
        q ^ n * gainPairRelativeDifference a b ∧
      gainPairRelativeDifference (path n 2).parameter (path n 3).parameter ≤
        q ^ n * gainPairRelativeDifference c d := by
    intro n
    induction n with
    | zero =>
      change gainPairRelativeDifference a b ≤ q ^ 0 * gainPairRelativeDifference a b ∧
        gainPairRelativeDifference c d ≤ q ^ 0 * gainPairRelativeDifference c d
      simp only [pow_zero, one_mul, le_refl, and_self]
    | succ n ih =>
      have hstep := gain_native_relative_difference_step remaining genGain memGain bound ceiling eps decay rate (path n)
        hgen hmem hgain hclip he heta (hparam n) (hmass n).1 (hmass n).2 hsmall
      have hg := le_trans hstep.1 (mul_le_mul_of_nonneg_left ih.1 hq.1)
      have hm := le_trans hstep.2 (mul_le_mul_of_nonneg_left ih.2 hq.1)
      constructor
      · simpa only [path, gainNativePath, pow_succ, mul_assoc, mul_comm, mul_left_comm] using hg
      · simpa only [path, gainNativePath, pow_succ, mul_assoc, mul_comm, mul_left_comm] using hm
  exact ⟨ceiling, hceiling, hq.1, hq.2, hbox, hmass, hpowers⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 / 200 ∧ (0 : ℝ) < 0 + 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by
  norm_num

/-- Source-style unbalanced positive-sum seeds approach relative pair
balance on the actual native path without individual parameter limits.
Sources: section 3's zero-first-factor seeds and true feedback at
d9b30bf; no future lower mass, box, coefficient or balance is supplied. -/
theorem gain_native_seeded_relative_difference_tendsto_zero (remaining : ℕ)
    (genGain memGain bound eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain)
    (hclip : 0 < bound) (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, b), (c, d)))
    Tendsto (fun n => gainPairRelativeDifference (path n 0).parameter (path n 1).parameter) atTop (nhds 0) ∧
      Tendsto (fun n => gainPairRelativeDifference (path n 2).parameter (path n 3).parameter) atTop (nhds 0) := by
  obtain ⟨ceiling, _, hq0, hq1, _, hmass, hpowers⟩ := gain_native_seeded_relative_path_bound remaining
    genGain memGain bound eps decay rate a b c d hgen hmem hgain hclip he hdecay heta ha hb hc hdseed hgMass hmMass hsmall
  have hpow := tendsto_pow_atTop_nhds_zero_of_lt_one hq0 hq1
  constructor
  · apply squeeze_zero (fun n => div_nonneg (abs_nonneg _) (le_of_lt (hmass n).1)) (fun n => (hpowers n).1)
    simpa using hpow.mul_const (gainPairRelativeDifference a b)
  · apply squeeze_zero (fun n => div_nonneg (abs_nonneg _) (le_of_lt (hmass n).2)) (fun n => (hpowers n).2)
    simpa using hpow.mul_const (gainPairRelativeDifference c d)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 / 200 ∧ (0 : ℝ) < 0 + 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by
  norm_num

/-- Product strength relative to its fixed-sum maximum is exactly one
minus squared relative asymmetry. Source: appendix C's product forward
and mass comparison at d9b30bf; the nonzero sum premise is explicit. -/
theorem gain_pair_relative_product (a b : ℝ) (hmass : a + b ≠ 0) :
    4 * a * b / (a + b) ^ 2 = 1 - gainPairRelativeDifference a b ^ 2 := by
  unfold gainPairRelativeDifference
  field_simp [hmass]
  rw [sq_abs]
  ring

example : (0 : ℝ) + 1 ≠ 0 := by norm_num

/-- Both actual pair products approach their fixed-sum maximum in
relative strength. Sources: appendix C's product forward and generated
relative balance at d9b30bf; this normalization does not assert positive
limiting product, individual parameter convergence or held-out success. -/
theorem gain_native_seeded_relative_products_tendsto_one (remaining : ℕ)
    (genGain memGain bound eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain)
    (hclip : 0 < bound) (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, b), (c, d)))
    Tendsto (fun n => 4 * (path n 0).parameter * (path n 1).parameter /
      ((path n 0).parameter + (path n 1).parameter) ^ 2) atTop (nhds 1) ∧
      Tendsto (fun n => 4 * (path n 2).parameter * (path n 3).parameter /
        ((path n 2).parameter + (path n 3).parameter) ^ 2) atTop (nhds 1) := by
  obtain ⟨_, _, _, _, _, hmass, _⟩ := gain_native_seeded_relative_path_bound remaining genGain memGain bound eps decay rate
    a b c d hgen hmem hgain hclip he hdecay heta ha hb hc hdseed hgMass hmMass hsmall
  have hbalance := gain_native_seeded_relative_difference_tendsto_zero remaining genGain memGain bound eps decay rate
    a b c d hgen hmem hgain hclip he hdecay heta ha hb hc hdseed hgMass hmMass hsmall
  have hg : Tendsto (fun n => 1 - gainPairRelativeDifference
      (gainNativePath remaining genGain memGain bound 0 0 eps decay rate (seededNativeSubweights ((a, b), (c, d))) n 0).parameter
      (gainNativePath remaining genGain memGain bound 0 0 eps decay rate (seededNativeSubweights ((a, b), (c, d))) n 1).parameter ^ 2)
      atTop (nhds 1) := by simpa using (hbalance.1.pow 2).const_sub 1
  have hm := (hbalance.2.pow 2).const_sub (1 : ℝ)
  constructor
  · exact hg.congr (fun n => (gain_pair_relative_product _ _ (ne_of_gt (hmass n).1)).symm)
  · simpa using hm.congr (fun n => (gain_pair_relative_product _ _ (ne_of_gt (hmass n).2)).symm)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 / 200 ∧ (0 : ℝ) < 0 + 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by
  norm_num

end Transformer.Grokking.CircuitEfficiency

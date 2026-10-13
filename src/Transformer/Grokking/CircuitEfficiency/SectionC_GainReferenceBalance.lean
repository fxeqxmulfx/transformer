import Transformer.Grokking.CircuitEfficiency.SectionC_GainNoncollapse

/-!
# Nonnegative native reference pairs and efficient feedback threshold

Sources: Varma et al., arXiv:2309.02390v1, section 3 Efficiency
and appendix C product CE; native balance at lab commit c5b8413.
Classify each nonnegative two-factor normalized balance: both
factors are zero, or both are positive and equal at positive decay.
The classification does not exclude the cold boundary by itself.

At an actual gained-CE reference with a positive Mem factor,
Mem balance forces positive decay and decay * epsilon below
the actual Mem feedback coefficient. Gen's larger physical gain
then puts its coefficient strictly above that threshold. No
future successful Gen amplitude or held-out margin is assumed.
Positive Gen mass and one positive Mem factor also suffice for
positivity of all four balanced physical coordinates.

Reference balance is a point property, obtained on actual paths
only from previously derived necessary finite-limit laws. This
leaf alone supplies no attraction, convergence or delayed crossing.
Uniform decoupled native decay/plain CE remain distinct from the
source's GD/coupled cost and from learned stochastic GPTMini.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- A nonnegative normalized partner balance is either fully cold
or a positive equal-factor pair. Sources: appendix C pair partials
and native normalized balance at c5b8413; cold pairs are retained. -/
theorem native_nonnegative_pair_balance (scale eps decay first second : ℝ)
    (hs : 0 < scale) (he : 0 < eps) (hf : 0 ≤ first) (ht : 0 ≤ second)
    (hfirst : decay * first = scale * second / (scale * second + eps))
    (hsecond : decay * second = scale * first / (scale * first + eps)) :
    (first = 0 ∧ second = 0) ∨
      (0 < decay ∧ 0 < first ∧ 0 < second ∧ first = second ∧ decay * (scale * first + eps) = scale) := by
  by_cases hz : first = 0
  · left
    have hd : scale * second + eps ≠ 0 := by positivity
    rw [hz, mul_zero] at hfirst
    have hprod := (div_eq_zero_iff.mp hfirst.symm).resolve_right hd
    exact ⟨hz, (mul_eq_zero.mp hprod).resolve_left (ne_of_gt hs)⟩
  · right
    have hfp : 0 < first := lt_of_le_of_ne hf (Ne.symm hz)
    have hp : 0 < decay * second := by
      rw [hsecond]
      exact div_pos (mul_pos hs hfp) (by positivity)
    have hd : 0 < decay := pos_of_mul_pos_left hp ht
    have htp : 0 < second := (mul_pos_iff_of_pos_left hd).mp hp
    have hb := native_positive_pair_balance scale eps decay first second hs he hfp htp hfirst hsecond
    exact ⟨hb.1, hfp, htp, hb.2.1, hb.2.2⟩

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (1 / 2 : ℝ) * 1 = 1 * 1 / (1 * 1 + 1) ∧
    (1 / 2 : ℝ) * 1 = 1 * 1 / (1 * 1 + 1) := by norm_num

/-- Actual nonnegative gained balance has the positive-partner
fraction form, including zero partners. Sources: appendix C true
CE and native normalization at c5b8413; reference buffers are irrelevant. -/
theorem gain_native_nonnegative_partner_balance (remaining : ℕ) (genGain memGain bound eps decay : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hp : ∀ i, 0 ≤ (state i).parameter)
    (hb : ∀ i, decay * (state i).parameter + appliedGainNativeGradient remaining genGain memGain bound state i /
      (|appliedGainNativeGradient remaining genGain memGain bound state i| + eps) = 0) :
    ∀ i, decay * (state i).parameter =
      (gainCEGradientScale remaining genGain memGain bound state * nativeFactorGain genGain memGain i) *
        (state (nativeFactorPartner i)).parameter /
      ((gainCEGradientScale remaining genGain memGain bound state * nativeFactorGain genGain memGain i) *
        (state (nativeFactorPartner i)).parameter + eps) := by
  intro i
  have hi := hb i
  have hg := gain_native_applied_gradient_nonpos remaining genGain memGain bound state i hclip
    (le_of_lt (gain_native_factor_gain_pos genGain memGain hgen hmem i)) (hp _)
  rw [abs_of_nonpos hg, gain_native_applied_gradient_scale, neg_neg, neg_div] at hi
  linarith only [hi]

example :
    let state := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 state
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (∀ i, 0 ≤ (state i).parameter) ∧
    (∀ i, (1 / 2 : ℝ) * (state i).parameter + appliedGainNativeGradient 111 3 2 1 state i /
      (|appliedGainNativeGradient 111 3 2 1 state i| + eps) = 0) := by
  dsimp only
  refine ⟨by norm_num, by norm_num, by norm_num, ?_, (gain_native_efficient_point_balance 111 1 (by norm_num)).2⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- Positive Mem balance places the actual efficient Gen coefficient
above decay * epsilon. Sources: section 3 Efficiency and appendix C
normalized partner equations at c5b8413; Gen positivity is not a premise. -/
theorem gain_native_mem_live_feedback_threshold (remaining : ℕ) (genGain memGain bound eps decay : ℝ)
    (state : NativeSubweightState) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hclip : 0 < bound) (he : 0 < eps) (hp : ∀ i, 0 ≤ (state i).parameter) (hm : 0 < (state 2).parameter)
    (hb : ∀ i, decay * (state i).parameter + appliedGainNativeGradient remaining genGain memGain bound state i /
      (|appliedGainNativeGradient remaining genGain memGain bound state i| + eps) = 0) :
    0 < decay ∧ 0 < (state 3).parameter ∧
      decay * eps < gainCEGradientScale remaining genGain memGain bound state * genGain := by
  let scale := gainCEGradientScale remaining genGain memGain bound state
  have hs : 0 < scale := gain_ce_gradient_scale_pos remaining genGain memGain bound state hclip
  have hn := gain_native_nonnegative_partner_balance remaining genGain memGain bound eps decay state
    (lt_trans hmem hgain) hmem hclip hp hb
  have hpair := native_nonnegative_pair_balance (scale * memGain) eps decay (state 2).parameter (state 3).parameter
    (mul_pos hs hmem) he (hp 2) (hp 3)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 2)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 3)
  rcases hpair with hz | ⟨hd, hp2, hp3, _, hbalance⟩
  · exfalso
    linarith [hz.1]
  · have hpositive := mul_pos (mul_pos hd (mul_pos hs hmem)) hp2
    have hgap : decay * eps < scale * memGain := by nlinarith only [hbalance, hpositive]
    exact ⟨hd, hp3, lt_trans hgap (mul_lt_mul_of_pos_left hgain hs)⟩

example :
    let state := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 state
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ 0 < eps ∧
    (∀ i, 0 ≤ (state i).parameter) ∧ 0 < (state 2).parameter ∧
    (∀ i, (1 / 2 : ℝ) * (state i).parameter + appliedGainNativeGradient 111 3 2 1 state i /
      (|appliedGainNativeGradient 111 3 2 1 state i| + eps) = 0) := by
  dsimp only
  have hw := gain_native_efficient_point_balance 111 1 (by norm_num)
  refine ⟨by norm_num, by norm_num, by norm_num, hw.1, ?_, by norm_num [seededNativeSubweights, seededScalarState], hw.2⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- Positive Gen mass and one positive Mem factor force all balanced
coordinates positive. Sources: appendix C partners and native balance
at c5b8413; neither pair positivity nor a successful margin is supplied. -/
theorem gain_native_live_pairs_positive (remaining : ℕ) (genGain memGain bound eps decay : ℝ)
    (state : NativeSubweightState) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hclip : 0 < bound) (he : 0 < eps) (hp : ∀ i, 0 ≤ (state i).parameter)
    (hg : 0 < gainGenParameterMass state) (hm : 0 < (state 2).parameter)
    (hb : ∀ i, decay * (state i).parameter + appliedGainNativeGradient remaining genGain memGain bound state i /
      (|appliedGainNativeGradient remaining genGain memGain bound state i| + eps) = 0) :
    ∀ i, 0 < (state i).parameter := by
  have hn := gain_native_nonnegative_partner_balance remaining genGain memGain bound eps decay state
    (lt_trans hmem hgain) hmem hclip hp hb
  have hs := gain_ce_gradient_scale_pos remaining genGain memGain bound state hclip
  have hpair := native_nonnegative_pair_balance
    (gainCEGradientScale remaining genGain memGain bound state * genGain) eps decay (state 0).parameter (state 1).parameter
    (mul_pos hs (lt_trans hmem hgain)) he (hp 0) (hp 1)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 0)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 1)
  have hmem3 := (gain_native_mem_live_feedback_threshold remaining genGain memGain bound eps decay
    state hmem hgain hclip he hp hm hb).2.1
  rcases hpair with hz | ⟨_, hp0, hp1, _, _⟩
  · exfalso
    unfold gainGenParameterMass at hg
    linarith [hz.1, hz.2]
  · intro i
    fin_cases i
    · exact hp0
    · exact hp1
    · exact hm
    · exact hmem3

example :
    let state := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 state
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ 0 < eps ∧ (∀ i, 0 ≤ (state i).parameter) ∧
    0 < gainGenParameterMass state ∧ 0 < (state 2).parameter ∧
    (∀ i, (1 / 2 : ℝ) * (state i).parameter + appliedGainNativeGradient 111 3 2 1 state i /
      (|appliedGainNativeGradient 111 3 2 1 state i| + eps) = 0) := by
  dsimp only
  have hw := gain_native_efficient_point_balance 111 1 (by norm_num)
  refine ⟨by norm_num, by norm_num, by norm_num, hw.1, ?_,
    by norm_num [gainGenParameterMass, seededNativeSubweights, seededScalarState],
    by norm_num [seededNativeSubweights, seededScalarState], hw.2⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

end Transformer.Grokking.CircuitEfficiency

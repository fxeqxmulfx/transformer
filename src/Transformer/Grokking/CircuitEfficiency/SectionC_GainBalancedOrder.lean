import Transformer.Grokking.CircuitEfficiency.SectionC_GainBalancedEscape

/-!
# Persistent balanced Gen factor dominance under actual native feedback

Sources: Varma et al., arXiv:2309.02390v1, section 3's efficient
circuits and appendix C product CE; native AdamW at lab commit
43155e5. Prove strict monotonicity of the instantaneous native
normalization x/(x+epsilon), then compare Gen and Mem updates
when Gen is already at least as strong in factor amplitude.

The actual full CE scale is shared and positive. Greater Gen gain
therefore preserves strict amplitude dominance at the next native
step with nonnegative remaining decay and positive rate/epsilon.
Join this with the previously derived finite crossing: from positive
balanced seeds the same actual initialized path is permanently
Gen-dominant after a finite update, without parameter convergence.

This is a legal zero-beta, fixed gained-table specialization, with
equal positive factors within each pair. It differs from appendix C's
coupled assigned norm/GD and first-zero-factor seeds, and from learned
nonzero-beta GPTMini. Only ordering is proved; no finite parameter
limit or quantitative observed transition time is asserted.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Native instantaneous normalization is strictly increasing on
nonnegative magnitudes at positive epsilon. Source: the actual
zero-beta parameter formula at 43155e5, not a softmax surrogate. -/
theorem gain_native_normalized_input_strict (eps x y : ℝ)
    (he : 0 < eps) (hx : 0 ≤ x) (hxy : x < y) :
    x / (x + eps) < y / (y + eps) := by
  have hy : 0 < y := lt_of_le_of_lt hx hxy
  have hdx : 0 < x + eps := by positivity
  have hdy : 0 < y + eps := by positivity
  apply (div_lt_div_iff₀ hdx hdy).mpr
  have hm := mul_pos he (show 0 < y - x by linarith only [hxy])
  nlinarith only [hm]

example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) < 2 := by
  norm_num

/-- Greater physical gain preserves already-ordered positive pair
amplitudes at a common current CE scale. Sources: appendix C's
gained products and the native zero-beta update at 43155e5. -/
theorem gain_balanced_factor_order_step (scale eps genGain memGain decay rate a b : ℝ)
    (hs : 0 < scale) (he : 0 < eps) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (heta : 0 < rate) (hd : 0 ≤ 1 - rate * decay) (hb : 0 < b) (hba : b ≤ a) :
    gainBalancedFactorStep memGain scale eps decay rate b <
      gainBalancedFactorStep genGain scale eps decay rate a := by
  have hg : 0 < genGain := lt_trans hmem hgain
  have hgainScale := mul_lt_mul_of_pos_left hgain hs
  have hfirst := mul_lt_mul_of_pos_right hgainScale hb
  have hsecond := mul_le_mul_of_nonneg_left hba (le_of_lt (mul_pos hs hg))
  have hinput : scale * memGain * b < scale * genGain * a := lt_of_lt_of_le hfirst hsecond
  have hnormal := gain_native_normalized_input_strict eps (scale * memGain * b) (scale * genGain * a)
    he (le_of_lt (mul_pos (mul_pos hs hmem) hb)) hinput
  have hscaled := mul_lt_mul_of_pos_left hnormal heta
  have hdecayed := mul_le_mul_of_nonneg_left hba hd
  have hmemGroup : rate * (scale * memGain * b) / (scale * memGain * b + eps) =
      rate * (scale * memGain * b / (scale * memGain * b + eps)) := by ring
  have hgenGroup : rate * (scale * genGain * a) / (scale * genGain * a + eps) =
      rate * (scale * genGain * a / (scale * genGain * a + eps)) := by ring
  unfold gainBalancedFactorStep
  rw [hmemGroup, hgenGroup]
  exact add_lt_add_of_le_of_lt hdecayed hscaled

example : (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 := by
  norm_num

/-- The actual native step keeps ordered balanced amplitudes strictly
Gen-dominant. Sources: appendix C true CE and native AdamW at 43155e5;
derive the positive current scale and all physical signs from the point. -/
theorem gain_native_balanced_order_step (remaining : ℕ)
    (genGain memGain bound eps decay rate : ℝ)
    (state : NativeSubweightState) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hclip : 0 < bound) (he : 0 < eps) (heta : 0 < rate) (hd : 0 ≤ 1 - rate * decay)
    (hs : NativePairSymmetry state) (hb : 0 < (state 2).parameter)
    (horder : (state 2).parameter ≤ (state 0).parameter) :
    (gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state 2).parameter <
      (gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state 0).parameter := by
  have hg : 0 < genGain := lt_trans hmem hgain
  have hgp : 0 < (state 0).parameter := lt_of_lt_of_le hb horder
  have hp : ∀ i, 0 ≤ (state i).parameter := by
    intro i
    fin_cases i
    · exact le_of_lt hgp
    · change 0 ≤ (state 1).parameter
      rw [← hs.1]
      exact le_of_lt hgp
    · exact le_of_lt hb
    · change 0 ≤ (state 3).parameter
      rw [← hs.2]
      exact le_of_lt hb
  have hstep := gain_native_zero_beta_balanced_step remaining genGain memGain bound eps decay rate
    state hg hmem hclip hp hs
  rw [hstep.1, hstep.2]
  exact gain_balanced_factor_order_step _ eps genGain memGain decay rate _ _
    (gain_ce_gradient_scale_pos remaining genGain memGain bound state hclip) he hmem hgain heta hd hb horder

example :
    let state := seededNativeSubweights ((1, 1), (1, 1))
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧
    0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ NativePairSymmetry state ∧
    0 < (state 2).parameter ∧ (state 2).parameter ≤ (state 0).parameter := by
  dsimp only
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ⟨rfl, rfl⟩,
    by norm_num [seededNativeSubweights, seededScalarState], by norm_num [seededNativeSubweights, seededScalarState]⟩

/-- Actual initialized balanced native paths are permanently factor
Gen-dominant after a finite clock. Sources: appendix C's competing
gained products and native AdamW at 43155e5; finite crossing and
its subsequent preservation are both derived from initial constants. -/
theorem gain_native_balanced_order_tail (remaining : ℕ) (genGain memGain bound eps decay rate a b : ℝ)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (ha : 0 < a) (hb : 0 < b) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, a), (b, b)))
    ∃ start, ∀ n, start ≤ n → (path n 2).parameter < (path n 0).parameter := by
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
    (seededNativeSubweights ((a, a), (b, b)))
  obtain ⟨start, hcross⟩ := gain_native_balanced_finite_crossing remaining genGain memGain bound eps decay rate a b
    hmem hgain hclip he hdecay heta hd ha hb
  have hg : 0 < genGain := lt_trans hmem hgain
  have hpos : ∀ n, 0 < (path n 2).parameter := by
    intro n
    exact (gain_native_seeded_balanced_positive remaining genGain memGain bound eps decay rate a b n 2
      hg hmem hclip he (le_of_lt heta) hd ha hb).1
  have htail : ∀ k, (path (start + k) 2).parameter < (path (start + k) 0).parameter := by
    intro k
    induction k with
    | zero => simpa only [Nat.add_zero] using hcross
    | succ k ih =>
      rw [show start + (k + 1) = (start + k) + 1 by omega]
      exact gain_native_balanced_order_step remaining genGain memGain bound eps decay rate (path (start + k))
        hmem hgain hclip he heta (le_of_lt hd)
        (gain_native_equal_pair_path remaining genGain memGain bound 0 0 eps decay rate a b (start + k))
        (hpos (start + k)) (le_of_lt ih)
  refine ⟨start, ?_⟩
  intro n hn
  have hindex : start + (n - start) = n := by omega
  simpa only [hindex] using htail (n - start)

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 10 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by
  norm_num

end Transformer.Grokking.CircuitEfficiency

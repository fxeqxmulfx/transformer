import Transformer.Grokking.CircuitEfficiency.SectionC_GainMassRates

/-!
# Strong native decay bounds every actual pair mass by a contraction

Sources: Varma et al., arXiv:2309.02390v1, section 3's CE/decay
competition and appendix C's product partials; original clipped
native zero-beta mass laws at lab commit 4e9829f.

The actual instantaneous CE multiplier lies in (0,1). Positive
epsilon therefore bounds each unequal pair's next mass above by
its current mass times remaining decay plus rate times gain/epsilon.
This estimate also holds at zero pair mass; no division by that
mass, supplied successful reference or future parameter limit is used.

If decay exceeds the larger gain/epsilon and remaining decay is
nonnegative, the common multiplier lies strictly between zero and
one. Both actual Gen/Mem mass updates have this common ceiling.
Clipping and the original shared CE are retained rather than replaced
by a prescribed input stream or a balanced feedback trajectory.

This prepares an initialized absolute-mass contraction alongside the
already-derived relative Gen selection. Strong absolute decay need
not obstruct exact-real strict decisions; positive confidence and
CE-loss improvement are distinct requirements. These current bounds
alone assert neither a path limit nor successful generalization.

Fixed gained tables, zero betas and uniform native decoupled decay
differ from appendix C's assigned coupled norm/GD and the preserved
learned nonzero-beta GPTMini. No floating-point transfer is claimed.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- Uniform instantaneous pair-mass multiplier ceiling. Sources:
appendix C's partner partials and native normalization at 4e9829f;
all fixed numerical arguments occur in the expression. -/
noncomputable def gainPairMassCeiling (gain eps decay rate : ℝ) : ℝ :=
  1 - rate * decay + rate * gain / eps

/-- Strong positive native decay makes the explicit mass ceiling a
strict contraction. Sources: section 3 CE/decay competition and the
actual epsilon normalization at 4e9829f; only static numerical
conditions are assumed, without a future parameter bound or limit. -/
theorem gain_pair_mass_ceiling_strong_decay (gain eps decay rate : ℝ)
    (hg : 0 < gain) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hstrong : gain / eps < decay) :
    0 < gainPairMassCeiling gain eps decay rate ∧ gainPairMassCeiling gain eps decay rate < 1 := by
  have hinput := div_pos (mul_pos heta hg) he
  have hdiff : 0 < decay - gain / eps := by linarith only [hstrong]
  have hscaled := mul_pos heta hdiff
  have hscaledEq : rate * (decay - gain / eps) = rate * decay - rate * gain / eps := by ring
  rw [hscaledEq] at hscaled
  unfold gainPairMassCeiling
  constructor
  · linarith only [hd, hinput]
  · linarith only [hscaled]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧
    0 ≤ 1 - (1 / 1000 : ℝ) * 10 ∧ (3 : ℝ) / 1 < 10 := by norm_num

/-- The pair-mass ceiling is monotone in physical gain. Sources:
section 3 efficiency and native epsilon normalization at 4e9829f;
the decay term remains common to both circuits. -/
theorem gain_pair_mass_ceiling_gain_monotone (genGain memGain eps decay rate : ℝ)
    (hgain : memGain ≤ genGain) (he : 0 < eps) (heta : 0 ≤ rate) :
    gainPairMassCeiling memGain eps decay rate ≤ gainPairMassCeiling genGain eps decay rate := by
  have hw := mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right hgain (le_of_lt he)) heta
  have hm : rate * (memGain / eps) = rate * memGain / eps := by ring
  have hg : rate * (genGain / eps) = rate * genGain / eps := by ring
  rw [hm, hg] at hw
  unfold gainPairMassCeiling
  linarith only [hw]

example : (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 := by norm_num

/-- Every nonnegative unequal pair, including a zero-mass pair, has
the fixed epsilon mass multiplier ceiling. Sources: appendix C
product inputs and exact same-scale native correction at 4e9829f;
no positive current/limiting mass or future feedback is assumed. -/
theorem gain_pair_step_mass_ceiling (gain scale eps decay rate a b : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (hu : scale ≤ 1) (he : 0 < eps)
    (heta : 0 ≤ rate) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    (gainPairStep gain scale eps decay rate a b).1 + (gainPairStep gain scale eps decay rate a b).2 ≤
      gainPairMassCeiling gain eps decay rate * (a + b) := by
  have hmass : 0 ≤ a + b := add_nonneg ha hb
  have hmean : 0 ≤ (a + b) / 2 := by positivity
  have hr := gain_balanced_relative_rate_ceiling gain scale eps ((a + b) / 2) hg hs hu he hmean
  have hw := mul_le_mul_of_nonneg_left hr (mul_nonneg heta hmass)
  have hgroup : rate * (a + b) * (gain / eps) = rate * gain / eps * (a + b) := by ring
  rw [hgroup] at hw
  have hew := mul_nonneg heta (gain_pair_balancing_error_nonneg gain scale eps a b hg hs he ha hb)
  rw [gain_pair_step_mass_multiplier gain scale eps decay rate a b hg hs he ha hb]
  unfold gainPairMassCeiling
  nlinarith only [hw, hew]

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 := by norm_num

/-- Both original actual native pair mass updates have the common
larger-gain epsilon ceiling. Sources: appendix C true gradients and
retained native feedback at 4e9829f; the present shared clipped CE
scale is derived and neither pair needs positive current mass. -/
theorem gain_native_pair_mass_ceiling (remaining : ℕ) (genGain memGain bound eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (he : 0 < eps) (heta : 0 ≤ rate)
    (hp : ∀ i, 0 ≤ (state i).parameter) :
    let next := gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state
    (next 0).parameter + (next 1).parameter ≤
        gainPairMassCeiling genGain eps decay rate * ((state 0).parameter + (state 1).parameter) ∧
      (next 2).parameter + (next 3).parameter ≤
        gainPairMassCeiling genGain eps decay rate * ((state 2).parameter + (state 3).parameter) := by
  have hs := gain_ce_gradient_scale_unit_interval remaining genGain memGain bound state hclip
  have hform := gain_native_zero_beta_pair_step remaining genGain memGain bound eps decay rate state hgen hmem hclip hp
  have hg := gain_pair_step_mass_ceiling genGain _ eps decay rate _ _
    (le_of_lt hgen) (le_of_lt hs.1) (le_of_lt hs.2) he heta (hp 0) (hp 1)
  have hm := gain_pair_step_mass_ceiling memGain _ eps decay rate _ _
    (le_of_lt hmem) (le_of_lt hs.1) (le_of_lt hs.2) he heta (hp 2) (hp 3)
  have hmGain := gain_pair_mass_ceiling_gain_monotone genGain memGain eps decay rate hgain he heta
  have hmMass := mul_le_mul_of_nonneg_right hmGain (add_nonneg (hp 2) (hp 3))
  rw [← hform.1] at hg
  rw [← hform.2] at hm
  exact ⟨hg, le_trans hm hmMass⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ ∀ i : Fin 4,
      0 ≤ (seededNativeSubweights ((0, 1 / 200), (1, 1)) i).parameter := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, Transformer.Grokking.AdamW.seededScalarState]

/-- Each actual next coordinate is below the common ceiling times
its current pair mass. Sources: appendix C partner partials and native
sign/mass laws at 4e9829f; full current retained sign data generate
next coordinate signs, with no future sign or limit hypothesis. -/
theorem gain_native_coordinate_mass_ceiling (remaining : ℕ) (genGain memGain bound eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (he : 0 < eps) (heta : 0 ≤ rate)
    (hd : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState state) :
    let next := gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state
    let q := gainPairMassCeiling genGain eps decay rate
    (next 0).parameter ≤ q * ((state 0).parameter + (state 1).parameter) ∧
      (next 1).parameter ≤ q * ((state 0).parameter + (state 1).parameter) ∧
      (next 2).parameter ≤ q * ((state 2).parameter + (state 3).parameter) ∧
      (next 3).parameter ≤ q * ((state 2).parameter + (state 3).parameter) := by
  have hn := gain_native_nonnegative_step remaining genGain memGain bound 0 0 eps decay rate state
    hgen hmem hclip (by norm_num) (by norm_num) (by norm_num) (by norm_num) he heta hd hs
  have hm := gain_native_pair_mass_ceiling remaining genGain memGain bound eps decay rate state
    hgen hmem hgain hclip he heta (fun i => (hs i).1)
  refine ⟨?_, ?_, ?_, ?_⟩
  · linarith only [hm.1, (hn 1).1]
  · linarith only [hm.1, (hn 0).1]
  · linarith only [hm.2, (hn 3).1]
  · linarith only [hm.2, (hn 2).1]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 10 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.CircuitEfficiency

import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdCeiling
import Transformer.Grokking.CircuitEfficiency.SectionC_GainFeedbackFloor
import Transformer.Grokking.AdamW.ScalarBiasError
import Transformer.Grokking.AdamW.GeometricError

/-!
# An actual weighted native observer at the cold threshold

Sources: Varma et al., arXiv:2309.02390v1, section 3 CE/decay
competition and appendix C product partials; actual cold feedback
ceiling at 696630b, scalar clock-error law at 34e9a6f and geometric
correction at 26ad2c9. Native AdamW keeps both buffers and clocks.

Use (1-beta1)*epsilon times total physical mass plus rate*beta1
times total negative first-moment mass. At or above the actual cold
threshold, the true four-coordinate update admits a geometric error
budget from its retained numerator bound and completed clock.

For standard nonnegative physical seeds, clipping, native sign
induction and the completed-clock law generate all required current
bounds. The actual weighted observer has a finite nonnegative limit
for arbitrary legal first/second betas, including beta1=0.9/beta2=0.98.
No future parameter, gradient, variance or denominator limit is assumed.

This observer limit need not yet be zero, nor does it by itself
prove physical coordinate convergence. The next step must combine
the actual CE-generated large-input floor with retained adaptive
dissipation to exclude recurrent positive physical mass.

The rate is allowed to be zero in this observer result. Positive
rate will be necessary for a collapse theorem. Fixed gained tables,
plain CE and uniform decoupled native decay differ from appendix C's
assigned coupled norm-cost GD. Exact-real proofs do not establish
learned GPTMini, numerical kernels or thermodynamic size scaling.
Frozen transformer optimizers and checkpoints remain unchanged.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Numerical weighted mass of actual parameters and retained first
moments. Sources: native clock-error weight at 34e9a6f and appendix C's
four gained factors; no future stability is encoded in the value. -/
def coldGainNativeWeightedMass (b1 eps rate : ℝ) (state : NativeSubweightState) : ℝ :=
  ((1 - b1) * eps) * gainNativeParameterMass state + rate * b1 * gainNativeNegativeMomentMass state

/-- Native numerical signs give nonnegative weighted mass. Sources:
appendix C's factors and retained mass signs at ada36f3; the zero
observer is permitted without assuming successful classification. -/
theorem cold_gain_native_weighted_mass_nonnegative (b1 eps rate : ℝ) (state : NativeSubweightState)
    (hb1 : 0 ≤ b1) (h1 : b1 ≤ 1) (he : 0 ≤ eps) (heta : 0 ≤ rate)
    (hs : NonnegativeNativeState state) : 0 ≤ coldGainNativeWeightedMass b1 eps rate state := by
  have hm := gain_native_masses_nonnegative state hs
  unfold coldGainNativeWeightedMass
  exact add_nonneg (mul_nonneg (mul_nonneg (by linarith only [h1]) he) hm.1)
    (mul_nonneg (mul_nonneg heta hb1) hm.2.1)

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- The actual weighted update admits a geometric error budget at
or above the cold threshold. Sources: appendix C true CE, native
cold ceiling at 696630b and scalar weight at 34e9a6f; present moment
bounds and a common completed clock are explicit current hypotheses. -/
theorem gain_native_cold_weighted_mass_step (remaining clock : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (state : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (hs : NonnegativeNativeState state) (hbound : ∀ i, -(state i).moment ≤ bound)
    (hclock : ∀ i, (state i).clock = clock)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    coldGainNativeWeightedMass b1 eps rate (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) ≤
      coldGainNativeWeightedMass b1 eps rate state + (4 * rate * bound) * b1 ^ clock := by
  have hcoordinate : ∀ i, let next := gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state i
      ((1 - b1) * eps) * next.parameter + rate * b1 * (-next.moment) ≤
        ((1 - b1) * eps) * (1 - rate * decay) * (state i).parameter +
          rate * (-next.moment) + rate * bound * b1 ^ clock := by
    intro i
    have hg := gain_native_applied_gradient_nonpos remaining genGain memGain bound state i hclip
      (le_of_lt (gain_native_factor_gain_pos genGain memGain hgen hmem i)) (hs _).1
    have hmag := gain_native_applied_gradient_abs_bound remaining genGain memGain bound state i hclip
    rw [abs_of_nonpos hg] at hmag
    have hold := mul_le_mul_of_nonneg_left (hbound i) hb1
    have hnew := mul_le_mul_of_nonneg_left hmag (show 0 ≤ 1 - b1 by linarith only [h1])
    have hnext : -(scalarNativeStep b1 b2 eps decay rate (state i)
        (appliedGainNativeGradient remaining genGain memGain bound state i)).moment ≤ bound := by
      change -(b1 * (state i).moment + (1 - b1) * appliedGainNativeGradient remaining genGain memGain bound state i) ≤ bound
      nlinarith only [hold, hnew]
    simpa only [gainNativeStep, hclock i] using scalar_parameter_moment_clock_ceiling b1 b2 eps decay rate _ bound (state i)
      hb1 h1 he heta (hs i).2.1 hg hnext
  have hsum : coldGainNativeWeightedMass b1 eps rate
      (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) ≤
        ((1 - b1) * eps) * (1 - rate * decay) * gainNativeParameterMass state +
          rate * gainNativeNegativeMomentMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) +
            (4 * rate * bound) * b1 ^ clock := by
    convert add_le_add (add_le_add (hcoordinate 0) (hcoordinate 1))
      (add_le_add (hcoordinate 2) (hcoordinate 3)) using 1
    all_goals dsimp only [coldGainNativeWeightedMass, gainNativeParameterMass, gainNativeNegativeMomentMass]
    all_goals ring
  have hm := mul_le_mul_of_nonneg_left
    (gain_native_cold_moment_mass_ceiling remaining genGain memGain bound b1 b2 eps decay rate state
      hgen hmem hgain hclip (le_of_lt h1) hs) heta
  have hscaled := mul_le_mul_of_nonneg_right hthreshold (gain_native_masses_nonnegative state hs).1
  have hfeedback := mul_le_mul_of_nonneg_left hscaled (mul_nonneg heta (show 0 ≤ 1 - b1 by linarith only [h1]))
  unfold coldGainNativeWeightedMass at hsum ⊢
  nlinarith only [hsum, hm, hfeedback]

example :
    let state := seededNativeSubweights ((0, 1 / 200), (0, 1))
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
      (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
      NonnegativeNativeState state ∧ (∀ i, -(state i).moment ≤ 1) ∧
      (∀ i, (state i).clock = 0) ∧ coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  dsimp only
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), ?_, ?_,
    by norm_num [coldGainCEGradientScale]⟩
  · intro i
    fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]
  · intro i
    fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- The actual initialized weighted native mass has a finite
nonnegative limit at or above the threshold with legal retained betas.
Sources: appendix C actual CE, native bounds at 3a44336 and weighted
error law above; clipping, signs and clocks generate the recurrence
needed for 26ad2c9, without any future convergence premise. -/
theorem gain_native_seeded_cold_weighted_mass_limit (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (hkeep : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate (seededNativeSubweights ((a, b), (c, d)))
    ∃ value : ℝ, 0 ≤ value ∧ Tendsto (fun n => coldGainNativeWeightedMass b1 eps rate (path n)) atTop (nhds value) := by
  dsimp only
  let initial := seededNativeSubweights ((a, b), (c, d))
  let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
  have hpositive := lt_of_lt_of_le
    (mul_pos (cold_gain_ce_gradient_scale_pos remaining bound hclip) hgen) hthreshold
  have hdecay : 0 < decay := by nlinarith only [hpositive, he]
  obtain ⟨_, _, hbounds⟩ := gain_native_seeded_bounded remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hclip hb1 h1 hb2 h2 he hdecay heta hkeep ha hb hc hd
  apply nonnegative_observer_tendsto_of_geometric_error b1 (4 * rate * bound)
    (fun n => coldGainNativeWeightedMass b1 eps rate (path n)) hb1 h1 (by positivity)
  · intro n
    exact cold_gain_native_weighted_mass_nonnegative b1 eps rate (path n) hb1 (le_of_lt h1) (le_of_lt he) heta (hbounds n).1
  · intro n
    have hclock : ∀ i, (path n i).clock = n := by
      intro i
      rw [gain_native_path_clocks]
      fin_cases i <;> simp [initial, seededNativeSubweights, seededScalarState]
    exact gain_native_cold_weighted_mass_step remaining n genGain memGain bound b1 b2 eps decay rate (path n)
      hgen hmem hgain hclip hb1 h1 he heta (hbounds n).1 (fun i => ((hbounds n).2 i).2.1) hclock hthreshold

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  norm_num [coldGainCEGradientScale]

end Transformer.Grokking.CircuitEfficiency

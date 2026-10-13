import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdBounds
import Mathlib.Topology.Order.MonotoneConvergence

/-!
# Actual critical mass monotonicity with legal zero first beta

Sources: Varma et al., arXiv:2309.02390v1, section 3 CE/decay
competition and appendix C product partials; complete native mass
bounds at 8420a09 and actual cold feedback ceiling at 696630b.

With first beta zero, every actual corrected denominator is at least
epsilon at every retained clock. The newly inserted first moment uses
the true clipped CE feedback. At or above the cold greater-gain
threshold, these actual bounds make total physical parameter mass
nonincreasing. This does not assume coordinate monotonicity.

Legal initialized signs make the whole mass nonnegative. Its actual
sequence is antitone and bounded below, so it has a finite nonnegative
limit. No future mass, input, variance or successful reference is a
premise. The second buffer and its legal beta remain retained; this
control is not the earlier replacement of both buffers by zero.

The finite total-mass limit is not yet individual parameter convergence
or proof that the limit is zero. A nonnegative zero rate is allowed
here, so a positive constant physical mass is not excluded. Positive
rate and true nonlinear dissipation are needed for the critical result.

First beta zero is an explicit legal mathematical control. It does
not describe or modify the frozen beta1=0.9/beta2=0.98 experiments.
Transfer to a nonzero retained numerator needs a separate argument.
Fixed tables/plain CE/uniform native decay differ from appendix C's
assigned coupled norm-cost GD; no learned transformer or thermodynamic
system-size claim is made, and no optimizer/checkpoint is changed.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- One actual step cannot increase total physical mass at or above
the cold threshold when first beta is zero. Sources: appendix C inputs,
cold ceiling at 696630b and native denominator bounds at 8420a09;
the second buffer and completed correction clock remain in the step. -/
theorem gain_native_zero_first_beta_cold_mass_step (remaining : ℕ)
    (genGain memGain bound b2 eps decay rate : ℝ) (state : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (he : 0 < eps) (heta : 0 ≤ rate) (hs : NonnegativeNativeState state)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    gainNativeParameterMass (gainNativeStep remaining genGain memGain bound 0 b2 eps decay rate state) ≤
      gainNativeParameterMass state := by
  have hden : ∀ i, eps ≤ nextBufferDenominator 0 b2 eps (state i).variance
      (appliedGainNativeGradient remaining genGain memGain bound state i) (state i).clock := by
    intro i
    simpa only [sub_zero, one_mul] using next_buffer_denominator_floor 0 b2 eps (state i).variance
      (appliedGainNativeGradient remaining genGain memGain bound state i) (state i).clock
      (by norm_num) (by norm_num) (le_of_lt he)
  have hp := gain_native_parameter_mass_floor_denominator remaining genGain memGain bound 0 b2 eps decay rate eps state
    hgen hmem hclip (by norm_num) (by norm_num) heta he hs hden
  have hm := gain_native_cold_moment_mass_ceiling remaining genGain memGain bound 0 b2 eps decay rate state
    hgen hmem hgain hclip (by norm_num) hs
  simp only [zero_mul, sub_zero, one_mul, zero_add] at hm
  have hmoment := mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right hm (le_of_lt he)) heta
  have hscaled := mul_le_mul_of_nonneg_right hthreshold (gain_native_masses_nonnegative state hs).1
  have hfeedback : (coldGainCEGradientScale remaining bound * genGain * gainNativeParameterMass state) / eps ≤
      decay * gainNativeParameterMass state := (div_le_iff₀ he).mpr (by nlinarith only [hscaled])
  have hlast := mul_le_mul_of_nonneg_left hfeedback heta
  nlinarith only [hp, hmoment, hlast]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

/-- Legal initialized zero-first-beta native CE generates an antitone
total physical mass sequence at or above the cold threshold. Sources:
section 3 competition, appendix C inputs and native sign induction at
65ce284; future sign and input tails are not independent hypotheses. -/
theorem gain_native_zero_first_beta_cold_mass_antitone (remaining : ℕ)
    (genGain memGain bound b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (hkeep : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    Antitone (fun n => gainNativeParameterMass (gainNativePath remaining genGain memGain bound 0 b2 eps decay rate initial n)) := by
  apply antitone_nat_of_succ_le
  intro n
  have hsign := gain_native_nonnegative_path remaining genGain memGain bound 0 b2 eps decay rate initial n
    hgen hmem hclip (by norm_num) (by norm_num) hb2 h2 he heta hkeep hs
  exact gain_native_zero_first_beta_cold_mass_step remaining genGain memGain bound b2 eps decay rate _
    hgen hmem hgain hclip he heta hsign hthreshold

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

/-- Every actual physical coordinate stays below the initialized total
mass at or above the cold threshold with first beta zero. Sources:
appendix C partners, native mass covering at ada36f3 and the generated
antitone path above; a future parameter box is a conclusion. -/
theorem gain_native_zero_first_beta_cold_parameter_box (remaining : ℕ)
    (genGain memGain bound b2 eps decay rate : ℝ) (initial : NativeSubweightState) (n : ℕ) (i : Fin 4)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (hkeep : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    (gainNativePath remaining genGain memGain bound 0 b2 eps decay rate initial n i).parameter ≤
      gainNativeParameterMass initial := by
  have hanti := gain_native_zero_first_beta_cold_mass_antitone remaining genGain memGain bound b2 eps decay rate initial
    hgen hmem hgain hclip hb2 h2 he heta hkeep hs hthreshold
  have hsign := gain_native_nonnegative_path remaining genGain memGain bound 0 b2 eps decay rate initial n
    hgen hmem hclip (by norm_num) (by norm_num) hb2 h2 he heta hkeep hs
  exact le_trans ((gain_native_masses_nonnegative _ hsign).2.2 i) (hanti (Nat.zero_le n))

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

/-- At or above the threshold the actual initialized zero-first-beta
physical mass has a finite nonnegative limit. Sources: appendix C
feedback and native monotonicity above; order completeness supplies
convergence of this numerical observer without assuming a future limit. -/
theorem gain_native_zero_first_beta_cold_mass_limit (remaining : ℕ)
    (genGain memGain bound b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (hkeep : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    ∃ value : ℝ, 0 ≤ value ∧ Tendsto
      (fun n => gainNativeParameterMass (gainNativePath remaining genGain memGain bound 0 b2 eps decay rate initial n))
      atTop (nhds value) := by
  let mass := fun n => gainNativeParameterMass (gainNativePath remaining genGain memGain bound 0 b2 eps decay rate initial n)
  have hanti := gain_native_zero_first_beta_cold_mass_antitone remaining genGain memGain bound b2 eps decay rate initial
    hgen hmem hgain hclip hb2 h2 he heta hkeep hs hthreshold
  have hn : ∀ n, 0 ≤ mass n := fun n => (gain_native_masses_nonnegative _
    (gain_native_nonnegative_path remaining genGain memGain bound 0 b2 eps decay rate initial n
      hgen hmem hclip (by norm_num) (by norm_num) hb2 h2 he heta hkeep hs)).1
  have hbounded : BddBelow (Set.range mass) := by
    apply bddBelow_def.mpr
    refine ⟨0, ?_⟩
    rintro value ⟨n, rfl⟩
    exact hn n
  have ht := tendsto_atTop_ciInf hanti hbounded
  exact ⟨_, ge_of_tendsto ht (Eventually.of_forall hn), ht⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

end Transformer.Grokking.CircuitEfficiency

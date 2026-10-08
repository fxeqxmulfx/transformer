import Transformer.Grokking.CircuitEfficiency.SectionC_GainSourceLimits

/-!
# Actual cold CE coefficient and a nonempty decay regime

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C
train CE at zero products; native shared clipping at lab commit
ca253e4. Compute the true CE feedback coefficient at zero physical
parameters, including the native norm-plus-1e-6 clipping strip.
Buffers and clocks need not be zero for this point computation.

The coefficient depends only on class count and the clip cap.
It is positive at positive cap. The actual CE coefficient at any
finite point lies strictly below one, regardless of forward gains
or parameter signs. This permits an explicit nonzero balanced
point to witness a positive decay/epsilon regime below the cold
Gen feedback. The point is not declared an attracting trajectory.

The intended boundary condition is stated using this computed
numerical coefficient, not a future margin or rule-success
predicate. Exclusion of an actual fully cold parameter limit and
removal of the live-Mem condition require the next path argument.
Plain CE/fixed physical gains/native decoupled decay differ from
the source's norm-cost GD, and no GPTMini transfer is asserted.

There are remaining + 2 train logits: the target and remaining + 1
equal competitors. At zero score the CE slope is the competitor
count divided by total count. The raw product partials vanish,
so the clip multiplier still includes its explicit numerical strip.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Numerical clipped CE magnitude at zero physical parameters.
Sources: appendix C train CE at zero score and native shared
clipping at ca253e4; all supplied arguments enter the expression. -/
noncomputable def coldGainCEGradientScale (remaining : ℕ) (bound : ℝ) : ℝ :=
  min 1 (bound / (1 / 1000000)) * (((remaining : ℝ) + 1) / ((remaining : ℝ) + 2))

/-- All zero physical parameters give exactly the computed cold
coefficient, irrespective of retained buffers/clocks. Sources:
appendix C product derivatives and native clipping at ca253e4;
zero current inputs do not imply zero subsequent momentum updates. -/
theorem gain_ce_gradient_scale_zero_parameters (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : NativeSubweightState) (hp : ∀ i, (state i).parameter = 0) :
    gainCEGradientScale remaining genGain memGain bound state = coldGainCEGradientScale remaining bound := by
  have hs : gainNativeTotalScore genGain memGain state = 0 := by
    unfold gainNativeTotalScore physicalCircuitScore
    rw [hp 0, hp 1, hp 2, hp 3]
    ring
  have hg : rawGainNativeGradient remaining genGain memGain state = fun _ => 0 := by
    funext i
    unfold rawGainNativeGradient
    rw [hp (nativeFactorPartner i)]
    ring
  unfold gainCEGradientScale
  rw [hg, hs]
  unfold coordinateClipFactor coordinateGradientNorm
  simp only [zero_pow (by norm_num : (2 : ℕ) ≠ 0), Finset.sum_const_zero, Real.sqrt_zero,
    zero_add, Real.exp_zero]
  unfold coldGainCEGradientScale
  congr 2
  ring

example : ∀ _ : Fin 4, ({ parameter := 0, moment := -1, variance := 1, clock := 37 } : ScalarState).parameter = 0 := by
  intro
  change (0 : ℝ) = 0
  rfl

/-- Positive native clip cap gives positive actual cold feedback.
Sources: appendix C finite class count and native clipping at
ca253e4; no physical gain or successful current prediction is needed. -/
theorem cold_gain_ce_gradient_scale_pos (remaining : ℕ) (bound : ℝ) (hb : 0 < bound) :
    0 < coldGainCEGradientScale remaining bound := by
  unfold coldGainCEGradientScale
  apply mul_pos
  · exact lt_min (by norm_num) (div_pos hb (by norm_num))
  · exact div_pos (by positivity) (by positivity)

example : (0 : ℝ) < 1 := by norm_num

/-- A unit native cap does not clip the zero raw norm; the cold
coefficient is the actual finite-class CE slope. Sources: appendix C
zero score and native clipping strip at ca253e4. -/
theorem cold_gain_ce_gradient_scale_unit_cap (remaining : ℕ) :
    coldGainCEGradientScale remaining 1 = ((remaining : ℝ) + 1) / ((remaining : ℝ) + 2) := by
  unfold coldGainCEGradientScale
  norm_num

/-- Every finite actual clipped CE slope is strictly less than one.
Sources: appendix C multiclass CE and native max-one clipping at
ca253e4; finite positive exponential rules out an exact unit slope. -/
theorem gain_ce_gradient_scale_unit_interval (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : NativeSubweightState) (hb : 0 < bound) :
    0 < gainCEGradientScale remaining genGain memGain bound state ∧
      gainCEGradientScale remaining genGain memGain bound state < 1 := by
  refine ⟨gain_ce_gradient_scale_pos remaining genGain memGain bound state hb, ?_⟩
  let slope := ((remaining : ℝ) + 1) /
    (Real.exp (gainNativeTotalScore genGain memGain state) + (remaining : ℝ) + 1)
  have hs : 0 < slope := div_pos (by positivity) (by positivity)
  have hexp := Real.exp_pos (gainNativeTotalScore genGain memGain state)
  have hlt : slope < 1 := by
    apply (div_lt_one (by positivity)).mpr
    linarith only [hexp]
  have hc := coordinate_clip_factor_le_one bound (rawGainNativeGradient remaining genGain memGain state)
  have hu := mul_le_mul_of_nonneg_right hc (le_of_lt hs)
  change coordinateClipFactor bound (rawGainNativeGradient remaining genGain memGain state) * slope < 1
  have hbound : coordinateClipFactor bound (rawGainNativeGradient remaining genGain memGain state) * slope ≤ slope := by
    simpa only [one_mul] using hu
  exact lt_of_le_of_lt hbound hlt

example : (0 : ℝ) < 1 := by norm_num

/-- A genuine nonzero native balanced point lies below the cold Gen
feedback threshold at positive decay/epsilon. Sources: appendix C
actual CE and section 3 Efficiency, native balance at ca253e4;
the epsilon is explicitly chosen, not the preserved GPTMini setting. -/
theorem gain_native_efficient_point_cold_regime :
    let state := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 state
    0 < eps ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2) * eps < coldGainCEGradientScale 111 1 * 3 ∧
      ∀ i, (1 / 2 : ℝ) * (state i).parameter + appliedGainNativeGradient 111 3 2 1 state i /
        (|appliedGainNativeGradient 111 3 2 1 state i| + eps) = 0 := by
  dsimp only
  let state := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
  have hw := gain_native_efficient_point_balance 111 1 (by norm_num)
  have hs := (gain_ce_gradient_scale_unit_interval 111 3 2 1 state (by norm_num)).2
  have hc : coldGainCEGradientScale 111 1 = 112 / 113 := by
    rw [cold_gain_ce_gradient_scale_unit_cap]
    norm_num
  refine ⟨hw.1, by norm_num, ?_, hw.2⟩
  rw [hc]
  linarith only [hs]

/-- Unit cap and gain three meet the cold feedback regime at decay
0.1 and epsilon 1e-8 for every finite class count. Sources: actual
appendix C zero-score CE and native clipping at ca253e4; these are
toy physical gains, not a gain certificate for the learned GPTMini. -/
theorem cold_gain_small_decay_epsilon_regime (remaining : ℕ) :
    (1 / 10 : ℝ) * (1 / 100000000) < coldGainCEGradientScale remaining 1 * 3 := by
  rw [cold_gain_ce_gradient_scale_unit_cap]
  have hn : (0 : ℝ) ≤ (remaining : ℝ) := by positivity
  have hden : (0 : ℝ) < (remaining : ℝ) + 2 := by positivity
  have hs : (1 / 2 : ℝ) ≤ ((remaining : ℝ) + 1) / ((remaining : ℝ) + 2) := by
    apply (le_div_iff₀ hden).mpr
    linarith only [hn]
  nlinarith only [hs]

end Transformer.Grokking.CircuitEfficiency

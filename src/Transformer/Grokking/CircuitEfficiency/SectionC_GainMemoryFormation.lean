import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemorySelection
import Transformer.Grokking.CircuitEfficiency.SectionC_GainDelaySeeds
import Transformer.Grokking.CircuitEfficiency.SectionC_GainFeedbackFloor

/-!
# Uniform native Mem formation from zero first factors in both pairs

Sources: Varma et al., arXiv:2309.02390v1, section 3 Slow vs fast
learning and appendix C product logits with zero initial first factors;
original retained native selection at lab commit 4c76b7d, current
feedback floors at ee02740 and native growth envelopes at 73c78eb.

Keep the pinned binary gains 3/2, beta1=0.9, beta2=0.98,
epsilon/cap 1, decay 100 and rate 0.001. Initialize Gen at (0,seed)
and Mem at (0,1), with 0 <= seed <= 1 and both buffers/clocks zero.
The same complete CE/clipping callback has a uniform positive scale
floor on this current physical box. The actual first Mem update and
its partner decay floor give two positive physical factors bounded
below by one fixed amplitude, independently of the small Gen seed.

The actual first retained Gen growth weight is bounded by 103/50
times its seed. The forward at initialization has zero logits; strict
training correctness cannot hold there. A later prefix must start
at actual clock one and retain its nonzero buffers and completed clock.
A full-state path shift is proved, with no reset to a fresh seeded state.

These present-state bounds prepare arbitrary wrong prefixes from the
source-style untrained initialization. They do not yet prove a delayed
transition or positive limiting confidence. Fixed gained lookup tables,
exact reals and uniform decoupled AdamW differ from appendix C's
coupled assigned norm/GD and learned stochastic/numerical GPTMini.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- The shared actual CE multiplier at double-zero initialization
has a uniform positive box floor below one. Sources: appendix C
zero initial products and complete native feedback floor at ee02740;
only current initial parameter bounds, not future convergence, enter. -/
theorem fixed_gain_memory_seed_formation_scale (seed : ℝ) (hseed : 0 ≤ seed) (hunit : seed ≤ 1) :
    0 < gainCEFeedbackFloor 0 3 2 1 1 ∧
      gainCEFeedbackFloor 0 3 2 1 1 ≤
        gainCEGradientScale 0 3 2 1 (seededNativeSubweights ((0, seed), (0, 1))) ∧
      gainCEFeedbackFloor 0 3 2 1 1 < 1 := by
  let initial := seededNativeSubweights ((0, seed), (0, 1))
  have hp : ∀ i, 0 ≤ (initial i).parameter ∧ (initial i).parameter ≤ 1 := by
    intro i
    fin_cases i <;> norm_num [initial, seededNativeSubweights, seededScalarState, hseed, hunit]
  have hfloor := gain_ce_gradient_scale_box_floor 0 3 2 1 1 initial
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hp
  have hupper := (gain_ce_gradient_scale_unit_interval 0 3 2 1 initial (by norm_num)).2
  exact ⟨gain_ce_feedback_floor_pos 0 3 2 1 1 (by norm_num) (by norm_num) (by norm_num),
    hfloor, lt_of_le_of_lt hfloor hupper⟩

example : (0 : ℝ) ≤ 1 / 200 ∧ (1 / 200 : ℝ) ≤ 1 := by norm_num

/-- Both actual Mem factors at clock one exceed a uniform positive
amplitude independent of the Gen seed. Sources: section 3 double-zero
formation, appendix C true partials and native first-step normalization
at 88aa892; the complete shared clip and legal beta corrections remain. -/
theorem fixed_gain_memory_seed_mem_formation_floor (seed : ℝ) (hseed : 0 ≤ seed) (hunit : seed ≤ 1) :
    let initial := seededNativeSubweights ((0, seed), (0, 1))
    gainCEFeedbackFloor 0 3 2 1 1 / 2000 ≤ (fixedGainMemoryPath initial 1 2).parameter ∧
      gainCEFeedbackFloor 0 3 2 1 1 / 2000 ≤ (fixedGainMemoryPath initial 1 3).parameter := by
  let initial := seededNativeSubweights ((0, seed), (0, 1))
  let gradient := appliedGainNativeGradient 0 3 2 1 initial 2
  have hf := fixed_gain_memory_seed_formation_scale seed hseed hunit
  have heq : gradient = -(gainCEGradientScale 0 3 2 1 initial * 2) := by
    dsimp only [gradient]
    rw [gain_native_applied_gradient_scale]
    norm_num [initial, nativeFactorGain, nativeFactorPartner, seededNativeSubweights, seededScalarState]
  have hinput : 2 * gainCEFeedbackFloor 0 3 2 1 1 ≤ -gradient := by
    rw [heq]
    linarith only [hf.2.1]
  have hnegative : 0 ≤ -gradient := by linarith only [hinput, hf.1]
  have habs : |gradient| ≤ 1 := gain_native_applied_gradient_abs_bound 0 3 2 1 initial 2 (by norm_num)
  have hdiv := div_le_div_of_nonneg_left
    (show 0 ≤ (1 / 1000 : ℝ) * (-gradient) from mul_nonneg (by norm_num) hnegative)
    (show 0 < |gradient| + 1 by positivity) (show |gradient| + 1 ≤ (2 : ℝ) by linarith only [habs])
  have hparameter : (fixedGainMemoryPath initial 1 2).parameter =
      (1 / 1000 : ℝ) * (-gradient) / (|gradient| + 1) := by
    change (scalarNativeStep (9 / 10) (49 / 50) 1 100 (1 / 1000) (seededScalarState 0) gradient).parameter = _
    rw [scalar_native_first_parameter]
    unfold firstUpdate
    rw [firstDirection_eq _ _ _ _ (by norm_num) (by norm_num)]
    ring
  have hpartner := gain_native_parameter_decay_floor 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000)
    initial 3 (by norm_num) (by norm_num [nativeFactorGain])
    (by norm_num [initial, nativeFactorPartner, seededNativeSubweights, seededScalarState])
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [initial, seededNativeSubweights, seededScalarState])
  have hpartnerFloor : (9 / 10 : ℝ) ≤ (fixedGainMemoryPath initial 1 3).parameter := by
    change (1 - (1 / 1000 : ℝ) * 100) * 1 ≤ (fixedGainMemoryPath initial 1 3).parameter at hpartner
    norm_num only at hpartner
    exact hpartner
  dsimp only
  constructor
  · rw [hparameter]
    linarith only [hinput, hdiv, hf.1]
  · linarith only [hpartnerFloor, hf.2.2]

example : (0 : ℝ) ≤ 1 / 200 ∧ (1 / 200 : ℝ) ≤ 1 := by norm_num

/-- One fixed positive formed Mem amplitude works for every admissible
Gen seed, before any future-path comparison is supplied. Sources:
section 3 untrained initialization and native actual first-step floors
at 88aa892/ee02740; the amplitude is chosen independently of the seed. -/
theorem fixed_gain_memory_uniform_mem_formation :
    ∃ amplitude : ℝ, 0 < amplitude ∧ ∀ seed : ℝ, 0 ≤ seed → seed ≤ 1 →
      amplitude ≤ (fixedGainMemoryPath (seededNativeSubweights ((0, seed), (0, 1))) 1 2).parameter ∧
        amplitude ≤ (fixedGainMemoryPath (seededNativeSubweights ((0, seed), (0, 1))) 1 3).parameter := by
  have hf := fixed_gain_memory_seed_formation_scale 0 (by norm_num) (by norm_num)
  refine ⟨gainCEFeedbackFloor 0 3 2 1 1 / 2000, div_pos hf.1 (by norm_num), ?_⟩
  intro seed hseed hunit
  exact fixed_gain_memory_seed_mem_formation_floor seed hseed hunit

/-- Actual first-step retained Gen parameter/moment growth is at most
103/50 times its initial partner seed. Sources: section 3 slow formation
and the original complete native envelope at 73c78eb; the formed state
keeps both actual moments/variance and the new completed clock. -/
theorem fixed_gain_memory_seed_gen_formation_ceiling (seed : ℝ) (hseed : 0 ≤ seed) :
    gainGenGrowthWeight (9 / 10) 1 (1 / 1000)
      (fixedGainMemoryPath (seededNativeSubweights ((0, seed), (0, 1))) 1) ≤ (103 / 50 : ℝ) * seed := by
  let initial := seededNativeSubweights ((0, seed), (0, 1))
  have hs := native_seeded_nonnegative 0 seed 0 1 le_rfl hseed le_rfl (by norm_num)
  have hu := gain_gen_growth_weight_step 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) initial
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) hs
  have hw : gainGenGrowthWeight (9 / 10) 1 (1 / 1000) initial = seed := by
    simp [gainGenGrowthWeight, gainGenParameterMass, gainGenNegativeMomentMass,
      initial, seededNativeSubweights, seededScalarState]
  simpa only [fixedGainMemoryPath, gainNativePath, hw,
    show gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 = (103 / 50 : ℝ) by norm_num [gainGenGrowthFactor]] using hu

example : (0 : ℝ) ≤ 1 / 200 := by norm_num

/-- Shifting the original path by one preserves the entire retained
formed state, including both buffers and the completed clock. Sources:
appendix C repeated updates and the original native recurrence at
c268c1f; restarting means a current full state, not a fresh seed/reset. -/
theorem fixed_gain_memory_path_shift_one (initial : NativeSubweightState) (n : ℕ) :
    fixedGainMemoryPath initial (n + 1) = fixedGainMemoryPath (fixedGainMemoryPath initial 1) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change gainNativeStep 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000)
      (fixedGainMemoryPath initial (n + 1)) = gainNativeStep 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000)
        (fixedGainMemoryPath (fixedGainMemoryPath initial 1) n)
    rw [ih]

end Transformer.Grokking.CircuitEfficiency

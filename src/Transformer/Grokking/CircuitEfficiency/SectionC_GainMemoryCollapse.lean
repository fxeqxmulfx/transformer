import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryContraction
import Mathlib.Analysis.SpecificLimits.Basic

/-!
# Initialized native mass contraction with legal retained beta memory

Sources: Varma et al., arXiv:2309.02390v1, section 3 CE/decay
competition and appendix C product coordinates; retained native
AdamW/clipping and corrected denominator at lab commit ada36f3.

Start with any nonnegative retained numerical state, legal first and
second betas, positive gains/cap/epsilon, nonnegative rate and remaining
decay. Actual sign induction and the current weighted-memory ceiling
give all-clock power bounds on that mass and every physical coordinate.

Under the static strong-decay threshold
GenGain*(2-beta1)/((1-beta1)*epsilon) < decay,
positive rate makes the computed ceiling a strict contraction. Squeezing
generates zero weighted-mass and all four physical parameter limits.
Both buffers and completed clocks are the original retained path;
zero-beta or zero-buffer initialization is not imposed.

No parameter convergence, future input/box/sign data, successful reference
or margin is supplied. The sufficient threshold uses the uniform
first-clock epsilon floor and need not be sharp. Convergence to zero
alone neither proves nor refutes any finite-clock task decisions.
This is a fixed-table exact-real decoupled-native result, differing from
appendix C coupled-cost GD. Learned GPTMini, stochastic minibatches and
floating-point kernels are not identified with these dynamics.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter

/-- The actual initialized legal-beta path has power ceilings on the
weighted memory mass and every physical coordinate. Sources: appendix C
factors and native mass/sign iteration at ada36f3; full retained signs
are generated, with no future box, reference or moment reset. -/
theorem gain_native_memory_mass_power_ceiling (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (hd : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial) :
    let q := gainNativeMemoryRatio b1 eps decay rate genGain
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    ∀ n, NonnegativeNativeState (path n) ∧
      gainNativeMemoryMass b1 eps rate (path n) ≤ q ^ n * gainNativeMemoryMass b1 eps rate initial ∧
      ∀ i, (path n i).parameter ≤ q ^ n * gainNativeMemoryMass b1 eps rate initial := by
  let q := gainNativeMemoryRatio b1 eps decay rate genGain
  let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
  have hbeta : 0 ≤ b1 * (2 - b1) := mul_nonneg hb1 (by linarith only [h1])
  have hq : 0 ≤ q := le_trans hbeta (le_max_right _ _)
  have hn : ∀ n, NonnegativeNativeState (path n) := by
    intro n
    exact gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
      hgen hmem hclip hb1 h1 hb2 h2 he heta hd hs
  have hpowers : ∀ n, gainNativeMemoryMass b1 eps rate (path n) ≤ q ^ n * gainNativeMemoryMass b1 eps rate initial := by
    intro n
    induction n with
    | zero =>
      change gainNativeMemoryMass b1 eps rate initial ≤ q ^ 0 * gainNativeMemoryMass b1 eps rate initial
      simp only [pow_zero, one_mul, le_refl]
    | succ n ih =>
      have hstep := gain_native_memory_mass_step_ceiling remaining genGain memGain bound b1 b2 eps decay rate (path n)
        hgen hmem hgain hclip hb1 h1 he heta (hn n)
      have hu := le_trans hstep (mul_le_mul_of_nonneg_left ih hq)
      simpa only [path, gainNativePath, q, pow_succ, mul_assoc, mul_comm, mul_left_comm] using hu
  dsimp only
  intro n
  exact ⟨hn n, hpowers n, fun i => le_trans
    ((gain_native_memory_mass_covers b1 eps rate (path n) he heta (hn n)).2.2 i) (hpowers n)⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧
    (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 100 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- Static strong native decay makes initialized weighted parameter/
moment mass tend to zero with legal nonzero betas. Sources: section 3
competition and native denominator/mass laws at ada36f3; convergence
is derived from retained initialization, not a future limiting premise. -/
theorem gain_native_memory_mass_tendsto_zero (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial)
    (hstrong : genGain * (2 - b1) / ((1 - b1) * eps) < decay) :
    Tendsto (fun n => gainNativeMemoryMass b1 eps rate
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n)) atTop (nhds 0) := by
  have hq := gain_native_memory_ratio_strong_decay b1 eps decay rate genGain h1 he heta hgen hd hstrong
  have hpowers := gain_native_memory_mass_power_ceiling remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he (le_of_lt heta) hd hs
  have hpow := tendsto_pow_atTop_nhds_zero_of_lt_one (le_of_lt hq.1) hq.2
  apply squeeze_zero
    (fun n => (gain_native_memory_mass_covers b1 eps rate _ he (le_of_lt heta) (hpowers n).1).1)
    (fun n => (hpowers n).2.1)
  simpa only [zero_mul] using hpow.mul_const (gainNativeMemoryMass b1 eps rate initial)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧
    (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 100 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    (3 : ℝ) * (2 - 9 / 10) / ((1 - 9 / 10) * 1) < 100 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num⟩

/-- All original physical parameter limits are zero under initialized
strong native decay with legal retained betas. Sources: appendix C
coordinates and native weighted contraction at ada36f3; no zero-beta,
zero-buffer or assumed individual parameter-convergence input is used. -/
theorem gain_native_memory_parameters_tendsto_zero (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial)
    (hstrong : genGain * (2 - b1) / ((1 - b1) * eps) < decay) :
    ∀ i, Tendsto (fun n => (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).parameter)
      atTop (nhds 0) := by
  have hmass := gain_native_memory_mass_tendsto_zero remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hd hs hstrong
  have hpowers := gain_native_memory_mass_power_ceiling remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he (le_of_lt heta) hd hs
  intro i
  exact squeeze_zero (fun n => ((hpowers n).1 i).1)
    (fun n => (gain_native_memory_mass_covers b1 eps rate _ he (le_of_lt heta) (hpowers n).1).2.2 i) hmass

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧
    (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 100 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    (3 : ℝ) * (2 - 9 / 10) / ((1 - 9 / 10) * 1) < 100 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num⟩

/-- Every actual clipped coordinate input vanishes on the initialized
strong-decay path with legal retained betas. Sources: appendix C true
partials and native mass cap at ada36f3; the generated mass limit
derives the input limit, without a prescribed or convergent stream. -/
theorem gain_native_memory_inputs_tendsto_zero (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial)
    (hstrong : genGain * (2 - b1) / ((1 - b1) * eps) < decay) :
    ∀ i, Tendsto (fun n => appliedGainNativeGradient remaining genGain memGain bound
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n) i) atTop (nhds 0) := by
  have hmass := gain_native_memory_mass_tendsto_zero remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hd hs hstrong
  have hpowers := gain_native_memory_mass_power_ceiling remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he (le_of_lt heta) hd hs
  intro i
  have hcap := fun n => gain_native_applied_gradient_mass_cap remaining genGain memGain bound _ i
    hgen hmem hgain hclip (hpowers n).1
  have hupper : ∀ n, -appliedGainNativeGradient remaining genGain memGain bound
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n) i ≤
      genGain * gainNativeMemoryMass b1 eps rate
        (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n) := by
    intro n
    exact le_trans (hcap n).2 (mul_le_mul_of_nonneg_left
      (gain_native_memory_mass_covers b1 eps rate _ he (le_of_lt heta) (hpowers n).1).2.1 (le_of_lt hgen))
  have hnegative := squeeze_zero (fun n => (hcap n).1) hupper
    (by simpa only [mul_zero] using hmass.const_mul genGain)
  simpa only [neg_neg, neg_zero] using hnegative.neg

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧
    (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 100 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    (3 : ℝ) * (2 - 9 / 10) / ((1 - 9 / 10) * 1) < 100 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num⟩

end Transformer.Grokking.CircuitEfficiency

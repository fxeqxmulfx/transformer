import Transformer.Grokking.CircuitEfficiency.SectionC_GainSigns
import Transformer.Grokking.AdamW.ScalarDenominator

/-!
# Bounded actual gained-CE parameters and retained buffers

Sources: Varma et al., arXiv:2309.02390v1, appendix C's physical
two-factor CE, and native PyTorch AdamW/clipping at lab commit
3a44336. Actual clipping bounds every generated input. The retained
moment insertions preserve the corresponding coordinate bounds.
The full epsilon denominator floor then gives a finite parameter
ceiling balanced against positive decoupled decay.

The entire argument follows the actual closed feedback recurrence,
without an independently supplied input stream or parameter limit.
For standard zero-buffer nonnegative seeds at positive decay, an
explicit finite ceiling exists. Parameters and both buffers are
bounded; completed clocks still advance and are not bounded here.
Boundedness alone implies neither convergence nor eventual Gen
selection. A later small-rate/attraction argument must establish
those separately on the same delayed-prefix trajectories. Exact
reals, fixed tables, gains and uniform native decay are explicit
deviations from appendix C's coupled-cost GD and from learned GPTMini.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- True clipped CE preserves numerical parameter and moment/variance
ceilings at every finite clock. Sources: appendix C partner feedback
and native AdamW at 3a44336; the decay-budget condition concerns a
constant ceiling and current initialization, not future boundedness. -/
theorem gain_native_bounded_path (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate ceiling : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hd : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial)
    (hi : ∀ i, (initial i).parameter ≤ ceiling ∧
      -(initial i).moment ≤ bound ∧ (initial i).variance ≤ bound ^ 2)
    (hcost : bound ≤ decay * ((1 - b1) * eps) * ceiling) :
    ∀ n, NonnegativeNativeState (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n) ∧
      ∀ i, let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i
        state.parameter ≤ ceiling ∧ -state.moment ≤ bound ∧ state.variance ≤ bound ^ 2 := by
  intro n
  induction n with
  | zero => exact ⟨hs, hi⟩
  | succ n ih =>
    refine ⟨gain_native_nonnegative_step remaining genGain memGain bound b1 b2 eps decay rate _
      hgen hmem hclip hb1 h1 hb2 h2 he heta hd ih.1, ?_⟩
    intro i
    let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n
    let gradient := appliedGainNativeGradient remaining genGain memGain bound state i
    let next := scalarNativeStep b1 b2 eps decay rate (state i) gradient
    have hg : gradient ≤ 0 := gain_native_applied_gradient_nonpos remaining genGain memGain bound state i hclip
      (le_of_lt (gain_native_factor_gain_pos genGain memGain hgen hmem i)) (ih.1 _).1
    have hmag : -gradient ≤ bound := by
      have habs := gain_native_applied_gradient_abs_bound remaining genGain memGain bound state i hclip
      change |gradient| ≤ bound at habs
      rw [abs_of_nonpos hg] at habs
      exact habs
    have hmold := mul_le_mul_of_nonneg_left (ih.2 i).2.1 hb1
    have hminput := mul_le_mul_of_nonneg_left hmag (show 0 ≤ 1 - b1 by linarith only [h1])
    have hmnext : -next.moment ≤ bound := by
      change -(b1 * (state i).moment + (1 - b1) * gradient) ≤ bound
      nlinarith only [hmold, hminput]
    have hsquare : gradient ^ 2 ≤ bound ^ 2 := by nlinarith only [hg, hmag, hclip]
    have hvold := mul_le_mul_of_nonneg_left (ih.2 i).2.2 hb2
    have hvinput := mul_le_mul_of_nonneg_left hsquare (show 0 ≤ 1 - b2 by linarith only [h2])
    have hvnext : next.variance ≤ bound ^ 2 := by
      change b2 * (state i).variance + (1 - b2) * gradient ^ 2 ≤ bound ^ 2
      nlinarith only [hvold, hvinput]
    have hf : 0 < (1 - b1) * eps := by positivity
    have hdiv : -next.moment / ((1 - b1) * eps) ≤ decay * ceiling := by
      apply (div_le_iff₀ hf).mpr
      nlinarith only [hmnext, hcost]
    have hadapt := mul_le_mul_of_nonneg_left hdiv heta
    have hpold := mul_le_mul_of_nonneg_left (ih.2 i).1 hd
    change (1 - rate * decay) * (state i).parameter ≤ (1 - rate * decay) * ceiling at hpold
    have hupper := scalar_parameter_denominator_ceiling b1 b2 eps decay rate gradient (state i)
      hb1 h1 he heta (ih.1 i).2.1 hg
    change next.parameter ≤ (1 - rate * decay) * (state i).parameter +
      rate * (-next.moment) / ((1 - b1) * eps) at hupper
    have hgroup : rate * (-next.moment) / ((1 - b1) * eps) =
        rate * (-next.moment / ((1 - b1) * eps)) := by ring
    rw [hgroup] at hupper
    change next.parameter ≤ ceiling ∧ -next.moment ≤ bound ∧ next.variance ≤ bound ^ 2
    exact ⟨by nlinarith only [hupper, hadapt, hpold], hmnext, hvnext⟩

example :
    let initial := seededNativeSubweights ((0, 1 / 200), (1, 1))
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 0 ≤ (1 / 1000 : ℝ) ∧
    0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ NonnegativeNativeState initial ∧
    (∀ i, (initial i).parameter ≤ 100 ∧ -(initial i).moment ≤ 1 ∧ (initial i).variance ≤ 1 ^ 2) ∧
    (1 : ℝ) ≤ (1 / 10) * ((1 - 9 / 10) * 1) * 100 := by
  dsimp only
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), ?_, by norm_num⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- Nonnegative zero-buffer physical seeds have an explicit finite
parameter/buffer ceiling at positive native decay. Sources: appendix
C's initialized factors and native clipping/decay at 3a44336;
no future input, denominator ceiling or parameter convergence is supplied. -/
theorem gain_native_seeded_bounded (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 ≤ rate)
    (hd : 0 ≤ 1 - rate * decay)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d) :
    ∃ ceiling : ℝ, 0 < ceiling ∧
      let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
        (seededNativeSubweights ((a, b), (c, d)))
      ∀ n, NonnegativeNativeState (path n) ∧
        ∀ i, (path n i).parameter ≤ ceiling ∧
          -(path n i).moment ≤ bound ∧ (path n i).variance ≤ bound ^ 2 := by
  let denominator := decay * ((1 - b1) * eps)
  let ceiling := a + b + c + d + bound / denominator
  let initial := seededNativeSubweights ((a, b), (c, d))
  have hden : 0 < denominator := by dsimp only [denominator]; positivity
  have hpart : 0 < bound / denominator := div_pos hclip hden
  have hceiling : 0 < ceiling := by dsimp only [ceiling]; linarith only [ha, hb, hc, hdseed, hpart]
  have hs : NonnegativeNativeState initial :=
    native_seeded_nonnegative _ _ _ _ ha hb hc hdseed
  have hi : ∀ i, (initial i).parameter ≤ ceiling ∧
      -(initial i).moment ≤ bound ∧ (initial i).variance ≤ bound ^ 2 := by
    intro i
    constructor
    · fin_cases i <;> norm_num [initial, ceiling, seededNativeSubweights, seededScalarState] <;>
        linarith only [ha, hb, hc, hdseed, hpart]
    · fin_cases i <;> norm_num [initial, seededNativeSubweights, seededScalarState] <;>
        exact ⟨le_of_lt hclip, sq_nonneg bound⟩
  have hcost : bound ≤ decay * ((1 - b1) * eps) * ceiling := by
    have hcancel := div_mul_cancel₀ bound (ne_of_gt hden)
    have hsum := mul_nonneg (le_of_lt hden) (show 0 ≤ a + b + c + d by positivity)
    change bound ≤ denominator * ceiling
    dsimp only [ceiling]
    nlinarith only [hcancel, hsum]
  exact ⟨ceiling, hceiling, gain_native_bounded_path remaining genGain memGain bound b1 b2 eps decay rate ceiling initial
    hgen hmem hclip hb1 h1 hb2 h2 he heta hd hs hi hcost⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 10 ∧
    0 ≤ (1 / 1000 : ℝ) ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 := by norm_num

end Transformer.Grokking.CircuitEfficiency

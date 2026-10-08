import Transformer.Grokking.CircuitEfficiency.SectionC_GainFeedbackBounds
import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdRegime

/-!
# A uniform positive actual CE scale on initialized native paths

Sources: Varma et al., arXiv:2309.02390v1, appendix C's actual
multiclass product CE, and native AdamW/clipping at lab commit ee02740.
Combine the proved physical score and raw-gradient norm ceilings
into a strictly positive numerical floor for the complete shared
clipped CE multiplier. The added clipping denominator 1e-6 remains.

At positive decay, actual initialized nonnegative paths already
have a finite physical box. Derive the floor on that same closed
feedback trajectory, including legal nonzero betas and retained
buffers/clocks, without a future boundedness, scale or convergence
premise. The scalar hyperparameters stay fixed for the entire path.

The bound is qualitative in exact reals: an exponential of the loose
parameter ceiling can make it extremely small. It is not an estimate
of the observed GPTMini transition time or a floating-point guarantee.
It will support a separate balanced zero-beta relative-growth argument;
it alone proves neither convergence nor eventual decision selection.
Fixed physical gains and tables with uniform decoupled AdamW differ
from appendix C's assigned coupled norm cost/GD and learned GPTMini.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Numerical floor from the actual norm and score box estimates.
Sources: appendix C true CE and native norm-two clipping at ee02740;
every argument enters the explicit bound, not a prescribed future path. -/
noncomputable def gainCEFeedbackFloor (remaining : ℕ) (genGain memGain bound ceiling : ℝ) : ℝ :=
  min 1 (bound / (2 * genGain * ceiling + 1 / 1000000)) *
    (((remaining : ℝ) + 1) / (Real.exp ((genGain + memGain) * ceiling ^ 2) + (remaining : ℝ) + 1))

/-- The numerical complete-feedback floor is strictly positive at
a positive cap and a finite nonnegative Gen box. Sources: actual
appendix C CE and native clipping at ee02740; no future margin is assumed. -/
theorem gain_ce_feedback_floor_pos (remaining : ℕ) (genGain memGain bound ceiling : ℝ)
    (hgen : 0 ≤ genGain) (hc : 0 ≤ ceiling) (hb : 0 < bound) :
    0 < gainCEFeedbackFloor remaining genGain memGain bound ceiling := by
  unfold gainCEFeedbackFloor
  apply mul_pos
  · apply lt_min (by norm_num)
    exact div_pos hb (by positivity)
  · positivity

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 100 ∧ (0 : ℝ) < 1 := by
  norm_num

/-- The full actual clipped CE multiplier dominates its explicit
physical-box floor. Sources: appendix C true score/partials and
native clipping at ee02740; current parameter bounds suffice. -/
theorem gain_ce_gradient_scale_box_floor (remaining : ℕ) (genGain memGain bound ceiling : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (horder : memGain ≤ genGain) (hc : 0 ≤ ceiling) (hb : 0 < bound)
    (hp : ∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ ceiling) :
    gainCEFeedbackFloor remaining genGain memGain bound ceiling ≤
      gainCEGradientScale remaining genGain memGain bound state := by
  have hclip := gain_native_clip_factor_box remaining genGain memGain bound ceiling state
    hgen hmem horder hc (le_of_lt hb) hp
  have hscore := gain_native_total_score_box genGain memGain ceiling state
    (le_of_lt hgen) (le_of_lt hmem) hc hp
  have hexp := Real.exp_le_exp.mpr hscore
  have hslope : ((remaining : ℝ) + 1) /
      (Real.exp ((genGain + memGain) * ceiling ^ 2) + (remaining : ℝ) + 1) ≤
      ((remaining : ℝ) + 1) /
        (Real.exp (gainNativeTotalScore genGain memGain state) + (remaining : ℝ) + 1) := by
    apply div_le_div_of_nonneg_left (by positivity) (by positivity)
    linarith only [hexp]
  have hraw : 0 < coordinateClipFactor bound (rawGainNativeGradient remaining genGain memGain state) :=
    coordinate_clip_factor_pos bound _ hb
  unfold gainCEFeedbackFloor gainCEGradientScale
  exact mul_le_mul hclip hslope (by positivity) (le_of_lt hraw)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 100 ∧ (0 : ℝ) < 1 ∧
    ∀ i : Fin 4, 0 ≤ (seededNativeSubweights ((1, 2), (3, 4)) i).parameter ∧
      (seededNativeSubweights ((1, 2), (3, 4)) i).parameter ≤ 100 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- Actual zero-buffer seeded native CE paths have a uniform strictly
positive scale and strict unit ceiling. Sources: appendix C's initialized
physical factors and native AdamW at ee02740; the parameter box and
feedback floor are derived, not supplied as future hypotheses. -/
theorem gain_native_seeded_uniform_scale (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (horder : memGain ≤ genGain)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 ≤ rate)
    (hd : 0 ≤ 1 - rate * decay)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d) :
    ∃ lower : ℝ, 0 < lower ∧
      let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
        (seededNativeSubweights ((a, b), (c, d)))
      ∀ n, lower ≤ gainCEGradientScale remaining genGain memGain bound (path n) ∧
        gainCEGradientScale remaining genGain memGain bound (path n) < 1 := by
  obtain ⟨ceiling, hceiling, hbox⟩ := gain_native_seeded_bounded remaining genGain memGain bound b1 b2 eps decay rate
    a b c d hgen hmem hclip hb1 h1 hb2 h2 he hdecay heta hd ha hb hc hdseed
  refine ⟨gainCEFeedbackFloor remaining genGain memGain bound ceiling,
    gain_ce_feedback_floor_pos remaining genGain memGain bound ceiling
      (le_of_lt hgen) (le_of_lt hceiling) hclip, ?_⟩
  dsimp only
  intro n
  let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
    (seededNativeSubweights ((a, b), (c, d))) n
  have hparam : ∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ ceiling := by
    intro i
    exact ⟨((hbox n).1 i).1, ((hbox n).2 i).1⟩
  exact ⟨gain_ce_gradient_scale_box_floor remaining genGain memGain bound ceiling state
      hgen hmem horder (le_of_lt hceiling) hclip hparam,
    (gain_ce_gradient_scale_unit_interval remaining genGain memGain bound state hclip).2⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (2 : ℝ) ≤ 3 ∧
    0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧ 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 10 ∧ 0 ≤ (1 / 1000 : ℝ) ∧
    0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 := by
  norm_num

/-- Every actual applied partial has a quantitative partner-weight
floor on the physical box. Sources: appendix C's true chain rule
and native clipping at ee02740; no independent gradient is supplied. -/
theorem gain_native_applied_gradient_box_floor (remaining : ℕ) (genGain memGain bound ceiling : ℝ)
    (state : NativeSubweightState) (i : Fin 4) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (horder : memGain ≤ genGain) (hc : 0 ≤ ceiling) (hb : 0 < bound)
    (hp : ∀ j, 0 ≤ (state j).parameter ∧ (state j).parameter ≤ ceiling) :
    gainCEFeedbackFloor remaining genGain memGain bound ceiling * nativeFactorGain genGain memGain i *
      (state (nativeFactorPartner i)).parameter ≤
        -appliedGainNativeGradient remaining genGain memGain bound state i := by
  have hf := gain_ce_gradient_scale_box_floor remaining genGain memGain bound ceiling state hgen hmem horder hc hb hp
  have hg := gain_native_factor_gain_pos genGain memGain hgen hmem i
  have hfirst := mul_le_mul_of_nonneg_right hf (le_of_lt hg)
  have hsecond := mul_le_mul_of_nonneg_right hfirst (hp (nativeFactorPartner i)).1
  rw [gain_native_applied_gradient_scale]
  simpa only [neg_neg] using hsecond

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 100 ∧ (0 : ℝ) < 1 ∧
    ∀ i : Fin 4, 0 ≤ (seededNativeSubweights ((1, 2), (3, 4)) i).parameter ∧
      (seededNativeSubweights ((1, 2), (3, 4)) i).parameter ≤ 100 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

end Transformer.Grokking.CircuitEfficiency

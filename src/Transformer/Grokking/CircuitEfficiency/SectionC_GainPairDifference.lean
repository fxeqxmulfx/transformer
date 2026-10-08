import Transformer.Grokking.CircuitEfficiency.SectionC_GainBalancedStep

/-!
# Difference contraction for unequal native zero-beta factor pairs

Sources: Varma et al., arXiv:2309.02390v1, section 3's unbalanced
zero-first-factor seeds and appendix C's product partials; actual
zero-beta coordinate reduction at lab commit dadb5ac.

Unequal factors receive each other's normalized true partner input.
Their difference after a step is the old difference times an explicit
coefficient. At a shared nonnegative scale at most one, positive
epsilon bounds the partner-input secant slope by gain/epsilon.
The sufficient small-rate condition rate*(decay+gain/epsilon) <= 1
makes the difference coefficient nonnegative and no greater than
the remaining pure-decay factor. Thus unequal pairs contract in
absolute difference without assuming current pair symmetry.

This file is scalar algebra for the already-derived actual-form step;
the next bridge must derive its scale/sign premises from actual CE
and iterate on initialized retained states. No future balance or
parameter convergence is assumed. Zero betas, physical gains and
uniform decoupled decay differ from appendix C's coupled-cost GD
and preserved GPTMini. These are exact-real, fixed-table statements.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- Unequal partner-input pair update. Sources: appendix C product
partials and the actual zero-beta formula at dadb5ac. Its equality
with actual native coordinates remains an explicit bridge obligation. -/
noncomputable def gainPairStep (gain scale eps decay rate a b : ℝ) : ℝ × ℝ :=
  ((1 - rate * decay) * a +
      rate * (scale * gain * b) / (scale * gain * b + eps),
    (1 - rate * decay) * b +
      rate * (scale * gain * a) / (scale * gain * a + eps))

/-- Exact secant slope of the native normalized partner input.
Sources: the positive-epsilon native formula at dadb5ac; it depends
on both present factors and contains no path property or conclusion. -/
noncomputable def gainPairInputSlope (gain scale eps a b : ℝ) : ℝ :=
  scale * gain * eps /
    ((scale * gain * a + eps) * (scale * gain * b + eps))

/-- The actual-form pair difference has the explicit secant coefficient.
Sources: appendix C's two product factors and native normalization
at dadb5ac; neither factor equality nor a future limit is required. -/
theorem gain_pair_step_difference (gain scale eps decay rate a b : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (he : 0 < eps) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    (gainPairStep gain scale eps decay rate a b).1 -
      (gainPairStep gain scale eps decay rate a b).2 =
        (a - b) * (1 - rate * decay - rate * gainPairInputSlope gain scale eps a b) := by
  have hda : 0 < scale * gain * a + eps := by positivity
  have hdb : 0 < scale * gain * b + eps := by positivity
  have hdiff : scale * gain * b / (scale * gain * b + eps) -
      scale * gain * a / (scale * gain * a + eps) =
        (b - a) * gainPairInputSlope gain scale eps a b := by
    unfold gainPairInputSlope
    generalize hdae : scale * gain * a + eps = da at hda ⊢
    generalize hdbe : scale * gain * b + eps = db at hdb ⊢
    field_simp [ne_of_gt hda, ne_of_gt hdb]
    rw [← hdae, ← hdbe]
    ring
  have hgroup : (gainPairStep gain scale eps decay rate a b).1 -
      (gainPairStep gain scale eps decay rate a b).2 =
        (1 - rate * decay) * (a - b) + rate *
          (scale * gain * b / (scale * gain * b + eps) -
            scale * gain * a / (scale * gain * a + eps)) := by
    unfold gainPairStep
    ring
  rw [hgroup, hdiff]
  ring

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by
  norm_num

/-- At the actual CE unit scale the secant slope lies between zero
and gain/epsilon. Source: normalized native inputs at dadb5ac;
the bound permits a sufficient rate condition independent of the path. -/
theorem gain_pair_input_slope_interval (gain scale eps a b : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (hu : scale ≤ 1) (he : 0 < eps)
    (ha : 0 ≤ a) (hb : 0 ≤ b) :
    0 ≤ gainPairInputSlope gain scale eps a b ∧
      gainPairInputSlope gain scale eps a b ≤ gain / eps := by
  have hsa : 0 ≤ scale * gain * a := by positivity
  have hsb : 0 ≤ scale * gain * b := by positivity
  have hda : 0 < scale * gain * a + eps := by positivity
  have hdb : 0 < scale * gain * b + eps := by positivity
  have hden : 0 < (scale * gain * a + eps) * (scale * gain * b + eps) := mul_pos hda hdb
  have hfloor : eps ^ 2 ≤ (scale * gain * a + eps) * (scale * gain * b + eps) := by
    nlinarith only [hsa, hsb, mul_nonneg hsa hsb, he]
  have hgain := mul_le_mul_of_nonneg_right hu hg
  have hnum := mul_le_mul_of_nonneg_right hgain (sq_nonneg eps)
  have hdenGain := mul_le_mul_of_nonneg_left hfloor hg
  constructor
  · unfold gainPairInputSlope
    positivity
  · unfold gainPairInputSlope
    apply (div_le_div_iff₀ hden he).mpr
    nlinarith only [hnum, hdenGain]

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by
  norm_num

/-- A nonempty sufficient small-rate region makes the difference
coefficient nonnegative and no greater than pure decay. Source:
the explicit native secant formula at dadb5ac; this rate restriction
is additional to the source's GD model and PyTorch's legal constants. -/
theorem gain_pair_difference_coefficient_interval (gain scale eps decay rate a b : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (hu : scale ≤ 1) (he : 0 < eps)
    (heta : 0 ≤ rate) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hsmall : rate * (decay + gain / eps) ≤ 1) :
    0 ≤ 1 - rate * decay - rate * gainPairInputSlope gain scale eps a b ∧
      1 - rate * decay - rate * gainPairInputSlope gain scale eps a b ≤
        1 - rate * decay := by
  have hslope := gain_pair_input_slope_interval gain scale eps a b hg hs hu he ha hb
  have hrate := mul_le_mul_of_nonneg_left hslope.2 heta
  have hnn := mul_nonneg heta hslope.1
  constructor <;> nlinarith only [hsmall, hrate, hnn]

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by
  norm_num

/-- An unequal actual-form pair contracts its absolute difference
under that sufficient rate condition. Sources: appendix C's factor
composition and native formula at dadb5ac; it assumes no attained
balance, positive limit, successful task margin or future input stream. -/
theorem gain_pair_step_difference_contraction (gain scale eps decay rate a b : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (hu : scale ≤ 1) (he : 0 < eps)
    (heta : 0 ≤ rate) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hsmall : rate * (decay + gain / eps) ≤ 1) :
    |(gainPairStep gain scale eps decay rate a b).1 -
      (gainPairStep gain scale eps decay rate a b).2| ≤
      (1 - rate * decay) * |a - b| := by
  have hc := gain_pair_difference_coefficient_interval gain scale eps decay rate a b
    hg hs hu he heta ha hb hsmall
  rw [gain_pair_step_difference gain scale eps decay rate a b hg hs he ha hb,
    abs_mul, abs_of_nonneg hc.1]
  simpa only [mul_comm] using mul_le_mul_of_nonneg_left hc.2 (abs_nonneg (a - b))

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by
  norm_num

end Transformer.Grokking.CircuitEfficiency

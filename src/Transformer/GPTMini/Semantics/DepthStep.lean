import Transformer.GPTMini.Semantics.CountSpline

/-!
# Ordinary homogeneous ReLU2 gates for bounded-depth prefix existence

Source: the unchanged bias-free ReLU2_FFN at f11b6e2 and the ordered
subsequence criterion for Basis E_2/E_4 at cbafbe9. Three ordinary
quadratic hinges form a continuous saturating step. A separate linear
type offset excludes a current token of the wrong raw alphabet type.

The computation is given by genuine linear forms and ReLU2, rather than
by a defined Boolean indicator or a prefix oracle. On inputs separated
from zero by a positive finite presence threshold, the output is exactly
zero or a common positive amplitude. Positive homogeneity preserves the
same computation through the actual position-dependent prenorm scale.

This scalar construction fits three ordinary FFN units per detector;
simultaneous matrices, raw encoder recurrence and the complete depth
readout are still needed. No convexity or AdamW success is claimed.
-/

namespace Transformer.GPTMini.Semantics

/-- Three ordinary ReLU2 hinges, using the protected constant coordinate for both threshold shifts.
Source: new continuous quadratic finite-difference step in the original bias-free FFN operator. -/
noncomputable def depthStep (a signal constant : ℝ) : ℝ :=
  relu2 signal - 2 * relu2 (signal - a * constant) + relu2 (signal - 2 * a * constant)

/-- A nonpositive signal has all three actual shifted hinges inactive.
Source: positive threshold times the protected nonnegative constant and the real ReLU2 branches. -/
theorem depthStep_off (a signal constant : ℝ) (ha : 0 ≤ a) (hc : 0 ≤ constant) (hs : signal ≤ 0) :
    depthStep a signal constant = 0 := by
  have hac := mul_nonneg ha hc
  have h1 : signal - a * constant ≤ 0 := by nlinarith
  have h2 : signal - 2 * a * constant ≤ 0 := by nlinarith
  rw [depthStep, relu2_of_nonpos _ hs, relu2_of_nonpos _ h1, relu2_of_nonpos _ h2]
  ring

example : (0 : ℝ) ≤ 1 / 256 ∧ (0 : ℝ) ≤ 1 ∧ (-1 : ℝ) ≤ 0 := by norm_num

/-- Above the positive threshold, the three actual quadratic branches cancel to a constant positive plateau.
Source: ordinary degree-two finite differences; the input is not replaced by a Boolean condition in the definition. -/
theorem depthStep_plateau (a signal constant : ℝ) (ha : 0 ≤ a) (hc : 0 ≤ constant)
    (hs : 2 * a * constant ≤ signal) : depthStep a signal constant = 2 * (a * constant) ^ 2 := by
  have hac := mul_nonneg ha hc
  have h0 : 0 ≤ signal := by nlinarith
  have h1 : 0 ≤ signal - a * constant := by nlinarith
  have h2 : 0 ≤ signal - 2 * a * constant := by nlinarith
  unfold depthStep relu2
  rw [max_eq_right h0, max_eq_right h1, max_eq_right h2]
  ring

example : (0 : ℝ) ≤ 1 / 256 ∧ (0 : ℝ) ≤ 1 ∧ 2 * (1 / 256 : ℝ) * 1 ≤ 1 / 128 := by norm_num

/-- Scaling both real input coordinates preserves exactly the quadratic FFN degree.
Source: ReLU2 positive homogeneity and the ordinary bias-free shifted linear forms. -/
theorem depthStep_scale (a signal constant scale : ℝ) (hscale : 0 ≤ scale) :
    depthStep a (scale * signal) (scale * constant) = scale ^ 2 * depthStep a signal constant := by
  have h1 : scale * signal - a * (scale * constant) = scale * (signal - a * constant) := by ring
  have h2 : scale * signal - 2 * a * (scale * constant) = scale * (signal - 2 * a * constant) := by ring
  unfold depthStep
  rw [h1, h2, relu2_mul_nonneg scale signal hscale, relu2_mul_nonneg scale _ hscale,
    relu2_mul_nonneg scale _ hscale]
  ring

example : (0 : ℝ) ≤ 3 / 19 := by norm_num

/-- The continuous FFN computes an exact discrete presence amplitude on a separated genuine signal.
Source: the derived zero and plateau computations, when actual softmax presence has a positive finite floor. -/
theorem depthStep_separated (a signal constant : ℝ) (ha : 0 < a) (hc : 0 < constant)
    (hs : signal = 0 ∨ 2 * a * constant ≤ signal) :
    depthStep a signal constant = if 0 < signal then 2 * (a * constant) ^ 2 else 0 := by
  rcases hs with hzero | hpositive
  · subst signal
    rw [depthStep_off a 0 constant ha.le hc.le (le_refl 0), ite_eq_right (lt_irrefl 0)]
  · have hp : 0 < signal := by have hac := mul_pos ha hc; nlinarith
    rw [ite_eq_left hp]
    exact depthStep_plateau a signal constant ha.le hc.le hpositive

example : (0 : ℝ) < 1 / 256 ∧ (0 : ℝ) < 1 ∧ ((1 / 128 : ℝ) = 0 ∨ 2 * (1 / 256 : ℝ) * 1 ≤ 1 / 128) := by
  exact ⟨by norm_num, by norm_num, Or.inr (by norm_num)⟩

/-- A type offset is another ordinary linear form in the actual presence, token-type and protected constant coordinates.
Source: new type-conditioned ordered-subsequence detector; no multiplication of learned features is inserted into W_in. -/
noncomputable def depthTypeStep (a cap signal kind constant : ℝ) : ℝ :=
  depthStep a (signal + cap * (kind - constant)) constant

/-- A missing current type suppresses even a positive preceding presence signal under its true upper bound.
Source: the actual linear type offset, with all three quadratic hinges forced inactive. -/
theorem depthTypeStep_wrong_type (a cap signal constant : ℝ) (ha : 0 ≤ a) (hc : 0 ≤ constant)
    (hs : signal ≤ cap * constant) : depthTypeStep a cap signal 0 constant = 0 := by
  unfold depthTypeStep
  apply depthStep_off a _ constant ha hc
  nlinarith

example : (0 : ℝ) ≤ 1 / 256 ∧ (0 : ℝ) ≤ 1 ∧ (1 / 128 : ℝ) ≤ 2 * 1 := by norm_num

/-- The complete type-conditioned linear forms commute with the actual common positive RMS multiplier.
Source: homogeneity of the three genuine shifted hinges, retaining the real token-type and constant coordinates. -/
theorem depthTypeStep_scale (a cap signal kind constant scale : ℝ) (hscale : 0 ≤ scale) :
    depthTypeStep a cap (scale * signal) (scale * kind) (scale * constant) =
      scale ^ 2 * depthTypeStep a cap signal kind constant := by
  have he : scale * signal + cap * (scale * kind - scale * constant) =
      scale * (signal + cap * (kind - constant)) := by ring
  unfold depthTypeStep
  rw [he, depthStep_scale a _ constant scale hscale]

example : (0 : ℝ) ≤ 3 / 19 := by norm_num

/-- Actual separated presence and the raw zero/unit type yield exactly the conjunction needed by the next ordered detector.
Source: derived type exclusion and the true three-hinge plateau; the Boolean result is proved, not defined as the FFN. -/
theorem depthTypeStep_binary (a cap signal kind constant : ℝ) (ha : 0 < a) (hc : 0 < constant)
    (hkind : kind = 0 ∨ kind = constant) (hs : signal = 0 ∨ 2 * a * constant ≤ signal)
    (hcap : signal ≤ cap * constant) : depthTypeStep a cap signal kind constant =
      if kind = constant ∧ 0 < signal then 2 * (a * constant) ^ 2 else 0 := by
  rcases hkind with hzero | hmatch
  · subst kind
    have hne : (0 : ℝ) ≠ constant := by linarith
    rw [depthTypeStep_wrong_type a cap signal constant ha.le hc.le hcap, ite_eq_right (by simp [hne])]
  · subst kind
    have he : depthTypeStep a cap signal constant constant = depthStep a signal constant := by
      unfold depthTypeStep
      ring_nf
    rw [he, depthStep_separated a signal constant ha hc hs]
    simp only [true_and]

example : (0 : ℝ) < 1 / 256 ∧ (0 : ℝ) < 1 ∧ ((1 : ℝ) = 0 ∨ 1 = 1) ∧
    ((1 / 128 : ℝ) = 0 ∨ 2 * (1 / 256 : ℝ) * 1 ≤ 1 / 128) ∧ (1 / 128 : ℝ) ≤ 2 * 1 := by
  exact ⟨by norm_num, by norm_num, Or.inr rfl, Or.inr (by norm_num), by norm_num⟩

/-- The original prenorm/ReLU2 composition still has a strictly positive exact amplitude on a matching true presence.
Source: the common positive real RMS multiplier, genuine type match and quantitative softmax presence floor. -/
theorem depthTypeStep_positive (a cap signal constant scale : ℝ) (ha : 0 < a) (hc : 0 < constant)
    (hscale : 0 < scale) (hs : 2 * a * constant ≤ signal) :
    depthTypeStep a cap (scale * signal) (scale * constant) (scale * constant) =
      2 * (scale * a * constant) ^ 2 ∧ 0 < depthTypeStep a cap (scale * signal) (scale * constant) (scale * constant) := by
  have he : depthTypeStep a cap signal constant constant = 2 * (a * constant) ^ 2 := by
    unfold depthTypeStep
    simpa only [sub_self, mul_zero, add_zero] using depthStep_plateau a signal constant ha.le hc.le hs
  rw [depthTypeStep_scale a cap signal constant constant scale hscale.le, he]
  constructor
  · ring
  · have hsc := sq_pos_of_pos hscale
    have hac := sq_pos_of_pos (mul_pos ha hc)
    nlinarith

example : (0 : ℝ) < 1 / 256 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 3 / 19 ∧
    2 * (1 / 256 : ℝ) * 1 ≤ 1 / 128 := by norm_num

end Transformer.GPTMini.Semantics

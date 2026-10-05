import Transformer.GPTMini.Sparsemax.BoundedGain
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-!
# Smooth bounded score coordinates

Derived score parameterization for arXiv:1602.02068v2, §2.2 and §2.5.
A centered sigmoid maps each independent real parameter onto `(-cap, cap)`.
It has a positive derivative and a smooth inverse there. These are actual
score coordinates, followed by the existing variational sparsemax; no
new weight function or custom backward rule is introduced.

Unlike `QKNormScores` at commit `73f8a0b`, the coordinates here are
independent. This explicit architecture change makes small raw score
transfers accessible to trainable parameters. It does not certify that
a query/key factorization has the same local freedom.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- A centered sigmoid score, with an independent finite parameter.
Derived restriction for arXiv:1602.02068v2, §2.2; this replaces the
query/key score coordinate, not the variational attention weights. -/
def boundedCoordinate (cap parameter : ℝ) : ℝ :=
  cap * (Real.exp parameter - 1) / (Real.exp parameter + 1)

/-- Every finite coordinate lies strictly inside its positive cap.
Source context: the derived score restriction for §2.2 of
arXiv:1602.02068v2; strict bounds permit local parameter inversion. -/
theorem boundedCoordinate_bounds (cap parameter : ℝ) (hc : 0 < cap) :
    -cap < boundedCoordinate cap parameter ∧ boundedCoordinate cap parameter < cap := by
  have he := Real.exp_pos parameter
  have hd : 0 < Real.exp parameter + 1 := by positivity
  unfold boundedCoordinate
  constructor
  · apply (lt_div_iff₀ hd).mpr
    nlinarith [mul_pos hc he]
  · apply (div_lt_iff₀ hd).mpr
    nlinarith

/-- A positive cap inhabits the strict-bounds premise.
Source context: arXiv:1602.02068v2, §2.2, derived score chart. -/
example : -(1 / 8 : ℝ) < boundedCoordinate (1 / 8) 0 ∧
    boundedCoordinate (1 / 8) 0 < (1 / 8 : ℝ) :=
  boundedCoordinate_bounds _ _ (by norm_num)

/-- The chart has its ordinary derivative at every finite parameter.
Source context: §2.5 of arXiv:1602.02068v2, upstream score restriction;
the derivative is computed from the exponential, without a surrogate. -/
theorem boundedCoordinate_hasDerivAt (cap parameter : ℝ) :
    HasDerivAt (boundedCoordinate cap)
      (2 * cap * Real.exp parameter / (Real.exp parameter + 1) ^ 2) parameter := by
  have hn := ((Real.hasDerivAt_exp parameter).sub_const 1).const_mul cap
  have hd := (Real.hasDerivAt_exp parameter).add_const 1
  have hne : Real.exp parameter + 1 ≠ 0 := ne_of_gt (by positivity)
  have h := hn.div hd hne
  convert h using 1 <;> first | rfl | ring

/-- A positive cap gives a strictly positive coordinate derivative.
Source context: the derived score chart for arXiv:1602.02068v2, §2.5. -/
theorem boundedCoordinate_derivative_pos (cap parameter : ℝ) (hc : 0 < cap) :
    0 < 2 * cap * Real.exp parameter / (Real.exp parameter + 1) ^ 2 := by
  positivity

/-- The positive derivative condition is inhabited at finite parameters.
Source context: arXiv:1602.02068v2, §2.5, derived score chart. -/
example : (0 : ℝ) < 2 * (1 / 8) * Real.exp 0 / (Real.exp 0 + 1) ^ 2 :=
  boundedCoordinate_derivative_pos _ _ (by norm_num)

/-- The inverse chart on the open score interval.
Derived parameter lift for arXiv:1602.02068v2, §2.5's active directions;
outside this interval no inverse property is asserted. -/
def inverseCoordinate (cap value : ℝ) : ℝ :=
  Real.log ((cap + value) / (cap - value))

/-- A score strictly inside the cap is recovered by the chart.
Source context: the derived bounded parameterization for §2.2 and
§2.5 of arXiv:1602.02068v2; this is a two-sided real calculation. -/
theorem boundedCoordinate_inverse (cap value : ℝ)
    (hl : -cap < value) (hu : value < cap) :
    boundedCoordinate cap (inverseCoordinate cap value) = value := by
  have hn : 0 < cap + value := by linarith
  have hd : 0 < cap - value := by linarith
  have hr : 0 < (cap + value) / (cap - value) := div_pos hn hd
  unfold boundedCoordinate inverseCoordinate
  rw [Real.exp_log hr]
  have hs : (cap + value) / (cap - value) + 1 ≠ 0 := ne_of_gt (by positivity)
  field_simp
  nlinarith

/-- Both inverse premises are realized by a finite bounded coordinate.
Source context: arXiv:1602.02068v2, §2.2, the derived score chart. -/
example : boundedCoordinate (1 / 8) (inverseCoordinate (1 / 8) 0) = 0 :=
  boundedCoordinate_inverse _ _ (by norm_num) (by norm_num)

/-- Inverting an actual chart value recovers its finite parameter.
Source context: §2.5 of arXiv:1602.02068v2, accessibility of score
directions for this independent-coordinate architecture. -/
theorem inverseCoordinate_bounded (cap parameter : ℝ) (hc : 0 < cap) :
    inverseCoordinate cap (boundedCoordinate cap parameter) = parameter := by
  have he : 0 < Real.exp parameter := Real.exp_pos _
  have hd : Real.exp parameter + 1 ≠ 0 := ne_of_gt (by positivity)
  have hcne : cap ≠ 0 := ne_of_gt hc
  have hden : cap - boundedCoordinate cap parameter ≠ 0 :=
    ne_of_gt (by linarith [(boundedCoordinate_bounds cap parameter hc).2])
  have hr : (cap + boundedCoordinate cap parameter) /
      (cap - boundedCoordinate cap parameter) = Real.exp parameter := by
    unfold boundedCoordinate at *
    field_simp
    ring
  rw [inverseCoordinate, hr, Real.log_exp]

/-- Finite parameters inhabit the parameter-recovery premise.
Source context: the derived chart for arXiv:1602.02068v2, §2.5. -/
example : inverseCoordinate (1 / 8) (boundedCoordinate (1 / 8) 3) = 3 :=
  inverseCoordinate_bounded _ _ (by norm_num)

/-- The inverse is differentiable throughout the open score interval.
Source context: the derived score lift for §2.5 of arXiv:1602.02068v2;
only the ordinary real logarithm and quotient are used. -/
theorem inverseCoordinate_differentiableAt (cap value : ℝ)
    (hl : -cap < value) (hu : value < cap) :
    DifferentiableAt ℝ (inverseCoordinate cap) value := by
  have hn : 0 < cap + value := by linarith
  have hd : 0 < cap - value := by linarith
  have hnum : DifferentiableAt ℝ (fun x : ℝ => cap + x) value :=
    (differentiableAt_const cap).add differentiableAt_id
  have hden : DifferentiableAt ℝ (fun x : ℝ => cap - x) value :=
    (differentiableAt_const cap).sub differentiableAt_id
  have hr := hnum.div hden (ne_of_gt hd)
  unfold inverseCoordinate
  exact hr.log (ne_of_gt (div_pos hn hd))

/-- The inverse differentiability premises hold at a chart value.
Source context: arXiv:1602.02068v2, §2.5, derived parameter lift. -/
example : DifferentiableAt ℝ (inverseCoordinate (1 / 8)) 0 :=
  inverseCoordinate_differentiableAt _ _ (by norm_num) (by norm_num)

/-- A lifted score path approaches the original parameter continuously.
Source context: §2.5 of arXiv:1602.02068v2, the derived inverse chart;
this property is needed to compare local task-loss derivatives. -/
theorem inverseCoordinate_continuousAt (cap value : ℝ)
    (hl : -cap < value) (hu : value < cap) :
    ContinuousAt (inverseCoordinate cap) value :=
  (inverseCoordinate_differentiableAt cap value hl hu).continuousAt

/-- A finite chart value inhabits the local continuity hypotheses.
Source context: arXiv:1602.02068v2, §2.5, derived parameter lift. -/
example : ContinuousAt (inverseCoordinate (1 / 8)) 0 :=
  inverseCoordinate_continuousAt _ _ (by norm_num) (by norm_num)

end Transformer.GPTMini.Sparsemax

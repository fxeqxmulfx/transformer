/-
# Scalar coefficients of a Dirac characteristic flow

For the stationary solution `μ(t) = δ_w` of `eq: mean.field.pde`, the
normalized velocity is `Proj_z w`. Write `c = ⟨z,w⟩` and `p = z - cw`.
The explicit trajectory is `C(t,c) w + S(t,c) p`, where
`D = (1+c) exp(2t) + (1-c)`, `C = ((1+c) exp(2t)-(1-c))/D`,
and `S = 2 exp(t)/D`. The denominator is positive even at `c = ±1`.
These formulas apply at every real time, including the initial time.

The identities `C² + (1-c²) S² = 1`, `C' = 1-C²` and `S' = -CS`
prove that this is a sphere-valued solution. Monotonicity in the initial
coordinate will identify the minimizer of each transported cap.
This is auxiliary to arXiv:2410.06833v1, §5, `eq: flow.map` and the
counterexample to `eq: v.small`; it is not an assumed ODE solver.
-/

import Transformer.Metastability.MeanField
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Inv

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Metastability

open Perspective

/-- Positive denominator of the explicit flow for `eq: flow.map` at a Dirac law.

Source: arXiv:2410.06833v1, §5. -/
noncomputable def diracDen (t c : ℝ) : ℝ :=
  (1 + c) * Real.exp (2 * t) + (1 - c)

/-- Coordinate along the stationary mass in the explicit flow of `eq: flow.map`.

Source: arXiv:2410.06833v1, §5. -/
noncomputable def diracAlong (t c : ℝ) : ℝ :=
  ((1 + c) * Real.exp (2 * t) - (1 - c)) / diracDen t c

/-- Multiplier of the initial orthogonal component in `eq: flow.map`.

Source: arXiv:2410.06833v1, §5. -/
noncomputable def diracAcross (t c : ℝ) : ℝ :=
  2 * Real.exp t / diracDen t c

/-- The characteristic-flow denominator never vanishes for a unit initial coordinate.

Source: arXiv:2410.06833v1, §5. -/
theorem diracDen_pos (t c : ℝ) (hc : c ∈ Set.Icc (-1 : ℝ) 1) :
    0 < diracDen t c := by
  rcases hc with ⟨hlo, hhi⟩
  by_cases h : c = -1
  · simp [diracDen, h]
  · have ha : 0 < 1 + c := by
      by_contra hn
      apply h
      linarith
    have hb : 0 ≤ 1 - c := by linarith
    exact add_pos_of_pos_of_nonneg (mul_pos ha (Real.exp_pos _)) hb

/-- The flow of `eq: flow.map` starts at its initial longitudinal coordinate.

Source: arXiv:2410.06833v1, §5. -/
theorem diracAlong_zero (c : ℝ) : diracAlong 0 c = c := by
  simp [diracAlong, diracDen]
  ring

/-- The initial orthogonal component is unchanged at time zero in `eq: flow.map`.

Source: arXiv:2410.06833v1, §5. -/
theorem diracAcross_zero (c : ℝ) : diracAcross 0 c = 1 := by
  norm_num [diracAcross, diracDen]

/-- The stationary center keeps longitudinal coordinate one in `eq: flow.map`.

Source: arXiv:2410.06833v1, §5. -/
theorem diracAlong_one (t : ℝ) : diracAlong t 1 = 1 := by
  simp [diracAlong, diracDen, Real.exp_ne_zero]

/-- The two coefficients preserve squared norm in the flow of `eq: flow.map`.

Source: arXiv:2410.06833v1, §5. -/
theorem dirac_coefficients_unit (t c : ℝ) (hc : c ∈ Set.Icc (-1 : ℝ) 1) :
    diracAlong t c ^ 2 + diracAcross t c ^ 2 * (1 - c ^ 2) = 1 := by
  have hd := ne_of_gt (diracDen_pos t c hc)
  unfold diracAlong diracAcross
  field_simp
  unfold diracDen
  have he : Real.exp (2 * t) = Real.exp t ^ 2 := by
    simpa using Real.exp_nat_mul t 2
  rw [he]
  ring

/-- Derivative of the denominator used to solve `eq: flow.map`.

Source: arXiv:2410.06833v1, §5. -/
theorem hasDerivAt_diracDen (t c : ℝ) :
    HasDerivAt (fun s => diracDen s c) (2 * (1 + c) * Real.exp (2 * t)) t := by
  have he := (Real.hasDerivAt_exp (2 * t)).comp t ((hasDerivAt_id t).const_mul 2)
  convert (he.const_mul (1 + c)).add_const (1 - c) using 1
  dsimp [diracDen]
  ring

/-- The longitudinal equation for `eq: flow.map` at a Dirac mass is `C′ = 1-C²`.

Source: arXiv:2410.06833v1, §5. -/
theorem hasDerivAt_diracAlong (t c : ℝ) (hc : c ∈ Set.Icc (-1 : ℝ) 1) :
    HasDerivAt (fun s => diracAlong s c) (1 - diracAlong t c ^ 2) t := by
  have he := (Real.hasDerivAt_exp (2 * t)).comp t ((hasDerivAt_id t).const_mul 2)
  have hn := (he.const_mul (1 + c)).sub_const (1 - c)
  have hd := ne_of_gt (diracDen_pos t c hc)
  have h := hn.div (hasDerivAt_diracDen t c) (ne_of_gt (diracDen_pos t c hc))
  convert h using 1
  · rfl
  · unfold diracAlong
    dsimp only [Function.comp_apply]
    field_simp [hd]
    unfold diracDen
    ring

/-- The orthogonal equation for `eq: flow.map` at a Dirac mass is `S′ = -CS`.

Source: arXiv:2410.06833v1, §5. -/
theorem hasDerivAt_diracAcross (t c : ℝ) (hc : c ∈ Set.Icc (-1 : ℝ) 1) :
    HasDerivAt (fun s => diracAcross s c) (-diracAlong t c * diracAcross t c) t := by
  have hd := ne_of_gt (diracDen_pos t c hc)
  have h := ((Real.hasDerivAt_exp t).const_mul 2).div
    (hasDerivAt_diracDen t c) (ne_of_gt (diracDen_pos t c hc))
  convert h using 1
  · rfl
  · unfold diracAlong diracAcross
    field_simp [hd]
    unfold diracDen
    ring

/-- Exact difference of longitudinal trajectories in `eq: flow.map`.

Source: arXiv:2410.06833v1, §5. -/
theorem diracAlong_sub (t a b : ℝ)
    (ha : a ∈ Set.Icc (-1 : ℝ) 1) (hb : b ∈ Set.Icc (-1 : ℝ) 1) :
    diracAlong t b - diracAlong t a =
      4 * Real.exp (2 * t) * (b - a) / (diracDen t b * diracDen t a) := by
  have hda := ne_of_gt (diracDen_pos t a ha)
  have hdb := ne_of_gt (diracDen_pos t b hb)
  unfold diracAlong
  field_simp
  unfold diracDen
  ring

/-- The Dirac characteristic flow preserves ordering of longitudinal coordinates.

Source: arXiv:2410.06833v1, §5. -/
theorem diracAlong_mono (t a b : ℝ)
    (ha : a ∈ Set.Icc (-1 : ℝ) 1) (hb : b ∈ Set.Icc (-1 : ℝ) 1) (hab : a ≤ b) :
    diracAlong t a ≤ diracAlong t b := by
  apply sub_nonneg.mp
  rw [diracAlong_sub t a b ha hb]
  exact div_nonneg (mul_nonneg (by positivity) (sub_nonneg.mpr hab))
    (le_of_lt (mul_pos (diracDen_pos t b hb) (diracDen_pos t a ha)))

/-- A coordinate strictly between the poles satisfies the denominator hypothesis. -/
example : (0 : ℝ) ∈ Set.Icc (-1 : ℝ) 1 ∧ 0 < diracDen 0 0 :=
  ⟨by norm_num, diracDen_pos 0 0 (by norm_num)⟩

/-- The unit identity's coordinate hypothesis is attained on the equator. -/
example : (0 : ℝ) ∈ Set.Icc (-1 : ℝ) 1 ∧
    diracAlong 0 0 ^ 2 + diracAcross 0 0 ^ 2 * (1 - (0 : ℝ) ^ 2) = 1 :=
  ⟨by norm_num, dirac_coefficients_unit 0 0 (by norm_num)⟩

/-- Both derivative statements have an admissible initial coordinate. -/
example : (0 : ℝ) ∈ Set.Icc (-1 : ℝ) 1 ∧
    HasDerivAt (fun s => diracAlong s 0) (1 - diracAlong 0 0 ^ 2) 0 ∧
    HasDerivAt (fun s => diracAcross s 0) (-diracAlong 0 0 * diracAcross 0 0) 0 :=
  ⟨by norm_num, hasDerivAt_diracAlong 0 0 (by norm_num),
    hasDerivAt_diracAcross 0 0 (by norm_num)⟩

/-- The difference and monotonicity hypotheses allow distinct coordinates. -/
example : (0 : ℝ) ∈ Set.Icc (-1 : ℝ) 1 ∧ (1 : ℝ) ∈ Set.Icc (-1 : ℝ) 1 ∧
    (0 : ℝ) ≤ 1 ∧ diracAlong 0 1 - diracAlong 0 0 =
      4 * Real.exp (2 * 0) * (1 - 0) / (diracDen 0 1 * diracDen 0 0) ∧
    diracAlong 0 0 ≤ diracAlong 0 1 :=
  ⟨by norm_num, by norm_num, by norm_num,
    diracAlong_sub 0 0 1 (by norm_num) (by norm_num),
    diracAlong_mono 0 0 1 (by norm_num) (by norm_num) (by norm_num)⟩

end Transformer.Metastability

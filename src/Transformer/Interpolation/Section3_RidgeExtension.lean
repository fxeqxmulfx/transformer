/-
# A complete ambient extension of the actual one-neuron field

Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2,
and the original vector field `eq: vf` in §1.

The actual spherical field is `max(b-⟨x,u⟩,0) • proj x p`.
Multiplication by `4‖x‖/(1+‖x‖²)²` gives an ambient extension with
linear growth. This factor is exactly one on the unit sphere.
It also removes the constant drift at the origin, allowing the global
ODE theorem for fields with homogeneous linear growth to apply.

The norm defect satisfies a scalar homogeneous ODE, so a unit initial
point stays on the sphere. Thus the extension constructs trajectories
of the original network field without changing its spherical dynamics.
-/

import Transformer.Interpolation.Section3_PerceptronVariation
import Transformer.Perspective.Section2_SphereFlow

open Real MeasureTheory Filter Set

namespace Transformer.Interpolation

/-- Ambient cutoff, equal to one on the unit sphere. Auxiliary for
arXiv:2411.04551v3, §3.2; the source's spherical field is preserved. -/
noncomputable def ridgeExtensionScale {d : ℕ} (x : EucSpace d) : ℝ :=
  4 * ‖x‖ / (1 + ‖x‖ ^ 2) ^ 2

/-- A complete ambient extension of the actual single-neuron field in
arXiv:2411.04551v3, §1, `eq: vf`. The cutoff only changes off-sphere values. -/
noncomputable def ridgeGlobalField {d : ℕ} (p u : EucSpace d) (b : ℝ) (x : EucSpace d) : EucSpace d :=
  (ridgeExtensionScale x * max (b - inner ℝ x u) 0) • proj d x p

/-- On a unit vector the extension is the original spherical field.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem ridgeGlobalField_eq_of_unit {d : ℕ} (p u : EucSpace d) (b : ℝ)
    (x : EucSpace d) (hx : ‖x‖ = 1) :
    ridgeGlobalField p u b x = max (b - inner ℝ x u) 0 • proj d x p := by
  norm_num [ridgeGlobalField, ridgeExtensionScale, hx]

/-- The extended field is locally Lipschitz, including at the origin
and at the ReLU kink. Auxiliary for arXiv:2411.04551v3, §3.2. -/
theorem locallyLipschitz_ridgeGlobalField {d : ℕ} (p u : EucSpace d) (b : ℝ) :
    LocallyLipschitz (ridgeGlobalField p u b) := by
  have hden : ContDiff ℝ 1 (fun x : EucSpace d => 4 / (1 + ‖x‖ ^ 2) ^ 2) :=
    contDiff_const.div ((contDiff_const.add (contDiff_norm_sq ℝ)).pow 2) (fun x => by positivity)
  have hnorm : LocallyLipschitz (fun x : EucSpace d => ‖x‖) := by
    have hn : LipschitzWith 1 (fun x : EucSpace d => ‖x‖) := by
      rw [lipschitzWith_iff_dist_le_mul]
      intro x y
      simpa only [NNReal.coe_one, one_mul, dist_eq_norm] using dist_norm_norm_le x y
    exact hn.locallyLipschitz
  have hs : LocallyLipschitz (fun x : EucSpace d => ridgeExtensionScale x * max (b - inner ℝ x u) 0) := by
    have hlin : ContDiff ℝ 1 (fun x : EucSpace d => b - inner ℝ x u) :=
      contDiff_const.sub (contDiff_id.inner ℝ contDiff_const)
    have hmul : LocallyLipschitz (fun y : ℝ × ℝ => y.1 * y.2) :=
      (by fun_prop : ContDiff ℝ 1 (fun y : ℝ × ℝ => y.1 * y.2)).locallyLipschitz
    have hfirst : LocallyLipschitz (fun x : EucSpace d => (4 / (1 + ‖x‖ ^ 2) ^ 2) * ‖x‖) :=
      hmul.comp (hden.locallyLipschitz.prodMk hnorm)
    have ht : LocallyLipschitz (fun x : EucSpace d =>
        (4 / (1 + ‖x‖ ^ 2) ^ 2) * ‖x‖ * max (b - inner ℝ x u) 0) :=
      hmul.comp (hfirst.prodMk (hlin.locallyLipschitz.max_const 0))
    have heq : (ridgeExtensionScale : EucSpace d → ℝ) =
        fun x => (4 / (1 + ‖x‖ ^ 2) ^ 2) * ‖x‖ := by
      funext x
      dsimp [ridgeExtensionScale]
      ring
    rw [heq]
    exact ht
  have hp : ContDiff ℝ 1 (fun x : EucSpace d => proj d x p) :=
    contDiff_const.sub ((contDiff_id.inner ℝ contDiff_const).smul contDiff_id)
  have hsmul : ContDiff ℝ 1 (fun y : ℝ × EucSpace d => y.1 • y.2) := by fun_prop
  change LocallyLipschitz (fun x : EucSpace d =>
    (ridgeExtensionScale x * max (b - inner ℝ x u) 0) • proj d x p)
  have hsmulL : LocallyLipschitz (fun y : ℝ × EucSpace d => y.1 • y.2) :=
    hsmul.locallyLipschitz
  have hpL : LocallyLipschitz (fun x : EucSpace d => proj d x p) := hp.locallyLipschitz
  have hpair : LocallyLipschitz (fun x : EucSpace d =>
      (ridgeExtensionScale x * max (b - inner ℝ x u) 0, proj d x p)) := hs.prodMk hpL
  have hcomp := hsmulL.comp hpair
  exact hcomp

/-- The original ridge has a global ambient extension of linear growth.
Auxiliary existence estimate for arXiv:2411.04551v3, §3.2. -/
theorem norm_ridgeGlobalField_le {d : ℕ} (p u : EucSpace d) (b : ℝ) (x : EucSpace d) :
    ‖ridgeGlobalField p u b x‖ ≤ 4 * (|b| + ‖u‖) * ‖p‖ * ‖x‖ := by
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  have hd : 0 < 1 + ‖x‖ ^ 2 := by positivity
  have hcoef : 0 ≤ ridgeExtensionScale x := by unfold ridgeExtensionScale; positivity
  have hw : max (b - inner ℝ x u) 0 ≤ |b| + ‖u‖ * ‖x‖ := by
    apply max_le
    · have hi := abs_real_inner_le_norm x u
      have hb := le_abs_self b
      have hminus := neg_le_abs (inner ℝ x u)
      nlinarith
    · positivity
  have hp : ‖proj d x p‖ ≤ (1 + ‖x‖ ^ 2) * ‖p‖ := by
    calc
      ‖proj d x p‖ ≤ ‖p‖ + ‖inner ℝ x p • x‖ := norm_sub_le _ _
      _ = ‖p‖ + |inner ℝ x p| * ‖x‖ := by rw [norm_smul, Real.norm_eq_abs]
      _ ≤ ‖p‖ + (‖x‖ * ‖p‖) * ‖x‖ := by gcongr; exact abs_real_inner_le_norm _ _
      _ = _ := by ring
  have hfrac : (|b| + ‖u‖ * ‖x‖) / (1 + ‖x‖ ^ 2) ≤ |b| + ‖u‖ := by
    apply (div_le_iff₀ hd).mpr
    have hx : ‖x‖ ≤ 1 + ‖x‖ ^ 2 := by nlinarith [sq_nonneg (‖x‖ - 1 / 2)]
    nlinarith [mul_nonneg (norm_nonneg u) (sub_nonneg.mpr hx), mul_nonneg (abs_nonneg b) (sq_nonneg ‖x‖)]
  calc
    ‖ridgeGlobalField p u b x‖ = ridgeExtensionScale x * max (b - inner ℝ x u) 0 * ‖proj d x p‖ := by
      rw [ridgeGlobalField, norm_smul, Real.norm_eq_abs, abs_of_nonneg (mul_nonneg hcoef (le_max_right _ _))]
    _ ≤ ridgeExtensionScale x * (|b| + ‖u‖ * ‖x‖) * ((1 + ‖x‖ ^ 2) * ‖p‖) := by gcongr
    _ = 4 * ‖x‖ * ((|b| + ‖u‖ * ‖x‖) / (1 + ‖x‖ ^ 2)) * ‖p‖ := by
      unfold ridgeExtensionScale
      field_simp
    _ ≤ 4 * (|b| + ‖u‖) * ‖p‖ * ‖x‖ := by
      have hm := mul_le_mul_of_nonneg_left hfrac (by positivity : 0 ≤ 4 * ‖x‖)
      have hm' := mul_le_mul_of_nonneg_right hm (norm_nonneg p)
      nlinarith

/-- A unit initial point stays on the sphere along every actual ambient
solution. The squared-norm defect solves a homogeneous scalar ODE.
Source: the tangent projection in arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem norm_eq_one_of_ridgeGlobalField {d : ℕ} (p u : EucSpace d) (b : ℝ)
    (Y : ℝ → EucSpace d) (hY : ∀ t : ℝ, HasDerivAt Y (ridgeGlobalField p u b (Y t)) t)
    (h0 : ‖Y 0‖ = 1) (t : ℝ) : ‖Y t‖ = 1 := by
  have hc : Continuous Y := continuous_iff_continuousAt.mpr fun s => (hY s).continuousAt
  let a (s : ℝ) := -2 * (ridgeExtensionScale (Y s) * max (b - inner ℝ (Y s) u) 0) *
    inner ℝ (Y s) p
  have ha : Continuous a := by
    have hs : Continuous (fun s => ridgeExtensionScale (Y s)) :=
      (continuous_const.mul hc.norm).div
        ((continuous_const.add (hc.norm.pow 2)).pow 2)
        (fun s => pow_ne_zero _ (by positivity))
    have hr : Continuous (fun s => max (b - inner ℝ (Y s) u) 0) :=
      (continuous_const.sub (hc.inner continuous_const)).max continuous_const
    exact (continuous_const.mul (hs.mul hr)).mul (hc.inner continuous_const)
  have hd (s : ℝ) : HasDerivAt (fun r => ‖Y r‖ ^ 2 - 1)
      (a s * (‖Y s‖ ^ 2 - 1)) s := by
    convert (hY s).norm_sq.sub_const 1 using 1
    simp only [ridgeGlobalField, real_inner_smul_right, proj, inner_sub_right,
      real_inner_self_eq_norm_sq]
    dsimp [a]
    ring
  have hz := Perspective.scalar_eq_zero_of_linearODE a _ ha hd (by rw [h0]; norm_num) t
  nlinarith [norm_nonneg (Y t)]

/-- A nonzero unit initial point and an actual stationary trajectory jointly
satisfy the unit-vector and ODE premises above. The neuron has a positive
activation; its tangent projection vanishes at its output direction. -/
example : ‖(basePoint 0 : EucSpace 1)‖ = 1 ∧
    (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (basePoint 0 : EucSpace 1))
      (ridgeGlobalField (basePoint 0 : EucSpace 1) 0 1 (basePoint 0)) t) := by
  have hunit : ‖(basePoint 0 : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hfield : ridgeGlobalField (basePoint 0 : EucSpace 1) 0 1 (basePoint 0) = 0 := by
    rw [ridgeGlobalField_eq_of_unit _ _ _ _ hunit]
    simp [proj, hunit]
  refine ⟨hunit, fun t => ?_⟩
  rw [hfield]
  exact hasDerivAt_const t _

end Transformer.Interpolation

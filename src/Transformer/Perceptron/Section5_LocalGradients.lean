/-
# Localized smooth gradients on the actual circle

ArXiv:2601.21366v2, §5.2, the localized tangent directions in
the proof of `thm: circle.gelu`. Multiply an ambient linear
function by a smooth bump centered at an actual support point.
Near the center the bump is one, so the actual gradient has the
prescribed direction. Outside its closed support the function
is locally zero, so its actual gradient vanishes.

Choosing the center's unit angular tangent gives scalar speed one
at the center. The resulting speed is continuous and is a genuine
smooth spherical gradient direction, supported inside any requested
positive-radius ball or open neighborhood. No support finiteness
or density of the measure is assumed in this construction.
-/

import Transformer.Perceptron.Section5_GradientFields
import Mathlib.Analysis.Calculus.BumpFunction.FiniteDimension

open Real Filter Set Metric
open scoped Topology

namespace Transformer.Perceptron

/-- The localized linear test potential is actually smooth. Source:
arXiv:2601.21366v2, §5.2, the smooth functions `uᵢ` generating
the localized directions in the proof of `thm: circle.gelu`. -/
theorem contDiff_bump_mul_inner (y v : EucSpace 2) (b : ContDiffBump y) :
    ContDiff ℝ (⊤ : ℕ∞) (fun z => b z * inner (𝕜 := ℝ) v z) :=
  b.contDiff.mul (innerSL ℝ v).contDiff

/-- At the center the actual gradient is the prescribed vector.
Source: arXiv:2601.21366v2, §5.2, the choice of nonvanishing
localized derivatives at points of the support. -/
theorem gradient_bump_mul_inner_center (y v : EucSpace 2) (b : ContDiffBump y) :
    gradient (fun z => b z * inner (𝕜 := ℝ) v z) y = v := by
  have heq : (fun z => b z * inner (𝕜 := ℝ) v z) =ᶠ[𝓝 y]
      (fun z => inner (𝕜 := ℝ) v z) := by
    filter_upwards [b.eventuallyEq_one] with z hz
    simp only [hz, Pi.one_apply, one_mul]
  have hlin : HasGradientAt (fun z => inner (𝕜 := ℝ) v z) v y := by
    rw [hasGradientAt_iff_hasFDerivAt]
    exact (innerSL ℝ v).hasFDerivAt
  calc
    _ = gradient (fun z => inner (𝕜 := ℝ) v z) y := by
      unfold gradient
      rw [heq.fderiv_eq]
    _ = v := hlin.gradient

/-- Outside the bump's closed support the actual gradient is zero.
Source: arXiv:2601.21366v2, §5.2, extending the local smooth
test functions by zero outside their arcs. -/
theorem gradient_bump_mul_inner_zero (y v : EucSpace 2) (b : ContDiffBump y)
    (x : EucSpace 2) (hx : x ∉ closedBall y b.rOut) :
    gradient (fun z => b z * inner (𝕜 := ℝ) v z) x = 0 := by
  have hb : (b : EucSpace 2 → ℝ) =ᶠ[𝓝 x] 0 :=
    notMem_tsupport_iff_eventuallyEq.mp (by rwa [b.tsupport_eq])
  have heq : (fun z => b z * inner (𝕜 := ℝ) v z) =ᶠ[𝓝 x] (fun _ => (0 : ℝ)) := by
    filter_upwards [hb] with z hz
    simp only [hz, Pi.zero_apply, zero_mul]
  unfold gradient
  rw [heq.fderiv_eq]
  simp

/-- Choosing the center's actual unit tangent gives angular speed one.
Source: arXiv:2601.21366v2, §5.2, ensuring a nonzero `L²` norm
for each localized gradient direction. -/
theorem circleGradientSpeed_bump_center (y : SSphere 2)
    (b : ContDiffBump (y : EucSpace 2)) :
    circleGradientSpeed (fun z => b z * inner (𝕜 := ℝ)
      (circleTangent (y : EucSpace 2)) z) y = 1 := by
  rw [circleGradientSpeed, gradient_bump_mul_inner_center,
    real_inner_self_eq_norm_sq, norm_circleTangent, mem_sphere_zero_iff_norm.mp y.2, one_pow]

/-- The actual spherical gradient has unit norm at its center.
Source: arXiv:2601.21366v2, §5.2, ensuring that localized
gradient fields do not vanish at their chosen support points. -/
theorem norm_bump_gradient_center (y : SSphere 2)
    (b : ContDiffBump (y : EucSpace 2)) :
    ‖proj 2 (y : EucSpace 2) (gradient (fun z => b z * inner (𝕜 := ℝ)
      (circleTangent (y : EucSpace 2)) z) (y : EucSpace 2))‖ = 1 := by
  rw [proj_gradient_eq_circleGradientSpeed, circleGradientSpeed_bump_center,
    one_smul, norm_circleTangent, mem_sphere_zero_iff_norm.mp y.2]

/-- An actual smooth gradient direction can be localized to any
positive-radius ball, with speed one at its actual center.
Source: arXiv:2601.21366v2, §5.2, the directions `ηᵢ` in the
proof of `thm: circle.gelu`. The bump is constructed explicitly. -/
theorem exists_local_circle_gradient (y : SSphere 2) (r : ℝ) (hr : 0 < r) :
    ∃ s : SSphere 2 → ℝ, Continuous s ∧
      IsGradientField (fun x => s x • circleTangent (x : EucSpace 2)) ∧
      s y = 1 ∧ ∀ x : SSphere 2, r ≤ dist x y → s x = 0 := by
  let b : ContDiffBump (y : EucSpace 2) :=
    { rIn := r / 4
      rOut := r / 2
      rIn_pos := by positivity
      rIn_lt_rOut := by linarith }
  let f : EucSpace 2 → ℝ := fun z => b z * inner (𝕜 := ℝ) (circleTangent (y : EucSpace 2)) z
  have hf : ContDiff ℝ (⊤ : ℕ∞) f := contDiff_bump_mul_inner _ _ b
  refine ⟨circleGradientSpeed f, continuous_circleGradientSpeed f hf,
    isGradientField_circleGradientSpeed f hf, circleGradientSpeed_bump_center y b, ?_⟩
  intro x hx
  have hxout : (x : EucSpace 2) ∉ closedBall (y : EucSpace 2) b.rOut := by
    rw [mem_closedBall]
    change ¬ dist (x : EucSpace 2) (y : EucSpace 2) ≤ r / 2
    change r ≤ dist (x : EucSpace 2) (y : EucSpace 2) at hx
    linarith
  unfold circleGradientSpeed
  rw [gradient_bump_mul_inner_zero (y : EucSpace 2) _ b (x : EucSpace 2) hxout]
  exact inner_zero_right _

/-- The localized speed has its actual function support inside the ball.
Source: arXiv:2601.21366v2, §5.2, the compactly localized
directions in the proof of `thm: circle.gelu`. -/
theorem exists_local_circle_gradient_in_ball (y : SSphere 2) (r : ℝ) (hr : 0 < r) :
    ∃ s : SSphere 2 → ℝ, Continuous s ∧
      IsGradientField (fun x => s x • circleTangent (x : EucSpace 2)) ∧
      s y = 1 ∧ Function.support s ⊆ ball y r := by
  obtain ⟨s, hs, hgrad, hsy, hout⟩ := exists_local_circle_gradient y r hr
  refine ⟨s, hs, hgrad, hsy, fun x hx => ?_⟩
  change dist x y < r
  apply lt_of_not_ge
  intro hdist
  exact (Function.mem_support.mp hx) (hout x hdist)

/-- The same construction is supported inside any open neighborhood.
Source: arXiv:2601.21366v2, §5.2, the small subarcs carrying
the localized gradient directions. Nonempty support is ensured
by speed one at the specified center. -/
theorem exists_local_circle_gradient_in_open (U : Set (SSphere 2)) (hU : IsOpen U)
    (y : SSphere 2) (hy : y ∈ U) :
    ∃ s : SSphere 2 → ℝ, Continuous s ∧
      IsGradientField (fun x => s x • circleTangent (x : EucSpace 2)) ∧
      s y = 1 ∧ Function.support s ⊆ U := by
  obtain ⟨r, hr, hball⟩ := isOpen_iff.mp hU y hy
  obtain ⟨s, hs, hgrad, hsy, hsupp⟩ := exists_local_circle_gradient_in_ball y r hr
  exact ⟨s, hs, hgrad, hsy, hsupp.trans hball⟩

/-- The outside-support input is satisfiable for an explicit smooth
bump with radii one and two, at a point of Euclidean norm three. -/
example : ∃ b : ContDiffBump (0 : EucSpace 2),
    (3 : ℝ) • ((basePoint 1 : SSphere 2) : EucSpace 2) ∉ closedBall 0 b.rOut := by
  refine ⟨{ rIn := 1, rOut := 2, rIn_pos := one_pos, rIn_lt_rOut := by norm_num }, ?_⟩
  change ¬ dist ((3 : ℝ) • ((basePoint 1 : SSphere 2) : EucSpace 2)) 0 ≤ 2
  rw [dist_zero_right, norm_smul, norm_basePoint_one, Real.norm_eq_abs]
  norm_num

/-- All radius and open-neighborhood hypotheses are jointly realized
at an actual point of the nonempty circle and its full open neighborhood. -/
example : (0 : ℝ) < 1 ∧ IsOpen (univ : Set (SSphere 2)) ∧
    (basePoint 1 : SSphere 2) ∈ (univ : Set (SSphere 2)) :=
  ⟨one_pos, isOpen_univ, mem_univ _⟩

end Transformer.Perceptron

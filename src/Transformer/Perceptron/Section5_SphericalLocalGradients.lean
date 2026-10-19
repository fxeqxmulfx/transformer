module

public import Transformer.Perceptron.Section5_SphereTransport
public import Transformer.Perceptron.Hyperplane
public import Mathlib.Analysis.Calculus.BumpFunction.FiniteDimension

/-!
# Actual localized smooth spherical gradients in every dimension

ArXiv:2601.21366v2, §5.3, the localization argument for
`thm: any.d` (i), extending the smooth test functions in §5.2.
An actual smooth ambient bump multiplies an actual linear
potential. At its center the original gradient equals the
prescribed tangent vector; outside the bump's closed support it
vanishes because the original test function is locally zero.

For every sphere point in dimension at least two there is a unit
tangent vector. Projecting the actual bump potential's gradient
therefore gives an original smooth spherical gradient field with
unit norm at that point, supported in any requested positive-radius
ball. This construction makes no assumption about measure support
or about the existence of a Hessian.
-/

@[expose] public section

open Real Filter Set Metric
open scoped Topology

namespace Transformer.Perceptron

/-- The actual localized linear potential is smooth in every dimension.
Source: arXiv:2601.21366v2, §5.3, the smooth-gradient localization
extending the test potentials `uᵢ` of §5.2. -/
theorem contDiff_bump_mul_inner_any_d {d : ℕ} (y v : EucSpace d) (b : ContDiffBump y) :
    ContDiff ℝ (⊤ : ℕ∞) (fun z => b z * inner (𝕜 := ℝ) v z) :=
  b.contDiff.mul (innerSL ℝ v).contDiff

/-- At the center the actual gradient equals its prescribed vector.
Source: arXiv:2601.21366v2, §5.3, choosing nonvanishing localized
gradient directions at actual support points. -/
theorem gradient_bump_mul_inner_center_any_d {d : ℕ}
    (y v : EucSpace d) (b : ContDiffBump y) :
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

/-- Outside the bump's closed support the original gradient vanishes.
Source: arXiv:2601.21366v2, §5.3, extending localized test potentials
by zero, as in the circle construction of §5.2. -/
theorem gradient_bump_mul_inner_zero_any_d {d : ℕ}
    (y v : EucSpace d) (b : ContDiffBump y) (x : EucSpace d)
    (hx : x ∉ closedBall y b.rOut) :
    gradient (fun z => b z * inner (𝕜 := ℝ) v z) x = 0 := by
  have hb : (b : EucSpace d → ℝ) =ᶠ[𝓝 x] 0 :=
    notMem_tsupport_iff_eventuallyEq.mp (by rwa [b.tsupport_eq])
  have heq : (fun z => b z * inner (𝕜 := ℝ) v z) =ᶠ[𝓝 x] (fun _ => (0 : ℝ)) := by
    filter_upwards [hb] with z hz
    simp only [hz, Pi.zero_apply, zero_mul]
  unfold gradient
  rw [heq.fderiv_eq]
  simp

/-- An actual smooth spherical gradient can be supported in any requested
ball and equal a prescribed tangent at its center.
Source: arXiv:2601.21366v2, §5.3, the localized admissible directions
used to contradict `eq:strict2order` in `thm: any.d` (i). -/
theorem exists_local_sphere_gradient {d : ℕ} (y : SSphere d) (v : EucSpace d)
    (hv : inner (𝕜 := ℝ) (y : EucSpace d) v = 0) (r : ℝ) (hr : 0 < r) :
    ∃ ξ : SSphere d → EucSpace d, IsGradientField ξ ∧ Continuous ξ ∧
      ξ y = v ∧ Function.support ξ ⊆ ball y r := by
  let b : ContDiffBump (y : EucSpace d) :=
    { rIn := r / 4
      rOut := r / 2
      rIn_pos := by positivity
      rIn_lt_rOut := by linarith }
  let f : EucSpace d → ℝ := fun z => b z * inner (𝕜 := ℝ) v z
  let ξ : SSphere d → EucSpace d := fun x => proj d (x : EucSpace d) (gradient f (x : EucSpace d))
  have hf : ContDiff ℝ (⊤ : ℕ∞) f := contDiff_bump_mul_inner_any_d _ _ b
  have hgrad : IsGradientField ξ := ⟨f, hf, fun _ => rfl⟩
  refine ⟨ξ, hgrad, continuous_of_isGradientField ξ hgrad, ?_, ?_⟩
  · change proj d (y : EucSpace d)
      (gradient (fun z => b z * inner (𝕜 := ℝ) v z) (y : EucSpace d)) = v
    rw [gradient_bump_mul_inner_center_any_d, proj, hv, zero_smul, sub_zero]
  · intro x hx
    change dist x y < r
    apply lt_of_not_ge
    intro hdist
    have hxout : (x : EucSpace d) ∉ closedBall (y : EucSpace d) b.rOut := by
      rw [mem_closedBall]
      change ¬ dist (x : EucSpace d) (y : EucSpace d) ≤ r / 2
      change r ≤ dist (x : EucSpace d) (y : EucSpace d) at hdist
      linarith
    apply Function.mem_support.mp hx
    change proj d (x : EucSpace d)
      (gradient (fun z => b z * inner (𝕜 := ℝ) v z) (x : EucSpace d)) = 0
    rw [gradient_bump_mul_inner_zero_any_d _ _ b _ hxout]
    simp only [proj, inner_zero_right, zero_smul, sub_zero]

/-- In every dimension at least two the localized original gradient has
unit norm at its center. Source: arXiv:2601.21366v2, §5.3,
the nonzero smooth-gradient directions in `thm: any.d` (i). -/
theorem exists_local_unit_sphere_gradient {d : ℕ} (hd : 2 ≤ d)
    (y : SSphere d) (r : ℝ) (hr : 0 < r) :
    ∃ ξ : SSphere d → EucSpace d, IsGradientField ξ ∧ Continuous ξ ∧
      ‖ξ y‖ = 1 ∧ Function.support ξ ⊆ ball y r := by
  have hy : (y : EucSpace d) ≠ 0 := by
    intro hz
    have hn := norm_eq_of_mem_sphere y
    rw [hz, norm_zero] at hn
    norm_num at hn
  obtain ⟨v, hv, hyv⟩ := exists_unit_inner_eq_zero hd hy
  obtain ⟨ξ, hgrad, hc, heq, hsupp⟩ := exists_local_sphere_gradient y v hyv r hr
  exact ⟨ξ, hgrad, hc, by rw [heq, hv], hsupp⟩

/-- The same actual unit gradient can be supported in any open
neighborhood of its center. Source: arXiv:2601.21366v2, §5.3,
localizing the Hessian directions, extending the subarcs of §5.2. -/
theorem exists_local_unit_sphere_gradient_in_open {d : ℕ} (hd : 2 ≤ d)
    (U : Set (SSphere d)) (hU : IsOpen U) (y : SSphere d) (hy : y ∈ U) :
    ∃ ξ : SSphere d → EucSpace d, IsGradientField ξ ∧ Continuous ξ ∧
      ‖ξ y‖ = 1 ∧ Function.support ξ ⊆ U := by
  obtain ⟨r, hr, hball⟩ := isOpen_iff.mp hU y hy
  obtain ⟨ξ, hgrad, hc, hnorm, hsupp⟩ := exists_local_unit_sphere_gradient hd y r hr
  exact ⟨ξ, hgrad, hc, hnorm, hsupp.trans hball⟩

/-- The outside-support hypothesis holds for a genuine smooth bump with
radii one and two, at an actual ambient point of norm three. -/
example : ∃ b : ContDiffBump (0 : EucSpace 2),
    (3 : ℝ) • ((basePoint 1 : SSphere 2) : EucSpace 2) ∉ closedBall 0 b.rOut := by
  refine ⟨{ rIn := 1, rOut := 2, rIn_pos := one_pos, rIn_lt_rOut := by norm_num }, ?_⟩
  change ¬ dist ((3 : ℝ) • ((basePoint 1 : SSphere 2) : EucSpace 2)) 0 ≤ 2
  rw [dist_zero_right, norm_smul, norm_basePoint_one, Real.norm_eq_abs]
  norm_num

/-- Dimension, positive radius, tangency and open-neighborhood inputs
are jointly realized by the actual circle and its nonzero unit tangent. -/
example : (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    inner (𝕜 := ℝ) (circlePoint 0 : EucSpace 2)
      (circleTangent (circlePoint 0 : EucSpace 2)) = 0 ∧
    IsOpen (univ : Set (SSphere 2)) ∧ circlePoint 0 ∈ (univ : Set (SSphere 2)) :=
  ⟨le_rfl, one_pos, inner_circleTangent _, isOpen_univ, mem_univ _⟩

end Transformer.Perceptron

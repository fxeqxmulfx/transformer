import Transformer.Grokking.EffectiveTheory.Section3_GroundDynamics

/-!
# Nonzero representation domain from initial data

Source: Liu et al., arXiv:2205.10343v2, section 3.2, equation `eq:l_eff`,
and appendix conservation laws, equation `eq:Z0_conservation`. The source
says a nonzero embedding cannot collapse to the origin. For the corrected
single-parallelogram quotient gradient, this follows from the actual ODE
off the origin and ordinary differentiability, without assuming that the
trajectory remains in the nonzero domain.

At the origin the derivative of the squared norm is zero for any
differentiable coordinate paths. Off the origin it is zero by the radial
gradient identity. Therefore it is zero everywhere, the norm is constant,
and a nonzero initial norm excludes the origin at every finite real time.
Existence of such a global differentiable path is still an input.
-/

namespace Transformer.Grokking.EffectiveTheory

/-- The effective quotient-gradient equation on its natural domain.
Source: arXiv:2205.10343v2, appendix, equation `eq:Ei_dynamics_app`.
This predicate specifies velocities, not conservation or convergence. -/
def FollowsGradientOffOrigin (x y z : ℝ → ℝ) : Prop :=
  ∀ t, squaredNorm (x t) (y t) (z t) ≠ 0 →
    HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t ∧
    HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t ∧
    HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t

/-- The representation is at the origin exactly when its squared norm
vanishes. Source: arXiv:2205.10343v2, section 3.2, `Z₀` specialized to
three scalar coordinates; nonnegativity excludes cancellation. -/
theorem squaredNorm_eq_zero_iff (x y z : ℝ) :
    squaredNorm x y z = 0 ↔ x = 0 ∧ y = 0 ∧ z = 0 := by
  constructor
  · intro h
    unfold squaredNorm at h
    have hx : x ^ 2 = 0 := by nlinarith [sq_nonneg x, sq_nonneg y, sq_nonneg z]
    have hy : y ^ 2 = 0 := by nlinarith [sq_nonneg x, sq_nonneg y, sq_nonneg z]
    have hz : z ^ 2 = 0 := by nlinarith [sq_nonneg x, sq_nonneg y, sq_nonneg z]
    exact ⟨sq_eq_zero_iff.mp hx, sq_eq_zero_iff.mp hy, sq_eq_zero_iff.mp hz⟩
  · rintro ⟨hx, hy, hz⟩
    rw [hx, hy, hz]
    unfold squaredNorm
    norm_num

/-- A nonzero initial ground coordinate already supplies a nonzero norm.
Source correction: arXiv:2205.10343v2, section 3.2, relative spectral
analysis; a separate initial-domain hypothesis is redundant here. -/
theorem squaredNorm_ne_zero_of_ground (x y z : ℝ) (hA : x - z ≠ 0) :
    squaredNorm x y z ≠ 0 := by
  intro hZ
  rcases (squaredNorm_eq_zero_iff x y z).mp hZ with ⟨hx, hy, hz⟩
  apply hA
  rw [hx, hz]
  ring

example : (1 : ℝ) - (-1) ≠ 0 := by norm_num

/-- The norm derivative is zero everywhere, including an origin that has
not yet been excluded. Source: arXiv:2205.10343v2, appendix norm law.
The extra differentiability premises cover the domain's missing point. -/
theorem norm_deriv_off_origin (x y z : ℝ → ℝ)
    (hdx : Differentiable ℝ x) (hdy : Differentiable ℝ y) (hdz : Differentiable ℝ z)
    (hflow : FollowsGradientOffOrigin x y z) (t : ℝ) :
    HasDerivAt (fun s => squaredNorm (x s) (y s) (z s)) 0 t := by
  by_cases hZ : squaredNorm (x t) (y t) (z t) = 0
  · rcases (squaredNorm_eq_zero_iff (x t) (y t) (z t)).mp hZ with ⟨hx, hy, hz⟩
    have h := (((hdx t).hasDerivAt.pow 2).add ((hdy t).hasDerivAt.pow 2)).add
      ((hdz t).hasDerivAt.pow 2)
    convert h using 1
    · rfl
    · simp only [hx, hy, hz]
      ring
  · exact norm_deriv_of_gradient_flow x y z t hZ
      (hflow t hZ).1 (hflow t hZ).2.1 (hflow t hZ).2.2

example : Differentiable ℝ (fun _ : ℝ => (1 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (0 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (-1 : ℝ)) ∧
    FollowsGradientOffOrigin (fun _ => 1) (fun _ => 0) (fun _ => -1) := by
  refine ⟨differentiable_const 1, differentiable_const 0, differentiable_const (-1), ?_⟩
  intro t hZ
  have hg := ground_gradient_zero 1 0 (-1) hZ (by unfold residual; norm_num)
  refine ⟨?_, ?_, ?_⟩ <;> simpa [hg] using hasDerivAt_const t _

/-- Complete norm conservation under the ODE only off the origin.
Source: arXiv:2205.10343v2, appendix, `eq:Z0_conservation`; global
differentiability replaces the previously assumed nonzero domain. -/
theorem norm_const_off_origin (x y z : ℝ → ℝ)
    (hdx : Differentiable ℝ x) (hdy : Differentiable ℝ y) (hdz : Differentiable ℝ z)
    (hflow : FollowsGradientOffOrigin x y z) (s t : ℝ) :
    squaredNorm (x t) (y t) (z t) = squaredNorm (x s) (y s) (z s) := by
  have hd := norm_deriv_off_origin x y z hdx hdy hdz hflow
  exact is_const_of_deriv_eq_zero (fun u => (hd u).differentiableAt)
    (fun u => (hd u).deriv) t s

example : Differentiable ℝ (fun _ : ℝ => (1 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (0 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (-1 : ℝ)) ∧
    FollowsGradientOffOrigin (fun _ => 1) (fun _ => 0) (fun _ => -1) := by
  refine ⟨differentiable_const 1, differentiable_const 0, differentiable_const (-1), ?_⟩
  intro t hZ
  have hg := ground_gradient_zero 1 0 (-1) hZ (by unfold residual; norm_num)
  refine ⟨?_, ?_, ?_⟩ <;> simpa [hg] using hasDerivAt_const t _

/-- A nonzero initial representation never reaches the excluded origin.
Source: arXiv:2205.10343v2, section 3.2, noncollapse claim. Initial
nonvanishing is sufficient; all-time nonvanishing is a conclusion. -/
theorem nonzero_domain_from_initial (x y z : ℝ → ℝ)
    (hdx : Differentiable ℝ x) (hdy : Differentiable ℝ y) (hdz : Differentiable ℝ z)
    (hflow : FollowsGradientOffOrigin x y z)
    (h0 : squaredNorm (x 0) (y 0) (z 0) ≠ 0) :
    ∀ t, squaredNorm (x t) (y t) (z t) ≠ 0 := by
  intro t hZ
  apply h0
  rw [← norm_const_off_origin x y z hdx hdy hdz hflow 0 t]
  exact hZ

example : Differentiable ℝ (fun _ : ℝ => (1 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (0 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (-1 : ℝ)) ∧
    FollowsGradientOffOrigin (fun _ => 1) (fun _ => 0) (fun _ => -1) ∧
    squaredNorm 1 0 (-1) ≠ 0 := by
  refine ⟨differentiable_const 1, differentiable_const 0, differentiable_const (-1), ?_, ?_⟩
  · intro t hZ
    have hg := ground_gradient_zero 1 0 (-1) hZ (by unfold residual; norm_num)
    refine ⟨?_, ?_, ?_⟩ <;> simpa [hg] using hasDerivAt_const t _
  · unfold squaredNorm
    norm_num

/-- The domain-restricted equation becomes the full classical ODE once
noncollapse is proved. Source correction: arXiv:2205.10343v2, appendix
effective dynamics. This supplies the earlier spectral theorems' premises. -/
theorem full_gradient_flow_from_initial (x y z : ℝ → ℝ)
    (hdx : Differentiable ℝ x) (hdy : Differentiable ℝ y) (hdz : Differentiable ℝ z)
    (hflow : FollowsGradientOffOrigin x y z)
    (h0 : squaredNorm (x 0) (y 0) (z 0) ≠ 0) : ∀ t,
    HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t ∧
    HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t ∧
    HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t := by
  have hZ := nonzero_domain_from_initial x y z hdx hdy hdz hflow h0
  intro t
  exact hflow t (hZ t)

example : Differentiable ℝ (fun _ : ℝ => (1 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (0 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (-1 : ℝ)) ∧
    FollowsGradientOffOrigin (fun _ => 1) (fun _ => 0) (fun _ => -1) ∧
    squaredNorm 1 0 (-1) ≠ 0 := by
  refine ⟨differentiable_const 1, differentiable_const 0, differentiable_const (-1), ?_, ?_⟩
  · intro t hZ
    have hg := ground_gradient_zero 1 0 (-1) hZ (by unfold residual; norm_num)
    refine ⟨?_, ?_, ?_⟩ <;> simpa [hg] using hasDerivAt_const t _
  · unfold squaredNorm
    norm_num

end Transformer.Grokking.EffectiveTheory

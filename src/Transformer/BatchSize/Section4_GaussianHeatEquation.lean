/-
# The heat equation for the Gaussian comparison operator

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
The time derivative of the actual Gaussian transition expectation equals
one half of its spatial second derivative. This supplies the backward
generator equation for constant scalar Brownian coefficients.
-/

import Transformer.BatchSize.Section4_GaussianSemigroup

noncomputable section

namespace Transformer.BatchSize

/-- The continuous Gaussian comparison satisfies the backward heat equation,
Section 4.3, equations (2)--(3). Both derivatives are actual derivatives of
the expectation operator, and the assertion applies to every initial state. -/
theorem gaussianTransition_heat_equation {f : ℝ → ℝ} (hf : ContDiff ℝ 2 f)
    {R : ℝ} (hR : ∀ j ≤ 2, ∀ x, |iteratedDeriv j f x| ≤ R)
    {t : ℝ} (ht : 0 < t) (x : ℝ) :
    HasDerivAt (fun s => gaussianTransition f s x)
      ((1 / 2) * iteratedDeriv 2 (gaussianTransition f t) x) t := by
  have hsmooth := contDiff_nat_iff_iteratedDeriv.mp hf
  have hd₀ : ∀ y, HasDerivAt f (iteratedDeriv 1 f y) y := by
    intro y
    rw [iteratedDeriv_one]
    exact (hf.differentiable (by norm_num) y).hasDerivAt
  have hd₁ : ∀ y, HasDerivAt (iteratedDeriv 1 f) (iteratedDeriv 2 f y) y := by
    intro y
    rw [iteratedDeriv_succ (n := 1)]
    exact (hsmooth.2 1 (by norm_num) y).hasDerivAt
  have hdshift₀ : ∀ y, HasDerivAt (fun z => f (x + z)) (iteratedDeriv 1 f (x + y)) y := by
    intro y
    convert (hd₀ (x + y)).comp y ((hasDerivAt_id y).const_add x) using 1 <;>
      simp only [Function.comp_def, mul_one]
  have hdshift₁ : ∀ y, HasDerivAt (fun z => iteratedDeriv 1 f (x + z))
      (iteratedDeriv 2 f (x + y)) y := by
    intro y
    convert (hd₁ (x + y)).comp y ((hasDerivAt_id y).const_add x) using 1 <;>
      simp only [Function.comp_def, mul_one]
  rw [gaussianTransition_iteratedDeriv hf hR t 2 le_rfl]
  exact gaussianFlow_hasDerivAt hdshift₀ hdshift₁
    ((hsmooth.1 2 le_rfl).comp (continuous_const.add continuous_id))
    (fun y => by simpa only [iteratedDeriv_zero] using hR 0 (by norm_num) (x + y))
    (fun y => hR 1 (by norm_num) (x + y)) (fun y => hR 2 le_rfl (x + y)) ht

/-- Joint nonvacuity of smoothness, all derivative bounds and positive
elapsed time in the heat equation, Section 4.3. -/
example : ContDiff ℝ 2 (fun _ : ℝ => (1 : ℝ)) ∧
    (∀ j ≤ 2, ∀ x : ℝ, |iteratedDeriv j (fun _ : ℝ => (1 : ℝ)) x| ≤ 1) ∧
    (0 : ℝ) < 1 := by
  refine ⟨contDiff_const, ?_, by norm_num⟩
  intro j hj x
  rw [iteratedDeriv_const]
  split_ifs <;> norm_num

/-- A uniform local generator expansion at every initial state,
Section 4.3, equations (2)--(3), for constant scalar Brownian coefficients.
The remainder follows from the proved integrated generator identity. -/
theorem gaussianTransition_local_generator_error {f : ℝ → ℝ} (hf : ContDiff ℝ 4 f)
    {R : ℝ} (hR : ∀ j ≤ 4, ∀ x, |iteratedDeriv j f x| ≤ R)
    {t : ℝ} (ht : 0 ≤ t) (x : ℝ) :
    |gaussianTransition f t x - f x - t / 2 * iteratedDeriv 2 f x| ≤ R * t ^ 2 / 4 := by
  have hsmooth := contDiff_nat_iff_iteratedDeriv.mp hf
  have hd (j : ℕ) (hj : j < 4) (y : ℝ) :
      HasDerivAt (fun z => iteratedDeriv j f (x + z))
        (iteratedDeriv (j + 1) f (x + y)) y := by
    have hdj : HasDerivAt (iteratedDeriv j f) (iteratedDeriv (j + 1) f (x + y)) (x + y) := by
      rw [iteratedDeriv_succ (n := j)]
      exact (hsmooth.2 j hj (x + y)).hasDerivAt
    convert hdj.comp y ((hasDerivAt_id y).const_add x) using 1 <;>
      simp only [Function.comp_def, mul_one]
  have hbound (j : ℕ) (hj : j ≤ 4) (y : ℝ) := hR j hj (x + y)
  have h := gaussianFlow_local_generator_error
    (hd 0 (by norm_num)) (hd 1 (by norm_num)) (hd 2 (by norm_num)) (hd 3 (by norm_num))
    ((hsmooth.1 4 le_rfl).comp (continuous_const.add continuous_id))
    (hbound 0 (by norm_num)) (hbound 1 (by norm_num)) (hbound 2 (by norm_num))
    (hbound 3 (by norm_num)) (hbound 4 le_rfl) ht
  simpa only [iteratedDeriv_zero, add_zero, gaussianTransition] using h

/-- Joint nonvacuity of all four derivative bounds and nonnegative time
in the local continuous expansion, Section 4.3. -/
example : ContDiff ℝ 4 (fun _ : ℝ => (1 : ℝ)) ∧
    (∀ j ≤ 4, ∀ x : ℝ, |iteratedDeriv j (fun _ : ℝ => (1 : ℝ)) x| ≤ 1) ∧
    (0 : ℝ) ≤ 1 := by
  refine ⟨contDiff_const, ?_, by norm_num⟩
  intro j hj x
  rw [iteratedDeriv_const]
  split_ifs <;> norm_num

end Transformer.BatchSize

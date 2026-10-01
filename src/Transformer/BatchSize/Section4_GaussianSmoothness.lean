/-
# Propagation of smooth observables by Gaussian transitions

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
The constant-coefficient comparison operator preserves each derivative
bound through any finite order. The proof identifies its actual iterated
derivatives with expectations of the original iterated derivatives.
-/

import Transformer.BatchSize.Section4_GaussianTransition

noncomputable section

namespace Transformer.BatchSize

/-- Every spatial derivative through order n commutes with Gaussian
transition, Section 4.3's weak comparison operator. -/
theorem gaussianTransition_iteratedDeriv {n : ℕ} {f : ℝ → ℝ}
    (hf : ContDiff ℝ n f) {R : ℝ}
    (hR : ∀ j ≤ n, ∀ x, |iteratedDeriv j f x| ≤ R) (t : ℝ) :
    ∀ j ≤ n, iteratedDeriv j (gaussianTransition f t) =
      gaussianTransition (iteratedDeriv j f) t := by
  have hsmooth := contDiff_nat_iff_iteratedDeriv.mp hf
  intro j
  induction j with
  | zero => intro hj; simp only [iteratedDeriv_zero]
  | succ j ih =>
    intro hj
    have hj' : j ≤ n := by omega
    have hdiff := hsmooth.2 j (by omega)
    have hd : ∀ x, HasDerivAt (iteratedDeriv j f) (iteratedDeriv (j + 1) f x) x := by
      intro x
      rw [iteratedDeriv_succ]
      exact (hdiff x).hasDerivAt
    rw [iteratedDeriv_succ, ih hj']
    funext x
    exact (gaussianTransition_hasDerivAt hd (hsmooth.1 (j + 1) hj)
      (hR j hj') (hR (j + 1) hj) t x).deriv

/-- Joint nonvacuity of smoothness and all derivative bounds, Section 4.3:
a nonzero bounded observable with six derivatives. -/
example : ContDiff ℝ 6 (fun _ : ℝ => (1 : ℝ)) ∧
    ∀ j ≤ 6, ∀ x : ℝ, |iteratedDeriv j (fun _ : ℝ => (1 : ℝ)) x| ≤ 1 := by
  refine ⟨contDiff_const, ?_⟩
  intro j hj x
  rw [iteratedDeriv_const]
  split_ifs <;> norm_num

/-- Gaussian comparison preserves Cn regularity and the same uniform
bounds on all derivatives through n, Section 4.3's weak approximation.
This resolves propagation of test functions for constant Brownian coefficients. -/
theorem gaussianTransition_preserves_smooth_bound {n : ℕ} {f : ℝ → ℝ}
    (hf : ContDiff ℝ n f) {R : ℝ}
    (hR : ∀ j ≤ n, ∀ x, |iteratedDeriv j f x| ≤ R) (t : ℝ) :
    ContDiff ℝ n (gaussianTransition f t) ∧
      ∀ j ≤ n, ∀ x, |iteratedDeriv j (gaussianTransition f t) x| ≤ R := by
  have hsmooth := contDiff_nat_iff_iteratedDeriv.mp hf
  have hcomm := gaussianTransition_iteratedDeriv hf hR t
  refine ⟨contDiff_nat_iff_iteratedDeriv.mpr ⟨?_, ?_⟩, ?_⟩
  · intro j hj
    rw [hcomm j hj]
    exact continuous_gaussianTransition (hsmooth.1 j hj) (hR j hj) t
  · intro j hj x
    rw [hcomm j (by omega)]
    have hd : ∀ y, HasDerivAt (iteratedDeriv j f) (iteratedDeriv (j + 1) f y) y := by
      intro y
      rw [iteratedDeriv_succ]
      exact (hsmooth.2 j hj y).hasDerivAt
    exact (gaussianTransition_hasDerivAt hd (hsmooth.1 (j + 1) (by omega))
      (hR j (by omega)) (hR (j + 1) (by omega)) t x).differentiableAt
  · intro j hj x
    rw [hcomm j hj]
    exact gaussianTransition_abs_le (hR j hj) t x

/-- Joint nonvacuity of every bound in the propagation theorem, Section 4.3. -/
example : ContDiff ℝ 6 (fun _ : ℝ => (1 : ℝ)) ∧
    ∀ j ≤ 6, ∀ x : ℝ, |iteratedDeriv j (fun _ : ℝ => (1 : ℝ)) x| ≤ 1 := by
  refine ⟨contDiff_const, ?_⟩
  intro j hj x
  rw [iteratedDeriv_const]
  split_ifs <;> norm_num

end Transformer.BatchSize

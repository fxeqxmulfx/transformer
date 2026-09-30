/-
# DASH — continuous Chebyshev orthogonality and cosine coefficients

arXiv:2602.02016v2, Appendix A. The first-kind polynomials are
orthogonal for the Chebyshev measure, and their weighted Fourier
coefficients equal those of the cosine-composed function.
-/

import Transformer.DASH.SectionA_ScalarChebyshev
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Chebyshev.Orthogonality

noncomputable section

namespace Transformer.DASH

/-- Products of the actual recurrence polynomials are integrable for the
first-kind Chebyshev measure. Source: arXiv:2602.02016v2, Appendix A, orthogonal basis. -/
theorem chebT_weighted_integrable (k l : ℕ) :
    MeasureTheory.Integrable (fun x : ℝ => chebT x k * chebT x l)
      Polynomial.Chebyshev.measureT := by
  apply Polynomial.Chebyshev.integrable_measureT
  simp only [chebT_eq_eval]
  fun_prop

/-- The first-kind recurrence basis is orthogonal for the weight
`1/sqrt(1-x²)` on `[-1,1]`; its constant mode has twice the squared norm
of every other mode. Source: arXiv:2602.02016v2, Appendix A, orthogonal basis. -/
theorem chebT_weighted_orthogonality (k l : ℕ) :
    (∫ x : ℝ, chebT x k * chebT x l ∂Polynomial.Chebyshev.measureT) =
      if k = l then (if k = 0 then Real.pi else Real.pi / 2) else 0 := by
  simp only [chebT_eq_eval]
  split_ifs with hkl hk
  · subst l
    subst k
    exact Polynomial.Chebyshev.integral_eval_T_real_mul_self_measureT_zero
  · subst l
    exact Polynomial.Chebyshev.integral_T_real_mul_self_measureT_of_ne_zero hk
  · exact Polynomial.Chebyshev.integral_eval_T_real_mul_eval_T_real_measureT_of_ne hkl

/-- Each weighted Chebyshev coefficient integral is the corresponding
Fourier cosine coefficient integral of `f(cos θ)`. This is the coefficient
identity underlying the source's series equivalence and discrete fitting.
Source: arXiv:2602.02016v2, Appendix A, the Fourier cosine series paragraph. -/
theorem chebyshev_cosine_coefficient_identity (f : ℝ → ℝ) (k : ℕ) :
    (∫ x : ℝ, f x * chebT x k ∂Polynomial.Chebyshev.measureT) =
      ∫ θ : ℝ in 0..Real.pi, f (Real.cos θ) * Real.cos ((k : ℝ) * θ) := by
  rw [Polynomial.Chebyshev.integral_measureT_eq_integral_cos]
  simp_rw [chebT_cos]

end Transformer.DASH

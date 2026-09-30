/-
# DASH — the Chebyshev basis and existence of uniform approximants

arXiv:2602.02016v2, Appendix A. Chebyshev polynomials span all real
polynomials. On a positive regularized interval there exist finite
Chebyshev expansions approximating inverse powers to any requested accuracy.
This does not identify them with a particular finite cosine-fit output.
-/

import Transformer.DASH.SectionA_ScalarChebyshev
import Mathlib.Topology.ContinuousMap.Weierstrass
import Mathlib.Analysis.SpecialFunctions.Pow.Continuity

noncomputable section

namespace Transformer.DASH

/-- The first-kind Chebyshev polynomial basis, arXiv:2602.02016v2,
Appendix A, the basis used for finite function approximations. -/
def chebyshevBasis : Module.Basis ℕ ℝ (Polynomial ℝ) :=
  (Polynomial.Chebyshev.chebyshevTsequence ℝ).basis (fun k => by
    change IsUnit (Polynomial.Chebyshev.T ℝ (k : ℤ)).leadingCoeff
    rw [Polynomial.Chebyshev.leadingCoeff_T]
    exact isUnit_iff_ne_zero.2 (pow_ne_zero _ (by norm_num)))

/-- The basis elements are exactly the source's first-kind Chebyshev
polynomials. Source: arXiv:2602.02016v2, Appendix A, `T_0,T_1,T_(n+1)`. -/
theorem chebyshevBasis_apply (k : ℕ) :
    chebyshevBasis k = Polynomial.Chebyshev.T ℝ (k : ℤ) := by
  apply Polynomial.Sequence.basis_eq_self

/-- Every real polynomial has a finite Chebyshev expansion.
Source: arXiv:2602.02016v2, Appendix A, the polynomial basis claim. -/
theorem exists_chebyshev_polynomial_expansion (p : Polynomial ℝ) :
    ∃ c : ℕ →₀ ℝ, p = c.sum (fun k a => a • Polynomial.Chebyshev.T ℝ (k : ℤ)) := by
  refine ⟨chebyshevBasis.repr p, ?_⟩
  have h := chebyshevBasis.linearCombination_repr p
  simpa only [Finsupp.linearCombination_apply, chebyshevBasis_apply] using h.symm

/-- Polynomial evaluation agrees with the finite scalar recurrence expansion.
Source: arXiv:2602.02016v2, Appendix A, the finite sum `Σ c_k T_k(x)`. -/
theorem chebyshev_expansion_eval (c : ℕ →₀ ℝ) (x : ℝ) :
    (c.sum (fun k a => a • Polynomial.Chebyshev.T ℝ (k : ℤ))).eval x =
      c.sum (fun k a => a * chebT x k) := by
  simp only [Finsupp.sum, Polynomial.eval_finsetSum, Polynomial.eval_smul,
    smul_eq_mul, chebT_eq_eval]

/-- Continuous functions on `[-1,1]` admit finite Chebyshev expansions with
arbitrarily small uniform error. This is an existence theorem, rather than
an unchecked assertion that every finite fitted expansion is accurate.
Source: arXiv:2602.02016v2, Appendix A, approximation by a Chebyshev basis. -/
theorem exists_chebyshev_uniform_approximation (f : ℝ → ℝ)
    (hf : ContinuousOn f (Set.Icc (-1) 1)) (δ : ℝ) (hδ : 0 < δ) :
    ∃ c : ℕ →₀ ℝ, ∀ x ∈ Set.Icc (-1 : ℝ) 1,
      |c.sum (fun k a => a * chebT x k) - f x| < δ := by
  obtain ⟨p, hp⟩ := exists_polynomial_near_of_continuousOn (-1) 1 f hf δ hδ
  obtain ⟨c, hc⟩ := exists_chebyshev_polynomial_expansion p
  refine ⟨c, ?_⟩
  intro x hx
  rw [← chebyshev_expansion_eval, ← hc]
  exact hp x hx

/-- The uniform-approximation assumptions are satisfiable,
arXiv:2602.02016v2, Appendix A. -/
example : ContinuousOn (fun x : ℝ => x) (Set.Icc (-1) 1) ∧ (0 : ℝ) < 1 :=
  ⟨continuous_id.continuousOn, by norm_num⟩

/-- Every real inverse-power exponent has uniform finite Chebyshev
approximants on the source's positive interval `[ε,1+ε]`, after its required
affine mapping. In particular this covers exponents `-1/2` and `-1/4`.
Source: arXiv:2602.02016v2, Appendix A, regularized inverse-root approximation. -/
theorem exists_chebyshev_inverse_power_approximation (ε exponent δ : ℝ)
    (hε : 0 < ε) (hδ : 0 < δ) :
    ∃ c : ℕ →₀ ℝ, ∀ t ∈ Set.Icc (-1 : ℝ) 1,
      |c.sum (fun k a => a * chebT t k) -
        (chebFromCoordinate ε (1 + ε) t) ^ exponent| < δ := by
  apply exists_chebyshev_uniform_approximation _ _ δ hδ
  intro t ht
  have hcoord : 0 < chebFromCoordinate ε (1 + ε) t := by
    unfold chebFromCoordinate
    linarith [ht.1]
  apply (Real.continuousAt_rpow_const _ exponent (Or.inl hcoord.ne')).comp_continuousWithinAt
  have hcont : Continuous (chebFromCoordinate ε (1 + ε)) := by
    unfold chebFromCoordinate
    fun_prop
  exact hcont.continuousWithinAt

/-- Positive regularization and a positive error tolerance coexist,
arXiv:2602.02016v2, Appendix A. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100 := by norm_num

end Transformer.DASH

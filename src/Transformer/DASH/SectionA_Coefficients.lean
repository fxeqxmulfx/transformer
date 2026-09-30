/-
# DASH — coefficient fitting and finite polynomial degree

arXiv:2602.02016v2, Appendix A, `algorithm:cbshv-coeff-fitting`.
The cosine transform determines coefficients, not an automatic
error bound for arbitrary degree and sample count.
-/

import Transformer.DASH.SectionA_ScalarChebyshev
import Mathlib.Algebra.Polynomial.BigOperators

open scoped BigOperators

noncomputable section

namespace Transformer.DASH

/-- Cosine-node angles, arXiv:2602.02016v2, Appendix A, coefficient fitting. -/
def chebAngle (N : ℕ) (i : Fin N) : ℝ := (2 * (i.val : ℝ) + 1) * Real.pi / (2 * N)

/-- Mapped cosine nodes, arXiv:2602.02016v2, Appendix A, coefficient fitting. -/
def chebNode (a b : ℝ) (N : ℕ) (i : Fin N) : ℝ :=
  chebFromCoordinate a b (Real.cos (chebAngle N i))

/-- The fitted coefficients, with the source's final halving of `c_0`.
Source: arXiv:2602.02016v2, Appendix A, `algorithm:cbshv-coeff-fitting`.
An actual fit uses `N>0`; total division returns zero when there are no samples. -/
def chebFitCoefficient (f : ℝ → ℝ) (a b : ℝ) (N k : ℕ) : ℝ :=
  (if k = 0 then 1 else 2) / (N : ℝ) *
    ∑ i : Fin N, f (chebNode a b N i) * Real.cos ((k : ℝ) * chebAngle N i)

/-- All `d+1` coefficients in ascending order,
arXiv:2602.02016v2, Appendix A, coefficient-fitting output. -/
def chebFit (f : ℝ → ℝ) (a b : ℝ) (N d : ℕ) : List ℝ :=
  List.ofFn (fun k : Fin (d + 1) => chebFitCoefficient f a b N k.val)

/-- Every fitted node is in `[a,b]`, arXiv:2602.02016v2, Appendix A. -/
theorem chebNode_bounds (a b : ℝ) (N : ℕ) (i : Fin N) (hab : a ≤ b) :
    a ≤ chebNode a b N i ∧ chebNode a b N i ≤ b :=
  chebFromCoordinate_bounds a b _ hab (Real.neg_one_le_cos _) (Real.cos_le_one _)

/-- A fitting interval exists, arXiv:2602.02016v2, Appendix A. -/
example : (1 : ℝ) ≤ 3 := by norm_num

/-- The halved constant coefficient is the sample mean,
arXiv:2602.02016v2, Appendix A, the final coefficient-fitting step. -/
theorem chebFitCoefficient_zero (f : ℝ → ℝ) (a b : ℝ) (N : ℕ) :
    chebFitCoefficient f a b N 0 =
      (1 / (N : ℝ)) * ∑ i : Fin N, f (chebNode a b N i) := by
  simp [chebFitCoefficient]

/-- Fitting a constant recovers its constant coefficient exactly, provided
there is at least one sample, arXiv:2602.02016v2, Appendix A. -/
theorem chebFit_constant_zero (a b c : ℝ) (N : ℕ) (hN : 0 < N) :
    chebFitCoefficient (fun _ => c) a b N 0 = c := by
  have hn : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  rw [chebFitCoefficient_zero]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-- Positive sample counts exist, arXiv:2602.02016v2, Appendix A. -/
example : 0 < (1000 : ℕ) := by norm_num

/-- Degree `d` requires `d+1` coefficients. The source calls `d` the
number of terms in prose but both the sum and pseudocode include index zero.
Source: arXiv:2602.02016v2, Appendix A, finite expansion and fitting algorithm. -/
theorem chebFit_length (f : ℝ → ℝ) (a b : ℝ) (N d : ℕ) :
    (chebFit f a b N d).length = d + 1 := by simp [chebFit]

/-- The polynomial represented by a degree-`d` coefficient vector,
arXiv:2602.02016v2, Appendix A, `Σ_(k=0)^d c_k T_k`. -/
def chebPolynomial (d : ℕ) (c : Fin (d + 1) → ℝ) : Polynomial ℝ :=
  ∑ k, Polynomial.C (c k) * Polynomial.Chebyshev.T ℝ (k.val : ℤ)

/-- A degree-`d` Chebyshev expansion has polynomial degree at most `d`.
Source: arXiv:2602.02016v2, Appendix A, the finite-expansion degree claim. -/
theorem chebPolynomial_degree (d : ℕ) (c : Fin (d + 1) → ℝ) :
    (chebPolynomial d c).natDegree ≤ d := by
  apply Polynomial.natDegree_sum_le_of_forall_le
  intro k hk
  have hdegree := Polynomial.natDegree_mul_le
    (p := Polynomial.C (c k)) (q := Polynomial.Chebyshev.T ℝ (k.val : ℤ))
  simp only [Polynomial.natDegree_C, Polynomial.Chebyshev.natDegree_T,
    Int.natAbs_natCast, zero_add] at hdegree
  exact hdegree.trans (Nat.le_of_lt_succ k.isLt)

/-- Finite cosine expansion with the same coefficients,
arXiv:2602.02016v2, Appendix A, the Fourier cosine interpretation. -/
def cosineSeriesFrom (θ : ℝ) (k : ℕ) : List ℝ → ℝ
  | [] => 0
  | c :: cs => Real.cos ((k : ℝ) * θ) * c + cosineSeriesFrom θ (k + 1) cs

/-- Under `x=cos θ`, the finite Chebyshev and cosine series agree exactly.
Source: arXiv:2602.02016v2, Appendix A, Fourier cosine series paragraph. -/
theorem chebSeries_cos (θ : ℝ) (cs : List ℝ) (k : ℕ) :
    chebSeriesFrom (Real.cos θ) k cs = cosineSeriesFrom θ k cs := by
  induction cs generalizing k with
  | nil => rfl
  | cons c cs ih => simp only [chebSeriesFrom, cosineSeriesFrom, chebT_cos, ih]

end Transformer.DASH

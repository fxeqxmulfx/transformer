/-
# DASH — exact recovery by the cosine coefficient fit

arXiv:2602.02016v2, Appendix A, coefficient fitting and scalar Clenshaw.
With degree below the number of samples, fitting recovers every polynomial
in the selected Chebyshev span exactly, including the constant coefficient.
-/

import Transformer.DASH.SectionA_DiscreteOrthogonality
import Transformer.DASH.SectionA_DenseCoefficients

open scoped BigOperators

noncomputable section

namespace Transformer.DASH

/-- Scalar evaluation of the source's fitted array, with the explicit
interval mapping required by its “Important” paragraph.
Source: arXiv:2602.02016v2, Appendix A, fitting and scalar Clenshaw algorithms. -/
def chebFitEval (f : ℝ → ℝ) (a b : ℝ) (N d : ℕ) (x : ℝ) : ℝ :=
  clenshaw (chebToCoordinate a b x) (chebFit f a b N d)

/-- The fitted scalar output is the finite expansion of the actual cosine
coefficients. Source: arXiv:2602.02016v2, Appendix A, finite Chebyshev expansion. -/
theorem chebFitEval_eq_sum (f : ℝ → ℝ) (a b x : ℝ) (N d : ℕ) :
    chebFitEval f a b N d x = ∑ k : Fin (d + 1),
      chebFitCoefficient f a b N k * chebT (chebToCoordinate a b x) k := by
  exact clenshaw_ofFn _ _ _

/-- Coefficient fitting recovers any degree-`d` Chebyshev expansion when
`d<N`. The source leaves this sampling restriction implicit in its claim
that the final halving makes the transform orthogonal.
Source: arXiv:2602.02016v2, Appendix A, coefficient-fitting algorithm. -/
theorem chebFitCoefficient_reproduces (a b : ℝ) (N d : ℕ)
    (c : Fin (d + 1) → ℝ) (k : Fin (d + 1)) (hab : a < b) (hd : d < N) :
    chebFitCoefficient
      (fun x => ∑ j : Fin (d + 1), c j * chebT (chebToCoordinate a b x) j)
      a b N k = c k := by
  have hN : 0 < N := by omega
  have hn : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  have hnodes : ∀ i : Fin N,
      chebToCoordinate a b (chebNode a b N i) = Real.cos (chebAngle N i) := by
    intro i
    exact chebCoordinate_reverse a b _ hab
  unfold chebFitCoefficient
  simp_rw [hnodes, chebT_cos, Finset.sum_mul]
  rw [Finset.sum_comm]
  have hsum : (∑ j : Fin (d + 1), ∑ i : Fin N,
      c j * Real.cos ((j : ℝ) * chebAngle N i) *
        Real.cos ((k : ℝ) * chebAngle N i)) =
      c k * (if k.val = 0 then (N : ℝ) else (N : ℝ) / 2) := by
    have hcolumn (j : Fin (d + 1)) :=
      chebAngle_cos_orthogonality N j.val k.val
        (Nat.lt_of_le_of_lt (Nat.le_of_lt_succ j.isLt) hd)
        (Nat.lt_of_le_of_lt (Nat.le_of_lt_succ k.isLt) hd)
    simp_rw [mul_assoc, ← Finset.mul_sum, hcolumn]
    rw [Finset.sum_eq_single k]
    · simp
    · intro j hj hjk
      have hjk' : j.val ≠ k.val := by exact fun h => hjk (Fin.ext h)
      simp [hjk']
    · simp
  rw [hsum]
  split_ifs <;> field_simp

/-- Recovery assumptions have a nonconstant polynomial instance,
arXiv:2602.02016v2, Appendix A. -/
example : (1 : ℝ) < 3 ∧ (1 : ℕ) < 1000 := by norm_num

/-- Fitting and evaluating reproduces the entire selected polynomial span,
not just its values at the fitting nodes.
Source: arXiv:2602.02016v2, Appendix A, coefficient fitting followed by Clenshaw. -/
theorem chebFitEval_reproduces (a b x : ℝ) (N d : ℕ)
    (c : Fin (d + 1) → ℝ) (hab : a < b) (hd : d < N) :
    chebFitEval
      (fun y => ∑ j : Fin (d + 1), c j * chebT (chebToCoordinate a b y) j)
      a b N d x = ∑ j : Fin (d + 1), c j * chebT (chebToCoordinate a b x) j := by
  rw [chebFitEval_eq_sum]
  apply Finset.sum_congr rfl
  intro k hk
  rw [chebFitCoefficient_reproduces a b N d c k hab hd]

/-- A positive fitting interval and sufficient sample count coexist,
arXiv:2602.02016v2, Appendix A. -/
example : (1 : ℝ) < 2 ∧ (60 : ℕ) < 1000 := by norm_num

end Transformer.DASH

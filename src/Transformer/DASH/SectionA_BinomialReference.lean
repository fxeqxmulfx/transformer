/-
# DASH — binomial reference polynomials for inverse powers

arXiv:2602.02016v2, Appendix A, approximation on `[ε,1+ε]`.
Expanding at the interval midpoint supplies degree-bounded reference
polynomials for estimates of the source's discrete cosine fit.
-/

import Transformer.DASH.SectionA_DegreeSpan
import Mathlib.Analysis.Analytic.Binomial
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open scoped BigOperators

noncomputable section

namespace Transformer.DASH

/-- Midpoint binomial reference polynomial in the Chebyshev coordinate.
The identity `x=(ε+1/2)(1+t/(2ε+1))` turns the power series for `(1+z)^r`
into a polynomial approximation on the source's interval.
Source: arXiv:2602.02016v2, Appendix A, inverse-power fitting interval. -/
def inversePowerReference (ε exponent : ℝ) (d : ℕ) : Polynomial ℝ :=
  Polynomial.C ((ε + 1 / 2) ^ exponent) *
    ∑ k : Fin (d + 1), Polynomial.C (Ring.choose exponent k) *
      (Polynomial.C (1 / (2 * ε + 1)) * Polynomial.X) ^ k.val

/-- The reference truncation has degree at most `d` and therefore belongs
to the source's first `d+1` Chebyshev modes.
Source: arXiv:2602.02016v2, Appendix A, the finite polynomial degree. -/
theorem inversePowerReference_degree (ε exponent : ℝ) (d : ℕ) :
    (inversePowerReference ε exponent d).natDegree ≤ d := by
  apply (Polynomial.natDegree_C_mul_le _ _).trans
  apply Polynomial.natDegree_sum_le_of_forall_le
  intro k hk
  apply (Polynomial.natDegree_C_mul_le _ _).trans
  calc
    _ ≤ k.val * (Polynomial.C (1 / (2 * ε + 1)) *
        (Polynomial.X : Polynomial ℝ)).natDegree := Polynomial.natDegree_pow_le
    _ ≤ k.val * 1 := Nat.mul_le_mul_left _
      ((Polynomial.natDegree_C_mul_le _ _).trans (by simp))
    _ ≤ d := by simpa using Nat.le_of_lt_succ k.isLt

/-- Evaluating the reference polynomial is the actual binomial partial
sum at the scaled coordinate, with midpoint output scaling included.
Source: arXiv:2602.02016v2, Appendix A, scalar inverse-power approximation. -/
theorem inversePowerReference_eval (ε exponent t : ℝ) (d : ℕ) :
    (inversePowerReference ε exponent d).eval t =
      (ε + 1 / 2) ^ exponent * (binomialSeries ℝ exponent).partialSum
        (d + 1) (t / (2 * ε + 1)) := by
  simp only [inversePowerReference, Polynomial.eval_mul, Polynomial.eval_C,
    Polynomial.eval_finsetSum, Polynomial.eval_pow, Polynomial.eval_X,
    one_div_mul_eq_div, FormalMultilinearSeries.partialSum, binomialSeries_apply,
    List.ofFn_const, List.prod_replicate, smul_eq_mul]
  congr 1
  exact Fin.sum_univ_eq_sum_range
    (fun k : ℕ => Ring.choose exponent k * (t / (2 * ε + 1)) ^ k) (d + 1)

/-- The affine fitting target factors into its positive midpoint scale
and a binomial-series argument. The lower endpoint bound suffices for
this real-power factorization.
Source: arXiv:2602.02016v2, Appendix A, mapping `[ε,1+ε]` to `[-1,1]`. -/
theorem inverse_power_midpoint_factor (ε exponent t : ℝ)
    (hε : 0 < ε) (ht : -1 ≤ t) :
    (chebFromCoordinate ε (1 + ε) t) ^ exponent =
      (ε + 1 / 2) ^ exponent * (1 + t / (2 * ε + 1)) ^ exponent := by
  have hden : (0 : ℝ) < 2 * ε + 1 := by linarith
  have hbase : 0 ≤ 1 + t / (2 * ε + 1) := by
    have h : -1 < t / (2 * ε + 1) := (lt_div_iff₀ hden).2 (by linarith)
    linarith
  have hcoord : chebFromCoordinate ε (1 + ε) t =
      (ε + 1 / 2) * (1 + t / (2 * ε + 1)) := by
    unfold chebFromCoordinate
    field_simp
    ring
  rw [hcoord, Real.mul_rpow (by linarith : 0 ≤ ε + 1 / 2) hbase]

/-- Positive regularization and a valid midpoint coordinate coexist,
arXiv:2602.02016v2, Appendix A. -/
example : (0 : ℝ) < 1 ∧ (-1 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by norm_num

end Transformer.DASH

/-
# DASH — scalar Chebyshev basis and interval mapping

arXiv:2602.02016v2, Appendix A. The input to Clenshaw must be
the mapped coordinate, although the scalar pseudocode calls it `x`.
-/

import Transformer.DASH.SectionA_Clenshaw
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Chebyshev.Basic

noncomputable section

namespace Transformer.DASH

/-- The recurrence used by the implementation is Mathlib's Chebyshev
polynomial evaluated at the argument, arXiv:2602.02016v2, Appendix A. -/
theorem chebT_eq_eval {R : Type*} [CommRing R] (x : R) (k : ℕ) :
    chebT x k = (Polynomial.Chebyshev.T R (k : ℤ)).eval x := by
  induction k using Nat.twoStepInduction with
  | zero => simp [chebT]
  | one => simp [chebT]
  | more k ih ih' =>
    rw [chebT, ih, ih']
    simp [Nat.cast_add, Polynomial.Chebyshev.T_add_two]

/-- The Chebyshev expansion is a cosine expansion under `x=cos θ`.
Source: arXiv:2602.02016v2, Appendix A, the Fourier cosine interpretation. -/
theorem chebT_cos (θ : ℝ) (k : ℕ) : chebT (Real.cos θ) k = Real.cos ((k : ℝ) * θ) := by
  rw [chebT_eq_eval, Polynomial.Chebyshev.T_real_cos]
  simp

/-- Map the reference interval `[-1,1]` into `[a,b]`,
arXiv:2602.02016v2, Appendix A, coefficient-fitting algorithm. -/
def chebFromCoordinate (a b t : ℝ) : ℝ := (b - a) * t / 2 + (b + a) / 2

/-- The inverse interval map omitted as an explicit step in the scalar
Clenshaw pseudocode, arXiv:2602.02016v2, Appendix A, “Important”. -/
def chebToCoordinate (a b x : ℝ) : ℝ := (2 * x - a - b) / (b - a)

/-- Fitting nodes lie in the intended interval after the affine map,
arXiv:2602.02016v2, Appendix A, `algorithm:cbshv-coeff-fitting`. -/
theorem chebFromCoordinate_bounds (a b t : ℝ) (hab : a ≤ b)
    (ht : -1 ≤ t) (ht' : t ≤ 1) :
    a ≤ chebFromCoordinate a b t ∧ chebFromCoordinate a b t ≤ b := by
  unfold chebFromCoordinate
  constructor <;> nlinarith [mul_nonneg (sub_nonneg.mpr hab) (by linarith : 0 ≤ t + 1),
    mul_nonneg (sub_nonneg.mpr hab) (sub_nonneg.mpr ht')]

/-- Fitting-interval hypotheses are satisfiable, arXiv:2602.02016v2, Appendix A. -/
example : (1 : ℝ) ≤ 3 ∧ (-1 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- Evaluation coordinates lie in `[-1,1]` for a nondegenerate fitting
interval, arXiv:2602.02016v2, Appendix A, scalar interval mapping. -/
theorem chebToCoordinate_bounds (a b x : ℝ) (hab : a < b) (hx : a ≤ x) (hx' : x ≤ b) :
    -1 ≤ chebToCoordinate a b x ∧ chebToCoordinate a b x ≤ 1 := by
  unfold chebToCoordinate
  constructor
  · apply (le_div_iff₀ (sub_pos.mpr hab)).mpr
    linarith
  · apply (div_le_iff₀ (sub_pos.mpr hab)).mpr
    linarith

/-- Evaluation-interval hypotheses are satisfiable, arXiv:2602.02016v2, Appendix A. -/
example : (1 : ℝ) < 3 ∧ (1 : ℝ) ≤ 2 ∧ (2 : ℝ) ≤ 3 := by norm_num

/-- Mapping back recovers the original scalar. The coefficients embed the
forward map; they do not remove the need to map the evaluation argument.
Source: arXiv:2602.02016v2, Appendix A, “Important”. -/
theorem chebCoordinate_roundtrip (a b x : ℝ) (hab : a < b) :
    chebFromCoordinate a b (chebToCoordinate a b x) = x := by
  have hd : b - a ≠ 0 := (sub_pos.mpr hab).ne'
  unfold chebFromCoordinate chebToCoordinate
  field_simp
  ring

/-- Nondegenerate intervals exist, arXiv:2602.02016v2, Appendix A. -/
example : (1 : ℝ) < 3 := by norm_num

/-- The inverse map also recovers the reference coordinate from a fitting
node. Source: arXiv:2602.02016v2, Appendix A, the affine fitting map. -/
theorem chebCoordinate_reverse (a b t : ℝ) (hab : a < b) :
    chebToCoordinate a b (chebFromCoordinate a b t) = t := by
  have hd : b - a ≠ 0 := (sub_pos.mpr hab).ne'
  unfold chebToCoordinate chebFromCoordinate
  field_simp
  ring

/-- Nondegenerate fitting maps exist, arXiv:2602.02016v2, Appendix A. -/
example : (1 : ℝ) < 3 := by norm_num

/-- Clenshaw returns an incorrect affine value if the mapping is omitted:
on `[1,3]`, coefficients `[2,1]` represent `f(x)=x` in coordinate `x-2`.
Source: arXiv:2602.02016v2, Appendix A, scalar pseudocode versus its preceding mapping rule. -/
theorem unmapped_scalar_counterexample :
    clenshaw (chebToCoordinate 1 3 2) ([2, 1] : List ℝ) = 2 ∧
      clenshaw (2 : ℝ) ([2, 1] : List ℝ) = 4 := by
  norm_num [chebToCoordinate, clenshaw, clenshawState]

end Transformer.DASH

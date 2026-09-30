/-
# DASH — stability bounds for the actual fitted coefficients

arXiv:2602.02016v2, Appendix A, coefficient fitting followed by scalar
Clenshaw. Sample errors are amplified by at most `1+2d` on the fitting
interval, without an unproved assumption about the fitted coefficients.
-/

import Transformer.DASH.SectionA_FitReproduction
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Chebyshev.RootsExtrema
import Mathlib.Algebra.BigOperators.Fin

open scoped BigOperators

noncomputable section

namespace Transformer.DASH

/-- Coefficient fitting is linear under subtraction of sampled functions.
Source: arXiv:2602.02016v2, Appendix A, the cosine scalar product for `c_k`. -/
theorem chebFitCoefficient_sub (f g : ℝ → ℝ) (a b : ℝ) (N k : ℕ) :
    chebFitCoefficient (fun x => f x - g x) a b N k =
      chebFitCoefficient f a b N k - chebFitCoefficient g a b N k := by
  simp only [chebFitCoefficient, sub_mul, Finset.sum_sub_distrib, mul_sub]

/-- A sample perturbation bounded by `η` changes the constant coefficient
by at most `η` and each other coefficient by at most `2η`.
Source: arXiv:2602.02016v2, Appendix A, the coefficient-fitting normalization. -/
theorem chebFitCoefficient_error (f g : ℝ → ℝ) (a b η : ℝ) (N k : ℕ)
    (hN : 0 < N) (hη : 0 ≤ η)
    (herror : ∀ i : Fin N, |f (chebNode a b N i) - g (chebNode a b N i)| ≤ η) :
    |chebFitCoefficient f a b N k - chebFitCoefficient g a b N k| ≤
      (if k = 0 then 1 else 2) * η := by
  have hn : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  have hweight : 0 ≤ (if k = 0 then (1 : ℝ) else 2) := by split_ifs <;> norm_num
  have hscale : 0 ≤ (if k = 0 then (1 : ℝ) else 2) / N := by positivity
  have hsum : |∑ i : Fin N, (f (chebNode a b N i) - g (chebNode a b N i)) *
      Real.cos ((k : ℝ) * chebAngle N i)| ≤ (N : ℝ) * η := by
    calc
      _ ≤ ∑ i : Fin N, |(f (chebNode a b N i) - g (chebNode a b N i)) *
          Real.cos ((k : ℝ) * chebAngle N i)| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _ : Fin N, η := by
        apply Finset.sum_le_sum
        intro i hi
        rw [abs_mul]
        exact (mul_le_mul (herror i) (Real.abs_cos_le_one _) (abs_nonneg _) hη).trans_eq
          (mul_one η)
      _ = (N : ℝ) * η := by simp
  rw [← chebFitCoefficient_sub, chebFitCoefficient, abs_mul, abs_of_nonneg hscale]
  calc
    _ ≤ ((if k = 0 then 1 else 2) / (N : ℝ)) * ((N : ℝ) * η) :=
      mul_le_mul_of_nonneg_left hsum hscale
    _ = (if k = 0 then 1 else 2) * η := by field_simp

/-- The perturbation assumptions have a nonzero instance,
arXiv:2602.02016v2, Appendix A. -/
example : 0 < (10 : ℕ) ∧ (0 : ℝ) ≤ 1 ∧
    (∀ i : Fin 10, |(fun _ => (1 : ℝ)) (chebNode 1 2 10 i) -
      (fun _ => (0 : ℝ)) (chebNode 1 2 10 i)| ≤ 1) := by norm_num

/-- Every first-kind Chebyshev basis element is bounded by one on the
mapped fitting interval. Source: arXiv:2602.02016v2, Appendix A, interval mapping. -/
theorem chebT_mapped_abs_le_one (a b x : ℝ) (k : ℕ)
    (hab : a < b) (hx : a ≤ x) (hx' : x ≤ b) :
    |chebT (chebToCoordinate a b x) k| ≤ 1 := by
  rw [chebT_eq_eval]
  exact Polynomial.Chebyshev.abs_eval_T_real_le_one _
    (abs_le.2 (chebToCoordinate_bounds a b x hab hx hx'))

/-- Mapped evaluation inside a fitting interval is possible,
arXiv:2602.02016v2, Appendix A. -/
example : (1 : ℝ) < 3 ∧ (1 : ℝ) ≤ 2 ∧ (2 : ℝ) ≤ 3 := by norm_num

/-- The actual fitted-and-evaluated polynomial amplifies uniformly bounded
sample perturbations by at most `2d+1` throughout `[a,b]`. This holds for
any degree; exact polynomial recovery additionally requires `d<N`.
Source: arXiv:2602.02016v2, Appendix A, coefficient fitting and scalar Clenshaw. -/
theorem chebFitEval_error (f g : ℝ → ℝ) (a b x η : ℝ) (N d : ℕ)
    (hab : a < b) (hx : a ≤ x) (hx' : x ≤ b) (hN : 0 < N) (hη : 0 ≤ η)
    (herror : ∀ i : Fin N, |f (chebNode a b N i) - g (chebNode a b N i)| ≤ η) :
    |chebFitEval f a b N d x - chebFitEval g a b N d x| ≤ (2 * (d : ℝ) + 1) * η := by
  have hsum : (∑ k : Fin (d + 1), (if k.val = 0 then (1 : ℝ) else 2) * η) =
      (2 * (d : ℝ) + 1) * η := by
    rw [Fin.sum_univ_succ]
    simp only [Fin.val_zero, ite_true, one_mul, Fin.val_succ, Nat.add_eq_zero_iff,
      one_ne_zero, and_false, ite_false, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul]
    ring
  calc
    _ = |∑ k : Fin (d + 1),
        (chebFitCoefficient f a b N k - chebFitCoefficient g a b N k) *
          chebT (chebToCoordinate a b x) k| := by
      rw [chebFitEval_eq_sum, chebFitEval_eq_sum, ← Finset.sum_sub_distrib]
      simp_rw [← sub_mul]
    _ ≤ ∑ k : Fin (d + 1),
        |(chebFitCoefficient f a b N k - chebFitCoefficient g a b N k) *
          chebT (chebToCoordinate a b x) k| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ k : Fin (d + 1), (if k.val = 0 then (1 : ℝ) else 2) * η := by
      apply Finset.sum_le_sum
      intro k hk
      rw [abs_mul]
      have hcoef := chebFitCoefficient_error f g a b η N k hN hη herror
      have hbound := chebT_mapped_abs_le_one a b x k hab hx hx'
      have hnonneg : 0 ≤ (if k.val = 0 then (1 : ℝ) else 2) * η := by
        split_ifs <;> positivity
      exact (mul_le_mul hcoef hbound (abs_nonneg _) hnonneg).trans_eq (mul_one _)
    _ = (2 * (d : ℝ) + 1) * η := hsum

/-- Fitted-output stability assumptions are jointly satisfiable,
arXiv:2602.02016v2, Appendix A. -/
example : (1 : ℝ) < 3 ∧ (1 : ℝ) ≤ 2 ∧ (2 : ℝ) ≤ 3 ∧ 0 < (1000 : ℕ) ∧
    (0 : ℝ) ≤ 1 ∧ (∀ i : Fin 1000, |(fun _ => (1 : ℝ)) (chebNode 1 3 1000 i) -
      (fun _ => (0 : ℝ)) (chebNode 1 3 1000 i)| ≤ 1) := by norm_num

end Transformer.DASH

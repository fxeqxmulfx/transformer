/-
# DASH — approximation bounds for the coefficient-fitting algorithm

arXiv:2602.02016v2, Appendix A. Exact polynomial reproduction and
sample stability give an error bound for the fitted array itself,
in terms of any verified degree-`d` reference approximation.
-/

import Transformer.DASH.SectionA_FitStability

open scoped BigOperators

noncomputable section

namespace Transformer.DASH

/-- A degree-`d` reference approximation of error `η` gives an error at
most `(2d+2)η` for the actual fitted-and-evaluated array, provided `d<N`.
This states the accuracy assumptions absent from the source's unqualified
description of a “good approximation”; they concern an independently
verified polynomial, not the algorithm's output.
Source: arXiv:2602.02016v2, Appendix A, coefficient fitting and scalar Clenshaw. -/
theorem chebFitEval_approximation (f : ℝ → ℝ) (a b x η : ℝ) (N d : ℕ)
    (c : Fin (d + 1) → ℝ) (hab : a < b) (hx : a ≤ x) (hx' : x ≤ b)
    (hd : d < N) (hη : 0 ≤ η)
    (happrox : ∀ y : ℝ, a ≤ y → y ≤ b →
      |(∑ j : Fin (d + 1), c j * chebT (chebToCoordinate a b y) j) - f y| ≤ η) :
    |chebFitEval f a b N d x - f x| ≤ (2 * (d : ℝ) + 2) * η := by
  let p : ℝ → ℝ := fun y =>
    ∑ j : Fin (d + 1), c j * chebT (chebToCoordinate a b y) j
  have hN : 0 < N := by omega
  have hsample : ∀ i : Fin N, |f (chebNode a b N i) - p (chebNode a b N i)| ≤ η := by
    intro i
    obtain ⟨hi, hi'⟩ := chebNode_bounds a b N i hab.le
    rw [abs_sub_comm]
    exact happrox _ hi hi'
  have hfit := chebFitEval_error f p a b x η N d hab hx hx' hN hη hsample
  have hrecover : chebFitEval p a b N d x = p x :=
    chebFitEval_reproduces a b x N d c hab hd
  rw [hrecover] at hfit
  calc
    _ ≤ |chebFitEval f a b N d x - p x| + |p x - f x| := abs_sub_le _ _ _
    _ ≤ (2 * (d : ℝ) + 1) * η + η := add_le_add hfit (happrox x hx hx')
    _ = (2 * (d : ℝ) + 2) * η := by ring

/-- The approximation hypotheses hold for a nonconstant exact reference.
Source: arXiv:2602.02016v2, Appendix A. -/
example : (-1 : ℝ) < 1 ∧ (-1 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℕ) < 1000 ∧
    (0 : ℝ) ≤ 0 ∧ (∀ y : ℝ, (-1 : ℝ) ≤ y → y ≤ 1 →
      |(∑ j : Fin 2, (if j.val = 1 then (1 : ℝ) else 0) *
        chebT (chebToCoordinate (-1) 1 y) j) - y| ≤ 0) := by
  norm_num [Fin.sum_univ_succ, chebT, chebToCoordinate]

/-- For the source's regularized inverse-power target, an explicit
reference expansion on `[-1,1]` bounds its actual fitted Clenshaw output.
Source: arXiv:2602.02016v2, Appendix A, scalar fitting on `[ε,1+ε]`. -/
theorem chebFit_inverse_power_approximation (ε exponent t η : ℝ) (N d : ℕ)
    (c : Fin (d + 1) → ℝ) (ht : -1 ≤ t) (ht' : t ≤ 1)
    (hd : d < N) (hη : 0 ≤ η)
    (happrox : ∀ u : ℝ, -1 ≤ u → u ≤ 1 →
      |(∑ j : Fin (d + 1), c j * chebT u j) -
        (chebFromCoordinate ε (1 + ε) u) ^ exponent| ≤ η) :
    |clenshaw t (chebFit (fun x : ℝ => x ^ exponent) ε (1 + ε) N d) -
      (chebFromCoordinate ε (1 + ε) t) ^ exponent| ≤ (2 * (d : ℝ) + 2) * η := by
  have hab : ε < 1 + ε := by linarith
  obtain ⟨hx, hx'⟩ := chebFromCoordinate_bounds ε (1 + ε) t hab.le ht ht'
  have h := chebFitEval_approximation (fun x : ℝ => x ^ exponent)
    ε (1 + ε) (chebFromCoordinate ε (1 + ε) t) η N d c hab hx hx' hd hη (by
      intro y hy hy'
      obtain ⟨hu, hu'⟩ := chebToCoordinate_bounds ε (1 + ε) y hab hy hy'
      have h := happrox _ hu hu'
      rwa [chebCoordinate_roundtrip ε (1 + ε) y hab] at h)
  simpa only [chebFitEval, chebCoordinate_reverse ε (1 + ε) t hab] using h

/-- Inverse-power fit assumptions have the exact exponent-zero instance.
Source: arXiv:2602.02016v2, Appendix A. -/
example : (-1 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℕ) < 1000 ∧ (0 : ℝ) ≤ 0 ∧
    (∀ u : ℝ, -1 ≤ u → u ≤ 1 →
      |(∑ j : Fin 1, (1 : ℝ) * chebT u j) -
        (chebFromCoordinate 1 2 u) ^ (0 : ℝ)| ≤ 0) := by
  norm_num [chebT]

end Transformer.DASH

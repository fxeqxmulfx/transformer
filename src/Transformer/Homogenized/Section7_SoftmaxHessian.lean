/-
# Actual first and second coordinate derivatives of the softmax

ArXiv:2604.01978v1, §7, the computations preceding
`eq:estimate_bound_h` and `eq:estimate_bound_g`. The actual derivatives
are taken in the original score coordinates. Differentiating the original
exponential quotient gives the first derivative with its correct sign;
the genuine product and evaluation chain rules then give the Hessian.

The source's first derivative in its earlier `claim:softmax_is_smooth`
has the sign reversed; `SoftmaxDerivatives.lean` already corrects it.
The formulas here agree with the correctly signed formulas in §7.
Their softmax factors remain attached to the original derivatives.
The first summed bound is independent of the number of scores. Such
summed estimates are needed for the actual covariance contraction in
Gram stability, rather than a separate bound on each Hessian entry.
-/

import Transformer.Homogenized.SoftmaxDerivatives

open Real
open scoped BigOperators

namespace Transformer.Homogenized

/-- The actual score derivative keeps its original softmax factor.
Source: arXiv:2604.01978v1, §7, the first derivative formula preceding
`eq:estimate_bound_h`, with the earlier sign error corrected. -/
theorem fderiv_softmaxWeight_factor {m : ℕ} (hm : 0 < m) (u : Idx m → ℝ) (j r : Idx m) :
    fderiv ℝ (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) u (Pi.single r 1) =
      Perspective.softmaxWeight u j * ((if r = j then (1 : ℝ) else 0) - Perspective.softmaxWeight u r) := by
  rw [fderiv_softmaxWeight hm]
  split_ifs <;> ring

/-- The original partial derivative is genuinely differentiable with
the stated derivative, by the product rule on the original softmax.
Source: arXiv:2604.01978v1, §7, the Hessian computation preceding
`eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem hasFDerivAt_softmaxWeight_partial {m : ℕ} (hm : 0 < m) (u : Idx m → ℝ) (j s : Idx m) :
    HasFDerivAt (fun v : Idx m → ℝ =>
        fderiv ℝ (fun w : Idx m → ℝ => Perspective.softmaxWeight w j) v (Pi.single s 1))
      (((if s = j then (1 : ℝ) else 0) - Perspective.softmaxWeight u s) •
          fderiv ℝ (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) u -
        Perspective.softmaxWeight u j •
          fderiv ℝ (fun v : Idx m → ℝ => Perspective.softmaxWeight v s) u) u := by
  have hχ (a : Idx m) : HasFDerivAt (fun v : Idx m → ℝ => Perspective.softmaxWeight v a)
      (fderiv ℝ (fun v : Idx m → ℝ => Perspective.softmaxWeight v a) u) u :=
    ((contDiff_softmaxWeight hm a).differentiable (by simp) u).hasFDerivAt
  have hfun : (fun v : Idx m → ℝ =>
      fderiv ℝ (fun w : Idx m → ℝ => Perspective.softmaxWeight w j) v (Pi.single s 1)) =
      fun v => Perspective.softmaxWeight v j *
        ((if s = j then (1 : ℝ) else 0) - Perspective.softmaxWeight v s) :=
    funext fun v => fderiv_softmaxWeight_factor hm v j s
  rw [hfun]
  have h := (hχ j).fun_mul ((hasFDerivAt_const (if s = j then (1 : ℝ) else 0) u).fun_sub (hχ s))
  convert h using 1
  ext v
  simp only [add_apply, sub_apply, smul_apply, zero_apply, smul_eq_mul]
  ring

/-- The actual second derivative on the original two score axes is
the source's explicit softmax Hessian formula.
Source: arXiv:2604.01978v1, §7, the formula for `χ''` preceding
`eq:estimate_bound_h`. Both arguments refer to the actual iterated derivative. -/
theorem iteratedFDeriv_two_softmaxWeight {m : ℕ} (hm : 0 < m) (u : Idx m → ℝ) (j r s : Idx m) :
    iteratedFDeriv ℝ 2 (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) u
      ![Pi.single r 1, Pi.single s 1] = Perspective.softmaxWeight u j *
        (((if r = j then (1 : ℝ) else 0) - Perspective.softmaxWeight u r) *
          ((if s = j then (1 : ℝ) else 0) - Perspective.softmaxWeight u s) -
          Perspective.softmaxWeight u s * ((if r = s then (1 : ℝ) else 0) - Perspective.softmaxWeight u r)) := by
  let φ : (Idx m → ℝ) → ℝ := fun v => Perspective.softmaxWeight v j
  have hD : ContDiff ℝ 1 (fderiv ℝ φ) := (contDiff_softmaxWeight hm j).fderiv_right (by simp)
  have heval := (((hD.differentiable (by norm_num)) u).hasFDerivAt.clm_apply
    (hasFDerivAt_const (Pi.single s (1 : ℝ)) u)).fderiv
  have haxis : fderiv ℝ (fun v => fderiv ℝ φ v (Pi.single s 1)) u (Pi.single r 1) =
      fderiv ℝ (fderiv ℝ φ) u (Pi.single r 1) (Pi.single s 1) := by
    rw [heval]
    simp only [ContinuousLinearMap.comp_zero, zero_add, ContinuousLinearMap.flip_apply]
  rw [iteratedFDeriv_two_apply]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
  rw [← haxis, (hasFDerivAt_softmaxWeight_partial hm u j s).fderiv]
  simp only [sub_apply, smul_apply, smul_eq_mul]
  rw [fderiv_softmaxWeight_factor hm u j r, fderiv_softmaxWeight_factor hm u s r]
  ring

/-- The absolute distance from a coordinate mass to any normalized
nonnegative weight vector sums to at most two.
Source: arXiv:2604.01978v1, §7, normalization used in the derivative
estimates preceding `eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem sum_abs_probability_coordinate_sub_le {m : ℕ} (p : Idx m → ℝ)
    (hp : ∀ r, 0 ≤ p r) (hs : ∑ r, p r = 1) (j : Idx m) :
    (∑ r : Idx m, |(if r = j then (1 : ℝ) else 0) - p r|) ≤ 2 := by
  calc
    _ ≤ ∑ r : Idx m, ((if r = j then (1 : ℝ) else 0) + p r) := by
      apply Finset.sum_le_sum
      intro r hr
      have hδ : 0 ≤ (if r = j then (1 : ℝ) else 0) := by split_ifs <;> norm_num
      simpa only [sub_zero, zero_sub, abs_neg, abs_of_nonneg hδ, abs_of_nonneg (hp r)] using
        abs_sub_le (if r = j then (1 : ℝ) else 0) 0 (p r)
    _ = 2 := by rw [Finset.sum_add_distrib, hs]; norm_num

/-- The absolute actual score derivative keeps a normalized coordinate
envelope as well as the original softmax factor.
Source: arXiv:2604.01978v1, §7, the pointwise first derivative
estimate used in `eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem abs_fderiv_softmaxWeight_coordinate_le {m : ℕ} (hm : 0 < m)
    (u : Idx m → ℝ) (j r : Idx m) :
    |fderiv ℝ (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) u (Pi.single r 1)| ≤
      Perspective.softmaxWeight u j * ((if r = j then (1 : ℝ) else 0) + Perspective.softmaxWeight u r) := by
  rw [fderiv_softmaxWeight_factor hm, abs_mul, abs_of_nonneg (Perspective.softmaxWeight_nonneg u j)]
  have hδ : 0 ≤ (if r = j then (1 : ℝ) else 0) := by split_ifs <;> norm_num
  apply mul_le_mul_of_nonneg_left _ (Perspective.softmaxWeight_nonneg u j)
  simpa only [sub_zero, zero_sub, abs_neg, abs_of_nonneg hδ,
    abs_of_nonneg (Perspective.softmaxWeight_nonneg u r)] using
    abs_sub_le (if r = j then (1 : ℝ) else 0) 0 (Perspective.softmaxWeight u r)

/-- The sum of absolute actual score derivatives is bounded by twice
the original softmax weight, independently of the number of scores.
Source: arXiv:2604.01978v1, §7, the first derivative estimate used
in `eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem sum_abs_fderiv_softmaxWeight_le {m : ℕ} (hm : 0 < m) (u : Idx m → ℝ) (j : Idx m) :
    (∑ r : Idx m, |fderiv ℝ (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) u (Pi.single r 1)|) ≤
      2 * Perspective.softmaxWeight u j := by
  simp only [fderiv_softmaxWeight_factor hm, abs_mul,
    abs_of_nonneg (Perspective.softmaxWeight_nonneg u j)]
  rw [← Finset.mul_sum]
  have h := sum_abs_probability_coordinate_sub_le (Perspective.softmaxWeight u)
    (Perspective.softmaxWeight_nonneg u) (Perspective.sum_softmaxWeight hm u) j
  exact (mul_le_mul_of_nonneg_left h (Perspective.softmaxWeight_nonneg u j)).trans_eq (mul_comm _ _)

/-- Two genuine biased softmax weights satisfy normalization and
positivity, and their actual coordinate Hessian is nonzero. -/
example : ∃ u : Idx 2 → ℝ,
    (0 : ℕ) < 2 ∧ (∀ r, 0 ≤ Perspective.softmaxWeight u r) ∧
    (∑ r : Idx 2, Perspective.softmaxWeight u r) = 1 ∧
    Perspective.softmaxWeight u (0 : Idx 2) = 3 / 4 ∧
    Perspective.softmaxWeight u (1 : Idx 2) = 1 / 4 ∧
    iteratedFDeriv ℝ 2 (fun v : Idx 2 → ℝ => Perspective.softmaxWeight v 0) u
      ![Pi.single (0 : Idx 2) 1, Pi.single (0 : Idx 2) 1] = -(3 / 32 : ℝ) := by
  let u : Idx 2 → ℝ := ![Real.log 3, 0]
  have hw0 : Perspective.softmaxWeight u (0 : Idx 2) = 3 / 4 := by
    norm_num [Perspective.softmaxWeight, u, Real.exp_log (by norm_num : (0 : ℝ) < 3), Fin.sum_univ_two]
  have hw1 : Perspective.softmaxWeight u (1 : Idx 2) = 1 / 4 := by
    norm_num [Perspective.softmaxWeight, u, Real.exp_log (by norm_num : (0 : ℝ) < 3), Fin.sum_univ_two]
  refine ⟨u, by norm_num, Perspective.softmaxWeight_nonneg u, Perspective.sum_softmaxWeight (by norm_num) u,
    hw0, hw1, ?_⟩
  rw [iteratedFDeriv_two_softmaxWeight (by norm_num), hw0]
  norm_num

end Transformer.Homogenized

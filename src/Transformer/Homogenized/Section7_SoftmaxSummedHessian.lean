module

public import Transformer.Homogenized.Section7_SoftmaxHessian

/-!
# Summed actual softmax Hessians without a factor from the number of scores

ArXiv:2604.01978v1, §7, `eq:estimate_bound_h` and
`eq:estimate_bound_g`. A normalized softmax coordinate envelope has mass
two, and the corresponding second derivative envelope has total mass six.
The actual Hessian formula therefore gives a sum of absolute coordinate
Hessians bounded by six times the original softmax weight.

Contracting the actual Hessian with a matrix bounded entrywise retains
that same constant. No power of the number of scores is introduced by
bounding individual entries separately. Every bound is pointwise in the
original scores; its softmax weight stays inside any later expectation.
The source's displayed expectation estimates put random softmax factors
outside the expectation. The pointwise estimates here are the inputs to
an aggregate estimate before taking the original Gaussian expectation.
-/

@[expose] public section

open Real
open scoped BigOperators

namespace Transformer.Homogenized

/-- The coordinate envelope has mass two whenever the weights sum to one.
Source: arXiv:2604.01978v1, §7, normalization in the computations
preceding `eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem sum_probability_coordinate_envelope {m : ℕ} (p : Idx m → ℝ)
    (hs : ∑ r, p r = 1) (j : Idx m) :
    (∑ r : Idx m, ((if r = j then (1 : ℝ) else 0) + p r)) = 2 := by
  rw [Finset.sum_add_distrib, hs]
  norm_num

/-- The explicit original Hessian coordinate envelope has total
mass six for every normalized probability vector.
Source: arXiv:2604.01978v1, §7, the summed `χ''` estimate used
in `eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem sum_probability_hessian_envelope {m : ℕ} (p : Idx m → ℝ)
    (hs : ∑ r, p r = 1) (j : Idx m) :
    (∑ r : Idx m, ∑ s : Idx m,
      (((if r = j then (1 : ℝ) else 0) + p r) * ((if s = j then (1 : ℝ) else 0) + p s) +
        p s * ((if r = s then (1 : ℝ) else 0) + p r))) = 6 := by
  have hfirst : (∑ r : Idx m, ∑ s : Idx m,
      ((if r = j then (1 : ℝ) else 0) + p r) * ((if s = j then (1 : ℝ) else 0) + p s)) = 4 := by
    simp_rw [← Finset.mul_sum]
    rw [← Finset.sum_mul, sum_probability_coordinate_envelope p hs j]
    norm_num
  have hsecond : (∑ r : Idx m, ∑ s : Idx m, p s * ((if r = s then (1 : ℝ) else 0) + p r)) = 2 := by
    rw [Finset.sum_comm]
    simp_rw [← Finset.mul_sum, sum_probability_coordinate_envelope p hs]
    rw [← Finset.sum_mul, hs]
    norm_num
  simp only [Finset.sum_add_distrib]
  rw [hfirst, hsecond]
  norm_num

/-- The absolute actual Hessian entry is dominated by the explicit
normalized envelope, retaining the original softmax factor.
Source: arXiv:2604.01978v1, §7, the pointwise `χ''` estimate
used in `eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem abs_iteratedFDeriv_two_softmaxWeight_coordinate_le {m : ℕ} (hm : 0 < m)
    (u : Idx m → ℝ) (j r s : Idx m) :
    |iteratedFDeriv ℝ 2 (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) u
      ![Pi.single r 1, Pi.single s 1]| ≤ Perspective.softmaxWeight u j *
        (((if r = j then (1 : ℝ) else 0) + Perspective.softmaxWeight u r) *
          ((if s = j then (1 : ℝ) else 0) + Perspective.softmaxWeight u s) +
          Perspective.softmaxWeight u s * ((if r = s then (1 : ℝ) else 0) + Perspective.softmaxWeight u r)) := by
  have hδ (a b : Idx m) : 0 ≤ (if a = b then (1 : ℝ) else 0) := by split_ifs <;> norm_num
  have hb (a b : Idx m) : |(if a = b then (1 : ℝ) else 0) - Perspective.softmaxWeight u a| ≤
      (if a = b then (1 : ℝ) else 0) + Perspective.softmaxWeight u a := by
    simpa only [sub_zero, zero_sub, abs_neg, abs_of_nonneg (hδ a b),
      abs_of_nonneg (Perspective.softmaxWeight_nonneg u a)] using
      abs_sub_le (if a = b then (1 : ℝ) else 0) 0 (Perspective.softmaxWeight u a)
  rw [iteratedFDeriv_two_softmaxWeight hm, abs_mul,
    abs_of_nonneg (Perspective.softmaxWeight_nonneg u j)]
  apply mul_le_mul_of_nonneg_left _ (Perspective.softmaxWeight_nonneg u j)
  have h := abs_sub_le
    (((if r = j then (1 : ℝ) else 0) - Perspective.softmaxWeight u r) *
      ((if s = j then (1 : ℝ) else 0) - Perspective.softmaxWeight u s)) 0
    (Perspective.softmaxWeight u s * ((if r = s then (1 : ℝ) else 0) - Perspective.softmaxWeight u r))
  simp only [sub_zero, zero_sub, abs_neg, abs_mul,
    abs_of_nonneg (Perspective.softmaxWeight_nonneg u s)] at h
  exact h.trans (add_le_add (mul_le_mul (hb r j) (hb s j) (abs_nonneg _)
    (add_nonneg (hδ r j) (Perspective.softmaxWeight_nonneg u r)))
    (mul_le_mul_of_nonneg_left (hb r s) (Perspective.softmaxWeight_nonneg u s)))

/-- The sum of all absolute actual coordinate Hessians is at most
six times the original softmax weight, independently of the score count.
Source: arXiv:2604.01978v1, §7, the summed Hessian estimate
underlying `eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem sum_abs_iteratedFDeriv_two_softmaxWeight_le {m : ℕ}
    (hm : 0 < m) (u : Idx m → ℝ) (j : Idx m) :
    (∑ r : Idx m, ∑ s : Idx m,
      |iteratedFDeriv ℝ 2 (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) u
        ![Pi.single r 1, Pi.single s 1]|) ≤ 6 * Perspective.softmaxWeight u j := by
  calc
    _ ≤ ∑ r : Idx m, ∑ s : Idx m, Perspective.softmaxWeight u j *
        (((if r = j then (1 : ℝ) else 0) + Perspective.softmaxWeight u r) *
          ((if s = j then (1 : ℝ) else 0) + Perspective.softmaxWeight u s) +
          Perspective.softmaxWeight u s * ((if r = s then (1 : ℝ) else 0) + Perspective.softmaxWeight u r)) :=
      Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun s hs =>
        abs_iteratedFDeriv_two_softmaxWeight_coordinate_le hm u j r s
    _ = _ := by
      simp_rw [← Finset.mul_sum]
      rw [sum_probability_hessian_envelope (Perspective.softmaxWeight u) (Perspective.sum_softmaxWeight hm u) j]
      ring

/-- An entrywise bounded matrix contracts the actual softmax Hessian
with the same universal constant. Source: arXiv:2604.01978v1,
§7, the covariance contractions in `eq:estimate_bound_h` and
`eq:estimate_bound_g`, with softmax factors retained pointwise. -/
theorem abs_sum_matrix_softmax_hessian_le {m : ℕ} (hm : 0 < m) (u : Idx m → ℝ)
    (j : Idx m) (B : Matrix (Idx m) (Idx m) ℝ) (ε : ℝ) (hB : ∀ r s, |B r s| ≤ ε) :
    |∑ r : Idx m, ∑ s : Idx m, B r s *
      iteratedFDeriv ℝ 2 (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) u
        ![Pi.single r 1, Pi.single s 1]| ≤ 6 * ε * Perspective.softmaxWeight u j := by
  have hε : 0 ≤ ε := (abs_nonneg (B j j)).trans (hB j j)
  calc
    _ ≤ ∑ r : Idx m, ∑ s : Idx m, |B r s *
        iteratedFDeriv ℝ 2 (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) u
          ![Pi.single r 1, Pi.single s 1]| :=
      (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun r hr => Finset.abs_sum_le_sum_abs _ _)
    _ ≤ ∑ r : Idx m, ∑ s : Idx m, ε *
        |iteratedFDeriv ℝ 2 (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) u
          ![Pi.single r 1, Pi.single s 1]| :=
      Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun s hs => by
        rw [abs_mul]
        exact mul_le_mul_of_nonneg_right (hB r s) (abs_nonneg _)
    _ = ε * ∑ r : Idx m, ∑ s : Idx m,
        |iteratedFDeriv ℝ 2 (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) u
          ![Pi.single r 1, Pi.single s 1]| := by simp only [Finset.mul_sum]
    _ ≤ ε * (6 * Perspective.softmaxWeight u j) :=
      mul_le_mul_of_nonneg_left (sum_abs_iteratedFDeriv_two_softmaxWeight_le hm u j) hε
    _ = _ := by ring

/-- A biased genuine two-score softmax and the nonzero identity
matrix satisfy every normalization, positivity and matrix-bound premise. -/
example : ∃ (u : Idx 2 → ℝ) (B : Matrix (Idx 2) (Idx 2) ℝ),
    (0 : ℕ) < 2 ∧ (∀ r, 0 ≤ Perspective.softmaxWeight u r) ∧
    (∑ r : Idx 2, Perspective.softmaxWeight u r) = 1 ∧
    B = 1 ∧ (∀ r s, |B r s| ≤ 1) ∧ Perspective.softmaxWeight u (0 : Idx 2) = 3 / 4 := by
  let u : Idx 2 → ℝ := ![Real.log 3, 0]
  refine ⟨u, 1, by norm_num, Perspective.softmaxWeight_nonneg u,
    Perspective.sum_softmaxWeight (by norm_num) u, rfl, ?_, ?_⟩
  · intro r s
    by_cases h : r = s <;> simp [Matrix.one_apply, h]
  · norm_num [Perspective.softmaxWeight, u, Real.exp_log (by norm_num : (0 : ℝ) < 3), Fin.sum_univ_two]

end Transformer.Homogenized

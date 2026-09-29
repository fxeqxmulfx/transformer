/-
# A convex atomic mixture changes the original score penalty

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §2–§3.1, equations
`eq:attention_only` and `eq:attention_only_obj`.  This module studies a
possible convexification using atoms that are themselves genuine softmax
heads, with trainable query-key score products.  It compares the positively
homogeneous atomic cost with the paper's four-matrix weight decay.
-/

import Transformer.Convexifying.Section3_TrainableScores

namespace Transformer.Convexifying

/-- Centered first-row softmax weight for the two-token input `[1, 0]`.
Source: arXiv:2211.11052v1, equation `eq:attention_only`. -/
noncomputable def centeredSoftmax (p : ℝ) : ℝ :=
  Real.exp p / (Real.exp p + 1) - 1 / 2

/-- The nonuniform component of a genuine head with score product `p` and
value-output product `t`.  Source: equations `eq:attention_only`–
`eq:attention_only_obj`, specialized to one scalar feature. -/
noncomputable def reducedHeadDifference (p t : ℝ) : ℝ :=
  t * centeredSoftmax p

/-- The product-based lower bound `|p| + |t|` for four-factor weight decay.
Source: equation `eq:attention_only_obj`, specialized to a scalar head and
`β = 1`. -/
def reducedHeadCost (p t : ℝ) : ℝ := |p| + |t|

/-- A finite mixture of genuine softmax heads, retaining trainable scores.
Source: arXiv:2211.11052v1, equation `eq:attention_only`, extended to a
finite number of heads. -/
noncomputable def reducedMixtureDifference (heads : List (ℝ × ℝ)) : ℝ :=
  (heads.map fun h => reducedHeadDifference h.1 h.2).sum

/-- The sum of product-based lower bounds on four-factor weight decay for a
finite mixture of scalar heads.  Source: equation `eq:attention_only_obj`. -/
def reducedMixtureCost (heads : List (ℝ × ℝ)) : ℝ :=
  (heads.map fun h => reducedHeadCost h.1 h.2).sum

/-- The centered two-token softmax weight grows at most linearly with the
query-key score product.  Source: equation `eq:attention_only`. -/
private theorem abs_centeredSoftmax_le (p : ℝ) :
    |centeredSoftmax p| ≤ |p| := by
  let e := Real.exp p
  have he : 0 < e := Real.exp_pos p
  have hd : 0 < 2 * (e + 1) := by positivity
  have hEq : centeredSoftmax p = (e - 1) / (2 * (e + 1)) := by
    dsimp [centeredSoftmax, e]
    field_simp
    ring
  rw [hEq, abs_div, abs_of_pos hd]
  by_cases hp : |p| ≤ 1
  · have hExp : |e - 1| ≤ 2 * |p| := Real.abs_exp_sub_one_le hp
    apply (div_le_iff₀ hd).2
    have hmul : 0 ≤ |p| * e := mul_nonneg (abs_nonneg _) (le_of_lt he)
    nlinarith
  · have hp' : 1 < |p| := lt_of_not_ge hp
    have hnum : |e - 1| ≤ e + 1 := by
      apply abs_le.mpr
      constructor <;> linarith
    apply (div_le_iff₀ hd).2
    nlinarith

/-- The reduced regularizer is nonnegative for every finite mixture.
Source: equation `eq:attention_only_obj`, specialized. -/
theorem reducedMixtureCost_nonneg (heads : List (ℝ × ℝ)) :
    0 ≤ reducedMixtureCost heads := by
  induction heads with
  | nil => simp [reducedMixtureCost]
  | cons h hs ih =>
      simp only [reducedMixtureCost, List.map_cons, List.sum_cons] at ih ⊢
      dsimp [reducedHeadCost]
      positivity

/-- The nonuniform output of any finite number of original softmax heads is
quadratic in the total product-based lower bound on their weight decay.
This obstructs a positively homogeneous exact atomic cost near the origin.
Source: equations `eq:attention_only`–`eq:attention_only_obj`, specialized. -/
theorem reducedMixtureDifference_quadratic (heads : List (ℝ × ℝ)) :
    |reducedMixtureDifference heads| ≤ reducedMixtureCost heads ^ 2 := by
  induction heads with
  | nil => simp [reducedMixtureDifference, reducedMixtureCost]
  | cons h hs ih =>
      have hh : |reducedHeadDifference h.1 h.2| ≤ |h.1| * |h.2| := by
        rw [reducedHeadDifference, abs_mul]
        simpa [mul_comm] using
          (mul_le_mul_of_nonneg_left (abs_centeredSoftmax_le h.1) (abs_nonneg h.2))
      have htail : 0 ≤ reducedMixtureCost hs := reducedMixtureCost_nonneg hs
      have hp : 0 ≤ |h.1| := abs_nonneg _
      have ht : 0 ≤ |h.2| := abs_nonneg _
      have htriangle :
          |reducedHeadDifference h.1 h.2 + reducedMixtureDifference hs| ≤
            |reducedHeadDifference h.1 h.2| + |reducedMixtureDifference hs| :=
        abs_add_le _ _
      change |reducedHeadDifference h.1 h.2 + reducedMixtureDifference hs| ≤
        (|h.1| + |h.2| + reducedMixtureCost hs) ^ 2
      nlinarith [mul_nonneg hp ht,
        mul_nonneg (add_nonneg hp ht) htail,
        sq_nonneg |h.1|, sq_nonneg |h.2|]

/-- A standard signed-atom penalty: each coefficient is charged linearly,
with the fixed score cost included in the atom's weight.  This is positively
homogeneous in the coefficient and yields a convex finite-mixture gauge.
Source: equations `eq:attention_only`–`eq:attention_only_obj`, candidate
convexification. -/
def atomicHeadCost (p t : ℝ) : ℝ := |t| * (1 + |p|)

/-- The genuine softmax atom with score `log 2` and coefficient `1/100`
has nonuniform output `1/600`.  Source: equation `eq:attention_only`. -/
theorem smallAtomicDifference :
    reducedHeadDifference (Real.log 2) (1 / 100) = 1 / 600 := by
  norm_num [reducedHeadDifference, centeredSoftmax,
    Real.exp_log (by norm_num : (0 : ℝ) < 2)]

/-- The same atom costs at most `1/50` under the positively homogeneous
penalty.  Source: equation `eq:attention_only_obj`, candidate atomic cost. -/
theorem smallAtomicCost :
    atomicHeadCost (Real.log 2) (1 / 100) ≤ 1 / 50 := by
  have hlog0 : 0 ≤ Real.log 2 := Real.log_nonneg (by norm_num)
  have hlog1 : Real.log 2 ≤ 1 := by
    have h := Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 2)
    norm_num at h
    exact h
  simp only [atomicHeadCost, abs_of_nonneg hlog0, abs_of_nonneg (by norm_num :
    (0 : ℝ) ≤ 1 / 100)]
  nlinarith

/-- **Gap for every finite number of trainable-score heads.**  A single
genuine softmax atom has nonuniform output `1/600` and atomic cost at most
`1/50`, but no finite mixture of original heads can achieve that output
within the same reduced four-factor weight-decay budget.  Hence this
positively homogeneous atomic mixture does not preserve the regularized
problem in equation `eq:attention_only_obj`. -/
theorem no_small_original_mixture (heads : List (ℝ × ℝ))
    (houtput : reducedMixtureDifference heads = 1 / 600) :
    1 / 50 < reducedMixtureCost heads := by
  have hbound := reducedMixtureDifference_quadratic heads
  have hnonneg := reducedMixtureCost_nonneg heads
  rw [houtput] at hbound
  rw [abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 600)] at hbound
  by_contra h
  have hle : reducedMixtureCost heads ≤ 1 / 50 := le_of_not_gt h
  nlinarith [sq_nonneg (reducedMixtureCost heads - 1 / 50)]

/-- The nonuniform-output hypothesis of the preceding theorem is realized
by a genuine score and coefficient.  Source: equation
`eq:attention_only`, specialized. -/
example : ∃ heads : List (ℝ × ℝ),
    reducedMixtureDifference heads = 1 / 600 := by
  refine ⟨[(Real.log 2, 1 / 100)], ?_⟩
  simpa [reducedMixtureDifference] using smallAtomicDifference

end Transformer.Convexifying

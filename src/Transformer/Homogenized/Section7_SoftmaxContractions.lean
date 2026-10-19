module

public import Transformer.Homogenized.Section7_SoftmaxSummedHessian

/-!
# Normalized softmax matrix and derivative contractions

ArXiv:2604.01978v1, §7, `eq:estimate_bound_h`,
`eq:estimate_bound_g` and `eq:Lispchitz_bound_softmax`. Entrywise matrix
bounds control actual coordinate products through their summed absolute
values. The actual softmax derivative sums therefore give the universal
constant four for mixed gradient contractions.

Normalization also aggregates the original Hessian and mixed gradient
contractions over the two original output weights. Their constants remain
six and four, with no factor from the score count. All these estimates
are pointwise before the Gaussian expectation, so the random softmax
weights are never moved outside an expectation. The last estimate is
the direct change of the original Gram coefficients at fixed scores.
-/

@[expose] public section

open Real
open scoped BigOperators

namespace Transformer.Homogenized

/-- Entrywise bounds control the actual bilinear coordinate sum.
Source: arXiv:2604.01978v1, §7, the finite contractions in
`eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem abs_sum_matrix_coordinate_products_le {m n : ℕ}
    (B : Matrix (Idx m) (Idx n) ℝ) (a : Idx m → ℝ) (b : Idx n → ℝ)
    (ε : ℝ) (hB : ∀ r s, |B r s| ≤ ε) :
    |∑ r : Idx m, ∑ s : Idx n, B r s * a r * b s| ≤
      ε * (∑ r : Idx m, |a r|) * (∑ s : Idx n, |b s|) := by
  calc
    _ ≤ ∑ r : Idx m, ∑ s : Idx n, |B r s * a r * b s| :=
      (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun r hr => Finset.abs_sum_le_sum_abs _ _)
    _ ≤ ∑ r : Idx m, ∑ s : Idx n, ε * |a r| * |b s| :=
      Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun s hs => by
        rw [abs_mul, abs_mul]
        exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right (hB r s) (abs_nonneg _)) (abs_nonneg _)
    _ = _ := by
      simp only [Finset.mul_sum, Finset.sum_mul]
      exact Finset.sum_comm

/-- The actual Gram-weighted pair of softmax rows is bounded by the
entrywise Gram bound. Source: arXiv:2604.01978v1, §7, the direct
Gram coefficient term following `eq:Lispchitz_bound_softmax`. -/
theorem abs_sum_matrix_softmax_pair_le {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (u : Idx m → ℝ) (v : Idx n → ℝ) (B : Matrix (Idx m) (Idx n) ℝ)
    (ε : ℝ) (hB : ∀ r s, |B r s| ≤ ε) :
    |∑ r : Idx m, ∑ s : Idx n, B r s * Perspective.softmaxWeight u r * Perspective.softmaxWeight v s| ≤ ε := by
  have ha (r : Idx m) : |Perspective.softmaxWeight u r| = Perspective.softmaxWeight u r :=
    abs_of_nonneg (Perspective.softmaxWeight_nonneg u r)
  have hb (s : Idx n) : |Perspective.softmaxWeight v s| = Perspective.softmaxWeight v s :=
    abs_of_nonneg (Perspective.softmaxWeight_nonneg v s)
  simpa only [ha, hb, Perspective.sum_softmaxWeight hm u, Perspective.sum_softmaxWeight hn v, mul_one] using
    abs_sum_matrix_coordinate_products_le B (Perspective.softmaxWeight u) (Perspective.softmaxWeight v) ε hB

/-- The actual mixed gradient contraction has universal constant
four and keeps both original softmax factors.
Source: arXiv:2604.01978v1, §7, the mixed derivative terms in
`eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem abs_sum_matrix_softmax_gradients_le {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (u : Idx m → ℝ) (v : Idx n → ℝ) (j : Idx m) (k : Idx n)
    (B : Matrix (Idx m) (Idx n) ℝ) (ε : ℝ) (hB : ∀ r s, |B r s| ≤ ε) :
    |∑ r : Idx m, ∑ s : Idx n, B r s *
      fderiv ℝ (fun w : Idx m → ℝ => Perspective.softmaxWeight w j) u (Pi.single r 1) *
      fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) v (Pi.single s 1)| ≤
      4 * ε * Perspective.softmaxWeight u j * Perspective.softmaxWeight v k := by
  have hε : 0 ≤ ε := (abs_nonneg (B j k)).trans (hB j k)
  apply (abs_sum_matrix_coordinate_products_le B _ _ ε hB).trans
  have h := mul_le_mul (sum_abs_fderiv_softmaxWeight_le hm u j)
    (sum_abs_fderiv_softmaxWeight_le hn v k) (Finset.sum_nonneg fun s hs => abs_nonneg _)
    (mul_nonneg (by norm_num) (Perspective.softmaxWeight_nonneg u j))
  convert mul_le_mul_of_nonneg_left h hε using 1 <;> ring

/-- Normalization aggregates an actual matrix whose entries are
controlled by the original pair of coordinate weights.
Source: arXiv:2604.01978v1, §7, the normalization step following
`eq:Lispchitz_bound_softmax`, performed before expectation. -/
theorem abs_sum_normalized_weighted_matrix_le {m n : ℕ}
    (p : Idx m → ℝ) (q : Idx n → ℝ) (hp : ∑ k, p k = 1) (hq : ∑ l, q l = 1)
    (A T : Matrix (Idx m) (Idx n) ℝ) (hA : ∀ k l, |A k l| ≤ 1)
    (C : ℝ) (hT : ∀ k l, |T k l| ≤ C * p k * q l) :
    |∑ k : Idx m, ∑ l : Idx n, A k l * T k l| ≤ C := by
  calc
    _ ≤ ∑ k : Idx m, ∑ l : Idx n, |A k l * T k l| :=
      (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k hk => Finset.abs_sum_le_sum_abs _ _)
    _ ≤ ∑ k : Idx m, ∑ l : Idx n, C * p k * q l :=
      Finset.sum_le_sum fun k hk => Finset.sum_le_sum fun l hl => by
        rw [abs_mul]
        exact ((mul_le_mul_of_nonneg_right (hA k l) (abs_nonneg _)).trans_eq (one_mul _)).trans (hT k l)
    _ = C := by
      simp_rw [← Finset.mul_sum]
      rw [hq]
      simp only [mul_one]
      rw [← Finset.mul_sum, hp, mul_one]

/-- Summing the actual Hessian contraction over both output weights
preserves the universal constant six.
Source: arXiv:2604.01978v1, §7, the aggregate Hessian contribution
underlying `eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem abs_sum_gram_softmax_hessians_le {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (u : Idx m → ℝ) (v : Idx n → ℝ) (A : Matrix (Idx m) (Idx n) ℝ) (hA : ∀ k l, |A k l| ≤ 1)
    (B : Matrix (Idx m) (Idx m) ℝ) (ε : ℝ) (hB : ∀ r s, |B r s| ≤ ε) :
    |∑ k : Idx m, ∑ l : Idx n, A k l * (Perspective.softmaxWeight v l *
      ∑ r : Idx m, ∑ s : Idx m, B r s *
        iteratedFDeriv ℝ 2 (fun w : Idx m → ℝ => Perspective.softmaxWeight w k) u
          ![Pi.single r 1, Pi.single s 1])| ≤ 6 * ε := by
  apply abs_sum_normalized_weighted_matrix_le (Perspective.softmaxWeight u) (Perspective.softmaxWeight v)
    (Perspective.sum_softmaxWeight hm u) (Perspective.sum_softmaxWeight hn v) A _ hA (6 * ε)
  intro k l
  rw [abs_mul, abs_of_nonneg (Perspective.softmaxWeight_nonneg v l)]
  convert mul_le_mul_of_nonneg_left (abs_sum_matrix_softmax_hessian_le hm u k B ε hB)
    (Perspective.softmaxWeight_nonneg v l) using 1; ring

/-- The actual mixed gradient contractions can be aggregated over
both original output rows with universal constant four.
Source: arXiv:2604.01978v1, §7, the aggregate mixed terms in
`eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem abs_sum_gram_softmax_gradients_le {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (u : Idx m → ℝ) (v : Idx n → ℝ) (A : Matrix (Idx m) (Idx n) ℝ) (hA : ∀ k l, |A k l| ≤ 1)
    (B : Matrix (Idx m) (Idx n) ℝ) (ε : ℝ) (hB : ∀ r s, |B r s| ≤ ε) :
    |∑ k : Idx m, ∑ l : Idx n, A k l * (∑ r : Idx m, ∑ s : Idx n, B r s *
      fderiv ℝ (fun w : Idx m → ℝ => Perspective.softmaxWeight w k) u (Pi.single r 1) *
      fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v (Pi.single s 1))| ≤ 4 * ε := by
  apply abs_sum_normalized_weighted_matrix_le (Perspective.softmaxWeight u) (Perspective.softmaxWeight v)
    (Perspective.sum_softmaxWeight hm u) (Perspective.sum_softmaxWeight hn v) A _ hA (4 * ε)
  exact fun k l => abs_sum_matrix_softmax_gradients_le hm hn u v k l B ε hB

/-- Changing the actual Gram coefficients at fixed original scores
costs only their entrywise change. Source: arXiv:2604.01978v1,
§7, the first term after `eq:Lispchitz_bound_softmax`. -/
theorem abs_sub_gram_softmax_pair_le {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (u : Idx m → ℝ) (v : Idx n → ℝ) (A B : Matrix (Idx m) (Idx n) ℝ)
    (ε : ℝ) (hAB : ∀ k l, |A k l - B k l| ≤ ε) :
    |(∑ k : Idx m, ∑ l : Idx n, A k l * Perspective.softmaxWeight u k * Perspective.softmaxWeight v l) -
      ∑ k : Idx m, ∑ l : Idx n, B k l * Perspective.softmaxWeight u k * Perspective.softmaxWeight v l| ≤ ε := by
  simpa only [Matrix.sub_apply, sub_mul, Finset.sum_sub_distrib] using
    abs_sum_matrix_softmax_pair_le hm hn u v (A - B) ε hAB

/-- Biased actual softmax scores, two positive normalized rows and
the identity matrix satisfy every coordinate and Gram bound premise. -/
example : ∃ (u v : Idx 2 → ℝ) (A B : Matrix (Idx 2) (Idx 2) ℝ),
    (0 : ℕ) < 2 ∧ (∑ k : Idx 2, Perspective.softmaxWeight u k) = 1 ∧
    (∑ l : Idx 2, Perspective.softmaxWeight v l) = 1 ∧
    (∀ k, 0 ≤ Perspective.softmaxWeight u k) ∧ (∀ l, 0 ≤ Perspective.softmaxWeight v l) ∧
    (∀ k l, |A k l| ≤ 1) ∧ (∀ k l, |B k l| ≤ 1) ∧
    (∀ k l, |A k l - B k l| ≤ 0) ∧
    (∀ k l, |Perspective.softmaxWeight u k * Perspective.softmaxWeight v l| ≤
      1 * Perspective.softmaxWeight u k * Perspective.softmaxWeight v l) := by
  let u : Idx 2 → ℝ := ![Real.log 3, 0]
  let v : Idx 2 → ℝ := ![Real.log 2, 0]
  refine ⟨u, v, 1, 1, by norm_num, Perspective.sum_softmaxWeight (by norm_num) u,
    Perspective.sum_softmaxWeight (by norm_num) v, Perspective.softmaxWeight_nonneg u,
    Perspective.softmaxWeight_nonneg v, ?_, ?_, fun k l => by simp, ?_⟩
  · intro k l
    by_cases h : k = l <;> simp [Matrix.one_apply, h]
  · intro k l
    by_cases h : k = l <;> simp [Matrix.one_apply, h]
  · intro k l
    rw [abs_of_nonneg (mul_nonneg (Perspective.softmaxWeight_nonneg u k) (Perspective.softmaxWeight_nonneg v l))]
    simp

end Transformer.Homogenized

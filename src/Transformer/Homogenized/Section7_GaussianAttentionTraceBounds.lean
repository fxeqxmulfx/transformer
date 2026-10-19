module

public import Transformer.Homogenized.Section7_GaussianAttentionQueryTrace

/-!
# Pointwise original query and key trace bounds after output normalization

ArXiv:2604.01978v1, §7, `eq:estimate_bound_h`,
`eq:estimate_bound_g` and `eq:Lispchitz_bound_softmax`.
The actual query trace identity is bounded using the original diagonal
and mixed perturbation entries and the proved original key Gram
contractions. It has the universal pointwise constant twenty.

The original attention-pair output factors are then normalized in the
finite Gram coefficient sum. This preserves twenty for the query trace
and ten for the key trace, independently of token count. Their combined
bound retains only the original feature squared norms and the actual
inverse-temperature square and perturbation size.
The source places random softmax factors outside expectation; the
proved normalization occurs pointwise before any expectation.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.Homogenized

/-- The actual diagonal and mixed query trace terms combine with
constants six, six and eight to give the universal constant twenty.
Source: arXiv:2604.01978v1, §7, the arithmetic of `eq:estimate_bound_h`. -/
theorem abs_query_trace_three_terms_le (U V T b0 b1 c0 c1 H J D ε : ℝ)
    (hU : 0 ≤ U) (hV : 0 ≤ V) (hε : 0 ≤ ε)
    (hb0 : |b0| ≤ ε) (hb1 : |b1| ≤ ε) (hc0 : |c0| ≤ ε) (hc1 : |c1| ≤ ε)
    (hH : |H| ≤ 6 * T * U) (hJ : |J| ≤ 6 * T * V) (hD : |D| ≤ 4 * T * U * V) :
    |V * b0 * H + U * b1 * J + (c0 + c1) * D| ≤ 20 * ε * T * U * V := by
  have h1 : |V * b0 * H| ≤ V * ε * (6 * T * U) := by
    rw [abs_mul, abs_mul, abs_of_nonneg hV]
    exact mul_le_mul (mul_le_mul_of_nonneg_left hb0 hV) hH (abs_nonneg _) (mul_nonneg hV hε)
  have h2 : |U * b1 * J| ≤ U * ε * (6 * T * V) := by
    rw [abs_mul, abs_mul, abs_of_nonneg hU]
    exact mul_le_mul (mul_le_mul_of_nonneg_left hb1 hU) hJ (abs_nonneg _) (mul_nonneg hU hε)
  have hc : |c0 + c1| ≤ 2 * ε :=
    (abs_add_le _ _).trans ((add_le_add hc0 hc1).trans_eq (by ring))
  have h3 : |(c0 + c1) * D| ≤ (2 * ε) * (4 * T * U * V) := by
    rw [abs_mul]
    exact mul_le_mul hc hD (abs_nonneg _) (by positivity)
  have ht : |V * b0 * H + U * b1 * J + (c0 + c1) * D| ≤
      |V * b0 * H| + |U * b1 * J| + |(c0 + c1) * D| :=
    (abs_add_le _ _).trans (add_le_add (abs_add_le _ _) le_rfl)
  exact (ht.trans (add_le_add (add_le_add h1 h2) h3)).trans_eq (by ring)

/-- The actual full query-block trace is bounded pointwise by twenty,
retaining both original softmax factors and original key squared norms.
Source: arXiv:2604.01978v1, §7, `eq:estimate_bound_h`, with its
random factors kept pointwise inside the bound before expectation. -/
theorem abs_gaussianAttention_product_query_trace_le {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i j k l : Idx n)
    (B : Matrix (Idx n) (Idx n) ℝ) (ε : ℝ) (hB : ∀ a b, |B a b| ≤ ε) :
    |∑ q : Fin d, ∑ a, ∑ b, B a b * iteratedFDeriv ℝ 2 (fun y =>
      Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
        Perspective.softmaxWeight (gaussianFeatureScores β y j) l) z
      ![EuclideanSpace.single (0, a, q) 1, EuclideanSpace.single (0, b, q) 1]| ≤
      20 * β ^ 2 * ε * Perspective.softmaxWeight (gaussianFeatureScores β z i) k *
        Perspective.softmaxWeight (gaussianFeatureScores β z j) l *
          (∑ m : Idx n, ∑ q : Fin d, z (1, m, q) ^ 2) := by
  have hε : 0 ≤ ε := (abs_nonneg (B k k)).trans (hB k k)
  have h := abs_query_trace_three_terms_le _ _ _ (B i i) (B j j) (B i j) (B j i) _ _ _ ε
    (Perspective.softmaxWeight_nonneg _ k) (Perspective.softmaxWeight_nonneg _ l)
    hε
    (hB i i) (hB j j) (hB i j) (hB j i)
    (abs_sum_query_softmax_hessians_le hn (gaussianFeatureScores β z i) k (fun m q => z (1, m, q)))
    (abs_sum_query_softmax_hessians_le hn (gaussianFeatureScores β z j) l (fun m q => z (1, m, q)))
    (abs_sum_query_softmax_gradient_products_le hn (gaussianFeatureScores β z i)
      (gaussianFeatureScores β z j) k l (fun m q => z (1, m, q)))
  rw [gaussianAttention_product_query_trace hn]
  dsimp only
  rw [abs_mul, abs_of_nonneg (sq_nonneg β)]
  exact (mul_le_mul_of_nonneg_left h (sq_nonneg β)).trans_eq (by ring)

/-- Normalizing the original output weights preserves twenty in the
actual Gram-weighted query trace, independently of token count.
Source: arXiv:2604.01978v1, §7, normalization after `eq:estimate_bound_h`. -/
theorem abs_sum_weighted_gaussianAttention_query_trace_le {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i j : Idx n)
    (A B : Matrix (Idx n) (Idx n) ℝ) (hA : ∀ k l, |A k l| ≤ 1)
    (ε : ℝ) (hB : ∀ a b, |B a b| ≤ ε) :
    |∑ k, ∑ l, A k l * (∑ q : Fin d, ∑ a, ∑ b, B a b * iteratedFDeriv ℝ 2 (fun y =>
      Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
        Perspective.softmaxWeight (gaussianFeatureScores β y j) l) z
      ![EuclideanSpace.single (0, a, q) 1, EuclideanSpace.single (0, b, q) 1])| ≤
      20 * β ^ 2 * ε * (∑ m : Idx n, ∑ q : Fin d, z (1, m, q) ^ 2) := by
  apply abs_sum_normalized_weighted_matrix_le (Perspective.softmaxWeight (gaussianFeatureScores β z i))
    (Perspective.softmaxWeight (gaussianFeatureScores β z j))
    (Perspective.sum_softmaxWeight hn _) (Perspective.sum_softmaxWeight hn _) A _ hA
  intro k l
  exact (abs_gaussianAttention_product_query_trace_le hn β z i j k l B ε hB).trans_eq (by ring)

/-- Normalizing the original output weights preserves ten in the
actual Gram-weighted key trace, independently of token count.
Source: arXiv:2604.01978v1, §7, normalization after `eq:estimate_bound_g`. -/
theorem abs_sum_weighted_gaussianAttention_key_trace_le {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i j : Idx n)
    (A B : Matrix (Idx n) (Idx n) ℝ) (hA : ∀ k l, |A k l| ≤ 1)
    (ε : ℝ) (hB : ∀ a b, |B a b| ≤ ε) :
    |∑ k, ∑ l, A k l * (∑ q : Fin d, ∑ a, ∑ b, B a b * iteratedFDeriv ℝ 2 (fun y =>
      Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
        Perspective.softmaxWeight (gaussianFeatureScores β y j) l) z
      ![EuclideanSpace.single (1, a, q) 1, EuclideanSpace.single (1, b, q) 1])| ≤
      10 * β ^ 2 * ε * ((∑ q : Fin d, z (0, i, q) ^ 2) + ∑ q : Fin d, z (0, j, q) ^ 2) := by
  apply abs_sum_normalized_weighted_matrix_le (Perspective.softmaxWeight (gaussianFeatureScores β z i))
    (Perspective.softmaxWeight (gaussianFeatureScores β z j))
    (Perspective.sum_softmaxWeight hn _) (Perspective.sum_softmaxWeight hn _) A _ hA
  intro k l
  exact (abs_gaussianAttention_product_key_trace_le hn β z i j k l B ε hB).trans_eq (by ring)

/-- The actual combined Gaussian block trace keeps only the original
feature squared norms after the original output weights are normalized.
Source: arXiv:2604.01978v1, §7, the two-block estimate used for
`eq:Lispchitz_bound_softmax`, with normalization before expectation. -/
theorem abs_sum_weighted_gaussianAttention_block_traces_le {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i j : Idx n)
    (A B : Matrix (Idx n) (Idx n) ℝ) (hA : ∀ k l, |A k l| ≤ 1)
    (ε : ℝ) (hB : ∀ a b, |B a b| ≤ ε) :
    |∑ k, ∑ l, A k l * (∑ c : Fin 2, ∑ q : Fin d, ∑ a, ∑ b,
      B a b * iteratedFDeriv ℝ 2 (fun y =>
        Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
          Perspective.softmaxWeight (gaussianFeatureScores β y j) l) z
        ![EuclideanSpace.single (c, a, q) 1, EuclideanSpace.single (c, b, q) 1])| ≤
      β ^ 2 * ε * (20 * (∑ m : Idx n, ∑ q : Fin d, z (1, m, q) ^ 2) +
        10 * ((∑ q : Fin d, z (0, i, q) ^ 2) + ∑ q : Fin d, z (0, j, q) ^ 2)) := by
  simp only [Fin.sum_univ_two, mul_add, Finset.sum_add_distrib]
  exact (abs_add_le _ _).trans ((add_le_add
    (abs_sum_weighted_gaussianAttention_query_trace_le hn β z i j A B hA ε hB)
    (abs_sum_weighted_gaussianAttention_key_trace_le hn β z i j A B hA ε hB)).trans_eq (by ring))

/-- Nonzero coefficients satisfy every original arithmetic-bound premise. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ b ∈ ({(1 : ℝ)} : Finset ℝ), |b| ≤ 1) ∧ |(1 : ℝ)| ≤ 6 * 1 * 1 ∧
    |(1 : ℝ)| ≤ 6 * 1 * 1 ∧ |(1 : ℝ)| ≤ 4 * 1 * 1 * 1 := by
  norm_num

/-- A biased original feature vector and identity Gram/perturbation
matrices satisfy every positive-count and entrywise-bound premise. -/
example : ∃ (z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2))
    (A B : Matrix (Idx 2) (Idx 2) ℝ), (0 : ℕ) < 2 ∧
    (∀ k l, |A k l| ≤ 1) ∧ (∀ a b, |B a b| ≤ 1) ∧
      gaussianFeatureScores 1 z (0 : Idx 2) = ![Real.log 3, 0] := by
  let z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2) :=
    EuclideanSpace.single (0, 0, 0) 1 + EuclideanSpace.single (1, 0, 0) (Real.log 3)
  have hI (a b : Idx 2) : |(1 : Matrix (Idx 2) (Idx 2) ℝ) a b| ≤ 1 := by
    simp only [Matrix.one_apply]
    split_ifs <;> norm_num
  refine ⟨z, 1, 1, by norm_num, hI, hI, ?_⟩
  ext j
  fin_cases j <;> simp [gaussianFeatureScores, z, PiLp.single_apply]

end Transformer.Homogenized

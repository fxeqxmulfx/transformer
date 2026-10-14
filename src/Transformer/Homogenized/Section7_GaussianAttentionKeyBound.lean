/-
# The pointwise bound on the actual original Gaussian key Hessian trace

ArXiv:2604.01978v1, §7, `eq:estimate_bound_g`.
The source places random softmax factors outside an expected trace.
The bound proved here holds pointwise for the actual original attention
product, and keeps both original factors inside the bound. They can
then be normalized in the finite output sum before taking expectation.

The exact actual key trace identity has already been proved.
The two score Hessian contractions each have constant six, and the two
mixed score-gradient contractions together have constant eight.
Cauchy-Schwarz bounds the mixed query product by half the sum of their
original squared norms. Combining these bounds gives the universal
constant ten, independent of the number of tokens and coordinates.
Self-pairs, zero coordinates and arbitrary real inverse temperatures
are included in the actual attention bound.

The perturbation matrix need not be symmetric: its transpose controls
the reversed mixed derivative. Entrywise boundedness forces its bound
to be nonnegative once there is a token. The coordinate sums on the
right are the original squared query norms. Their Gaussian expectations
are evaluated after the original softmax factors are normalized.
-/

import Transformer.Homogenized.Section7_SoftmaxContractions
import Transformer.Homogenized.Section7_GaussianAttentionKeyTrace

open scoped BigOperators

namespace Transformer.Homogenized

/-- The two actual mixed score-gradient contractions retain both
original softmax factors, with universal constant eight.
Source: arXiv:2604.01978v1, §7, the mixed terms in `eq:estimate_bound_g`. -/
theorem abs_sum_matrix_softmax_gradient_cross_le {n : ℕ} (hn : 0 < n) (u v : Idx n → ℝ) (k l : Idx n)
    (B : Matrix (Idx n) (Idx n) ℝ) (ε : ℝ) (hB : ∀ a b, |B a b| ≤ ε) :
    |∑ a, ∑ b, B a b *
      (fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u (Pi.single a 1) *
        fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v (Pi.single b 1) +
      fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u (Pi.single b 1) *
        fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v (Pi.single a 1))| ≤
      8 * ε * Perspective.softmaxWeight u k * Perspective.softmaxWeight v l := by
  change Idx n → Idx n → ℝ at B
  have h1 := abs_sum_matrix_softmax_gradients_le hn hn u v k l B ε hB
  have h2 := abs_sum_matrix_softmax_gradients_le hn hn u v k l (Matrix.transpose B) ε (fun a b => hB b a)
  have heq : (∑ a, ∑ b, B a b *
      fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u (Pi.single b 1) *
        fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v (Pi.single a 1)) =
      ∑ a, ∑ b, Matrix.transpose B a b *
        fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u (Pi.single a 1) *
          fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v (Pi.single b 1) := by
    rw [Finset.sum_comm]
    rfl
  simp only [mul_add, Finset.sum_add_distrib, ← mul_assoc]
  rw [heq]
  exact (abs_add_le _ _).trans ((add_le_add h1 h2).trans_eq (by ring))

/-- The actual mixed coordinate product is bounded by half the sum
of the two original squared norms. Source: arXiv:2604.01978v1,
§7, the Cauchy-Schwarz step in `eq:estimate_bound_g`. -/
theorem two_mul_abs_sum_coordinate_products_le {d : ℕ} (a b : Fin d → ℝ) :
    2 * |∑ q, a q * b q| ≤ (∑ q, a q ^ 2) + ∑ q, b q ^ 2 := by
  have h (q : Fin d) : 2 * |a q * b q| ≤ a q ^ 2 + b q ^ 2 := by
    rw [abs_mul]
    nlinarith [sq_nonneg (|a q| - |b q|), sq_abs (a q), sq_abs (b q)]
  calc
    _ ≤ 2 * ∑ q, |a q * b q| := mul_le_mul_of_nonneg_left (Finset.abs_sum_le_sum_abs _ _) (by norm_num)
    _ ≤ ∑ q, (a q ^ 2 + b q ^ 2) := by
      rw [Finset.mul_sum]
      exact Finset.sum_le_sum fun q hq => h q
    _ = _ := Finset.sum_add_distrib

/-- The actual Hessian and mixed-gradient bounds combine with the
original squared norms and their Cauchy-Schwarz bound to give ten.
Source: arXiv:2604.01978v1, §7, the arithmetic of `eq:estimate_bound_g`. -/
theorem abs_key_trace_three_terms_le (U V a b c H J G ε : ℝ)
    (hU : 0 ≤ U) (hV : 0 ≤ V) (ha : 0 ≤ a) (hb : 0 ≤ b) (hε : 0 ≤ ε)
    (hH : |H| ≤ 6 * ε * U) (hJ : |J| ≤ 6 * ε * V) (hG : |G| ≤ 8 * ε * U * V)
    (hc : 2 * |c| ≤ a + b) :
    |V * a * H + U * b * J + c * G| ≤ 10 * ε * U * V * (a + b) := by
  have ht : |V * a * H + U * b * J + c * G| ≤
      |V * a * H| + |U * b * J| + |c * G| :=
    (abs_add_le _ _).trans (add_le_add (abs_add_le _ _) le_rfl)
  have h1 : |V * a * H| ≤ V * a * (6 * ε * U) := by
    rw [abs_mul, abs_of_nonneg (mul_nonneg hV ha)]
    exact mul_le_mul_of_nonneg_left hH (mul_nonneg hV ha)
  have h2 : |U * b * J| ≤ U * b * (6 * ε * V) := by
    rw [abs_mul, abs_of_nonneg (mul_nonneg hU hb)]
    exact mul_le_mul_of_nonneg_left hJ (mul_nonneg hU hb)
  have h3 : |c * G| ≤ |c| * (8 * ε * U * V) := by
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left hG (abs_nonneg c)
  have hcross := mul_le_mul_of_nonneg_left hc
    (by positivity : 0 ≤ 4 * ε * U * V)
  have hall := ht.trans (add_le_add (add_le_add h1 h2) h3)
  nlinarith

/-- The actual full key-block trace is bounded pointwise with constant
ten, keeping both original softmax factors inside the bound.
Source: arXiv:2604.01978v1, §7, `eq:estimate_bound_g`, corrected
by retaining the random factors before expectation. -/
theorem abs_gaussianAttention_product_key_trace_le {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i j k l : Idx n)
    (B : Matrix (Idx n) (Idx n) ℝ) (ε : ℝ) (hB : ∀ a b, |B a b| ≤ ε) :
    |∑ q : Fin d, ∑ a, ∑ b, B a b * iteratedFDeriv ℝ 2 (fun y =>
      Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
        Perspective.softmaxWeight (gaussianFeatureScores β y j) l) z
      ![EuclideanSpace.single (1, a, q) 1, EuclideanSpace.single (1, b, q) 1]| ≤
      10 * β ^ 2 * ε * Perspective.softmaxWeight (gaussianFeatureScores β z i) k *
        Perspective.softmaxWeight (gaussianFeatureScores β z j) l *
          ((∑ q : Fin d, z (0, i, q) ^ 2) + ∑ q : Fin d, z (0, j, q) ^ 2) := by
  have hε : 0 ≤ ε := (abs_nonneg (B k k)).trans (hB k k)
  have h := abs_key_trace_three_terms_le _ _ _ _ _ _ _ _ ε
    (Perspective.softmaxWeight_nonneg _ k) (Perspective.softmaxWeight_nonneg _ l)
    (Finset.sum_nonneg fun q hq => sq_nonneg (z (0, i, q)))
    (Finset.sum_nonneg fun q hq => sq_nonneg (z (0, j, q))) hε
    (abs_sum_matrix_softmax_hessian_le hn (gaussianFeatureScores β z i) k B ε hB)
    (abs_sum_matrix_softmax_hessian_le hn (gaussianFeatureScores β z j) l B ε hB)
    (abs_sum_matrix_softmax_gradient_cross_le hn (gaussianFeatureScores β z i) (gaussianFeatureScores β z j) k l B ε hB)
    (two_mul_abs_sum_coordinate_products_le (fun q => z (0, i, q)) (fun q => z (0, j, q)))
  rw [gaussianAttention_product_key_trace hn]
  dsimp only
  rw [abs_mul, abs_of_nonneg (sq_nonneg β)]
  exact (mul_le_mul_of_nonneg_left h (sq_nonneg β)).trans_eq (by ring)

/-- Positive nonzero coefficients satisfy every arithmetic-bound premise. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    |(1 : ℝ)| ≤ 6 * 1 * 1 ∧ |(1 : ℝ)| ≤ 6 * 1 * 1 ∧ |(1 : ℝ)| ≤ 8 * 1 * 1 * 1 ∧
    2 * |(1 : ℝ)| ≤ 1 + 1 := by
  norm_num

/-- A biased original nonzero feature vector, identity perturbation
and positive score count satisfy the actual derivative-bound premises. -/
example : ∃ (z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2))
    (B : Matrix (Idx 2) (Idx 2) ℝ), (0 : ℕ) < 2 ∧
    (∀ a b, |B a b| ≤ 1) ∧ gaussianFeatureScores 1 z (0 : Idx 2) = ![Real.log 3, 0] ∧
    z (0, 0, 0) = 1 ∧ z (1, 0, 0) = Real.log 3 := by
  let z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2) :=
    EuclideanSpace.single (0, 0, 0) 1 + EuclideanSpace.single (1, 0, 0) (Real.log 3)
  refine ⟨z, 1, by norm_num, ?_, ?_, ?_, ?_⟩
  · intro a b
    simp only [Matrix.one_apply]
    split_ifs <;> norm_num
  · ext j
    fin_cases j <;> simp [gaussianFeatureScores, z, PiLp.single_apply]
  · simp [z]
  · simp [z]

end Transformer.Homogenized


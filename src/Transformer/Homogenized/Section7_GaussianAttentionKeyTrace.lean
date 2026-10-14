/-
# The actual key-block Hessian trace of an original Gaussian attention pair

ArXiv:2604.01978v1, §7, the key-block trace calculation preceding
`eq:estimate_bound_g`.
The original attention pair is differentiated before taking expectations.
Its actual four-term Hessian is contracted with the original matrix,
then summed over matching feature coordinates.

The resulting identity retains the two original query squared norms,
their original mixed product, and the actual score Hessian and gradient
contractions. Every identity holds pointwise for arbitrary coefficients,
original features and pairs of tokens, including equal tokens.
The biased self-pair example gives a nonzero actual full trace.
-/

import Transformer.Homogenized.Section7_GaussianAttentionProduct

open scoped BigOperators

namespace Transformer.Homogenized

/-- The actual key-coordinate Hessian of the original attention pair,
including self-pairs, has all four genuine terms.
Source: arXiv:2604.01978v1, §7, the computation before `eq:estimate_bound_g`. -/
theorem iteratedFDeriv_two_gaussianAttention_product_key_axes {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i j k l a b : Idx n) (q r : Fin d) :
    let u := gaussianFeatureScores β z i
    let v := gaussianFeatureScores β z j
    iteratedFDeriv ℝ 2 (fun y =>
      Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
        Perspective.softmaxWeight (gaussianFeatureScores β y j) l) z
      ![EuclideanSpace.single (1, a, q) 1, EuclideanSpace.single (1, b, r) 1] =
      β ^ 2 * (Perspective.softmaxWeight v l * z (0, i, q) * z (0, i, r) *
          iteratedFDeriv ℝ 2 (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u
            ![Pi.single a 1, Pi.single b 1] +
        Perspective.softmaxWeight u k * z (0, j, q) * z (0, j, r) *
          iteratedFDeriv ℝ 2 (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v
            ![Pi.single a 1, Pi.single b 1] +
        z (0, i, q) * z (0, j, r) *
          fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u (Pi.single a 1) *
          fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v (Pi.single b 1) +
        z (0, i, r) * z (0, j, q) *
          fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u (Pi.single b 1) *
          fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v (Pi.single a 1)) := by
  rw [iteratedFDeriv_two_gaussianAttention_product hn]
  dsimp only
  simp only [iteratedFDeriv_two_gaussianAttention_key_axes hn, fderiv_gaussianAttention_key_axis hn]
  ring

/-- Three actual matrix terms distribute through the original finite contraction.
Source: arXiv:2604.01978v1, §7, the three groups in `eq:estimate_bound_g`. -/
theorem sum_three_weighted_matrix_terms {n : ℕ} (B M N T : Idx n → Idx n → ℝ) (c d e : ℝ) :
    (∑ a, ∑ b, B a b * (c * M a b + d * N a b + e * T a b)) =
      c * (∑ a, ∑ b, B a b * M a b) + d * (∑ a, ∑ b, B a b * N a b) +
        e * (∑ a, ∑ b, B a b * T a b) := by
  calc
    _ = ∑ a, ∑ b, (c * (B a b * M a b) + d * (B a b * N a b) + e * (B a b * T a b)) :=
      Finset.sum_congr rfl fun a ha => Finset.sum_congr rfl fun b hb => by ring
    _ = _ := by simp only [Finset.sum_add_distrib, ← Finset.mul_sum]

/-- The actual common scalar factors from the finite matrix contraction.
Source: arXiv:2604.01978v1, §7, the `β²` factor in `eq:estimate_bound_g`. -/
theorem sum_weighted_matrix_scalar {n : ℕ} (B M : Idx n → Idx n → ℝ) (c : ℝ) :
    (∑ a, ∑ b, B a b * (c * M a b)) = c * (∑ a, ∑ b, B a b * M a b) := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro a ha
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun b hb => by ring

/-- The actual diagonal coordinate sum retains its three original coefficients.
Source: arXiv:2604.01978v1, §7, the trace calculation before `eq:estimate_bound_g`. -/
theorem sum_three_weighted_trace_terms {d : ℕ} (a b e : Fin d → ℝ) (c h f j g : ℝ) :
    (∑ q, (c * a q * h + f * b q * j + e q * g)) =
      c * (∑ q, a q) * h + f * (∑ q, b q) * j + (∑ q, e q) * g := by
  simp only [Finset.sum_add_distrib, ← Finset.sum_mul, ← Finset.mul_sum]

/-- Contracting the actual original key Hessian at one feature coordinate
retains both original squared queries and their mixed coordinate product.
Source: arXiv:2604.01978v1, §7, the terms before `eq:estimate_bound_g`. -/
theorem gaussianAttention_product_key_contraction_coordinate {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i j k l : Idx n)
    (B : Matrix (Idx n) (Idx n) ℝ) (q : Fin d) :
    let u := gaussianFeatureScores β z i
    let v := gaussianFeatureScores β z j
    (∑ a, ∑ b, B a b * iteratedFDeriv ℝ 2 (fun y =>
      Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
        Perspective.softmaxWeight (gaussianFeatureScores β y j) l) z
      ![EuclideanSpace.single (1, a, q) 1, EuclideanSpace.single (1, b, q) 1]) =
      β ^ 2 * (Perspective.softmaxWeight v l * z (0, i, q) ^ 2 *
          (∑ a, ∑ b, B a b *
            iteratedFDeriv ℝ 2 (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u
              ![Pi.single a 1, Pi.single b 1]) +
        Perspective.softmaxWeight u k * z (0, j, q) ^ 2 *
          (∑ a, ∑ b, B a b *
            iteratedFDeriv ℝ 2 (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v
              ![Pi.single a 1, Pi.single b 1]) +
        z (0, i, q) * z (0, j, q) * (∑ a, ∑ b, B a b *
          (fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u (Pi.single a 1) *
            fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v (Pi.single b 1) +
          fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u (Pi.single b 1) *
            fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v (Pi.single a 1)))) := by
  change Idx n → Idx n → ℝ at B
  dsimp only
  have hterm (a b : Idx n) := iteratedFDeriv_two_gaussianAttention_product_key_axes hn β z i j k l a b q q
  dsimp only at hterm
  simp_rw [hterm]
  rw [sum_weighted_matrix_scalar B _ (β ^ 2)]
  congr 1
  calc
    _ = ∑ a, ∑ b, B a b *
      ((Perspective.softmaxWeight (gaussianFeatureScores β z j) l * z (0, i, q) ^ 2) *
        iteratedFDeriv ℝ 2 (fun w : Idx n → ℝ => Perspective.softmaxWeight w k)
          (gaussianFeatureScores β z i) ![Pi.single a 1, Pi.single b 1] +
      (Perspective.softmaxWeight (gaussianFeatureScores β z i) k * z (0, j, q) ^ 2) *
        iteratedFDeriv ℝ 2 (fun w : Idx n → ℝ => Perspective.softmaxWeight w l)
          (gaussianFeatureScores β z j) ![Pi.single a 1, Pi.single b 1] +
      (z (0, i, q) * z (0, j, q)) *
        (fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k)
          (gaussianFeatureScores β z i) (Pi.single a 1) *
          fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l)
            (gaussianFeatureScores β z j) (Pi.single b 1) +
        fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k)
          (gaussianFeatureScores β z i) (Pi.single b 1) *
          fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l)
            (gaussianFeatureScores β z j) (Pi.single a 1))) :=
      Finset.sum_congr rfl fun a ha => Finset.sum_congr rfl fun b hb => by ring
    _ = _ := sum_three_weighted_matrix_terms B _ _ _ _ _ _

/-- The actual full key-block trace contraction is exactly the original
three-term expression in query second moments and mixed query products.
Source: arXiv:2604.01978v1, §7, the trace identity before `eq:estimate_bound_g`. -/
theorem gaussianAttention_product_key_trace {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i j k l : Idx n)
    (B : Matrix (Idx n) (Idx n) ℝ) :
    let u := gaussianFeatureScores β z i
    let v := gaussianFeatureScores β z j
    (∑ q : Fin d, ∑ a, ∑ b, B a b * iteratedFDeriv ℝ 2 (fun y =>
      Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
        Perspective.softmaxWeight (gaussianFeatureScores β y j) l) z
      ![EuclideanSpace.single (1, a, q) 1, EuclideanSpace.single (1, b, q) 1]) =
      β ^ 2 * (Perspective.softmaxWeight v l * (∑ q : Fin d, z (0, i, q) ^ 2) *
          (∑ a, ∑ b, B a b *
            iteratedFDeriv ℝ 2 (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u
              ![Pi.single a 1, Pi.single b 1]) +
        Perspective.softmaxWeight u k * (∑ q : Fin d, z (0, j, q) ^ 2) *
          (∑ a, ∑ b, B a b *
            iteratedFDeriv ℝ 2 (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v
              ![Pi.single a 1, Pi.single b 1]) +
        (∑ q : Fin d, z (0, i, q) * z (0, j, q)) * (∑ a, ∑ b, B a b *
          (fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u (Pi.single a 1) *
            fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v (Pi.single b 1) +
          fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u (Pi.single b 1) *
            fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v (Pi.single a 1)))) := by
  change Idx n → Idx n → ℝ at B
  dsimp only
  simp_rw [gaussianAttention_product_key_contraction_coordinate hn β z i j k l B]
  rw [← Finset.mul_sum]
  congr 1
  exact sum_three_weighted_trace_terms _ _ _ _ _ _ _ _

/-- The actual biased self-pair has full key trace `-9/64`,
with nonzero original features and a positive score count. -/
example : ∃ z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2), (0 : ℕ) < 2 ∧
    (∑ q : Fin 2, ∑ a : Idx 2, ∑ b : Idx 2, (1 : Matrix (Idx 2) (Idx 2) ℝ) a b *
      iteratedFDeriv ℝ 2 (fun y =>
        Perspective.softmaxWeight (gaussianFeatureScores 1 y (0 : Idx 2)) 0 *
          Perspective.softmaxWeight (gaussianFeatureScores 1 y (0 : Idx 2)) 0) z
        ![EuclideanSpace.single (1, a, q) 1, EuclideanSpace.single (1, b, q) 1]) = -(9 / 64 : ℝ) := by
  let z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2) :=
    EuclideanSpace.single (0, 0, 0) 1 + EuclideanSpace.single (1, 0, 0) (Real.log 3)
  have hs : gaussianFeatureScores 1 z (0 : Idx 2) = ![Real.log 3, 0] := by
    ext j
    fin_cases j <;> simp [gaussianFeatureScores, z, PiLp.single_apply]
  have hw : Perspective.softmaxWeight ![Real.log 3, 0] (0 : Idx 2) = (3 / 4 : ℝ) := by
    norm_num [Perspective.softmaxWeight, Real.exp_log (by norm_num : (0 : ℝ) < 3), Fin.sum_univ_two]
  have hw' : Perspective.softmaxWeight ![Real.log 3, 0] (1 : Idx 2) = (1 / 4 : ℝ) := by
    norm_num [Perspective.softmaxWeight, Real.exp_log (by norm_num : (0 : ℝ) < 3), Fin.sum_univ_two]
  refine ⟨z, by norm_num, ?_⟩
  rw [gaussianAttention_product_key_trace (by norm_num)]
  dsimp only
  rw [hs]
  simp only [Fin.sum_univ_two, Matrix.one_apply]
  simp only [iteratedFDeriv_two_softmaxWeight (m := 2) (by norm_num),
    fderiv_softmaxWeight_factor (m := 2) (by norm_num)]
  norm_num [z, PiLp.single_apply, hw, hw']

end Transformer.Homogenized


module

public import Transformer.Homogenized.Section7_GaussianQueryContractions

/-!
# The actual query-block Hessian trace of the original Gaussian attention pair

ArXiv:2604.01978v1, §7, the query trace calculation preceding
`eq:estimate_bound_h`.
The genuine original attention derivatives have already been proved.
Their actual token support selects the two diagonal and two mixed
entries of the original perturbation matrix in each feature coordinate.

The original feature-coordinate sum then gives the exact query trace
identity, preserving the actual original score Hessians, first
derivatives, key coordinate vectors and inverse-temperature square.
The formula allows coincident original tokens and arbitrary matrices,
so it also gives the self-pairs used by diagonal overlaps in Gram drift.
The biased self-pair example has a nonzero actual query trace.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.Homogenized

/-- The original finite token support selects the actual matrix entry.
Source: arXiv:2604.01978v1, §7, the supported query-block trace
before `eq:estimate_bound_h`. -/
theorem sum_matrix_two_token_indicator {n : ℕ} (B : Idx n → Idx n → ℝ) (i j : Idx n) (c : ℝ) :
    (∑ a, ∑ b, B a b * (if a = i ∧ b = j then c else 0)) = B i j * c := by
  calc
    _ = ∑ b, B i b * (if i = i ∧ b = j then c else 0) :=
      Finset.sum_eq_single i (fun a ha hai => by simp [hai]) (by simp)
    _ = _ := by simp only [true_and, mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ite_true]

/-- The actual query derivative acts on its original key coordinate
vector, with its original token support and inverse temperature.
Source: arXiv:2604.01978v1, §7, the derivatives in `eq:estimate_bound_h`. -/
theorem fderiv_gaussianAttention_query_direction {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i k a : Idx n) (q : Fin d) :
    fderiv ℝ (fun y => Perspective.softmaxWeight (gaussianFeatureScores β y i) k) z
      (EuclideanSpace.single (0, a, q) 1) =
      if a = i then β * fderiv ℝ (fun u : Idx n → ℝ => Perspective.softmaxWeight u k)
        (gaussianFeatureScores β z i) (fun r => z (1, r, q)) else 0 := by
  rw [fderiv_gaussianAttention_query_axis hn]
  rw [← linear_pi_coordinate_expansion
    (fderiv ℝ (fun u : Idx n → ℝ => Perspective.softmaxWeight u k) (gaussianFeatureScores β z i))
    (fun r => z (1, r, q))]

/-- The actual query Hessian contraction at one feature coordinate
retains both diagonal terms and both mixed token entries.
Source: arXiv:2604.01978v1, §7, the contraction before `eq:estimate_bound_h`. -/
theorem gaussianAttention_product_query_contraction_coordinate {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i j k l : Idx n)
    (B : Matrix (Idx n) (Idx n) ℝ) (q : Fin d) :
    let u := gaussianFeatureScores β z i
    let v := gaussianFeatureScores β z j
    let G := fun r : Idx n => z (1, r, q)
    (∑ a, ∑ b, B a b * iteratedFDeriv ℝ 2 (fun y =>
      Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
        Perspective.softmaxWeight (gaussianFeatureScores β y j) l) z
      ![EuclideanSpace.single (0, a, q) 1, EuclideanSpace.single (0, b, q) 1]) =
      β ^ 2 * (Perspective.softmaxWeight v l * B i i *
          iteratedFDeriv ℝ 2 (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u ![G, G] +
        Perspective.softmaxWeight u k * B j j *
          iteratedFDeriv ℝ 2 (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v ![G, G] +
        (B i j + B j i) *
          (fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u G *
            fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v G)) := by
  change Idx n → Idx n → ℝ at B
  dsimp only
  let H := iteratedFDeriv ℝ 2 (fun w : Idx n → ℝ => Perspective.softmaxWeight w k)
    (gaussianFeatureScores β z i) ![(fun r => z (1, r, q)), (fun r => z (1, r, q))]
  let J := iteratedFDeriv ℝ 2 (fun w : Idx n → ℝ => Perspective.softmaxWeight w l)
    (gaussianFeatureScores β z j) ![(fun r => z (1, r, q)), (fun r => z (1, r, q))]
  let D := fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k)
    (gaussianFeatureScores β z i) (fun r => z (1, r, q)) *
    fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l)
      (gaussianFeatureScores β z j) (fun r => z (1, r, q))
  have hp (a b : Idx n) :
      iteratedFDeriv ℝ 2 (fun y =>
        Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
          Perspective.softmaxWeight (gaussianFeatureScores β y j) l) z
        ![EuclideanSpace.single (0, a, q) 1, EuclideanSpace.single (0, b, q) 1] =
        β ^ 2 * ((if a = i ∧ b = i then Perspective.softmaxWeight (gaussianFeatureScores β z j) l * H else 0) +
          (if a = j ∧ b = j then Perspective.softmaxWeight (gaussianFeatureScores β z i) k * J else 0) +
          (if a = i ∧ b = j then D else 0) + (if a = j ∧ b = i then D else 0)) := by
    rw [iteratedFDeriv_two_gaussianAttention_product hn]
    dsimp only
    simp only [iteratedFDeriv_two_gaussianAttention_query_axes hn, fderiv_gaussianAttention_query_direction hn]
    dsimp only [H, J, D]
    by_cases hai : a = i <;> by_cases hbi : b = i <;>
      by_cases haj : a = j <;> by_cases hbj : b = j <;>
      simp_all only [ite_true, ite_false, and_true, and_false] <;> ring
  simp_rw [hp]
  rw [sum_weighted_matrix_scalar B _ (β ^ 2)]
  simp only [mul_add, Finset.sum_add_distrib]
  rw [sum_matrix_two_token_indicator, sum_matrix_two_token_indicator,
    sum_matrix_two_token_indicator, sum_matrix_two_token_indicator]
  dsimp only [H, J, D]
  ring

/-- The three original query trace groups retain their actual coefficients
when summed over feature coordinates.
Source: arXiv:2604.01978v1, §7, the summation in `eq:estimate_bound_h`. -/
theorem sum_three_coordinate_contractions {d : ℕ} (f g h : Fin d → ℝ) (a b c : ℝ) :
    (∑ q, (a * f q + b * g q + c * h q)) = a * (∑ q, f q) + b * (∑ q, g q) + c * ∑ q, h q := by
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum]

/-- The actual full query-block trace has the source's three terms,
with genuine score Hessians and mixed gradients on original key vectors.
Source: arXiv:2604.01978v1, §7, the identity before `eq:estimate_bound_h`. -/
theorem gaussianAttention_product_query_trace {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i j k l : Idx n)
    (B : Matrix (Idx n) (Idx n) ℝ) :
    let u := gaussianFeatureScores β z i
    let v := gaussianFeatureScores β z j
    let G := fun (q : Fin d) (r : Idx n) => z (1, r, q)
    (∑ q : Fin d, ∑ a, ∑ b, B a b * iteratedFDeriv ℝ 2 (fun y =>
      Perspective.softmaxWeight (gaussianFeatureScores β y i) k *
        Perspective.softmaxWeight (gaussianFeatureScores β y j) l) z
      ![EuclideanSpace.single (0, a, q) 1, EuclideanSpace.single (0, b, q) 1]) =
      β ^ 2 * (Perspective.softmaxWeight v l * B i i *
          (∑ q : Fin d, iteratedFDeriv ℝ 2 (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u ![G q, G q]) +
        Perspective.softmaxWeight u k * B j j *
          (∑ q : Fin d, iteratedFDeriv ℝ 2 (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v ![G q, G q]) +
        (B i j + B j i) * (∑ q : Fin d,
          fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u (G q) *
            fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v (G q))) := by
  change Idx n → Idx n → ℝ at B
  dsimp only
  simp_rw [gaussianAttention_product_query_contraction_coordinate hn β z i j k l B]
  rw [← Finset.mul_sum]
  congr 1
  exact sum_three_coordinate_contractions _ _ _ _ _ _

/-- Nonzero original features and two scores give the biased self-pair
query trace `(-9/128)(log 3)²`, with every positive-count premise met. -/
example : ∃ z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2), (0 : ℕ) < 2 ∧
    (∑ q : Fin 2, ∑ a : Idx 2, ∑ b : Idx 2, (1 : Matrix (Idx 2) (Idx 2) ℝ) a b *
      iteratedFDeriv ℝ 2 (fun y =>
        Perspective.softmaxWeight (gaussianFeatureScores 1 y (0 : Idx 2)) 0 *
          Perspective.softmaxWeight (gaussianFeatureScores 1 y (0 : Idx 2)) 0) z
        ![EuclideanSpace.single (0, a, q) 1, EuclideanSpace.single (0, b, q) 1]) =
          -(9 / 128 : ℝ) * (Real.log 3) ^ 2 := by
  let z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2) :=
    EuclideanSpace.single (0, 0, 0) 1 + EuclideanSpace.single (1, 0, 0) (Real.log 3)
  have hs : gaussianFeatureScores 1 z (0 : Idx 2) = ![Real.log 3, 0] := by
    ext j
    fin_cases j <;> simp [gaussianFeatureScores, z, PiLp.single_apply]
  have hw : Perspective.softmaxWeight ![Real.log 3, 0] (0 : Idx 2) = (3 / 4 : ℝ) := by
    norm_num [Perspective.softmaxWeight, Real.exp_log (by norm_num : (0 : ℝ) < 3), Fin.sum_univ_two]
  have hh : (∑ q : Fin 2,
      iteratedFDeriv ℝ 2 (fun w : Idx 2 → ℝ => Perspective.softmaxWeight w 0) ![Real.log 3, 0]
        ![(fun r => z (1, r, q)), (fun r => z (1, r, q))]) = -(3 / 32 : ℝ) * (Real.log 3) ^ 2 := by
    rw [sum_bilinear_key_gram]
    simp only [Fin.sum_univ_two, iteratedFDeriv_two_softmaxWeight (m := 2) (by norm_num)]
    simp [z, hw]
    ring
  have hd : (∑ q : Fin 2,
      fderiv ℝ (fun w : Idx 2 → ℝ => Perspective.softmaxWeight w 0) ![Real.log 3, 0] (fun r => z (1, r, q)) *
        fderiv ℝ (fun w : Idx 2 → ℝ => Perspective.softmaxWeight w 0) ![Real.log 3, 0] (fun r => z (1, r, q))) =
          (9 / 256 : ℝ) * (Real.log 3) ^ 2 := by
    rw [sum_linear_key_gram]
    simp only [Fin.sum_univ_two, fderiv_softmaxWeight_factor (m := 2) (by norm_num)]
    simp [z, hw]
    ring
  refine ⟨z, by norm_num, ?_⟩
  rw [gaussianAttention_product_query_trace (by norm_num)]
  dsimp only
  rw [hs]
  simp only [Matrix.one_apply, ite_true, one_pow, one_mul, hh, hd, hw]
  ring

end Transformer.Homogenized

module

public import Transformer.Homogenized.Section7_GaussianSoftmaxGrowth

/-!
# Actual derivatives of the original bilinear Gaussian feature scores

ArXiv:2604.01978v1, §7, the original scores before `eq:SIGMA_R`
and the Hessian computations before `eq:estimate_bound_h` and
`eq:estimate_bound_g`.
All derivatives below are genuine derivatives of the original bilinear
score function used by the exact original attention probabilities.

The first derivative differentiates both original feature factors.
The second derivative is independent of the base point, retains both
mixed directions, and vanishes on directions within one feature block.
The separate query and key coordinate formulas retain their actual
support in the token indices. These identities supply the chain-rule
terms for the actual softmax covariance contraction.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.Homogenized

/-- The actual original bilinear score vector has its stated genuine
Fréchet derivative, with both original feature blocks retained.
Source: arXiv:2604.01978v1, §7, the scores before `eq:SIGMA_R`
and their chain rules before `eq:estimate_bound_h`. -/
theorem hasFDerivAt_gaussianFeatureScores {d n : ℕ} (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i : Idx n) :
    HasFDerivAt (fun w => gaussianFeatureScores β w i)
      (ContinuousLinearMap.pi fun j => β • ∑ q : Fin d,
        (z (0, i, q) • EuclideanSpace.proj (𝕜 := ℝ) (1, j, q) +
          z (1, j, q) • EuclideanSpace.proj (𝕜 := ℝ) (0, i, q))) z := by
  apply hasFDerivAt_pi.mpr
  intro j
  have h := (HasFDerivAt.sum (x := z) (u := Finset.univ) fun q hq =>
    (EuclideanSpace.proj (𝕜 := ℝ) (ι := GaussianFeatureIdx d n) (0, i, q)).hasFDerivAt.fun_mul
      (EuclideanSpace.proj (𝕜 := ℝ) (ι := GaussianFeatureIdx d n) (1, j, q)).hasFDerivAt).const_mul β
  simpa [gaussianFeatureScores] using h

/-- The genuine original score derivative acts by differentiating
both factors of the original bilinear scores.
Source: arXiv:2604.01978v1, §7, the score differentiation used
before `eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem fderiv_gaussianFeatureScores_apply {d n : ℕ} (β : ℝ)
    (z v : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i j : Idx n) :
    fderiv ℝ (fun w => gaussianFeatureScores β w i) z v j =
      β * ∑ q : Fin d, (z (0, i, q) * v (1, j, q) + z (1, j, q) * v (0, i, q)) := by
  rw [(hasFDerivAt_gaussianFeatureScores β z i).fderiv]
  simp

/-- The original quadratic score Hessian is independent of its base
point and is the linear first derivative evaluated at a direction.
Source: arXiv:2604.01978v1, §7, the actual bilinear score derivative
calculation before `eq:estimate_bound_h`. -/
theorem iteratedFDeriv_two_gaussianFeatureScores {d n : ℕ} (β : ℝ)
    (z v w : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i : Idx n) :
    iteratedFDeriv ℝ 2 (fun y => gaussianFeatureScores β y i) z ![v, w] =
      fderiv ℝ (fun y => gaussianFeatureScores β y i) w v := by
  let f := fun y : EuclideanSpace ℝ (GaussianFeatureIdx d n) => gaussianFeatureScores β y i
  have hfun : (fun y => fderiv ℝ f y w) = fderiv ℝ f w := by
    funext y
    ext j
    rw [fderiv_gaussianFeatureScores_apply, fderiv_gaussianFeatureScores_apply]
    congr 1
    exact Finset.sum_congr rfl fun q hq => by ring
  have hp : HasFDerivAt (fun y => fderiv ℝ f y w) (fderiv ℝ f w) z := by
    rw [hfun]
    exact (fderiv ℝ f w).hasFDerivAt
  have hD : ContDiff ℝ 1 (fderiv ℝ f) :=
    (hasTemperateGrowth_gaussianFeatureScores β i).1.fderiv_right (by simp)
  have heval := (((hD.differentiable (by norm_num)) z).hasFDerivAt.clm_apply
    (hasFDerivAt_const w z)).fderiv
  have haxis : fderiv ℝ (fun y => fderiv ℝ f y w) z v = fderiv ℝ (fderiv ℝ f) z v w := by
    rw [heval]
    simp only [ContinuousLinearMap.comp_zero, zero_add, ContinuousLinearMap.flip_apply]
  rw [iteratedFDeriv_two_apply]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
  rw [← haxis, hp.fderiv]

/-- Only the original query token's coordinates affect its score row.
Source: arXiv:2604.01978v1, §7, the query-coordinate differentiation
used before `eq:estimate_bound_h`. The derivative keeps the genuine
original key coordinates and the exact inverse temperature. -/
theorem fderiv_gaussianFeatureScores_query_axis {d n : ℕ} (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i a : Idx n) (q : Fin d) :
    fderiv ℝ (fun y => gaussianFeatureScores β y i) z (EuclideanSpace.single (0, a, q) 1) =
      if a = i then (fun j => β * z (1, j, q)) else 0 := by
  ext j
  rw [fderiv_gaussianFeatureScores_apply]
  by_cases hai : a = i
  · subst a
    simp [PiLp.single_apply]
  · simp [hai, Ne.symm hai]

/-- An original key coordinate changes exactly its own score column.
Source: arXiv:2604.01978v1, §7, the key-coordinate differentiation
used before `eq:estimate_bound_g`. The derivative keeps the genuine
original query coordinate and the exact inverse temperature. -/
theorem fderiv_gaussianFeatureScores_key_axis {d n : ℕ} (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i a : Idx n) (q : Fin d) :
    fderiv ℝ (fun y => gaussianFeatureScores β y i) z (EuclideanSpace.single (1, a, q) 1) =
      Pi.single a (β * z (0, i, q)) := by
  ext j
  rw [fderiv_gaussianFeatureScores_apply]
  by_cases haj : a = j
  · subst a
    simp [PiLp.single_apply]
  · simp [Ne.symm haj]

/-- The actual original score Hessian retains both mixed directions.
Source: arXiv:2604.01978v1, §7, the bilinear score chain rule
before `eq:estimate_bound_h` and `eq:estimate_bound_g`.
The formula applies to arbitrary, independently chosen directions. -/
theorem iteratedFDeriv_two_gaussianFeatureScores_apply {d n : ℕ} (β : ℝ)
    (z v w : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i j : Idx n) :
    iteratedFDeriv ℝ 2 (fun y => gaussianFeatureScores β y i) z ![v, w] j =
      β * ∑ q : Fin d, (v (0, i, q) * w (1, j, q) + v (1, j, q) * w (0, i, q)) := by
  rw [iteratedFDeriv_two_gaussianFeatureScores, fderiv_gaussianFeatureScores_apply]
  congr 1
  exact Finset.sum_congr rfl fun q hq => by ring

/-- The actual score Hessian vanishes along two original coordinates
in the same feature block. Source: arXiv:2604.01978v1, §7, the
absence of second score-derivative terms in its same-block Hessian
formulas before `eq:estimate_bound_h` and `eq:estimate_bound_g`. -/
theorem iteratedFDeriv_two_gaussianFeatureScores_same_block {d n : ℕ} (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i a b : Idx n)
    (c : Fin 2) (q r : Fin d) :
    iteratedFDeriv ℝ 2 (fun y => gaussianFeatureScores β y i) z
      ![EuclideanSpace.single (c, a, q) 1, EuclideanSpace.single (c, b, r) 1] = 0 := by
  rw [iteratedFDeriv_two_gaussianFeatureScores]
  ext j
  rw [fderiv_gaussianFeatureScores_apply]
  fin_cases c <;> simp [PiLp.single_apply]

/-- A nonzero actual feature vector gives a nonzero first derivative
and a nonzero mixed Hessian, witnessing the original bilinear terms. -/
example : ∃ z : EuclideanSpace ℝ (GaussianFeatureIdx 2 2),
    fderiv ℝ (fun y => gaussianFeatureScores 1 y (0 : Idx 2)) z
      (EuclideanSpace.single (0, 0, 0) 1) (1 : Idx 2) = 3 ∧
    iteratedFDeriv ℝ 2 (fun y => gaussianFeatureScores 1 y (0 : Idx 2)) z
      ![EuclideanSpace.single (0, 0, 0) 1, EuclideanSpace.single (1, 1, 0) 1] (1 : Idx 2) = 1 := by
  refine ⟨EuclideanSpace.single (1, 1, 0) 3, ?_, ?_⟩
  · rw [fderiv_gaussianFeatureScores_query_axis]
    simp
  · rw [iteratedFDeriv_two_gaussianFeatureScores, fderiv_gaussianFeatureScores_apply]
    norm_num [PiLp.single_apply, Fin.sum_univ_two]
end Transformer.Homogenized

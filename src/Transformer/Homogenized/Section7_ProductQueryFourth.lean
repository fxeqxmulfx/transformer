module

public import Transformer.Homogenized.Section7_ConditionalQueryMoments
public import Transformer.Homogenized.Section7_GaussianQueryMoments

/-!
# Fourth moments of the actual original product-matrix query

ArXiv:2604.01978v1, §2.3.3, initialization (G), and §7.3,
`lem:delta_method`. The original matrix is `A = WW′ᵀ`, with both
factors independent and their entries standard Gaussian here.

For a coordinate unit query, the actual squared norm is the sum of
squared row linear forms, conditional on one row of the second
factor. Its fourth norm moment is `d² (d + 2)²`. This integrates
both original Gaussian factors rather than replacing their product
by a Gaussian matrix. Its mean squared norm is the previously proved
`d²`, so its exact squared-norm variance is `4 d² (d + 1)`.

The source normalizes the matrix by `1/d` at unit entry variance.
The normalized squared-norm variance consequently has leading term
`4/d`, whereas the displayed covariance in `lem:delta_method`
asserts `2/d + O(d⁻²)`. The normalized comparison is proved in the
next module; no normalized-correlation remainder is assumed here.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The actual independent key-row/matrix product has an integrable
fourth query-norm moment. Source: arXiv:2604.01978v1, §7.3,
the two original Gaussian factors of `lem:delta_method`. -/
theorem integrable_gaussianRow_matrix_query_fourth (d : ℕ) :
    Integrable (fun p : (Fin d → ℝ) × Matrix (Fin d) (Fin d) ℝ =>
      (∑ i : Fin d, dotProduct p.1 (p.2 i) ^ 2) ^ 2)
        ((gaussianRowLaw d).prod (gaussianMatrixLaw d)) := by
  apply (integrable_prod_iff (by
    dsimp only [dotProduct]
    fun_prop)).mpr
  refine ⟨ae_of_all _ (fun g => integrable_gaussianMatrix_query_norm_fourth g), ?_⟩
  have hi := (integrable_gaussianRow_norm_fourth d).const_mul ((d : ℝ) * (d + 2))
  simpa only [Real.norm_of_nonneg (sq_nonneg _), integral_gaussianMatrix_query_norm_fourth] using hi

/-- Fubini under the original independent factors gives the exact
fourth query-norm moment. Source: arXiv:2604.01978v1, §7.3,
the covariance calculation for `lem:delta_method`. -/
theorem integral_gaussianRow_matrix_query_fourth (d : ℕ) :
    (∫ p : (Fin d → ℝ) × Matrix (Fin d) (Fin d) ℝ,
      (∑ i : Fin d, dotProduct p.1 (p.2 i) ^ 2) ^ 2
        ∂(gaussianRowLaw d).prod (gaussianMatrixLaw d)) = (d : ℝ) ^ 2 * (d + 2) ^ 2 := by
  rw [integral_prod _ (integrable_gaussianRow_matrix_query_fourth d)]
  simp_rw [integral_gaussianMatrix_query_norm_fourth]
  rw [integral_const_mul, integral_gaussianRow_norm_fourth]
  ring

/-- The original head's squared unit-coordinate query norm has
exactly the original independent key-row/matrix law. Source:
arXiv:2604.01978v1, initialization (G), §7.3, `lem:delta_method`. -/
theorem gaussianHead_query_norm_sq_map {d : ℕ} (j : Fin d) :
    (gaussHeadLaw d 1 1).map (fun θ => ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2) =
      ((gaussianRowLaw d).prod (gaussianMatrixLaw d)).map
        (fun p : (Fin d → ℝ) × Matrix (Fin d) (Fin d) ℝ =>
          ∑ i : Fin d, dotProduct p.1 (p.2 i) ^ 2) := by
  have hm : Measurable (fun θ : HeadParam d =>
      ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2) := by
    simp_rw [EuclideanSpace.real_norm_sq_eq, qkMap_single]
    fun_prop
  rw [gaussHeadLaw, Measure.map_map hm (measurable_gaussHead d)]
  rw [← originalGaussianRow_matrix_pair_map (1 : Fin 3) 2 (by decide) j,
    Measure.map_map (by dsimp only [dotProduct]; fun_prop)
      ((measurable_originalGaussianRow 2 j).prodMk (measurable_gaussBlock 1))]
  congr 1
  funext z
  simp only [Function.comp_apply, EuclideanSpace.real_norm_sq_eq, qkMap_single,
    Matrix.mul_apply, Matrix.transpose_apply, gaussBlock_apply, originalGaussianRow, dotProduct]
  apply Finset.sum_congr rfl
  intro i hi
  congr 1
  exact Finset.sum_congr rfl fun k hk => mul_comm _ _

/-- The actual original head query has finite fourth norm moment.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integrable_gaussianHead_query_norm_fourth {d : ℕ} (j : Fin d) :
    Integrable (fun θ : HeadParam d => ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 4)
      (gaussHeadLaw d 1 1) := by
  have hi : Integrable (fun u : ℝ => u ^ 2)
      ((gaussHeadLaw d 1 1).map (fun θ => ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2)) := by
    rw [gaussianHead_query_norm_sq_map]
    exact (integrable_map_measure (continuous_pow 2).aestronglyMeasurable
      (by dsimp only [dotProduct]; fun_prop)).mpr (integrable_gaussianRow_matrix_query_fourth d)
  have h := (integrable_map_measure (continuous_pow 2).aestronglyMeasurable
    (by
      simp_rw [EuclideanSpace.real_norm_sq_eq, qkMap_single]
      fun_prop : Measurable (fun θ : HeadParam d =>
        ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2)).aemeasurable).mp hi
  simpa only [Function.comp_def, ← pow_mul] using h

/-- **Exact fourth moment for the original two-factor matrix.**
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`.
The fourth query-norm moment is `d² (d + 2)²`; both original
Gaussian factors contribute to the squared-norm fluctuations. -/
theorem integral_gaussianHead_query_norm_fourth {d : ℕ} (j : Fin d) :
    (∫ θ : HeadParam d, ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 4
      ∂gaussHeadLaw d 1 1) = (d : ℝ) ^ 2 * (d + 2) ^ 2 := by
  calc
    _ = ∫ u : ℝ, u ^ 2 ∂(gaussHeadLaw d 1 1).map
        (fun θ => ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2) := by
      rw [integral_map (by
        simp_rw [EuclideanSpace.real_norm_sq_eq, qkMap_single]
        fun_prop) (continuous_pow 2).aestronglyMeasurable]
      simp only [← pow_mul]
    _ = _ := by
      rw [gaussianHead_query_norm_sq_map, integral_map
        (by dsimp only [dotProduct]; fun_prop) (continuous_pow 2).aestronglyMeasurable]
      exact integral_gaussianRow_matrix_query_fourth d

/-- The actual squared query norm lies in `L²` under the original
head law. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem memLp_gaussianHead_query_norm_sq {d : ℕ} (j : Fin d) :
    MemLp (fun θ : HeadParam d => ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2)
      2 (gaussHeadLaw d 1 1) := by
  apply (memLp_two_iff_integrable_sq
    (integrable_gaussianHead_query_norm_sq j).aestronglyMeasurable).mpr
  simpa only [← pow_mul] using integrable_gaussianHead_query_norm_fourth j

/-- **Corrected exact squared-norm variance.** The original
unnormalized product query has variance `4 d² (d + 1)`.
The source's displayed covariance in `lem:delta_method` instead
amounts to `2 d³ + O(d²)` at this scaling. This identity retains
the original initialization and corrects that leading coefficient.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem variance_gaussianHead_query_norm_sq {d : ℕ} (j : Fin d) :
    variance (fun θ : HeadParam d => ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2)
      (gaussHeadLaw d 1 1) = 4 * (d : ℝ) ^ 2 * (d + 1) := by
  rw [variance_eq_sub (memLp_gaussianHead_query_norm_sq j)]
  change (∫ θ : HeadParam d, (‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2) ^ 2
    ∂gaussHeadLaw d 1 1) -
      (∫ θ : HeadParam d, ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2
        ∂gaussHeadLaw d 1 1) ^ 2 = _
  simp only [← pow_mul]
  rw [integral_gaussianHead_query_norm_fourth, integral_gaussianHead_query_norm_sq]
  ring

/-- In dimension two the actual original product query has fourth
norm moment 64 and squared-norm variance 48. No Gaussian law is
postulated for its product matrix. -/
example : (∫ θ : HeadParam 2, ‖qkMap θ (EuclideanSpace.single 0 (1 : ℝ))‖ ^ 4
      ∂gaussHeadLaw 2 1 1) = 64 ∧
    variance (fun θ : HeadParam 2 => ‖qkMap θ (EuclideanSpace.single 0 (1 : ℝ))‖ ^ 2)
      (gaussHeadLaw 2 1 1) = 48 := by
  constructor
  · rw [integral_gaussianHead_query_norm_fourth]; norm_num
  · rw [variance_gaussianHead_query_norm_sq]; norm_num

end Transformer.Homogenized

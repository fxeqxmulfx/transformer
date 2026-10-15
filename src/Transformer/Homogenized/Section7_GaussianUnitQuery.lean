/-
# The actual Gaussian matrix query law for every unit input

ArXiv:2604.01978v1, §2.3.3, initialization (G), and §7.3,
`lem:delta_method`. With one original Gaussian matrix factor fixed
as the random matrix, its image of any deterministic unit vector
has exactly the original standard Gaussian row law.

Each output coordinate is a genuine Gaussian linear form. Joint
independence of the actual matrix rows proves independence of all
output coordinates, and the input's unit norm gives their variance
exactly one. This derives the full query law, without postulating
rotational invariance or a covariance for the product of two factors.

The original query is measurable and almost surely nonzero in positive
dimension. Its exact second and fourth norm moments, and the uniform
third absolute moment of its dimension-normalized squared-norm
deviation, follow by the actual pushforward law. All unit inputs use
the same constants. These results concern one original Gaussian factor;
the source's product query requires composing the two independent laws.
-/

import Transformer.Homogenized.Section7_GaussianMatrixLaw
import Transformer.Homogenized.Section7_GaussianRowDeviationThird

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- An actual matrix-query coordinate is the original row linear
form. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem gaussianMatrix_query_coordinate {d : ℕ} (M : Matrix (Fin d) (Fin d) ℝ)
    (x : EucSpace d) (i : Fin d) :
    Matrix.toEuclideanLin M x i = dotProduct (WithLp.ofLp x) (M i) := by
  change (∑ j : Fin d, M i j * x j) = ∑ j : Fin d, x j * M i j
  exact Finset.sum_congr rfl fun j hj => mul_comm _ _

/-- The original coordinate query vector is measurable. Source:
arXiv:2604.01978v1, initialization (G) and §7.3, `lem:delta_method`. -/
theorem measurable_gaussianMatrix_query_coordinates {d : ℕ} (x : EucSpace d) :
    Measurable (fun M : Matrix (Fin d) (Fin d) ℝ => WithLp.ofLp (Matrix.toEuclideanLin M x)) := by
  apply Measurable.of_eval
  intro i
  change Measurable (fun M : Matrix (Fin d) (Fin d) ℝ => Matrix.toEuclideanLin M x i)
  simp only [gaussianMatrix_query_coordinate]
  fun_prop

/-- The original Euclidean query vector itself is measurable.
Source: arXiv:2604.01978v1, §7.3, the actual query in `lem:delta_method`. -/
theorem measurable_gaussianMatrix_query {d : ℕ} (x : EucSpace d) :
    Measurable (fun M : Matrix (Fin d) (Fin d) ℝ => Matrix.toEuclideanLin M x) := by
  have h := (WithLp.measurable_toLp (p := 2) (X := Fin d → ℝ)).comp
    (measurable_gaussianMatrix_query_coordinates x)
  simpa only [Function.comp_def, WithLp.toLp_ofLp] using h

/-- **Actual query law for every original unit input.** Independent
original Gaussian rows give a standard Gaussian output vector.
Source: arXiv:2604.01978v1, initialization (G), §7.3, `lem:delta_method`. -/
theorem gaussianMatrix_unit_query_hasLaw {d : ℕ} (x : EucSpace d) (hx : ‖x‖ = 1) :
    HasLaw (fun M : Matrix (Fin d) (Fin d) ℝ => WithLp.ofLp (Matrix.toEuclideanLin M x))
      (gaussianRowLaw d) (gaussianMatrixLaw d) := by
  let g := WithLp.ofLp x
  have hg : (∑ i : Fin d, g i ^ 2) = 1 := by
    change (∑ i : Fin d, x i ^ 2) = 1
    rw [← EuclideanSpace.real_norm_sq_eq, hx]
    norm_num
  have hc (i : Fin d) : (gaussianMatrixLaw d).map
      (fun M : Matrix (Fin d) (Fin d) ℝ => dotProduct g (M i)) = gaussianReal 0 1 := by
    have h := (gaussianMatrix_row_linear_hasLaw g i).map_eq
    rw [hg, Real.toNNReal_one] at h
    exact h
  refine ⟨(measurable_gaussianMatrix_query_coordinates x).aemeasurable, ?_⟩
  have he : (fun M : Matrix (Fin d) (Fin d) ℝ => WithLp.ofLp (Matrix.toEuclideanLin M x)) =
      fun M i => dotProduct g (M i) := by
    funext M i
    exact gaussianMatrix_query_coordinate M x i
  rw [he, (iIndepFun_gaussianMatrix_row_linear g).map_fun_eq_pi_map
    (fun i => (gaussianMatrix_row_linear_hasLaw g i).aemeasurable)]
  simp_rw [hc]
  rfl

/-- Every actual original unit-input matrix query is almost surely
nonzero in positive dimension. Source: arXiv:2604.01978v1,
§7.3, the query normalization in `lem:delta_method`. -/
theorem ae_gaussianMatrix_unit_query_ne_zero {d : ℕ} (hd : 0 < d)
    (x : EucSpace d) (hx : ‖x‖ = 1) :
    ∀ᵐ M ∂gaussianMatrixLaw d, Matrix.toEuclideanLin M x ≠ 0 := by
  have h := (gaussianMatrix_unit_query_hasLaw x hx).ae_iff
    (by fun_prop : Measurable (fun z : Fin d → ℝ => z ≠ 0)) |>.mpr (ae_gaussianRow_ne_zero hd)
  filter_upwards [h] with M hM
  intro he
  apply hM
  rw [he]
  rfl

/-- The actual image of every unit input has norm second moment
`d`. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integral_gaussianMatrix_unit_query_norm_sq {d : ℕ} (x : EucSpace d) (hx : ‖x‖ = 1) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ, ‖Matrix.toEuclideanLin M x‖ ^ 2 ∂gaussianMatrixLaw d) = d := by
  have h := (gaussianMatrix_unit_query_hasLaw x hx).integral_comp
    (f := fun z => ∑ i : Fin d, z i ^ 2) (gaussianRow_norm_sq_memLp_two d).aestronglyMeasurable
  rw [integral_gaussianRow_norm_sq] at h
  simpa only [Function.comp_def, ← EuclideanSpace.real_norm_sq_eq] using h

/-- The actual unit-input image has fourth norm moment `d(d+2)`.
Source: arXiv:2604.01978v1, §7.3, Gaussian moments in `lem:delta_method`. -/
theorem integral_gaussianMatrix_unit_query_norm_fourth {d : ℕ} (x : EucSpace d) (hx : ‖x‖ = 1) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ, ‖Matrix.toEuclideanLin M x‖ ^ 4 ∂gaussianMatrixLaw d) =
      (d : ℝ) * (d + 2) := by
  have h := (gaussianMatrix_unit_query_hasLaw x hx).integral_comp
    (f := fun z => (∑ i : Fin d, z i ^ 2) ^ 2) (integrable_gaussianRow_norm_fourth d).aestronglyMeasurable
  rw [integral_gaussianRow_norm_fourth] at h
  simpa only [Function.comp_def, ← EuclideanSpace.real_norm_sq_eq, ← pow_mul] using h

/-- Every actual unit query has normalized squared-norm deviation
second moment `2/d`. Source: arXiv:2604.01978v1, §7.3,
the single-factor covariance input to `lem:delta_method`. -/
theorem integral_gaussianMatrix_unit_query_deviation_sq {d : ℕ} (hd : 0 < d)
    (x : EucSpace d) (hx : ‖x‖ = 1) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ, (‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1) ^ 2
      ∂gaussianMatrixLaw d) = 2 / (d : ℝ) := by
  have hX := gaussianRow_normalized_norm_memLp_two d hd
  have hi := (memLp_two_iff_integrable_sq hX.aestronglyMeasurable).mp hX
  have h := (gaussianMatrix_unit_query_hasLaw x hx).integral_comp hi.aestronglyMeasurable
  rw [integral_gaussianRow_normalized_norm_sq d hd] at h
  simpa only [Function.comp_def, ← EuclideanSpace.real_norm_sq_eq] using h

/-- **Uniform third deviation moment for every original unit query.**
The row moment rate transfers under the actual matrix-query law,
with the same constant for all unit inputs and positive dimensions.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`, original Gaussian factors. -/
theorem gaussianMatrix_unit_query_deviation_third_uniform :
    ∃ C : ℝ, 0 < C ∧ ∀ d : ℕ, 0 < d → ∀ x : EucSpace d, ‖x‖ = 1 →
      Integrable (fun M : Matrix (Fin d) (Fin d) ℝ => |‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1| ^ 3)
        (gaussianMatrixLaw d) ∧
      (∫ M : Matrix (Fin d) (Fin d) ℝ, |‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1| ^ 3
        ∂gaussianMatrixLaw d) ≤ C * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
  obtain ⟨C, hC, h⟩ := gaussianRow_normalized_norm_third_uniform
  refine ⟨C, hC, ?_⟩
  intro d hd x hx
  have hlaw := gaussianMatrix_unit_query_hasLaw x hx
  have hi := hlaw.integrable_comp (h d hd).1
  have he := hlaw.integral_comp (h d hd).1.aestronglyMeasurable
  constructor
  · simpa only [Function.comp_def, ← EuclideanSpace.real_norm_sq_eq] using hi
  · simpa only [Function.comp_def, ← EuclideanSpace.real_norm_sq_eq] using he.trans_le (h d hd).2

/-- A genuine nonzero unit input in positive dimension satisfies
the law and normalization hypotheses jointly under the actual matrix law. -/
example : (0 : ℕ) < 2 ∧ ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧
    HasLaw (fun M : Matrix (Fin 2) (Fin 2) ℝ =>
      WithLp.ofLp (Matrix.toEuclideanLin M (EuclideanSpace.single 0 (1 : ℝ))))
        (gaussianRowLaw 2) (gaussianMatrixLaw 2) := by
  exact ⟨by decide, by simp, gaussianMatrix_unit_query_hasLaw _ (by simp)⟩

end Transformer.Homogenized

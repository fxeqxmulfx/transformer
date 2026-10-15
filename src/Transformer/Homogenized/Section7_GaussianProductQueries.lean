/-
# Actual product queries and conditioning on the first Gaussian factor

ArXiv:2604.01978v1, §2.3.3, initialization (G), and §7.3,
`lem:delta_method`. The original query matrix has two independent
Gaussian factors. Its actual normalized correlation can be computed
by first applying one factor, normalizing its images, and then applying
the other factor and normalizing again.

The deterministic identity includes zero intermediate images. Under
the original independent Gaussian matrix laws in positive dimension,
the two intermediate normalized images of original unit inputs are
jointly unit almost surely. This is obtained from the actual query
laws, rather than an assumed nondegeneracy condition.

Joint measurability of matrix application to a varying vector gives
measurability and integrability of the actual product correlation.
Fubini then expresses its expectation as the actual iterated factor
expectation. These identities keep both original factors explicit;
no Gaussian law is imposed on the product matrix itself.
-/

import Transformer.Homogenized.Section7_GaussianCorrelationDeviation
import Mathlib.MeasureTheory.Integral.Prod

open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- Matrix application remains measurable when both the original
matrix and its input vary measurably. Source: arXiv:2604.01978v1,
§7.3, the composed original queries in `lem:delta_method`. -/
theorem measurable_gaussianMatrix_query_application {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (A : Ω → Matrix (Fin d) (Fin d) ℝ) (z : Ω → EucSpace d)
    (hA : Measurable A) (hz : Measurable z) :
    Measurable (fun ω => Matrix.toEuclideanLin (A ω) (z ω)) := by
  have hcoords : Measurable (fun ω => WithLp.ofLp (Matrix.toEuclideanLin (A ω) (z ω))) := by
    apply Measurable.of_eval
    intro i
    change Measurable (fun ω => Matrix.toEuclideanLin (A ω) (z ω) i)
    simp only [gaussianMatrix_query_coordinate, dotProduct]
    have hzc : Measurable (fun ω => WithLp.ofLp (z ω)) :=
      (WithLp.measurable_ofLp (p := 2) (X := Fin d → ℝ)).comp hz
    fun_prop
  have h := (WithLp.measurable_toLp (p := 2) (X := Fin d → ℝ)).comp hcoords
  simpa only [Function.comp_def, WithLp.toLp_ofLp] using h

/-- Normalizing the original intermediate vector before applying
the next factor preserves the final actual normalization.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
composition of the original normalized-query factors. -/
theorem gaussianMatrix_normalize_intermediate {d : ℕ}
    (M : Matrix (Fin d) (Fin d) ℝ) (z : EucSpace d) :
    normalizeLayer (Matrix.toEuclideanLin M (normalizeLayer z)) =
      normalizeLayer (Matrix.toEuclideanLin M z) := by
  by_cases hz : z = 0
  · subst z
    simp [normalizeLayer]
  change normalizeLayer (Matrix.toEuclideanLin M (‖z‖⁻¹ • z)) = _
  rw [(Matrix.toEuclideanLin M).map_smul,
    normalizeLayer_smul (inv_pos.mpr (norm_pos_iff.mpr hz))]

/-- The actual normalized product correlation equals the correlation
obtained by feeding the first factor's normalized original images
into the second factor. The identity includes all zero images.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem gaussian_product_normalized_query_identity {d : ℕ}
    (N M : Matrix (Fin d) (Fin d) ℝ) (x y : EucSpace d) :
    inner ℝ (normalizeLayer (Matrix.toEuclideanLin M (normalizeLayer (Matrix.toEuclideanLin N x))))
      (normalizeLayer (Matrix.toEuclideanLin M (normalizeLayer (Matrix.toEuclideanLin N y)))) =
        inner ℝ (normalizeLayer (Matrix.toEuclideanLin M (Matrix.toEuclideanLin N x)))
          (normalizeLayer (Matrix.toEuclideanLin M (Matrix.toEuclideanLin N y))) := by
  rw [gaussianMatrix_normalize_intermediate, gaussianMatrix_normalize_intermediate]

/-- The two original intermediate normalized queries are jointly
unit almost surely under their actual first-factor law.
Source: arXiv:2604.01978v1, initialization (G), §7.3,
the random inputs to the second factor in `lem:delta_method`. -/
theorem ae_gaussianMatrix_normalized_queries_unit {d : ℕ} (hd : 0 < d)
    (x y : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    ∀ᵐ N ∂gaussianMatrixLaw d,
      ‖normalizeLayer (Matrix.toEuclideanLin N x)‖ = 1 ∧
        ‖normalizeLayer (Matrix.toEuclideanLin N y)‖ = 1 := by
  filter_upwards [ae_gaussianMatrix_unit_query_ne_zero hd x hx,
    ae_gaussianMatrix_unit_query_ne_zero hd y hy] with N hNx hNy
  exact ⟨norm_normalizeLayer hNx, norm_normalizeLayer hNy⟩

/-- An actual two-factor query is jointly measurable in both
original matrices. Source: arXiv:2604.01978v1, initialization (G),
§7.3, the product query in `lem:delta_method`. -/
theorem measurable_gaussian_product_query {d : ℕ} (x : EucSpace d) :
    Measurable (fun P : Matrix (Fin d) (Fin d) ℝ × Matrix (Fin d) (Fin d) ℝ =>
      Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 x)) := by
  exact measurable_gaussianMatrix_query_application _ _ measurable_snd
    ((measurable_gaussianMatrix_query x).comp measurable_fst)

/-- The actual product correlation is integrable under the two
original independent matrix laws. Source: arXiv:2604.01978v1,
§7.3, the expectation in `lem:delta_method`. -/
theorem integrable_gaussian_product_correlation {d : ℕ} (x y : EucSpace d) :
    Integrable (fun P : Matrix (Fin d) (Fin d) ℝ × Matrix (Fin d) (Fin d) ℝ =>
      inner ℝ (normalizeLayer (Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 x)))
        (normalizeLayer (Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 y))))
          ((gaussianMatrixLaw d).prod (gaussianMatrixLaw d)) := by
  exact integrable_normalized_inner _ _ _
    (measurable_gaussian_product_query x) (measurable_gaussian_product_query y)

/-- **Actual conditioning identity for the original product correlation.**
Fubini gives the original iterated factor expectation; no conditional
covariance or independence of the two query images is assumed.
Source: arXiv:2604.01978v1, initialization (G), §7.3, `lem:delta_method`. -/
theorem integral_gaussian_product_correlation {d : ℕ} (x y : EucSpace d) :
    (∫ P : Matrix (Fin d) (Fin d) ℝ × Matrix (Fin d) (Fin d) ℝ,
      inner ℝ (normalizeLayer (Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 x)))
        (normalizeLayer (Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 y)))
          ∂(gaussianMatrixLaw d).prod (gaussianMatrixLaw d)) =
      ∫ N : Matrix (Fin d) (Fin d) ℝ, ∫ M : Matrix (Fin d) (Fin d) ℝ,
        inner ℝ (normalizeLayer (Matrix.toEuclideanLin M (normalizeLayer (Matrix.toEuclideanLin N x))))
          (normalizeLayer (Matrix.toEuclideanLin M (normalizeLayer (Matrix.toEuclideanLin N y))))
            ∂gaussianMatrixLaw d ∂gaussianMatrixLaw d := by
  rw [integral_prod _ (integrable_gaussian_product_correlation x y)]
  simp_rw [gaussian_product_normalized_query_identity]

/-- The actual conditional correlation expectation is integrable
under the original first-factor matrix law. Its integrability follows
from the bounded original product correlation and Fubini, with no
regularity assumption on the conditional expectation itself.
Source: arXiv:2604.01978v1, initialization (G), §7.3,
the iterated original-factor expectation in `lem:delta_method`. -/
theorem integrable_gaussian_product_conditional_correlation {d : ℕ} (x y : EucSpace d) :
    Integrable (fun N : Matrix (Fin d) (Fin d) ℝ =>
      ∫ M : Matrix (Fin d) (Fin d) ℝ,
        inner ℝ (normalizeLayer (Matrix.toEuclideanLin M (normalizeLayer (Matrix.toEuclideanLin N x))))
          (normalizeLayer (Matrix.toEuclideanLin M (normalizeLayer (Matrix.toEuclideanLin N y))))
            ∂gaussianMatrixLaw d) (gaussianMatrixLaw d) := by
  have h := (integrable_gaussian_product_correlation x y).integral_prod_left
  simpa only [gaussian_product_normalized_query_identity] using h

/-- Positive dimension, two actual coordinate unit inputs and the
measurable original matrix and vector projections satisfy all
query, conditioning and intermediate-unit hypotheses jointly. -/
example : (0 : ℕ) < 2 ∧ ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧
    ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ = 1 ∧
    Measurable (Prod.snd : Matrix (Fin 2) (Fin 2) ℝ × Matrix (Fin 2) (Fin 2) ℝ → Matrix (Fin 2) (Fin 2) ℝ) ∧
    Measurable (fun P : Matrix (Fin 2) (Fin 2) ℝ × Matrix (Fin 2) (Fin 2) ℝ =>
      Matrix.toEuclideanLin P.1 (EuclideanSpace.single 0 (1 : ℝ))) := by
  exact ⟨by decide, by simp, by simp, measurable_snd,
    (measurable_gaussianMatrix_query _).comp measurable_fst⟩

end Transformer.Homogenized

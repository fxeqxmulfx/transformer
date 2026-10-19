module

public import Transformer.Homogenized.Section7_GaussianHeadDelta
public import Transformer.Homogenized.Section7_ConditionalQueryMoments

/-!
# Exact second moments of the original two-factor Gaussian query

ArXiv:2604.01978v1, §2.3.3, initialization (G), and §7,
the Gaussian query estimate in `thm:clustering_small_beta`.
The original query matrix is `A = WW′ᵀ`, with independent Gaussian
entry scales `σA` in each factor. For every deterministic input `x`,
its actual query therefore satisfies

  `E ‖A x‖² = d² σA⁴ ‖x‖²`.

Each original factor contributes exactly `d` at unit entry variance.
The integrability proof first proves the one-factor squared-norm
moment, then uses the actual independent product law and its actual
conditional integral. The head pushforward retains the original
transpose. The proved original block-scaling and canonical-law
identities transfer this calculation to every original Gaussian head
law, including zero value variance and zero attention variance.

This exact query estimate avoids the full matrix Frobenius moment,
which would introduce an unnecessary extra dimension factor in the
temperature Taylor remainder. No product matrix is declared Gaussian.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The actual squared query norm of one original standard Gaussian
factor is integrable for every input. Source: arXiv:2604.01978v1,
§2.3.3, initialization (G), used in §7's small-temperature estimate. -/
theorem integrable_gaussianMatrix_query_norm_sq_raw {d : ℕ} (x : EucSpace d) :
    Integrable (fun M : Matrix (Fin d) (Fin d) ℝ => ‖Matrix.toEuclideanLin M x‖ ^ 2)
      (gaussianMatrixLaw d) := by
  simp_rw [gaussianMatrix_query_norm_sq]
  exact integrable_finsetSum _ fun i hi => integrable_gaussianMatrix_row_linear_pow (WithLp.ofLp x) i 2

/-- A single actual standard Gaussian factor contributes exactly
dimension times the original squared input norm.
Source: arXiv:2604.01978v1, §2.3.3, initialization (G). -/
theorem integral_gaussianMatrix_query_norm_sq_raw {d : ℕ} (x : EucSpace d) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ, ‖Matrix.toEuclideanLin M x‖ ^ 2 ∂gaussianMatrixLaw d) =
      (d : ℝ) * ‖x‖ ^ 2 := by
  simp_rw [gaussianMatrix_query_norm_sq]
  rw [integral_finsetSum _ fun i hi => integrable_gaussianMatrix_row_linear_pow (WithLp.ofLp x) i 2]
  simp only [integral_gaussianMatrix_row_linear_sq, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  rw [← EuclideanSpace.real_norm_sq_eq]

/-- The original two-factor squared query norm is integrable under
the actual independent matrix product law.
Source: arXiv:2604.01978v1, §7, the two-factor Gaussian query estimate. -/
theorem integrable_gaussian_product_query_norm_sq {d : ℕ} (x : EucSpace d) :
    Integrable (fun P : Matrix (Fin d) (Fin d) ℝ × Matrix (Fin d) (Fin d) ℝ =>
      ‖Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 x)‖ ^ 2)
      ((gaussianMatrixLaw d).prod (gaussianMatrixLaw d)) := by
  apply (integrable_prod_iff ((measurable_gaussian_product_query x).norm.pow_const 2).aestronglyMeasurable).mpr
  constructor
  · exact ae_of_all _ fun N => integrable_gaussianMatrix_query_norm_sq_raw (Matrix.toEuclideanLin N x)
  · simp_rw [Real.norm_of_nonneg (sq_nonneg _), integral_gaussianMatrix_query_norm_sq_raw]
    exact (integrable_gaussianMatrix_query_norm_sq_raw x).const_mul d

/-- Two actual independent standard Gaussian factors contribute
`d²` to the original squared input norm.
Source: arXiv:2604.01978v1, §7, the query estimate after the
small-temperature Taylor expansion. -/
theorem integral_gaussian_product_query_norm_sq {d : ℕ} (x : EucSpace d) :
    (∫ P : Matrix (Fin d) (Fin d) ℝ × Matrix (Fin d) (Fin d) ℝ,
      ‖Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 x)‖ ^ 2
      ∂(gaussianMatrixLaw d).prod (gaussianMatrixLaw d)) = (d : ℝ) ^ 2 * ‖x‖ ^ 2 := by
  rw [integral_prod _ (integrable_gaussian_product_query_norm_sq x)]
  simp_rw [integral_gaussianMatrix_query_norm_sq_raw]
  rw [integral_const_mul, integral_gaussianMatrix_query_norm_sq_raw]
  ring

/-- The original head's actual query vector has precisely the actual
independent two-factor query law, retaining `WW′ᵀ`.
Source: arXiv:2604.01978v1, §2.3.3, initialization (G). -/
theorem gaussHead_unit_query_map {d : ℕ} (x : EucSpace d) :
    (gaussHeadLaw d 1 1).map (fun θ => qkMap θ x) =
      ((gaussianMatrixLaw d).prod (gaussianMatrixLaw d)).map
        (fun P => Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 x)) := by
  have hq : Measurable (fun θ : HeadParam d => qkMap θ x) :=
    (measurable_gaussianMatrix_query x).comp measurable_snd
  have ht : Measurable (fun M : Matrix (Fin d) (Fin d) ℝ => M.transpose) := by fun_prop
  have hp : Measurable (fun z : GaussIdx d → ℝ => ((gaussBlock (2 : Fin 3) z).transpose, gaussBlock (1 : Fin 3) z)) :=
    (ht.comp (measurable_gaussBlock (2 : Fin 3))).prodMk (measurable_gaussBlock (1 : Fin 3))
  rw [gaussHeadLaw, Measure.map_map hq (measurable_gaussHead d),
    ← originalGaussianMatrix_transpose_pair_map (d := d) 1 2 (by decide),
    Measure.map_map (measurable_gaussian_product_query x) hp]
  congr 1
  funext z
  exact gaussianMatrix_product_query _ _ x

/-- The original standard head's actual squared query norm is
integrable. Source: arXiv:2604.01978v1, §7, its Gaussian query estimate. -/
theorem integrable_gaussHead_unit_query_norm_sq {d : ℕ} (x : EucSpace d) :
    Integrable (fun θ : HeadParam d => ‖qkMap θ x‖ ^ 2) (gaussHeadLaw d 1 1) := by
  have hf : Measurable (fun u : EucSpace d => ‖u‖ ^ 2) := by fun_prop
  have hp := (integrable_map_measure hf.aestronglyMeasurable
    (measurable_gaussian_product_query x).aemeasurable).mpr (integrable_gaussian_product_query_norm_sq x)
  rw [← gaussHead_unit_query_map] at hp
  exact (integrable_map_measure hf.aestronglyMeasurable
    ((measurable_gaussianMatrix_query x).comp measurable_snd).aemeasurable).mp hp

/-- The original standard head has exact query second moment `d² ‖x‖²`.
Source: arXiv:2604.01978v1, §7, the small-temperature Gaussian query estimate. -/
theorem integral_gaussHead_unit_query_norm_sq {d : ℕ} (x : EucSpace d) :
    (∫ θ : HeadParam d, ‖qkMap θ x‖ ^ 2 ∂gaussHeadLaw d 1 1) = (d : ℝ) ^ 2 * ‖x‖ ^ 2 := by
  have hf : Measurable (fun u : EucSpace d => ‖u‖ ^ 2) := by fun_prop
  have hq : Measurable (fun θ : HeadParam d => qkMap θ x) :=
    (measurable_gaussianMatrix_query x).comp measurable_snd
  rw [← integral_map hq.aemeasurable hf.aestronglyMeasurable,
    gaussHead_unit_query_map, integral_map (measurable_gaussian_product_query x).aemeasurable hf.aestronglyMeasurable]
  exact integral_gaussian_product_query_norm_sq x

/-- The original head's actual query norm scales by the fourth power
of the original Gaussian entry scale, including scale zero.
Source: arXiv:2604.01978v1, initialization (G), `A = WW′ᵀ`. -/
theorem norm_sq_qkMap_block_scales {d : ℕ} (σV σA : ℝ≥0) (θ : HeadParam d) (x : EucSpace d) :
    ‖qkMap (((σV : ℝ) • θ.1, (σA : ℝ) ^ 2 • θ.2) : HeadParam d) x‖ ^ 2 =
      (σA : ℝ) ^ 4 * ‖qkMap θ x‖ ^ 2 := by
  change ‖Matrix.toEuclideanLin ((σA : ℝ) ^ 2 • θ.2) x‖ ^ 2 =
    (σA : ℝ) ^ 4 * ‖Matrix.toEuclideanLin θ.2 x‖ ^ 2
  rw [LinearEquiv.map_smul, LinearMap.smul_apply, norm_smul,
    Real.norm_of_nonneg (sq_nonneg _), mul_pow]
  ring

/-- **Every original Gaussian head has its genuine exact query second
moment, with both original factors and separate variances retained.**
Source: arXiv:2604.01978v1, §2.3.3, initialization (G), and §7,
the Gaussian query estimate in `thm:clustering_small_beta`.
No nondegeneracy or unit-input restriction is needed. -/
theorem gaussian_head_query_second_moment {d : ℕ} {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (x : EucSpace d) :
    Integrable (fun θ : HeadParam d => ‖qkMap θ x‖ ^ 2) ρ ∧
      (∫ θ, ‖qkMap θ x‖ ^ 2 ∂ρ) = (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 * ‖x‖ ^ 2 := by
  have hm : Measurable (fun θ : HeadParam d => ‖qkMap θ x‖ ^ 2) :=
    ((measurable_gaussianMatrix_query x).comp measurable_snd).norm.pow_const 2
  rw [gaussianHeadLaw_eq_canonical hρ, gaussHeadLaw_eq_map_unit_block_scales]
  constructor
  · apply (integrable_map_measure hm.aestronglyMeasurable (by fun_prop)).mpr
    simpa only [Function.comp_def, norm_sq_qkMap_block_scales] using
      (integrable_gaussHead_unit_query_norm_sq x).const_mul ((σA : ℝ) ^ 4)
  · rw [integral_map (by fun_prop) hm.aestronglyMeasurable]
    simp_rw [norm_sq_qkMap_block_scales]
    rw [integral_const_mul, integral_gaussHead_unit_query_norm_sq]
    ring

/-- Positive original attention variance, zero value variance and a
nonzero unit query jointly satisfy every original-law hypothesis. -/
example : IsGaussianHeadLaw 2 0 1 (gaussHeadLaw 2 0 1) ∧
    ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧
    (∫ θ : HeadParam 2, ‖qkMap θ (EuclideanSpace.single (0 : Fin 2) 1)‖ ^ 2
      ∂gaussHeadLaw 2 0 1) = 4 := by
  refine ⟨isGaussianHeadLaw_gaussHeadLaw 2 0 1, by simp, ?_⟩
  rw [(gaussian_head_query_second_moment (isGaussianHeadLaw_gaussHeadLaw 2 0 1) _).2]
  norm_num

end Transformer.Homogenized

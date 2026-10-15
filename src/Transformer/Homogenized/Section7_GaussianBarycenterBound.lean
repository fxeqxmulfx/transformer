/-
# A cutoff bound for the original Gaussian uniform-law barycenter

ArXiv:2604.01978v1, §7.3, `lemma:Laplace_method`, Step 1.
The original uniform spherical law has density one, so it satisfies
every lower-density, upper-density, smoothness and gradient bound of
`ass:low-temperature` with constants `(1,1,0)`.

The actual barycenter has norm at most one. At queries of norm at
most `R`, the proved isotropic vector-integral bound improves this
to squared norm at most `exp(2R)/d`. At all larger queries the unit
bound is at most the query's squared norm divided by `R²`. These
pointwise alternatives yield one integrable polynomial majorant.

Integrating against the original Gaussian product head at a coordinate
unit query and using its actual second moment gives
`E ‖m‖² ≤ exp(2R)/d + β²d²/R²` at unit entry variances.
The actual normalized-query self-correlation is one, giving the error
lower bound below. The proof uses the source's original barycenter,
original law and original product matrix, with no postulated limit
or Gaussian approximation to the product-matrix query.
-/

import Transformer.Homogenized.Section7_GaussianQueryMoments
import Transformer.Homogenized.Section7_UniformBarycenter

open Real MeasureTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The original uniform law realizes all low-temperature density
conditions. Source: arXiv:2604.01978v1, §2.3.3, `ass:low-temperature`. -/
theorem isLowTemperature_map_spherical_uniform {d : ℕ} (ν : Measure (SSphere d))
    (hν : Metastability.IsUniformOn d ν) :
    IsLowTemperature d (ν.map (Subtype.val : SSphere d → EucSpace d))
      (fun _ => ν.map Subtype.val) (fun _ _ => 1) 1 1 0 := by
  refine ⟨isUniformAmbient_map_spherical ν hν, one_pos, ?_, ?_, ?_, ?_, ?_⟩
  · intro t ht
    simp
  · intro t ht
    exact contDiff_const
  · intro t ht y hy
    exact le_rfl
  · intro t ht y hy
    exact le_rfl
  · intro t ht y hy
    simp [proj]

/-- The actual uniform-law barycenter lies in the unit ball.
Source: arXiv:2604.01978v1, §7.3, `lemma:Laplace_method`. -/
theorem norm_softBary_map_spherical_uniform_le_one {d : ℕ} (ν : Measure (SSphere d))
    (hν : Metastability.IsUniformOn d ν) (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ) (x : EucSpace d) :
    ‖softBary β A (ν.map (Subtype.val : SSphere d → EucSpace d)) x‖ ≤ 1 := by
  have hσ := isUniformAmbient_map_spherical ν hν
  have : IsProbabilityMeasure ν := hν.1
  let : IsProbabilityMeasure (ν.map (Subtype.val : SSphere d → EucSpace d)) := hσ.1
  apply norm_softBary_le_one
  · rw [ae_iff]
    exact hσ.2.1
  · apply (integrable_map_measure
      (by dsimp only [softWeight]; fun_prop : Continuous
        (fun y : EucSpace d => softWeight β A x y)).aestronglyMeasurable
      continuous_subtype_val.measurable.aemeasurable).mpr
    exact Perspective.integrable_of_continuous_compact
      (by dsimp only [softWeight]; fun_prop) ν

/-- The dimension-sensitive bound on small queries and the unit bound
on large queries give one explicit polynomial majorant.
Source: arXiv:2604.01978v1, §7.3, `lemma:Laplace_method`, Step 1. -/
theorem softBary_uniform_query_cutoff_bound {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (β R : ℝ) (hR : 0 < R) (A : Matrix (Fin d) (Fin d) ℝ) (x : EucSpace d) :
    ‖softBary β A (ν.map (Subtype.val : SSphere d → EucSpace d)) x‖ ^ 2 ≤
      Real.exp (2 * R) / d + (β ^ 2 / R ^ 2) * ‖Matrix.toEuclideanLin A x‖ ^ 2 := by
  have hdR : (0 : ℝ) < d := by exact_mod_cast hd
  by_cases hq : ‖β • Matrix.toEuclideanLin A x‖ ≤ R
  · have hm := (norm_softBary_uniform_spherical_sq_le hd ν hν β A x).trans
      (Real.exp_le_exp.mpr (by linarith : 2 * ‖β • Matrix.toEuclideanLin A x‖ ≤ 2 * R))
    have hn : ‖softBary β A (ν.map (Subtype.val : SSphere d → EucSpace d)) x‖ ^ 2 ≤
        Real.exp (2 * R) / d := (le_div_iff₀ hdR).mpr (by nlinarith)
    exact hn.trans (le_add_of_nonneg_right (by positivity))
  · have hn := norm_softBary_map_spherical_uniform_le_one ν hν β A x
    have hm : ‖softBary β A (ν.map (Subtype.val : SSphere d → EucSpace d)) x‖ ^ 2 ≤ 1 := by
      nlinarith [norm_nonneg (softBary β A (ν.map (Subtype.val : SSphere d → EucSpace d)) x)]
    have hnorm : ‖β • Matrix.toEuclideanLin A x‖ ^ 2 = β ^ 2 * ‖Matrix.toEuclideanLin A x‖ ^ 2 := by
      simp [norm_smul, mul_pow]
    have hlarge : 1 ≤ (β ^ 2 / R ^ 2) * ‖Matrix.toEuclideanLin A x‖ ^ 2 := by
      have hs : R ^ 2 ≤ β ^ 2 * ‖Matrix.toEuclideanLin A x‖ ^ 2 := by
        nlinarith [norm_nonneg (β • Matrix.toEuclideanLin A x)]
      have h : 1 ≤ β ^ 2 * ‖Matrix.toEuclideanLin A x‖ ^ 2 / R ^ 2 :=
        (le_div_iff₀ (sq_pos_of_pos hR)).mpr (by simpa using hs)
      convert h using 1
      ring
    exact hm.trans (hlarge.trans (le_add_of_nonneg_left (by positivity)))

/-- The original Gaussian second moment controls the original
uniform-law self-correlation. Source: arXiv:2604.01978v1,
§7.3, `lemma:Laplace_method`, retaining dimension in the estimate. -/
theorem baryCorr_uniform_gaussian_query_cutoff_bound {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (β R : ℝ) (hR : 0 < R) (j : Fin d) :
    baryCorr β (gaussHeadLaw d 1 1) (ν.map (Subtype.val : SSphere d → EucSpace d))
      (EuclideanSpace.single j 1) (EuclideanSpace.single j 1) ≤
        Real.exp (2 * R) / d + β ^ 2 * (d : ℝ) ^ 2 / R ^ 2 := by
  rw [baryCorr_self]
  have hi : Integrable (fun θ : HeadParam d => Real.exp (2 * R) / d +
      (β ^ 2 / R ^ 2) * ‖qkMap θ (EuclideanSpace.single j 1)‖ ^ 2) (gaussHeadLaw d 1 1) :=
    (integrable_const (Real.exp (2 * R) / d)).add
    ((integrable_gaussianHead_query_norm_sq j).const_mul (β ^ 2 / R ^ 2))
  have hb : (∫ θ : HeadParam d, ‖softBary β θ.2
      (ν.map (Subtype.val : SSphere d → EucSpace d)) (EuclideanSpace.single j 1)‖ ^ 2
      ∂gaussHeadLaw d 1 1) ≤ ∫ θ : HeadParam d,
      (Real.exp (2 * R) / d + (β ^ 2 / R ^ 2) * ‖qkMap θ (EuclideanSpace.single j 1)‖ ^ 2)
      ∂gaussHeadLaw d 1 1 :=
    integral_mono_of_nonneg (ae_of_all _ (fun θ => sq_nonneg
    ‖softBary β θ.2 (ν.map (Subtype.val : SSphere d → EucSpace d)) (EuclideanSpace.single j 1)‖)) hi
    (ae_of_all _ (fun θ => softBary_uniform_query_cutoff_bound hd ν hν β R hR θ.2
      (EuclideanSpace.single j 1)))
  rw [integral_add (integrable_const _)
      ((integrable_gaussianHead_query_norm_sq j).const_mul (β ^ 2 / R ^ 2)),
    integral_const, probReal_univ, one_smul, integral_const_mul,
    integral_gaussianHead_query_norm_sq] at hb
  convert hb using 1
  ring

/-- The actual Laplace self-correlation error has this explicit lower
bound. Source: arXiv:2604.01978v1, §7.3, `lemma:Laplace_method`.
Both sides of its claimed approximation retain their original integrals. -/
theorem one_sub_cutoff_le_laplace_self_error {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (β R : ℝ) (hR : 0 < R) (j : Fin d) :
    1 - (Real.exp (2 * R) / d + β ^ 2 * (d : ℝ) ^ 2 / R ^ 2) ≤
      |baryCorr β (gaussHeadLaw d 1 1) (ν.map (Subtype.val : SSphere d → EucSpace d))
        (EuclideanSpace.single j 1) (EuclideanSpace.single j 1) -
      ∫ θ : HeadParam d, inner ℝ
        (normalizeLayer (qkMap θ (EuclideanSpace.single j (1 : ℝ))))
        (normalizeLayer (qkMap θ (EuclideanSpace.single j (1 : ℝ)))) ∂gaussHeadLaw d 1 1| := by
  rw [integral_gaussianHead_normalized_query_self]
  have hb := baryCorr_uniform_gaussian_query_cutoff_bound hd ν hν β R hR j
  linarith [neg_le_abs (baryCorr β (gaussHeadLaw d 1 1)
    (ν.map (Subtype.val : SSphere d → EucSpace d))
    (EuclideanSpace.single j 1) (EuclideanSpace.single j 1) - 1)]

/-- Positive dimension, genuine uniform probability and a positive
cutoff hold jointly for the actual two-point spherical uniform law.
The original coordinate query has unit norm and the original head
ensemble has the nondegenerate Gaussian initialization. -/
example : (0 : ℕ) < 1 ∧ Metastability.IsUniformOn 1 Normalization.oneDimUniform ∧
    (0 : ℝ) < 1 ∧ IsGaussianHeadLaw 1 1 1 (gaussHeadLaw 1 1 1) ∧
    ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ) : EucSpace 1)‖ = 1 :=
  ⟨by decide, ⟨inferInstance, Normalization.oneDimUniform_invariant⟩,
    one_pos, isGaussianHeadLaw_gaussHeadLaw 1 1 1, by simp⟩

end Transformer.Homogenized

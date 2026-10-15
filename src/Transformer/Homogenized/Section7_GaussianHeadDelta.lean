/-
# Corrected delta method for every original Gaussian head law

ArXiv:2604.01978v1, §2.3.3, initialization (G), and §7.3,
`lem:delta_method`. The source's correction `(r³-r)/(2d)` is
replaced by the proved two-factor correction `(r³-r)/d`.
All original dimension, unit-query and Gaussian-head quantifiers
are retained, with one constant independent of their choices.

The actual original entry ensemble is obtained from independent
standard entries by multiplying each value entry by `σV` and each
query/key entry by `σA`. This identity permits zero value variance.
The resulting original head has value scale `σV` and query matrix
scale `σA²`, reflecting its genuine product of two Gaussian factors.

Every law satisfying the original Gaussian initialization is the
canonical original head pushforward. For positive query scale,
normalization removes its common positive matrix scale. Its actual
normalized-correlation expectation is therefore the original standard
head expectation, regardless of the original value variance.

The corrected uniform expansion follows from the proved standard-head
expansion. No covariance, remainder or law-transfer identity is assumed.
-/

import Transformer.Homogenized.Section7_GaussianMatrixPair
import Transformer.Homogenized.GaussianDrift

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The genuine original entry law scales its value and query blocks
separately, including either zero scale.
Source: arXiv:2604.01978v1, initialization (G). -/
theorem gaussEnsemble_eq_map_unit_block_scales (d : ℕ) (σV σA : ℝ≥0) :
    gaussEnsemble d σV σA = (gaussEnsemble d 1 1).map
      (fun z : GaussIdx d → ℝ => fun p => (blockSigma σV σA p : ℝ) * z p) := by
  simp only [gaussEnsemble, entryVar, one_pow, ite_self]
  rw [Measure.pi_map_pi (f := fun p : GaussIdx d => fun u : ℝ => (blockSigma σV σA p : ℝ) * u)
    (fun p => (measurable_const_mul (blockSigma σV σA p : ℝ)).aemeasurable)]
  congr 1
  funext p
  rw [gaussianReal_map_const_mul, mul_zero, mul_one]
  by_cases hp : p.1 = 0 <;> simp [blockSigma, hp]
  all_goals congr 1

/-- The actual original head has linear value scaling and quadratic
query-matrix scaling for arbitrary separate original entry scales.
Source: arXiv:2604.01978v1, initialization (G), `A=WW′ᵀ`. -/
theorem gaussHeadLaw_eq_map_unit_block_scales (d : ℕ) (σV σA : ℝ≥0) :
    gaussHeadLaw d σV σA = (gaussHeadLaw d 1 1).map
      (fun θ : HeadParam d => ((σV : ℝ) • θ.1, (σA : ℝ) ^ 2 • θ.2)) := by
  rw [gaussHeadLaw, gaussEnsemble_eq_map_unit_block_scales,
    Measure.map_map (measurable_gaussHead d) (by fun_prop)]
  rw [gaussHeadLaw, Measure.map_map (by fun_prop) (measurable_gaussHead d)]
  congr 1
  funext z
  apply Prod.ext
  · ext i j
    simp [gaussBlock, blockSigma, smul_eq_mul]
  · ext i j
    simp only [Function.comp_apply, gaussBlock, Matrix.of_apply, Matrix.mul_apply,
      Matrix.transpose_apply, Matrix.smul_apply, smul_eq_mul]
    norm_num only [blockSigma, show (1 : Fin 3) ≠ 0 by decide, show (2 : Fin 3) ≠ 0 by decide, ite_false]
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun q hq => by ring

/-- Every original Gaussian head law is its canonical entry-law
pushforward, with all original variance parameters unchanged.
Source: arXiv:2604.01978v1, initialization (G). -/
theorem gaussianHeadLaw_eq_canonical {d : ℕ} {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) : ρ = gaussHeadLaw d σV σA := by
  have h := map_headOfBlocks_pi hρ
  have he : (fun p : GaussIdx d => gaussianReal 0 (blockSigma σV σA p ^ 2)) =
      fun p => gaussianReal 0 (entryVar σV σA p) := by
    funext p
    by_cases hp : p.1 = 0 <;> simp [blockSigma, entryVar, hp]
  rw [he] at h
  change gaussHeadLaw d σV σA = ρ at h
  exact h.symm

/-- Positive query variance makes the actual normalized correlation
independent of both original entry scales. Zero value variance is allowed.
Source: arXiv:2604.01978v1, initialization (G), §7.3,
the normalized-query expectation in `lem:delta_method`. -/
theorem integral_gaussHead_correlation_block_scales {d : ℕ} (σV σA : ℝ≥0)
    (hσA : 0 < σA) (x y : EucSpace d) :
    (∫ θ : HeadParam d,
      inner ℝ (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ y)) ∂gaussHeadLaw d σV σA) =
      ∫ θ : HeadParam d,
        inner ℝ (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ y)) ∂gaussHeadLaw d 1 1 := by
  have hm : Measurable (fun θ : HeadParam d =>
      inner ℝ (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ y))) :=
    measurable_normalized_inner _ _ ((measurable_gaussianMatrix_query x).comp measurable_snd)
      ((measurable_gaussianMatrix_query y).comp measurable_snd)
  rw [gaussHeadLaw_eq_map_unit_block_scales,
    integral_map (by fun_prop) hm.aestronglyMeasurable]
  apply integral_congr_ae
  filter_upwards with θ
  have hc : (0 : ℝ) < (σA : ℝ) ^ 2 := sq_pos_of_pos (by exact_mod_cast hσA)
  have hquery (z : EucSpace d) :
      qkMap (((σV : ℝ) • θ.1, (σA : ℝ) ^ 2 • θ.2) : HeadParam d) z = (σA : ℝ) ^ 2 • qkMap θ z := by
    change Matrix.toEuclideanLin ((σA : ℝ) ^ 2 • θ.2) z =
      (σA : ℝ) ^ 2 • Matrix.toEuclideanLin θ.2 z
    rw [LinearEquiv.map_smul, LinearMap.smul_apply]
  rw [hquery, hquery, normalizeLayer_smul hc, normalizeLayer_smul hc]

/-- Every original Gaussian head law with positive query scale has
the actual standard-head normalized-correlation expectation.
Arbitrary value variance, including zero, is retained in the hypothesis.
The observable uses the original head's actual query/key matrix.
Source: arXiv:2604.01978v1, initialization (G), §7.3,
the scale-invariant normalized-query expectation in `lem:delta_method`. -/
theorem integral_gaussian_head_correlation_eq_unit {d : ℕ} {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hσA : 0 < σA) (hρ : IsGaussianHeadLaw d σV σA ρ)
    (x y : EucSpace d) :
    (∫ θ : HeadParam d,
      inner ℝ (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ y)) ∂ρ) =
      ∫ θ : HeadParam d,
        inner ℝ (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ y)) ∂gaussHeadLaw d 1 1 := by
  rw [gaussianHeadLaw_eq_canonical hρ, integral_gaussHead_correlation_block_scales σV σA hσA]

/-- **Corrected uniform delta method with every original head-law quantifier.**
The source states correction `(r³-r)/(2d)` for two factors; the proved
correction is `(r³-r)/d`. All original unit queries, positive dimensions,
arbitrary value scales, positive query scales and original Gaussian head
laws share the same positive constant and actual `d^(-3/2)` remainder.
Source: arXiv:2604.01978v1, initialization (G), §7.3, `lem:delta_method`. -/
theorem gaussian_head_delta_method :
    ∃ C : ℝ, 0 < C ∧
      ∀ d : ℕ, 0 < d → ∀ σV σA : ℝ≥0, 0 < σA →
      ∀ ρ : Measure (HeadParam d), IsGaussianHeadLaw d σV σA ρ →
      ∀ x y : EucSpace d, ‖x‖ = 1 → ‖y‖ = 1 →
        |(∫ θ, inner ℝ (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ y)) ∂ρ) -
            (inner ℝ x y + (inner ℝ x y ^ 3 - inner ℝ x y) / (d : ℝ))| ≤
          C * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
  obtain ⟨C, hC, h⟩ := gaussHead_unit_delta_method
  refine ⟨C, hC, ?_⟩
  intro d hd σV σA hσA ρ hρ x y hx hy
  rw [integral_gaussian_head_correlation_eq_unit hσA hρ]
  exact h d hd x y hx hy

/-- Positive query scale and zero value scale satisfy every original
head-law hypothesis jointly with positive dimension and two genuine
unit queries. Thus the full statement also covers the allowed zero-value case. -/
example : (0 : ℕ) < 2 ∧ (0 : ℝ≥0) < 1 ∧
    IsGaussianHeadLaw 2 0 1 (gaussHeadLaw 2 0 1) ∧
    ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧ ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ = 1 := by
  exact ⟨by decide, by norm_num, isGaussianHeadLaw_gaussHeadLaw 2 0 1, by simp, by simp⟩

end Transformer.Homogenized

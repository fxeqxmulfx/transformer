/-
# Exact rescaling of the original Gaussian-product head

ArXiv:2604.01978v1, §2.3, initialization (G), and §7.3,
`lemma:Laplace_method`. Scaling each original independent Gaussian
entry by `c` scales the value matrix by `c` and the original product
attention matrix by `c²`. The original head pushforward is proved
from its entry product law, rather than assuming a Gaussian law for
the product attention matrix.

For positive `c`, the corresponding measurable equivalence gives an
exact change of variables in every original head integral. The actual
softmax barycenter at matrix `c²A` and temperature `β` is the actual
one at matrix `A` and temperature `βc²`. Positive scaling leaves the
normalized original query unchanged, with self-correlation one.

At the paper's common entry-variance scale `1/d`, both value and
query/key entry standard deviations are `stdSigmaV d`. The original
effective temperature then equals the actual temperature `β`, while
the barycenter is exactly the unit-entry-variance one at `β/d`.
These equalities retain all original laws, integrals and unit queries.
-/

import Transformer.Homogenized.Section7_GaussianBarycenterBound
import Transformer.Homogenized.Metastability
import Mathlib.MeasureTheory.Group.MeasurableEquiv

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The actual original entry product law scales coordinatewise.
Source: arXiv:2604.01978v1, `eq: tformers.at.initialization`. -/
theorem gaussEnsemble_eq_map_unit_scale (d : ℕ) (c : ℝ≥0) :
    gaussEnsemble d c c = (gaussEnsemble d 1 1).map
      (fun z : GaussIdx d → ℝ => fun p => (c : ℝ) * z p) := by
  simp only [gaussEnsemble, entryVar, ite_self, one_pow]
  rw [Measure.pi_map_pi (f := fun _ : GaussIdx d => fun u : ℝ => (c : ℝ) * u)
    (fun p => (measurable_const_mul (c : ℝ)).aemeasurable)]
  congr 1
  funext p
  rw [gaussianReal_map_const_mul, mul_zero, mul_one]
  congr 1

/-- The original head pushforward retains the quadratic scaling of
the original matrix product. Source: arXiv:2604.01978v1, initialization (G). -/
theorem gaussHeadLaw_eq_map_unit_scale (d : ℕ) (c : ℝ≥0) :
    gaussHeadLaw d c c = (gaussHeadLaw d 1 1).map
      (fun θ : HeadParam d => ((c : ℝ) • θ.1, (c : ℝ) ^ 2 • θ.2)) := by
  rw [gaussHeadLaw, gaussEnsemble_eq_map_unit_scale,
    Measure.map_map (measurable_gaussHead d) (by fun_prop)]
  rw [gaussHeadLaw, Measure.map_map (by fun_prop) (measurable_gaussHead d)]
  congr 1
  funext z
  apply Prod.ext
  · ext i j
    rfl
  · ext i j
    simp only [Function.comp_apply, gaussBlock, Matrix.of_apply, Matrix.mul_apply,
      Matrix.transpose_apply, Matrix.smul_apply, smul_eq_mul]
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl (fun q hq => by ring)

/-- Actual matrix scaling is actual temperature scaling in the
original barycenter. Source: arXiv:2604.01978v1, `eq: mbetaA`. -/
theorem softBary_smul_matrix {d : ℕ} (β c : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (μ : Measure (EucSpace d)) (x : EucSpace d) :
    softBary β (c • A) μ x = softBary (β * c) A μ x := by
  have he (y : EucSpace d) : softWeight β (c • A) x y = softWeight (β * c) A x y := by
    rw [softWeight, softWeight]
    change Real.exp (β * inner ℝ ((Matrix.toEuclideanLin (c • A)) x) y) = _
    rw [map_smul, LinearMap.smul_apply, real_inner_smul_left]
    congr 1
    ring
  simp only [softBary, he]

/-- Positive entry scaling gives an exact change of variables in the
original head integral. Source: arXiv:2604.01978v1, initialization (G). -/
theorem integral_gaussHead_scaled {d : ℕ} (c : ℝ≥0) (hc : 0 < c) (F : HeadParam d → ℝ) :
    (∫ θ, F θ ∂gaussHeadLaw d c c) =
      ∫ θ, F ((c : ℝ) • θ.1, (c : ℝ) ^ 2 • θ.2) ∂gaussHeadLaw d 1 1 := by
  rw [gaussHeadLaw_eq_map_unit_scale]
  have hcR : (c : ℝ) ≠ 0 := (show (0 : ℝ) < c from hc).ne'
  let e : HeadParam d ≃ᵐ HeadParam d :=
    (MeasurableEquiv.smul₀ (c : ℝ) hcR).prodCongr
      (MeasurableEquiv.smul₀ ((c : ℝ) ^ 2) (pow_ne_zero 2 hcR))
  have he : (e : HeadParam d → HeadParam d) =
      (fun θ : HeadParam d => ((c : ℝ) • θ.1, (c : ℝ) ^ 2 • θ.2)) := by
    funext θ
    change ((MeasurableEquiv.smul₀ (c : ℝ) hcR) θ.1,
      (MeasurableEquiv.smul₀ ((c : ℝ) ^ 2) (pow_ne_zero 2 hcR)) θ.2) = _
    simp only [MeasurableEquiv.coe_smul₀]
  rw [← he, integral_map_equiv e]
  simp only [he]

/-- The original correlation rescales at temperature `βc²` exactly.
Source: arXiv:2604.01978v1, §7.3, `lemma:Laplace_method`. -/
theorem baryCorr_gaussHead_scaled {d : ℕ} (c : ℝ≥0) (hc : 0 < c) (β : ℝ)
    (μ : Measure (EucSpace d)) (x y : EucSpace d) :
    baryCorr β (gaussHeadLaw d c c) μ x y =
      baryCorr (β * (c : ℝ) ^ 2) (gaussHeadLaw d 1 1) μ x y := by
  rw [baryCorr, integral_gaussHead_scaled c hc]
  simp only [softBary_smul_matrix, baryCorr]

/-- The actual original query scales quadratically with entry scale.
Source: arXiv:2604.01978v1, initialization (G). -/
theorem qkMap_scaled_gaussianHead {d : ℕ} (c : ℝ) (θ : HeadParam d) (x : EucSpace d) :
    qkMap (c • θ.1, c ^ 2 • θ.2) x = c ^ 2 • qkMap θ x := by
  simp only [qkMap, map_smul, LinearMap.smul_apply]

/-- Normalized-query self-correlation remains exactly one under
positive entry scaling. Source: arXiv:2604.01978v1, §7.3, `lemma:Laplace_method`. -/
theorem integral_scaled_gaussianHead_normalized_query_self {d : ℕ}
    (c : ℝ≥0) (hc : 0 < c) (j : Fin d) :
    (∫ θ : HeadParam d, inner ℝ
      (normalizeLayer (qkMap θ (EuclideanSpace.single j (1 : ℝ))))
      (normalizeLayer (qkMap θ (EuclideanSpace.single j (1 : ℝ)))) ∂gaussHeadLaw d c c) = 1 := by
  rw [integral_gaussHead_scaled c hc]
  simp only [qkMap_scaled_gaussianHead,
    normalizeLayer_smul (sq_pos_of_pos (show (0 : ℝ) < c from hc))]
  exact integral_gaussianHead_normalized_query_self j

/-- The original `1/d` variance scale has positive standard deviation.
Source: arXiv:2604.01978v1, §2.3, the common Gaussian scaling. -/
theorem stdSigmaV_pos {d : ℕ} (hd : 0 < d) : 0 < stdSigmaV d := by
  change (0 : ℝ) < (stdSigmaV d : ℝ)
  have hdR : (0 : ℝ) < d := by exact_mod_cast hd
  nlinarith [NNReal.coe_nonneg (stdSigmaV d), stdSigmaV_sq d, one_div_pos.mpr hdR]

/-- At that actual query/key variance, effective and actual temperature
coincide. Source: arXiv:2604.01978v1, §7.3, `lemma:Laplace_method`. -/
theorem effBeta_stdSigmaV {d : ℕ} (hd : 0 < d) (β : ℝ) : effBeta d β (stdSigmaV d) = β := by
  have hdR : (d : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
  rw [effBeta, stdSigmaV_sq]
  field_simp

/-- The common original scaling is the exact unit-entry-variance
barycenter at temperature `β/d`. Source: arXiv:2604.01978v1,
§2.3 and §7.3, `lemma:Laplace_method`. -/
theorem baryCorr_std_gaussianHead {d : ℕ} (hd : 0 < d) (β : ℝ)
    (μ : Measure (EucSpace d)) (x y : EucSpace d) :
    baryCorr β (gaussHeadLaw d (stdSigmaV d) (stdSigmaV d)) μ x y =
      baryCorr (β / d) (gaussHeadLaw d 1 1) μ x y := by
  rw [baryCorr_gaussHead_scaled (stdSigmaV d) (stdSigmaV_pos hd), stdSigmaV_sq]
  congr 1
  ring

/-- Positive entry scale and positive dimension hold jointly for a
nondegenerate original Gaussian head; the common `1/d` scale is positive too. -/
example : (0 : ℝ≥0) < 1 ∧ (0 : ℕ) < 2 ∧
    IsGaussianHeadLaw 2 1 1 (gaussHeadLaw 2 1 1) ∧ 0 < stdSigmaV 2 :=
  ⟨one_pos, by decide, isGaussianHeadLaw_gaussHeadLaw 2 1 1, stdSigmaV_pos (by decide)⟩

end Transformer.Homogenized

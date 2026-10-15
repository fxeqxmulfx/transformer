/-
# The original uniform-law softmax barycenter and dimension

ArXiv:2604.01978v1, §7.3, `lemma:Laplace_method`, Step 1.
The displayed Laplace expansion is applied to a uniform spherical law
whose density is identically one. The estimates here retain dimension
in that exact setting and use the original exponential barycenter.

The isotropy identity from `Normalization.InitialSphereMoments` gives
a Bessel bound for the vector integral against any continuous scalar
weight: its squared norm times dimension is at most the scalar second
moment. The proof integrates an actual nonnegative square, so no
normalization or covariance identity is added to the weight definition.
The original uniform law is centered, giving an exponential normalizer
of at least one. Together these yield the explicit bound
`d ‖m‖² ≤ exp(2 ‖β A x‖)`, without any Gaussian-law hypothesis.

The ambient pushforward below satisfies precisely the manuscript's
uniform-law predicate. Every bound therefore concerns its actual
token measure, rather than a replacement finite sample or a postulated
Laplace limit. The estimate is useful when the query norm grows more
slowly than the logarithm of dimension.
-/

import Transformer.Homogenized.SlowMotion
import Transformer.Homogenized.Barycenter
import Transformer.Normalization.InitialSphereMoments
import Transformer.Metastability.InitialUniform

open Real MeasureTheory
open scoped BigOperators

namespace Transformer.Homogenized

/-- The actual spherical uniform law, viewed in ambient coordinates,
satisfies the original uniform predicate. Source: arXiv:2604.01978v1,
§2.3.3, `ass:low-temperature`. -/
theorem isUniformAmbient_map_spherical {d : ℕ} (ν : Measure (SSphere d))
    (hν : Metastability.IsUniformOn d ν) :
    IsUniformAmbient d (ν.map (Subtype.val : SSphere d → EucSpace d)) := by
  have : IsProbabilityMeasure ν := hν.1
  refine ⟨(Measure.isProbabilityMeasure_map_iff continuous_subtype_val.measurable.aemeasurable).mpr
    inferInstance, ?_, ?_⟩
  · rw [Measure.map_apply continuous_subtype_val.measurable]
    · have he : (Subtype.val : SSphere d → EucSpace d) ⁻¹'
          {y : EucSpace d | ‖y‖ = 1}ᶜ = ∅ := by
        ext x
        simp
      rw [he, measure_empty]
    · exact (isClosed_eq continuous_norm continuous_const).measurableSet.compl
  · intro U
    rw [Measure.map_map U.continuous.measurable continuous_subtype_val.measurable]
    have he : (U : EucSpace d → EucSpace d) ∘ Subtype.val =
        (Subtype.val : SSphere d → EucSpace d) ∘ Perspective.sphereMap d U := rfl
    rw [he, ← Measure.map_map continuous_subtype_val.measurable
      (Perspective.measurable_sphereMap d U), hν.2 U]

/-- Isotropy bounds the actual weighted vector integral by the original
scalar second moment, retaining the exact dimension factor.
Source: arXiv:2604.01978v1, §7.3, `lemma:Laplace_method`, Step 1;
the isotropy identity is `Normalization.integral_sphere_inner_sq`. -/
theorem norm_integral_spherical_smul_sq_le {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (f : SSphere d → ℝ) (hf : Continuous f) :
    (d : ℝ) * ‖∫ y : SSphere d, f y • (y : EucSpace d) ∂ν‖ ^ 2 ≤ ∫ y, f y ^ 2 ∂ν := by
  have : IsProbabilityMeasure ν := hν.1
  let m : EucSpace d := ∫ y : SSphere d, f y • (y : EucSpace d) ∂ν
  have hint : Integrable (fun y : SSphere d => f y • (y : EucSpace d)) ν :=
    Perspective.integrable_of_continuous_compact (hf.smul continuous_subtype_val) ν
  have hcross : (∫ y : SSphere d, f y * inner ℝ m (y : EucSpace d) ∂ν) = ‖m‖ ^ 2 := by
    calc
      _ = ∫ y : SSphere d, inner ℝ m (f y • (y : EucSpace d)) ∂ν := by
        simp only [real_inner_smul_right]
      _ = inner ℝ m m := integral_inner hint m
      _ = _ := real_inner_self_eq_norm_sq m
  have hsq : (∫ y : SSphere d, inner ℝ m (y : EucSpace d) ^ 2 ∂ν) = ‖m‖ ^ 2 / (d : ℝ) :=
    Normalization.integral_sphere_inner_sq hd ν hν.2 m
  have hp : 0 ≤ ∫ y : SSphere d, (f y - (d : ℝ) * inner ℝ m (y : EucSpace d)) ^ 2 ∂ν :=
    integral_nonneg (fun y => sq_nonneg _)
  have he : (∫ y : SSphere d, (f y - (d : ℝ) * inner ℝ m (y : EucSpace d)) ^ 2 ∂ν) =
      (∫ y, f y ^ 2 ∂ν) - 2 * d * ‖m‖ ^ 2 + (d : ℝ) ^ 2 * (‖m‖ ^ 2 / d) := by
    have heq : ∀ y : SSphere d,
        (f y - (d : ℝ) * inner ℝ m (y : EucSpace d)) ^ 2 =
        (f y ^ 2 - (2 * d) * (f y * inner ℝ m (y : EucSpace d))) +
          (d : ℝ) ^ 2 * inner ℝ m (y : EucSpace d) ^ 2 := by intro y; ring
    simp_rw [heq]
    rw [integral_add, integral_sub, integral_const_mul, integral_const_mul, hcross, hsq]
    all_goals exact Perspective.integrable_of_continuous_compact (by fun_prop) ν
  rw [he] at hp
  have hdim : (d : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
  have hcancel : (d : ℝ) ^ 2 * (‖m‖ ^ 2 / d) = d * ‖m‖ ^ 2 := by field_simp
  rw [hcancel] at hp
  change (d : ℝ) * ‖m‖ ^ 2 ≤ _
  linarith

/-- The original exponential normalizer is at least one under the
centered uniform probability law. Source: arXiv:2604.01978v1,
§7.3, the denominator in `lemma:Laplace_method`. -/
theorem one_le_spherical_exp_integral {d : ℕ}
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) (q : EucSpace d) :
    1 ≤ ∫ y : SSphere d, Real.exp (inner ℝ q (y : EucSpace d)) ∂ν := by
  have : IsProbabilityMeasure ν := hν.1
  have hc : (∫ y : SSphere d, inner ℝ q (y : EucSpace d) ∂ν) = 0 := by
    rw [integral_inner (Perspective.integrable_of_continuous_compact continuous_subtype_val ν),
      Normalization.integral_sphere_eq_zero ν hν.2, inner_zero_right]
  have h := integral_mono
    (Perspective.integrable_of_continuous_compact (by fun_prop : Continuous
      (fun y : SSphere d => inner ℝ q (y : EucSpace d) + 1)) ν)
    (Perspective.integrable_of_continuous_compact (by fun_prop : Continuous
      (fun y : SSphere d => Real.exp (inner ℝ q (y : EucSpace d)))) ν)
    (fun y => Real.add_one_le_exp _)
  rw [integral_add (Perspective.integrable_of_continuous_compact (by fun_prop) ν)
    (integrable_const 1), hc] at h
  simpa using h

/-- The actual uniform-law barycenter satisfies an explicit bound
with dimension retained. Source: arXiv:2604.01978v1, §7.3,
`lemma:Laplace_method`, Step 1. This estimate is an auxiliary inequality,
not the source's asserted dimension-free Laplace error. -/
theorem norm_softBary_uniform_spherical_sq_le {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ) (x : EucSpace d) :
    (d : ℝ) * ‖softBary β A (ν.map (Subtype.val : SSphere d → EucSpace d)) x‖ ^ 2 ≤
      Real.exp (2 * ‖β • Matrix.toEuclideanLin A x‖) := by
  have : IsProbabilityMeasure ν := hν.1
  let q : EucSpace d := β • Matrix.toEuclideanLin A x
  let f : SSphere d → ℝ := fun y => Real.exp (inner ℝ q (y : EucSpace d))
  let Z : ℝ := ∫ y, f y ∂ν
  let m : EucSpace d := ∫ y : SSphere d, f y • (y : EucSpace d) ∂ν
  have hf : Continuous f := by fun_prop
  have hZ : 1 ≤ Z := one_le_spherical_exp_integral ν hν q
  have hm : (d : ℝ) * ‖m‖ ^ 2 ≤ Real.exp (2 * ‖q‖) := by
    apply (norm_integral_spherical_smul_sq_le hd ν hν f hf).trans
    have hf2 : Integrable (fun y => f y ^ 2) ν :=
      Perspective.integrable_of_continuous_compact (hf.pow 2) ν
    calc
      (∫ y, f y ^ 2 ∂ν) ≤ ∫ _y : SSphere d, Real.exp (2 * ‖q‖) ∂ν := by
        apply integral_mono hf2 (integrable_const _)
        intro y
        dsimp only [f]
        rw [← Real.exp_nat_mul]
        apply Real.exp_le_exp.mpr
        have hi := real_inner_le_norm q (y : EucSpace d)
        rw [mem_sphere_zero_iff_norm.mp y.2, mul_one] at hi
        norm_num
        linarith
      _ = _ := by simp
  have hvalue : softBary β A (ν.map (Subtype.val : SSphere d → EucSpace d)) x = Z⁻¹ • m := by
    rw [softBary]
    rw [integral_map continuous_subtype_val.measurable.aemeasurable
        (by dsimp only [softWeight]; fun_prop : Continuous
          (fun y : EucSpace d => softWeight β A x y)).aestronglyMeasurable,
      integral_map continuous_subtype_val.measurable.aemeasurable
        (by dsimp only [softWeight]; fun_prop : Continuous
          (fun y : EucSpace d => softWeight β A x y • y)).aestronglyMeasurable]
    simp only [Z, m, f, q, softWeight, real_inner_smul_left]
  have hi : 0 ≤ Z⁻¹ := inv_nonneg.mpr (zero_le_one.trans hZ)
  have hi1 : Z⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hZ
  rw [hvalue, norm_smul, Real.norm_of_nonneg hi, mul_pow]
  calc
    (d : ℝ) * ((Z⁻¹) ^ 2 * ‖m‖ ^ 2) ≤ (d : ℝ) * ‖m‖ ^ 2 := by
      have hs : (Z⁻¹) ^ 2 ≤ 1 := by nlinarith
      exact mul_le_mul_of_nonneg_left
        ((mul_le_mul_of_nonneg_right hs (sq_nonneg ‖m‖)).trans_eq (one_mul _)) (Nat.cast_nonneg d)
    _ ≤ _ := hm

/-- Positive dimension, genuine uniform probability, full isometry
invariance and a continuous nonzero weight hold jointly for the actual
two-point sphere law. This includes the ambient pushforward and both
normalizer and vector-integral hypotheses. -/
example : (0 : ℕ) < 1 ∧ Metastability.IsUniformOn 1 Normalization.oneDimUniform ∧
    Continuous (fun _ : SSphere 1 => (1 : ℝ)) :=
  ⟨by decide, ⟨inferInstance, Normalization.oneDimUniform_invariant⟩, continuous_const⟩

end Transformer.Homogenized

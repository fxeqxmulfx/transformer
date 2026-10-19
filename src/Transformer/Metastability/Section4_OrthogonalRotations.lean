module

public import Transformer.Metastability.Section4_UniformLaw
public import Mathlib.Analysis.InnerProductSpace.Projection.Reflection
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# Actual coordinate-plane rotations of the original uniform law

ArXiv:2410.06833v1, §4, `prop: concentration unif`, and
arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
The projection estimate used there can be obtained from the
uniform law's actual polynomial moments. Its moment recurrence
uses actual orthogonal rotations in a prescribed two-dimensional
plane, leaving the orthogonal complement fixed.

Two original Householder reflections give a bundled linear isometry
at every real parameter. The moving normal is `cos(t)u+sin(t)v`;
for orthonormal `u,v` it has norm one. The two coordinates rotate
by angle `2t`, with the actual cosine and sine coefficients below.
Every original uniform law therefore preserves the actual power
integrals along these rotations. No moment value is assumed.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Metastability

/-- The original moving unit normal in the prescribed coordinate plane.
Source: arXiv:2410.06833v1, §4, `prop: concentration unif`,
the orthogonal invariance used to estimate projections. -/
noncomputable def rotationNormal {d : ℕ} (u v : EucSpace d) (t : ℝ) : EucSpace d :=
  Real.cos t • u + Real.sin t • v

/-- Two actual reflections give an actual bundled orthogonal rotation.
Source: arXiv:2410.06833v1, §4, the original rotation-invariant law.
Its coordinate angle is `2t` when the two normals are orthonormal. -/
noncomputable def planeRotation {d : ℕ} (u v : EucSpace d) (t : ℝ) :
    EucSpace d ≃ₗᵢ[ℝ] EucSpace d :=
  ((ℝ ∙ u)ᗮ.reflection).trans ((ℝ ∙ rotationNormal u v t)ᗮ.reflection)

/-- The moving original normal has unit norm at every real parameter.
Source: arXiv:2410.06833v1, §4, orthogonal invariance in the
projection estimate preceding `eq: upto-t`. -/
theorem norm_rotationNormal {d : ℕ} {u v : EucSpace d} (hu : ‖u‖ = 1) (hv : ‖v‖ = 1)
    (huv : inner (𝕜 := ℝ) u v = 0) (t : ℝ) : ‖rotationNormal u v t‖ = 1 := by
  have hs : ‖rotationNormal u v t‖ ^ 2 = 1 := by
    rw [rotationNormal, norm_add_sq_real, norm_smul, norm_smul,
      real_inner_smul_left, real_inner_smul_right, huv, mul_zero, mul_zero,
      hu, hv, mul_one, mul_one, Real.norm_eq_abs, Real.norm_eq_abs, sq_abs, sq_abs]
    nlinarith [Real.sin_sq_add_cos_sq t]
  nlinarith [norm_nonneg (rotationNormal u v t)]

/-- Reflection perpendicular to an actual unit normal has the original
Householder formula. Source: arXiv:2410.06833v1, §4, the actual
orthogonal maps preserving the uniform initialization law. -/
theorem reflection_unit_apply {d : ℕ} {u : EucSpace d} (hu : ‖u‖ = 1) (z : EucSpace d) :
    (ℝ ∙ u)ᗮ.reflection z = z - (2 * inner (𝕜 := ℝ) u z) • u := by
  rw [Submodule.reflection_orthogonal_apply, Submodule.reflection_singleton_apply,
    hu]
  simp only [RCLike.ofReal_real_eq_id, id_eq, one_pow, div_one, neg_sub, mul_smul]
  simp only [two_smul]

/-- The actual two-reflection orbit is written in its original normals.
Source: arXiv:2410.06833v1, §4, the orthogonal projection-moment computation. -/
theorem planeRotation_apply {d : ℕ} {u v : EucSpace d} (hu : ‖u‖ = 1) (hv : ‖v‖ = 1)
    (huv : inner (𝕜 := ℝ) u v = 0) (t : ℝ) (z : EucSpace d) :
    planeRotation u v t z = (z - (2 * inner (𝕜 := ℝ) u z) • u) -
      (2 * inner (𝕜 := ℝ) (rotationNormal u v t)
        (z - (2 * inner (𝕜 := ℝ) u z) • u)) • rotationNormal u v t := by
  rw [planeRotation, LinearIsometryEquiv.trans_apply,
    reflection_unit_apply (norm_rotationNormal hu hv huv t), reflection_unit_apply hu]

/-- The first original coordinate rotates by the actual angle `2t`.
Source: arXiv:2410.06833v1, §4, the moments used in the original
concentration estimate. Both original coordinate vectors are retained. -/
theorem inner_planeRotation_left {d : ℕ} {u v : EucSpace d} (hu : ‖u‖ = 1) (hv : ‖v‖ = 1)
    (huv : inner (𝕜 := ℝ) u v = 0) (t : ℝ) (z : EucSpace d) :
    inner (𝕜 := ℝ) u (planeRotation u v t z) =
      Real.cos (2 * t) * inner (𝕜 := ℝ) u z - Real.sin (2 * t) * inner (𝕜 := ℝ) v z := by
  have huu : inner (𝕜 := ℝ) u u = 1 := by rw [real_inner_self_eq_norm_sq, hu, one_pow]
  have hvu : inner (𝕜 := ℝ) v u = 0 := by rw [real_inner_comm, huv]
  rw [planeRotation_apply hu hv huv]
  simp only [rotationNormal, inner_sub_right, inner_add_left, inner_add_right,
    real_inner_smul_left, real_inner_smul_right, huu, huv, hvu, mul_zero,
    mul_one, add_zero, Real.cos_two_mul, Real.sin_two_mul]
  ring

/-- The second original coordinate has the same actual cosine and sine
rotation. Source: arXiv:2410.06833v1, §4, the original uniform
projection moments used before `eq: upto-t`. -/
theorem inner_planeRotation_right {d : ℕ} {u v : EucSpace d} (hu : ‖u‖ = 1) (hv : ‖v‖ = 1)
    (huv : inner (𝕜 := ℝ) u v = 0) (t : ℝ) (z : EucSpace d) :
    inner (𝕜 := ℝ) v (planeRotation u v t z) =
      Real.sin (2 * t) * inner (𝕜 := ℝ) u z + Real.cos (2 * t) * inner (𝕜 := ℝ) v z := by
  have hvv : inner (𝕜 := ℝ) v v = 1 := by rw [real_inner_self_eq_norm_sq, hv, one_pow]
  have huu : inner (𝕜 := ℝ) u u = 1 := by rw [real_inner_self_eq_norm_sq, hu, one_pow]
  have hvu : inner (𝕜 := ℝ) v u = 0 := by rw [real_inner_comm, huv]
  have hsin : Real.sin t ^ 2 = 1 - Real.cos t ^ 2 := by nlinarith [Real.sin_sq_add_cos_sq t]
  rw [planeRotation_apply hu hv huv]
  simp only [rotationNormal, inner_sub_right, inner_add_left, inner_add_right,
    real_inner_smul_left, real_inner_smul_right, huu, hvv, hvu, mul_zero,
    mul_one, add_zero, zero_add, Real.cos_two_mul, Real.sin_two_mul]
  ring_nf
  rw [hsin]
  ring

/-- The orthogonal complement of the actual coordinate plane is fixed.
Source: arXiv:2410.06833v1, §4, the projection-moment recurrence
derived by rotating an included and an excluded coordinate. -/
theorem planeRotation_fix_of_perp {d : ℕ} {u v z : EucSpace d}
    (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) (huv : inner (𝕜 := ℝ) u v = 0)
    (huz : inner (𝕜 := ℝ) u z = 0) (hvz : inner (𝕜 := ℝ) v z = 0) (t : ℝ) :
    planeRotation u v t z = z := by
  rw [planeRotation_apply hu hv huv, huz, mul_zero, zero_smul, sub_zero]
  have hn : inner (𝕜 := ℝ) (rotationNormal u v t) z = 0 := by
    simp only [rotationNormal, inner_add_left, real_inner_smul_left, huz, hvz,
      mul_zero, add_zero]
  rw [hn, mul_zero, zero_smul, sub_zero]

/-- The actual rotation starts at the actual identity.
Source: arXiv:2410.06833v1, §4, differentiating original moments
at the identity in the projection estimate. -/
theorem planeRotation_zero_apply {d : ℕ} (u v z : EucSpace d) :
    planeRotation u v 0 z = z := by
  simp only [planeRotation, rotationNormal, Real.cos_zero, Real.sin_zero,
    one_smul, zero_smul, add_zero, LinearIsometryEquiv.trans_apply]
  exact Submodule.reflection_involutive ((ℝ ∙ u)ᗮ) z

/-- Every original uniform law preserves the actual coordinate power
integral along the actual rotations. Source: arXiv:2410.06833v1,
§4, the rotational invariance underlying `prop: concentration unif`. -/
theorem integral_inner_pow_planeRotation_eq {d : ℕ} (ν : Measure (SSphere d))
    (hν : IsUniformOn d ν) (u v : EucSpace d) (t : ℝ) (n : ℕ) :
    (∫ x : SSphere d, inner (𝕜 := ℝ) u (planeRotation u v t (x : EucSpace d)) ^ n ∂ν) =
      ∫ x : SSphere d, inner (𝕜 := ℝ) u (x : EucSpace d) ^ n ∂ν := by
  let f : SSphere d → ℝ := fun x => inner (𝕜 := ℝ) u (x : EucSpace d) ^ n
  have hf : Continuous f := (continuous_const.inner continuous_subtype_val).pow n
  have h := integral_map (μ := ν) (Perspective.measurable_sphereMap d (planeRotation u v t)).aemeasurable
    hf.aestronglyMeasurable
  rw [hν.2 (planeRotation u v t)] at h
  exact h.symm

/-- The actual uniform circle law and its two coordinate axes satisfy
every uniformity, unit-normal, orthogonality and perpendicular-vector
hypothesis of the rotation identities. Source: arXiv:2410.06833v1, §4. -/
example : IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    ‖(EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2)‖ = 1 ∧
    ‖(EuclideanSpace.single (1 : Fin 2) 1 : EucSpace 2)‖ = 1 ∧
    inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2)
      (EuclideanSpace.single (1 : Fin 2) 1) = 0 ∧
    inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2) 0 = 0 ∧
    inner (𝕜 := ℝ) (EuclideanSpace.single (1 : Fin 2) 1 : EucSpace 2) 0 = 0 := by
  refine ⟨isUniformOn_uniformLaw (by norm_num), by simp, by simp, ?_, by simp, by simp⟩
  simp [EuclideanSpace.inner_single_left]

end Transformer.Metastability

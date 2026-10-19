module

public import Transformer.Perceptron.Section6_AmbientGradient
public import Transformer.Perceptron.Transform
public import Transformer.Metastability.Section4_UniformLaw
public import Mathlib.Analysis.InnerProductSpace.Projection.Reflection

/-!
# Orthogonal invariance of the original uniform interaction

ArXiv:2601.21366v2, §5.1, `eq:FH-basic` and `lem: quadpol`.
The degree-zero harmonic is constant: the original uniform law's
exponential transform is constant on the original sphere. This step
is proved directly from the source's orthogonal-invariance predicate,
without assuming spherical harmonic eigenvalues or a polynomial inverse.

The actual ambient interaction weight is invariant under every original
orthogonal map. Its actual vector integral is equivariant under that
map. Reflection in the line through a unit point fixes the point and
therefore fixes that vector integral. The reflection's fixed space is
that same line, so its original tangent projection vanishes.

The resulting original scalar attention field vanishes on the uniform
law at every nonzero temperature, including the source's positive regime.
This is the actual stationary uniform example needed for the paper's
linear-activation density statement, before any inverse-transform argument.
The original measure, exponential kernel and projected field are retained.
The constructed uniform circle law supplies the complete joint witness.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- Every orthogonal change of the original ambient argument preserves
the uniform law's actual interaction weight. Source: arXiv:2601.21366v2,
§5.1, `eq:FH-basic`, the degree-zero part of `lem: quadpol`.
The original pushforward identity is used before restricting to the sphere. -/
theorem attentionWeightMap_isometry_of_uniform {d : ℕ} (β : ℝ)
    (μ : Perspective.ProbSphere d) (hμ : Metastability.IsUniformOn d (μ : Measure (SSphere d)))
    (x : EucSpace d) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    attentionWeightMap (β • LinearMap.id) μ (U x) =
      attentionWeightMap (β • LinearMap.id) μ x := by
  let f : SSphere d → ℝ := fun y => Real.exp (inner (𝕜 := ℝ) (U x) (β • (y : EucSpace d)))
  have hf : Continuous f := Real.continuous_exp.comp
    (continuous_const.inner (continuous_subtype_val.const_smul β))
  calc
    _ = ∫ y, f y ∂(μ : Measure (SSphere d)).map (Perspective.sphereMap d U) := by
      rw [hμ.2 U]
      rfl
    _ = ∫ y, f (Perspective.sphereMap d U y) ∂(μ : Measure (SSphere d)) :=
      integral_map (Perspective.continuous_sphereMap d U).measurable.aemeasurable
        hf.aestronglyMeasurable
    _ = attentionWeightMap (β • LinearMap.id) μ x := by
      apply integral_congr_ae
      exact ae_of_all _ fun y => by
        simp only [f, Perspective.sphereMap, LinearMap.smul_apply, LinearMap.id_apply,
          ← U.map_smul, U.inner_map_map]

/-- The original uniform vector integral transforms by the original
orthogonal map. Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`.
The identity is for the actual vector-valued exponential integral, with
the same original probability law; it is not an assumed gradient symmetry. -/
theorem ambientAttentionGradMap_isometry_of_uniform {d : ℕ} (β : ℝ)
    (μ : Perspective.ProbSphere d) (hμ : Metastability.IsUniformOn d (μ : Measure (SSphere d)))
    (x : EucSpace d) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    ambientAttentionGradMap (β • LinearMap.id) μ (U x) =
      U (ambientAttentionGradMap (β • LinearMap.id) μ x) := by
  let f : SSphere d → EucSpace d := fun y =>
    Real.exp (inner (𝕜 := ℝ) (U x) (β • (y : EucSpace d))) • (β • (y : EucSpace d))
  have hf : Continuous f := (Real.continuous_exp.comp
    (continuous_const.inner (continuous_subtype_val.const_smul β))).smul
      (continuous_subtype_val.const_smul β)
  calc
    _ = ∫ y, f y ∂(μ : Measure (SSphere d)).map (Perspective.sphereMap d U) := by
      rw [hμ.2 U]
      rfl
    _ = ∫ y, f (Perspective.sphereMap d U y) ∂(μ : Measure (SSphere d)) :=
      integral_map (Perspective.continuous_sphereMap d U).measurable.aemeasurable
        hf.aestronglyMeasurable
    _ = ∫ y, U (Real.exp (inner (𝕜 := ℝ) x (β • (y : EucSpace d))) •
        (β • (y : EucSpace d))) ∂(μ : Measure (SSphere d)) := by
      apply integral_congr_ae
      exact ae_of_all _ fun y => by
        simp only [f, Perspective.sphereMap, ← U.map_smul, U.inner_map_map]
    _ = U (ambientAttentionGradMap (β • LinearMap.id) μ x) := by
      simpa only [ambientAttentionGradMap, LinearMap.smul_apply, LinearMap.id_apply,
        LinearIsometryEquiv.coe_toLinearIsometry] using
        U.toLinearIsometry.integral_comp_comm
          (fun y : SSphere d => Real.exp (inner (𝕜 := ℝ) x (β • (y : EucSpace d))) •
            (β • (y : EucSpace d)))

/-- The actual uniform interaction has zero original projected matrix
field at every original unit point. Source: arXiv:2601.21366v2,
§5.1, the constant degree-zero harmonic in `eq:FH-basic`.
The reflection fixing the point puts the actual ambient vector in its
radial line, which the original tangent projection kills. -/
theorem attentionGradMap_zero_of_uniform {d : ℕ} (β : ℝ)
    (μ : Perspective.ProbSphere d) (hμ : Metastability.IsUniformOn d (μ : Measure (SSphere d)))
    (x : SSphere d) : attentionGradMap (β • LinearMap.id) μ (x : EucSpace d) = 0 := by
  let K : Submodule ℝ (EucSpace d) := ℝ ∙ (x : EucSpace d)
  have hfix : K.reflection (x : EucSpace d) = (x : EucSpace d) :=
    Submodule.reflection_mem_subspace_eq_self (Submodule.mem_span_singleton_self _)
  have hG := ambientAttentionGradMap_isometry_of_uniform β μ hμ (x : EucSpace d) K.reflection
  rw [hfix] at hG
  have hmem : ambientAttentionGradMap (β • LinearMap.id) μ (x : EucSpace d) ∈ K :=
    (Submodule.reflection_eq_self_iff _).mp hG.symm
  obtain ⟨c, hc⟩ := Submodule.mem_span_singleton.mp hmem
  rw [attentionGradMap_eq_proj_ambientGradMap, ← hc]
  exact proj_smul_self (norm_eq_of_mem_sphere x) c

/-- The actual scalar attention field vanishes on the original uniform
law for every nonzero temperature. Source: arXiv:2601.21366v2,
§5.1, `eq:FH-basic`, and §2.3, the original stationarity field.
The source fixes positive temperature; the matrix identity also permits
negative nonzero temperature without changing the original law. -/
theorem attentionGrad_zero_of_uniform {d : ℕ} (β : ℝ) (hβ : β ≠ 0)
    (μ : Perspective.ProbSphere d) (hμ : Metastability.IsUniformOn d (μ : Measure (SSphere d)))
    (x : SSphere d) : attentionGrad β μ (x : EucSpace d) = 0 := by
  have hz := attentionGradMap_zero_of_uniform β μ hμ x
  rw [attentionGradMap_smul_id] at hz
  exact (smul_eq_zero.mp hz).resolve_left hβ

/-- The original uniform law's exponential transform is constant on
the original sphere. Source: arXiv:2601.21366v2, §5.1,
`eq:FH-basic`, the degree-zero case of `lem: quadpol`.
The actual reflection exchanging two equal-norm points supplies transitivity,
including in dimension one. No eigenvalue is supplied as an assumption. -/
theorem attentionTransform_eq_of_uniform {d : ℕ} (β : ℝ)
    (μ : Perspective.ProbSphere d) (hμ : Metastability.IsUniformOn d (μ : Measure (SSphere d)))
    (x y : SSphere d) : attentionTransform β μ x = attentionTransform β μ y := by
  let U := (ℝ ∙ ((x : EucSpace d) - (y : EucSpace d)))ᗮ.reflection
  have hxy : U (x : EucSpace d) = (y : EucSpace d) :=
    Submodule.reflection_sub ((norm_eq_of_mem_sphere x).trans (norm_eq_of_mem_sphere y).symm)
  have h := attentionWeightMap_isometry_of_uniform β μ hμ (x : EucSpace d) U
  rw [hxy] at h
  change attentionTransformMap (β • LinearMap.id) μ y =
    attentionTransformMap (β • LinearMap.id) μ x at h
  simpa only [attentionTransformMap_smul_id] using h.symm

/-- Uniformity, a nonzero original temperature, an actual orthogonal map
and actual sphere points hold jointly for the constructed circle law.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` and `eq:FH-basic`.
The law is a genuine probability on the original nonempty sphere. -/
example : ∃ μ : Perspective.ProbSphere 2,
    Metastability.IsUniformOn 2 (μ : Measure (SSphere 2)) ∧ (1 : ℝ) ≠ 0 ∧
    ∃ x y : SSphere 2, ∃ U : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2,
      ‖(x : EucSpace 2)‖ = 1 ∧ U (x : EucSpace 2) = (y : EucSpace 2) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  exact ⟨μ, hν, one_ne_zero, basePoint 1, basePoint 1, LinearIsometryEquiv.refl ℝ _,
    norm_eq_of_mem_sphere _, rfl⟩

end Transformer.Perceptron

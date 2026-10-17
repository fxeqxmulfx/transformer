/-
# The uniform law on the sphere is unique

`Perspective.UniformTuple` describes the uniform law `σ_d` on `𝕊^{d-1}` by its
rotation invariance instead of constructing it.  This file shows that the
description determines it: two rotation-invariant probability measures on the
sphere coincide (`eq_of_rotation_invariant`), as the docstring of `sphereMap`
asserts.

The proof compares characteristic functions in `ℝ^d`.  Invariance makes the
characteristic function `φ_σ` of an invariant `σ` radial: `φ_σ(ξ) = φ_σ(‖ξ‖ y)`
for every unit vector `y` (`charFun_map_coe_norm_smul`).  Averaging over
`y ∼ τ` and exchanging the integrals, with `⟨x, ‖ξ‖ y⟩ = ⟨y, ‖ξ‖ x⟩`, turns
`φ_σ(ξ)` into `φ_τ(ξ)`; a finite measure is determined by its characteristic
function.  So every uniform tuple law is the product of shell laws
(`eq_pi_shellLaw`).

Source: arXiv:2312.10794v5, §4 (the uniform law `σ_d` of `p:beta0`).
-/

import Transformer.Perspective.UniformShell
import Mathlib.MeasureTheory.Measure.CharacteristicFunction.Basic

open MeasureTheory Complex

namespace Transformer
namespace Perspective

variable {d : ℕ}

/-- The characteristic function of a measure on the sphere, seen in `ℝ^d`. -/
theorem charFun_map_coe (σ : Measure (SSphere d)) (ξ : EucSpace d) :
    charFun (σ.map ((↑) : SSphere d → EucSpace d)) ξ
      = ∫ x : SSphere d, exp (inner ℝ (x : EucSpace d) ξ * I) ∂σ := by
  rw [charFun_apply, integral_map continuous_subtype_val.aemeasurable
    (Continuous.aestronglyMeasurable (by fun_prop))]

/-- **The characteristic function of a rotation-invariant measure on the sphere
is rotation invariant.** -/
theorem charFun_map_coe_isometry (σ : Measure (SSphere d))
    (hσ : ∀ U, σ.map (sphereMap d U) = σ) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (ξ : EucSpace d) :
    charFun (σ.map ((↑) : SSphere d → EucSpace d)) (U ξ)
      = charFun (σ.map ((↑) : SSphere d → EucSpace d)) ξ := by
  rw [charFun_map_coe, charFun_map_coe]
  conv_rhs => rw [← hσ U.symm]
  rw [integral_map (measurable_sphereMap d U.symm).aemeasurable
    (Continuous.aestronglyMeasurable (by fun_prop))]
  congr 1
  funext x
  have h := U.symm.inner_map_map (x : EucSpace d) (U ξ)
  rw [U.symm_apply_apply] at h
  simp only [sphereMap, h]

/-- The hypothesis of `charFun_map_coe_isometry` holds for the zero measure. -/
example : ∀ U, (0 : Measure (SSphere 2)).map (sphereMap 2 U) = 0 :=
  fun _ => Measure.map_zero _

/-- **The characteristic function of a rotation-invariant measure on the sphere
is radial:** `φ(ξ) = φ(‖ξ‖ y)` for every unit vector `y`, a rotation carrying
`ξ / ‖ξ‖` to `y`. -/
theorem charFun_map_coe_norm_smul (σ : Measure (SSphere d))
    (hσ : ∀ U, σ.map (sphereMap d U) = σ) (ξ : EucSpace d) (y : SSphere d) :
    charFun (σ.map ((↑) : SSphere d → EucSpace d)) (‖ξ‖ • (y : EucSpace d))
      = charFun (σ.map ((↑) : SSphere d → EucSpace d)) ξ := by
  rcases eq_or_ne ξ 0 with rfl | hξ
  · simp
  have hn : ‖ξ‖ ≠ 0 := norm_ne_zero_iff.mpr hξ
  have hx : ‖ξ‖⁻¹ • ξ ∈ Metric.sphere (0 : EucSpace d) 1 := by
    rw [mem_sphere_zero_iff_norm, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hn]
  obtain ⟨U, hU⟩ := exists_sphereMap_apply_eq d ⟨_, hx⟩ y
  have hUy : (y : EucSpace d) = ‖ξ‖⁻¹ • U ξ := by
    rw [← hU]
    simp [sphereMap]
  rw [hUy, smul_smul, mul_inv_cancel₀ hn, one_smul, charFun_map_coe_isometry σ hσ]

/-- The hypothesis of `charFun_map_coe_norm_smul` holds for the zero measure. -/
example : ∀ U, (0 : Measure (SSphere 2)).map (sphereMap 2 U) = 0 :=
  fun _ => Measure.map_zero _

/-- **The uniform law on the sphere is unique:** two rotation-invariant
probability measures on `𝕊^{d-1}` coincide. -/
theorem eq_of_rotation_invariant (σ τ : Measure (SSphere d)) [IsProbabilityMeasure σ]
    [IsProbabilityMeasure τ] (hσ : ∀ U, σ.map (sphereMap d U) = σ)
    (hτ : ∀ U, τ.map (sphereMap d U) = τ) : σ = τ := by
  have hφ : charFun (σ.map ((↑) : SSphere d → EucSpace d))
      = charFun (τ.map ((↑) : SSphere d → EucSpace d)) := by
    funext ξ
    have hf : Integrable (Function.uncurry fun (x y : SSphere d) =>
        exp (inner ℝ (x : EucSpace d) (‖ξ‖ • (y : EucSpace d)) * I)) (σ.prod τ) :=
      Integrable.of_bound (Continuous.aestronglyMeasurable (by fun_prop)) 1
        (Filter.Eventually.of_forall fun p => (norm_exp_ofReal_mul_I _).le)
    calc charFun (σ.map ((↑) : SSphere d → EucSpace d)) ξ
        = ∫ y : SSphere d,
            charFun (σ.map ((↑) : SSphere d → EucSpace d)) (‖ξ‖ • (y : EucSpace d)) ∂τ := by
          simp [charFun_map_coe_norm_smul σ hσ]
      _ = ∫ y : SSphere d, ∫ x : SSphere d,
            exp (inner ℝ (x : EucSpace d) (‖ξ‖ • (y : EucSpace d)) * I) ∂σ ∂τ := by
          simp_rw [charFun_map_coe]
      _ = ∫ x : SSphere d, ∫ y : SSphere d,
            exp (inner ℝ (x : EucSpace d) (‖ξ‖ • (y : EucSpace d)) * I) ∂τ ∂σ :=
          (integral_integral_swap hf).symm
      _ = ∫ x : SSphere d, ∫ y : SSphere d,
            exp (inner ℝ (y : EucSpace d) (‖ξ‖ • (x : EucSpace d)) * I) ∂τ ∂σ := by
          simp_rw [real_inner_smul_right, real_inner_comm]
      _ = ∫ x : SSphere d,
            charFun (τ.map ((↑) : SSphere d → EucSpace d)) (‖ξ‖ • (x : EucSpace d)) ∂σ := by
          simp_rw [charFun_map_coe]
      _ = charFun (τ.map ((↑) : SSphere d → EucSpace d)) ξ := by
          simp [charFun_map_coe_norm_smul τ hτ]
  rw [← measurableEmbedding_coe_sphere.comap_map σ, ← measurableEmbedding_coe_sphere.comap_map τ,
    Measure.ext_of_charFun hφ]

/-- The hypotheses of `eq_of_rotation_invariant` hold for two copies of the shell
law on the circle. -/
example : IsProbabilityMeasure (shellLaw (d := 2) (by norm_num)) ∧
    ∀ U, (shellLaw (d := 2) (by norm_num)).map (sphereMap 2 U) = shellLaw (by norm_num) :=
  ⟨inferInstance, shellLaw_isometry _⟩

/-- **The uniform tuple law is unique:** every `UniformTuple` is the product of
shell laws (`d ≥ 1`). -/
theorem eq_pi_shellLaw (hd : 1 ≤ d) {n : ℕ} (P : Measure (SphereTuple d n))
    (hP : UniformTuple d n P) : P = Measure.pi fun _ : Idx n => shellLaw hd := by
  obtain ⟨σ, hσp, hσ, rfl⟩ := hP
  rw [eq_of_rotation_invariant σ (shellLaw hd) hσ (shellLaw_isometry hd)]

/-- The hypotheses of `eq_pi_shellLaw` hold for the product of shell laws. -/
example : UniformTuple 2 3 (Measure.pi fun _ : Idx 3 => shellLaw (d := 2) (by norm_num)) :=
  uniformTuple_shellLaw _

end Perspective
end Transformer

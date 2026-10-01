/-
# The first two moments of uniform spherical initialization

The centered and isotropic spherical law used in Appendix B, proof of
Theorem 4.2 of arXiv:2510.22026v2. Only isometry invariance is assumed.
-/

import Transformer.Perspective.SphereInvariant
import Transformer.Perspective.Section2_EnergyMoments
import Transformer.Perspective.WendelOneDimBasic
import Mathlib.MeasureTheory.Integral.Bochner.ContinuousLinearMap

open scoped BigOperators
open MeasureTheory

namespace Transformer.Normalization

/-- The uniform law on the two-point unit sphere. Source:
arXiv:2510.22026v2, §4.2, i.i.d. uniform initialization. -/
noncomputable def oneDimUniform : Measure (SSphere 1) :=
  (1 / 2 : ENNReal) • Measure.dirac Perspective.eOne +
    (1 / 2 : ENNReal) • Measure.dirac Perspective.eNeg

instance : IsProbabilityMeasure oneDimUniform := by
  constructor
  simp [oneDimUniform, ENNReal.inv_two_add_inv_two]

/-- Every isometry preserves the equally weighted two-point law. Source:
arXiv:2510.22026v2, §4.2, the uniform spherical initialization. -/
theorem oneDimUniform_invariant (U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1) :
    oneDimUniform.map (Perspective.sphereMap 1 U) = oneDimUniform := by
  have hinj : Function.Injective (Perspective.sphereMap 1 U) := by
    intro x y h
    exact Subtype.ext (U.injective (congrArg Subtype.val h))
  have hne : Perspective.sphereMap 1 U Perspective.eOne ≠
      Perspective.sphereMap 1 U Perspective.eNeg :=
    fun h => Perspective.eOne_ne_eNeg (hinj h)
  rcases Perspective.sphere_one_cases (Perspective.sphereMap 1 U Perspective.eOne) with hpos | hneg
  · have hn : Perspective.sphereMap 1 U Perspective.eNeg = Perspective.eNeg :=
      (Perspective.sphere_one_cases _).resolve_left (fun h => hne (hpos.trans h.symm))
    dsimp [oneDimUniform]
    rw [Measure.map_add _ _ (Perspective.measurable_sphereMap 1 U)]
    simp only [Measure.map_smul _ (Perspective.measurable_sphereMap 1 U).aemeasurable,
      Measure.map_dirac, hpos, hn]
  · have hp : Perspective.sphereMap 1 U Perspective.eNeg = Perspective.eOne :=
      (Perspective.sphere_one_cases _).resolve_right (fun h => hne (hneg.trans h.symm))
    dsimp [oneDimUniform]
    rw [Measure.map_add _ _ (Perspective.measurable_sphereMap 1 U)]
    simp only [Measure.map_smul _ (Perspective.measurable_sphereMap 1 U).aemeasurable,
      Measure.map_dirac, hp, hneg]
    exact add_comm _ _


variable {d : ℕ}

/-- An invariant spherical law is centered. Source: arXiv:2510.22026v2,
Appendix B, the centered initialization used in `eq:herbstpr2`. -/
theorem integral_sphere_eq_zero (σ : Measure (SSphere d)) [IsFiniteMeasure σ]
    (hσ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d,
      σ.map (Perspective.sphereMap d U) = σ) :
    (∫ x : SSphere d, (x : EucSpace d) ∂σ) = 0 := by
  have hmap := integral_map (μ := σ)
    (Perspective.measurable_sphereMap d (LinearIsometryEquiv.neg ℝ)).aemeasurable
    continuous_subtype_val.aestronglyMeasurable
  rw [hσ (LinearIsometryEquiv.neg ℝ)] at hmap
  change (∫ x : SSphere d, (x : EucSpace d) ∂σ) = ∫ x : SSphere d, -(x : EucSpace d) ∂σ
    at hmap
  rw [integral_neg] at hmap
  have h : (2 : ℝ) • (∫ x : SSphere d, (x : EucSpace d) ∂σ) = 0 := by
    rw [two_smul]
    exact eq_neg_iff_add_eq_zero.mp hmap
  exact (smul_eq_zero.mp h).resolve_left (by norm_num)

/-- The zero finite measure satisfies isometry invariance. -/
example : ∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1,
    (0 : Measure (SSphere 1)).map (Perspective.sphereMap 1 U) = 0 :=
  fun _ => Measure.map_zero _

/-- A rotation does not change a spherical directional second moment.
Source: arXiv:2510.22026v2, Appendix B, isotropy of the uniform law. -/
theorem integral_sphere_inner_sq_isometry (σ : Measure (SSphere d)) [IsFiniteMeasure σ]
    (hσ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d,
      σ.map (Perspective.sphereMap d U) = σ)
    (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) (u : EucSpace d) :
    (∫ x : SSphere d, inner (𝕜 := ℝ) (U u) (x : EucSpace d) ^ 2 ∂σ) =
      ∫ x : SSphere d, inner (𝕜 := ℝ) u (x : EucSpace d) ^ 2 ∂σ := by
  have hf : Continuous (fun x : SSphere d => inner (𝕜 := ℝ) (U u) (x : EucSpace d) ^ 2) :=
    by fun_prop
  have hmap := integral_map (μ := σ) (Perspective.measurable_sphereMap d U).aemeasurable
    hf.aestronglyMeasurable
  rw [hσ U] at hmap
  simpa only [Perspective.sphereMap, LinearIsometryEquiv.inner_map_map] using hmap

/-- A zero measure realizes the invariance hypothesis. -/
example : ∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1,
    (0 : Measure (SSphere 1)).map (Perspective.sphereMap 1 U) = 0 :=
  fun _ => Measure.map_zero _

/-- The directional second moment of a uniform unit vector is `‖u‖²/d`.
Source: arXiv:2510.22026v2, Appendix B, the variance proxy and expectation
estimate for `X_k`. -/
theorem integral_sphere_inner_sq (hd : 0 < d) (σ : Measure (SSphere d))
    [IsProbabilityMeasure σ]
    (hσ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d,
      σ.map (Perspective.sphereMap d U) = σ) (u : EucSpace d) :
    (∫ x : SSphere d, inner (𝕜 := ℝ) u (x : EucSpace d) ^ 2 ∂σ) =
      ‖u‖ ^ 2 / (d : ℝ) := by
  let e : EucSpace d := EuclideanSpace.single (⟨0, hd⟩ : Fin d) 1
  have he : ‖e‖ = 1 := by simp [e]
  let c : ℝ := ∫ x : SSphere d, inner (𝕜 := ℝ) e (x : EucSpace d) ^ 2 ∂σ
  have hunit : ∀ v : EucSpace d, ‖v‖ = 1 →
      (∫ x : SSphere d, inner (𝕜 := ℝ) v (x : EucSpace d) ^ 2 ∂σ) = c := by
    intro v hv
    obtain ⟨U, hU⟩ := Perspective.exists_sphereMap_apply_eq d
      ⟨v, mem_sphere_zero_iff_norm.mpr hv⟩ ⟨e, mem_sphere_zero_iff_norm.mpr he⟩
    have hUv : U v = e := congrArg Subtype.val hU
    simpa only [hUv] using (integral_sphere_inner_sq_isometry σ hσ U v).symm
  have hc : (d : ℝ) * c = 1 := by
    have hs : (∑ i : Fin d, ∫ x : SSphere d, (x : EucSpace d) i ^ 2 ∂σ) = 1 := by
      rw [← integral_finsetSum _ (fun i _ =>
        Perspective.integrable_of_continuous_compact (by fun_prop) σ)]
      have heq : (fun x : SSphere d => ∑ i : Fin d, (x : EucSpace d) i ^ 2) = fun _ => 1 := by
        funext x
        rw [← EuclideanSpace.real_norm_sq_eq,
          mem_sphere_zero_iff_norm.mp x.property, one_pow]
      rw [heq]
      simp
    have hi : ∀ i : Fin d, (∫ x : SSphere d, (x : EucSpace d) i ^ 2 ∂σ) = c := by
      intro i
      simpa [EuclideanSpace.inner_single_left] using
        hunit (EuclideanSpace.single i 1) (by simp)
    simpa only [hi, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] using hs
  have hd' : (d : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
  have hc' : c = (d : ℝ)⁻¹ := by
    apply mul_left_cancel₀ hd'
    rw [hc, mul_inv_cancel₀ hd']
  by_cases hu : u = 0
  · simp [hu]
  have hn : 0 < ‖u‖ := norm_pos_iff.mpr hu
  have hv : ‖‖u‖⁻¹ • u‖ = 1 := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hn), inv_mul_cancel₀ hn.ne']
  calc (∫ x : SSphere d, inner (𝕜 := ℝ) u (x : EucSpace d) ^ 2 ∂σ)
      = ‖u‖ ^ 2 * c := by
        have hs := hunit (‖u‖⁻¹ • u) hv
        simp only [real_inner_smul_left, mul_pow, integral_const_mul] at hs
        have h := congrArg (fun a : ℝ => ‖u‖ ^ 2 * a) hs
        rw [← mul_assoc, ← mul_pow, mul_inv_cancel₀ hn.ne', one_pow, one_mul] at h
        exact h
    _ = ‖u‖ ^ 2 / (d : ℝ) := by rw [hc', div_eq_mul_inv]

/-- The two-point uniform law realizes the positive dimension, probability,
and invariance hypotheses. -/
example : 0 < (1 : ℕ) ∧ IsProbabilityMeasure oneDimUniform ∧
    ∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1,
      oneDimUniform.map (Perspective.sphereMap 1 U) = oneDimUniform :=
  ⟨by decide, inferInstance, oneDimUniform_invariant⟩

end Transformer.Normalization

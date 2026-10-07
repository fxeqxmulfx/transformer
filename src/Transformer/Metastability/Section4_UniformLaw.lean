/-
# A uniform spherical law with all hypotheses exhibited

The expectation in arXiv:2410.06833v1, §4, `sec: energy.levels`, is over a
rotation-invariant probability law. We use the already constructed
`MeanField.uniformLaw`, the normalized spherical part of Lebesgue measure.
Here its invariance is proved by taking cones over measurable spherical
sets and applying the measure-preserving ambient linear isometry.

The sphere's subspace and Borel measurable spaces are equal but not
identical Lean terms. The cone argument uses the subspace instance locally;
the final statement transports the resulting set masses to the Borel law.
This constructs a witness for `IsUniformFamily` in every positive dimension.

Invariance holds under every linear isometry, including reflections, and
for every measurable set, not just spherical caps. The finite spherical
measure is normalized before transporting it. In dimension zero the same
construction gives zero mass; only the probability claim requires `d ≥ 1`.
No uniqueness theorem is needed to exhibit these hypotheses.
-/

import Transformer.MeanField.UniformLaw
import Transformer.Metastability.InitialUniform
import Transformer.Perspective.SphereInvariant

open MeasureTheory
open scoped Pointwise

namespace Transformer.Metastability
variable {d : ℕ}

section Subspace
attribute [local instance 2000] Subtype.instMeasurableSpace

/-- The part of the cone over `s` with radii in `(0,1)`.
Source: arXiv:2410.06833v1, §4, `sec: energy.levels` (uniform-law construction);
the cone-volume law is `Measure.toSphere` used in `Causal.CapMass`. -/
def sphereCone (s : Set (SSphere d)) : Set (EucSpace d) :=
  Set.Ioo (0 : ℝ) 1 • (Subtype.val '' s)

/-- Polar coordinates express the cone as an injective image of a
measurable product, allowing arbitrary measurable spherical sets.
Source: arXiv:2410.06833v1, §4, `sec: energy.levels` (auxiliary construction). -/
theorem sphereCone_radial_image (s : Set (SSphere d)) :
    sphereCone s = Subtype.val '' ((homeomorphUnitSphereProd (EucSpace d)).symm ''
      (s ×ˢ {r : Set.Ioi (0 : ℝ) | (r : ℝ) < 1})) := by
  ext z
  constructor
  · rintro ⟨r, hr, _, ⟨w, hw, rfl⟩, rfl⟩
    let p : SSphere d × Set.Ioi (0 : ℝ) := (w, ⟨r, hr.1⟩)
    refine ⟨(homeomorphUnitSphereProd (EucSpace d)).symm p, ⟨p, ⟨hw, hr.2⟩, rfl⟩, ?_⟩
    exact homeomorphUnitSphereProd_symm_apply_coe (EucSpace d) p
  · rintro ⟨_, ⟨⟨w, r⟩, ⟨hw, hr⟩, rfl⟩, rfl⟩
    refine ⟨(r : ℝ), ⟨r.2, hr⟩, _, ⟨w, hw, rfl⟩, ?_⟩
    exact (homeomorphUnitSphereProd_symm_apply_coe (EucSpace d) (w, r)).symm

/-- A measurable set on the sphere has a measurable truncated cone.
Source: arXiv:2410.06833v1, §4, `sec: energy.levels` (auxiliary construction). -/
theorem measurableSet_sphereCone {s : Set (SSphere d)} (hs : MeasurableSet s) :
    MeasurableSet (sphereCone s) := by
  rw [sphereCone_radial_image]
  apply (MeasurableEmbedding.subtype_coe
    (measurableSet_singleton (0 : EucSpace d)).compl).measurableSet_image.mpr
  apply (homeomorphUnitSphereProd (EucSpace d)).symm.measurableEmbedding.measurableSet_image.mpr
  exact hs.prod (measurableSet_lt measurable_subtype_coe measurable_const)

/-- An ambient linear isometry commutes with forming spherical cones.
Source: arXiv:2410.06833v1, §4, `sec: energy.levels` (auxiliary construction). -/
theorem sphereCone_preimage (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) (s : Set (SSphere d)) :
    sphereCone ((Perspective.sphereMap d U) ⁻¹' s) = U ⁻¹' sphereCone s := by
  ext z
  constructor
  · rintro ⟨r, hr, _, ⟨w, hw, rfl⟩, rfl⟩
    refine ⟨r, hr, _, ⟨Perspective.sphereMap d U w, hw, rfl⟩, ?_⟩
    exact (map_smul U r (w : EucSpace d)).symm
  · rintro ⟨r, hr, _, ⟨w, hw, rfl⟩, hz⟩
    refine ⟨r, hr, _, ⟨Perspective.sphereMap d U.symm w, ?_, rfl⟩, ?_⟩
    · change Perspective.sphereMap d U (Perspective.sphereMap d U.symm w) ∈ s
      have he : Perspective.sphereMap d U (Perspective.sphereMap d U.symm w) = w := by
        apply Subtype.ext
        exact U.apply_symm_apply _
      rwa [he]
    · apply U.injective
      simpa only [map_smul, Perspective.sphereMap, LinearIsometryEquiv.apply_symm_apply] using hz

/-- The spherical part of Lebesgue measure is isometry invariant.
Source: arXiv:2410.06833v1, §4, `sec: energy.levels` (the uniform law). -/
theorem map_toSphere_volume (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    (volume : Measure (EucSpace d)).toSphere.map (Perspective.sphereMap d U) =
      (volume : Measure (EucSpace d)).toSphere := by
  apply Measure.ext
  intro s hs
  have hm : Measurable (Perspective.sphereMap d U) := by
    unfold Perspective.sphereMap
    fun_prop
  rw [Measure.map_apply hm hs,
    Measure.toSphere_apply' _ (hs.preimage hm),
    Measure.toSphere_apply' _ hs]
  change _ * volume (sphereCone ((Perspective.sphereMap d U) ⁻¹' s)) =
    _ * volume (sphereCone s)
  rw [sphereCone_preimage]
  congr 1
  exact U.measurePreserving.measure_preimage (measurableSet_sphereCone hs).nullMeasurableSet

/-- Normalization preserves the spherical law's isometry invariance.
Source: arXiv:2410.06833v1, §4, `sec: energy.levels` (the uniform law). -/
theorem map_uniformSphere (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    (Causal.uniformSphere d).map (Perspective.sphereMap d U) = Causal.uniformSphere d := by
  have hm : Measurable (Perspective.sphereMap d U) := by
    unfold Perspective.sphereMap
    fun_prop
  unfold Causal.uniformSphere
  rw [Measure.map_smul _ hm.aemeasurable, map_toSphere_volume]

end Subspace

/-- The constructed Borel probability law satisfies exactly the
uniformity predicate used for the energy window in every `d ≥ 1`.
Source: arXiv:2410.06833v1, §4, `sec: energy.levels`. -/
theorem isUniformOn_uniformLaw (hd : 1 ≤ d) : IsUniformOn d (MeanField.uniformLaw d) := by
  refine ⟨MeanField.isProbabilityMeasure_uniformLaw d hd, ?_⟩
  intro U
  apply Measure.ext
  intro s hs
  rw [Measure.map_apply (Perspective.measurable_sphereMap d U) hs,
    MeanField.uniformLaw_apply, MeanField.uniformLaw_apply]
  have hs' : @MeasurableSet (SSphere d) Subtype.instMeasurableSpace s :=
    MeanField.sphereMeasurableSpace_eq_subtype d ▸ hs
  have hm : @Measurable (SSphere d) (SSphere d)
      Subtype.instMeasurableSpace Subtype.instMeasurableSpace (Perspective.sphereMap d U) := by
    unfold Perspective.sphereMap
    fun_prop
  have he := congrArg (fun μ => μ s) (map_uniformSphere U)
  rw [Measure.map_apply hm hs'] at he
  exact he

/-- A uniform family exists simultaneously in every positive dimension.
The empty sphere in dimension zero is excluded by `IsUniformFamily`.
Source: arXiv:2410.06833v1, §4, uniform initialization. -/
theorem exists_isUniformFamily :
    ∃ σ : ∀ d : ℕ, Measure (SSphere d), IsUniformFamily σ :=
  ⟨MeanField.uniformLaw, fun _ hd => isUniformOn_uniformLaw hd⟩

/-- The whole circle is measurable, witnessing `measurableSet_sphereCone`. -/
example : MeasurableSet (Set.univ : Set (SSphere 2)) := MeasurableSet.univ

/-- Dimension two is positive and has an actual invariant probability law. -/
example : 1 ≤ 2 ∧ IsUniformOn 2 (MeanField.uniformLaw 2) :=
  ⟨by norm_num, isUniformOn_uniformLaw (by norm_num)⟩

end Transformer.Metastability

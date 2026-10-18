/-
# The surface measure of the sphere, and polar coordinates

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

Mathlib's surface measure `volume.toSphere` of `𝕊^{d-1}` lives on the subspace
σ-algebra; `surfaceMeasure` carries it to the Borel σ-algebra
`sphereMeasurableSpace` of `SSphere d`, which is the same σ-algebra, though
not by definition.  On it:

* polar coordinates `x ↦ (x/‖x‖, ‖x‖)` carry Lebesgue measure on `ℝ^d ∖ 0` to
  `surfaceMeasure ⊗ r^{d-1} dr` (`measurePreserving_polar`, from
  `Measure.measurePreserving_homeomorphUnitSphereProd`);
* the surface measure is rotation invariant (`surfaceMeasure_isometry`), finite
  and nonzero, so normalized it is the uniform law (`shellLaw_eq_surfaceMeasure`,
  by `eq_of_rotation_invariant`), and the uniform tuple law is a multiple of the
  product of surface measures (`pi_shellLaw_eq`).

`Perspective.SphereNull` takes the product of `n` polar coordinates and computes
the null sets of the uniform law with Lebesgue measure on `(ℝ^d)^n`.

Source: arXiv:2312.10794v5, §4 (the uniform law `σ_d`: "the Lebesgue measure on
`𝕊^{d-1}`, normalized to be a probability measure", footnote to
`prop: existence.uniqueness.energy`).
-/

import Transformer.MeanField.UniformLaw
import Transformer.Perspective.UniformLipschitz
import Mathlib.MeasureTheory.Constructions.HaarToSphere

open MeasureTheory Set Metric
open scoped ENNReal Pointwise

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- `ℝ^d` has a nonzero vector once `d ≥ 1`. -/
theorem nontrivial_eucSpace (hd : 1 ≤ d) : Nontrivial (EucSpace d) :=
  nontrivial_of_ne (firstBasisPoint hd : EucSpace d) 0
    (ne_zero_of_mem_unit_sphere (firstBasisPoint hd))

/-- The hypothesis of `nontrivial_eucSpace` holds for `d = 3`. -/
example : 1 ≤ 3 := by norm_num

/-- The identity of `SSphere d`, from the subspace σ-algebra to the Borel one, is
measurable: the two agree (`MeanField.sphereMeasurableSpace_eq_subtype`). -/
theorem measurable_id_subtype_sphere :
    @Measurable (SSphere d) (SSphere d) Subtype.instMeasurableSpace (sphereMeasurableSpace d)
      id := fun _ hs => MeanField.sphereMeasurableSpace_eq_subtype d ▸ hs

/-- **The surface measure** `volume.toSphere` of `𝕊^{d-1}`, on the Borel
σ-algebra of `SSphere d`. -/
noncomputable def surfaceMeasure (d : ℕ) : Measure (SSphere d) :=
  @Measure.map _ _ Subtype.instMeasurableSpace _ id (volume : Measure (EucSpace d)).toSphere

/-- `surfaceMeasure d s = d · vol((0, 1) • s)`: the volume of the cone over `s`
cut at radius `1`, times `d`. -/
theorem surfaceMeasure_apply {s : Set (SSphere d)} (hs : MeasurableSet s) :
    surfaceMeasure d s = d * volume (Ioo (0 : ℝ) 1 • ((↑) '' s : Set (EucSpace d))) := by
  rw [surfaceMeasure, @Measure.map_apply _ _ Subtype.instMeasurableSpace _ _ _
      measurable_id_subtype_sphere _ hs, preimage_id,
    Measure.toSphere_apply' _ (MeanField.sphereMeasurableSpace_eq_subtype d ▸ hs),
    finrank_euclideanSpace_fin]

/-- The hypothesis of `surfaceMeasure_apply` holds for the whole sphere. -/
example : MeasurableSet (univ : Set (SSphere 3)) := MeasurableSet.univ

theorem surfaceMeasure_univ :
    surfaceMeasure d univ = (volume : Measure (EucSpace d)).toSphere univ := by
  rw [surfaceMeasure, @Measure.map_apply _ _ Subtype.instMeasurableSpace _ _ _
    measurable_id_subtype_sphere _ MeasurableSet.univ, preimage_id]

instance : IsFiniteMeasure (surfaceMeasure d) :=
  ⟨by rw [surfaceMeasure_univ]; exact measure_lt_top _ _⟩

/-- Registered directly: found through local finiteness, the instance is too large
for the products of `Perspective.SphereNull`. -/
instance : SigmaFinite (surfaceMeasure d) := IsFiniteMeasure.toSigmaFinite _

theorem surfaceMeasure_univ_ne_zero (hd : 1 ≤ d) : surfaceMeasure d univ ≠ 0 := by
  have := nontrivial_eucSpace hd
  rw [surfaceMeasure_univ]
  exact (Measure.measure_univ_ne_zero).2 (Measure.toSphere_ne_zero _)

/-- The hypothesis of `surfaceMeasure_univ_ne_zero` holds for `d = 3`. -/
example : 1 ≤ 3 := by norm_num

/-- **The surface measure is rotation invariant.** -/
theorem surfaceMeasure_isometry (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    (surfaceMeasure d).map (sphereMap d U) = surfaceMeasure d := by
  ext s hs
  rw [Measure.map_apply (measurable_sphereMap d U) hs,
    surfaceMeasure_apply (measurable_sphereMap d U hs), surfaceMeasure_apply hs]
  have hset : Ioo (0 : ℝ) 1 • ((↑) '' (sphereMap d U ⁻¹' s) : Set (EucSpace d)) =
      U ⁻¹' (Ioo (0 : ℝ) 1 • ((↑) '' s)) := by
    ext y
    simp only [Set.mem_smul, mem_image, mem_preimage]
    constructor
    · rintro ⟨r, hr, _, ⟨u, hu, rfl⟩, rfl⟩
      exact ⟨r, hr, _, ⟨_, hu, rfl⟩, by simp [sphereMap]⟩
    · rintro ⟨r, hr, _, ⟨v, hv, rfl⟩, hy⟩
      refine ⟨r, hr, _, ⟨sphereMap d U.symm v, ?_, rfl⟩, ?_⟩
      · convert hv
        exact Subtype.ext (by simp [sphereMap])
      · rw [show ((sphereMap d U.symm v : SSphere d) : EucSpace d) = U.symm v from rfl, ← map_smul,
          hy, LinearIsometryEquiv.symm_apply_apply]
  rw [hset, U.measurePreserving.measure_preimage_emb U.toHomeomorph.measurableEmbedding]

/-- **Polar coordinates on `ℝ^d ∖ 0`:** `x ↦ (x/‖x‖, ‖x‖)` carries Lebesgue
measure to `surfaceMeasure ⊗ r^{d-1} dr`.  This is
`Measure.measurePreserving_homeomorphUnitSphereProd` with the Borel σ-algebra of
`SSphere d` for the subspace one. -/
theorem measurePreserving_polar :
    MeasurePreserving (homeomorphUnitSphereProd (EucSpace d))
      ((volume : Measure (EucSpace d)).comap (↑))
      ((surfaceMeasure d).prod (Measure.volumeIoiPow (d - 1))) := by
  suffices h : ∀ m : MeasurableSpace (SSphere d), m = Subtype.instMeasurableSpace →
      @MeasurePreserving _ _ _ (@Prod.instMeasurableSpace _ _ m _)
        (homeomorphUnitSphereProd (EucSpace d)) ((volume : Measure (EucSpace d)).comap (↑))
        (@Measure.prod _ _ m _
          (@Measure.map _ _ Subtype.instMeasurableSpace m id
            (volume : Measure (EucSpace d)).toSphere)
          (Measure.volumeIoiPow (d - 1))) from
    h _ (MeanField.sphereMeasurableSpace_eq_subtype d)
  rintro m rfl
  have h := Measure.measurePreserving_homeomorphUnitSphereProd (volume : Measure (EucSpace d))
  rw [finrank_euclideanSpace_fin] at h
  rwa [Measure.map_id]

/-- **The uniform law is the normalized surface measure:** both are
rotation-invariant probability measures (`eq_of_rotation_invariant`). -/
theorem shellLaw_eq_surfaceMeasure (hd : 1 ≤ d) :
    shellLaw hd = (surfaceMeasure d univ)⁻¹ • surfaceMeasure d := by
  have : IsProbabilityMeasure ((surfaceMeasure d univ)⁻¹ • surfaceMeasure d) :=
    ⟨by rw [Measure.smul_apply, smul_eq_mul,
      ENNReal.inv_mul_cancel (surfaceMeasure_univ_ne_zero hd) (measure_ne_top _ _)]⟩
  exact eq_of_rotation_invariant _ _ (shellLaw_isometry hd) fun U => by
    rw [Measure.map_smul _ (measurable_sphereMap d U).aemeasurable, surfaceMeasure_isometry]

/-- The hypothesis of `shellLaw_eq_surfaceMeasure` holds for `d = 3`. -/
example : 1 ≤ 3 := by norm_num

/-- **The uniform tuple law is a multiple of the product of surface measures,**
by `(surfaceMeasure d univ)⁻¹ ^ n`. -/
theorem pi_shellLaw_eq (hd : 1 ≤ d) :
    Measure.pi (fun _ : Idx n => shellLaw hd) =
      (surfaceMeasure d univ)⁻¹ ^ n • Measure.pi fun _ : Idx n => surfaceMeasure d := by
  refine Measure.pi_eq fun s _ => ?_
  rw [Measure.smul_apply, Measure.pi_pi, shellLaw_eq_surfaceMeasure hd, smul_eq_mul]
  simp only [Measure.smul_apply, smul_eq_mul, Finset.prod_mul_distrib, Finset.prod_const,
    Finset.card_univ, Fintype.card_fin]

/-- The hypothesis of `pi_shellLaw_eq` holds for `d = 3`. -/
example : 1 ≤ 3 := by norm_num

end Perspective
end Transformer

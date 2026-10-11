/-
# Rotation invariance of the actual cluster law

ArXiv:2411.04551v3, §5, the rotation argument in `prop:generic-discrete`.

The reference tuple law is invariant under a common orthogonal map.
The almost-sure empirical limit hypothesis therefore holds both at an
input and at its rotation. The proved rotation identity of the actual
flow and uniqueness of limits give equivariance of the cluster map
almost everywhere. Its proved measurability then makes its pushforward
law a rotation-invariant probability law.

The reference hypotheses have an actual witness: the normalized spherical
Lebesgue law constructed in `MeanField.UniformLaw`, whose invariance is
proved in `Metastability.Section4_UniformLaw`. No unproved uniformity or
clustering claim is invoked. The latter is an explicit input throughout.
-/

import Transformer.Interpolation.Section5_EmpiricalLimits
import Transformer.Metastability.Section4_UniformLaw
import Mathlib.Dynamics.Ergodic.MeasurePreserving

open Real MeasureTheory Filter Topology

namespace Transformer.Interpolation

variable {d n : ℕ}

/-- The constructed spherical law realizes the uniform-tuple predicate.
Source: arXiv:2411.04551v3, §5, the uniform reference in
`prop:generic-discrete`; the construction is proved in
`Metastability.Section4_UniformLaw`. -/
theorem uniformTuple_uniformLaw (hd : 1 ≤ d) (n : ℕ) :
    Perspective.UniformTuple d n (Measure.pi fun _ : Idx n => MeanField.uniformLaw d) := by
  have hσ := Metastability.isUniformOn_uniformLaw hd
  exact ⟨MeanField.uniformLaw d, hσ.1, hσ.2, rfl⟩

example : 1 ≤ 3 := by norm_num

/-- A uniform tuple law has probability mass one.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem isProbabilityMeasure_of_uniformTuple (ref : Measure (SphereTuple d n))
    (href : Perspective.UniformTuple d n ref) : IsProbabilityMeasure ref := by
  obtain ⟨σ, hσ, _, rfl⟩ := href
  have := hσ
  infer_instance

example : Perspective.UniformTuple 3 1
    (Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3) :=
  uniformTuple_uniformLaw (by norm_num) 1

/-- A common rotation preserves the entire uniform tuple law.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem map_sphereTupleMap_uniform (ref : Measure (SphereTuple d n))
    (href : Perspective.UniformTuple d n ref) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    ref.map (Perspective.sphereTupleMap U) = ref := by
  obtain ⟨σ, hσ, hσinv, rfl⟩ := href
  have := hσ
  have : IsProbabilityMeasure (σ.map (Perspective.sphereMap d U)) := by
    rw [hσinv U]
    infer_instance
  change (Measure.pi fun _ : Idx n => σ).map
    (fun X i => Perspective.sphereMap d U (X i)) = Measure.pi fun _ : Idx n => σ
  rw [Measure.pi_map_pi (fun _ => (Perspective.measurable_sphereMap d U).aemeasurable)]
  simp only [hσinv U]

example : Perspective.UniformTuple 3 2
    (Measure.pi fun _ : Idx 2 => MeanField.uniformLaw 3) :=
  uniformTuple_uniformLaw (by norm_num) 2

/-- The rotation action is measurable and preserves the source's law.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem measurePreserving_sphereTupleMap (ref : Measure (SphereTuple d n))
    (href : Perspective.UniformTuple d n ref) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    MeasurePreserving (Perspective.sphereTupleMap U) ref ref :=
  ⟨Perspective.measurable_sphereTupleMap U, map_sphereTupleMap_uniform ref href U⟩

example : Perspective.UniformTuple 3 1
    (Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3) :=
  uniformTuple_uniformLaw (by norm_num) 1

/-- A full empirical-limit witness for every one-token reference law.
Source: arXiv:2411.04551v3, §5, the one-token case of
`prop:generic-discrete`; its actual trajectory is stationary by `SA`. -/
theorem empirical_saFlow_one_clusters (β : ℝ) (ref : Measure (SphereTuple d 1)) :
    ∀ᵐ X ∂ref, Tendsto
      (fun t : ℝ => Perspective.empiricalProb one_pos (Perspective.saFlow β X t))
      atTop (nhds (Perspective.diracProb d (X 0))) := by
  apply ae_of_all
  intro X
  simpa only [saFlow_one, Perspective.empiricalProb_one] using
    (tendsto_const_nhds : Tendsto (fun _ : ℝ => Perspective.diracProb d (X 0)) atTop _)

/-- **The actual cluster map is rotation equivariant almost everywhere.**
The two empirical limits are obtained on a common event of full measure.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem clusterMap_sphereTupleMap_ae (β : ℝ) (hn : 0 < n)
    (ref : Measure (SphereTuple d n)) (href : Perspective.UniformTuple d n ref)
    (f : SphereTuple d n → SSphere d)
    (hf : ∀ᵐ X ∂ref, Tendsto (fun t => Perspective.empiricalProb hn
      (Perspective.saFlow β X t)) atTop (nhds (Perspective.diracProb d (f X))))
    (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    ∀ᵐ X ∂ref, f (Perspective.sphereTupleMap U X) = Perspective.sphereMap d U (f X) := by
  have hrot := (measurePreserving_sphereTupleMap ref href U).quasiMeasurePreserving.ae hf
  filter_upwards [hf, hrot] with X hX hUX
  let i : Idx n := ⟨0, hn⟩
  have hlim := tokens_tendsto_of_empiricalProb_tendsto_dirac hn _ (f X) hX i
  have hrotlim := (Perspective.continuous_sphereMap d U).continuousAt.tendsto.comp hlim
  have hlimU := tokens_tendsto_of_empiricalProb_tendsto_dirac hn _
    (f (Perspective.sphereTupleMap U X)) hUX i
  apply tendsto_nhds_unique hlimU
  simpa only [Perspective.saFlow_sphereTupleMap, Perspective.sphereTupleMap,
    Function.comp_def] using hrotlim

/-- The nonempty tuple, uniform law and full empirical cluster hypothesis
are all exhibited by the actual one-token flow in dimension three. -/
example : (0 : ℕ) < 1 ∧ Perspective.UniformTuple 3 1
    (Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3) ∧
    ∀ᵐ X ∂(Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3),
      Tendsto (fun t : ℝ => Perspective.empiricalProb one_pos (Perspective.saFlow 0 X t))
        atTop (nhds (Perspective.diracProb 3 (X 0))) :=
  ⟨one_pos, uniformTuple_uniformLaw (by norm_num) 1, empirical_saFlow_one_clusters 0 _⟩

/-- **The pushforward law of the actual cluster map is rotation invariant.**
Its measurability is derived from the actual flow, and its equivariance
is used only on an event of full measure.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem map_clusterLaw_rotation (β : ℝ) (hβ : 0 ≤ β) (hn : 0 < n)
    (ref : Measure (SphereTuple d n)) (href : Perspective.UniformTuple d n ref)
    (f : SphereTuple d n → SSphere d)
    (hf : ∀ᵐ X ∂ref, Tendsto (fun t => Perspective.empiricalProb hn
      (Perspective.saFlow β X t)) atTop (nhds (Perspective.diracProb d (f X))))
    (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    (ref.map f).map (Perspective.sphereMap d U) = ref.map f := by
  have hmf := aemeasurable_of_empirical_saFlow_cluster β hβ hn ref f hf
  have hmfrot : AEMeasurable f (ref.map (Perspective.sphereTupleMap U)) := by
    rw [map_sphereTupleMap_uniform ref href U]
    exact hmf
  calc (ref.map f).map (Perspective.sphereMap d U) =
      ref.map (Perspective.sphereMap d U ∘ f) :=
        (Perspective.measurable_sphereMap d U).aemeasurable.map_map_of_aemeasurable hmf
    _ = ref.map (f ∘ Perspective.sphereTupleMap U) :=
      Measure.map_congr
        (Filter.EventuallyEq.symm (clusterMap_sphereTupleMap_ae β hn ref href f hf U))
    _ = (ref.map (Perspective.sphereTupleMap U)).map f :=
      (hmfrot.map_map_of_aemeasurable (Perspective.measurable_sphereTupleMap U).aemeasurable).symm
    _ = ref.map f := by rw [map_sphereTupleMap_uniform ref href U]

example : (0 : ℝ) ≤ 0 ∧ (0 : ℕ) < 1 ∧ Perspective.UniformTuple 3 1
    (Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3) ∧
    ∀ᵐ X ∂(Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3),
      Tendsto (fun t : ℝ => Perspective.empiricalProb one_pos (Perspective.saFlow 0 X t))
        atTop (nhds (Perspective.diracProb 3 (X 0))) :=
  ⟨le_rfl, one_pos, uniformTuple_uniformLaw (by norm_num) 1,
    empirical_saFlow_one_clusters 0 _⟩

end Transformer.Interpolation

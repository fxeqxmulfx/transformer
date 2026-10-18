/-
# Null sets of the uniform law are cones of Lebesgue-null sets

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

Polar coordinates `x_i = r_i u_i`, one for each of the `n` particles, identify
`(ℝ^d ∖ 0)^n` with `(𝕊^{d-1})^n × (0, ∞)^n` (`tuplePolar`) and carry Lebesgue
measure to the product of the surface measures and of `n` copies of
`r^{d-1} dr` (`measurePreserving_tuplePolar`).  The cone over a set `A` of
configurations — the tuples of nonzero vectors whose directions lie in `A`
(`sphereCone`) — is the preimage of `A × (0, ∞)^n`.  So its Lebesgue measure is
the surface measure of `A` times that of `(0, ∞)^n`, which is not `0`
(`volume_sphereCone`), and `A` is null for the uniform law exactly when its cone
is Lebesgue-null (`measure_eq_zero_iff_volume_sphereCone`), for every `A`,
measurable or not.  Hence:

* a property of the directions `x_i / ‖x_i‖` holds for Lebesgue-almost every
  `x ∈ (ℝ^d)^n` exactly when it holds `P`-almost surely (`ae_iff_ae_volume`);
* a set of tuples of nonzero vectors invariant under positive rescaling of each
  vector is Lebesgue-null exactly when its trace on `(𝕊^{d-1})^n` is `P`-null
  (`volume_eq_zero_iff_measure`).

This is the measure half of the bridge M5.3 of `landscape_plan.md`: an "almost
every initial configuration" on `(𝕊^{d-1})^n` is an "almost every initial
point" of `(ℝ^d)^n`, where `Perspective.SphereRadial` puts the flows.

Source: arXiv:2312.10794v5, §4 (the "Lebesgue almost any
`(x_i(0))_{i∈[n]} ∈ (𝕊^{d-1})^n`" of `thm: beta.tiny`) and Appendix A (the null
sets of `l:nosaddleconv`).
-/

import Transformer.Perspective.SpherePolar

open MeasureTheory Set Metric

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **The cone over a set of configurations:** the tuples of nonzero vectors of
`ℝ^d` whose directions `x_i / ‖x_i‖` form a configuration in `A`. -/
def sphereCone (hd : 1 ≤ d) (A : Set (SphereTuple d n)) : Set (Idx n → EucSpace d) :=
  {x | (∀ i, x i ≠ 0) ∧ tupleNormalize hd x ∈ A}

/-- **Polar coordinates on `(ℝ^d ∖ 0)^n`,** onto `(𝕊^{d-1})^n × (0, ∞)^n`. -/
noncomputable def tuplePolar (d n : ℕ) :
    (Idx n → ({0}ᶜ : Set (EucSpace d))) ≃ᵐ SphereTuple d n × (Idx n → Ioi (0 : ℝ)) :=
  (MeasurableEquiv.piCongrRight fun _ =>
    (homeomorphUnitSphereProd (EucSpace d)).toMeasurableEquiv).trans
    (MeasurableEquiv.arrowProdEquivProdArrow _ _ _)

/-- The angular part of `tuplePolar` is radial projection. -/
theorem tuplePolar_fst (hd : 1 ≤ d) (y : Idx n → ({0}ᶜ : Set (EucSpace d))) :
    (tuplePolar d n y).1 = tupleNormalize hd fun i => (y i : EucSpace d) := by
  funext i
  apply Subtype.ext
  exact (homeomorphUnitSphereProd_apply_fst_coe _ _).trans
    (coe_sphereNormalize hd (y i).2).symm

/-- **Polar coordinates carry Lebesgue measure on `(ℝ^d ∖ 0)^n`** to the product
of the surface measures and of `n` copies of `r^{d-1} dr`. -/
theorem measurePreserving_tuplePolar :
    MeasurePreserving (tuplePolar d n)
      (Measure.pi fun _ => (volume : Measure (EucSpace d)).comap (↑))
      ((Measure.pi fun _ => surfaceMeasure d).prod
        (Measure.pi fun _ => Measure.volumeIoiPow (d - 1))) :=
  (measurePreserving_arrowProdEquivProdArrow _ _ _ _ _).comp
    (measurePreserving_pi _ _ fun _ => measurePreserving_polar)

/-- `(0, ∞)^n` is not null for `r^{d-1} dr`: `(0, 1)` already has mass
`1 / (d - 1 + 1) > 0`. -/
theorem pi_volumeIoiPow_univ_ne_zero :
    (Measure.pi fun _ : Idx n => Measure.volumeIoiPow (d - 1)) univ ≠ 0 := by
  rw [Measure.pi_univ, Finset.prod_ne_zero_iff]
  intro i _ h
  have h1 := measure_mono_null (subset_univ (Iio ⟨1, mem_Ioi.2 one_pos⟩)) h
  rw [Measure.volumeIoiPow_apply_Iio, ENNReal.ofReal_eq_zero] at h1
  have : (0 : ℝ) < 1 ^ (d - 1 + 1) / ((d - 1 : ℕ) + 1 : ℝ) := by positivity
  linarith

/-- **The cone over `A` has the surface measure of `A` times that of `(0, ∞)^n`,**
for every `A`, measurable or not. -/
theorem volume_sphereCone (hd : 1 ≤ d) (A : Set (SphereTuple d n)) :
    volume (sphereCone hd A) = (Measure.pi fun _ => surfaceMeasure d) A *
      (Measure.pi fun _ : Idx n => Measure.volumeIoiPow (d - 1)) univ := by
  have hS : MeasurableSet ({0}ᶜ : Set (EucSpace d)) := (measurableSet_singleton 0).compl
  have hC : MeasurableSet {x : Idx n → EucSpace d | ∀ i, x i ∈ ({0}ᶜ : Set (EucSpace d))} := by
    rw [← iInter_ofPred]
    exact MeasurableSet.iInter fun i => measurable_pi_apply i hS
  have hι : MeasurableEmbedding
      fun (y : Idx n → ({0}ᶜ : Set (EucSpace d))) i => (y i : EucSpace d) :=
    (MeasurableEmbedding.subtype_coe hC).comp
      MeasurableEquiv.subtypePiEquivPi.symm.measurableEmbedding
  have : SigmaFinite ((volume : Measure (EucSpace d)).comap
      ((↑) : ({0}ᶜ : Set (EucSpace d)) → EucSpace d)) :=
    SigmaFinite.of_map _ measurePreserving_polar.measurable.aemeasurable
      (by rw [measurePreserving_polar.map_eq]; infer_instance)
  have : SigmaFinite (((volume : Measure (EucSpace d)).comap
      ((↑) : ({0}ᶜ : Set (EucSpace d)) → EucSpace d)).map
        ((↑) : ({0}ᶜ : Set (EucSpace d)) → EucSpace d)) := by
    rw [map_comap_subtype_coe hS]
    infer_instance
  have hmap : (Measure.pi fun _ : Idx n => (volume : Measure (EucSpace d)).comap (↑)).map
      (fun (y : Idx n → ({0}ᶜ : Set (EucSpace d))) i => (y i : EucSpace d)) =
      volume.restrict (univ.pi fun _ => ({0}ᶜ : Set (EucSpace d))) := by
    rw [Measure.pi_map_pi fun _ => measurable_subtype_coe.aemeasurable]
    simp_rw [map_comap_subtype_coe hS]
    rw [volume_pi, Measure.restrict_pi_pi]
  have hcone : sphereCone hd A = (fun (y : Idx n → ({0}ᶜ : Set (EucSpace d))) i =>
      (y i : EucSpace d)) '' (tuplePolar d n ⁻¹' (A ×ˢ univ)) := by
    ext x
    constructor
    · rintro ⟨hx, hxA⟩
      refine ⟨fun i => ⟨x i, hx i⟩, ?_, rfl⟩
      rw [mem_preimage, mem_prod, tuplePolar_fst hd]
      exact ⟨hxA, mem_univ _⟩
    · rintro ⟨y, hy, rfl⟩
      rw [mem_preimage, mem_prod, tuplePolar_fst hd] at hy
      exact ⟨fun i => (y i).2, hy.1⟩
  have hsub : sphereCone hd A ⊆ univ.pi fun _ => ({0}ᶜ : Set (EucSpace d)) :=
    fun x hx i _ => hx.1 i
  rw [← Measure.prod_prod, ← measurePreserving_tuplePolar.measure_preimage_equiv,
    ← hι.injective.preimage_image (tuplePolar d n ⁻¹' _), ← hι.map_apply, hmap, ← hcone,
    Measure.restrict_apply' (MeasurableSet.univ_pi fun _ => hS), inter_eq_left.2 hsub]

/-- The hypothesis of `tuplePolar_fst` and `volume_sphereCone` holds for `d = 3`. -/
example : 1 ≤ 3 := by norm_num

/-- **A set of configurations is null for the uniform law exactly when its cone
is Lebesgue-null,** for every set, measurable or not. -/
theorem measure_eq_zero_iff_volume_sphereCone (hd : 1 ≤ d) {P : Measure (SphereTuple d n)}
    (hP : UniformTuple d n P) (A : Set (SphereTuple d n)) :
    P A = 0 ↔ volume (sphereCone hd A) = 0 := by
  rw [volume_sphereCone hd, mul_eq_zero, or_iff_left pi_volumeIoiPow_univ_ne_zero,
    eq_pi_shellLaw hd P hP, pi_shellLaw_eq hd, Measure.smul_apply, smul_eq_mul, mul_eq_zero,
    or_iff_right (pow_ne_zero _ (ENNReal.inv_ne_zero.2 (measure_ne_top _ _)))]

/-- **Almost every configuration is almost every direction:** a property of the
directions `x_i / ‖x_i‖` holds for Lebesgue-almost every `x ∈ (ℝ^d)^n` exactly
when it holds `P`-almost surely.  The tuples with a zero vector are null. -/
theorem ae_iff_ae_volume (hd : 1 ≤ d) {P : Measure (SphereTuple d n)} (hP : UniformTuple d n P)
    {p : SphereTuple d n → Prop} :
    (∀ᵐ X ∂P, p X) ↔ ∀ᵐ x ∂(volume : Measure (Idx n → EucSpace d)), p (tupleNormalize hd x) := by
  have := nontrivial_eucSpace hd
  rw [ae_iff, ae_iff, measure_eq_zero_iff_volume_sphereCone hd hP]
  refine ⟨fun h => measure_mono_null
    (t := sphereCone hd {X | ¬p X} ∪ ⋃ i, Function.eval i ⁻¹' {0}) (fun x hx => ?_)
    (measure_union_null h (measure_iUnion_null fun i => ?_)),
    fun h => measure_mono_null (fun x hx => hx.2) h⟩
  · by_cases h0 : ∀ i, x i ≠ 0
    · exact Or.inl ⟨h0, hx⟩
    · obtain ⟨i, hi⟩ := not_forall.1 h0
      exact Or.inr (mem_iUnion.2 ⟨i, not_not.1 hi⟩)
  · rw [volume_pi]
    exact Measure.pi_eval_preimage_null _ (measure_singleton 0)

/-- The hypotheses of `measure_eq_zero_iff_volume_sphereCone` and
`ae_iff_ae_volume` hold for the product of shell laws on `𝕊²`. -/
example : 1 ≤ 3 ∧ UniformTuple 3 2 (Measure.pi fun _ : Idx 2 => shellLaw (d := 3) (by norm_num)) :=
  ⟨by norm_num, uniformTuple_shellLaw _⟩

/-- **Radially invariant null sets:** a set `B` of tuples of nonzero vectors,
invariant under positive rescaling of each vector, is Lebesgue-null exactly when
the configurations in it are `P`-null. -/
theorem volume_eq_zero_iff_measure (hd : 1 ≤ d) {P : Measure (SphereTuple d n)}
    (hP : UniformTuple d n P) {B : Set (Idx n → EucSpace d)} (hB0 : ∀ x ∈ B, ∀ i, x i ≠ 0)
    (hB : ∀ x ∈ B, ∀ r : Idx n → ℝ, (∀ i, 0 < r i) → (fun i => r i • x i) ∈ B) :
    volume B = 0 ↔ P {X | (fun i => (X i : EucSpace d)) ∈ B} = 0 := by
  have hcone : sphereCone hd {X | (fun i => (X i : EucSpace d)) ∈ B} = B := by
    ext x
    constructor
    · rintro ⟨hx0, hx⟩
      convert hB _ hx (fun i => ‖x i‖) fun i => norm_pos_iff.2 (hx0 i) using 1
      funext i
      show x i = ‖x i‖ • (sphereNormalize hd (x i) : EucSpace d)
      rw [coe_sphereNormalize hd (hx0 i), smul_smul,
        mul_inv_cancel₀ (norm_ne_zero_iff.2 (hx0 i)), one_smul]
    · intro hx
      refine ⟨hB0 x hx, ?_⟩
      show (fun i => (tupleNormalize hd x i : EucSpace d)) ∈ B
      convert hB x hx (fun i => ‖x i‖⁻¹) (fun i => inv_pos.2 (norm_pos_iff.2 (hB0 x hx i)))
        using 1
      funext i
      exact coe_sphereNormalize hd (hB0 x hx i)
  rw [measure_eq_zero_iff_volume_sphereCone hd hP, hcone]

/-- The hypotheses of `volume_eq_zero_iff_measure` hold for the set of all pairs of
nonzero vectors of `ℝ³`, and the product of shell laws on `𝕊²`. -/
example : UniformTuple 3 2 (Measure.pi fun _ : Idx 2 => shellLaw (d := 3) (by norm_num)) ∧
    (∀ x ∈ {x : Idx 2 → EucSpace 3 | ∀ i, x i ≠ 0}, ∀ i, x i ≠ 0) ∧
    ∀ x ∈ {x : Idx 2 → EucSpace 3 | ∀ i, x i ≠ 0}, ∀ r : Idx 2 → ℝ, (∀ i, 0 < r i) →
      (fun i => r i • x i) ∈ {x : Idx 2 → EucSpace 3 | ∀ i, x i ≠ 0} :=
  ⟨uniformTuple_shellLaw _, fun _ hx => hx, fun _ hx _ hr i => smul_ne_zero (hr i).ne' (hx i)⟩

end Perspective
end Transformer

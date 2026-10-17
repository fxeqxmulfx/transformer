/-
# Lipschitz maps preserve the null sets of the uniform law

The uniform law of `n` points on `𝕊^{d-1}` is the radial projection of a Haar
measure on `(ℝ^d)^n` restricted to the product of the shells `1 < ‖y‖ < 2`
(`exists_addHaar_eq_map`, from the uniqueness of `Perspective.SphereUnique`).  A
Lipschitz map `f` of the configurations lifts to the shells as
`Y ↦ (‖Y_k‖ f(Y/‖Y‖)_k)_k`, Lipschitz there (`lipschitzOnWith_radialLift`), and
Haar measure has the null sets of the Hausdorff measure of full dimension, which
a Lipschitz map cannot make positive (`addHaar_image_eq_zero_of_lipschitzOnWith`).
So a Lipschitz map sends a closed null set of the uniform law to a null set
(`measure_image_eq_zero_of_lipschitz`): how the proof of `l:nosaddleconv` pulls
null sets back along the flow of `e:gradfl`.

Source: arXiv:2312.10794v5, Appendix A, `l:nosaddleconv`.
-/

import Transformer.Perspective.SphereUnique
import Mathlib.MeasureTheory.Measure.Haar.Unique
import Mathlib.MeasureTheory.Measure.Hausdorff

open MeasureTheory Set
open scoped NNReal ENNReal

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **Lipschitz maps send null sets to null sets** for a Haar measure on a
finite-dimensional real normed space: Haar measure has the null sets of the
Hausdorff measure of full dimension `m`, which a `K`-Lipschitz map enlarges at
most by `K^m`. -/
theorem addHaar_image_eq_zero_of_lipschitzOnWith {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]
    (μ : Measure E) [μ.IsAddHaarMeasure] {K : ℝ≥0} {f : E → E} {s : Set E}
    (hf : LipschitzOnWith K f s) (hs : μ s = 0) : μ (f '' s) = 0 := by
  have h1 : μH[Module.finrank ℝ E] s = 0 :=
    Measure.absolutelyContinuous_isAddHaarMeasure _ μ hs
  have h2 : μH[Module.finrank ℝ E] (f '' s) = 0 :=
    le_antisymm ((hf.hausdorffMeasure_image_le (Nat.cast_nonneg _)).trans
      (by rw [h1, mul_zero])) zero_le
  exact Measure.absolutelyContinuous_isAddHaarMeasure μ _ h2

/-- The hypotheses of `addHaar_image_eq_zero_of_lipschitzOnWith` hold for the
identity of `ℝ²` and the empty set. -/
example : LipschitzOnWith 1 (id : EucSpace 2 → EucSpace 2) ∅ ∧
    (volume : Measure (EucSpace 2)) ∅ = 0 :=
  ⟨LipschitzWith.id.lipschitzOnWith, measure_empty⟩

/-- The product of the shells `1 < ‖y_k‖ < 2` in `(ℝ^d)^n`. -/
def tupleShell (d n : ℕ) : Set (Idx n → EucSpace d) := univ.pi fun _ => sphereShell d

theorem mem_tupleShell {Y : Idx n → EucSpace d} :
    Y ∈ tupleShell d n ↔ ∀ k, Y k ∈ sphereShell d := by
  simp [tupleShell]

theorem measurableSet_tupleShell : MeasurableSet (tupleShell d n) :=
  MeasurableSet.univ_pi fun _ => isOpen_sphereShell.measurableSet

/-- Radial projection of a tuple onto the sphere, point by point. -/
noncomputable def tupleNormalize (hd : 1 ≤ d) (Y : Idx n → EucSpace d) : SphereTuple d n :=
  fun k => sphereNormalize hd (Y k)

theorem measurable_tupleNormalize (hd : 1 ≤ d) :
    Measurable (tupleNormalize (n := n) hd) :=
  Measurable.of_eval fun k => (measurable_sphereNormalize hd).comp (measurable_pi_apply k)

/-- **The uniform tuple law is the radial projection of a Haar measure** on
`(ℝ^d)^n` restricted to the product of the shells: the product of the shell
laws, which is the only uniform tuple law (`eq_pi_shellLaw`). -/
theorem exists_addHaar_eq_map (hd : 1 ≤ d) (P : Measure (SphereTuple d n))
    (hP : UniformTuple d n P) :
    ∃ μ : Measure (Idx n → EucSpace d), μ.IsAddHaarMeasure ∧
      P = (μ.restrict (tupleShell d n)).map (tupleNormalize hd) := by
  have : ((volume (sphereShell d))⁻¹ • (volume : Measure (EucSpace d))).IsAddHaarMeasure :=
    Measure.IsAddHaarMeasure.smul _ (ENNReal.inv_ne_zero.mpr volume_sphereShell_lt_top.ne)
      (ENNReal.inv_ne_top.mpr (volume_sphereShell_pos hd).ne')
  have hlaw : shellLaw hd = (((volume (sphereShell d))⁻¹ • volume).restrict
      (sphereShell d)).map (sphereNormalize hd) := by
    rw [shellLaw, shellMeasure, Measure.restrict_smul,
      Measure.map_smul _ (measurable_sphereNormalize hd).aemeasurable]
  have : SigmaFinite ((((volume (sphereShell d))⁻¹ • (volume : Measure (EucSpace d))).restrict
      (sphereShell d)).map (sphereNormalize hd)) := by
    rw [← hlaw]
    infer_instance
  refine ⟨Measure.pi fun _ => (volume (sphereShell d))⁻¹ • volume, inferInstance, ?_⟩
  rw [eq_pi_shellLaw hd P hP, tupleShell, Measure.restrict_pi_pi,
    show tupleNormalize (n := n) hd = fun Y k => sphereNormalize hd (Y k) from rfl,
    Measure.pi_map_pi fun _ => (measurable_sphereNormalize hd).aemeasurable, hlaw]

/-- The hypotheses of `exists_addHaar_eq_map` hold for the product of shell laws
on the circle. -/
example : UniformTuple 2 3 (Measure.pi fun _ : Idx 3 => shellLaw (d := 2) (by norm_num)) :=
  uniformTuple_shellLaw _

/-- **The radial lift of a map of configurations:** `Y ↦ (‖Y_k‖ f(Y/‖Y‖)_k)_k`,
which keeps the radii and moves the directions by `f`. -/
noncomputable def radialLift (hd : 1 ≤ d) (f : SphereTuple d n → SphereTuple d n)
    (Y : Idx n → EucSpace d) : Idx n → EucSpace d :=
  fun k => ‖Y k‖ • (f (tupleNormalize hd Y) k : EucSpace d)

/-- **The radial lift of a `K`-Lipschitz map is `(1 + 4K)`-Lipschitz on the
shells.** -/
theorem lipschitzOnWith_radialLift (hd : 1 ≤ d) {K : ℝ≥0}
    {f : SphereTuple d n → SphereTuple d n} (hf : LipschitzWith K f) :
    LipschitzOnWith (1 + 4 * K) (radialLift hd f) (tupleShell d n) := by
  refine LipschitzOnWith.of_dist_le_mul fun Y hY Y' hY' => ?_
  rw [mem_tupleShell] at hY hY'
  have hG : dist (tupleNormalize hd Y) (tupleNormalize hd Y') ≤ 2 * dist Y Y' :=
    (dist_pi_le_iff (by positivity)).mpr fun k =>
      (dist_sphereNormalize_le hd (hY k).1.le (hY' k).1.le).trans
        (mul_le_mul_of_nonneg_left (dist_le_pi_dist Y Y' k) zero_le_two)
  have hF := hf.dist_le_mul (tupleNormalize hd Y) (tupleNormalize hd Y')
  refine (dist_pi_le_iff (by positivity)).mpr fun k => ?_
  set a : EucSpace d := ↑(f (tupleNormalize hd Y) k)
  set a' : EucSpace d := ↑(f (tupleNormalize hd Y') k)
  have ha : ‖a‖ = 1 := mem_sphere_zero_iff_norm.mp (f _ k).2
  have hsplit : ‖Y k‖ • a - ‖Y' k‖ • a' = (‖Y k‖ - ‖Y' k‖) • a + ‖Y' k‖ • (a - a') := by
    rw [sub_smul, smul_sub]
    abel
  have h1 : |‖Y k‖ - ‖Y' k‖| ≤ dist Y Y' :=
    (abs_norm_sub_norm_le _ _).trans (by rw [← dist_eq_norm]; exact dist_le_pi_dist Y Y' k)
  have h3 : ‖a - a'‖ ≤ K * (2 * dist Y Y') := by
    rw [← dist_eq_norm, ← Subtype.dist_eq]
    exact (dist_le_pi_dist _ _ k).trans (hF.trans (mul_le_mul_of_nonneg_left hG K.2))
  have h4 := mul_le_mul (hY' k).2.le h3 (norm_nonneg _) zero_le_two
  rw [radialLift, radialLift, dist_eq_norm, hsplit]
  refine (norm_add_le _ _).trans ?_
  rw [norm_smul, norm_smul, ha, Real.norm_eq_abs, mul_one, norm_norm]
  push_cast
  linarith

/-- The hypothesis of `lipschitzOnWith_radialLift` holds for the identity. -/
example : LipschitzWith 1 (id : SphereTuple 2 3 → SphereTuple 2 3) := LipschitzWith.id

/-- **The radial lift covers the preimage of an image:** a tuple of the shells
whose projection lies in `f '' N` is the lift of the tuple with the same radii
and the directions of a preimage in `N`. -/
theorem preimage_image_subset_radialLift (hd : 1 ≤ d) (f : SphereTuple d n → SphereTuple d n)
    (N : Set (SphereTuple d n)) :
    tupleNormalize hd ⁻¹' (f '' N) ∩ tupleShell d n ⊆
      radialLift hd f '' (tupleNormalize hd ⁻¹' N ∩ tupleShell d n) := by
  rintro Y ⟨⟨X, hXN, hfX⟩, hY⟩
  rw [mem_tupleShell] at hY
  have hY0 : ∀ k, ‖Y k‖ ≠ 0 := fun k => norm_ne_zero_iff.mpr (ne_zero_of_mem_sphereShell (hY k))
  have hnorm : ∀ k, ‖‖Y k‖ • (X k : EucSpace d)‖ = ‖Y k‖ := fun k => by
    rw [norm_smul, norm_norm, mem_sphere_zero_iff_norm.mp (X k).2, mul_one]
  have hX : tupleNormalize hd (fun k => ‖Y k‖ • (X k : EucSpace d)) = X := by
    funext k
    apply Subtype.ext
    rw [tupleNormalize, coe_sphereNormalize hd (norm_ne_zero_iff.mp (by rw [hnorm]; exact hY0 k)),
      hnorm, smul_smul, inv_mul_cancel₀ (hY0 k), one_smul]
  refine ⟨fun k => ‖Y k‖ • (X k : EucSpace d), ⟨by rw [mem_preimage, hX]; exact hXN,
    mem_tupleShell.mpr fun k => ?_⟩, ?_⟩
  · simp only [sphereShell, mem_ofPred_eq, hnorm]
    exact hY k
  · funext k
    simp only [radialLift, hX, hfX, hnorm]
    rw [tupleNormalize, coe_sphereNormalize hd (norm_ne_zero_iff.mp (hY0 k)), smul_smul,
      mul_inv_cancel₀ (hY0 k), one_smul]

/-- **Lipschitz maps send closed null sets of the uniform law to null sets**
(`d ≥ 1`): on the shells the lifted map is Lipschitz, and Haar-null sets stay
null. -/
theorem measure_image_eq_zero_of_lipschitz (hd : 1 ≤ d) (P : Measure (SphereTuple d n))
    (hP : UniformTuple d n P) {N : Set (SphereTuple d n)} (hN : IsClosed N) (hPN : P N = 0)
    {K : ℝ≥0} {f : SphereTuple d n → SphereTuple d n} (hf : LipschitzWith K f) :
    P (f '' N) = 0 := by
  obtain ⟨μ, hμ, rfl⟩ := exists_addHaar_eq_map hd P hP
  rw [Measure.map_apply (measurable_tupleNormalize hd) hN.measurableSet,
    Measure.restrict_apply' measurableSet_tupleShell] at hPN
  rw [Measure.map_apply (measurable_tupleNormalize hd)
    (hN.isCompact.image hf.continuous).isClosed.measurableSet,
    Measure.restrict_apply' measurableSet_tupleShell]
  exact measure_mono_null (preimage_image_subset_radialLift hd f N)
    (addHaar_image_eq_zero_of_lipschitzOnWith μ
      ((lipschitzOnWith_radialLift hd hf).mono inter_subset_right) hPN)

/-- The hypotheses of `measure_image_eq_zero_of_lipschitz` hold for the empty set
and the identity, under the product of shell laws on the circle. -/
example : UniformTuple 2 3 (Measure.pi fun _ : Idx 3 => shellLaw (d := 2) (by norm_num)) ∧
    IsClosed (∅ : Set (SphereTuple 2 3)) ∧
    (Measure.pi fun _ : Idx 3 => shellLaw (d := 2) (by norm_num)) ∅ = 0 ∧
    LipschitzWith 1 (id : SphereTuple 2 3 → SphereTuple 2 3) :=
  ⟨uniformTuple_shellLaw _, isClosed_empty, measure_empty, LipschitzWith.id⟩

end Perspective
end Transformer

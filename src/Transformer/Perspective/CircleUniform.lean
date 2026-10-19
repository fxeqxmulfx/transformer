module

public import Transformer.Perspective.CircleAngles
public import Transformer.Perspective.SphereUnique
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.Periodic
public import Mathlib.MeasureTheory.Function.SpecialFunctions.Basic
public import Mathlib.MeasureTheory.Measure.Haar.Unique

/-!
# The uniform law on the circle in angles

On the circle the uniform law `σ_2` of `UniformTuple` is the image of the
normalized Lebesgue measure on `[0, 2π)` (`angleLaw`) under
`θ ↦ (cos θ, sin θ)` (`circleLaw`). That image is a rotation-invariant
probability measure (`circleLaw_isometry`): every isometry of the plane acts on
angles by a translation or a reflection (`exists_sphereMap_circlePoint`), and
the image of Lebesgue measure on an interval `(t, t + 2π]` does not depend on
`t` (`map_circlePoint_restrict_Ioc`). The uniform law is unique
(`eq_of_rotation_invariant`), so every `UniformTuple 2 n P` is the image under
`anglesToSphere` of the normalized Lebesgue measure on `[0, 2π)ⁿ`
(`eq_map_angleCube`).

Source: arXiv:2312.10794v5, §3.2 (footnote: `σ_d` is the Lebesgue measure on
the sphere, normalized), §4 (the uniform law of `p:beta0` and
`thm: beta.tiny`), §7.1 (particles on the circle as angles).
-/

@[expose] public section

open MeasureTheory Complex Set
open scoped Real ENNReal

namespace Transformer
namespace Perspective

/-- The normalized Lebesgue measure on `[0, 2π)`: the law of a uniform angle. -/
noncomputable def angleLaw : Measure ℝ :=
  (ENNReal.ofReal (2 * π))⁻¹ • volume.restrict (Ico 0 (2 * π))

instance isProbabilityMeasure_angleLaw : IsProbabilityMeasure angleLaw :=
  ⟨by rw [angleLaw, Measure.smul_apply, Measure.restrict_apply_univ, Real.volume_Ico, sub_zero,
    smul_eq_mul, ENNReal.inv_mul_cancel (ENNReal.ofReal_pos.2 Real.two_pi_pos).ne'
      ENNReal.ofReal_ne_top]⟩

theorem measurable_circlePoint : Measurable circlePoint :=
  continuous_circlePoint.measurable

/-- The law of `(cos θ, sin θ)` for a uniform angle `θ`. -/
noncomputable def circleLaw : Measure (SSphere 2) :=
  angleLaw.map circlePoint

instance isProbabilityMeasure_circleLaw : IsProbabilityMeasure circleLaw :=
  (Measure.isProbabilityMeasure_map_iff measurable_circlePoint.aemeasurable).2 inferInstance

/-- **One period of the circle**: the image of Lebesgue measure on `(t, t + 2π]`
under `θ ↦ (cos θ, sin θ)` does not depend on `t`. Both are the image of the
Haar measure of `ℝ / 2πℤ` under the map that `θ ↦ (cos θ, sin θ)` induces. -/
theorem map_circlePoint_restrict_Ioc (t : ℝ) :
    (volume.restrict (Ioc t (t + 2 * π))).map circlePoint =
      (volume.restrict (Ioc 0 (2 * π))).map circlePoint := by
  have : Fact (0 < 2 * π) := ⟨Real.two_pi_pos⟩
  have key : ∀ s : ℝ, ∀ A, MeasurableSet A →
      (volume.restrict (Ioc s (s + 2 * π))).map circlePoint A =
        ∫⁻ b : AddCircle (2 * π), A.indicator 1 (circlePoint_periodic.lift b) := by
    intro s A hA
    rw [Measure.map_apply measurable_circlePoint hA,
      ← lintegral_indicator_one (measurable_circlePoint hA),
      ← AddCircle.lintegral_preimage (2 * π) s]
    rfl
  ext A hA
  have h0 := key 0 A hA
  rw [zero_add] at h0
  rw [key t A hA, h0]

/-- Lebesgue measure on `(0, 2π]` translated by `α` is Lebesgue measure on
`(α, α + 2π]`. -/
theorem map_restrict_const_add (α : ℝ) :
    (volume.restrict (Ioc 0 (2 * π))).map (α + ·) = volume.restrict (Ioc α (α + 2 * π)) := by
  have h := ((measurePreserving_add_left volume α).restrict_preimage
    (measurableSet_Ioc (a := α) (b := α + 2 * π))).map_eq
  rwa [preimage_const_add_Ioc, sub_self, add_sub_cancel_left] at h

/-- Lebesgue measure on `(0, 2π]` reflected by `θ ↦ α - θ` is Lebesgue measure
on `(α - 2π, α]`. -/
theorem map_restrict_const_sub (α : ℝ) :
    (volume.restrict (Ioc 0 (2 * π))).map (α - ·) =
      volume.restrict (Ioc (α - 2 * π) (α - 2 * π + 2 * π)) := by
  have : (volume : Measure ℝ).IsNegInvariant :=
    Measure.IsAddHaarMeasure.isNegInvariant_of_regular volume
  have h := ((Measure.measurePreserving_sub_left volume α).restrict_preimage
    (measurableSet_Ico (a := α - 2 * π) (b := α))).map_eq
  rw [sub_add_cancel]
  rwa [preimage_const_sub_Ico, sub_self, sub_sub_cancel, Measure.restrict_congr_set Ico_ae_eq_Ioc]
    at h

/-- The circle law, with the interval `(0, 2π]`. -/
theorem circleLaw_eq :
    circleLaw = (ENNReal.ofReal (2 * π))⁻¹ • (volume.restrict (Ioc 0 (2 * π))).map circlePoint := by
  rw [circleLaw, angleLaw, Measure.restrict_congr_set Ico_ae_eq_Ioc,
    Measure.map_smul _ measurable_circlePoint.aemeasurable]

/-- **The circle law is rotation invariant**: every linear isometry of the plane
preserves it. -/
theorem circleLaw_isometry (U : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2) :
    circleLaw.map (sphereMap 2 U) = circleLaw := by
  have hU := measurable_sphereMap 2 U
  rw [circleLaw_eq, Measure.map_smul _ hU.aemeasurable, Measure.map_map hU measurable_circlePoint]
  congr 1
  obtain ⟨α, hα | hα⟩ := exists_sphereMap_circlePoint U
  · rw [show sphereMap 2 U ∘ circlePoint = circlePoint ∘ (α + ·) from funext hα,
      ← Measure.map_map measurable_circlePoint (measurable_const_add α), map_restrict_const_add,
      map_circlePoint_restrict_Ioc]
  · rw [show sphereMap 2 U ∘ circlePoint = circlePoint ∘ (α - ·) from funext hα,
      ← Measure.map_map measurable_circlePoint (measurable_const_sub α), map_restrict_const_sub,
      map_circlePoint_restrict_Ioc]

/-- The product of circle laws is a uniform tuple law. -/
theorem uniformTuple_circleLaw (n : ℕ) : UniformTuple 2 n (Measure.pi fun _ : Idx n => circleLaw) :=
  ⟨circleLaw, inferInstance, circleLaw_isometry, rfl⟩

/-- The product of `n` uniform angles is the normalized Lebesgue measure on
`[0, 2π)ⁿ`. -/
theorem pi_angleLaw (n : ℕ) :
    Measure.pi (fun _ : Idx n => angleLaw) =
      (ENNReal.ofReal (2 * π) ^ n)⁻¹ •
        volume.restrict (univ.pi fun _ : Idx n => Ico 0 (2 * π)) := by
  refine Measure.pi_eq fun s hs => ?_
  rw [Measure.smul_apply, Measure.restrict_apply (MeasurableSet.univ_pi hs),
    ← Set.pi_inter_distrib, volume_pi_pi, smul_eq_mul, ENNReal.inv_pow]
  simp only [angleLaw, Measure.smul_apply, Measure.restrict_apply (hs _), smul_eq_mul,
    Finset.prod_mul_distrib, Finset.prod_const, Finset.card_univ, Fintype.card_fin]

/-- **The uniform law on `(𝕊¹)ⁿ` is normalized Lebesgue measure in angles**:
every `UniformTuple 2 n P` is the image under `θ ↦ ((cos θ_i, sin θ_i))_i` of the
normalized Lebesgue measure on `[0, 2π)ⁿ`. -/
theorem eq_map_angleCube {n : ℕ} {P : Measure (SphereTuple 2 n)} (hP : UniformTuple 2 n P) :
    P = ((ENNReal.ofReal (2 * π) ^ n)⁻¹ •
      volume.restrict (univ.pi fun _ : Idx n => Ico 0 (2 * π))).map anglesToSphere := by
  obtain ⟨σ, hσp, hσ, rfl⟩ := hP
  rw [← pi_angleLaw, eq_of_rotation_invariant σ circleLaw hσ circleLaw_isometry]
  exact (Measure.pi_map_pi (f := fun _ => circlePoint)
    fun _ => measurable_circlePoint.aemeasurable).symm

/-- The hypothesis of `eq_map_angleCube` is satisfiable: the product of circle
laws is a uniform tuple law. -/
example : UniformTuple 2 3 (Measure.pi fun _ : Idx 3 => circleLaw) :=
  uniformTuple_circleLaw 3

end Perspective
end Transformer

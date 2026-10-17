/-
# Saddle avoidance — preimages of null sets

A `C¹` map `g` of a finite-dimensional real normed space into itself whose
derivative has nonzero determinant everywhere pulls null sets back to null
sets (`measure_preimage_eq_zero`), and so do its iterates
(`measure_preimage_iterate_eq_zero`).

By the inverse function theorem every point has an open neighbourhood `U`
on which `g` is injective; countably many of them cover the space.  On
`s = g⁻¹(N) ∩ U` the change of variables formula gives
`∫_s |det Dg| = μ(g(s)) ≤ μ(N) = 0`, and `|det Dg| > 0`, so `μ(s) = 0`.

This is `lem:measure-0-local-diff` of arXiv:1710.07406, §4.1, for the space
`ℝᵈ` itself in place of a manifold; the source argues through local inverses,
which are `C¹` and so locally Lipschitz.
-/

import Transformer.SaddleAvoidance.Defs
import Mathlib.Analysis.Calculus.InverseFunctionTheorem.ContDiff
import Mathlib.MeasureTheory.Function.Jacobian

open Set MeasureTheory

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]

/-- **Inverse function theorem, injectivity part.**  A `C¹` map whose
derivative at `x` has nonzero determinant is injective on an open
neighbourhood of `x`. -/
theorem exists_isOpen_injOn {g : E → E} (hg : ContDiff ℝ 1 g) {x : E}
    (hdet : (fderiv ℝ g x).det ≠ 0) : ∃ U : Set E, IsOpen U ∧ x ∈ U ∧ InjOn g U := by
  let f' : E ≃L[ℝ] E :=
    (LinearMap.equivOfIsUnitDet (isUnit_iff_ne_zero.2 hdet)).toContinuousLinearEquiv
  have hf' : HasFDerivAt g (f' : E →L[ℝ] E) x := by
    convert (hg.differentiable one_ne_zero x).hasFDerivAt using 1
    ext v
    simp [f']
  exact ⟨_, (hg.contDiffAt.toOpenPartialHomeomorph g hf' one_ne_zero).open_source,
    hg.contDiffAt.mem_toOpenPartialHomeomorph_source hf' one_ne_zero,
    (hg.contDiffAt.toOpenPartialHomeomorph g hf' one_ne_zero).injOn⟩

variable [MeasurableSpace E] [BorelSpace E]

/-- **Preimages of null sets are null** under a `C¹` map whose derivative has
nonzero determinant everywhere (`lem:measure-0-local-diff` of
arXiv:1710.07406). -/
theorem measure_preimage_eq_zero (μ : Measure E) [μ.IsAddHaarMeasure] {g : E → E}
    (hg : ContDiff ℝ 1 g) (hdet : ∀ x, (fderiv ℝ g x).det ≠ 0) {N : Set E} (hN : μ N = 0) :
    μ (g ⁻¹' N) = 0 := by
  obtain ⟨N', hNN', hN'm, hN'0⟩ := exists_measurable_superset_of_null hN
  choose U hUo hxU hUinj using fun x => exists_isOpen_injOn hg (hdet x)
  obtain ⟨T, hTc, hTU⟩ := TopologicalSpace.isOpen_iUnion_countable U hUo
  have hsub : g ⁻¹' N ⊆ ⋃ x ∈ T, (g ⁻¹' N' ∩ U x) := by
    intro y hy
    have hyU : y ∈ ⋃ x ∈ T, U x := by
      rw [hTU]
      exact mem_iUnion.2 ⟨y, hxU y⟩
    obtain ⟨x, hxT, hyx⟩ := mem_iUnion₂.1 hyU
    exact mem_iUnion₂.2 ⟨x, hxT, hNN' hy, hyx⟩
  refine measure_mono_null hsub ((measure_biUnion_null_iff hTc).2 fun x _ => ?_)
  have hs : MeasurableSet (g ⁻¹' N' ∩ U x) :=
    (hN'm.preimage hg.continuous.measurable).inter (hUo x).measurableSet
  have hcv := lintegral_abs_det_fderiv_eq_addHaar_image μ hs
    (fun y _ => (hg.differentiable one_ne_zero y).hasFDerivAt.hasFDerivWithinAt)
    ((hUinj x).mono inter_subset_right)
  have himg : μ (g '' (g ⁻¹' N' ∩ U x)) = 0 :=
    measure_mono_null (by rintro _ ⟨y, ⟨hy, _⟩, rfl⟩; exact hy) hN'0
  rw [himg] at hcv
  have hmeas : Measurable fun y => ENNReal.ofReal |(fderiv ℝ g y).det| :=
    ENNReal.measurable_ofReal.comp
      ((ContinuousLinearMap.continuous_det.comp (hg.continuous_fderiv one_ne_zero)).abs.measurable)
  rw [setLIntegral_eq_zero_iff hs hmeas] at hcv
  rw [measure_eq_zero_iff_ae_notMem]
  filter_upwards [hcv] with y hy hys
  have h := hy hys
  rw [ENNReal.ofReal_eq_zero, abs_nonpos_iff] at h
  exact hdet y h

/-- Preimages of null sets under the iterates of such a map are null. -/
theorem measure_preimage_iterate_eq_zero (μ : Measure E) [μ.IsAddHaarMeasure] {g : E → E}
    (hg : ContDiff ℝ 1 g) (hdet : ∀ x, (fderiv ℝ g x).det ≠ 0) {N : Set E} (hN : μ N = 0)
    (m : ℕ) : μ (g^[m] ⁻¹' N) = 0 := by
  induction m with
  | zero => simpa using hN
  | succ m ih =>
    rw [Function.iterate_succ, preimage_comp]
    exact measure_preimage_eq_zero μ hg hdet ih

/-- The hypotheses of the three theorems above hold for the identity of the
line, whose derivative has determinant `1`, and the null set `{0}`. -/
example : ContDiff ℝ 1 (id : ℝ → ℝ) ∧ (∀ x : ℝ, (fderiv ℝ (id : ℝ → ℝ) x).det ≠ 0) ∧
    volume ({0} : Set ℝ) = 0 := by
  refine ⟨contDiff_id, fun x => ?_, measure_singleton 0⟩
  rw [fderiv_id, ContinuousLinearMap.det, ContinuousLinearMap.coe_id, LinearMap.det_id]
  exact one_ne_zero

end Transformer.SaddleAvoidance

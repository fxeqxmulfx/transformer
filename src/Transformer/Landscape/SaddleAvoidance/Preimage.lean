module

public import Transformer.Landscape.SaddleAvoidance.Defs
public import Mathlib.Analysis.Calculus.InverseFunctionTheorem.ContDiff
public import Mathlib.MeasureTheory.Function.Jacobian

/-!
# Saddle avoidance — preimages of null sets

A `C¹` map `g` of a finite-dimensional real normed space into itself whose
derivative has nonzero determinant everywhere pulls null sets back to null
sets (`measure_preimage_eq_zero`), and so do its iterates
(`measure_preimage_iterate_eq_zero`). Locally: the points of a set `S` where
the determinant is nonzero that `g` maps into a null set form a null set
(`measure_inter_preimage_eq_zero`), and the same holds for the iterates when
`g` maps `S` into itself (`measure_inter_preimage_iterate_eq_zero`).

By the inverse function theorem every point where the determinant is nonzero
has an open neighbourhood `U` on which `g` is injective; countably many of
them cover the open set of such points. On `s = g⁻¹(N) ∩ U` the change of
variables formula gives `∫_s |det Dg| = μ(g(s)) ≤ μ(N) = 0`, and
`|det Dg| > 0`, so `μ(s) = 0`.

This is `lem:measure-0-local-diff` of arXiv:1710.07406, §4.1, for the space
`ℝᵈ` itself in place of a manifold; the source argues through local inverses,
which are `C¹` and so locally Lipschitz.
-/

@[expose] public section

open Set MeasureTheory

namespace Transformer.Landscape.SaddleAvoidance

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

/-- **Preimages of null sets are null where the derivative is invertible.**
If `g` is `C¹` and its derivative has nonzero determinant at every point of
`S`, the points of `S` that `g` maps into a null set form a null set. -/
theorem measure_inter_preimage_eq_zero (μ : Measure E) [μ.IsAddHaarMeasure] {g : E → E}
    (hg : ContDiff ℝ 1 g) {S : Set E} (hdet : ∀ x ∈ S, (fderiv ℝ g x).det ≠ 0) {N : Set E}
    (hN : μ N = 0) : μ (S ∩ g ⁻¹' N) = 0 := by
  set O := {x | (fderiv ℝ g x).det ≠ 0}
  have hcont : Continuous fun x => (fderiv ℝ g x).det :=
    ContinuousLinearMap.continuous_det.comp (hg.continuous_fderiv one_ne_zero)
  have hO : IsOpen O := isOpen_ne_fun hcont continuous_const
  obtain ⟨N', hNN', hN'm, hN'0⟩ := exists_measurable_superset_of_null hN
  choose U hUo hxU hUinj using fun x : O => exists_isOpen_injOn hg x.2
  obtain ⟨T, hTc, hTU⟩ := TopologicalSpace.isOpen_iUnion_countable (fun x : O => U x ∩ O)
    fun x => (hUo x).inter hO
  have hsub : S ∩ g ⁻¹' N ⊆ ⋃ x ∈ T, (g ⁻¹' N' ∩ (U x ∩ O)) := by
    rintro y ⟨hyS, hy⟩
    have hyU : y ∈ ⋃ x ∈ T, (U x ∩ O) := by
      rw [hTU]
      exact mem_iUnion.2 ⟨⟨y, hdet y hyS⟩, hxU _, hdet y hyS⟩
    obtain ⟨x, hxT, hyx⟩ := mem_iUnion₂.1 hyU
    exact mem_iUnion₂.2 ⟨x, hxT, hNN' hy, hyx⟩
  refine measure_mono_null hsub ((measure_biUnion_null_iff hTc).2 fun x _ => ?_)
  have hs : MeasurableSet (g ⁻¹' N' ∩ (U x ∩ O)) :=
    (hN'm.preimage hg.continuous.measurable).inter ((hUo x).inter hO).measurableSet
  have hcv := lintegral_abs_det_fderiv_eq_addHaar_image μ hs
    (fun y _ => (hg.differentiable one_ne_zero y).hasFDerivAt.hasFDerivWithinAt)
    ((hUinj x).mono (inter_subset_right.trans inter_subset_left))
  have himg : μ (g '' (g ⁻¹' N' ∩ (U x ∩ O))) = 0 :=
    measure_mono_null (by rintro _ ⟨y, ⟨hy, _⟩, rfl⟩; exact hy) hN'0
  rw [himg] at hcv
  have hmeas : Measurable fun y => ENNReal.ofReal |(fderiv ℝ g y).det| :=
    ENNReal.measurable_ofReal.comp hcont.abs.measurable
  rw [setLIntegral_eq_zero_iff hs hmeas] at hcv
  rw [measure_eq_zero_iff_ae_notMem]
  filter_upwards [hcv] with y hy hys
  have h := hy hys
  rw [ENNReal.ofReal_eq_zero, abs_nonpos_iff] at h
  exact hys.2.2 h

/-- **Preimages of null sets are null** under a `C¹` map whose derivative has
nonzero determinant everywhere (`lem:measure-0-local-diff` of
arXiv:1710.07406). -/
theorem measure_preimage_eq_zero (μ : Measure E) [μ.IsAddHaarMeasure] {g : E → E}
    (hg : ContDiff ℝ 1 g) (hdet : ∀ x, (fderiv ℝ g x).det ≠ 0) {N : Set E} (hN : μ N = 0) :
    μ (g ⁻¹' N) = 0 := by
  simpa using measure_inter_preimage_eq_zero μ hg (S := univ) (fun x _ => hdet x) hN

/-- Preimages of null sets under the iterates of `g` are null on a set that
`g` maps into itself and where its derivative has nonzero determinant. -/
theorem measure_inter_preimage_iterate_eq_zero (μ : Measure E) [μ.IsAddHaarMeasure]
    {g : E → E} (hg : ContDiff ℝ 1 g) {S : Set E} (hS : MapsTo g S S)
    (hdet : ∀ x ∈ S, (fderiv ℝ g x).det ≠ 0) {N : Set E} (hN : μ N = 0) (m : ℕ) :
    μ (S ∩ g^[m] ⁻¹' N) = 0 := by
  induction m with
  | zero => exact measure_mono_null inter_subset_right (by simpa using hN)
  | succ m ih =>
    refine measure_mono_null (fun x hx => ?_) (measure_inter_preimage_eq_zero μ hg hdet ih)
    refine ⟨hx.1, hS hx.1, ?_⟩
    simpa only [mem_preimage, Function.iterate_succ_apply] using hx.2

/-- Preimages of null sets under the iterates of such a map are null. -/
theorem measure_preimage_iterate_eq_zero (μ : Measure E) [μ.IsAddHaarMeasure] {g : E → E}
    (hg : ContDiff ℝ 1 g) (hdet : ∀ x, (fderiv ℝ g x).det ≠ 0) {N : Set E} (hN : μ N = 0)
    (m : ℕ) : μ (g^[m] ⁻¹' N) = 0 := by
  simpa using measure_inter_preimage_iterate_eq_zero μ hg (mapsTo_univ g univ)
    (fun x _ => hdet x) hN m

/-- The hypotheses of the five theorems above hold for the identity of the
line, whose derivative has determinant `1`, the null set `{0}`, and the
interval `[0, 1]`, which the identity maps into itself. -/
example : ContDiff ℝ 1 (id : ℝ → ℝ) ∧ (∀ x : ℝ, (fderiv ℝ (id : ℝ → ℝ) x).det ≠ 0) ∧
    volume ({0} : Set ℝ) = 0 ∧ MapsTo (id : ℝ → ℝ) (Icc 0 1) (Icc 0 1) := by
  refine ⟨contDiff_id, fun x => ?_, measure_singleton 0, mapsTo_id _⟩
  rw [fderiv_id, ContinuousLinearMap.det, ContinuousLinearMap.coe_id, LinearMap.det_id]
  exact one_ne_zero

end Transformer.Landscape.SaddleAvoidance

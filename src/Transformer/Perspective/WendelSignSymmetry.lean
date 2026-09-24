/-
# Independent sign symmetry of a uniform spherical sample

The product law in Wendel's theorem is invariant under independently
reflecting any of its sampled points through the origin. This is the
probabilistic half of Wendel's sign-pattern counting argument.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelGeneralPosition
import Transformer.Perspective.WendelOneDimBasic
import Mathlib.MeasureTheory.Constructions.Pi

open MeasureTheory

namespace Transformer.Perspective

/-- Reflect exactly the coordinates selected by the Boolean mask.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
noncomputable def flipSigns (d n : ℕ) (mask : Idx n → Bool)
    (X : SphereTuple d n) : SphereTuple d n :=
  fun i => if mask i then sphereMap d (LinearIsometryEquiv.neg ℝ) (X i) else X i

/-- The common-open-hemisphere event is open, hence measurable. It is the
union over possible directions of finite intersections of open half-spaces.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem isOpen_hemisphereEvent (d n : ℕ) :
    IsOpen {X : SphereTuple d n | ∃ w : SSphere d, ∀ i : Idx n,
      0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))} := by
  have hfixed (w : SSphere d) :
      IsOpen {X : SphereTuple d n | ∀ i : Idx n,
        0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))} := by
    have heq : {X : SphereTuple d n | ∀ i : Idx n,
        0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))} =
        ⋂ i ∈ (Finset.univ : Finset (Idx n)),
          {X : SphereTuple d n | 0 < inner (𝕜 := ℝ) ((X i : EucSpace d))
            ((w : EucSpace d))} := by
      ext X
      simp
    rw [heq]
    apply isOpen_biInter_finset
    intro i _
    have hc : Continuous (fun X : SphereTuple d n =>
        inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))) := by
      fun_prop
    exact isOpen_Ioi.preimage hc
  have heq : {X : SphereTuple d n | ∃ w : SSphere d, ∀ i : Idx n,
      0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))} =
      ⋃ w : SSphere d, {X : SphereTuple d n | ∀ i : Idx n,
        0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))} := by
    ext X
    simp
  rw [heq]
  exact isOpen_iUnion hfixed

/-- Each independently chosen sign pattern preserves the product law of an
isometry-invariant spherical measure.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem measurePreserving_flipSigns (d n : ℕ) (σ : Measure (SSphere d))
    [SigmaFinite σ]
    (hσ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, σ.map (sphereMap d U) = σ)
    (mask : Idx n → Bool) :
    MeasurePreserving (flipSigns d n mask)
      (Measure.pi fun _ : Idx n => σ) (Measure.pi fun _ : Idx n => σ) := by
  have hlocal (i : Idx n) :
      MeasurePreserving
        (fun x : SSphere d => if mask i then sphereMap d (LinearIsometryEquiv.neg ℝ) x else x)
        σ σ := by
    cases h : mask i with
    | false =>
      change MeasurePreserving id σ σ
      exact MeasurePreserving.id σ
    | true =>
      simpa [h] using
        (⟨measurable_sphereMap d (LinearIsometryEquiv.neg ℝ),
          hσ (LinearIsometryEquiv.neg ℝ)⟩ :
          MeasurePreserving (sphereMap d (LinearIsometryEquiv.neg ℝ)) σ σ)
  have hp := measurePreserving_pi (fun _ : Idx n => σ) (fun _ : Idx n => σ) hlocal
  change MeasurePreserving (fun X i => if mask i then sphereMap d (LinearIsometryEquiv.neg ℝ) (X i)
    else X i) (Measure.pi fun _ : Idx n => σ) (Measure.pi fun _ : Idx n => σ)
  exact hp

/-- Under a uniform product law, the probability of the hemisphere event is
unchanged if an arbitrary fixed sign pattern is applied to the sample.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem measure_hemisphereEvent_flipSigns (d n : ℕ)
    (P : Measure (SphereTuple d n)) (hP : UniformTuple d n P)
    (mask : Idx n → Bool) :
    P ((flipSigns d n mask) ⁻¹' {X : SphereTuple d n |
      ∃ w : SSphere d, ∀ i : Idx n,
        0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))}) =
      P {X : SphereTuple d n | ∃ w : SSphere d, ∀ i : Idx n,
        0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))} := by
  obtain ⟨σ, hσprob, hσinv, rfl⟩ := hP
  let _ : IsProbabilityMeasure σ := hσprob
  exact (measurePreserving_flipSigns d n σ hσinv mask).measure_preimage
    (isOpen_hemisphereEvent d n).measurableSet.nullMeasurableSet

/-- Independent sign flips preserve linear general position. The selected
vectors are rescaled by units `±1`, which leaves independence unchanged.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem linearGeneralPosition_flipSigns (d n : ℕ) (mask : Idx n → Bool)
    (X : SphereTuple d n)
    (hX : ∀ I : Finset (Idx n), I.card ≤ d →
      LinearIndependent ℝ fun i : I => (X i : EucSpace d)) :
    ∀ I : Finset (Idx n), I.card ≤ d →
      LinearIndependent ℝ fun i : I => (flipSigns d n mask X i : EucSpace d) := by
  intro I hI
  let w : I → ℝˣ := fun i => if mask i then -1 else 1
  have heq : (fun i : I => (flipSigns d n mask X i : EucSpace d)) =
      w • (fun i : I => (X i : EucSpace d)) := by
    funext i
    cases h : mask i with
    | false => simp [flipSigns, w, h]
    | true => simp [flipSigns, w, h, sphereMap]
  rw [heq]
  exact (LinearIndependent.units_smul_iff _ w).2 (hX I hI)

/-- A concrete general-position configuration for the sign-flip theorem:
the single positive point of the one-dimensional sphere.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
example (mask : Idx 1 → Bool) :
    ∀ I : Finset (Idx 1), I.card ≤ 1 →
      LinearIndependent ℝ fun i : I =>
        (flipSigns 1 1 mask (fun _ => eOne) i : EucSpace 1) := by
  apply linearGeneralPosition_flipSigns
  intro I _
  have hbase : LinearIndependent ℝ (fun _ : Idx 1 => (eOne : EucSpace 1)) := by
    rw [linearIndependent_unique_iff]
    simp [eOne]
  exact hbase.comp Subtype.val Subtype.val_injective

/-- The invariance hypothesis is satisfiable: the zero measure is
σ-finite and isometry-invariant. Source: arXiv:2312.10794v5, §6.1,
`r:wendel`. -/
example : SigmaFinite (0 : Measure (SSphere 1)) ∧
    (∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1,
      (0 : Measure (SSphere 1)).map (sphereMap 1 U) = 0) :=
  ⟨inferInstance, fun _ => Measure.map_zero _⟩

end Transformer.Perspective

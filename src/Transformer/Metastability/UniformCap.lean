/-
# Metastability — Open caps carry positive uniform mass

The uniform law on `𝕊^{d-1}` is pinned down by `IsUniformOn` rather than
constructed, so every lower bound on the mass of a set has to come from the
invariance alone.  For open caps it does: the isometries act transitively on
the sphere (`Perspective.exists_sphereMap_apply_eq`), so all caps of one
radius carry the same mass, and finitely many of them cover the compact
sphere, so that common mass is positive.

Source: arXiv:2410.06833v1, §4 (the uniform law on `𝕊^{d-1}` of the
low-dimensional bound).
-/

import Transformer.Metastability.InitialUniform
import Transformer.Perspective.SphereInvariant

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Metastability

variable {d : ℕ}

/-- The open cap `{y ∈ 𝕊^{d-1} : ⟨y, x⟩ > 1 - η}` around `x`. -/
def openCap (x : SSphere d) (η : ℝ) : Set (SSphere d) :=
  { y | 1 - η < inner (𝕜 := ℝ) (y : EucSpace d) (x : EucSpace d) }

theorem isOpen_openCap (x : SSphere d) (η : ℝ) : IsOpen (openCap x η) :=
  isOpen_lt continuous_const (continuous_subtype_val.inner continuous_const)

/-- `‖y - x‖² = 2 - 2⟨y, x⟩` for unit vectors. -/
theorem norm_sub_sq_sphere (x y : SSphere d) :
    ‖(y : EucSpace d) - (x : EucSpace d)‖ ^ 2
      = 2 - 2 * inner (𝕜 := ℝ) (y : EucSpace d) (x : EucSpace d) := by
  rw [norm_sub_sq_real, mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2]
  ring

/-- `-1 ≤ ⟨x, y⟩ ≤ 1` for unit vectors. -/
theorem inner_mem_Icc_sphere (x y : SSphere d) :
    -1 ≤ inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d) ∧
      inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d) ≤ 1 := by
  have h := abs_real_inner_le_norm (x : EucSpace d) (y : EucSpace d)
  rw [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2, mul_one] at h
  exact abs_le.1 h

/-- **Open caps of one radius have one mass** under an isometry-invariant
measure: an isometry `U` with `U x = y` pulls the cap around `y` back to the
cap around `x`. -/
theorem measure_openCap_eq (ν : Measure (SSphere d))
    (hν : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, ν.map (Perspective.sphereMap d U) = ν)
    (x y : SSphere d) (η : ℝ) : ν (openCap x η) = ν (openCap y η) := by
  obtain ⟨U, hU⟩ := Perspective.exists_sphereMap_apply_eq d x y
  have hpre : Perspective.sphereMap d U ⁻¹' openCap y η = openCap x η := by
    ext z
    simp only [Set.mem_preimage, openCap, Set.mem_ofPred_eq]
    rw [← hU]
    show 1 - η < inner (𝕜 := ℝ) (U (z : EucSpace d)) (U (x : EucSpace d)) ↔ _
    rw [LinearIsometryEquiv.inner_map_map]
  have := Measure.map_apply (μ := ν) (Perspective.measurable_sphereMap d U)
    (isOpen_openCap y η).measurableSet
  rw [hν U, hpre] at this
  exact this.symm

/-- The hypothesis of `measure_openCap_eq` is satisfiable: the zero measure is
invariant under every `sphereMap`. -/
example : ∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1,
    (0 : Measure (SSphere 1)).map (Perspective.sphereMap 1 U) = 0 := fun _ =>
  Measure.map_zero _

/-- **Every open cap has positive uniform mass.**  The caps of radius `η`
around all points cover the sphere; a finite subfamily does, by compactness,
and they all carry the same mass, so that mass cannot vanish. -/
theorem measure_openCap_pos (ν : Measure (SSphere d)) (hν : IsUniformOn d ν)
    (x : SSphere d) {η : ℝ} (hη : 0 < η) : 0 < ν (openCap x η) := by
  have : IsProbabilityMeasure ν := hν.1
  have hcov : (Set.univ : Set (SSphere d)) ⊆ ⋃ y : SSphere d, openCap y η := by
    intro y _
    refine Set.mem_iUnion.2 ⟨y, ?_⟩
    show 1 - η < inner (𝕜 := ℝ) (y : EucSpace d) (y : EucSpace d)
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp y.2]
    linarith
  obtain ⟨t, ht⟩ := isCompact_univ.elim_finite_subcover (fun y : SSphere d => openCap y η)
    (fun y => isOpen_openCap y η) hcov
  refine pos_iff_ne_zero.2 fun h0 => ?_
  have hle := (measure_mono (μ := ν) ht).trans
    (measure_biUnion_finset_le t fun y => openCap y η)
  simp only [measure_univ, measure_openCap_eq ν hν.2 _ x η, h0, Finset.sum_const_zero] at hle
  exact absurd hle (by norm_num)

/-- The hypothesis `0 < η` of `measure_openCap_pos` is satisfiable; the
uniformity hypothesis is `IsUniformOn`, witnessed as `InitialUniform` explains
(the law is pinned down, not constructed). -/
example : (0 : ℝ) < 1 := one_pos

/-- `|⟨x, y⟩ - ⟨e, f⟩| ≤ ‖x - e‖ + ‖y - f‖` for unit vectors. -/
theorem abs_inner_sub_inner_le (x y e f : SSphere d) :
    |inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)
        - inner (𝕜 := ℝ) (e : EucSpace d) (f : EucSpace d)|
      ≤ ‖(x : EucSpace d) - e‖ + ‖(y : EucSpace d) - f‖ := by
  have hy : ‖(y : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp y.2
  have he : ‖(e : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp e.2
  have h : inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)
        - inner (𝕜 := ℝ) (e : EucSpace d) (f : EucSpace d)
      = inner (𝕜 := ℝ) ((x : EucSpace d) - e) (y : EucSpace d)
        + inner (𝕜 := ℝ) (e : EucSpace d) ((y : EucSpace d) - f) := by
    rw [inner_sub_left, inner_sub_right]; ring
  rw [h]
  refine (abs_add_le _ _).trans (add_le_add ?_ ?_)
  · exact (abs_real_inner_le_norm _ _).trans (by rw [hy, mul_one])
  · exact (abs_real_inner_le_norm _ _).trans (by rw [he, one_mul])

/-- A point of the open cap `openCap e (ε²/8)` is within `ε/2` of `e`. -/
theorem norm_sub_lt_of_mem_openCap {e x : SSphere d} {ε : ℝ} (hε : 0 < ε)
    (hx : x ∈ openCap e (ε ^ 2 / 8)) : ‖(x : EucSpace d) - e‖ < ε / 2 := by
  have hx' : 1 - ε ^ 2 / 8 < inner (𝕜 := ℝ) (x : EucSpace d) (e : EucSpace d) := hx
  have h := norm_sub_sq_sphere e x
  nlinarith [norm_nonneg ((x : EucSpace d) - e)]

end Metastability
end Transformer

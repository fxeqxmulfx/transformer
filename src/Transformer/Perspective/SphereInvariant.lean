/-
# Rotation-invariant measures on the sphere have no atoms

`Perspective.UniformTuple` pins the uniform law down by rotation invariance
instead of constructing it: its marginal `σ` is a probability measure on
`𝕊^{d-1}` invariant under every linear isometry of `ℝ^d`.  This file draws
from that description the fact the almost-sure statements of §4 and §6 rest
on: such a `σ` charges no point, as soon as `d ≥ 2`.

The argument is elementary.  The isometry group acts transitively on the
sphere, a Householder reflection carrying any unit vector to any other, so an
invariant measure gives all singletons the same mass; and the sphere is
infinite for `d ≥ 2`, so `k` points of common mass `a` force `k a ≤ 1` for
every `k`, hence `a = 0`.

The tuple-level consequences are in `Perspective.UniformAtomless`.

Source: arXiv:2312.10794v5, §4 (the uniform law `σ_d` of `p:beta0`).
-/

import Transformer.Perspective.Section2_FlowMap

open scoped BigOperators ENNReal
open MeasureTheory

namespace Transformer
namespace Perspective

variable (d : ℕ)

/-! ### Transitivity of the isometry group on the sphere -/

/-- **The isometries of `ℝ^d` act transitively on `𝕊^{d-1}`.**

The Householder reflection in the hyperplane orthogonal to `x - y` exchanges
two unit vectors `x` and `y`; `Submodule.reflection_sub` is exactly that
statement in Mathlib. -/
theorem exists_sphereMap_apply_eq (x y : SSphere d) :
    ∃ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, sphereMap d U x = y := by
  refine ⟨Submodule.reflection (ℝ ∙ ((x : EucSpace d) - (y : EucSpace d)))ᗮ, ?_⟩
  apply Subtype.ext
  show Submodule.reflection _ (x : EucSpace d) = (y : EucSpace d)
  refine Submodule.reflection_sub ?_
  rw [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2]

/-- The hypotheses of `exists_sphereMap_apply_eq` are satisfiable: `𝕊^0 ⊂ ℝ^1`
contains the first standard basis vector. -/
example : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 1 := by
  simp [PiLp.norm_single]

/-- `sphereMap d U` is the restriction of a continuous map to a subtype. -/
theorem continuous_sphereMap (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    Continuous (sphereMap d U) :=
  Continuous.subtype_mk (U.continuous.comp continuous_subtype_val) _

/-- `sphereMap d U` is measurable. -/
theorem measurable_sphereMap (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    Measurable (sphereMap d U) :=
  (continuous_sphereMap d U).measurable

/-- `sphereMap d U` is injective, `U` being a linear equivalence. -/
theorem injective_sphereMap (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    Function.Injective (sphereMap d U) := fun _ _ h =>
  Subtype.ext (U.injective (congrArg Subtype.val h))

/-- **A rotation-invariant measure gives every point the same mass.**

Transitivity turns invariance into a symmetry between singletons: if
`U` carries `x` to `y` then `{x}` is the preimage of `{y}`. -/
theorem measure_singleton_eq_of_invariant
    (σ : Measure (SSphere d))
    (hσ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, σ.map (sphereMap d U) = σ)
    (x y : SSphere d) : σ {x} = σ {y} := by
  obtain ⟨U, hU⟩ := exists_sphereMap_apply_eq d x y
  have hpre : sphereMap d U ⁻¹' {y} = {x} := by
    ext z
    simp only [Set.mem_preimage, Set.mem_singleton_iff]
    exact ⟨fun hz => injective_sphereMap d U (hz.trans hU.symm), fun hz => hz ▸ hU⟩
  have := Measure.map_apply (μ := σ) (measurable_sphereMap d U) (measurableSet_singleton y)
  rw [hσ U, hpre] at this
  exact this.symm

/-- The hypothesis of `measure_singleton_eq_of_invariant` is satisfiable: the
zero measure is invariant under every `sphereMap`. -/
example : ∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1,
    (0 : Measure (SSphere 1)).map (sphereMap 1 U) = 0 := fun _ => Measure.map_zero _

/-! ### The sphere is infinite in dimension `d ≥ 2` -/

/-- A point of `ℝ^d` (`d ≥ 2`) with first coordinate `a`, on the unit sphere
as soon as `|a| ≤ 1`: the two-dimensional parametrisation
`(a, √(1 - a²), 0, …, 0)`. -/
noncomputable def spherePt (hd : 2 ≤ d) (a : ℝ) : EucSpace d :=
  EuclideanSpace.single (⟨0, by omega⟩ : Fin d) a
    + EuclideanSpace.single (⟨1, by omega⟩ : Fin d) (Real.sqrt (1 - a ^ 2))

/-- `spherePt` is a unit vector whenever `|a| ≤ 1`, its two nonzero
coordinates being orthogonal with squares `a²` and `1 - a²`. -/
theorem norm_spherePt (hd : 2 ≤ d) (a : ℝ) (ha : |a| ≤ 1) :
    ‖spherePt d hd a‖ = 1 := by
  have hortho : inner (𝕜 := ℝ) (EuclideanSpace.single (⟨0, by omega⟩ : Fin d) a)
      (EuclideanSpace.single (⟨1, by omega⟩ : Fin d) (Real.sqrt (1 - a ^ 2))) = 0 := by
    simp [EuclideanSpace.inner_single_left]
  have h := norm_add_sq_real (EuclideanSpace.single (⟨0, by omega⟩ : Fin d) a)
      (EuclideanSpace.single (⟨1, by omega⟩ : Fin d) (Real.sqrt (1 - a ^ 2)))
  rw [hortho] at h
  have ha2 : a ^ 2 ≤ 1 := by nlinarith [abs_nonneg a, sq_abs a]
  have hs : Real.sqrt (1 - a ^ 2) ^ 2 = 1 - a ^ 2 := Real.sq_sqrt (by linarith)
  rw [PiLp.norm_single, PiLp.norm_single] at h
  simp only [Real.norm_eq_abs, mul_zero, add_zero] at h
  rw [sq_abs, sq_abs, hs] at h
  have hsq : ‖spherePt d hd a‖ ^ 2 = 1 := by rw [spherePt, h]; ring
  nlinarith [norm_nonneg (spherePt d hd a)]

/-- The first coordinate of `spherePt`, read off as an inner product. -/
theorem inner_spherePt (hd : 2 ≤ d) (a : ℝ) :
    inner (𝕜 := ℝ) (EuclideanSpace.single (⟨0, by omega⟩ : Fin d) (1 : ℝ))
      (spherePt d hd a) = a := by
  simp [spherePt, inner_add_right, EuclideanSpace.inner_single_right]

/-- **`𝕊^{d-1}` is infinite for `d ≥ 2`**: the points `spherePt (1/(k+1))`,
`k ∈ ℕ`, are pairwise distinct, their first coordinates being. -/
theorem infinite_sSphere (hd : 2 ≤ d) : Infinite (SSphere d) := by
  have hmem : ∀ k : ℕ, spherePt d hd (1 / (k + 1 : ℝ)) ∈ Metric.sphere (0 : EucSpace d) 1 := by
    intro k
    have hk : (0 : ℝ) < (k : ℝ) + 1 := by positivity
    have h1 : |1 / ((k : ℝ) + 1)| ≤ 1 := by
      rw [abs_of_pos (by positivity)]
      rw [div_le_one hk]
      linarith [Nat.cast_nonneg (α := ℝ) k]
    exact mem_sphere_zero_iff_norm.mpr (norm_spherePt d hd _ h1)
  refine Infinite.of_injective (fun k : ℕ => (⟨_, hmem k⟩ : SSphere d)) ?_
  intro k m hkm
  have := congrArg (fun z : SSphere d =>
    inner (𝕜 := ℝ) (EuclideanSpace.single (⟨0, by omega⟩ : Fin d) (1 : ℝ))
      (z : EucSpace d)) hkm
  simp only [inner_spherePt] at this
  have hk : (0 : ℝ) < (k : ℝ) + 1 := by positivity
  have hm : (0 : ℝ) < (m : ℝ) + 1 := by positivity
  have hcast : (k : ℝ) = (m : ℝ) := by
    field_simp at this
    linarith
  exact_mod_cast hcast

/-- **A rotation-invariant probability measure on `𝕊^{d-1}` has no atoms**
when `d ≥ 2`: all singletons carry the same mass `a`, and `k` distinct points
carry `k a ≤ 1` for every `k`, which forces `a = 0`. -/
theorem measure_singleton_eq_zero_of_invariant (hd : 2 ≤ d)
    (σ : Measure (SSphere d)) [IsProbabilityMeasure σ]
    (hσ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, σ.map (sphereMap d U) = σ)
    (x : SSphere d) : σ {x} = 0 := by
  by_contra hne
  have := infinite_sSphere d hd
  obtain ⟨k, hk⟩ := ENNReal.exists_nat_mul_gt hne (b := 1) ENNReal.one_ne_top
  obtain ⟨s, hs⟩ := Infinite.exists_subset_card_eq (SSphere d) k
  have hsum : σ ↑s = (k : ℝ≥0∞) * σ {x} := by
    rw [← sum_measure_singleton (μ := σ) (s := s),
      Finset.sum_congr rfl (fun p _ => measure_singleton_eq_of_invariant d σ hσ p x),
      Finset.sum_const, hs, nsmul_eq_mul]
  exact absurd (hsum ▸ prob_le_one) (not_le.mpr hk)

end Perspective
end Transformer

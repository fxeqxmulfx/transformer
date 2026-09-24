/-
# Two-point sphere and invariant law in dimension one

The one-dimensional unit sphere has two points. A rotation-invariant
probability measure gives each mass `1/2`, so the hemisphere event is exactly
the union of the two constant configurations.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.UniformHemisphere

open MeasureTheory

namespace Transformer.Perspective

/-- The positive point of the one-dimensional sphere.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
noncomputable def eOne : SSphere 1 :=
  ⟨EuclideanSpace.single (0 : Fin 1) (1 : ℝ), by simp⟩

/-- The negative point of the one-dimensional sphere.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
noncomputable def eNeg : SSphere 1 :=
  ⟨EuclideanSpace.single (0 : Fin 1) (-1 : ℝ), by simp⟩

/-- These are the only two points of the one-dimensional unit sphere.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem sphere_one_cases (x : SSphere 1) : x = eOne ∨ x = eNeg := by
  have hx : (x : EucSpace 1) = EuclideanSpace.single (0 : Fin 1) (x.1.ofLp 0) := by
    ext i
    fin_cases i
    simp
  have hnorm : ‖(x : EucSpace 1)‖ = 1 := by
    exact mem_sphere_zero_iff_norm.mp x.2
  have habs : |x.1.ofLp 0| = 1 := by
    rw [hx] at hnorm
    simp at hnorm ⊢
    exact hnorm
  rcases (abs_eq (by norm_num : (0 : ℝ) ≤ 1)).mp habs with hp | hm
  · left
    apply Subtype.ext
    rw [hx, hp]
    rfl
  · right
    apply Subtype.ext
    rw [hx, hm]
    rfl

/-- The two points are distinct. Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem eOne_ne_eNeg : eOne ≠ eNeg := by
  intro h
  have hval : (eOne : EucSpace 1).ofLp 0 = (eNeg : EucSpace 1).ofLp 0 :=
    congrArg (fun x : SSphere 1 => (x : EucSpace 1).ofLp 0) h
  norm_num [eOne, eNeg] at hval

/-- The positive point has positive self inner product.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem inner_eOne_eOne :
    inner (𝕜 := ℝ) (eOne : EucSpace 1) (eOne : EucSpace 1) = 1 := by
  norm_num [eOne, EuclideanSpace.inner_single_left]

/-- The two points have negative inner product.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem inner_eNeg_eOne :
    inner (𝕜 := ℝ) (eNeg : EucSpace 1) (eOne : EucSpace 1) = -1 := by
  norm_num [eOne, eNeg, EuclideanSpace.inner_single_left]

/-- The reversed inner product is also negative.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem inner_eOne_eNeg :
    inner (𝕜 := ℝ) (eOne : EucSpace 1) (eNeg : EucSpace 1) = -1 := by
  norm_num [eOne, eNeg, EuclideanSpace.inner_single_left]

/-- The negative point has positive self inner product.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem inner_eNeg_eNeg :
    inner (𝕜 := ℝ) (eNeg : EucSpace 1) (eNeg : EucSpace 1) = 1 := by
  norm_num [eNeg, EuclideanSpace.inner_single_left]

/-- Reflection swaps the positive point with the negative point.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem sphereMap_neg_eOne :
    sphereMap 1 (LinearIsometryEquiv.neg ℝ) eOne = eNeg := by
  apply Subtype.ext
  ext i
  fin_cases i
  simp [sphereMap, eOne, eNeg]

/-- Reflection swaps the negative point with the positive point.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem sphereMap_neg_eNeg :
    sphereMap 1 (LinearIsometryEquiv.neg ℝ) eNeg = eOne := by
  apply Subtype.ext
  ext i
  fin_cases i
  simp [sphereMap, eOne, eNeg]

/-- Reflection invariance gives equal masses to the two points.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem sphere_one_atom_eq (σ : Measure (SSphere 1))
    (hσ : ∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1, σ.map (sphereMap 1 U) = σ) :
    σ {eOne} = σ {eNeg} := by
  have heq : (sphereMap 1 (LinearIsometryEquiv.neg ℝ)) ⁻¹' {eNeg} = {eOne} := by
    ext x
    rcases sphere_one_cases x with rfl | rfl
    · simp [sphereMap_neg_eOne]
    · have hne : eNeg ≠ eOne := Ne.symm eOne_ne_eNeg
      simp [sphereMap_neg_eNeg, eOne_ne_eNeg, hne]
  have hm := hσ (LinearIsometryEquiv.neg ℝ)
  have hmeas : Measurable (sphereMap 1 (LinearIsometryEquiv.neg ℝ)) :=
    measurable_sphereMap 1 _
  calc
    σ {eOne} = (Measure.map (sphereMap 1 (LinearIsometryEquiv.neg ℝ)) σ) {eNeg} := by
      rw [Measure.map_apply hmeas (measurableSet_singleton eNeg), heq]
    _ = σ {eNeg} := by rw [hm]

/-- Under an invariant probability law, each point has mass one half.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem sphere_one_atom_half (σ : Measure (SSphere 1))
    [IsProbabilityMeasure σ]
    (hσ : ∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1, σ.map (sphereMap 1 U) = σ) :
    (σ {eOne}).toReal = 1 / 2 ∧ (σ {eNeg}).toReal = 1 / 2 := by
  have huniv : ({eOne} ∪ {eNeg} : Set (SSphere 1)) = Set.univ := by
    ext x
    rcases sphere_one_cases x with rfl | rfl <;> simp
  have hdisj : Disjoint ({eOne} : Set (SSphere 1)) {eNeg} := by
    simp [eOne_ne_eNeg]
  have heq := sphere_one_atom_eq σ hσ
  have hadd := measure_union hdisj (measurableSet_singleton eNeg) (μ := σ)
  rw [huniv, measure_univ, ← heq] at hadd
  have hfin : σ {eOne} ≠ ⊤ := measure_ne_top σ _
  have hreal := congrArg ENNReal.toReal hadd
  rw [ENNReal.toReal_add hfin hfin, ENNReal.toReal_one] at hreal
  constructor
  · linarith
  · rw [← heq]
    linarith

end Transformer.Perspective

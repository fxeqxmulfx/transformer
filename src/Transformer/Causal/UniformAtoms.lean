/-
# Causal attention — the uniform law on the sphere has no atoms

The uniform probability measure `uniformSphere d` on `𝕊^{d-1}` gives no mass to a point once
`d ≥ 2`, and positive mass to every nonempty open set.  These are the two facts about it that the
diffuse initial measures of the interpolation problems are built from.
-/

import Transformer.Causal.CapMass

open scoped ENNReal Pointwise
open Real MeasureTheory

namespace Transformer
namespace Causal

/-- **The spherical part of Lebesgue measure has no atoms** in dimension `d ≥ 2`: the cone
`(0,1)·{y}` over a point lies in the line `ℝ y`, which is a proper subspace and so null. -/
theorem toSphere_singleton (d : ℕ) (hd : 2 ≤ d) (y : SSphere d) :
    (volume : Measure (EucSpace d)).toSphere {y} = 0 := by
  have hne : (Submodule.span ℝ {(y : EucSpace d)}) ≠ ⊤ := by
    intro h
    have h1 := finrank_span_singleton (K := ℝ) (ne_zero_of_mem_unit_sphere y)
    rw [h, finrank_top, finrank_euclideanSpace_fin] at h1
    omega
  have hnull : volume (Set.Ioo (0 : ℝ) 1 • (((↑) : SSphere d → EucSpace d) '' {y})) = 0 := by
    refine measure_mono_null ?_ (Measure.addHaar_submodule volume _ hne)
    rw [Set.image_singleton, Set.smul_singleton]
    rintro _ ⟨r, -, rfl⟩
    exact Submodule.smul_mem _ r (Submodule.mem_span_singleton_self _)
  rw [Measure.toSphere_apply' _ (measurableSet_singleton y), hnull, mul_zero]

/-- The hypothesis of `toSphere_singleton` is satisfiable: the circle, `d = 2`. -/
example : 2 ≤ 2 := le_rfl

/-- **The uniform law on the sphere has no atoms** in dimension `d ≥ 2`. -/
theorem uniformSphere_singleton (d : ℕ) (hd : 2 ≤ d) (y : SSphere d) :
    uniformSphere d {y} = 0 := by
  simp only [uniformSphere, Measure.smul_apply, toSphere_singleton d hd y, smul_zero]

/-- The hypothesis of `uniformSphere_singleton` is satisfiable: the circle, `d = 2`. -/
example : 2 ≤ 2 := le_rfl

/-- **The uniform law on the sphere charges every nonempty open set**: the spherical part of
Lebesgue measure does, and the normalizing constant is finite. -/
theorem uniformSphere_pos_of_isOpen (d : ℕ) {U : Set (SSphere d)} (hU : IsOpen U)
    (hne : U.Nonempty) : 0 < uniformSphere d U := by
  have htop : (volume : Measure (EucSpace d)).toSphere Set.univ ≠ ⊤ := measure_ne_top _ _
  have hpos : 0 < (volume : Measure (EucSpace d)).toSphere U :=
    hU.measure_pos _ hne
  simp only [uniformSphere, Measure.smul_apply, smul_eq_mul]
  exact ENNReal.mul_pos (ENNReal.inv_ne_zero.mpr htop) hpos.ne'

/-- The hypotheses of `uniformSphere_pos_of_isOpen` are satisfiable: the whole circle is open and
nonempty. -/
example : IsOpen (Set.univ : Set (SSphere 2)) ∧ (Set.univ : Set (SSphere 2)).Nonempty :=
  ⟨isOpen_univ, ⟨⟨EuclideanSpace.single 0 1, by simp⟩, trivial⟩⟩

end Causal
end Transformer

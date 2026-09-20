/-
# Perceptrons and attention's mean-field landscape — atomic measures on `𝕊¹`

The vocabulary `thm: bound` and `cor: bound` of arXiv:2601.21366v2 are written
in: a finite convex combination of Dirac masses, the point of `𝕊¹` at a given
angle, and `eq: atomic.thm.bound` — the presentation of a measure as `N ≥ 2`
atoms of positive mass at pairwise distinct angles of `[0,2π)`.

`atomicProb` is the constructor every witness of a finitely atomic critical
point is built from; `IsAtomicOnCircle` is a predicate of its arguments, not a
claim, and `isAtomicOnCircle_atomicProb` says the constructor satisfies it.

Source: arXiv:2601.21366v2, `eq: atomic.thm.bound`.
-/

import Transformer.Perceptron.Atomicity

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### Finite convex combinations of Dirac masses -/

/-- `Σ_i m_i δ_{z_i}` as a probability measure on `𝕊^{d-1}`, for non-negative
weights summing to `1`. -/
noncomputable def atomicProb {N : ℕ} (m : Idx N → ℝ) (z : Idx N → SSphere d)
    (hm : ∀ i, 0 ≤ m i) (hsum : ∑ i, m i = 1) : Perspective.ProbSphere d :=
  ⟨∑ i, ENNReal.ofReal (m i) • Measure.dirac (z i), by
    refine ⟨?_⟩
    simp only [Measure.coe_finsetSum, Finset.sum_apply, Measure.smul_apply, smul_eq_mul,
      measure_univ, mul_one]
    rw [← ENNReal.ofReal_sum_of_nonneg (fun i _ => hm i), hsum, ENNReal.ofReal_one]⟩

@[simp] theorem coe_atomicProb {N : ℕ} (m : Idx N → ℝ) (z : Idx N → SSphere d)
    (hm : ∀ i, 0 ≤ m i) (hsum : ∑ i, m i = 1) :
    (atomicProb m z hm hsum : Measure (SSphere d))
      = ∑ i, ENNReal.ofReal (m i) • Measure.dirac (z i) := rfl

/-- A finite convex combination of Dirac masses is purely atomic with finite
support. -/
theorem isFinitelyAtomic_atomicProb {N : ℕ} (m : Idx N → ℝ) (z : Idx N → SSphere d)
    (hm : ∀ i, 0 ≤ m i) (hsum : ∑ i, m i = 1) :
    IsFinitelyAtomic (atomicProb m z hm hsum) := by
  classical
  refine ⟨Finset.univ.image z, fun w => ∑ i ∈ Finset.univ.filter fun i => z i = w,
    ENNReal.ofReal (m i), ?_⟩
  rw [coe_atomicProb, ← Finset.sum_fiberwise_of_maps_to
    (t := Finset.univ.image z) (fun i _ => Finset.mem_image_of_mem z (Finset.mem_univ i))
    (fun i => ENNReal.ofReal (m i) • Measure.dirac (z i))]
  refine Finset.sum_congr rfl fun w _ => ?_
  rw [Finset.sum_smul]
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [(Finset.mem_filter.mp hi).2]

/-! ### The circle -/

/-- The point of `𝕊¹` at angle `θ`, in the standard frame of `ℝ²`:
`(cos θ, sin θ)`. -/
noncomputable def circlePoint (θ : ℝ) : SSphere 2 :=
  ⟨greatCircle ((basePoint 1 : SSphere 2) : EucSpace 2) secondAxis θ,
    mem_sphere_zero_iff_norm.mpr
      (norm_greatCircle norm_basePoint_one norm_secondAxis inner_basePoint_secondAxis θ)⟩

@[simp] theorem coe_circlePoint (θ : ℝ) :
    (circlePoint θ : EucSpace 2)
      = greatCircle ((basePoint 1 : SSphere 2) : EucSpace 2) secondAxis θ := rfl

/-- The cosine of the angle between two points of `𝕊¹` read off their angles is
`cos θ` at the base point. -/
theorem inner_basePoint_circlePoint (θ : ℝ) :
    inner (𝕜 := ℝ) ((basePoint 1 : SSphere 2) : EucSpace 2) (circlePoint θ : EucSpace 2)
      = Real.cos θ :=
  inner_greatCircle norm_basePoint_one inner_basePoint_secondAxis θ

/-! ### `eq: atomic.thm.bound` -/

/-- **`eq: atomic.thm.bound`.**  `μ = Σ_{i ≤ N} m_i δ_{θ_i} ∈ P(𝕊¹)` with
`N ≥ 2` atoms, weights `m_i > 0` summing to `1`, and pairwise distinct angles
`θ_i ∈ [0, 2π)`.

This is a predicate of `N, m, θ, μ`, the shape `thm: bound` and `cor: bound`
assume their critical point to have.

Source: arXiv:2601.21366v2, `eq: atomic.thm.bound`. -/
structure IsAtomicOnCircle (N : ℕ) (m θ : Idx N → ℝ)
    (μ : Perspective.ProbSphere 2) : Prop where
  /-- At least two atoms. -/
  two_le : 2 ≤ N
  /-- Every atom carries positive mass. -/
  mass_pos : ∀ i, 0 < m i
  /-- The masses sum to one. -/
  mass_sum : ∑ i, m i = 1
  /-- The angles lie in `[0, 2π)`. -/
  angle_mem : ∀ i, θ i ∈ Set.Ico 0 (2 * π)
  /-- The angles are pairwise distinct. -/
  angle_inj : Function.Injective θ
  /-- `μ` is the corresponding combination of Dirac masses. -/
  eq_sum : (μ : Measure (SSphere 2))
    = ∑ i, ENNReal.ofReal (m i) • Measure.dirac (circlePoint (θ i))

/-- `atomicProb` realizes `eq: atomic.thm.bound`: every admissible list of
masses and angles is the presentation of a measure. -/
theorem isAtomicOnCircle_atomicProb {N : ℕ} (hN : 2 ≤ N) (m θ : Idx N → ℝ)
    (hpos : ∀ i, 0 < m i) (hsum : ∑ i, m i = 1)
    (hmem : ∀ i, θ i ∈ Set.Ico 0 (2 * π)) (hinj : Function.Injective θ) :
    IsAtomicOnCircle N m θ
      (atomicProb m (fun i => circlePoint (θ i)) (fun i => (hpos i).le) hsum) :=
  { two_le := hN, mass_pos := hpos, mass_sum := hsum, angle_mem := hmem,
    angle_inj := hinj, eq_sum := rfl }

end Perceptron
end Transformer

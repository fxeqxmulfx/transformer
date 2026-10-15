/-
# Positive directions avoiding one prescribed point

ArXiv:2411.04551v3, §3.1, proof of `lem: first.quadrant`.
The second contraction needs a positive direction distinct from the
missing point of the original compact set. In dimension at least two,
normalizing two strictly positive vectors gives two distinct choices.
The positive quadrant is open, so each choice has a spherical ball
contained in the quadrant. Antipodal identities track the repelling
poles of the two original projection fields.
-/

import Transformer.Interpolation.Section3_ProjectionContraction

open Real

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A positive-coordinate candidate with one distinguished entry.
Source: arXiv:2411.04551v3, §3.1, the choice of `α` in
`lem: first.quadrant`. -/
noncomputable def quadrantSeed (i : Fin d) (r : ℝ) : EucSpace d :=
  (EuclideanSpace.equiv (Fin d) ℝ).symm (fun j => if j = i then r else 1)

/-- The distinguished coordinate records the free positive parameter.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
@[simp] theorem quadrantSeed_apply (i j : Fin d) (r : ℝ) :
    quadrantSeed i r j = if j = i then r else 1 := rfl

/-- Strictly positive parameters give strictly positive coordinates.
Source: arXiv:2411.04551v3, §3.1, `α ∈ ℚ₁` in
`lem: first.quadrant`. -/
theorem quadrantSeed_pos (i j : Fin d) (r : ℝ) (hr : 0 < r) :
    0 < quadrantSeed i r j := by
  rw [quadrantSeed_apply]
  split_ifs
  · exact hr
  · exact one_pos

/-- A positive seed is nonzero before its normalization.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem quadrantSeed_ne_zero (i : Fin d) (r : ℝ) (hr : 0 < r) :
    quadrantSeed i r ≠ 0 := by
  intro h
  have hi := congrArg (fun x : EucSpace d => x i) h
  have hp := quadrantSeed_pos i i r hr
  rw [hi] at hp
  exact (lt_irrefl 0) hp

/-- The unit normalization of the explicit seed.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
noncomputable def quadrantPoint (i : Fin d) (r : ℝ) (hr : 0 < r) : SSphere d :=
  ⟨‖quadrantSeed i r‖⁻¹ • quadrantSeed i r,
    mem_sphere_zero_iff_norm.mpr (norm_smul_inv_norm (quadrantSeed_ne_zero i r hr))⟩

/-- The normalized point has the expected coordinate ratios.
Source: arXiv:2411.04551v3, §3.1, the choice of the positive
unit direction in `lem: first.quadrant`. -/
theorem quadrantPoint_apply (i j : Fin d) (r : ℝ) (hr : 0 < r) :
    (quadrantPoint i r hr : EucSpace d) j =
      ‖quadrantSeed i r‖⁻¹ * (if j = i then r else 1) := by
  change ‖quadrantSeed i r‖⁻¹ * quadrantSeed i r j = _
  rw [quadrantSeed_apply]

/-- Normalization preserves the strictly positive coordinates.
Source: arXiv:2411.04551v3, §3.1, `α ∈ ℚ₁` in
`lem: first.quadrant`. -/
theorem quadrantPoint_mem (i : Fin d) (r : ℝ) (hr : 0 < r) :
    quadrantPoint i r hr ∈ positiveQuadrant d := by
  intro j
  change 0 < (‖quadrantSeed i r‖⁻¹ : ℝ) * quadrantSeed i r j
  exact mul_pos (inv_pos.mpr (norm_pos_iff.mpr (quadrantSeed_ne_zero i r hr)))
    (quadrantSeed_pos i j r hr)

/-- Changing just one entry from one to two changes the unit direction
when another coordinate is available.
Source: arXiv:2411.04551v3, §3.1, the choice of `α` in
`lem: first.quadrant`. -/
theorem quadrantPoint_one_ne_two (i j : Fin d) (hij : j ≠ i) :
    quadrantPoint i 1 one_pos ≠ quadrantPoint i 2 (by norm_num) := by
  intro h
  have hi := congrArg (fun x : SSphere d => (x : EucSpace d) i) h
  have hj := congrArg (fun x : SSphere d => (x : EucSpace d) j) h
  change ‖quadrantSeed i 1‖⁻¹ * quadrantSeed i 1 i =
    ‖quadrantSeed i 2‖⁻¹ * quadrantSeed i 2 i at hi
  change ‖quadrantSeed i 1‖⁻¹ * quadrantSeed i 1 j =
    ‖quadrantSeed i 2‖⁻¹ * quadrantSeed i 2 j at hj
  simp only [quadrantSeed_apply, ite_true, mul_one] at hi
  simp only [quadrantSeed_apply, hij, ite_false, mul_one] at hj
  have hp : 0 < ‖quadrantSeed i 2‖⁻¹ :=
    inv_pos.mpr (norm_pos_iff.mpr (quadrantSeed_ne_zero i 2 (by norm_num)))
  nlinarith

/-- There is always a positive direction other than a prescribed point.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`.
The explicit dimension hypothesis excludes the one-point positive
quadrant of the zero-dimensional sphere. -/
theorem exists_positiveQuadrant_ne (hd : 2 ≤ d) (ω : SSphere d) :
    ∃ α ∈ positiveQuadrant d, α ≠ ω := by
  let i : Fin d := ⟨0, by omega⟩
  let j : Fin d := ⟨1, by omega⟩
  have hij : j ≠ i := by intro h; have := congrArg Fin.val h; simp [i, j] at this
  by_cases h : quadrantPoint i 1 one_pos = ω
  · refine ⟨quadrantPoint i 2 (by norm_num), quadrantPoint_mem i 2 (by norm_num), ?_⟩
    intro h'
    exact quadrantPoint_one_ne_two i j hij (h.trans h'.symm)
  · exact ⟨quadrantPoint i 1 one_pos, quadrantPoint_mem i 1 one_pos, h⟩

/-- The source's target quadrant is open in the original sphere.
Source: arXiv:2411.04551v3, §3.1, the small cap around `α` in
`lem: first.quadrant`. -/
theorem isOpen_positiveQuadrant : IsOpen (positiveQuadrant d) := by
  have heq : positiveQuadrant d =
      ⋂ i : Fin d, {x : SSphere d | 0 < (x : EucSpace d) i} := by
    ext x
    simp only [positiveQuadrant, Set.mem_ofPred_eq, Set.mem_iInter]
    rfl
  rw [heq]
  apply isOpen_iInter_of_finite
  intro i
  exact isOpen_lt continuous_const ((EuclideanSpace.proj i).continuous.comp continuous_subtype_val)

/-- Each positive direction has a ball entirely in the positive quadrant.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem exists_ball_positiveQuadrant (α : SSphere d) (hα : α ∈ positiveQuadrant d) :
    ∃ ε : ℝ, 0 < ε ∧ Metric.ball α ε ⊆ positiveQuadrant d :=
  Metric.isOpen_iff.mp isOpen_positiveQuadrant α hα

/-- Applying the original spherical antipode twice returns the point.
Source: arXiv:2411.04551v3, §3.1, the two repelling poles in
`lem: first.quadrant`. -/
@[simp] theorem antipode_antipode (x : SSphere d) : antipode d (antipode d x) = x := by
  apply Subtype.ext
  exact neg_neg (x : EucSpace d)

/-- Distinct axes have distinct repelling poles.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem antipode_injective : Function.Injective (antipode d) :=
  Function.Involutive.injective antipode_antipode

/-- A positive target cannot contain its own repelling pole.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem antipode_notMem_quadrant_of_mem (i : Fin d) (α : SSphere d)
    (hα : α ∈ positiveQuadrant d) : antipode d α ∉ positiveQuadrant d := by
  intro hneg
  have hp := hα i
  have hn := hneg i
  change 0 < (α : EucSpace d) i at hp
  change 0 < -((α : EucSpace d) i) at hn
  linarith

/-- The normalization and avoidance hypotheses hold for genuine
positive-coordinate points on the circle. -/
example : (0 : ℝ) < 1 ∧ (1 : Fin 2) ≠ 0 ∧ 2 ≤ 2 ∧
    quadrantPoint (0 : Fin 2) 1 one_pos ∈ positiveQuadrant 2 := by
  exact ⟨one_pos, by decide, le_rfl, quadrantPoint_mem 0 1 one_pos⟩

end Transformer.Interpolation

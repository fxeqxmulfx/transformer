/-
# Basic definitions shared across the formalization

We work over `ℝ^d` realized as `EuclideanSpace ℝ (Fin d)`.  The unit sphere
`S(d) = 𝕊^{d-1} ⊂ ℝ^d` is the codomain on which Transformer particles live.

We also collect the projection operator `Proj x : ℝ^d → T_x S(d)` and the
linear-algebraic notation used throughout the survey:
* `Q, K, V` parameter matrices,
* `β > 0` inverse temperature,
* `[n] = {1,…,n}` indexed via `Fin n`.

Everything in this file is auxiliary: the actual paper-by-paper formalization
lives in the other files.
-/

import Mathlib.Analysis.InnerProductSpace.EuclideanDist
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import Mathlib.Topology.MetricSpace.Pseudo.Defs

open scoped BigOperators

namespace Transformer

/-- The ambient Euclidean space `ℝ^d`. -/
abbrev EucSpace (d : ℕ) : Type := EuclideanSpace ℝ (Fin d)

/-- The unit sphere `𝕊^{d-1} ⊂ ℝ^d`. -/
abbrev Sphere (d : ℕ) : Set (EucSpace d) := Metric.sphere (0 : EucSpace d) 1

/-- The sphere as a subtype, useful when we want a bundled object. -/
abbrev SSphere (d : ℕ) : Type := Metric.sphere (0 : EucSpace d) 1

/-- The orthogonal projection `Proj_x y = y - ⟨x,y⟩ x` of a vector `y ∈ ℝ^d`
onto the tangent space `T_x 𝕊^{d-1}`.  Defined in display below
`eq: transformerSd.QKV` in §2 of the paper:

  `Proj_x y = y - ⟨x, y⟩ x`. -/
noncomputable def proj (d : ℕ) (x y : EucSpace d) : EucSpace d :=
  y - (inner (𝕜 := ℝ) x y) • x

@[inherit_doc] notation "𝐏" => proj

/-- `Proj_x y` is orthogonal to `x` whenever `x` is a unit vector: the
projection really does land in the tangent space `T_x 𝕊^{d-1}`. -/
theorem inner_proj_eq_zero {d : ℕ} {x : EucSpace d} (hx : ‖x‖ = 1) (y : EucSpace d) :
    inner (𝕜 := ℝ) x (proj d x y) = 0 := by
  rw [proj, inner_sub_right, real_inner_smul_right,
    real_inner_self_eq_norm_mul_norm, hx]
  ring

/-- The hypothesis is satisfiable: the first standard basis vector of `ℝ^1`
has norm `1`. -/
example : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 1 := by
  simp [PiLp.norm_single]

/-- The "indexing set" `[n] = {1,…,n}`, realized as `Fin n`. -/
abbrev Idx (n : ℕ) : Type := Fin n

/-- Convenience: `n`-tuples of sphere points (particles). -/
abbrev SphereTuple (d n : ℕ) : Type := Idx n → SSphere d

/-- The first standard basis vector of `ℝ^{d+1}`, as a point of `𝕊^d`.

Every sphere of positive dimension has one, so `SSphere (d+1)` and
`SphereTuple (d+1) n` are inhabited; that is what the statements below need it
for. -/
noncomputable def basePoint (d : ℕ) : SSphere (d + 1) :=
  ⟨EuclideanSpace.single (0 : Fin (d + 1)) (1 : ℝ), by simp [PiLp.norm_single]⟩

/-- Coordinate map from a sphere tuple to a tuple of vectors. -/
noncomputable def tupleCoe {d n : ℕ} (X : SphereTuple d n) : Idx n → EucSpace d :=
  fun i => (X i : EucSpace d)

/-! ### Linear parameters

The Transformer architecture has `Query`, `Key`, `Value` matrices.  In the
abstract analysis these are arbitrary linear maps on `ℝ^d`.
-/

/-- A `d × d` parameter matrix is a continuous linear endomorphism. -/
abbrev ParamMatrix (d : ℕ) : Type := EucSpace d →L[ℝ] EucSpace d

/-- A *time-dependent* parameter matrix is a (smooth) curve of `ParamMatrix d`. -/
abbrev TimeParam (d : ℕ) : Type := ℝ → ParamMatrix d

end Transformer

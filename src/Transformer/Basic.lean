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

/-- **The projection kills the radial direction:** `Proj_x (c x) = 0` for a
unit vector `x`.  This is what makes a configuration whose attention average
is parallel to the particle itself a stationary point of the dynamics. -/
theorem proj_smul_self {d : ℕ} {x : EucSpace d} (hx : ‖x‖ = 1) (c : ℝ) :
    proj d x (c • x) = 0 := by
  rw [proj, real_inner_smul_right, real_inner_self_eq_norm_mul_norm, hx,
    mul_one, mul_one, sub_self]

/-- The hypothesis is satisfiable: the first standard basis vector of `ℝ^1`
has norm `1`. -/
example : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 1 := by
  simp [PiLp.norm_single]

/-- **The projection is a contraction:** `‖Proj_x y‖ ≤ ‖y‖` for a unit vector
`x`, since `‖Proj_x y‖² = ‖y‖² - ⟨x, y⟩²`.  This is what gives the `β = 0`
drift of §4 its Lipschitz constant. -/
theorem norm_proj_le {d : ℕ} {x : EucSpace d} (hx : ‖x‖ = 1) (y : EucSpace d) :
    ‖proj d x y‖ ≤ ‖y‖ := by
  have hsq : ‖proj d x y‖ ^ 2 = ‖y‖ ^ 2 - (inner (𝕜 := ℝ) x y) ^ 2 := by
    rw [proj, norm_sub_sq_real, real_inner_smul_right, norm_smul,
      real_inner_comm y x, hx]
    simp
    ring
  nlinarith [norm_nonneg (proj d x y), norm_nonneg y,
    sq_nonneg (inner (𝕜 := ℝ) x y), hsq]

/-- The hypothesis is satisfiable: the first standard basis vector of `ℝ^1`
has norm `1`. -/
example : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 1 := by
  simp [PiLp.norm_single]

/-- **How far two projections can be apart.**

For unit vectors `a`, `b` and a vector `v` with `‖v‖ ≤ 1`,

  `Proj_a u - Proj_b v = Proj_a (u - v) - ⟨a, v⟩ (a - b) - ⟨a - b, v⟩ b`,

whose three summands are bounded by `‖u - v‖`, `‖v‖ ‖a - b‖ ` and
`‖a - b‖ ‖v‖ ‖b‖`.  So the difference of the projections is at most
`‖u - v‖ + 2 ‖a - b‖`.  The naive split
`(u - v) - (⟨a, u⟩ a - ⟨b, v⟩ b)` gives `3` in place of `2`.

This is the estimate behind every Grönwall constant in the formalization: it
turns a bound on the drifts' arguments into a bound on the drifts.

Source: arXiv:2312.10794v5, §4 and Appendix D, the Lipschitz step preceding
`e:approxsphere` and `eq: stability.4ortho`. -/
theorem norm_proj_sub_proj_le {d : ℕ} {a b u v : EucSpace d} {δ r : ℝ}
    (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hv : ‖v‖ ≤ 1)
    (hab : ‖a - b‖ ≤ δ) (huv : ‖u - v‖ ≤ r) :
    ‖proj d a u - proj d b v‖ ≤ r + 2 * δ := by
  have hd0 : (0 : ℝ) ≤ δ := le_trans (norm_nonneg _) hab
  have hdecomp : proj d a u - proj d b v
      = proj d a (u - v)
        - ((inner (𝕜 := ℝ) a v) • (a - b) + (inner (𝕜 := ℝ) (a - b) v) • b) := by
    simp only [proj, inner_sub_right, inner_sub_left, sub_smul]
    module
  have hA : ‖proj d a (u - v)‖ ≤ r := (norm_proj_le ha _).trans huv
  have hB : ‖(inner (𝕜 := ℝ) a v) • (a - b)‖ ≤ δ := by
    rw [norm_smul, Real.norm_eq_abs]
    have h1 : |inner (𝕜 := ℝ) a v| ≤ 1 := by
      have h2 := abs_real_inner_le_norm a v
      rw [ha, one_mul] at h2
      exact h2.trans hv
    calc |inner (𝕜 := ℝ) a v| * ‖a - b‖ ≤ 1 * δ :=
          mul_le_mul h1 hab (norm_nonneg _) zero_le_one
      _ = δ := one_mul δ
  have hC : ‖(inner (𝕜 := ℝ) (a - b) v) • b‖ ≤ δ := by
    rw [norm_smul, Real.norm_eq_abs, hb, mul_one]
    refine (abs_real_inner_le_norm (a - b) v).trans ?_
    calc ‖a - b‖ * ‖v‖ ≤ δ * 1 := mul_le_mul hab hv (norm_nonneg _) hd0
      _ = δ := mul_one δ
  rw [hdecomp]
  calc ‖proj d a (u - v)
          - ((inner (𝕜 := ℝ) a v) • (a - b) + (inner (𝕜 := ℝ) (a - b) v) • b)‖
      ≤ ‖proj d a (u - v)‖
          + ‖(inner (𝕜 := ℝ) a v) • (a - b)
              + (inner (𝕜 := ℝ) (a - b) v) • b‖ := norm_sub_le _ _
    _ ≤ ‖proj d a (u - v)‖
          + (‖(inner (𝕜 := ℝ) a v) • (a - b)‖
              + ‖(inner (𝕜 := ℝ) (a - b) v) • b‖) := by
          gcongr
          exact norm_add_le _ _
    _ ≤ r + (δ + δ) := by gcongr
    _ = r + 2 * δ := by ring

/-- The hypotheses are satisfiable: `a = b = u = v` the first standard basis
vector of `ℝ^1`, with `δ = r = 0`. -/
example : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 1 ∧
    ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ ≤ 1 ∧
    ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))
      - (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ ≤ (0 : ℝ) :=
  ⟨by simp [PiLp.norm_single], by simp [PiLp.norm_single], by simp⟩

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

module

public import Transformer.Clusters.Section3_Rescaled

/-!
# The four planar corners in the codimension test

Source: arXiv:2305.05465v6, §4 `c:codim` and `eq:trans_dyn`.
The test uses four tokens with coordinates
`(-u,-u)`, `(-u,u)`, `(u,-u)`, and `(u,u)` in that order.
They are vectors in the same Euclidean space as the paper's dynamics.

The two sign functions below record the coordinates of these four
fixed directions. Multiplication by the scalar amplitude is the
ordinary vector scalar multiplication; it encodes no differential
equation or limiting behavior.

The coordinate and inner-product identities allow direct evaluation
of the attention scores. In particular, each score splits into the
sum of the two one-dimensional sign scores. The scalar multiplication
identity also verifies the paper's rescaling for the identity matrix.

For nonzero amplitude all four corners are distinct. Consequently
their image has cardinality four. This is the finite obstruction to
the conjecture's bound of three codimension-two affine subspaces:
in the plane, such subspaces are single points.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology
namespace Transformer.Clusters

/-- The first coordinates of the four fixed square directions.
Source: arXiv:2305.05465v6, §4 `c:codim`, its planar four-corner test. -/
def squareSignX : Idx 4 → ℝ := ![-1, -1, 1, 1]

/-- The second coordinates of the four fixed square directions.
Source: arXiv:2305.05465v6, §4 `c:codim`, its planar four-corner test. -/
def squareSignY : Idx 4 → ℝ := ![-1, 1, -1, 1]

/-- A corner of the ordinary planar square with amplitude `u`.
Source: arXiv:2305.05465v6, §4 `c:codim` and `eq:trans_dyn`,
the literal token configuration used to test the conjecture. -/
noncomputable def squareCorner (u : ℝ) (i : Idx 4) : EucSpace 2 :=
  u • (EuclideanSpace.single 0 (squareSignX i) + EuclideanSpace.single 1 (squareSignY i))

/-- The first coordinate of a square token.
Source: arXiv:2305.05465v6, §4 `c:codim`, the explicit test configuration. -/
theorem squareCorner_apply_zero (u : ℝ) (i : Idx 4) :
    squareCorner u i 0 = u * squareSignX i := by
  simp [squareCorner]

/-- The second coordinate of a square token.
Source: arXiv:2305.05465v6, §4 `c:codim`, the explicit test configuration. -/
theorem squareCorner_apply_one (u : ℝ) (i : Idx 4) :
    squareCorner u i 1 = u * squareSignY i := by
  simp [squareCorner]

/-- Scaling the actual square scales its amplitude.
Source: arXiv:2305.05465v6, the rescaling preceding `e:Rres`. -/
theorem smul_squareCorner (a u : ℝ) (i : Idx 4) :
    a • squareCorner u i = squareCorner (a * u) i := by
  simp only [squareCorner, smul_smul]

/-- Zero amplitude is the collapsed configuration.
Source: arXiv:2305.05465v6, `eq:trans_dyn`, the zero initial datum. -/
theorem squareCorner_zero (i : Idx 4) : squareCorner 0 i = 0 := by
  simp only [squareCorner, zero_smul]

/-- A square token depends continuously on its scalar amplitude.
Source: arXiv:2305.05465v6, `eq:trans_dyn`, the actual four-corner parameterization. -/
theorem continuous_squareCorner (i : Idx 4) : Continuous (fun u : ℝ => squareCorner u i) := by
  unfold squareCorner
  fun_prop

/-- The attention score is the sum of the two coordinate-sign scores.
Source: arXiv:2305.05465v6, `eq:P`, evaluated at two square corners. -/
theorem inner_squareCorner (u v : ℝ) (i j : Idx 4) :
    inner (𝕜 := ℝ) (squareCorner u i) (squareCorner v j) =
      u * v * (squareSignX i * squareSignX j + squareSignY i * squareSignY j) := by
  rw [PiLp.inner_apply, Fin.sum_univ_two]
  simp only [Real.inner_apply, squareCorner_apply_zero, squareCorner_apply_one]
  ring

/-- For one square amplitude the two coordinate scores each have the factor `u²`.
Source: arXiv:2305.05465v6, `eq:P`, its identity-matrix score. -/
theorem inner_squareCorner_same (u : ℝ) (i j : Idx 4) :
    inner (𝕜 := ℝ) (squareCorner u i) (squareCorner u j) =
      u ^ 2 * squareSignX i * squareSignX j + u ^ 2 * squareSignY i * squareSignY j := by
  rw [inner_squareCorner]
  ring

/-- The actual attention numerator factors into the two coordinate sign weights.
Source: arXiv:2305.05465v6, `eq:P`, the four-corner specialization. -/
theorem exp_inner_squareCorner (u : ℝ) (i j : Idx 4) :
    Real.exp (inner (𝕜 := ℝ) (squareCorner u i) (squareCorner u j)) =
      Real.exp (u ^ 2 * squareSignX i * squareSignX j) *
        Real.exp (u ^ 2 * squareSignY i * squareSignY j) := by
  rw [inner_squareCorner_same, Real.exp_add]

/-- Each coordinate direction is a sign, so its square is one.
Source: arXiv:2305.05465v6, §4 `c:codim`, the four combinations `(±1,±1)`. -/
theorem squareSign_sq (i : Idx 4) : squareSignX i ^ 2 = 1 ∧ squareSignY i ^ 2 = 1 := by
  fin_cases i <;> norm_num [squareSignX, squareSignY]

/-- Every corner has the Euclidean length of a planar diagonal.
Source: arXiv:2305.05465v6, §4 `c:codim`, the geometry of the four-corner test. -/
theorem norm_squareCorner_sq (u : ℝ) (i : Idx 4) : ‖squareCorner u i‖ ^ 2 = 2 * u ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, inner_squareCorner]
  have hx := (squareSign_sq i).1
  have hy := (squareSign_sq i).2
  nlinarith

/-- Differentiating a square token is differentiating its actual scalar amplitude.
Source: arXiv:2305.05465v6, `eq:trans_dyn`, the four-corner parameterization. -/
theorem hasDerivAt_squareCorner {u : ℝ → ℝ} {u' t : ℝ}
    (hu : HasDerivAt u u' t) (i : Idx 4) :
    HasDerivAt (fun s => squareCorner (u s) i) (squareCorner u' i) t := by
  exact hu.smul_const _

example : HasDerivAt (fun t : ℝ => t) 1 0 := hasDerivAt_id' (x := 0)

/-- Taking a scalar limit gives the literal vector limit of every corner.
Source: arXiv:2305.05465v6, §4 `c:codim`, the limits of its four-corner test. -/
theorem tendsto_squareCorner {u : ℝ → ℝ} {q : ℝ}
    (hu : Tendsto u atTop (nhds q)) (i : Idx 4) :
    Tendsto (fun t => squareCorner (u t) i) atTop (nhds (squareCorner q i)) := by
  exact hu.smul_const _

example : Tendsto (fun _ : ℝ => (1 : ℝ)) atTop (nhds 1) := tendsto_const_nhds

/-- Nonzero amplitude gives four distinct corners.
Source: arXiv:2305.05465v6, §4 `c:codim`, the finite obstruction
in the planar four-corner test of its bound of three subspaces. -/
theorem squareCorner_injective {q : ℝ} (hq : q ≠ 0) : Function.Injective (squareCorner q) := by
  intro i j hij
  have hx := congrArg (fun z : EucSpace 2 => z 0) hij
  have hy := congrArg (fun z : EucSpace 2 => z 1) hij
  simp only [squareCorner_apply_zero, squareCorner_apply_one] at hx hy
  fin_cases i <;> fin_cases j <;> norm_num [squareSignX, squareSignY] at hx hy ⊢ <;>
    exfalso <;> apply hq <;> linarith

example : (1 : ℝ) ≠ 0 := one_ne_zero

/-- The finite set of limiting corners has cardinality four whenever the amplitude is nonzero.
Source: arXiv:2305.05465v6, §4 `c:codim`, the square test of its numerical bound. -/
theorem card_squareCorner_image {q : ℝ} (hq : q ≠ 0) :
    (Finset.univ.image (squareCorner q)).card = 4 := by
  classical
  rw [Finset.card_image_of_injective _ (squareCorner_injective hq)]
  norm_num

example : (1 : ℝ) ≠ 0 := one_ne_zero

end Transformer.Clusters

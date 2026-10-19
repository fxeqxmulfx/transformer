module

public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Analysis.SpecialFunctions.Exp
public import Mathlib.Analysis.Complex.ExponentialBounds
public import Mathlib.Tactic

/-!
# Weighted velocity differences and the source's strictness constant

ArXiv:2601.21366v2, §5.5, `rem:strictSOPD-perceptron`.
The displayed probability identity for `(ξ(θ)-ξ(φ))²` removes the
extra factor that a pointwise estimate of each square would lose.
This file proves that identity for arbitrary finite weights of
total mass one, then applies it to the finite-sum Hessian formula.

The matrix of pair curvatures and the potential curvatures are
explicit inputs to this algebraic step. They must subsequently be
identified with the actual derivatives of the kernel and potential.
The lower bound uses `K_β'' ≥ -β exp β` and `V'' ≥ 2 exp β+2κ`,
and retains exactly the coefficient `κ` claimed by the source.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.Perceptron

variable {ι : Type*}

/-- **The exact weighted pair-square identity.**
Source: arXiv:2601.21366v2, §5.5, the probability identity displayed
in `rem:strictSOPD-perceptron`. Only total mass one is required
for the equality; nonnegativity enters the ensuing estimates. -/
theorem weighted_pair_square_identity (F : Finset ι) (m s : ι → ℝ)
    (hmass : ∑ i ∈ F, m i = 1) :
    (∑ i ∈ F, ∑ j ∈ F, m i * m j * (s i - s j) ^ 2) =
      2 * (∑ i ∈ F, m i * s i ^ 2) - 2 * (∑ i ∈ F, m i * s i) ^ 2 := by
  have hexpand (i j : ι) : m i * m j * (s i - s j) ^ 2 =
      m i * s i ^ 2 * m j + m i * (m j * s j ^ 2) -
        (2 * (m i * s i)) * (m j * s j) := by ring
  simp_rw [hexpand, Finset.sum_sub_distrib, Finset.sum_add_distrib,
    ← Finset.mul_sum, ← Finset.sum_mul]
  rw [hmass]
  rw [← Finset.mul_sum]
  ring

/-- **The pair-square sum is at most twice the weighted squared norm.**
Source: arXiv:2601.21366v2, §5.5, the inequality following the
probability identity in `rem:strictSOPD-perceptron`. -/
theorem weighted_pair_square_le (F : Finset ι) (m s : ι → ℝ)
    (hmass : ∑ i ∈ F, m i = 1) :
    (∑ i ∈ F, ∑ j ∈ F, m i * m j * (s i - s j) ^ 2) ≤
      2 * (∑ i ∈ F, m i * s i ^ 2) := by
  rw [weighted_pair_square_identity F m s hmass]
  nlinarith [sq_nonneg (∑ i ∈ F, m i * s i)]

/-- **The weighted squared norm is nonnegative.**
Source: arXiv:2601.21366v2, §5.5, the `L²(μ)` norm in
`rem:strictSOPD-perceptron`. Zero weights and speeds are allowed. -/
theorem weighted_speed_square_nonneg (F : Finset ι) (m s : ι → ℝ)
    (hm : ∀ i ∈ F, 0 ≤ m i) : 0 ≤ ∑ i ∈ F, m i * s i ^ 2 := by
  exact Finset.sum_nonneg fun i hi => mul_nonneg (hm i hi) (sq_nonneg _)

/-- **The weighted pair-square sum is nonnegative.**
Source: arXiv:2601.21366v2, §5.5, the squared velocity difference
in `rem:strictSOPD-perceptron`. This is valid without normalizing
the total mass, since every actual probability weight is nonnegative. -/
theorem weighted_pair_square_nonneg (F : Finset ι) (m s : ι → ℝ)
    (hm : ∀ i ∈ F, 0 ≤ m i) :
    0 ≤ ∑ i ∈ F, ∑ j ∈ F, m i * m j * (s i - s j) ^ 2 := by
  exact Finset.sum_nonneg fun i hi => Finset.sum_nonneg fun j hj =>
    mul_nonneg (mul_nonneg (hm i hi) (hm j hj)) (sq_nonneg _)

/-- **The source's kernel curvature bound controls the pair sum.**
Source: arXiv:2601.21366v2, §5.5, the Hessian estimate in
`rem:strictSOPD-perceptron`. Here `C` is the absolute lower-bound
constant before the interaction prefactor `1/(2β)` is applied. -/
theorem weighted_pair_curvature_lower (F : Finset ι) (m s : ι → ℝ)
    (k : ι → ι → ℝ) (C : ℝ) (hC : 0 ≤ C)
    (hm : ∀ i ∈ F, 0 ≤ m i) (hmass : ∑ i ∈ F, m i = 1)
    (hk : ∀ i ∈ F, ∀ j ∈ F, -C ≤ k i j) :
    -(2 * C) * (∑ i ∈ F, m i * s i ^ 2) ≤
      ∑ i ∈ F, ∑ j ∈ F, m i * m j * k i j * (s i - s j) ^ 2 := by
  have hpair := weighted_pair_square_le F m s hmass
  have hscaled := mul_le_mul_of_nonpos_left hpair (neg_nonpos.mpr hC)
  have hpoint : (-C) * (∑ i ∈ F, ∑ j ∈ F, m i * m j * (s i - s j) ^ 2) ≤
      ∑ i ∈ F, ∑ j ∈ F, m i * m j * k i j * (s i - s j) ^ 2 := by
    simp_rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i hi
    apply Finset.sum_le_sum
    intro j hj
    have h := mul_le_mul_of_nonneg_right (hk i hi j hj)
      (mul_nonneg (mul_nonneg (hm i hi) (hm j hj)) (sq_nonneg (s i - s j)))
    convert h using 1; ring
  calc
    -(2 * C) * (∑ i ∈ F, m i * s i ^ 2) =
        (-C) * (2 * (∑ i ∈ F, m i * s i ^ 2)) := by ring
    _ ≤ (-C) * (∑ i ∈ F, ∑ j ∈ F, m i * m j * (s i - s j) ^ 2) := hscaled
    _ ≤ _ := hpoint

/-- **The source's potential bound yields strictness for the finite Hessian.**
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`, and
§5.5, `eq:strictSOPD-sufficient`. This is the quantitative algebra
after the actual second derivative has been computed.
There is no positivity assumption on `κ` in this algebraic step. -/
theorem weighted_hessian_lower (F : Finset ι) (m s : ι → ℝ)
    (k : ι → ι → ℝ) (V : ι → ℝ) (β κ : ℝ) (hβ : 0 < β)
    (hm : ∀ i ∈ F, 0 ≤ m i) (hmass : ∑ i ∈ F, m i = 1)
    (hk : ∀ i ∈ F, ∀ j ∈ F, -(β * Real.exp β) ≤ k i j)
    (hV : ∀ i ∈ F, 2 * Real.exp β + 2 * κ ≤ V i) :
    κ * (∑ i ∈ F, m i * s i ^ 2) ≤
      (2 * β)⁻¹ * (∑ i ∈ F, ∑ j ∈ F, m i * m j * k i j * (s i - s j) ^ 2) +
        (1 / 2 : ℝ) * (∑ i ∈ F, m i * V i * s i ^ 2) := by
  have hpair := weighted_pair_curvature_lower F m s k (β * Real.exp β)
    (by positivity) hm hmass hk
  have hinteraction := mul_le_mul_of_nonneg_left hpair
    (show 0 ≤ (2 * β)⁻¹ by positivity)
  have hcancel : (2 * β)⁻¹ * (-(2 * (β * Real.exp β)) *
      (∑ i ∈ F, m i * s i ^ 2)) = -Real.exp β * (∑ i ∈ F, m i * s i ^ 2) := by
    field_simp
  rw [hcancel] at hinteraction
  have hpotential : (2 * Real.exp β + 2 * κ) * (∑ i ∈ F, m i * s i ^ 2) ≤
      ∑ i ∈ F, m i * V i * s i ^ 2 := by
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i hi
    have h := mul_le_mul_of_nonneg_right (hV i hi)
      (mul_nonneg (hm i hi) (sq_nonneg (s i)))
    convert h using 1; ring
  linarith

/-- All weighted-identity and pair-curvature hypotheses are jointly
realized by a singleton probability with arbitrary actual speed.
The pair curvature is the sharp negative self-curvature at `β=1`. -/
example (s : Unit → ℝ) : let F : Finset Unit := {()}
    let m : Unit → ℝ := fun _ => 1
    (∑ i ∈ F, m i = 1) ∧ (∀ i ∈ F, 0 ≤ m i) ∧ (0 : ℝ) ≤ Real.exp 1 ∧
    (∀ i ∈ F, ∀ j ∈ F, -Real.exp 1 ≤ (fun _ _ : Unit => -Real.exp 1) i j) ∧
    0 ≤ ∑ i ∈ F, m i * s i ^ 2 := by
  simp [sq_nonneg, Real.exp_nonneg]

/-- The complete finite-Hessian inputs are realized by one unit-mass
atom, the self-curvature `-exp 1`, and potential curvature eight.
These are the curvatures of the quadratic Dirac witness used above. -/
example : let F : Finset Unit := {()}
    let m : Unit → ℝ := fun _ => 1
    (0 : ℝ) < 1 ∧ (∀ i ∈ F, 0 ≤ m i) ∧ (∑ i ∈ F, m i = 1) ∧
    (∀ i ∈ F, ∀ j ∈ F, -(1 * Real.exp 1) ≤
      (fun _ _ : Unit => -Real.exp 1) i j) ∧
    (∀ i ∈ F, 2 * Real.exp 1 + 2 * 1 ≤ (fun _ : Unit => (8 : ℝ)) i) := by
  dsimp
  refine ⟨one_pos, fun i hi => zero_le_one, by simp,
    fun i hi j hj => by simp, fun i hi => ?_⟩
  linarith [Real.exp_one_lt_three]

end Transformer.Perceptron

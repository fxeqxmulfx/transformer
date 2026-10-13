/-
# ReLU activation patterns and their actual circle forces

ArXiv:2601.21366v2, §6, proof of `prop: unified.log`.
The source works on components cut out by neuron hyperplanes.
Here the finitely many activation patterns use weak inequalities.
A zero neuron or a point on an activation boundary consequently
belongs to valid patterns, and its zero contribution is retained.

The branch potential is half the source's quadratic polynomial and
the branch force is its actual angular derivative.
At an angle with the supplied pattern, that force agrees with
the original ReLU perceptron force. Every infinite set of angles
contains infinitely many angles in one pattern, by finite covering.
-/

import Transformer.Perceptron.Section5_CirclePotential
import Transformer.Perceptron.SignedGram
import Mathlib.Data.Set.Finite.Lattice

open Real Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- An activation pattern allows zero inner products on either side.
Source: arXiv:2601.21366v2, §6, the active index set `J_I`.
Weak inequalities include hyperplane boundaries and zero neurons;
their actual ReLU contributions agree in both adjacent branches. -/
def IsActivationPattern {d : ℕ} (a : Idx d → EucSpace d) (J : Finset (Idx d))
    (x : EucSpace d) : Prop :=
  (∀ j ∈ J, 0 ≤ inner (𝕜 := ℝ) (a j) x) ∧
    ∀ j, j ∉ J → inner (𝕜 := ℝ) (a j) x ≤ 0

/-- Every actual point has an activation pattern, including boundary
points. Source: arXiv:2601.21366v2, §6, the active index sets `J_I`.
The strictly positive inner products give one canonical choice. -/
theorem exists_activationPattern {d : ℕ} (a : Idx d → EucSpace d) (x : EucSpace d) :
    ∃ J : Finset (Idx d), IsActivationPattern a J x := by
  classical
  refine ⟨Finset.univ.filter (fun j => 0 < inner (𝕜 := ℝ) (a j) x), ?_, ?_⟩
  · intro j hj
    exact (Finset.mem_filter.mp hj).2.le
  · intro j hj
    exact le_of_not_gt (fun hp => hj (Finset.mem_filter.mpr ⟨Finset.mem_univ j, hp⟩))

/-- Negating the actual point gives the complementary activation pattern.
Source: arXiv:2601.21366v2, §6, the antipodal component `I'`.
The claim also holds when one or more neuron inner products are zero. -/
theorem isActivationPattern_compl_neg {d : ℕ} (a : Idx d → EucSpace d)
    (J : Finset (Idx d)) (x : EucSpace d) (hx : IsActivationPattern a J x) :
    IsActivationPattern a Jᶜ (-x) := by
  classical
  constructor
  · intro j hj
    rw [inner_neg_right]
    exact neg_nonneg.mpr (hx.2 j (Finset.mem_compl.mp hj))
  · intro j hj
    rw [inner_neg_right]
    exact neg_nonpos.mpr (hx.1 j (by simpa only [Finset.mem_compl, not_not] using hj))

/-- One activation pattern contains infinitely many of any infinite set
of actual angles. Source: arXiv:2601.21366v2, §6, the finite component
selection in part (b) of the proof of `prop: unified.log`.
The finite pattern cover includes all boundaries and zero neurons. -/
theorem exists_infinite_activation_angles (a : Idx 2 → EucSpace 2)
    (S : Set ℝ) (hS : S.Infinite) :
    ∃ J : Finset (Idx 2), (S ∩ {θ | IsActivationPattern a J (circlePoint θ : EucSpace 2)}).Infinite := by
  classical
  by_contra h
  have hfinite : ∀ J : Finset (Idx 2),
      (S ∩ {θ | IsActivationPattern a J (circlePoint θ : EucSpace 2)}).Finite := by
    intro J
    by_contra hJ
    exact h ⟨J, hJ⟩
  apply hS
  apply (finite_iUnion hfinite).subset
  intro θ hθ
  obtain ⟨J, hJ⟩ := exists_activationPattern a (circlePoint θ : EucSpace 2)
  exact mem_iUnion.mpr ⟨J, hθ, hJ⟩

/-- One half of the branch's actual quadratic potential in the angle.
Source: arXiv:2601.21366v2, §6, the branch term in `H_log`. -/
noncomputable def circleBranchHalfPotential (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (J : Finset (Idx 2)) (θ : ℝ) : ℝ :=
  (1 / 2 : ℝ) * ∑ j ∈ J, ω j * inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2) ^ 2

/-- The branch's actual scalar angular force. Source: arXiv:2601.21366v2,
§6, the perceptron term of `g_log`, before division by the positive weight. -/
noncomputable def circleBranchForce (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (J : Finset (Idx 2)) (θ : ℝ) : ℝ :=
  ∑ j ∈ J, ω j * (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2) *
    inner (𝕜 := ℝ) (a j) (circleVel θ))

/-- The branch force is the actual derivative of the half potential.
Source: arXiv:2601.21366v2, §6, differentiating the branch term of
`H_log`. No activation or support assumption is needed for the polynomial. -/
theorem hasDerivAt_circleBranchHalfPotential (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (J : Finset (Idx 2)) (θ : ℝ) :
    HasDerivAt (circleBranchHalfPotential ω a J) (circleBranchForce ω a J θ) θ := by
  have hterm (j : Idx 2) := ((hasDerivAt_inner_circlePoint (a j) θ).pow 2).const_mul (ω j)
  have h := (HasDerivAt.fun_sum (u := J) (fun j _ => hterm j)).const_mul (1 / 2 : ℝ)
  apply h.congr_deriv
  unfold circleBranchForce
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- At an angle with a valid pattern the original ReLU force equals the
actual polynomial branch force. Source: arXiv:2601.21366v2, §6,
the branch restriction of `g_log`. Boundary contributions are actually zero. -/
theorem circlePotentialForce_relu_eq_branch (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (J : Finset (Idx 2)) (θ : ℝ)
    (hθ : IsActivationPattern a J (circlePoint θ : EucSpace 2)) :
    circlePotentialForce (fun s => max s 0) ω a (circlePoint θ) = circleBranchForce ω a J θ := by
  classical
  unfold circlePotentialForce circleBranchForce
  rw [circleTangent_circlePoint]
  symm
  calc
    _ = ∑ j ∈ J, ω j * (max (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) 0 *
        inner (𝕜 := ℝ) (a j) (circleVel θ)) :=
      Finset.sum_congr rfl (fun j hj => by rw [max_eq_left (hθ.1 j hj)])
    _ = _ := Finset.sum_subset (Finset.subset_univ J) (fun j _ hj => by
      rw [max_eq_right (hθ.2 j hj), zero_mul, mul_zero])

/-- All pattern hypotheses hold at the actual first basis vector for
the source's non-degenerate one-neuron ReLU example, with empty active set.
The second zero neuron is on its boundary. The infinite-set selection input
is realized on the full angle line. Source: arXiv:2601.21366v2, §6,
`eq: nondeg.normalized` and the proof of `prop: unified.log`.
All inputs of the pattern and infinite-set statements are realized together. -/
example : IsNonDegenerate (Pi.single 0 1)
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∧
    IsActivationPattern
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∅
      ((basePoint 1 : SSphere 2) : EucSpace 2) ∧
    IsActivationPattern
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∅
      (circlePoint 0 : EucSpace 2) ∧ (univ : Set ℝ).Infinite := by
  have h (x : EucSpace 2) (hx : x = ((basePoint 1 : SSphere 2) : EucSpace 2)) :
      IsActivationPattern (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∅ x := by
    subst x
    refine ⟨fun j hj => by simp at hj, ?_⟩
    intro j _
    fin_cases j <;> simp [basePoint]
  exact ⟨isNonDegenerate_pin, h _ rfl,
    h _ (by simp [circlePoint, greatCircle]), infinite_univ⟩

end Transformer.Perceptron

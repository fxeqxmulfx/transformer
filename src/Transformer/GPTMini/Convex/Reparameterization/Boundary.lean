import Transformer.GPTMini.Convex.Reparameterization.LogOdds

/-!
# No convex likelihood coordinates for the unchanged scalar-head class

New prediction-class counterexample derived from arXiv:2211.11052v1, §3's
shared-key softmax product and Boyd and Vandenberghe (2004), §3.5.
Any parameterization into the original two-query, three-key scalar-head
class that covers the two concrete endpoint heads has at least one
nonconvex category negative log likelihood.

The coordinates may be nonlinear, redundant, and arbitrarily numerous.
No continuity, smoothness, injectivity, affine-logit, or finite-dimension
assumption is made. A midpoint must remain in the original prediction
class, but likelihood convexity forces its log-odds minor to be positive.
Thus merely changing the coordinates cannot convexify this whole class.

This is a fixed-one-hot-value shared-memory control, not a theorem about
every causal GPTMini, every head width, learned values, or a particular
dataset's aggregate loss. A changed prediction class remains admissible.
-/

noncomputable section

namespace Transformer.GPTMini.Convex.Reparameterization

open scoped BigOperators

/-- Membership in the original physical head class entails actual positive predictions.
Source: §3's finite softmax formula, not an additional probability-domain hypothesis. -/
theorem scalarHeadClass_pos {p : SharedProbability} (hp : ScalarHeadClass p)
    (r : Fin 2) (c : Fin 3) : 0 < p r c := by
  obtain ⟨q, k, h⟩ := hp
  rw [← h]
  exact scalarHeadProbability_pos q k r c

example : ScalarHeadClass leftProbability := by
  exact ⟨leftQueries, leftKeys, rfl⟩

/-- Membership in the physical head class entails actual row normalization.
Source: §3's finite softmax normalizer, independent of factor coordinates. -/
theorem scalarHeadClass_sum {p : SharedProbability} (hp : ScalarHeadClass p)
    (r : Fin 2) : ∑ c, p r c = 1 := by
  obtain ⟨q, k, h⟩ := hp
  rw [← h]
  exact scalarHeadProbability_sum q k r

example : ScalarHeadClass rightProbability := by
  exact ⟨rightQueries, rightKeys, rfl⟩

/-- With log probabilities as logits, ordinary log-sum-exp cross entropy is the tested NLL.
Source: the new shared-memory control's normalized predictions and §3.5.
This specifies the control's readout; it does not identify it with GPTMini's learned readout. -/
theorem scalarHeadClass_crossEntropy {p : SharedProbability} (hp : ScalarHeadClass p)
    (r : Fin 2) (c : Fin 3) :
    Real.log (∑ j, Real.exp (Real.log (p r j))) - Real.log (p r c) =
      -Real.log (p r c) := by
  have he : ∀ j, Real.exp (Real.log (p r j)) = p r j := by
    intro j
    exact Real.exp_log (scalarHeadClass_pos hp r j)
  simp only [he]
  rw [scalarHeadClass_sum hp r, Real.log_one, zero_sub]

example : ScalarHeadClass leftProbability := by
  exact ⟨leftQueries, leftKeys, rfl⟩

/-- The original free factor parameter space of the shared-memory scalar-head control.
Source: arXiv:2211.11052v1, §3; this control has two Q factors and three K factors. -/
abbrev ScalarFactors := (Fin 2 → ℝ) × (Fin 3 → ℝ)

/-- The actual normalized head predictions evaluated at all original free factors.
Source: §3's scalar Q/K softmax formula; values are fixed category basis vectors. -/
def factorProbability (θ : ScalarFactors) : SharedProbability :=
  scalarHeadProbability θ.1 θ.2

/-- Every original factor assignment belongs to the unchanged physical prediction class.
Source: the actual Q/K factorization used in the class's existential predicate. -/
theorem factorProbability_physical (θ : ScalarFactors) :
    ScalarHeadClass (factorProbability θ) := by
  exact ⟨θ.1, θ.2, rfl⟩

/-- The first endpoint is attained by finite original factor tables.
Source: the new exact softmax control, with scores log 2 and zero. -/
theorem factorProbability_covers_left :
    ∃ θ : ScalarFactors, factorProbability θ = leftProbability := by
  exact ⟨(leftQueries, leftKeys), rfl⟩

/-- The second endpoint is attained by finite original factor tables.
Source: the new exact softmax control, with scores log 2 and zero. -/
theorem factorProbability_covers_right :
    ∃ θ : ScalarFactors, factorProbability θ = rightProbability := by
  exact ⟨(rightQueries, rightKeys), rfl⟩

/-- No parameterization staying in the scalar-head class and covering both witnesses
has convex category likelihoods throughout its real parameter space.
Source: the new coordinate-independent prediction-class counterexample from §3 and §3.5.
The parameter space may have any dimension and the prediction map need not be affine. -/
theorem scalarHeadClass_no_convex_coordinates {E : Type*} [AddCommGroup E] [Module ℝ E]
    (p : E → SharedProbability) (hphysical : ∀ θ, ScalarHeadClass (p θ))
    (hleft : ∃ θ, p θ = leftProbability) (hright : ∃ θ, p θ = rightProbability) :
    ¬∀ r c, ConvexOn ℝ Set.univ (fun θ => -Real.log (p θ r c)) := by
  intro hconvex
  obtain ⟨x, hx⟩ := hleft
  obtain ⟨y, hy⟩ := hright
  let m : E := (1 / 2 : ℝ) • x + (1 / 2 : ℝ) • y
  have hproduct : ∀ r c, leftProbability r c * rightProbability r c ≤ p m r c ^ 2 := by
    intro r c
    have h := nll_midpoint_product (fun θ => p θ r c)
      (fun θ => scalarHeadClass_pos (hphysical θ) r c) (hconvex r c) x y
    rw [hx, hy] at h
    exact h
  exact bridge_not_scalarHead (p m)
    (scalarHeadClass_pos (hphysical m)) (scalarHeadClass_sum (hphysical m))
    hproduct (hphysical m)

example :
    (∀ θ : ScalarFactors, ScalarHeadClass (factorProbability θ)) ∧
    (∃ θ : ScalarFactors, factorProbability θ = leftProbability) ∧
    (∃ θ : ScalarFactors, factorProbability θ = rightProbability) := by
  exact ⟨factorProbability_physical, factorProbability_covers_left,
    factorProbability_covers_right⟩

/-- At least one category likelihood is nonconvex in the original free Q/K factors.
Source: the new class obstruction, witnessed by two actual physical heads rather than
only the simultaneous sign symmetry of the original factors. -/
theorem factorProbability_not_all_convex :
    ¬∀ r c, ConvexOn ℝ Set.univ (fun θ : ScalarFactors =>
      -Real.log (factorProbability θ r c)) := by
  exact scalarHeadClass_no_convex_coordinates factorProbability
    factorProbability_physical factorProbability_covers_left factorProbability_covers_right

/-- Even nonlinear coordinates mapping back to original factors cannot fix both endpoint heads.
Source: the new unchanged-class obstruction. Only endpoint coverage is needed;
surjectivity onto all factors, differentiability and coordinate invertibility are unnecessary. -/
theorem factorProbability_coordinate_change {E : Type*} [AddCommGroup E] [Module ℝ E]
    (F : E → ScalarFactors)
    (hleft : ∃ θ, F θ = (leftQueries, leftKeys))
    (hright : ∃ θ, F θ = (rightQueries, rightKeys)) :
    ¬∀ r c, ConvexOn ℝ Set.univ (fun θ =>
      -Real.log (factorProbability (F θ) r c)) := by
  apply scalarHeadClass_no_convex_coordinates (fun θ => factorProbability (F θ))
    (fun θ => factorProbability_physical (F θ))
  · obtain ⟨θ, hθ⟩ := hleft
    refine ⟨θ, ?_⟩
    rw [hθ]
    rfl
  · obtain ⟨θ, hθ⟩ := hright
    refine ⟨θ, ?_⟩
    rw [hθ]
    rfl

example :
    (∃ θ : ScalarFactors, id θ = (leftQueries, leftKeys)) ∧
    (∃ θ : ScalarFactors, id θ = (rightQueries, rightKeys)) := by
  exact ⟨⟨(leftQueries, leftKeys), rfl⟩, ⟨(rightQueries, rightKeys), rfl⟩⟩

end Transformer.GPTMini.Convex.Reparameterization

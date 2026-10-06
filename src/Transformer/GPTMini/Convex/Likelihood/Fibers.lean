import Transformer.GPTMini.Convex.Likelihood.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Convex categorical likelihoods force convex prediction fibers

Derived from Boyd and Vandenberghe (2004), §3.5's log-concave probability
criterion, applied to strictly positive normalized categorical predictions.
If every possible observed-category negative log likelihood is convex, two
parameter assignments with identical predictions have identical predictions
throughout their connecting segment. Normalization is essential: convexity
first makes every interior probability at least its common endpoint value,
and total mass one forces equality in every coordinate.

This does not assume affine logits. It gives a necessary test for the compact
nonlinear route, without claiming every particular dataset loss is nonconvex.
In particular an even prediction family on the entire unrestricted parameter
space must be constant if all its category likelihoods are convex. Symmetric
query/key products therefore need a changed parameterization, not merely
a new positive normalizer, to escape this test.
-/

noncomputable section

namespace Transformer.GPTMini.Convex.Likelihood

open scoped BigOperators

/-- A normalized likelihood family preserves equal predictions along parameter segments.
Source: the new categorical consequence of Boyd and Vandenberghe (2004), §3.5.
All observed categories must have convex losses; this is stronger than one dataset's sum. -/
theorem normalizedNLL_fiber {E : Type*} [AddCommGroup E] [Module ℝ E] {C : ℕ}
    (p : E → Fin C → ℝ) (hpos : ∀ θ c, 0 < p θ c)
    (hsum : ∀ θ, ∑ c, p θ c = 1)
    (hconvex : ∀ c, ConvexOn ℝ Set.univ (fun θ => -Real.log (p θ c)))
    (x y : E) (hxy : p x = p y) (a b : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    p (a • x + b • y) = p x := by
  have hle : ∀ c, p x c ≤ p (a • x + b • y) c := by
    intro c
    have h := (hconvex c).2 (Set.mem_univ x) (Set.mem_univ y) ha hb hab
    have hy : p y c = p x c := congrFun hxy.symm c
    simp only [hy, smul_eq_mul] at h
    have he : a * (-Real.log (p x c)) + b * (-Real.log (p x c)) =
        -Real.log (p x c) := by rw [← add_mul, hab, one_mul]
    rw [he, neg_le_neg_iff] at h
    exact (Real.log_le_log_iff (hpos x c) (hpos (a • x + b • y) c)).mp h
  have he : (∑ c, p x c) = ∑ c, p (a • x + b • y) c := by
    rw [hsum, hsum]
  have hc := (Finset.sum_eq_sum_iff_of_le (fun c hc => hle c)).mp he
  funext c
  exact (hc c (Finset.mem_univ c)).symm

example :
    (∀ θ c, 0 < threeRouteProbability θ c) ∧
    (∀ θ, ∑ c, threeRouteProbability θ c = 1) ∧
    (∀ c, ConvexOn ℝ Set.univ (fun θ => -Real.log (threeRouteProbability θ c))) ∧
    threeRouteProbability (0, 0) = threeRouteProbability (0, 0) ∧
    (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) + 1 / 2 = 1 := by
  refine ⟨threeRouteProbability_pos, threeRouteProbability_sum, ?_, rfl,
    by norm_num, by norm_num, by norm_num⟩
  intro c
  exact threeRouteNLL_convex c

/-- Equal endpoint predictions give a flat loss along the whole segment for every category.
Source: the new §3.5 normalized fiber consequence. Convex likelihoods do
not eliminate flat directions arising from identical predictions. -/
theorem normalizedNLL_equal_loss_segment {E : Type*}
    [AddCommGroup E] [Module ℝ E] {C : ℕ}
    (p : E → Fin C → ℝ) (hpos : ∀ θ c, 0 < p θ c)
    (hsum : ∀ θ, ∑ c, p θ c = 1)
    (hconvex : ∀ c, ConvexOn ℝ Set.univ (fun θ => -Real.log (p θ c)))
    (x y : E) (hxy : p x = p y) (a b : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) (c : Fin C) :
    -Real.log (p (a • x + b • y) c) = -Real.log (p x c) := by
  exact congrArg (fun v => -Real.log (v c))
    (normalizedNLL_fiber p hpos hsum hconvex x y hxy a b ha hb hab)

example :
    (∀ θ c, 0 < threeRouteProbability θ c) ∧
    (∀ θ, ∑ c, threeRouteProbability θ c = 1) ∧
    (∀ c, ConvexOn ℝ Set.univ (fun θ => -Real.log (threeRouteProbability θ c))) ∧
    threeRouteProbability (0, 0) = threeRouteProbability (0, 0) ∧
    (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) + 1 / 2 = 1 := by
  refine ⟨threeRouteProbability_pos, threeRouteProbability_sum, ?_, rfl,
    by norm_num, by norm_num, by norm_num⟩
  intro c
  exact threeRouteNLL_convex c

/-- Each set of parameter assignments with a specified prediction is convex.
Source: the new normalized categorical fiber theorem, with no linearity assumption. -/
theorem normalizedNLL_convex_fibers {E : Type*} [AddCommGroup E] [Module ℝ E] {C : ℕ}
    (p : E → Fin C → ℝ) (hpos : ∀ θ c, 0 < p θ c)
    (hsum : ∀ θ, ∑ c, p θ c = 1)
    (hconvex : ∀ c, ConvexOn ℝ Set.univ (fun θ => -Real.log (p θ c)))
    (z : Fin C → ℝ) : Convex ℝ {θ | p θ = z} := by
  intro x hx y hy a b ha hb hab
  change p (a • x + b • y) = z
  exact (normalizedNLL_fiber p hpos hsum hconvex x y (hx.trans hy.symm) a b ha hb hab).trans hx

example :
    (∀ θ c, 0 < threeRouteProbability θ c) ∧
    (∀ θ, ∑ c, threeRouteProbability θ c = 1) ∧
    (∀ c, ConvexOn ℝ Set.univ (fun θ => -Real.log (threeRouteProbability θ c))) := by
  refine ⟨threeRouteProbability_pos, threeRouteProbability_sum, ?_⟩
  intro c
  exact threeRouteNLL_convex c

/-- Equal predictions at opposite parameters must equal the zero-parameter prediction.
Source: the new §3.5 categorical fiber consequence at the midpoint.
This is a local symmetry test; no global evenness hypothesis is needed. -/
theorem normalizedNLL_opposite {E : Type*} [AddCommGroup E] [Module ℝ E] {C : ℕ}
    (p : E → Fin C → ℝ) (hpos : ∀ θ c, 0 < p θ c)
    (hsum : ∀ θ, ∑ c, p θ c = 1)
    (hconvex : ∀ c, ConvexOn ℝ Set.univ (fun θ => -Real.log (p θ c)))
    (θ : E) (hsym : p θ = p (-θ)) : p θ = p 0 := by
  have h := normalizedNLL_fiber p hpos hsum hconvex θ (-θ) hsym
    (1 / 2) (1 / 2) (by norm_num) (by norm_num) (by norm_num)
  have hm : (1 / 2 : ℝ) • θ + (1 / 2 : ℝ) • (-θ) = 0 := by
    rw [smul_neg, add_neg_cancel]
  rw [hm] at h
  exact h.symm

example :
    (∀ θ c, 0 < threeRouteProbability θ c) ∧
    (∀ θ, ∑ c, threeRouteProbability θ c = 1) ∧
    (∀ c, ConvexOn ℝ Set.univ (fun θ => -Real.log (threeRouteProbability θ c))) ∧
    threeRouteProbability (0 : ℝ × ℝ) = threeRouteProbability (-(0 : ℝ × ℝ)) := by
  refine ⟨threeRouteProbability_pos, threeRouteProbability_sum, ?_, ?_⟩
  · intro c
    exact threeRouteNLL_convex c
  · rw [neg_zero]

/-- A globally even normalized classifier cannot learn its parameters under convex losses for all labels.
Source: the new §3.5 categorical symmetry obstruction. The conclusion is prediction
constancy, not an impossibility theorem for every compact nonlinear attention family. -/
theorem even_normalizedNLL_constant {E : Type*} [AddCommGroup E] [Module ℝ E] {C : ℕ}
    (p : E → Fin C → ℝ) (hpos : ∀ θ c, 0 < p θ c)
    (hsum : ∀ θ, ∑ c, p θ c = 1)
    (hconvex : ∀ c, ConvexOn ℝ Set.univ (fun θ => -Real.log (p θ c)))
    (heven : ∀ θ, p (-θ) = p θ) (θ : E) : p θ = p 0 := by
  exact normalizedNLL_opposite p hpos hsum hconvex θ (heven θ).symm

example :
    (∀ θ : ℝ, ∀ c, 0 < (fun _ : ℝ => threeRouteProbability (0, 0)) θ c) ∧
    (∀ θ : ℝ, ∑ c, (fun _ : ℝ => threeRouteProbability (0, 0)) θ c = 1) ∧
    (∀ c, ConvexOn ℝ (Set.univ : Set ℝ)
      (fun θ => -Real.log ((fun _ : ℝ => threeRouteProbability (0, 0)) θ c))) ∧
    (∀ θ : ℝ, (fun _ : ℝ => threeRouteProbability (0, 0)) (-θ) =
      (fun _ : ℝ => threeRouteProbability (0, 0)) θ) := by
  refine ⟨fun θ c => threeRouteProbability_pos (0, 0) c,
    fun θ => threeRouteProbability_sum (0, 0), ?_, fun θ => rfl⟩
  intro c
  exact convexOn_const _ convex_univ

end Transformer.GPTMini.Convex.Likelihood

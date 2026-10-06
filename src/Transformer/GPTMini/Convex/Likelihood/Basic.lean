import Transformer.MagnitudeDirection.Section4_PositiveGains
import Mathlib.Analysis.Convex.Deriv
import Mathlib.LinearAlgebra.Prod
import Mathlib.LinearAlgebra.AffineSpace.AffineMap
import Mathlib.Tactic

/-!
# A compact nonlinear probability family with convex ordinary likelihoods

New sequential Bernoulli construction, using Boyd and Vandenberghe (2004),
§3.1.5's log-sum-exp convexity and §3.5's log-concave probability criterion.
The actual three probabilities are normalized and strictly positive. Their
negative logarithms are jointly convex in two unconstrained real parameters.
Thus affine output logits are not necessary for one particular convex loss.

This is a likelihood control, not an embedding/attention replacement. It
has neither token inputs nor learned Q/K/value coupling; those are the
remaining construction problem. Its three categories are observable
outcomes, not a hidden attention route marginalized by a downstream loss.
The reused softplus function and derivative are the existing declarations
in `MagnitudeDirection.Section4_PositiveGains` (arXiv:2606.25971v2, §4.1.3).
No optimizer, constraint projection, or auxiliary target is introduced.
-/

noncomputable section

namespace Transformer.GPTMini.Convex.Likelihood

open scoped BigOperators
open Transformer.MagnitudeDirection

/-- A strictly interior Bernoulli probability at an unrestricted real logit.
Source: the new sequential control; Boyd and Vandenberghe (2004), §3.5. -/
def routeSigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

/-- Both Bernoulli outcomes have positive probabilities at finite parameters.
Source: the actual sigmoid formula, rather than an assumed probability domain. -/
theorem routeSigmoid_pos (x : ℝ) : 0 < routeSigmoid x := by
  unfold routeSigmoid
  positivity

/-- The two sigmoid branches have total mass one.
Source: the new control's complementary Bernoulli outcomes. -/
theorem routeSigmoid_complement (x : ℝ) : routeSigmoid x + routeSigmoid (-x) = 1 := by
  unfold routeSigmoid
  rw [neg_neg, Real.exp_neg]
  have hp := Real.exp_pos x
  field_simp
  ring

/-- A Bernoulli negative log likelihood is the existing softplus at the negative logit.
Source: Boyd and Vandenberghe (2004), §3.1.5, specialized two-term log-sum-exp. -/
theorem routeSigmoid_nll (x : ℝ) : -Real.log (routeSigmoid x) = softplus (-x) := by
  unfold routeSigmoid softplus
  rw [Real.log_div (by norm_num) (by positivity), Real.log_one]
  ring

/-- The softplus derivative is monotone on the entire unconstrained real line.
Source: the existing exact derivative at §4.1.3, and §3.1.5's convex log-sum-exp. -/
theorem softplusDerivative_monotone : Monotone softplusDerivative := by
  intro x y hxy
  unfold softplusDerivative
  apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
  nlinarith [Real.exp_le_exp.mpr hxy]

/-- Actual softplus is convex, with no restriction on raw trained parameters.
Source: Boyd and Vandenberghe (2004), §3.1.5; proved via the reused actual derivative. -/
theorem softplus_convex : ConvexOn ℝ Set.univ softplus := by
  apply Monotone.convexOn_univ_of_deriv
  · intro x
    exact (softplus_hasDerivAt x).differentiableAt
  · have hd : deriv softplus = softplusDerivative := by
      funext x
      exact (softplus_hasDerivAt x).deriv
    rw [hd]
    exact softplusDerivative_monotone

/-- Three observable categories, obtained from two sequential learned Bernoulli logits.
Source: the new probability control, using §3.5's product log-concavity rule.
The third category is the remaining branch, not an unnormalized extra score. -/
def threeRouteProbability (p : ℝ × ℝ) (c : Fin 3) : ℝ :=
  if c = 0 then routeSigmoid p.1 else
    if c = 1 then routeSigmoid (-p.1) * routeSigmoid p.2 else
      routeSigmoid (-p.1) * routeSigmoid (-p.2)

/-- Every actual category has positive mass for every simultaneous parameter assignment.
Source: the new sequential control's sigmoid products. -/
theorem threeRouteProbability_pos (p : ℝ × ℝ) (c : Fin 3) :
    0 < threeRouteProbability p c := by
  unfold threeRouteProbability
  split_ifs
  · exact routeSigmoid_pos _
  · exact mul_pos (routeSigmoid_pos _) (routeSigmoid_pos _)
  · exact mul_pos (routeSigmoid_pos _) (routeSigmoid_pos _)

/-- The actual three probabilities normalize exactly, without a learned or projected normalizer.
Source: the new sequential control and the two complementary branch identities. -/
theorem threeRouteProbability_sum (p : ℝ × ℝ) : ∑ c, threeRouteProbability p c = 1 := by
  norm_num [Fin.sum_univ_three, threeRouteProbability]
  calc
    routeSigmoid p.1 + routeSigmoid (-p.1) * routeSigmoid p.2 +
        routeSigmoid (-p.1) * routeSigmoid (-p.2) =
      routeSigmoid p.1 + routeSigmoid (-p.1) *
        (routeSigmoid p.2 + routeSigmoid (-p.2)) := by ring
    _ = 1 := by rw [routeSigmoid_complement, mul_one, routeSigmoid_complement]

/-- The ordinary categorical negative log likelihood of an observed category.
Source: §3.5's negative logarithm of a positive normalized probability, without route supervision. -/
def threeRouteNLL (c : Fin 3) (p : ℝ × ℝ) : ℝ := -Real.log (threeRouteProbability p c)

/-- Actual categorical loss reduces to one or two softplus terms, depending on the observed category.
Source: the new sequential probability control. Both raw logits remain jointly trained. -/
theorem threeRouteNLL_eq (c : Fin 3) (p : ℝ × ℝ) :
    threeRouteNLL c p = if c = 0 then softplus (-p.1) else
      if c = 1 then softplus p.1 + softplus (-p.2) else
        softplus p.1 + softplus p.2 := by
  unfold threeRouteNLL threeRouteProbability
  by_cases h₀ : c = 0
  · simp only [h₀, ite_true]
    exact routeSigmoid_nll _
  · simp only [h₀, ite_false]
    by_cases h₁ : c = 1
    · simp only [h₁, ite_true]
      rw [Real.log_mul (routeSigmoid_pos _).ne' (routeSigmoid_pos _).ne', neg_add,
        routeSigmoid_nll, routeSigmoid_nll, neg_neg]
    · simp only [h₁, ite_false]
      rw [Real.log_mul (routeSigmoid_pos _).ne' (routeSigmoid_pos _).ne', neg_add,
        routeSigmoid_nll, routeSigmoid_nll, neg_neg, neg_neg]

/-- Ordinary categorical likelihood training is jointly convex in both unrestricted raw parameters.
Source: the new control, using §3.1.5 and the actual loss equality above.
This does not claim convexity after learned Q/K products or learned value mixing. -/
theorem threeRouteNLL_convex (c : Fin 3) : ConvexOn ℝ Set.univ (threeRouteNLL c) := by
  have hf : ConvexOn ℝ Set.univ (fun p : ℝ × ℝ => softplus p.1) := by
    refine ⟨convex_univ, ?_⟩
    intro p hp r hr a b ha hb hab
    change softplus (a * p.1 + b * r.1) ≤ a * softplus p.1 + b * softplus r.1
    exact softplus_convex.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab
  have hnf : ConvexOn ℝ Set.univ (fun p : ℝ × ℝ => softplus (-p.1)) := by
    refine ⟨convex_univ, ?_⟩
    intro p hp r hr a b ha hb hab
    have h := softplus_convex.2 (Set.mem_univ (-p.1)) (Set.mem_univ (-r.1)) ha hb hab
    change softplus (-(a * p.1 + b * r.1)) ≤ a * softplus (-p.1) + b * softplus (-r.1)
    simpa only [smul_eq_mul, mul_neg, neg_add] using h
  have hs : ConvexOn ℝ Set.univ (fun p : ℝ × ℝ => softplus p.2) := by
    refine ⟨convex_univ, ?_⟩
    intro p hp r hr a b ha hb hab
    change softplus (a * p.2 + b * r.2) ≤ a * softplus p.2 + b * softplus r.2
    exact softplus_convex.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab
  have hns : ConvexOn ℝ Set.univ (fun p : ℝ × ℝ => softplus (-p.2)) := by
    refine ⟨convex_univ, ?_⟩
    intro p hp r hr a b ha hb hab
    have h := softplus_convex.2 (Set.mem_univ (-p.2)) (Set.mem_univ (-r.2)) ha hb hab
    change softplus (-(a * p.2 + b * r.2)) ≤ a * softplus (-p.2) + b * softplus (-r.2)
    simpa only [smul_eq_mul, mul_neg, neg_add] using h
  refine ⟨convex_univ, ?_⟩
  intro p hp r hr a b ha hb hab
  simp_rw [threeRouteNLL_eq]
  by_cases h₀ : c = 0
  · simp only [h₀, ite_true]
    exact hnf.2 hp hr ha hb hab
  · by_cases h₁ : c = 1
    · simp only [h₀, ite_eq_left h₁]
      exact (hf.add hns).2 hp hr ha hb hab
    · simp only [h₀, h₁, ite_false]
      exact (hf.add hs).2 hp hr ha hb hab

/-- Any simultaneous affine utility parameterization retains the nonlinear control's convex losses.
Source: Boyd and Vandenberghe (2004), §3.2.2's affine composition rule,
applied to the actual normalized sequential probabilities. Products of
two learned utilities are outside this sufficient condition. -/
theorem threeRouteNLL_affine_convex {E : Type*} [AddCommGroup E] [Module ℝ E]
    (F : E →ᵃ[ℝ] ℝ × ℝ) (c : Fin 3) :
    ConvexOn ℝ Set.univ (fun θ => threeRouteNLL c (F θ)) := by
  have h := ConvexOn.comp_affineMap F (threeRouteNLL_convex c)
  refine ⟨convex_univ, ?_⟩
  intro p hp r hr a b ha hb hab
  exact h.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

end Transformer.GPTMini.Convex.Likelihood

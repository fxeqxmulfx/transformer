import Transformer.GPTMini.Convex.Reparameterization.Basic

/-!
# Likelihood interpolation between actual shared-key heads

Derived from Boyd and Vandenberghe (2004), §3.5's log-concavity criterion.
Convex negative log likelihood requires each midpoint probability to
dominate the geometric mean of its two endpoint probabilities.

The endpoints here are the actual scalar softmax heads of Basic, not
postulated probability tables. Their arithmetic mean witnesses that the
interpolation inequalities themselves are consistent. Rational lower and
upper bounds will expose the obstruction imposed by their rank-one class.
This control has fixed one-hot values and two queries over shared memory;
it does not assert a boundary for the entire causal GPTMini architecture.
The first endpoint favors the first category in only the first query row;
the second favors the second category in only the second row. All other
endpoint rows are uniform, with no zero or infinite scores involved.
-/

noncomputable section

namespace Transformer.GPTMini.Convex.Reparameterization

open scoped BigOperators

/-- Convex negative log likelihood imposes the squared geometric-mean midpoint bound.
Source: Boyd and Vandenberghe (2004), §3.5, specialized to two equal weights.
No affine, continuous, or injective probability parameterization is assumed. -/
theorem nll_midpoint_product {E : Type*} [AddCommGroup E] [Module ℝ E]
    (p : E → ℝ) (hpos : ∀ θ, 0 < p θ)
    (hconvex : ConvexOn ℝ Set.univ (fun θ => -Real.log (p θ))) (x y : E) :
    p x * p y ≤ (p ((1 / 2 : ℝ) • x + (1 / 2 : ℝ) • y)) ^ 2 := by
  have h := hconvex.2 (Set.mem_univ x) (Set.mem_univ y)
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  have hl : Real.log (p x * p y) ≤
      Real.log ((p ((1 / 2 : ℝ) • x + (1 / 2 : ℝ) • y)) ^ 2) := by
    rw [Real.log_mul (hpos x).ne' (hpos y).ne', Real.log_pow]
    simp only [smul_eq_mul] at h
    nlinarith
  exact (Real.log_le_log_iff (mul_pos (hpos x) (hpos y)) (pow_pos (hpos _) 2)).mp hl

example :
    (∀ θ, 0 < Likelihood.threeRouteProbability θ 0) ∧
    ConvexOn ℝ Set.univ (fun θ => -Real.log (Likelihood.threeRouteProbability θ 0)) := by
  exact ⟨fun θ => Likelihood.threeRouteProbability_pos θ 0,
    Likelihood.threeRouteNLL_convex 0⟩

/-- Each endpoint's nonuniform row favors the category with the same row index.
Source: the two actual scalar-head witnesses derived from §3 of arXiv:2211.11052v1. -/
def peakCategory (r : Fin 2) : Fin 3 := if r = 0 then 0 else 1

/-- The coordinate products of the two physical endpoint predictions.
Source: the new shared-memory softmax control's exact rational endpoint evaluation. -/
theorem endpoint_product (r : Fin 2) (c : Fin 3) :
    leftProbability r c * rightProbability r c =
      if c = peakCategory r then 1 / 6 else 1 / 12 := by
  fin_cases r <;> fin_cases c <;>
    norm_num [leftProbability_eq, rightProbability_eq, peakCategory]

/-- An actual arithmetic mean of the two endpoint predictions, used as a feasibility witness.
Source: the new interpolation control; this is not claimed to be an original scalar head. -/
def meanProbability (r : Fin 2) (c : Fin 3) : ℝ :=
  (leftProbability r c + rightProbability r c) / 2

/-- The interpolation witness's exact rational table.
Source: the two actual softmax endpoints, averaged entry by entry. -/
theorem meanProbability_eq (r : Fin 2) (c : Fin 3) :
    meanProbability r c = if c = peakCategory r then 5 / 12 else 7 / 24 := by
  fin_cases r <;> fin_cases c <;>
    norm_num [meanProbability, leftProbability_eq, rightProbability_eq, peakCategory]

/-- The consistent interpolation witness has positive mass in every entry.
Source: the exact arithmetic mean of two physical softmax predictions. -/
theorem meanProbability_pos (r : Fin 2) (c : Fin 3) : 0 < meanProbability r c := by
  rw [meanProbability_eq]
  split <;> norm_num

/-- The consistent interpolation witness remains normalized in each row.
Source: the two physical endpoint probability rows and their arithmetic mean. -/
theorem meanProbability_sum (r : Fin 2) : ∑ c, meanProbability r c = 1 := by
  fin_cases r <;> norm_num [Fin.sum_univ_three, meanProbability_eq, peakCategory]

/-- The arithmetic mean meets every required geometric-mean midpoint inequality.
Source: Boyd and Vandenberghe (2004), §3.5's required interpolation bound.
This supplies a satisfiable witness independently of the later head-class obstruction. -/
theorem meanProbability_product (r : Fin 2) (c : Fin 3) :
    leftProbability r c * rightProbability r c ≤ meanProbability r c ^ 2 := by
  rw [endpoint_product, meanProbability_eq]
  split <;> norm_num

/-- Every positive table meeting the likelihood midpoint bounds has explicit lower masses.
Source: the new rational relaxation of §3.5's geometric-mean condition for these heads.
The peak bound is 2/5 and each other category has mass at least 7/25. -/
theorem bridge_lower (p : SharedProbability) (hpos : ∀ r c, 0 < p r c)
    (hproduct : ∀ r c, leftProbability r c * rightProbability r c ≤ p r c ^ 2)
    (r : Fin 2) (c : Fin 3) :
    (if c = peakCategory r then 2 / 5 else 7 / 25) ≤ p r c := by
  have h := hproduct r c
  rw [endpoint_product] at h
  have hp := hpos r c
  by_cases hc : c = peakCategory r
  · rw [ite_eq_left hc] at h ⊢
    by_contra hn
    have ht : p r c < 2 / 5 := by linarith
    have hh : 0 ≤ (2 / 5 - p r c) * (2 / 5 + p r c) := by positivity
    nlinarith
  · rw [ite_eq_right hc] at h ⊢
    by_contra hn
    have ht : p r c < 7 / 25 := by linarith
    have hh : 0 ≤ (7 / 25 - p r c) * (7 / 25 + p r c) := by positivity
    nlinarith

example :
    (∀ r c, 0 < meanProbability r c) ∧
    (∀ r c, leftProbability r c * rightProbability r c ≤ meanProbability r c ^ 2) := by
  exact ⟨meanProbability_pos, meanProbability_product⟩

/-- Normalization and the lower masses bound every nonpeak entry by 8/25.
Source: the new finite interpolation control; the remaining two entries consume
at least 2/5 + 7/25 of the row's probability mass. -/
theorem bridge_small_upper (p : SharedProbability) (hsum : ∀ r, ∑ c, p r c = 1)
    (hlower : ∀ r c, (if c = peakCategory r then 2 / 5 else 7 / 25) ≤ p r c)
    (r : Fin 2) (c : Fin 3) (hc : c ≠ peakCategory r) : p r c ≤ 8 / 25 := by
  have hs := hsum r
  have h0 := hlower r 0
  have h1 := hlower r 1
  have h2 := hlower r 2
  rw [Fin.sum_univ_three] at hs
  fin_cases r <;> fin_cases c <;> norm_num [peakCategory] at * <;> linarith

example :
    (∀ r, ∑ c, meanProbability r c = 1) ∧
    (∀ r c, (if c = peakCategory r then 2 / 5 else 7 / 25) ≤ meanProbability r c) ∧
    (1 : Fin 3) ≠ peakCategory 0 := by
  refine ⟨meanProbability_sum, ?_, by norm_num [peakCategory]⟩
  intro r c
  exact bridge_lower meanProbability meanProbability_pos meanProbability_product r c

/-- The consistent midpoint witness satisfies both explicit bounds for every nonpeak entry.
Source: the new rational interpolation certificate; neither interval hypothesis is vacuous. -/
theorem meanProbability_small_bounds (r : Fin 2) (c : Fin 3) (hc : c ≠ peakCategory r) :
    7 / 25 ≤ meanProbability r c ∧ meanProbability r c ≤ 8 / 25 := by
  rw [meanProbability_eq, ite_eq_right hc]
  constructor <;> norm_num

example : (1 : Fin 3) ≠ peakCategory 0 := by norm_num [peakCategory]

end Transformer.GPTMini.Convex.Reparameterization

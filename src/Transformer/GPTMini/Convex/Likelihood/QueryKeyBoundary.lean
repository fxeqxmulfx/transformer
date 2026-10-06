import Transformer.GPTMini.Convex.Likelihood.Fibers

/-!
# Genuine nonlinear logits, and the boundary at learned query/key products

New controls derived from Boyd and Vandenberghe (2004), §3.1.5 and §3.5.
The sequential model's log probabilities are genuine categorical logits:
ordinary log-sum-exp cross entropy equals the convex loss already proved.
Their class differences are nonlinear, so this is not just an affine model
with a nonlinear common shift. A strict midpoint witness certifies this.

Reintroducing even one free scalar query times one free scalar key destroys
the guarantee. Two opposite parameter assignments have identical actual
three-category probabilities, but their midpoint has worse cross entropy
for category zero. This is a changed likelihood toy, not a claim that the
sequential model implements an attention/value block. No route teacher,
new loss, optimizer, fixed feature bank, or head-search oracle is added.
-/

noncomputable section

namespace Transformer.GPTMini.Convex.Likelihood

open scoped BigOperators
open Transformer.MagnitudeDirection

/-- Normalized log probabilities serve as ordinary categorical logits.
Source: the new likelihood control and §3.5's strictly positive probabilities. -/
def threeRouteLogits (p : ℝ × ℝ) (c : Fin 3) : ℝ := Real.log (threeRouteProbability p c)

/-- The same log-sum-exp categorical cross entropy used by the ordinary training loop.
Source: GPTMini's PyTorch cross entropy at cbafbe9, evaluated on the new control's logits. -/
def threeRouteCrossEntropy (c : Fin 3) (p : ℝ × ℝ) : ℝ :=
  Real.log (∑ j, Real.exp (threeRouteLogits p j)) - threeRouteLogits p c

/-- Ordinary cross entropy equals the already proved likelihood, with no auxiliary objective.
Source: the new control's actual normalization, rather than an assumed logit normalizer. -/
theorem threeRouteCrossEntropy_eq (c : Fin 3) (p : ℝ × ℝ) :
    threeRouteCrossEntropy c p = threeRouteNLL c p := by
  have he : ∀ j, Real.exp (threeRouteLogits p j) = threeRouteProbability p j := by
    intro j
    exact Real.exp_log (threeRouteProbability_pos p j)
  unfold threeRouteCrossEntropy
  simp_rw [he]
  rw [threeRouteProbability_sum, Real.log_one]
  unfold threeRouteLogits threeRouteNLL
  ring

/-- Cross entropy stays jointly convex in the two free raw parameters despite nonlinear logits.
Source: the actual categorical identity and the §3.1.5 sequential likelihood proof. -/
theorem threeRouteCrossEntropy_convex (c : Fin 3) :
    ConvexOn ℝ Set.univ (threeRouteCrossEntropy c) := by
  refine ⟨convex_univ, ?_⟩
  intro p hp r hr a b ha hb hab
  simp_rw [threeRouteCrossEntropy_eq]
  exact (threeRouteNLL_convex c).2 hp hr ha hb hab

/-- The difference of the two softplus branches is the raw logit.
Source: §3.1.5's two-term log-sum-exp, factored using the actual exponential. -/
theorem softplus_difference (x : ℝ) : softplus x - softplus (-x) = x := by
  have he : 1 + Real.exp x = Real.exp x * (1 + Real.exp (-x)) := by
    rw [Real.exp_neg]
    have hp := Real.exp_pos x
    field_simp
    ring
  unfold softplus
  rw [he, Real.log_mul (Real.exp_pos x).ne' (by positivity), Real.log_exp]
  ring

/-- A class-logit contrast depends nonlinearly on the second independently learned parameter.
Source: the new sequential probability control; the nonlinearity survives common-logit shifts. -/
theorem threeRouteLogits_difference (p : ℝ × ℝ) :
    threeRouteLogits p 0 - threeRouteLogits p 2 = p.1 + softplus p.2 := by
  have h₀ := threeRouteNLL_eq 0 p
  have h₂ := threeRouteNLL_eq 2 p
  norm_num only [Fin.reduceEq, ite_true, ite_false] at h₀ h₂
  calc
    threeRouteLogits p 0 - threeRouteLogits p 2 =
        threeRouteNLL 2 p - threeRouteNLL 0 p := by
      unfold threeRouteLogits threeRouteNLL
      ring
    _ = p.1 + softplus p.2 := by
      rw [h₀, h₂]
      linarith [softplus_difference p.1]

/-- A strict midpoint gap shows that the actual logit contrast is not affine.
Source: the new sequential control at parameters (0, ±log 3), whose midpoint is (0, 0).
This refutes affine logits as a necessary condition for convex categorical training. -/
theorem threeRouteLogits_non_affine_witness :
    threeRouteLogits (0, 0) 0 - threeRouteLogits (0, 0) 2 <
      ((threeRouteLogits (0, Real.log 3) 0 - threeRouteLogits (0, Real.log 3) 2) +
       (threeRouteLogits (0, -Real.log 3) 0 - threeRouteLogits (0, -Real.log 3) 2)) / 2 := by
  have h₀ : softplus 0 = Real.log 2 := by norm_num [softplus]
  have hplus : softplus (Real.log 3) = Real.log 4 := by
    unfold softplus
    rw [Real.exp_log (by norm_num : (0 : ℝ) < 3)]
    norm_num
  have hminus : softplus (-Real.log 3) = Real.log (4 / 3) := by
    unfold softplus
    rw [Real.exp_neg, Real.exp_log (by norm_num : (0 : ℝ) < 3)]
    norm_num
  have hl : Real.log 4 = 2 * Real.log 2 := by
    rw [show (4 : ℝ) = 2 * 2 by norm_num]
    rw [Real.log_mul (by norm_num) (by norm_num)]
    ring
  rw [threeRouteLogits_difference, threeRouteLogits_difference, threeRouteLogits_difference]
  simp only [zero_add, h₀, hplus, hminus]
  linarith [Real.log_pos (by norm_num : (1 : ℝ) < 4 / 3)]

/-- Replace the first free logit by a genuine product of a learned query and a learned key.
Source: arXiv:2211.11052v1, §3's scalar Q/K product, inserted into the new control.
The second branch stays at zero; there is no claimed attention/value implementation. -/
def productRouteParameters (p : ℝ × ℝ) : ℝ × ℝ := (p.1 * p.2, 0)

/-- The actual categorical cross entropy after reintroducing the Q/K factorization.
Source: the new likelihood control with §3's two simultaneously trained scalar factors. -/
def productRouteCrossEntropy (c : Fin 3) (p : ℝ × ℝ) : ℝ :=
  threeRouteCrossEntropy c (productRouteParameters p)

/-- Category zero's loss is the ordinary softplus of the negative learned query/key product.
Source: the actual cross entropy identity, with no independently declared loss surface. -/
theorem productRouteCrossEntropy_zero (p : ℝ × ℝ) :
    productRouteCrossEntropy 0 p = softplus (-(p.1 * p.2)) := by
  unfold productRouteCrossEntropy
  rw [threeRouteCrossEntropy_eq, threeRouteNLL_eq]
  norm_num [productRouteParameters]

/-- Simultaneously negating Q and K leaves every actual category probability unchanged.
Source: §3's bilinear matching symmetry, exhibited in the new normalized control. -/
theorem productRouteProbability_opposite (p : ℝ × ℝ) :
    threeRouteProbability (productRouteParameters (-p)) =
      threeRouteProbability (productRouteParameters p) := by
  have he : productRouteParameters (-p) = productRouteParameters p := by
    apply Prod.ext
    · change (-p.1) * (-p.2) = p.1 * p.2
      ring
    · change (0 : ℝ) = 0
      rfl
  rw [he]

/-- Useful nonzero matching fits category zero better than its zero-product midpoint.
Source: the new actual categorical control; exponent and logarithm monotonicity give a strict gap. -/
theorem productRouteCrossEntropy_matching_better :
    productRouteCrossEntropy 0 (1, 1) < productRouteCrossEntropy 0 (0, 0) := by
  rw [productRouteCrossEntropy_zero, productRouteCrossEntropy_zero]
  norm_num only [one_mul, zero_mul, neg_zero]
  unfold softplus
  apply Real.log_lt_log (by positivity)
  have h := Real.exp_lt_exp.mpr (by norm_num : (-1 : ℝ) < 0)
  linarith

/-- Reintroducing the two learned scalar Q/K factors destroys convex ordinary cross entropy.
Source: the new control's actual opposite predictions and strict midpoint loss gap.
Changing the sigmoid normalizer alone cannot fix this particular factorization. -/
theorem productRouteCrossEntropy_not_convex :
    ¬ ConvexOn ℝ Set.univ (productRouteCrossEntropy 0) := by
  intro h
  have hj := h.2 (Set.mem_univ (1, 1)) (Set.mem_univ (-1, -1))
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  have hm : (1 / 2 : ℝ) • ((1, 1) : ℝ × ℝ) +
      (1 / 2 : ℝ) • ((-1, -1) : ℝ × ℝ) = (0, 0) := by
    apply Prod.ext <;> norm_num
  rw [hm, productRouteCrossEntropy_zero, productRouteCrossEntropy_zero,
    productRouteCrossEntropy_zero] at hj
  norm_num only [smul_eq_mul, one_mul, neg_mul_neg, zero_mul, neg_zero] at hj
  have hb := productRouteCrossEntropy_matching_better
  rw [productRouteCrossEntropy_zero, productRouteCrossEntropy_zero] at hb
  norm_num only [one_mul, zero_mul, neg_zero] at hb
  linarith

end Transformer.GPTMini.Convex.Likelihood

module

public import Transformer.GPTMini.Convex.Likelihood.Basic
public import Transformer.Landscape.Neural.Loss

/-!
# Complete path likelihood and ordinary next-object likelihood differ

Source: `review/flat_spectrum.md`, CM74. At one fixed input, a learned
Bernoulli encoder chooses a hidden bit. A learned decoder keeps or flips it.
The actual four path probabilities are normalized. Their complete negative
log likelihoods are convex in both raw parameters, without a frozen encoder.
The answer probability sums over the hidden bit; it is a different objective.
This is a finite architecture control, not a raw-sequence learning result.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.HiddenPath

open scoped BigOperators
open Transformer.GPTMini.Convex.Likelihood
open Transformer.MagnitudeDirection (softplus)
open Transformer.Landscape.Neural (crossEntropy)

/-- Signed outcome logit of a learned binary stage, CM74. -/
def signed (b : Bool) (x : ℝ) : ℝ := if b then x else -x

/-- Actual outcome probability of that stage, CM74. -/
noncomputable def bernoulli (x : ℝ) (b : Bool) : ℝ := routeSigmoid (signed b x)

/-- Encoder and decoder probabilities multiply along a complete hidden path,
CM74. The second stage is a learned keep/flip channel, not an answer label. -/
noncomputable def pathProbability (p : ℝ × ℝ) (z : Bool × Bool) : ℝ :=
  bernoulli p.1 z.1 * bernoulli p.2 (decide (z.1 = z.2))

/-- The actual next-object distribution marginalizes the unobserved state, CM74. -/
noncomputable def answerProbability (p : ℝ × ℝ) (c : Bool) : ℝ :=
  ∑ h, pathProbability p (h, c)

/-- The complete-path objective requires the internal state as well as the
answer, CM74; it is not the requested answer-only supervision. -/
noncomputable def pathLoss (z : Bool × Bool) (p : ℝ × ℝ) : ℝ :=
  -Real.log (pathProbability p z)

/-- Usual full-vocabulary CE on the actual marginal log probabilities, CM74. -/
noncomputable def answerLoss (c : Bool) (p : ℝ × ℝ) : ℝ :=
  crossEntropy (fun d => Real.log (answerProbability p d)) c

/-- Ordinary CE of positive normalized log probabilities is their NLL,
CM74. Normalization is proved for each actual model before this is used. -/
theorem crossEntropy_log_probability {C : Type*} [Fintype C] (p : C → ℝ)
    (hpos : ∀ c, 0 < p c) (hsum : ∑ c, p c = 1) (c : C) :
    crossEntropy (fun d => Real.log (p d)) c = -Real.log (p c) := by
  unfold crossEntropy
  simp_rw [Real.exp_log (hpos _)]
  rw [hsum, Real.log_one, zero_sub]

/-- Both encoder and decoder retain positive probability on every path, CM74. -/
theorem pathProbability_pos (p : ℝ × ℝ) (z : Bool × Bool) :
    0 < pathProbability p z :=
  mul_pos (routeSigmoid_pos _) (routeSigmoid_pos _)

/-- The complete learned path distribution actually sums to one, CM74. -/
theorem pathProbability_sum (p : ℝ × ℝ) : ∑ z, pathProbability p z = 1 := by
  simp only [Fintype.sum_prod_type, Fintype.sum_bool]
  change routeSigmoid p.1 * routeSigmoid p.2 +
    routeSigmoid p.1 * routeSigmoid (-p.2) +
    (routeSigmoid (-p.1) * routeSigmoid (-p.2) + routeSigmoid (-p.1) * routeSigmoid p.2) = 1
  calc
    _ = (routeSigmoid p.1 + routeSigmoid (-p.1)) *
        (routeSigmoid p.2 + routeSigmoid (-p.2)) := by ring
    _ = 1 := by rw [routeSigmoid_complement, routeSigmoid_complement]; norm_num

/-- Complete-path NLL is itself ordinary CE on the four normalized path
categories, CM74. Observing those categories reveals the internal state. -/
theorem pathLoss_crossEntropy (p : ℝ × ℝ) (z : Bool × Bool) :
    crossEntropy (fun w => Real.log (pathProbability p w)) z = pathLoss z p :=
  crossEntropy_log_probability _ (pathProbability_pos p) (pathProbability_sum p) z

/-- Marginalization keeps every next-object probability positive, CM74. -/
theorem answerProbability_pos (p : ℝ × ℝ) (c : Bool) :
    0 < answerProbability p c := by
  exact Finset.sum_pos (fun h _ => pathProbability_pos p (h, c)) Finset.univ_nonempty

/-- Marginalization keeps the entire original output vocabulary normalized,
CM74. No internal-path target enters this computation. -/
theorem answerProbability_sum (p : ℝ × ℝ) : ∑ c, answerProbability p c = 1 := by
  unfold answerProbability
  rw [Finset.sum_comm]
  simpa only [Fintype.sum_prod_type] using pathProbability_sum p

/-- The actual answer CE equals marginal NLL, CM74; complete-path NLL cannot
be substituted for it when internal states are hidden. -/
theorem answerLoss_eq (c : Bool) (p : ℝ × ℝ) :
    answerLoss c p = -Real.log (answerProbability p c) :=
  crossEntropy_log_probability _ (answerProbability_pos p) (answerProbability_sum p) c

/-- Actual complete-path loss is the sum of the two learned-stage losses,
CM74. This identity uses the observed hidden state. -/
theorem pathLoss_eq (z : Bool × Bool) (p : ℝ × ℝ) :
    pathLoss z p = softplus (-signed z.1 p.1) +
      softplus (-signed (decide (z.1 = z.2)) p.2) := by
  unfold pathLoss pathProbability bernoulli
  rw [Real.log_mul (routeSigmoid_pos _).ne' (routeSigmoid_pos _).ne', neg_add,
    routeSigmoid_nll, routeSigmoid_nll]

/-- Each outcome sign preserves all parameter combinations, CM74. -/
theorem signed_combination (v : Bool) (a b x y : ℝ) :
    signed v (a * x + b * y) = a * signed v x + b * signed v y := by
  cases v <;> simp [signed, add_comm]

/-- Complete-path NLL is jointly convex in both learned stages, CM74.
The following module tests the different answer-only objective. -/
theorem pathLoss_convex (z : Bool × Bool) : ConvexOn ℝ Set.univ (pathLoss z) := by
  refine ⟨convex_univ, ?_⟩
  intro p _ q _ a b ha hb hab
  simp only [pathLoss_eq]
  change softplus (-signed z.1 (a * p.1 + b * q.1)) +
    softplus (-signed (decide (z.1 = z.2)) (a * p.2 + b * q.2)) ≤ _
  rw [signed_combination, signed_combination]
  simp only [neg_add, ← mul_neg]
  have h₁ := softplus_convex.2 (Set.mem_univ (-signed z.1 p.1))
    (Set.mem_univ (-signed z.1 q.1)) ha hb hab
  have h₂ := softplus_convex.2
    (Set.mem_univ (-signed (decide (z.1 = z.2)) p.2))
    (Set.mem_univ (-signed (decide (z.1 = z.2)) q.2)) ha hb hab
  simp only [smul_eq_mul] at h₁ h₂ ⊢
  nlinarith

example := crossEntropy_log_probability (fun _ : Bool => (1 / 2 : ℝ))
  (by intro c; norm_num) (by norm_num [Fintype.sum_bool]) true

end Transformer.Grokking.FlatSpectrum.ConvexChannel.HiddenPath

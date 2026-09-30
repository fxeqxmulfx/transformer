/-
# AdaFisher: likelihood and the training objective

arXiv:2405.16397v3, §2, the maximum-likelihood/negative-log-likelihood
equivalence. The likelihoods are positive on the observations in question.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

open scoped BigOperators

noncomputable section

namespace Transformer.AdaFisher

variable {n : ℕ} {Θ : Type*}

/-- Joint observed likelihood for independent observations, §2.
`p θ k` is the conditional probability of observation k at parameter θ. -/
def jointLikelihood (p : Θ → Fin n → ℝ) (θ : Θ) : ℝ := ∏ k, p θ k

/-- Training objective in §2: the sum of negative log-likelihood losses. -/
def negativeLogLikelihood (p : Θ → Fin n → ℝ) (θ : Θ) : ℝ :=
  ∑ k, -Real.log (p θ k)

/-- The objective is the negative logarithm of the joint likelihood,
§2, under the positivity assumption implicit in the paper's logarithms. -/
theorem negativeLogLikelihood_eq (p : Θ → Fin n → ℝ) (θ : Θ)
    (hp : ∀ k, 0 < p θ k) :
    negativeLogLikelihood p θ = -Real.log (jointLikelihood p θ) := by
  rw [jointLikelihood, Real.log_prod (fun k _ => ne_of_gt (hp k))]
  simp [negativeLogLikelihood]

example : ∀ k : Fin 1, 0 < (fun (_ : ℝ) (_ : Fin 1) => (1 : ℝ)) 0 k := by
  norm_num

/-- Maximizing the likelihood and minimizing the negative log-likelihood
are equivalent, §2. The theorem preserves the inequality direction and
requires positive likelihoods, rather than using Lean's total log at zero. -/
theorem maximumLikelihood_iff (p : Θ → Fin n → ℝ) (θ φ : Θ)
    (hθ : ∀ k, 0 < p θ k) (hφ : ∀ k, 0 < p φ k) :
    negativeLogLikelihood p θ ≤ negativeLogLikelihood p φ ↔
      jointLikelihood p φ ≤ jointLikelihood p θ := by
  rw [negativeLogLikelihood_eq p θ hθ, negativeLogLikelihood_eq p φ hφ,
    neg_le_neg_iff]
  exact Real.log_le_log_iff
    (Finset.prod_pos (fun k _ => hφ k)) (Finset.prod_pos (fun k _ => hθ k))

example :
    (∀ k : Fin 1, 0 < (fun (_ : ℝ) (_ : Fin 1) => (1 : ℝ)) 0 k) ∧
    (∀ k : Fin 1, 0 < (fun (_ : ℝ) (_ : Fin 1) => (1 : ℝ)) 1 k) := by
  norm_num

end Transformer.AdaFisher

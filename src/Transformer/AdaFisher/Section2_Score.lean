/-
# AdaFisher: finite probabilistic scores are centered

arXiv:2405.16397v3, §2, `eq:fishermatrix`.
The score is the derivative of log p. Differentiation of a finite
normalized positive probability mass function gives a zero expected score.
-/

import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Tactic

open scoped BigOperators

noncomputable section

namespace Transformer.AdaFisher

variable {n : ℕ}

/-- Coordinate score `∂θ log pθ(y) = (∂θ pθ(y))/pθ(y)`, §2,
the logarithmic gradient in `eq:fishermatrix`. -/
def probabilityScore (p dp : Fin n → ℝ) (k : Fin n) : ℝ := dp k / p k

/-- The score is the actual derivative of the log probability, §2.
Positive mass makes the derivative of the logarithm well-defined. -/
theorem probabilityScore_hasDerivAt (p : ℝ → Fin n → ℝ) (dp : Fin n → ℝ)
    (θ : ℝ) (k : Fin n) (hp : 0 < p θ k)
    (hdp : HasDerivAt (fun u => p u k) (dp k) θ) :
    HasDerivAt (fun u => Real.log (p u k)) (probabilityScore (p θ) dp k) θ :=
  hdp.log (ne_of_gt hp)

example : (0 : ℝ) < 1 / 2 ∧
    HasDerivAt (fun _ : ℝ => (1 / 2 : ℝ)) 0 0 :=
  ⟨by norm_num, hasDerivAt_const 0 (1 / 2)⟩

/-- The derivatives of the normalized probabilities sum to zero, §2,
the regularity condition behind centered scores and Fisher additivity.
For finite label spaces differentiation and the sum commute. -/
theorem normalized_probability_derivatives_sum (p : ℝ → Fin n → ℝ)
    (dp : Fin n → ℝ) (θ : ℝ)
    (hdp : ∀ k, HasDerivAt (fun u => p u k) (dp k) θ)
    (hnorm : ∀ u, ∑ k, p u k = 1) : ∑ k, dp k = 0 := by
  have hsum := HasDerivAt.sum (u := Finset.univ) (fun k _ => hdp k)
  have hsum' : HasDerivAt (fun _ : ℝ => (1 : ℝ)) (∑ k, dp k) θ := by
    convert hsum using 1
    funext u
    simp [Finset.sum_apply, hnorm]
  exact hsum'.unique (hasDerivAt_const θ 1)

example :
    (∀ k : Fin 2, HasDerivAt (fun u : ℝ =>
      (fun (_ : ℝ) (_ : Fin 2) => (1 / 2 : ℝ)) u k) 0 0) ∧
    (∀ u : ℝ, ∑ k : Fin 2, (fun (_ : ℝ) (_ : Fin 2) => (1 / 2 : ℝ)) u k = 1) := by
  constructor
  · intro k
    exact hasDerivAt_const 0 (1 / 2)
  · intro u
    norm_num [Fin.sum_univ_two]

/-- Expected score is zero for a positive differentiable finite
probability law, §2, `eq:fishermatrix`. The sign-reversed negative-log-loss
gradient is also centered. This justifies the centered-score condition,
rather than assuming all activation second moments are covariance matrices. -/
theorem expected_probabilityScore_zero (p : ℝ → Fin n → ℝ)
    (dp : Fin n → ℝ) (θ : ℝ) (hp : ∀ k, 0 < p θ k)
    (hdp : ∀ k, HasDerivAt (fun u => p u k) (dp k) θ)
    (hnorm : ∀ u, ∑ k, p u k = 1) :
    (∑ k, p θ k * probabilityScore (p θ) dp k) = 0 := by
  have heq : (∑ k, p θ k * probabilityScore (p θ) dp k) = ∑ k, dp k := by
    apply Finset.sum_congr rfl
    intro k hk
    unfold probabilityScore
    field_simp [ne_of_gt (hp k)]
  rw [heq]
  exact normalized_probability_derivatives_sum p dp θ hdp hnorm

example :
    (∀ k : Fin 2, 0 < (fun (_ : ℝ) (_ : Fin 2) => (1 / 2 : ℝ)) 0 k) ∧
    (∀ k : Fin 2, HasDerivAt (fun u : ℝ =>
      (fun (_ : ℝ) (_ : Fin 2) => (1 / 2 : ℝ)) u k) 0 0) ∧
    (∀ u : ℝ, ∑ k : Fin 2, (fun (_ : ℝ) (_ : Fin 2) => (1 / 2 : ℝ)) u k = 1) := by
  refine ⟨by norm_num, ?_, ?_⟩
  · intro k
    exact hasDerivAt_const 0 (1 / 2)
  · intro u
    norm_num [Fin.sum_univ_two]

end Transformer.AdaFisher

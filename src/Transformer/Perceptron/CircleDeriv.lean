/-
# Perceptrons and attention's mean-field landscape — the potential along the
  circle

The angular derivatives of `v_ϑ ∘ x` on `𝕊¹`, in the standard parametrization
`x(θ) = circlePoint θ` of `Atoms.lean`.  They are what
`rem:strictSOPD-perceptron` of arXiv:2601.21366v2 states its sufficient
condition for strictness in, and its closing display

  `∂_θ²(v_ϑ ∘ x)(θ) = 2 Σ_j ω_j (σ'(a_j·x(θ))(a_j·x'(θ))² + σ(a_j·x(θ)) a_j·x''(θ))`

is `secondDeriv_potential_circlePoint`, proved: the factor `2` is the `φ' = 2σ`
convention of `eq: primitive.field`, and `a_j·x''(θ) = -a_j·x(θ)` because the
unit circle has unit curvature (`hasDerivAt_circleVel`).

Source: arXiv:2601.21366v2, `rem:strictSOPD-perceptron`, `eq: primitive.field`.
-/

import Transformer.Perceptron.Atoms

open scoped BigOperators
open Real

namespace Transformer
namespace Perceptron

/-! ### The potential along the circle -/

/-- The angular derivative of `v_ϑ ∘ x`:
`∂_θ(v_ϑ ∘ x)(θ) = 2 Σ_j ω_j σ(a_j·x(θ)) (a_j·x'(θ))`, by `φ' = 2σ`. -/
theorem hasDerivAt_potential_circlePoint (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (θ : ℝ) :
    HasDerivAt (fun s : ℝ => potential φ ω a (circlePoint s : EucSpace 2))
      (2 * ∑ j : Idx 2, ω j *
        (σ (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) *
          inner (𝕜 := ℝ) (a j) (circleVel θ))) θ := by
  have h : ∀ j : Idx 2, HasDerivAt
      (fun s : ℝ => ω j * φ (inner (𝕜 := ℝ) (a j) (circlePoint s : EucSpace 2)))
      (ω j * (2 * σ (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) *
        inner (𝕜 := ℝ) (a j) (circleVel θ))) θ := fun j =>
    (((hφ _).comp θ (hasDerivAt_inner_circlePoint (a j) θ))).const_mul (ω j)
  have hsum := HasDerivAt.sum (fun j (_ : j ∈ Finset.univ) => h j)
  rw [Finset.mul_sum] at *
  refine hsum.congr_deriv (Finset.sum_congr rfl fun j _ => by ring)

/-- The angular derivative of `v_ϑ ∘ x` as a function. -/
theorem deriv_potential_circlePoint (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) :
    (deriv fun s : ℝ => potential φ ω a (circlePoint s : EucSpace 2))
      = fun θ => 2 * ∑ j : Idx 2, ω j *
        (σ (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) *
          inner (𝕜 := ℝ) (a j) (circleVel θ)) :=
  funext fun θ => (hasDerivAt_potential_circlePoint φ σ hφ ω a θ).deriv

/-- **The simplified display of `rem:strictSOPD-perceptron`.**  The angular
second derivative of the potential is
`∂_θ²(v_ϑ ∘ x)(θ) = 2 Σ_j ω_j (σ'(a_j·x(θ))(a_j·x'(θ))² + σ(a_j·x(θ)) a_j·x''(θ))`,
and `a_j·x''(θ) = -a_j·x(θ)` on the unit circle.

Source: arXiv:2601.21366v2, `rem:strictSOPD-perceptron`. -/
theorem secondDeriv_potential_circlePoint (φ σ σ' : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (hσ : ∀ s : ℝ, HasDerivAt σ (σ' s) s)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (θ : ℝ) :
    deriv^[2] (fun s : ℝ => potential φ ω a (circlePoint s : EucSpace 2)) θ
      = 2 * ∑ j : Idx 2, ω j *
        (σ' (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) *
            inner (𝕜 := ℝ) (a j) (circleVel θ) ^ 2 +
          σ (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) *
            -inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) := by
  have h : ∀ j : Idx 2, HasDerivAt
      (fun s : ℝ => ω j * (σ (inner (𝕜 := ℝ) (a j) (circlePoint s : EucSpace 2)) *
        inner (𝕜 := ℝ) (a j) (circleVel s)))
      (ω j * (σ' (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) *
          inner (𝕜 := ℝ) (a j) (circleVel θ) ^ 2 +
        σ (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) *
          -inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2))) θ := by
    intro j
    have hmul := (((hσ _).comp θ (hasDerivAt_inner_circlePoint (a j) θ))).mul
      (hasDerivAt_inner_circleVel (a j) θ)
    exact (hmul.congr_deriv (by simp only [Function.comp_apply]; ring)).const_mul (ω j)
  have hsum := (HasDerivAt.sum (fun j (_ : j ∈ Finset.univ) => h j)).const_mul (2 : ℝ)
  rw [Function.iterate_succ, Function.iterate_one, Function.comp_apply,
    deriv_potential_circlePoint φ σ hφ ω a]
  exact hsum.deriv

/-- The hypotheses `φ' = 2σ` and `σ' = (σ)'` are satisfiable: the quadratic
`φ(s) = s²` is a primitive of `2σ` for the linear `σ(s) = s`, whose derivative
is the constant `1`. -/
example : (∀ s : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * (fun s : ℝ => s) s) s) ∧
    (∀ s : ℝ, HasDerivAt (fun t : ℝ => t) ((fun _ : ℝ => (1 : ℝ)) s) s) :=
  ⟨fun s => by simpa using hasDerivAt_pow 2 s, fun s => hasDerivAt_id' (𝕜 := ℝ) (x := s)⟩

end Perceptron
end Transformer

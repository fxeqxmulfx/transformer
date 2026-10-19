module

public import Mathlib.Analysis.Calculus.Gradient.Basic
public import Mathlib.LinearAlgebra.Eigenspace.Basic

/-!
# Saddle avoidance — definitions

The objects of the saddle-avoidance theorem for a map `g` of a real inner
product space into itself:

* the starts whose forward orbit stays in a closed ball (`stayNear`);
* the fixed points where the derivative of `g` is symmetric and stretches
  some direction (`IsSymmetricUnstable`);
* a splitting of a linear map into a stretching and a non-stretching part
  (`IsExpandingSplitting`);
* for a function `f`, the gradient step (`gradientStep`) and the strict
  saddles (`IsStrictSaddle`).

The sources are the gradient-descent corollary in §5.1 of arXiv:1710.07406
(Lee, Panageas, Piliouras, Simchowitz, Jordan, Recht, *First-order methods
almost always avoid strict saddle points*) and `thm:main` of arXiv:1602.04915
(Lee, Simchowitz, Jordan, Recht, *Gradient descent only converges to
minimizers*). Both rest on the center-stable manifold theorem of Shub; the
proof here replaces it by a cone estimate (`SaddleAvoidance/Cone.lean`).
-/

@[expose] public section

open Metric

namespace Transformer.Landscape.SaddleAvoidance

/-- The starts whose whole forward orbit under `g` stays in the closed ball of
radius `r` about `z`: the set `⋂ₖ g⁻ᵏ(B̄(z, r))` of the proof of `thm:main-general`
in §4 of arXiv:1710.07406. -/
def stayNear {X : Type*} [PseudoMetricSpace X] (g : X → X) (z : X) (r : ℝ) : Set X :=
  {x | ∀ k : ℕ, g^[k] x ∈ closedBall z r}

section Splitting

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- `P` splits off directions that `A` stretches by at least `c`: `P` commutes
with `A`, neither `P` nor `1 - P` lengthens a vector, `A` stretches every vector
of the form `P v` by at least the factor `c`, and lengthens no vector of the
form `v - P v`. -/
structure IsExpandingSplitting (A P : E →ₗ[ℝ] E) (c : ℝ) : Prop where
  comm : ∀ v, P (A v) = A (P v)
  norm_le : ∀ v, ‖P v‖ ≤ ‖v‖
  norm_sub_le : ∀ v, ‖v - P v‖ ≤ ‖v‖
  expand : ∀ v, c * ‖P v‖ ≤ ‖A (P v)‖
  contract : ∀ v, ‖A (v - P v)‖ ≤ ‖v - P v‖

end Splitting

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- `z` is a fixed point of `g` at which the derivative of `g` is symmetric and
has an eigenvalue of modulus greater than one.

The unstable fixed points of §4 of arXiv:1710.07406 are the fixed points whose
derivative has an eigenvalue of modulus greater than one; this predicate adds
that the derivative be symmetric, which holds for gradient descent, whose
derivative `1 - η ∇²f` is symmetric. -/
def IsSymmetricUnstable (g : E → E) (z : E) : Prop :=
  g z = z ∧ (fderiv ℝ g z : E →ₗ[ℝ] E).IsSymmetric ∧
    ∃ μ : ℝ, 1 < |μ| ∧ Module.End.HasEigenvalue (fderiv ℝ g z : E →ₗ[ℝ] E) μ

variable [CompleteSpace E]

/-- One step of gradient descent on `f` with step size `η`: `x - η ∇f(x)`, the
map `g` of `eq:g-gd` in §5.1 of arXiv:1710.07406. -/
noncomputable def gradientStep (f : E → ℝ) (η : ℝ) (x : E) : E :=
  x - η • gradient f x

/-- `z` is a strict saddle of `f`: a critical point at which the derivative of
the gradient, the Hessian, has a negative eigenvalue. Definition
`def:strict_saddle` of arXiv:1710.07406 asks for `λ_min(∇²f(z)) < 0`, which is
the same for the symmetric Hessian of a `C²` function; as there, local maxima
with a nondegenerate negative direction count as strict saddles. -/
def IsStrictSaddle (f : E → ℝ) (z : E) : Prop :=
  gradient f z = 0 ∧
    ∃ μ < 0, Module.End.HasEigenvalue (fderiv ℝ (gradient f) z : E →ₗ[ℝ] E) μ

end Transformer.Landscape.SaddleAvoidance

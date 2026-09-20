/-
# Perceptrons and attention's mean-field landscape — second-order critical points

Formalization of §2.2 of arXiv:2601.21366v2: the Wasserstein Hessian, and the
SOPD (second-order positive-definite) critical points `eq:2order`,
`eq:strict2order` that `thm: circle.gelu`, `thm: any.d` and `thm: bound` are
about.

**What the source says and what is carried here.**

* The geodesic the Hessian is read along, `μ(t) = (T(t))_# μ` with
  `T(t)(x) = exp_x(t ∇φ(x))`, is carried as data: a curve `ν` together with a
  measurable map realizing the pushforward at each time.  The source itself
  notes that for singular `μ` a geodesic need not be determined by its `L²(μ)`
  velocity field, so quantifying over every such curve is the faithful reading:
  `IsSOPD` asks for the inequality along every geodesic issued from `μ`.

* `exp_x` is the spherical exponential map, written out — there is no
  `Exp` for `Metric.sphere` in Mathlib, and `sphereExp` is a formula, not a
  difficulty parked in a definition: `norm_sphereExp` proves it lands on the
  sphere.

* The tangent space `eq:otto.tangent` is the `L²(μ)`-closure of the spherical
  gradients `∇φ` of smooth `φ`; the Hessian, however, is *defined* by the
  source only along `ξ = ∇φ`, and `eq:2order` is tested there.  `IsGradientField`
  is that dense subspace, and no closure is taken: testing on fewer directions
  is the weaker hypothesis, hence the stronger theorem.

* `Hess_μ E(ξ,ξ)` is `(d²/dt²)|_{t=0} E(ν t)`, carried as "for every `H` which
  is the second derivative at `0`" rather than as `iteratedDeriv 2`: the source
  assumes the Hessian is well-defined, and an `H` that no curve produces makes
  the condition vacuous in the right direction — a measure for which no second
  derivative exists is not asserted to be SOPD by anything downstream.

Source: arXiv:2601.21366v2, §2.1–2.2, `eq:otto.tangent`, `eq:2order`,
`eq:strict2order`.
-/

import Transformer.Perceptron.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### The spherical exponential map -/

/-- The exponential map of `𝕊^{d-1}` at `x` in the direction `v ⟂ x`:

  `exp_x(v) = cos‖v‖ · x + (sin‖v‖/‖v‖) · v`,

the great circle through `x` with initial velocity `v`.  At `v = 0` the second
summand is `(0/0) • 0 = 0`, so `exp_x(0) = x`.

Source: arXiv:2601.21366v2, §2.2 (footnote). -/
noncomputable def sphereExp (x v : EucSpace d) : EucSpace d :=
  Real.cos ‖v‖ • x + (Real.sin ‖v‖ / ‖v‖) • v

@[simp] theorem sphereExp_zero (x : EucSpace d) : sphereExp x (0 : EucSpace d) = x := by
  simp [sphereExp]

/-- The exponential map lands on the sphere. -/
theorem norm_sphereExp {x v : EucSpace d} (hx : ‖x‖ = 1)
    (hv : inner (𝕜 := ℝ) x v = 0) : ‖sphereExp x v‖ = 1 := by
  rcases eq_or_ne v 0 with rfl | hv0
  · simpa using hx
  have hvn : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv0
  have h0 : inner (𝕜 := ℝ) (Real.cos ‖v‖ • x) ((Real.sin ‖v‖ / ‖v‖) • v) = (0 : ℝ) := by
    rw [real_inner_smul_left, real_inner_smul_right, hv]; ring
  have hsq : ‖sphereExp x v‖ ^ 2 = 1 := by
    rw [sphereExp, norm_add_sq_real, h0, norm_smul, norm_smul, hx, Real.norm_eq_abs,
      Real.norm_eq_abs, mul_one, mul_pow, sq_abs, sq_abs, div_pow,
      div_mul_cancel₀ _ (pow_ne_zero 2 hvn)]
    linarith [Real.sin_sq_add_cos_sq ‖v‖]
  have hfac : (‖sphereExp x v‖ - 1) * (‖sphereExp x v‖ + 1) = 0 := by
    linear_combination hsq
  rcases mul_eq_zero.mp hfac with h | h
  · linarith
  · linarith [norm_nonneg (sphereExp x v)]

/-- Moving along a geodesic from `x`, the cosine of the angle to `x` is the
arclength: `⟪x, exp_x(t v)⟫ = cos(t‖v‖)` for `v ⟂ x`, `‖x‖ = 1`.

This is the whole second-order content of the exponential map that the
statements below need. -/
theorem inner_sphereExp_smul {x v : EucSpace d} (hx : ‖x‖ = 1)
    (hv : inner (𝕜 := ℝ) x v = 0) (t : ℝ) :
    inner (𝕜 := ℝ) x (sphereExp x (t • v)) = Real.cos (t * ‖v‖) := by
  have hxx : inner (𝕜 := ℝ) x x = (1 : ℝ) := by
    rw [real_inner_self_eq_norm_sq, hx]; norm_num
  have habs : ‖t • v‖ = |t * ‖v‖| := by
    rw [norm_smul, Real.norm_eq_abs, abs_mul, abs_of_nonneg (norm_nonneg v)]
  rw [sphereExp, inner_add_right, real_inner_smul_right, real_inner_smul_right,
    real_inner_smul_right, hv, hxx, habs, Real.cos_abs]
  ring

/-! ### Tangent fields and geodesics -/

/-- **`eq:otto.tangent`.**  `ξ` is the spherical gradient `∇φ` of a smooth
function: an element of the subspace whose `L²(μ)`-closure is the tangent space
`T_μ P(𝕊^{d-1})`, and the only kind of direction along which the source defines
the Wasserstein Hessian.

Source: arXiv:2601.21366v2, `eq:otto.tangent`. -/
def IsGradientField (ξ : SSphere d → EucSpace d) : Prop :=
  ∃ f : EucSpace d → ℝ, ContDiff ℝ (⊤ : ℕ∞) f ∧
    ∀ x : SSphere d, ξ x = proj d (x : EucSpace d) (gradient f (x : EucSpace d))

/-- The zero field is a gradient field, so `IsGradientField` is satisfiable. -/
theorem isGradientField_zero : IsGradientField (fun _ : SSphere d => (0 : EucSpace d)) :=
  ⟨fun _ => 0, contDiff_const, fun x => by simp [gradient, proj]⟩

/-- A gradient field is tangent to the sphere. -/
theorem inner_isGradientField {ξ : SSphere d → EucSpace d} (hξ : IsGradientField ξ)
    (x : SSphere d) : inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0 := by
  obtain ⟨f, _, hf⟩ := hξ
  rw [hf x]
  exact inner_proj_eq_zero (mem_sphere_zero_iff_norm.mp x.2) _

/-- **The `W₂`-geodesic issued from `μ` with velocity `ξ`**: `ν t` is the
pushforward of `μ` under `x ↦ exp_x(t ξ(x))`, the map being carried explicitly
because it is a map into the sphere.

Source: arXiv:2601.21366v2, §2.2. -/
def IsGeodesicFrom (ξ : SSphere d → EucSpace d) (μ : Perspective.ProbSphere d)
    (ν : ℝ → Perspective.ProbSphere d) : Prop :=
  ∀ t : ℝ, ∃ T : SSphere d → SSphere d, Measurable T ∧
    (∀ x : SSphere d, (T x : EucSpace d) = sphereExp (x : EucSpace d) (t • ξ x)) ∧
    (ν t : Measure (SSphere d)) = (μ : Measure (SSphere d)).map T

/-! ### SOPD critical points -/

/-- **`eq:2order`.**  A *second-order positive-definite* Wasserstein critical
point: stationary, with a non-negative Wasserstein Hessian in every direction.

Source: arXiv:2601.21366v2, `eq:2order`. -/
def IsSOPD (β : ℝ) (φ σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) : Prop :=
  IsStationary β σ ω a μ ∧
    ∀ ξ : SSphere d → EucSpace d, IsGradientField ξ →
      ∀ ν : ℝ → Perspective.ProbSphere d, IsGeodesicFrom ξ μ ν →
        ∀ H : ℝ, HasDerivAt (deriv fun t => energy β φ ω a (ν t)) H 0 → 0 ≤ H

/-- **`eq:strict2order`.**  A *strictly* SOPD Wasserstein critical point: the
Hessian dominates `κ‖ξ‖²_{L²(μ)}` for some `κ > 0`.

Source: arXiv:2601.21366v2, `eq:strict2order`. -/
def IsStrictSOPD (β : ℝ) (φ σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) : Prop :=
  IsStationary β σ ω a μ ∧
    ∃ κ : ℝ, 0 < κ ∧
      ∀ ξ : SSphere d → EucSpace d, IsGradientField ξ →
        ∀ ν : ℝ → Perspective.ProbSphere d, IsGeodesicFrom ξ μ ν →
          ∀ H : ℝ, HasDerivAt (deriv fun t => energy β φ ω a (ν t)) H 0 →
            κ * ∫ x, ‖ξ x‖ ^ 2 ∂(μ : Measure (SSphere d)) ≤ H

/-- A strictly SOPD critical point is SOPD: `κ‖ξ‖²_{L²(μ)} ≥ 0`. -/
theorem IsSOPD_of_IsStrictSOPD {β : ℝ} {φ σ : ℝ → ℝ} {ω : Idx d → ℝ}
    {a : Idx d → EucSpace d} {μ : Perspective.ProbSphere d}
    (h : IsStrictSOPD β φ σ ω a μ) : IsSOPD β φ σ ω a μ := by
  obtain ⟨hstat, κ, hκ, hH⟩ := h
  refine ⟨hstat, fun ξ hξ ν hν H hd => ?_⟩
  refine le_trans ?_ (hH ξ hξ ν hν H hd)
  have : (0 : ℝ) ≤ ∫ x, ‖ξ x‖ ^ 2 ∂(μ : Measure (SSphere d)) :=
    integral_nonneg fun x => by positivity
  positivity

end Perceptron
end Transformer

/-
# Perceptrons and attention's mean-field landscape — a general attention matrix

Formalization of `rem: general-attention` of arXiv:2601.21366v2: for an
invertible `B`, the map `μ ↦ f_B^μ`, `f_B^μ(x) = ∫ e^{xᵀ B y} dμ(y)`, is
injective on `𝒫(𝕊^{d-1})`, and `f_B^μ` is even exactly when `μ` is
antipodally symmetric.

**What the source says and what is carried here.**

* The source fixes a *symmetric* nonsingular `B`.  Symmetry is never used and
  is dropped: `xᵀ B y = (Bᵀ x)·y`, and only the invertibility of `Bᵀ` enters —
  the source's own ellipsoid is `{Bᵀ x : ‖x‖ < 1}`.  Dropping a hypothesis only
  strengthens a statement.

* The source's standing `d ≥ 2` is dropped, as in `Transform`.

* The source solves the Dirichlet problem for `Δ - 1` on the ellipsoid,
  continues `F_ν` analytically from a neighbourhood of `0`, and reads off the
  moments.  Here no derivative is taken.  With `w_i = (Bᵀ)⁻¹ e_i`, the second
  differences of `f(x) = ∫ e^{xᵀ B y} dμ(y)` satisfy
  `Σ_i [f(x + t w_i) + f(x - t w_i) - 2 f(x)] = t² f(x) + O(t⁴ f(x))`
  (`abs_secondDiff_sub_le`), the discrete form of `Δ F = F`; at an interior
  positive maximum of `f_{μ₁} - f_{μ₂}` the left side is `≤ 0`
  (`le_zero_of_secondDiff_ge`), so `f_{μ₁} = f_{μ₂}` on the closed unit ball.
  The ball contains `r (Bᵀ)⁻¹ 𝕊^{d-1}` for some `r > 0`, where `f_B^μ` is the
  isotropic transform at `β = r`, and `eq_of_integral_exp_inner_eq` finishes:
  the analytic continuation is not needed.  The second differences are
  `TransformExt`, the maximum principle `SecondDiff`.

Source: arXiv:2601.21366v2, `rem: general-attention`.
-/

import Transformer.Perceptron.TransformExt
import Mathlib.Analysis.InnerProductSpace.Adjoint

open scoped BigOperators
open Real MeasureTheory Filter Topology

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### Injectivity and parity -/

/-- **Remark (rem: general-attention), injectivity.**  For an invertible `B` the
map `μ ↦ f_B^μ` is injective on `𝒫(𝕊^{d-1})`.

The source fixes `B` symmetric as well; symmetry is not used (module
docstring).  With `D = (B⁻¹)ᵀ`, `(D v)·(B y) = v·y`: the directions
`w_i = D e_i` feed `attentionTransformMapExt_eq_of_eq_on_sphere`, and on
`D(r 𝕊^{d-1})`, inside the ball for small `r > 0`, `f_B^μ` is the isotropic
transform at `β = r`.

Source: arXiv:2601.21366v2, `rem: general-attention`. -/
theorem injective_attentionTransformMap (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (hB : Function.Bijective B) : Function.Injective (attentionTransformMap B) := by
  intro μ₁ μ₂ h
  set D : EucSpace d →ₗ[ℝ] EucSpace d :=
    LinearMap.adjoint (LinearEquiv.ofBijective B hB).symm.toLinearMap with hD_def
  have hD : ∀ v y : EucSpace d, inner (𝕜 := ℝ) (D v) (B y) = inner (𝕜 := ℝ) v y :=
    fun v y => by
      rw [hD_def, LinearMap.adjoint_inner_left, LinearEquiv.coe_toLinearMap]
      congr 1
      exact (LinearEquiv.ofBijective B hB).symm_apply_apply y
  have hball : ∀ x : EucSpace d, ‖x‖ ≤ 1 →
      attentionTransformMapExt B μ₁ x = attentionTransformMapExt B μ₂ x := fun x hx =>
    attentionTransformMapExt_eq_of_eq_on_sphere B (fun i => D (EuclideanSpace.single i 1))
      (fun i y => by rw [hD, EuclideanSpace.inner_single_left]; simp) _ _
      (fun x hx => by
        have := congrFun h ⟨x, mem_sphere_zero_iff_norm.2 hx⟩
        unfold attentionTransformMap at this
        exact this) hx
  obtain ⟨δ, hδ, hδD⟩ := Metric.continuousAt_iff.1
    (D.continuous_of_finiteDimensional.continuousAt (x := 0)) 1 one_pos
  refine ProbabilityMeasure.toMeasure_injective
    (eq_of_integral_exp_inner_eq (δ / 2) (by positivity) _ _ fun u => ?_)
  have hu : ‖D ((δ / 2) • (u : EucSpace d))‖ ≤ 1 := by
    have := @hδD ((δ / 2) • (u : EucSpace d)) (by
      rw [dist_zero_right, norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity),
        norm_eq_of_mem_sphere u, mul_one]
      linarith)
    rw [map_zero, dist_zero_right] at this
    exact this.le
  have := hball _ hu
  simp only [attentionTransformMapExt, hD, real_inner_smul_left] at this
  exact this

/-- **Remark (rem: general-attention), parity.**  For an invertible `B`,
`f_B^μ` is even if and only if `μ(A) = μ(-A)` for every Borel `A`.

The source fixes `B` symmetric as well; symmetry is not used.

Source: arXiv:2601.21366v2, `rem: general-attention`, `lem: quadpol` (ii). -/
theorem even_attentionTransformMap_iff (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (hB : Function.Bijective B) (μ : Perspective.ProbSphere d) :
    (∀ x : SSphere d, attentionTransformMap B μ (antipodeMap d x)
        = attentionTransformMap B μ x)
      ↔ (μ : Measure (SSphere d)).map (antipodeMap d) = (μ : Measure (SSphere d)) :=
  even_attentionTransformMap_iff_of_injective B (injective_attentionTransformMap B hB) μ

/-- The hypothesis of `injective_attentionTransformMap` and
`even_attentionTransformMap_iff` is satisfiable: `B = id`. -/
example : Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace d)) :=
  Function.bijective_id

end Perceptron
end Transformer

/-
# Appendix B — the Hessian at a critical point is intrinsic

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`e:Hessianincoord`: in coordinates `Hess(f)_{ij} = ∂²f/∂y_i∂y_j - Γ_{ij}^k
∂f/∂y_k`, and at a critical point the Christoffel term drops out — so the
second derivative of `𝖤_β` along a curve depends on the curve only through its
initial velocity.

The proof is the second-order expansion `selfEnergy_expansion_aux` of
`Perspective.AppendixB_Expansion`, whose quadratic coefficient `hessQuad` is
written in the velocity alone, against the uniqueness of a Peano expansion,
`Perspective.eq_zero_of_isLittleO_pow`.
-/

import Transformer.Perspective.AppendixB_Expansion
import Transformer.Perspective.PeanoTaylor

open scoped BigOperators
open Real Asymptotics Filter

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- The quadratic coefficient of `𝖤_β` along any curve through `X` with
velocity `v`, at a critical point `X`:

  `(2β)⁻¹ [ β (Σ_{i,j} K_{ij} ⟨v_i, v_j⟩ - Σ_i λ_i ‖v_i‖²)
            + Σ_{i,j} K_{ij} β² (⟨x_i, v_j⟩ + ⟨v_i, x_j⟩)² / 2 ]`,

with `K_{ij} = e^{β⟨x_i,x_j⟩}` and `λ_i = ⟨x_i, Σ_j K_{ij} x_j⟩`.  It is half
the Riemannian Hessian `Hess 𝖤_β(X)[v, v]`.

Source: arXiv:2312.10794v5, Appendix B, `e:Hessianincoord`. -/
noncomputable def hessQuad (β : ℝ) (X : SphereTuple d n) (v : Idx n → EucSpace d) : ℝ :=
  (2 * β)⁻¹ *
    (β * ((∑ i : Idx n, ∑ j : Idx n,
              Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
                * inner (𝕜 := ℝ) (v i) (v j))
            - ∑ i : Idx n,
                inner (𝕜 := ℝ) ((X i : EucSpace d))
                  (∑ j : Idx n,
                    Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
                      • ((X j : EucSpace d))) * ‖v i‖ ^ 2)
      + ∑ i : Idx n, ∑ j : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
            * (β ^ 2 * (inner (𝕜 := ℝ) ((X i : EucSpace d)) (v j)
                + inner (𝕜 := ℝ) (v i) ((X j : EucSpace d))) ^ 2 / 2))

/-- **The second-order expansion of `𝖤_β` at a critical point.**

  `𝖤_β(Y t) = 𝖤_β(X) + hessQuad(X, v) t² + o(t²)`

for every curve `Y` through the critical point `X` with velocity `v`.  The
first-order term is absent — that is criticality — and the second-order one
depends on `Y` only through `v`.

Source: arXiv:2312.10794v5, Appendix B, `e:Hessianincoord`. -/
theorem isLittleO_selfEnergy_secondOrder
    (β : ℝ) (X : SphereTuple d n) (hX : IsCriticalEBeta d n β X)
    (Y : ℝ → SphereTuple d n) (v : Idx n → EucSpace d) (hY0 : Y 0 = X)
    (hY : ∀ i : Idx n, HasDerivAt (fun s => (Y s i : EucSpace d)) (v i) 0) :
    (fun t : ℝ => selfEnergy d n β (Y t) - selfEnergy d n β X - hessQuad d n β X v * t ^ 2)
      =o[nhds 0] fun t : ℝ => t ^ 2 := by
  have haux := selfEnergy_expansion_aux d n β (fun i => ((X i : EucSpace d))) v
    (fun t i => ((Y t i : EucSpace d)))
    (fun i j => Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d))))
    (fun i => inner (𝕜 := ℝ) ((X i : EucSpace d))
      (∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
          • ((X j : EucSpace d))))
    (fun i j => inner (𝕜 := ℝ) ((X i : EucSpace d)) (v j)
      + inner (𝕜 := ℝ) (v i) ((X j : EucSpace d)))
    (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl)
    (fun i => mem_sphere_zero_iff_norm.mp (X i).2)
    (fun t i => mem_sphere_zero_iff_norm.mp (Y t i).2)
    (fun i => by rw [hY0]) hY hX
  refine (haux.const_mul_left (2 * β)⁻¹).congr' ?_ (by rfl)
  filter_upwards with t
  simp only [selfEnergy, particleEnergy, hessQuad, ContinuousLinearMap.id_apply]
  ring

/-- **Equation (e:Hessianincoord).** *The Hessian at a critical point is
intrinsic.*

At a critical point of `𝖤_β` the second derivative of `𝖤_β` along a curve
depends on the curve only through its initial velocity — which is exactly why
the strict-saddle property does not depend on the metric.

The proof: both `c` and `c'` equal `2 · hessQuad(X, v)`.  The curve supplies a
Taylor–Young expansion with coefficient `c/2`, `isLittleO_selfEnergy_secondOrder`
supplies one with coefficient `hessQuad(X, v)`, and a Peano expansion has at
most one pair of coefficients.

Source: arXiv:2312.10794v5, Appendix B, `e:Hessianincoord`. -/
theorem hessian_at_critical_intrinsic
    (β : ℝ) (X : SphereTuple d n) (hX : IsCriticalEBeta d n β X)
    (Y Z : ℝ → SphereTuple d n) (v : Idx n → EucSpace d)
    (hY0 : Y 0 = X) (hZ0 : Z 0 = X)
    (hY : ∀ i : Idx n, HasDerivAt (fun s => (Y s i : EucSpace d)) (v i) 0)
    (hZ : ∀ i : Idx n, HasDerivAt (fun s => (Z s i : EucSpace d)) (v i) 0)
    (c c' : ℝ) (hc : SecondDerivEBetaAt d n β Y c)
    (hc' : SecondDerivEBetaAt d n β Z c') :
    c = c' := by
  have key : ∀ (W : ℝ → SphereTuple d n) (cc : ℝ), W 0 = X →
      (∀ i : Idx n, HasDerivAt (fun s => (W s i : EucSpace d)) (v i) 0) →
      SecondDerivEBetaAt d n β W cc → cc = 2 * hessQuad d n β X v := by
    rintro W cc hW0 hW ⟨f', hf', hcc⟩
    have hexp := isLittleO_selfEnergy_secondOrder d n β X hX W v hW0 hW
    have htay := isLittleO_secondOrder hf' hcc
    have hf0 : selfEnergy d n β (W 0) = selfEnergy d n β X := by rw [hW0]
    have hsub : (fun t : ℝ => f' 0 * t + (cc / 2 - hessQuad d n β X v) * t ^ 2)
        =o[nhds 0] fun t : ℝ => t ^ 2 := by
      refine (hexp.sub htay).congr' ?_ (by rfl)
      filter_upwards with t
      rw [hf0]; ring
    have hlin : (fun t : ℝ => f' 0 * t ^ 1) =o[nhds 0] fun t : ℝ => t ^ 1 := by
      have h1 : (fun t : ℝ => f' 0 * t + (cc / 2 - hessQuad d n β X v) * t ^ 2)
          =o[nhds 0] fun t : ℝ => t := hsub.trans_isBigO isLittleO_sq_id.isBigO
      have h2 : (fun t : ℝ => (cc / 2 - hessQuad d n β X v) * t ^ 2)
          =o[nhds 0] fun t : ℝ => t := isLittleO_sq_id.const_mul_left _
      refine (h1.sub h2).congr' ?_ ?_
      · filter_upwards with t; ring
      · filter_upwards with t; ring
    have hf'0 : f' 0 = 0 := eq_zero_of_isLittleO_pow hlin
    have hquad : (fun t : ℝ => (cc / 2 - hessQuad d n β X v) * t ^ 2)
        =o[nhds 0] fun t : ℝ => t ^ 2 := by
      refine hsub.congr' ?_ (by rfl)
      filter_upwards with t
      rw [hf'0]; ring
    have := eq_zero_of_isLittleO_pow hquad
    linarith
  rw [key Y c hY0 hY hc, key Z c' hZ0 hZ hc']

/-! ### The hypotheses are satisfiable -/

/-- The hypotheses of `isLittleO_selfEnergy_secondOrder` and of
`hessian_at_critical_intrinsic` are satisfiable: the single token sits still,
so both curves are the constant one, their common velocity is `0`, and the
energy they carry is constant, hence has second derivative `0` along each of
them. -/
example :
    IsCriticalEBeta 1 1 1 singleToken ∧
      ((fun _ : ℝ => singleToken) 0 = singleToken) ∧
      (∀ i : Idx 1,
        HasDerivAt (fun s => (((fun _ : ℝ => singleToken) s i : SSphere 1) : EucSpace 1))
          ((fun _ : Idx 1 => (0 : EucSpace 1)) i) 0) ∧
      SecondDerivEBetaAt 1 1 1 (fun _ => singleToken) 0 := by
  refine ⟨singleToken_isSkew_critical_hessianNonPos.2.1, rfl,
    fun _ => hasDerivAt_const _ _, ⟨fun _ => 0, fun t => ?_, hasDerivAt_const _ _⟩⟩
  exact hasDerivAt_const t _

end Perspective
end Transformer

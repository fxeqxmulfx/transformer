/-
# Appendix B — the `d`-dimensional statements

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The part of Appendix B that runs on `(𝕊^{d-1})^n` rather than on the torus:

* `eq: claim.yury`   — the sub-block inequality in dimension `d`,
* `eq: dr1`          — the skew-symmetric perturbation inequality, proved,
* `e:Hessianincoord` — the Hessian at a critical point is intrinsic.

The comparison of the modified metric of §3.4 with the round one,
`eq: metric.grad` and `eq: metric.hess`, is proved in
`Perspective.AppendixB_MetricGrad` and `Perspective.AppendixB_MetricHess`,
which are built on this file.

The energy `𝖤_β`, its critical points and the non-positive-Hessian hypothesis
are in `Perspective.AppendixB_EBeta`; the second derivative of `𝖤_β` along a
block rotation is computed in `Perspective.AppendixB_HessBeta`, and `eq: dr1`
is that computation read against the hypothesis.
-/

import Transformer.Perspective.AppendixB_BetaInterval
import Transformer.Perspective.AppendixB_HessBeta

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **Equation (eq: claim.yury).** *Higher-dimensional generalization of
`eq: taylor3`.*

For a critical point `(x_1,…,x_n) ∈ (𝕊^{d-1})^n` of `𝖤_β` with non-positive
Hessian, and `θ_{ij} = arccos ⟨x_i, x_j⟩ ∈ [0, π]` the geodesic distance,

  `Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} g_β(θ_{ij}) ≥ 0`.

Source: arXiv:2312.10794v5, Appendix B, `eq: claim.yury`. -/
theorem claim_yury
    (β : ℝ) (X : SphereTuple d n) (𝒮 : Finset (Idx n))
    (h_crit : IsCriticalEBeta d n β X) (h_hess : EBetaHessianNonPos d n β X) :
    0 ≤ ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
      g_β_d d β
        (Real.arccos (inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))) := by
  sorry

/-- **Equation (eq:dr1).**  The second-order term of the block rotation by a
skew-symmetric `B`, written out at a point with non-positive Hessian:

  `Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} e^{β ⟨x_i, x_j⟩}
        (β ⟨B x_i, x_j⟩² + ⟨B² x_i, x_j⟩) ≤ 0`.

The proof is `secondDeriv_selfEnergy` — the sum above is `𝖤_β''(0)` up to the
positive factor `(2β)⁻¹ · 2β = 1` — against `EBetaHessianNonPos`, applied to
the rotation `exists_perturbationBy` produces.

**What the source says and what is changed here.**  Two deviations.

*A hypothesis is added.*  The survey states `eq: dr1` for `β > 0`, where
`𝖤_β = (2β)⁻¹ Σ e^{β⟨x_i,x_j⟩}` is what it is; here `β` ranges over all of
`ℝ`, and at `β = 0` Lean reads `(2·0)⁻¹` as `0`, so `𝖤_0` in this sense is the
constant `0`, every second derivative vanishes, `EBetaHessianNonPos` holds
vacuously — and the conclusion is then false, since `Perspective.yury_lemma`
produces an `X`, a `B` and an `𝒮` with `Σ_{i∈𝒮} Σ_{j∈𝒮^c} ⟨B² x_i, x_j⟩ > 0`.
The hypothesis `β ≠ 0` is what the statement needs and what the paper has.

*A hypothesis is dropped.*  The survey states `eq: dr1` at a critical point,
because that is where it uses it; criticality enters the proof nowhere — the
second derivative alone carries the inequality — so `IsCriticalEBeta` is not
assumed, exactly as in `Perspective.hessian_at_critical` at `β = 0`.

Source: arXiv:2312.10794v5, Appendix B, `eq: dr1`. -/
theorem dr1_skew_inequality
    (β : ℝ) (hβ : β ≠ 0) (X : SphereTuple d n) (𝒮 : Finset (Idx n)) (B : ParamMatrix d)
    (h_skew : IsSkew d B) (h_hess : EBetaHessianNonPos d n β X) :
    (∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
        Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
          * (β * (inner (𝕜 := ℝ) (B (X i)) ((X j : EucSpace d)))^2
              + inner (𝕜 := ℝ) (B (B (X i))) ((X j : EucSpace d)))) ≤ 0 := by
  obtain ⟨Y, hY⟩ := exists_perturbationBy d n B h_skew 𝒮 X
  have h := h_hess B 𝒮 h_skew Y hY _ (secondDeriv_selfEnergy d n X β B 𝒮 h_skew Y hY)
  rwa [inv_mul_cancel_left₀ (mul_ne_zero (two_ne_zero) hβ)] at h

/-- **Equation (e:Hessianincoord).** *The Hessian at a critical point is
intrinsic.*

In coordinates `Hess(f)_{ij} = ∂²f/∂y_i ∂y_j - Γ_{ij}^k ∂f/∂y_k`, and at a
critical point the Christoffel term drops out.  Stated without Christoffel
symbols: at a critical point the second derivative of `𝖤_β` along a curve
depends on the curve only through its initial velocity — which is exactly why
the strict-saddle property does not depend on the metric.

Not proved here: the second-order expansion is not carried out.

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
  sorry

/-! ### The hypotheses are satisfiable -/

/-- The hypotheses of `claim_yury`, `dr1_skew_inequality` and
`metric_hess_comparison` are satisfiable: `β = 1` is non-zero, and the single
token is a critical point with non-positive Hessian. -/
example :
    (1 : ℝ) ≠ 0 ∧ IsSkew 1 0 ∧ IsCriticalEBeta 1 1 1 singleToken ∧
      EBetaHessianNonPos 1 1 1 singleToken :=
  ⟨one_ne_zero, singleToken_isSkew_critical_hessianNonPos⟩

/-- The hypotheses of `hessian_at_critical_intrinsic` are satisfiable: the
single token sits still, so both curves are the constant one, their common
velocity is `0`, and the energy they carry is constant, hence has second
derivative `0` along each of them. -/
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

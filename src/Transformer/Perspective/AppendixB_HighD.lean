/-
# Appendix B — the `d`-dimensional statements

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The part of Appendix B that runs on `(𝕊^{d-1})^n` rather than on the torus:

* `eq: claim.yury`   — the sub-block inequality in dimension `d`,
* `eq: dr1`          — the skew-symmetric perturbation inequality,
* `e:Hessianincoord` — the Hessian at a critical point is intrinsic.

The comparison of the modified metric of §3.4 with the round one,
`eq: metric.grad` and `eq: metric.hess`, is proved in
`Perspective.AppendixB_MetricGrad` and `Perspective.AppendixB_MetricHess`,
which are built on this file.

The energy is `Perspective.particleEnergy` at `V = Id`; the skew-symmetric
perturbations and `𝖤_0` come from `Perspective.AppendixA_Saddle`.
-/

import Transformer.Perspective.AppendixB_BetaInterval
import Transformer.Perspective.AppendixA_Saddle
import Transformer.Perspective.Section2_GradientFlow

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- The interaction energy of Appendix B,

  `𝖤_β(X) = (2β)⁻¹ Σ_i Σ_j e^{β ⟨x_i, x_j⟩}`,

that is `Perspective.particleEnergy` at `V = Id`. -/
noncomputable def selfEnergy (β : ℝ) (X : SphereTuple d n) : ℝ :=
  particleEnergy d n β (ContinuousLinearMap.id ℝ (EucSpace d)) X

/-- With a single token the only interaction is the token with itself, so
`𝖤_β` is the constant `(2β)⁻¹ e^β`. -/
theorem selfEnergy_one (β : ℝ) (X : SphereTuple d 1) :
    selfEnergy d 1 β X = (2 * β)⁻¹ * Real.exp β := by
  have hx : ‖(X 0 : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (X 0).2
  simp [selfEnergy, particleEnergy, hx]

/-- A critical point of `𝖤_β` on `(𝕊^{d-1})^n`: the Riemannian gradient

  `Proj_{x_i} Σ_j e^{β ⟨x_i, x_j⟩} x_j`

vanishes at every token.  It is the right-hand side of `USA`, so the critical
points are exactly the stationary configurations of the dynamics. -/
def IsCriticalEBeta (β : ℝ) (X : SphereTuple d n) : Prop :=
  ∀ i : Idx n,
    proj d ((X i : EucSpace d))
      (∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
          • ((X j : EucSpace d))) = 0

/-- `c` is the second derivative of `t ↦ 𝖤_β(Y t)` at `t = 0`, in the same
sense as `Perspective.SecondDerivE0At` for `𝖤_0`. -/
def SecondDerivEBetaAt (β : ℝ) (Y : ℝ → SphereTuple d n) (c : ℝ) : Prop :=
  ∃ f' : ℝ → ℝ,
    (∀ t : ℝ, HasDerivAt (fun s => selfEnergy d n β (Y s)) (f' t) t) ∧
    HasDerivAt f' c 0

/-- *Non-positive Hessian at `X`*, along the block rotations Appendix B
perturbs by: for every skew-symmetric `B` and every `𝒮 ⊂ [n]`, rotating the
tokens of `𝒮` by `e^{tB}` does not increase `𝖤_β` to second order.

This is the hypothesis placed on a critical point that is not a strict saddle.

Source: arXiv:2312.10794v5, Appendix B. -/
def EBetaHessianNonPos (β : ℝ) (X : SphereTuple d n) : Prop :=
  ∀ (B : ParamMatrix d) (𝒮 : Finset (Idx n)), IsSkew d B →
    ∀ Y : ℝ → SphereTuple d n, PerturbationBy d n B 𝒮 X Y →
      ∀ c : ℝ, SecondDerivEBetaAt d n β Y c → c ≤ 0

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
skew-symmetric `B`, written out at a critical point with non-positive Hessian:

  `Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} e^{β ⟨x_i, x_j⟩}
        (β ⟨B x_i, x_j⟩² + ⟨B² x_i, x_j⟩) ≤ 0`.

Source: arXiv:2312.10794v5, Appendix B, `eq: dr1`. -/
theorem dr1_skew_inequality
    (β : ℝ) (X : SphereTuple d n) (𝒮 : Finset (Idx n)) (B : ParamMatrix d)
    (h_skew : IsSkew d B)
    (h_crit : IsCriticalEBeta d n β X) (h_hess : EBetaHessianNonPos d n β X) :
    (∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
        Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
          * (β * (inner (𝕜 := ℝ) (B (X i)) ((X j : EucSpace d)))^2
              + inner (𝕜 := ℝ) (B (B (X i))) ((X j : EucSpace d)))) ≤ 0 := by
  sorry

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

/-- The single token of `𝕊^0 ⊂ ℝ^1`. -/
noncomputable def singleToken : SphereTuple 1 1 :=
  fun _ => ⟨EuclideanSpace.single (0 : Fin 1) (1 : ℝ), by simp⟩

/-- A single token is a critical point of `𝖤_β` — its gradient is a multiple
of `Proj_x x = 0` — and `𝖤_β` is constant on `(𝕊^{d-1})^1`, so every second
derivative vanishes and the Hessian is non-positive.  The zero matrix is
skew-symmetric.  This is the witness the statements of this file are tested
against. -/
theorem singleToken_isSkew_critical_hessianNonPos :
    IsSkew 1 0 ∧ IsCriticalEBeta 1 1 1 singleToken ∧
      EBetaHessianNonPos 1 1 1 singleToken := by
  refine ⟨fun x y => by simp, ?_, ?_⟩
  · intro i
    have h0 : ∀ j : Idx 1, (singleToken j : EucSpace 1) = (singleToken i : EucSpace 1) := by
      intro j; rw [Subsingleton.elim j i]
    have hx : ‖(singleToken i : EucSpace 1)‖ = 1 :=
      mem_sphere_zero_iff_norm.mp (singleToken i).2
    simp only [h0, Finset.sum_const, Finset.card_univ, Fintype.card_fin, one_smul]
    rw [proj_smul]
    have hself : proj 1 (singleToken i : EucSpace 1) (singleToken i : EucSpace 1) = 0 := by
      simp [proj, hx]
    rw [hself, smul_zero]
  · rintro B 𝒮 _ Y _ c ⟨f', hf', hc⟩
    have hconst : (fun s : ℝ => selfEnergy 1 1 1 (Y s))
        = fun _ : ℝ => (2 * (1 : ℝ))⁻¹ * Real.exp 1 := by
      funext s
      exact selfEnergy_one 1 1 (Y s)
    have hf0 : f' = fun _ : ℝ => (0 : ℝ) := by
      funext s
      exact (hf' s).unique (by rw [hconst]; exact hasDerivAt_const s _)
    rw [hf0] at hc
    have hc0 : c = 0 := hc.unique (hasDerivAt_const (0 : ℝ) (0 : ℝ))
    simp [hc0]

/-- The hypotheses of `claim_yury`, `dr1_skew_inequality` and
`metric_hess_comparison` are satisfiable. -/
example :
    IsSkew 1 0 ∧ IsCriticalEBeta 1 1 1 singleToken ∧
      EBetaHessianNonPos 1 1 1 singleToken :=
  singleToken_isSkew_critical_hessianNonPos

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

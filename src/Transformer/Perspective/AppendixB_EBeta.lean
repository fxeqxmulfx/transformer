/-
# Appendix B — the energy `𝖤_β` on `(𝕊^{d-1})^n`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The vocabulary the `d`-dimensional part of Appendix B is written in: the
interaction energy

  `𝖤_β(X) = (2β)⁻¹ Σ_i Σ_j e^{β ⟨x_i, x_j⟩}`,

which is `Perspective.particleEnergy` at `V = Id`, its critical points, the
second derivative of it along a curve, and the hypothesis "the Hessian at `X`
is non-positive" that a critical point which is not a strict saddle satisfies.

The statements built on this are in `Perspective.AppendixB_HessBeta` (the
Hessian along a block rotation, `eq: dr1`), `Perspective.AppendixB_HighD`
(`eq: claim.yury`) and `Perspective.AppendixB_Intrinsic`
(`e:Hessianincoord`).
-/

import Transformer.Perspective.AppendixA_Rotation
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

/-! ### The hypotheses are satisfiable -/

/-- The single token of `𝕊^0 ⊂ ℝ^1`. -/
noncomputable def singleToken : SphereTuple 1 1 :=
  fun _ => ⟨EuclideanSpace.single (0 : Fin 1) (1 : ℝ), by simp⟩

/-- A single token is a critical point of `𝖤_β` — its gradient is a multiple
of `Proj_x x = 0` — and `𝖤_β` is constant on `(𝕊^{d-1})^1`, so every second
derivative vanishes and the Hessian is non-positive.  The zero matrix is
skew-symmetric.  This is the witness the statements of Appendix B are tested
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

end Perspective
end Transformer

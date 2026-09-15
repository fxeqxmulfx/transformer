/-
# §3.3 — the particle version of the Wasserstein gradient flow

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, `rem: particle.gf`.

At an empirical measure `μ(t) = n⁻¹ Σ_i δ_{x_i(t)}` the interaction energy
`eq: interaction.energy` takes the form

  `𝖤_β(X) = (1/(2βn²)) Σ_i Σ_j e^{β⟨x_i,x_j⟩}`,

and `USA` is its gradient *ascent* flow for the round metric on
`(𝕊^{d-1})^n`, sped up by `n`:

  `Ẋ(t) = n ∇_X 𝖤_β(X(t))`.                                  (`e:dynonX`)

The Riemannian gradient on `(𝕊^{d-1})^n` is not carried by this development,
so — as in `Perspective.SAIsGradientFlow` — the identity is stated through
directional derivatives along curves, which is what `∇` abbreviates: the
differential of `𝖤_β` at `X(t)` in a tangent direction `b` is `n⁻¹ ⟨Ẋ(t), b⟩`.
Against `SAIsGradientFlow` the only difference is the metric: the round one
here, `e:scalarproduct` there.  That is exactly the difference between `USA`
and `SA`.
-/

import Transformer.Perspective.Section2_GradientFlow

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- The empirical interaction energy of `rem: particle.gf`,

  `𝖤_β(X) = (1/(2βn²)) Σ_i Σ_j e^{β⟨x_i,x_j⟩}`,

i.e. `Perspective.particleEnergy` at `V = Id` carrying the `n⁻²` of the
empirical measure `n⁻¹ Σ_i δ_{x_i}`. -/
noncomputable def empiricalEnergy (β : ℝ) (X : SphereTuple d n) : ℝ :=
  ((n : ℝ) ^ 2)⁻¹ * particleEnergy d n β (ContinuousLinearMap.id ℝ (EucSpace d)) X

/-- **Equation (e:dynonX).**  *`USA` is a gradient ascent flow for the round
metric.*

For every curve `Y` through `X(t)` with velocity `b`,

  `d/ds 𝖤_β(Y(s))|_{s=0} = n⁻¹ Σ_i ⟨ẋ_i(t), b_i⟩`,

which is `Ẋ(t) = n ∇_X 𝖤_β(X(t))` read against the round metric
`⟨a, b⟩ = Σ_i ⟨a_i, b_i⟩`.  Equivalently, per particle,
`∂_i 𝖤_β(X(t)) = n⁻¹ ẋ_i(t)`.

Taking `b = Ẋ(t)` gives `d/dt 𝖤_β(X(t)) = n⁻¹ Σ_i ‖ẋ_i(t)‖² ≥ 0`, the
particle counterpart of `lem: dissipation`.

Source: arXiv:2312.10794v5, §3.3, `rem: particle.gf`, `e:dynonX`. -/
theorem usa_isGradientFlow
    (β : ℝ) (X : ℝ → SphereTuple d n) (hX : USA d n β X)
    (t : ℝ) (Y : ℝ → SphereTuple d n) (b : Idx n → EucSpace d)
    (hY : Y 0 = X t)
    (hb : ∀ i : Idx n, HasDerivAt (fun s => (Y s i : EucSpace d)) (b i) 0) :
    HasDerivAt (fun s => empiricalEnergy d n β (Y s))
      ((n : ℝ)⁻¹ *
        ∑ i : Idx n,
          inner (𝕜 := ℝ) (deriv (fun s => (X s i : EucSpace d)) t) (b i)) 0 := by
  sorry

/-- The single token of `𝕊^0 ⊂ ℝ^1`. -/
noncomputable def oneToken : SphereTuple 1 1 :=
  fun _ => ⟨EuclideanSpace.single (0 : Fin 1) (1 : ℝ), by simp⟩

/-- The hypotheses of `usa_isGradientFlow` are satisfiable: a single token is
a stationary solution of `USA`, since its velocity is `Proj_x x = 0`, and the
constant curve through it has velocity `b = 0`. -/
example :
    USA 1 1 1 (fun _ => oneToken) ∧
      ∀ i : Idx 1,
        HasDerivAt (fun s : ℝ => (((fun _ => oneToken) s i : EucSpace 1))) 0 0 := by
  have hx : ‖(oneToken 0 : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (oneToken 0).2
  have hinner : inner (𝕜 := ℝ) ((oneToken 0 : EucSpace 1)) ((oneToken 0 : EucSpace 1)) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx, mul_one]
  have hself : ∀ c : ℝ,
      proj 1 ((oneToken 0 : EucSpace 1)) (c • (oneToken 0 : EucSpace 1)) = 0 := by
    intro c
    rw [proj, real_inner_smul_right, hinner, mul_one, sub_self]
  refine ⟨fun t i => ?_, fun i => hasDerivAt_const (0 : ℝ) _⟩
  have hi : (oneToken i : EucSpace 1) = (oneToken 0 : EucSpace 1) := by
    rw [Subsingleton.elim i 0]
  simpa [hi, hinner, hself] using hasDerivAt_const t ((oneToken 0 : EucSpace 1))

end Perspective
end Transformer

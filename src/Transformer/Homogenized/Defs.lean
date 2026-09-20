/-
# Homogenized Transformers — the model

Formalization of the setup of arXiv:2604.01978v1, *Homogenized Transformers*
(Geshkovski, Koubbi, Rigollet), §2.1 and §5.

An encoder, attention-only Transformer whose head parameters
`θ = (V, A) ∈ Θ = (ℝ^{d×d})²` are i.i.d. across heads and layers.  Tokens live
on `𝕊^{d-1}` and move by `eq:update_tokens`,

  `x_i^{ℓ+1} = N(x_i^ℓ + (η/H) Σ_{h=1}^H B_{θ_h^ℓ}[μ_{X^ℓ}](x_i^ℓ))`,

with `N(x) = x/‖x‖` and `μ_X` the empirical measure of the tokens.

This file carries the vocabulary: the attention field `EQ:VELOCITY_FIELD_SELF_ATTENTION`,
the layer update, its piecewise-constant interpolation `X^η(t) = X^{⌊t/η⌋}`, the
drift-fluctuation splitting `B = b_{ρ*} + ξ_θ`, the projected kernels `b` and
`G` of `eq:G_def`, the variance proxy `σ²` of `eq: defining.alpha` and
`α = η σ²/H` of `eq:Alpha_Sec2`.
-/

import Transformer.Basic
import Mathlib.Analysis.Matrix.MeasurableSpace
import Mathlib.Analysis.Normed.Module.RCLike.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-- `Θ = (ℝ^{d×d})²`, the parameters `θ = (V, A)` of one attention head: the
value matrix and the collapsed query-key product `A = Qᵀ K`.

Source: arXiv:2604.01978v1, §2.1. -/
abbrev HeadParam (d : ℕ) : Type :=
  Matrix (Fin d) (Fin d) ℝ × Matrix (Fin d) (Fin d) ℝ

/-- The value matrix `V` of a head, as a map on `ℝ^d`. -/
noncomputable def valueMap {d : ℕ} (θ : HeadParam d) (y : EucSpace d) : EucSpace d :=
  Matrix.toEuclideanLin θ.1 y

/-- The query-key matrix `A` of a head, as a map on `ℝ^d`. -/
noncomputable def qkMap {d : ℕ} (θ : HeadParam d) (x : EucSpace d) : EucSpace d :=
  Matrix.toEuclideanLin θ.2 x

/-- The unnormalized attention weight `e^{β⟨A x, y⟩}`.

Source: arXiv:2604.01978v1, `EQ:VELOCITY_FIELD_SELF_ATTENTION`. -/
noncomputable def attnWeight {d : ℕ} (β : ℝ) (θ : HeadParam d) (x y : EucSpace d) : ℝ :=
  Real.exp (β * inner (𝕜 := ℝ) (qkMap θ x) y)

theorem attnWeight_pos {d : ℕ} (β : ℝ) (θ : HeadParam d) (x y : EucSpace d) :
    0 < attnWeight β θ x y :=
  Real.exp_pos _

/-- The attention-induced velocity field at the empirical measure `μ_X`,

  `B_θ[μ_X](z) = (Σ_k e^{β⟨A z, x_k⟩})⁻¹ Σ_k e^{β⟨A z, x_k⟩} V x_k`.

The `1/n` of `μ_X = (1/n) Σ_k δ_{x_k}` appears in both the numerator and the
normalizer `𝒵_A[μ_X](z)` of `EQ:VELOCITY_FIELD_SELF_ATTENTION` and cancels, so
the two plain sums below are that field.

Source: arXiv:2604.01978v1, `EQ:VELOCITY_FIELD_SELF_ATTENTION`. -/
noncomputable def attnField {d n : ℕ} (β : ℝ) (θ : HeadParam d)
    (x : Idx n → EucSpace d) (z : EucSpace d) : EucSpace d :=
  (∑ k : Idx n, attnWeight β θ z (x k))⁻¹ •
    ∑ k : Idx n, attnWeight β θ z (x k) • valueMap θ (x k)
/-- The normalization layer `N(x) = x/‖x‖`.

Source: arXiv:2604.01978v1, `eq:update_tokens`. -/
noncomputable def normalizeLayer {d : ℕ} (v : EucSpace d) : EucSpace d := ‖v‖⁻¹ • v
/-- One layer of `eq:update_tokens`, at the head parameters `θ = (θ_1,…,θ_H)`
of that layer:

  `x_i ↦ N(x_i + (η/H) Σ_{h=1}^H B_{θ_h}[μ_X](x_i))`.

Source: arXiv:2604.01978v1, `eq:update_tokens`. -/
noncomputable def layerUpdate {d n H : ℕ} (η β : ℝ) (θ : Idx H → HeadParam d)
    (x : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  normalizeLayer (x i + (η / (H : ℝ)) • ∑ h : Idx H, attnField β (θ h) x (x i))

/-- `X = (X^ℓ)_{ℓ≥0}` is the Markov chain of `eq:update_tokens` driven by the
head parameters `θ^ℓ_h`: a tuple of unit vectors at every layer, each obtained
from the previous one by `layerUpdate`.

Source: arXiv:2604.01978v1, `eq:update_tokens`. -/
def IsLayerChain {d n H : ℕ} (η β : ℝ) (θ : ℕ → Idx H → HeadParam d)
    (X : ℕ → Idx n → EucSpace d) : Prop :=
  (∀ ℓ : ℕ, ∀ i : Idx n, ‖X ℓ i‖ = 1) ∧
    ∀ ℓ : ℕ, ∀ i : Idx n, X (ℓ + 1) i = layerUpdate η β (θ ℓ) (X ℓ) i

/-- The piecewise-constant interpolation `X^η(t) = X^{⌊t/η⌋}` of the chain.

Source: arXiv:2604.01978v1, §2.2.3. -/
noncomputable def interpChain {d n : ℕ} (η : ℝ) (X : ℕ → Idx n → EucSpace d) (t : ℝ) :
    Idx n → EucSpace d :=
  X ⌊t / η⌋₊

/-- The mean field `b_{ρ*}[μ_X](z) = E_{θ∼ρ*} B_θ[μ_X](z)`.

Source: arXiv:2604.01978v1, §2.2.1. -/
noncomputable def meanField {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (x : Idx n → EucSpace d) (z : EucSpace d) : EucSpace d :=
  ∫ θ, attnField β θ x z ∂ρ

/-- The fluctuation `ξ_θ[μ_X](z) = B_θ[μ_X](z) - b_{ρ*}[μ_X](z)`.

Source: arXiv:2604.01978v1, §2.2.1. -/
noncomputable def fluct {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d)) (θ : HeadParam d)
    (x : Idx n → EucSpace d) (z : EucSpace d) : EucSpace d :=
  attnField β θ x z - meanField β ρ x z

/-- The projected drift `b(X)_i = Proj_{x_i} b_{ρ*}[μ_X](x_i)`.

Source: arXiv:2604.01978v1, §5. -/
noncomputable def bField {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (x : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  proj d (x i) (meanField β ρ x (x i))

/-- The projected fluctuation kernel `G(X,θ)_i = Proj_{x_i} ξ_θ[μ_X](x_i)`.

Source: arXiv:2604.01978v1, `eq:G_def`. -/
noncomputable def Gfield {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (x : Idx n → EucSpace d) (θ : HeadParam d) (i : Idx n) : EucSpace d :=
  proj d (x i) (fluct β ρ θ x (x i))

/-- `σ` is the variance proxy of `eq: defining.alpha`:

  `σ² = max_{X ∈ (𝕊^{d-1})^n} max_{j ∈ [n]} E ‖Proj_{x_j} ξ_θ[μ_X](x_j)‖²`.

Written as `IsGreatest` so that the maximum is asserted to be attained, as the
source's `max` asserts, rather than replaced by a supremum with a junk value.

Source: arXiv:2604.01978v1, `eq: defining.alpha`. -/
def IsVarianceProxy (d n : ℕ) (β : ℝ) (ρ : Measure (HeadParam d)) (s : ℝ) : Prop :=
  0 ≤ s ∧
    IsGreatest {v : ℝ | ∃ x : Idx n → EucSpace d, (∀ i : Idx n, ‖x i‖ = 1) ∧
      ∃ j : Idx n, v = ∫ θ, ‖Gfield β ρ x θ j‖ ^ 2 ∂ρ} (s ^ 2)
/-- `α = η σ²/H`, the diffusive scaling parameter.

Source: arXiv:2604.01978v1, `eq:Alpha_Sec2`. -/
noncomputable def alphaOf (η s : ℝ) (H : ℕ) : ℝ := η * s ^ 2 / (H : ℝ)

end Homogenized
end Transformer

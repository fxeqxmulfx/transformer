/-
# §2 — Interacting particle system

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes the equations from §2 of the survey.  The main objects:

* `eq: resnet`           — discrete-time ResNet,
* `eq: neural.ode`       — its continuous-time limit (neural ODE),
* `eq: transformerSd.QKV`— the full Transformer ODE on `(𝕊^{d-1})^n`,
* `eq: SA.QKV`           — partition function for the QKV model,
* `eq:P`                 — self-attention matrix,
* `SA`                   — the simplified Q=K=V=I_d model,
* `eq: SA`               — its partition function,
* `eq: multihead`        — multi-headed self-attention,
* `eq: albert`           — the full Transformer with feed-forward layers.

Proofs in this section are mostly trivial unfoldings; the substantive
analytic results (existence/uniqueness of the ODE flow, etc.) appear
in later sections and the appendices.
-/

import Transformer.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace SectionIPS

variable (d n : ℕ)

/-! ### §2.1 — Residual networks -/

/-- **Equation (eq: resnet).**  The discrete-time ResNet update with `L` hidden
layers:

  `x(k+1) = x(k) + w(k) · σ(a(k) x(k) + b(k))`,  `x(0) = x`. -/
def resnetStep
    (σ : ℝ → ℝ)
    (w : ℕ → (EucSpace d →L[ℝ] EucSpace d))
    (a : ℕ → (EucSpace d →L[ℝ] EucSpace d))
    (b : ℕ → EucSpace d)
    (x : ℕ → EucSpace d) (k : ℕ) : Prop :=
  x (k+1) = x k + (w k) (EuclideanSpace.equiv _ ℝ |>.symm
                          (fun i => σ ((EuclideanSpace.equiv _ ℝ ((a k) (x k) + b k)) i)))

/-- **Equation (eq: neural.ode).**  Continuous-time analogue (a neural ODE):

  `ẋ(t) = w(t) · σ(a(t) x(t) + b(t))`,  `x(0) = x`. -/
def neuralODE
    (σ : ℝ → ℝ)
    (w : ℝ → (EucSpace d →L[ℝ] EucSpace d))
    (a : ℝ → (EucSpace d →L[ℝ] EucSpace d))
    (b : ℝ → EucSpace d)
    (x : ℝ → EucSpace d) : Prop :=
  ∀ t : ℝ, HasDerivAt x
    ((w t) (EuclideanSpace.equiv _ ℝ |>.symm
            (fun i => σ ((EuclideanSpace.equiv _ ℝ ((a t) (x t) + b t)) i)))) t

/-! ### §2.2 — The interacting particle system (Transformer ODE) -/

/-- **Equation (eq: SA.QKV).** Partition function:

  `Z_{β,i}(t) = Σ_k exp(β ⟨Q(t) x_i(t), K(t) x_k(t)⟩)`. -/
noncomputable def partitionQKV
    (β : ℝ) (Q K : TimeParam d)
    (X : ℝ → SphereTuple d n) (t : ℝ) (i : Idx n) : ℝ :=
  ∑ k : Idx n,
    Real.exp (β * inner (𝕜 := ℝ)
      ((Q t) ((X t i : EucSpace d)))
      ((K t) ((X t k : EucSpace d))))

/-- **Equation (eq:P).** Self-attention matrix entry:

  `A_{ij}(t) = exp(β ⟨Q x_i, K x_j⟩) / Z_{β,i}(t)`. -/
noncomputable def attention
    (β : ℝ) (Q K : TimeParam d)
    (X : ℝ → SphereTuple d n) (t : ℝ) (i j : Idx n) : ℝ :=
  Real.exp (β * inner (𝕜 := ℝ)
              ((Q t) ((X t i : EucSpace d)))
              ((K t) ((X t j : EucSpace d))))
  / partitionQKV d n β Q K X t i

/-- **Equation (eq: transformerSd.QKV).** Idealised Transformer ODE on
`(𝕊^{d-1})^n`:

  `ẋ_i(t) = Proj_{x_i(t)} ( Z_{β,i}(t)^{-1} Σ_j exp(β ⟨Q x_i, K x_j⟩) V x_j )`. -/
def transformerODE
    (β : ℝ) (Q K V : TimeParam d)
    (X : ℝ → SphereTuple d n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => (X s i : EucSpace d))
      (proj d ((X t i : EucSpace d))
        ((partitionQKV d n β Q K X t i)⁻¹ •
          ∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ)
                        ((Q t) ((X t i : EucSpace d)))
                        ((K t) ((X t j : EucSpace d))))
            • (V t) ((X t j : EucSpace d)))) t

/-! ### Simplified `SA` model with `Q = K = V = I_d` -/

/-- **Equation (eq: SA).** Partition function in the simplified model:

  `Z_{β,i}(t) = Σ_k exp(β ⟨x_i(t), x_k(t)⟩)`. -/
noncomputable def partitionSA
    (β : ℝ) (X : ℝ → SphereTuple d n) (t : ℝ) (i : Idx n) : ℝ :=
  ∑ k : Idx n,
    Real.exp (β * inner (𝕜 := ℝ)
                ((X t i : EucSpace d))
                ((X t k : EucSpace d)))

/-- **Equation (SA).** Simplified Transformer dynamics (Q = K = V = I_d):

  `ẋ_i(t) = Proj_{x_i(t)} ( Z_{β,i}(t)^{-1} Σ_j exp(β ⟨x_i, x_j⟩) x_j )`. -/
def SA
    (β : ℝ) (X : ℝ → SphereTuple d n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => (X s i : EucSpace d))
      (proj d ((X t i : EucSpace d))
        ((partitionSA d n β X t i)⁻¹ •
          ∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ)
                        ((X t i : EucSpace d))
                        ((X t j : EucSpace d)))
            • ((X t j : EucSpace d)))) t

/-! ### §2.2 — Boundedness of the partition function

The text right after `eq: dissipation.softmax` notes that
`e^{-β} ≤ Z_{β,μ}(x) ≤ e^{β}` for all `x ∈ 𝕊^{d-1}`.  We record the
particle-system analogue. -/

lemma partitionSA_bounds (β : ℝ) (X : ℝ → SphereTuple d n) (t : ℝ) (i : Idx n)
    (hβ : 0 ≤ β) (hn : 0 < n) :
    Real.exp (-β) * n ≤ partitionSA d n β X t i
      ∧ partitionSA d n β X t i ≤ Real.exp β * n := by
  -- The proof rests on the bound `|⟨x_i, x_j⟩| ≤ 1` for unit-norm vectors
  -- on the sphere.  This in turn gives `-β ≤ β⟨x_i, x_j⟩ ≤ β`, hence
  -- `exp(-β) ≤ exp(β⟨x_i, x_j⟩) ≤ exp(β)`, and summing over `n` terms
  -- yields the claimed bounds.
  --
  -- The substantive analytic content (Cauchy-Schwarz on the sphere,
  -- monotonicity of `Real.exp`) is in Mathlib; the orchestration
  -- requires bookkeeping the `SSphere d` ↪ `EucSpace d` coercion.
  -- Deferred — connected separately via `GPTMini.QKNorm.score_bounded`.
  sorry

/-! ### §2.3 — Toward the complete Transformer -/

/-- **Equation (eq: multihead).** Multi-headed self-attention:

  `ẋ_i = Proj_{x_i} ( Σ_h Σ_j exp(β ⟨Q_h x_i, K_h x_j⟩)/Z_{β,i,h} · V_h x_j )`. -/
def multiHeadSA
    (H : ℕ) (β : ℝ)
    (Q K V : Idx H → TimeParam d)
    (X : ℝ → SphereTuple d n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => (X s i : EucSpace d))
      (proj d ((X t i : EucSpace d))
        (∑ h : Idx H,
          (partitionQKV d n β (Q h) (K h) X t i)⁻¹ •
            ∑ j : Idx n,
              Real.exp (β * inner (𝕜 := ℝ)
                          ((Q h t) ((X t i : EucSpace d)))
                          ((K h t) ((X t j : EucSpace d))))
              • (V h t) ((X t j : EucSpace d)))) t

/-- **Equation (eq: albert).**  Full Transformer dynamics combining multi-head
self-attention with a feed-forward layer:

  `ẋ_i = Proj_{x_i} ( Σ_h Σ_j ⋯ V_h x_j  +  w σ(a x_i + b) )`. -/
def fullTransformer
    (H : ℕ) (β : ℝ)
    (Q K V : Idx H → TimeParam d)
    (σ : ℝ → ℝ)
    (w a : ℝ → (EucSpace d →L[ℝ] EucSpace d))
    (b : ℝ → EucSpace d)
    (X : ℝ → SphereTuple d n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => (X s i : EucSpace d))
      (proj d ((X t i : EucSpace d))
        ((∑ h : Idx H,
          (partitionQKV d n β (Q h) (K h) X t i)⁻¹ •
            ∑ j : Idx n,
              Real.exp (β * inner (𝕜 := ℝ)
                          ((Q h t) ((X t i : EucSpace d)))
                          ((K h t) ((X t j : EucSpace d))))
              • (V h t) ((X t j : EucSpace d)))
          +
          (w t) (EuclideanSpace.equiv _ ℝ |>.symm
                  (fun i' => σ ((EuclideanSpace.equiv _ ℝ
                                  ((a t) ((X t i : EucSpace d)) + b t)) i'))))) t

/-! ### Permutation equivariance (Remark after `SA`)

The Transformer `SA` is permutation-equivariant on `(𝕊^{d-1})^n`.
-/

/-- The flow induced by `SA` is permutation-equivariant: if `X(·)` is a
solution and `π : [n] ≃ [n]` is a permutation, then `i ↦ X(t)(π i)` is also a
solution. -/
theorem SA_permutation_equivariant
    (β : ℝ) (X : ℝ → SphereTuple d n) (σπ : Idx n ≃ Idx n) :
    SA d n β X → SA d n β (fun t => (X t) ∘ σπ) := by
  sorry

end SectionIPS
end Transformer

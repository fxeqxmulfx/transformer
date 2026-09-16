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
namespace Perspective

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

/-- **The consensus configurations are the equilibria of `SA`.**

When every token sits at the same point `x`, the attention average is `x`
itself — the weights are equal and sum to one — and `Proj_x x = 0`, so the
constant curve solves `eq: SA`.  It is the one solution of `SA` available in
closed form, and it is what makes the hypotheses of the Appendix D estimates
satisfiable. -/
theorem SA_const_consensus (hn : 0 < n) (β : ℝ) (x : SSphere d) :
    SA d n β (fun _ _ => x) := by
  intro t i
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hxx : inner (𝕜 := ℝ) ((x : EucSpace d)) ((x : EucSpace d)) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  have hne : ((n : ℝ) * Real.exp β) ≠ 0 :=
    mul_ne_zero (Nat.cast_ne_zero.mpr hn.ne') (Real.exp_ne_zero β)
  have hZ : partitionSA d n β (fun _ _ => x) t i = (n : ℝ) * Real.exp β := by
    simp [partitionSA]
  have hsum : ∑ _j : Idx n,
      Real.exp (β * inner (𝕜 := ℝ) ((x : EucSpace d)) ((x : EucSpace d)))
        • ((x : EucSpace d)) = ((n : ℝ) * Real.exp β) • ((x : EucSpace d)) := by
    simp only [hxx, mul_one, Finset.sum_const, Finset.card_univ, Fintype.card_fin]
    rw [← Nat.cast_smul_eq_nsmul ℝ, smul_smul]
  have hval : proj d ((x : EucSpace d))
      ((partitionSA d n β (fun _ _ => x) t i)⁻¹ •
        ∑ _j : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) ((x : EucSpace d)) ((x : EucSpace d)))
            • ((x : EucSpace d))) = 0 := by
    rw [hZ, hsum, smul_smul, inv_mul_cancel₀ hne, one_smul, proj, hxx, one_smul,
      sub_self]
  exact (hasDerivAt_const t ((x : EucSpace d))).congr_deriv hval.symm

/-- The hypothesis `0 < n` of `SA_const_consensus` is satisfiable. -/
example : 0 < 1 := one_pos

/-! ### §2.2 — Boundedness of the partition function

The text right after `eq: dissipation.softmax` notes that
`e^{-β} ≤ Z_{β,μ}(x) ≤ e^{β}` for all `x ∈ 𝕊^{d-1}`.  We record the
particle-system analogue. -/

/-- **Boundedness of the partition function.**

`e^{-β} n ≤ Z_{β,i}(t) ≤ e^{β} n`: Cauchy-Schwarz on the sphere gives
`|⟨x_i, x_k⟩| ≤ 1`, so each of the `n` exponents lies in `[-β, β]`.

Source: arXiv:2312.10794v5, §2, the text after `eq: dissipation.softmax`. -/
lemma partitionSA_bounds (β : ℝ) (X : ℝ → SphereTuple d n) (t : ℝ) (i : Idx n)
    (hβ : 0 ≤ β) :
    Real.exp (-β) * n ≤ partitionSA d n β X t i
      ∧ partitionSA d n β X t i ≤ Real.exp β * n := by
  have hnorm : ∀ k : Idx n, ‖(X t k : EucSpace d)‖ = 1 := fun k =>
    mem_sphere_zero_iff_norm.mp (X t k).2
  have hbound : ∀ k : Idx n,
      |inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t k : EucSpace d))| ≤ 1 := fun k => by
    simpa [hnorm i, hnorm k] using
      abs_real_inner_le_norm ((X t i : EucSpace d)) ((X t k : EucSpace d))
  have hconst : ∀ c : ℝ, (∑ _k : Idx n, c) = c * n := by
    intro c
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    ring
  refine ⟨?_, ?_⟩
  · rw [← hconst (Real.exp (-β)), partitionSA]
    refine Finset.sum_le_sum fun k _ => Real.exp_le_exp.mpr ?_
    nlinarith [neg_le_of_abs_le (hbound k)]
  · rw [← hconst (Real.exp β), partitionSA]
    refine Finset.sum_le_sum fun k _ => Real.exp_le_exp.mpr ?_
    nlinarith [le_of_abs_le (hbound k)]

/-- The hypothesis `0 ≤ β` of `partitionSA_bounds` is satisfiable. -/
example : (0 : ℝ) ≤ 1 := zero_le_one

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
  intro h t i
  have hZ : partitionSA d n β (fun s => (X s) ∘ σπ) t i
      = partitionSA d n β X t (σπ i) :=
    Fintype.sum_equiv σπ _ _ fun _ => rfl
  have hsum : (∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ)
        ((X t (σπ i) : EucSpace d)) ((X t (σπ j) : EucSpace d)))
        • ((X t (σπ j) : EucSpace d)))
      = ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ)
        ((X t (σπ i) : EucSpace d)) ((X t j : EucSpace d)))
        • ((X t j : EucSpace d)) :=
    Fintype.sum_equiv σπ _ _ fun _ => rfl
  simpa [Function.comp_apply, hZ, hsum] using h t (σπ i)

end Perspective
end Transformer

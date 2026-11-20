module

public import Mathlib.Basic.Complex.Basic
public import Mathlib.LinearAlgebra.Matrix.ToLin

/-!
# Product layers

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV7. The FFN of the main
model on an ordinary residual, without normalizations or attention, is the layer
`z ↦ z + W_out ((W_g z) ⊙ (W_u z))` over `ℂ`: hidden unit `u` multiplies two linear readings of
the state and writes the product along column `u` of `W_out` (`IsProductLayer`; the matrix form
is `isProductLayer_iff_matrix`). Layers compose in a row (`IsProductNet`, `IsProductNet.comp`)
and side by side on disjoint parts of the stream, where the hidden units add up
(`IsProductLayer.sideBySide`, `IsProductNet.sideBySide`); the identity is a layer without units
(`isProductLayer_id`, `IsProductNet.id_net`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset
open scoped Matrix

variable {ι ι₁ ι₂ : Type*}

/-- `F` is one product layer with at most `m` hidden units: `F z = z + ∑_u (g_u z)(w_u z) o_u`,
each unit `u` multiplying two linear readings `g_u`, `w_u` of the state and writing along
`o_u`. -/
def IsProductLayer (m : ℕ) (F : (ι → ℂ) → ι → ℂ) : Prop :=
  ∃ (η : Type) (_ : Fintype η), Fintype.card η ≤ m ∧
    ∃ (gate up : η → (ι → ℂ) →ₗ[ℂ] ℂ) (out : η → ι → ℂ),
      ∀ z, F z = z + ∑ u, (gate u z * up u z) • out u

/-- **The matrix form**: a product layer is `z ↦ z + W_out ((W_g z) ⊙ (W_u z))` with
`W_g, W_u ∈ ℂ^{η × ι}` and `W_out ∈ ℂ^{ι × η}`. -/
theorem isProductLayer_iff_matrix [Fintype ι] {m : ℕ} {F : (ι → ℂ) → ι → ℂ} :
    IsProductLayer m F ↔ ∃ (η : Type) (_ : Fintype η), Fintype.card η ≤ m ∧
      ∃ (Wg Wu : Matrix η ι ℂ) (Wo : Matrix ι η ℂ),
        ∀ z, F z = z + Wo *ᵥ ((Wg *ᵥ z) * (Wu *ᵥ z)) := by
  classical
  constructor
  · rintro ⟨η, _, hm, gate, up, out, hF⟩
    refine ⟨η, inferInstance, hm, LinearMap.toMatrix' (LinearMap.pi gate),
      LinearMap.toMatrix' (LinearMap.pi up), Matrix.of fun c u => out u c, fun z => ?_⟩
    rw [hF, LinearMap.toMatrix'_mulVec, LinearMap.toMatrix'_mulVec]
    ext c
    simp [Matrix.mulVec, dotProduct, Finset.sum_apply, mul_comm]
  · rintro ⟨η, _, hm, Wg, Wu, Wo, hF⟩
    refine ⟨η, inferInstance, hm, fun u => (LinearMap.proj u).comp Wg.mulVecLin,
      fun u => (LinearMap.proj u).comp Wu.mulVecLin, fun u c => Wo c u, fun z => ?_⟩
    rw [hF]
    ext c
    simp [Matrix.mulVec, dotProduct, Finset.sum_apply, mul_comm]

/-- The identity is a product layer without hidden units. -/
lemma isProductLayer_id (m : ℕ) : IsProductLayer m (id : (ι → ℂ) → ι → ℂ) :=
  ⟨Empty, inferInstance, by simp, fun u => u.elim, fun u => u.elim, fun u => u.elim,
    fun z => by simp⟩

lemma IsProductLayer.mono {m m' : ℕ} (hm : m ≤ m') {F : (ι → ℂ) → ι → ℂ}
    (h : IsProductLayer m F) : IsProductLayer m' F := by
  obtain ⟨η, i, hc, rest⟩ := h
  exact ⟨η, i, hc.trans hm, rest⟩

example : IsProductLayer 1 (id : (Unit → ℂ) → Unit → ℂ) :=
  (isProductLayer_id 0).mono zero_le_one

/-- Two maps on disjoint parts of the stream, side by side. -/
def sideBySide (F₁ : (ι₁ → ℂ) → ι₁ → ℂ) (F₂ : (ι₂ → ℂ) → ι₂ → ℂ) :
    (ι₁ ⊕ ι₂ → ℂ) → ι₁ ⊕ ι₂ → ℂ :=
  fun z => Sum.elim (F₁ (z ∘ Sum.inl)) (F₂ (z ∘ Sum.inr))

/-- **Two product layers side by side are one**, with the hidden units of both. -/
theorem IsProductLayer.sideBySide {m₁ m₂ : ℕ} {F₁ : (ι₁ → ℂ) → ι₁ → ℂ}
    {F₂ : (ι₂ → ℂ) → ι₂ → ℂ} (h₁ : IsProductLayer m₁ F₁) (h₂ : IsProductLayer m₂ F₂) :
    IsProductLayer (m₁ + m₂) (sideBySide F₁ F₂) := by
  obtain ⟨η₁, _, hc₁, g₁, w₁, o₁, hF₁⟩ := h₁
  obtain ⟨η₂, _, hc₂, g₂, w₂, o₂, hF₂⟩ := h₂
  let r₁ : (ι₁ ⊕ ι₂ → ℂ) →ₗ[ℂ] ι₁ → ℂ := LinearMap.funLeft ℂ ℂ Sum.inl
  let r₂ : (ι₁ ⊕ ι₂ → ℂ) →ₗ[ℂ] ι₂ → ℂ := LinearMap.funLeft ℂ ℂ Sum.inr
  refine ⟨η₁ ⊕ η₂, inferInstance, by rw [Fintype.card_sum]; omega,
    Sum.elim (fun u => (g₁ u).comp r₁) (fun u => (g₂ u).comp r₂),
    Sum.elim (fun u => (w₁ u).comp r₁) (fun u => (w₂ u).comp r₂),
    Sum.elim (fun u => Sum.elim (o₁ u) 0) (fun u => Sum.elim 0 (o₂ u)), fun z => ?_⟩
  have e₁ (z : ι₁ ⊕ ι₂ → ℂ) : r₁ z = z ∘ Sum.inl := rfl
  have e₂ (z : ι₁ ⊕ ι₂ → ℂ) : r₂ z = z ∘ Sum.inr := rfl
  funext c
  rcases c with c | c
  · simp [FlatSpectrum.sideBySide, hF₁, Fintype.sum_sum_type, Finset.sum_apply, e₁]
  · simp [FlatSpectrum.sideBySide, hF₂, Fintype.sum_sum_type, Finset.sum_apply, e₂]

/-- `F` is `d` product layers in a row, each with at most `m` hidden units. -/
inductive IsProductNet (m : ℕ) : ℕ → ((ι → ℂ) → ι → ℂ) → Prop
  | zero : IsProductNet m 0 id
  | succ {d : ℕ} {F G : (ι → ℂ) → ι → ℂ} :
      IsProductNet m d F → IsProductLayer m G → IsProductNet m (d + 1) (G ∘ F)

namespace IsProductNet

lemma mono {m m' : ℕ} (hm : m ≤ m') {d : ℕ} {F : (ι → ℂ) → ι → ℂ} (h : IsProductNet m d F) :
    IsProductNet m' d F := by
  induction h with
  | zero => exact zero
  | succ _ hG ih => exact succ ih (hG.mono hm)

/-- The identity is a net of every depth, without hidden units. -/
lemma id_net (m d : ℕ) : IsProductNet m d (id : (ι → ℂ) → ι → ℂ) := by
  induction d with
  | zero => exact zero
  | succ d ih => exact succ ih (isProductLayer_id m)

/-- One layer is a net of depth one. -/
lemma of_layer {m : ℕ} {G : (ι → ℂ) → ι → ℂ} (h : IsProductLayer m G) : IsProductNet m 1 G :=
  succ zero h

/-- A net of depth one is one layer. -/
lemma isProductLayer {m : ℕ} {G : (ι → ℂ) → ι → ℂ} (h : IsProductNet m 1 G) :
    IsProductLayer m G := by
  cases h with
  | succ h₀ hG => cases h₀; exact hG

/-- Nets in a row: first `F`, then `G`. -/
lemma comp {m d₁ d₂ : ℕ} {F G : (ι → ℂ) → ι → ℂ} (hF : IsProductNet m d₁ F)
    (hG : IsProductNet m d₂ G) : IsProductNet m (d₁ + d₂) (G ∘ F) := by
  induction hG with
  | zero => exact hF
  | @succ d G' L _ hL ih => exact succ (F := G' ∘ F) (G := L) ih hL

/-- **Two nets of one depth side by side are one**, with the hidden units of both. -/
lemma sideBySide {m₁ m₂ d : ℕ} {F₁ : (ι₁ → ℂ) → ι₁ → ℂ} {F₂ : (ι₂ → ℂ) → ι₂ → ℂ}
    (h₁ : IsProductNet m₁ d F₁) (h₂ : IsProductNet m₂ d F₂) :
    IsProductNet (m₁ + m₂) d (FlatSpectrum.sideBySide F₁ F₂) := by
  induction h₁ generalizing F₂ with
  | zero =>
    cases h₂
    have : FlatSpectrum.sideBySide (id : (ι₁ → ℂ) → ι₁ → ℂ) (id : (ι₂ → ℂ) → ι₂ → ℂ) = id := by
      funext z c
      rcases c with c | c <;> rfl
    rw [this]
    exact zero
  | succ _ hG₁ ih =>
    cases h₂ with
    | succ hF₂ hG₂ => exact succ (ih hF₂) (hG₁.sideBySide hG₂)

end IsProductNet

example : IsProductLayer 0 (sideBySide (id : (Unit → ℂ) → Unit → ℂ) (id : (Bool → ℂ) → Bool → ℂ)) :=
  (isProductLayer_id 0).sideBySide (isProductLayer_id 0)

example : IsProductLayer 0 (id : (Unit → ℂ) → Unit → ℂ) :=
  (IsProductNet.of_layer (isProductLayer_id 0)).isProductLayer

example : IsProductNet 1 2 (id ∘ id : (Unit → ℂ) → Unit → ℂ) :=
  ((IsProductNet.of_layer (isProductLayer_id 0)).comp
    (IsProductNet.of_layer (isProductLayer_id 0))).mono zero_le_one

example : IsProductNet 0 3 (FlatSpectrum.sideBySide (id : (Unit → ℂ) → Unit → ℂ)
    (id : (Bool → ℂ) → Bool → ℂ)) :=
  (IsProductNet.id_net 0 3).sideBySide (IsProductNet.id_net 0 3)

end Transformer.Grokking.FlatSpectrum

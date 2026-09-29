/-
# §10 — Approximation, control, training

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

Section 10 of the survey is a short review: it states no theorem.  Of the approximation results it
cites, two had been written here as sorried theorems of the survey, and neither is one:

* for discrete-time Transformers (Yun et al.), `universal_approximation_discrete` claimed uniform
  approximation of *every* continuous sequence-to-sequence map on a compact set by residual
  attention-plus-feed-forward layers.  Such layers are permutation-equivariant, so the claim is false
  as soon as there are two tokens: `not_universal_approximation_discrete`.  The cited theorem uses
  positional encodings, which this layer does not have.
* for measure-to-measure flow maps, `universal_approximation_measure`: refuted in
  `Transformer.Perspective.Section9_ApproximationMeasure`.

The false statements are deleted and the counterexamples kept: nothing in the survey is left
unproved here.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section2_FlowMap

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-! ### Discrete-time Transformers -/

/-- Partition function of one discrete self-attention block:

  `Z_{β,i}(x) = Σ_j exp(β ⟨Q x_i, K x_j⟩)`. -/
noncomputable def discretePartition
    (β : ℝ) (Q K : ParamMatrix d) (x : Idx n → EucSpace d) (i : Idx n) : ℝ :=
  ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Q (x i)) (K (x j)))

/-- One discrete Transformer block in the residual form of `eq: resnet`:
self-attention followed by a one-hidden-layer feed-forward map,

  `x_i ↦ x_i + Z_{β,i}(x)⁻¹ Σ_j exp(β ⟨Q x_i, K x_j⟩) V x_j + w σ(a x_i + b)`.

This is the discrete-time counterpart of `fullTransformer` at one head. -/
noncomputable def discreteLayer
    (β : ℝ) (Q K V : ParamMatrix d) (σ : ℝ → ℝ)
    (w a : ParamMatrix d) (b : EucSpace d)
    (x : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  x i
    + (discretePartition d n β Q K x i)⁻¹ •
        ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Q (x i)) (K (x j))) • V (x j)
    + w (EuclideanSpace.equiv _ ℝ |>.symm
          (fun k => σ ((EuclideanSpace.equiv _ ℝ (a (x i) + b)) k)))

/-- **A block commutes with permutations of the tokens.**  Renaming the tokens by `τ` renames the
outputs by `τ`: the attention weights of token `i` are a sum over all tokens, which does not see
their order, and the feed-forward part acts on each token alone.  A stack of such blocks therefore
carries no information on the position of a token. -/
theorem discreteLayer_comp_perm (β : ℝ) (Q K V : ParamMatrix d) (σ : ℝ → ℝ)
    (w a : ParamMatrix d) (b : EucSpace d) (x : Idx n → EucSpace d) (τ : Equiv.Perm (Idx n))
    (i : Idx n) :
    discreteLayer d n β Q K V σ w a b (x ∘ τ) i = discreteLayer d n β Q K V σ w a b x (τ i) := by
  unfold discreteLayer discretePartition
  simp only [Function.comp_apply]
  rw [Equiv.sum_comp τ (fun j => Real.exp (β * inner (𝕜 := ℝ) (Q (x (τ i))) (K (x j)))),
    Equiv.sum_comp τ (fun j => Real.exp (β * inner (𝕜 := ℝ) (Q (x (τ i))) (K (x j))) • V (x j))]

/-- **Discrete-time universal approximation, as formerly stated, is false.**

The claim was: for every continuous sequence-to-sequence map `f`, every compact `S` and every
`ε > 0`, some stack of residual attention-plus-feed-forward blocks (`discreteLayer`, any continuous
activation, per-layer parameters, any depth) moves each `x₀ ∈ S` to within `ε` of `f x₀`.  It fails
with two tokens on the line: blocks are permutation-equivariant (`discreteLayer_comp_perm`), so the
stack sends the swapped sequence `y ∘ τ` to the swap of the image of `y`, while the constant map
`f ≡ (0, 1)` asks for the same output at `y = (0, 1)` and at `(1, 0)`.  Both cannot be within
`1/2` of it.

The survey states no such theorem: §10 only says that universal approximation "has been shown to
hold" for discrete-time Transformers "making use of a variant of the architecture with translation
parameters".  The translation parameters are positional encodings, which break the symmetry; this
layer has none.

Source: arXiv:2312.10794v5, §10, the paragraph citing Yun et al. -/
theorem not_universal_approximation_discrete :
    ¬ ∀ (d n : ℕ) (β : ℝ) (f : (Idx n → EucSpace d) → (Idx n → EucSpace d)), Continuous f →
      ∀ (S : Set (Idx n → EucSpace d)), IsCompact S → ∀ ε : ℝ, 0 < ε →
      ∃ (L : ℕ) (Q K V w a : ℕ → ParamMatrix d) (b : ℕ → EucSpace d) (σ : ℝ → ℝ),
      Continuous σ ∧
      ∀ x₀ ∈ S, ∀ x : ℕ → Idx n → EucSpace d,
        x 0 = x₀ →
        (∀ k : ℕ,
          x (k + 1) = discreteLayer d n β (Q k) (K k) (V k) σ (w k) (a k) (b k) (x k)) →
        ∀ i : Idx n, ‖x L i - f x₀ i‖ < ε := by
  intro h
  set u : EucSpace 1 := EuclideanSpace.single 0 1 with hu
  have hu1 : ‖u‖ = 1 := by simp [hu]
  set y : Idx 2 → EucSpace 1 := ![0, u] with hy
  set τ : Equiv.Perm (Idx 2) := Equiv.swap 0 1 with hτ
  have hS : IsCompact ({y, y ∘ τ} : Set (Idx 2 → EucSpace 1)) :=
    ((Set.finite_singleton _).insert _).isCompact
  obtain ⟨L, Q, K, V, w, a, b, σ, -, hall⟩ :=
    h 1 2 0 (fun _ => y) continuous_const {y, y ∘ τ} hS (1 / 2) (by norm_num)
  let orb : (Idx 2 → EucSpace 1) → ℕ → Idx 2 → EucSpace 1 := fun x₀ k =>
    Nat.rec (motive := fun _ => Idx 2 → EucSpace 1) x₀
      (fun k xk => discreteLayer 1 2 0 (Q k) (K k) (V k) σ (w k) (a k) (b k) xk) k
  have hperm : ∀ k, orb (y ∘ τ) k = orb y k ∘ τ := by
    intro k
    induction k with
    | zero => rfl
    | succ k ih =>
      show discreteLayer 1 2 0 (Q k) (K k) (V k) σ (w k) (a k) (b k) (orb (y ∘ τ) k) = _
      rw [ih]
      funext i
      exact discreteLayer_comp_perm 1 2 0 _ _ _ σ _ _ _ _ τ i
  have h1 := hall y (Or.inl rfl) (orb y) rfl (fun k => rfl)
  have h2 := hall (y ∘ τ) (Or.inr rfl) (orb (y ∘ τ)) rfl (fun k => rfl)
  have h1' := h1 1
  have h2' := h2 0
  rw [hperm L] at h2'
  simp only [Function.comp_apply, hτ, hy, Equiv.swap_apply_left, Matrix.cons_val_zero,
    Matrix.cons_val_one, sub_zero] at h1' h2'
  have hsum : ‖u‖ < 1 := by
    calc ‖u‖ = ‖(u - orb y L 1) + (orb y L 1 - 0)‖ := by congr 1; abel
      _ ≤ ‖u - orb y L 1‖ + ‖orb y L 1 - 0‖ := norm_add_le _ _
      _ = ‖orb y L 1 - u‖ + ‖orb y L 1 - 0‖ := by rw [norm_sub_rev]
      _ < 1 / 2 + 1 / 2 := add_lt_add h1' (by simpa using h2')
      _ = 1 := by norm_num
  exact absurd hu1 (ne_of_lt hsum)

end Perspective
end Transformer

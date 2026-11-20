module

public import Transformer.Grokking.FlatSpectrum.ProductCopies
public import Transformer.Grokking.FlatSpectrum.ProductCorners

/-!
# Depth `1 + ⌈log₂ k⌉` is needed, at any width

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV7. Every linear readout of
`d` product layers on the one-hot features has order at most `2^d` (`readout_mem_orderSpan`), so
its corner sum vanishes when `2^d < 2k`, while the corner sum of a product over the pairs is the
product of their mixed differences (`cornerSum_prod`). So a product net whose linear readout is
such a product, with no mixed difference zero, has `2k ≤ 2^d` at any width
(`two_mul_card_le_of_productNet`): `ψ(∑_i q(x_i, y_i))` needs depth `1 + ⌈log₂ k⌉`
(`le_depth_of_productNet_copies`), the depth of `exists_productNet_copies`, and `q = P(x) + x y`
qualifies for `p ≥ 2` (`mixed_stdAddChar_ne_zero`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {κ α β : Type} [Fintype κ] [DecidableEq κ]

variable [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β]

/-- **Depth is needed, at any width**: a product net of `d` layers on the one-hot features of the
pairs, whose linear readout is `∏_i φ_i(x_i, y_i)` with a nonzero mixed difference for every
pair, has `2k ≤ 2^d`. -/
theorem two_mul_card_le_of_productNet {T : Type} [Fintype T] {m d : ℕ}
    {F : (((κ × α) ⊕ (κ × β)) ⊕ T → ℂ) → ((κ × α) ⊕ (κ × β)) ⊕ T → ℂ}
    (hF : IsProductNet m d F) (ℓ : (((κ × α) ⊕ (κ × β)) ⊕ T → ℂ) →ₗ[ℂ] ℂ)
    (φ : κ → α → β → ℂ) (hφ : ∀ i, ∃ a₀ a₁ b₀ b₁, φ i a₀ b₀ - φ i a₁ b₀ - φ i a₀ b₁ + φ i a₁ b₁ ≠ 0)
    (hval : ∀ x y, ℓ (F (Sum.elim (pairFeatures x y) 0)) = ∏ i, φ i (x i) (y i)) :
    2 * Fintype.card κ ≤ 2 ^ d := by
  by_contra h
  choose a₀ a₁ b₀ b₁ hne using hφ
  have hmem := readout_mem_orderSpan hF ℓ
  rw [show (fun x y => ℓ (F (Sum.elim (pairFeatures x y) 0))) = fun x y => ∏ i, φ i (x i) (y i)
    from funext fun x => funext fun y => hval x y] at hmem
  have h0 := cornerSum_eq_zero a₀ a₁ b₀ b₁ (not_le.1 h) hmem
  rw [cornerSum_prod] at h0
  exact prod_ne_zero_iff.2 (fun i _ => hne i) h0

/-- The mixed difference of `ψ(P(x) + x y)` at the corners `0`, `1` is `ψ(P 1)(ψ(1) − 1)`, not
zero for `p ≥ 2`. -/
lemma mixed_stdAddChar_ne_zero {p : ℕ} [NeZero p] (hp : 1 < p) (P : ZMod p → ZMod p) :
    ZMod.stdAddChar (P 0 + 0 * 0) - ZMod.stdAddChar (P 1 + 1 * 0) -
      ZMod.stdAddChar (P 0 + 0 * 1) + ZMod.stdAddChar (P 1 + 1 * 1) ≠ 0 := by
  have : Fact (1 < p) := ⟨hp⟩
  have h1 : ZMod.stdAddChar (1 : ZMod p) ≠ 1 := fun h =>
    one_ne_zero (ZMod.injective_stdAddChar (h.trans (AddChar.map_zero_eq_one _).symm))
  have hn : ‖ZMod.stdAddChar (P 1)‖ = 1 := by simp
  have hP : ZMod.stdAddChar (P 1) ≠ 0 := fun h => by rw [h, norm_zero] at hn; exact zero_ne_one hn
  have e : ZMod.stdAddChar (P 0 + 0 * 0) - ZMod.stdAddChar (P 1 + 1 * 0) -
      ZMod.stdAddChar (P 0 + 0 * 1) + ZMod.stdAddChar (P 1 + 1 * 1) =
        ZMod.stdAddChar (P 1) * (ZMod.stdAddChar (1 : ZMod p) - 1) := by
    simp only [mul_zero, add_zero, mul_one, AddChar.map_add_eq_mul]
    ring
  rw [e]
  exact mul_ne_zero hP (sub_ne_zero.2 h1)

example := mixed_stdAddChar_ne_zero (p := 7) (by norm_num) fun a => a ^ 3

/-- **`k` copies need depth `1 + ⌈log₂ k⌉`**, at any width: if a linear readout of `d` product
layers on the features of `k` pairs is `ψ(∑_i q(x_i, y_i))`, and `ψ ∘ q` has a nonzero mixed
difference, then `1 + ⌈log₂ k⌉ ≤ d`. -/
theorem le_depth_of_productNet_copies {p k : ℕ} [NeZero p] (hk : 0 < k) (q : α → β → ZMod p)
    (hq : ∃ a₀ a₁ b₀ b₁, ZMod.stdAddChar (q a₀ b₀) - ZMod.stdAddChar (q a₁ b₀) -
      ZMod.stdAddChar (q a₀ b₁) + ZMod.stdAddChar (q a₁ b₁) ≠ 0)
    {T : Type} [Fintype T] {m d : ℕ}
    {F : (((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ) → ((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ}
    (hF : IsProductNet m d F) (ℓ : (((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ) →ₗ[ℂ] ℂ)
    (hval : ∀ x y, ℓ (F (Sum.elim (pairFeatures x y) 0)) =
      ZMod.stdAddChar (∑ i, q (x i) (y i))) :
    1 + Nat.clog 2 k ≤ d := by
  have h := two_mul_card_le_of_productNet hF ℓ (fun _ a b => ZMod.stdAddChar (q a b))
    (fun _ => hq) fun x y => by rw [hval, stdAddChar_sum]
  rw [Fintype.card_fin] at h
  obtain _ | d := d
  · rw [pow_zero] at h
    omega
  · have hk' : k ≤ 2 ^ d := by rw [pow_succ] at h; omega
    have := Nat.clog_le_of_le_pow hk'
    omega

/-- The network of `exists_productNet_copies` for two copies has depth `1 + ⌈log₂ 2⌉ = 2`, the
least possible. -/
example : 1 + Nat.clog 2 2 ≤ 1 + Nat.clog 2 2 := by
  obtain ⟨T, _, root, F, -, hF, hval⟩ := exists_productNet_copies (p := 7) (α := ZMod 7)
    (β := ZMod 7) (k := 2) (by norm_num) fun a b => a ^ 3 + a * b
  exact le_depth_of_productNet_copies (by norm_num) (fun a b => a ^ 3 + a * b)
    ⟨0, 1, 0, 1, mixed_stdAddChar_ne_zero (by norm_num) fun a => a ^ 3⟩ hF
    (LinearMap.proj (Sum.inr root)) hval

example : 2 * Fintype.card (Fin 2) ≤ 2 ^ (1 + Nat.clog 2 2) := by
  obtain ⟨T, _, root, F, -, hF, hval⟩ := exists_productNet_copies (p := 7) (α := ZMod 7)
    (β := ZMod 7) (k := 2) (by norm_num) fun a b => a ^ 3 + a * b
  exact two_mul_card_le_of_productNet hF (LinearMap.proj (Sum.inr root))
    (fun _ a b => ZMod.stdAddChar (a ^ 3 + a * b))
    (fun _ => ⟨0, 1, 0, 1, mixed_stdAddChar_ne_zero (by norm_num) fun a => a ^ 3⟩)
    fun x y => by rw [LinearMap.proj_apply, hval, stdAddChar_sum]

end Transformer.Grokking.FlatSpectrum

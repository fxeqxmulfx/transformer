module

public import Transformer.Grokking.FlatSpectrum.BilinearKernel
public import Transformer.Grokking.FlatSpectrum.ProductCharacters
public import Transformer.Grokking.FlatSpectrum.ProductRank
public import Transformer.Grokking.FlatSpectrum.ProductTree

/-!
# `k` copies by a product network of depth `1 + ⌈log₂ k⌉`

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV7. Pad the `k` pairs to
`2^{⌈log₂ k⌉} < 2k` leaves with the constant `1` (`two_pow_clog_lt`): if each pair's value is
`∑_u G_u(x) H_u(y)`, then `1 + ⌈log₂ k⌉` product layers with at most `2k|U|` hidden units each,
on the one-hot features of the pairs and fewer than `4k` scratch coordinates, write the product
of the values (`exists_productNet_pairs`). With `U = β` and `H_b(y) = [y = b]`, the value is any
function of a pair, so `ψ(∑_i q(x_i, y_i))` for any `q` takes `2k|β|` units
(`exists_productNet_copies`), and the readout `Re(ψ(f) conj ψ(c))` is largest at `c = f`
(`exists_productNet_copies_readout`). Through the characters, `ψ(P(x) + x y)` is
`ψ(P x) · 1 + ∑_χ c(χ) ψ(P x) χ(x) · χ(y)`, `p` units per pair (`exists_productNet_mulChar`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

/-- `2^{⌈log₂ k⌉} < 2k` for `k ≥ 1`. -/
lemma two_pow_clog_lt {k : ℕ} (hk : 0 < k) : 2 ^ Nat.clog 2 k < 2 * k := by
  rcases Nat.lt_or_ge 1 k with h | h
  · have h1 := Nat.pow_pred_clog_lt_self 2 h
    obtain ⟨L, hL⟩ : ∃ L, Nat.clog 2 k = L + 1 :=
      ⟨_, (Nat.succ_pred_eq_of_pos (Nat.clog_pos (by norm_num) h)).symm⟩
    rw [hL] at h1 ⊢
    rw [pow_succ]
    simp only [Nat.pred_eq_sub_one, Nat.add_sub_cancel] at h1
    omega
  · obtain rfl : k = 1 := by omega
    simp

example := two_pow_clog_lt (k := 5) (by norm_num)

variable {α β : Type} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β]

/-- **`k` pairs**: if each pair's value is `∑_u G_u(x) H_u(y)`, then `1 + ⌈log₂ k⌉` product
layers with at most `2k|U|` hidden units each, on the features of the pairs and fewer than `4k`
scratch coordinates that start at zero, write `∏_i ∑_u G_u(x_i) H_u(y_i)` into one coordinate. -/
theorem exists_productNet_pairs {k : ℕ} (hk : 0 < k) (U : Type) [Fintype U] [Nonempty U]
    (G : U → α → ℂ) (H : U → β → ℂ) :
    ∃ (T : Type) (_ : Fintype T) (root : T)
      (F : (((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ) → ((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ),
      Fintype.card T < 4 * k ∧ IsProductNet (2 * k * Fintype.card U) (1 + Nat.clog 2 k) F ∧
      ∀ x y, F (Sum.elim (pairFeatures x y) 0) (Sum.inr root) =
        ∏ i, ∑ u, G u (x i) * H u (y i) := by
  set L := Nat.clog 2 k
  have hkL : k ≤ 2 ^ L := Nat.le_pow_clog (by norm_num) k
  have hLk : 2 ^ L < 2 * k := two_pow_clog_lt hk
  have hU : (Fintype.card U : ℂ) ≠ 0 := Nat.cast_ne_zero.2 Fintype.card_ne_zero
  let π : Fin (2 ^ L) → Fin k := fun j => if h : (j : ℕ) < k then ⟨j, h⟩ else ⟨0, hk⟩
  let G' : Fin (2 ^ L) → U → α → ℂ := fun j u a =>
    if (j : ℕ) < k then G u a else (Fintype.card U : ℂ)⁻¹
  let H' : Fin (2 ^ L) → U → β → ℂ := fun j u b => if (j : ℕ) < k then H u b else 1
  obtain ⟨T, _, root, F, hT, hF, hval⟩ := exists_productNet L U π G' H'
  refine ⟨T, inferInstance, root, F, ?_, hF.mono (Nat.mul_le_mul_right _ hLk.le),
    fun x y => ?_⟩
  · rw [pow_succ] at hT
    omega
  let φ : ℕ → ℂ := fun j => if h : j < k then ∑ u, G u (x ⟨j, h⟩) * H u (y ⟨j, h⟩) else 1
  have hφ (j : Fin (2 ^ L)) : ∑ u, G' j u (x (π j)) * H' j u (y (π j)) = φ j := by
    by_cases h : (j : ℕ) < k
    · simp [G', H', π, φ, h]
    · simp [G', H', π, φ, h, hU]
  have hIco : ∏ j ∈ Ico k (2 ^ L), φ j = 1 :=
    prod_eq_one fun j hj => by simp [φ, not_lt.2 (mem_Ico.1 hj).1]
  rw [hval, Fintype.prod_congr _ _ hφ, Fin.prod_univ_eq_prod_range φ,
    ← prod_range_mul_prod_Ico φ hkL, hIco, mul_one, ← Fin.prod_univ_eq_prod_range φ]
  exact Fintype.prod_congr _ _ fun i => by simp [φ, i.2]

example := exists_productNet_pairs (α := Bool) (β := Bool) (k := 2) (by norm_num) Unit
  (fun _ _ => 1) (fun _ _ => 1)

/-- **`k` copies of any `q`**: `1 + ⌈log₂ k⌉` product layers with at most `2k|β|` hidden units
each, on the features of the pairs and fewer than `4k` scratch coordinates that start at zero,
write `ψ(∑_i q(x_i, y_i))` into one coordinate. -/
theorem exists_productNet_copies {p k : ℕ} [NeZero p] [Nonempty β] (hk : 0 < k)
    (q : α → β → ZMod p) :
    ∃ (T : Type) (_ : Fintype T) (root : T)
      (F : (((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ) → ((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ),
      Fintype.card T < 4 * k ∧ IsProductNet (2 * k * Fintype.card β) (1 + Nat.clog 2 k) F ∧
      ∀ x y, F (Sum.elim (pairFeatures x y) 0) (Sum.inr root) =
        ZMod.stdAddChar (∑ i, q (x i) (y i)) := by
  obtain ⟨T, _, root, F, hT, hF, hval⟩ := exists_productNet_pairs hk β
    (fun b a => ZMod.stdAddChar (q a b)) (fun b y => if y = b then 1 else 0)
  refine ⟨T, inferInstance, root, F, hT, hF, fun x y => ?_⟩
  rw [hval, stdAddChar_sum]
  exact Fintype.prod_congr _ _ fun i => by simp

/-- Three copies of `x³ + x y` at `p = 7`. -/
example := exists_productNet_copies (p := 7) (α := ZMod 7) (β := ZMod 7) (k := 3) (by norm_num)
  fun a b => a ^ 3 + a * b

/-- **The readout of `k` copies**: the coordinate of `exists_productNet_copies` has
`Re(out · conj ψ(c))` largest exactly at `c = f`. -/
theorem exists_productNet_copies_readout {p k : ℕ} [NeZero p] [Nonempty β] (hk : 0 < k)
    (q : α → β → ZMod p) :
    ∃ (T : Type) (_ : Fintype T) (root : T)
      (F : (((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ) → ((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ),
      Fintype.card T < 4 * k ∧ IsProductNet (2 * k * Fintype.card β) (1 + Nat.clog 2 k) F ∧
      ∀ x y c, c ≠ ∑ i, q (x i) (y i) →
        (F (Sum.elim (pairFeatures x y) 0) (Sum.inr root) * star (ZMod.stdAddChar c)).re <
          (F (Sum.elim (pairFeatures x y) 0) (Sum.inr root) *
            star (ZMod.stdAddChar (∑ i, q (x i) (y i)))).re := by
  obtain ⟨T, _, root, F, hT, hF, hval⟩ := exists_productNet_copies hk q
  refine ⟨T, inferInstance, root, F, hT, hF, fun x y c hc => ?_⟩
  rw [hval, re_stdAddChar_mul_star_self]
  exact re_stdAddChar_mul_star_lt hc

example := exists_productNet_copies_readout (p := 7) (α := ZMod 7) (β := ZMod 7) (k := 3)
  (by norm_num) fun a b => a ^ 3 + a * b

/-- **The route through characters**: for `q = P(x) + x y` over a prime, the units of a pair are
`ψ(P x) · 1` and `c(χ) ψ(P x) χ(x) · χ(y)` for the `p − 1` characters, `p` in all, so the network
has at most `2kp` hidden units per layer. -/
theorem exists_productNet_mulChar {p k : ℕ} [Fact p.Prime] (hk : 0 < k) (P : ZMod p → ZMod p) :
    ∃ (T : Type) (_ : Fintype T) (root : T)
      (F : (((Fin k × ZMod p) ⊕ (Fin k × ZMod p)) ⊕ T → ℂ) →
        ((Fin k × ZMod p) ⊕ (Fin k × ZMod p)) ⊕ T → ℂ),
      Fintype.card T < 4 * k ∧ IsProductNet (2 * k * p) (1 + Nat.clog 2 k) F ∧
      ∀ x y, F (Sum.elim (pairFeatures x y) 0) (Sum.inr root) =
        ZMod.stdAddChar (∑ i, (P (x i) + x i * y i)) := by
  obtain ⟨T, _, root, F, hT, hF, hval⟩ := exists_productNet_pairs hk
    (Option (MulChar (ZMod p) ℂ))
    (fun u a => u.elim (ZMod.stdAddChar (P a)) fun χ => mulCoeff 1 χ * ZMod.stdAddChar (P a) * χ a)
    (fun u b => u.elim 1 fun χ => χ b)
  have hcard : Fintype.card (Option (MulChar (ZMod p) ℂ)) = p := by
    rw [Fintype.card_option, card_mulChar, Nat.sub_add_cancel (Fact.out : p.Prime).one_le]
  refine ⟨T, inferInstance, root, F, hT, hF.mono (Nat.mul_le_mul_left _ hcard.le), fun x y => ?_⟩
  rw [hval, stdAddChar_sum]
  refine Fintype.prod_congr _ _ fun i => ?_
  have h := stdAddChar_mul_eq 1 (x i) (y i)
  rw [one_mul] at h
  rw [Fintype.sum_option, AddChar.map_add_eq_mul, h]
  simp only [Option.elim, mul_one, mul_add, mul_sum]
  exact congrArg _ (sum_congr rfl fun χ _ => by ring)

example := @exists_productNet_mulChar 7 3 ⟨Nat.prime_seven⟩ (by norm_num) fun a => a ^ 3

end Transformer.Grokking.FlatSpectrum

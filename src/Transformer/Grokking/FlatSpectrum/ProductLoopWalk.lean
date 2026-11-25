module

public import Transformer.Grokking.FlatSpectrum.ProductLoop

/-!
# Following `2K` links: the passes of a loop count as depth

Source: `review/flat_spectrum.md`, *Loops*, item LP3. An `N`-hop question gives `N` lookup tables,
and its answer is the key that the chain of lookups reaches. Here the `N = 2K` tables are the
operands, maps `γ → γ` on one set of keys, followed in the order `x 0, y 0, x 1, y 1, …` from a
fixed key (`pairWalk`); a varying start is the first of them applied to the fixed key.

On the corners of the box between the identity and the swap of two keys `a`, `b`, every table is
a power of the swap, so the walk is the swap to the number of swaps (`pairWalk_corner`). A readout
`g` of the reached key with `g a ≠ g b` then has the corner sum `2^{2K − 1} (g a − g b)`
(`two_mul_cornerSum_pairWalk`), so its order is the full `2K`, and any function with a nonzero
corner sum of order at most `2^D` has `1 + ⌈log₂ K⌉ ≤ D` (`one_add_clog_le_of_cornerSum_ne_zero`).

LP1's count of the order then bounds every loop on features that read one table each, as the
one-hot features of the links do (`linkFeatures_mem_orderSpan`): a prelude of depth `d₀`, `n + 1`
passes of a core of depth `d₁` and a coda of depth `d₂` answer `2K` hops only if
`1 + ⌈log₂ K⌉ ≤ d₀ + (n + 1) d₁ + d₂`, at any width (`le_passes_of_loop_walk`). With
`d₀ = d₂ = 0` and `d₁ = 1` this is `⌈log₂ N⌉` passes.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {γ : Type}

/-- The key reached from `c` through the `2K` maps `x 0, y 0, x 1, y 1, …, x (K − 1), y (K − 1)`. -/
def pairWalk : (K : ℕ) → (Fin K → γ → γ) → (Fin K → γ → γ) → γ → γ
  | 0, _, _, c => c
  | K + 1, x, y, c => pairWalk K (Fin.tail x) (Fin.tail y) (y 0 (x 0 c))

/-- If every map of the walk is a power of `τ`, the walk is `τ` to the sum of the powers. -/
lemma pairWalk_iterate (τ : γ → γ) : ∀ (K : ℕ) (x y : Fin K → γ → γ) (ex ey : Fin K → ℕ),
    (∀ i, x i = τ^[ex i]) → (∀ i, y i = τ^[ey i]) → ∀ c,
      pairWalk K x y c = τ^[∑ i, (ex i + ey i)] c
  | 0, _, _, _, _, _, _, c => by simp [pairWalk]
  | K + 1, x, y, ex, ey, hx, hy, c => by
    rw [pairWalk, pairWalk_iterate τ K (Fin.tail x) (Fin.tail y) (Fin.tail ex) (Fin.tail ey)
      (fun i => hx i.succ) (fun i => hy i.succ), hx 0, hy 0, ← Function.iterate_add_apply,
      ← Function.iterate_add_apply, Fin.sum_univ_succ]
    congr 1
    simp only [Fin.tail]
    omega

/-- On a corner of the box between the identity and `τ`, the table `x i` is `τ` or the identity. -/
lemma cornerX_id (τ : γ → γ) {K : ℕ} (ε : Fin K ⊕ Fin K → Bool) (i : Fin K) :
    cornerX (fun _ => id) (fun _ => τ) ε i = τ^[if ε (Sum.inl i) then 1 else 0] := by
  unfold cornerX
  split_ifs <;> rfl

/-- On a corner of the box between the identity and `τ`, the table `y i` is `τ` or the identity. -/
lemma cornerY_id (τ : γ → γ) {K : ℕ} (ε : Fin K ⊕ Fin K → Bool) (i : Fin K) :
    cornerY (fun _ => id) (fun _ => τ) ε i = τ^[if ε (Sum.inr i) then 1 else 0] := by
  unfold cornerY
  split_ifs <;> rfl

/-- **On a corner the walk is `τ` to the number of tables at `τ`.** -/
lemma pairWalk_corner (τ : γ → γ) {K : ℕ} (ε : Fin K ⊕ Fin K → Bool) (c : γ) :
    pairWalk K (cornerX (fun _ => id) (fun _ => τ) ε) (cornerY (fun _ => id) (fun _ => τ) ε) c =
      τ^[∑ o, if ε o then 1 else 0] c := by
  rw [pairWalk_iterate τ K _ _ _ _ (cornerX_id τ ε) (cornerY_id τ ε), Fintype.sum_sum_type,
    sum_add_distrib]

/-- The sign of a corner is `−1` to the number of operands at their second value. -/
lemma cornerSign_eq_pow {κ : Type} [Fintype κ] [DecidableEq κ] (ε : κ ⊕ κ → Bool) :
    cornerSign ε = (-1) ^ ∑ o, if ε o then 1 else 0 := by
  rw [cornerSign, ← prod_pow_eq_pow_sum]
  exact prod_congr rfl fun o _ => by split_ifs <;> simp

/-- The signs of the corners cancel. -/
lemma sum_cornerSign {κ : Type} [Fintype κ] [DecidableEq κ] [Nonempty κ] :
    ∑ ε : κ ⊕ κ → Bool, cornerSign ε = 0 := by
  simp only [cornerSign]
  rw [← Fintype.prod_sum (fun (_ : κ ⊕ κ) (b : Bool) => if b then (-1 : ℂ) else 1)]
  exact prod_eq_zero (mem_univ (Sum.inl (Classical.arbitrary κ))) (by simp)

variable [DecidableEq γ]

/-- The swap of `a` and `b` to the power `n` sends `a` to `a` or `b` by the parity of `n`. -/
lemma swap_iterate (a b : γ) (n : ℕ) : (Equiv.swap a b)^[n] a = if Even n then a else b := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Function.iterate_succ_apply', ih]
    by_cases h : Even n <;> simp [h, Nat.even_add_one]

/-- **The corner sum of a readout of the walk** is `2^{2K − 1} (g a − g b)`. -/
theorem two_mul_cornerSum_pairWalk {K : ℕ} (hK : 0 < K) (a b : γ) (g : γ → ℂ) :
    2 * cornerSum (fun _ => id) (fun _ => ⇑(Equiv.swap a b)) (fun _ => id)
      (fun _ => ⇑(Equiv.swap a b)) (fun x y => g (pairWalk K x y a)) =
        2 ^ (2 * K) * (g a - g b) := by
  have : Nonempty (Fin K) := ⟨⟨0, hK⟩⟩
  have key : ∀ ε : Fin K ⊕ Fin K → Bool, 2 * (cornerSign ε *
      g (pairWalk K (cornerX (fun _ => id) (fun _ => ⇑(Equiv.swap a b)) ε)
        (cornerY (fun _ => id) (fun _ => ⇑(Equiv.swap a b)) ε) a)) =
        (g a - g b) + cornerSign ε * (g a + g b) := by
    intro ε
    rw [pairWalk_corner, swap_iterate, cornerSign_eq_pow]
    rcases Nat.even_or_odd (∑ o, if ε o then 1 else 0) with h | h
    · simp only [h, ↓reduceIte, h.neg_one_pow]
      ring
    · have h' : ¬Even (∑ o, if ε o then 1 else 0) := Nat.not_even_iff_odd.2 h
      simp only [h', ↓reduceIte, h.neg_one_pow]
      ring
  rw [cornerSum_apply, mul_sum, sum_congr rfl fun ε _ => key ε, sum_add_distrib, ← sum_mul,
    sum_cornerSign, zero_mul, add_zero, sum_const, card_univ, Fintype.card_fun,
    Fintype.card_bool, Fintype.card_sum, Fintype.card_fin, nsmul_eq_mul]
  push_cast
  ring_nf

/-- A readout of the walk that tells `a` from `b` has a nonzero corner sum. -/
lemma cornerSum_pairWalk_ne_zero {K : ℕ} (hK : 0 < K) {a b : γ} (g : γ → ℂ) (hg : g a ≠ g b) :
    cornerSum (fun _ => id) (fun _ => ⇑(Equiv.swap a b)) (fun _ => id)
      (fun _ => ⇑(Equiv.swap a b)) (fun x y => g (pairWalk K x y a)) ≠ 0 := fun h0 => by
  have hc := two_mul_cornerSum_pairWalk hK a b g
  rw [h0, mul_zero] at hc
  exact hg (sub_eq_zero.1 ((mul_eq_zero.1 hc.symm).resolve_left (by simp)))

/-- **A nonzero corner sum is full order**: with order at most `2^D`, `1 + ⌈log₂ |κ|⌉ ≤ D`. -/
lemma one_add_clog_le_of_cornerSum_ne_zero {κ α β : Type} [Fintype κ] [DecidableEq κ] {D : ℕ}
    (hK : 0 < Fintype.card κ) {φ : (κ → α) → (κ → β) → ℂ} (hφ : φ ∈ orderSpan (2 ^ D))
    {x₀ x₁ : κ → α} {y₀ y₁ : κ → β} (h : cornerSum x₀ x₁ y₀ y₁ φ ≠ 0) :
    1 + Nat.clog 2 (Fintype.card κ) ≤ D := by
  have h2 : 2 * Fintype.card κ ≤ 2 ^ D := by
    by_contra hlt
    exact h (cornerSum_eq_zero x₀ x₁ y₀ y₁ (by omega) hφ)
  obtain _ | D := D
  · rw [pow_zero] at h2
    omega
  · have hk : Fintype.card κ ≤ 2 ^ D := by rw [pow_succ] at h2; omega
    have := Nat.clog_le_of_le_pow hk
    omega

/-- **Passes count as depth for `2K` hops**: if a linear readout of a coda of depth `d₂` after
`n + 1` passes of a core of depth `d₁`, behind a prelude of depth `d₀` on features that read one
table each, tells the keys `a` and `b` apart at the end of every walk from `a`, then
`1 + ⌈log₂ K⌉ ≤ d₀ + (n + 1) d₁ + d₂`, at any width. -/
theorem le_passes_of_loop_walk {K : ℕ} (hK : 0 < K) {a b : γ} (g : γ → ℂ) (hg : g a ≠ g b)
    {ι ν : Type} [Fintype ι] [Fintype ν] {m₀ m₁ m₂ d₀ d₁ d₂ : ℕ}
    (Z : (Fin K → γ → γ) → (Fin K → γ → γ) → ι → ℂ) (hZ : ∀ c, (fun x y => Z x y c) ∈ orderSpan 1)
    {P : (ι → ℂ) → ι → ℂ} (hP : IsProductNet m₀ d₀ P)
    {L : (ι ⊕ ν → ℂ) → ι ⊕ ν → ℂ} (hL : IsProductNet m₁ d₁ L) (R : (ι → ℂ) →ₗ[ℂ] ν → ℂ)
    {C : (ι → ℂ) → ι → ℂ} (hC : IsProductNet m₂ d₂ C) (ℓ : (ι → ℂ) →ₗ[ℂ] ℂ) (n : ℕ)
    (hval : ∀ x y, ℓ (C (loopState L R (P (Z x y)) n)) = g (pairWalk K x y a)) :
    1 + Nat.clog 2 K ≤ d₀ + (n + 1) * d₁ + d₂ := by
  have h1 := loopState_mem_orderSpan hL R (Z := fun x y => P (Z x y)) (hP.order hZ) n
  have h3 := linear_mem_orderSpan (hC.order (Z := fun x y => loopState L R (P (Z x y)) n) h1) ℓ
  rw [show (fun x y => ℓ (C (loopState L R (P (Z x y)) n))) = fun x y => g (pairWalk K x y a) from
    funext fun x => funext (hval x), show 2 ^ d₂ * (2 ^ (d₁ * (n + 1)) * (2 ^ d₀ * 1)) =
      2 ^ (d₀ + (n + 1) * d₁ + d₂) by ring] at h3
  simpa using one_add_clog_le_of_cornerSum_ne_zero (by simpa using hK) h3
    (cornerSum_pairWalk_ne_zero hK g hg)

/-- The one-hot features of the links: the coordinate `(i, c, v)` of table `x i` is
`[x i c = v]`, and likewise for `y i`. -/
def linkFeatures {K : ℕ} (x y : Fin K → γ → γ) : (Fin K × γ × γ) ⊕ (Fin K × γ × γ) → ℂ :=
  Sum.elim (fun t => if x t.1 t.2.1 = t.2.2 then 1 else 0)
    fun t => if y t.1 t.2.1 = t.2.2 then 1 else 0

/-- The features of the links, and a zero scratch, read one table per coordinate. -/
lemma linkFeatures_mem_orderSpan {K : ℕ} {T : Type}
    (c : ((Fin K × γ × γ) ⊕ (Fin K × γ × γ)) ⊕ T) :
    (fun x y => (Sum.elim (linkFeatures x y) 0 : _ → ℂ) c) ∈ orderSpan 1 := by
  rcases c with (⟨i, c, v⟩ | ⟨i, c, v⟩) | t
  · refine mem_orderSpan_of_readsOnly (S := {Sum.inl i}) (by simp) fun x x' y y' hx _ => ?_
    simp [linkFeatures, hx i (mem_singleton_self _)]
  · refine mem_orderSpan_of_readsOnly (S := {Sum.inr i}) (by simp) fun x x' y y' _ hy => ?_
    simp [linkFeatures, hy i (mem_singleton_self _)]
  · exact mem_orderSpan_of_readsOnly (S := ∅) (by simp) fun _ _ _ _ _ _ => rfl

/-- Four links on two keys: the walk from `false` through swaps ends at `false` on an even number
of swaps, and the corner sum of its indicator is `2^3 = 8`. -/
example : 2 * cornerSum (fun _ => id) (fun _ => ⇑(Equiv.swap false true)) (fun _ => id)
    (fun _ => ⇑(Equiv.swap false true))
    (fun x y => if pairWalk 2 x y false = false then (1 : ℂ) else 0) = 16 := by
  rw [two_mul_cornerSum_pairWalk (by norm_num) false true fun c => if c = false then 1 else 0]
  norm_num

/-- **Four links need two passes** of a one-layer core with no prelude and no coda, at any width. -/
example {T ν : Type} [Fintype T] [Fintype ν] {m : ℕ}
    {L : ((((Fin 2 × Bool × Bool) ⊕ (Fin 2 × Bool × Bool)) ⊕ T) ⊕ ν → ℂ) →
      (((Fin 2 × Bool × Bool) ⊕ (Fin 2 × Bool × Bool)) ⊕ T) ⊕ ν → ℂ}
    (hL : IsProductNet m 1 L) (R : (((Fin 2 × Bool × Bool) ⊕ (Fin 2 × Bool × Bool)) ⊕ T → ℂ) →ₗ[ℂ]
      ν → ℂ) (ℓ : (((Fin 2 × Bool × Bool) ⊕ (Fin 2 × Bool × Bool)) ⊕ T → ℂ) →ₗ[ℂ] ℂ) (n : ℕ)
    (hval : ∀ x y, ℓ (loopState L R (Sum.elim (linkFeatures x y) 0) n) =
      if pairWalk 2 x y false = false then 1 else 0) : 2 ≤ n + 1 := by
  have h := le_passes_of_loop_walk (by norm_num) (a := false) (b := true)
    (fun c => if c = false then 1 else 0) (by simp) _ linkFeatures_mem_orderSpan
    (IsProductNet.id_net 0 0) hL R (IsProductNet.id_net 0 0) ℓ n hval
  rw [show Nat.clog 2 2 = 1 by decide] at h
  omega

end Transformer.Grokking.FlatSpectrum

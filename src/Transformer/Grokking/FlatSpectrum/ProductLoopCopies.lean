module

public import Transformer.Grokking.FlatSpectrum.ProductLoopTree

/-!
# `k` copies in exactly `⌈log₂ k⌉` passes of one shared layer

Source: `review/flat_spectrum.md`, *Loops*, item LP2. One prelude layer writes the start of the
looped tree from the one-hot features of the pairs: unit `(t, b)` multiplies the reading
`∑_a G_t(a, b) [x_{π t} = a]` by `[y_{π t} = b]`, so scratch slot `t` gets `G_t(x_{π t}, y_{π t})`
(`leafLayer`, `leafLayer_features`). For `k` copies of `q`, leaf slot `l < k` gets
`ψ(q(x_l, y_l))`, the other leaf slots get `1`, and the product slots get the start of
`TreeStart` (`copiesStart`). Then pass `n + 1 ≥ ⌈log₂ k⌉` of `treeCore`, with `k` units, writes
`ψ(∑_i q(x_i, y_i))` into the root (`loopCopies_root`), and no later pass changes the state
(`loopCopies_add`). With LP1 this is an equivalence (`loopCopies_root_iff`): the root holds
`ψ(∑_i q(x_i, y_i))` on all inputs after pass `n + 1` iff `⌈log₂ k⌉ ≤ n + 1`, so a prelude layer
and `⌈log₂ k⌉` passes of one shared layer reach NV7's depth `1 + ⌈log₂ k⌉`, the least possible.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {κ α β T : Type} [DecidableEq κ] [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β]
  [Fintype T] [DecidableEq T]

/-- The gate of unit `(t, b)` of the leaf layer: `∑_a G_t(a, b) [x_{π t} = a]`. -/
def leafGate (π : T → κ) (G : T → α → β → ℂ) (tb : T × β) :
    (((κ × α) ⊕ (κ × β)) ⊕ T → ℂ) →ₗ[ℂ] ℂ :=
  ∑ a, G tb.1 a tb.2 • LinearMap.proj (Sum.inl (Sum.inl (π tb.1, a)))

/-- The up reading of unit `(t, b)` of the leaf layer: `[y_{π t} = b]`. -/
def leafUp (π : T → κ) (tb : T × β) : (((κ × α) ⊕ (κ × β)) ⊕ T → ℂ) →ₗ[ℂ] ℂ :=
  LinearMap.proj (Sum.inl (Sum.inr (π tb.1, tb.2)))

/-- **The leaf layer**: unit `(t, b)` writes the product of its two readings into scratch
slot `t`. -/
def leafLayer (π : T → κ) (G : T → α → β → ℂ) :
    (((κ × α) ⊕ (κ × β)) ⊕ T → ℂ) → ((κ × α) ⊕ (κ × β)) ⊕ T → ℂ :=
  fun z => z + ∑ tb, (leafGate π G tb z * leafUp (α := α) π tb z) • Pi.single (Sum.inr tb.1) 1

lemma isProductLayer_leafLayer (π : T → κ) (G : T → α → β → ℂ) :
    IsProductLayer (Fintype.card T * Fintype.card β) (leafLayer π G) :=
  ⟨T × β, inferInstance, (Fintype.card_prod T β).le, leafGate π G, leafUp π,
    fun tb => Pi.single (Sum.inr tb.1) 1, fun _ => rfl⟩

/-- On the features and a zero scratch, slot `t` of the leaf layer gets `G_t(x_{π t}, y_{π t})`. -/
lemma leafLayer_features (π : T → κ) (G : T → α → β → ℂ) (x : κ → α) (y : κ → β) (t : T) :
    leafLayer π G (Sum.elim (pairFeatures x y) 0) (Sum.inr t) = G t (x (π t)) (y (π t)) := by
  have hg (tb : T × β) :
      leafGate π G tb (Sum.elim (pairFeatures x y) 0) = G tb.1 (x (π tb.1)) tb.2 := by
    simp [leafGate, pairFeatures, oneHot]
  have hu (tb : T × β) : leafUp (α := α) π tb (Sum.elim (pairFeatures x y) 0) =
      if y (π tb.1) = tb.2 then 1 else 0 := by
    simp [leafUp, pairFeatures, oneHot]
  simp [leafLayer, Finset.sum_apply, Pi.single_apply, hg, hu, Fintype.sum_prod_type]

variable {p k : ℕ} [NeZero p]

/-- The pair that scratch slot `t` reads: pair `l` for the leaf slot `l < k`, pair `0` for the
rest. -/
def copiesPair (hk : 0 < k) : Fin (2 * k) ⊕ Fin (2 * k) → Fin k
  | Sum.inl l => if h : (l : ℕ) < k then ⟨l, h⟩ else ⟨0, hk⟩
  | Sum.inr _ => ⟨0, hk⟩

/-- What the prelude writes into scratch slot `t` from the pair `(a, b)` it reads: `ψ(q(a, b))`
into the leaf slots `l < k`, `1` into the other leaf slots, and the start of the tree into the
product slots. -/
noncomputable def copiesValue (q : α → β → ZMod p) : Fin (2 * k) ⊕ Fin (2 * k) → α → β → ℂ
  | Sum.inl l, a, b => if (l : ℕ) < k then ZMod.stdAddChar (q a b) else 1
  | Sum.inr i, _, _ => if (i : ℕ) < k then 0 else 1

/-- **The prelude state of `k` copies**: one leaf layer, with `4k|β|` units, on the features of
the pairs and a zero scratch of `2k` leaf slots and `2k` product slots. -/
noncomputable def copiesStart (hk : 0 < k) (q : α → β → ZMod p) (x : Fin k → α) (y : Fin k → β) :
    ((Fin k × α) ⊕ (Fin k × β)) ⊕ (Fin (2 * k) ⊕ Fin (2 * k)) → ℂ :=
  leafLayer (copiesPair hk) (copiesValue q) (Sum.elim (pairFeatures x y) 0)

lemma treeStart_copiesStart (hk : 0 < k) (q : α → β → ZMod p) (x : Fin k → α) (y : Fin k → β) :
    TreeStart (copiesStart hk q x y) :=
  fun i => leafLayer_features _ _ x y (Sum.inr i)

lemma treeLeaves_copiesStart (hk : 0 < k) (q : α → β → ZMod p) (x : Fin k → α) (y : Fin k → β)
    (i : Fin k) : treeLeaves (copiesStart hk q x y) i = ZMod.stdAddChar (q (x i) (y i)) := by
  have h2 : (i : ℕ) < 2 * k := by omega
  simp [treeLeaves, h2, copiesStart, leafLayer_features, copiesValue, copiesPair]

lemma treeLeaves_copiesStart_of_le (hk : 0 < k) (q : α → β → ZMod p) (x : Fin k → α)
    (y : Fin k → β) {l : ℕ} (hl : k ≤ l) : treeLeaves (copiesStart hk q x y) l = 1 := by
  by_cases h2 : l < 2 * k
  · simp [treeLeaves, h2, copiesStart, leafLayer_features, copiesValue, not_lt.2 hl]
  · simp [treeLeaves, h2]

/-- **`⌈log₂ k⌉` passes are enough**: after pass `n + 1 ≥ ⌈log₂ k⌉` of `treeCore` with `k` units,
the root holds `ψ(∑_i q(x_i, y_i))`. -/
theorem loopCopies_root (hk : 0 < k) (q : α → β → ZMod p) {n : ℕ} (hn : Nat.clog 2 k ≤ n + 1)
    (x : Fin k → α) (y : Fin k → β) :
    loopState treeCore treeRead (copiesStart hk q x y) n (prodSlot ⟨0, by omega⟩) =
      ZMod.stdAddChar (∑ i, q (x i) (y i)) := by
  rw [loopState_treeCore_root (treeStart_copiesStart hk q x y) hk
    (fun _ hl => treeLeaves_copiesStart_of_le hk q x y hl)
    ((Nat.clog_le_iff_le_pow one_lt_two).1 hn), ← Fin.prod_univ_eq_prod_range, stdAddChar_sum]
  exact Fintype.prod_congr _ _ fun i => treeLeaves_copiesStart hk q x y i

/-- **Then the loop stands still**: every later pass writes what pass `n + 1` wrote. -/
theorem loopCopies_add (hk : 0 < k) (q : α → β → ZMod p) {n : ℕ} (hn : Nat.clog 2 k ≤ n + 1)
    (x : Fin k → α) (y : Fin k → β) (j : ℕ) :
    loopState treeCore treeRead (copiesStart hk q x y) (n + j) =
      loopState treeCore treeRead (copiesStart hk q x y) n :=
  loopState_treeCore_add (treeStart_copiesStart hk q x y)
    (fun _ hl => treeLeaves_copiesStart_of_le hk q x y hl)
    ((Nat.clog_le_iff_le_pow one_lt_two).1 hn) j

/-- **Exactly `⌈log₂ k⌉` passes**: when `ψ ∘ q` has a nonzero mixed difference, the root holds
`ψ(∑_i q(x_i, y_i))` on all inputs after pass `n + 1` iff `⌈log₂ k⌉ ≤ n + 1`. The bound is LP1
with a prelude and a core of one layer each and no coda. -/
theorem loopCopies_root_iff (hk : 0 < k) (q : α → β → ZMod p)
    (hq : ∃ a₀ a₁ b₀ b₁, ZMod.stdAddChar (q a₀ b₀) - ZMod.stdAddChar (q a₁ b₀) -
      ZMod.stdAddChar (q a₀ b₁) + ZMod.stdAddChar (q a₁ b₁) ≠ 0) (n : ℕ) :
    (∀ x y, loopState treeCore treeRead (copiesStart hk q x y) n (prodSlot ⟨0, by omega⟩) =
      ZMod.stdAddChar (∑ i, q (x i) (y i))) ↔ Nat.clog 2 k ≤ n + 1 := by
  refine ⟨fun h => ?_, fun hn x y => loopCopies_root hk q hn x y⟩
  have := le_passes_of_loop_copies hk q hq
    (IsProductNet.of_layer (isProductLayer_leafLayer (copiesPair hk) (copiesValue q)))
    (IsProductNet.of_layer isProductLayer_treeCore) treeRead (IsProductNet.id_net 0 0)
    (LinearMap.proj (prodSlot ⟨0, by omega⟩)) n h
  omega

/-- The start of three copies: the product slots as `TreeStart` asks, leaf `2` holds
`ψ(q(x₂, y₂))`, and leaf `3` holds `1`. -/
example (x y : Fin 3 → ZMod 7) :
    TreeStart (copiesStart (by norm_num) (fun a b : ZMod 7 => a ^ 3 + a * b) x y) ∧
      treeLeaves (copiesStart (by norm_num) (fun a b : ZMod 7 => a ^ 3 + a * b) x y) 2 =
        ZMod.stdAddChar (x 2 ^ 3 + x 2 * y 2) ∧
      treeLeaves (copiesStart (by norm_num) (fun a b : ZMod 7 => a ^ 3 + a * b) x y) 3 = 1 :=
  ⟨treeStart_copiesStart _ _ x y, treeLeaves_copiesStart _ _ x y 2,
    treeLeaves_copiesStart_of_le _ _ x y le_rfl⟩

/-- Three copies of `x³ + x y` at `p = 7`: two passes write `ψ(f)` into the root, and a third
changes nothing. -/
example (x y : Fin 3 → ZMod 7) :
    loopState treeCore treeRead (copiesStart (by norm_num) (fun a b : ZMod 7 => a ^ 3 + a * b) x y)
      1 (prodSlot ⟨0, by norm_num⟩) = ZMod.stdAddChar (∑ i, (x i ^ 3 + x i * y i)) ∧
    loopState treeCore treeRead (copiesStart (by norm_num) (fun a b : ZMod 7 => a ^ 3 + a * b) x y)
      (1 + 1) =
    loopState treeCore treeRead (copiesStart (by norm_num) (fun a b : ZMod 7 => a ^ 3 + a * b) x y)
      1 :=
  ⟨loopCopies_root _ _ (Nat.clog_le_of_le_pow (by norm_num)) x y,
    loopCopies_add _ _ (Nat.clog_le_of_le_pow (by norm_num)) x y 1⟩

/-- One pass is not enough for three copies, and by LP1 no width would help. -/
example : ¬ ∀ x y : Fin 3 → ZMod 7,
    loopState treeCore treeRead (copiesStart (by norm_num) (fun a b : ZMod 7 => a ^ 3 + a * b) x y)
      0 (prodSlot ⟨0, by norm_num⟩) = ZMod.stdAddChar (∑ i, (x i ^ 3 + x i * y i)) := fun h => by
  have := (loopCopies_root_iff (k := 3) (by norm_num) (fun a b : ZMod 7 => a ^ 3 + a * b)
    ⟨0, 1, 0, 1, mixed_stdAddChar_ne_zero (by norm_num) fun a => a ^ 3⟩ 0).1 h
  rw [Nat.clog_le_iff_le_pow one_lt_two] at this
  norm_num at this

end Transformer.Grokking.FlatSpectrum

module

public import Transformer.Grokking.FlatSpectrum.ProductLoop

/-!
# The looped tree: one layer multiplies adjacent pairs on every pass

Source: `review/flat_spectrum.md`, *Loops*, item LP2. Every level of NV7's tree does the same
thing: it multiplies adjacent pairs. So one shared layer can do all levels. Take `2h` leaf slots
`P` and `2h` product slots `S` on the prelude's stream: unit `i < h` multiplies the readings of
`P(2i)` and `P(2i + 1)` into `S(i)` (`treeCore`). Every pass restarts from the prelude state,
where `P` holds the leaves and `S` holds `0` on its first half and `1` on its second
(`TreeStart`), so the message `m = S − P` (`treeRead`) turns the reading `z(P l) + m(l)` of a leaf
slot into the product the last pass wrote into `S(l)`. Pass `n` then writes into `S(i)` the
product of the leaves `[i 2^n, (i + 1) 2^n)`, the leaves past the `2h` slots counting as `1`
(`treeLevel`, `loopState_treeCore`). So the root `S(0)` holds the product of the first `2^n`
leaves (`loopState_treeCore_root`), and once `2^n` covers every leaf that is not `1`, the next pass
changes nothing (`loopState_treeCore_succ`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

/-- The product of the leaves `[i 2^n, (i + 1) 2^n)`: slot `i` of level `n` of the tree. -/
def treeLevel (v : ℕ → ℂ) (n i : ℕ) : ℂ := ∏ l ∈ Ico (i * 2 ^ n) ((i + 1) * 2 ^ n), v l

lemma treeLevel_zero (v : ℕ → ℂ) (i : ℕ) : treeLevel v 0 i = v i := by
  simp [treeLevel]

/-- **A level is the products of adjacent pairs of the level below.** -/
lemma treeLevel_succ (v : ℕ → ℂ) (n i : ℕ) :
    treeLevel v (n + 1) i = treeLevel v n (2 * i) * treeLevel v n (2 * i + 1) := by
  have e₁ : i * 2 ^ (n + 1) = 2 * i * 2 ^ n := by ring
  have e₂ : (i + 1) * 2 ^ (n + 1) = (2 * i + 1 + 1) * 2 ^ n := by ring
  rw [treeLevel, treeLevel, treeLevel, e₁, e₂, prod_Ico_consecutive]
  · exact Nat.mul_le_mul_right _ (by omega)
  · exact Nat.mul_le_mul_right _ (by omega)

/-- A slot whose leaves all lie at or past `K` holds `1` when the leaves from `K` on are `1`. -/
lemma treeLevel_eq_one {v : ℕ → ℂ} {K : ℕ} (hv : ∀ l, K ≤ l → v l = 1) {n i : ℕ}
    (hi : K ≤ i * 2 ^ n) : treeLevel v n i = 1 :=
  prod_eq_one fun l hl => hv l (hi.trans (mem_Ico.1 hl).1)

/-- The root of level `n` holds the product of the first `K` leaves once `K ≤ 2^n`, when the
leaves from `K` on are `1`. -/
lemma treeLevel_root {v : ℕ → ℂ} {K : ℕ} (hv : ∀ l, K ≤ l → v l = 1) {n : ℕ} (hn : K ≤ 2 ^ n) :
    treeLevel v n 0 = ∏ l ∈ range K, v l := by
  rw [treeLevel, zero_mul, zero_add, one_mul, Nat.Ico_zero_eq_range,
    ← prod_range_mul_prod_Ico _ hn, prod_eq_one fun l hl => hv l (mem_Ico.1 hl).1, mul_one]

example : treeLevel (fun l => (l : ℂ) + 1) 1 0 = 2 := by
  rw [treeLevel_succ, treeLevel_zero, treeLevel_zero]
  norm_num

variable {F : Type*} [DecidableEq F] {h : ℕ}

/-- Leaf slot `l` of the stream `F ⊕ (P ⊕ S)`. -/
abbrev leafSlot (l : Fin (2 * h)) : F ⊕ (Fin (2 * h) ⊕ Fin (2 * h)) := Sum.inr (Sum.inl l)

/-- Product slot `i` of the stream `F ⊕ (P ⊕ S)`. -/
abbrev prodSlot (i : Fin (2 * h)) : F ⊕ (Fin (2 * h) ⊕ Fin (2 * h)) := Sum.inr (Sum.inr i)

/-- The reading of leaf slot `l` through the message: `z(P l) + m(l)`. -/
def slotReading (l : Fin (2 * h)) :
    ((F ⊕ (Fin (2 * h) ⊕ Fin (2 * h))) ⊕ Fin (2 * h) → ℂ) →ₗ[ℂ] ℂ :=
  LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (Sum.inl (leafSlot l)) +
    LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (Sum.inr l)

/-- **The shared layer**: unit `i < h` multiplies the readings of the leaf slots `2i` and `2i + 1`
and writes the product into the product slot `i`. -/
def treeCore : ((F ⊕ (Fin (2 * h) ⊕ Fin (2 * h))) ⊕ Fin (2 * h) → ℂ) →
    (F ⊕ (Fin (2 * h) ⊕ Fin (2 * h))) ⊕ Fin (2 * h) → ℂ :=
  fun u => u + ∑ i : Fin h,
    (slotReading ⟨2 * i, by omega⟩ u * slotReading ⟨2 * i + 1, by omega⟩ u) •
      Pi.single (Sum.inl (prodSlot ⟨i, by omega⟩)) 1

lemma isProductLayer_treeCore : IsProductLayer h (treeCore (F := F) (h := h)) :=
  ⟨Fin h, inferInstance, by simp, fun i => slotReading ⟨2 * i, by omega⟩,
    fun i => slotReading ⟨2 * i + 1, by omega⟩,
    fun i => Pi.single (Sum.inl (prodSlot ⟨i, by omega⟩)) 1, fun _ => rfl⟩

/-- **The message**: the products minus the leaves, `m(l) = s(S l) − s(P l)`. -/
def treeRead : (F ⊕ (Fin (2 * h) ⊕ Fin (2 * h)) → ℂ) →ₗ[ℂ] Fin (2 * h) → ℂ :=
  LinearMap.pi fun l => LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (prodSlot l) -
    LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (leafSlot l)

/-- The leaves of a state, `1` past the `2h` slots. -/
def treeLeaves (z : F ⊕ (Fin (2 * h) ⊕ Fin (2 * h)) → ℂ) (l : ℕ) : ℂ :=
  if hl : l < 2 * h then z (leafSlot ⟨l, hl⟩) else 1

/-- A prelude state the tree loop starts from: the product slots hold `0` on their first half and
`1` on their second. -/
def TreeStart (z : F ⊕ (Fin (2 * h) ⊕ Fin (2 * h)) → ℂ) : Prop :=
  ∀ i : Fin (2 * h), z (prodSlot i) = if (i : ℕ) < h then 0 else 1

/-- The prelude state with level `n` of the tree in the product slots. -/
def treeState (z : F ⊕ (Fin (2 * h) ⊕ Fin (2 * h)) → ℂ) (n : ℕ) :
    F ⊕ (Fin (2 * h) ⊕ Fin (2 * h)) → ℂ :=
  Sum.elim (fun f => z (Sum.inl f))
    (Sum.elim (fun l => z (leafSlot l)) fun i => treeLevel (treeLeaves z) n i)

/-- One pass whose leaf readings are level `n` writes level `n + 1` into the product slots and
leaves the rest of the prelude state as it was. -/
lemma treeCore_pass {z : F ⊕ (Fin (2 * h) ⊕ Fin (2 * h)) → ℂ} (hz : TreeStart z) {n : ℕ}
    {m : Fin (2 * h) → ℂ}
    (hm : ∀ l : Fin (2 * h), z (leafSlot l) + m l = treeLevel (treeLeaves z) n l) :
    treeCore (Sum.elim z m) ∘ Sum.inl = treeState z (n + 1) := by
  have hread (l : Fin (2 * h)) : slotReading l (Sum.elim z m) = treeLevel (treeLeaves z) n l :=
    hm l
  funext c
  simp only [Function.comp_apply, treeCore, Pi.add_apply, Finset.sum_apply, Pi.smul_apply,
    smul_eq_mul, hread, Sum.elim_inl, Pi.single_apply]
  rcases c with f | l | i
  · simp [treeState]
  · simp [treeState]
  · simp [treeState, Fin.ext_iff]
    rw [hz i]
    by_cases hi : (i : ℕ) < h
    · rw [ite_eq_left hi, zero_add, Fintype.sum_eq_single (⟨i, hi⟩ : Fin h) fun x hx =>
        ite_eq_right fun (e : (i : ℕ) = x) => hx (Fin.ext e.symm)]
      simp [treeLevel_succ]
    · have h2 : 2 ≤ 2 ^ (n + 1) := by
        rw [pow_succ]
        have := Nat.one_le_two_pow (n := n)
        omega
      rw [ite_eq_right hi, Fintype.sum_eq_zero _ fun (x : Fin h) =>
        ite_eq_right fun (e : (i : ℕ) = x) => hi (e ▸ x.2), add_zero]
      exact (treeLevel_eq_one (K := 2 * h) (fun l hl => by simp [treeLeaves, not_lt.2 hl])
        (by have := Nat.mul_le_mul (not_lt.1 hi) h2; omega)).symm

/-- **The looped tree**: pass `n + 1` writes level `n + 1` of the tree into the product slots and
leaves the rest of the prelude state as it was. -/
theorem loopState_treeCore {z : F ⊕ (Fin (2 * h) ⊕ Fin (2 * h)) → ℂ} (hz : TreeStart z) (n : ℕ) :
    loopState treeCore treeRead z n = treeState z (n + 1) := by
  induction n with
  | zero => exact treeCore_pass hz fun l => by simp [treeLevel_zero, treeLeaves]
  | succ n ih =>
    refine treeCore_pass hz fun l => ?_
    rw [ih]
    simp [treeRead, treeState]

/-- **The root** holds the product of the first `K` leaves after pass `n + 1` once
`K ≤ 2^{n+1}`, when the leaves from `K` on are `1`. -/
theorem loopState_treeCore_root {z : F ⊕ (Fin (2 * h) ⊕ Fin (2 * h)) → ℂ} (hz : TreeStart z)
    (hh : 0 < h) {K : ℕ} (hv : ∀ l, K ≤ l → treeLeaves z l = 1) {n : ℕ} (hn : K ≤ 2 ^ (n + 1)) :
    loopState treeCore treeRead z n (prodSlot ⟨0, by omega⟩) = ∏ l ∈ range K, treeLeaves z l := by
  rw [loopState_treeCore hz]
  exact treeLevel_root hv hn

/-- **The fixed point**: once `K ≤ 2^{n+1}`, pass `n + 2` writes what pass `n + 1` wrote, when the
leaves from `K` on are `1`. -/
theorem loopState_treeCore_succ {z : F ⊕ (Fin (2 * h) ⊕ Fin (2 * h)) → ℂ} (hz : TreeStart z)
    {K : ℕ} (hv : ∀ l, K ≤ l → treeLeaves z l = 1) {n : ℕ} (hn : K ≤ 2 ^ (n + 1)) :
    loopState treeCore treeRead z (n + 1) = loopState treeCore treeRead z n := by
  rw [loopState_treeCore hz, loopState_treeCore hz]
  funext c
  rcases c with f | l | i
  · rfl
  · rfl
  · have hn' : K ≤ 2 ^ (n + 1 + 1) := hn.trans (Nat.pow_le_pow_right (by norm_num) (by omega))
    show treeLevel _ (n + 1 + 1) (i : ℕ) = treeLevel _ (n + 1) (i : ℕ)
    rcases Nat.eq_zero_or_pos (i : ℕ) with h0 | hpos
    · rw [h0, treeLevel_root hv hn', treeLevel_root hv hn]
    · rw [treeLevel_eq_one hv (hn'.trans (Nat.le_mul_of_pos_left _ hpos)),
        treeLevel_eq_one hv (hn.trans (Nat.le_mul_of_pos_left _ hpos))]

/-- Every later pass agrees with the fixed point, by `loopState_add_of_succ_eq`. -/
theorem loopState_treeCore_add {z : F ⊕ (Fin (2 * h) ⊕ Fin (2 * h)) → ℂ} (hz : TreeStart z)
    {K : ℕ} (hv : ∀ l, K ≤ l → treeLeaves z l = 1) {n : ℕ} (hn : K ≤ 2 ^ (n + 1)) (j : ℕ) :
    loopState treeCore treeRead z (n + j) = loopState treeCore treeRead z n :=
  loopState_add_of_succ_eq (loopState_treeCore_succ hz hv hn) j

/-- Two leaves `2` on one unit: the root holds `2 · 2` from the first pass on. -/
example (j : ℕ) :
    let z : Unit ⊕ (Fin (2 * 1) ⊕ Fin (2 * 1)) → ℂ :=
      Sum.elim 0 (Sum.elim (fun _ => 2) fun i => if (i : ℕ) < 1 then 0 else 1)
    treeCore (Sum.elim z 0) ∘ Sum.inl = treeState z 1 ∧
      loopState treeCore treeRead z 0 = treeState z 1 ∧
      loopState treeCore treeRead z 0 (prodSlot ⟨0, by omega⟩) = ∏ l ∈ range 2, treeLeaves z l ∧
      loopState treeCore treeRead z 1 = loopState treeCore treeRead z 0 ∧
      loopState treeCore treeRead z (0 + j) = loopState treeCore treeRead z 0 := by
  intro z
  have hz : TreeStart z := fun _ => rfl
  have hv (l : ℕ) (hl : 2 ≤ l) : treeLeaves z l = 1 := by simp [treeLeaves, not_lt.2 hl]
  exact ⟨treeCore_pass hz fun l => by simp [treeLevel_zero, treeLeaves, l.2],
    loopState_treeCore hz 0, loopState_treeCore_root hz (by norm_num) hv (by norm_num),
    loopState_treeCore_succ hz hv (by norm_num), loopState_treeCore_add hz hv (by norm_num) j⟩

example :
    treeLevel (fun l => if l < 2 then 2 else 1) 1 0 = ∏ l ∈ range 2, (if l < 2 then 2 else 1) ∧
    treeLevel (fun l => if l < 2 then (2 : ℂ) else 1) 1 1 = 1 :=
  ⟨treeLevel_root (fun l hl => ite_eq_right (not_lt.2 hl)) (by norm_num),
    treeLevel_eq_one (K := 2) (fun l hl => ite_eq_right (not_lt.2 hl)) (by norm_num)⟩

end Transformer.Grokking.FlatSpectrum

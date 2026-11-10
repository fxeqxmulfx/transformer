module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Tactic.Linarith
public import Transformer.Grokking.FlatSpectrum.BlockCut.Defs

/-!
# `L_k` is a first letter and a number of blocks

Source: `review/flat_spectrum.md`, *The block languages across a cut* (added), items BL1 and
BL2, for the languages `L_k` of arXiv:2506.16055v3, §2.4, Equation `eq:altplus`, in the form
`Transformer.CRASP.altPlus`.

BL2. A word that is not empty is in `L_k` exactly when it starts with `a` and has `k` blocks
(`mem_altPlus_iff`; `mem_altPlus_succ_iff` for `k + 1` and every word). The alternating word of
length `n` has `n` blocks (`blocks_altList`), and one of odd length ends with its first letter
(`getLast?_altList_odd`).

BL1 in numbers. Write `ℓ(u) = [u ends with b]` and, for a suffix `v` that is not empty,
`f(v) = [v starts with b]`. Then `[last u = first v] = 1 − ℓ − f + 2 ℓ f`, so for a prefix `u`
that is not empty

    blocks (u ++ v) = blocks u + shift v + ℓ(u) · slope v           (`cast_blocks_append`),

with `shift v = blocks v − 1 + f(v)` and `slope v = 1 − 2 f(v)`, both `0` for the empty suffix:
the suffix acts on the two numbers `blocks u` and `ℓ(u)` of the prefix.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.BlockCut

open Transformer.CRASP

/-- A word of `L_{k+1}` that starts with `s` starts with `s`. -/
lemma head?_of_mem_altPlus {s : Bool} {k : ℕ} {w : List Bool} (h : w ∈ altPlus s (k + 1)) :
    w.head? = some s := by
  obtain ⟨m, hm, v, -, rfl⟩ := h
  obtain ⟨m, rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  rfl

/-- **`L_{k+1}` is a first letter and a number of blocks** (`review/flat_spectrum.md`, BL2). -/
theorem mem_altPlus_succ_iff {k : ℕ} : ∀ {s : Bool} {w : List Bool},
    w ∈ altPlus s (k + 1) ↔ w.head? = some s ∧ blocks w = k + 1 := by
  induction k with
  | zero =>
    intro s w
    constructor
    · rintro ⟨m, hm, v, hv, rfl⟩
      obtain ⟨m, rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
      rw [Set.mem_singleton_iff.1 hv]
      exact ⟨rfl, by rw [blocks_replicate_append]; rfl⟩
    · rintro ⟨hh, hb⟩
      obtain ⟨m, v, rfl, hv⟩ := exists_replicate_append s w hh
      rw [blocks_replicate_append, ite_eq_right hv] at hb
      obtain rfl : v = [] := blocks_eq_zero.1 (by omega)
      exact ⟨m + 1, Nat.succ_pos m, [], rfl, rfl⟩
  | succ k ih =>
    intro s w
    constructor
    · rintro ⟨m, hm, v, hv, rfl⟩
      obtain ⟨m, rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
      obtain ⟨hvh, hvb⟩ := ih.1 hv
      refine ⟨rfl, ?_⟩
      rw [blocks_replicate_append, hvh, hvb, ite_eq_right (by cases s <;> simp)]
      omega
    · rintro ⟨hh, hb⟩
      obtain ⟨m, v, rfl, hv⟩ := exists_replicate_append s w hh
      rw [blocks_replicate_append, ite_eq_right hv] at hb
      have hvb : blocks v = k + 1 := by omega
      have hvh : v.head? = some (!s) := by
        cases v with
        | nil => simp at hvb
        | cons c v =>
          have hc : c ≠ s := fun h => hv (by rw [h]; rfl)
          rw [List.head?_cons]
          cases c <;> cases s <;> simp_all
      exact ⟨m + 1, Nat.succ_pos m, v, ih.2 ⟨hvh, hvb⟩, rfl⟩

/-- **A word that is not empty is in `L_k` exactly when it starts with `a` and has `k` blocks**
(`review/flat_spectrum.md`, BL2); stated for both first letters. -/
theorem mem_altPlus_iff {s : Bool} {k : ℕ} {w : List Bool} (hw : w ≠ []) :
    w ∈ altPlus s k ↔ w.head? = some s ∧ blocks w = k := by
  cases k with
  | zero =>
    exact ⟨fun h => absurd (Set.mem_singleton_iff.1 h) hw,
      fun h => absurd (blocks_eq_zero.1 h.2) hw⟩
  | succ k => exact mem_altPlus_succ_iff

/-- `abba` is in `L_3`. -/
example : [false, true, true, false] ∈ altPlus false 3 :=
  (mem_altPlus_iff (by simp)).2 ⟨rfl, by decide⟩

example {w : List Bool} (h : w ∈ altPlus false 1) := head?_of_mem_altPlus h

/-- The alternating word of length `n` has `n` blocks. -/
lemma blocks_altList : ∀ (n : ℕ) (s : Bool), blocks (altList s n) = n
  | 0, _ => rfl
  | 1, _ => rfl
  | n + 2, s => by
    rw [altList_succ, blocks_cons, blocks_altList (n + 1) (!s), altList_succ, List.head?_cons,
      ite_eq_right (by cases s <;> simp)]
    omega

/-- The alternating word of odd length ends with its first letter. -/
lemma getLast?_altList_odd (s : Bool) : ∀ i : ℕ, (altList s (2 * i + 1)).getLast? = some s
  | 0 => rfl
  | i + 1 => by
    rw [show 2 * (i + 1) + 1 = 2 * i + 1 + 1 + 1 by omega, altList_succ, altList_not_succ,
      altList_succ, List.getLast?_cons_cons, List.getLast?_cons_cons, ← altList_succ]
    exact getLast?_altList_odd s i

/-- `[u ends with b]` as a number. -/
noncomputable def lastB (u : List Bool) : ℝ := if u.getLast? = some true then 1 else 0

/-- `blocks v − 1 + [v starts with b]`, and `0` for the empty suffix. -/
noncomputable def shift (v : List Bool) : ℝ :=
  if v = [] then 0 else (blocks v : ℝ) - 1 + if v.head? = some true then 1 else 0

/-- `1 − 2 [v starts with b]`, and `0` for the empty suffix. -/
noncomputable def slope (v : List Bool) : ℝ :=
  if v = [] then 0 else 1 - 2 * if v.head? = some true then 1 else 0

/-- **Blocks across a cut, in numbers** (`review/flat_spectrum.md`, BL1): the suffix acts on the
number of blocks of the prefix and on its last letter. -/
theorem cast_blocks_append {u : List Bool} (hu : u ≠ []) (v : List Bool) :
    (blocks (u ++ v) : ℝ) = blocks u + shift v + lastB u * slope v := by
  by_cases hv : v = []
  · simp [hv, shift, slope]
  have h := congrArg (Nat.cast : ℕ → ℝ) (blocks_append u hv)
  obtain ⟨x, hx⟩ : ∃ x, u.getLast? = some x :=
    Option.isSome_iff_exists.1 (by rwa [List.getLast?_isSome])
  obtain ⟨y, v, rfl⟩ := List.exists_cons_of_ne_nil hv
  rw [hx, List.head?_cons] at h
  simp only [shift, slope, lastB, hx, List.head?_cons, hv, ite_false]
  push_cast at h
  cases x <;> cases y <;> simp at h ⊢ <;> linarith

/-- `ab` before `ba`: `abba` has `2 + (2 − 1 + 1) + 1 · (1 − 2) = 3` blocks. -/
example : (blocks ([false, true] ++ [true, false]) : ℝ) =
    blocks [false, true] + shift [true, false] + lastB [false, true] * slope [true, false] :=
  cast_blocks_append (by simp) _

end Transformer.Grokking.FlatSpectrum.BlockCut

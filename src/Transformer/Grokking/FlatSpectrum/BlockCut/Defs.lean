module

public import Transformer.CRASP.Alternating

/-!
# The number of blocks of a word

Source: `review/flat_spectrum.md`, *The block languages across a cut* (added), item BL1, for
the languages `L_k` of Yang, Cadilhac and Chiang, arXiv:2506.16055v3, §2.4, Equation
`eq:altplus`: "the language of strings with alternating *blocks* of `a`'s and `b`'s, starting
with `a`". The alphabet is `Bool`, `false` for `a`, as in `Transformer.CRASP.altPlus`.

`blocks w` is the number of maximal blocks of equal letters of `w`. A letter starts a new block
unless the word after it starts with the same letter (`blocks_cons`). Across a cut the blocks
add, and the two blocks at the cut are one when their letters agree:

    blocks (u ++ v) + [last u = first v] = blocks u + blocks v      (`blocks_append`).

A word that starts with `s` is a run of `s` and a word that does not start with `s`
(`exists_replicate_append`), and the run is one block (`blocks_replicate_append`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.BlockCut

/-- **The number of maximal blocks of equal letters** of a word over `{a, b}`
(`review/flat_spectrum.md`, BL1). -/
def blocks : List Bool → ℕ
  | [] => 0
  | a :: l => (if l.head? = some a then 0 else 1) + blocks l

@[simp] lemma blocks_nil : blocks [] = 0 := rfl

/-- A letter starts a new block unless the word after it starts with the same letter. -/
lemma blocks_cons (a : Bool) (l : List Bool) :
    blocks (a :: l) = (if l.head? = some a then 0 else 1) + blocks l := rfl

@[simp] lemma blocks_singleton (a : Bool) : blocks [a] = 1 := rfl

/-- `abba` has three blocks. -/
example : blocks [false, true, true, false] = 3 := by decide

/-- A word that is not empty has a block. -/
lemma blocks_pos : ∀ {l : List Bool}, l ≠ [] → 0 < blocks l
  | [], h => absurd rfl h
  | [_], _ => Nat.one_pos
  | a :: b :: l, _ => by
    rw [blocks_cons]
    exact Nat.add_pos_right _ (blocks_pos (List.cons_ne_nil b l))

example : 0 < blocks [true] := blocks_pos (by simp)

/-- Only the empty word has no block. -/
lemma blocks_eq_zero {l : List Bool} : blocks l = 0 ↔ l = [] :=
  ⟨fun h => by_contra fun hl => (blocks_pos hl).ne' h, fun h => h ▸ rfl⟩

/-- **Blocks add across a cut** (`review/flat_spectrum.md`, BL1): the last block of `u` and the
first block of `v` are one block of `u ++ v` exactly when their letters agree. -/
theorem blocks_append (u : List Bool) {v : List Bool} (hv : v ≠ []) :
    blocks (u ++ v) + (if u.getLast? = v.head? then 1 else 0) = blocks u + blocks v := by
  obtain ⟨y, v, rfl⟩ := List.exists_cons_of_ne_nil hv
  induction u with
  | nil => simp
  | cons a u ih =>
    cases u with
    | nil =>
      rw [List.singleton_append, blocks_cons a (y :: v), blocks_singleton]
      by_cases h : y = a
      · simp only [h, List.head?_cons, List.getLast?_singleton, ite_true]
        omega
      · simp [h, eq_comm]
    | cons b u =>
      rw [List.cons_append, blocks_cons a, blocks_cons a (b :: u), List.getLast?_cons_cons]
      rw [List.cons_append] at ih ⊢
      simp only [List.head?_cons] at ih ⊢
      exact (Nat.add_assoc _ _ _).trans ((congrArg _ ih).trans (Nat.add_assoc _ _ _).symm)

/-- `ab` and `ba`: the cut joins no blocks; `ab` and `ba` reversed, `ab · b`, joins two. -/
example : blocks ([false, true] ++ [true]) + 1 = blocks [false, true] + blocks [true] := by
  decide

example := blocks_append [false, true] (v := [true, false]) (by simp)

/-- A run of `s` before `v` is one more block, unless `v` starts with `s`. -/
lemma blocks_replicate_append (s : Bool) (v : List Bool) :
    ∀ m, blocks (List.replicate (m + 1) s ++ v) = (if v.head? = some s then 0 else 1) + blocks v
  | 0 => rfl
  | m + 1 => by
    rw [List.replicate_succ, List.cons_append, blocks_cons, blocks_replicate_append s v m,
      List.replicate_succ, List.cons_append, List.head?_cons, ite_eq_left rfl, Nat.zero_add]

example : blocks (List.replicate 3 false ++ [true]) = 2 := by
  rw [blocks_replicate_append]
  rfl

/-- A word that starts with `s` is a run of `s` and a word that does not start with `s`. -/
lemma exists_replicate_append (s : Bool) :
    ∀ w : List Bool, w.head? = some s →
      ∃ m v, w = List.replicate (m + 1) s ++ v ∧ v.head? ≠ some s
  | [], h => by simp at h
  | a :: w, h => by
    obtain rfl : a = s := by simpa using h
    by_cases hw : w.head? = some a
    · obtain ⟨m, v, rfl, hv⟩ := exists_replicate_append a w hw
      exact ⟨m + 1, v, by simp [List.replicate_succ], hv⟩
    · exact ⟨0, w, by simp, hw⟩

example := exists_replicate_append false [false, false, true] rfl

end Transformer.Grokking.FlatSpectrum.BlockCut

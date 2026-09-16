/-
# The blocks of the words of `L_k`

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §2.4, Equation `eq:altplus`.

What the depth lower bounds read off a word of `L_k`: the number of its blocks
is determined by the word (`eq_of_mem_altPlus`), and so is its last letter
(`getElem?_of_mem_altPlus`); a nonempty prefix keeps the blocks before the cut
(`take_mem_altPlus`); letters equal to the last one lengthen the last block,
and a word starting with the other letter adds its blocks
(`append_mem_altPlus`); and an alternating subsequence starting with the letter
of the second block is shorter than the number of blocks, while every shorter
one occurs (`altList_not_sublist_iff`).
-/

import Transformer.CRASP.Alternating

namespace Transformer
namespace CRASP

/-- The alternating words starting with a fixed symbol are subsequences of the
longer ones (§2.4). -/
theorem altList_sublist_of_le (s : Bool) {m m' : ℕ} (h : m ≤ m') :
    (altList s m).Sublist (altList s m') := by
  induction h with
  | refl => exact List.Sublist.refl _
  | step _ ih => exact ih.trans (altList_sublist_succ s _)

/-- The hypothesis of `altList_sublist_of_le` is satisfiable: `1 ≤ 2`. -/
example : 1 ≤ 2 := by decide

/-- **A word of `L_j` beginning with `s` has an alternating subsequence of
length `m` starting with `!s` exactly when `m < j`**: such a subsequence can
use every block but the first (§2.4, the count behind
`lem:piecewise_testable`). -/
theorem altList_not_sublist_iff {s : Bool} {j : ℕ} (hj : 0 < j) {w : List Bool}
    (hw : w ∈ altPlus s j) (m : ℕ) : (altList (!s) m).Sublist w ↔ m < j := by
  constructor
  · intro h
    by_contra hm
    exact not_altList_not_sublist_of_mem_altPlus j hj s w hw
      ((altList_sublist_of_le (!s) (not_lt.1 hm)).trans h)
  · intro hm
    obtain ⟨i, rfl⟩ : ∃ i, j = i + 1 := ⟨j - 1, by omega⟩
    obtain ⟨p, -, v, hv, rfl⟩ := hw
    exact (altList_sublist_of_le (!s) (by omega)).trans
      ((altList_sublist_of_mem_altPlus i (!s) v hv).trans (List.sublist_append_right _ _))

/-- **A word has one number of blocks** (§2.4). -/
theorem eq_of_mem_altPlus {s : Bool} {j n : ℕ} {w : List Bool} (hj : w ∈ altPlus s j)
    (hn : w ∈ altPlus s n) : j = n := by
  by_contra hne
  rcases Nat.lt_or_gt_of_ne hne with h | h
  · exact not_altList_succ_sublist j s w hj
      ((altList_sublist_of_le s h).trans (altList_sublist_of_mem_altPlus n s w hn))
  · exact not_altList_succ_sublist n s w hn
      ((altList_sublist_of_le s h).trans (altList_sublist_of_mem_altPlus j s w hj))

/-- The hypotheses of `altList_not_sublist_iff` and `eq_of_mem_altPlus` are
satisfiable: `ab` has two alternating blocks starting with `a`. -/
example : 0 < 2 ∧ ([false, true] : List Bool) ∈ altPlus false 2 :=
  ⟨by omega, 1, Nat.one_pos, [true], ⟨1, Nat.one_pos, [], rfl, rfl⟩, rfl⟩

/-- **A nonempty prefix of a word of `L_n` lies in `L_j` for some
`0 < j ≤ n`**: a cut inside a block keeps the blocks before it (§2.4). -/
theorem take_mem_altPlus (n : ℕ) :
    ∀ (s : Bool) (w : List Bool), w ∈ altPlus s n →
      ∀ i, 0 < i → i ≤ w.length → ∃ j, 0 < j ∧ j ≤ n ∧ w.take i ∈ altPlus s j := by
  induction n with
  | zero =>
      intro s w hw i hi hiw
      rw [Set.mem_singleton_iff.1 hw, List.length_nil] at hiw
      omega
  | succ n ih =>
      rintro s w ⟨m, hm, v, hv, rfl⟩ i hi hiw
      rw [List.length_append, List.length_replicate] at hiw
      by_cases him : i ≤ m
      · refine ⟨1, Nat.one_pos, by omega, ?_⟩
        rw [List.take_append_of_le_length (by rw [List.length_replicate]; exact him),
          List.take_replicate, altPlus_one]
        exact ⟨min i m, by omega, rfl⟩
      · obtain ⟨j, hj, hjn, hv'⟩ := ih (!s) v hv (i - m) (by omega) (by omega)
        refine ⟨j + 1, j.succ_pos, by omega, m, hm, _, hv', ?_⟩
        rw [List.take_append, List.length_replicate, List.take_of_length_le (by simp; omega)]

/-- **The last symbol of a word of `L_{j+1}` beginning with `s`** is `s` when
`j` is even and `!s` when `j` is odd: the blocks alternate (§2.4). -/
theorem getElem?_of_mem_altPlus (j : ℕ) :
    ∀ (s : Bool) (w : List Bool), w ∈ altPlus s (j + 1) →
      w[w.length - 1]? = some (s ^^ decide (j % 2 = 1)) := by
  induction j with
  | zero =>
      rintro s w ⟨m, hm, v, hv, rfl⟩
      rw [Set.mem_singleton_iff.1 hv, List.append_nil,
        List.getElem?_replicate_of_lt (by rw [List.length_replicate]; omega)]
      simp
  | succ j ih =>
      rintro s w ⟨m, hm, v, hv, rfl⟩
      have hlen : 0 < v.length := by
        obtain ⟨p, hp, _, _, rfl⟩ := hv
        rw [List.length_append, List.length_replicate]
        omega
      rw [List.length_append, List.length_replicate,
        List.getElem?_append_right (by rw [List.length_replicate]; omega), List.length_replicate,
        show m + v.length - 1 - m = v.length - 1 by omega, ih (!s) v hv]
      rcases Nat.mod_two_eq_zero_or_one j with h | h <;> cases s <;> simp [h, Nat.add_mod]

/-- The hypothesis of `take_mem_altPlus` and `getElem?_of_mem_altPlus` is
satisfiable: `ab` has two alternating blocks starting with `a`, and its first
symbol is a prefix. -/
example : ([false, true] : List Bool) ∈ altPlus false (1 + 1) ∧ 0 < 1 ∧
    1 ≤ ([false, true] : List Bool).length :=
  ⟨⟨1, Nat.one_pos, [true], ⟨1, Nat.one_pos, [], rfl, rfl⟩, rfl⟩, Nat.one_pos, by decide⟩

/-- **Blocks appended to a word of `L_{j+1}`.**  Letters equal to the last one
lengthen its last block, and a word of `n` blocks starting with the other letter
adds `n` blocks (§2.4, Equation `eq:altplus`). -/
theorem append_mem_altPlus (j : ℕ) :
    ∀ (t : Bool) (w : List Bool), w ∈ altPlus t (j + 1) →
      ∀ p n (v : List Bool), v ∈ altPlus (!(t ^^ decide (j % 2 = 1))) n →
        w ++ List.replicate p (t ^^ decide (j % 2 = 1)) ++ v ∈ altPlus t (j + 1 + n) := by
  induction j with
  | zero =>
      rintro t w ⟨m, hm, v₀, hv₀, rfl⟩ p n v hv
      rw [Set.mem_singleton_iff.1 hv₀, Nat.zero_add, Nat.add_comm]
      simp only [Nat.zero_mod, Nat.zero_ne_one, decide_false, Bool.xor_false] at hv ⊢
      exact ⟨m + p, by omega, v, hv, by rw [List.append_nil, List.replicate_add]⟩
  | succ j ih =>
      rintro t w ⟨m, hm, w', hw', rfl⟩ p n v hv
      have e : (t ^^ decide ((j + 1) % 2 = 1)) = ((!t) ^^ decide (j % 2 = 1)) := by
        rcases Nat.mod_two_eq_zero_or_one j with h | h <;> cases t <;> simp [h, Nat.add_mod]
      rw [e] at hv ⊢
      rw [show j + 1 + 1 + n = j + 1 + n + 1 by omega]
      exact ⟨m, hm, _, ih (!t) w' hw' p n v hv, by simp only [List.append_assoc]⟩

/-- The hypotheses of `append_mem_altPlus` are satisfiable: `a` has one block,
and `b` has one block starting with the other letter. -/
example : ([false] : List Bool) ∈ altPlus false (0 + 1) ∧
    ([true] : List Bool) ∈ altPlus (!(false ^^ decide (0 % 2 = 1))) 1 :=
  ⟨⟨1, Nat.one_pos, [], rfl, rfl⟩, ⟨1, Nat.one_pos, [], rfl, rfl⟩⟩

/-- `c̄cc̄` and `c̄c̄cc̄` have three blocks (§2.4). -/
theorem mem_altPlus_three (c : Bool) :
    [!c, c, !c] ∈ altPlus (!c) 3 ∧ [!c, !c, c, !c] ∈ altPlus (!c) 3 := by
  cases c
  · exact ⟨⟨1, Nat.one_pos, [false, true], ⟨1, Nat.one_pos, [true], ⟨1, Nat.one_pos, [], rfl, rfl⟩,
      rfl⟩, rfl⟩, ⟨2, by decide, [false, true], ⟨1, Nat.one_pos, [true],
        ⟨1, Nat.one_pos, [], rfl, rfl⟩, rfl⟩, rfl⟩⟩
  · exact ⟨⟨1, Nat.one_pos, [true, false], ⟨1, Nat.one_pos, [false], ⟨1, Nat.one_pos, [], rfl, rfl⟩,
      rfl⟩, rfl⟩, ⟨2, by decide, [true, false], ⟨1, Nat.one_pos, [false],
        ⟨1, Nat.one_pos, [], rfl, rfl⟩, rfl⟩, rfl⟩⟩

end CRASP
end Transformer

/-
# The blocks of the words of `L_k`

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §2.4, Equation `eq:altplus`.

What the depth lower bounds read off a word of `L_k`: the number of its blocks
is determined by the word (`eq_of_mem_altPlus`), and so is its last letter
(`getElem?_of_mem_altPlus`); a nonempty prefix keeps the blocks before the cut
(`take_mem_altPlus`); and an alternating subsequence starting with the letter
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

end CRASP
end Transformer

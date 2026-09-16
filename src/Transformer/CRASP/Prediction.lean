/-
# The next-token prediction problem

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §5.1: `def:prediction_task` and `cor:prediction_task_depth`, with
the proof in Appendix C.4.

A formula solves the prediction problem for `L_k` when it recognizes, among the
prefixes of the strings of `L_k`, exactly those in `L_k`.  That is easier than
recognition by one level of depth and two blocks: a prefix of a string of
`L_{k+3}` starts with `a` and has at most `k + 3` blocks, so it is in `L_{k+3}`
when it has at least `k + 2` blocks — the alternating word of length `k + 1`
starting with `b` is a subsequence — and its last symbol is that of block
`k + 3`.  This is the paper's formula `φ_{B_{k+1}} ∧ Q_a` (`k` even) or
`φ_{B_{k+1}} ∧ Q_b` (`k` odd), `predictAltPlus` below.

The negative half is derived in the paper from `lem:cropping_oneway` and
`lem:reduction`, both false as stated (`Transformer.CRASP.Depth`), and is left
open.
-/

import Transformer.CRASP.Depth

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- **Definition `def:prediction_task`.**  A `TL[◁#]` formula solves the
next-token prediction problem for `L` when, on every prefix of every string of
`L`, it says exactly whether that prefix is itself in `L`.  Only prefixes of
strings of `L` are considered, which is what separates prediction from
recognition. -/
def SolvesPrediction [DecidableEq σ] (φ : Form σ) (L : Set (List σ)) : Prop :=
  ∀ w ∈ L, ∀ i, 1 ≤ i → i ≤ w.length → (φ.models (w.take i) ↔ w.take i ∈ L)

/-- The prediction problem is not vacuous: the formula `⊤` — written `¬(1 < 1)`
— solves the prediction problem for `Σ*`, in which every prefix lies. -/
example : SolvesPrediction (σ := Bool) (.neg (.lt .one .one)) Set.univ :=
  fun _ _ _ _ _ => by simp [Form.models, Form.sat, Term.val]

/-! ## Prefixes of alternating words -/

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

/-! ## The prediction problem for `L_{k+3}` at depth `k + 1` -/

/-- The formula of `cor:prediction_task_depth`: `φ_{B_{k+1}} ∧ Q_a` for even
`k`, `φ_{B_{k+1}} ∧ Q_b` for odd `k`, with `φ_{B_{k+1}}` the test of
`lem:piecewise_testable_depth` for the alternating word of length `k + 1`
starting with `b` (Appendix C.4). -/
def predictAltPlus (k : ℕ) : Form Bool :=
  .and (subseqAt (altList true (k + 1)).reverse) (.sym (decide (k % 2 = 1)))

/-- `predictAltPlus k` is a past-only formula of depth `k + 1` without PNPs
(Appendix C.4). -/
theorem predictAltPlus_mem (k : ℕ) : predictAltPlus k ∈ TLCl Bool (k + 1) :=
  ⟨by simp [predictAltPlus, Form.past], by simp [predictAltPlus, Form.pnpFree],
    by simp [predictAltPlus, Form.depth]⟩

/-- **`predictAltPlus k` solves the prediction problem for `L_{k+3}`**
(arXiv:2506.16055v3, Appendix C.4, the first half of the proof of
`cor:prediction_task_depth`). -/
theorem solvesPrediction_predictAltPlus (k : ℕ) :
    SolvesPrediction (predictAltPlus k) (altPlus false (k + 3)) := by
  intro w hw i hi hiw
  obtain ⟨j, hj, hjn, hu⟩ := take_mem_altPlus (k + 3) false w hw i hi hiw
  obtain ⟨j, rfl⟩ : ∃ j', j = j' + 1 := ⟨j - 1, by omega⟩
  have hsub := altList_not_sublist_iff hj hu (k + 1)
  rw [Bool.not_false] at hsub
  rw [Form.models, predictAltPlus, Form.sat, Bool.and_eq_true, sat_subseqAt,
    List.reverse_reverse, List.take_length, Form.sat, getElem?_of_mem_altPlus j false _ hu, hsub]
  simp only [decide_eq_true_eq, Bool.false_xor, Option.some.injEq, decide_eq_decide]
  constructor
  · rintro ⟨h1, h2⟩
    obtain rfl : j = k + 2 := by
      rcases Nat.mod_two_eq_zero_or_one k with hk | hk <;>
        rcases Nat.mod_two_eq_zero_or_one j with hj' | hj' <;> simp [hk, hj'] at h2 <;> omega
    exact hu
  · intro h
    obtain rfl : j = k + 2 := by have := eq_of_mem_altPlus hu h; omega
    exact ⟨by omega, by omega⟩

/-- **Corollary `cor:prediction_task_depth`.**  A depth-`(k+1)` `TL[◁#]`
formula solves the next-token prediction problem for `L_{k+3}`, and no
depth-`k` formula does.

The positive half is `solvesPrediction_predictAltPlus`.  The negative half is
the one the paper derives from `lem:cropping_oneway` and `lem:reduction`, both
false as stated, and only it is left open. -/
theorem prediction_task_depth (k : ℕ) (hk : 0 < k) :
    (∃ φ ∈ TLCl Bool (k + 1), SolvesPrediction φ (altPlus false (k + 3))) ∧
      ∀ φ ∈ TLCl Bool k, ¬ SolvesPrediction φ (altPlus false (k + 3)) :=
  ⟨⟨predictAltPlus k, predictAltPlus_mem k, solvesPrediction_predictAltPlus k⟩, sorry⟩

end CRASP
end Transformer

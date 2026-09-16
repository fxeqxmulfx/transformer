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

import Transformer.CRASP.Blocks
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

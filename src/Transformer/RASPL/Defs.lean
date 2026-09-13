/-
# RASP-L: the causal core

Zhou et al. — arXiv:2310.16028v1, "What Algorithms can Transformers Learn?  A
Study in Length Generalization" (ICLR 2024), Appendix C (Listing 3, "the
RASP-L core") and §5.

RASP-L is RASP restricted so that every program it admits is one a
transformer can plausibly *learn*, not merely represent.  Three of its
restrictions are typing disciplines the paper itself calls open —
"fully formalizing the intended index restrictions, and proving they hold
under the appropriate type system, is an open question for future work"
(Appendix C) — so what is formalized here is the part that is not open: the
core operations themselves, which differ from the RASP of `Transformer.RASP`
in being *causal*.

Their `select` builds "a causal binary attention matrix": `A[i][j]` is set
only for `j ≤ i`.  Note also that the reference implementation writes
`A[i, j] = pred(k[j], q[i])`, keys against queries — the same convention
`Transformer.RASP` had to recover from arXiv:2106.06981's worked example
rather than from its prose.
-/

import Transformer.RASP.Basic

namespace Transformer
namespace RASPL

open RASP

variable {n : ℕ} {α β γ : Type*}

/-- `select(k, q, pred)`: the causal selector, which attends only backwards
(Listing 3, `select`). -/
def select (k : Seq n α) (q : Seq n β) (p : α → β → Bool) : Selector n :=
  fun i j => decide ((j : ℕ) ≤ (i : ℕ)) && p (k j) (q i)

/-- `full(x, c)`: the constant sequence. -/
def full (n : ℕ) (c : α) : Seq n α := fun _ => c

/-- `indices(x)`: the position of each token, at RASP-L's index type, which
is `ℕ` and not `ℝ`. -/
def idx (n : ℕ) : Seq n ℕ := fun i => (i : ℕ)

/-- `tok_map(x, func)`: a tokenwise map. -/
def tokMap (f : α → β) (x : Seq n α) : Seq n β := fun i => f (x i)

/-- `seq_map(x, y, func)`: a tokenwise map of two sequences. -/
def seqMap (f : α → β → γ) (x : Seq n α) (y : Seq n β) : Seq n γ := fun i => f (x i) (y i)

/-- `sel_width(A)`: how many keys each query selects (Listing 3). -/
def selWidth (A : Selector n) : Seq n ℕ := fun i => (selected A i).card

/-- `aggr_mean(A, v, default)`: the mean of the selected values, or the
default where nothing is selected. -/
noncomputable def aggrMean (A : Selector n) (v : Seq n ℝ) (d : ℝ) : Seq n ℝ :=
  aggregate A v d

/-- `aggr_max(A, v, default)`: the largest selected value, or the default
where nothing is selected. -/
noncomputable def aggrMax (A : Selector n) (v : Seq n ℝ) (d : ℝ) : Seq n ℝ :=
  fun i => if h : (selected A i).Nonempty then (selected A i).sup' h v else d

/-- `kqv(k, q, v, pred, default)`: one attention layer, mean reduction. -/
noncomputable def kqv (k : Seq n α) (q : Seq n β) (v : Seq n ℝ) (p : α → β → Bool)
    (d : ℝ) : Seq n ℝ :=
  aggrMean (select k q p) v d

/-- `kqv(..., reduction='max')`: one attention layer, max reduction. -/
noncomputable def kqvMax (k : Seq n α) (q : Seq n β) (v : Seq n ℝ) (p : α → β → Bool)
    (d : ℝ) : Seq n ℝ :=
  aggrMax (select k q p) v d

@[simp] lemma mem_selected_select (k : Seq n α) (q : Seq n β) (p : α → β → Bool)
    (i j : Fin n) :
    j ∈ selected (select k q p) i ↔ (j : ℕ) ≤ (i : ℕ) ∧ p (k j) (q i) = true := by
  simp [select, RASP.mem_selected]

/-- **Every RASP-L selector is causal**: a query never attends to a later
key (Listing 3, "constructs a causal binary attention matrix"). -/
theorem le_of_mem_selected_select {k : Seq n α} {q : Seq n β} {p : α → β → Bool}
    {i j : Fin n} (h : j ∈ selected (select k q p) i) : (j : ℕ) ≤ (i : ℕ) :=
  ((mem_selected_select k q p i j).1 h).1

/-- A query always selects itself when the predicate holds there, so a
selector whose predicate is reflexive at `i` is never empty. -/
theorem self_mem_selected_select {k : Seq n α} {q : Seq n β} {p : α → β → Bool}
    {i : Fin n} (h : p (k i) (q i) = true) : i ∈ selected (select k q p) i :=
  (mem_selected_select k q p i i).2 ⟨le_rfl, h⟩

/-- Selector width counts exactly the prior positions the predicate holds
at. -/
theorem selWidth_eq (k : Seq n α) (q : Seq n β) (p : α → β → Bool) (i : Fin n) :
    selWidth (select k q p) i
      = (Finset.univ.filter fun j : Fin n => (j : ℕ) ≤ (i : ℕ) ∧ p (k j) (q i) = true).card := by
  refine congrArg Finset.card (Finset.ext fun j => ?_)
  rw [mem_selected_select]
  simp

/-- `aggr_max` at a nonempty row is the maximum of the selected values. -/
theorem aggrMax_of_mem {A : Selector n} {v : Seq n ℝ} {d : ℝ} {i j : Fin n}
    (hj : j ∈ selected A i) : aggrMax A v d i = (selected A i).sup' ⟨j, hj⟩ v := by
  rw [aggrMax, dif_pos ⟨j, hj⟩]

/-- The causal core is not vacuous: on three positions, `select` with `<`
selects strictly earlier positions only, and its width is the index. -/
example : selWidth (select (idx 3) (idx 3) (fun a b => decide (a < b))) = ![0, 1, 2] := by
  funext i
  fin_cases i <;> decide

end RASPL
end Transformer

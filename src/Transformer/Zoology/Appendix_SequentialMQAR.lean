/-
# Correctness of sequential associative recall

Arora et al., arXiv:2312.04927v1, Appendix Proposition
`prop: seq-gen-ar`, Algorithm `algo: seq-gen-ar`. The algorithm queries
the key-value table before inserting the current pair. This file models
the table as a finite function updated in sequence order and proves its
functional correctness for distinct keys. The proposition's balanced-tree
runtime bound is not represented by this function-level result.
-/

import Transformer.Zoology.Section4_PairedRecall

namespace Transformer.Zoology

/-- Insert each indexed key-value pair into a finite associative table,
processing the index list from left to right. Source: Appendix Algorithm
`algo: seq-gen-ar`, `insert` step. -/
def sequentialTable {n c : ℕ} (x : MQARInstance n c) :
    List (Fin n) → (Fin c → Option (Fin c)) → Fin c → Option (Fin c)
  | [], table => table
  | j :: rest, table =>
      sequentialTable x rest
        (fun q => if q = x.key j then some (x.value j) else table q)

/-- Indices inserted before the query at position `i`.
Source: Appendix Algorithm `algo: seq-gen-ar`, query-before-insert order. -/
def priorIndices {n : ℕ} (i : Fin n) : List (Fin n) :=
  (List.finRange n).filter fun j => decide (j < i)

/-- The prior-index list contains exactly the earlier positions.
Source: Appendix Algorithm `algo: seq-gen-ar`. -/
theorem mem_priorIndices_iff {n : ℕ} (i j : Fin n) :
    j ∈ priorIndices i ↔ j < i := by
  simp [priorIndices]

/-- The sequential algorithm's answer at a query, including `none` when
no prior key matches. Source: Appendix Algorithm `algo: seq-gen-ar`. -/
def sequentialAnswer {n c : ℕ} (x : MQARInstance n c)
    (i : Fin n) : Option (Fin c) :=
  sequentialTable x (priorIndices i) (fun _ => none) (x.query i)

/-- If none of the inserted keys equals `q`, the table preserves its
previous answer for `q`. Source: Appendix Algorithm `algo: seq-gen-ar`. -/
theorem sequentialTable_no_match {n c : ℕ} (x : MQARInstance n c)
    (js : List (Fin n)) (table : Fin c → Option (Fin c)) (q : Fin c)
    (h : ∀ j ∈ js, x.key j ≠ q) :
    sequentialTable x js table q = table q := by
  induction js generalizing table with
  | nil => rfl
  | cons j rest ih =>
      have hj : x.key j ≠ q := h j (by simp)
      have hrest : ∀ k ∈ rest, x.key k ≠ q := by
        intro k hk
        exact h k (by simp [hk])
      simp only [sequentialTable]
      rw [ih _ hrest]
      simp [Ne.symm hj]

/-- With distinct keys, the inserted list returns the value of its sole
matching key, independently of the previous table state. Source: Appendix
Algorithm `algo: seq-gen-ar`, lookup after prior insertions. -/
theorem sequentialTable_unique_match {n c : ℕ} (x : MQARInstance n c)
    (hx : UniqueKeys x) (js : List (Fin n)) (hnd : js.Nodup)
    (table : Fin c → Option (Fin c)) (j : Fin n) (hj : j ∈ js)
    (q : Fin c) (hkey : x.key j = q) :
    sequentialTable x js table q = some (x.value j) := by
  induction js generalizing table with
  | nil => simp at hj
  | cons k rest ih =>
      obtain ⟨hknot, hrestnd⟩ := List.nodup_cons.mp hnd
      rcases List.mem_cons.mp hj with hkj | hjrest
      · subst j
        have hnomatch : ∀ m ∈ rest, x.key m ≠ q := by
          intro m hm hmq
          have hmk : m = k := hx (hmq.trans hkey.symm)
          exact hknot (hmk ▸ hm)
        simp only [sequentialTable]
        rw [sequentialTable_no_match x rest _ q hnomatch]
        simp [hkey]
      · simp only [sequentialTable]
        exact ih hrestnd _ hjrest

/-- The prior-index list contains no duplicate pair positions.
Source: Appendix Algorithm `algo: seq-gen-ar`. -/
theorem priorIndices_nodup {n : ℕ} (i : Fin n) :
    (priorIndices i).Nodup := by
  exact List.Nodup.filter _ (List.nodup_finRange n)

/-- Sequential lookup returns the earlier matching value, provided the
keys are distinct. Source: Appendix Proposition `prop: seq-gen-ar`,
functional correctness component. -/
theorem sequentialAnswer_match {n c : ℕ} (x : MQARInstance n c)
    (hx : UniqueKeys x) (i j : Fin n) (hj : j < i)
    (hkey : x.key j = x.query i) :
    sequentialAnswer x i = some (x.value j) := by
  exact sequentialTable_unique_match x hx (priorIndices i)
    (priorIndices_nodup i) (fun _ => none) j
    ((mem_priorIndices_iff i j).mpr hj) (x.query i) hkey

/-- With no earlier matching key, sequential lookup returns `none`.
Source: Appendix Proposition `prop: seq-gen-ar`, no-match case. -/
theorem sequentialAnswer_no_match {n c : ℕ} (x : MQARInstance n c)
    (i : Fin n) (h : ∀ j : Fin n, j < i → x.key j ≠ x.query i) :
    sequentialAnswer x i = none := by
  apply sequentialTable_no_match
  intro j hj
  exact h j ((mem_priorIndices_iff i j).mp hj)

/-- The option-valued algorithm solves all queries: a result exists exactly
when an earlier key matches, and it is the associated value. Source:
Appendix Proposition `prop: seq-gen-ar`, functional correctness component. -/
theorem sequentialAnswer_iff_priorAnswer {n c : ℕ}
    (x : MQARInstance n c) (hx : UniqueKeys x)
    (i : Fin n) (v : Fin c) :
    sequentialAnswer x i = some v ↔ PriorAnswer x i v := by
  constructor
  · intro h
    by_cases hm : ∃ j : Fin n, j < i ∧ x.key j = x.query i
    · obtain ⟨j, hj, hkey⟩ := hm
      rw [sequentialAnswer_match x hx i j hj hkey] at h
      have hv : x.value j = v := Option.some.inj h
      exact ⟨j, hj, hkey, hv⟩
    · have hnone : ∀ j : Fin n, j < i → x.key j ≠ x.query i := by
        intro j hj hkey
        exact hm ⟨j, hj, hkey⟩
      rw [sequentialAnswer_no_match x i hnone] at h
      cases h
  · rintro ⟨j, hj, hkey, rfl⟩
    exact sequentialAnswer_match x hx i j hj hkey

/-- The distinct-key hypothesis is satisfiable with a genuine prior query.
Source: Appendix Algorithm `algo: seq-gen-ar`. -/
example : ∃ x : MQARInstance 2 2,
    UniqueKeys x ∧ sequentialAnswer x 1 = some 0 := by
  let x : MQARInstance 2 2 := {
    key := id
    value := id
    query := fun _ => 0
  }
  refine ⟨x, ?_, ?_⟩
  · intro i j h
    exact h
  · exact sequentialAnswer_match x (by
      intro i j h
      exact h) 1 0 (by decide) rfl

end Transformer.Zoology

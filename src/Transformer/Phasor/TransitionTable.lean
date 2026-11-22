module

public import Transformer.Phasor.TransitionValues

/-!
# An independent hidden value table for the number of completed changes

Source: the transition-table trial in `experiments/phasor_value_transport/README.md`.
A learned vector indexed by the modular change count is added to the full
existing hidden read before RMS. The table may learn any compatible vector
on each bucket. Its transpose groups every contributing row by that bucket.
-/

@[expose] public section

namespace Transformer.Phasor

open scoped RealInnerProductSpace
open Transformer

variable {α β ι : Type*} {p D : ℕ}

/-- At position `i`, the last completed edge ends at `i-1`. The first
two positions read bucket zero. Longer prompts can wrap around the table. -/
def transitionBucket [DecidableEq α] (p : ℕ) [NeZero p] (xs : ℕ → α) (i : ℕ) : Fin p :=
  Fin.ofNat p (transitionCount xs (i-1))

/-- An injective change of token labels, including one into a larger
alphabet, preserves every bucket for any fixed table width. -/
theorem transitionBucket_map [DecidableEq α] [DecidableEq β] (p : ℕ) [NeZero p]
    (f : α → β) (hf : Function.Injective f) (xs : ℕ → α) (i : ℕ) :
    transitionBucket p (fun j => f (xs j)) i = transitionBucket p xs i := by
  rw [transitionBucket, transitionBucket, transitionCount_map f hf]

/-- All vectors requested at finitely many prefixes can be represented if
and only if prefixes sharing a count residue request the same vector.
The previous hidden read is arbitrary at every prefix. -/
theorem exists_transition_table_iff [DecidableEq α] [NeZero p] (active : Finset ι)
    (xs : ι → ℕ → α) (pos : ι → ℕ) (base desired : ι → EucSpace D) :
    (∃ table : Fin p → EucSpace D, ∀ i ∈ active,
      base i + table (transitionBucket p (xs i) (pos i)) = desired i) ↔
    ∀ i ∈ active, ∀ j ∈ active,
      transitionBucket p (xs i) (pos i) = transitionBucket p (xs j) (pos j) →
      desired i - base i = desired j - base j := by
  classical
  let key := fun i => transitionBucket p (xs i) (pos i)
  constructor
  · rintro ⟨table, fits⟩ i hi j hj same
    rw [← fits i hi, ← fits j hj, same]
    abel
  · intro consistent
    let table (b : Fin p) := if h : ∃ j ∈ active, key j = b then
      desired h.choose - base h.choose else 0
    refine ⟨table, ?_⟩
    intro i hi
    have hit : ∃ j ∈ active, key j = key i := ⟨i, hi, rfl⟩
    have selected := consistent hit.choose hit.choose_spec.1 i hi hit.choose_spec.2
    change base i + table (key i) = desired i
    rw [show table (key i) = desired i - base i by simpa [table, hit] using selected]
    abel

/-- Zero initialization embeds every existing hidden read on every stream,
including all positions of a supervised sequence. -/
theorem transitionTable_zero [DecidableEq α] [NeZero p] (xs : ℕ → α) (pos : ℕ)
    (base : EucSpace D) : base + (0 : Fin p → EucSpace D) (transitionBucket p xs pos) = base := by
  simp

/-- The linear variation of the added table pairs exactly with the hidden
gradient summed over each bucket, including repeated reads after wraparound. -/
theorem transitionTable_adjoint [DecidableEq α] [NeZero p] (active : Finset ι)
    (xs : ι → ℕ → α) (pos : ι → ℕ) (grad : ι → EucSpace D) (change : Fin p → EucSpace D) :
    ∑ i ∈ active, ⟪grad i, change (transitionBucket p (xs i) (pos i))⟫ =
      ∑ b, ⟪∑ i ∈ active with transitionBucket p (xs i) (pos i) = b, grad i, change b⟫ := by
  let key := fun i => transitionBucket p (xs i) (pos i)
  rw [← Finset.sum_fiberwise active key (fun i => ⟪grad i, change (key i)⟫)]
  apply Finset.sum_congr rfl
  intro b _
  rw [sum_inner]
  apply Finset.sum_congr rfl
  intro i hi
  rw [(Finset.mem_filter.mp hi).2]

private def samples (row : Fin 2) (i : ℕ) : Fin 2 :=
  if row = 0 then (if i = 0 ∨ i = 2 then 0 else 1)
  else (if i < 3 then 0 else 1)

/-- Two equal-length prompts end in the same token pair. Four buckets
distinguish the earlier one and three changes, while two buckets alias them. -/
example : transitionBucket 4 (samples 0) 5 = 3 ∧ transitionBucket 4 (samples 1) 5 = 1 ∧
    transitionBucket 2 (samples 0) 5 = transitionBucket 2 (samples 1) 5 := by decide

example : transitionBucket 4 (fun i => (samples 0 i).val) 5 = transitionBucket 4 (samples 0) 5 :=
  transitionBucket_map _ Fin.val Fin.val_injective _ _

example (base desired : Fin 2 → EucSpace 8) :
    ∃ table : Fin 4 → EucSpace 8, ∀ i ∈ Finset.univ,
      base i + table (transitionBucket 4 (samples i) 5) = desired i := by
  apply (exists_transition_table_iff _ samples (fun _ => 5) base desired).mpr
  intro i _ j _ eq
  have distinct : ∀ i j : Fin 2,
      transitionBucket 4 (samples i) 5 = transitionBucket 4 (samples j) 5 → i = j := by decide
  rw [distinct i j eq]

end Transformer.Phasor

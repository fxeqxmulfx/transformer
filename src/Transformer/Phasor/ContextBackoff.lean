module

public import Transformer.Phasor.SingleWrite
public import Transformer.Phasor.SuccessorAgreement

/-!
# Causal backoff from a two-token context to an ordinary successor

Source: the context-backoff trial in `experiments/phasor_value_transport/README.md`.
When there is no resolved closed relation, the latest completed write for
the last two query tokens supplies the address. A missing context falls
back to the ordinary successor. Every contextual proposal is a witnessed
successor of the current token, so consistent existing bindings are preserved.
-/

@[expose] public section

namespace Transformer.Phasor

variable {α β : Type*}

/-- A value returned by the context table was written by an actual triple
wholly inside the specified completed prefix. -/
theorem latestRelation_witness [DecidableEq α] (xs : ℕ → α) (n : ℕ) (a k v : α)
    (hit : latestRelation xs n a k = some v) :
    ∃ j < n, xs j = a ∧ xs (j+1) = k ∧ xs (j+2) = v := by
  induction n with
  | zero => simp [latestRelation] at hit
  | succ n ih =>
    by_cases hw : xs n = a ∧ xs (n+1) = k
    · have hv : xs (n+2) = v := by simpa [latestRelation, hw] using hit
      exact ⟨n, by omega, hw.1, hw.2, hv⟩
    · obtain ⟨j, hj, h⟩ := ih (by simpa [latestRelation, hw] using hit)
      exact ⟨j, by omega, h⟩

/-- With `n` completed triples there are `n+1` completed token pairs.
This read can be used with any query context; the caller uses the final
two input tokens and skips the table when the prefix is shorter. -/
def contextBackoff [DecidableEq α] (xs : ℕ → α) (n : ℕ) (a k : α) : α :=
  (latestRelation xs n a k).getD ((latestSuccessor xs (n+1) k).getD k)

/-- Context selection cannot introduce a destination that never followed
the current key in the completed prefix. -/
theorem contextHit_mem [DecidableEq α] (xs : ℕ → α) (n : ℕ) (a k v : α)
    (hit : latestRelation xs n a k = some v) : v ∈ completedSuccessors xs (n+1) k := by
  obtain ⟨j, hj, _, hk, hv⟩ := latestRelation_witness xs n a k v hit
  exact (mem_completedSuccessors xs (n+1) k v).mpr ⟨j+1, by omega, hk, hv⟩

/-- A singleton set of possible destinations fixes the answer even after
arbitrarily many agreeing writes and regardless of the preceding query token. -/
theorem contextBackoff_agrees [DecidableEq α] (xs : ℕ → α) (n : ℕ) (a k v : α)
    (agree : completedSuccessors xs (n+1) k = {v}) : contextBackoff xs n a k = v := by
  unfold contextBackoff
  cases h : latestRelation xs n a k with
  | some y =>
    have mem := contextHit_mem xs n a k y h
    simpa [agree] using mem
  | none =>
    have history := latestSuccessor_history xs (n+1) k
    cases hs : latestSuccessor xs (n+1) k with
    | none => simp [hs, agree] at history
    | some y => simpa [hs, agree] using history

/-- One written occurrence is sufficient for exact agreement with the
ordinary address for every two-token context. -/
theorem contextBackoff_singleWrite [DecidableEq α] (xs : ℕ → α) (n : ℕ) (a k : α)
    (one : sourceUses xs (n+1) k = 1) :
    contextBackoff xs n a k = (latestSuccessor xs (n+1) k).getD k := by
  obtain ⟨j, hj, hk, latest, unique⟩ := singleWrite_value xs (n+1) k one
  have agree : completedSuccessors xs (n+1) k = {xs (j+1)} := by
    ext v
    rw [mem_completedSuccessors, Finset.mem_singleton]
    constructor
    · rintro ⟨m, hm, hmk, hv⟩
      exact (unique m hm hmk ▸ hv).symm
    · rintro rfl
      exact ⟨j, hj, hk, rfl⟩
  rw [contextBackoff_agrees xs n a k _ agree, latest, Option.getD_some]

/-- The fallback order uses no distinguished vocabulary labels. -/
theorem contextBackoff_map [DecidableEq α] [DecidableEq β] (f : α → β) (hf : Function.Injective f)
    (xs : ℕ → α) (n : ℕ) (a k : α) :
    contextBackoff (fun i => f (xs i)) n (f a) (f k) = f (contextBackoff xs n a k) := by
  simp only [contextBackoff, latestRelation_map f hf, latestSuccessor_map f hf, Option.getD_map]

private def word (i : ℕ) : Fin 9 :=
  (![4,1,7,2,1,8,5,6] : Fin 8 → Fin 9) ⟨i % 8, by omega⟩

/-- The two earlier values of key one disagree. The preceding token four
selects seven, while a missing longer context recovers the latest value eight. -/
example : latestRelation word 6 4 1 = some 7 ∧ contextBackoff word 6 4 1 = 7 ∧
    contextBackoff word 6 3 1 = 8 := by decide

example : completedSuccessors word 3 1 = {7} ∧ contextBackoff word 2 3 1 = 7 := by decide

example : sourceUses word 3 1 = 1 ∧
    contextBackoff word 2 4 1 = (latestSuccessor word 3 1).getD 1 := by decide

example : contextBackoff (fun i => (word i).val) 6 4 1 = (contextBackoff word 6 4 1).val :=
  contextBackoff_map Fin.val Fin.val_injective word 6 4 1

end Transformer.Phasor

module

public import Transformer.Phasor.TransitionTable

/-!
# A bounded causal change counter with an absorbing last bucket

Source: the saturated transition-table trial in `experiments/phasor_value_transport/README.md`.
A vocabulary-sized table keeps each small count distinct and shares its last
entry across all larger counts. Bounded summaries compose over adjacent
segments. On observations below capacity the modular and saturated tables
have the same reads and the same directional derivative.
-/

@[expose] public section

namespace Transformer.Phasor

open scoped RealInnerProductSpace
open Transformer

variable {α ι : Type*} {p D : ℕ}

/-- A finite counter whose last state absorbs further changes. -/
def saturatedChanges (p : ℕ) [NeZero p] (n : ℕ) : Fin p :=
  ⟨min n (p-1), by have := NeZero.pos p; omega⟩

/-- Two counts share an address precisely when equal or both in the last
bucket. In particular, a small count can never recur after saturation. -/
theorem saturatedChanges_eq_iff (p : ℕ) [NeZero p] (m n : ℕ) :
    saturatedChanges p m = saturatedChanges p n ↔ m = n ∨ (p-1 ≤ m ∧ p-1 ≤ n) := by
  rw [Fin.ext_iff]
  change min m (p-1) = min n (p-1) ↔ _
  omega

/-- Merging two independently capped edge counts and capping the sum gives
the same state as processing all edges together, for any chunk sizes. -/
theorem saturatedChanges_add (p : ℕ) [NeZero p] (m n : ℕ) :
    saturatedChanges p ((saturatedChanges p m).val + (saturatedChanges p n).val) =
      saturatedChanges p (m+n) := by
  apply Fin.ext
  change min (min m (p-1) + min n (p-1)) (p-1) = min (m+n) (p-1)
  omega

/-- At a cut, each group counts its own completed edges. The shared
endpoint supplies the first letter of the second group. -/
theorem saturatedTransition_split [DecidableEq α] (p : ℕ) [NeZero p] (xs : ℕ → α) (m n : ℕ) :
    saturatedChanges p (transitionCount xs (m+n)) =
      saturatedChanges p ((saturatedChanges p (transitionCount xs m)).val +
        (saturatedChanges p (transitionCount (fun j => xs (m+j)) n)).val) := by
  rw [transitionCount_split, saturatedChanges_add]

/-- Below capacity the same trained vector is retrieved in both models.
The completed prefix at position `i` ends at its predecessor. -/
theorem saturatedTransition_eq_bucket [DecidableEq α] [NeZero p] (xs : ℕ → α) (i : ℕ)
    (h : transitionCount xs (i-1) < p) :
    saturatedChanges p (transitionCount xs (i-1)) = transitionBucket p xs i := by
  apply Fin.ext
  change min _ (p-1) = _ % p
  rw [Nat.mod_eq_of_lt h, Nat.min_eq_left (by omega)]

/-- The two table reads also give the same directional derivative through
every active row below capacity, for arbitrary upstream hidden gradients. -/
theorem saturatedTransition_agrees_on_batch [DecidableEq α] [NeZero p] (active : Finset ι)
    (xs : ι → ℕ → α) (pos : ι → ℕ) (grad : ι → EucSpace D) (change : Fin p → EucSpace D)
    (bounded : ∀ i ∈ active, transitionCount (xs i) (pos i - 1) < p) :
    ∑ i ∈ active, ⟪grad i, change (saturatedChanges p (transitionCount (xs i) (pos i - 1)))⟫ =
      ∑ i ∈ active, ⟪grad i, change (transitionBucket p (xs i) (pos i))⟫ := by
  apply Finset.sum_congr rfl
  intro i hi
  rw [saturatedTransition_eq_bucket _ _ (bounded i hi)]

private def alternating (i : ℕ) : ℕ := i % 2

/-- One change and five changes address the same modular vector at width
four, while the saturated table separates them. -/
example : transitionBucket 4 alternating 2 = transitionBucket 4 alternating 6 ∧
    saturatedChanges 4 (transitionCount alternating 1) ≠
      saturatedChanges 4 (transitionCount alternating 5) := by decide

example : saturatedChanges 4 (transitionCount alternating 3) = transitionBucket 4 alternating 4 :=
  saturatedTransition_eq_bucket (p := 4) alternating 4 (by decide)

example (grad : Fin 2 → EucSpace 8) (change : Fin 4 → EucSpace 8) :
    ∑ i : Fin 2, ⟪grad i, change (saturatedChanges 4 (transitionCount alternating i.val))⟫ =
      ∑ i : Fin 2, ⟪grad i, change (transitionBucket 4 alternating (i.val+1))⟫ := by
  have fits : ∀ i : Fin 2, transitionCount alternating (i.val+1-1) < 4 := by decide
  simpa using saturatedTransition_agrees_on_batch Finset.univ (fun _ : Fin 2 => alternating)
    (fun i => i.val+1) grad change (fun i _ => fits i)

end Transformer.Phasor

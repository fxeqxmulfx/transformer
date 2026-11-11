module

public import Mathlib.Data.Fintype.Perm
public import Mathlib.Data.List.FinRange
public import Mathlib.Tactic

/-!
# The split and the batches of a run, as functions of its shuffles

Source: openai/grok at 3d64b1d8, `grok/data.py`: `ArithmeticDataset.splits` and `make_data`,
which shuffle the equations and take the first `k` for training, and `ArithmeticIterator`,
which draws a permutation of the training set for each epoch and cuts it into batches in
order; `experiments/grok_division/PROTOCOL.md`, *The split* and *The optimizer*.

The equations are listed as `eqs : Fin N → X`. A shuffle `π` of the list puts the equation
`eqs (π j)` in slot `j`; the first `k` slots are the training set (`trainAt`) and the rest the
held-out set (`heldout`), which is nonempty when `k < N` (`heldout_nonempty`).

An epoch has `⌈k / b⌉` batches (`batchesPerEpoch`). Update `t`, counted from `1`, is batch
`(t - 1) mod ⌈k / b⌉` of epoch `(t - 1) / ⌈k / b⌉`: the slots `j b, …, j b + b - 1` of that
epoch's permutation, cut off at `k` (`batchSlots`). A batch has at most `b` slots and none
twice (`length_batchSlots_le`, `nodup_batchSlots`).

For the paper's sizes, `k = 4656` and `b = 512`, an epoch has `10` batches and the tenth has
`48` slots.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {X : Type*} {N k : ℕ}

/-- The training equation in slot `i` of the list shuffled by `π`. -/
def trainAt (eqs : Fin N → X) (hk : k ≤ N) (π : Equiv.Perm (Fin N)) (i : Fin k) : X :=
  eqs (π (Fin.castLE hk i))

/-- The held-out equations: those in the slots from `k` on. -/
def heldout [DecidableEq X] (eqs : Fin N → X) (k : ℕ) (π : Equiv.Perm (Fin N)) : Finset X :=
  (Finset.univ.filter fun j : Fin N => k ≤ (j : ℕ)).image fun j => eqs (π j)

/-- **The held-out set is nonempty** when not every equation is trained on. -/
lemma heldout_nonempty [DecidableEq X] (eqs : Fin N → X) (hk : k < N) (π : Equiv.Perm (Fin N)) :
    (heldout eqs k π).Nonempty :=
  ⟨eqs (π ⟨k, hk⟩),
    Finset.mem_image.2 ⟨⟨k, hk⟩, Finset.mem_filter.2 ⟨Finset.mem_univ _, le_rfl⟩, rfl⟩⟩

/-- Batches of an epoch over `k` equations at the batch size `b`: `⌈k / b⌉`. -/
def batchesPerEpoch (k b : ℕ) : ℕ := (k + b - 1) / b

/-- **The slots of the batch of update `t`**, counted from `1`: the epochs' permutations are
`σ 0, σ 1, …`. -/
def batchSlots (k b : ℕ) (σ : ℕ → Equiv.Perm (Fin k)) (t : ℕ) : List (Fin k) :=
  (((List.finRange k).map (σ ((t - 1) / batchesPerEpoch k b))).drop
    ((t - 1) % batchesPerEpoch k b * b)).take b

lemma length_batchSlots_le (k b : ℕ) (σ : ℕ → Equiv.Perm (Fin k)) (t : ℕ) :
    (batchSlots k b σ t).length ≤ b :=
  List.length_take_le _ _

/-- **No slot is twice in a batch.** -/
lemma nodup_batchSlots (k b : ℕ) (σ : ℕ → Equiv.Perm (Fin k)) (t : ℕ) :
    (batchSlots k b σ t).Nodup :=
  (((List.nodup_finRange k).map (Equiv.injective _)).sublist (List.drop_sublist _ _)).sublist
    (List.take_sublist _ _)

/-- The length of batch `j` of an epoch: `b`, or what is left of the `k` slots. -/
lemma length_batchSlots (k b : ℕ) (σ : ℕ → Equiv.Perm (Fin k)) (t : ℕ) :
    (batchSlots k b σ t).length = min b (k - (t - 1) % batchesPerEpoch k b * b) := by
  simp [batchSlots]

/-! The witnesses: the paper's sizes, and an epoch of four equations in batches of three. -/

example : batchesPerEpoch 4656 512 = 10 := by decide

/-- The tenth batch of an epoch, update `10`, has `48` slots; the first has `512`. -/
example (σ : ℕ → Equiv.Perm (Fin 4656)) :
    (batchSlots 4656 512 σ 10).length = 48 ∧ (batchSlots 4656 512 σ 1).length = 512 := by
  constructor <;> rw [length_batchSlots] <;> decide

/-- Update `11` starts the second epoch: its batch is cut from `σ 1`. -/
example : (11 - 1) / batchesPerEpoch 4656 512 = 1 ∧ (11 - 1) % batchesPerEpoch 4656 512 = 0 := by
  decide

/-- With the identity shuffles, four equations in batches of three: updates `1` and `2` see
the slots `0, 1, 2` and `3`, and update `3` starts over. -/
example : batchSlots 4 3 (fun _ => 1) 1 = [0, 1, 2] ∧ batchSlots 4 3 (fun _ => 1) 2 = [3] ∧
    batchSlots 4 3 (fun _ => 1) 3 = [0, 1, 2] := by
  decide

example := length_batchSlots_le 4 3 (fun _ => 1) 2

example := nodup_batchSlots 4 3 (fun _ => 1) 2

/-- Two equations, one trained on: with the identity shuffle the first is the training
equation and the second the held-out set. -/
example : trainAt (id : Fin 2 → Fin 2) (by norm_num : 1 ≤ 2) 1 0 = 0 ∧
    heldout (id : Fin 2 → Fin 2) 1 1 = {1} := by
  decide

example := heldout_nonempty (id : Fin 2 → Fin 2) (by norm_num : 1 < 2) 1

end Transformer.Grokking.Protocol

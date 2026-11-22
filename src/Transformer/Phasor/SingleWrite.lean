module

public import Transformer.Phasor.ClosedRelations
public import Transformer.Phasor.DisplacementGain

/-!
# Identifying a successor written exactly once

Source: the single-write gain trial in `experiments/phasor_value_transport/README.md`.
An ordinary read can distinguish one completed occurrence of its key from
repeated writes, even when all of those writes carry the same value. A
saturating counter suffices at every prefix length. A resolved closed-relation
read and an ordinary moving singleton read share the larger gain.
-/

@[expose] public section

namespace Transformer.Phasor

variable {α β : Type*}

/-- Completed writes out of the given key among the first `n` transitions. -/
def sourceCount [DecidableEq α] (xs : ℕ → α) (n : ℕ) (k : α) : ℕ :=
  ((Finset.range n).filter fun j => xs j = k).card

/-- A completed write increments exactly one source count, including on a self-loop. -/
theorem sourceCount_succ [DecidableEq α] (xs : ℕ → α) (n : ℕ) (k : α) :
    sourceCount xs (n+1) k = sourceCount xs n k + if xs n = k then 1 else 0 := by
  have fresh : n ∉ (Finset.range n).filter (fun j => xs j = k) := by simp
  by_cases h : xs n = k <;>
    simp [sourceCount, Finset.range_add_one, Finset.filter_insert, h, fresh]

/-- Two is the absorbing state of the two-bit per-key counter. -/
def sourceUses [DecidableEq α] (xs : ℕ → α) : ℕ → α → ℕ
  | 0, _ => 0
  | n+1, k => if xs n = k then min 2 (sourceUses xs n k + 1) else sourceUses xs n k

/-- Saturating after every individual write gives exactly the truncated
count of all matching source positions, for arbitrary histories. -/
theorem sourceUses_eq_min [DecidableEq α] (xs : ℕ → α) (n : ℕ) (k : α) :
    sourceUses xs n k = min 2 (sourceCount xs n k) := by
  induction n with
  | zero => simp [sourceUses, sourceCount]
  | succ n ih =>
    rw [sourceUses, sourceCount_succ, ih]
    by_cases h : xs n = k
    · simp [h]
      omega
    · simp [h]

/-- A singleton count identifies a unique transition position, including
when several occurrences of another key share the same destination. -/
theorem sourceUses_one_iff [DecidableEq α] (xs : ℕ → α) (n : ℕ) (k : α) :
    sourceUses xs n k = 1 ↔ ∃! j, j < n ∧ xs j = k := by
  rw [sourceUses_eq_min]
  have h : min 2 (sourceCount xs n k) = 1 ↔ sourceCount xs n k = 1 := by omega
  rw [h, sourceCount, Finset.card_eq_one_iff_existsUnique]
  simp

/-- The value delivered by a singleton is the target of that specific
transition, independent of all writes to other keys. -/
theorem singleWrite_value [DecidableEq α] (xs : ℕ → α) (n : ℕ) (k : α)
    (one : sourceUses xs n k = 1) :
    ∃ j, j < n ∧ xs j = k ∧ latestSuccessor xs n k = some (xs (j+1)) ∧
      ∀ m, m < n → xs m = k → m = j := by
  obtain ⟨j, ⟨hj, hk⟩, only⟩ := (sourceUses_one_iff xs n k).mp one
  refine ⟨j, hj, hk, latestSuccessor_lookup xs k j n hj hk ?_, fun m hm hmkey => only m ⟨hm, hmkey⟩⟩
  intro m later hm same
  exact later.ne (only m ⟨hm, same⟩).symm

/-- The single-write condition is invariant under any injective recoding
of tokens, even into a larger alphabet. -/
theorem sourceUses_map [DecidableEq α] [DecidableEq β] (f : α → β) (hf : Function.Injective f)
    (xs : ℕ → α) (n : ℕ) (k : α) : sourceUses (f ∘ xs) n (f k) = sourceUses xs n k := by
  simp [sourceUses_eq_min, sourceCount, Function.comp_def, hf.eq_iff]

/-- The closed-relation evidence has priority. An ordinary successor receives
the same multiplier only on a changed read with one completed write. -/
def singleWriteGain [DecidableEq α] (gain factor : ℝ) (closed : Bool) (xs : ℕ → α) (i : ℕ) : ℝ :=
  if closed then factor * gain else if sourceUses xs (i-1) (xs i) = 1
    then retrievalGain gain factor xs i else gain

/-- Once the closed-relation vote is given, token renaming leaves every
downstream gain decision unchanged. -/
theorem singleWriteGain_map [DecidableEq α] [DecidableEq β] (f : α → β) (hf : Function.Injective f)
    (gain factor : ℝ) (closed : Bool) (xs : ℕ → α) (i : ℕ) :
    singleWriteGain gain factor closed (f ∘ xs) i = singleWriteGain gain factor closed xs i := by
  simp only [singleWriteGain, Function.comp_apply, sourceUses_map f hf]
  rw [show retrievalGain gain factor (f ∘ xs) i = retrievalGain gain factor xs i from
    retrievalGain_map f hf gain factor xs i]

/-- Two-token inputs have no completed writes at either position. -/
theorem singleWriteGain_start [DecidableEq α] (gain factor : ℝ) (xs : ℕ → α) (i : ℕ) (hi : i ≤ 1) :
    singleWriteGain gain factor false xs i = gain := by
  have h : i-1 = 0 := by omega
  simp [singleWriteGain, sourceUses, h]

private def word (i : ℕ) : Fin 3 := (![0, 1, 2, 0, 1, 0] : Fin 6 → Fin 3) ⟨i % 6, by omega⟩

/-- The first read of key zero finds its only earlier value. A second
completed write of the same value makes the later ordinary read use base gain. -/
example : sourceUses word 2 0 = 1 ∧ latestSuccessor word 2 0 = some 1 ∧
    sourceUses word 4 0 = 2 ∧ latestSuccessor word 4 0 = some 1 := by decide

example : singleWriteGain 1 2 false word 3 = 2 ∧ singleWriteGain 1 2 false word 5 = 1 ∧
    singleWriteGain 1 2 true word 5 = 2 := by
  norm_num [singleWriteGain, retrievalGain, successorAddress, latestSuccessor, sourceUses, word]

example : sourceUses ((fun a : Fin 3 => a.val) ∘ word) 4 0 = sourceUses word 4 0 :=
  sourceUses_map Fin.val Fin.val_injective word 4 0

example : singleWriteGain 1 2 false word 1 = 1 := singleWriteGain_start _ _ _ _ (by decide)

/-- Evidence for a residual gain. A moving closed-relation vote takes
priority over any simultaneously available ordinary singleton. -/
inductive ReadSupport where
  | ordinary | singleton | relation
  deriving DecidableEq

/-- Classify a position using its causal closed vote and completed writes.
A self-read contributes no singleton evidence, even when it was written once. -/
def classifyReadSupport [DecidableEq α] (closed : Bool) (xs : ℕ → α) (i : ℕ) : ReadSupport :=
  if closed then .relation else
    if sourceUses xs (i-1) (xs i) = 1 ∧ successorAddress xs i ≠ xs i then .singleton else .ordinary

/-- The separately learned increment belongs only to a moving singleton
without a closed vote. Closed votes keep twice the learned base gain. -/
def ReadSupport.amplitude (e : ReadSupport) (base bonus : ℝ) : ℝ :=
  match e with
  | .ordinary => base
  | .singleton => base + bonus
  | .relation => 2 * base

/-- The exclusive singleton-gain trial in the experiment README. Zero bonus
recovers every closed-only gain. A bonus equal to the base recovers every
gain of the inclusive single-write model, including where both kinds of
evidence occur. -/
theorem readSupport_gain_endpoints [DecidableEq α] (closed : Bool) (xs : ℕ → α) (i : ℕ)
    (base : ℝ) :
    (classifyReadSupport closed xs i).amplitude base 0 = (if closed then 2*base else base) ∧
    (classifyReadSupport closed xs i).amplitude base base = singleWriteGain base 2 closed xs i := by
  cases closed <;> by_cases one : sourceUses xs (i-1) (xs i) = 1 <;>
    by_cases same : successorAddress xs i = xs i <;>
    simp [classifyReadSupport, ReadSupport.amplitude, singleWriteGain, retrievalGain, one, same, two_mul]

/-- The exclusive singleton-gain trial in the experiment README. A closed
vote and a moving singleton can be assigned any two real amplitudes, with
a unique parameter pair, despite using a single gain on ordinary reads. -/
theorem existsUnique_readSupport_amplitudes {ι : Type*} (support : ι → ReadSupport) (i j : ι)
    (hi : support i = .relation) (hj : support j = .singleton) (r s : ℝ) :
    ∃! θ : ℝ × ℝ, (support i).amplitude θ.1 θ.2 = r ∧ (support j).amplitude θ.1 θ.2 = s := by
  rw [hi, hj]
  refine ⟨(r/2, s-r/2), ?_, ?_⟩
  · constructor <;> dsimp [ReadSupport.amplitude] <;> ring
  · rintro ⟨g,b⟩ ⟨hr,hs⟩
    change 2*g = r at hr
    change g+b = s at hs
    apply Prod.ext <;> dsimp <;> linarith

/-- The exclusive singleton-gain trial in the experiment README. For
arbitrary upstream derivatives, a closed vote doubles the base derivative
and every exclusive singleton contributes once to the bonus derivative.
Both sums preserve all repeated and masked rows. -/
theorem readSupport_batch_adjoint {ι : Type*} [Fintype ι] (support : ι → ReadSupport)
    (upstream : ι → ℝ) (g b dg db : ℝ) :
    (∑ i, upstream i * ((support i).amplitude (g+dg) (b+db) - (support i).amplitude g b)) =
      (∑ i, if support i = .relation then 2 * upstream i else upstream i) * dg +
      (∑ i, if support i = .singleton then upstream i else 0) * db := by
  rw [Finset.sum_mul, Finset.sum_mul, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  cases support i <;> simp [ReadSupport.amplitude] <;> ring

example :
    (classifyReadSupport false word 3, classifyReadSupport false word 5,
      classifyReadSupport true word 3) = (.singleton, .ordinary, .relation) := by decide

example (r s : ℝ) :
    let support := fun vote : Bool => classifyReadSupport vote word 3
    ∃! θ : ℝ × ℝ, (support true).amplitude θ.1 θ.2 = r ∧ (support false).amplitude θ.1 θ.2 = s :=
  existsUnique_readSupport_amplitudes (fun vote => classifyReadSupport vote word 3)
    true false (by decide) (by decide) r s

end Transformer.Phasor

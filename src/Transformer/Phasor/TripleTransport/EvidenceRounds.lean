module

public import Transformer.Phasor.TripleTransport

/-!
# Reusing predictions supported by learned hidden correspondences

Source: “Repeated witness reads” in the triple-transport experiment. Each
round may use a different reference budget. It adds answers to a local copy
of the learned evidence. The bound on the number of rounds is arbitrary.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.EvidenceRounds

variable {I V : Type*}

abbrev Bank (I V : Type*) := I → Option V

/-- A round keeps all existing correspondences and fills only abstentions. -/
def fill (bank : Bank I V) (infer : Bank I V → Bank I V) : Bank I V :=
  fun i => (bank i).or (infer bank i)

def rounds (infer : ℕ → Bank I V → Bank I V) (initial : Bank I V) : ℕ → Bank I V
  | 0 => initial
  | n + 1 => fill (rounds infer initial n) (infer n)

/-- A fixed but unknown total interpretation certifies a partial graph. -/
def Agrees (truth : I → V) (bank : Bank I V) : Prop := ∀ i v, bank i = some v → truth i = v

/-- If a local inference is sound whenever the available evidence is sound,
each finite round preserves the unknown total interpretation. Only predictions
on previously unanswered keys need to satisfy the local condition. -/
theorem rounds_agree (truth : I → V) (initial : Bank I V) (infer : ℕ → Bank I V → Bank I V)
    (base : Agrees truth initial)
    (localSound : ∀ n b, Agrees truth b → ∀ i v, b i = none → infer n b i = some v → truth i = v)
    (n : ℕ) : Agrees truth (rounds infer initial n) := by
  induction n with
  | zero => exact base
  | succ n ih =>
    intro i v h
    change ((rounds infer initial n i).or _) = some v at h
    cases prior : rounds infer initial n i with
    | none => exact localSound n _ ih i v prior (by simpa [prior] using h)
    | some w =>
      have same : w = v := by simpa [prior] using h
      exact (ih i w prior).trans same

/-- Adding further rounds never overwrites a previously answered query,
independently of the quality of any new proposals. -/
theorem retained (initial : Bank I V) (infer : ℕ → Bank I V → Bank I V)
    (first extra : ℕ) (i : I) (v : V) (known : rounds infer initial first i = some v) :
    rounds infer initial (first + extra) i = some v := by
  induction extra with
  | zero => exact known
  | succ n ih =>
    change (rounds infer initial (first + n) i).or _ = some v
    simp [ih]

private def seed (i : ℕ) : Option ℕ := if i = 0 then some 0 else none
private def next (b : Bank ℕ ℕ) : Bank ℕ ℕ
  | 0 => none
  | i + 1 => (b i).map Nat.succ

example : rounds (fun _ => next) seed 1 2 = none ∧ rounds (fun _ => next) seed 2 2 = some 2 := by decide

example (n : ℕ) : Agrees id (rounds (fun _ => next) seed n) := by
  apply rounds_agree
  · intro i v seen
    by_cases h : i = 0 <;> simp_all [seed]
  · intro _ b hb i v _ proposed
    cases i with
    | zero => simp [next] at proposed
    | succ i =>
      cases h : b i with
      | none => simp [next, h] at proposed
      | some w =>
        have val := hb i w h
        simpa [next, h, ← val] using proposed

example : rounds (fun _ => next) seed 4 2 = some 2 :=
  retained seed (fun _ => next) 2 2 2 2 (by decide)

end Transformer.Phasor.TripleTransport.EvidenceRounds

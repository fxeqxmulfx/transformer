module

public import Transformer.Phasor.LookupQuestion

/-! The synthetic lookup study scores the freely generated answer letter.
Each row has a reference and an optional valid prediction. Concentration
in a single predicted letter limits attainable accuracy on a balanced split. -/
@[expose] public section

namespace Transformer.Phasor.LookupQuestion.AnswerCounts

variable {A : Type*} [DecidableEq A]

def correct (rows : List (A × Option A)) : ℕ :=
  (rows.filter fun p => p.2 == some p.1).length

def gold (c : A) (rows : List (A × Option A)) : ℕ :=
  (rows.filter fun p => p.1 == c).length

def other (c : A) (rows : List (A × Option A)) : ℕ :=
  (rows.filter fun p => p.2 != some c).length

/-- Each correct answer has reference c or is predicted away from c.
The bound includes invalid and empty generations. -/
theorem concentrated (rows : List (A × Option A)) (c : A) :
    correct rows ≤ gold c rows + other c rows := by
  induction rows with
  | nil => simp [correct, gold, other]
  | cons entry rest ih =>
    obtain ⟨g, p⟩ := entry
    by_cases h : g = c <;> by_cases k : p = some g <;> by_cases model : p = some c <;>
      simp_all [correct, gold, other] <;> omega

/-- Always predicting c gives exactly the number of occurrences of c in
the reference answers, for any choice of split. -/
theorem constant (truth : List A) (c : A) :
    correct (truth.map fun g => (g, some c)) = truth.count c := by
  induction truth with
  | nil => simp [correct]
  | cons g rest ih =>
    by_cases h : g = c
    · subst g
      simpa [correct] using congrArg Nat.succ ih
    · have different : c ≠ g := Ne.symm h
      simpa [correct, h, different] using ih

private def sample : List (ℕ × Option ℕ) :=
  [(0, some 0), (1, some 0), (2, none), (3, some 3)]

example : correct sample ≤ gold 0 sample + other 0 sample := concentrated _ _
example : correct ([0, 1, 2, 3, 4].map fun g : ℕ => (g, some 2)) = 1 := by decide

end Transformer.Phasor.LookupQuestion.AnswerCounts

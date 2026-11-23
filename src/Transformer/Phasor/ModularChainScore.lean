module

public import Mathlib.Data.ZMod.Basic
public import Mathlib.Tactic

/-!
# Final-answer supervision on the modular-chain GPT catalog cases

Source: DiffGPT `research/current_grokking/adapters.py`, function `chain`,
at 5a37c515, and `ModularCompositionTask` at 07a39ac2. The data adapter
constructs short and doubled-length examples with one target per row.
-/

@[expose] public section

namespace Transformer.Phasor.ModularChainScore

variable {A : Type*} {p : ℕ}

/-- DiffGPT's chain adapter at 5a37c515 gives the model the initial residue
and the two raw coefficients of each step in their original order. -/
def coefficients : List (A × A) → List A
  | [] => []
  | (b,c) :: steps => b :: c :: coefficients steps

def inputs (z : A) (steps : List (A × A)) := z :: coefficients steps

theorem input_length (z : A) (steps : List (A × A)) :
    (inputs z steps).length = 2 * steps.length + 1 := by
  have h : (coefficients steps).length = 2 * steps.length := by
    induction steps with
    | nil => rfl
    | cons pair steps ih =>
      rcases pair with ⟨b,c⟩
      simp only [coefficients, List.length_cons] at *
      omega
  simp [inputs, h]

/-- The generator calculates the final residue in the coefficient ring.
This recurrence lives outside the trained model. -/
def answer (z : ZMod p) (steps : List (ZMod p × ZMod p)) : ZMod p :=
  steps.foldl (fun x bc => x^3 + bc.1*x + bc.2) z

/-- The doubled-length generator uses the same recurrence on twice as many
coefficient pairs. Its second half begins at the first half's true state. -/
theorem answer_append (z : ZMod p) (a b : List (ZMod p × ZMod p)) :
    answer z (a ++ b) = answer (answer z a) b := List.foldl_append

/-- Only the final position receives the target. All earlier logits are
ignored by the source exact-sequence scorer. -/
def targets (n : ℕ) (y : A) : List (Option A) := List.replicate n none ++ [some y]

theorem supervised_iff (n i : ℕ) (y a : A) :
    (targets n y)[i]? = some (some a) ↔ i = n ∧ y = a := by
  simp only [targets, List.getElem?_append, List.length_replicate, List.getElem?_replicate,
    List.getElem?_singleton]
  by_cases lt : i < n
  · have ne : i ≠ n := by omega
    simp [lt, ne]
  · by_cases eq : i = n
    · simp [eq]
    · have diff : i - n ≠ 0 := by omega
      simp [lt, eq, diff]

/-- The dense row passes whole-sequence accuracy exactly when a prediction
from the complete original prefix gives the final answer. This holds at
the original and doubled lengths for any prefix predictor. -/
theorem exact_iff (predict : List (ZMod p) → ZMod p) (z : ZMod p)
    (steps : List (ZMod p × ZMod p)) :
    (∀ i a, (targets (2 * steps.length) (answer z steps))[i]? = some (some a) →
      predict ((inputs z steps).take (i+1)) = a) ↔
        predict (inputs z steps) = answer z steps := by
  have full : (inputs z steps).take (2*steps.length+1) = inputs z steps :=
    List.take_of_length_le (by rw [input_length])
  constructor
  · intro h
    simpa only [full] using h _ _ ((supervised_iff _ _ _ _).2 ⟨rfl,rfl⟩)
  · intro h i a accept
    obtain ⟨rfl,rfl⟩ := (supervised_iff _ _ _ _).1 accept
    simpa only [full] using h

example : inputs (2 : ZMod 17) [(3,4),(5,6)] = [2,3,4,5,6] ∧
    answer (2 : ZMod 17) [(3,4),(5,6)] = 12 := by decide

end Transformer.Phasor.ModularChainScore

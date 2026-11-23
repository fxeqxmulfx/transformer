module

public import Transformer.Phasor.TripleTransport.FixedEvaluation

/-!
# Reusing the first greedy prediction from an identical prompt batch

Phasor's batched free-generation scorer already has predictions from its initial
forward pass. With the same packed input, the last real position of each row has
the same tied logits regardless of the labels supplied to the outer loss. Subsequent
passes receive the tokens that the model actually generated, up to EOS or the limit.
-/

@[expose] public section
namespace Transformer.Phasor.TripleTransport.FirstGeneration
open FixedEvaluation
variable {A : Type*} [DecidableEq A]

/-- Supply an already computed first token. For a positive limit the usual
rollout handles the remaining positions with its original stop rule. -/
def primed (first stop : A) (next : ℕ → List A → A) (start : ℕ) (prompt : List A) : ℕ → List A
  | 0 => []
  | fuel+1 => first :: (if first = stop then [] else
      rollout next stop fuel (start+1) (prompt++[first]))

/-- Only the actually queried initial prefix must agree with the saved
prediction. Subsequent free decoding can use any extensionally equal
implementation inside the remaining window, including after row compaction. -/
theorem same_tokens (direct later : ℕ → List A → A) (stop first : A)
    (fuel start : ℕ) (prompt : List A) (initial : first = direct start prompt)
    (future : ∀ i, start < i → i < start+fuel → ∀ p, later i p = direct i p) :
    primed first stop later start prompt fuel = rollout direct stop fuel start prompt := by
  cases fuel with
  | zero => rfl
  | succ n =>
    subst first
    simp only [primed, rollout]
    split
    · rfl
    · congr 1
      apply rollout_eq_of_window
      intro i lo hi p
      exact future i (by omega) (by omega) p

example : primed 2 4 (fun _ (p : List ℕ) => p.length + 1) 0 [7] 6 =
    rollout (fun _ p => p.length+1) 4 6 0 [7] := by
  apply same_tokens
  · rfl
  · intro i lo hi p
    rfl

/-- A token obtained on a different input can terminate the trajectory early,
even if limits, weights and subsequent greedy decoding agree. -/
example : primed 2 4 (fun _ (p : List ℕ) => p.length+1) 0 [7] 6 = [2,3,4] ∧
    primed 4 4 (fun _ (p : List ℕ) => p.length+1) 0 [7] 6 = [4] := by decide

end Transformer.Phasor.TripleTransport.FirstGeneration

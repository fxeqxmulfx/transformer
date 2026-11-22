module

public import Transformer.Phasor.TripleTransport

/-!
# Reusing fixed weights during evaluation

Source: “Fixed-weight evaluation” in the triple-transport experiment. The
original scorer uses multiple batches and greedily extends each causal prefix.
A function of the frozen parameters can be prepared once for that score.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.FixedEvaluation

variable {A W H : Type*} [DecidableEq A]

/-- Greedy free decoding emits the stop token and never appends it to a later
prefix. Each invocation has its own generation limit. -/
def rollout (next : ℕ → List A → A) (stop : A) : ℕ → ℕ → List A → List A
  | 0, _, _ => []
  | fuel + 1, i, prompt =>
    let a := next i prompt
    a :: (if a = stop then [] else rollout next stop fuel (i + 1) (prompt ++ [a]))

/-- Agreement of each next-token function inside the current scoring window
preserves the emitted sequence, including its stopping time. -/
theorem rollout_eq_of_window (next cached : ℕ → List A → A) (stop : A)
    (fuel first : ℕ) (prompt : List A)
    (same : ∀ i, first ≤ i → i < first + fuel → ∀ p, next i p = cached i p) :
    rollout next stop fuel first prompt = rollout cached stop fuel first prompt := by
  induction fuel generalizing first prompt with
  | zero => rfl
  | succ n ih =>
    have here := same first (by omega) (by omega) prompt
    simp only [rollout, here]
    split
    · rfl
    · congr 1
      apply ih
      intro i lower upper
      exact same i (by omega) (by omega)

/-- A dense read may precompute any function of the fixed weights once per
evaluation. Every input-dependent calculation still runs on the generated
prefix. The assumption needs to hold only from entry to the limit. -/
theorem reuse_preparation (prepare : W → H) (read : H → List A → A)
    (current : ℕ → W) (snapshot : W) (stop : A) (fuel first : ℕ) (prompt : List A)
    (stable : ∀ i, first ≤ i → i < first + fuel → prepare (current i) = prepare snapshot) :
    rollout (fun i => read (prepare (current i))) stop fuel first prompt =
      rollout (fun _ => read (prepare snapshot)) stop fuel first prompt := by
  apply rollout_eq_of_window
  intro i lower upper p
  rw [stable i lower upper]

private def readAt (w : ℕ) (p : List ℕ) := w + p.length

example : rollout (fun i => readAt (if i < 4 then 3 else i)) 5 4 0 [] =
    [3, 4, 5] := by decide

example : rollout (fun i => readAt (if i < 4 then 3 else i)) 5 4 0 [] =
    rollout (fun _ => readAt 3) 5 4 0 [] := by
  apply reuse_preparation id readAt (fun i => if i < 4 then 3 else i)
  intro i lo hi
  simp [show i < 4 by omega]

/-- Carrying the previous snapshot across a parameter change can alter the
first prediction and whether the scorer emits its stop token. -/
example : rollout (fun _ => readAt 2) 3 3 0 [] = [2, 3] ∧
    rollout (fun _ => readAt 1) 3 3 0 [] = [1, 2, 3] := by decide

end Transformer.Phasor.TripleTransport.FixedEvaluation

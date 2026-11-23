module
public import Transformer.Phasor.TripleTransport.TypedLookupPaths

/-! The disjoint-layer depth study. A terminating succession of different
keys can distinguish pass counts only while its endpoint is still moving.
A deterministic optimizer retains its entire state when every minibatch
gives equal gradients along the common trajectory. -/
@[expose] public section
namespace Transformer.Phasor.TripleTransport.TrainingReadDepth
open TypedLookupPaths

/-- A chain with h directed links and a missing edge at its final key. -/
def linear (h k : ℕ) : Option ℕ := if k<h then some (k+1) else none

/-- The same shared read makes exactly the available progress on this
example, providing an explicit witness whenever a budget is too short. -/
theorem progress (h n : ℕ) : follow (linear h) 0 n = min n h := by
  induction n with
  | zero => simp [follow]
  | succ n ih =>
    rw [follow,ih]
    by_cases step : n<h
    · simp [linear,Nat.min_eq_left (Nat.le_of_lt step),step]
    · have bound : h≤n := by omega
      have after : h≤n+1 := by omega
      simp [linear,Nat.min_eq_right bound,Nat.min_eq_right after]

/-- An optimizer state can include moments, parameters and the batch
schedule. Both arms have the same initial state and update program. -/
def trajectory {S G : Type*} (step : ℕ → S → G → S) (grad : ℕ → S → G) (initial : S) : ℕ → S
  | 0 => initial
  | n+1 => let s := trajectory step grad initial n; step n s (grad n s)

/-- Equality of gradients is required only on the small-budget trajectory
at the updates made so far. Matching predictions alone is insufficient. -/
theorem same_trajectory {S G : Type*} (step : ℕ → S → G → S) (a b : ℕ → S → G)
    (init : S) (n : ℕ) (equal : ∀ i<n, let s := trajectory step a init i; a i s = b i s) :
    trajectory step a init n = trajectory step b init n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have prev := ih (fun i hi => equal i (by omega))
    simp only [trajectory, ←prev, equal n (by omega)]

example : follow (linear 5) 0 10 = follow (linear 5) 0 40 ∧
    follow (linear 20) 0 10 ≠ follow (linear 20) 0 20 := by simp [progress]
example : trajectory (fun (i s g : ℕ) => s+g+i) (fun i s => s+i+1) 2 4 =
    trajectory (fun (i s g : ℕ) => s+g+i) (fun i s => if i<4 then s+i+1 else 0) 2 4 := by
  apply same_trajectory
  intro i hi
  simp [hi]

end Transformer.Phasor.TripleTransport.TrainingReadDepth

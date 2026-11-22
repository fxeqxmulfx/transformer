module

public import Transformer.Phasor.TripleTransport

/-!
# Splitting the original AdamW observation schedule into work windows

Source: “Resuming the original update budget” in the triple-transport
experiment. A work window carries the state of the model, optimizer and
sampler, and the index in a potentially nonstationary update schedule.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.ResumeSchedule

variable {S O : Type*}

/-- State includes parameters, both AdamW moments and the next sampler
position. The update for an absolute step can use its prescribed WSD rate. -/
def run (update : ℕ → S → S) : ℕ → S → ℕ → S
  | _, s, 0 => s
  | t, s, n + 1 => run update (t + 1) (update t s) n

/-- A later work window starts at the exact optimizer and sampler state of
the earlier window. The absolute update index prevents the schedule from
starting over when a process reloads. -/
theorem run_split (u : ℕ → S → S) (t : ℕ) (s : S) (lo hi : ℕ) :
    run u t s (lo + hi) = run u (t + lo) (run u t s lo) hi := by
  induction lo generalizing t s with
  | zero => simp [run]
  | succ n ih =>
    rw [Nat.succ_add]
    simp only [run]
    rw [ih]
    rw [show t + 1 + n = t + (n + 1) by omega]

/-- The optional scorer is called just at its original observation indices.
The half-open window convention records the shared boundary only once. -/
def observations (u : ℕ → S → S) (score : ℕ → S → Option O) : ℕ → S → ℕ → List O
  | _, _, 0 => []
  | t, s, n + 1 => (score t s).toList ++ observations u score (t + 1) (u t s) n

theorem observations_split (u : ℕ → S → S) (score : ℕ → S → Option O)
    (t : ℕ) (s : S) (lo hi : ℕ) :
    observations u score t s (lo + hi) = observations u score t s lo ++
      observations u score (t + lo) (run u t s lo) hi := by
  induction lo generalizing t s with
  | zero => simp [observations, run]
  | succ n ih =>
    rw [Nat.succ_add]
    simp only [observations, run, List.append_assoc]
    rw [ih]
    have boundary : t + 1 + n = t + (n + 1) := by omega
    rw [boundary]

private def update (t : ℕ) (s : ℕ × ℕ) : ℕ × ℕ := (s.1 + s.2, s.2 + t + 1)
private def score (t : ℕ) (s : ℕ × ℕ) : Option Bool :=
  if t % 2 = 0 then some (decide (6 ≤ s.1)) else none

example : run update 0 (0, 1) 5 = run update 2 (run update 0 (0, 1) 2) 3 :=
  run_split update 0 (0, 1) 2 3

example : observations update score 0 (0, 1) 6 =
    observations update score 0 (0, 1) 2 ++ observations update score 2 (run update 0 (0, 1) 2) 4 :=
  observations_split update score 0 (0, 1) 2 4

/-- Persisting just the first coordinate can change a later scheduled
observation, even when the parameter at the boundary agrees. -/
example : (run update 2 (3, 4) 2).1 ≠ (run update 2 (3, 0) 2).1 ∧
    score 4 (run update 2 (3, 4) 2) = some true := by decide

end Transformer.Phasor.TripleTransport.ResumeSchedule

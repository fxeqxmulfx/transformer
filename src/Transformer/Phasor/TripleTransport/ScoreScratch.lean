module

public import Transformer.Phasor.TripleTransport.FixedEvaluation

/-!
# Reusing attention scores across frozen forward layers

Source: the Phasor fixed-weight evaluation and the pinned Kernite v0.6.0
forward scorer. Each layer overwrites its active slots before reading them.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.ScoreScratch

variable {A I V T : Type*} [DecidableEq I]

/-- A scoring layer writes its active set of probabilities and imaginary
scores. Its forward consumer can address that active set. Inactive padding
may contain scores left by an earlier layer or an earlier scoring batch. -/
structure Stage (A I V : Type*) where
  active : Finset I
  make : A → I → V
  finish : A → (↥active → V) → A

def write (s : Stage A I V) (h : A) (old : I → V) (i : I) : V :=
  if i ∈ s.active then s.make h i else old i

def consume (s : Stage A I V) (h : A) (buf : I → V) : A :=
  s.finish h (fun i => buf i.val)

theorem consume_written (s : Stage A I V) (h : A) (old other : I → V) :
    consume s h (write s h old) = consume s h (write s h other) := by
  unfold consume
  congr 1
  funext i
  simp [write,i.property]

/-- A training tape retains each layer's freshly allocated score buffer. -/
def retained (fresh : I → V) : List (Stage A I V) → A → A × List (I → V)
  | [], h => (h,[])
  | s :: later, h =>
    let scores := write s h fresh
    let rest := retained fresh later (consume s h scores)
    (rest.1,scores::rest.2)

/-- A frozen forward keeps one scratch buffer across all layers. -/
def shared : List (Stage A I V) → A → (I → V) → A × (I → V)
  | [], h, old => (h,old)
  | s :: later, h, old =>
    let scores := write s h old
    shared later (consume s h scores) scores

theorem forward_eq (layers : List (Stage A I V)) (fresh old : I → V) (h : A) :
    (retained fresh layers h).1 = (shared layers h old).1 := by
  induction layers generalizing h old with
  | nil => rfl
  | cons s later ih =>
    simp only [retained,shared]
    rw [ih (write s h old) (consume s h (write s h fresh)), consume_written s h fresh old]

/-- The same comparison holds when every generation step changes the active
shape and the contents of scratch memory. Greedy termination is preserved. -/
theorem rollout_eq [DecidableEq T] (layers : ℕ → List T → List (Stage A I V))
    (hidden : List T → A) (token : A → T) (fresh : I → V)
    (old : ℕ → List T → I → V) (stop : T) (fuel start : ℕ) (prompt : List T) :
    FixedEvaluation.rollout (fun n p => token (retained fresh (layers n p) (hidden p)).1)
        stop fuel start prompt =
      FixedEvaluation.rollout (fun n p => token (shared (layers n p) (hidden p) (old n p)).1)
        stop fuel start prompt := by
  apply FixedEvaluation.rollout_eq_of_window
  intro n _ _ p
  rw [forward_eq (layers n p) fresh (old n p) (hidden p)]

private def indexed (i delta : ℕ) : Stage ℕ ℕ ℕ where
  active := {i}
  make := fun h j => h + j + delta
  finish := fun h scores => h + scores ⟨i,by simp⟩

example : (retained (fun _ => 0) [indexed 7 2,indexed 1 3] 4).1 = 38 ∧
    (shared [indexed 7 2,indexed 1 3] 4 (fun _ => 99)).1 = 38 := by decide

/-- A consumer reading beyond its active region could see stale padding. -/
example : write (indexed 4 2) 3 (fun _ => 0) 7 ≠ write (indexed 4 2) 3 (fun _ => 99) 7 := by decide

end Transformer.Phasor.TripleTransport.ScoreScratch

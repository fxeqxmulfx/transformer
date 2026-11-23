module

public import Mathlib.Tactic

/-!
# Causal replay of the middle blocks of Inkling 2 Mini

Thinking Machines Lab, *Introducing Inkling 2 Mini*, “Looping design”, and
`Inkling2LoopTextModel.forward` in Transformers. A pass starts from the
saved prelude. Its only other input is a message computed from the preceding
pass's block states. The state type can retain the entire block history.
The four role-specific gates below use the equation of `Inkling2LoopMessageGate`.
-/

@[expose] public section
namespace Transformer.InklingLoop

/-- Equality through the query token, including the query itself. -/
def Agree {A : Type*} (t : ℕ) (x y : ℕ → A) : Prop := ∀ i ≤ t, x i = y i

/-- One pass with no message followed by `extra` replays of the same core.
The first argument of the core always contains the original prelude. -/
def passes {A S M : Type*} (core : (ℕ → A) → (ℕ → Option M) → ℕ → S)
    (read : (ℕ → S) → ℕ → M) (pre : ℕ → A) : ℕ → ℕ → S
  | 0 => core pre (fun _ => none)
  | n + 1 => core pre (fun t => some (read (passes core read pre n) t))

/-- “Looping design”. Every number of replays preserves prefix causality
when the shared middle blocks and the previous-pass message reader do. -/
theorem passes_agree {A S M : Type*} (core : (ℕ → A) → (ℕ → Option M) → ℕ → S)
    (read : (ℕ → S) → ℕ → M)
    (hc : ∀ t x y m m', Agree t x y → Agree t m m' → Agree t (core x m) (core y m'))
    (hr : ∀ t x y, Agree t x y → Agree t (read x) (read y))
    (t : ℕ) (x y : ℕ → A) (h : Agree t x y) (n : ℕ) :
    Agree t (passes core read x n) (passes core read y n) := by
  induction n with
  | zero => exact hc _ _ _ _ _ h (fun _ _ => rfl)
  | succ n ih =>
    apply hc _ _ _ _ _ h
    intro i hi
    exact congrArg some (hr _ _ _ ih i hi)

/-- The output of one reference-model gate at one token, after its two
learned projections and tanh. Four such gates serve Q, K, V and the MLP. -/
def gate {d : ℕ} (forget admit x m : Fin d → ℝ) : Fin d → ℝ :=
  fun j => (1 - forget j) * x j + admit j * m j

/-- Zero-initializing both output projections lets all four roles take
their original inputs regardless of the previous pass's message. -/
theorem closed_gates {d : ℕ} (x m : Fin 4 → Fin d → ℝ) :
    (fun r => gate (fun _ => 0) (fun _ => 0) (x r) (m r)) = x := by
  funext r j
  simp [gate]

/-- If closing the message gates makes the core independent of a supplied
message, extra applications agree at every token with the first pass.
The prelude is reused even when the first pass changes its input. -/
theorem closed_replay {A S M : Type*} (core : (ℕ → A) → (ℕ → Option M) → ℕ → S)
    (read : (ℕ → S) → ℕ → M) (pre : ℕ → A)
    (closed : ∀ m : ℕ → M, core pre (fun t => some (m t)) = core pre (fun _ => none))
    (n : ℕ) : passes core read pre n = passes core read pre 0 := by
  cases n with
  | zero => rfl
  | succ n => exact closed (read (passes core read pre n))

private def adding : (ℕ → ℕ) → (ℕ → Option ℕ) → ℕ → ℕ :=
  fun p m t => p t + (m t).getD 0

example (t n : ℕ) (x y : ℕ → ℕ) (h : Agree t x y) :
    Agree t (passes adding id x n) (passes adding id y n) := by
  apply passes_agree _ _ _ _ _ _ _ h
  · intro k a b m m' hab hmm i hi
    simp [adding, hab i hi, hmm i hi]
  · exact fun _ _ _ hxy => hxy

example : passes adding id (fun _ => 2) 2 0 = 6 := by decide

example (n : ℕ) : passes (fun (p : ℕ → ℕ) (_ : ℕ → Option ℕ) t => p t + 1)
    id (fun t => 2*t) n = fun t => 2*t + 1 := by
  apply closed_replay
  exact fun _ => rfl

/-- The text decoder's sparse MoE sums only the router's selected experts.
At fixed routing, summing their coordinate outputs gives exactly the dense
weighted mixture with zeros outside the selected set. The signed output of
each expert can be arbitrary. Source: `Inkling2LoopExperts.forward`. -/
theorem routed_sum {e d : ℕ} (selected : Finset (Fin e)) (weight : Fin e → ℝ)
    (value : Fin e → Fin d → ℝ) (outside : ∀ i ∉ selected, weight i = 0) (j : Fin d) :
    ∑ i ∈ selected, weight i * value i j = ∑ i, weight i * value i j := by
  apply Finset.sum_subset (Finset.subset_univ selected)
  intro i _ hi
  rw [outside i hi, zero_mul]

example (v : Fin 4 → Fin 2 → ℝ) (j : Fin 2) :
    ∑ i ∈ ({1, 3} : Finset (Fin 4)), (if i = 1 then 3 else if i = 3 then 5 else 0) * v i j =
      ∑ i, (if i = 1 then 3 else if i = 3 then 5 else 0) * v i j := by
  apply routed_sum
  intro i hi
  simp only [Finset.mem_insert, Finset.mem_singleton, not_or] at hi
  simp [hi.1, hi.2]

end Transformer.InklingLoop

module

public import Transformer.Phasor.TripleTransport.ResumeSchedule

/-!
# Exact gradients from the addressed rows of a backward tape

Source: “Addressed row gradients” in the triple-transport experiment. The
claim concerns the order of contributions at each parameter between the
outer loss and the unmodified dense AdamW update.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.RowAdjoint
variable {K V G : Type*} [DecidableEq K]

/-- Each event carries one addressed contribution. The accumulator may
round, saturate or process noncommuting updates. -/
def push (add : V → G → V) (bank : K → V) (event : K × G) (i : K) : V :=
  if event.1 = i then add (bank i) event.2 else bank i

def dense (add : V → G → V) (bank : K → V) (events : List (K × G)) : K → V :=
  events.foldl (push add) bank

def atRow (add : V → G → V) (start : V) (events : List (K × G)) (i : K) : V :=
  (events.filterMap (fun e => if e.1 = i then some e.2 else none)).foldl add start

/-- Packing the rows preserves the order of additions within each address
for arbitrary arithmetic. -/
theorem exact (add : V → G → V) (bank : K → V) (events : List (K × G)) (i : K) :
    dense add bank events i = atRow add (bank i) events i := by
  induction events generalizing bank with
  | nil => rfl
  | cons ev es ih =>
    change dense add (push add bank ev) es i = _
    rw [ih]
    by_cases h : ev.1 = i <;> simp [atRow, push, h]

/-- Rows absent from the compact adjoint remain at the original initial
gradient. In the native backward this is zero even when AdamW carries
nonzero momentum for that parameter from earlier updates. -/
theorem untouched (add : V → G → V) (bank : K → V) (events : List (K × G)) (i : K)
    (missing : ∀ e ∈ events, e.1 ≠ i) : dense add bank events i = bank i := by
  induction events generalizing bank with
  | nil => rfl
  | cons ev es ih =>
    change dense add (push add bank ev) es i = _
    rw [ih _ (fun e he => missing e (List.mem_cons_of_mem _ he))]
    simp [push, missing ev (by simp)]

/-- Sparse storage only needs the addresses occurring in the tape.
Scattering over a freshly cleared full bank gives the dense update at
both touched and untouched parameters. -/
theorem scatter (add : V → G → V) (bank : K → V) (events : List (K × G)) (used : Finset K)
    (covers : ∀ e ∈ events, e.1 ∈ used) (i : K) :
    (if i ∈ used then atRow add (bank i) events i else bank i) = dense add bank events i := by
  split
  · exact (exact add bank events i).symm
  · rename_i absent
    exact (untouched add bank events i (fun e he eq => absent (eq ▸ covers e he))).symm

private def accumulate (x g : ℕ) := 2*x+g
private def events : List (Bool × ℕ) := [(true,3), (false,9), (true,5)]

example : dense accumulate (fun _ => 0) events true = 11 := by decide

example : dense accumulate (fun _ => 0) [(true,5),(true,3)] true ≠
    dense accumulate (fun _ => 0) events true := by decide

example : dense accumulate (fun _ : Fin 3 => 8) [(0,3),(1,9),(0,5)] 2 = 8 := by
  apply untouched
  decide

example : (if true ∈ ({true,false} : Finset Bool) then atRow accumulate 0 events true else 0) =
    dense accumulate (fun _ => 0) events true := by
  apply scatter accumulate (fun _ => 0) events ({true, false} : Finset Bool) _ true
  intro e _
  cases e.1 <;> simp

end Transformer.Phasor.TripleTransport.RowAdjoint

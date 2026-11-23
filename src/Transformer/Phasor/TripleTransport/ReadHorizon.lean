module
public import Transformer.Phasor.TripleTransport.TypedLookupPaths

/-! The forty-hop budget test. One call reveals the successor of the current
key. A decoder may use the entire history of revealed states. This contract
excludes inspecting other rows or precomputing longer jumps while encoding
the input. -/
@[expose] public section
namespace Transformer.Phasor.TripleTransport.ReadHorizon
open TypedLookupPaths

/-- Two stores can agree on h links and give any chosen value at link h+1. -/
def fork (h value k : ℕ) : Option ℕ :=
  if k<h then some (k+1) else if k=h then some value else none

/-- All intermediate keys seen through the last common link are independent
of the chosen terminal answer. -/
theorem shared (h value n : ℕ) (early : n≤h) : follow (fork h value) 0 n = n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have less : n<h := by omega
    rw [follow, ih (by omega)]
    simp [fork,less]

/-- Exactly one additional call distinguishes the two possible last edges. -/
theorem revealed (h value : ℕ) : follow (fork h value) 0 (h+1) = value := by
  rw [follow, shared h value h (by omega)]
  simp [fork]

/-- Even a decoder receiving every earlier key must fail on one of the
two memories when they differ only at the next edge. -/
theorem insufficient (h u v : ℕ) (different : u≠v) (answer : (Fin (h+1) → ℕ) → ℕ) :
    ¬ (answer (fun i => follow (fork h u) 0 i.val) = u ∧
       answer (fun i => follow (fork h v) 0 i.val) = v) := by
  have same : (fun i : Fin (h+1) => follow (fork h u) 0 i.val) =
      (fun i : Fin (h+1) => follow (fork h v) 0 i.val) := by
    funext i
    have bound : i.val≤h := by omega
    rw [shared _ _ _ bound, shared _ _ _ bound]
  rintro ⟨left,right⟩
  exact different (left.symm.trans ((congrArg answer same).trans right))

example : follow (fork 39 77) 0 39 = 39 := shared 39 77 39 (by decide)
example (reply : (Fin 40 → ℕ) → ℕ) :
    ¬ (reply (fun i => follow (fork 39 77) 0 i.val) = 77 ∧
       reply (fun i => follow (fork 39 88) 0 i.val) = 88) :=
  insufficient 39 77 88 (by decide) reply
example : follow (fork 39 77) 0 40 = 77 ∧ follow (fork 39 88) 0 40 = 88 :=
  ⟨revealed 39 77,revealed 39 88⟩

/-- Given the entire transition table, one parallel round can compose
each stored successor with its own stored successor. -/
def double {K : Type*} (f : K → K) : ℕ → K → K
  | 0 => f
  | n+1 => fun k => double f n (double f n k)

/-- The table constructed by j doubling rounds contains jumps of length
2^j. A separate table is needed for each power that will be queried. -/
theorem double_reads {K : Type*} (f : K → K) (j : ℕ) (x : K) :
    double f j x = f^[2^j] x := by
  induction j generalizing x with
  | zero => simp [double]
  | succ j ih =>
    change double f j (double f j x) = _
    rw [ih,ih,←Function.iterate_add_apply,Nat.pow_succ,Nat.mul_two]

/-- After constructing the tables, two lookups of lengths 8 and 32
compute exactly forty transitions for any fixed successor map. -/
theorem forty_from_two_jumps {K : Type*} (f : K → K) (x : K) :
    double f 5 (double f 3 x) = f^[40] x := by
  rw [double_reads,double_reads,←Function.iterate_add_apply]
  rfl

/-- The absolute read counts obtained by spending each successive budget increment. -/
def totals (past : ℕ) : List ℕ → List ℕ
  | [] => []
  | n :: ns => (n + past) :: totals (n + past) ns

/-- Keep the current state when evaluating a series of increasing read budgets. -/
def sample {S : Type*} (next : S → S) (s : S) : List ℕ → List S
  | [] => []
  | n :: ns => let t := next^[n] s; t :: sample next t ns

/-- Sharing the common prefix of the trajectory gives each requested state
in exactly its original order. The state can include a write-time cursor. -/
theorem sample_eq_restarts {S : Type*} (next : S → S) (s : S) (gaps : List ℕ) (past : ℕ) :
    sample next (next^[past] s) gaps = (totals past gaps).map (fun n => next^[n] s) := by
  induction gaps generalizing past with
  | nil => rfl
  | cons n ns ih =>
    simp only [sample, totals, List.map_cons, ← Function.iterate_add_apply, ih]

example : sample (fun n : ℕ => n + 1) 0 [20, 19, 1, 1] = [20, 39, 40, 41] := by decide

end Transformer.Phasor.TripleTransport.ReadHorizon
